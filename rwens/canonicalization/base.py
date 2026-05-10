"""
Base class for canonicalization modules.

Canonicalization modules transform Lean proof states into a canonical form
(e.g., consistent variable names like h₀, h₁) for use with models trained
on augmented data.
"""

from abc import ABC, abstractmethod
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple, Union

import leanclient as lc

from rwens.logger import get_logger
from rwens.utils.applier import TacticApplier

logger = get_logger(__name__)


class CanonicalizationModule(ABC):
    """
    Abstract base class for canonicalization modules.

    A canonicalization module maintains state for a Lean proof file and provides:
    - project_root: Root of the Lean project
    - reset: Initialize with imports and theorem statement
    - update: Append proof lines
    - get_states: Get current and augmented states
    - get_current_state: Get current state without augmentation
    """

    @property
    @abstractmethod
    def project_root(self) -> Path:
        """Root of the Lean project."""
        pass

    @abstractmethod
    def reset(self, imports: str, theorem_statement: str) -> None:
        """Clear the file and write imports + theorem statement (including := by)."""
        pass

    @abstractmethod
    def update(self, code_lines: Union[str, Sequence[str]]) -> None:
        """Append proof lines to the current file content (raw, no indentation)."""
        pass

    def apply_tactic(self, tactic: str) -> Optional[str]:
        """
        Apply a single tactic with correct indentation using the internal TacticApplier.
        Returns the state after application, or None if extraction failed.
        Default implementation falls back to update() for modules that do not support it.
        """
        self.update(tactic)
        return self.get_current_state()

    @abstractmethod
    def get_states(
        self, keep_augmentation: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (current_state, augmented_state)."""
        pass

    def get_augmentation(
        self,
        imports: str,
        theorem_stmt: str,
        theorem_name: Optional[str] = None,
    ) -> Tuple[Optional[str], List[str], str]:
        """
        Return the best augmented theorem statement (statement domain), with cache key.

        Works like get_states but takes imports + theorem statement and returns
        (augmented_statement, rw_tactics, cache_key). Uses StateProblemConverter
        for state->statement conversion. Optional; not all modules implement this.
        """
        raise NotImplementedError(
            "get_augmentation is not implemented for this canonicalization module"
        )

    @abstractmethod
    def get_current_state(self) -> Optional[str]:
        """Return current state at EOF without augmentation."""
        pass

    @abstractmethod
    def get_file_content(self) -> str:
        """Return current file content."""
        pass

    def get_diagnostics(self) -> Any:
        """
        Return Lean diagnostics (e.g. from LSP) for the current file.
        Used to detect proof success and tactic failures.
        Default returns a conservative result (success=False) when unavailable.
        """
        return SimpleNamespace(success=False, diagnostics=[])

    def invalidate_state_cache(self) -> None:
        """
        Invalidate any cached state so the next get_states/get_current_state
        fetches fresh data. No-op for modules that do not cache.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources (e.g., temp files)."""
        pass


class SimpleCanonicalizationModule(CanonicalizationModule):
    """
    Base implementation for canonicalization modules that use a single
    temporary Lean file and one Lean LSP client.

    This class provides the common lifecycle: a temp file under ``.temp/``,
    an LSP client that keeps that file open, a TacticApplier for applying
    tactics and reading goal state, and cached state to avoid repeated
    LSP calls. Subclasses (e.g. IdentityModule, VariableRenamer) inherit
    reset, update, apply_tactic, get_current_state, get_file_content,
    get_diagnostics, and close; they must implement get_states to define
    how the augmented state is produced (e.g. identity vs. revert/intro
    renaming).

    This class is not registered in the canonicalization factory and
    cannot be instantiated from config; it exists only as a shared base.

    Attributes:
        project_root: Resolved path to the Lean project (with lakefile).
        timeout_seconds: Timeout for get_goal / state-fetch calls.
        sfc: The Lean single-file client for the temp file.
    """

    def __init__(
        self,
        project_root: str,
        initial_imports: str = "import Mathlib\n",
        timeout_seconds: float = 120.0,
        temp_suffix: str = "temp",
    ) -> None:
        """
        Initialize the module with a temp file and LSP client.

        Args:
            project_root: Path to the Lean project root (must contain
                lakefile.lean or lakefile.toml). Used for the LSP and
                for the temp file under ``.temp/_temp_{temp_suffix}_{uuid}.lean``.
            initial_imports: Initial contents of the temp file (e.g.
                ``"import Mathlib\\n"``). Written to disk and sent to the LSP.
            timeout_seconds: Timeout in seconds for goal/state fetches
                when using the internal TacticApplier.
            temp_suffix: Suffix in the temp filename (e.g. ``"identity"``
                or ``"iter"``) so different subclasses use different files.
        """
        self._project_root = Path(project_root).resolve()
        self.timeout_seconds = timeout_seconds
        os.makedirs(self._project_root / ".temp", exist_ok=True)
        self._temp_file = self._project_root / f".temp/_temp_{temp_suffix}_{uuid.uuid4()}.lean"
        self.rel_path = self._temp_file.relative_to(self._project_root).as_posix()
        self._content = initial_imports if initial_imports.endswith("\n") else initial_imports + "\n"
        self._initial_imports = self._content
        self._temp_file.write_text(self._content, encoding="utf-8")
        self._client = lc.LeanLSPClient(
            str(self._project_root), prevent_cache_get=True, max_opened_files=8
        )
        self._client.open_file(self.rel_path)
        self.sfc: lc.SingleFileClient = self._client.create_file_client(self.rel_path)
        self._applier: Any = None
        self._cached_state: Optional[str] = None

    @property
    def project_root(self) -> Path:
        """Root of the Lean project."""
        return self._project_root

    def _eof_position(self, content: Optional[str] = None) -> Tuple[int, int]:
        """
        Return the (line, character) position at the end of the file.

        Used to build LSP DocumentContentChange ranges for appending or
        deleting at EOF. If content is None, uses the cached self._content.

        Returns:
            (end_line, end_char): 0-based line and column of the position
            immediately after the last character (for insertion at EOF).
        """
        text = self._content if content is None else content
        lines = text.split("\n")
        end_line = len(lines) - 1
        end_char = len(lines[-1]) if lines else 0
        return end_line, end_char

    def _recreate_file_client(self) -> None:
        """
        Recreate the Lean LSP client and single-file client after a failure.

        Writes the current cached content to the temp file, closes the old
        client (if possible), creates a new LeanLSPClient, reopens the file,
        and syncs the LSP buffer. Resets _applier and _cached_state so the
        next operation rebuilds state. Used by _replace_entire_file,
        _append_to_file, and get_diagnostics when an LSP call raises.
        """
        initial_content = self._content if self._content else "import Mathlib\n\n"
        if not self._temp_file.exists():
            logger.debug(f"Temp file {self._temp_file} does not exist, recreating")
        self._temp_file.write_text(initial_content, encoding="utf-8")
        try:
            if hasattr(self._client, "close"):
                self._client.close()
        except Exception:
            pass
        self._client = lc.LeanLSPClient(
            str(self._project_root), prevent_cache_get=True, max_opened_files=8
        )
        self.rel_path = self._temp_file.relative_to(self._project_root).as_posix()
        self._client.open_file(self.rel_path)
        self.sfc = self._client.create_file_client(self.rel_path)
        lines = initial_content.split("\n")
        change = lc.DocumentContentChange(
            text=initial_content,
            start=[0, 0],
            end=[max(0, len(lines) - 1), len(lines[-1]) if lines else 0],
        )
        self.sfc.update_file(changes=[change])
        self._applier = None
        self._cached_state = None

    def _char_offset_to_line_char(self, text: str, offset: int) -> Tuple[int, int]:
        """Convert 0-based character offset into (line, char) for LSP (0-based)."""
        if offset <= 0:
            return (0, 0)
        lines = text.split("\n")
        remaining = offset
        for line_idx, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            if remaining <= line_len:
                return (line_idx, min(remaining, len(line)))
            remaining -= line_len
        return (len(lines) - 1, len(lines[-1]) if lines else 0)

    def _replace_entire_file(self, new_content: str) -> None:
        """
        Replace the temp file content via LSP document change(s).

        If new_content starts with the same initial_imports as at creation,
        only the part after the imports is replaced (smaller edit, preserves
        imports in the LSP buffer). Otherwise the entire file is replaced.
        Updates self._content. On LSP failure, calls _recreate_file_client
        and retries once; raises RuntimeError on second failure.
        """
        if not new_content.endswith("\n"):
            new_content += "\n"
        prefix = self._initial_imports
        try:
            old = self.sfc.get_file_content()
            do_partial = (
                len(prefix) > 0
                and len(new_content) >= len(prefix)
                and new_content[: len(prefix)] == prefix
                and len(old) >= len(prefix)
                and old[: len(prefix)] == prefix
            )
            old_lines = old.split("\n")
            end = [len(old_lines) - 1, len(old_lines[-1]) if old_lines else 0]
            if do_partial:
                suffix = new_content[len(prefix) :]
                start_line, start_char = self._char_offset_to_line_char(old, len(prefix))
                change = lc.DocumentContentChange(
                    text=suffix,
                    start=[start_line, start_char],
                    end=end,
                )
            else:
                change = lc.DocumentContentChange(
                    text=new_content, start=[0, 0], end=end
                )
            self.sfc.update_file(changes=[change])
            self._content = new_content
        except Exception as e:
            logger.warning(f"File client failed in _replace_entire_file, recreating: {e}")
            self._recreate_file_client()
            try:
                old = self.sfc.get_file_content()
                old_lines = old.split("\n")
                end = [len(old_lines) - 1, len(old_lines[-1]) if old_lines else 0]
                do_partial = (
                    len(prefix) > 0
                    and len(new_content) >= len(prefix)
                    and new_content[: len(prefix)] == prefix
                    and len(old) >= len(prefix)
                    and old[: len(prefix)] == prefix
                )
                if do_partial:
                    suffix = new_content[len(prefix) :]
                    start_line, start_char = self._char_offset_to_line_char(old, len(prefix))
                    change = lc.DocumentContentChange(
                        text=suffix,
                        start=[start_line, start_char],
                        end=end,
                    )
                else:
                    change = lc.DocumentContentChange(
                        text=new_content, start=[0, 0], end=end
                    )
                self.sfc.update_file(changes=[change])
                self._content = new_content
            except Exception as e2:
                raise RuntimeError(f"Failed to replace file: {e2}") from e2

    def _append_to_file(self, text: str) -> None:
        """
        Append text at the end of the file via an LSP document change.

        Inserts at the position returned by _eof_position(); ensures text
        ends with a newline. Updates self._content. On LSP failure,
        recreates the file client and retries once; raises RuntimeError
        on second failure.
        """
        if not text:
            return
        if not text.endswith("\n"):
            text += "\n"
        try:
            end_line, end_char = self._eof_position()
            change = lc.DocumentContentChange(
                text=text,
                start=[end_line, end_char],
                end=[end_line, end_char],
            )
            self.sfc.update_file(changes=[change])
            self._content = self._content + text
        except Exception as e:
            logger.warning(f"File client failed in _append_to_file, recreating: {e}")
            self._recreate_file_client()
            try:
                end_line, end_char = self._eof_position()
                change = lc.DocumentContentChange(
                    text=text,
                    start=[end_line, end_char],
                    end=[end_line, end_char],
                )
                self.sfc.update_file(changes=[change])
                self._content = self._content + text
            except Exception as e2:
                raise RuntimeError(f"Failed to append to file: {e2}") from e2

    def _delete_range(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        """
        Delete text in [start, end) (LSP range). Updates cached _content.
        """
        try:
            change = lc.DocumentContentChange(
                text="",
                start=[start[0], start[1]],
                end=[end[0], end[1]],
            )
            self.sfc.update_file(changes=[change])
            self._content = self.sfc.get_file_content()
        except Exception as e:
            logger.warning(f"File client failed in _delete_range, recreating: {e}")
            self._recreate_file_client()
            try:
                change = lc.DocumentContentChange(
                    text="",
                    start=[start[0], start[1]],
                    end=[end[0], end[1]],
                )
                self.sfc.update_file(changes=[change])
                self._content = self.sfc.get_file_content()
            except Exception as e2:
                raise RuntimeError(f"Failed to delete range: {e2}") from e2

    def reset(self, imports: str, theorem_statement: str) -> None:
        """Clear the file and write imports + theorem statement (including := by)."""
        imports_s = imports.rstrip("\n")
        thm_s = theorem_statement.strip("\n")
        new_content = imports_s + "\n" + thm_s + "\n"
        self._replace_entire_file(new_content)
        self._applier = TacticApplier(self.sfc, timeout_seconds=self.timeout_seconds)
        self._cached_state = self._applier._build_hypothesis_stack_from_code(reset_stack=True)

    def update(self, code_lines: Union[str, Sequence[str]]) -> None:
        """
        Append proof lines to the end of the temp file.

        code_lines can be a single string (possibly with newlines) or a
        sequence of lines; it is appended as-is with no indentation
        normalization. After appending, the hypothesis stack is rebuilt
        and the current state is cached.
        """
        if isinstance(code_lines, str):
            text = code_lines
        else:
            text = "\n".join(code_lines)
        self._append_to_file(text)
        if self._applier is None:
            self._applier = TacticApplier(self.sfc, timeout_seconds=self.timeout_seconds)
        self._cached_state = self._applier._build_hypothesis_stack_from_code(reset_stack=True)

    def apply_tactic(self, tactic: str) -> Optional[str]:
        """
        Apply a single tactic at the current proof position and return the new state.

        Uses the internal TacticApplier to insert the tactic with correct
        indentation and to fetch the goal state after application. Updates
        cached content and state. Returns the state string, or None if
        the applier could not extract it.
        """
        if self._applier is None or self._cached_state is None:
            self._applier = TacticApplier(
                self.sfc, timeout_seconds=self.timeout_seconds
            )
            self._cached_state = self._applier._build_hypothesis_stack_from_code(
                reset_stack=True
            )
        state_after, updated_code = self._applier.update(tactic)
        self._content = updated_code
        self._cached_state = state_after
        return state_after

    def get_current_state(self) -> Optional[str]:
        """
        Return the current goal state at EOF, without any augmentation.

        Uses the cached state if valid; otherwise rebuilds the hypothesis
        stack from the current file content and caches the result. Returns
        None if state fetch fails (e.g. timeout).
        """
        if self._applier is None or self._cached_state is None:
            self._applier = self._applier or TacticApplier(
                self.sfc, timeout_seconds=self.timeout_seconds
            )
            self._cached_state = self._applier._build_hypothesis_stack_from_code(
                reset_stack=True
            )
        return self._cached_state

    @abstractmethod
    def get_states(
        self, keep_augmentation: bool = False
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (current_state, augmented_state). Subclasses must implement.

        The augmented state is the form presented to the model (e.g. after
        variable renaming). keep_augmentation controls whether to leave
        any temporary proof lines in the file (e.g. for revert/intro).
        """
        pass

    def get_file_content(self) -> str:
        """Return the current full content of the temp file from the LSP."""
        return self.sfc.get_file_content()

    def invalidate_state_cache(self) -> None:
        """
        Invalidate the cached goal state.

        The next get_current_state or get_states will rebuild the
        hypothesis stack from the current file content.
        """
        self._cached_state = None

    def get_diagnostics(self) -> Any:
        """
        Return Lean LSP diagnostics for the temp file.

        On failure (e.g. client disconnected), recreates the file client
        and retries once with a short inactivity timeout. Raises
        RuntimeError if the retry fails.
        """
        try:
            return self.sfc.get_diagnostics()
        except Exception as e:
            logger.warning(f"File client failed in get_diagnostics, recreating: {e}")
            self._recreate_file_client()
            try:
                return self.sfc.get_diagnostics(inactivity_timeout=5.0)
            except Exception as e2:
                raise RuntimeError(f"Failed to get diagnostics: {e2}") from e2

    def close(self) -> None:
        """
        Clean up resources: remove the temp Lean file from disk.

        Best-effort only; exceptions are swallowed. Does not close the
        LSP client explicitly (subclasses may override to add that).
        """
        try:
            self._temp_file.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

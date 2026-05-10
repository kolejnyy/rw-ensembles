"""
Variable renaming utilities.

This module provides `VariableRenamer`, a stateful wrapper around a single
Lean LSP file client. It is intended for workflows where we process many checkpoints
from the *same* source file in order, and want to avoid restarting Lean / reopening
files between queries.

Core idea:
- Keep one temp Lean file open via Lean LSP.
- `reset(...)` writes imports + theorem header (":= by") into that file.
- `update(...)` appends additional proof lines to the end of the file.
- `get_states(...)`:
    * reads the current goal state at EOF,
    * temporarily appends a revert/intro "augmentation" script,
    * returns the augmented state,
    * then deletes those temporary lines, restoring the file to the previous state.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from rwens.canonicalization.utils import (
    build_intro_tactic,
    classify_declarations,
    create_renaming_map,
    detect_intro_prefix,
    extract_all_declarations,
    get_next_prefixed_name,
    normalize_superscript_numbers,
    prev_shadowed,
    rename_have_hypotheses,
    rename_intro_line,
    replace_known_names,
    split_prefixed_subscript,
    _RENAMING_PREFIXES,
)
from rwens.canonicalization.base import SimpleCanonicalizationModule
from rwens.logger import get_logger
from rwens.utils.applier import TacticApplier, StateFetchAbort, GOAL_TIMEOUT_SECONDS

logger = get_logger(__name__)


_MAX_ROUNDS = 7


class VariableRenamer(SimpleCanonicalizationModule):
    """
    Stateful renamer that reuses a single Lean LSP file client across many queries.
    Implements CanonicalizationModule for variable canonicalization.

    This class does *not* attempt to be clever about incremental hypothesis-stack
    maintenance; it focuses on avoiding LSP restarts and whole-file reinitializations.
    """

    def __init__(
        self,
        project_root: str,
        initial_imports: str = "import Mathlib\n",
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            project_root=project_root,
            initial_imports=initial_imports,
            timeout_seconds=timeout_seconds,
            temp_suffix="iter",
        )
        self._last_var_mapping: Dict[str, str] = {}

    # -------------------------
    # Public API (get_states is VariableRenamer-specific)
    # -------------------------
    def get_states(
        self,
        keep_augmentation: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Return (current_state_at_eof, augmented_state).

        Augmentation is done by temporarily appending revert/intro lines to the file,
        extracting the resulting state, and then deleting those appended lines so the
        file returns to the exact previous content (unless keep_augmentation=True).

        Args:
            keep_augmentation: If True, do not delete the augmentation lines after
                extraction. The proof file will retain revert/intro so the state
                remains in canonical form. Skips adding "sorry" so the goal stays open.

        Raises StateFetchAbort if the cached state is timeout (None) or error ("");
        caller should skip to next problem.
        """
        if self._applier is None or self._cached_state is None:
            self._applier = self._applier or TacticApplier(
                self.sfc, timeout_seconds=self.timeout_seconds
            )
            self._cached_state = self._applier._build_hypothesis_stack_from_code(
                reset_stack=True
            )
        state = self._cached_state
        if state is None:
            raise StateFetchAbort("state fetch timed out", is_timeout=True)

        # Record the insertion point so we can delete temporary lines afterwards.
        try:
            current_file_content = self.sfc.get_file_content()
        except Exception as e:
            # File/client might be broken - try to recreate it
            logger.warning(f"File client operation failed in get_states, recreating: {e}")
            self._recreate_file_client()
            current_file_content = self.sfc.get_file_content()
        insert_start = self._eof_position(current_file_content)

        try:
            aug_state, mapping = self._augment_state_in_place(
                state, skip_sorry=keep_augmentation
            )
            self._last_var_mapping = mapping

            # Check if mapping is an identity mapping; if so, set keep_augmentation to False
            if mapping == {k: k for k in mapping.keys()}:
                keep_augmentation = False

            result = (state, aug_state)
        except StateFetchAbort as e:
            logger.warning("state fetch timed out: %s", e)
            raise
        finally:
            if keep_augmentation:
                # Refresh content and state cache so update() appends correctly.
                self._content = self.sfc.get_file_content()
                self._applier = TacticApplier(
                    self.sfc, timeout_seconds=self.timeout_seconds
                )
                self._cached_state = self._applier._build_hypothesis_stack_from_code(
                    reset_stack=True
                )
            else:
                try:
                    cur = self.sfc.get_file_content()
                except Exception as e:
                    # File/client might be broken - try to recreate it
                    logger.warning(f"File client operation failed in get_states cleanup, recreating: {e}")
                    self._recreate_file_client()
                    cur = self.sfc.get_file_content()
                cur_end = self._eof_position(cur)
                if cur_end != insert_start:
                    self._delete_range(insert_start, cur_end)
        return result




    # -------------------------
    # Augmentation logic (revert/intro)
    # -------------------------
    def _build_intro_tactic(self, names: List[str], var_mapping: Dict[str, str]) -> str:
        return build_intro_tactic(names, var_mapping)

    @staticmethod
    def _check_state_result(s: Optional[str], where: str = "update") -> str:
        """Raise StateFetchAbort if state is timeout (None) or error ("")."""
        if s is None:
            raise StateFetchAbort(f"state fetch timed out during {where}", is_timeout=True)
        return s

    def _augment_state_in_place(
        self, state: str, skip_sorry: bool = False
    ) -> Tuple[str, Dict[str, str]]:
        """
        Apply revert/intro augmentation at the end of the current file, and return
        (augmented_state, var_mapping). The caller is responsible for cleaning up
        the appended lines (unless skip_sorry=True and caller keeps them).

        Args:
            skip_sorry: If True, do not add "sorry" to close goals. Use when
                keeping augmentation lines in the proof (goal stays open).
        """
        empty_mapping: Dict[str, str] = {}
        state_parts = [part.strip() for part in state.split("\n\n") if part.strip()]
        if not state_parts:
            return "", empty_mapping

        applier = TacticApplier(self.sfc, timeout_seconds=self.timeout_seconds)
        if len(state_parts) > 1:
            return self._augment_multiple_cases(applier, state_parts, skip_sorry)
        return self._augment_single_case(applier, state_parts[0], skip_sorry)

    def _augment_multiple_cases(
        self,
        applier: TacticApplier,
        state_parts: List[str],
        skip_sorry: bool = False,
    ) -> Tuple[str, Dict[str, str]]:
        empty_mapping: Dict[str, str] = {}
        aug_states: List[str] = []
        first_mapping: Optional[Dict[str, str]] = None

        for state_part in state_parts:
            all_decls = extract_all_declarations(state_part)
            if not all_decls or not all_decls[0]:
                if skip_sorry:
                    return state_part, empty_mapping
                aug_states.append(state_part)
                if first_mapping is None:
                    first_mapping = empty_mapping
                # Close this goal to move on.
                close_state, _ = applier.update("· sorry")
                self._check_state_result(close_state, "close goal")
                continue

            variables_list, hypotheses_list = classify_declarations(state_part)
            intro_name_for: Dict[str, str] = create_renaming_map(variables_list[0], hypotheses_list[0])
            if first_mapping is None:
                first_mapping = dict(intro_name_for)

            intros_gathered: List[str] = []
            current_state = state_part

            first_revert_in_case = True
            rounds = _MAX_ROUNDS
            while rounds > 0:
                rounds -= 1
                if rounds == 1:
                    return ("", {"warning": "_MAX_ROUNDS reached"})

                current_decls = extract_all_declarations(current_state)
                if not current_decls or not current_decls[0]:
                    break

                declarations = current_decls[0]
                names = [name for name, _, _ in declarations]
                visible = [n for n in names if "✝" not in n]
                if not visible:
                    break

                revert_tactic = "revert " + " ".join(visible)
                if first_revert_in_case:
                    revert_line = "· " + revert_tactic
                    first_revert_in_case = False
                else:
                    revert_line = revert_tactic

                state_after_revert, _ = applier.update(revert_line)
                state_after_revert = self._check_state_result(state_after_revert, "revert")

                after_decls = extract_all_declarations(state_after_revert or "") if state_after_revert else []
                after_names = set()
                if after_decls and after_decls[0]:
                    after_names = {name for name, _, _ in after_decls[0]}

                reverted = [n for n in names if ("✝" not in n or (prev_shadowed(n) not in after_names and prev_shadowed(n) in names))]

                intro_tactic = self._build_intro_tactic(reverted, intro_name_for)
                intros_gathered.append(intro_tactic)

                for n in reverted:
                    short_name = n.split("✝")[0]
                    if short_name not in intro_name_for:
                        normalized = normalize_superscript_numbers(n)
                        intro_name_for[n] = "shadowed_" + normalized

                if not after_decls or not after_decls[0]:
                    break
                current_state = state_after_revert or ""

            state_after_intros: Optional[str] = None
            for intro_t in reversed(intros_gathered):
                state_after_intros, _ = applier.update(intro_t)
                state_after_intros = self._check_state_result(state_after_intros or "", "intro")
            aug_states.append(state_after_intros or "")
            if not skip_sorry:
                sorry_state, _ = applier.update("sorry")
                self._check_state_result(sorry_state, "sorry")

            if skip_sorry or "\n\n" in aug_states[-1]:
                break

        joined = "\n\n".join([x for x in aug_states if x])
        return joined, (first_mapping if first_mapping is not None else empty_mapping)

    def _augment_single_case(
        self,
        applier: TacticApplier,
        state_part: str,
        skip_sorry: bool = False,
    ) -> Tuple[str, Dict[str, str]]:
        empty_mapping: Dict[str, str] = {}

        all_decls = extract_all_declarations(state_part)
        if not all_decls or not all_decls[0]:
            return state_part, empty_mapping

        variables_list, hypotheses_list = classify_declarations(state_part)
        intro_name_for: Dict[str, str] = create_renaming_map(variables_list[0], hypotheses_list[0])

        intros_gathered: List[str] = []
        current_state = state_part

        rounds = _MAX_ROUNDS
        while rounds > 0:
            rounds -= 1
            if rounds == 1:
                return ("", {"warning": "_MAX_ROUNDS reached"})

            current_decls = extract_all_declarations(current_state)
            if not current_decls or not current_decls[0]:
                break

            declarations = current_decls[0]
            names = [name for name, _, _ in declarations]
            visible = [n for n in names if "✝" not in n]
            if not visible:
                break

            revert_tactic = "revert " + " ".join(visible)
            state_after_revert, _ = applier.update(revert_tactic)

            after_decls = extract_all_declarations(state_after_revert or "") if state_after_revert else []
            after_names = set()
            if after_decls and after_decls[0]:
                after_names = {name for name, _, _ in after_decls[0]}

            reverted = [n for n in names if ("✝" not in n or (prev_shadowed(n) not in after_names and prev_shadowed(n) in names))]

            intro_tactic = self._build_intro_tactic(reverted, intro_name_for)
            intros_gathered.append(intro_tactic)

            for n in reverted:
                short_name = n.split("✝")[0]
                if short_name not in intro_name_for:
                    normalized = normalize_superscript_numbers(n)
                    intro_name_for[n] = "shadowed_" + normalized

            if not after_decls or not after_decls[0]:
                break
            current_state = state_after_revert or ""

        state_after_intros: Optional[str] = None
        for intro_t in reversed(intros_gathered):
            state_after_intros, _ = applier.update(intro_t)

        if not skip_sorry:
            applier.update("sorry")
        return (state_after_intros or ""), intro_name_for

    # -------------------------
    # Tactic renaming (have / intro / known names)
    # -------------------------
    def _rename_tactic(
        self, tactic: str, var_mapping: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Rename variable/hypothesis names in a tactic to match the augmented state.
        Uses the mapping from the last get_states() call if var_mapping is not provided.
        """
        if not tactic.strip():
            return tactic
        mapping = var_mapping if var_mapping is not None else self._last_var_mapping

        text = replace_known_names(tactic, mapping)
        used_numbers: Dict[str, set] = {p: set() for p in _RENAMING_PREFIXES}
        for value in mapping.values():
            prefix, num = split_prefixed_subscript(value)
            if prefix in used_numbers and num is not None:
                used_numbers[prefix].add(num)

        text = rename_have_hypotheses(text, used_numbers)
        text = rename_intro_line(text, used_numbers)
        return text

    def close(self) -> None:
        """Best-effort cleanup: close the temp file on disk."""
        try:
            super().close()
        except Exception:
            pass


"""
Lean proof verifier using a single persistent file and LSP client.

Keeps a fixed preamble (e.g. imports) loaded so only the theorem + proof
part is updated when verifying, avoiding repeated reload of Mathlib etc.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import leanclient as lc

from invpro.utils.metrics import diagnostics_indicate_success
from invpro.utils.text_positions import eof_line_col, offset_to_line_col

logger = logging.getLogger(__name__)


class ProofVerifier:
    """
    Verifies Lean proof code using a single persistent LSP file client.

    The file is initialized with a fixed preamble (e.g. ``import Mathlib\\n``).
    When verifying, only the content after the preamble is replaced if
    full_code starts with that preamble, so imports are not reloaded.
    """

    def __init__(
        self,
        project_root: str,
        initial_imports: str = "import Mathlib\n",
        timeout_seconds: float = 120.0,
    ):
        self._project_root = Path(project_root).resolve()
        self._timeout_seconds = timeout_seconds
        self._preamble = (
            initial_imports.rstrip("\n\r") + "\n"
            if initial_imports else ""
        )
        self._temp_dir = self._project_root / ".temp"
        _temp_lean_name = f"_temp_verifier_{uuid.uuid4().hex[:12]}.lean"
        self._temp_path = self._temp_dir / _temp_lean_name
        self._lsp_client: Optional[lc.LeanLSPClient] = None
        self._sfc: Optional[lc.SingleFileClient] = None

        if not self._project_root.exists():
            raise ValueError(f"Project root does not exist: {self._project_root}")
        if not (self._project_root / "lakefile.lean").exists() and not (
            self._project_root / "lakefile.toml"
        ).exists():
            raise ValueError(
                "Project root must contain lakefile.lean or lakefile.toml"
            )

    @classmethod
    def is_lsp_server_crash_error(cls, exc: BaseException) -> bool:
        """
        True if the exception indicates the Lean LSP server died (recoverable by
        disposing clients and creating a new LeanLSPClient).

        Matches messages like:
          LSP Error: {... 'Server process for file://... crashed ...' 'code': -32901}
        """
        s = str(exc).lower()
        r = repr(exc).lower()
        blob = f"{s} {r}"
        return (
            "-32901" in blob
            or ("server process" in s and "crashed" in s)
            or "lsp error" in s
            or "broken pipe" in s
            or "connection reset" in s
        )

    def _get_sfc(self) -> lc.SingleFileClient:
        """Create and cache LSP client and SingleFileClient on first use."""
        if self._sfc is not None:
            return self._sfc
        os.makedirs(self._temp_dir, exist_ok=True)
        initial_content = self._preamble if self._preamble else "\n"
        self._temp_path.write_text(initial_content, encoding="utf-8")
        rel_path = self._temp_path.relative_to(self._project_root).as_posix()
        self._lsp_client = lc.LeanLSPClient(
            str(self._project_root),
            prevent_cache_get=True,
            max_opened_files=8,
        )
        self._lsp_client.open_file(rel_path)
        self._sfc = self._lsp_client.create_file_client(rel_path)
        return self._sfc

    def _dispose_lsp_clients(self) -> None:
        """
        Close the Lean LSP client and clear cached handles so the next call
        creates a new server process and file client.
        """
        self._sfc = None
        client = self._lsp_client
        self._lsp_client = None
        if client is None:
            return
        try:
            close = getattr(client, "close", None)
            if close is not None:
                close()
        except Exception as e:
            logger.debug("ProofVerifier: error while closing LeanLSPClient: %s", e)

    def close(self) -> None:
        """Release Lean LSP resources held by this verifier."""
        self._dispose_lsp_clients()

    def __del__(self) -> None:
        # Best-effort cleanup when callers forget explicit close().
        try:
            self._dispose_lsp_clients()
        except Exception:
            pass

    def _apply_code_and_get_diagnostics(self, full_code: str):
        """
        Push ``full_code`` into the temp buffer and return Lean diagnostics.
        ``full_code`` must already end with a newline.
        """
        sfc = self._get_sfc()
        current = sfc.get_file_content()
        eof_line, eof_char = eof_line_col(current)

        if (
            self._preamble
            and full_code.startswith(self._preamble)
            and current.startswith(self._preamble)
        ):
            # Replace only the part after the preamble so imports are not reloaded.
            start_line, start_col = offset_to_line_col(current, len(self._preamble))
            replacement = full_code[len(self._preamble) :]
            change = lc.DocumentContentChange(
                text=replacement,
                start=[start_line, start_col],
                end=[eof_line, eof_char],
            )
        else:
            # Full replace (e.g. first run or preamble mismatch).
            change = lc.DocumentContentChange(
                text=full_code,
                start=[0, 0],
                end=[eof_line, eof_char],
            )
        sfc.update_file(changes=[change])
        self._temp_path.write_text(full_code, encoding="utf-8")
        return sfc.get_diagnostics(inactivity_timeout=self._timeout_seconds)

    def verify(
        self,
        full_code: str,
        *,
        diagnostics_ok: Optional[Callable[[Any], bool]] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Verify full Lean file content. Uses a minimal edit when full_code
        starts with the verifier's preamble so imports are not reloaded.

        If the Lean LSP server crashes (e.g. code -32901), clients are
        disposed, a warning is logged, and verification is retried once with a
        fresh server.

        Parameters
        ----------
        full_code
            Full Lean source (typically ends with newline).
        diagnostics_ok
            Predicate on the leanclient diagnostics object (same shape as for
            :func:`invpro.utils.metrics.diagnostics_indicate_success`). Default:
            :func:`~invpro.utils.metrics.diagnostics_indicate_success`.

        Returns
        -------
        (True, None) if diagnostics indicate success per ``diagnostics_ok``,
        (False, error_message) otherwise.
        """
        ok = diagnostics_ok or diagnostics_indicate_success

        if not full_code.endswith("\n"):
            full_code += "\n"

        diag = None
        last_err: Optional[str] = None
        for attempt in range(2):
            try:
                diag = self._apply_code_and_get_diagnostics(full_code)
                break
            except Exception as e:
                last_err = str(e)
                if self.is_lsp_server_crash_error(e) and attempt == 0:
                    logger.warning(
                        "Lean LSP server crashed for temp file %s; "
                        "disposing LSP client and retrying verify once. Error: %s",
                        self._temp_path,
                        last_err,
                    )
                    self._dispose_lsp_clients()
                    continue
                return False, last_err
        else:
            return False, last_err or "ProofVerifier: verify failed without diagnostics"

        if ok(diag):
            # Confirm success once more to reduce rare stale/late diagnostic races.
            # If the second read disagrees, treat as failure (or retry once on crash).
            try:
                confirm = self._get_sfc().get_diagnostics(
                    inactivity_timeout=self._timeout_seconds
                )
                if ok(confirm):
                    return True, None
                diag = confirm
            except Exception as e:
                if self.is_lsp_server_crash_error(e):
                    self._dispose_lsp_clients()
                    try:
                        diag2 = self._apply_code_and_get_diagnostics(full_code)
                        if ok(diag2):
                            return True, None
                        diag = diag2
                    except Exception as e2:
                        return False, str(e2)
                else:
                    return False, str(e)
        diagnostics = getattr(diag, "diagnostics", None) or []
        err_parts = [str(d) for d in diagnostics if d]
        return (
            False,
            "; ".join(err_parts) if err_parts else "Lean diagnostics indicated failure",
        )

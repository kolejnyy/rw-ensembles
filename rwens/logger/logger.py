"""File-based logger implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

_RWENS_ROOT = "rwens"
_handler: Optional[logging.FileHandler] = None
_log_path: Optional[Path] = None
_disabled: bool = False


def set_log_file(path: Optional[Path]) -> None:
    """
    Set the log file for rwens loggers. All get_logger() loggers write here.
    Pass None to disable file logging.
    """
    global _handler, _log_path, _disabled

    _disabled = path is None

    root = logging.getLogger(_RWENS_ROOT)
    if _handler is not None:
        root.removeHandler(_handler)
        try:
            _handler.close()
        except Exception:
            pass
        _handler = None
    _log_path = None

    if path is None:
        return

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    except Exception:
        return

    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(_handler)
    root.setLevel(logging.DEBUG)
    _log_path = path


def _ensure_handler() -> None:
    """Use RWENS_LOG_FILE or default rwens.log if no file set and not disabled."""
    global _handler
    if _handler is not None or _disabled:
        return
    path = os.environ.get("RWENS_LOG_FILE")
    if path:
        set_log_file(Path(path))
    else:
        set_log_file(Path("rwens.log"))


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes to the rwens log file.
    Safe to use from subprocesses and threads.
    """
    _ensure_handler()
    if name.startswith("rwens.") or name == "rwens":
        full = name
    else:
        full = f"{_RWENS_ROOT}.{name}"
    return logging.getLogger(full)

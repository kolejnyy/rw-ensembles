"""
Character offset ↔ (line, column) helpers for LSP-style document edits.
"""

from __future__ import annotations


def offset_to_line_col(text: str, offset: int) -> tuple[int, int]:
    """Return (0-based line, column) for character offset in ``text``."""
    if offset <= 0:
        return 0, 0
    lines = text[:offset].split("\n")
    line = len(lines) - 1
    col = len(lines[-1]) if lines else 0
    return line, col


def eof_line_col(text: str) -> tuple[int, int]:
    """Return (0-based line, column) of the position after the last character."""
    lines = text.split("\n")
    end_line = max(0, len(lines) - 1)
    end_char = len(lines[-1]) if lines else 0
    return end_line, end_char

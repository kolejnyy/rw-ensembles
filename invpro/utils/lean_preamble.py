"""Lean file preamble helpers (imports / options)."""

from __future__ import annotations

import re

_AESOP_IMPORT_LINE = re.compile(r"(?m)^\s*import\s+Aesop\s*$")


def sanitize_preamble_remove_aesop(preamble: str) -> str:
    """
    Drop standalone ``import Aesop`` lines from a problem header.

    Those imports add build/load cost and are unnecessary for many proof pipelines
    when Mathlib alone suffices.
    """
    cleaned = _AESOP_IMPORT_LINE.sub("", preamble)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned

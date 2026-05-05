"""
Formatting and extraction for Lean theorem snippets (single-theorem files).

For tooling that builds or parses ``theorem … := by`` blocks (equivalence checks, eval
scripts, etc.).
"""

from __future__ import annotations

import re
from typing import Sequence

from invpro.dataset.utils import split_declarations_theorem_proof

_THEOREM_END_BY = re.compile(r":=\s*by\s*$", re.IGNORECASE | re.MULTILINE)


def append_tactics_block(theorem_through_by: str, tactics: Sequence[str]) -> str:
    """
    Turn ``theorem ... := by`` into one proof block: each tactic on its own line as
    ``try <tactic>`` (see ``.todo/RWData.md``). Pass **bare** tactic names only
    (e.g. ``ring_nf``), not an outer ``try``.
    """
    tacs = tuple(t.strip() for t in tactics if t.strip())
    if not tacs:
        raise ValueError("Need at least one non-empty tactic.")
    raw = theorem_through_by.rstrip()
    if not _THEOREM_END_BY.search(raw + "\n"):
        raise ValueError("Expected theorem string ending with ':= by'.")
    lines = "\n".join(f"  try {t}" for t in tacs)
    replaced = _THEOREM_END_BY.sub(f":= by\n{lines}\n", raw, count=1)
    return replaced if replaced.endswith("\n") else replaced + "\n"


def append_tactic_proof(theorem_through_by: str, tactic: str) -> str:
    """Single-tactic convenience wrapper for :func:`append_tactics_block` (emits ``try <tactic>``)."""
    return append_tactics_block(theorem_through_by, (tactic,))


def extract_proof_body_after_by(full_lean_file: str) -> str:
    """Return the proof text after the first ``:= by`` in a single-theorem file."""
    _, _, proof = split_declarations_theorem_proof(full_lean_file)
    return proof


def _leading_ws_width(ln: str) -> int:
    """Count leading spaces and tabs (each column = one char)."""
    i = 0
    n = len(ln)
    while i < n and ln[i] in " \t":
        i += 1
    return i


def normalize_proof_indent(proof_body: str) -> str:
    """
    Indent a proof body for placement directly under ``:= by``.

    Preserves **relative** indentation between lines (nested ``have`` / ``·`` blocks).
    The previous implementation stripped every line and re-indented with two spaces only,
    which broke multi-level proofs.
    """
    lines = proof_body.replace("\r\n", "\n").split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return "\n"
    min_indent = min(_leading_ws_width(ln) for ln in nonempty)
    base = 2
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        w = _leading_ws_width(ln)
        if w >= min_indent:
            rel = ln[min_indent:]
        else:
            rel = ln.lstrip(" \t")
        out.append(" " * base + rel)
    return "\n".join(out).rstrip() + "\n"


def theorem_header_through_by(theorem_through_by: str) -> str:
    """``theorem ... := by`` including newline, without proof body."""
    t = theorem_through_by.rstrip()
    if not _THEOREM_END_BY.search(t + "\n"):
        raise ValueError("Expected theorem string ending with ':= by'.")
    return _THEOREM_END_BY.sub(":= by\n", t, count=1)


def theorem_with_sorry(theorem_through_by: str) -> str:
    """Replace the proof stub with ``sorry`` (one theorem, ends with ``:= by``)."""
    t = theorem_through_by.rstrip()
    if not _THEOREM_END_BY.search(t + "\n"):
        raise ValueError("Expected theorem string ending with ':= by'.")
    return _THEOREM_END_BY.sub(":= by\n  sorry\n", t, count=1)

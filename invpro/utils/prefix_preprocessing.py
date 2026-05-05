"""
Prefix preprocessing utilities.

Currently mirrors the indentation-fixing logic used in `scripts/_test_renamer.py`.
We will iterate on this as we encounter more formatting edge-cases.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


def _align_indentation(text: str) -> str:
    """
    Apply 4-space reduction and realigning to a string.

    Step 1: If all indents are multiples of 4 and min >= 4, reduce by half.
    Step 2: If min indent > 2, reduce by (min - 2) so smallest level is 2.
    """
    lines = text.split("\n")
    indent_counts = [
        len(l) - len(l.lstrip())
        for l in lines
        if l.strip() and (len(l) - len(l.lstrip())) > 0
    ]
    use_4_space = False
    if indent_counts:
        min_indent = min(indent_counts)
        if min_indent >= 4 and all(c % 4 == 0 for c in indent_counts):
            use_4_space = True

    step1: list[str] = []
    for line in lines:
        if use_4_space and line.strip():
            lead = len(line) - len(line.lstrip())
            if lead > 0 and lead % 4 == 0:
                step1.append(" " * (lead // 2) + line.lstrip())
            else:
                step1.append(line)
        else:
            step1.append(line)

    s = "\n".join(step1)
    lines = s.split("\n")
    indent_counts = [
        len(l) - len(l.lstrip())
        for l in lines
        if l.strip() and (len(l) - len(l.lstrip())) > 0
    ]
    if not indent_counts:
        return s

    min_indent = min(indent_counts)
    if min_indent <= 2:
        return s

    reduction = min_indent - 2
    out: list[str] = []
    for line in lines:
        if line.strip():
            lead = len(line) - len(line.lstrip())
            if lead > 0:
                out.append(" " * max(0, lead - reduction) + line.lstrip())
            else:
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def fix_indentation(prefix: str) -> str:
    """
    Normalize indentation in the part of prefix *after* the first "theorem ... := by".

    Everything up to and including the matched `... by` is left unchanged.
    The proof body after that is processed with:
    Step 1 – 4-space case: reduce by half if all indents multiples of 4.
    Step 2 – Realigning: reduce so smallest indent is 2 spaces.
    """
    before, proof_body = _split_prefix_at_theorem_by(prefix)
    if before is None or proof_body is None:
        return prefix

    return before + "\n" + _align_indentation(proof_body)


_THEOREM_BY_RE = re.compile(r"theorem\s+.*?:=\s*by", re.MULTILINE | re.DOTALL)


def _split_prefix_at_theorem_by(prefix: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split prefix into (before, proof_body) at the first `theorem ... := <ws> by` match.
    `before` includes the matched `... by`.
    """
    m = _THEOREM_BY_RE.search(prefix)
    if m is None:
        return None, None
    before = prefix[: m.end()]
    after = prefix[m.end() :].lstrip("\n")
    if not after:
        return before, None
    return before, after


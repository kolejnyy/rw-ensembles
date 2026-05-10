"""
Unified JSONL record shape for rewrite / augmentation datasets (extends ``data/minif2f.jsonl`` fields).

Each line is one variant (one ``theorem``). ``name`` is the **augmented** theorem identifier
(``seed_v2``, ``seed_v3_renamed``, …) matching ``formal_statement``; ``original_name`` is the
benchmark seed name. ``variable_map`` / ``hypothesis_map`` are set for α-renamed rows; otherwise
``null``. After equivalence verification, rows may gain a ``certificate`` field.
"""

from __future__ import annotations

import re
from typing import Any

_AESOP_LINE_RE = re.compile(r"^\s*import\s+Aesop\s*$", re.MULTILINE)
# Proof start in benchmark-style statements; drop any following tactic/sorry body.
_THEOREM_BY_HEAD = re.compile(r":=\s*by\b")


def sanitize_header_remove_aesop(header: str) -> str:
    """Drop ``import Aesop`` lines; trim trailing whitespace."""
    lines = []
    for line in (header or "").splitlines():
        if _AESOP_LINE_RE.match(line):
            continue
        lines.append(line)
    text = "\n".join(lines).rstrip()
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def truncate_theorem_after_by(s: str) -> str:
    """
    Keep the declaration through ``:= by`` only.

    Models sometimes emit ``sorry`` or tactic scripts after ``by``; benchmarks use an empty
    proof shell ending at ``:= by``.
    """
    t = (s or "").strip()
    m = _THEOREM_BY_HEAD.search(t)
    if not m:
        return t
    return t[: m.end()].rstrip()


def normalize_formal_statement(s: str) -> str:
    """Theorem text through ``:= by`` only, single trailing newline (benchmark style)."""
    return truncate_theorem_after_by(s).rstrip() + "\n"


def make_rewrite_dataset_record(
    *,
    name: str,
    original_name: str,
    split: str,
    informal_prefix: str,
    formal_statement: str,
    header: str,
    variant: str,
    dataset_name: str,
    original_formal_statement: str,
    variable_map: dict[str, str] | None = None,
    hypothesis_map: dict[str, str] | None = None,
    goal: str | None = None,
    certificate: str | None = None,
) -> dict[str, Any]:
    """
    Build one JSON-serializable record for the aggregated rewrites dataset.

    Parameters
    ----------
    name
        Declared theorem name for this row (e.g. ``problem_v2``, ``problem_v3_renamed``). Matches
        the identifier in ``formal_statement`` so downstream tools get unique path segments.
    original_name
        Seed benchmark theorem name (e.g. ``problem``).
    variant
        ``\"original\"`` for the seed theorem row, else e.g. ``\"v2\"``, ``\"v11\"``, ``\"v3_renamed\"``.
    original_formal_statement
        The seed ``theorem … := by`` block (for manual review and equivalence checks).
    certificate
        Optional; set after verification to the full certificate Lean text (or ``null``).
    """
    rec: dict[str, Any] = {
        "name": name,
        "original_name": original_name,
        "split": split,
        "dataset_name": dataset_name,
        "informal_prefix": informal_prefix,
        "formal_statement": normalize_formal_statement(formal_statement),
        "header": sanitize_header_remove_aesop(header),
        "variant": variant,
        "original_formal_statement": normalize_formal_statement(original_formal_statement),
        "variable_map": variable_map,
        "hypothesis_map": hypothesis_map,
    }
    if goal is not None:
        rec["goal"] = goal
    if certificate is not None:
        rec["certificate"] = certificate
    return rec


def empty_maps_to_none(m: dict[str, str] | None) -> dict[str, str] | None:
    """Normalize empty dicts to None for storage in JSONL."""
    if m is None:
        return None
    return None if len(m) == 0 else dict(m)

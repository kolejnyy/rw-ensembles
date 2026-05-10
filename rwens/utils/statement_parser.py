"""
Parse Lean theorem statement heads (through ``:= by``) and classify binders.

Classification of **hypotheses** vs **variable / data** binders is **heuristic**:
Lean does not mark them differently in syntax. We treat a binder as a *variable
definition* when its type looks like a non-``Prop`` type (e.g. ``ℝ``, ``ℕ``,
unary function types ``A → B``, and curried **data** function types
``X → Y → Z`` where each component is an allowed atomic type); binders whose
types look like propositions (inequalities, equalities, ``∀ …``, etc.) are
*hypotheses*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

_THEOREM_THROUGH_BY_RE = re.compile(r":=\s*by\s*$", re.IGNORECASE | re.MULTILINE)

# Leading token(s) that usually denote data / Type, not a proposition to prove.
_DATA_TYPE_HEAD = re.compile(
    r"^(ℝ|ℕ|ℤ|ℂ|ℚ|NNReal|Bool|Nat|Int|Real|Rat|Complex|Prop|Type|Sort|True|False)\b"
)
_FIN_LIST_SET = re.compile(r"^(Fin|List|Set|Finset|Array|Option|Subtype|Equiv)\b")
_NUMERIC_BASE_TYPES = {"ℂ", "ℝ", "ℚ", "NNReal", "ℤ", "ℕ+", "ℕ"}


class BinderRole(str, Enum):
    """Heuristic role of a parenthesized binder."""

    VARIABLE = "variable"
    """Parameter / data (e.g. ``(x : ℝ)``, ``(n : ℕ)``)."""

    HYPOTHESIS = "hypothesis"
    """Assumed proposition (e.g. ``(h : x > 0)``)."""


@dataclass(frozen=True)
class ParsedTheorem:
    """A single ``theorem ... := by`` head (no proof body)."""

    name: str
    binder_segment: str
    conclusion: str
    raw_through_by: str


@dataclass(frozen=True)
class ParsedBinder:
    """One parenthesized binder in a theorem header."""

    name: str
    type_str: str
    role: BinderRole
    raw_chunk: str


def _strip_theorem_through_by(theorem_through_by: str) -> str:
    t = theorem_through_by.strip()
    if not _THEOREM_THROUGH_BY_RE.search(t):
        raise ValueError(
            "Expected theorem string ending with ':= by' (possibly on last line)."
        )
    return _THEOREM_THROUGH_BY_RE.sub("", t).strip()


def first_top_level_colon_index(s: str) -> Optional[int]:
    """Index of the first ``:`` at paren depth 0 (splits binder name from type inside ``(...)``)."""
    depth = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
    return None


def parse_theorem_through_by(theorem_through_by: str) -> ParsedTheorem:
    """
    Parse ``theorem Name ... : conclusion := by`` into name, binders, conclusion.

    Uses paren/bracket/brace depth so ``:`` inside binders (e.g. ``(x : ℝ)``)
    does not end the binder segment.
    """
    head = _strip_theorem_through_by(theorem_through_by)
    m = re.match(r"^theorem\s+(\S+)\s*(.*)$", head, re.DOTALL)
    if not m:
        raise ValueError(f"Could not parse theorem head: {head[:80]!r}...")
    name, rest = m.group(1), m.group(2).strip()
    idx = first_top_level_colon_index(rest)
    if idx is None:
        raise ValueError("Could not find top-level ':' separating binders from conclusion.")
    binder_segment = rest[:idx].strip()
    conclusion = rest[idx + 1 :].strip()
    raw = theorem_through_by if theorem_through_by.endswith("\n") else theorem_through_by + "\n"
    return ParsedTheorem(
        name=name,
        binder_segment=binder_segment,
        conclusion=conclusion,
        raw_through_by=raw,
    )


def split_binder_chunks(binder_segment: str) -> list[str]:
    """Split ``binder_segment`` into top-level ``(...)`` chunks."""
    segment = binder_segment.strip()
    if not segment:
        return []
    chunks: list[str] = []
    i = 0
    n = len(segment)
    while i < n:
        while i < n and segment[i].isspace():
            i += 1
        if i >= n:
            break
        if segment[i] != "(":
            raise ValueError(
                f"Expected '(' at start of binder in {segment[i : i + 40]!r}..."
            )
        depth = 0
        start = i
        for j in range(i, n):
            c = segment[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    chunks.append(segment[start : j + 1])
                    i = j + 1
                    break
        else:
            raise ValueError("Unbalanced parentheses in binder segment.")
    return chunks


def binder_name(chunk: str) -> str:
    """Binder name(s) in a ``(...)`` chunk (before the first top-level ``:``)."""
    inner = chunk.strip()
    if not (inner.startswith("(") and inner.endswith(")")):
        raise ValueError("Binder chunk must be parenthesized.")
    inner = inner[1:-1].strip()
    idx = first_top_level_colon_index(inner)
    if idx is None:
        raise ValueError(f"Could not find ':' in binder chunk {chunk!r}")
    return inner[:idx].strip()


def binder_type(chunk: str) -> str:
    """Type part of a ``(name : type)`` chunk (depth-aware; first ``:`` separates name/type)."""
    inner = chunk.strip()
    if not (inner.startswith("(") and inner.endswith(")")):
        raise ValueError("Binder chunk must be parenthesized.")
    inner = inner[1:-1].strip()
    idx = first_top_level_colon_index(inner)
    if idx is None:
        raise ValueError(f"Could not find ':' in binder chunk {chunk!r}")
    return inner[idx + 1 :].strip()


def _looks_like_proposition(type_str: str) -> bool:
    """
    True if ``type_str`` likely denotes a ``Prop`` (heuristic).

    Uses character checks at paren depth 0 for typical prop syntax.
    """
    if _is_numeric_arrow_function_type(type_str):
        return False
    if _is_numeric_curried_function_type(type_str):
        return False

    depth = 0
    for ch in type_str:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            if ch in "=≠<>≤≥∧∨→↔∀∃":
                return True
    return False


def _strip_outer_parens(s: str) -> str:
    t = s.strip()
    while t.startswith("(") and t.endswith(")"):
        depth = 0
        balanced = True
        for i, ch in enumerate(t):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if depth == 0 and i < len(t) - 1:
                balanced = False
                break
        if not balanced:
            break
        t = t[1:-1].strip()
    return t


def _is_numeric_atom_type(s: str) -> bool:
    return _strip_outer_parens(s) in _NUMERIC_BASE_TYPES


def _top_level_single_arrow_parts(type_str: str) -> tuple[str, str] | None:
    """
    Return ``(lhs, rhs)`` splitting at the **first** top-level arrow (``->`` or
    ``→``), else ``None``.

    Further arrows may appear in ``rhs`` (curried types). Older code rejected
    multiple top-level arrows, which broke recognition of e.g.
    ``ℕ → NNReal → ℝ``.
    """
    s = type_str.strip()
    depth = 0
    arrow_idx: Optional[int] = None
    arrow_width = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
            depth -= 1
            i += 1
            continue
        if depth == 0:
            if ch == "→":
                arrow_idx = i
                arrow_width = 1
                break
            if ch == "-" and i + 1 < len(s) and s[i + 1] == ">":
                arrow_idx = i
                arrow_width = 2
                break
        i += 1
    if arrow_idx is None:
        return None
    lhs = s[:arrow_idx].strip()
    rhs = s[arrow_idx + arrow_width :].strip()
    if not lhs or not rhs:
        return None
    return lhs, rhs


def _is_numeric_arrow_function_type(type_str: str) -> bool:
    """
    True for simple function types ``A -> B`` / ``A → B`` with A,B in:
    ``ℂ, ℝ, NNReal, ℤ, ℕ+, ℕ``.

    Superseded for classification by :func:`_is_numeric_curried_function_type`
    (binary case); kept for callers that only need the non-recursive check.
    """
    parts = _top_level_single_arrow_parts(type_str)
    if parts is None:
        return False
    lhs, rhs = parts
    return _is_numeric_atom_type(lhs) and _is_numeric_atom_type(rhs)


def _is_numeric_curried_function_type(type_str: str) -> bool:
    """
    True for curried function types ``X₁ → X₂ → … → Xₙ`` (and ``->``) where each
    ``Xᵢ`` is an allowed atomic data type (``_NUMERIC_BASE_TYPES``).

    This classifies e.g. ``ℕ → NNReal → ℝ`` and ``ℝ → ℝ → ℝ`` as *data*
    (variable binders), not hypotheses. Implication or dependent types that
    use ``→`` are still detected elsewhere via ``∀``, ``=``, etc.
    """
    t = _strip_outer_parens(type_str.strip())
    if _is_numeric_atom_type(t):
        return True
    parts = _top_level_single_arrow_parts(t)
    if parts is None:
        return False
    lhs, rhs = parts
    if not _is_numeric_atom_type(lhs):
        return False
    return _is_numeric_curried_function_type(rhs)


def _looks_like_data_type(type_str: str) -> bool:
    """True if type looks like a data / Type family head (heuristic)."""
    t = type_str.strip()
    if not t:
        return False
    if _looks_like_proposition(t):
        return False
    if _DATA_TYPE_HEAD.match(t):
        return True
    if _FIN_LIST_SET.match(t):
        return True
    # Function types to data: `ℝ → ℝ`, `ℕ → ℤ`
    if "→" in t or "->" in t:
        if not _looks_like_proposition(t):
            return True
    return False


def classify_binder_type(type_str: str) -> BinderRole:
    """
    Classify binder as VARIABLE vs HYPOTHESIS from its type string alone.

    Unknown single identifiers (e.g. ``P``) default to **hypothesis** (often ``P : Prop``).
    """
    t = type_str.strip()
    if _looks_like_proposition(t):
        return BinderRole.HYPOTHESIS
    if _looks_like_data_type(t):
        return BinderRole.VARIABLE
    # Single simple name: likely Prop
    if re.match(r"^[\w.']+$", t):
        return BinderRole.HYPOTHESIS
    return BinderRole.HYPOTHESIS


class StatementParser:
    """
    Extract structured information from a Lean theorem statement (header only).
    """

    @classmethod
    def parse_binders(cls, theorem_through_by: str) -> list[ParsedBinder]:
        """
        All parenthesized binders in order, each with a heuristic :attr:`BinderRole`.
        """
        parsed = parse_theorem_through_by(theorem_through_by)
        out: list[ParsedBinder] = []
        for chunk in split_binder_chunks(parsed.binder_segment):
            ty = binder_type(chunk)
            role = classify_binder_type(ty)
            out.append(
                ParsedBinder(
                    name=binder_name(chunk),
                    type_str=ty,
                    role=role,
                    raw_chunk=chunk,
                )
            )
        return out

    @classmethod
    def hypotheses(cls, theorem_through_by: str) -> list[ParsedBinder]:
        """
        Binders classified as hypotheses (excludes variable / data binders).

        Parameters
        ----------
        theorem_through_by
            Full theorem line(s) through ``:= by``, as in miniF2F ``formal_statement``.
        """
        return [b for b in cls.parse_binders(theorem_through_by) if b.role == BinderRole.HYPOTHESIS]

    @classmethod
    def variable_binders(cls, theorem_through_by: str) -> list[ParsedBinder]:
        """Binders classified as variable / data parameters."""
        return [b for b in cls.parse_binders(theorem_through_by) if b.role == BinderRole.VARIABLE]


def all_binder_name_tokens(theorem_through_by: str) -> list[str]:
    """
    All binder names in theorem order; splits multi-name chunks (e.g. ``x p`` in ``(x p : ℝ)``).
    """
    toks: list[str] = []
    for b in StatementParser.parse_binders(theorem_through_by):
        toks.extend(b.name.split())
    return toks


def variable_binder_name_tokens(theorem_through_by: str) -> list[str]:
    """Names from variable/data binders only, in order (split like :func:`all_binder_name_tokens`)."""
    toks: list[str] = []
    for b in StatementParser.variable_binders(theorem_through_by):
        toks.extend(b.name.split())
    return toks


def normalize_binder_segment(s: str) -> str:
    """Whitespace-normalize binder segment for equality comparisons."""
    return " ".join(s.split())

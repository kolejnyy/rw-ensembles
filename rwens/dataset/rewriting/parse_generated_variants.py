"""
Parse OpenAI / model output into individual Lean theorem variant blocks.

Handles fenced ```lean / ```lean4 segments and multiple theorems without blank lines between them.
"""

from __future__ import annotations

import re
from typing import Pattern

# Fenced code blocks (lean / lean4 / generic)
_LEAN_FENCE = re.compile(r"```(?:lean4?)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Start of a theorem declaration at beginning of line (optional indent)
_THEOREM_HEAD: Pattern[str] = re.compile(r"(?m)^\s*theorem\s+\S+")


def preprocess_model_output(raw: str) -> str:
    """Strip prose; prefer Lean inside markdown fences if present."""
    if not raw or not raw.strip():
        return ""
    chunks = [m.group(1).strip() for m in _LEAN_FENCE.finditer(raw)]
    if chunks:
        return "\n\n".join(c for c in chunks if c)
    return raw.strip()


def theorem_declared_name(block: str) -> str | None:
    first = (block.splitlines() or [""])[0].strip()
    m = re.match(r"^theorem\s+(\S+)", first)
    return m.group(1) if m else None


def extract_theorem_blocks(raw_text: str) -> list[str]:
    """
    Return one string per complete theorem (through `:= by`), in order.

    Splits on every line that begins a new `theorem` declaration, so multiple
    theorems in one fence without blank lines are separated correctly.
    """
    text = preprocess_model_output(raw_text)
    if not text:
        return []

    positions = [m.start() for m in _THEOREM_HEAD.finditer(text)]
    if not positions:
        return []

    blocks: list[str] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk = text[start:end].strip()
        if ":= by" not in chunk:
            continue
        blocks.append(chunk)
    return blocks


def filter_variant_blocks(blocks: list[str], problem_name: str) -> list[str]:
    """Drop repeats of the original theorem name (model often echoes the base stmt)."""
    out: list[str] = []
    for b in blocks:
        name = theorem_declared_name(b)
        if name is not None and name == problem_name:
            continue
        out.append(b)
    return out


def variant_tag_from_declared_name(decl_name: str, problem_name: str) -> str:
    """Derive a short variant tag (e.g. ``v2``, ``v3_renamed``) from a declared theorem name."""
    prefix = f"{problem_name}_v"
    if decl_name.startswith(prefix):
        rest = decl_name[len(prefix):]
        if rest.isdigit():
            return f"v{rest}"
    if decl_name.startswith(problem_name + "_"):
        tail = decl_name[len(problem_name) + 1:]
        if tail:
            return tail
    return "v2"

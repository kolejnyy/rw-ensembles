"""
Prompt formatter: variable and hypothesis renamings for an existing variant theorem.

Asks the model for a single JSON object: original theorem text, renamed theorem,
and explicit maps for variables and hypotheses.
"""

from __future__ import annotations

import json
import re
from typing import Any

from invpro.prompt.base import PromptFormatter

_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class VariantRenamingPromptFormatter(PromptFormatter):
    """
    Produce a renaming of binder names in a Lean ``theorem ... := by`` head.

    Output must be JSON with keys ``original``, ``renamed``, ``variable_map``,
    ``hypothesis_map`` (see :func:`parse_variant_renaming_response`).
    """

    def format(
        self,
        formal_statement: str,
        theorem_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        label = theorem_name or "<variant_theorem>"
        stmt = formal_statement.strip()
        return (
            "You are an expert Lean 4 assistant.\n\n"
            "Task:\n"
            "Given a single Lean 4 theorem declaration (through `:= by`), produce a "
            "**renaming** of its variables and hypothesis names only. The mathematical "
            "meaning must stay the same (α-equivalence): only binder identifiers change.\n\n"
            "Rules:\n"
            "1) Keep the same theorem name as in the input.\n"
            "2) Do not change the conclusion or hypothesis **types** except where a name "
            "appears as a binder (you may rename bound variables consistently).\n"
            "3) `variable_map` maps old variable names → new names (e.g. `x` → `u`).\n"
            "4) `hypothesis_map` maps old hypothesis names → new names (e.g. `h₀` → `h1` or `h0`). "
            "Use the exact Unicode subscript names from the input if present (e.g. `h₀`).\n"
            "5) If you rename nothing in a category, use an empty object `{}` for that map.\n"
            "6) `original` must be exactly the input theorem string (after trimming).\n"
            "7) `renamed` must be the full theorem line(s) ending with `:= by` only — no `sorry`, "
            "tactics, or proof body after `by`.\n\n"
            "Output format:\n"
            "- Output **only** one JSON object, no markdown fences, no commentary.\n"
            "- Schema:\n"
            '  {"original": "...", "renamed": "...", "variable_map": {...}, "hypothesis_map": {...}}\n\n'
            f"-- theorem id: {label}\n"
            "Input theorem:\n```lean4\n"
            f"{stmt}\n"
            "```\n"
        )

    def format_answer(self, answer: str) -> str:
        return answer.strip()

    def extract_answer(self, response: str, **kwargs: Any) -> str:
        obj = parse_variant_renaming_response(response)
        return json.dumps(obj, ensure_ascii=False)


def parse_variant_renaming_response(raw: str) -> dict[str, Any]:
    """
    Parse model output into a dict with keys ``original``, ``renamed``,
    ``variable_map``, ``hypothesis_map``.

    Accepts optional ```json ... ``` fences. Validates required keys and that maps
    are JSON objects (dicts with string keys and string values).
    """
    text = raw.strip() if raw else ""
    if not text:
        raise ValueError("Empty renaming response.")
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Renaming JSON must be an object.")
    for key in ("original", "renamed", "variable_map", "hypothesis_map"):
        if key not in obj:
            raise ValueError(f"Renaming JSON missing key: {key!r}")
    if not isinstance(obj["original"], str) or not isinstance(obj["renamed"], str):
        raise ValueError("original and renamed must be strings.")
    vm = obj["variable_map"]
    hm = obj["hypothesis_map"]
    if not isinstance(vm, dict) or not isinstance(hm, dict):
        raise ValueError("variable_map and hypothesis_map must be JSON objects.")
    for label, m in ("variable_map", vm), ("hypothesis_map", hm):
        for k, v in m.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(f"{label} keys and values must be strings.")
    return {
        "original": obj["original"].strip(),
        "renamed": obj["renamed"].strip(),
        "variable_map": dict(vm),
        "hypothesis_map": dict(hm),
    }

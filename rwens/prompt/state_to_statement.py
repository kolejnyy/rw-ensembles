"""
Prompt formatter for converting Lean proof states to theorem statements.
"""

from __future__ import annotations

import re

from rwens.prompt.base import PromptFormatter

_LEAN_FENCE_RE = re.compile(r"```lean4\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class StateToStatementPromptFormatter(PromptFormatter):
    """Build prompts for state -> statement conversion."""

    def format(
        self,
        original_statement: str,
        original_state: str,
        augmented_state: str,
        **kwargs,
    ) -> str:
        return (
            "You are an expert Lean4 formalization assistant.\n\n"
            "Task:\n"
            "Given an original Lean4 theorem statement for reference, and a tactic state created after "
            "applying some augmentations to the original theorem, create a theorem statement corresponding "
            "to the provided augmented state.\n\n"
            "Requirements:\n"
            "1) Keep theorem name and binder structure aligned with the original statement unless the augmented state requires changes.\n"
            "2) Preserve hypothesis order from the augmented state.\n"
            "3) Preserve the augmented state's mathematics exactly (do not simplify away changes).\n"
            "4) Replace shorthand notation with explicit Lean forms when appropriate (e.g. `√2` -> `Real.sqrt 2`, `π` -> `Real.pi`).\n"
            "5) Output must contain exactly one block in Lean markdown fence format:\n"
            "   ```lean4\n"
            "   theorem ... := by\n"
            "   ```\n"
            "6) Do not output any explanation or extra text outside that block.\n"
            "7) Do not include a proof body, tactics, or sorry; stop exactly at `:= by`.\n\n"
            "Original statement:\n"
            f"{original_statement.strip()}\n\n"
            "Original state:\n"
            f"{original_state.strip()}\n\n"
            "Augmented state:\n"
            f"{augmented_state.strip()}\n"
        )

    def format_answer(self, answer: str) -> str:
        return answer.strip()

    def extract_answer(self, response: str, **kwargs) -> str:
        text = (response or "").strip()
        m = _LEAN_FENCE_RE.search(text)
        if not m:
            raise ValueError("Response must contain a ```lean4 ... ``` block.")
        return m.group(1).strip()

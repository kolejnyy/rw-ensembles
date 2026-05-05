"""
Prompt formatter for Kimina-Prover-Distill-8B (full-proof generation).
See: https://huggingface.co/AI-MO/Kimina-Prover-Distill-8B
"""

from __future__ import annotations

import re

from invpro.prompt.base import PromptFormatter


class KiminaProverPromptFormatter(PromptFormatter):
    """Format prompts for Kimina-Prover-Distill-8B using model-card style user instruction."""

    PROMPT = """Think about and solve the following problem step by step in Lean 4.
# Problem:{problem}
# Formal statement:
```lean4
{formal_statement}
```
"""

    USER_PREFIX_BEFORE_THEOREM = (
        "Think about and solve the following problem step by step in Lean 4.\n"
        "# Problem:"
    )

    _LEAN4_BLOCK = re.compile(r"```\s*lean4\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    _BY = re.compile(r":=\s*by\b", re.MULTILINE | re.DOTALL)
    _BY_OR_STMT_END = re.compile(r":=\s*(?:by\b|\n)", re.MULTILINE | re.DOTALL)
    _DOCSTRING = re.compile(r"/--\s*(.*?)\s*-/", re.DOTALL)

    def _extract_problem_text(self, formal_statement: str) -> str:
        """Extract natural-language problem from Lean docstring; fallback to placeholder."""
        m = self._DOCSTRING.search(formal_statement)
        if not m:
            return "Formal theorem proving task in Lean 4."
        text = re.sub(r"\s+", " ", m.group(1)).strip()
        return text or "Formal theorem proving task in Lean 4."

    def format(self, formal_statement: str, **kwargs) -> str:
        theorem = formal_statement.strip()
        problem = kwargs.get("problem") or self._extract_problem_text(theorem)
        return self.PROMPT.format(problem=problem, formal_statement=theorem)

    def get_user_prefix_before_theorem(self) -> str:
        return self.USER_PREFIX_BEFORE_THEOREM

    def format_answer(self, answer: str) -> str:
        return f"```lean4\n{answer.strip()}\n```"

    def extract_answer(self, response: str, **kwargs) -> str:
        """
        Extract proof body from model response.
        Prefer largest Lean4 fenced block if present; otherwise fall back to ':= by' parsing.
        """
        text = response
        blocks = self._LEAN4_BLOCK.findall(response)
        if blocks:
            text = max(blocks, key=lambda x: len(x.strip()))
        m = self._BY.search(text)
        if m:
            return text[m.end() :].rstrip("\n\r")
        m2 = self._BY_OR_STMT_END.search(text)
        if m2:
            return text[m2.end() :].rstrip("\n\r")
        return response.rstrip("\n\r")

"""
Prompt formatter for DeepSeek Prover (full-proof generation).
See: https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B
"""

import re
from invpro.prompt.base import PromptFormatter


class DeepSeekProverPromptFormatter(PromptFormatter):
    """Format prompts for DeepSeek-Prover-V2: complete Lean 4 code given a formal statement."""

    PROMPT = """Complete the following Lean 4 code:

```lean4
{formal_statement}
```
"""

    # User message content before the theorem (for scoring P(theorem tokens | prefix))
    USER_PREFIX_BEFORE_THEOREM = "Complete the following Lean 4 code:\n\n```lean4\n"

    _LEAN4_BLOCK = re.compile(r"```\s*lean4\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    _BY = re.compile(r":=\s*by\b", re.MULTILINE | re.DOTALL)
    # ProofNet-style: statement ends with ":=" then newline, no " by"
    _BY_OR_STMT_END = re.compile(r":=\s*(?:by\b|\n)", re.MULTILINE | re.DOTALL)

    def format(self, formal_statement: str, **kwargs) -> str:
        return self.PROMPT.format(formal_statement=formal_statement.strip())

    def get_user_prefix_before_theorem(self) -> str:
        """User message content that appears before the theorem. Used to score P(theorem | prefix)."""
        return self.USER_PREFIX_BEFORE_THEOREM

    def format_answer(self, answer: str) -> str:
        return f"```lean4\n{answer.strip()}\n```"

    def extract_answer(self, response: str, **kwargs) -> str:
        """Find the theorem in the response and return everything after it (the proof body). No indent changes.
        Supports both ':= by' (minif2f-style) and ':= ' then newline (ProofNet-style) as end of statement.
        """
        text = response
        blocks = self._LEAN4_BLOCK.findall(response)
        if blocks:
            text = max(blocks, key=lambda x: len(x.strip()))
        m = self._BY.search(text)
        if m:
            return text[m.end() :].rstrip("\n\r")
        # ProofNet-style: model output may have statement ending with ":=" then newline
        m2 = self._BY_OR_STMT_END.search(text)
        if m2:
            return text[m2.end() :].rstrip("\n\r")
        return response.rstrip("\n\r")


class DeepSeekProverCoTPromptFormatter(PromptFormatter):
    """Format prompts for DeepSeek-Prover-V2 CoT style from the model card."""

    PROMPT = """Complete the following Lean 4 code:

```lean4
{formal_statement}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""

    # User message content before the theorem (for scoring P(theorem tokens | prefix))
    USER_PREFIX_BEFORE_THEOREM = "Complete the following Lean 4 code:\n\n```lean4\n"

    _LEAN4_BLOCK = re.compile(r"```\s*lean4\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    _BY = re.compile(r":=\s*by\b", re.MULTILINE | re.DOTALL)
    # ProofNet-style: statement ends with ":=" then newline, no " by"
    _BY_OR_STMT_END = re.compile(r":=\s*(?:by\b|\n)", re.MULTILINE | re.DOTALL)

    def format(self, formal_statement: str, **kwargs) -> str:
        return self.PROMPT.format(formal_statement=formal_statement.strip())

    def get_user_prefix_before_theorem(self) -> str:
        """User message content that appears before the theorem. Used to score P(theorem | prefix)."""
        return self.USER_PREFIX_BEFORE_THEOREM

    def format_answer(self, answer: str) -> str:
        return f"```lean4\n{answer.strip()}\n```"

    def extract_answer(self, response: str, **kwargs) -> str:
        """Find the theorem in the response and return everything after it (the proof body). No indent changes.
        Supports both ':= by' (minif2f-style) and ':= ' then newline (ProofNet-style) as end of statement.
        """
        text = response
        blocks = self._LEAN4_BLOCK.findall(response)
        if blocks:
            text = max(blocks, key=lambda x: len(x.strip()))
        m = self._BY.search(text)
        if m:
            return text[m.end() :].rstrip("\n\r")
        # ProofNet-style: model output may have statement ending with ":=" then newline
        m2 = self._BY_OR_STMT_END.search(text)
        if m2:
            return text[m2.end() :].rstrip("\n\r")
        return response.rstrip("\n\r")

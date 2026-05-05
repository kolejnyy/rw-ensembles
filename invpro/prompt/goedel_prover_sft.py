"""
Prompt formatter for Goedel-Prover-SFT (full-proof generation).
Prompt shape follows Goedel's inference script and skips informal text.
"""

import re

from invpro.prompt.deepseek_prover import DeepSeekProverPromptFormatter


class GoedelProverSFTPromptFormatter(DeepSeekProverPromptFormatter):
    """
    Format prompts for Goedel-Prover-SFT.

    Matches the structure from Goedel-LM's inference script, but intentionally
    omits informal_prefix / informal statement blocks.
    """

    # Match Goedel-LM inference script: prompt ends right after formal_statement
    # (no closing ``` in the input prompt).
    PROMPT = """Complete the following Lean 4 code with explanatory comments preceding each line of code:

```lean4
{formal_statement}"""

    USER_PREFIX_BEFORE_THEOREM = (
        "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n"
    )

    _LEAN_BLOCK_ANY = re.compile(r"```\s*(?:lean4)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

    def extract_answer(self, response: str, **kwargs) -> str:
        """
        Extract proof body from model output.

        Goedel's public script extracts Lean code from fenced blocks in the final
        model text. We do the same (largest ``` block, strip fences).

        We do **not** slice at the first ``:= by`` in the completion: vLLM returns
        a continuation after the prompt (which already ends with ``:= by``), so the
        first ``:= by`` in the decoded text is often an inner ``have … := by``; cutting
        there drops valid lines. DeepSeek-style ``extract_answer`` slicing is wrong
        for this formatter.
        """
        text = response
        blocks = self._LEAN_BLOCK_ANY.findall(response)
        if blocks:
            text = max(blocks, key=lambda x: len(x.strip()))
        # Some generations emit only a trailing fence in the completion.
        text = text.replace("```lean4", "").replace("```", "").rstrip("\n\r")

        return text

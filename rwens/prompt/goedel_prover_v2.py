"""
Prompt formatter for Goedel-Prover-V2 (full-proof generation with proof plan).
See: https://huggingface.co/Goedel-LM/Goedel-Prover-V2-8B
"""

from rwens.prompt.deepseek_prover import DeepSeekProverCoTPromptFormatter


class GoedelProverV2PromptFormatter(DeepSeekProverCoTPromptFormatter):
    """
    User message matches the Goedel-Prover-V2 model card: Lean 4 block plus instructions
    to produce a detailed proof plan before the formal proof (same structure as DeepSeek CoT).
    """

    pass

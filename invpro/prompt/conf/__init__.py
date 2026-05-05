"""Configuration system for prompt formatters."""

from invpro.prompt.conf.types import (
    BaseStateToTacPromptFormatterConfig,
    DeepSeekProverCoTPromptFormatterConfig,
    DeepSeekProverPromptFormatterConfig,
    KiminaProverPromptFormatterConfig,
    GoedelProverSFTPromptFormatterConfig,
    GoedelProverV2PromptFormatterConfig,
    RewriteAugmentationPromptFormatterConfig,
    StateToStatementPromptFormatterConfig,
    VariantRenamingPromptFormatterConfig,
    PromptFormatterConfig,
)
from invpro.prompt.conf.factory import dict_to_config, prompt_formatter_from_config

__all__ = [
    "PromptFormatterConfig",
    "BaseStateToTacPromptFormatterConfig",
    "DeepSeekProverCoTPromptFormatterConfig",
    "DeepSeekProverPromptFormatterConfig",
    "KiminaProverPromptFormatterConfig",
    "GoedelProverSFTPromptFormatterConfig",
    "GoedelProverV2PromptFormatterConfig",
    "RewriteAugmentationPromptFormatterConfig",
    "StateToStatementPromptFormatterConfig",
    "VariantRenamingPromptFormatterConfig",
    "dict_to_config",
    "prompt_formatter_from_config",
]

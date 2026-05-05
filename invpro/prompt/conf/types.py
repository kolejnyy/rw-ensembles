"""
Configuration classes for prompt formatters.

This module defines dataclasses that represent the configuration for different
types of prompt formatters.
"""

from dataclasses import dataclass
from typing import Any, Union


@dataclass
class BaseStateToTacPromptFormatterConfig:
    """Configuration for BaseStateToTacPromptFormatter.
    
    This formatter currently has no parameters, so this is an empty config class.
    """
    def __init__(self, **kwargs: Any) -> None:
        pass


@dataclass
class DeepSeekProverPromptFormatterConfig:
    """Configuration for DeepSeekProverPromptFormatter (full-proof prompt for DeepSeek-Prover-V2)."""
    def __init__(self, **kwargs: Any) -> None:
        pass


@dataclass
class DeepSeekProverCoTPromptFormatterConfig:
    """Configuration for DeepSeekProverCoTPromptFormatter (CoT full-proof prompt for DeepSeek-Prover-V2)."""
    def __init__(self, **kwargs: Any) -> None:
        pass


@dataclass
class KiminaProverPromptFormatterConfig:
    """Configuration for KiminaProverPromptFormatter."""
    def __init__(self, **kwargs: Any) -> None:
        pass


@dataclass
class GoedelProverSFTPromptFormatterConfig:
    """Configuration for GoedelProverSFTPromptFormatter."""
    def __init__(self, **kwargs: Any) -> None:
        pass


@dataclass
class GoedelProverV2PromptFormatterConfig:
    """Configuration for GoedelProverV2PromptFormatter (Goedel-Prover-V2 model card prompt)."""
    def __init__(self, **kwargs: Any) -> None:
        pass


@dataclass
class RewriteAugmentationPromptFormatterConfig:
    """Configuration for RewriteAugmentationPromptFormatter."""
    def __init__(self, **kwargs: Any) -> None:
        pass


@dataclass
class VariantRenamingPromptFormatterConfig:
    """Configuration for VariantRenamingPromptFormatter."""
    def __init__(self, **kwargs: Any) -> None:
        pass


@dataclass
class StateToStatementPromptFormatterConfig:
    """Configuration for StateToStatementPromptFormatter."""
    def __init__(self, **kwargs: Any) -> None:
        pass


# Create a single type for all configuration classes
PromptFormatterConfig = Union[
    BaseStateToTacPromptFormatterConfig,
    DeepSeekProverPromptFormatterConfig,
    DeepSeekProverCoTPromptFormatterConfig,
    KiminaProverPromptFormatterConfig,
    GoedelProverSFTPromptFormatterConfig,
    GoedelProverV2PromptFormatterConfig,
    RewriteAugmentationPromptFormatterConfig,
    VariantRenamingPromptFormatterConfig,
    StateToStatementPromptFormatterConfig,
]

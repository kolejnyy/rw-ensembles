"""
Factory functions for creating prompt formatters from configuration.

This module provides functions to:
1. Convert a dictionary (from YAML) to a configuration object
2. Create a prompt formatter instance from a configuration object
"""

from typing import Dict, Any, Union

from rwens.prompt.conf.types import (
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
from rwens.prompt.tactic_prediction import BaseStateToTacPromptFormatter
from rwens.prompt.deepseek_prover import DeepSeekProverPromptFormatter
from rwens.prompt.deepseek_prover import DeepSeekProverCoTPromptFormatter
from rwens.prompt.kimina_prover import KiminaProverPromptFormatter
from rwens.prompt.goedel_prover_sft import GoedelProverSFTPromptFormatter
from rwens.prompt.goedel_prover_v2 import GoedelProverV2PromptFormatter
from rwens.prompt.rewrite_augmentation import RewriteAugmentationPromptFormatter
from rwens.prompt.state_to_statement import StateToStatementPromptFormatter
from rwens.prompt.variant_renaming import VariantRenamingPromptFormatter
from rwens.prompt.base import PromptFormatter


# Mapping from class name to config class
_CONFIG_CLASSES = {
    "BaseStateToTacPromptFormatter": BaseStateToTacPromptFormatterConfig,
    "DeepSeekProverPromptFormatter": DeepSeekProverPromptFormatterConfig,
    "DeepSeekProverCoTPromptFormatter": DeepSeekProverCoTPromptFormatterConfig,
    "KiminaProverPromptFormatter": KiminaProverPromptFormatterConfig,
    "GoedelProverSFTPromptFormatter": GoedelProverSFTPromptFormatterConfig,
    "GoedelProverV2PromptFormatter": GoedelProverV2PromptFormatterConfig,
    "RewriteAugmentationPromptFormatter": RewriteAugmentationPromptFormatterConfig,
    "StateToStatementPromptFormatter": StateToStatementPromptFormatterConfig,
    "VariantRenamingPromptFormatter": VariantRenamingPromptFormatterConfig,
}

# Mapping from class name to formatter class
_FORMATTER_CLASSES = {
    "BaseStateToTacPromptFormatter": BaseStateToTacPromptFormatter,
    "DeepSeekProverPromptFormatter": DeepSeekProverPromptFormatter,
    "DeepSeekProverCoTPromptFormatter": DeepSeekProverCoTPromptFormatter,
    "KiminaProverPromptFormatter": KiminaProverPromptFormatter,
    "GoedelProverSFTPromptFormatter": GoedelProverSFTPromptFormatter,
    "GoedelProverV2PromptFormatter": GoedelProverV2PromptFormatter,
    "RewriteAugmentationPromptFormatter": RewriteAugmentationPromptFormatter,
    "StateToStatementPromptFormatter": StateToStatementPromptFormatter,
    "VariantRenamingPromptFormatter": VariantRenamingPromptFormatter,
}

def dict_to_config(config_dict: Dict[str, Any]) -> PromptFormatterConfig:
    """
    Convert a dictionary (from YAML) to a configuration object.
    
    Args:
        config_dict: Dictionary with 'class' and 'parameters' keys
            Example: {
                "class": "BaseStateToTacPromptFormatter",
                "parameters": {}
            }
    
    Returns:
        Configuration object instance
    
    Raises:
        ValueError: If the class name is not recognized
    """
    class_name = config_dict.get("class")
    if class_name is None:
        raise ValueError("Config dictionary must have a 'class' key")
    
    if class_name == "BaseStateToTacPromptFormatter":
        return BaseStateToTacPromptFormatterConfig()
    if class_name == "DeepSeekProverPromptFormatter":
        return DeepSeekProverPromptFormatterConfig(**config_dict.get("parameters", {}))
    if class_name == "DeepSeekProverCoTPromptFormatter":
        return DeepSeekProverCoTPromptFormatterConfig(**config_dict.get("parameters", {}))
    if class_name == "KiminaProverPromptFormatter":
        return KiminaProverPromptFormatterConfig(**config_dict.get("parameters", {}))
    if class_name == "GoedelProverSFTPromptFormatter":
        return GoedelProverSFTPromptFormatterConfig(**config_dict.get("parameters", {}))
    if class_name == "GoedelProverV2PromptFormatter":
        return GoedelProverV2PromptFormatterConfig(**config_dict.get("parameters", {}))
    if class_name == "RewriteAugmentationPromptFormatter":
        return RewriteAugmentationPromptFormatterConfig(**config_dict.get("parameters", {}))
    if class_name == "StateToStatementPromptFormatter":
        return StateToStatementPromptFormatterConfig(**config_dict.get("parameters", {}))
    if class_name == "VariantRenamingPromptFormatter":
        return VariantRenamingPromptFormatterConfig(**config_dict.get("parameters", {}))
    else:
        raise ValueError(
            f"Unknown prompt formatter class: {class_name}. "
            f"Supported classes: {list(_CONFIG_CLASSES.keys())}"
        )


def prompt_formatter_from_config(config: PromptFormatterConfig) -> PromptFormatter:
    """
    Create a prompt formatter instance from a configuration object.
    
    Args:
        config: Configuration object (e.g., BaseStateToTacPromptFormatterConfig)
    
    Returns:
        Prompt formatter instance
    
    Raises:
        ValueError: If the config type is not recognized
    """
    # Find the config class name
    config_class_name = None
    for name, cls in _CONFIG_CLASSES.items():
        if isinstance(config, cls):
            config_class_name = name
            break

    if config_class_name == "BaseStateToTacPromptFormatter":
        return BaseStateToTacPromptFormatter()
    if config_class_name == "DeepSeekProverPromptFormatter":
        return DeepSeekProverPromptFormatter()
    if config_class_name == "DeepSeekProverCoTPromptFormatter":
        return DeepSeekProverCoTPromptFormatter()
    if config_class_name == "KiminaProverPromptFormatter":
        return KiminaProverPromptFormatter()
    if config_class_name == "GoedelProverSFTPromptFormatter":
        return GoedelProverSFTPromptFormatter()
    if config_class_name == "GoedelProverV2PromptFormatter":
        return GoedelProverV2PromptFormatter()
    if config_class_name == "RewriteAugmentationPromptFormatter":
        return RewriteAugmentationPromptFormatter()
    if config_class_name == "StateToStatementPromptFormatter":
        return StateToStatementPromptFormatter()
    if config_class_name == "VariantRenamingPromptFormatter":
        return VariantRenamingPromptFormatter()
    else:
        raise ValueError(
            f"Unknown config type: {config_class_name}. "
            f"Supported config types: {list(_CONFIG_CLASSES.keys())}"
        )

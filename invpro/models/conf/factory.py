"""
Factory functions for creating formal provers from configuration.

This module provides functions to:
1. Convert a dictionary (from YAML) to a configuration object
2. Create a formal prover instance from a configuration object
"""

from typing import Dict, Any

from invpro.models.conf.types import (
    NaiveStepByStepProverConfig,
    CanonicalProverConfig,
    CanonicalLLMProverConfig,
    EnsembleLLMProverConfig,
    SinglePassProverConfig,
    ProverConfig,
)
from invpro.models.base import FormalProver
from invpro.models.llm.conf import dict_to_config as llm_dict_to_config, llm_from_config
from invpro.canonicalization.conf import (
    dict_to_config as canonicalization_dict_to_config,
    canonicalization_from_config,
)


# Mapping from class name to config class
_CONFIG_CLASSES = {
    "NaiveStepByStepProver": NaiveStepByStepProverConfig,
    "CanonicalProver": CanonicalProverConfig,
    "CanonicalLLMProver": CanonicalLLMProverConfig,
    "EnsembleLLMProver": EnsembleLLMProverConfig,
    "SinglePassProver": SinglePassProverConfig,
}


def dict_to_config(config_dict: Dict[str, Any]) -> ProverConfig:
    """
    Convert a dictionary (from YAML) to a configuration object.
    
    Args:
        config_dict: Dictionary with 'class' and 'parameters' keys
            Example: {
                "class": "NaiveStepByStepProver",
                "parameters": {
                    "llm": {...},
                    "project_root": ".",
                    "max_iterations": 100,
                    "timeout_seconds": 90.0
                }
            }
    
    Returns:
        Configuration object instance
    
    Raises:
        ValueError: If the class name is not recognized
    """
    class_name = config_dict.get("class")
    parameters = config_dict.get("parameters", {})
    
    if class_name is None:
        raise ValueError("Config dictionary must have a 'class' key")
    
    if class_name == "NaiveStepByStepProver":
        llm_config_dict = parameters.pop("llm", {})
        llm_config = llm_dict_to_config(llm_config_dict)
        return NaiveStepByStepProverConfig(llm_config=llm_config, **parameters)
    if class_name == "CanonicalProver":
        llm_config_dict = parameters.pop("llm", {})
        llm_config = llm_dict_to_config(llm_config_dict)
        canonicalization_config_dict = parameters.pop("canonicalization", {})
        canonicalization_config = canonicalization_dict_to_config(
            canonicalization_config_dict
        )
        return CanonicalProverConfig(
            llm_config=llm_config,
            canonicalization_config=canonicalization_config,
            **parameters
        )
    if class_name == "SinglePassProver":
        from invpro.prompt.conf import (
            dict_to_config as prompt_dict_to_config,
            prompt_formatter_from_config,
        )
        llm_config_dict = parameters.pop("llm", {})
        llm_config = llm_dict_to_config(llm_config_dict)
        pf_dict = parameters.pop("prompt_formatter", {})
        pf_config = prompt_dict_to_config(pf_dict)
        return SinglePassProverConfig(
            llm_config=llm_config,
            prompt_formatter_config=pf_config,
            **parameters
        )
    if class_name == "CanonicalLLMProver":
        rewriting_config_dict = parameters.pop("rewriting", {})
        rewriting_config = canonicalization_dict_to_config(rewriting_config_dict)
        single_pass_config_dict = parameters.pop("single_pass_prover", {})
        single_pass_config = dict_to_config(single_pass_config_dict)
        if not isinstance(single_pass_config, SinglePassProverConfig):
            raise ValueError(
                "CanonicalLLMProver requires single_pass_prover to be SinglePassProver config"
            )
        return CanonicalLLMProverConfig(
            rewriting_config=rewriting_config,
            single_pass_prover_config=single_pass_config,
            **parameters
        )
    if class_name == "EnsembleLLMProver":
        rewriting_config_dict = parameters.pop("rewriting", {})
        rewriting_config = canonicalization_dict_to_config(rewriting_config_dict)
        single_pass_config_dict = parameters.pop("single_pass_prover", {})
        single_pass_config = dict_to_config(single_pass_config_dict)
        if not isinstance(single_pass_config, SinglePassProverConfig):
            raise ValueError(
                "EnsembleLLMProver requires single_pass_prover to be SinglePassProver config"
            )
        return EnsembleLLMProverConfig(
            rewriting_config=rewriting_config,
            single_pass_prover_config=single_pass_config,
            **parameters
        )
    else:
        raise ValueError(
            f"Unknown prover class: {class_name}. "
            f"Supported classes: {list(_CONFIG_CLASSES.keys())}"
        )


def prover_from_config(config: ProverConfig) -> FormalProver:
    """
    Create a formal prover instance from a configuration object.
    
    Args:
        config: Configuration object (e.g., NaiveStepByStepProverConfig)
    
    Returns:
        Formal prover instance
    
    Raises:
        ValueError: If the config type is not recognized
    """
    
    from invpro.models.naive_st import NaiveStepByStepProver
    from invpro.models.canonical_st import CanonicalProver
    from invpro.models.single_pass_prover import SinglePassProver
    from invpro.prompt.conf import prompt_formatter_from_config

    # Find the config class name
    config_class_name = None
    for name, cls in _CONFIG_CLASSES.items():
        if isinstance(config, cls):
            config_class_name = name
            break

    if config_class_name == "NaiveStepByStepProver":
        llm = llm_from_config(config.llm_config)
        return NaiveStepByStepProver(
            llm=llm,
            project_root=config.project_root,
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
        )
    if config_class_name == "CanonicalProver":
        llm = llm_from_config(config.llm_config)
        canonicalization_module = canonicalization_from_config(
            config.canonicalization_config, llm=llm
        )
        return CanonicalProver(
            llm=llm,
            canonicalization_module=canonicalization_module,
            full_aug=config.full_aug,
            max_iterations=config.max_iterations,
        )
    if config_class_name == "SinglePassProver":
        llm = llm_from_config(config.llm_config)
        prompt_formatter = prompt_formatter_from_config(config.prompt_formatter_config)
        return SinglePassProver(
            llm=llm,
            prompt_formatter=prompt_formatter,
            project_root=config.project_root,
            timeout_seconds=getattr(config, "timeout_seconds", 120.0),
            initial_imports=getattr(config, "initial_imports", None) or "import Mathlib\n",
        )
    if config_class_name == "CanonicalLLMProver":
        from invpro.models.canonical_llm import CanonicalLLMProver
        single_pass_prover = prover_from_config(config.single_pass_prover_config)
        canonicalization_module = canonicalization_from_config(
            config.rewriting_config,
            llm=getattr(single_pass_prover, "llm", None),
        )
        return CanonicalLLMProver(
            canonicalization_module=canonicalization_module,
            single_pass_prover=single_pass_prover,
        )
    if config_class_name == "EnsembleLLMProver":
        from invpro.models.ensemble_llm import EnsembleLLMProver
        single_pass_prover = prover_from_config(config.single_pass_prover_config)
        canonicalization_module = canonicalization_from_config(
            config.rewriting_config,
            llm=getattr(single_pass_prover, "llm", None),
        )
        return EnsembleLLMProver(
            canonicalization_module=canonicalization_module,
            single_pass_prover=single_pass_prover,
            ensemble_size=getattr(config, "ensemble_size", 4),
            shuffle_seed=getattr(config, "shuffle_seed", None),
            state_to_statement_mode=getattr(config, "state_to_statement_mode", "naive"),
            gpt_state_to_statement_model=getattr(
                config, "gpt_state_to_statement_model", "gpt-5.4-mini"
            ),
            gpt_state_api_key_env=getattr(config, "gpt_state_api_key_env", "OPENAI_API_KEY"),
            gpt_state_max_output_tokens=getattr(
                config, "gpt_state_max_output_tokens", 300
            ),
        )
    else:
        raise ValueError(
            f"Unknown config type: {config_class_name}. "
            f"Supported config types: {list(_CONFIG_CLASSES.keys())}"
        )

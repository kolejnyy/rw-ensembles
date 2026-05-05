"""
Factory functions for creating LLM models from configuration.

This module provides functions to:
1. Convert a dictionary (from YAML) to a configuration object
2. Create an LLM model instance from a configuration object
"""

from typing import Dict, Any

from invpro.models.llm.conf.types import (
    QwenCoderS2TConfig,
    HFChatLLMConfig,
    VLLMChatLLMConfig,
    OpenAIGPTLLMConfig,
    LLMConfig,
)
from invpro.models.llm.base import BaseLLM
from invpro.prompt.conf import dict_to_config as prompt_dict_to_config, prompt_formatter_from_config


# Mapping from class name to config class
_CONFIG_CLASSES = {
    "QwenCoderS2T": QwenCoderS2TConfig,
    "HFChatLLM": HFChatLLMConfig,
    "VLLMChatLLM": VLLMChatLLMConfig,
    "OpenAIGPTLLM": OpenAIGPTLLMConfig,
}


def dict_to_config(config_dict: Dict[str, Any]) -> LLMConfig:
    """
    Convert a dictionary (from YAML) to a configuration object.
    
    Args:
        config_dict: Dictionary with 'class' and 'parameters' keys
            Example: {
                "class": "QwenCoderS2T",
                "parameters": {
                    "checkpoint": "path/to/checkpoint",
                    "base_model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "use_4bit": true,
                    "prompt_formatter": {...}
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
    
    if class_name == "QwenCoderS2T":
        prompt_formatter_config = prompt_dict_to_config(parameters.pop("prompt_formatter", {}))
        return QwenCoderS2TConfig(prompt_formatter_config=prompt_formatter_config, **parameters)
    if class_name == "HFChatLLM":
        return HFChatLLMConfig(**parameters)
    if class_name == "VLLMChatLLM":
        return VLLMChatLLMConfig(**parameters)
    if class_name == "OpenAIGPTLLM":
        return OpenAIGPTLLMConfig(**parameters)
    else:
        raise ValueError(
            f"Unknown LLM class: {class_name}. "
            f"Supported classes: {list(_CONFIG_CLASSES.keys())}"
        )


def llm_from_config(config: LLMConfig) -> BaseLLM:
    """
    Create an LLM model instance from a configuration object.
    
    Args:
        config: Configuration object (e.g., QwenCoderS2TConfig)
    
    Returns:
        LLM model instance
    
    Raises:
        ValueError: If the config type is not recognized
    """
    from invpro.models.llm.qwen_coder import QwenCoderS2T
    from invpro.models.llm.hf_chat import HFChatLLM
    from invpro.models.llm.vllm_chat import VLLMChatLLM
    from invpro.models.llm.openai_gpt import OpenAIGPTLLM

    # Find the config class name
    config_class_name = None
    for name, cls in _CONFIG_CLASSES.items():
        if isinstance(config, cls):
            config_class_name = name
            break

    if config_class_name == "QwenCoderS2T":
        prompt_formatter = prompt_formatter_from_config(config.prompt_formatter_config)
        return QwenCoderS2T.from_config(config, prompt_formatter)
    if config_class_name == "HFChatLLM":
        import torch
        dtype = getattr(config, "torch_dtype", None)
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.bfloat16
        return HFChatLLM.from_pretrained(
            model_name_or_path=config.model_name_or_path,
            max_new_tokens=getattr(config, "max_new_tokens", 8192),
            device_map=getattr(config, "device_map", "auto"),
            torch_dtype=torch_dtype,
            temperature=getattr(config, "temperature", 0.7),
            top_p=getattr(config, "top_p", None),
            top_k=getattr(config, "top_k", None),
        )
    if config_class_name == "VLLMChatLLM":
        return VLLMChatLLM.from_pretrained(
            model_name_or_path=config.model_name_or_path,
            max_new_tokens=getattr(config, "max_new_tokens", 8192),
            max_model_len=getattr(config, "max_model_len", 8192),
            max_num_seqs=getattr(config, "max_num_seqs", 8),
            tensor_parallel_size=getattr(config, "tensor_parallel_size", 1),
            gpu_memory_utilization=getattr(config, "gpu_memory_utilization", 0.9),
            temperature=getattr(config, "temperature", 0.7),
            top_p=getattr(config, "top_p", None),
            top_k=getattr(config, "top_k", None),
            use_chat_template=getattr(config, "use_chat_template", True),
        )
    if config_class_name == "OpenAIGPTLLM":
        return OpenAIGPTLLM.from_pretrained(
            model_name_or_path=getattr(config, "model_name_or_path", "gpt-5.4"),
            temperature=getattr(config, "temperature", 0.2),
            max_output_tokens=getattr(config, "max_output_tokens", 5000),
            api_key=getattr(config, "api_key", None),
            api_key_env=getattr(config, "api_key_env", "OPENAI_API_KEY"),
            env_file_path=getattr(config, "env_file_path", None),
        )
    else:
        raise ValueError(
            f"Unknown config type: {config_class_name}. "
            f"Supported config types: {list(_CONFIG_CLASSES.keys())}"
        )

"""
Configuration classes for LLM models.

This module defines dataclasses that represent the configuration for different
types of LLM models.
"""

from pathlib import Path
from typing import Any, Union, Optional, Dict

from invpro.prompt.conf.types import PromptFormatterConfig

class QwenCoderS2TConfig:
    """Configuration for QwenCoderS2T model.

    Args:
        checkpoint: Optional path to checkpoint directory (for loading trained models)
        base_model_name: Base model name (default: "Qwen/Qwen2.5-Coder-7B-Instruct")
        use_4bit: Whether to use 4-bit quantization (default: True)
        prompt_formatter: Dictionary configuration for the prompt formatter
    """
    checkpoint: Optional[str] = None
    base_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    use_4bit: bool = True
    prompt_formatter_config: PromptFormatterConfig = None

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        base_model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        use_4bit: bool = True,
        prompt_formatter_config: Optional[PromptFormatterConfig] = None,
        **kwargs: Any
    ) -> None:
        self.checkpoint = checkpoint
        self.base_model_name = base_model_name
        self.use_4bit = use_4bit
        self.prompt_formatter_config = prompt_formatter_config


class VLLMChatLLMConfig:
    """Configuration for VLLMChatLLM (vLLM-backed chat model, faster inference).

    Args:
        model_name_or_path: HuggingFace model id or local path
        max_new_tokens: Max tokens to generate (default 8192)
        max_model_len: Max sequence length for vLLM (default 8192; lower for smaller GPUs)
        max_num_seqs: Max sequences per batch (default 8; lower for smaller GPUs)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default 1)
        gpu_memory_utilization: Fraction of GPU memory to use (default 0.9)
        temperature: Sampling temperature (default 0.7)
        top_p: Optional nucleus sampling
        top_k: Optional top-k sampling
        use_chat_template: If False, run plain completion on prompt text
    """
    model_name_or_path: str
    max_new_tokens: int = 8192
    max_model_len: Optional[int] = 8192
    max_num_seqs: int = 8
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    use_chat_template: bool = True

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 8192,
        max_model_len: Optional[int] = 8192,
        max_num_seqs: int = 8,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        use_chat_template: bool = True,
        **kwargs: Any
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_chat_template = use_chat_template


class HFChatLLMConfig:
    """Configuration for HFChatLLM (generic HF chat model, e.g. DeepSeek-Prover-V2).

    Args:
        model_name_or_path: HuggingFace model id or local path
        max_new_tokens: Max tokens to generate (default 8192)
        device_map: Device map for model (default "auto")
        torch_dtype: Optional dtype (default bfloat16)
        temperature: Sampling temperature (default 0.7). Used when do_sample=True.
        top_p: Optional nucleus sampling (e.g. 0.9).
        top_k: Optional top-k sampling.
    """
    model_name_or_path: str
    max_new_tokens: int = 8192
    device_map: Optional[str] = None
    torch_dtype: Optional[str] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 8192,
        device_map: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


class OpenAIGPTLLMConfig:
    """Configuration for OpenAIGPTLLM (OpenAI Responses API wrapper)."""

    model_name_or_path: str = "gpt-5.4"
    temperature: float = 0.2
    max_output_tokens: int = 5000
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    env_file_path: Optional[str] = None

    def __init__(
        self,
        model_name_or_path: str = "gpt-5.4",
        temperature: float = 0.2,
        max_output_tokens: int = 5000,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        env_file_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.env_file_path = env_file_path


# Create a single type for all LLM configuration classes
LLMConfig = Union[
    QwenCoderS2TConfig,
    HFChatLLMConfig,
    VLLMChatLLMConfig,
    OpenAIGPTLLMConfig,
]

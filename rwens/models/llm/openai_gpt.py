from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rwens.models.llm.base import BaseLLM


@dataclass
class OpenAIUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


class OpenAIGPTLLM(BaseLLM):
    """
    OpenAI-backed chat LLM wrapper using the Responses API.

    Initialization handles API key setup; callers can then invoke:
    - generate(prompt) -> processed text output
    - generate_with_usage(prompt) -> (processed text, token usage)
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt-5.4",
        temperature: float = 0.2,
        max_output_tokens: int = 5000,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        env_file_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.api_key_env = api_key_env
        self.env_file_path = env_file_path

        key = api_key or os.getenv(api_key_env) or self._read_env_key(api_key_env, env_file_path)
        if not key:
            raise ValueError(
                f"Missing API key '{api_key_env}'. Provide api_key, set env var, or set env_file_path."
            )

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "OpenAI package is not available. Install with: conda run -n invpro pip install openai"
            ) from e

        self._client = OpenAI(api_key=key)

    @staticmethod
    def _read_env_key(api_key_env: str, env_file_path: Optional[str]) -> str:
        if not env_file_path:
            return ""
        p = Path(env_file_path)
        if not p.is_file():
            return ""
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == api_key_env:
                return v.strip().strip('"').strip("'")
        return ""

    def get_model_cache_id(self) -> Optional[str]:
        blob = f"OpenAIGPTLLM:{self.model_name_or_path}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @staticmethod
    def _extract_processed_text(response) -> str:
        # Responses API exposes `output_text`; strip for downstream parser stability.
        return (getattr(response, "output_text", "") or "").strip()

    @staticmethod
    def _extract_usage(response) -> OpenAIUsage:
        usage_obj = getattr(response, "usage", None)
        input_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage_obj, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
        return OpenAIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def generate(self, prompt: str) -> str:
        text, _usage = self.generate_with_usage(prompt)
        return text

    def generate_with_usage(self, prompt: str) -> tuple[str, OpenAIUsage]:
        req = {
            "model": self.model_name_or_path,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        # Some models (including parts of GPT-5 line) may reject temperature.
        if self.temperature is not None:
            req["temperature"] = self.temperature
        try:
            resp = self._client.responses.create(**req)
        except Exception as e:
            msg = str(e)
            if "temperature" in msg and "not supported" in msg and "temperature" in req:
                req.pop("temperature", None)
                resp = self._client.responses.create(**req)
            else:
                raise
        return self._extract_processed_text(resp), self._extract_usage(resp)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "gpt-5.4",
        temperature: float = 0.2,
        max_output_tokens: int = 5000,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        env_file_path: Optional[str] = None,
        **kwargs,
    ) -> "OpenAIGPTLLM":
        return cls(
            model_name_or_path=model_name_or_path,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            api_key_env=api_key_env,
            env_file_path=env_file_path,
            **kwargs,
        )

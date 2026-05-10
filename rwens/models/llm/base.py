from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def get_model_cache_id(self) -> Optional[str]:
        """
        Return a stable id for cache keys (e.g. first-tactic / theorem-surprise / rewriting).
        Should encode backend implementation (HF vs vLLM, etc.) plus model path so different
        runtimes do not share entries for the same weights. Default None; override to enable caching.
        """
        return None
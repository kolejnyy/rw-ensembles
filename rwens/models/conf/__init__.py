"""Configuration system for formal provers."""

from rwens.models.conf.types import (
    ProverConfig,
    NaiveStepByStepProverConfig,
    CanonicalProverConfig,
    CanonicalLLMProverConfig,
    EnsembleLLMProverConfig,
)
from rwens.models.conf.factory import dict_to_config, prover_from_config

__all__ = [
    "ProverConfig",
    "NaiveStepByStepProverConfig",
    "CanonicalProverConfig",
    "CanonicalLLMProverConfig",
    "EnsembleLLMProverConfig",
    "dict_to_config",
    "prover_from_config",
]

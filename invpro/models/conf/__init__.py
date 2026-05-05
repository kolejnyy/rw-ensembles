"""Configuration system for formal provers."""

from invpro.models.conf.types import (
    ProverConfig,
    NaiveStepByStepProverConfig,
    CanonicalProverConfig,
    CanonicalLLMProverConfig,
    EnsembleLLMProverConfig,
)
from invpro.models.conf.factory import dict_to_config, prover_from_config

__all__ = [
    "ProverConfig",
    "NaiveStepByStepProverConfig",
    "CanonicalProverConfig",
    "CanonicalLLMProverConfig",
    "EnsembleLLMProverConfig",
    "dict_to_config",
    "prover_from_config",
]

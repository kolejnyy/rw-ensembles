"""Rewriting configuration (YAML → RewritingCanonicalizationModule)."""

from rwens.rewriting.conf.types import (
    CanonicalizationConfig,
    RewritingCanonicalizationConfig,
)
from rwens.rewriting.conf.factory import dict_to_config, canonicalization_from_config

__all__ = [
    "CanonicalizationConfig",
    "RewritingCanonicalizationConfig",
    "dict_to_config",
    "canonicalization_from_config",
]

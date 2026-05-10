"""Canonicalization configuration system."""

from rwens.canonicalization.conf.types import (
    VariableRenamerConfig,
    IdentityModuleConfig,
    SimpModuleConfig,
    CanonicalizationConfig,
)
from rwens.canonicalization.conf.factory import (
    dict_to_config,
    canonicalization_from_config,
)

__all__ = [
    "VariableRenamerConfig",
    "IdentityModuleConfig",
    "SimpModuleConfig",
    "CanonicalizationConfig",
    "dict_to_config",
    "canonicalization_from_config",
]

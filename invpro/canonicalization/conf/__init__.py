"""Canonicalization configuration system."""

from invpro.canonicalization.conf.types import (
    VariableRenamerConfig,
    IdentityModuleConfig,
    SimpModuleConfig,
    CanonicalizationConfig,
)
from invpro.canonicalization.conf.factory import (
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

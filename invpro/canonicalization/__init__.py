"""
Canonicalization utilities for Lean proof states.

This module provides functions to canonicalize Lean proof states by renaming
variables and hypotheses. The main interface is CanonicalizationModule, with
VariableRenamer as the default implementation.
"""

from invpro.canonicalization.base import CanonicalizationModule
from invpro.canonicalization.identity import IdentityModule
from invpro.canonicalization.renaming import VariableRenamer
from invpro.canonicalization.simp import SimpModule
from invpro.canonicalization.rewriting import RewritingCanonicalizationModule

__all__ = [
    "CanonicalizationModule",
    "IdentityModule",
    "VariableRenamer",
    "SimpModule",
    "RewritingCanonicalizationModule",
]

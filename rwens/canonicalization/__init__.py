"""
Canonicalization utilities for Lean proof states.

This module provides functions to canonicalize Lean proof states by renaming
variables and hypotheses. The main interface is CanonicalizationModule, with
VariableRenamer as the default implementation.
"""

from rwens.canonicalization.base import CanonicalizationModule
from rwens.canonicalization.identity import IdentityModule
from rwens.canonicalization.renaming import VariableRenamer
from rwens.canonicalization.simp import SimpModule
from rwens.canonicalization.rewriting import RewritingCanonicalizationModule

__all__ = [
    "CanonicalizationModule",
    "IdentityModule",
    "VariableRenamer",
    "SimpModule",
    "RewritingCanonicalizationModule",
]

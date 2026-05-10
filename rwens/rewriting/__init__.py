"""
Lean rewriting augmentation (rwcnc pipeline) and shared module base classes.

Public surface: :class:`CanonicalizationModule`, :class:`RewritingCanonicalizationModule`,
and helpers under ``rwens.rewriting.rewrites`` / ``rwens.rewriting.conf``.
"""

from rwens.rewriting.base import CanonicalizationModule, SimpleCanonicalizationModule
from rwens.rewriting.module import RewritingCanonicalizationModule

__all__ = [
    "CanonicalizationModule",
    "SimpleCanonicalizationModule",
    "RewritingCanonicalizationModule",
]

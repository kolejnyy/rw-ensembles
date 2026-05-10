"""
Theorem equivalence checking (RWData-style bridge) and certificate construction.

Prefer importing from this package, e.g. ``from rwens.equivalence import EquivalenceChecker``.
Lean snippet helpers live in :mod:`rwens.utils.lean_proof_text` and
:mod:`rwens.utils.lean_preamble` and are re-exported here for convenience.
"""

from __future__ import annotations

from rwens.equivalence.certificate import (
    build_equivalence_certificate_lean,
    compose_rwdata_bridge_proof_lines,
)
from rwens.equivalence.checker import (
    DEFAULT_TACTIC_SEQUENCE,
    EquivalenceChecker,
    append_tactic_proof,
    append_tactics_block,
    sanitize_preamble_remove_aesop,
)
from rwens.equivalence.types import (
    AuxiliaryGoal,
    AuxiliaryGoalKind,
    AuxiliaryGoalsBundle,
    AuxiliaryVerificationResult,
    EquivalenceCertificateStatus,
    EquivalenceCheckResult,
)

__all__ = [
    "DEFAULT_TACTIC_SEQUENCE",
    "AuxiliaryGoal",
    "AuxiliaryGoalKind",
    "AuxiliaryGoalsBundle",
    "AuxiliaryVerificationResult",
    "EquivalenceCertificateStatus",
    "EquivalenceCheckResult",
    "EquivalenceChecker",
    "append_tactic_proof",
    "append_tactics_block",
    "build_equivalence_certificate_lean",
    "compose_rwdata_bridge_proof_lines",
    "sanitize_preamble_remove_aesop",
]

"""Shared dataclasses and enums for equivalence / RWData bridge checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from invpro.utils.statement_parser import ParsedTheorem


class AuxiliaryGoalKind(str, Enum):
    """Proof obligation category for rewrite equivalence."""

    CONDITION_FORWARD = "condition_forward"
    """Original context ⊢ each augmented hypothesis (``cond_i`` in RWData methodology)."""

    GOAL_AUG_TO_ORIG = "goal_augmented_to_original"
    """Augmented conclusion as hypothesis ⊢ original conclusion (``goal_to_goal``)."""


@dataclass(frozen=True)
class AuxiliaryGoal:
    """One auxiliary proof obligation (theorem declaration through ``:= by``)."""

    kind: AuxiliaryGoalKind
    theorem_statement: str
    """Lean snippet: ``theorem ... := by`` plus trailing newline."""


@dataclass
class AuxiliaryGoalsBundle:
    """Result of :meth:`EquivalenceChecker.create_auxiliary_goals`."""

    preamble: str
    """Imports / opens / options for this problem only (caller-supplied)."""
    original: ParsedTheorem
    variant: ParsedTheorem
    goals: list[AuxiliaryGoal] = field(default_factory=list)


@dataclass(frozen=True)
class AuxiliaryVerificationResult:
    """Outcome of verifying one auxiliary goal with a tactic block."""

    kind: AuxiliaryGoalKind
    theorem_statement: str
    success: bool
    winning_tactic: Optional[str]
    """On tactic success, joined tactic names; on LLM success, ``\"llm\"``."""
    last_error: Optional[str]
    proof_source: str = "tactics"
    """``\"tactics\"`` or ``\"llm\"`` when :attr:`success` is True."""
    llm_fallback_attempted: bool = False
    """True if cheap tactics failed and an LLM fallback was run (when configured)."""
    verified_proof_body: Optional[str] = None
    """Text after ``:= by`` for this goal when verification succeeded (tactics or LLM)."""


class EquivalenceCertificateStatus(str, Enum):
    """End-to-end equivalence certificate state (auxiliary lemmas + composed file)."""

    AUXILIARY_FAILED = "auxiliary_failed"
    """At least one auxiliary goal did not verify."""

    UNCONFIRMED = "unconfirmed"
    """All auxiliary goals verified, but the full RWData certificate file did not verify."""

    ACCEPTED = "accepted"
    """Auxiliary lemmas and the composed certificate (``invpro.lean``-style) both verify."""


@dataclass(frozen=True)
class EquivalenceCheckResult:
    """Outcome of :meth:`EquivalenceChecker.check_equivalence`."""

    bundle: AuxiliaryGoalsBundle
    auxiliary_results: list[AuxiliaryVerificationResult]
    certificate_lean: Optional[str] = None
    """Full single-file certificate (header + variant ``sorry`` + auxiliaries + original bridge)."""
    certificate_verified: Optional[bool] = None
    """Whether :class:`ProofVerifier` accepted :attr:`certificate_lean` (``None`` if not built)."""
    certificate_verify_error: Optional[str] = None
    """Lean error when certificate verification failed; or build exception message."""

    @property
    def success(self) -> bool:
        """True iff every auxiliary bridge goal verified (cheap tactics and/or LLM)."""
        return all(r.success for r in self.auxiliary_results)

    @property
    def certificate_status(self) -> EquivalenceCertificateStatus:
        if not self.success:
            return EquivalenceCertificateStatus.AUXILIARY_FAILED
        if self.certificate_verified is True:
            return EquivalenceCertificateStatus.ACCEPTED
        return EquivalenceCertificateStatus.UNCONFIRMED

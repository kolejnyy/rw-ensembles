"""
Equivalence checking for pairs of Lean theorem statements (e.g. original vs rewrite).

**Sufficiency (RWData bridge):** Write the original theorem as
``Γ ⊢ C`` and the augmented (variant) theorem as ``Γ' ⊢ C'``. To show that a proof
of the variant yields the original, it is enough to prove ``Γ ⊢ h`` for each
hypothesis ``h`` in ``Γ'`` (the ``cond_i`` goals) and ``C' ⊢ C`` with ``C'`` as an
assumption (the ``goal_to_goal`` goal). There is **no** need for the reverse
``C ⊢ C'`` (original conclusion implying the augmented conclusion).

There is no single dataset-wide header: each problem may use different imports,
opens, and options. Callers must pass the exact ``preamble`` (everything before
the first ``theorem`` / ``lemma`` / ``def``) they want wrapped around generated
goals — typically copied from the same source as the original statement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from rwens.equivalence.certificate import (
    build_equivalence_certificate_lean,
    compose_rwdata_bridge_proof_lines,
)
from rwens.utils.lean_preamble import sanitize_preamble_remove_aesop
from rwens.utils.lean_proof_text import (
    append_tactic_proof,
    append_tactics_block,
    extract_proof_body_after_by,
)
from rwens.equivalence.types import (
    AuxiliaryGoal,
    AuxiliaryGoalKind,
    AuxiliaryGoalsBundle,
    AuxiliaryVerificationResult,
    EquivalenceCertificateStatus,
    EquivalenceCheckResult,
)
from rwens.models.single_pass_prover import SinglePassProver
from rwens.utils.metrics import equivalence_certificate_diagnostics_acceptable
from rwens.utils.statement_parser import (
    StatementParser,
    parse_theorem_through_by,
)
from rwens.utils.verifier import ProofVerifier

# Re-export for callers that import from this module (tests, scripts).
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

# Same order as ``rwens.lean`` (cond_2): linear / simp-style tactics before ``ring_nf``
# to avoid undesirable ``ring_nf`` behaviour on early goals.
DEFAULT_TACTIC_SEQUENCE: tuple[str, ...] = (
    "decide",
    "native_decide",
    "trivial",
    "linarith",
    "nlinarith",
    "omega",
    "norm_num",
    "field_simp",
    "simp",
    "ring_nf",
    "ring",
    "exact?",
)


def _resolved_preamble(bundle: AuxiliaryGoalsBundle, default_preamble: str) -> str:
    p = bundle.preamble.strip()
    if not p:
        p = default_preamble.strip()
    if not p.endswith("\n"):
        p += "\n"
    return sanitize_preamble_remove_aesop(p)


def _try_llm_prove_auxiliary_goal(
    preamble: str,
    goal: AuxiliaryGoal,
    verifier: ProofVerifier,
    prover: SinglePassProver,
    n_attempts: int,
    batch_size: int,
) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Generate full-file candidates with :meth:`SinglePassProver.generate_batch` and
    return the first that :meth:`ProofVerifier.verify` accepts.

    Returns
    -------
    success, winning_tactic, last_error, verified_proof_body
        ``verified_proof_body`` is set on success (text after ``:= by``).
    """
    problem_statement = preamble + goal.theorem_statement
    if not problem_statement.endswith("\n"):
        problem_statement += "\n"
    candidates = prover.generate_batch(
        problem_statement,
        n_attempts=n_attempts,
        batch_size=batch_size,
    )
    last_err: Optional[str] = None
    for code in candidates:
        if code is None:
            continue
        if not code.endswith("\n"):
            code += "\n"
        ok, err = verifier.verify(code)
        if ok:
            try:
                pb = extract_proof_body_after_by(code)
            except ValueError:
                pb = ""
            return True, "llm", None, pb
        last_err = err
    return False, None, last_err or "LLM produced no verifiable candidate", None


class EquivalenceChecker:
    """
    Bridge lemmas for showing an augmented theorem suffices for the original (see
    module docstring: forward hypotheses + ``augmented goal → original goal`` only).
    Auxiliary theorem heads are generated explicitly; proof search is separate.
    """

    def __init__(self, single_pass_prover: SinglePassProver | None = None) -> None:
        """
        Parameters
        ----------
        single_pass_prover
            Optional :class:`SinglePassProver` for LLM fallback after cheap tactics
            fail on a goal (:meth:`verify_auxiliary_goals`, :meth:`check_equivalence`).
        """
        self.single_pass_prover = single_pass_prover

    def create_auxiliary_goals(
        self,
        preamble: str,
        original_theorem: str,
        variant_theorem: str,
        *,
        bridge_variant_theorem: str | None = None,
    ) -> AuxiliaryGoalsBundle:
        """
        Build Lean theorem heads (through ``:= by``) that encode standard bridge steps.

        **Preamble:** must be exactly what you need before theorems for this problem
        (imports, ``open``, ``set_option``, etc.). It is not read or transformed; it is
        only stored on the bundle for downstream code to prepend when building files.

        **Inputs:** ``original_theorem`` and ``variant_theorem`` should each be a single
        theorem ending with ``:= by`` (same shape as in miniF2F ``formal_statement``).

        **Methodology** (see ``.todo/RWData.md`` and the module docstring): only the
        obligations below are required; there is no ``original conclusion → variant
        conclusion`` goal.

        1. For each **hypothesis** binder of the variant (in order), emit
           ``theorem cond_i <original binders> : <variant type for that hypothesis> := by``.
        2. Emit ``theorem goal_to_goal <original variable binders> (aug_goal : <variant conclusion>)
           : <original conclusion> := by``.

        Original binders are taken verbatim from the original theorem (problem-specific shape).
        """
        orig = parse_theorem_through_by(original_theorem)
        variant_for_bridge = bridge_variant_theorem or variant_theorem
        var = parse_theorem_through_by(variant_for_bridge)
        goals: list[AuxiliaryGoal] = []
        ob = orig.binder_segment.strip()

        for i, pb in enumerate(StatementParser.hypotheses(variant_for_bridge), start=1):
            ty = pb.type_str.strip()
            stmt = f"theorem cond_{i} {ob} : {ty} := by\n"
            goals.append(
                AuxiliaryGoal(
                    kind=AuxiliaryGoalKind.CONDITION_FORWARD,
                    theorem_statement=stmt,
                )
            )

        var_chunks = " ".join(
            v.raw_chunk.strip() for v in StatementParser.variable_binders(original_theorem)
        )
        oc = orig.conclusion.strip()
        vc = var.conclusion.strip()
        goal_stmt = (
            f"theorem goal_to_goal {var_chunks} (aug_goal : {vc}) : {oc} := by\n"
        )
        goals.append(
            AuxiliaryGoal(
                kind=AuxiliaryGoalKind.GOAL_AUG_TO_ORIG,
                theorem_statement=goal_stmt,
            )
        )

        return AuxiliaryGoalsBundle(
            preamble=preamble,
            original=orig,
            variant=var,
            goals=goals,
        )

    def check_equivalence(
        self,
        preamble: str,
        original_theorem: str,
        variant_theorem: str,
        project_root: str | Path,
        *,
        bridge_variant_theorem: str | None = None,
        tactics: Sequence[str] | None = None,
        timeout_seconds: float = 120.0,
        default_preamble: str = "import Mathlib\n",
        llm_attempts_per_failed_goal: int = 16,
        llm_batch_size: int = 16,
    ) -> EquivalenceCheckResult:
        """
        First-stage equivalence check: generate auxiliary bridge goals and verify each
        with the default ``try`` tactic block (same as :meth:`verify_auxiliary_goals`).

        Constructs a single :class:`ProofVerifier` (one LSP session) for all goals.
        If :attr:`single_pass_prover` was set on this checker, it runs after cheap
        tactics fail on a goal (:meth:`SinglePassProver.generate_batch` + re-verify
        with the same verifier).
        """
        bundle = self.create_auxiliary_goals(
            preamble,
            original_theorem,
            variant_theorem,
            bridge_variant_theorem=bridge_variant_theorem,
        )
        resolved = _resolved_preamble(bundle, default_preamble)
        verifier = ProofVerifier(
            project_root=str(Path(project_root).resolve()),
            initial_imports=resolved,
            timeout_seconds=timeout_seconds,
        )
        try:
            results = self.verify_auxiliary_goals(
                bundle,
                project_root,
                tactics=tactics,
                timeout_seconds=timeout_seconds,
                default_preamble=default_preamble,
                verifier=verifier,
                llm_attempts_per_failed_goal=llm_attempts_per_failed_goal,
                llm_batch_size=llm_batch_size,
            )
            if not all(r.success for r in results):
                return EquivalenceCheckResult(
                    bundle=bundle,
                    auxiliary_results=results,
                    certificate_lean=None,
                    certificate_verified=None,
                    certificate_verify_error=None,
                )

            try:
                cert = build_equivalence_certificate_lean(
                    resolved,
                    original_theorem,
                    variant_theorem,
                    bundle,
                    results,
                    bridge_variant_theorem=bridge_variant_theorem,
                )
            except Exception as ex:
                return EquivalenceCheckResult(
                    bundle=bundle,
                    auxiliary_results=results,
                    certificate_lean=None,
                    certificate_verified=False,
                    certificate_verify_error=str(ex),
                )

            if not cert.endswith("\n"):
                cert += "\n"
            ok_cert, err_cert = verifier.verify(
                cert,
                diagnostics_ok=equivalence_certificate_diagnostics_acceptable,
            )
            return EquivalenceCheckResult(
                bundle=bundle,
                auxiliary_results=results,
                certificate_lean=cert,
                certificate_verified=ok_cert,
                certificate_verify_error=None if ok_cert else err_cert,
            )
        finally:
            verifier.close()

    def verify_auxiliary_goals(
        self,
        bundle: AuxiliaryGoalsBundle,
        project_root: str | Path,
        *,
        tactics: Sequence[str] | None = None,
        timeout_seconds: float = 120.0,
        default_preamble: str = "import Mathlib\n",
        verifier: ProofVerifier | None = None,
        llm_attempts_per_failed_goal: int = 16,
        llm_batch_size: int = 16,
    ) -> list[AuxiliaryVerificationResult]:
        """
        Check each auxiliary goal with :class:`ProofVerifier` in **one** LSP round-trip
        per goal.

        Builds ``full_code = preamble + theorem`` where the theorem proof is
        ``by`` followed by ``try <tactic>`` on each line (same pattern as ``.todo/RWData.md``).

        Parameters
        ----------
        bundle
            Output of :meth:`create_auxiliary_goals`.
        project_root
            Lean project root (directory containing ``lakefile.lean`` / ``lakefile.toml``).
            Used only when ``verifier`` is omitted (to construct a new client).
        tactics
            Tactic names (no ``by``), one line each under ``by``. Defaults to
            :data:`DEFAULT_TACTIC_SEQUENCE`.
        default_preamble
            Used when ``bundle.preamble`` is empty.
        verifier
            If provided, used for every goal (avoids a new LSP client). Must have been
            created with ``initial_imports`` equal to the resolved preamble for this
            bundle. When omitted, a new :class:`ProofVerifier` is created for this call.
        llm_attempts_per_failed_goal
            Number of LLM samples per failed goal (batched when ``llm_batch_size`` > 1).
        llm_batch_size
            Batch size for underlying LLM generation when supported.

        If :attr:`single_pass_prover` is set on this checker, each goal that still
        fails after the cheap tactic phase is passed to
        :meth:`SinglePassProver.generate_batch`; candidates are verified with the
        same ``verifier`` (not the prover's internal client).
        """
        tac_seq: tuple[str, ...] = (
            tuple(tactics) if tactics is not None else DEFAULT_TACTIC_SEQUENCE
        )
        preamble = _resolved_preamble(bundle, default_preamble)
        if verifier is None:
            verifier = ProofVerifier(
                project_root=str(Path(project_root).resolve()),
                initial_imports=preamble,
                timeout_seconds=timeout_seconds,
            )
        results: list[AuxiliaryVerificationResult] = []
        tactics_label = "; ".join(tac_seq)
        for goal in bundle.goals:
            body = append_tactics_block(goal.theorem_statement, tac_seq)
            full_code = preamble + body
            if not full_code.endswith("\n"):
                full_code += "\n"
            ok, err = verifier.verify(full_code)
            vpb: Optional[str] = None
            if ok:
                try:
                    vpb = extract_proof_body_after_by(full_code)
                except ValueError:
                    vpb = None
            results.append(
                AuxiliaryVerificationResult(
                    kind=goal.kind,
                    theorem_statement=goal.theorem_statement,
                    success=ok,
                    winning_tactic=tactics_label if ok else None,
                    last_error=None if ok else err,
                    proof_source="tactics",
                    llm_fallback_attempted=False,
                    verified_proof_body=vpb,
                )
            )

        if self.single_pass_prover is None:
            return results

        updated: list[AuxiliaryVerificationResult] = []
        for r, goal in zip(results, bundle.goals):
            if r.success:
                updated.append(r)
                continue
            ok, wt, llm_err, vpb = _try_llm_prove_auxiliary_goal(
                preamble,
                goal,
                verifier,
                self.single_pass_prover,
                llm_attempts_per_failed_goal,
                llm_batch_size,
            )
            updated.append(
                AuxiliaryVerificationResult(
                    kind=r.kind,
                    theorem_statement=r.theorem_statement,
                    success=ok,
                    winning_tactic=wt if ok else None,
                    last_error=None if ok else llm_err,
                    proof_source="llm" if ok else "tactics",
                    llm_fallback_attempted=True,
                    verified_proof_body=vpb,
                )
            )
        return updated

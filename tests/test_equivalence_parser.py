"""
Tests for equivalence auxiliary goal generation and verification helpers.

The first test pins the RWData.md amc12 example; add further parser / verification tests below.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rwens.equivalence import (
    AuxiliaryGoalKind,
    AuxiliaryVerificationResult,
    EquivalenceCertificateStatus,
    EquivalenceChecker,
    EquivalenceCheckResult,
    append_tactic_proof,
    append_tactics_block,
    build_equivalence_certificate_lean,
)
from rwens.utils.metrics import equivalence_certificate_diagnostics_acceptable

# --- RWData.md — 50×20 example (amc12_2000_p5 vs v11) ---------------------------------

ORIGINAL_AMC12_2000_P5 = """theorem amc12_2000_p5 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : x - p = 2 - 2 * p := by
"""

VARIANT_AMC12_2000_P5_V11 = """theorem amc12_2000_p5_v11 (x p : ℝ) (h₀ : 2 > x) (h₁ : abs (x - 2) = p) :
  x + p = 2 := by
"""

EXPECTED_COND_1 = """theorem cond_1 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : 2 > x := by
"""

EXPECTED_COND_2 = """theorem cond_2 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : abs (x - 2) = p := by
"""

EXPECTED_GOAL_TO_GOAL = """theorem goal_to_goal (x p : ℝ) (aug_goal : x + p = 2) : x - p = 2 - 2 * p := by
"""

# --- ``data/rewritings/minif2f/valid/amc12_2000_p5`` — v3 (reordered equality in ``h₁``) -----------------

VARIANT_AMC12_2000_P5_V3 = """theorem amc12_2000_p5_v3 (x p : ℝ) (h₀ : x < 2) (h₁ : p = abs (x - 2)) :
  x - p = 2 - 2 * p := by
"""

VARIANT_AMC12_2000_P5_V11_RENAMED = """theorem amc12_2000_p5_v11_renamed (a b : ℝ) (ha : 2 > a) (hb : abs (a - 2) = b) :
  a + b = 2 := by
"""

EXPECTED_V3_COND_1 = """theorem cond_1 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : x < 2 := by
"""

EXPECTED_V3_COND_2 = """theorem cond_2 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : p = abs (x - 2) := by
"""

EXPECTED_V3_GOAL_TO_GOAL = """theorem goal_to_goal (x p : ℝ) (aug_goal : x - p = 2 - 2 * p) : x - p = 2 - 2 * p := by
"""


def test_rwdata_amc12_auxiliary_goals_match_doc() -> None:
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble="",
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V11,
    )
    assert len(bundle.goals) == 3
    assert bundle.goals[0].kind == AuxiliaryGoalKind.CONDITION_FORWARD
    assert bundle.goals[0].theorem_statement == EXPECTED_COND_1
    assert bundle.goals[1].kind == AuxiliaryGoalKind.CONDITION_FORWARD
    assert bundle.goals[1].theorem_statement == EXPECTED_COND_2
    assert bundle.goals[2].kind == AuxiliaryGoalKind.GOAL_AUG_TO_ORIG
    assert bundle.goals[2].theorem_statement == EXPECTED_GOAL_TO_GOAL


def test_amc12_v3_auxiliary_goals_match_valid_file() -> None:
    """Pins bridge lemmas for ``amc12_2000_p5`` vs ``amc12_2000_p5_v3`` (valid rewrite file)."""
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble="",
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V3,
    )
    assert len(bundle.goals) == 3
    assert bundle.goals[0].theorem_statement == EXPECTED_V3_COND_1
    assert bundle.goals[1].theorem_statement == EXPECTED_V3_COND_2
    assert bundle.goals[2].theorem_statement == EXPECTED_V3_GOAL_TO_GOAL


def test_auxiliary_goals_use_explicit_bridge_variant_for_renamed_rows() -> None:
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble="",
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V11_RENAMED,
        bridge_variant_theorem=VARIANT_AMC12_2000_P5_V11,
    )
    assert len(bundle.goals) == 3
    assert bundle.goals[0].theorem_statement == EXPECTED_COND_1
    assert bundle.goals[1].theorem_statement == EXPECTED_COND_2
    assert bundle.goals[2].theorem_statement == EXPECTED_GOAL_TO_GOAL


def test_append_tactic_proof_inserts_tactic_after_by() -> None:
    head = "theorem t (n : ℕ) : n = n := by\n"
    out = append_tactic_proof(head, "simp")
    assert ":= by\n" in out
    assert "  try simp" in out


def test_equivalence_certificate_diagnostics_accepts_sorry_and_benign_linter() -> None:
    class _Diag:
        success = False
        diagnostics = [
            {"message": "declaration uses 'sorry'"},
            {"message": "unused variable `h₁`\nnote: disable with set_option"},
        ]

    assert equivalence_certificate_diagnostics_acceptable(_Diag())


def test_equivalence_certificate_diagnostics_rejects_real_failure() -> None:
    class _Diag:
        success = False
        diagnostics = [
            {"message": "declaration uses 'sorry'"},
            {"message": "type mismatch"},
        ]

    assert not equivalence_certificate_diagnostics_acceptable(_Diag())


def test_equivalence_certificate_diagnostics_empty_uses_success_flag() -> None:
    class Ok:
        success = True
        diagnostics: list = []

    class Bad:
        success = False
        diagnostics: list = []

    assert equivalence_certificate_diagnostics_acceptable(Ok())
    assert not equivalence_certificate_diagnostics_acceptable(Bad())


def test_append_tactics_block_joins_multiple_lines() -> None:
    head = "theorem t : True := by\n"
    out = append_tactics_block(head, ("simp", "trivial"))
    assert "  try simp" in out
    assert "  try trivial" in out
    assert out.index("try simp") < out.index("try trivial")


def test_verify_auxiliary_goals_one_tactic_per_goal_when_verify_succeeds() -> None:
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble="import Mathlib\n",
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V11,
    )
    with patch("rwens.equivalence.checker.ProofVerifier") as MockV:
        inst = MockV.return_value
        inst.verify.return_value = (True, None)

        results = EquivalenceChecker().verify_auxiliary_goals(
            bundle,
            project_root="/tmp",
            tactics=("simp",),
        )

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.winning_tactic == "simp" for r in results)
        assert inst.verify.call_count == 3  # one full block per goal


def test_check_equivalence_uses_single_verifier_for_all_goals() -> None:
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble="import Mathlib\n",
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V11,
    )
    with patch("rwens.equivalence.checker.ProofVerifier") as MockV:
        inst = MockV.return_value
        inst.verify.return_value = (True, None)

        result = EquivalenceChecker().check_equivalence(
            preamble="import Mathlib\n",
            original_theorem=ORIGINAL_AMC12_2000_P5,
            variant_theorem=VARIANT_AMC12_2000_P5_V11,
            project_root="/tmp",
            tactics=("simp",),
        )

        assert MockV.call_count == 1
        assert inst.verify.call_count == 4  # three aux goals + full certificate file
        assert isinstance(result, EquivalenceCheckResult)
        assert result.success
        assert len(result.auxiliary_results) == 3
        assert result.bundle.goals == bundle.goals
        assert result.certificate_verified is True
        assert result.certificate_status == EquivalenceCertificateStatus.ACCEPTED
        assert result.certificate_lean is not None


def test_check_equivalence_amc12_v3() -> None:
    """``check_equivalence`` on original vs v3: one verifier, three auxiliary checks (mocked Lean)."""
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble="import Mathlib\n",
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V3,
    )
    with patch("rwens.equivalence.checker.ProofVerifier") as MockV:
        inst = MockV.return_value
        inst.verify.return_value = (True, None)

        result = EquivalenceChecker().check_equivalence(
            preamble="import Mathlib\n",
            original_theorem=ORIGINAL_AMC12_2000_P5,
            variant_theorem=VARIANT_AMC12_2000_P5_V3,
            project_root="/tmp",
            tactics=("simp",),
        )

        assert MockV.call_count == 1
        assert inst.verify.call_count == 4
        assert isinstance(result, EquivalenceCheckResult)
        assert result.success
        assert len(result.auxiliary_results) == 3
        assert result.bundle.goals == bundle.goals
        assert result.certificate_status == EquivalenceCertificateStatus.ACCEPTED


def test_verify_auxiliary_goals_skips_verifier_constructor_when_given() -> None:
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble="import Mathlib\n",
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V11,
    )
    with patch("rwens.equivalence.checker.ProofVerifier") as MockV:
        v = MagicMock()
        v.verify.return_value = (True, None)
        EquivalenceChecker().verify_auxiliary_goals(
            bundle,
            "/tmp",
            tactics=("simp",),
            verifier=v,
        )
        MockV.assert_not_called()
        assert v.verify.call_count == 3


# --- Real Lean (optional): RWData.md amc12 example ------------------------------------

LEAN_PREAMBLE_AMC12 = """import Mathlib

set_option maxHeartbeats 0

open Real
"""


def test_build_equivalence_certificate_amc12_v11_shape() -> None:
    """Certificate text matches RWData layout (variant ``sorry``, aux, original bridge)."""
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble=LEAN_PREAMBLE_AMC12,
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V11,
    )
    stub = "linarith\n"
    aux: list[AuxiliaryVerificationResult] = [
        AuxiliaryVerificationResult(
            kind=AuxiliaryGoalKind.CONDITION_FORWARD,
            theorem_statement=bundle.goals[0].theorem_statement,
            success=True,
            winning_tactic="t",
            last_error=None,
            proof_source="tactics",
            verified_proof_body=stub,
        ),
        AuxiliaryVerificationResult(
            kind=AuxiliaryGoalKind.CONDITION_FORWARD,
            theorem_statement=bundle.goals[1].theorem_statement,
            success=True,
            winning_tactic="t",
            last_error=None,
            proof_source="tactics",
            verified_proof_body=stub,
        ),
        AuxiliaryVerificationResult(
            kind=AuxiliaryGoalKind.GOAL_AUG_TO_ORIG,
            theorem_statement=bundle.goals[2].theorem_statement,
            success=True,
            winning_tactic="t",
            last_error=None,
            proof_source="tactics",
            verified_proof_body=stub,
        ),
    ]
    cert = build_equivalence_certificate_lean(
        LEAN_PREAMBLE_AMC12,
        ORIGINAL_AMC12_2000_P5,
        VARIANT_AMC12_2000_P5_V11,
        bundle,
        aux,
    )
    assert "sorry" in cert
    assert "theorem cond_1" in cert and "theorem cond_2" in cert and "theorem goal_to_goal" in cert
    assert "have aug_0" in cert and "have aug_goal" in cert
    assert "exact goal_to_goal" in cert


def test_build_equivalence_certificate_renamed_variant_uses_bridge_variant() -> None:
    bundle = EquivalenceChecker().create_auxiliary_goals(
        preamble=LEAN_PREAMBLE_AMC12,
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V11_RENAMED,
        bridge_variant_theorem=VARIANT_AMC12_2000_P5_V11,
    )
    stub = "linarith\n"
    aux: list[AuxiliaryVerificationResult] = [
        AuxiliaryVerificationResult(
            kind=g.kind,
            theorem_statement=g.theorem_statement,
            success=True,
            winning_tactic="t",
            last_error=None,
            proof_source="tactics",
            verified_proof_body=stub,
        )
        for g in bundle.goals
    ]
    cert = build_equivalence_certificate_lean(
        LEAN_PREAMBLE_AMC12,
        ORIGINAL_AMC12_2000_P5,
        VARIANT_AMC12_2000_P5_V11_RENAMED,
        bundle,
        aux,
        bridge_variant_theorem=VARIANT_AMC12_2000_P5_V11,
    )
    assert "theorem amc12_2000_p5_v11_renamed (a b : ℝ)" in cert
    assert "theorem cond_1 (x p : ℝ)" in cert
    assert "have aug_goal : x + p = 2 := amc12_2000_p5_v11_renamed x p aug_0 aug_1" in cert


@pytest.mark.skipif(
    not os.environ.get("RWENS_RUN_LEAN_VERIFY"),
    reason="Set RWENS_RUN_LEAN_VERIFY=1 to run ProofVerifier + LSP (slow; needs lake project).",
)
def test_verify_auxiliary_goals_rwdata_amc12_real_lean() -> None:
    """End-to-end check for ``.todo/RWData.md`` lines 38–41: tactics close all three goals."""
    repo_root = Path(__file__).resolve().parents[1]
    result = EquivalenceChecker().check_equivalence(
        preamble=LEAN_PREAMBLE_AMC12,
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V11,
        project_root=repo_root,
        timeout_seconds=180.0,
    )
    assert len(result.auxiliary_results) == 3
    assert result.success, [
        r.last_error for r in result.auxiliary_results if not r.success
    ]


@pytest.mark.skipif(
    not os.environ.get("RWENS_RUN_LEAN_VERIFY"),
    reason="Set RWENS_RUN_LEAN_VERIFY=1 to run ProofVerifier + LSP (slow; needs lake project).",
)
def test_check_equivalence_amc12_v3_real_lean() -> None:
    """Default ``try`` tactic block closes all bridge goals for original vs ``amc12_2000_p5_v3``."""
    repo_root = Path(__file__).resolve().parents[1]
    result = EquivalenceChecker().check_equivalence(
        preamble=LEAN_PREAMBLE_AMC12,
        original_theorem=ORIGINAL_AMC12_2000_P5,
        variant_theorem=VARIANT_AMC12_2000_P5_V3,
        project_root=repo_root,
        timeout_seconds=180.0,
    )
    assert len(result.auxiliary_results) == 3
    assert result.success, [
        r.last_error for r in result.auxiliary_results if not r.success
    ]

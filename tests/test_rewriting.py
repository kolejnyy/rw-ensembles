"""
Tests for rewriting canonicalization: filtering (filter_rewrites_by_namespace, namespace helpers)
and reranking (rerank_top_k_shortest_states, rerank_top_k_complexity_depth, get_reranking_heuristic).
"""

from __future__ import annotations

import pytest

from rwens.canonicalization.rewrites import (
    filter_rewrites_by_namespace,
    get_reranking_heuristic,
    rerank_top_k_complexity_depth,
    rerank_top_k_shortest_states,
)
from rwens.canonicalization.rewrites.cache import RewriteEntry
from rwens.canonicalization.rewrites.filters import (
    lemma_namespaces_from_tactic,
    namespaces_from_premise,
)


def test_filter_rewrites_by_namespace() -> None:
    """Verify that bare lemmas (add_comm) are allowed and qualified ones are filtered correctly."""
    allowed = ["Real", "Int", "Nat", "Complex", "Finset"]
    rewrites: list[RewriteEntry] = [
        RewriteEntry("rw [add_comm] at h₂", "a + b = b + a", None, None),
        RewriteEntry("rw [mul_comm] at h₂", "a * b = b * a", None, None),
        RewriteEntry("rw [← Int.natAbs_sq] at h₂", "Int.natAbs x ^ 2 = (Int.natAbs x) ^ 2", None, None),
        RewriteEntry("rw [← Polynomial.eval_monomial] at h₂", "Polynomial.eval x (Polynomial.monomial 2) 6 = ...", None, None),
        RewriteEntry("rw [add_comm, ← Int.mul_comm] at h₂", "y * x = x * y", None, None),
        RewriteEntry("rw [sq, ← RCLike.normSq_to_real] at h₀", "RCLike.normSq x = x * conj x", None, None),
        RewriteEntry("rw [sq, ← normSq_to_real] at h₀", "RCLike.normSq x = x * conj x", None, None),
    ]
    filtered = filter_rewrites_by_namespace(rewrites, allowed)
    tactics = [e.tactic for e in filtered]
    assert "rw [add_comm] at h₂" in tactics, "bare add_comm should be allowed"
    assert "rw [mul_comm] at h₂" in tactics, "bare mul_comm should be allowed"
    assert "rw [← Int.natAbs_sq] at h₂" in tactics, "Int.natAbs_sq should be allowed (Int in list)"
    assert "rw [← Polynomial.eval_monomial] at h₂" not in tactics, "Polynomial should be filtered out"
    assert "rw [add_comm, ← Int.mul_comm] at h₂" in tactics, "add_comm + Int allowed"
    assert "rw [sq, ← RCLike.normSq_to_real] at h₀" not in tactics, "RCLike in tactic should be filtered"
    assert "rw [sq, ← normSq_to_real] at h₀" not in tactics, "RCLike in premise should be filtered"


def test_filter_rewrites_by_namespace_mixed_disallowed() -> None:
    """Mix with disallowed namespace (e.g. Polynomial) should filter out the whole rewrite."""
    allowed = ["Real", "Int", "Nat", "Complex", "Finset"]
    rewrites2: list[RewriteEntry] = [
        RewriteEntry("rw [add_comm, ← Polynomial.foo] at h₂", "some state", None, None),
    ]
    filtered2 = filter_rewrites_by_namespace(rewrites2, allowed)
    assert len(filtered2) == 0, "mix with Polynomial should be filtered out"


# ---------- Filter helpers (lemma_namespaces_from_tactic, namespaces_from_premise) ----------


def test_lemma_namespaces_from_tactic_bare() -> None:
    """Bare lemmas (no dot) yield no namespaces."""
    assert lemma_namespaces_from_tactic("rw [add_comm] at h") == []
    assert lemma_namespaces_from_tactic("rw [mul_comm, add_comm] at h₂") == []


def test_lemma_namespaces_from_tactic_qualified() -> None:
    """Qualified lemmas yield their namespace prefix."""
    assert lemma_namespaces_from_tactic("rw [← Int.natAbs_sq] at h₂") == ["Int"]
    assert lemma_namespaces_from_tactic("rw [Polynomial.eval_monomial] at h") == ["Polynomial"]
    assert lemma_namespaces_from_tactic("rw [add_comm, ← Int.mul_comm] at h") == ["Int"]


def test_lemma_namespaces_from_tactic_not_rw() -> None:
    """Non-rw tactics yield no namespaces."""
    assert lemma_namespaces_from_tactic("simp only [foo]") == []


def test_namespaces_from_premise() -> None:
    """Premise type strings yield uppercase namespace prefixes."""
    assert namespaces_from_premise("Int.natAbs x ^ 2 = (Int.natAbs x) ^ 2") == ["Int", "Int"]
    assert namespaces_from_premise("RCLike.normSq x = x * conj x") == ["RCLike"]
    assert namespaces_from_premise("a + b = b + a") == []


# ---------- Reranking ----------


def test_rerank_top_k_shortest_states_orders_by_length() -> None:
    """Shortest state string first, then tie-break by tactic/type."""
    # StateCandidate = (state_str, rw_tactic, rw_type, complexity, depth)
    candidates: list = [
        ("long_state_here", "rw [foo]", "type_a", None, None),
        ("x", "rw [bar]", "type_b", None, None),
        ("mid", "rw [baz]", "type_c", None, None),
    ]
    out = rerank_top_k_shortest_states(candidates, 2)
    assert len(out) == 2
    assert out[0][0] == "x"
    assert out[1][0] == "mid"


def test_rerank_top_k_shortest_states_tie_break() -> None:
    """Same length: tie-break by tactic then type."""
    candidates = [
        ("ab", "rw [b]", "t2", None, None),
        ("ab", "rw [a]", "t1", None, None),
    ]
    out = rerank_top_k_shortest_states(candidates, 2)
    assert out[0][1] == "rw [a]"
    assert out[1][1] == "rw [b]"


def test_rerank_top_k_shortest_states_k_larger_than_list() -> None:
    """Requesting more than available returns all."""
    candidates = [("a", "", "", None, None), ("bb", "", "", None, None)]
    out = rerank_top_k_shortest_states(candidates, 10)
    assert len(out) == 2


def test_rerank_top_k_complexity_depth_orders_by_compl_plus_depth() -> None:
    """Lower complexity+depth first."""
    candidates = [
        ("s1", "t1", "ty1", 10, 2),   # 12
        ("s2", "t2", "ty2", 1, 1),    # 2
        ("s3", "t3", "ty3", 5, 0),    # 5
    ]
    out = rerank_top_k_complexity_depth(candidates, 2)
    assert len(out) == 2
    assert out[0][3] == 1 and out[0][4] == 1
    assert out[1][3] == 5 and out[1][4] == 0


def test_rerank_top_k_complexity_depth_none_treated_as_zero() -> None:
    """None complexity/depth treated as 0."""
    candidates = [
        ("long", "t", "ty", 100, None),
        ("x", "t", "ty", None, None),
    ]
    out = rerank_top_k_complexity_depth(candidates, 2)
    assert out[0][0] == "x"  # 0+0 then len tie-break
    assert out[1][0] == "long"


def test_rerank_top_k_complexity_depth_k_one() -> None:
    """Top-1 returns single best by complexity+depth."""
    candidates = [
        ("s1", "t1", "ty1", 3, 1),
        ("s2", "t2", "ty2", 1, 0),
    ]
    out = rerank_top_k_complexity_depth(candidates, 1)
    assert len(out) == 1
    assert out[0][3] == 1 and out[0][4] == 0


def test_get_reranking_heuristic_shortest_states() -> None:
    """get_reranking_heuristic with type shortest_states returns shortest-state reranker."""
    fn = get_reranking_heuristic({"type": "shortest_states"})
    candidates = [("bb", "t", "ty", None, None), ("a", "t", "ty", None, None)]
    out = fn(candidates, 1)
    assert len(out) == 1 and out[0][0] == "a"


def test_get_reranking_heuristic_complexity_depth() -> None:
    """get_reranking_heuristic with type complexity_depth returns complexity-depth reranker."""
    fn = get_reranking_heuristic({"type": "complexity_depth"})
    candidates = [("s1", "t", "ty", 10, 0), ("s2", "t", "ty", 1, 0)]
    out = fn(candidates, 1)
    assert len(out) == 1 and out[0][3] == 1


def test_get_reranking_heuristic_missing_type_raises() -> None:
    """get_reranking_heuristic raises when type is missing."""
    with pytest.raises(ValueError, match="reranking config must specify 'type'"):
        get_reranking_heuristic({})
    with pytest.raises(ValueError, match="reranking config must specify 'type'"):
        get_reranking_heuristic({"other": "key"})


def test_get_reranking_heuristic_unknown_type_raises() -> None:
    """get_reranking_heuristic raises for unknown type."""
    with pytest.raises(ValueError, match="Unknown reranking heuristic type"):
        get_reranking_heuristic({"type": "unknown_kind"})

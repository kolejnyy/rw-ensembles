"""
Reranking and other heuristics used in the rewriting pipeline (not the energy function).

Stage 2: narrow many candidate states to top-k, e.g. by shortest state length or complexity+depth.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

# Type: (state_str, rw_tactic, rw_type, complexity, depth) for one candidate. complexity/depth may be None.
StateCandidate = Tuple[str, str, str, Optional[int], Optional[int]]


def rerank_top_k_shortest_states(
    candidates: List[StateCandidate], k: int
) -> List[StateCandidate]:
    """
    Sort candidates by length of the state string (shortest first), then return top-k.
    """
    sorted_candidates = sorted(
        candidates, key=lambda x: (len(x[0]), x[1] or "", x[2] or "")
    )
    return sorted_candidates[:k]


def rerank_top_k_complexity_depth(
    candidates: List[StateCandidate], k: int
) -> List[StateCandidate]:
    """
    Sort candidates by (complexity + depth) ascending, then return top-k.
    Uses state length as tie-breaker when complexity/depth are None or equal.
    """
    def key(x: StateCandidate) -> Tuple[int, int, str, str]:
        compl = x[3] if x[3] is not None else 0
        depth = x[4] if x[4] is not None else 0
        return (compl + depth, len(x[0]), x[1] or "", x[2] or "")

    sorted_candidates = sorted(candidates, key=key)
    return sorted_candidates[:k]


def get_reranking_heuristic(
    config: Dict[str, Any],
) -> Callable[[List[StateCandidate], int], List[StateCandidate]]:
    """
    Build a reranking function from config.
    Config must contain type: "shortest_states" or "complexity_depth".
    """
    cfg = config or {}
    heuristic_type = cfg.get("type")
    if not heuristic_type:
        raise ValueError(
            "reranking config must specify 'type' (e.g. 'shortest_states', 'complexity_depth'). "
            "No default is applied."
        )
    if heuristic_type == "shortest_states":
        return rerank_top_k_shortest_states
    if heuristic_type == "complexity_depth":
        return rerank_top_k_complexity_depth
    raise ValueError(f"Unknown reranking heuristic type: {heuristic_type!r}")

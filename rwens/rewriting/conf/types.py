"""Configuration for rewriting modules."""

from typing import Any, Optional

from rwens.utils.applier import GOAL_TIMEOUT_SECONDS


class RewritingCanonicalizationConfig:
    """Configuration for RewritingCanonicalizationModule (rwcnc + LLM confidence).
    Requires an LLM instance at build time (passed by prover factory).
    Sampling params (max_per_step, depth, only_simplifying_rewrites, reverse_order)
    live under the "sampling" dict.
    """

    project_root: str
    timeout_seconds: float
    top_rewrites: int = 10
    filter_rewrite_namespaces: Optional[list] = None
    namespace_blacklist: Optional[list] = None
    sampling: Optional[dict] = None
    reranking: Optional[dict] = None
    energy: Optional[dict] = None

    def __init__(
        self,
        project_root: str,
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
        top_rewrites: int = 10,
        filter_rewrite_namespaces: Optional[list] = None,
        namespace_blacklist: Optional[list] = None,
        sampling: Optional[dict] = None,
        reranking: Optional[dict] = None,
        energy: Optional[dict] = None,
        confidence_aggregation: Optional[str] = None,
        max_per_step: Optional[int] = None,
        depth: Optional[int] = None,
        reverse_order: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.project_root = project_root
        self.timeout_seconds = timeout_seconds
        self.top_rewrites = top_rewrites
        self.filter_rewrite_namespaces = filter_rewrite_namespaces
        self.namespace_blacklist = namespace_blacklist
        self.reranking = reranking
        self.sampling = dict(sampling) if sampling else {}
        if max_per_step is not None and "max_per_step" not in self.sampling:
            self.sampling["max_per_step"] = max_per_step
        if depth is not None and "depth" not in self.sampling:
            self.sampling["depth"] = depth
        if reverse_order is not None and "reverse_order" not in self.sampling:
            self.sampling["reverse_order"] = reverse_order
        self.energy = dict(energy) if energy else {}
        if confidence_aggregation and "confidence_aggregation" not in self.energy:
            self.energy.setdefault("type", "confidence")
            self.energy["confidence_aggregation"] = confidence_aggregation


# Alias kept for prover YAML (`canonicalization:` block) and model config types.
CanonicalizationConfig = RewritingCanonicalizationConfig

"""
Rewriting helpers: cache, heuristics, energy, filters.

Use get_energy_heuristic(config, project_root) and get_reranking_heuristic(config)
to build pipeline components. Cache keys and structured entries live in cache;
filtering in filters.
"""

from invpro.canonicalization.rewrites.cache import (
    CacheEntry,
    ConfidenceCacheEntry,
    FirstTacticCacheEntry,
    GetStatesCacheEntry,
    RewriteEntry,
    TheoremSurpriseCacheEntry,
    cache_config_tag_from_energy,
    confidence_cache_key,
    first_tactic_cache_key,
    get_states_cache_key,
    load_get_states_entry,
    load_rewrites_cache,
    read_cache,
    rewrites_cache_key,
    save_get_states_entry,
    save_rewrites_cache,
    theorem_surprise_cache_key,
)
from invpro.canonicalization.rewrites.energy import (
    get_energy_heuristic,
    make_confidence_energy_cached,
    make_confidence_energy_uncached,
    make_single_pass_confidence_energy,
    make_theorem_surprise_energy,
)
from invpro.canonicalization.rewrites.filters import (
    filter_rewrites_by_namespace,
    filter_rewrites_by_namespace_blacklist,
    lemma_namespaces_from_tactic,
    namespaces_from_premise,
)
from invpro.canonicalization.rewrites.energy import aggregate_probs
from invpro.canonicalization.rewrites.heuristics import (
    StateCandidate,
    get_reranking_heuristic,
    rerank_top_k_complexity_depth,
    rerank_top_k_shortest_states,
)

__all__ = [
    # cache
    "CacheEntry",
    "ConfidenceCacheEntry",
    "FirstTacticCacheEntry",
    "GetStatesCacheEntry",
    "RewriteEntry",
    "TheoremSurpriseCacheEntry",
    "cache_config_tag_from_energy",
    "confidence_cache_key",
    "first_tactic_cache_key",
    "get_states_cache_key",
    "load_get_states_entry",
    "load_rewrites_cache",
    "read_cache",
    "rewrites_cache_key",
    "save_get_states_entry",
    "save_rewrites_cache",
    "theorem_surprise_cache_key",
    # energy
    "get_energy_heuristic",
    "make_confidence_energy_cached",
    "make_confidence_energy_uncached",
    "make_single_pass_confidence_energy",
    "make_theorem_surprise_energy",
    # filters
    "filter_rewrites_by_namespace",
    "filter_rewrites_by_namespace_blacklist",
    "lemma_namespaces_from_tactic",
    "namespaces_from_premise",
    # heuristics
    "StateCandidate",
    "aggregate_probs",
    "get_reranking_heuristic",
    "rerank_top_k_complexity_depth",
    "rerank_top_k_shortest_states",
]

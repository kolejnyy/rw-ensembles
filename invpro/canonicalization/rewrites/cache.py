"""
Cache keys, entry types, and load/save helpers for rewriting and energy caches.

Provides structured dataclasses for cache contents (get_states, rewrites, confidence)
so callers work with typed objects instead of raw tuples/dicts.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Energy type tags for cache key blobs (so caches are not reused across energy functions).
ENERGY_TAG_CONFIDENCE = "confidence"
ENERGY_TAG_FIRST_TACTIC = "first_tactic_confidence"
ENERGY_TAG_THEOREM_SURPRISE = "theorem_surprise"

# Prefix for rwcnc rewrites cache files in .cache/rw (consistent with gs_, ap_, etc.).
PREFIX_REWRITES = "rw_"


# ---------- Cache key functions ----------


def rewrites_cache_key(
    content: str,
    depth: int,
    max_per_step: int,
    only_simplify: bool = False,
    reverse_order: bool = False,
) -> str:
    """Stable hash for rwcnc config (state, depth, max_per_step, simplify, reverse)."""
    blob = (
        content
        + "\n---\n"
        + str(depth)
        + "\n---\n"
        + str(max_per_step)
        + "\n---\n"
        + ("true" if only_simplify else "false")
        + "\n---\n"
        + ("true" if reverse_order else "false")
    ).encode("utf-8")
    return PREFIX_REWRITES + hashlib.sha256(blob).hexdigest()


def cache_config_tag_from_energy(energy_cfg: Optional[dict]) -> str:
    """Stable string identifying the energy/config for get_states cache key."""
    if energy_cfg is None:
        return "energy:none"
    if not isinstance(energy_cfg, dict):
        return "energy:unknown"
    t = energy_cfg.get("type") or "none"
    if t == "none":
        return "energy:none"
    if t == "theorem_surprise":
        return "energy:theorem_surprise"
    if t == "confidence":
        agg = energy_cfg.get("confidence_aggregation", "mean")
        return f"energy:confidence:{agg}"
    return f"energy:{t}"


def get_states_cache_key(
    content: str,
    depth: int,
    max_per_step: int,
    only_simplify: bool,
    reverse_order: bool,
    top_rewrites: int,
    filter_namespaces: Optional[List[str]],
    model_cache_id: str = "",
    cache_config_tag: str = "",
) -> str:
    """Stable hash for get_states result cache."""
    ns_part = (
        ",".join(sorted(filter_namespaces))
        if filter_namespaces
        else ""
    )
    blob = (
        content
        + "\n---\n"
        + str(depth)
        + "\n---\n"
        + str(max_per_step)
        + "\n---\n"
        + ("true" if only_simplify else "false")
        + "\n---\n"
        + ("true" if reverse_order else "false")
        + "\n---\n"
        + str(top_rewrites)
        + "\n---\n"
        + ns_part
        + "\n---\n"
        + (model_cache_id or "")
        + "\n---\n"
        + (cache_config_tag or "")
    ).encode("utf-8")
    return "gs_" + hashlib.sha256(blob).hexdigest()


def confidence_cache_key(
    model_cache_id: str, statement: str, confidence_aggregation: str
) -> str:
    """Stable hash for (model_id, energy_type, confidence_aggregation, statement)."""
    blob = (
        model_cache_id
        + "\n"
        + ENERGY_TAG_CONFIDENCE
        + "\n"
        + confidence_aggregation
        + "\n"
        + statement
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def first_tactic_cache_key(model_cache_id: str, problem_stmt: str) -> str:
    """Stable hash for (model_id, energy_type, problem_stmt) for first-tactic cache."""
    blob = (
        model_cache_id + "\n" + ENERGY_TAG_FIRST_TACTIC + "\n" + problem_stmt
    ).encode("utf-8")
    return "ft_" + hashlib.sha256(blob).hexdigest()


def theorem_surprise_cache_key(model_cache_id: str, problem_stmt: str) -> str:
    """Stable hash for (model_id, energy_type, problem_stmt) for theorem-surprise cache."""
    blob = (
        model_cache_id + "\n" + ENERGY_TAG_THEOREM_SURPRISE + "\n" + problem_stmt
    ).encode("utf-8")
    return "ts_" + hashlib.sha256(blob).hexdigest()


# ---------- Structured cache entries ----------

# Prefixes used in cache filenames (stem) to identify entry type for read_cache.
PREFIX_GET_STATES = "gs_"
PREFIX_FIRST_TACTIC = "ft_"
PREFIX_THEOREM_SURPRISE = "ts_"
# Confidence cache uses raw hash (no prefix).


class CacheEntry(ABC):
    """Base class for all cache entry types (file-level cache contents)."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Parse a cache entry from JSON-loaded dict. Override in subclasses."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dict for JSON. Override in subclasses."""
        raise NotImplementedError


@dataclass
class RewriteEntry(CacheEntry):
    """Single rewrite: tactic, premise/type, optional complexity and depth."""

    tactic: str
    premise: str = ""
    complexity: Optional[int] = None
    depth: Optional[int] = None

    def to_tuple(self) -> tuple:
        return (self.tactic, self.premise, self.complexity, self.depth)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tactic": self.tactic,
            "premise": self.premise,
            "complexity": self.complexity,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewriteEntry":
        return cls(
            tactic=data.get("tactic", ""),
            premise=data.get("premise", ""),
            complexity=data.get("complexity"),
            depth=data.get("depth"),
        )

    @classmethod
    def from_tuple(cls, t: tuple) -> "RewriteEntry":
        if len(t) >= 4:
            return cls(t[0], t[1], t[2], t[3])
        if len(t) >= 2:
            return cls(t[0], t[1], None, None)
        return cls(t[0] if t else "", "")


@dataclass
class GetStatesCacheEntry(CacheEntry):
    """Cached result of get_states: current state, chosen best state, and rw tactics."""

    current: str
    best_state: str
    rw_tactics: List[str] = field(default_factory=list)
    use_original_problem: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": self.current,
            "best_state": self.best_state,
            "rw_tactics": self.rw_tactics,
            "use_original_problem": self.use_original_problem,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GetStatesCacheEntry":
        return cls(
            current=data.get("current", ""),
            best_state=data.get("best_state", ""),
            rw_tactics=data.get("rw_tactics", []),
            use_original_problem=data.get("use_original_problem", True),
        )


@dataclass
class ConfidenceCacheEntry(CacheEntry):
    """Step-by-step confidence cache: tactic and aggregated confidence."""

    tactic: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"tactic": self.tactic, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfidenceCacheEntry":
        return cls(
            tactic=data.get("tactic", ""),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class FirstTacticCacheEntry(CacheEntry):
    """First-tactic confidence cache entry."""

    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FirstTacticCacheEntry":
        return cls(confidence=float(data.get("confidence", 0.0)))


@dataclass
class TheoremSurpriseCacheEntry(CacheEntry):
    """Theorem-surprise cache entry (mean log P)."""

    mean_log_p: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"mean_log_p": self.mean_log_p}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TheoremSurpriseCacheEntry":
        return cls(mean_log_p=float(data.get("mean_log_p", 0.0)))


# ---------- Load / save helpers ----------


def load_rewrites_cache(
    cache_dir: Path,
    content: str,
    depth: int,
    max_per_step: int,
    only_simplify: bool = False,
    reverse_order: bool = False,
) -> Optional[Dict[str, List[RewriteEntry]]]:
    """
    Load rewrites cache if present. Returns dict[hyp -> list[RewriteEntry]] or None.
    Supports legacy 2-tuple entries (tactic, type_str) -> RewriteEntry(tactic, type_str, None, None).
    """
    key = rewrites_cache_key(content, depth, max_per_step, only_simplify, reverse_order)
    path = cache_dir / f"{key}.cache"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, List[RewriteEntry]] = {}
        for hyp, pairs in data.items():
            out[hyp] = [
                RewriteEntry.from_tuple(
                    (p[0], p[1], p[2], p[3]) if len(p) >= 4 else (p[0], p[1], None, None)
                )
                for p in pairs
            ]
        return out
    except (json.JSONDecodeError, OSError):
        return None


def save_rewrites_cache(
    cache_dir: Path,
    content: str,
    depth: int,
    max_per_step: int,
    result: Dict[str, List[RewriteEntry]],
    only_simplify: bool = False,
    reverse_order: bool = False,
) -> None:
    """Save rewrites result to cache_dir. Uses UTF-8."""
    key = rewrites_cache_key(content, depth, max_per_step, only_simplify, reverse_order)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.cache"
    data = {
        hyp: [list(entry.to_tuple()) for entry in entries]
        for hyp, entries in result.items()
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass


def load_get_states_entry(path: Path) -> Optional[GetStatesCacheEntry]:
    """Load get_states cache entry from path. Returns None on error."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return GetStatesCacheEntry.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def save_get_states_entry(path: Path, entry: GetStatesCacheEntry) -> None:
    """Write get_states cache entry to path. Uses UTF-8."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(entry.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def read_cache(path: Path) -> Optional[CacheEntry]:
    """
    Read a cache file and return the appropriate CacheEntry instance based on filename prefix.

    Uses the file stem (filename without extension) to decide the type:
    - gs_*  -> GetStatesCacheEntry
    - ft_*  -> FirstTacticCacheEntry
    - ts_*  -> TheoremSurpriseCacheEntry
    - no prefix (plain hash) -> ConfidenceCacheEntry

    Returns None if the file does not exist or parsing fails.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stem = path.stem
    if stem.startswith(PREFIX_GET_STATES):
        return GetStatesCacheEntry.from_dict(data)
    if stem.startswith(PREFIX_FIRST_TACTIC):
        return FirstTacticCacheEntry.from_dict(data)
    if stem.startswith(PREFIX_THEOREM_SURPRISE):
        return TheoremSurpriseCacheEntry.from_dict(data)
    # Plain hash (confidence cache)
    return ConfidenceCacheEntry.from_dict(data)

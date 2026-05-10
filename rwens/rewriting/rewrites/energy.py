"""
Energy function for the rewriting pipeline: score problem statements (e.g. confidence, theorem surprise).

Energy has signature (llm, statement: str) -> float. Caller converts states to statements before calling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from rwens.dataset.utils import split_declarations_theorem_proof
from rwens.models.llm.base import BaseLLM
from rwens.utils.cache_paths import get_conf_cache_dir

from rwens.rewriting.rewrites.cache import (
    confidence_cache_key,
    first_tactic_cache_key,
    theorem_surprise_cache_key,
)


def _probs_to_list(probs: Union[List[float], Any]) -> List[float]:
    """Convert probs (list, tensor, or array-like) to a list of floats."""
    if isinstance(probs, list):
        return [float(p) for p in probs]
    if hasattr(probs, "tolist"):
        return [float(p) for p in probs.tolist()]
    return [float(p) for p in probs]


def aggregate_probs(
    probs: Union[List[float], Any],
    aggregation: str = "mean",
) -> float:
    """
    Aggregate per-token probabilities into a single score.
    aggregation: "mean" (arithmetic mean) or "log_mean" (mean of log(prob)).
    """
    plist = _probs_to_list(probs)
    if not plist:
        return 0.0
    if aggregation == "log_mean":
        return sum(math.log(max(p, 1e-10)) for p in plist) / len(plist)
    return sum(plist) / len(plist)


def _extract_first_tactic_confidence_impl(
    llm: BaseLLM,
    problem_statement: str,
    prompt_formatter: Optional[Any],
    verbose: bool = False,
) -> float:
    """
    Extract average token confidence for the first tactic from a single-pass model.
    When verbose=True, passes verbose to the LLM's generate_first_tactic_confidence.
    """
    if prompt_formatter is None:
        return 0.0
    try:
        prompt = prompt_formatter.format(problem_statement)
    except Exception:
        return 0.0
    if hasattr(llm, "generate_first_tactic_confidence"):
        try:
            return llm.generate_first_tactic_confidence(prompt, verbose=verbose)
        except Exception:
            return 0.0
    return 0.0


def _extract_theorem_surprise_impl(
    llm: BaseLLM,
    problem_statement: str,
    prompt_formatter: Optional[Any],
    verbose: bool = False,
) -> float:
    """
    Measure the surprise of the model when "generating" the theorem content.
    Returns mean(log P(token)) over theorem tokens (higher = less surprise).
    Returns -inf when required methods or data are missing.
    """
    if prompt_formatter is None:
        return float("-inf")
    if not hasattr(prompt_formatter, "get_user_prefix_before_theorem"):
        return float("-inf")
    if not hasattr(llm, "get_log_probs_for_continuation"):
        return float("-inf")
    try:
        decls, theorem_stmt, _ = split_declarations_theorem_proof(problem_statement)
    except ValueError:
        return float("-inf")
    instruction_prefix = prompt_formatter.get_user_prefix_before_theorem()
    prefix = instruction_prefix + decls if decls.strip() else instruction_prefix
    log_probs = llm.get_log_probs_for_continuation(prefix, theorem_stmt, verbose=verbose)
    if not log_probs:
        return float("-inf")
    mean_log_p = sum(log_probs) / len(log_probs)
    return mean_log_p


def _extract_first_tactic_confidence(
    llm: BaseLLM,
    problem_statement: str,
    prompt_formatter: Optional[Any],
    cache_dir: Optional[Path] = None,
) -> float:
    """
    Extract first-tactic confidence, with optional cache. When cache_dir is set
    and llm has get_model_cache_id, uses read-through/write-through cache.
    """
    cache_id = getattr(llm, "get_model_cache_id", None) and (llm.get_model_cache_id() or "")
    if cache_dir and cache_id:
        key = first_tactic_cache_key(cache_id, problem_statement)
        path = cache_dir / f"{key}.cache"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return float(data["confidence"])
            except (OSError, KeyError, TypeError, ValueError):
                pass
    score = _extract_first_tactic_confidence_impl(llm, problem_statement, prompt_formatter)
    if cache_dir and cache_id:
        try:
            key = first_tactic_cache_key(cache_id, problem_statement)
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / f"{key}.cache"
            path.write_text(
                json.dumps({"confidence": score}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
    return score


def make_confidence_energy_cached(
    project_root: str,
    confidence_aggregation: str = "mean",
) -> Callable[[BaseLLM, str], float]:
    """
    Return an energy function (llm, statement) -> float that uses model confidence,
    with read-through/write-through cache under project_root/.cache/conf.
    """
    cache_dir = get_conf_cache_dir(project_root)

    def energy(llm: BaseLLM, statement: str) -> float:
        get_id = getattr(llm, "get_model_cache_id", None)
        cache_id = get_id() if get_id is not None else None
        if cache_id:
            key = confidence_cache_key(cache_id, statement, confidence_aggregation)
            path = cache_dir / f"{key}.cache"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return float(data["confidence"])
                except (OSError, KeyError, TypeError, ValueError):
                    pass
        if hasattr(llm, "generate_greedy_with_confidence"):
            tactic, probs = llm.generate_greedy_with_confidence(statement)
            conf = aggregate_probs(probs, confidence_aggregation)
            if cache_id:
                cache_dir.mkdir(parents=True, exist_ok=True)
                path = cache_dir / f"{key}.cache"
                path.write_text(
                    json.dumps(
                        {"tactic": tactic, "confidence": conf},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            return conf
        return 0.0

    return energy


def make_confidence_energy_uncached(
    confidence_aggregation: str = "mean",
) -> Callable[[BaseLLM, str], float]:
    """Return an energy function (llm, statement) -> float with no cache."""
    def energy(llm: BaseLLM, statement: str) -> float:
        if hasattr(llm, "generate_greedy_with_confidence"):
            _, probs = llm.generate_greedy_with_confidence(statement)
            return aggregate_probs(probs, confidence_aggregation)
        return 0.0
    return energy


def make_single_pass_confidence_energy(
    project_root: Optional[str] = None,
    prompt_formatter: Optional[Any] = None,
    extract_confidence_fn: Optional[
        Callable[[BaseLLM, str, Optional[Any]], float]
    ] = None,
    llm: Optional[BaseLLM] = None,
    verbose: bool = False,
) -> Callable[[BaseLLM, str], float]:
    """
    Return an energy function (llm, statement) -> float for single-pass models.
    Scores a problem statement by confidence in the first tactic.
    """
    cache_dir = (
        get_conf_cache_dir(project_root) if project_root else None
    )
    _raw_extract = extract_confidence_fn
    _verbose = verbose
    if _raw_extract is None:
        _raw_extract = lambda llm, prob, pf: _extract_first_tactic_confidence_impl(
            llm, prob, pf, verbose=_verbose
        )
    _llm = llm

    def _cached_extract(llm: BaseLLM, problem: str, pf: Optional[Any]) -> float:
        cache_id = getattr(llm, "get_model_cache_id", None) and (
            llm.get_model_cache_id() or ""
        )
        if cache_dir and cache_id:
            key = first_tactic_cache_key(cache_id, problem)
            path = cache_dir / f"{key}.cache"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return float(data["confidence"])
                except (OSError, KeyError, TypeError, ValueError):
                    pass
        score = _raw_extract(llm, problem, pf)
        if cache_dir and cache_id:
            try:
                key = first_tactic_cache_key(cache_id, problem)
                cache_dir.mkdir(parents=True, exist_ok=True)
                path = cache_dir / f"{key}.cache"
                path.write_text(
                    json.dumps({"confidence": score}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass
        return score

    def energy(call_llm: BaseLLM, statement: str) -> float:
        model = _llm if _llm is not None else call_llm
        if not statement or not statement.strip():
            return 0.0
        return _cached_extract(model, statement, prompt_formatter)

    return energy


def make_theorem_surprise_energy(
    project_root: Optional[str] = None,
    prompt_formatter: Optional[Any] = None,
    verbose: bool = False,
    use_cache: bool = True,
    llm: Optional[BaseLLM] = None,
) -> Callable[[BaseLLM, str], float]:
    """
    Return an energy function (llm, statement) -> float that scores statements by theorem surprise.
    Computes mean(log P(theorem | prefix)); higher = less surprise.
    """
    cache_dir = (
        get_conf_cache_dir(project_root) if (project_root and use_cache) else None
    )
    _verbose = verbose
    _llm = llm

    def _cached_extract(callee_llm: BaseLLM, problem: str) -> float:
        cache_id = getattr(callee_llm, "get_model_cache_id", None) and (
            callee_llm.get_model_cache_id() or ""
        )
        if cache_dir and cache_id:
            key = theorem_surprise_cache_key(cache_id, problem)
            path = cache_dir / f"{key}.cache"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return float(data["mean_log_p"])
                except (OSError, KeyError, TypeError, ValueError):
                    pass
        score = _extract_theorem_surprise_impl(
            callee_llm, problem, prompt_formatter, verbose=_verbose
        )
        if cache_dir and cache_id and score > float("-inf"):
            try:
                key = theorem_surprise_cache_key(cache_id, problem)
                cache_dir.mkdir(parents=True, exist_ok=True)
                path = cache_dir / f"{key}.cache"
                path.write_text(
                    json.dumps({"mean_log_p": score}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass
        return score

    def energy(call_llm: BaseLLM, statement: str) -> float:
        model = _llm if _llm is not None else call_llm
        if not statement or not statement.strip():
            return float("-inf")
        return _cached_extract(model, statement)

    return energy


def get_energy_heuristic(
    config: Dict[str, Any],
    project_root: Optional[str] = None,
    module: Optional[Any] = None,
) -> Callable[[BaseLLM, str], float]:
    """
    Build an energy function (llm, statement) -> float from config.
    Config keys: type ("confidence" | "single_pass_confidence"), confidence_aggregation ("mean" | "log_mean").
    Energy scores problem statements; caller converts states to statements before calling.
    """
    cfg = config or {}
    heuristic_type = cfg.get("type", "confidence")
    agg = cfg.get("confidence_aggregation", "mean")
    if heuristic_type == "confidence":
        if project_root:
            return make_confidence_energy_cached(
                project_root,
                confidence_aggregation=agg,
            )
        return make_confidence_energy_uncached(confidence_aggregation=agg)
    if heuristic_type == "single_pass_confidence":
        return make_single_pass_confidence_energy(
            project_root=project_root,
            prompt_formatter=cfg.get("prompt_formatter"),
            extract_confidence_fn=cfg.get("extract_confidence_fn"),
        )
    raise ValueError(f"Unknown energy heuristic type: {heuristic_type!r}")

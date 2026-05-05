"""Core data loading and statistical computation for reverification analysis.

Terminology
-----------
CP  : correctness probability -- the empirical fraction of LLM proof attempts
      that Lean accepted, estimated from the reverification JSON files.
SR  : success rate under a given budget B -- the probability that at least one
      of B independent proof attempts succeeds.  For a single variant with CP c:
          SR(c, B) = 1 - (1 - c)^B
      For an ensemble of k variants each receiving B/k attempts:
          SR(c_1..c_k, B) = 1 - prod_i (1 - c_i)^(B/k)

Configurations
--------------
seed       : use only the original minif2f statement (no variants)
random     : pick one variant uniformly at random (averaged analytically)
ensemble-k : pick k variants uniformly at random, split budget equally
ensemble-1+j: force-include the original + pick j additional variants at random

Variance / std convention
-------------------------
* **seed** : one SR per problem; ``std_sr`` is undefined (NaN) — no ± in tables.
* **random** : for each problem, SR values are computed for every statement in
  the pool; the reported mean is their average.  The reported ``std_sr`` is the
  mean across problems of the sample standard deviation of those per-pool SRs
  (NaN when every pool has size 1).
* **ensemble** : for each problem, ``n_mc`` Monte Carlo draws give one SR each;
  the reported mean is the mean of per-problem MC means; ``std_sr`` is the mean
  across problems of the sample std of those ``n_mc`` SRs (NaN when the draw is
  deterministic — no variant sampling).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

VARIANT_RE = re.compile(r"^(?P<base>.+)_v(?P<num>\d+)(?P<renamed>_renamed)?$")

BUDGETS: list[int] = [1, 4, 8, 16, 32, 64]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class SplitStats(NamedTuple):
    """avg SR and spread metric for one (config, budget) cell (see module doc)."""

    avg_sr: float
    std_sr: float  # NaN when no ± is shown (e.g. seed)
    n_problems: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _sr_from_row(row: dict) -> float | None:
    successes = row.get("successes")
    if isinstance(successes, list) and successes:
        return sum(bool(x) for x in successes) / len(successes)
    vsc = row.get("verified_success_count")
    n = row.get("n_attempts")
    if isinstance(vsc, int) and isinstance(n, int) and n > 0:
        return vsc / n
    return None


def load_cp_rates(reverification_dir: Path) -> dict[str, float]:
    """Return {theorem_name: CP} from all JSON files in *reverification_dir*."""
    rates: dict[str, float] = {}
    for p in sorted(reverification_dir.glob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = str(row.get("problem") or p.stem)
        sr = _sr_from_row(row)
        if sr is None:
            continue
        rates[name] = sr
    return rates


def group_by_base(rates: dict[str, float]) -> dict[str, dict]:
    """Group CP values by base problem name.

    Returns
    -------
    dict mapping base_name -> {'original': float|None, 'variants': list[float]}

    'original' is the CP for the theorem that has no _vN suffix.
    'variants' are the CPs of all _vN (and _vN_renamed) forms.
    """
    groups: dict[str, dict] = {}
    for name, cp in rates.items():
        m = VARIANT_RE.match(name)
        base = str(m.group("base")) if m else name
        if base not in groups:
            groups[base] = {"original": None, "variants": []}
        if m:
            groups[base]["variants"].append(cp)
        else:
            groups[base]["original"] = cp
    return groups


# ---------------------------------------------------------------------------
# Core SR / variance helpers
# ---------------------------------------------------------------------------


def sr_from_cp(cp: float, budget: float) -> float:
    """Success rate: probability at least one of *budget* attempts succeeds."""
    return 1.0 - (1.0 - cp) ** budget


# ---------------------------------------------------------------------------
# Config implementations
# ---------------------------------------------------------------------------


def compute_seed_stats(groups: dict[str, dict], budget: int) -> SplitStats:
    """Exact computation for the 'seed' (original statement) configuration."""
    srs: list[float] = []
    for data in groups.values():
        cp = data["original"]
        if cp is None:
            continue
        srs.append(sr_from_cp(cp, budget))

    if not srs:
        return SplitStats(float("nan"), float("nan"), 0)

    N = len(srs)
    avg = sum(srs) / N
    return SplitStats(avg, float("nan"), N)


def compute_random_stats(groups: dict[str, dict], budget: int) -> SplitStats:
    """Mean SR per problem (uniform over pool); std = mean of per-pool std of SRs."""
    problem_means: list[float] = []
    problem_stds: list[float] = []

    for data in groups.values():
        pool = list(data["variants"])
        if data["original"] is not None:
            pool.append(data["original"])
        if not pool:
            continue
        srs = [sr_from_cp(cp, budget) for cp in pool]
        K = len(srs)
        problem_means.append(sum(srs) / K)
        if K > 1:
            problem_stds.append(float(np.std(srs, ddof=1)))
        else:
            problem_stds.append(float("nan"))

    if not problem_means:
        return SplitStats(float("nan"), float("nan"), 0)

    N = len(problem_means)
    avg = sum(problem_means) / N
    agg_std = float(np.nanmean(problem_stds))
    return SplitStats(avg, agg_std, N)


def ensemble_sr_mean_std_one_problem(
    fixed: np.ndarray,
    pool: np.ndarray,
    k_sample: int,
    budget: int,
    n_mc: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Per-problem ensemble SR: MC mean and sample std over ``n_mc`` draws; else (sr, nan)."""
    K = len(pool)
    effective_k = len(fixed) + k_sample
    q, r = divmod(budget, effective_k)
    base_exps = np.where(np.arange(effective_k) < r, q + 1, q).astype(np.float64)

    if k_sample == 0 or k_sample == K:
        all_cps = np.concatenate([fixed, pool]) if K > 0 else fixed.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_terms = np.log1p(-all_cps) * base_exps
        sr = 1.0 - np.exp(float(np.nansum(log_terms)))
        return (float(sr), float("nan"))

    noise = rng.random((n_mc, K))
    order = np.argsort(noise, axis=1)[:, :k_sample]
    chosen = pool[order]
    if len(fixed) > 0:
        all_cps = np.concatenate([np.broadcast_to(fixed, (n_mc, len(fixed))), chosen], axis=1)
    else:
        all_cps = chosen
    with np.errstate(divide="ignore", invalid="ignore"):
        log_fail = np.nansum(np.log1p(-all_cps) * base_exps, axis=1)
    srs = 1.0 - np.exp(log_fail)
    return (float(np.mean(srs)), float(np.std(srs, ddof=1)))


def compute_ensemble_stats(
    groups: dict[str, dict],
    budget: int,
    n_variants: int,
    include_seed: bool,
    n_mc: int = 20_000,
    rng: np.random.Generator | None = None,
) -> SplitStats:
    """Monte Carlo approximation for ensemble configurations.

    Parameters
    ----------
    n_variants   : nominal ensemble size (includes the seed slot when include_seed)
    include_seed : if True, the original is always in the ensemble and we draw
                   up to n_variants-1 additional variants at random from the
                   variants pool.
    n_mc         : number of Monte Carlo draws per problem

    Budget allocation
    -----------------
    Let k = effective ensemble size for a problem.  Each variant receives
    q = budget // k attempts; the remaining r = budget % k attempts are
    distributed one-each to r randomly chosen variants, so every variant
    gets either q or q+1 attempts.  In the MC path the random assignment is
    implicit: the sampled variants are already in a random order, so the
    first r positions in that order receive the extra attempt.

    Pool composition
    ----------------
    include_seed=False : pool = original (if present) + all variants
    include_seed=True  : original is always used; pool = all variants

    Undersized pools
    ----------------
    When a problem's pool has fewer items than needed to fill the ensemble,
    we use *all* available items and distribute the budget equally among them
    (so effective budget-per-variant = budget / actual_k).  This means no
    problem is dropped; the randomness in variant selection simply disappears
    for those problems and they contribute a deterministic SR.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Separate problems into deterministic (pool too small to sample) and MC.
    # Each entry: (fixed_cps_array, pool_for_sampling, k_to_sample)
    #   fixed_cps_array  : CPs that are always in the ensemble (may be empty)
    #   pool_for_sampling: CPs to sample from (may be empty → deterministic)
    #   k_to_sample      : how many to draw from pool_for_sampling
    ProblemEntry = tuple[np.ndarray, np.ndarray, int]
    problem_list: list[ProblemEntry] = []

    for data in groups.values():
        variant_arr = np.asarray(data["variants"], dtype=np.float64)
        orig_cp = data["original"]

        if include_seed:
            if orig_cp is None:
                continue  # can't force-include a seed that doesn't exist
            fixed = np.array([orig_cp], dtype=np.float64)
            pool = variant_arr
            k_need = n_variants - 1
        else:
            # Pool includes original (if present) + all variants
            pool_list = list(variant_arr)
            if orig_cp is not None:
                pool_list = [orig_cp] + pool_list
            if not pool_list:
                continue
            fixed = np.empty(0, dtype=np.float64)
            pool = np.asarray(pool_list, dtype=np.float64)
            k_need = n_variants

        k_avail = len(pool)
        k_sample = min(k_need, k_avail)  # how many we will actually draw
        problem_list.append((fixed, pool, k_sample))

    if not problem_list:
        return SplitStats(float("nan"), float("nan"), 0)

    N = len(problem_list)
    per_mean: list[float] = []
    per_std: list[float] = []

    for fixed, pool, k_sample in problem_list:
        m, s = ensemble_sr_mean_std_one_problem(fixed, pool, k_sample, budget, n_mc, rng)
        per_mean.append(m)
        per_std.append(s)

    avg_sr = float(np.mean(per_mean))
    agg_std = float(np.nanmean(per_std))
    return SplitStats(avg_sr, agg_std, N)

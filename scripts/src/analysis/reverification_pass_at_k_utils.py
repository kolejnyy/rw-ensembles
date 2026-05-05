"""Pass@k analysis from reverification JSON (hypergeometric sampling, no replacement).

Terminology
-----------
For a single statement with n verifier attempts and c successes, pass@b is the
probability that a uniformly random subset of b distinct attempts contains at
least one success:

    pass@b = 1 - C(n-c, b) / C(n, b)

Edge case (per project convention): if c > 0 and b > n, pass@b = 1.

Ensembles
---------
As in ``reverification_utils`` for SR, the total budget b is split across k
selected variants (q = b // k, remainder r gives r variants one extra draw).
Segments are treated as independent for the purpose of pass@k:

    pass = 1 - prod_i [ C(n_i - c_i, b_i) / C(n_i, b_i) ]

where (n_i, c_i) come from each variant's reverification JSON and b_i is the
integer draw count for that variant (same allocation as SR analysis).

Variance / std (random & ensemble rows): ``n_mc`` **simulated runs**. Each run picks
independently per problem (one random pool variant for *random*; one random variant
subset for *ensemble*), computes pass@b per problem, then **averages over problems**
to get one scalar score. Reported ``avg_pass`` / ``std_pass`` are the mean and
sample standard deviation (``ddof=1``) of those run scores across simulations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np

from reverification_utils import VARIANT_RE


class PassAtKSplitStats(NamedTuple):
    """Mean pass@b and spread metric for one (config, budget) cell."""

    avg_pass: float
    std_pass: float  # NaN when no ± is shown (e.g. seed)
    n_problems: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _counts_from_row(row: dict) -> tuple[int, int] | None:
    """Return (n_attempts, success_count) if the row supports pass@k."""
    n = row.get("n_attempts")
    if not isinstance(n, int) or n <= 0:
        return None
    successes = row.get("successes")
    if isinstance(successes, list) and successes:
        c = sum(1 for x in successes if x)
    else:
        vsc = row.get("verified_success_count")
        if not isinstance(vsc, int):
            return None
        c = vsc
    if c < 0 or c > n:
        return None
    return n, c


def load_attempt_counts(reverification_dir: Path) -> dict[str, tuple[int, int]]:
    """Return {theorem_name: (n, c)} from all JSON files in *reverification_dir*."""
    out: dict[str, tuple[int, int]] = {}
    for p in sorted(reverification_dir.glob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = str(row.get("problem") or p.stem)
        cnt = _counts_from_row(row)
        if cnt is None:
            continue
        out[name] = cnt
    return out


def group_counts_by_base(counts: dict[str, tuple[int, int]]) -> dict[str, dict]:
    """Group (n,c) by base problem (same naming rules as ``group_by_base``).

    Returns
    -------
    dict mapping base_name -> {'original': (n,c)|None, 'variants': list[(n,c)]}
    """
    groups: dict[str, dict] = {}
    for name, nc in counts.items():
        m = VARIANT_RE.match(name)
        base = str(m.group("base")) if m else name
        if base not in groups:
            groups[base] = {"original": None, "variants": []}
        if m:
            groups[base]["variants"].append(nc)
        else:
            groups[base]["original"] = nc
    return groups


# ---------------------------------------------------------------------------
# Hypergeometric pass@b
# ---------------------------------------------------------------------------


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_fail_prob(n: int, c: int, b: int) -> float:
    """P(no success among b draws without replacement) = C(n-c,b)/C(n,b)."""
    if b <= 0:
        return 1.0
    if n <= 0:
        return 1.0
    if c <= 0:
        return 1.0
    if c > 0 and b > n:
        return 0.0
    if b > n:
        return 1.0
    if b > n - c:
        return 0.0
    # b <= n, use log ratio for stability
    logp = _log_comb(n - c, b) - _log_comb(n, b)
    return math.exp(logp)


def pass_at_b(n: int, c: int, b: int) -> float:
    """pass@b = 1 - C(n-c,b)/C(n,b); if c > 0 and b > n, return 1."""
    return 1.0 - hypergeom_fail_prob(n, c, b)


# ---------------------------------------------------------------------------
# Config implementations
# ---------------------------------------------------------------------------


def compute_seed_pass_stats(groups: dict[str, dict], budget: int) -> PassAtKSplitStats:
    """pass@budget on the original statement only."""
    vals: list[float] = []
    for data in groups.values():
        orig = data["original"]
        if orig is None:
            continue
        n, c = orig
        vals.append(pass_at_b(n, c, budget))

    if not vals:
        return PassAtKSplitStats(float("nan"), float("nan"), 0)

    N = len(vals)
    avg = sum(vals) / N
    return PassAtKSplitStats(avg, float("nan"), N)


def compute_random_pass_stats(
    groups: dict[str, dict],
    budget: int,
    n_mc: int,
    rng: np.random.Generator,
) -> PassAtKSplitStats:
    """
    Monte Carlo over simulated runs. Each run: for every base problem with a non-empty
    pool, draw one (n, c) uniformly from the pool, compute pass@b, then average those
    values over problems. Return mean and sample std of the run scores (``n_mc`` runs).
    """
    pools: list[list[tuple[int, int]]] = []
    for data in groups.values():
        pool: list[tuple[int, int]] = list(data["variants"])
        if data["original"] is not None:
            pool.append(data["original"])
        if not pool:
            continue
        pools.append(pool)

    if not pools:
        return PassAtKSplitStats(float("nan"), float("nan"), 0)

    n_problems = len(pools)
    rollout_scores = np.empty(n_mc, dtype=np.float64)
    for t in range(n_mc):
        total = 0.0
        for pool in pools:
            idx = int(rng.integers(len(pool)))
            n, c = pool[idx]
            total += pass_at_b(n, c, budget)
        rollout_scores[t] = total / n_problems

    avg_pass = float(np.mean(rollout_scores))
    std_pass = float(np.std(rollout_scores, ddof=1)) if n_mc > 1 else float("nan")
    return PassAtKSplitStats(avg_pass, std_pass, n_problems)


def _ensemble_pass_from_nc_arrays(
    n_arr: np.ndarray,
    c_arr: np.ndarray,
    base_exps: np.ndarray,
) -> float:
    """1 - prod_i hypergeom_fail(n_i, c_i, b_i); arrays are 1D matching length."""
    prod_fail = 1.0
    for i in range(len(n_arr)):
        bi = int(base_exps[i])
        fi = hypergeom_fail_prob(int(n_arr[i]), int(c_arr[i]), bi)
        prod_fail *= fi
    return 1.0 - prod_fail


def _ensemble_pass_single_draw(
    fixed_n: np.ndarray,
    fixed_c: np.ndarray,
    pool_n: np.ndarray,
    pool_c: np.ndarray,
    k_sample: int,
    budget: int,
    rng: np.random.Generator,
) -> float:
    """One random variant subset for this problem; return ensemble pass@b."""
    K = len(pool_n)
    effective_k = len(fixed_n) + k_sample
    q, r = divmod(budget, effective_k)
    base_exps = np.where(np.arange(effective_k) < r, q + 1, q).astype(np.float64)

    if k_sample == 0 or k_sample == K:
        if K > 0:
            all_n = np.concatenate([fixed_n, pool_n])
            all_c = np.concatenate([fixed_c, pool_c])
        else:
            all_n = fixed_n.copy()
            all_c = fixed_c.copy()
        return float(_ensemble_pass_from_nc_arrays(all_n, all_c, base_exps))

    noise = rng.random(K)
    order = np.argsort(noise)[:k_sample]
    chosen_n = pool_n[order]
    chosen_c = pool_c[order]
    if len(fixed_n) > 0:
        all_n = np.concatenate([fixed_n, chosen_n])
        all_c = np.concatenate([fixed_c, chosen_c])
    else:
        all_n = chosen_n
        all_c = chosen_c
    return float(_ensemble_pass_from_nc_arrays(all_n, all_c, base_exps))


def ensemble_pass_mean_std_one_problem(
    fixed_n: np.ndarray,
    fixed_c: np.ndarray,
    pool_n: np.ndarray,
    pool_c: np.ndarray,
    k_sample: int,
    budget: int,
    n_mc: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """MC mean and sample std of pass@b over ``n_mc`` draws *for this problem only*.

    Used by per-problem breakdown tables. Split-level ensemble stats use
    :func:`compute_ensemble_pass_stats` (rollout-averaged over all problems).
    Deterministic subset returns ``(p, nan)``.
    """
    K = len(pool_n)
    effective_k = len(fixed_n) + k_sample
    q, r = divmod(budget, effective_k)
    base_exps = np.where(np.arange(effective_k) < r, q + 1, q).astype(np.float64)

    if k_sample == 0 or k_sample == K:
        if K > 0:
            all_n = np.concatenate([fixed_n, pool_n])
            all_c = np.concatenate([fixed_c, pool_c])
        else:
            all_n = fixed_n.copy()
            all_c = fixed_c.copy()
        p = _ensemble_pass_from_nc_arrays(all_n, all_c, base_exps)
        return (float(p), float("nan"))

    noise = rng.random((n_mc, K))
    order = np.argsort(noise, axis=1)[:, :k_sample]
    chosen_n = pool_n[order]
    chosen_c = pool_c[order]

    if len(fixed_n) > 0:
        fn = np.broadcast_to(fixed_n, (n_mc, len(fixed_n)))
        fc = np.broadcast_to(fixed_c, (n_mc, len(fixed_c)))
        all_n = np.concatenate([fn, chosen_n], axis=1)
        all_c = np.concatenate([fc, chosen_c], axis=1)
    else:
        all_n = chosen_n
        all_c = chosen_c

    ps = np.empty(n_mc, dtype=np.float64)
    for t in range(n_mc):
        ps[t] = _ensemble_pass_from_nc_arrays(all_n[t], all_c[t], base_exps)
    return (float(np.mean(ps)), float(np.std(ps, ddof=1)))


def compute_ensemble_pass_stats(
    groups: dict[str, dict],
    budget: int,
    n_variants: int,
    include_seed: bool,
    n_mc: int = 20_000,
    rng: np.random.Generator | None = None,
) -> PassAtKSplitStats:
    """Monte Carlo over simulated runs (same budget split as SR ensembles).

    Each run: for every base problem, draw one random variant subset (independently
    across problems), compute ensemble pass@b for that problem, then average over
    problems. ``avg_pass`` / ``std_pass`` are mean and sample std of those run scores.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ProblemEntry = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
    problem_list: list[ProblemEntry] = []

    for data in groups.values():
        variant_ns = np.asarray([x[0] for x in data["variants"]], dtype=np.int64)
        variant_cs = np.asarray([x[1] for x in data["variants"]], dtype=np.int64)
        orig = data["original"]

        if include_seed:
            if orig is None:
                continue
            fixed_n = np.array([orig[0]], dtype=np.int64)
            fixed_c = np.array([orig[1]], dtype=np.int64)
            pool_n, pool_c = variant_ns, variant_cs
            k_need = n_variants - 1
        else:
            if orig is not None:
                pool_n = np.concatenate([[orig[0]], variant_ns])
                pool_c = np.concatenate([[orig[1]], variant_cs])
            else:
                pool_n, pool_c = variant_ns, variant_cs
            if len(pool_n) == 0:
                continue
            fixed_n = np.empty(0, dtype=np.int64)
            fixed_c = np.empty(0, dtype=np.int64)
            k_need = n_variants

        k_avail = len(pool_n)
        k_sample = min(k_need, k_avail)
        problem_list.append((fixed_n, fixed_c, pool_n, pool_c, k_sample))

    if not problem_list:
        return PassAtKSplitStats(float("nan"), float("nan"), 0)

    n_problems = len(problem_list)
    rollout_scores = np.empty(n_mc, dtype=np.float64)
    for t in range(n_mc):
        total = 0.0
        for fixed_n, fixed_c, pool_n, pool_c, k_sample in problem_list:
            total += _ensemble_pass_single_draw(
                fixed_n, fixed_c, pool_n, pool_c, k_sample, budget, rng
            )
        rollout_scores[t] = total / n_problems

    avg_pass = float(np.mean(rollout_scores))
    std_pass = float(np.std(rollout_scores, ddof=1)) if n_mc > 1 else float("nan")
    return PassAtKSplitStats(avg_pass, std_pass, n_problems)

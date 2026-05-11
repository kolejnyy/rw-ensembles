#!/usr/bin/env python3
"""Produce a LaTeX comparison table of proof-search configurations.

For each combination of (split, configuration, budget) the script computes
the average success rate (SR) across all base problems and an optional spread
metric (see ``reverification_utils``), then emits a LaTeX tabular environment
ready to paste into Overleaf.  The seed row omits ± (single SR per problem).

Configurations
--------------
  seed        : original minif2f statement, no variants
  random      : pick one variant uniformly at random
  ens-2/4/8   : pick k variants at random, split budget equally
  ens-1+1/1+3/1+7 : original + k random variants, split budget equally

Usage
-----
    python make_analysis_table.py \\
        --valid-dir results/minif2f-rw/valid/deepseek-miniF2F-rw-valid-noncot \\
        --test-dir  results/minif2f-rw/test/deepseek-miniF2F-rw-test-noncot  \\
        --out-dir   results/minif2f-rw/.analysis

Both --valid-dir and --test-dir accept either:
  * the split result directory (auto-appends /.reverification_by_problem), or
  * the .reverification_by_problem directory directly.

Either flag may be omitted to produce a single-split table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from reverification_utils import (  # noqa: E402
    BUDGETS,
    SplitStats,
    compute_ensemble_stats,
    compute_random_stats,
    compute_seed_stats,
    ensemble_sr_mean_std_one_problem,
    group_by_base,
    load_cp_rates,
    sr_from_cp,
)

_REVERIFICATION_SUBDIR = ".reverification_by_problem"


def resolve_reverification_dir(path: Path) -> Path:
    """Accept either the split root or the .reverification_by_problem subdir."""
    if (path / _REVERIFICATION_SUBDIR).is_dir():
        return path / _REVERIFICATION_SUBDIR
    if path.name == _REVERIFICATION_SUBDIR and path.is_dir():
        return path
    # Last resort: try as-is
    return path


# ---------------------------------------------------------------------------
# Configuration descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    name: str
    label: str  # short LaTeX label for the row
    kind: str   # "seed" | "random" | "ensemble"
    n_variants: int = 0
    include_seed: bool = False


CONFIGS: list[Config] = [
    Config("seed",    r"\textit{seed}",   "seed"),
    Config("random",  r"\textit{random}", "random"),
    Config("ens-2",   r"ens-2",           "ensemble", n_variants=2,  include_seed=False),
    Config("ens-4",   r"ens-4",           "ensemble", n_variants=4,  include_seed=False),
    Config("ens-8",   r"ens-8",           "ensemble", n_variants=8,  include_seed=False),
    Config("ens-1+1", r"ens-1+1",         "ensemble", n_variants=2,  include_seed=True),
    Config("ens-1+3", r"ens-1+3",         "ensemble", n_variants=4,  include_seed=True),
    Config("ens-1+7", r"ens-1+7",         "ensemble", n_variants=8,  include_seed=True),
]


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def compute_all_stats(
    groups: dict[str, dict],
    configs: list[Config],
    budgets: list[int],
    n_mc: int,
    rng: np.random.Generator,
) -> dict[tuple[str, int], SplitStats]:
    """Return {(config_name, budget): SplitStats} for a single split."""
    results: dict[tuple[str, int], SplitStats] = {}

    for cfg in configs:
        print(f"  Computing '{cfg.name}' ...", end="", flush=True)
        for budget in budgets:
            if cfg.kind == "ensemble" and cfg.n_variants > budget:
                results[(cfg.name, budget)] = None
                continue
            if cfg.kind == "seed":
                stats = compute_seed_stats(groups, budget)
            elif cfg.kind == "random":
                stats = compute_random_stats(groups, budget)
            else:  # ensemble
                stats = compute_ensemble_stats(
                    groups,
                    budget,
                    n_variants=cfg.n_variants,
                    include_seed=cfg.include_seed,
                    n_mc=n_mc,
                    rng=rng,
                )
            results[(cfg.name, budget)] = stats
        first_valid = next(s for s in (results.get((cfg.name, b)) for b in budgets) if s is not None)
        print(f" done (N={first_valid.n_problems})")

    return results


# ---------------------------------------------------------------------------
# LaTeX table formatting
# ---------------------------------------------------------------------------


def _config_group(cfg: Config) -> int:
    """Return a group index used to insert \\midrule between groups of rows."""
    if cfg.kind in ("seed", "random"):
        return 0
    if cfg.kind == "ensemble" and not cfg.include_seed:
        return 1
    return 2  # ensemble with forced seed


def _fmt_cell(stats: SplitStats | None, bold: bool = False) -> str:
    if stats is None:
        return "--"
    if stats.n_problems == 0:
        return r"\textemdash"
    pct_avg = 100.0 * stats.avg_sr
    if not math.isfinite(stats.std_sr):
        inner = f"{pct_avg:.1f}"
    else:
        pct_std = 100.0 * stats.std_sr
        inner = rf"{pct_avg:.1f}_{{\pm{pct_std:.1f}}}"
    if bold:
        # \bm{} bolds all math symbols including \pm; requires \usepackage{bm}
        return rf"$\bm{{{inner}}}$"
    return rf"${inner}$"


def _build_latex_table(
    split_results: dict[str, dict[tuple[str, int], SplitStats | None]],
    splits: list[tuple[str, str]],  # [(split_name, split_display_label), ...]
    configs: list[Config],
    budgets: list[int],
) -> str:
    n_splits = len(splits)
    n_b = len(budgets)

    col_spec = "l" + ("|" + "c" * n_b) * n_splits
    budget_header = " & ".join(str(b) for b in budgets)

    # Pre-compute the best avg_sr per (split_name, budget) column so we can bold it.
    col_best: dict[tuple[str, int], float] = {}
    for split_name, _ in splits:
        split_res = split_results.get(split_name, {})
        for budget in budgets:
            vals = [
                s.avg_sr
                for cfg in configs
                if (s := split_res.get((cfg.name, budget))) is not None
                and s.n_problems > 0
            ]
            if vals:
                col_best[(split_name, budget)] = max(vals)

    lines: list[str] = [
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
    ]

    # Top header: split names spanning n_b columns each
    header_cells = []
    for i, (_, label) in enumerate(splits):
        align = "c|" if i < n_splits - 1 else "c"
        header_cells.append(rf"\multicolumn{{{n_b}}}{{{align}}}{{{label}}}")
    lines.append(r" & " + " & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Budget sub-header
    sub_cells = [budget_header] * n_splits
    lines.append(r"\textbf{Budget} & " + " & ".join(sub_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows, with \midrule between config groups
    prev_group: int | None = None
    for cfg in configs:
        group = _config_group(cfg)
        if prev_group is not None and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group

        row_cells = [cfg.label]
        for split_name, _ in splits:
            split_res = split_results.get(split_name, {})
            for budget in budgets:
                stats = split_res.get((cfg.name, budget))
                best = col_best.get((split_name, budget))
                is_best = (
                    stats is not None
                    and stats.n_problems > 0
                    and best is not None
                    and abs(stats.avg_sr - best) < 1e-9
                )
                row_cells.append(_fmt_cell(stats, bold=is_best))
        lines.append(" & ".join(row_cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-problem breakdown
# ---------------------------------------------------------------------------


def _ens_sr_one_problem(
    fixed: np.ndarray,
    pool: np.ndarray,
    k_sample: int,
    budget: int,
    n_mc: int,
    rng: np.random.Generator,
) -> float:
    """Mean SR for a single problem under one ensemble config (per-problem CSV)."""
    m, _ = ensemble_sr_mean_std_one_problem(fixed, pool, k_sample, budget, n_mc, rng)
    return m


def compute_per_problem_breakdown(
    groups: dict[str, dict],
    configs: list[Config],
    budgets: list[int],
    n_mc: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Return one dict per base problem with SR for every valid (config, budget).

    Each dict contains:
      problem, original_cp, n_variants, n_pool
      sr_{cfg.name}_{budget}  for every config × budget (nan when n_variants > budget
                               for ensemble, or original missing for seed)
      delta_best_{budget}      max-ensemble-SR minus seed-SR at that budget
    """
    rows: list[dict] = []

    for base, data in sorted(groups.items()):
        orig_cp = data["original"]
        variant_arr = np.asarray(data["variants"], dtype=np.float64)

        row: dict = {
            "problem": base,
            "original_cp": orig_cp if orig_cp is not None else float("nan"),
            "n_variants": len(variant_arr),
            "n_pool": len(variant_arr) + (1 if orig_cp is not None else 0),
        }

        for budget in budgets:
            ens_srs: list[float] = []

            for cfg in configs:
                col = f"sr_{cfg.name}_{budget}"

                if cfg.kind == "seed":
                    if orig_cp is None:
                        row[col] = float("nan")
                    else:
                        row[col] = sr_from_cp(orig_cp, budget)

                elif cfg.kind == "random":
                    pool_list = list(variant_arr)
                    if orig_cp is not None:
                        pool_list = [orig_cp] + pool_list
                    if not pool_list:
                        row[col] = float("nan")
                    else:
                        row[col] = sum(sr_from_cp(cp, budget) for cp in pool_list) / len(pool_list)

                else:  # ensemble
                    if cfg.n_variants > budget:
                        row[col] = float("nan")
                        continue

                    if cfg.include_seed:
                        if orig_cp is None:
                            row[col] = float("nan")
                            continue
                        fixed = np.array([orig_cp], dtype=np.float64)
                        pool = variant_arr
                        k_extra_need = cfg.n_variants - 1
                    else:
                        pool_list = list(variant_arr)
                        if orig_cp is not None:
                            pool_list = [orig_cp] + pool_list
                        if not pool_list:
                            row[col] = float("nan")
                            continue
                        fixed = np.empty(0, dtype=np.float64)
                        pool = np.asarray(pool_list, dtype=np.float64)
                        k_extra_need = cfg.n_variants

                    k_sample = min(k_extra_need, len(pool))
                    sr = _ens_sr_one_problem(fixed, pool, k_sample, budget, n_mc, rng)
                    row[col] = sr
                    ens_srs.append(sr)

            # Improvement of best ensemble over seed at this budget
            seed_sr = row.get(f"sr_seed_{budget}", float("nan"))
            if ens_srs and not (seed_sr != seed_sr):  # seed_sr is not nan
                row[f"delta_best_{budget}"] = max(ens_srs) - seed_sr
            else:
                row[f"delta_best_{budget}"] = float("nan")

        rows.append(row)

    return rows


def save_per_problem_csv(
    rows: list[dict],
    out_path: Path,
    sort_budget: int = 32,
) -> None:
    """Write per-problem breakdown CSV, sorted by delta_best_{sort_budget} descending."""
    if not rows:
        return

    sort_key = f"delta_best_{sort_budget}"
    rows_sorted = sorted(
        rows,
        key=lambda r: (-(r.get(sort_key) or float("-inf")), r["problem"]),
    )

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({
                k: ("" if (v != v) else (f"{v:.6f}" if isinstance(v, float) else v))
                for k, v in row.items()
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--valid-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Valid split directory (split root or .reverification_by_problem subdir).",
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Test split directory (split root or .reverification_by_problem subdir).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory to write analysis outputs (JSON + .tex). "
            "Defaults to <split_dir>/.analysis when a single split is given, "
            "or the common parent's .analysis subdir for two splits."
        ),
    )
    p.add_argument(
        "--n-mc",
        type=int,
        default=20_000,
        metavar="N",
        help="Number of Monte Carlo samples for ensemble configurations (default: 20000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for Monte Carlo sampling (default: 42).",
    )
    return p.parse_args()


def _default_out_dir(dirs: list[Path]) -> Path:
    if len(dirs) == 1:
        return dirs[0].parent / ".analysis"
    # Try common parent
    try:
        common = Path(*[str(d) for d in dirs])  # may fail; fall back below
    except TypeError:
        common = dirs[0].parent
    # Walk up to find a directory that contains all split dirs
    parents = [d.resolve() for d in dirs]
    candidate = parents[0].parent
    while candidate != candidate.parent:
        if all(str(p).startswith(str(candidate)) for p in parents):
            return candidate / ".analysis"
        candidate = candidate.parent
    return dirs[0].parent / ".analysis"


def main() -> int:
    args = parse_args()

    if args.valid_dir is None and args.test_dir is None:
        # Attempt defaults
        default_valid = Path("results/minif2f-rw/valid/deepseek-miniF2F-rw-valid-noncot")
        default_test = Path("results/minif2f-rw/test/deepseek-miniF2F-rw-test-noncot")
        if default_valid.exists():
            args.valid_dir = default_valid
        if default_test.exists():
            args.test_dir = default_test
        if args.valid_dir is None and args.test_dir is None:
            sys.exit(
                "Error: provide at least one of --valid-dir / --test-dir "
                "(or run from the project root where defaults exist)."
            )

    # Resolve split directories and labels
    splits_input: list[tuple[str, str, Path]] = []  # (name, label, reveri_dir)
    if args.valid_dir is not None:
        revdir = resolve_reverification_dir(args.valid_dir.resolve())
        if not revdir.is_dir():
            sys.exit(f"Directory not found: {revdir}")
        splits_input.append(("valid", r"\textit{minif2f-rw-valid}", revdir))
    if args.test_dir is not None:
        revdir = resolve_reverification_dir(args.test_dir.resolve())
        if not revdir.is_dir():
            sys.exit(f"Directory not found: {revdir}")
        splits_input.append(("test", r"\textit{minif2f-rw-test}", revdir))

    # Output directory
    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        out_dir = _default_out_dir([d for _, _, d in splits_input])
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Compute stats for each split
    all_results: dict[str, dict[tuple[str, int], SplitStats]] = {}
    all_splits_display: list[tuple[str, str]] = []

    for split_name, split_label, revdir in splits_input:
        print(f"\n[{split_name}] Loading CP rates from: {revdir}")
        rates = load_cp_rates(revdir)
        groups = group_by_base(rates)
        print(f"  {len(groups)} base problems, {len(rates)} total theorem files")

        print(f"[{split_name}] Computing statistics (n_mc={args.n_mc}) ...")
        results = compute_all_stats(groups, CONFIGS, BUDGETS, n_mc=args.n_mc, rng=rng)
        all_results[split_name] = results
        all_splits_display.append((split_name, split_label))

        # Per-problem breakdown CSV
        print(f"[{split_name}] Computing per-problem breakdown ...")
        pp_rows = compute_per_problem_breakdown(groups, CONFIGS, BUDGETS, n_mc=args.n_mc, rng=rng)
        pp_path = out_dir / f"{split_name}_per_problem.csv"
        save_per_problem_csv(pp_rows, pp_path, sort_budget=32)
        print(f"  Saved: {pp_path}")

        # Save per-split JSON
        json_path = out_dir / f"{split_name}_stats.json"
        serialisable = {
            f"{cfg_name}_budget{b}": (
                None if stats is None else {
                    "avg_sr": stats.avg_sr,
                    "std_sr": (
                        None
                        if isinstance(stats.std_sr, float) and not math.isfinite(stats.std_sr)
                        else stats.std_sr
                    ),
                    "n_problems": stats.n_problems,
                }
            )
            for (cfg_name, b), stats in results.items()
        }
        json_path.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")
        print(f"  Saved: {json_path}")

    # Build and save LaTeX table
    print("\nBuilding LaTeX table ...")
    latex = _build_latex_table(all_results, all_splits_display, CONFIGS, BUDGETS)

    tex_path = out_dir / "analysis_table.tex"
    tex_path.write_text(latex + "\n", encoding="utf-8")
    print(f"Saved: {tex_path}")

    # Also print to stdout for convenience
    print("\n" + "=" * 70)
    print(latex)
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""LaTeX table of pass@b metrics from reverification JSON (hypergeometric).

Unlike ``make_analysis_table.py`` (SR from correctness probability), this script
uses counts (n attempts, c successes) per theorem and:

  pass@b = 1 - C(n-c, b) / C(n, b)

with pass@b = 1 when c > 0 and b > n.  Ensembles use the product of segment
failure probabilities with the same budget split as SR analysis; see
``reverification_pass_at_k_utils.py``.

For *random* and *ensemble* rows, ``--n-mc`` controls how many **simulated runs**
are averaged: each run scores all problems (with independent random choices per
problem), then mean pass@b across problems is one scalar; reported mean ± std are
over those run scores.

Usage
-----
    python make_pass_at_k_table.py \\
        --valid-dir results/minif2f-rw/valid/deepseek-miniF2F-rw-valid-noncot \\
        --test-dir  results/minif2f-rw/test/deepseek-miniF2F-rw-test-noncot  \\
        --out-dir   results/minif2f-rw/.analysis-pass-at-k

Path resolution for --valid-dir / --test-dir matches ``make_analysis_table.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from reverification_utils import BUDGETS  # noqa: E402
from reverification_pass_at_k_utils import (  # noqa: E402
    PassAtKSplitStats,
    compute_ensemble_pass_stats,
    compute_random_pass_stats,
    compute_seed_pass_stats,
    ensemble_pass_mean_std_one_problem,
    group_counts_by_base,
    load_attempt_counts,
    pass_at_b,
)

_REVERIFICATION_SUBDIR = ".reverification_by_problem"


def resolve_reverification_dir(path: Path) -> Path:
    """Accept either the split root or the .reverification_by_problem subdir."""
    if (path / _REVERIFICATION_SUBDIR).is_dir():
        return path / _REVERIFICATION_SUBDIR
    if path.name == _REVERIFICATION_SUBDIR and path.is_dir():
        return path
    return path


@dataclass(frozen=True)
class Config:
    name: str
    label: str
    kind: str
    n_variants: int = 0
    include_seed: bool = False


CONFIGS: list[Config] = [
    Config("seed", r"\textit{seed}", "seed"),
    # Config("random", r"\textit{random}", "random"),
    # Config("ens-2", r"ens-2", "ensemble", n_variants=2, include_seed=False),
    # Config("ens-4", r"ens-4", "ensemble", n_variants=4, include_seed=False),
    Config("ens-8", r"ens-8", "ensemble", n_variants=8, include_seed=False),
    # Config("ens-1+1", r"ens-1+1", "ensemble", n_variants=2, include_seed=True),
    # Config("ens-1+3", r"ens-1+3", "ensemble", n_variants=4, include_seed=True),
    Config("ens-1+7", r"ens-1+7", "ensemble", n_variants=8, include_seed=True),
]


def compute_all_stats(
    groups: dict[str, dict],
    configs: list[Config],
    budgets: list[int],
    n_mc: int,
    rng: np.random.Generator,
) -> dict[tuple[str, int], PassAtKSplitStats | None]:
    results: dict[tuple[str, int], PassAtKSplitStats | None] = {}

    for cfg in configs:
        print(f"  Computing '{cfg.name}' ...", end="", flush=True)
        for budget in tqdm(budgets):
            if cfg.kind == "ensemble" and cfg.n_variants > budget:
                results[(cfg.name, budget)] = None
                continue
            if cfg.kind == "seed":
                stats = compute_seed_pass_stats(groups, budget)
            elif cfg.kind == "random":
                stats = compute_random_pass_stats(groups, budget, n_mc=n_mc, rng=rng)
            else:
                stats = compute_ensemble_pass_stats(
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


def _config_group(cfg: Config) -> int:
    if cfg.kind in ("seed", "random"):
        return 0
    if cfg.kind == "ensemble" and not cfg.include_seed:
        return 1
    return 2


def _fmt_cell(stats: PassAtKSplitStats | None, bold: bool = False) -> str:
    if stats is None:
        return "--"
    if stats.n_problems == 0:
        return r"\textemdash"
    pct_avg = 100.0 * stats.avg_pass
    if not math.isfinite(stats.std_pass):
        inner = f"{pct_avg:.1f}"
    else:
        pct_std = 100.0 * stats.std_pass
        inner = rf"{pct_avg:.1f}_{{\pm{pct_std:.1f}}}"
    if bold:
        return rf"$\bm{{{inner}}}$"
    return rf"${inner}$"


def _build_latex_table(
    split_results: dict[str, dict[tuple[str, int], PassAtKSplitStats | None]],
    splits: list[tuple[str, str]],
    configs: list[Config],
    budgets: list[int],
) -> str:
    n_splits = len(splits)
    n_b = len(budgets)

    col_spec = "l" + ("|" + "c" * n_b) * n_splits
    budget_header = " & ".join(str(b) for b in budgets)

    col_best: dict[tuple[str, int], float] = {}
    for split_name, _ in splits:
        split_res = split_results.get(split_name, {})
        for budget in budgets:
            vals = [
                s.avg_pass
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

    header_cells = []
    for i, (_, label) in enumerate(splits):
        align = "c|" if i < n_splits - 1 else "c"
        header_cells.append(rf"\multicolumn{{{n_b}}}{{{align}}}{{{label}}}")
    lines.append(r" & " + " & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    sub_cells = [budget_header] * n_splits
    lines.append(r"\textbf{Budget} & " + " & ".join(sub_cells) + r" \\")
    lines.append(r"\midrule")

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
                    and abs(stats.avg_pass - best) < 1e-9
                )
                row_cells.append(_fmt_cell(stats, bold=is_best))
        lines.append(" & ".join(row_cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def compute_per_problem_breakdown(
    groups: dict[str, dict],
    configs: list[Config],
    budgets: list[int],
    n_mc: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []

    for base, data in sorted(groups.items()):
        orig = data["original"]
        variants = data["variants"]

        row: dict = {
            "problem": base,
            "original_n": orig[0] if orig is not None else "",
            "original_c": orig[1] if orig is not None else "",
            "n_variants": len(variants),
            "n_pool": len(variants) + (1 if orig is not None else 0),
        }

        for budget in budgets:
            ens_pass: list[float] = []

            for cfg in configs:
                col = f"pass_{cfg.name}_{budget}"

                if cfg.kind == "seed":
                    if orig is None:
                        row[col] = float("nan")
                    else:
                        n, c = orig
                        row[col] = pass_at_b(n, c, budget)

                elif cfg.kind == "random":
                    pool = list(variants)
                    if orig is not None:
                        pool = [orig] + pool
                    if not pool:
                        row[col] = float("nan")
                    else:
                        row[col] = sum(pass_at_b(n, c, budget) for n, c in pool) / len(pool)

                else:
                    if cfg.n_variants > budget:
                        row[col] = float("nan")
                        continue

                    variant_ns = np.asarray([x[0] for x in variants], dtype=np.int64)
                    variant_cs = np.asarray([x[1] for x in variants], dtype=np.int64)

                    if cfg.include_seed:
                        if orig is None:
                            row[col] = float("nan")
                            continue
                        fixed_n = np.array([orig[0]], dtype=np.int64)
                        fixed_c = np.array([orig[1]], dtype=np.int64)
                        pool_n, pool_c = variant_ns, variant_cs
                        k_need = cfg.n_variants - 1
                    else:
                        if orig is not None:
                            pool_n = np.concatenate([[orig[0]], variant_ns])
                            pool_c = np.concatenate([[orig[1]], variant_cs])
                        else:
                            pool_n, pool_c = variant_ns, variant_cs
                        if len(pool_n) == 0:
                            row[col] = float("nan")
                            continue
                        fixed_n = np.empty(0, dtype=np.int64)
                        fixed_c = np.empty(0, dtype=np.int64)
                        k_need = cfg.n_variants

                    k_sample = min(k_need, len(pool_n))
                    p, _ = ensemble_pass_mean_std_one_problem(
                        fixed_n, fixed_c, pool_n, pool_c, k_sample, budget, n_mc, rng
                    )
                    row[col] = p
                    ens_pass.append(p)

            seed_p = row.get(f"pass_seed_{budget}", float("nan"))
            if ens_pass and not (seed_p != seed_p):
                row[f"delta_best_{budget}"] = max(ens_pass) - seed_p
            else:
                row[f"delta_best_{budget}"] = float("nan")

        rows.append(row)

    return rows


def save_per_problem_csv(
    rows: list[dict],
    out_path: Path,
    sort_budget: int = 32,
) -> None:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--valid-dir", type=Path, default=None, metavar="DIR")
    p.add_argument("--test-dir", type=Path, default=None, metavar="DIR")
    p.add_argument("--out-dir", type=Path, default=None, metavar="DIR")
    p.add_argument(
        "--n-mc",
        type=int,
        default=20_000,
        metavar="N",
        help=(
            "Monte Carlo samples: simulated runs for random + ensemble split stats "
            "(each run averages pass@b over all problems; default: 20000). "
            "Per-problem ensemble breakdown still uses this many draws per problem."
        ),
    )
    p.add_argument("--seed", type=int, default=42, metavar="SEED")
    return p.parse_args()


def _default_out_dir(dirs: list[Path]) -> Path:
    if len(dirs) == 1:
        return dirs[0].parent / ".analysis-pass-at-k"
    parents = [d.resolve() for d in dirs]
    candidate = parents[0].parent
    while candidate != candidate.parent:
        if all(str(p).startswith(str(candidate)) for p in parents):
            return candidate / ".analysis-pass-at-k"
        candidate = candidate.parent
    return dirs[0].parent / ".analysis-pass-at-k"


def main() -> int:
    args = parse_args()

    if args.valid_dir is None and args.test_dir is None:
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

    splits_input: list[tuple[str, str, Path]] = []
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

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        out_dir = _default_out_dir([d for _, _, d in splits_input])
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    all_results: dict[str, dict[tuple[str, int], PassAtKSplitStats | None]] = {}
    all_splits_display: list[tuple[str, str]] = []

    for split_name, split_label, revdir in splits_input:
        print(f"\n[{split_name}] Loading attempt counts from: {revdir}")
        counts = load_attempt_counts(revdir)
        groups = group_counts_by_base(counts)
        print(f"  {len(groups)} base problems, {len(counts)} total theorem files")

        print(f"[{split_name}] Computing pass@b statistics (n_mc={args.n_mc}) ...")
        results = compute_all_stats(groups, CONFIGS, BUDGETS, n_mc=args.n_mc, rng=rng)
        all_results[split_name] = results
        all_splits_display.append((split_name, split_label))

        # print(f"[{split_name}] Computing per-problem breakdown ...")
        # pp_rows = compute_per_problem_breakdown(groups, CONFIGS, BUDGETS, n_mc=args.n_mc, rng=rng)
        # pp_path = out_dir / f"{split_name}_per_problem_pass_at_k.csv"
        # save_per_problem_csv(pp_rows, pp_path, sort_budget=32)
        # print(f"  Saved: {pp_path}")

        json_path = out_dir / f"{split_name}_pass_at_k_stats.json"
        serialisable = {
            f"{cfg_name}_budget{b}": (
                None if stats is None else {
                    "avg_pass": stats.avg_pass,
                    "std_pass": (
                        None
                        if isinstance(stats.std_pass, float) and not math.isfinite(stats.std_pass)
                        else stats.std_pass
                    ),
                    "n_problems": stats.n_problems,
                }
            )
            for (cfg_name, b), stats in results.items()
        }
        json_path.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")
        print(f"  Saved: {json_path}")

    print("\nBuilding LaTeX table ...")
    latex = _build_latex_table(all_results, all_splits_display, CONFIGS, BUDGETS)

    tex_path = out_dir / "analysis_pass_at_k_table.tex"
    tex_path.write_text(latex + "\n", encoding="utf-8")
    print(f"Saved: {tex_path}")

    print("\n" + "=" * 70)
    print(latex)
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

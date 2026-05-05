#!/usr/bin/env python3
"""
Re-verify success rate and PASS@k for a folder of solutions.

Expected directory structure:
  <solutions_dir>/
    <problem_name>/
      attempts.jsonl   # one JSON object per attempt, must include `final_code`

For every problem:
  - verify every `final_code` in attempts.jsonl using `invpro.utils.verifier.ProofVerifier`
  - compute per-problem PASS@k from the verified success booleans

Progress reporting:
  - tqdm over problems
  - per-problem tqdm over attempts (verification completion)
  - per-problem tqdm "theorems solved so far" bar (counts successful verifications)

Notes:
- This script re-verifies regardless of the stored `success` field in attempts.jsonl.
- It uses multiprocessing where each worker process owns a persistent ProofVerifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Optional

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# Allow running the script from arbitrary working directories when
# `invpro` isn't installed as a site-package.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from invpro.utils.metrics import pass_at_k
from invpro.utils.verifier import ProofVerifier


_WORKER_VERIFIER: Optional[ProofVerifier] = None
_WORKER_CURRENT_PROBLEM: Optional[str] = None
_WORKER_PROJECT_ROOT: Optional[str] = None
_WORKER_PREAMBLE: Optional[str] = None
_WORKER_TIMEOUT_SECONDS: Optional[float] = None


@dataclass(frozen=True)
class _VerifyTask:
    problem_name: str
    idx: int
    code: str


def _pool_worker_init(project_root: str, preamble: str, timeout_seconds: float) -> None:
    global _WORKER_VERIFIER, _WORKER_CURRENT_PROBLEM
    global _WORKER_PROJECT_ROOT, _WORKER_PREAMBLE, _WORKER_TIMEOUT_SECONDS
    _WORKER_PROJECT_ROOT = project_root
    _WORKER_PREAMBLE = preamble
    _WORKER_TIMEOUT_SECONDS = timeout_seconds
    _WORKER_CURRENT_PROBLEM = None
    _WORKER_VERIFIER = None


def _pool_worker_verify(task: _VerifyTask) -> tuple[int, bool]:
    # Keep function top-level for multiprocessing pickling.
    global _WORKER_VERIFIER, _WORKER_CURRENT_PROBLEM
    assert _WORKER_PROJECT_ROOT is not None
    assert _WORKER_PREAMBLE is not None
    assert _WORKER_TIMEOUT_SECONDS is not None

    def _dispose_and_recreate_verifier() -> None:
        global _WORKER_VERIFIER
        if _WORKER_VERIFIER is not None:
            dispose = getattr(_WORKER_VERIFIER, "_dispose_lsp_clients", None)
            if callable(dispose):
                try:
                    dispose()
                except Exception:
                    pass
        _WORKER_VERIFIER = ProofVerifier(
            project_root=_WORKER_PROJECT_ROOT,
            initial_imports=_WORKER_PREAMBLE,
            timeout_seconds=_WORKER_TIMEOUT_SECONDS,
        )

    def _is_lsp_crash_like_error(err: Optional[str]) -> bool:
        if not err:
            return False
        s = str(err).lower()
        return (
            "-32901" in s
            or "-32902" in s
            or ("server process" in s and "crashed" in s)
            or "future exception was never retrieved" in s
            or "lsp error" in s
            or "broken pipe" in s
            or "connection reset" in s
        )

    # Re-instantiate per problem boundary.
    if _WORKER_CURRENT_PROBLEM != task.problem_name:
        _dispose_and_recreate_verifier()
        _WORKER_CURRENT_PROBLEM = task.problem_name

    assert _WORKER_VERIFIER is not None, "Worker verifier not initialized"

    # First try
    try:
        ok, err = _WORKER_VERIFIER.verify(task.code)
    except Exception as e:
        ok = False
        err = str(e)

    # If we hit crash-like signals, recreate immediately and retry once.
    if (not ok) and _is_lsp_crash_like_error(err):
        _dispose_and_recreate_verifier()
        try:
            ok_retry, _err_retry = _WORKER_VERIFIER.verify(task.code)
            ok = ok_retry
        except Exception:
            ok = False

    return task.idx, ok


def _parse_k_values(csv: str) -> list[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            k = int(p)
        except ValueError as e:
            raise SystemExit(f"Invalid k value: {p!r}") from e
        if k <= 0:
            raise SystemExit(f"k must be positive; got {k}")
        out.append(k)
    # Deduplicate while preserving order.
    seen = set()
    dedup: list[int] = []
    for k in out:
        if k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    return dedup


def _read_attempts(attempts_path: Path, max_attempts: Optional[int]) -> list[tuple[int, str]]:
    attempts: list[tuple[int, str]] = []
    with attempts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = rec.get("final_code") or ""
            if not isinstance(code, str) or not code.strip():
                continue
            attempt_num_raw = rec.get("attempt_num", None)
            try:
                attempt_num = int(attempt_num_raw) if attempt_num_raw is not None else len(attempts)
            except ValueError:
                attempt_num = len(attempts)
            attempts.append((attempt_num, code))
            if max_attempts is not None and len(attempts) >= max_attempts:
                break
    # Stable ordering by attempt_num (not required for PASS@k, but makes output clearer).
    attempts.sort(key=lambda x: x[0])
    return attempts


def _iter_problem_dirs(solutions_dir: Path) -> list[Path]:
    dirs = [p for p in sorted(solutions_dir.iterdir()) if p.is_dir()]
    return dirs


def _fmt_percent(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x * 100.0:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solutions_dir", type=Path, help="Folder with per-problem subfolders")
    parser.add_argument(
        "--attempts-file",
        type=str,
        default="attempts.jsonl",
        help="Filename inside each problem folder",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Lean project root used by ProofVerifier",
    )
    parser.add_argument(
        "--preamble",
        type=str,
        default="import Mathlib\n",
        help="initial_imports for ProofVerifier (optimization to reuse Mathlib)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Per-verification timeout passed to ProofVerifier",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Multiprocessing workers",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated k values for PASS@k",
    )
    parser.add_argument(
        "--max-attempts-per-problem",
        type=int,
        default=None,
        help="Optional cap on number of attempts verified per problem",
    )
    parser.add_argument(
        "--limit-problems",
        type=int,
        default=None,
        help="Optional cap on number of problems (debugging)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <solutions_dir>/reverified_pass_at_k.csv)",
    )
    args = parser.parse_args()

    solutions_dir = args.solutions_dir.resolve()
    if not solutions_dir.is_dir():
        raise SystemExit(f"Not a directory: {solutions_dir}")

    k_values = _parse_k_values(args.k_values)
    problem_dirs = _iter_problem_dirs(solutions_dir)
    if args.limit_problems is not None:
        problem_dirs = problem_dirs[: int(args.limit_problems)]

    if not problem_dirs:
        raise SystemExit(f"No problem subdirectories found in: {solutions_dir}")

    output_csv = args.output_csv or (solutions_dir / "reverified_pass_at_k.csv")

    use_tqdm = tqdm is not None
    outer_iter: Iterable[Path] = problem_dirs
    if use_tqdm:
        outer_iter = tqdm(problem_dirs, desc="Problems", unit="problem")

    # Build worker pool once (amortize LSP startup per process).
    ctx = get_context("spawn")
    pool = ctx.Pool(
        processes=int(args.num_workers),
        initializer=_pool_worker_init,
        initargs=(str(args.project_root), args.preamble, float(args.timeout_seconds)),
    )

    per_problem_metrics: dict[str, dict[str, float]] = {}
    passk_sums: dict[int, float] = {k: 0.0 for k in k_values}
    total_problems = 0

    try:
        for problem_dir in outer_iter:
            problem_name = problem_dir.name
            attempts_path = problem_dir / args.attempts_file
            if not attempts_path.exists():
                # Skip directories without attempts.jsonl.
                continue

            attempts = _read_attempts(attempts_path, args.max_attempts_per_problem)
            if not attempts:
                continue

            indexed_tasks = [
                _VerifyTask(problem_name=problem_name, idx=i, code=code)
                for i, (_attempt_num, code) in enumerate(attempts)
            ]
            n = len(indexed_tasks)

            # Inner progress bars live only in the main process.
            solved_bar = None
            if use_tqdm:
                # position=1 avoids clobbering the outer bar
                attempts_bar = tqdm(total=n, desc=f"{problem_name}: verify", unit="attempt", position=1, leave=False)
                solved_bar = tqdm(total=n, desc=f"{problem_name}: solved", unit="theorem", position=2, leave=False)
            else:  # pragma: no cover
                attempts_bar = None

            successes: list[Optional[bool]] = [None] * n
            success_count = 0

            # Feed tasks to pool and update progress as results return.
            for idx, ok in pool.imap_unordered(_pool_worker_verify, indexed_tasks, chunksize=1):
                successes[idx] = ok
                if use_tqdm:
                    attempts_bar.update(1)
                    if ok:
                        success_count += 1
                        solved_bar.update(1)

            if use_tqdm:
                attempts_bar.close()
                if solved_bar is not None:
                    solved_bar.close()

            # Convert to a plain boolean list (all should be filled).
            success_bools = [bool(x) for x in successes if x is not None]
            if len(success_bools) != n:
                raise SystemExit(f"Internal error: expected {n} successes, got {len(success_bools)} for {problem_name}")

            per_k: dict[str, float] = {}
            for k in k_values:
                per_k[f"pass@{k}"] = pass_at_k(success_bools, k)

            per_problem_metrics[problem_name] = per_k
            total_problems += 1
            for k in k_values:
                passk_sums[k] += per_k[f"pass@{k}"]

            # Print per-problem line(s).
            # (Printing after inner bars closes keeps terminal readable.)
            metrics_str = " ".join(f"pass@{k}={per_k[f'pass@{k}']*100.0:.2f}%" for k in k_values)
            print(f"[{problem_name}] n_attempts={n} solved={success_count}/{n} {metrics_str}")

        # Save per-problem table to CSV.
        csv_fieldnames = ["problem"] + [f"pass@{k}" for k in k_values]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            for problem_name in sorted(per_problem_metrics.keys()):
                row = {"problem": problem_name}
                row.update(per_problem_metrics[problem_name])
                writer.writerow(row)

        # Total averages.
        print()
        print("=== Overall ===")
        print(f"Problems evaluated: {total_problems}")
        for k in k_values:
            avg = passk_sums[k] / total_problems if total_problems > 0 else float("nan")
            print(f"Average pass@{k}: {avg*100.0:.4f}%")
        print(f"Per-problem CSV: {output_csv}")

    finally:
        pool.close()
        pool.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


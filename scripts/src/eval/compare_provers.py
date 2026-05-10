#!/usr/bin/env python3
"""
Run multiple prover configs (baseline, variable renaming, rewriting variants) on a
single minif2f problem with many attempts each; write results to a timestamped
experiment folder with per-prover JSONL logs and a summary JSON.

Usage (from repo root):
  python scripts/src/eval/compare_provers.py [--problem mathd_numbertheory_780] [--attempts 50] [--output DIR]

Output: results/compare_provers/<timestamp>/ (or --output DIR) containing:
  - <prover_name>.jsonl: one JSON object per attempt, appended as obtained
  - summary.json: problem info, PASS@k per prover, and full attempts list

Prover configs are read from configs/testing/compare_provers/ (baseline_24k,
varenamer_24k, rewrite_100_1_mean, rewrite_100_1_log_mean, rewrite_10_2_mean,
rewrite_10_2_log_mean, rewrite_15_2_mean, rewrite_15_2_log_mean).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_project_root))

from rwens.models.conf import dict_to_config as prover_dict_to_config, prover_from_config
from rwens.utils.metrics import pass_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PROBLEM = "mathd_numbertheory_780"
DEFAULT_DATASET = Path("data/minif2f.jsonl")
DEFAULT_SPLIT = "valid"
DEFAULT_ATTEMPTS = 50
CONFIG_DIR = Path("configs/testing/compare_provers")
PROVER_CONFIG_NAMES = [
    "baseline_24k",
    "varenamer_24k",
    "rewrite_100_1_mean",
    "rewrite_100_1_log_mean",
    "rewrite_10_2_mean",
    "rewrite_10_2_log_mean",
    "rewrite_15_2_mean",
    "rewrite_15_2_log_mean",
]


def load_dataset(dataset_path: Path, split: str) -> List[Dict[str, Any]]:
    problems = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p.get("split") == split:
                    problems.append(p)
    return problems


def find_problem_by_name(problems: List[Dict], name: str) -> Dict[str, Any] | None:
    for p in problems:
        if p.get("name") == name:
            return p
    return None


def build_problem_statement(problem: Dict[str, Any]) -> str:
    header = problem.get("header", "")
    stmt = problem.get("formal_statement", "")
    text = (header + stmt).replace("import Aesop\n", "")
    if "rwens.lean.Rewrites" not in text:
        text = text.replace("import Mathlib\n", "import Mathlib\nimport rwens.lean.Rewrites\n", 1)
    return text


def load_prover_from_config_path(prover_config_path: Path):
    with open(prover_config_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    config = prover_dict_to_config(d)
    return prover_from_config(config)


def calculate_k_values(n: int) -> List[int]:
    k_values = [1]
    k = 2
    while k <= n:
        k_values.append(k)
        k *= 2
    k_values = [k for k in k_values if k <= n]
    if k_values[-1] != n:
        k_values.append(n)
    return k_values


def run_attempts(
    prover,
    problem_statement: str,
    problem_name: str,
    num_attempts: int,
    prover_label: str,
    attempt_log_path: Optional[Path] = None,
) -> tuple[List[bool], List[Dict[str, Any]], Dict[int, float]]:
    successes = []
    attempts_log = []
    for attempt_num in range(1, num_attempts + 1):
        try:
            result = prover.prove(problem_statement)
            ok = result is not None and result.get("success", False)
            successes.append(ok)
            record = {
                "problem_name": problem_name,
                "prover": prover_label,
                "attempt_num": attempt_num,
                "success": ok,
                "final_code": result.get("final_code", "") if result else "",
                "error": result.get("error") if result else None,
                "steps_count": len(result.get("steps", [])) if result else 0,
            }
            attempts_log.append(record)
        except Exception as e:
            successes.append(False)
            record = {
                "problem_name": problem_name,
                "prover": prover_label,
                "attempt_num": attempt_num,
                "success": False,
                "final_code": "",
                "error": str(e),
                "steps_count": 0,
            }
            attempts_log.append(record)
        if attempt_log_path is not None:
            with open(attempt_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\r[{prover_label}] Attempts: {''.join('✓' if s else '✗' for s in successes)}", end="", flush=True)
    print()
    k_values = calculate_k_values(num_attempts)
    pass_at_k_dict = {k: pass_at_k(successes, k) for k in k_values}
    return successes, attempts_log, pass_at_k_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        type=str,
        default=DEFAULT_PROBLEM,
        help=f"minif2f problem name (default: {DEFAULT_PROBLEM})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to minif2f JSONL (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help=f"Attempts per prover (default: {DEFAULT_ATTEMPTS})",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Directory containing prover YAML configs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Experiment output directory (default: results/compare_provers/<timestamp>)",
    )
    args = parser.parse_args()

    config_dir = args.config_dir if args.config_dir.is_absolute() else _project_root / args.config_dir
    dataset_path = args.dataset if args.dataset.is_absolute() else _project_root / args.dataset

    problems = load_dataset(dataset_path, DEFAULT_SPLIT)
    problem = find_problem_by_name(problems, args.problem)
    if problem is None:
        logger.error(f"Problem '{args.problem}' not found in {dataset_path} (split={DEFAULT_SPLIT})")
        return 1

    problem_statement = build_problem_statement(problem)
    num_attempts = args.attempts
    k_values = calculate_k_values(num_attempts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = _project_root / "results" / "compare_provers"
    if args.output is not None:
        experiment_dir = Path(args.output) if args.output.is_absolute() else _project_root / args.output
    else:
        experiment_dir = base_out / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment directory: {experiment_dir}")

    results = {
        "problem_name": args.problem,
        "dataset_path": str(dataset_path),
        "split": DEFAULT_SPLIT,
        "num_attempts_per_prover": num_attempts,
        "k_values": k_values,
        "experiment_dir": str(experiment_dir),
        "provers": {},
    }

    all_attempts: List[Dict[str, Any]] = []

    for name in PROVER_CONFIG_NAMES:
        config_path = config_dir / f"{name}.yaml"
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, skipping")
            continue
        logger.info(f"Loading prover: {name} ({config_path})")
        try:
            prover = load_prover_from_config_path(config_path)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            results["provers"][name] = {"error": str(e), "config_path": str(config_path)}
            continue
        attempt_log_path = experiment_dir / f"{name}.jsonl"
        logger.info(f"Running {num_attempts} attempts for {name} (logging to {attempt_log_path})...")
        successes, attempts_log, pass_at_k_dict = run_attempts(
            prover,
            problem_statement,
            args.problem,
            num_attempts,
            name,
            attempt_log_path=attempt_log_path,
        )
        all_attempts.extend(attempts_log)
        num_success = sum(successes)
        results["provers"][name] = {
            "config_path": str(config_path),
            "successes": successes,
            "num_success": num_success,
            "pass_at_k": pass_at_k_dict,
        }
        for k in k_values:
            logger.info(f"  {name} PASS@{k}: {pass_at_k_dict[k]:.4f}")

    results["attempts"] = all_attempts

    summary_path = experiment_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, data in results["provers"].items():
        if "error" in data:
            logger.info(f"  {name}: ERROR {data['error']}")
        else:
            p = data["pass_at_k"]
            k1 = p.get(1, 0.0)
            k_last = p.get(num_attempts, 0.0)
            logger.info(f"  {name}: PASS@1={k1:.4f} PASS@{num_attempts}={k_last:.4f}")
    logger.info(f"\nResults in: {experiment_dir}")
    logger.info(f"  - Prover attempt logs: {experiment_dir}/<prover_name>.jsonl")
    logger.info(f"  - Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

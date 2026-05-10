#!/usr/bin/env python3
"""
Generate solution attempts (without verification) from a config file.

This script uses the same config format as `scripts/src/eval/test_prover.py`:
  - prover: path to prover YAML
  - dataset_path: JSONL dataset path
  - split: valid/test
  - num_attempts: attempts per problem
  - limit: optional number of problems
  - start_from: optional global offset (0-indexed)
  - dataset_name: optional override (default: dataset_path stem)

Outputs:
  results/<dataset_name>/<split>/<experiment_id>/<problem>/attempts.jsonl
Each line contains generated code and generation metadata. Correctness is not checked.

The subdirectory `experiment_id` is taken from (in order): CLI `--experiment-id`, YAML key
`experiment_id`, or a timestamp — use the first two when a driver must know the path in advance.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

from rwens.models.conf import dict_to_config as prover_dict_to_config, prover_from_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prover_from_config(prover_config_path: Path):
    logger.info("Loading prover configuration from %s", prover_config_path)
    with open(prover_config_path, "r", encoding="utf-8") as f:
        prover_config_dict = yaml.safe_load(f)
    prover_config = prover_dict_to_config(prover_config_dict)
    return prover_from_config(prover_config)


def load_dataset(dataset_path: Path, split: str) -> List[Dict]:
    problems: List[Dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("split") == split:
                problems.append(obj)
    return problems


def count_existing_attempt_rows(attempts_path: Path) -> int:
    """Count non-empty JSONL rows already present in attempts file."""
    if not attempts_path.exists():
        return 0
    count = 0
    with open(attempts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to test_prover-style config YAML",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help=(
            "Override the results subdirectory name (default: from YAML `experiment_id` "
            "or a timestamp). Use a fixed id when a driver script must know the output path."
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prover_config_path = Path(config["prover"])
    dataset_path = Path(config["dataset_path"])
    split = config["split"]
    num_attempts = int(config.get("num_attempts", 16))
    batch_size = int(config.get("batch_size", 1))
    limit = config.get("limit")
    start_from = int(config.get("start_from", 0))
    dataset_name = config.get("dataset_name") or dataset_path.stem
    ensemble_size_override = config.get("ensemble_size")

    prover = load_prover_from_config(prover_config_path)
    if ensemble_size_override is not None and hasattr(prover, "set_ensemble_size"):
        prover.set_ensemble_size(int(ensemble_size_override))
        logger.info("Applied ensemble_size override from test config: %s", ensemble_size_override)
    experiment_id = (
        args.experiment_id
        or config.get("experiment_id")
        or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    results_dir = Path("results") / dataset_name / split / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)

    all_problems = load_dataset(dataset_path, split)
    if start_from > 0:
        if start_from >= len(all_problems):
            raise SystemExit(f"start_from ({start_from}) is >= total problems ({len(all_problems)})")
        problems = all_problems[start_from:]
    else:
        problems = all_problems
    if limit is not None:
        problems = problems[: int(limit)]

    logger.info("Dataset: %s", dataset_name)
    logger.info("Split: %s", split)
    logger.info("Problems selected: %d (out of %d)", len(problems), len(all_problems))
    logger.info("Attempts per problem: %d", num_attempts)
    logger.info("Batch size: %d", batch_size)
    logger.info("Writing outputs to: %s", results_dir)

    total_attempts_written = 0
    total_non_empty_codes = 0
    total_problem_elapsed_s = 0.0
    all_raw_response_token_counts: List[int] = []

    for local_idx, problem in enumerate(problems):
        global_idx = start_from + local_idx + 1
        name = problem["name"]
        header = problem["header"].replace("import Aesop\n", "")
        statement = header + problem["formal_statement"]

        problem_dir = results_dir / name
        problem_dir.mkdir(parents=True, exist_ok=True)
        out_file = problem_dir / "attempts.jsonl"
        existing_rows = count_existing_attempt_rows(out_file)
        if existing_rows >= num_attempts:
            logger.info(
                "[Problem %03d/%03d] %s - skip (existing attempts=%d >= requested=%d)",
                global_idx,
                len(problems),
                name,
                existing_rows,
                num_attempts,
            )
            continue

        remaining_attempts = num_attempts - existing_rows
        attempt_start = existing_rows + 1
        logger.info("[Problem %03d/%03d] %s", global_idx, len(problems), name)
        success_count = 0
        gen_t0 = time.perf_counter()

        generated_codes: List[str] = []
        errors: List[str | None] = []
        raw_response_token_counts: List[int | None] = []
        try:
            if hasattr(prover, "generate_batch_with_metadata"):
                batch_rows = prover.generate_batch_with_metadata(
                    statement,
                    n_attempts=remaining_attempts,
                    batch_size=batch_size,
                )
                generated_codes = [str(r.get("final_code") or "") for r in batch_rows]
                errors = [r.get("error") for r in batch_rows]
                raw_response_token_counts = [r.get("raw_response_num_tokens") for r in batch_rows]
            elif hasattr(prover, "generate_batch"):
                batch_out = prover.generate_batch(
                    statement,
                    n_attempts=remaining_attempts,
                    batch_size=batch_size,
                )
                generated_codes = [g or "" for g in batch_out]
                errors = [None] * len(generated_codes)
                raw_response_token_counts = [None] * len(generated_codes)
            else:
                for _ in range(remaining_attempts):
                    try:
                        generated = prover.generate(statement)
                        generated_codes.append(generated or "")
                        errors.append(None)
                        raw_response_token_counts.append(None)
                    except Exception as e:
                        generated_codes.append("")
                        errors.append(str(e))
                        raw_response_token_counts.append(None)
        except Exception as e:
            generated_codes = ["" for _ in range(remaining_attempts)]
            errors = [str(e) for _ in range(remaining_attempts)]
            raw_response_token_counts = [None for _ in range(remaining_attempts)]

        with open(out_file, "a", encoding="utf-8") as f:
            for i in range(remaining_attempts):
                attempt_num = attempt_start + i
                final_code = generated_codes[i] if i < len(generated_codes) else ""
                err = errors[i] if i < len(errors) else "Generation failed"
                raw_resp_tokens = (
                    raw_response_token_counts[i] if i < len(raw_response_token_counts) else None
                )
                generated_ok = bool(final_code.strip())
                if generated_ok:
                    success_count += 1
                    total_non_empty_codes += 1
                if isinstance(raw_resp_tokens, int):
                    all_raw_response_token_counts.append(raw_resp_tokens)
                total_attempts_written += 1

                rec = {
                    "problem_name": name,
                    "problem_idx": global_idx,
                    "attempt_num": attempt_num,
                    "success": generated_ok,
                    "final_code": final_code,
                    "error": err if not generated_ok else None,
                    "raw_response_num_tokens": raw_resp_tokens,
                    "generation_only": True,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        gen_elapsed_s = time.perf_counter() - gen_t0
        total_problem_elapsed_s += gen_elapsed_s
        problem_tokens = [x for x in raw_response_token_counts if isinstance(x, int)]
        avg_problem_tokens = (
            sum(problem_tokens) / len(problem_tokens) if problem_tokens else math.nan
        )
        logger.info(
            "  generated non-empty codes: %d/%d (new attempts, elapsed: %.2fs, avg raw response tokens: %s)",
            success_count,
            remaining_attempts,
            gen_elapsed_s,
            f"{avg_problem_tokens:.2f}" if not math.isnan(avg_problem_tokens) else "n/a",
        )

    overall_avg_tokens = (
        (sum(all_raw_response_token_counts) / len(all_raw_response_token_counts))
        if all_raw_response_token_counts
        else math.nan
    )
    avg_problem_elapsed = (
        (total_problem_elapsed_s / len(problems)) if problems else math.nan
    )
    logger.info(
        "Summary: attempts=%d, non-empty=%d, avg problem elapsed=%.2fs, avg raw response tokens=%s",
        total_attempts_written,
        total_non_empty_codes,
        avg_problem_elapsed if not math.isnan(avg_problem_elapsed) else 0.0,
        f"{overall_avg_tokens:.2f}" if not math.isnan(overall_avg_tokens) else "n/a",
    )
    logger.info("Done. Results: %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


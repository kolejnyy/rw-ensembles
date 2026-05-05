"""
Test script for evaluating FormalProver on a JSONL dataset (minif2f, ProofNet, etc.).

Expects each line to have: name, split, header, formal_statement (same schema as minif2f/ProofNet).

This script:
1. Reads configuration from a YAML file
2. Loads the formal prover from the config
3. Reads the dataset JSONL and filters by split (valid or test)
4. Runs the prover multiple times on each problem
5. Calculates PASS@k metrics for k=1,2,4,8,16
6. Logs all attempts and saves results under results/<dataset_name>/<split>/<experiment_id>

Config YAML: prover, dataset_path, split, num_attempts, limit, start_from.
Optional: dataset_name (default: stem of dataset_path), batch_size (default 16; used when
prover has prove_batch to run all attempts in batched mode).
Successes per problem are printed only at the end of each problem (after all attempts).
"""

import argparse
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from invpro.models.conf import dict_to_config as prover_dict_to_config, prover_from_config
from invpro.utils.metrics import pass_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prover_from_config(prover_config_path: Path):
    """
    Load and instantiate a formal prover from a configuration file.
    
    Args:
        prover_config_path: Path to the prover configuration YAML file
        
    Returns:
        FormalProver instance
    """
    logger.info(f"Loading prover configuration from {prover_config_path}")
    with open(prover_config_path, "r", encoding="utf-8") as f:
        prover_config_dict = yaml.safe_load(f)
    
    logger.info("Initializing prover from configuration...")
    prover_config = prover_dict_to_config(prover_config_dict)
    prover = prover_from_config(prover_config)
    
    return prover


def load_dataset(dataset_path: Path, split: str) -> List[Dict]:
    """
    Load and filter a JSONL dataset by split (same schema as minif2f/ProofNet).
    
    Args:
        dataset_path: Path to the dataset JSONL file
        split: Split to filter by ("valid" or "test")
        
    Returns:
        List of problem dictionaries with name, header, formal_statement
    """
    logger.info(f"Loading dataset from {dataset_path}")
    problems = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problem = json.loads(line)
                if problem.get("split") == split:
                    problems.append(problem)
    
    logger.info(f"Found {len(problems)} problems in {split} split")
    return problems


def calculate_k_values(num_attempts: int) -> List[int]:
    """
    Calculate k values for PASS@k metrics.
    
    Uses powers of 2 up to num_attempts, plus num_attempts itself.
    
    Args:
        num_attempts: Total number of attempts
        
    Returns:
        List of k values
    """
    k_values = [1]
    k = 2
    while k <= num_attempts:
        k_values.append(k)
        k *= 2
    # Ensure we don't exceed num_attempts
    k_values = [k for k in k_values if k <= num_attempts]
    if k_values[-1] != num_attempts:
        k_values.append(num_attempts)
    
    return k_values


def process_problem(
    prover,
    problem: Dict,
    problem_idx: int,
    total_problems: int,
    num_attempts: int,
    batch_size: int,
    problem_dir: Path,
    k_values: List[int],
) -> tuple[List[bool], Dict[int, float]]:
    """
    Process a single problem: run attempts (batched if prover has prove_batch), log results, calculate PASS@k.
    Prints successes for this problem only once at the end.
    """
    problem_name = problem["name"]
    header = problem["header"].replace("import Aesop\n", "")
    formal_statement = problem["formal_statement"]
    problem_statement = header + formal_statement

    problem_num_str = f"{problem_idx:03d}/{total_problems}"
    problem_name_display = (problem_name[:20] if len(problem_name) >= 20 else problem_name.ljust(20))
    log_file = problem_dir / "attempts.jsonl"

    def result_to_success(r: Optional[Dict]) -> bool:
        return r is not None and r.get("success", False)

    def write_attempt_log(attempt_num: int, success: bool, result: Optional[Dict], error: Optional[str] = None):
        attempt_log = {
            "problem_name": problem_name,
            "problem_idx": problem_idx,
            "attempt_num": attempt_num,
            "success": success,
            "final_code": result.get("final_code", "") if result else "",
            "error": result.get("error") if result else error,
            "steps_count": len(result.get("steps", [])) if result else 0,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(attempt_log, ensure_ascii=False) + "\n")

    successes: List[bool] = []

    if hasattr(prover, "prove_batch"):
        # Run all attempts at once with prove_batch
        try:
            results = prover.prove_batch(
                problem_statement,
                n_attempts=num_attempts,
                batch_size=batch_size,
            )
            for attempt_num, result in enumerate(results, start=1):
                success = result_to_success(result)
                successes.append(success)
                write_attempt_log(attempt_num, success, result)
        except Exception as e:
            for attempt_num in range(1, num_attempts + 1):
                successes.append(False)
                write_attempt_log(attempt_num, False, None, error=str(e))
    else:
        # Sequential prove() per attempt
        for attempt_num in range(1, num_attempts + 1):
            try:
                result = prover.prove(problem_statement)
                success = result_to_success(result)
                successes.append(success)
                write_attempt_log(attempt_num, success, result)
            except Exception as e:
                successes.append(False)
                write_attempt_log(attempt_num, False, None, error=str(e))

    attempt_symbols = ["✓" if s else "✗" for s in successes]
    attempts_str = "".join(attempt_symbols)
    line = f"[Problem {problem_num_str}][{problem_name_display}] Attempts: [{attempts_str}]"
    print(line)

    problem_pass_at_k = {k: pass_at_k(successes, k) for k in k_values}
    logger.info(f"  PASS@k for {problem_name}:")
    for k in k_values:
        logger.info(f"    PASS@{k}: {problem_pass_at_k[k]:.4f}")

    return successes, problem_pass_at_k


def save_final_metrics(
    results_dir: Path,
    total_problems: int,
    cumulative_pass_at_k_sum: Dict[int, float],
    k_values: List[int],
    prover_config: Optional[Dict] = None,
):
    """
    Calculate, save, and display final metrics.
    
    Args:
        results_dir: Directory to save metrics
        total_problems: Total number of problems processed
        cumulative_pass_at_k_sum: Cumulative sum of PASS@k values
        k_values: List of k values
        prover_config: Optional prover configuration dict to include in metrics
    """
    # Calculate final metrics
    final_metrics = {
        "total_problems": total_problems,
        "pass_at_k": {},
    }
    
    # Add prover configuration if provided
    if prover_config is not None:
        final_metrics["prover_config"] = prover_config
    
    for k in k_values:
        avg_pass_at_k = cumulative_pass_at_k_sum[k] / total_problems if total_problems > 0 else 0.0
        final_metrics["pass_at_k"][k] = {
            "average": avg_pass_at_k,
        }
    
    # Save final metrics
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    
    # Display final results
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total problems: {total_problems}")
    for k in k_values:
        avg_pass_at_k = cumulative_pass_at_k_sum[k] / total_problems if total_problems > 0 else 0.0
        logger.info(f"PASS@{k}: {avg_pass_at_k:.4f} ({avg_pass_at_k*100:.2f}%)")
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"  - Problem subfolders: {results_dir}/<problem_name>/attempts.jsonl")
    logger.info(f"  - Metrics: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test FormalProver on a JSONL dataset (minif2f, ProofNet, etc.)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file (e.g., configs/testing/minif2f-naive-qwen7B-sft.yaml)",
    )
    args = parser.parse_args()
    
    # Load configuration from YAML
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Extract parameters from config
    prover_config_path = Path(config["prover"])
    dataset_path = Path(config["dataset_path"])
    split = config["split"]
    num_attempts = config.get("num_attempts", 16)
    batch_size = config.get("batch_size", 16)  # used when prover has prove_batch
    limit = config.get("limit")
    start_from = config.get("start_from", 0)  # 0-indexed starting point
    dataset_name = config.get("dataset_name") or dataset_path.stem
    ensemble_size_override = config.get("ensemble_size")
    
    # Load prover configuration as dict for metrics
    prover_config_dict = None
    try:
        with open(prover_config_path, "r", encoding="utf-8") as f:
            prover_config_dict = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load prover config for metrics: {e}")
    
    # Load and instantiate prover
    prover = load_prover_from_config(prover_config_path)
    if ensemble_size_override is not None and hasattr(prover, "set_ensemble_size"):
        prover.set_ensemble_size(int(ensemble_size_override))
        logger.info(f"Applied ensemble_size override from test config: {ensemble_size_override}")
    
    # Create experiment ID from datetime
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory (e.g. results/proofnet/valid/... or results/minif2f/valid/...)
    results_dir = Path("results") / dataset_name / split / experiment_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Dataset: {dataset_name} | Results will be saved to: {results_dir}")
    
    # Load dataset
    problems = load_dataset(dataset_path, split)
    
    # Apply start_from offset (0-indexed)
    if start_from > 0:
        if start_from >= len(problems):
            logger.error(f"start_from ({start_from}) is >= total problems ({len(problems)})")
            return 1
        problems = problems[start_from:]
        logger.info(f"Starting from problem index {start_from} (skipping first {start_from} problems)")
    
    # Apply limit if specified
    if limit is not None:
        problems = problems[: limit]
        logger.info(f"Limited to {len(problems)} problems")
    
    total_problems_in_dataset = len(load_dataset(dataset_path, split))  # Original count for display
    logger.info(f"Processing {len(problems)} problems (out of {total_problems_in_dataset} total in {split} split)")
    if hasattr(prover, "prove_batch"):
        logger.info(f"Prover has prove_batch; running all {num_attempts} attempts per problem in batched mode (batch_size={batch_size})")

    # Calculate k values for metrics
    k_values = calculate_k_values(num_attempts)
    
    # Initialize metrics
    total_problems = len(problems)
    cumulative_pass_at_k_sum: Dict[int, float] = {k: 0.0 for k in k_values}
    
    # Process each problem (problem_idx is 1-indexed for display, but accounts for start_from offset)
    for local_idx, problem in enumerate(problems):
        problem_idx = start_from + local_idx + 1  # Global 1-indexed position
        # Create subfolder for this problem
        problem_dir = results_dir / problem["name"]
        problem_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the problem
        successes, problem_pass_at_k = process_problem(
            prover=prover,
            problem=problem,
            problem_idx=problem_idx,
            total_problems=total_problems,
            num_attempts=num_attempts,
            batch_size=batch_size,
            problem_dir=problem_dir,
            k_values=k_values,
        )
        
        # Update cumulative metrics
        for k in k_values:
            cumulative_pass_at_k_sum[k] += problem_pass_at_k[k]
        
        # Display cumulative averages (based on problems processed so far, not global index)
        problems_processed = local_idx + 1
        for k in k_values:
            cumulative_avg = cumulative_pass_at_k_sum[k] / problems_processed
            logger.info(f"    PASS@{k} cumulative avg: {cumulative_avg:.4f}")
    
    # Save and display final metrics
    save_final_metrics(
        results_dir=results_dir,
        total_problems=total_problems,
        cumulative_pass_at_k_sum=cumulative_pass_at_k_sum,
        k_values=k_values,
        prover_config=prover_config_dict,
    )


if __name__ == "__main__":
    main()

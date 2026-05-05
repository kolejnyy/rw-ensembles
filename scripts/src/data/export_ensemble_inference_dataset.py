#!/usr/bin/env python3
"""
Export a JSONL dataset of EnsembleLLMProver inference candidates (one row per variant).

For each benchmark problem, enumerates the same candidate set used at inference time
(``list_inference_candidates`` with shuffle disabled: original first, then augmentations).

Rows use the benchmark ``name`` for the original statement (variant index 0) and
``<original_problem_name>_v<n>`` for augmentations (``n >= 1``). If a rewrite closes the goal immediately,
``formal_statement`` is stored as the literal ``no goals`` (per project convention for that case).

Each row includes ``energy_score`` (float or null) from the rewriting module's energy heuristic
when configured (e.g. theorem_surprise mean log-prob), plus ``energy_config_tag`` identifying
which heuristic was active. Instant-rewrite variants omit numeric energy (null).

Example:
  python scripts/src/data/export_ensemble_inference_dataset.py \\
    --experiment minif2f-ensemble-candidates-valid \\
    --prover-config configs/models/ensemble-llm-prover-15-2-theorem-surprise.yaml \\
    --split valid --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import string
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from invpro.dataset.utils import split_declarations_theorem_proof
from invpro.models.conf import dict_to_config as prover_dict_to_config, prover_from_config
from invpro.models.ensemble_llm import EnsembleLLMProver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _random_suffix(length: int = 16) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def load_dataset_rows(dataset_path: Path, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("split") == split:
                rows.append(obj)
    return rows


def build_problem_statement(problem: Dict[str, Any]) -> str:
    header = str(problem["header"]).replace("import Aesop\n", "")
    return header + str(problem["formal_statement"])


def row_for_variant(
    base: Dict[str, Any],
    orig_name: str,
    variant_id: int,
    full_statement: str,
    rw_tactics: List[str],
    instant_rewrite_solved: bool,
    energy_score: Optional[float],
    energy_config_tag: str,
) -> Dict[str, Any]:
    out = dict(base)
    out["name"] = orig_name if variant_id == 0 else f"{orig_name}_v{variant_id}"
    out["original_name"] = orig_name
    out["rw_tactics"] = rw_tactics
    out["instant_rewrite_solved"] = instant_rewrite_solved
    out["ensemble_full_statement"] = full_statement
    out["energy_score"] = energy_score
    out["energy_config_tag"] = energy_config_tag

    if instant_rewrite_solved:
        out["header"] = str(base["header"]).replace("import Aesop\n", "")
        out["formal_statement"] = "no goals"
        out["goal"] = "no goals"
        return out

    try:
        decls, theorem_stmt, _proof_body = split_declarations_theorem_proof(full_statement)
    except ValueError:
        logger.warning(
            "Could not split candidate for %s v%s; storing stub formal_statement",
            orig_name,
            variant_id,
        )
        out["formal_statement"] = ""
        out["goal"] = ""
        return out

    out["header"] = decls
    out["formal_statement"] = theorem_stmt
    out["goal"] = ""
    return out


def resolve_output_path(data_dir: Path, experiment: str) -> Path:
    name = experiment.strip()
    base_stem = name[:-6] if name.endswith(".jsonl") else name
    path = data_dir / f"{base_stem}.jsonl"
    if path.exists():
        path = data_dir / f"{base_stem}_{_random_suffix()}.jsonl"
        logger.info("Output exists; using %s", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/minif2f.jsonl"),
        help="Input JSONL (default: data/minif2f.jsonl)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Logical experiment name; output file data/<name>.jsonl",
    )
    parser.add_argument("--split", type=str, default="valid", help="Dataset split filter")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of problems to process (default: all)",
    )
    parser.add_argument(
        "--prover-config",
        type=Path,
        required=True,
        help="YAML prover config (class: EnsembleLLMProver)",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Override ensemble_size on the prover after load",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for output JSONL (default: data/)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Exact output JSONL file (overrides experiment/data-dir and collision rename)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (plain logs only)",
    )
    args = parser.parse_args()

    with open(args.prover_config, "r", encoding="utf-8") as f:
        prover_yaml = yaml.safe_load(f)
    prover = prover_from_config(prover_dict_to_config(prover_yaml))
    if not isinstance(prover, EnsembleLLMProver):
        raise SystemExit(
            f"Expected EnsembleLLMProver from {args.prover_config}, got {type(prover).__name__}"
        )
    if args.ensemble_size is not None:
        prover.set_ensemble_size(int(args.ensemble_size))

    energy_tag = getattr(prover._canonicalization_module, "_cache_config_tag", "") or ""

    problems = load_dataset_rows(args.dataset, args.split)
    if args.limit is not None:
        problems = problems[: max(0, int(args.limit))]

    if args.output_path is not None:
        out_path = args.output_path.resolve()
    else:
        out_path = resolve_output_path(args.data_dir, args.experiment)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    try:
        with open(out_path, "w", encoding="utf-8") as out:
            bar = tqdm(
                problems,
                desc="Ensemble candidates",
                unit="problem",
                disable=args.no_progress,
            )
            for problem in bar:
                orig_name = str(problem["name"])
                statement = build_problem_statement(problem)
                candidates = prover.list_inference_candidates(statement, shuffle=False)
                if not candidates:
                    logger.warning("No candidates for %s (parse or rewrite pipeline empty)", orig_name)
                    n_skipped += 1
                    if not args.no_progress:
                        bar.set_postfix_str(f"skip:{orig_name[:24]}…")
                    continue
                for vid, (full_s, rw, instant, energy) in enumerate(candidates):
                    row = row_for_variant(
                        problem,
                        orig_name,
                        vid,
                        full_s,
                        rw,
                        instant,
                        energy,
                        energy_tag,
                    )
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_written += 1
                if not args.no_progress:
                    bar.set_postfix(rows=n_written, variants=len(candidates))
    finally:
        if hasattr(prover, "close"):
            prover.close()

    logger.info(
        "Wrote %d rows to %s (%d problems, %d skipped)",
        n_written,
        out_path,
        len(problems) - n_skipped,
        n_skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

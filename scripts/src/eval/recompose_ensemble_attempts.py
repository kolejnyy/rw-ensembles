#!/usr/bin/env python3
"""
Recompose SinglePassProver outputs on ensemble *variant* problems into proofs of the original theorems.

Each result directory is named after a variant row (e.g. ``problem_v3`` or the seed ``problem``).
For every line in ``attempts.jsonl``, replaces ``final_code`` (proof of the augmented statement) with
``EnsembleLLMProver._compose_to_original`` or, for instant-rewrite / ``no goals`` variants
(``instant_rewrite_solved`` or ``formal_statement == "no goals"``), only the **rw_tactics** lines
as the proof under the original theorem (same as ensemble inference for closed goals).

Uses ``rw_tactics`` and ``original_name`` from the variant row and the **seed** row for imports / theorem.

Writes ``attempts_recomposed.jsonl`` beside ``attempts.jsonl`` unless ``--in-place`` is set.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from invpro.dataset.utils import split_declarations_theorem_proof
from invpro.models.ensemble_llm import EnsembleLLMProver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_statement(problem: Dict[str, Any]) -> str:
    header = str(problem["header"]).replace("import Aesop\n", "")
    return header + str(problem["formal_statement"])


def load_dataset_by_name(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            by_name[str(o["name"])] = o
    return by_name


def original_imports_and_theorem(
    by_name: Dict[str, Dict[str, Any]], original_name: str
) -> Optional[tuple[str, str]]:
    seed = by_name.get(original_name)
    if seed is None:
        return None
    try:
        stmt = _build_statement(seed)
        imports, theorem_stmt, _ = split_declarations_theorem_proof(stmt)
        return imports, theorem_stmt
    except ValueError:
        logger.warning("Could not parse original problem %s for recomposition", original_name)
        return None


def _use_rw_tactics_only_as_proof(
    variant_row: Dict[str, Any],
    final_code: str,
    rw_tactics: List[str],
) -> bool:
    """
    Instant-rewrite variants (``no goals`` in Lean) have no real model proof: the whole proof of the
    original theorem is the ``rw_tactics`` block. Also treat unparseable / empty / ``no goals`` proof
    bodies like the workspace state string.
    """
    if variant_row.get("instant_rewrite_solved"):
        return True
    if str(variant_row.get("formal_statement") or "").strip() == "no goals":
        return True
    if not rw_tactics:
        return False
    try:
        _, _, proof_body = split_declarations_theorem_proof(final_code)
    except ValueError:
        return True
    ps = (proof_body or "").strip().lower()
    if not ps or ps == "no goals" or ps.startswith("no goals"):
        return True
    return False


def recompose_record(
    final_code: str,
    imports: str,
    theorem_stmt: str,
    rw_tactics: List[str],
    variant_row: Dict[str, Any],
) -> str:
    rw = list(rw_tactics or [])
    if _use_rw_tactics_only_as_proof(variant_row, final_code, rw):
        if rw:
            return EnsembleLLMProver.original_statement_plus_rw_proof(
                imports, theorem_stmt, rw
            )
        return final_code if final_code else ""
    if not final_code or not final_code.strip():
        return final_code
    return EnsembleLLMProver._compose_to_original(
        imports, theorem_stmt, rw, final_code
    )


def process_attempts_file(
    attempts_path: Path,
    variant_row: Dict[str, Any],
    orig_parts: tuple[str, str],
    in_place: bool,
    backup: bool,
) -> int:
    imports, theorem_stmt = orig_parts
    rw_tactics = list(variant_row.get("rw_tactics") or [])

    lines_out: List[str] = []
    n_changed = 0
    with open(attempts_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fc = rec.get("final_code") or ""
            new_fc = recompose_record(fc, imports, theorem_stmt, rw_tactics, variant_row)
            if new_fc != fc:
                n_changed += 1
            rec["final_code_augmented"] = fc
            rec["final_code"] = new_fc
            rec["recomposed_to_original"] = True
            rec["recompose_original_name"] = variant_row.get("original_name")
            lines_out.append(json.dumps(rec, ensure_ascii=False))

    out_path = attempts_path if in_place else attempts_path.with_name("attempts_recomposed.jsonl")
    if in_place and backup and attempts_path.exists():
        bak = attempts_path.with_suffix(attempts_path.suffix + ".bak")
        shutil.copy2(attempts_path, bak)
        logger.info("Wrote backup %s", bak)

    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines_out:
            f.write(ln + "\n")

    if not in_place:
        logger.info("Wrote %s (%d lines recomposed from %s)", out_path, len(lines_out), attempts_path)
    else:
        logger.info("Updated %s in place (%d lines changed)", attempts_path, n_changed)
    return len(lines_out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Variant JSONL (same file used for generate_solutions)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory results/<dataset_stem>/<split>/<experiment_id>/ (contains per-problem subdirs)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite attempts.jsonl instead of writing attempts_recomposed.jsonl",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="With --in-place, copy attempts.jsonl to attempts.jsonl.bak first",
    )
    args = parser.parse_args()

    by_name = load_dataset_by_name(args.dataset)
    root = args.results_root
    if not root.is_dir():
        raise SystemExit(f"results-root not a directory: {root}")

    n_dirs = 0
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        pname = sub.name
        variant = by_name.get(pname)
        if variant is None:
            logger.warning("No dataset row for problem dir %s — skip", pname)
            continue
        orig_name = str(variant.get("original_name") or "")
        if not orig_name:
            logger.warning("Variant %s missing original_name — skip", pname)
            continue
        orig_parts = original_imports_and_theorem(by_name, orig_name)
        if orig_parts is None:
            continue
        attempts = sub / "attempts.jsonl"
        if not attempts.exists():
            logger.warning("No attempts.jsonl in %s — skip", sub)
            continue
        process_attempts_file(attempts, variant, orig_parts, args.in_place, args.backup)
        n_dirs += 1

    logger.info("Processed %d problem directories under %s", n_dirs, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

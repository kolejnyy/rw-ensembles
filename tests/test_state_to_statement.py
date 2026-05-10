#!/usr/bin/env python3
"""
Test StateProblemConverter round-trip: state -> problem -> verify states match.

For each theorem in data/minif2f.jsonl:
1. Insert header + theorem statement into a temp Lean file
2. Extract the initial state via LSP
3. Use StateProblemConverter.convert() to transform and verify round-trip
4. If convert returns None, count as failure; otherwise success

Usage:
  python tests/test_state_to_statement.py [--dataset PATH] [--limit N] [--project-root PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from rwens.canonicalization.identity import IdentityModule
from rwens.dataset.utils import split_declarations_theorem_proof
from rwens.utils.state_to_statement import (
    StateProblemConverter,
    extract_theorem_name,
)

SKIP_EXTRACT_FAILURE: set[str] = set()


def build_problem_statement(problem: dict) -> str:
    """Build full Lean problem (header + formal_statement) with := by. Removes import Aesop for speed."""
    header = problem.get("header", "")
    header = header.replace("import Aesop\n", "")
    stmt = problem.get("formal_statement", "")
    return (header.rstrip("\n") + "\n" + stmt.rstrip("\n") + "\n").strip() + "\n"


def load_dataset(dataset_path: Path, split: str = "valid", limit: int | None = None) -> list[dict]:
    """Load problems from JSONL, filtered by split."""
    problems = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p.get("split") == split:
                    problems.append(p)
                    if limit is not None and len(problems) >= limit:
                        break
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_project_root / "data/minif2f.jsonl",
        help="Path to minif2f JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of problems to test",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_project_root,
        help="Lean project root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        help="Dataset split (valid or test)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset if args.dataset.is_absolute() else _project_root / args.dataset
    project_root = args.project_root

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    if not (project_root / "lakefile.lean").exists() and not (project_root / "lakefile.toml").exists():
        print(f"Project root has no lakefile: {project_root}", file=sys.stderr)
        return 1

    problems = load_dataset(dataset_path, args.split, args.limit)
    print(f"Testing {len(problems)} problems from {dataset_path} (split={args.split})")
    print()

    module = IdentityModule(project_root=str(project_root), timeout_seconds=120.0)
    converter = StateProblemConverter(project_root=str(project_root), timeout_seconds=120.0)

    try:
        failures = 0
        for i, problem in enumerate(problems):
            name = problem.get("name", f"problem_{i}")
            if name in SKIP_EXTRACT_FAILURE:
                continue
            try:
                problem_statement = build_problem_statement(problem)
            except Exception as e:
                print(f"[{name}] Failed to build problem: {e}")
                failures += 1
                continue

            try:
                decls, theorem_stmt, _ = split_declarations_theorem_proof(problem_statement)
            except ValueError as e:
                print(f"[{name}] Failed to parse: {e}")
                failures += 1
                continue

            module.reset(decls, theorem_stmt)
            state1 = module.get_current_state()

            if state1 is None or not state1.strip():
                failures += 1
                print(f"[{name}] Could not extract initial state (timeout or empty)")
                continue

            theorem_name = extract_theorem_name(theorem_stmt)
            new_problem = converter.convert(decls, state1, theorem_name)

            if new_problem is None:
                failures += 1
                print(f"[{name}] convert returned None (states don't match or failed round-trip)")
                continue

            # Optional: verify the returned problem parses
            try:
                split_declarations_theorem_proof(new_problem)
            except ValueError as e:
                failures += 1
                print(f"[{name}] convert produced invalid output: {e}")
                continue

        print()
        print(f"Done: {failures} failure(s) out of {len(problems)} problems")
        return 0 if failures == 0 else 1

    finally:
        module.close()
        converter.close()


if __name__ == "__main__":
    sys.exit(main())

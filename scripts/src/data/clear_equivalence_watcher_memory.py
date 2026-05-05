#!/usr/bin/env python3
"""
Remove selected problems from equivalence watcher memory files.

Targets under a run directory:
- logs/equivalence_watcher_state.json
- logs/equivalence_watcher_results.jsonl

This allows re-processing those problems on the next watcher run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "run_dir",
        type=Path,
        help="Run directory, e.g. data/rewritings_pipeline/minif2f_aug_test_split",
    )
    p.add_argument(
        "problems",
        nargs="+",
        help="Problem names to clear (e.g. imo_1977_p6 aime_1983_p1).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually write changes (default is dry-run).",
    )
    return p.parse_args()


def _path_matches_problem(path_s: str, problems: set[str]) -> bool:
    p = Path(path_s)
    # expected shape includes .../<split>/<problem>/<variant>.lean
    if len(p.parts) >= 2 and p.parts[-2] in problems:
        return True
    return any(seg in problems for seg in p.parts)


def _load_state(path: Path) -> list[str]:
    if not path.is_file():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    raw = obj.get("processed", [])
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw]


def _save_state(path: Path, processed: list[str]) -> None:
    payload = {"processed": sorted(processed)}
    path.write_text(json.dumps(payload, indent=0, sort_keys=True) + "\n", encoding="utf-8")


def _load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _save_results(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    logs_dir = run_dir / "logs"
    state_path = logs_dir / "equivalence_watcher_state.json"
    results_path = logs_dir / "equivalence_watcher_results.jsonl"
    problems = set(args.problems)

    processed_before = _load_state(state_path)
    processed_after = [p for p in processed_before if not _path_matches_problem(p, problems)]
    state_removed = len(processed_before) - len(processed_after)

    results_before = _load_results(results_path)
    results_after: list[dict[str, Any]] = []
    results_removed = 0
    for row in results_before:
        name = str(row.get("name", ""))
        lean_path = str(row.get("lean_path", ""))
        if name in problems or _path_matches_problem(lean_path, problems):
            results_removed += 1
            continue
        results_after.append(row)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Run dir: {run_dir}")
    print(f"Problems: {sorted(problems)}")
    print(f"State file: {state_path} | removed={state_removed} kept={len(processed_after)}")
    print(f"Results file: {results_path} | removed={results_removed} kept={len(results_after)}")

    if args.execute:
        logs_dir.mkdir(parents=True, exist_ok=True)
        _save_state(state_path, processed_after)
        _save_results(results_path, results_after)
        print("Wrote updated watcher memory files.")
    else:
        print("No files modified. Re-run with --execute to apply changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


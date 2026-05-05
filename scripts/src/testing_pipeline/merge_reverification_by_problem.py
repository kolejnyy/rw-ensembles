#!/usr/bin/env python3
"""Merge per-problem JSON files from verify_solutions_one_problem_mp into one CSV + summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solutions_dir", type=Path)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    solutions_dir = args.solutions_dir.resolve()
    indir = args.input_dir or (solutions_dir / ".reverification_by_problem")
    if not indir.is_dir():
        raise SystemExit(f"Not a directory: {indir}")

    paths = sorted(indir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No JSON files in {indir}")

    rows: list[dict[str, object]] = []
    key_set: set[str] = set()
    for json_path in paths:
        with json_path.open("r", encoding="utf-8") as f:
            rec = json.load(f)
        problem = rec.get("problem")
        pak = rec.get("pass_at_k")
        if not isinstance(problem, str) or not isinstance(pak, dict):
            continue
        row: dict[str, object] = {"problem": problem}
        for key, val in pak.items():
            if isinstance(key, str) and key.startswith("pass@"):
                row[key] = val
                key_set.add(key)
        rows.append(row)

    if not rows:
        raise SystemExit("No valid records found in JSON files.")
    all_k_keys = sorted(key_set, key=lambda s: int(s.split("@")[1]))
    out_csv = args.output_csv or (solutions_dir / "reverified_pass_at_k.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem"] + all_k_keys)
        w.writeheader()
        for row in sorted(rows, key=lambda r: str(r["problem"])):
            w.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

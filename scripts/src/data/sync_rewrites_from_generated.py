#!/usr/bin/env python3
"""
Rebuild selected rows in ``rewrites_dataset.jsonl`` from on-disk ``generated/`` theorem files.

For each problem name, reads ``<generated-root>/<dataset>/<split>/<problem>`` (same layout as
``generate_rewrites_openai``), extracts theorem blocks, and emits records compatible with
:func:`~rwens.dataset.rewriting.jsonl_records.make_rewrite_dataset_record`.  Replaces the
contiguous block of lines for that ``original_name`` while preserving global JSONL order.

See :mod:`rwens.dataset.rewriting.sync_rewrites_from_generated` for the library functions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rwens.dataset.rewriting.sync_rewrites_from_generated import (
    load_benchmark_index,
    load_jsonl,
    merge_rewrites_jsonl,
    write_jsonl,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing rewrites_dataset.jsonl and generated/",
    )
    p.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=root / "data" / "minif2f.jsonl",
        help="Benchmark JSONL (split, name, formal_statement, header, …).",
    )
    p.add_argument(
        "--rewrites-jsonl",
        type=Path,
        default=None,
        help="Default: <run-dir>/rewrites_dataset.jsonl",
    )
    p.add_argument(
        "--generated-subdir",
        type=str,
        default="generated",
        help="Subdirectory of run-dir containing dataset/split/problem files.",
    )
    p.add_argument("problems", nargs="+", help="Problem names (benchmark seed names).")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    rewrites_path = (args.rewrites_jsonl or (run_dir / "rewrites_dataset.jsonl")).resolve()
    generated_root = (run_dir / args.generated_subdir).resolve()
    problems = set(args.problems)

    if not args.dataset_jsonl.is_file():
        raise SystemExit(f"Missing --dataset-jsonl: {args.dataset_jsonl}")
    if not rewrites_path.is_file():
        raise SystemExit(f"Missing rewrites JSONL: {rewrites_path}")
    if not generated_root.is_dir():
        raise SystemExit(f"Missing generated root: {generated_root}")

    benchmark_index = load_benchmark_index(args.dataset_jsonl.resolve())
    rows = load_jsonl(rewrites_path)
    new_rows, logs = merge_rewrites_jsonl(
        rows=rows,
        problems=problems,
        generated_root=generated_root,
        benchmark_index=benchmark_index,
    )

    for line in logs:
        print(line)
    print(f"Total lines: {len(rows)} -> {len(new_rows)}")

    if args.dry_run:
        print("Dry-run: not writing.")
        return 0

    backup = rewrites_path.with_suffix(rewrites_path.suffix + ".bak")
    backup.write_text(rewrites_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Backup: {backup}")
    write_jsonl(rewrites_path, new_rows)
    print(f"Wrote {rewrites_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

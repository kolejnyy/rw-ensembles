#!/usr/bin/env python3
"""
Truncate selected attempts.jsonl files to the first N non-empty rows.

Default behavior:
- scan one level under results root: <root>/*/attempts.jsonl
- match files with exactly 128 non-empty rows
- keep first 64 non-empty rows
- dry-run by default

Use --execute to apply changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def count_non_empty_lines(lines: list[str]) -> int:
    return sum(1 for line in lines if line.strip())


def first_k_non_empty_lines(lines: list[str], k: int) -> list[str]:
    out: list[str] = []
    kept = 0
    for line in lines:
        if not line.strip():
            continue
        out.append(line.rstrip("\n"))
        kept += 1
        if kept >= k:
            break
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("results/minif2f-aug/valid/deepseek-single-pass"),
        help="Root directory containing per-problem subdirs with attempts.jsonl",
    )
    p.add_argument(
        "--attempts-file",
        type=str,
        default="attempts.jsonl",
        help="Attempt filename under each problem directory",
    )
    p.add_argument(
        "--match-rows",
        type=int,
        default=128,
        help="Only process files with exactly this many non-empty rows",
    )
    p.add_argument(
        "--keep-rows",
        type=int,
        default=64,
        help="Number of non-empty rows to keep",
    )
    p.add_argument(
        "--backup-ext",
        type=str,
        default=".bak",
        help="Backup extension written next to modified files",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Apply modifications (default is dry-run)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    if args.match_rows <= 0 or args.keep_rows <= 0:
        raise SystemExit("--match-rows and --keep-rows must be > 0")
    if args.keep_rows > args.match_rows:
        raise SystemExit("--keep-rows must be <= --match-rows")

    changed = 0
    scanned = 0
    for attempts_path in sorted(root.glob(f"*/{args.attempts_file}")):
        scanned += 1
        text = attempts_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        n = count_non_empty_lines(lines)
        if n != args.match_rows:
            continue

        truncated_lines = first_k_non_empty_lines(lines, args.keep_rows)
        if len(truncated_lines) != args.keep_rows:
            print(f"SKIP malformed (cannot keep {args.keep_rows}): {attempts_path}")
            continue

        print(f"MATCH {attempts_path} rows={n} -> keep={args.keep_rows}")
        changed += 1
        if not args.execute:
            continue

        backup_path = attempts_path.with_suffix(attempts_path.suffix + args.backup_ext)
        backup_path.write_text(text, encoding="utf-8")
        attempts_path.write_text("\n".join(truncated_lines) + "\n", encoding="utf-8")

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(
        f"{mode} complete: scanned={scanned} matched={changed} "
        f"(match_rows={args.match_rows}, keep_rows={args.keep_rows})"
    )
    if not args.execute:
        print("Re-run with --execute to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


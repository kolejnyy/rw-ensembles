#!/usr/bin/env python3
"""
Augment inequality JSONL datasets with split/header metadata.

For each record in input datasets:
- set ``split`` to the configured value (default: ``test``)
- set ``header`` to the shared header loaded from a reference JSONL
  (default: ``data/minif2f-ver.jsonl``)

Writes files in place by default using atomic replace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUTS = [
    Path("data/ineq-seed.jsonl"),
    Path("data/ineq-type1.jsonl"),
    Path("data/ineq-type2.jsonl"),
]
DEFAULT_HEADER_SOURCE = Path("data/minif2f-ver.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def read_shared_header(path: Path) -> str:
    rows = load_jsonl(path)
    headers = {
        str(row["header"])
        for row in rows
        if isinstance(row, dict) and "header" in row and row["header"] is not None
    }
    if not headers:
        raise SystemExit(f"No non-empty 'header' found in {path}")
    if len(headers) > 1:
        raise SystemExit(
            f"Expected exactly one unique header in {path}, found {len(headers)}."
        )
    return next(iter(headers))


def augment_rows(
    rows: list[dict[str, Any]], split_value: str, header_value: str
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    changed = 0
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("Encountered non-object JSON row.")
        new_row = dict(row)
        old_split = new_row.get("split")
        old_header = new_row.get("header")
        new_row["split"] = split_value
        new_row["header"] = header_value
        if old_split != split_value or old_header != header_value:
            changed += 1
        out.append(new_row)
    return out, changed


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl_files",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help=(
            "Input dataset JSONL files. Defaults to "
            "data/ineq-seed.jsonl data/ineq-type1.jsonl data/ineq-type2.jsonl"
        ),
    )
    parser.add_argument(
        "--header-source",
        type=Path,
        default=DEFAULT_HEADER_SOURCE,
        help="JSONL file from which to read a shared `header` value.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Value to write into each row's `split` field (default: test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print what would change.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    header_source = args.header_source.resolve()
    if not header_source.is_file():
        print(f"ERROR: header source not found: {header_source}", file=sys.stderr)
        return 2

    shared_header = read_shared_header(header_source)
    any_changed = False

    for path in args.jsonl_files:
        path = path.resolve()
        if not path.is_file():
            print(f"ERROR: not a file: {path}", file=sys.stderr)
            return 2

        rows = load_jsonl(path)
        out_rows, changed = augment_rows(rows, split_value=args.split, header_value=shared_header)
        status = "would update" if args.dry_run else "updated"
        print(f"{path}: {status} {changed}/{len(rows)} row(s)")
        if changed and not args.dry_run:
            write_jsonl(path, out_rows)
        any_changed = any_changed or changed > 0

    if not any_changed:
        print("No changes needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

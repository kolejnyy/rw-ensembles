#!/usr/bin/env python3
"""
Pad ensemble-export JSONL rows so each benchmark has enough variant slots for evaluation.

When a problem has at most three rows (original + augmentations), duplicate existing
rows with new ``name`` suffixes so the total attempt budget can reach 64 with 16
attempts per variant (needs at least 4 distinct variant rows).

Rules (``name`` pattern: original has no ``_v`` suffix; variants are ``<original_name>_v<n>``):

- 1 row (original only): add three copies named ``_v2``, ``_v3``, ``_v4`` (same payload as original).
- 2 rows (original + ``_v1``): add ``_v3`` (copy of original) and ``_v4`` (copy of ``_v1``).
- 3 rows (original + ``_v1`` + ``_v2``): add ``_v4``, ``_v5``, ``_v6`` copying original, ``_v1``, ``_v2`` respectively.

Problems with four or more rows are left unchanged.

Writes each input file in place by default (atomic replace). Prints paths of files that
were modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def variant_index(name: str, original_name: str) -> int:
    if name == original_name:
        return 0
    prefix = original_name + "_v"
    if name.startswith(prefix):
        return int(name[len(prefix) :])
    raise ValueError(
        f"Row name {name!r} does not match original_name {original_name!r} "
        "(expected equality or ``<original_name>_v<int>``)."
    )


def sort_rows_by_variant(rows: list[dict[str, Any]], original_name: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: variant_index(str(r["name"]), original_name))


def augment_group(
    rows: list[dict[str, Any]],
    original_name: str,
) -> tuple[list[dict[str, Any]], int]:
    """Return (possibly extended rows, number of rows added)."""
    if len(rows) > 3:
        return rows, 0
    sorted_rows = sort_rows_by_variant(rows, original_name)
    n = len(sorted_rows)
    added: list[dict[str, Any]] = []

    def clone_from(src: dict[str, Any], new_suffix: int) -> dict[str, Any]:
        out = copy.deepcopy(src)
        out["name"] = f"{original_name}_v{new_suffix}"
        out["original_name"] = original_name
        return out

    if n == 1:
        base = sorted_rows[0]
        for suf in (2, 3, 4):
            added.append(clone_from(base, suf))
    elif n == 2:
        orig, v1 = sorted_rows[0], sorted_rows[1]
        added.append(clone_from(orig, 3))
        added.append(clone_from(v1, 4))
    elif n == 3:
        orig, a, b = sorted_rows[0], sorted_rows[1], sorted_rows[2]
        added.append(clone_from(orig, 4))
        added.append(clone_from(a, 5))
        added.append(clone_from(b, 6))
    else:
        # n == 0 should not happen for a non-empty group
        pass

    return sorted_rows + added, len(added)


def group_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """
    Return (ordered keys, groups), where key = (split, original_name).

    Order follows first occurrence in ``rows``.
    """
    order: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "original_name" not in row:
            raise SystemExit(f"Row missing original_name: {row.get('name')!r}")
        if "split" not in row:
            raise SystemExit(f"Row missing split: {row.get('name')!r}")
        split = str(row["split"])
        on = str(row["original_name"])
        key = (split, on)
        groups[key].append(row)
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order, dict(groups)


def process_file(path: Path, dry_run: bool) -> tuple[bool, int, list[str]]:
    """
    Returns (modified, total_rows_added, problems_augmented_names).
    """
    rows = load_jsonl(path)
    order, groups = group_rows(rows)
    out_rows: list[dict[str, Any]] = []
    total_added = 0
    augmented: list[str] = []

    for split, on in order:
        bucket = groups[(split, on)]
        names = [str(r["name"]) for r in bucket]
        if len(names) != len(set(names)):
            raise SystemExit(
                f"Duplicate ``name`` in group (split={split!r}, original_name={on!r}): {names}"
            )

        new_bucket, k = augment_group(bucket, on)
        total_added += k
        if k:
            augmented.append(f"{split}:{on}")
        out_rows.extend(new_bucket)

    if len(out_rows) != len(rows) + total_added:
        raise RuntimeError("internal row count mismatch")

    if dry_run:
        return total_added > 0, total_added, augmented

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return total_added > 0, total_added, augmented


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "jsonl_files",
        nargs="+",
        type=Path,
        help="Ensemble candidate JSONL files (e.g. data/foo.jsonl)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write; report per-file stats only.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    augmented_files: list[Path] = []
    for path in args.jsonl_files:
        path = path.resolve()
        if not path.is_file():
            print(f"ERROR: not a file: {path}", file=sys.stderr)
            return 2
        modified, n_add, problems = process_file(path, dry_run=args.dry_run)
        tag = "[dry-run] " if args.dry_run else ""
        if modified:
            augmented_files.append(path)
            print(
                f"{tag}{path}: added {n_add} row(s) across {len(problems)} problem(s): "
                + ", ".join(problems)
            )
        else:
            print(f"{tag}{path}: unchanged")

    if augmented_files:
        print("\nAugmented files:")
        for fp in augmented_files:
            print(fp)
    else:
        print("\nNo files were augmented.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

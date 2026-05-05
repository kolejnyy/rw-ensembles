#!/usr/bin/env python3
"""
Copy per-problem result folders from a source run tree into a destination tree, for every
``name`` field present in ``confirmed_rewrites_dataset.jsonl``.

Each problem directory (e.g. ``<name>/attempts.jsonl``) is copied with :func:`shutil.copytree`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ordered_unique_names(confirmed_jsonl: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with confirmed_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n = row.get("name")
            if not n:
                continue
            s = str(n)
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def main() -> int:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--confirmed-jsonl",
        type=Path,
        default=root / "data" / "rewritings_pipeline" / "minif2f_aug_test_split" / "confirmed_rewrites_dataset.jsonl",
        help="JSONL whose ``name`` fields select which problem dirs to copy.",
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=root / "results" / "minif2f-aug-test" / "test" / "deepseek-single-pass",
        help="Base directory containing subfolders named after ``name``.",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        default=root / "results" / "minif2f-aug" / "test" / "deepseek-single-pass",
        help="Destination base directory (created if missing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying.",
    )
    args = p.parse_args()

    confirmed = args.confirmed_jsonl.resolve()
    src_root = args.source_dir.resolve()
    dst_root = args.dest_dir.resolve()

    if not confirmed.is_file():
        print(f"Missing confirmed JSONL: {confirmed}", file=sys.stderr)
        return 1
    if not src_root.is_dir():
        print(f"Missing source dir: {src_root}", file=sys.stderr)
        return 1

    names = ordered_unique_names(confirmed)
    if not names:
        print("No names found in confirmed JSONL.", file=sys.stderr)
        return 1

    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for name in names:
        src = src_root / name
        dst = dst_root / name
        if not src.is_dir():
            print(f"SKIP (no source dir): {src}", file=sys.stderr)
            missing += 1
            continue
        if args.dry_run:
            print(f"would copy: {src} -> {dst}")
            copied += 1
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"copied: {dst}")
        copied += 1

    print(
        f"Done. names={len(names)} copied={copied} missing_source={missing} dest={dst_root}",
        file=sys.stderr if args.dry_run else sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

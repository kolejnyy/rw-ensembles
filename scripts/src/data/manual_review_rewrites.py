#!/usr/bin/env python3
"""
Interactive terminal review for generated theorem rewrites.

For each variant theorem:
- show original theorem and candidate rewrite
- accept/reject/skip in terminal
- write accepted/rejected variants to separate stage folders
- append a machine-readable JSONL decision log

Supports resume from previous decisions and optional watch mode.

When ``--rewrites-jsonl`` is set, candidates are read from the aggregated dataset
(``rewrites_dataset.jsonl``) instead of scanning ``generated/`` problem files.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from rwens.dataset.rewriting import extract_theorem_blocks, theorem_declared_name

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "rewritings_pipeline"
# Matches generate_openai layout: {output_root}/{run_id}/generated|accepted|discarded|logs/
DEFAULT_RUN_ID = "manual_v6_gpt54_valid_10"
GENERATED_SUBDIR = "generated"
ACCEPTED_SUBDIR = "accepted"
DISCARDED_SUBDIR = "discarded"

# ``foo_v3`` or ``foo_v3_renamed`` (file-based generated layout)
VARIANT_SUFFIX_RE = re.compile(r"^(?P<base>.+)_v(?P<num>\d+)(?P<suffix>_renamed)?$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--run-id",
        type=str,
        default=DEFAULT_RUN_ID,
        help="Experiment folder under output-root; reads generated/, writes accepted/ and discarded/.",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="JSONL decision log. Default: <output-root>/<run-id>/logs/manual_review_decisions.jsonl",
    )
    p.add_argument(
        "--pending-root",
        type=Path,
        default=None,
        help="Override directory to scan for pending problems (default: <output-root>/<run-id>/generated). "
        "Use while migrating old flat layouts.",
    )
    p.add_argument("--dataset-name", type=str, default=None, help="Optional filter (e.g. minif2f)")
    p.add_argument("--split", type=str, default=None, help="Optional filter (e.g. valid)")
    p.add_argument("--start-from", type=int, default=0)
    p.add_argument("--limit-problems", type=int, default=0, help="0 means no limit")
    p.add_argument("--preview-lines", type=int, default=80)
    p.add_argument(
        "--watch-seconds",
        type=int,
        default=0,
        help="If >0, poll for new pending files periodically after reaching current end.",
    )
    p.add_argument(
        "--copy-original-on-accept",
        action="store_true",
        help="Also write original theorem block into accepted folder when first variant is accepted.",
    )
    p.add_argument(
        "--rewrites-jsonl",
        type=Path,
        default=None,
        help=(
            "If set, read variant rows from this JSONL (e.g. rewrites_dataset.jsonl) "
            "instead of scanning generated/ files. Uses fields: dataset_name, split, name, variant, "
            "original_formal_statement, formal_statement."
        ),
    )
    return p.parse_args()


def load_reviewed_keys(path: Path) -> set[tuple[str, str, str, str]]:
    reviewed: set[tuple[str, str, str, str]] = set()
    if not path.is_file():
        return reviewed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            decision = str(row.get("decision", ""))
            if decision not in {"accept", "reject"}:
                continue
            dataset = str(row.get("dataset_name", ""))
            split = str(row.get("split", ""))
            problem = str(
                row.get("problem_name", "")
                or row.get("original_name", "")
                or row.get("name", "")
            )
            variant = str(row.get("variant_name", "") or row.get("variant", ""))
            if dataset and split and problem and variant:
                reviewed.add((dataset, split, problem, variant))
    return reviewed


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_pending_problem_files(
    pending_root: Path,
    dataset_name: str | None,
    split: str | None,
) -> list[tuple[str, str, str, Path]]:
    out: list[tuple[str, str, str, Path]] = []
    if not pending_root.is_dir():
        return out
    for path in sorted(pending_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(pending_root)
        if len(rel.parts) < 3:
            continue
        ds = rel.parts[0]
        sp = rel.parts[1]
        name = path.name
        if dataset_name is not None and ds != dataset_name:
            continue
        if split is not None and sp != split:
            continue
        out.append((ds, sp, name, path))
    return out


def _variant_order_key(theorem_name: str) -> tuple[int, int, str]:
    m = VARIANT_SUFFIX_RE.match(theorem_name)
    if not m:
        return (10**9, 0, theorem_name)
    num = int(m.group("num"))
    # Base ``…_vN`` before ``…_vN_renamed``
    sub = 1 if m.group("suffix") else 0
    return (num, sub, theorem_name)


def split_original_and_variants(problem_name: str, text: str) -> tuple[str | None, list[tuple[str, str]]]:
    original: str | None = None
    variants: list[tuple[str, str]] = []
    for block in extract_theorem_blocks(text):
        tname = theorem_declared_name(block)
        if tname is None:
            continue
        if tname == problem_name:
            original = block
        else:
            variants.append((tname, block))
    variants.sort(key=lambda x: _variant_order_key(x[0]))
    return original, variants


def print_block(title: str, block: str | None, preview_lines: int) -> None:
    print(f"\n===== {title} =====")
    if not block:
        print("<missing>")
        return
    lines = block.rstrip().splitlines()
    max_lines = max(1, preview_lines)
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"... ({len(lines) - max_lines} more lines)")


def clear_screen() -> None:
    """Clear terminal for a fresh side-by-side read cycle."""
    print("\033[2J\033[H", end="", flush=True)


def write_variant(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(block.strip() + "\n", encoding="utf-8")


def _safe_variant_filename(variant: str) -> str:
    return variant.replace("/", "_").replace("\\", "_")


def load_rewrites_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def review_once_jsonl(args: argparse.Namespace) -> tuple[int, int]:
    """One pass over ``--rewrites-jsonl`` rows (each row = one variant)."""
    assert args.rewrites_jsonl is not None
    run_root = args.output_root / args.run_id
    accepted_root = run_root / ACCEPTED_SUBDIR
    discarded_root = run_root / DISCARDED_SUBDIR
    reviewed = load_reviewed_keys(args.log_file)

    all_rows = load_rewrites_jsonl(args.rewrites_jsonl)
    rows: list[dict[str, object]] = []
    for row in all_rows:
        ds = str(row.get("dataset_name", ""))
        sp = str(row.get("split", ""))
        if args.dataset_name is not None and ds != args.dataset_name:
            continue
        if args.split is not None and sp != args.split:
            continue
        rows.append(row)

    if args.start_from > 0:
        rows = rows[args.start_from :]
    if args.limit_problems > 0:
        rows = rows[: args.limit_problems]

    total_actions = 0
    total_candidates = 0
    originals_written: set[tuple[str, str, str]] = set()

    for row_idx, row in enumerate(rows, start=1):
        dataset_name = str(row.get("dataset_name", ""))
        split = str(row.get("split", ""))
        original_name = str(row.get("original_name", "") or "").strip()
        problem_name = original_name if original_name else str(row.get("name", ""))
        variant = str(row.get("variant", ""))
        original = str(row.get("original_formal_statement", "")).strip()
        candidate = str(row.get("formal_statement", "")).strip()

        if not problem_name or not variant:
            continue

        total_candidates += 1
        key = (dataset_name, split, problem_name, variant)
        if key in reviewed:
            continue

        clear_screen()
        print(f"### [{row_idx}/{len(rows)}] {dataset_name}/{split}/{problem_name}  ({variant})")
        print(f"Source: {args.rewrites_jsonl}")
        print_block("ORIGINAL", original or None, args.preview_lines)
        print_block(f"CANDIDATE ({variant})", candidate or None, args.preview_lines)
        print("\n[a] accept   [r] reject   [s] skip   [q] quit")

        while True:
            try:
                choice = input("> ").strip().lower()
            except EOFError:
                choice = "q"
            if choice in {"a", "r", "s", "q"}:
                break
            print("Invalid choice. Use one of: a / r / s / q")

        if choice == "q":
            return total_actions, total_candidates

        now = datetime.now(timezone.utc).isoformat()
        destination_file = ""
        decision = {"a": "accept", "r": "reject", "s": "skip"}[choice]
        safe_v = _safe_variant_filename(variant)
        if choice == "a":
            destination = accepted_root / dataset_name / split / problem_name / f"{safe_v}.lean"
            write_variant(destination, candidate)
            destination_file = str(destination)
            if args.copy_original_on_accept and original:
                ok = (dataset_name, split, problem_name)
                if ok not in originals_written:
                    original_out = accepted_root / dataset_name / split / problem_name / "original.lean"
                    if not original_out.exists():
                        write_variant(original_out, original)
                    originals_written.add(ok)
            reviewed.add(key)
        elif choice == "r":
            destination = discarded_root / dataset_name / split / problem_name / f"{safe_v}.lean"
            write_variant(destination, candidate)
            destination_file = str(destination)
            reviewed.add(key)

        record = {
            "timestamp_utc": now,
            "run_id": args.run_id,
            "run_root": str(run_root),
            "generated_subdir": GENERATED_SUBDIR,
            "accepted_subdir": ACCEPTED_SUBDIR,
            "discarded_subdir": DISCARDED_SUBDIR,
            "dataset_name": dataset_name,
            "split": split,
            "problem_name": problem_name,
            "name": str(row.get("name", "")),
            "original_name": original_name or problem_name,
            "variant_name": variant,
            "variant": variant,
            "decision": decision,
            "source_jsonl": str(args.rewrites_jsonl),
            "destination_file": destination_file,
        }
        append_jsonl(args.log_file, record)
        total_actions += 1
        print(f"Recorded: {decision} ({variant})")

    return total_actions, total_candidates


def review_once(args: argparse.Namespace) -> tuple[int, int]:
    if args.rewrites_jsonl is not None:
        return review_once_jsonl(args)

    run_root = args.output_root / args.run_id
    pending_root = (
        args.pending_root
        if args.pending_root is not None
        else run_root / GENERATED_SUBDIR
    )
    accepted_root = run_root / ACCEPTED_SUBDIR
    discarded_root = run_root / DISCARDED_SUBDIR
    reviewed = load_reviewed_keys(args.log_file)

    files = iter_pending_problem_files(
        pending_root=pending_root,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    if args.start_from > 0:
        files = files[args.start_from :]
    if args.limit_problems > 0:
        files = files[: args.limit_problems]

    total_actions = 0
    total_candidates = 0
    for file_idx, (dataset_name, split, problem_name, path) in enumerate(files, start=1):
        text = path.read_text(encoding="utf-8")
        original, variants = split_original_and_variants(problem_name, text)
        if not variants:
            continue
        print(f"\n\n### [{file_idx}/{len(files)}] {dataset_name}/{split}/{problem_name}")
        print(f"Source: {path}")
        for variant_name, variant_block in variants:
            total_candidates += 1
            key = (dataset_name, split, problem_name, variant_name)
            if key in reviewed:
                continue

            clear_screen()
            print(f"### [{file_idx}/{len(files)}] {dataset_name}/{split}/{problem_name}  ({variant_name})")
            print(f"Source: {path}")
            print_block("ORIGINAL", original, args.preview_lines)
            print_block(f"CANDIDATE {variant_name}", variant_block, args.preview_lines)
            print("\n[a] accept   [r] reject   [s] skip   [q] quit")

            while True:
                try:
                    choice = input("> ").strip().lower()
                except EOFError:
                    choice = "q"
                if choice in {"a", "r", "s", "q"}:
                    break
                print("Invalid choice. Use one of: a / r / s / q")

            if choice == "q":
                return total_actions, total_candidates

            now = datetime.now(timezone.utc).isoformat()
            destination_file = ""
            decision = {"a": "accept", "r": "reject", "s": "skip"}[choice]
            if choice == "a":
                destination = accepted_root / dataset_name / split / problem_name / f"{variant_name}.lean"
                write_variant(destination, variant_block)
                destination_file = str(destination)
                if args.copy_original_on_accept and original:
                    original_out = accepted_root / dataset_name / split / problem_name / "original.lean"
                    if not original_out.exists():
                        write_variant(original_out, original)
                reviewed.add(key)
            elif choice == "r":
                destination = discarded_root / dataset_name / split / problem_name / f"{variant_name}.lean"
                write_variant(destination, variant_block)
                destination_file = str(destination)
                reviewed.add(key)

            record = {
                "timestamp_utc": now,
                "run_id": args.run_id,
                "run_root": str(run_root),
                "generated_subdir": GENERATED_SUBDIR,
                "accepted_subdir": ACCEPTED_SUBDIR,
                "discarded_subdir": DISCARDED_SUBDIR,
                "dataset_name": dataset_name,
                "split": split,
                "problem_name": problem_name,
                "variant_name": variant_name,
                "decision": decision,
                "source_file": str(path),
                "destination_file": destination_file,
            }
            append_jsonl(args.log_file, record)
            total_actions += 1
            print(f"Recorded: {decision} ({variant_name})")

    return total_actions, total_candidates


def main() -> int:
    args = parse_args()
    if args.log_file is None:
        args.log_file = args.output_root / args.run_id / "logs" / "manual_review_decisions.jsonl"
    run_root = args.output_root / args.run_id
    if args.start_from < 0:
        raise SystemExit("start-from must be >= 0")
    if args.limit_problems < 0:
        raise SystemExit("limit-problems must be >= 0")
    if args.preview_lines <= 0:
        raise SystemExit("preview-lines must be > 0")
    if args.watch_seconds < 0:
        raise SystemExit("watch-seconds must be >= 0")

    print("Manual rewrite review")
    print(f"Run id:    {args.run_id}")
    if args.rewrites_jsonl is not None:
        print(f"JSONL:     {args.rewrites_jsonl}")
    else:
        print(
            f"Pending:   {args.pending_root if args.pending_root is not None else run_root / GENERATED_SUBDIR}"
        )
    print(f"Accepted:  {run_root / ACCEPTED_SUBDIR}")
    print(f"Discarded: {run_root / DISCARDED_SUBDIR}")
    print(f"Log:       {args.log_file}")

    while True:
        actions, candidates = review_once(args)
        print(f"\nPass complete. actions={actions}, candidates_seen={candidates}")
        if args.watch_seconds <= 0:
            break
        print(f"Waiting {args.watch_seconds}s for new pending rewrites...")
        try:
            time.sleep(args.watch_seconds)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

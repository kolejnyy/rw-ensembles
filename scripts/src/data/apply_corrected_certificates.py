"""
Apply manually corrected certificates from ``unverified/`` into ``equivalence_certificates/``.

Intended workflow:
1) Watcher writes unresolved drafts under ``unverified/``.
2) You manually edit a draft in place and add a marker line ``**CORRECTED**``.
3) This script scans ``unverified/``, picks only marked files, strips the marker,
   verifies each file in Lean, and moves verified certificates to
   ``equivalence_certificates/``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rwens.utils.metrics import equivalence_certificate_diagnostics_acceptable
from rwens.utils.verifier import ProofVerifier


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_destination_rel(rel: Path) -> Optional[Path]:
    name = rel.name
    if name.endswith(".certificate.unverified.txt"):
        name = name[: -len(".certificate.unverified.txt")] + ".certificate.lean"
    elif name.endswith(".certificate.txt"):
        name = name[: -len(".certificate.txt")] + ".certificate.lean"
    elif name.endswith(".certificate.lean"):
        pass
    elif name.endswith(".lean"):
        stem = name[: -len(".lean")]
        name = f"{stem}.certificate.lean"
    else:
        return None
    return rel.with_name(name)


def _parse_cert_rel_key(rel: Path) -> Optional[tuple[str, str, str, str]]:
    parts = rel.parts
    if len(parts) != 4:
        return None
    dataset, split, problem, fname = parts
    if not fname.endswith(".certificate.lean"):
        return None
    variant = fname[: -len(".certificate.lean")]
    return dataset, split, problem, variant


def _iter_unverified_files(unverified_root: Path) -> list[Path]:
    if not unverified_root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(unverified_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(unverified_root)
        if _normalize_destination_rel(rel) is None:
            continue
        out.append(p)
    return out


def _is_expected_rel(rel: Path) -> bool:
    """
    Require files to mirror dataset/split/problem/file layout.

    Expected:
      unverified/<dataset>/<split>/<problem>/<variant>.certificate.unverified.txt
    """
    return len(rel.parts) == 4


def _strip_corrected_marker(text: str) -> tuple[bool, str]:
    """
    If the first non-empty line is ``**CORRECTED**``, remove it and return (True, cleaned_text).
    Otherwise return (False, original_text).
    """
    lines = text.splitlines()
    first_non_empty = None
    for i, line in enumerate(lines):
        if line.strip():
            first_non_empty = i
            break
    if first_non_empty is None:
        return False, text
    if lines[first_non_empty].strip() != "**CORRECTED**":
        return False, text
    cleaned = lines[:first_non_empty] + lines[first_non_empty + 1 :]
    out = "\n".join(cleaned).strip() + "\n"
    return True, out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=root / "data" / "rewritings_pipeline")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--project-root", type=Path, default=root, help="Lean project root (lake root).")
    p.add_argument("--unverified-root", type=Path, default=None, help="Default: <run-root>/unverified")
    p.add_argument(
        "--cert-root",
        type=Path,
        default=None,
        help="Default: <run-root>/equivalence_certificates",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Default: <run-root>/logs/corrected_certificates_applied.jsonl",
    )
    p.add_argument("--timeout", type=float, default=180.0, help="ProofVerifier timeout per file.")
    p.add_argument("--dry-run", action="store_true", help="Verify only; do not move files.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = (args.output_root / args.run_id).resolve()
    unverified_root = (args.unverified_root or (run_root / "unverified")).resolve()
    cert_root = (args.cert_root or (run_root / "equivalence_certificates")).resolve()
    log_file = (args.log_file or (run_root / "logs" / "corrected_certificates_applied.jsonl")).resolve()

    files = _iter_unverified_files(unverified_root)
    if not files:
        print(f"No candidate files under: {unverified_root}")
        return 0

    verifier = ProofVerifier(
        project_root=str(args.project_root.resolve()),
        initial_imports="import Mathlib\n",
        timeout_seconds=args.timeout,
    )

    marked = 0
    moved = 0
    failed = 0
    skipped_bad_layout = 0
    skipped_unmarked = 0
    for src in files:
        rel = src.relative_to(unverified_root)
        if not _is_expected_rel(rel):
            skipped_bad_layout += 1
            print(
                f"[SKIP layout] {src} "
                "(expected unverified/<dataset>/<split>/<problem>/<file>)"
            )
            continue
        dest_rel = _normalize_destination_rel(rel)
        if dest_rel is None:
            continue
        dst = cert_root / dest_rel
        key = _parse_cert_rel_key(dest_rel)

        row: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "source_file": str(src),
            "destination_file": str(dst),
            "source": "manual_correction_marker",
            "dry_run": bool(args.dry_run),
        }
        if key is not None:
            dataset, split, problem, variant = key
            row.update(
                {
                    "dataset_name": dataset,
                    "split": split,
                    "problem_name": problem,
                    "variant": variant,
                }
            )

        original_text = src.read_text(encoding="utf-8")
        is_marked, text = _strip_corrected_marker(original_text)
        if not is_marked:
            skipped_unmarked += 1
            continue
        marked += 1

        ok, err = verifier.verify(text, diagnostics_ok=equivalence_certificate_diagnostics_acceptable)
        if not ok:
            failed += 1
            row["status"] = "verify_failed"
            row["verify_error"] = err
            _append_jsonl(log_file, row)
            print(f"[FAIL] {src}")
            continue

        row["status"] = "verified"
        if args.dry_run:
            _append_jsonl(log_file, row)
            print(f"[OK dry-run] {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        src.unlink()
        moved += 1
        row["status"] = "moved"
        _append_jsonl(log_file, row)
        print(f"[MOVED] {src} -> {dst}")

    print(
        f"Processed={len(files)} marked={marked} moved={moved} failed={failed} "
        f"skipped_unmarked={skipped_unmarked} skipped_bad_layout={skipped_bad_layout}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

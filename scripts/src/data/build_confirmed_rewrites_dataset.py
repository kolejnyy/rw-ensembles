"""
Build ``confirmed_rewrites_dataset.jsonl`` by attaching certificates to rewrite rows.

Inputs combined:
- run ``rewrites_dataset.jsonl`` (base rows)
- manual review log (accepted/rejected decisions)
- watcher results log (certificate status/path)
- optional corrected-certificate apply log
- files present under ``equivalence_certificates/``

Output keeps only rows with an available certificate (default), and writes the
certificate text into the ``certificate`` field.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _parse_cert_rel_key(rel: Path) -> Optional[tuple[str, str, str, str]]:
    parts = rel.parts
    if len(parts) != 4:
        return None
    dataset, split, problem, fname = parts
    if not fname.endswith(".certificate.lean"):
        return None
    variant = fname[: -len(".certificate.lean")]
    return dataset, split, problem, variant


def _read_cert_index(cert_root: Path) -> dict[tuple[str, str, str, str], Path]:
    out: dict[tuple[str, str, str, str], Path] = {}
    if not cert_root.is_dir():
        return out
    for p in sorted(cert_root.rglob("*.certificate.lean")):
        key = _parse_cert_rel_key(p.relative_to(cert_root))
        if key is None:
            continue
        out[key] = p
    return out


def _decision_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], str]:
    out: dict[tuple[str, str, str, str], str] = {}
    for r in rows:
        decision = str(r.get("decision", ""))
        if decision not in {"accept", "reject"}:
            continue
        dataset = str(r.get("dataset_name", ""))
        split = str(r.get("split", ""))
        problem = str(r.get("problem_name", ""))
        variant = str(r.get("variant", r.get("variant_name", "")))
        if not (dataset and split and problem and variant):
            continue
        out[(dataset, split, problem, variant)] = decision
    return out


def _watcher_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for r in rows:
        dataset = str(r.get("dataset_name", ""))
        split = str(r.get("split", ""))
        problem = str(r.get("name", ""))
        variant = str(r.get("variant", ""))
        if not (dataset and split and problem and variant):
            # backward-compat fallback: infer dataset from path
            lp = str(r.get("lean_path", ""))
            parts = Path(lp).parts
            if len(parts) >= 4:
                dataset = dataset or parts[-4]
        if not (dataset and split and problem and variant):
            continue
        out[(dataset, split, problem, variant)] = r
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=root / "data" / "rewritings_pipeline")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--input-jsonl", type=Path, default=None, help="Default: <run-root>/rewrites_dataset.jsonl")
    p.add_argument(
        "--manual-log",
        type=Path,
        default=None,
        help="Default: <run-root>/logs/manual_review_decisions.jsonl",
    )
    p.add_argument(
        "--watcher-log",
        type=Path,
        default=None,
        help="Default: <run-root>/logs/equivalence_watcher_results.jsonl",
    )
    p.add_argument(
        "--corrected-log",
        type=Path,
        default=None,
        help="Default: <run-root>/logs/corrected_certificates_applied.jsonl",
    )
    p.add_argument(
        "--cert-root",
        type=Path,
        default=None,
        help="Default: <run-root>/equivalence_certificates",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Default: <run-root>/confirmed_rewrites_dataset.jsonl",
    )
    p.add_argument(
        "--include-unaccepted-review",
        action="store_true",
        help="Do not require manual decision == accept.",
    )
    p.add_argument(
        "--include-unconfirmed-watcher",
        action="store_true",
        help="Include rows even if watcher status is not accepted, if certificate file exists.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = (args.output_root / args.run_id).resolve()

    input_jsonl = (args.input_jsonl or (run_root / "rewrites_dataset.jsonl")).resolve()
    manual_log = (args.manual_log or (run_root / "logs" / "manual_review_decisions.jsonl")).resolve()
    watcher_log = (args.watcher_log or (run_root / "logs" / "equivalence_watcher_results.jsonl")).resolve()
    corrected_log = (args.corrected_log or (run_root / "logs" / "corrected_certificates_applied.jsonl")).resolve()
    cert_root = (args.cert_root or (run_root / "equivalence_certificates")).resolve()
    output_jsonl = (args.output_jsonl or (run_root / "confirmed_rewrites_dataset.jsonl")).resolve()

    base_rows = _load_jsonl(input_jsonl)
    decisions = _decision_index(_load_jsonl(manual_log))
    watcher = _watcher_index(_load_jsonl(watcher_log))
    corrected_rows = _load_jsonl(corrected_log)
    corrected_ok: set[tuple[str, str, str, str]] = set()
    for r in corrected_rows:
        if str(r.get("status", "")) not in {"moved", "verified"}:
            continue
        dataset = str(r.get("dataset_name", ""))
        split = str(r.get("split", ""))
        problem = str(r.get("problem_name", ""))
        variant = str(r.get("variant", ""))
        if dataset and split and problem and variant:
            corrected_ok.add((dataset, split, problem, variant))

    cert_index = _read_cert_index(cert_root)

    out_rows: list[dict[str, Any]] = []
    stats = defaultdict(int)
    for row in base_rows:
        dataset = str(row.get("dataset_name", ""))
        split = str(row.get("split", ""))
        problem = str(row.get("original_name", row.get("name", "")))
        variant = str(row.get("variant", ""))
        if not (dataset and split and problem and variant):
            stats["skip_bad_key"] += 1
            continue
        key = (dataset, split, problem, variant)
        is_original = variant == "original"

        decision = decisions.get(key)
        if (not is_original) and (not args.include_unaccepted_review) and decision != "accept":
            stats["skip_not_accepted"] += 1
            continue

        cert_path = cert_index.get(key)
        if (not is_original) and cert_path is None:
            stats["skip_no_certificate_file"] += 1
            continue

        wrow = watcher.get(key)
        watcher_status = str(wrow.get("certificate_status", "")) if wrow else ""
        is_corrected = key in corrected_ok
        if (
            (not is_original)
            and
            not args.include_unconfirmed_watcher
            and (wrow is not None)
            and (watcher_status != "accepted")
            and not is_corrected
        ):
            stats["skip_not_confirmed"] += 1
            continue

        cert_text = cert_path.read_text(encoding="utf-8") if cert_path is not None else ""
        out = dict(row)
        out["certificate"] = cert_text
        out["review_decision"] = decision
        out["watcher_certificate_status"] = watcher_status or None
        if is_original and cert_path is None:
            out["certificate_source"] = "original_placeholder"
        else:
            out["certificate_source"] = "corrected" if is_corrected else "watcher"
        out_rows.append(out)
        stats["kept"] += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows to {output_jsonl}")
    for k in sorted(stats):
        print(f"{k}: {stats[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

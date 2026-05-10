"""
Poll a manual-review ``accepted/`` tree and run :class:`~rwens.equivalence.checker.EquivalenceChecker`
on each new variant ``*.lean`` file (skipping ``original.lean``).

Expected layout (same as :mod:`rwens.dataset.rewriting.manual_review`)::

    {output_root}/{run_id}/accepted/{dataset}/{split}/{problem}/{variant}.lean

Benchmark rows are resolved from ``--dataset-jsonl`` by ``(split, name)`` matching ``problem``.
Optional ``original.lean`` in the same folder is ignored for checking (canonical original comes
from the JSONL ``formal_statement`` unless ``--prefer-original-lean`` is set).

State is persisted under ``{run_root}/logs/equivalence_watcher_state.json``; each run appends a
line to ``equivalence_watcher_results.jsonl``. Successful certificate text can be written under
``{run_root}/equivalence_certificates/...`` mirroring the accepted path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rwens.equivalence.certificate import compose_rwdata_bridge_proof_lines
from rwens.utils.lean_preamble import sanitize_preamble_remove_aesop
from rwens.utils.lean_proof_text import normalize_proof_indent, theorem_header_through_by, theorem_with_sorry
from rwens.utils.statement_parser import parse_theorem_through_by

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_benchmark_index(jsonl_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["split"], row["name"])
            out[key] = row
    return out


def _load_rewrites_index(jsonl_path: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dataset = str(row.get("dataset_name", ""))
            split = str(row.get("split", ""))
            original_name = str(row.get("original_name", ""))
            variant = str(row.get("variant", ""))
            out[(dataset, split, original_name, variant)] = row
    return out


def _load_single_pass_prover(yaml_path: Path, project_root: Path):
    from rwens.models.conf import dict_to_config as prover_dict_to_config
    from rwens.models.conf import prover_from_config
    from rwens.models.conf.types import SinglePassProverConfig
    from rwens.models.single_pass_prover import SinglePassProver

    with yaml_path.open(encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}
    config = prover_dict_to_config(cfg_dict)
    if not isinstance(config, SinglePassProverConfig):
        raise ValueError(
            f"Expected SinglePassProver YAML, got {type(config).__name__}. "
            "Use a file with top-level class: SinglePassProver"
        )
    config.project_root = str(project_root.resolve())
    prover = prover_from_config(config)
    if not isinstance(prover, SinglePassProver):
        raise ValueError(f"Expected SinglePassProver instance, got {type(prover).__name__}")
    return prover


def _parse_accepted_relative(rel: Path) -> tuple[str, str, str, str] | None:
    """Return (dataset, split, problem, variant_stem) or None if layout is not recognized."""
    parts = rel.parts
    if len(parts) != 4:
        return None
    dataset, split, problem, fname = parts
    if not fname.endswith(".lean"):
        return None
    if fname == "original.lean":
        return None
    return dataset, split, problem, Path(fname).stem


def _list_variant_leans(accepted_root: Path) -> list[Path]:
    if not accepted_root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(accepted_root.rglob("*.lean")):
        if p.name == "original.lean":
            continue
        rel = p.relative_to(accepted_root)
        if _parse_accepted_relative(rel) is None:
            continue
        out.append(p)
    return out


def _load_processed(state_path: Path) -> set[str]:
    if not state_path.is_file():
        return set()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    raw = data.get("processed")
    if not isinstance(raw, list):
        return set()
    return {str(x) for x in raw}


def _save_processed(state_path: Path, processed: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"processed": sorted(processed)}
    state_path.write_text(json.dumps(payload, indent=0, sort_keys=True) + "\n", encoding="utf-8")


def _build_unverified_certificate_text(
    *,
    preamble: str,
    original_theorem: str,
    variant_theorem: str,
    bridge_variant_theorem: str | None,
    auxiliary: list[dict[str, Any]],
    certificate_verify_error: str | None,
) -> str:
    """
    Build a manual-debug certificate draft:
    - variant theorem gets ``sorry``
    - failed auxiliary goals get ``sorry``
    - successful auxiliary goals keep their verified proof body
    - original bridge theorem is included unchanged in shape
    """
    clean_preamble = sanitize_preamble_remove_aesop(preamble)
    parts: list[str] = [clean_preamble.rstrip() + "\n\n", theorem_with_sorry(variant_theorem), "\n"]

    if certificate_verify_error:
        parts.append("/- certificate_verify_error\n")
        for line in certificate_verify_error.strip().splitlines():
            parts.append(f"{line}\n")
        parts.append("-/\n\n")

    for a in auxiliary:
        thm = str(a.get("theorem_statement", ""))
        head = theorem_header_through_by(thm)
        ok = bool(a.get("success"))
        if ok:
            body = normalize_proof_indent(str(a.get("verified_proof_body", "") or ""))
        else:
            body = "  sorry\n"
        parts.append(head)
        parts.append(body)
        last_error = str(a.get("last_error", "") or "").strip()
        if last_error:
            parts.append("  /- aux_error\n")
            for line in last_error.splitlines():
                parts.append(f"  {line}\n")
            parts.append("  -/\n")
        parts.append("\n")

    try:
        bridge_src = bridge_variant_theorem or variant_theorem
        displayed_variant_name = parse_theorem_through_by(variant_theorem).name
        bridge_lines = compose_rwdata_bridge_proof_lines(
            original_theorem,
            bridge_src,
            variant_name_override=displayed_variant_name,
        )
        orig = parse_theorem_through_by(original_theorem).raw_through_by.rstrip()
        parts.append(orig + "\n")
        parts.append("\n".join(bridge_lines) + "\n")
    except Exception as ex:
        parts.append("/- failed to compose original bridge theorem\n")
        parts.append(f"{ex}\n")
        parts.append("-/\n")
    return "".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=root / "data" / "rewritings_pipeline")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--accepted-subdir", type=str, default="accepted")
    p.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=root / "data" / "minif2f.jsonl",
        help="JSONL with split, name, formal_statement, header (default: data/minif2f.jsonl).",
    )
    p.add_argument(
        "--rewrites-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional rewrites_dataset.jsonl for recovering paired non-renamed variants "
            "when checking *_renamed files. Default: <run-root>/rewrites_dataset.jsonl if present."
        ),
    )
    p.add_argument("--project-root", type=Path, default=root, help="Lean project root (lake root).")
    p.add_argument("--watch-seconds", type=float, default=30.0, help="Poll interval when watching.")
    p.add_argument(
        "--once",
        action="store_true",
        help="Scan once and exit (no loop).",
    )
    p.add_argument("--timeout", type=float, default=180.0, help="ProofVerifier timeout per check.")
    p.add_argument(
        "--prover-config",
        type=Path,
        default=None,
        help="Optional SinglePassProver YAML for LLM fallback on failed auxiliary goals.",
    )
    p.add_argument(
        "--llm-attempts",
        type=int,
        default=8,
        help="When using --prover-config: n_attempts and batch_size per failed goal.",
    )
    p.add_argument(
        "--write-certificates",
        action="store_true",
        default=True,
        help="Write certificate .lean files under run_root/equivalence_certificates/ (default: on).",
    )
    p.add_argument(
        "--no-write-certificates",
        action="store_false",
        dest="write_certificates",
        help="Do not write certificate files (still log JSONL).",
    )
    p.add_argument(
        "--prefer-original-lean",
        action="store_true",
        help="Use sibling original.lean as original theorem instead of JSONL formal_statement.",
    )
    p.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Override path for processed-files state JSON (default: run logs dir).",
    )
    p.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help="Override path for append-only results log (default: run logs dir).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))

    run_root = (args.output_root / args.run_id).resolve()
    accepted_root = run_root / args.accepted_subdir
    logs_dir = run_root / "logs"
    state_path = args.state_file or (logs_dir / "equivalence_watcher_state.json")
    results_path = args.results_jsonl or (logs_dir / "equivalence_watcher_results.jsonl")
    cert_root = run_root / "equivalence_certificates"
    unverified_root = run_root / "unverified"

    if not args.dataset_jsonl.is_file():
        print(f"Missing dataset JSONL: {args.dataset_jsonl}", file=sys.stderr)
        return 1

    index = _load_benchmark_index(args.dataset_jsonl)
    rewrites_path = args.rewrites_jsonl or (run_root / "rewrites_dataset.jsonl")
    rewrites_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if rewrites_path.is_file():
        rewrites_index = _load_rewrites_index(rewrites_path)

    single_pass = None
    if args.prover_config is not None:
        cfg_path = args.prover_config if args.prover_config.is_absolute() else root / args.prover_config
        if not cfg_path.is_file():
            print(f"Missing prover config: {cfg_path}", file=sys.stderr)
            return 1
        try:
            single_pass = _load_single_pass_prover(cfg_path, root)
        except Exception as ex:
            print(f"Failed to load SinglePassProver: {ex}", file=sys.stderr)
            return 1

    from rwens.equivalence import EquivalenceCertificateStatus, EquivalenceChecker

    checker = EquivalenceChecker(single_pass_prover=single_pass)

    processed = _load_processed(state_path)
    ec = 0

    def process_one(lean_path: Path) -> None:
        nonlocal ec
        rel = lean_path.relative_to(accepted_root)
        parsed = _parse_accepted_relative(rel)
        if parsed is None:
            print(f"SKIP (bad path layout): {lean_path}", file=sys.stderr)
            ec = 1
            return
        _dataset, split, problem, variant = parsed
        key = (split, problem)
        row = index.get(key)
        if row is None:
            print(
                f"SKIP (no benchmark row for split={split!r} name={problem!r}): {lean_path}",
                file=sys.stderr,
            )
            ec = 1
            return

        variant_theorem = lean_path.read_text(encoding="utf-8")
        bridge_variant_theorem: str | None = None
        if variant.endswith("_renamed"):
            base_variant = variant[: -len("_renamed")]
            rw_row = rewrites_index.get((_dataset, split, problem, base_variant))
            if rw_row is not None:
                bridge_variant_theorem = str(rw_row.get("formal_statement", "")).strip()
                if bridge_variant_theorem and not bridge_variant_theorem.endswith("\n"):
                    bridge_variant_theorem += "\n"
        orig_path = lean_path.parent / "original.lean"
        if args.prefer_original_lean and orig_path.is_file():
            original_theorem = orig_path.read_text(encoding="utf-8")
        else:
            original_theorem = row["formal_statement"]

        preamble_raw = row.get("header") or "import Mathlib\n"

        if not original_theorem.strip().endswith("by"):
            print(f"SKIP (original does not end with := by): {lean_path}", file=sys.stderr)
            ec = 1
            return

        ts = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "ts": ts,
            "lean_path": str(lean_path),
            "split": split,
            "name": problem,
            "variant": variant,
        }
        if bridge_variant_theorem is not None:
            record["bridge_variant"] = variant[: -len("_renamed")] if variant.endswith("_renamed") else None

        try:
            eq_result = checker.check_equivalence(
                preamble=preamble_raw,
                original_theorem=original_theorem,
                variant_theorem=variant_theorem,
                project_root=root,
                bridge_variant_theorem=bridge_variant_theorem,
                timeout_seconds=args.timeout,
                default_preamble=preamble_raw,
                llm_attempts_per_failed_goal=args.llm_attempts,
                llm_batch_size=args.llm_attempts,
            )
        except Exception as ex:
            record["error"] = str(ex)
            record["certificate_status"] = None
            print(f"[{problem}/{variant}] check_equivalence raised: {ex}", file=sys.stderr)
            ec = 1
        else:
            cert_stat = eq_result.certificate_status
            record["certificate_status"] = cert_stat.value
            record["auxiliary_success"] = eq_result.success
            record["certificate_verified"] = eq_result.certificate_verified
            record["certificate_verify_error"] = eq_result.certificate_verify_error
            record["auxiliary"] = [
                {
                    "kind": r.kind.value,
                    "success": r.success,
                    "proof_source": r.proof_source,
                    "llm_fallback_attempted": r.llm_fallback_attempted,
                    "last_error": r.last_error,
                    "theorem_statement": r.theorem_statement,
                    "verified_proof_body": r.verified_proof_body,
                }
                for r in eq_result.auxiliary_results
            ]
            if args.write_certificates and eq_result.certificate_lean:
                out_cert = cert_root / rel.parent / f"{variant}.certificate.lean"
                out_cert.parent.mkdir(parents=True, exist_ok=True)
                out_cert.write_text(eq_result.certificate_lean, encoding="utf-8")
                record["certificate_path"] = str(out_cert)

            if cert_stat != EquivalenceCertificateStatus.ACCEPTED:
                unverified_text = _build_unverified_certificate_text(
                    preamble=preamble_raw,
                    original_theorem=original_theorem,
                    variant_theorem=variant_theorem,
                    bridge_variant_theorem=bridge_variant_theorem,
                    auxiliary=record["auxiliary"],
                    certificate_verify_error=eq_result.certificate_verify_error,
                )
                unverified_path = unverified_root / rel.parent / f"{variant}.certificate.unverified.txt"
                unverified_path.parent.mkdir(parents=True, exist_ok=True)
                unverified_path.write_text(unverified_text, encoding="utf-8")
                record["unverified_path"] = str(unverified_path)
                ec = 1
            print(
                f"[{problem}/{variant}] status={cert_stat.value} "
                f"aux_ok={eq_result.success} cert_verified={eq_result.certificate_verified}",
                file=sys.stderr,
            )

        logs_dir.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        processed.add(str(lean_path.resolve()))
        _save_processed(state_path, processed)

    while True:
        if not accepted_root.is_dir():
            print(
                f"Waiting for accepted directory: {accepted_root}",
                file=sys.stderr,
            )
        else:
            for lean_path in _list_variant_leans(accepted_root):
                key = str(lean_path.resolve())
                if key in processed:
                    continue
                process_one(lean_path)

        if args.once:
            break
        time.sleep(args.watch_seconds)

    return ec


if __name__ == "__main__":
    raise SystemExit(main())

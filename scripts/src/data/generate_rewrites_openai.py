#!/usr/bin/env python3
"""
Generate theorem rewrites from a prepared queue using OpenAI API.

Flow per problem:
1) Rewrite pass — GPT produces ``theorem <name>_v2``, ``_v3``, …
2) Renaming pass — a configurable fraction of those variants get a second GPT call
   (α-renames only); renamed rows are **appended** (base variants are kept).
3) JSONL rows use the **original** theorem ``name`` in ``formal_statement``; headers have
   ``import Aesop`` stripped.

Default model is gpt-5.4 (see .todo/RWData_TODO.md).

Safety defaults:
- Dry-run by default (no API calls).
- Explicit --execute required to call OpenAI.
- Per-call usage and estimated cost logging.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invpro.dataset.rewriting.jsonl_records import (
    empty_maps_to_none,
    make_rewrite_dataset_record,
    sanitize_header_remove_aesop,
    truncate_theorem_after_by,
)
from invpro.dataset.rewriting.parse_generated_variants import (
    extract_theorem_blocks,
    filter_variant_blocks,
    theorem_declared_name,
    variant_tag_from_declared_name,
)
from invpro.models.llm.openai_gpt import OpenAIGPTLLM, OpenAIUsage
from invpro.prompt.variant_renaming import (
    VariantRenamingPromptFormatter,
    parse_variant_renaming_response,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUEUE = PROJECT_ROOT / "data" / "rewritings_pipeline" / "queues" / "minif2f_test_queue_batch01.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "rewritings_pipeline"
DEFAULT_RUN_ID = "manual_v6_gpt54_valid_10"
GENERATED_SUBDIR = "generated"
CHEAP_TEST_MODEL = "gpt-5.4-nano"

# OpenAI list prices (USD per 1M tokens), short context — https://openai.com/api/pricing
OPENAI_MODEL_PRICES_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-nano": (0.20, 1.25),
}

THEOREM_NAME_RE = re.compile(r"^(theorem\s+)([^\s(]+)(.*)$")

# (split, problem_name) -> informal docstring; keyed by queue ``dataset_name`` (e.g. minif2f → data/minif2f.jsonl).
_INFORMAL_PREFIX_CACHE: dict[str, dict[tuple[str, str], str]] = {}


def _load_informal_prefix_map(dataset_name: str) -> dict[tuple[str, str], str]:
    if dataset_name in _INFORMAL_PREFIX_CACHE:
        return _INFORMAL_PREFIX_CACHE[dataset_name]
    path = PROJECT_ROOT / "data" / f"{dataset_name}.jsonl"
    m: dict[tuple[str, str], str] = {}
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                sp = str(o.get("split", ""))
                nm = str(o.get("name", ""))
                m[(sp, nm)] = str(o.get("informal_prefix", "") or "")
    _INFORMAL_PREFIX_CACHE[dataset_name] = m
    return m


def resolve_informal_prefix(dataset_name: str, split: str, problem_name: str, from_queue: str) -> str:
    """Use queue value when set; otherwise look up ``data/<dataset_name>.jsonl`` (miniF2F / ProofNet)."""
    if from_queue and str(from_queue).strip():
        return str(from_queue)
    return _load_informal_prefix_map(dataset_name).get((split, problem_name), "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue-jsonl", type=Path, default=DEFAULT_QUEUE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--run-id",
        type=str,
        default=DEFAULT_RUN_ID,
        help=f"Experiment folder under output-root; writes to {GENERATED_SUBDIR}/<dataset>/<split>/<problem>.",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="JSONL API log. Default: <output-root>/<run-id>/logs/openai_rewrite_calls.jsonl",
    )
    p.add_argument("--model", type=str, default="gpt-5.4")
    p.add_argument(
        "--use-cheap-test-model",
        action="store_true",
        help=f"Override --model with low-cost smoke-test model ({CHEAP_TEST_MODEL}).",
    )
    p.add_argument("--start-from", type=int, default=0)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-output-tokens", type=int, default=5000)
    p.add_argument(
        "--max-output-tokens-renaming",
        type=int,
        default=4000,
        help="Output token limit for the optional renaming pass (second GPT call per selected variant).",
    )
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=None,
        help=(
            "USD per 1M input tokens for cost estimates. "
            "If omitted together with --output-cost-per-1m, uses the table for "
            f"gpt-5.4 ({OPENAI_MODEL_PRICES_USD_PER_1M['gpt-5.4'][0]}) or "
            f"gpt-5.4-nano ({OPENAI_MODEL_PRICES_USD_PER_1M['gpt-5.4-nano'][0]}) depending on --model."
        ),
    )
    p.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=None,
        help=(
            "USD per 1M output tokens for cost estimates. "
            "If omitted together with --input-cost-per-1m, uses the table for the resolved model "
            f"(gpt-5.4 out={OPENAI_MODEL_PRICES_USD_PER_1M['gpt-5.4'][1]}, "
            f"gpt-5.4-nano out={OPENAI_MODEL_PRICES_USD_PER_1M['gpt-5.4-nano'][1]})."
        ),
    )
    p.add_argument(
        "--rename-fraction",
        type=float,
        default=0.2,
        help="Fraction of rewrite variants to send through the renaming GPT pass (0 disables).",
    )
    p.add_argument(
        "--rename-seed",
        type=int,
        default=0,
        help="Extra seed mixed into per-problem RNG for which variants get renaming.",
    )
    p.add_argument(
        "--no-renaming",
        action="store_true",
        help="Skip the second-pass renaming API calls (only rewrites are emitted).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Actually call OpenAI API.")
    mode.add_argument("--dry-run", action="store_true", help="Force no-API mode.")
    p.add_argument(
        "--aggregated-jsonl",
        type=Path,
        default=None,
        help=(
            "When using --execute: append one JSON record per variant (benchmark-like schema + "
            "dataset_name, variant, maps, original_formal_statement). "
            "Default: <output-root>/<run-id>/rewrites_dataset.jsonl"
        ),
    )
    return p.parse_args()


def load_queue(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Queue file not found: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"Queue file is empty: {path}")
    return rows


def rename_theorem_to_suffix(block: str, theorem_name: str) -> str:
    lines = block.splitlines()
    if not lines:
        return block
    m = THEOREM_NAME_RE.match(lines[0].strip())
    if not m:
        return block
    lines[0] = f"{m.group(1)}{theorem_name}{m.group(3)}"
    return "\n".join(lines).strip()


def theorem_name_for_renamed_file_block(problem_name: str, variant_tag: str) -> str:
    """Unique Lean name for a renamed variant in the generated review file."""
    if variant_tag.startswith("v") and variant_tag[1:].isdigit():
        return f"{problem_name}_v{variant_tag[1:]}_renamed"
    return f"{problem_name}_{variant_tag}_renamed"


def normalize_variants(raw_text: str, problem_name: str, variants_min: int, variants_max: int) -> list[str]:
    blocks = extract_theorem_blocks(raw_text)
    blocks = filter_variant_blocks(blocks, problem_name)
    selected = blocks[:variants_max]
    renamed: list[str] = []
    for idx, block in enumerate(selected, start=2):
        renamed.append(rename_theorem_to_suffix(block, f"{problem_name}_v{idx}"))
    if len(renamed) < variants_min:
        return renamed
    return renamed


def select_rename_indices(n: int, fraction: float, problem_name: str, run_id: str, extra_seed: int) -> list[int]:
    if n <= 0 or fraction <= 0:
        return []
    k = min(n, max(1, int(math.ceil(n * fraction))))
    h = zlib.adler32(f"{run_id}\0{problem_name}".encode("utf-8")) & 0xFFFFFFFF
    rng = random.Random(h ^ (extra_seed & 0xFFFFFFFF))
    indices = list(range(n))
    rng.shuffle(indices)
    return sorted(indices[:k])


def estimate_cost_usd(usage: OpenAIUsage, input_cost_per_1m: float, output_cost_per_1m: float) -> float:
    return (usage.input_tokens / 1_000_000.0) * input_cost_per_1m + (
        usage.output_tokens / 1_000_000.0
    ) * output_cost_per_1m


def resolve_pricing_defaults(args: argparse.Namespace) -> None:
    """
    If both cost flags were omitted, use OPENAI_MODEL_PRICES_USD_PER_1M for ``args.model``.
    If either was set, missing side defaults to 0.0.
    """
    if args.input_cost_per_1m is not None or args.output_cost_per_1m is not None:
        if args.input_cost_per_1m is None:
            args.input_cost_per_1m = 0.0
        if args.output_cost_per_1m is None:
            args.output_cost_per_1m = 0.0
        return
    inp, out = OPENAI_MODEL_PRICES_USD_PER_1M.get(args.model, (0.0, 0.0))
    args.input_cost_per_1m = inp
    args.output_cost_per_1m = out


def write_problem_file(path: Path, original_statement: str, variants: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [original_statement.strip()] + [v.strip() for v in variants]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks).rstrip() + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.use_cheap_test_model:
        args.model = CHEAP_TEST_MODEL
    resolve_pricing_defaults(args)
    do_execute = bool(args.execute and not args.dry_run)
    if args.log_file is None:
        args.log_file = args.output_root / args.run_id / "logs" / "openai_rewrite_calls.jsonl"

    aggregated_jsonl = args.aggregated_jsonl
    if aggregated_jsonl is None:
        aggregated_jsonl = args.output_root / args.run_id / "rewrites_dataset.jsonl"

    if args.start_from < 0:
        raise SystemExit("start-from must be >= 0")
    if args.limit <= 0:
        raise SystemExit("limit must be > 0")
    if not (0.0 <= args.rename_fraction <= 1.0):
        raise SystemExit("rename-fraction must be in [0, 1]")

    rows = load_queue(args.queue_jsonl)
    if args.start_from >= len(rows):
        raise SystemExit(f"start-from ({args.start_from}) >= queue length ({len(rows)})")
    rows = rows[args.start_from : args.start_from + args.limit]

    llm: OpenAIGPTLLM | None = None
    max_toks = max(int(args.max_output_tokens), int(args.max_output_tokens_renaming))
    if do_execute:
        try:
            llm = OpenAIGPTLLM.from_pretrained(
                model_name_or_path=args.model,
                temperature=args.temperature,
                max_output_tokens=max_toks,
                api_key_env=args.api_key_env,
                env_file_path=str(PROJECT_ROOT / ".env"),
            )
        except Exception as e:
            raise SystemExit(f"Failed to initialize OpenAIGPTLLM: {e}")

    renaming_formatter = VariantRenamingPromptFormatter()

    total_input_tokens = 0
    total_output_tokens = 0
    total_estimated_cost = 0.0

    print(f"Mode: {'EXECUTE' if do_execute else 'DRY-RUN'}")
    print(f"Queue file: {args.queue_jsonl}")
    print(f"Items selected: {len(rows)}")
    print(f"Model: {args.model}")
    print(
        f"Cost estimate ($/1M tok): input={args.input_cost_per_1m:g} output={args.output_cost_per_1m:g}"
    )
    print(f"Run id: {args.run_id}")
    print(f"Generated dir: {args.output_root / args.run_id / GENERATED_SUBDIR}")
    if do_execute:
        aggregated_jsonl.parent.mkdir(parents=True, exist_ok=True)
        aggregated_jsonl.write_text("", encoding="utf-8")
        print(f"Aggregated dataset JSONL (empty start): {aggregated_jsonl}")

    for i, row in enumerate(rows, start=1):
        problem_name = str(row["problem_name"])
        dataset_name = str(row.get("dataset_name", "dataset"))
        split = str(row.get("split", "split"))
        variants_min = int(row.get("variants_min", 5))
        variants_max = int(row.get("variants_max", 15))
        prompt = str(row.get("prompt", ""))
        original_statement = str(row.get("formal_statement", "")).strip()
        informal_prefix = resolve_informal_prefix(
            dataset_name,
            split,
            problem_name,
            str(row.get("informal_prefix", "")),
        )
        header = str(row.get("header", ""))
        goal = row.get("goal")
        goal_str = str(goal) if goal is not None else None

        output_file = (
            args.output_root / args.run_id / GENERATED_SUBDIR / dataset_name / split / problem_name
        )
        now = datetime.now(timezone.utc).isoformat()

        if do_execute:
            assert llm is not None
            response_text, usage = llm.generate_with_usage(prompt)
            variants = normalize_variants(
                raw_text=response_text,
                problem_name=problem_name,
                variants_min=variants_min,
                variants_max=variants_max,
            )
            est_cost = estimate_cost_usd(usage, args.input_cost_per_1m, args.output_cost_per_1m)
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_estimated_cost += est_cost

            san_header = sanitize_header_remove_aesop(header)

            jsonl_records: list[dict[str, Any]] = []
            file_variant_blocks: list[str] = []

            # Benchmark row (seed theorem) as its own JSONL line so the file contains originals, not only augments.
            jsonl_records.append(
                make_rewrite_dataset_record(
                    name=problem_name,
                    original_name=problem_name,
                    split=split,
                    informal_prefix=informal_prefix,
                    formal_statement=truncate_theorem_after_by(original_statement),
                    header=san_header,
                    variant="original",
                    dataset_name=dataset_name,
                    original_formal_statement=original_statement,
                    variable_map=None,
                    hypothesis_map=None,
                    goal=goal_str,
                    certificate=None,
                )
            )

            base_meta: list[tuple[str, str]] = []
            for block in variants:
                block = truncate_theorem_after_by(block)
                decl = theorem_declared_name(block)
                vtag = variant_tag_from_declared_name(decl or "", problem_name)
                augmented_name = decl if decl else f"{problem_name}_v2"
                jsonl_records.append(
                    make_rewrite_dataset_record(
                        name=augmented_name,
                        original_name=problem_name,
                        split=split,
                        informal_prefix=informal_prefix,
                        formal_statement=block,
                        header=san_header,
                        variant=vtag,
                        dataset_name=dataset_name,
                        original_formal_statement=original_statement,
                        variable_map=None,
                        hypothesis_map=None,
                        goal=goal_str,
                        certificate=None,
                    )
                )
                file_variant_blocks.append(block)
                base_meta.append((vtag, block))

            rename_fraction = 0.0 if args.no_renaming else float(args.rename_fraction)
            rename_ix = select_rename_indices(
                len(base_meta),
                rename_fraction,
                problem_name,
                args.run_id,
                args.rename_seed,
            )

            if llm is not None and rename_ix and not args.no_renaming:
                for ri in rename_ix:
                    vtag, block_for_rename = base_meta[ri]
                    decl_rename = theorem_declared_name(block_for_rename) or ""
                    r_prompt = renaming_formatter.format(
                        formal_statement=block_for_rename,
                        theorem_name=decl_rename or problem_name,
                    )
                    response_r, usage_r = llm.generate_with_usage(r_prompt)
                    total_input_tokens += usage_r.input_tokens
                    total_output_tokens += usage_r.output_tokens
                    total_estimated_cost += estimate_cost_usd(
                        usage_r, args.input_cost_per_1m, args.output_cost_per_1m
                    )
                    log_rename = {
                        "timestamp_utc": now,
                        "kind": "variant_renaming",
                        "mode": "execute",
                        "run_id": args.run_id,
                        "queue_idx": row.get("queue_idx"),
                        "problem_name": problem_name,
                        "variant": vtag,
                        "model": args.model,
                        "usage_input_tokens": usage_r.input_tokens,
                        "usage_output_tokens": usage_r.output_tokens,
                        "usage_total_tokens": usage_r.total_tokens,
                        "estimated_cost_usd": round(
                            estimate_cost_usd(usage_r, args.input_cost_per_1m, args.output_cost_per_1m),
                            8,
                        ),
                        "raw_response_preview": response_r[:500],
                    }
                    append_jsonl(args.log_file, log_rename)
                    try:
                        parsed = parse_variant_renaming_response(response_r)
                    except Exception as ex:
                        err_log = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "kind": "variant_renaming_error",
                            "problem_name": problem_name,
                            "variant": vtag,
                            "error": str(ex),
                        }
                        append_jsonl(args.log_file, err_log)
                        print(f"  renaming {vtag}: PARSE ERROR {ex}")
                        continue

                    new_variant = f"{vtag}_renamed"
                    renamed_stmt = truncate_theorem_after_by(parsed["renamed"])
                    file_name = theorem_name_for_renamed_file_block(problem_name, vtag)
                    formal_renamed = rename_theorem_to_suffix(renamed_stmt, file_name)
                    jsonl_records.append(
                        make_rewrite_dataset_record(
                            name=file_name,
                            original_name=problem_name,
                            split=split,
                            informal_prefix=informal_prefix,
                            formal_statement=formal_renamed,
                            header=san_header,
                            variant=new_variant,
                            dataset_name=dataset_name,
                            original_formal_statement=original_statement,
                            variable_map=empty_maps_to_none(parsed["variable_map"]),
                            hypothesis_map=empty_maps_to_none(parsed["hypothesis_map"]),
                            goal=goal_str,
                            certificate=None,
                        )
                    )
                    file_variant_blocks.append(formal_renamed)

            for rec in jsonl_records:
                append_jsonl(aggregated_jsonl, rec)

        else:
            response_text = ""
            usage = OpenAIUsage(input_tokens=0, output_tokens=0, total_tokens=0)
            variants = []
            est_cost = 0.0
            file_variant_blocks = []

        if not do_execute:
            write_problem_file(output_file, original_statement=original_statement, variants=variants)
        else:
            write_problem_file(output_file, original_statement=original_statement, variants=file_variant_blocks)

        log_record = {
            "timestamp_utc": now,
            "mode": "execute" if do_execute else "dry-run",
            "run_id": args.run_id,
            "queue_idx": row.get("queue_idx"),
            "problem_name": problem_name,
            "dataset_name": dataset_name,
            "split": split,
            "model": args.model,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "variants_min": variants_min,
            "variants_max": variants_max,
            "variants_generated": len(variants) if do_execute else 0,
            "output_file": str(output_file),
            "usage_input_tokens": usage.input_tokens,
            "usage_output_tokens": usage.output_tokens,
            "usage_total_tokens": usage.total_tokens,
            "estimated_cost_usd": round(est_cost, 8),
            "raw_response_preview": response_text[:500],
        }
        append_jsonl(args.log_file, log_record)

        n_out = len(variants) if do_execute else 0
        print(
            f"[{i}/{len(rows)}] {problem_name}: generated={n_out} "
            f"input_toks={usage.input_tokens} output_toks={usage.output_tokens} "
            f"est_cost=${est_cost:.6f}"
        )

    print("-----")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Estimated total cost (USD): ${total_estimated_cost:.6f}")
    print(f"Log file: {args.log_file}")
    if do_execute:
        print(f"Aggregated dataset JSONL: {aggregated_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

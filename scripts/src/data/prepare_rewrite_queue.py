#!/usr/bin/env python3
"""
Prepare a rewrite-generation task queue from a benchmark JSONL dataset.
Queue files default to ``data/rewritings_pipeline/queues/*.jsonl`` (next to generated runs).

This module does not call any API. It creates a JSONL queue with one entry
per selected theorem and (optionally) a prompt text file per theorem.
Default use is aligned with the rewrite MVP:
- start with miniF2F test split
- run in small batches
- skip problems that already have rewrite files

Each queue line is one JSON object. Important fields:

- ``dataset_name``, ``split``, ``problem_name``, ``queue_idx``
- ``formal_statement``, ``header`` — from the benchmark row
- ``informal_prefix`` — copied from the benchmark ``informal_prefix`` (Lean ``/-- ... -/`` doc text)
- ``variants_min``, ``variants_max``
- ``prompt`` — text passed to the rewrite model (includes informal context when present)
- ``goal`` — optional, when the benchmark row has a ``goal`` field
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from invpro.prompt.conf import dict_to_config as prompt_dict_to_config
from invpro.prompt.conf import prompt_formatter_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MINIF2F_DATASET = PROJECT_ROOT / "data" / "minif2f.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rewritings_pipeline" / "queues" / "minif2f_test_queue.jsonl"
DEFAULT_EXISTING_REWRITES = PROJECT_ROOT / "data" / "rewritings" / "minif2f" / "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_MINIF2F_DATASET)
    parser.add_argument("--dataset-name", type=str, default="minif2f")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--existing-rewrites-dir",
        type=Path,
        default=DEFAULT_EXISTING_REWRITES,
        help="Skip problems with an existing rewrite file at <dir>/<problem_name>",
    )
    parser.add_argument("--start-from", type=int, default=0, help="0-index in filtered split")
    parser.add_argument("--limit", type=int, default=20, help="Max number of queue entries")
    parser.add_argument("--seed", type=int, default=0, help="Used only with --shuffle")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--variants-min",
        type=int,
        default=5,
        help="Target lower bound for generated equivalent versions",
    )
    parser.add_argument(
        "--variants-max",
        type=int,
        default=15,
        help="Target upper bound for generated equivalent versions",
    )
    parser.add_argument(
        "--prompt-template-path",
        type=Path,
        default=None,
        help="Optional legacy text template with '{formal_statement}' placeholder",
    )
    parser.add_argument(
        "--prompt-formatter-config",
        type=Path,
        default=None,
        help="Optional YAML for prompt formatter config. Defaults to RewriteAugmentationPromptFormatter.",
    )
    parser.add_argument(
        "--write-prompts-dir",
        type=Path,
        default=None,
        help="If set, writes one prompt txt per queued problem",
    )
    return parser.parse_args()


def load_filtered_dataset(dataset_path: Path, split: str) -> list[dict[str, Any]]:
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    rows: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("split") != split:
                continue
            if not isinstance(obj.get("name"), str):
                continue
            if not isinstance(obj.get("formal_statement"), str):
                continue
            rows.append(obj)
    return rows


def load_prompt_template(path: Path | None) -> str:
    if not path.is_file():
        raise SystemExit(f"Prompt template file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_prompt_formatter(config_path: Path | None):
    if config_path is None:
        cfg_dict = {"class": "RewriteAugmentationPromptFormatter", "parameters": {}}
    else:
        if not config_path.is_file():
            raise SystemExit(f"Prompt formatter config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}
    return prompt_formatter_from_config(prompt_dict_to_config(cfg_dict))


def main() -> int:
    args = parse_args()
    if args.variants_min < 1 or args.variants_max < args.variants_min:
        raise SystemExit("Require 1 <= variants_min <= variants_max.")
    if args.start_from < 0:
        raise SystemExit("start-from must be >= 0.")
    if args.limit <= 0:
        raise SystemExit("limit must be > 0.")
    rows = load_filtered_dataset(args.dataset_path, args.split)
    if not rows:
        raise SystemExit(f"No rows found for split '{args.split}' in {args.dataset_path}")
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.start_from >= len(rows):
        raise SystemExit(f"start-from ({args.start_from}) >= available rows ({len(rows)})")
    rows = rows[args.start_from :]
    existing_dir = args.existing_rewrites_dir
    selected: list[dict[str, Any]] = []
    for row in rows:
        problem_name = str(row["name"])
        existing_file = existing_dir / problem_name
        if existing_file.exists():
            continue
        selected.append(row)
        if len(selected) >= args.limit:
            break
    if not selected:
        raise SystemExit("No new problems selected after filtering existing rewrites.")

    prompt_formatter = load_prompt_formatter(args.prompt_formatter_config)
    template = load_prompt_template(args.prompt_template_path) if args.prompt_template_path is not None else None

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.write_prompts_dir
    if prompts_dir is not None:
        prompts_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for idx, row in enumerate(selected):
            problem_name = str(row["name"])
            formal_statement = str(row["formal_statement"]).rstrip() + "\n"
            header = str(row.get("header", ""))
            informal_prefix = str(row.get("informal_prefix") or "")
            if template is not None:
                fmt_kwargs: dict[str, Any] = {
                    "formal_statement": formal_statement,
                    "variants_min": args.variants_min,
                    "variants_max": args.variants_max,
                }
                if "{informal_prefix}" in template:
                    fmt_kwargs["informal_prefix"] = informal_prefix
                prompt = template.format(**fmt_kwargs)
            else:
                prompt = prompt_formatter.format(
                    formal_statement=formal_statement,
                    theorem_name=problem_name,
                    variants_min=args.variants_min,
                    variants_max=args.variants_max,
                    informal_prefix=informal_prefix,
                )
            entry = {
                "queue_idx": idx,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "problem_name": problem_name,
                "variants_min": args.variants_min,
                "variants_max": args.variants_max,
                "formal_statement": formal_statement,
                "informal_prefix": informal_prefix,
                "header": header,
                "prompt": prompt,
            }
            if isinstance(row.get("goal"), str):
                entry["goal"] = row["goal"]
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if prompts_dir is not None:
                (prompts_dir / f"{problem_name}.txt").write_text(prompt, encoding="utf-8")
    print(f"Wrote queue: {args.output_jsonl}")
    print(f"Selected problems: {len(selected)}")
    print(f"Dataset/split: {args.dataset_name}/{args.split}")
    if prompts_dir is not None:
        print(f"Wrote prompts in: {prompts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Second-stage GPT pass: fill ``variable_map`` / ``hypothesis_map`` and optionally replace
``formal_statement`` using :class:`VariantRenamingPromptFormatter`.

Reads a JSONL produced by ``generate_openai`` (``rewrites_dataset.jsonl``), calls OpenAI
per row, writes an updated JSONL. Maps empty ``{}`` from the model are stored as ``null``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rwens.dataset.rewriting.jsonl_records import empty_maps_to_none, normalize_formal_statement
from rwens.models.llm.openai_gpt import OpenAIGPTLLM
from rwens.prompt.variant_renaming import (
    VariantRenamingPromptFormatter,
    parse_variant_renaming_response,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--model", type=str, default="gpt-5.4")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-output-tokens", type=int, default=4000)
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument(
        "--skip-renamed-rows",
        action="store_true",
        help="Skip rows that already have a ``renaming_timestamp_utc`` field (from a prior run).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    do_execute = bool(args.execute and not args.dry_run)
    fmt = VariantRenamingPromptFormatter()

    rows = load_jsonl(args.input_jsonl)
    if not rows:
        raise SystemExit(f"No rows in {args.input_jsonl}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("", encoding="utf-8")

    llm: OpenAIGPTLLM | None = None
    if do_execute:
        llm = OpenAIGPTLLM.from_pretrained(
            model_name_or_path=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            api_key_env=args.api_key_env,
            env_file_path=str(PROJECT_ROOT / ".env"),
        )

    for i, row in enumerate(rows, start=1):
        name = str(row.get("name", ""))
        if args.skip_renamed_rows and row.get("renaming_timestamp_utc"):
            append_jsonl(args.output_jsonl, row)
            print(f"[{i}/{len(rows)}] {name}: skip (already renamed)")
            continue

        formal = str(row.get("formal_statement", ""))
        prompt = fmt.format(formal_statement=formal, theorem_name=name)

        if not do_execute:
            out = dict(row)
            out["_renaming_prompt_preview"] = prompt[:400]
            append_jsonl(args.output_jsonl, out)
            print(f"[{i}/{len(rows)}] {name}: dry-run")
            continue

        assert llm is not None
        response_text, usage = llm.generate_with_usage(prompt)
        try:
            parsed = parse_variant_renaming_response(response_text)
        except Exception as ex:
            out = dict(row)
            out["renaming_error"] = str(ex)
            out["renaming_raw_preview"] = response_text[:800]
            append_jsonl(args.output_jsonl, out)
            print(f"[{i}/{len(rows)}] {name}: PARSE ERROR {ex}")
            continue

        out = dict(row)
        out["formal_statement"] = normalize_formal_statement(parsed["renamed"])
        out["variable_map"] = empty_maps_to_none(parsed["variable_map"])
        out["hypothesis_map"] = empty_maps_to_none(parsed["hypothesis_map"])
        out["name"] = name
        out["renaming_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        out["renaming_usage_input_tokens"] = usage.input_tokens
        out["renaming_usage_output_tokens"] = usage.output_tokens
        append_jsonl(args.output_jsonl, out)
        print(f"[{i}/{len(rows)}] {name}: ok in={usage.input_tokens} out={usage.output_tokens}")

    print(f"Wrote: {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

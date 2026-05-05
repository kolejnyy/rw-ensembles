"""
Rebuild selected rows in ``rewrites_dataset.jsonl`` from on-disk ``generated/`` theorem files.

For each problem name, reads ``<generated-root>/<dataset>/<split>/<problem>`` (same layout as
``generate_openai``), extracts theorem blocks, and emits records compatible with
:func:`make_rewrite_dataset_record` / the OpenAI generator. Replaces the contiguous block of
lines for that ``original_name`` while preserving global JSONL order.

Metadata ``header``, ``informal_prefix``, and ``goal`` come from ``data/<dataset_name>.jsonl``
via ``(split, name)``. The seed statement ``original_formal_statement`` (and the ``original``
row's ``formal_statement``) is taken from the **first** ``theorem <problem_name>`` block in the
generated file, not from the benchmark JSONL. For ``*_renamed`` variants,
``variable_map`` / ``hypothesis_map`` are copied from the previous JSONL row when the variant
string matches (Lean alone does not encode those maps).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from invpro.dataset.rewriting.jsonl_records import (
    empty_maps_to_none,
    make_rewrite_dataset_record,
    truncate_theorem_after_by,
)
from invpro.dataset.rewriting.parse_generated_variants import (
    extract_theorem_blocks,
    theorem_declared_name,
    variant_tag_from_declared_name,
)


def load_benchmark_index(dataset_jsonl: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with dataset_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[(str(row["split"]), str(row["name"]))] = row
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_generated_file(generated_root: Path, problem_name: str) -> Path:
    hits = [p for p in generated_root.rglob(problem_name) if p.is_file() and p.name == problem_name]
    if not hits:
        raise FileNotFoundError(f"No generated file named {problem_name!r} under {generated_root}")
    if len(hits) > 1:
        raise ValueError(f"Multiple generated files named {problem_name!r}: {hits[:5]}…")
    return hits[0]


def _maps_from_old(
    old_by_variant: dict[str, dict[str, Any]], variant: str
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    old = old_by_variant.get(variant)
    if not old:
        return None, None
    vm = old.get("variable_map")
    hm = old.get("hypothesis_map")
    if isinstance(vm, dict):
        vm = empty_maps_to_none(vm)
    else:
        vm = None
    if isinstance(hm, dict):
        hm = empty_maps_to_none(hm)
    else:
        hm = None
    return vm, hm


def build_records_for_problem(
    *,
    problem_name: str,
    generated_file: Path,
    benchmark_row: dict[str, Any],
    dataset_name: str,
    split: str,
    old_by_variant: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    text = generated_file.read_text(encoding="utf-8")
    blocks = extract_theorem_blocks(text)
    original_block: str | None = None
    variant_blocks: list[str] = []
    for block in blocks:
        tname = theorem_declared_name(block)
        if tname is None:
            continue
        if tname == problem_name:
            if original_block is None:
                original_block = block
            continue
        variant_blocks.append(block)

    if original_block is None:
        raise ValueError(f"{generated_file}: no theorem block named {problem_name!r}")

    # Seed text for every row: normalized first theorem in the generated file (not data/*.jsonl).
    seed_statement = truncate_theorem_after_by(original_block)
    informal_prefix = str(benchmark_row.get("informal_prefix", "") or "")
    header = str(benchmark_row.get("header", "") or "")
    goal = benchmark_row.get("goal")
    goal_str = str(goal) if goal is not None else None

    out: list[dict[str, Any]] = []
    out.append(
        make_rewrite_dataset_record(
            name=problem_name,
            original_name=problem_name,
            split=split,
            informal_prefix=informal_prefix,
            formal_statement=seed_statement,
            header=header,
            variant="original",
            dataset_name=dataset_name,
            original_formal_statement=seed_statement,
            variable_map=None,
            hypothesis_map=None,
            goal=goal_str,
            certificate=None,
        )
    )

    for block in variant_blocks:
        block = truncate_theorem_after_by(block)
        decl = theorem_declared_name(block) or ""
        vtag = variant_tag_from_declared_name(decl, problem_name)
        augmented_name = decl if decl else f"{problem_name}_v2"
        vm, hm = _maps_from_old(old_by_variant, vtag)
        out.append(
            make_rewrite_dataset_record(
                name=augmented_name,
                original_name=problem_name,
                split=split,
                informal_prefix=informal_prefix,
                formal_statement=block,
                header=header,
                variant=vtag,
                dataset_name=dataset_name,
                original_formal_statement=seed_statement,
                variable_map=vm,
                hypothesis_map=hm,
                goal=goal_str,
                certificate=None,
            )
        )
    return out


def index_old_rows_by_variant(rows: list[dict[str, Any]], problem_name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if str(r.get("original_name", "")) != problem_name:
            continue
        v = str(r.get("variant", ""))
        if v:
            out[v] = r
    return out


def merge_rewrites_jsonl(
    *,
    rows: list[dict[str, Any]],
    problems: set[str],
    generated_root: Path,
    benchmark_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Return new row list and log messages. Each problem in ``problems`` must appear contiguously
    in ``rows`` (as in generator output).
    """
    logs: list[str] = []
    emitted: set[str] = set()
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(rows):
        r = rows[i]
        on = str(r.get("original_name") or r.get("name") or "")
        if on in problems:
            if on not in emitted:
                gen_path = find_generated_file(generated_root, on)
                rel = gen_path.relative_to(generated_root)
                if len(rel.parts) < 3:
                    raise ValueError(f"Unexpected generated path shape: {gen_path}")
                ds = str(rel.parts[0])
                sp = str(rel.parts[1])
                key = (sp, on)
                bench = benchmark_index.get(key)
                if bench is None:
                    raise KeyError(f"No benchmark row for split={sp!r} name={on!r} in dataset JSONL")
                old_by = index_old_rows_by_variant(rows, on)
                new_block = build_records_for_problem(
                    problem_name=on,
                    generated_file=gen_path,
                    benchmark_row=bench,
                    dataset_name=ds,
                    split=sp,
                    old_by_variant=old_by,
                )
                out.extend(new_block)
                emitted.add(on)
                logs.append(
                    f"Replaced {sum(1 for x in rows if str(x.get('original_name')) == on)} "
                    f"-> {len(new_block)} rows for {on} ({gen_path})"
                )
            i += 1
        else:
            out.append(r)
            i += 1

    missing = problems - emitted
    if missing:
        raise ValueError(f"No rows found in JSONL for problem(s): {sorted(missing)}")
    return out, logs



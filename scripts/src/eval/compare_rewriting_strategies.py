#!/usr/bin/env python3
"""
Compare different rewriting strategies on minif2f (or other) problems.

Loads a config that lists named strategies (each a RewritingCanonicalization
parameters block), builds each module (with optional LLM and energy injection),
then for each problem runs each strategy and records the rewritten state and
tactics. Uses get_augmentation when the module supports it so behaviour matches
CanonicalLLMProver exactly; otherwise falls back to get_states + state-to-statement
conversion. Outputs a comparison table or JSON.

Usage:
  python scripts/compare_rewriting_strategies.py --config configs/compare_rewriting_strategies.yaml
  python scripts/compare_rewriting_strategies.py --config configs/compare_rewriting_strategies.yaml --dataset data/minif2f.jsonl --split valid --limit 3 --output results/rewriting_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from invpro.canonicalization.conf import dict_to_config as canonicalization_dict_to_config
from invpro.canonicalization.conf import canonicalization_from_config
from invpro.canonicalization.rewrites import make_theorem_surprise_energy
from invpro.canonicalization.rewrites import get_states_cache_key
from invpro.canonicalization.utils import ensure_rewrites_import
from invpro.dataset.utils import split_declarations_theorem_proof
from invpro.utils.cache_paths import get_rw_cache_dir
from invpro.models.llm.conf import dict_to_config as llm_dict_to_config
from invpro.models.llm.conf import llm_from_config
from invpro.prompt.conf import dict_to_config as prompt_dict_to_config
from invpro.prompt.conf import prompt_formatter_from_config
from invpro.utils.state_to_statement import StateProblemConverter, extract_theorem_name


def _state_to_theorem_statement(decls: str, state_str: str | None, theorem_name: str | None) -> str | None:
    """Convert goal state to full augmented theorem statement (imports + theorem line). Returns None on failure."""
    if not state_str or not state_str.strip():
        return None
    try:
        return StateProblemConverter.state_to_problem_pure(decls, state_str, theorem_name or "anon")
    except (ValueError, Exception):
        return None


def load_problems(dataset_path: Path, split: str, limit: int | None):
    """Load problems from JSONL by split."""
    problems = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("split") != split:
                continue
            problems.append(obj)
            if limit is not None and len(problems) >= limit:
                break
    return problems


def inject_energy_if_needed(module, llm, prompt_formatter, project_root):
    """If module has _inject_energy_type (e.g. theorem_surprise), inject the energy and set _model_cache_id."""
    inject_type = getattr(module, "_inject_energy_type", None)
    if not inject_type:
        if llm:
            module._model_cache_id = getattr(llm, "get_model_cache_id", lambda: None)() or ""
        return
    if inject_type == "theorem_surprise":
        if not prompt_formatter:
            raise ValueError("theorem_surprise energy requires prompt_formatter in config")
        energy = make_theorem_surprise_energy(
            project_root=str(project_root),
            prompt_formatter=prompt_formatter,
            verbose=False,
            use_cache=True,
            llm=llm,
        )
        module._energy_heuristic = energy
    module._model_cache_id = getattr(llm, "get_model_cache_id", lambda: None)() or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare rewriting strategies on problems")
    parser.add_argument("--config", type=Path, default=Path("configs/compare_rewriting_strategies.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/minif2f.jsonl"))
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--limit", type=int, default=None, help="Max problems to run")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON results here")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM load (only works if all strategies have energy: none)")
    parser.add_argument("--no-cache", action="store_true", help="Disable reading get_augmentation/get_states cache (always run rewriting)")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1
    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    strategies_config = config.get("strategies", [])
    if not strategies_config:
        print("No strategies in config.", file=sys.stderr)
        return 1

    # Build LLM and prompt formatter (required for RewritingCanonicalization and for theorem_surprise)
    llm = None
    prompt_formatter = None
    if not args.no_llm:
        has_rewriting = any(
            (s.get("canonicalization") or s.get("rewriting") or {}).get("class") == "RewritingCanonicalization"
            for s in strategies_config
        )
        if has_rewriting and "llm" not in config:
            print("Config has no 'llm'; required for RewritingCanonicalization strategies.", file=sys.stderr)
            return 1
        llm_config = llm_dict_to_config(config["llm"])
        llm = llm_from_config(llm_config)
        if "prompt_formatter" in config:
            pf_config = prompt_dict_to_config(config["prompt_formatter"])
            prompt_formatter = prompt_formatter_from_config(pf_config)

    # Build each strategy module (canonicalization or rewriting key)
    modules = []
    for s in strategies_config:
        name = s.get("name", "unnamed")
        canon_dict = s.get("canonicalization") or s.get("rewriting")
        if not canon_dict:
            print(f"Strategy {name!r} has no 'canonicalization' or 'rewriting'.", file=sys.stderr)
            return 1
        canon_config = canonicalization_dict_to_config(canon_dict)
        # Only RewritingCanonicalization needs LLM
        need_llm = canon_config.__class__.__name__ == "RewritingCanonicalizationConfig"
        module = canonicalization_from_config(canon_config, llm=llm if need_llm else None)
        if need_llm and llm:
            inject_energy_if_needed(module, llm, prompt_formatter, module.project_root)
        modules.append((name, module))

    problems = load_problems(args.dataset, args.split, args.limit)
    if not problems:
        print("No problems found.", file=sys.stderr)
        return 1

    print(f"Comparing {len(modules)} strategies on {len(problems)} problem(s).")

    results = []
    for prob in problems:
        name = prob.get("name", "?")
        header = prob.get("header", "").replace("import Aesop\n", "")
        formal = prob.get("formal_statement", "")
        problem_stmt = header + formal
        try:
            decls, theorem_stmt, _ = split_declarations_theorem_proof(problem_stmt)
        except ValueError as e:
            print(f"  Skip {name}: {e}")
            continue

        theorem_name = extract_theorem_name(theorem_stmt)
        row = {"problem": name, "strategies": {}}
        for strategy_name, module in modules:
            t0 = time.perf_counter()
            from_cache = False
            cache_id = None
            try:
                current, best_state, rw_tactics = None, None, []
                use_get_augmentation = hasattr(module, "get_augmentation")

                if use_get_augmentation:
                    # Match CanonicalLLMProver: use get_augmentation (statement domain).
                    # Cache is inside get_states() called by get_augmentation; report hit if gs_path exists.
                    if not args.no_cache and hasattr(module, "_depth") and hasattr(module, "_rwcache_dir"):
                        content = ensure_rewrites_import(decls).rstrip("\n") + "\n" + theorem_stmt.strip("\n")
                        model_cache_id = getattr(module, "_model_cache_id", "") or ""
                        gs_key = get_states_cache_key(
                            content,
                            module._depth,
                            module._max_per_step,
                            module._only_simplifying_rewrites,
                            getattr(module, "_reverse_order", False),
                            module._top_rewrites,
                            getattr(module, "_filter_rewrite_namespaces", None),
                            model_cache_id=model_cache_id,
                            cache_config_tag=getattr(module, "_cache_config_tag", ""),
                        )
                        cache_dir = get_rw_cache_dir(module.project_root)
                        gs_path = cache_dir / f"{gs_key}.cache"
                        from_cache = gs_path.exists()
                        cache_id = gs_key
                    augmented_problem, rw_tactics, cache_id = module.get_augmentation(
                        decls, theorem_stmt, theorem_name
                    )
                    augmented_theorem = augmented_problem
                    current = None
                    best_state = None
                    if augmented_problem:
                        try:
                            _, th_line, _ = split_declarations_theorem_proof(augmented_problem)
                            best_state = th_line.strip()
                        except ValueError:
                            best_state = augmented_problem
                else:
                    # Fallback: get_states + state-to-statement (e.g. non-rewriting modules).
                    if not args.no_cache and hasattr(module, "_depth") and hasattr(module, "_rwcache_dir"):
                        content = ensure_rewrites_import(decls).rstrip("\n") + "\n" + theorem_stmt.strip("\n")
                        model_cache_id = getattr(module, "_model_cache_id", "") or ""
                        gs_key = get_states_cache_key(
                            content,
                            module._depth,
                            module._max_per_step,
                            module._only_simplifying_rewrites,
                            getattr(module, "_reverse_order", False),
                            module._top_rewrites,
                            getattr(module, "_filter_rewrite_namespaces", None),
                            model_cache_id=model_cache_id,
                            cache_config_tag=getattr(module, "_cache_config_tag", ""),
                        )
                        cache_dir = get_rw_cache_dir(module.project_root)
                        gs_path = cache_dir / f"{gs_key}.cache"
                        if gs_path.exists():
                            try:
                                data = json.loads(gs_path.read_text(encoding="utf-8"))
                                current = data.get("current")
                                best_state = data.get("best_state")
                                rw_tactics = data.get("rw_tactics", [])
                                if best_state is not None:
                                    from_cache = True
                                    cache_id = gs_key
                            except (json.JSONDecodeError, OSError, KeyError):
                                pass
                    if not from_cache:
                        module.reset(decls, theorem_stmt)
                        out = module.get_states(keep_augmentation=False)
                        if len(out) >= 4:
                            current, best_state, rw_tactics, _ = out[:4]
                        elif len(out) == 3:
                            current, best_state, rw_tactics = out
                        else:
                            current, best_state = out[:2]
                            rw_tactics = []
                    augmented_theorem = _state_to_theorem_statement(decls, best_state, theorem_name)

                elapsed = time.perf_counter() - t0
                row["strategies"][strategy_name] = {
                    "current_len": len(current) if current else 0,
                    "best_state_len": len(best_state) if best_state else 0,
                    "num_rw_tactics": len(rw_tactics) if rw_tactics else 0,
                    "elapsed_sec": round(elapsed, 2),
                    "from_cache": from_cache,
                    "cache_id": cache_id,
                    "best_state": best_state[:500] + "..." if best_state and len(best_state) > 500 else best_state,
                    "augmented_theorem_statement": augmented_theorem,
                }
            except Exception as e:
                row["strategies"][strategy_name] = {"error": str(e), "elapsed_sec": round(time.perf_counter() - t0, 2), "from_cache": False, "cache_id": None}
        results.append(row)

        # Print one-line summary per problem
        parts = [f"  {name}:"]
        for strategy_name, data in row["strategies"].items():
            if "error" in data:
                parts.append(f" {strategy_name}=err")
            else:
                if data.get("from_cache") and data.get("cache_id"):
                    cache_mark = f" [cached: {data['cache_id']}]"
                else:
                    cache_mark = " [cached]" if data.get("from_cache") else ""
                parts.append(f" {strategy_name}=len={data['best_state_len']} rw={data['num_rw_tactics']} {data['elapsed_sec']}s{cache_mark}")
        print("".join(parts))

        # Print original problem (theorem statement) for reference
        print("  Original (theorem statement):")
        orig_snippet = theorem_stmt.strip() if len(theorem_stmt) < 300 else theorem_stmt.strip()[:297] + "..."
        for line in orig_snippet.split("\n"):
            print(f"      {line}")

        # Print augmented theorem statement for each strategy (each on new lines for readability)
        print("  Augmented theorem statements:")
        for strategy_name, data in row["strategies"].items():
            elapsed = data.get("elapsed_sec", 0)
            time_str = f" ({elapsed}s)" if isinstance(elapsed, (int, float)) else ""
            if data.get("from_cache"):
                cid = data.get("cache_id")
                time_str += f" [cached: {cid}]" if cid else " [cached]"
            if "error" in data:
                print(f"    [{strategy_name}]{time_str}")
                print(f"      error: {data['error']}")
            else:
                aug = data.get("augmented_theorem_statement")
                if aug:
                    try:
                        _, th_line, _ = split_declarations_theorem_proof(aug)
                        snippet = th_line.strip() if len(th_line) < 300 else th_line.strip()[:297] + "..."
                    except ValueError:
                        snippet = aug[:400] + "..." if len(aug) > 400 else aug
                    print(f"    [{strategy_name}]{time_str}")
                    for line in snippet.split("\n"):
                        print(f"      {line}")
                else:
                    print(f"    [{strategy_name}]{time_str}")
                    print("      (no augmented statement)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.output}")

    # Close any StateProblemConverter we attached
    for _name, module in modules:
        conv = getattr(module, "_converter", None)
        if conv is not None and hasattr(conv, "close"):
            conv.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

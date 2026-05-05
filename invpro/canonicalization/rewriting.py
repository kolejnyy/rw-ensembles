"""
Rewriting canonicalization module: three-stage retrieval (sampling → reranking → energy).

Stage 1 (sampling): rwcnc at each hypothesis, parametrized by branching factor and depth.
Stage 2 (reranking): heuristic to narrow candidates to ~10–30 (e.g. shortest states).
Stage 3 (energy): model-dependent scoring (e.g. confidence in first-tactic prediction).

Reranking and energy are pluggable via callables; use .factory(config, llm) to build from config.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from invpro.canonicalization.base import SimpleCanonicalizationModule
from invpro.canonicalization.rewrites import (
    GetStatesCacheEntry,
    StateCandidate,
    cache_config_tag_from_energy,
    filter_rewrites_by_namespace,
    filter_rewrites_by_namespace_blacklist,
    get_energy_heuristic,
    get_reranking_heuristic,
    get_states_cache_key,
    load_rewrites_cache,
    save_get_states_entry,
    save_rewrites_cache,
)
from invpro.canonicalization.rewrites.cache import RewriteEntry

# Type for the rewrites dict used in the pipeline (hyp name -> list of rewrite entries).
RewritesMap = dict[str, list[RewriteEntry]]
from invpro.utils.cache_paths import get_rw_cache_dir
from invpro.canonicalization.utils import (
    GOAL_REWRITE_KEY,
    build_fully_augmented_goal_from_best_types,
    build_states_for_goal,
    build_states_for_hypothesis,
    collect_try_this_from_diagnostics,
    deduplicate_by_state,
    ensure_rewrites_import,
    get_goal_expression,
    get_hypothesis_type,
    hyps_from_goal,
)
from invpro.models.llm.base import BaseLLM
from invpro.utils.applier import GOAL_TIMEOUT_SECONDS, StateFetchAbort, TacticApplier
from invpro.utils.state_to_statement import (
    StateProblemConverter,
    extract_theorem_name,
)

# Tactics applied at proof start to get extra state candidates.
TRY_AT_STAR_TACTICS = [
    "try ring_nf at *",
    "try simp at *",
    "try norm_num at *",
    "try field_simp at *",
]

# Optional explicit commutativity/symmetry rewrites (from invpro.lean.Rewrites).
EXPLICIT_COMM_RW_RULES = [
    "ge_iff_le",
    "← ge_iff_le",
    "← gt_iff_lt",
    "gt_iff_lt",
    "eq_comm",
    "add_comm",
    "mul_comm",
    "and_comm",
    "or_comm",
]

# Type aliases for module constructor (StateCandidate imported from rewrites).
RerankingHeuristic = Callable[[List[StateCandidate], int], List[StateCandidate]]
EnergyHeuristic = Callable[[BaseLLM, str], float]

class RewritingCanonicalizationModule(SimpleCanonicalizationModule):
    """
    Three-stage retrieval: (1) rewriting sampling via rwcnc, (2) reranking heuristic,
    (3) energy evaluation. Goal: find an equivalent state that maximizes the model's
    probability of generating a successful proof.

    Stage 1 stays in-module (uses SingleFileClient). Stages 2 and 3 are pluggable
    via reranking_heuristic and energy_heuristic. Use .factory(config, llm) to build
    from a config dict.
    """

    def __init__(
        self,
        project_root: str,
        llm: BaseLLM,
        initial_imports: str = "import Mathlib\n",
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
        top_rewrites: int = 10,
        max_per_step: int = 10,
        depth: int = 2,
        reverse_order: bool = False,
        wait_after_update: float = 0.1,
        retry_wait: float = 1.0,
        inactivity_timeout: float = 15.0,
        filter_rewrite_namespaces: Optional[List[str]] = None,
        namespace_blacklist: Optional[List[str]] = None,
        only_simplifying_rewrites: bool = False,
        use_explicit_comm: bool = False,
        use_combined: bool = False,
        num_combined: int = 20,
        reranking_heuristic: Optional[RerankingHeuristic] = None,
        energy_heuristic: Optional[EnergyHeuristic] = None,
        inject_energy_type: Optional[str] = None,
        cache_config_tag: Optional[str] = None,
    ):
        """
        inject_energy_type: When set (e.g. "theorem_surprise"), the prover (CanonicalLLMProver)
        will inject this energy instead of building it here. Leave None for "first_tactic_confidence".
        cache_config_tag: Stable string for get_states cache key (e.g. "energy:none", "energy:theorem_surprise")
        so different strategies do not share cache entries.
        """
        imports = ensure_rewrites_import(
            initial_imports if initial_imports.endswith("\n") else initial_imports + "\n"
        )
        super().__init__(
            project_root=project_root,
            initial_imports=imports,
            timeout_seconds=timeout_seconds,
            temp_suffix="rwcnc",
        )
        self._llm = llm
        self._top_rewrites = top_rewrites
        self._max_per_step = max_per_step
        self._depth = depth
        self._reverse_order = reverse_order
        self._wait_after_update = wait_after_update
        self._retry_wait = retry_wait
        self._inactivity_timeout = inactivity_timeout
        self._filter_rewrite_namespaces = filter_rewrite_namespaces
        self._namespace_blacklist = namespace_blacklist
        self._only_simplifying_rewrites = only_simplifying_rewrites
        self._use_explicit_comm = bool(use_explicit_comm)
        self._use_combined = bool(use_combined)
        self._num_combined = max(1, int(num_combined))
        self._combined_rng = random.Random(0)
        if reranking_heuristic is None:
            raise ValueError(
                "RewritingCanonicalizationModule requires reranking_heuristic; "
                "set canonicalization.reranking in config (e.g. { type: 'shortest_states' })."
            )
        self._reranking_heuristic = reranking_heuristic
        # When None: no LLM scoring; pick top reranked candidate only; no fully-augmented candidate.
        self._energy_heuristic = energy_heuristic
        self._inject_energy_type = inject_energy_type
        self._cache_config_tag = cache_config_tag or ""
        self._state_converter: Optional[StateProblemConverter] = None
        self._current_imports_raw: Optional[str] = None  # imports without rewrites; used for original-statement energy scoring

    def reset(self, imports: str, theorem_statement: str) -> None:
        """
        Reset with imports and theorem. Ensures invpro.lean.Rewrites is imported
        so rwcnc is available (required for rewriting canonicalization).
        Stores imports and theorem for energy functions that need context
        (e.g. single_pass_confidence converts state -> problem).
        """
        self._current_imports_raw = imports
        imports_with_rewrites = ensure_rewrites_import(imports)
        self._current_imports = imports_with_rewrites
        self._current_theorem_stmt = theorem_statement
        super().reset(imports_with_rewrites, theorem_statement)

    def close(self) -> None:
        """Close the state converter if used by get_augmentation, then run base cleanup."""
        if self._state_converter is not None:
            try:
                self._state_converter.close()
            except Exception:
                pass
            self._state_converter = None
        super().close()

    @classmethod
    def factory(
        cls,
        config: dict,
        llm: BaseLLM,
        initial_imports: Optional[str] = None,
    ) -> "RewritingCanonicalizationModule":
        """
        Build a RewritingCanonicalizationModule from a config dict (e.g. from YAML).
        config: canonicalization parameters (project_root, timeout_seconds, top_rewrites,
                sampling: { max_per_step, depth, only_simplifying_rewrites, reverse_order }, reranking, energy, filter_rewrite_namespaces).
                energy may include type and confidence_aggregation (e.g. { type: "confidence", confidence_aggregation: "mean" }).
        """
        if "parameters" in config and isinstance(config["parameters"], dict):
            params = config["parameters"]
        else:
            params = config
        project_root = params["project_root"]
        timeout_seconds = params.get("timeout_seconds", 90.0)
        top_rewrites = params.get("top_rewrites", 10)
        sampling = params.get("sampling") or {}
        max_per_step = sampling.get("max_per_step", params.get("max_per_step", 10))
        depth = sampling.get("depth", params.get("depth", 2))
        filter_rewrite_namespaces = params.get("filter_rewrite_namespaces")
        namespace_blacklist = params.get("namespace_blacklist")
        only_simplifying_rewrites = sampling.get(
            "only_simplifying_rewrites",
            params.get("only_simplifying_rewrites", False),
        )
        use_explicit_comm = sampling.get(
            "use_explicit_comm",
            params.get("use_explicit_comm", False),
        )
        use_combined = sampling.get(
            "use_combined",
            params.get("use_combined", False),
        )
        num_combined = sampling.get(
            "num_combined",
            params.get("num_combined", 20),
        )
        reverse_order = sampling.get(
            "reverse_order",
            params.get("reverse_order", False),
        )
        reranking_heuristic = get_reranking_heuristic(params.get("reranking") or {})
        energy_cfg = params.get("energy")
        inject_energy_type = None
        if energy_cfg is None or (
            isinstance(energy_cfg, dict) and energy_cfg.get("type") == "none"
        ):
            energy_heuristic = None
        elif isinstance(energy_cfg, dict) and energy_cfg.get("type") == "theorem_surprise":
            # Prover will inject this (energy scores statements; no state_to_problem in energy).
            energy_heuristic = None
            inject_energy_type = "theorem_surprise"
        elif isinstance(energy_cfg, dict) and energy_cfg.get("type") == "confidence":
            # Next-tactic (step-by-step) confidence. Prover may keep it if LLM has generate_greedy_with_confidence.
            energy_heuristic = get_energy_heuristic(energy_cfg or {}, project_root)
            inject_energy_type = "confidence"
        else:
            energy_heuristic = get_energy_heuristic(
                energy_cfg or {}, project_root
            )
        cache_config_tag = cache_config_tag_from_energy(energy_cfg)
        return cls(
            project_root=project_root,
            llm=llm,
            initial_imports=initial_imports or "import Mathlib\n",
            timeout_seconds=timeout_seconds,
            top_rewrites=top_rewrites,
            max_per_step=max_per_step,
            depth=depth,
            reverse_order=reverse_order,
            filter_rewrite_namespaces=filter_rewrite_namespaces,
            namespace_blacklist=namespace_blacklist,
            only_simplifying_rewrites=only_simplifying_rewrites,
            use_explicit_comm=use_explicit_comm,
            use_combined=use_combined,
            num_combined=num_combined,
            reranking_heuristic=reranking_heuristic,
            energy_heuristic=energy_heuristic,
            inject_energy_type=inject_energy_type,
            cache_config_tag=cache_config_tag,
        )

    def _build_rwcnc_line(self, hyp: str) -> str:
        """Return the proof line that runs rwcnc at the given hypothesis.
        When only_simplifying_rewrites is True, adds ' true' so rwcnc returns only
        rewrites that do not increase complexity or depth.
        """
        middle = (
            " true" if self._only_simplifying_rewrites else ""
        )
        reverse = " reverse" if self._reverse_order else ""
        return (
            "\n  rwcnc "
            + str(self._max_per_step)
            + " "
            + str(self._depth)
            + middle
            + reverse
            + " at "
            + hyp
            + "\n"
        )

    def _build_rwcnc_goal_line(self) -> str:
        """Return the proof line that runs rwcnc on the goal (no 'at h' part)."""
        middle = (
            " true" if self._only_simplifying_rewrites else ""
        )
        reverse = " reverse" if self._reverse_order else ""
        return (
            "\n  rwcnc "
            + str(self._max_per_step)
            + " "
            + str(self._depth)
            + middle
            + reverse
            + "\n"
        )

    def _rwcache_dir(self) -> Path:
        """Directory for rwcnc rewrite cache and get_states result cache (.cache/rw)."""
        return get_rw_cache_dir(self.project_root)

    def _try_get_rewrites_from_cache(
        self,
        content: str,
        depth: int,
        max_per_step: int,
        only_simplify: bool = False,
    ) -> Optional[RewritesMap]:
        """
        If a cache entry exists for (content, depth, max_per_step, only_simplify), load and return it.
        Otherwise return None. Returns dict[hyp -> list of RewriteEntry].
        """
        return load_rewrites_cache(
            self._rwcache_dir(), content, depth, max_per_step, only_simplify, self._reverse_order
        )

    def _store_rewrites_cache(
        self,
        content: str,
        depth: int,
        max_per_step: int,
        result: RewritesMap,
        only_simplify: bool = False,
    ) -> None:
        """Save result to .cache/rw/<key>.cache. Uses UTF-8."""
        save_rewrites_cache(
            self._rwcache_dir(), content, depth, max_per_step, result, only_simplify, self._reverse_order
        )

    def _get_rewrites(
        self,
        file_content: str,
        goal: str,
        hyps: list[str],
        filter_propext: bool = True,
    ) -> RewritesMap:
        """Run rwcnc at each hypothesis and on the goal; restore file after each. When hyps is empty, only goal rewrites are run."""
        content = ensure_rewrites_import(file_content).rstrip("\n")
        cached = self._try_get_rewrites_from_cache(
            content,
            self._depth,
            self._max_per_step,
            self._only_simplifying_rewrites,
        )
        if cached is not None:
            if self._namespace_blacklist:
                cached = {
                    h: filter_rewrites_by_namespace_blacklist(
                        entries, self._namespace_blacklist
                    )
                    for h, entries in cached.items()
                }
            if self._filter_rewrite_namespaces:
                cached = {
                    h: filter_rewrites_by_namespace(
                        entries, self._filter_rewrite_namespaces
                    )
                    for h, entries in cached.items()
                }
            return cached
        result: RewritesMap = {}

        for i, hyp in enumerate(hyps):
            # Append only the rwcnc line so we don't resend imports and retrigger reloads.
            insert_start = self._eof_position()
            rwcnc_line = self._build_rwcnc_line(hyp)
            self._append_to_file(rwcnc_line)
            time.sleep(self._wait_after_update)
            try:
                diag = self.sfc.get_diagnostics(
                    inactivity_timeout=self._inactivity_timeout
                )
            except Exception:
                diag = None
            diag_list = getattr(diag, "diagnostics", None) or []
            if not diag_list and self._retry_wait > 0:
                time.sleep(self._retry_wait)
                try:
                    diag = self.sfc.get_diagnostics(
                        inactivity_timeout=self._inactivity_timeout
                    )
                    diag_list = getattr(diag, "diagnostics", None) or []
                except Exception:
                    diag_list = []

            raw = collect_try_this_from_diagnostics(diag_list)
            if filter_propext:
                raw = [x for x in raw if "propext" not in x[0]]
            deduped = deduplicate_by_state(raw)
            # Stage 1: keep up to max_per_step rewrites per hyp (reranking will narrow later).
            result[hyp] = [
                RewriteEntry.from_tuple(p) for p in deduped[: self._max_per_step]
            ]

            # Remove the rwcnc line so the file is back to previous state.
            self._delete_range(insert_start, self._eof_position())

        # Run rwcnc on the goal (no "at h"): same params, rewrite the ⊢ target.
        insert_start = self._eof_position()
        rwcnc_goal_line = self._build_rwcnc_goal_line()
        self._append_to_file(rwcnc_goal_line)
        time.sleep(self._wait_after_update)
        try:
            diag = self.sfc.get_diagnostics(
                inactivity_timeout=self._inactivity_timeout
            )
        except Exception:
            diag = None
        diag_list = getattr(diag, "diagnostics", None) or []
        if not diag_list and self._retry_wait > 0:
            time.sleep(self._retry_wait)
            try:
                diag = self.sfc.get_diagnostics(
                    inactivity_timeout=self._inactivity_timeout
                )
                diag_list = getattr(diag, "diagnostics", None) or []
            except Exception:
                diag_list = []
        raw_goal = collect_try_this_from_diagnostics(diag_list)
        if filter_propext:
            raw_goal = [x for x in raw_goal if "propext" not in x[0]]
        deduped_goal = deduplicate_by_state(raw_goal)
        result[GOAL_REWRITE_KEY] = [
            RewriteEntry.from_tuple(p) for p in deduped_goal[: self._max_per_step]
        ]
        self._delete_range(insert_start, self._eof_position())

        self.invalidate_state_cache()
        self._store_rewrites_cache(
            content,
            self._depth,
            self._max_per_step,
            result,
            self._only_simplifying_rewrites,
        )
        if self._namespace_blacklist:
            result = {
                h: filter_rewrites_by_namespace_blacklist(
                    lst, self._namespace_blacklist
                )
                for h, lst in result.items()
            }
        if self._filter_rewrite_namespaces:
            result = {
                h: filter_rewrites_by_namespace(
                    lst, self._filter_rewrite_namespaces
                )
                for h, lst in result.items()
            }
        return result

    def _rewrite_options_for_target(
        self,
        hyp: str,
        current: str,
        rewrites: RewritesMap,
    ) -> list[tuple[str, str]]:
        """
        Return reranked rewrite options for one target as (rewritten_type, tactic).
        Original/no-rewrite is handled later via phantom options in combined sampling.
        """
        top_list = list(rewrites.get(hyp, []))
        if self._use_explicit_comm:
            top_list.extend(self._build_explicit_comm_rewrites(hyp))
        two_tuples = [(e.tactic, e.premise) for e in top_list]
        if hyp == GOAL_REWRITE_KEY:
            all_states = build_states_for_goal(current, two_tuples, top_k=len(two_tuples))
        else:
            all_states = build_states_for_hypothesis(
                current, hyp, two_tuples, top_k=len(two_tuples)
            )
        full_list: List[StateCandidate] = [
            (all_states[0][0], all_states[0][1] or "", all_states[0][2] or "", None, None)
        ]
        seen_state: set[str] = {all_states[0][0]}
        for i in range(1, len(all_states)):
            s = all_states[i][0]
            if s not in seen_state:
                seen_state.add(s)
                entry = top_list[i - 1]
                full_list.append(
                    (
                        all_states[i][0],
                        all_states[i][1],
                        all_states[i][2],
                        entry.complexity,
                        entry.depth,
                    )
                )
        reranked = self._reranking_heuristic(full_list[1:], self._top_rewrites)
        out: list[tuple[str, str]] = []
        for _state_str, rw_tactic, rw_type, _c, _d in reranked:
            rw_tactic = (rw_tactic or "").strip()
            rw_type = (rw_type or "").strip()
            if rw_tactic and rw_type:
                out.append((rw_type, rw_tactic))
        return out

    def _get_state_after_try_tactic(self, tactic: str) -> Optional[str]:
        """
        Append "try X at *" at proof start, retrieve state, then remove the line.
        Same pattern as SimpModule. Returns the state string or None if fetch fails.
        """
        if self._applier is None or self._cached_state is None:
            self._applier = self._applier or TacticApplier(
                self.sfc, timeout_seconds=self.timeout_seconds
            )
            try:
                self._cached_state = self._applier._build_hypothesis_stack_from_code(
                    reset_stack=True
                )
            except StateFetchAbort:
                # If state fetch times out, skip "try at *" candidate generation.
                self._cached_state = None
                return None
        insert_start = self._eof_position()
        try:
            self._append_to_file("  " + tactic + "\n")
            aug_state = self._applier._build_hypothesis_stack_from_code(
                reset_stack=True
            )
            return aug_state
        except StateFetchAbort:
            # Timeouts during exploratory "try at *" tactics should not abort export.
            return None
        finally:
            cur = self.sfc.get_file_content()
            cur_end = self._eof_position(cur)
            if cur_end != insert_start:
                self._delete_range(insert_start, cur_end)
            self._content = self.sfc.get_file_content()
            try:
                self._cached_state = self._applier._build_hypothesis_stack_from_code(
                    reset_stack=True
                )
            except StateFetchAbort:
                # Keep running; just invalidate cached state.
                self._cached_state = None

    def _build_explicit_comm_rewrites(self, hyp: str) -> list[RewriteEntry]:
        """
        Apply explicit commutativity/symmetry rewrites and convert successful results
        into RewriteEntry items that are merged with rwcnc suggestions.
        """
        out: list[RewriteEntry] = []
        seen_pairs: set[tuple[str, str]] = set()
        for rule in EXPLICIT_COMM_RW_RULES:
            if hyp == GOAL_REWRITE_KEY:
                tactic = f"try rw [{rule}]"
                aug_state = self._get_state_after_try_tactic(tactic)
                if aug_state and aug_state.strip() == "no goals":
                    continue
                rewritten = get_goal_expression(aug_state or "").strip()
            else:
                tactic = f"try rw [{rule}] at {hyp}"
                aug_state = self._get_state_after_try_tactic(tactic)
                if aug_state and aug_state.strip() == "no goals":
                    continue
                rewritten = (get_hypothesis_type(aug_state or "", hyp) or "").strip()

            if not rewritten:
                continue
            key = (tactic, rewritten)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            out.append(
                RewriteEntry(
                    tactic=tactic,
                    premise=rewritten,
                    complexity=None,
                    depth=None,
                )
            )
        return out

    def _build_state_candidates_from_rewrites(
        self,
        current: str,
        hyps: list[str],
        rewrites: RewritesMap,
    ) -> list[tuple[str, list[str]]]:
        """Build rewrite-derived state candidates (single-target or combined mode)."""
        hyps_and_goal: list[str] = list(hyps) + [GOAL_REWRITE_KEY]
        state_candidates: list[tuple[str, list[str]]] = []

        if self._use_combined:
            option_pool_per_target: dict[str, list[Optional[tuple[str, str]]]] = {}
            option_weights_per_target: dict[str, list[float]] = {}
            for target in hyps_and_goal:
                rewrite_options = self._rewrite_options_for_target(target, current, rewrites)
                option_pool_per_target[target] = [None] + rewrite_options
                # Weighted sampling: original ~ 1, reranked aug_i ~ 1/(i+1), i starts at 1.
                option_weights_per_target[target] = [1.0] + [
                    1.0 / (i + 1) for i in range(1, len(rewrite_options) + 1)
                ]

            for _ in range(self._num_combined):
                selected_types: dict[str, str] = {}
                selected_tactics: list[str] = []
                for target in hyps_and_goal:
                    chosen = self._combined_rng.choices(
                        option_pool_per_target[target],
                        weights=option_weights_per_target[target],
                        k=1,
                    )[0]
                    if chosen is None:
                        continue
                    rw_type, rw_tactic = chosen
                    selected_types[target] = rw_type
                    if rw_tactic:
                        selected_tactics.append(rw_tactic)
                combined_state = build_fully_augmented_goal_from_best_types(
                    current, hyps, selected_types
                )
                rw_list = ["\n".join(selected_tactics)] if selected_tactics else []
                state_candidates.append((combined_state, rw_list))
        else:
            # Default mode: keep single-target augmented states only.
            for target in hyps_and_goal:
                rewrite_options = self._rewrite_options_for_target(target, current, rewrites)
                for rw_type, rw_tactic in rewrite_options:
                    if target == GOAL_REWRITE_KEY:
                        state_str = build_fully_augmented_goal_from_best_types(
                            current, hyps, {GOAL_REWRITE_KEY: rw_type}
                        )
                    else:
                        state_str = build_fully_augmented_goal_from_best_types(
                            current, hyps, {target: rw_type}
                        )
                    rw_list = [rw_tactic] if rw_tactic else []
                    state_candidates.append((state_str, rw_list))

        # Keep existing extra exploratory tactics in both modes.
        for tactic in TRY_AT_STAR_TACTICS:
            aug_state = self._get_state_after_try_tactic(tactic)
            if aug_state is not None:
                state_candidates.append((aug_state, [tactic]))

        by_state: dict[str, tuple[str, list[str]]] = {}
        for state_str, rw_list in state_candidates:
            if state_str not in by_state or len(rw_list) > len(by_state[state_str][1]):
                by_state[state_str] = (state_str, rw_list)
        return list(by_state.values())

    def _pick_best_by_energy(
        self,
        state_candidates: list[tuple[str, list[str]]],
        current: str,
    ) -> tuple[bool, Optional[str], list[str]]:
        """Convert states to statements, score with energy, return (use_original_problem, best_state, rw_tactics)."""
        use_original_problem = True
        best_state = current
        rw_tactics: list[str] = []

        # If any candidate is "no goals", the tactic(s) already proved the theorem; return it immediately.
        for state_str, rw_list in state_candidates:
            if state_str and state_str.strip() == "no goals":
                return (False, state_str, rw_list)

        if self._energy_heuristic is None or not self._current_imports or not self._current_theorem_stmt:
            return (use_original_problem, best_state, rw_tactics)

        # Use raw imports (no rewrites line) for the original statement so energy sees the user's actual problem.
        imports_for_original = getattr(self, "_current_imports_raw", None) or self._current_imports or ""
        original_statement = (
            imports_for_original.rstrip("\n")
            + "\n"
            + (self._current_theorem_stmt or "").strip("\n")
        )
        statement_candidates: list[tuple[Optional[str], str, list[str]]] = [
            (None, original_statement, []),
        ]
        theorem_name = extract_theorem_name(self._current_theorem_stmt or "")
        if self._state_converter is None:
            self._state_converter = StateProblemConverter(
                project_root=str(self.project_root)
            )
        for state_str, rw_list in state_candidates:
            stmt = self._state_converter.convert(
                self._current_imports or "",
                state_str,
                theorem_name,
            )
            if stmt is not None:
                statement_candidates.append((state_str, stmt, rw_list))

        scores = [self._energy_heuristic(self._llm, stmt) for _s, stmt, _r in statement_candidates]
        finite = [(i, s) for i, s in enumerate(scores) if math.isfinite(s)]
        if finite:
            best_i = max(finite, key=lambda x: (x[1], -len(statement_candidates[x[0]][2])))
            best_idx = best_i[0]
            use_original_problem = best_idx == 0
            best_state = current if use_original_problem else statement_candidates[best_idx][0]
            rw_tactics = statement_candidates[best_idx][2]

        return (use_original_problem, best_state, rw_tactics)

    def _save_get_states_cache(
        self,
        gs_key: str,
        current: str,
        best_state: Optional[str],
        rw_tactics: list[str],
        use_original_problem: bool,
    ) -> None:
        """Persist get_states result to cache dir. Silently ignores OSError."""
        try:
            cache_dir = self._rwcache_dir()
            gs_path = cache_dir / f"{gs_key}.cache"
            save_get_states_entry(
                gs_path,
                GetStatesCacheEntry(
                    current=current,
                    best_state=best_state or "",
                    rw_tactics=rw_tactics,
                    use_original_problem=use_original_problem,
                ),
            )
        except OSError:
            pass

    def get_states(
        self, keep_augmentation: bool = False
    ) -> Tuple[Optional[str], Optional[str], List[str], bool]:
        """Run rewriting pipeline (rewrites, rerank, energy), return (current, best_state, rw_tactics, use_original_problem)."""
        current = self.get_current_state()
        if current is None or not current.strip():
            return (current, current, [], False)
        hyps = hyps_from_goal(current)

        file_content = self.get_file_content()
        content = ensure_rewrites_import(file_content).rstrip("\n")
        model_cache_id = getattr(self, "_model_cache_id", "") or ""
        gs_key = get_states_cache_key(
            content,
            self._depth,
            self._max_per_step,
            self._only_simplifying_rewrites,
            self._reverse_order,
            self._top_rewrites,
            self._filter_rewrite_namespaces,
            model_cache_id=model_cache_id,
            cache_config_tag=getattr(self, "_cache_config_tag", ""),
        )

        rewrites = self._get_rewrites(file_content, current, hyps)
        state_candidates = self._build_state_candidates_from_rewrites(current, hyps, rewrites)
        use_original_problem, best_state, rw_tactics = self._pick_best_by_energy(
            state_candidates, current
        )

        self._save_get_states_cache(
            gs_key, current, best_state, rw_tactics, use_original_problem
        )

        if rw_tactics and keep_augmentation:
            for t in rw_tactics:
                st = self.apply_tactic(t)
                if st is None:
                    return (current, current, [], use_original_problem)
            augmented = self.get_current_state()
            return (current, augmented, rw_tactics, use_original_problem)
        return (current, best_state, rw_tactics, use_original_problem)

    def get_state_candidates(
        self,
        sorted_by_energy: bool = False,
    ) -> Optional[Tuple[str, List[Tuple[str, List[str]]]]]:
        """
        Return (current_state, state_candidates) for external consumers.
        state_candidates is the list of (state_str, rw_list) passed to energy
        (same as in _pick_best_by_energy, excluding the original).

        When sorted_by_energy=True, candidates are ordered by descending energy
        preference (best-first) using this module's configured energy function.
        """
        current = self.get_current_state()
        if current is None or not current.strip():
            return None
        hyps = hyps_from_goal(current)
        file_content = self.get_file_content()
        rewrites = self._get_rewrites(file_content, current, hyps)
        candidates = self._build_state_candidates_from_rewrites(
            current, hyps, rewrites
        )
        if sorted_by_energy:
            energy_fn = self._energy_heuristic
            if energy_fn is not None and self._current_imports and self._current_theorem_stmt:
                theorem_name = extract_theorem_name(self._current_theorem_stmt or "")
                scored: list[tuple[float, tuple[str, list[str]]]] = []
                for state_str, rw_list in candidates:
                    # A "no goals" candidate means rw_tactics already prove the theorem.
                    # Put it first when present.
                    if state_str and state_str.strip() == "no goals":
                        scored.append((float("inf"), (state_str, rw_list)))
                        continue
                    try:
                        stmt = StateProblemConverter.state_to_problem_pure(
                            self._current_imports or "",
                            state_str,
                            theorem_name,
                        )
                        score = float(energy_fn(self._llm, stmt))
                    except Exception:
                        score = float("-inf")
                    scored.append((score, (state_str, rw_list)))
                scored.sort(key=lambda x: x[0], reverse=True)
                candidates = [c for _s, c in scored]
        return (current, candidates)

    def get_augmentation(
        self,
        imports: str,
        theorem_stmt: str,
        theorem_name: Optional[str] = None,
    ) -> Tuple[Optional[str], List[str], str]:
        """
        Return the best augmented theorem statement (statement domain).

        Same pipeline as get_states (rewrites, reranking, energy), but takes
        imports + theorem statement and returns (augmented_statement, rw_tactics, cache_key).
        Uses StateProblemConverter for state->statement conversion.
        Cache key uses same content as get_states() (with rewrites import) so keys match.
        """
        raw_content = imports.rstrip("\n") + "\n" + theorem_stmt.strip("\n")
        content = ensure_rewrites_import(raw_content).rstrip("\n")
        model_cache_id = getattr(self, "_model_cache_id", "") or ""
        gs_key = get_states_cache_key(
            content,
            self._depth,
            self._max_per_step,
            self._only_simplifying_rewrites,
            self._reverse_order,
            self._top_rewrites,
            self._filter_rewrite_namespaces,
            model_cache_id=model_cache_id,
            cache_config_tag=getattr(self, "_cache_config_tag", ""),
        )
        self.reset(imports, theorem_stmt)
        _current, best_state, rw_tactics, use_original_problem = self.get_states(
            keep_augmentation=True
        )
        if use_original_problem:
            original_problem = imports.rstrip("\n") + "\n" + theorem_stmt.strip("\n")
            return (original_problem, [], gs_key)
        if best_state is None or not best_state.strip():
            return (None, [], gs_key)
        # "no goals" means the tactic(s) already proved the theorem; no statement to convert.
        if best_state.strip() == "no goals":
            return (None, rw_tactics, gs_key)
        if theorem_name is None:
            theorem_name = extract_theorem_name(theorem_stmt)
        if self._state_converter is None:
            self._state_converter = StateProblemConverter(project_root=str(self.project_root))
        augmented = self._state_converter.convert(imports, best_state, theorem_name)
        return (augmented, rw_tactics, gs_key)

    def get_candidate_states_for_external_scoring(self, top_k: int) -> List[str]:
        """
        Get candidate states for external energy scoring (e.g. single-pass LLM).

        Runs rewriting pipeline (sampling, filtering by only_simplifying_rewrites,
        reranking) and returns the top_k state strings. Use when the energy
        function is evaluated externally (e.g. by running a single-pass model on
        problem statements derived from these states).

        Returns:
            List of state strings (goal state format), including the original.
            Empty only if no current state. When there are no hypotheses, only goal rewrites are considered.
        """
        current = self.get_current_state()
        if current is None or not current.strip():
            return []
        hyps = hyps_from_goal(current)
        # When hyps is empty we still run goal-only rewriting.

        file_content = self.get_file_content()
        rewrites = self._get_rewrites(file_content, current, hyps)

        all_candidates: List[StateCandidate] = []
        seen_states: set[str] = set()

        for hyp in list(hyps) + [GOAL_REWRITE_KEY]:
            top_list = rewrites.get(hyp, [])
            two_tuples = [(e.tactic, e.premise) for e in top_list]
            if hyp == GOAL_REWRITE_KEY:
                all_states = build_states_for_goal(
                    current, two_tuples, top_k=len(two_tuples)
                )
            else:
                all_states = build_states_for_hypothesis(
                    current, hyp, two_tuples, top_k=len(two_tuples)
                )
            for i in range(1, len(all_states)):
                state_str, rw_tac, rw_typ = all_states[i][0], all_states[i][1], all_states[i][2]
                if state_str in seen_states:
                    continue
                seen_states.add(state_str)
                entry = top_list[i - 1]
                all_candidates.append(
                    (state_str, rw_tac or "", rw_typ or "", entry.complexity, entry.depth)
                )

        original_state = (current, "", "", None, None)
        reranked = self._reranking_heuristic(all_candidates, top_k - 1)
        result = [current] + [s[0] for s in reranked]
        return result[:top_k]

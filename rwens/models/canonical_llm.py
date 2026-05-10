"""
Canonical single-pass prover: convert problem to a favourable form via rewriting,
then solve with a single-pass LLM.

Uses RewritingCanonicalizationModule to produce candidate states (filtered by
complexity/depth, reranked), scores them with energy inside the module
(e.g. first-tactic confidence from single-pass model), and runs the single-pass
prover on the best candidate.
"""

from __future__ import annotations

from pathlib import Path
from tqdm import tqdm
from typing import Callable, List, Optional

from rwens.canonicalization.rewrites import (
    get_states_cache_key,
    load_get_states_entry,
    make_single_pass_confidence_energy,
    make_theorem_surprise_energy,
)
from rwens.canonicalization.rewrites.cache import GetStatesCacheEntry
from rwens.canonicalization.utils import ensure_rewrites_import
from rwens.utils.cache_paths import get_rw_cache_dir
from rwens.dataset.utils import split_declarations_theorem_proof
from rwens.models.base import FormalProver
from rwens.utils.verifier import ProofVerifier
from rwens.utils.state_to_statement import (
    StateProblemConverter,
    extract_theorem_name,
)


class CanonicalLLMProver(FormalProver):
    """
    Single-pass prover that first converts the problem into a favourable form
    via rewriting, then solves with a single-pass LLM.

    Flow:
    1. Parse problem into imports and theorem statement
    2. Get augmented theorem statement via module.get_augmentation (statement domain;
       energy and state->statement conversion inside the module)
    3. Run single_pass_prover.prove(augmented_problem)
    """

    def __init__(
        self,
        canonicalization_module: object,  # e.g. RewritingCanonicalizationModule
        single_pass_prover: FormalProver,
        state_to_problem_fn: Optional[
            Callable[[str, str, Optional[str]], Optional[str]]
        ] = None,
        extract_confidence_fn: Optional[
            Callable[[object, str, Optional[object]], float]
        ] = None,
    ):
        """
        Args:
            canonicalization_module: Canonicalization module (e.g. RewritingCanonicalizationModule).
                Must have get_augmentation(imports, theorem_stmt, theorem_name) (statement domain).
                Energy is injected by this prover when applicable. If module._inject_energy_type is
                "theorem_surprise" (from rewriting.parameters.energy.type), that energy is used.
            single_pass_prover: Prover that generates full proof in one pass (e.g. SinglePassProver).
            state_to_problem_fn: (imports, state_str, theorem_name?) -> problem or None.
                Default: StateProblemConverter with round-trip verification.
            extract_confidence_fn: (llm, problem, prompt_formatter?) -> float for first-tactic
                confidence (used when inject_energy_type is first_tactic_confidence).
        """
        self._canonicalization_module = canonicalization_module
        self._single_pass_prover = single_pass_prover
        if state_to_problem_fn is None:
            converter = StateProblemConverter(
                project_root=str(canonicalization_module.project_root),
            )
            self._state_to_problem_fn = converter.convert
            self._converter = converter
        else:
            self._state_to_problem_fn = state_to_problem_fn
            self._converter = None

        prompt_formatter = getattr(single_pass_prover, "prompt_formatter", None)
        single_pass_llm = getattr(single_pass_prover, "llm", None)
        module = canonicalization_module
        inject_energy_type = getattr(module, "_inject_energy_type", None)
        project_root_str = str(canonicalization_module.project_root)
        if inject_energy_type == "theorem_surprise":
            energy = make_theorem_surprise_energy(
                project_root=project_root_str,
                prompt_formatter=prompt_formatter,
                verbose=False,
                use_cache=True,
                llm=single_pass_llm,
            )
            module._cache_config_tag = "energy:theorem_surprise"
            module._energy_heuristic = energy
        elif (
            inject_energy_type == "confidence"
            and single_pass_llm is not None
            and hasattr(single_pass_llm, "generate_greedy_with_confidence")
        ):
            # Keep the module's next-tactic (step-by-step) confidence energy.
            module._cache_config_tag = (
                getattr(module, "_cache_config_tag", "") or "energy:confidence"
            )
        else:
            energy = make_single_pass_confidence_energy(
                project_root=project_root_str,
                prompt_formatter=prompt_formatter,
                extract_confidence_fn=extract_confidence_fn,
                llm=single_pass_llm,
            )
            module._cache_config_tag = "energy:first_tactic_confidence"
            module._energy_heuristic = energy
        module._model_cache_id = (
            single_pass_llm.get_model_cache_id() or ""
            if single_pass_llm and hasattr(single_pass_llm, "get_model_cache_id")
            else ""
        )
        self._verifier = ProofVerifier(
            project_root=str(canonicalization_module.project_root),
            initial_imports="import Mathlib\n",
        )

    @property
    def project_root(self) -> Path:
        return Path(self._canonicalization_module.project_root)

    def close(self) -> None:
        """Clean up resources (e.g. StateProblemConverter's Lean session)."""
        if self._converter is not None:
            self._converter.close()
            self._converter = None

    def generate(self, problem_statement: str) -> Optional[str]:
        """
        Generate full Lean code without verification.

        Flow mirrors prove() up to generation, but skips verifier calls.
        """
        generated = self.generate_batch(
            problem_statement=problem_statement,
            n_attempts=-1,
            batch_size=1,
        )
        if isinstance(generated, list):
            return generated[0] if generated else None
        return generated

    def _get_or_compute_augmentation(
        self,
        imports: str,
        theorem_stmt: str,
        theorem_name: Optional[str],
        original_problem: str,
    ) -> tuple[Optional[str], list[str]]:
        """
        Get best augmented theorem statement and rewrite tactics via get_augmentation.

        Uses module.get_augmentation (statement domain). Caches augmented problem to
        ap_path; rw_tactics come from the module's get_states cache (gs_path) on cache hit.
        Returns (augmented_problem, rw_tactics). augmented_problem may be None.
        """
        module = self._canonicalization_module
        if not hasattr(module, "get_augmentation"):
            return None, []

        # Use same content as get_states() (after reset): includes rewrites import so cache key matches.
        raw_content = imports.rstrip("\n") + "\n" + theorem_stmt.strip("\n")
        content = ensure_rewrites_import(raw_content).rstrip("\n")
        model_cache_id = getattr(module, "_model_cache_id", "") or ""
        gs_key = get_states_cache_key(
            content,
            module._depth,
            module._max_per_step,
            module._only_simplifying_rewrites,
            getattr(module, "_reverse_order", False),
            module._top_rewrites,
            module._filter_rewrite_namespaces,
            model_cache_id=model_cache_id,
            cache_config_tag=getattr(module, "_cache_config_tag", ""),
        )
        cache_dir = get_rw_cache_dir(module.project_root)
        ap_key = "ap_" + gs_key[3:]
        ap_path = cache_dir / f"{ap_key}.cache"
        gs_path = cache_dir / f"{gs_key}.cache"

        augmented_problem = None
        rw_tactics: list[str] = []

        if ap_path.exists():
            try:
                augmented_problem = ap_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass
            if augmented_problem:
                entry = load_get_states_entry(gs_path)
                if entry is not None:
                    rw_tactics = entry.rw_tactics
                # If we have augmented from cache but no rw_tactics (gs_path missing/failed),
                # re-run get_augmentation so _verify_with_rewrites gets rw_tactics and
                # final_code uses original + rewrites + proof consistently (attempt 1 vs 2+).
                elif augmented_problem.strip():
                    augmented_problem = None

        # No augmented-problem file: check get_states cache for "no goals" (single-tactic proof).
        # get_states() always writes gs_path when it runs, so a prior no-goals result is stored there.
        if augmented_problem is None and gs_path.exists():
            entry = load_get_states_entry(gs_path)
            if (
                entry is not None
                and entry.best_state
                and entry.best_state.strip() == "no goals"
                and entry.rw_tactics
            ):
                return (None, entry.rw_tactics)

        if augmented_problem is None:
            augmented_problem, rw_tactics, _ = module.get_augmentation(
                imports, theorem_stmt, theorem_name
            )
            # "no goals" returns (None, rw_tactics): proof is just those tactics; don't overwrite.
            if augmented_problem is None and not rw_tactics:
                augmented_problem = original_problem
            if augmented_problem is not None:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    ap_path.write_text(augmented_problem, encoding="utf-8")
                except OSError:
                    pass

        return augmented_problem, rw_tactics

    def _verify_with_rewrites(
        self,
        imports: str,
        theorem_stmt: str,
        rw_tactics: list[str],
        result: dict,
    ) -> dict:
        """
        When rewrites were used, verify against original statement + rewrites + proof
        instead of augmented statement + proof. Returns updated result.
        """
        if not rw_tactics or not result.get("final_code"):
            return result
        try:
            _, _, proof_body = split_declarations_theorem_proof(result["final_code"])
        except ValueError:
            return result
        if not proof_body:
            return result
        rw_block = "  " + "\n  ".join(rw_tactics) + "\n"
        combined_body = rw_block + proof_body
        combined_code = imports.rstrip("\n") + "\n" + theorem_stmt + combined_body
        success, error = self._verifier.verify(combined_code)
        out = {
            "success": success,
            "final_code": combined_code,
            "error": None if success else error,
            "steps": result.get("steps", []),
        }
        if "timings" in result:
            out["timings"] = result["timings"]
        return out

    def prove(self, problem_statement: str) -> dict:
        """
        Generate once, then verify.

        Returns:
            Same shape as single_pass_prover.prove: {success, final_code, error, steps}.
        """
        final_code = self.generate(problem_statement)
        if final_code is None:
            return {
                "success": False,
                "final_code": problem_statement,
                "error": "Generation failed",
                "steps": [],
            }
        success, error = self._verifier.verify(final_code)
        return {
            "success": success,
            "final_code": final_code,
            "error": None if success else error,
            "steps": [],
        }

    def generate_batch(
        self,
        problem_statement: str,
        n_attempts: int,
        batch_size: int = 1,
    ) -> Optional[str] | List[Optional[str]]:
        """
        Generate n_attempts full Lean code candidates without verification.

        Special case: when n_attempts <= 0, runs single-attempt generation and
        returns a single code value (or None) instead of a list.
        """
        single_mode = n_attempts <= 0
        attempts = 1 if single_mode else n_attempts

        try:
            decls, theorem_stmt, _ = split_declarations_theorem_proof(problem_statement)
        except ValueError:
            return None if single_mode else [None for _ in range(attempts)]

        imports = decls
        theorem_name = extract_theorem_name(theorem_stmt)
        augmented_problem, rw_tactics = self._get_or_compute_augmentation(
            imports, theorem_stmt, theorem_name, problem_statement
        )

        # "no goals": rewriting itself already gives a proof body.
        if augmented_problem is None and rw_tactics:
            proof_body = "  " + "\n  ".join(rw_tactics) + "\n"
            combined_code = (
                imports.rstrip("\n") + "\n" + theorem_stmt.rstrip("\n") + "\n" + proof_body
            )
            return combined_code if single_mode else [combined_code for _ in range(attempts)]

        target = (
            problem_statement
            if augmented_problem is None or not augmented_problem.strip()
            else augmented_problem
        )
        single_pass = self._single_pass_prover
        if hasattr(single_pass, "generate_batch"):
            generated = single_pass.generate_batch(
                target, n_attempts=attempts, batch_size=batch_size
            )
        else:
            generated = [single_pass.generate(target) for _ in range(attempts)]

        # If we solved an augmented theorem with rewrites, translate each generated
        # candidate back to original theorem + rewrites + proof body.
        if augmented_problem and augmented_problem.strip() and rw_tactics:
            out: List[Optional[str]] = []
            rw_block = "  " + "\n  ".join(rw_tactics) + "\n"
            for code in generated:
                if code is None:
                    out.append(None)
                    continue
                try:
                    _, _, proof_body = split_declarations_theorem_proof(code)
                except ValueError:
                    out.append(code)
                    continue
                if not proof_body:
                    out.append(code)
                    continue
                combined_body = rw_block + proof_body
                out.append(imports.rstrip("\n") + "\n" + theorem_stmt + combined_body)
            return out[0] if single_mode else out

        return generated[0] if single_mode else generated

    def prove_batch(
        self,
        problem_statement: str,
        n_attempts: int,
        batch_size: int = 1,
    ) -> List[dict]:
        """
        Generate n_attempts, then verify each candidate.
        """
        if n_attempts <= 0:
            return []

        generated = self.generate_batch(
            problem_statement=problem_statement,
            n_attempts=n_attempts,
            batch_size=batch_size,
        )
        if not isinstance(generated, list):
            generated = [generated]
        results: List[dict] = []
        for code in tqdm(generated):
            if code is None:
                results.append(
                    {
                        "success": False,
                        "final_code": problem_statement,
                        "error": "Generation failed",
                        "steps": [],
                    }
                )
                continue
            success, error = self._verifier.verify(code)
            results.append(
                {
                    "success": success,
                    "final_code": code,
                    "error": None if success else error,
                    "steps": [],
                }
            )
        return results

"""
Ensemble single-pass prover over multiple rewrite augmentations.

Unlike CanonicalLLMProver (pick-one augmentation), this prover samples up to K
augmentation candidates, shuffles them, and allocates attempts uniformly across
them. Augmented tactic states are turned into theorem text via
``StateProblemConverter.convert_no_compile`` (naive), or optionally via GPT using
``StateToStatementPromptFormatter`` (original statement + original state +
augmented state), with fallback to naive on failure.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, List, Optional, Tuple

from invpro.canonicalization.rewrites import (
    make_single_pass_confidence_energy,
    make_theorem_surprise_energy,
)
from invpro.dataset.utils import split_declarations_theorem_proof
from invpro.models.base import FormalProver
from invpro.prompt.state_to_statement import StateToStatementPromptFormatter
from invpro.utils.state_to_statement import (
    StateProblemConverter,
    extract_theorem_name,
)
from invpro.utils.verifier import ProofVerifier


class EnsembleLLMProver(FormalProver):
    def __init__(
        self,
        canonicalization_module: object,
        single_pass_prover: FormalProver,
        ensemble_size: int = 4,
        shuffle_seed: Optional[int] = None,
        state_to_statement_mode: str = "naive",
        gpt_state_to_statement_model: str = "gpt-5.4-mini",
        gpt_state_api_key_env: str = "OPENAI_API_KEY",
        gpt_state_max_output_tokens: int = 300,
    ):
        self._canonicalization_module = canonicalization_module
        self._single_pass_prover = single_pass_prover
        self._ensemble_size = max(1, int(ensemble_size))
        self._rng = random.Random(shuffle_seed)
        self._state_to_statement_mode = (state_to_statement_mode or "naive").strip().lower()
        self._gpt_state_model = gpt_state_to_statement_model
        self._gpt_state_api_key_env = gpt_state_api_key_env
        self._gpt_state_max_output_tokens = int(gpt_state_max_output_tokens)
        self._gpt_state_llm: Optional[Any] = None
        self._gpt_state_llm_init_failed = False
        self._state_to_stmt_formatter = StateToStatementPromptFormatter()

        # Match CanonicalLLMProver: RewritingCanonicalization leaves energy_heuristic unset for
        # theorem_surprise until injection here, so get_state_candidates(sorted_by_energy=True)
        # and exported energy scores use the same LLM-based heuristic as canonical proving.
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
            module._cache_config_tag = (
                getattr(module, "_cache_config_tag", "") or "energy:confidence"
            )
        else:
            energy = make_single_pass_confidence_energy(
                project_root=project_root_str,
                prompt_formatter=prompt_formatter,
                extract_confidence_fn=None,
                llm=single_pass_llm,
            )
            module._cache_config_tag = "energy:first_tactic_confidence"
            module._energy_heuristic = energy
        module._model_cache_id = (
            single_pass_llm.get_model_cache_id() or ""
            if single_pass_llm and hasattr(single_pass_llm, "get_model_cache_id")
            else ""
        )

        self._converter = StateProblemConverter(
            project_root=str(canonicalization_module.project_root),
        )
        self._verifier = ProofVerifier(
            project_root=str(canonicalization_module.project_root),
            initial_imports="import Mathlib\n",
        )

    @property
    def project_root(self) -> Path:
        return Path(self._canonicalization_module.project_root)

    def close(self) -> None:
        if self._converter is not None:
            self._converter.close()

    def _get_gpt_state_llm(self) -> Optional[Any]:
        """Lazy OpenAI client for GPT state→statement; None if mode is not gpt or init failed."""
        if self._state_to_statement_mode != "gpt":
            return None
        if self._gpt_state_llm_init_failed:
            return None
        if self._gpt_state_llm is not None:
            return self._gpt_state_llm
        try:
            from invpro.models.llm.openai_gpt import OpenAIGPTLLM

            self._gpt_state_llm = OpenAIGPTLLM.from_pretrained(
                model_name_or_path=self._gpt_state_model,
                temperature=0.0,
                max_output_tokens=self._gpt_state_max_output_tokens,
                api_key_env=self._gpt_state_api_key_env,
                env_file_path=str(self.project_root / ".env"),
            )
        except Exception:
            self._gpt_state_llm_init_failed = True
            return None
        return self._gpt_state_llm

    def _augmented_state_to_problem_statement(
        self,
        imports: str,
        theorem_stmt: str,
        original_state: str,
        augmented_state: str,
    ) -> Optional[str]:
        """
        Turn augmented tactic state into header + theorem using GPT; validate parse.
        Returns None on any failure.
        """
        llm = self._get_gpt_state_llm()
        if llm is None:
            return None
        if not original_state.strip() or not augmented_state.strip():
            return None
        fmt = self._state_to_stmt_formatter
        prompt = fmt.format(
            original_statement=theorem_stmt.strip(),
            original_state=original_state.strip(),
            augmented_state=augmented_state.strip(),
        )
        try:
            raw = llm.generate(prompt)
            lean = fmt.extract_answer(raw)
        except Exception:
            return None
        lean = lean.strip()
        if not lean.startswith("theorem"):
            return None
        full = imports.rstrip("\n") + "\n" + lean + "\n"
        try:
            split_declarations_theorem_proof(full)
        except ValueError:
            return None
        return full

    def set_ensemble_size(self, ensemble_size: int) -> None:
        """Set ensemble size at runtime (must be >= 1)."""
        self._ensemble_size = max(1, int(ensemble_size))

    def get_ensemble_size(self) -> int:
        """Current ensemble size."""
        return int(self._ensemble_size)

    @staticmethod
    def _allocate_attempts_uniform(total_attempts: int, num_buckets: int) -> List[int]:
        if total_attempts <= 0 or num_buckets <= 0:
            return [0 for _ in range(max(0, num_buckets))]
        base = total_attempts // num_buckets
        rem = total_attempts % num_buckets
        return [base + (1 if i < rem else 0) for i in range(num_buckets)]

    def _score_full_statement_energy(
        self,
        full_statement: str,
        instant_rewrite_solved: bool,
    ) -> Optional[float]:
        """
        Evaluate the rewriting module's energy heuristic on the full candidate Lean text.

        Returns None when there is no energy configured, when the candidate is an instant
        rewrite proof (not scored like ``get_state_candidates``), or on failure.
        """
        if instant_rewrite_solved:
            return None
        module = self._canonicalization_module
        energy_fn = getattr(module, "_energy_heuristic", None)
        if energy_fn is None:
            return None
        llm = getattr(self._single_pass_prover, "llm", None)
        try:
            return float(energy_fn(llm, full_statement))
        except Exception:
            return None

    def _build_ensemble_candidates(
        self,
        imports: str,
        theorem_stmt: str,
        *,
        shuffle: bool = True,
    ) -> List[Tuple[str, List[str], bool, Optional[float]]]:
        """
        Return up to K candidates as
        (problem_statement, rw_tactics, instant_rewrite_solved, energy_score).

        ``energy_score`` is the configured energy heuristic on ``problem_statement`` when
        available (mirrors scoring semantics used to sort in ``get_state_candidates``).

        ``instant_rewrite_solved`` is True when rewrites closed the goal ("no goals"), matching
        the path that skips the LLM during ``generate_batch``.
        Augmented states use ``convert_no_compile`` or optional GPT (see ``state_to_statement_mode``).
        """
        module = self._canonicalization_module
        theorem_name = extract_theorem_name(theorem_stmt)
        module.reset(imports, theorem_stmt)

        original_stmt = imports.rstrip("\n") + "\n" + theorem_stmt.strip("\n")
        candidates: List[Tuple[str, List[str], bool]] = []
        debug = module.get_state_candidates(sorted_by_energy=True)
        if debug is not None:
            current_state, state_candidates = debug
            for state_str, rw_list in state_candidates:
                if state_str and state_str.strip() == "no goals":
                    proof_body = "  " + "\n  ".join(rw_list) + "\n"
                    combined = (
                        imports.rstrip("\n")
                        + "\n"
                        + theorem_stmt.rstrip("\n")
                        + "\n"
                        + proof_body
                    )
                    candidates.append((combined, rw_list, True))
                    continue
                stmt: Optional[str] = None
                if self._state_to_statement_mode == "gpt":
                    stmt = self._augmented_state_to_problem_statement(
                        imports, theorem_stmt, current_state, state_str
                    )
                if stmt is None:
                    stmt = self._converter.convert_no_compile(imports, state_str, theorem_name)
                if stmt is not None:
                    candidates.append((stmt, rw_list, False))

        # Deduplicate by statement text while keeping first seen rw_tactics / instant flag.
        by_stmt: dict[str, Tuple[List[str], bool]] = {}
        for stmt, rw, instant in candidates:
            if stmt not in by_stmt:
                by_stmt[stmt] = (rw, instant)
        deduped = [(stmt, rw, inst) for stmt, (rw, inst) in by_stmt.items()]

        # Always include original statement within the variant budget K.
        selected: List[Tuple[str, List[str], bool]] = []
        if self._ensemble_size > 0:
            selected.append((original_stmt, [], False))
        remaining_budget = max(0, self._ensemble_size - len(selected))

        # Remove original from augmentation pool if present. At this point deduped
        # order is already energy-sorted (best first) when module supports it.
        aug_pool = [(stmt, rw, inst) for stmt, rw, inst in deduped if stmt != original_stmt]
        if remaining_budget > 0 and aug_pool:
            selected.extend(aug_pool[:remaining_budget])

        # Final fallback when K=0-like edge cases: ensure at least original is available.
        if not selected:
            selected = [(original_stmt, [], False)]

        if shuffle:
            self._rng.shuffle(selected)

        scored: List[Tuple[str, List[str], bool, Optional[float]]] = []
        for stmt, rw, inst in selected:
            scored.append(
                (stmt, rw, inst, self._score_full_statement_energy(stmt, inst))
            )
        return scored

    def list_inference_candidates(
        self,
        problem_statement: str,
        *,
        shuffle: bool = False,
    ) -> List[Tuple[str, List[str], bool, Optional[float]]]:
        """
        Enumerate ensemble candidates for ``problem_statement`` (header + formal theorem).

        Same candidate construction as inference-time ``generate_batch``, but with optional
        shuffle disabled for stable dataset export (original first, then energy-ordered augments).

        Each tuple includes ``energy_score`` when the rewriting module has an energy heuristic
        (e.g. theorem_surprise); None if unconfigured, instant-rewrite, or scoring failed.
        """
        try:
            imports, theorem_stmt, _ = split_declarations_theorem_proof(problem_statement)
        except ValueError:
            return []
        return self._build_ensemble_candidates(imports, theorem_stmt, shuffle=shuffle)

    @staticmethod
    def original_statement_plus_rw_proof(
        imports: str,
        theorem_stmt: str,
        rw_tactics: List[str],
    ) -> str:
        """
        Full Lean file for the *original* theorem with proof body = only rewrite tactics (two-space indent).
        Same shape as the instant-rewrite branch in ``_build_ensemble_candidates`` (``no goals`` states).
        """
        proof_body = "  " + "\n  ".join(rw_tactics) + "\n"
        return (
            imports.rstrip("\n")
            + "\n"
            + theorem_stmt.rstrip("\n")
            + "\n"
            + proof_body
        )

    @staticmethod
    def _compose_to_original(
        imports: str,
        theorem_stmt: str,
        rw_tactics: List[str],
        generated_code: str,
    ) -> str:
        if not rw_tactics:
            return generated_code
        try:
            _, _, proof_body = split_declarations_theorem_proof(generated_code)
        except ValueError:
            return EnsembleLLMProver.original_statement_plus_rw_proof(
                imports, theorem_stmt, rw_tactics
            )
        pb = (proof_body or "").strip()
        if not pb:
            return EnsembleLLMProver.original_statement_plus_rw_proof(
                imports, theorem_stmt, rw_tactics
            )
        rw_block = "  " + "\n  ".join(rw_tactics) + "\n"
        return imports.rstrip("\n") + "\n" + theorem_stmt + rw_block + proof_body

    def generate(self, problem_statement: str) -> Optional[str]:
        generated = self.generate_batch(
            problem_statement=problem_statement,
            n_attempts=-1,
            batch_size=1,
        )
        if isinstance(generated, list):
            return generated[0] if generated else None
        return generated

    def generate_batch(
        self,
        problem_statement: str,
        n_attempts: int,
        batch_size: int = 1,
    ) -> Optional[str] | List[Optional[str]]:
        single_mode = n_attempts <= 0
        attempts = 1 if single_mode else n_attempts

        try:
            imports, theorem_stmt, _ = split_declarations_theorem_proof(problem_statement)
        except ValueError:
            return None if single_mode else [None for _ in range(attempts)]

        candidates = self._build_ensemble_candidates(imports, theorem_stmt, shuffle=True)
        if not candidates:
            return None if single_mode else [None for _ in range(attempts)]

        alloc = self._allocate_attempts_uniform(attempts, len(candidates))
        outputs: List[Optional[str]] = []

        for (candidate_problem, rw_tactics, _instant, _energy), n_for_candidate in zip(
            candidates, alloc
        ):
            if n_for_candidate <= 0:
                continue
            # If candidate already has a proof body (e.g. "no goals" from try-tactic
            # rewrites), do not call the LLM; reuse it directly.
            try:
                _d, _t, candidate_proof_body = split_declarations_theorem_proof(
                    candidate_problem
                )
            except ValueError:
                candidate_proof_body = ""
            if candidate_proof_body.strip():
                outputs.extend([candidate_problem] * n_for_candidate)
                continue

            single_pass = self._single_pass_prover
            if hasattr(single_pass, "generate_batch"):
                gen_list = single_pass.generate_batch(
                    candidate_problem,
                    n_attempts=n_for_candidate,
                    batch_size=batch_size,
                )
            else:
                gen_list = [single_pass.generate(candidate_problem) for _ in range(n_for_candidate)]
            for g in gen_list:
                if g is None:
                    outputs.append(None)
                    continue
                outputs.append(self._compose_to_original(imports, theorem_stmt, rw_tactics, g))

        # Keep output size exactly attempts.
        if len(outputs) < attempts:
            outputs.extend([None] * (attempts - len(outputs)))
        outputs = outputs[:attempts]
        return outputs[0] if single_mode else outputs

    def prove(self, problem_statement: str) -> dict:
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

    def prove_batch(
        self,
        problem_statement: str,
        n_attempts: int,
        batch_size: int = 1,
    ) -> List[dict]:
        if n_attempts <= 0:
            return []
        generated = self.generate_batch(
            problem_statement=problem_statement,
            n_attempts=n_attempts,
            batch_size=batch_size,
        )
        if not isinstance(generated, list):
            generated = [generated]
        out: List[dict] = []
        for code in generated:
            if code is None:
                out.append(
                    {
                        "success": False,
                        "final_code": problem_statement,
                        "error": "Generation failed",
                        "steps": [],
                    }
                )
                continue
            success, error = self._verifier.verify(code)
            out.append(
                {
                    "success": success,
                    "final_code": code,
                    "error": None if success else error,
                    "steps": [],
                }
            )
        return out


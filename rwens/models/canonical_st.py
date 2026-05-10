"""
Canonical step-by-step formal proving model.

Step-by-step prover that uses a CanonicalizationModule to present augmented
(canonical) states to the LLM. Two modes:
- full_aug=False: Augment only the initial state. The module may add proof
  lines so subsequent states stay in canonical form.
- full_aug=True: Re-augment at each step so the LLM always sees the
  canonical form of the current state.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from rwens.canonicalization.base import CanonicalizationModule
from rwens.dataset.utils import split_declarations_theorem_proof
from rwens.models.base import FormalProver
from rwens.models.llm.base import BaseLLM
from rwens.utils.applier import StateFetchAbort
from rwens.utils.metrics import diagnostics_indicate_success, get_tactic_failure_diagnostic

logger = logging.getLogger(__name__)


class CanonicalProver(FormalProver):
    """
    Step-by-step prover that uses canonical (augmented) states for the LLM.

    The LLM is shown augmented states from the canonicalization module. This prover:
    1. Uses the module to get augmented states
    2. Feeds augmented state to the LLM
    3. Applies tactics via the module
    4. When full_aug=True, re-augments at each step so the LLM always sees
       canonical form; when full_aug=False, augments only at the start.
    """

    def __init__(
        self,
        llm: BaseLLM,
        canonicalization_module: CanonicalizationModule,
        full_aug: bool = False,
        max_iterations: int = 100,
    ):
        """
        Initialize the canonical prover.

        Args:
            llm: The language model (trained on augmented states)
            canonicalization_module: Canonicalization module instance.
                Reused for each prove() call; reset() is called with the
                problem's imports and theorem at the start.
            full_aug: If False, augment only the initial state. If True,
                augment at each step.
            max_iterations: Maximum tactic steps before stopping
        """
        self.llm = llm
        self.canonicalization_module = canonicalization_module
        self.project_root = Path(canonicalization_module.project_root).resolve()
        self.full_aug = full_aug
        self.max_iterations = max_iterations

        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")
        if not (self.project_root / "lakefile.lean").exists() and not (
            self.project_root / "lakefile.toml"
        ).exists():
            raise ValueError(
                f"Project root must contain lakefile.lean or lakefile.toml: {self.project_root}"
            )

    def _extract_hypothesis_name_from_have(self, tactic: str) -> Optional[str]:
        tactic = tactic.strip()
        if not tactic.startswith("have"):
            return None
        match = re.match(r"have\s+([^\s:]+)", tactic)
        return match.group(1) if match else None

    def _find_available_hypothesis_name(self, known_hypotheses: set[str]) -> str:
        counter = 0
        while True:
            candidate = f"h{counter}"
            if candidate not in known_hypotheses:
                return candidate
            counter += 1

    def _rename_hypothesis_in_tactic(
        self, tactic: str, old_name: str, new_name: str
    ) -> str:
        pattern = rf"\bhave\s+{re.escape(old_name)}\b"
        return re.sub(pattern, f"have {new_name}", tactic, count=1)

    def _extract_hypothesis_names_from_state(self, state: str) -> set[str]:
        if not state or not state.strip():
            return set()
        names = set()
        for line in state.split("\n"):
            line = line.strip()
            if not line or line.startswith("⊢"):
                continue
            colon_pos = line.find(":")
            if colon_pos != -1:
                hyp = line[:colon_pos].strip().strip("()")
                if hyp:
                    names.add(hyp)
        return names

    def generate(self, problem_statement: str) -> Optional[str]:
        result = self.prove(problem_statement)
        return result["final_code"] if result["success"] else None

    def prove(self, problem_statement: str) -> dict:
        """
        Attempt to prove a problem statement using augmented states.

        Returns:
            Dictionary with success, steps, final_code, error.
        """
        steps = []
        decls, theorem_stmt, _ = split_declarations_theorem_proof(problem_statement)
        imports = decls

        canonicalizer = self.canonicalization_module
        canonicalizer.reset(imports, theorem_stmt)

        try:

            try:
                _, augmented_state, *_ = canonicalizer.get_states(
                    keep_augmentation=True
                )
            except StateFetchAbort as e:
                return {
                    "success": False,
                    "steps": [],
                    "final_code": decls + theorem_stmt,
                    "error": str(e),
                }
            if augmented_state is None or not augmented_state.strip():
                return {
                    "success": False,
                    "steps": [],
                    "final_code": canonicalizer.get_file_content(),
                    "error": "Could not get augmented state",
                }
            state_for_llm = augmented_state
            known_hypotheses = self._extract_hypothesis_names_from_state(
                augmented_state
            )

            for iteration in range(self.max_iterations):
                # Generate tactic from LLM
                try:
                    predicted_tactic = self.llm.generate(state_for_llm)
                except Exception as e:
                    return {
                        "success": False,
                        "steps": steps,
                        "final_code": canonicalizer.get_file_content(),
                        "error": f"Error generating tactic: {e}",
                    }

                tactic_to_apply = predicted_tactic

                # Resolve have-name conflicts
                hyp_name = self._extract_hypothesis_name_from_have(tactic_to_apply)
                if hyp_name and hyp_name in known_hypotheses:
                    new_name = self._find_available_hypothesis_name(known_hypotheses)
                    tactic_to_apply = self._rename_hypothesis_in_tactic(
                        tactic_to_apply, hyp_name, new_name
                    )
                    hyp_name = new_name
                if hyp_name:
                    known_hypotheses.add(hyp_name)

                state_before = state_for_llm
                state_after = canonicalizer.apply_tactic(tactic_to_apply)

                if state_after is None:
                    return {
                        "success": False,
                        "steps": steps,
                        "final_code": canonicalizer.get_file_content(),
                        "error": "Could not get state after tactic",
                    }

                # If state unchanged, run diagnostics; tactic failure phrases indicate
                # incorrectly applied tactic
                if state_before == state_after:
                    diag = canonicalizer.get_diagnostics()
                    failure_d = get_tactic_failure_diagnostic(diag)
                    if failure_d is not None:
                        return {
                            "success": False,
                            "steps": steps,
                            "final_code": canonicalizer.get_file_content(),
                            "error": f"Tactic applied but state unchanged; diagnostic: {failure_d}",
                        }

                steps.append({
                    "iteration": iteration + 1,
                    "tactic": tactic_to_apply,
                    "state_before": state_before,
                    "state_after": state_after,
                })

                if state_after == "no goals":
                    diag = canonicalizer.get_diagnostics()
                    success = diagnostics_indicate_success(diag)
                    return {
                        "success": success,
                        "steps": steps,
                        "final_code": canonicalizer.get_file_content(),
                        "error": None if success else diag.diagnostics,
                    }

                if self.full_aug:
                    canonicalizer.invalidate_state_cache()
                    try:
                        _, augmented_state, *_ = canonicalizer.get_states(
                            keep_augmentation=True
                        )
                    except StateFetchAbort as e:
                        return {
                            "success": False,
                            "steps": steps,
                            "final_code": canonicalizer.get_file_content(),
                            "error": str(e),
                        }
                    if augmented_state is None or not augmented_state.strip():
                        return {
                            "success": False,
                            "steps": steps,
                            "final_code": canonicalizer.get_file_content(),
                            "error": "Could not get augmented state after tactic",
                        }
                    state_for_llm = augmented_state
                else:
                    state_for_llm = state_after
                known_hypotheses = self._extract_hypothesis_names_from_state(state_for_llm)

            return {
                "success": False,
                "steps": steps,
                "final_code": canonicalizer.get_file_content(),
                "error": f"Max iterations ({self.max_iterations}) reached",
            }

        except Exception as e:
            return {
                "success": False,
                "steps": steps,
                "final_code": decls + theorem_stmt,
                "error": f"Error during proving: {e}",
            }

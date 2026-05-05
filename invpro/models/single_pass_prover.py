"""
Single-pass prover: LLM generates the full proof in one call; Lean verifies via LSP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from invpro.models.base import FormalProver
from invpro.models.llm.base import BaseLLM
from invpro.models.utils import merge_proof_into_statement
from invpro.prompt.base import PromptFormatter
from invpro.utils.verifier import ProofVerifier

logger = logging.getLogger(__name__)


class SinglePassProver(FormalProver):
    """Prover: format problem -> LLM.generate -> extract Lean code -> merge -> verify with Lean LSP."""

    def __init__(
        self,
        llm: BaseLLM,
        prompt_formatter: PromptFormatter,
        project_root: str,
        timeout_seconds: float = 120.0,
        initial_imports: str = "import Mathlib\n",
    ):
        self.llm = llm
        self.prompt_formatter = prompt_formatter
        self._project_root = Path(project_root).resolve()
        self._verifier = ProofVerifier(
            project_root=str(self._project_root),
            initial_imports=initial_imports,
            timeout_seconds=timeout_seconds,
        )

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _format_prompt(self, problem_statement: str) -> str:
        """Format problem statement into the prompt sent to the LLM. Raises on failure."""
        return self.prompt_formatter.format(problem_statement)

    def _merge_proof(self, problem_statement: str, proof_body: str) -> str:
        """Merge extracted proof body into the problem statement to form full Lean code."""
        return merge_proof_into_statement(problem_statement, proof_body)

    def _verify(self, full_code: str) -> tuple[bool, Optional[str]]:
        """Verify full Lean code with the verifier. Returns (success, error_message)."""
        return self._verifier.verify(full_code)

    def generate(self, problem_statement: str) -> Optional[str]:
        """
        Generate full Lean code without verification.

        Flow: format prompt -> LLM.generate -> extract answer -> merge into
        problem statement. Returns merged code or None on generation/extraction
        failure.
        """
        try:
            prompt = self._format_prompt(problem_statement)
            response = self.llm.generate(prompt)
            extracted = self.prompt_formatter.extract_answer(response)
            if not extracted or not extracted.strip():
                return None
            return self._merge_proof(problem_statement, extracted)
        except Exception:
            return None

    def prove(self, problem_statement: str) -> dict:
        """Generate once, then verify. Returns {success, final_code, error, steps}."""
        full_code = self.generate(problem_statement)
        if full_code is None:
            return {
                "success": False,
                "final_code": problem_statement,
                "error": "Generation failed",
                "steps": [],
            }

        success, error = self._verify(full_code)

        return {
            "success": success,
            "final_code": full_code,
            "error": None if success else error,
            "steps": [],
        }

    def generate_batch(
        self,
        problem_statement: str,
        n_attempts: int,
        batch_size: int = 1,
    ) -> List[Optional[str]]:
        """
        Generate n_attempts full Lean code candidates without verification.
        Returns one entry per attempt (code or None).
        """
        try:
            prompt = self._format_prompt(problem_statement)
        except Exception:
            return [None for _ in range(n_attempts)]

        responses: List[str] = []
        try:
            if hasattr(self.llm, "generate_batch") and batch_size > 1:
                for start in range(0, n_attempts, batch_size):
                    k = min(batch_size, n_attempts - start)
                    responses.extend(self.llm.generate_batch(prompt, k))
            else:
                for _ in range(n_attempts):
                    responses.append(self.llm.generate(prompt))
        except Exception:
            return [None for _ in range(n_attempts)]

        out: List[Optional[str]] = []
        for response in responses:
            extracted = self.prompt_formatter.extract_answer(response)
            if not extracted or not extracted.strip():
                out.append(None)
                continue
            out.append(self._merge_proof(problem_statement, extracted))

        if len(out) < n_attempts:
            out.extend([None for _ in range(n_attempts - len(out))])
        return out[:n_attempts]

    def generate_batch_with_metadata(
        self,
        problem_statement: str,
        n_attempts: int,
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate n_attempts full Lean code candidates without verification, returning
        per-attempt metadata including full raw-response token counts when available.
        """
        try:
            prompt = self._format_prompt(problem_statement)
        except Exception:
            return [{
                "final_code": None,
                "raw_response_num_tokens": None,
                "error": "Prompt formatting failed",
            } for _ in range(n_attempts)]

        responses: List[Dict[str, Any]] = []
        try:
            if hasattr(self.llm, "generate_batch_with_metadata") and batch_size > 1:
                for start in range(0, n_attempts, batch_size):
                    k = min(batch_size, n_attempts - start)
                    chunk = self.llm.generate_batch_with_metadata(prompt, k)
                    for row in chunk:
                        responses.append({
                            "text": row.get("text", ""),
                            "raw_response_num_tokens": row.get("num_generated_tokens"),
                        })
            elif hasattr(self.llm, "generate_with_metadata"):
                for _ in range(n_attempts):
                    row = self.llm.generate_with_metadata(prompt)
                    responses.append({
                        "text": row.get("text", ""),
                        "raw_response_num_tokens": row.get("num_generated_tokens"),
                    })
            elif hasattr(self.llm, "generate_batch") and batch_size > 1:
                for start in range(0, n_attempts, batch_size):
                    k = min(batch_size, n_attempts - start)
                    chunk = self.llm.generate_batch(prompt, k)
                    for text in chunk:
                        responses.append({"text": text, "raw_response_num_tokens": None})
            else:
                for _ in range(n_attempts):
                    responses.append({
                        "text": self.llm.generate(prompt),
                        "raw_response_num_tokens": None,
                    })
        except Exception as e:
            return [{
                "final_code": None,
                "raw_response_num_tokens": None,
                "error": str(e),
            } for _ in range(n_attempts)]

        out: List[Dict[str, Any]] = []
        for row in responses:
            response_text = row.get("text", "")
            extracted = self.prompt_formatter.extract_answer(response_text)
            if not extracted or not extracted.strip():
                out.append({
                    "final_code": None,
                    "raw_response_num_tokens": row.get("raw_response_num_tokens"),
                    "error": "Extraction failed",
                })
                continue
            out.append({
                "final_code": self._merge_proof(problem_statement, extracted),
                "raw_response_num_tokens": row.get("raw_response_num_tokens"),
                "error": None,
            })

        if len(out) < n_attempts:
            out.extend([{
                "final_code": None,
                "raw_response_num_tokens": None,
                "error": "Generation failed",
            } for _ in range(n_attempts - len(out))])
        return out[:n_attempts]

    def prove_batch(
        self,
        problem_statement: str,
        n_attempts: int,
        batch_size: int = 1,
    ) -> List[dict]:
        """
        Run n_attempts proof attempts with batched LLM generation when supported.

        Flow: format prompt once -> batch generate (or sequential) -> extract & merge
        each response -> verify each full code one-by-one. Returns a list of result
        dicts (same shape as prove(): success, final_code, error, steps, timings).
        """
        generated_codes = self.generate_batch(
            problem_statement=problem_statement,
            n_attempts=n_attempts,
            batch_size=batch_size,
        )
        results: List[dict] = []
        for full_code in generated_codes:
            if full_code is None:
                results.append({
                    "success": False,
                    "final_code": problem_statement,
                    "error": "Generation failed",
                    "steps": [],
                })
                continue
            success, error = self._verify(full_code)
            results.append({
                "success": success,
                "final_code": full_code,
                "error": None if success else error,
                "steps": [],
            })
        return results

"""
Naive step-by-step formal proving model.

This module implements a "naive" version of a step-by-step formal proving model
that, given a formal problem statement in Lean4, will:
1. Convert it into the corresponding state
2. Process the state using an LLM to generate the next tactic
3. Apply the tactic (virtually, by adding it to a file)
4. Get the next state
5. Reiterate until the proof is finished
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import leanclient as lc

from rwens.dataset.utils import extract_state_text
from rwens.models.base import FormalProver
from rwens.models.llm.base import BaseLLM
from rwens.models.conf.types import GOAL_TIMEOUT_SECONDS
from rwens.multithreading.utils import run_with_timeout
from rwens.utils.applier import TacticApplier
from rwens.utils.metrics import diagnostics_indicate_success, get_tactic_failure_diagnostic

logger = logging.getLogger(__name__)


class NaiveStepByStepProver(FormalProver):
    """
    A naive step-by-step formal proving model that uses an LLM to generate tactics.
    
    This prover works by:
    1. Creating a temporary file with the problem statement
    2. Extracting the initial state (after `:= by`)
    3. Using an LLM to predict the next tactic
    4. Appending the tactic to the file
    5. Extracting the new state
    6. Repeating until the proof is complete (no goals left) or max iterations reached
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        project_root: Path,
        max_iterations: int = 100,
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
    ):
        """
        Initialize the naive step-by-step prover.
        
        Args:
            llm: The language model to use for tactic prediction
            project_root: Root of the Lean project (where lakefile.lean lives)
            max_iterations: Maximum number of tactic steps before stopping
            timeout_seconds: Timeout for get_goal calls
        """
        self.llm = llm
        self.project_root = Path(project_root).resolve()
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        
        # Verify project root exists and contains lakefile.lean or lakefile.toml
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")
        if not (self.project_root / "lakefile.lean").exists() and not (self.project_root / "lakefile.toml").exists():
            raise ValueError(
                f"Project root does not contain lakefile.lean or lakefile.toml: {self.project_root}"
            )

        # Create a single persistent temp file and file client so Mathlib is not recompiled each call
        initial_content = "import Mathlib\n\n"
        file_id = str(uuid.uuid4())[:8]
        self._temp_file = self.project_root / f"_temp_proof_{file_id}.lean"
        self._temp_file.write_text(initial_content, encoding="utf-8")
        self._client = lc.LeanLSPClient(str(self.project_root), prevent_cache_get=True)
        self._rel_path = self._temp_file.relative_to(self.project_root).as_posix()
        self._client.open_file(self._rel_path)
        self.sfc = self._client.create_file_client(self._rel_path)
        # Ensure LSP buffer matches: replace full content with initial
        lines = initial_content.split("\n")
        change = lc.DocumentContentChange(
            text=initial_content,
            start=[0, 0],
            end=[max(0, len(lines) - 1), len(lines[-1]) if lines else 0],
        )
        self.sfc.update_file(changes=[change])
    
    def _recreate_file_client(self):
        """Recreate the temp file and file client if they become broken."""
        initial_content = "import Mathlib\n\n"
        # Ensure the file exists and has correct content
        if not self._temp_file.exists():
            logger.debug(f"Temp file {self._temp_file} does not exist, recreating")
        self._temp_file.write_text(initial_content, encoding="utf-8")
        # Recreate the client (close old one if possible)
        try:
            if hasattr(self._client, 'close'):
                self._client.close()
        except Exception:
            pass
        self._client = lc.LeanLSPClient(str(self.project_root), prevent_cache_get=True)
        self._rel_path = self._temp_file.relative_to(self.project_root).as_posix()
        self._client.open_file(self._rel_path)
        self.sfc = self._client.create_file_client(self._rel_path)
        # Ensure LSP buffer matches
        lines = initial_content.split("\n")
        change = lc.DocumentContentChange(
            text=initial_content,
            start=[0, 0],
            end=[max(0, len(lines) - 1), len(lines[-1]) if lines else 0],
        )
        self.sfc.update_file(changes=[change])
    
    def _strip_import_mathlib(self, problem_statement: str) -> str:
        """Remove a leading 'import Mathlib' line so it is not duplicated (we keep it in the file)."""
        stripped = problem_statement.lstrip()
        if stripped.startswith("import Mathlib"):
            rest = stripped[len("import Mathlib") :].lstrip("\n\r")
            return rest
        return problem_statement
    
    def _create_temp_file(self, problem_statement: str) -> Path:
        """
        Create a temporary file with the problem statement.
        
        Args:
            problem_statement: The Lean4 problem statement (should include imports, etc.)
            
        Returns:
            Path to the temporary file
        """
        # Generate a unique filename
        file_id = str(uuid.uuid4())[:8]
        temp_file = self.project_root / f"_temp_proof_{file_id}.lean"
        
        # Write the problem statement
        temp_file.write_text(problem_statement, encoding="utf-8")
        # logger.debug(f"Created temporary file: {temp_file}")
        
        return temp_file
    
    def _extract_hypothesis_names_from_state(self, state: str) -> set[str]:
        """
        Extract all hypothesis names from a proof state.
        
        Extracts names by checking the beginning of each non-goal line until the first ":".
        Goal lines start with "⊢".
        
        Args:
            state: The proof state string
            
        Returns:
            Set of hypothesis names found in the state
        """
        if not state or not state.strip():
            return set()
        
        hypothesis_names = set()
        lines = state.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip goal lines (they start with "⊢")
            if line.startswith("⊢"):
                continue
            
            # Extract hypothesis name: everything before the first ":"
            colon_pos = line.find(':')
            if colon_pos != -1:
                hyp_name = line[:colon_pos].strip()
                # Remove any leading/trailing whitespace or parentheses
                hyp_name = hyp_name.strip('()')
                if hyp_name:
                    hypothesis_names.add(hyp_name)
        
        return hypothesis_names
    
    def _extract_hypothesis_name_from_have(self, tactic: str) -> Optional[str]:
        """
        Extract the hypothesis name from a 'have' tactic.
        
        Args:
            tactic: The tactic string (should start with 'have')
            
        Returns:
            The hypothesis name (e.g., 'h7') or None if not found
        """
        tactic = tactic.strip()
        if not tactic.startswith("have"):
            return None
        
        # Match "have" followed by whitespace, then capture everything until space or ":"
        # This allows for Unicode characters in hypothesis names (e.g., h₂, h₀)
        match = re.match(r"have\s+([^\s:]+)", tactic)
        if match:
            return match.group(1)
        return None
    
    def _find_available_hypothesis_name(self, known_hypotheses: set[str]) -> str:
        """
        Find the first available hypothesis name in the format "hY" where Y is a number.
        
        Args:
            known_hypotheses: Set of known hypothesis names
            
        Returns:
            First available name like "h0", "h1", "h2", etc.
        """
        counter = 0
        while True:
            candidate = f"h{counter}"
            if candidate not in known_hypotheses:
                return candidate
            counter += 1
    
    def _rename_hypothesis_in_tactic(self, tactic: str, old_name: str, new_name: str) -> str:
        """
        Rename a hypothesis in a tactic string.
        
        Args:
            tactic: The tactic string
            old_name: The old hypothesis name
            new_name: The new hypothesis name
            
        Returns:
            The tactic string with the name replaced
        """
        # Replace the first occurrence of "have old_name" with "have new_name"
        # We need to be careful to match only at word boundaries
        pattern = rf"\bhave\s+{re.escape(old_name)}\b"
        replacement = f"have {new_name}"
        return re.sub(pattern, replacement, tactic, count=1)
    
    def generate(self, problem_statement: str) -> Optional[str]:
        """
        Generate a proof for the given problem statement.
        
        This method implements the FormalProver interface.
        
        Args:
            problem_statement: The formal problem statement (e.g., Lean4 code ending with `:= by`)
            
        Returns:
            The generated proof as a string (full code with tactics applied), or None if the proof generation failed
        """
        result = self.prove(problem_statement)
        if result["success"]:
            return result["final_code"]
        return None


    def prove(
        self,
        problem_statement: str,
    ) -> dict:
        """
        Attempt to prove a problem statement step-by-step.
        
        Args:
            problem_statement: The Lean4 problem statement (should end with `:= by` or similar)
            
        Returns:
            Dictionary with keys:
                - success: bool, whether the proof was completed
                - steps: list of dicts with 'tactic', 'state_before', 'state_after'
                - final_code: str, the final code with all tactics applied
                - error: Optional[str], error message if failed
        """
        # Initialize return values
        steps = []
        current_code = problem_statement

        # Replace file content from line 1 with problem body (keep "import Mathlib" on line 0)
        body = self._strip_import_mathlib(problem_statement)
        if body and not body.endswith("\n"):
            body = body + "\n"
        
        # Try to get file content and update; if it fails, recreate the file/client
        try:
            current_content = self.sfc.get_file_content()
            lines = current_content.split("\n")
            end_line = len(lines) - 1
            end_char = len(lines[-1]) if lines else 0
            change = lc.DocumentContentChange(
                text=body,
                start=[1, 0],
                end=[end_line, end_char],
            )
            self.sfc.update_file(changes=[change])
        except Exception as e:
            # File/client might be broken - try to recreate it
            logger.warning(f"File client operation failed, recreating: {e}")
            try:
                self._recreate_file_client()
                # Retry the update
                current_content = self.sfc.get_file_content()
                lines = current_content.split("\n")
                end_line = len(lines) - 1
                end_char = len(lines[-1]) if lines else 0
                change = lc.DocumentContentChange(
                    text=body,
                    start=[1, 0],
                    end=[end_line, end_char],
                )
                self.sfc.update_file(changes=[change])
            except Exception as e2:
                # If recreation also fails, return error
                return {
                    "success": False,
                    "steps": [],
                    "final_code": current_code,
                    "error": f"Failed to access file client: {e2}",
                }

        try:
            # Initialize TacticApplier (reuse self.sfc)
            applier = TacticApplier(
                file_client=self.sfc,
                main_goal_name=None,  # Will be extracted from code automatically
                timeout_seconds=self.timeout_seconds,
            )
            
            # Get initial state (at the start of the proof)
            initial_state = applier.get_state()
            if initial_state is None:
                return {
                    "success": False,
                    "steps": [],
                    "final_code": current_code,
                    "error": "Could not get initial state",
                }
            
            # Extract all known hypothesis names from the initial state
            known_hypotheses = self._extract_hypothesis_names_from_state(initial_state)
            # logger.debug(f"Initial known hypotheses: {known_hypotheses}")
            
            # Track steps
            current_state = initial_state
            
            # Iterate until proof is complete or max iterations reached
            for iteration in range(self.max_iterations):
                # Check if proof is complete (no goals - state should be empty)
                if current_state == "no goals":
                    # logger.info(f"Proof completed in {iteration} steps")
                    diag = self.sfc.get_diagnostics()
                    success = diagnostics_indicate_success(diag)
                    return {
                        "success": success,
                        "steps": steps,
                        "final_code": applier.current_code,
                        "error": None if success else diag.diagnostics,
                    }
                
                # Generate next tactic using LLM
                # logger.debug(f"Iteration {iteration + 1}: Generating tactic for state (length: {len(current_state)})")
                try:
                    predicted_tactic = self.llm.generate(current_state)
                    # logger.debug(f"Predicted tactic: {predicted_tactic}")
                except Exception as e:
                    # logger.error(f"Error generating tactic: {e}")
                    return {
                        "success": False,
                        "steps": steps,
                        "final_code": applier.current_code,
                        "error": f"Error generating tactic: {e}",
                    }
                
                # print(("predicted_tactic")
                # print((predicted_tactic)
                # Store state before applying tactic
                state_before = current_state
                
                # Check if tactic starts with "have" and handle name conflicts
                hypothesis_name = self._extract_hypothesis_name_from_have(predicted_tactic)
                if hypothesis_name:
                    # Check if the name conflicts with known hypotheses
                    if hypothesis_name in known_hypotheses:
                        # Rename to avoid conflict
                        new_name = self._find_available_hypothesis_name(known_hypotheses)
                        predicted_tactic = self._rename_hypothesis_in_tactic(
                            predicted_tactic, 
                            hypothesis_name, 
                            new_name
                        )
                        # logger.debug(
                        #     f"Renamed hypothesis '{hypothesis_name}' to '{new_name}' "
                        #     f"to avoid conflict with known hypotheses"
                        # )
                        # Use the new name going forward
                        hypothesis_name = new_name
                    
                    # Add the (new) name to known hypotheses
                    known_hypotheses.add(hypothesis_name)

                # print(("hypothesis_name")
                # print((hypothesis_name)
                # print(("updated predicted_tactic")
                # print((predicted_tactic)
                # Apply tactic using TacticApplier
                # This handles indentation, file update, and state evaluation
                state_after, current_code = applier.update(predicted_tactic)
                
                
                # print(("state_after")
                # print((state_after)
                # print(("current_code")
                # print((current_code)
                # print(('\n\n\n')

                # Handle case where state extraction failed
                if state_after is None:
                    # logger.error("Could not get state after applying tactic")
                    return {
                        "success": False,
                        "steps": steps,
                        "final_code": applier.current_code,
                        "error": "Could not get state after applying tactic",
                    }

                # If state unchanged, run diagnostics; tactic failure phrases indicate
                # incorrectly applied tactic (e.g. omega, linarith, rewrite)
                if state_before == state_after:
                    diag = self.sfc.get_diagnostics()
                    failure_d = get_tactic_failure_diagnostic(diag)
                    if failure_d is not None:
                        return {
                            "success": False,
                            "steps": steps,
                            "final_code": applier.current_code,
                            "error": f"Tactic applied but state unchanged; diagnostic: {failure_d}",
                        }

                # Record step
                steps.append({
                    "iteration": iteration + 1,
                    "tactic": predicted_tactic,
                    "state_before": state_before,
                    "state_after": state_after,
                })
                
                # Update for next iteration
                current_state = state_after
            
            # After max iterations, check if proof was completed
            if current_state == "no goals":
                # logger.info(f"Proof completed in {iteration} steps")
                diag = self.sfc.get_diagnostics()
                success = diagnostics_indicate_success(diag)
                return {
                    "success": success,
                    "steps": steps,
                    "final_code": applier.current_code,
                    "error": None if success else diag.diagnostics,
                }
            
            # Max iterations reached without completing proof
            # logger.warning(f"Max iterations ({self.max_iterations}) reached")
            return {
                "success": False,
                "steps": steps,
                "final_code": applier.current_code,
                "error": f"Max iterations ({self.max_iterations}) reached",
            }
            
        except Exception as e:
            # logger.error(f"Error during proving: {e}", exc_info=True)
            return {
                "success": False,
                "steps": steps,
                "final_code": current_code,
                "error": f"Error during proving: {e}",
            }

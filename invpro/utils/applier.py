"""
Tactic application utilities for Lean proofs.

This module provides the TacticApplier class for applying tactics to Lean proofs
with proper indentation tracking based on hypothesis stacks.
"""

import logging
import re
from dataclasses import dataclass
from textwrap import indent
from typing import Optional, Union, Dict

from time import time

import leanclient as lc
from numpy.strings import startswith
from torch._dynamo.utils import state_dict_hook_names

from invpro.dataset.utils import extract_state_text
from invpro.multithreading.utils import run_with_timeout

logger = logging.getLogger(__name__)

# Timeout for each `get_goal` call (in seconds)
GOAL_TIMEOUT_SECONDS = 120.0


@dataclass
class HypothesisEntry:
    name: str
    indent_level: int
    is_case: bool
    is_case_split: bool = False
    is_bracket: bool = False  # True if this is a bracket/try situation (line ending with {)


class StateFetchAbort(Exception):
    """Raised when state fetch times out or errors; caller should skip to next problem."""

    def __init__(self, message: str, is_timeout: bool = False) -> None:
        super().__init__(message)
        self.is_timeout = is_timeout

class TacticApplier:
    """
    Class for applying tactics to Lean proofs with proper indentation tracking.
    
    Maintains a hypothesis stack to determine correct indentation levels
    and applies tactics via the Lean LSP client.
    """
    
    def __init__(
        self,
        file_client: lc.SingleFileClient,
        main_goal_name: Optional[str] = None,
        timeout_seconds: float = GOAL_TIMEOUT_SECONDS,
    ):
        """
        Initialize the TacticApplier.
        
        Args:
            file_client: The Lean LSP file client
            main_goal_name: Optional main goal name (extracted from code if not provided)
            timeout_seconds: Timeout for get_goal calls
        """
        self.file_client = file_client
        self.current_code = file_client.get_file_content()
        self.timeout_seconds = timeout_seconds
        
        # Initialize hypothesis stack by parsing existing code
        if main_goal_name is None:
            main_goal_name = self._get_main_goal_name(self.current_code)
        
        self.hypothesis_stack: list[HypothesisEntry] = []
        if main_goal_name:
            # Stack stores HypothesisEntry objects tracking open goals/cases
            self.hypothesis_stack.append(HypothesisEntry(main_goal_name, 0, False))
        
        # Track pending case branches from case split tactics (induction/cases)
        # Format: list of (case_line, indent_level) tuples
        self.pending_case_branches = []
        
        # Build hypothesis stack from existing code
        self._build_hypothesis_stack_from_code(reset_stack=False)
    
    def _calculate_indent_level(self, line: str) -> int:
        """
        Calculate the indentation level of a line (number of 2-space indents).
        
        Args:
            line: The line of code
            
        Returns:
            The indentation level (number of 2-space indents)
        """
        stripped = line.lstrip()
        if not stripped:
            return 0
        leading_spaces = len(line) - len(stripped)
        return leading_spaces // 2
    
    def _is_hypothesis_active(self, hypothesis_name: str, indent_level: int) -> bool:
        """
        Check if a hypothesis is still active (not yet proven) by checking the state
        at the indent level where its proof would be (indent_level + 1).
        
        If the state at indent_level + 1 shows "no goals", the hypothesis is proven
        and therefore not active.
        
        Args:
            hypothesis_name: The name of the hypothesis
            indent_level: The indentation level where the hypothesis was declared
            
        Returns:
            True if the hypothesis is still active (not proven), False if proven
        """
        # Check the state at the proof level (indent_level + 1)
        # If it shows "no goals", the hypothesis is proven (not active)
        proof_state = self._get_state_at_indent(indent_level + 1)
        if proof_state is None:
            # If we can't get the state, assume it's active to be safe
            return True
        
        proof_state_stripped = proof_state.strip()
        
        # If the proof state shows "no goals", the hypothesis is proven (not active)
        if proof_state_stripped == "no goals":
            return False
        
        # Otherwise, the hypothesis is still active
        return True
    
    def _build_hypothesis_stack_from_code(self, reset_stack: bool = True) -> Optional[str]:
        """
        Build the hypothesis stack by parsing existing code.
        
        This method analyzes the code to find all "have" statements and case tactics,
        then checks which ones are still active by examining the state at their
        proof levels (indent_level + 1). If the state shows "no goals", the hypothesis
        is proven and not active.
        """
        # Always refresh from the file client (this class may be reused across edits).
        self.current_code = self.file_client.get_file_content()

        if reset_stack:
            # Reset state derived from code.
            self.pending_case_branches = []
            self.hypothesis_stack = []

            main_goal_name = self._get_main_goal_name(self.current_code)
            if main_goal_name:
                self.hypothesis_stack.append(HypothesisEntry(main_goal_name, 0, False))

        lines = self.current_code.split("\n")
        
        # Track all potential hypotheses and cases found in the code
        # Format: (name, indent_level, is_case, line_num)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            
            indent_level = self._calculate_indent_level(line)
            
            while len(self.hypothesis_stack) > 1:
                top_entry = self.hypothesis_stack[-1]
                if top_entry.indent_level < indent_level:
                    break
                self.hypothesis_stack.pop()
            
            is_case_tactic = False
            # Check for case tactics (lines starting with "·")
            if stripped.startswith('·') or stripped.startswith('. ') or stripped == '.':
                case_name = f"bullet_{i}"
                self.hypothesis_stack.append(HypothesisEntry(case_name, indent_level, True))
                is_case_tactic = True
            if stripped.startswith("|"):
                # Case branches are created by induction/cases/match splits.
                # Extract the name if possible; otherwise use a stable fallback.
                m = re.match(r"\|\s*([a-zA-Z_][a-zA-Z0-9_']*)", stripped)
                case_name = m.group(1) if m else f"case_{i}"
                self.hypothesis_stack.append(HypothesisEntry(case_name, indent_level, True, True))

            # Check for bracket/try blocks (line ending with "{")
            if stripped.endswith("{"):
                bracket_name = f"bracket_{i}"
                self.hypothesis_stack.append(
                    HypothesisEntry(bracket_name, indent_level, False, False, True)
                )

            # Check for "have" statements (including "have :" without name)
            hypothesis_name = self._extract_hypothesis_name_from_have(stripped)
            if hypothesis_name is not None:
                offset = 1 if is_case_tactic else 0
                
                # Strip bullet prefix for suffices checks
                stripped_no_bullet = stripped.lstrip("·").lstrip(". ").lstrip()
                if stripped.startswith("suffices") and not stripped.endswith("by"):
                    i += 1
                    continue
                
                # Collect consecutive lines until we find one with ":="
                # Start with the current line
                have_lines = [line]
                j = i + 1
                # If ":=" is not in the current line, collect subsequent lines until we find ":="
                # But don't collect for suffices patterns (they don't use :=)
                if ":=" not in line and not stripped_no_bullet.startswith("suffices"):
                    while j < len(lines) and ":=" not in lines[j]:
                        have_lines.append(lines[j])
                        j += 1
                    # Include the line with ":=" if found
                    if j < len(lines):
                        have_lines.append(lines[j])
                        j += 1
                
                # If the last collected line ends with ":=", append the next line (proof term)
                # This handles: have H0: P := \n proof_term
                if have_lines and have_lines[-1].rstrip().endswith(":=") and j < len(lines):
                    have_lines.append(lines[j])
                    j += 1
                
                # Combine all lines and check if completed in same line
                combined_have = '\n'.join(have_lines)
                if not self._is_hypothesis_completed_in_same_line(combined_have):
                    self.hypothesis_stack.append(HypothesisEntry(hypothesis_name, indent_level + offset, False))
                
                # Skip the lines we've already processed
                i = j
            else:
                i += 1

        # Pop the hypotheses / cases that are already resolved
        # Remember about the methodology of case splits
        # Check each entry from top to bottom and remove resolved ones
        force_pop = False

        # Try to get the file diagnostics, to prevent the bug with get_goal timing out
        diag = self.file_client.get_diagnostics(inactivity_timeout=60.0)

        while len(self.hypothesis_stack) > 1:
            top_entry = self.hypothesis_stack[-1]
            
            # Check if the hypothesis/case is resolved by checking the state at its proof level
            # For initial stack building, we don't have a tactic parameter, so pass None
            is_resolved = self._check_hypothesis_resolved_at_indent(
                top_entry.indent_level,
                top_entry.name,
                top_entry.is_case,
                is_case_split=top_entry.is_case_split,
                tactic=None,
            )
            
            if is_resolved or force_pop:
                
                force_pop = False
                popped_entry = self.hypothesis_stack.pop()
                
                # If this is a case_split, also remove the next hypothesis/case (the parent)
                if (popped_entry.is_case_split and len(self.hypothesis_stack) > 1):
                    force_pop = True
            else:
                # If the top entry is not resolved, we can't pop anything below it
                break

        # Return the state at the current active goal (deepest non-resolved entry).
        if not self.hypothesis_stack:
            return self.get_state() or ""
        top = self.hypothesis_stack[-1]
        state = self._get_state_at_indent(top.indent_level + 1)
        return state or ""
    
    def _extract_hypothesis_name_from_have(self, tactic: str) -> Optional[str]:
        """
        Extract the hypothesis name from a 'have' tactic.
        
        Args:
            tactic: The tactic string (should start with 'have')
            
        Returns:
            The hypothesis name (e.g., 'h7'), 'this' if no name is provided (have : ...),
            or None if not a 'have' statement
        """
        tactic = tactic.strip()

        # Handle cases like
        # >>  norm_num [this]; replace this : 1 - √3 * tan α ≠ 0 := by
        # where we apply some tactics before proposing the next declaration
        tactic = tactic.split(";")[-1].strip()
        if tactic.startswith('·'):
            tactic = tactic[1:].strip()
        if tactic.startswith('. '):
            tactic = tactic[2:].strip()

        start_list = ["have", "suffices", "obtain", "replace"]
        
        bad_pattern = True
        for start in start_list:
            if tactic.startswith(start):
                bad_pattern = False
                break
        if bad_pattern:
            return None
        
        # Match "have" followed by whitespace, then capture everything until space or ":"
        match = re.match(r"have\s+([^\s:]+)", tactic)
        if match:
            return match.group(1)
        # Check for "have :" pattern (no name, just "have" followed by whitespace and ":")
        if re.match(r"have\s*:", tactic):
            return "this"
        
        match = re.match(r"suffices\s+([^\s:]+)", tactic)
        if match:
            return match.group(1)
        if re.match(r"suffices\s+", tactic):
            if " by " not in tactic and " by\n" not in tactic:
                return None 
            return "this"

        match = re.match(r"replace\s+([^\s:]+)", tactic)
        if match:
            return match.group(1)
        # Handle both "replace :" and "replace:" (with or without space)
        if re.match(r"replace\s*:", tactic):
            return "this"

        # Also match the "obtain" pattern, if the line ends with ":= ... by", as in
        # obtain ⟨z, hz⟩ : ∃ z : ℤ, x ^ 2 + x - z = 0 := by
        # or obtain ⟨k, hk₁, hk₂⟩ := show ∃k, k = b^(k - 2) ∧ k > 2 by
        # The pattern can be simple like "z" or complex like "⟨z, hz⟩"
        # The ":" after the pattern is optional
        obtain_match = re.match(r"obtain\s+(.+?)(?:\s+:\s+.*?)?\s+:=.*?by\s*$", tactic)
        if obtain_match:
            # Extract the pattern part (e.g., "⟨z, hz⟩" or "z")
            pattern_part = obtain_match.group(1).strip()
            return pattern_part
        return None
    
    def _get_main_goal_name(self, problem_statement: str) -> Optional[str]:
        """
        Extract the main goal name from the problem statement.
        
        Args:
            problem_statement: The Lean code with the problem statement
            
        Returns:
            The goal name (e.g., 'h2') or None if not found
        """
        patterns = [
            r"lemma\s+([a-zA-Z_][a-zA-Z0-9_']*)",
            r"theorem\s+([a-zA-Z_][a-zA-Z0-9_']*)",
            r"def\s+([a-zA-Z_][a-zA-Z0-9_']*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, problem_statement)
            if match:
                return match.group(1)
        return None
    
    def _get_state_at_indent(self, indent_level: int) -> Optional[str]:
        """
        Get the state at a specific indentation level by temporarily inserting an empty line.
        
        Args:
            indent_level: The indentation level to check
            
        Returns:
            The state text at that indent level, or None if extraction failed
        """
        # Calculate the position at the end of the current code
        lines = self.current_code.rstrip().split('\n')
        last_line = lines[-1] if lines else ""
        
        # Position is 0-indexed for line and character
        end_line = len(lines) - 1 if lines else 0
        end_char = len(last_line)
        
        # Create an empty line with the correct indentation
        indent = "  " * indent_level
        empty_line = indent + "\n"
        
        # Temporarily insert the empty line
        change = lc.DocumentContentChange(
            text=empty_line,
            start=[end_line+1, 0],
            end=[end_line+1, 0],
        )
        
        # Update the file through LSP client
        self.file_client.update_file(changes=[change])
        
        # Get state at this line
        state = self._get_state_at_line(end_line + 1, character=len(indent))
        if state is None:
            return None
        
        # Remove the temporary empty line
        delete_change = lc.DocumentContentChange(
            text="",
            start=[end_line+1, 0],
            end=[end_line + 2, 0],
        )
        self.file_client.update_file(changes=[delete_change])
        
        return state
    
    def _extract_case_name_from_state(self, state: str) -> Optional[str]:
        """
        Extract the case name from the state.
        
        Case names appear as the first line of the state, e.g., "case left" or "case right".
        
        Args:
            state: The proof state string
            
        Returns:
            The case name (e.g., "left", "right") or None if not found
        """
        if not state or not state.strip():
            return None
        
        lines = state.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        
        match = re.match(r"case\s+([a-zA-Z_][a-zA-Z0-9_]*)", first_line)
        if match:
            return match.group(1)
        
        return None
    
    def _is_hypothesis_completed_in_same_line(self, tactic: str) -> bool:
        """
        Check if a hypothesis is completed in the same line (e.g., "have h1 : ... := by linarith"
        or "suffices P by linarith").

        - have: completed in same line if ":=" in tactic and last line does NOT end with
          ":=" or " by" or ":=by" (so the proof is on the same line).
        - suffices: completed in same line if single line, contains " by ", and does NOT end
          with " by" (so the proof follows " by" on the same line).

        Args:
            tactic: The tactic string

        Returns:
            True if the hypothesis is completed in the same line (proof is on same line)
        """
        tactic_lines = tactic.split('\n')
        if not tactic_lines:
            return False

        last_line = tactic_lines[-1].strip()

        # have / obtain / replace: use ":="
        # Multi-line: proof on next line(s), may be incomplete -> not completed in same line.
        if ":=" in tactic:
            return not (
                last_line.endswith(":=")
                or last_line.endswith(" by")
                or last_line.endswith(":=by")
            )

        # suffices: "suffices P by <proof>" on a single line = completed in same line
        suffices_line = last_line.lstrip("·").lstrip(". ").lstrip()
        if len(tactic_lines) == 1 and suffices_line.startswith("suffices"):
            return " by " in last_line and not last_line.endswith(" by")

        return False
    
    def _is_case_split_tactic(self, tactic: str) -> bool:
        """
        Check if a tactic is a case split (induction or cases with pattern branches).
        
        Args:
            tactic: The tactic string
            
        Returns:
            True if this is a case split tactic
        """
        tactic_lines = tactic.split('\n')
        if not tactic_lines:
            return False
        
        first_line = tactic_lines[0].strip()
        # Check if it is a "match" tactic (ends with "match ... with")
        is_match_tactic = bool(re.search(r"match\s+.*\s+with\n$", first_line))
        # Check if it starts with "induction" or "cases"
        if not (first_line.startswith("induction") or first_line.startswith("cases") or is_match_tactic):
            return False
        
        # Check if there are any lines starting with "|"
        for line in tactic_lines[1:]:
            if line.strip().startswith("|"):
                return True
        
        return False
    
    def _split_case_split_tactic(self, tactic: str) -> tuple[str, list[str]]:
        """
        Split a case split tactic into the main line and case branches.
        
        Args:
            tactic: The case split tactic string
            
        Returns:
            Tuple of (main_line, case_branches) where:
            - main_line: The first line (e.g., "induction k with")
            - case_branches: List of case branch lines (e.g., ["| zero =>", "| succ k ih =>"])
        """
        tactic_lines = tactic.split('\n')
        if not tactic_lines:
            return "", []
        
        main_line = tactic_lines[0].strip()
        case_branches = []
        
        for line in tactic_lines[1:]:
            stripped = line.strip()
            if stripped.startswith("|"):
                case_branches.append(stripped)
            else:
                main_line += "\n" + stripped
        
        return main_line, case_branches
    
    def _check_hypothesis_resolved_at_indent(
        self,
        indent_level: int,
        hypothesis_name: str,
        is_case: bool,
        is_case_split: bool,
        tactic: Optional[str] = None,
    ) -> bool:
        """
        Check if a hypothesis/case is resolved by checking the state at the proof level.
        If the hypothesis is completed in the same line (ends with ":=" or " by"), check at
        the current indent level. Otherwise, check at indent_level + 1.
        
        Args:
            indent_level: The indentation level where the hypothesis/case was declared
            hypothesis_name: The name of the hypothesis/case
            is_case: True if this is a case, False if it's a hypothesis
            tactic: Optional tactic string to check if hypothesis is completed in same line
        Returns:
            True if the hypothesis/case is resolved (proven)
        """
        # If we have the tactic and it's completed in the same line, check at current indent
        if tactic is not None and not is_case and self._is_hypothesis_completed_in_same_line(tactic):
            # Hypothesis is completed in the same line, check at current indent level
            proof_state = self._get_state_at_indent(indent_level)
        else:
            # Check the state at the proof level (indent_level + 1)
            # This is where the proof of the hypothesis/case would be
            proof_state = self._get_state_at_indent(indent_level + 1)
        
        # Check if state extraction failed
        if proof_state is None:
            return False
        
        proof_state_stripped = proof_state.strip()
        
        # If the proof state shows "no goals", the hypothesis/case is proven
        if proof_state_stripped == "no goals":
            return True
        
        # # For case split tactics, also check if we've moved to a different case
        # if is_case and is_case_split:
        #     current_case = self._extract_case_name_from_state(proof_state)
        #     if current_case is not None and current_case != hypothesis_name:
        #         return True
        
        return False
    
    def _get_state_at_line(self, line_number: int, character: int = 0) -> Optional[str]:
        """
        Get the state at a specific line using Lean LSP.
        
        Args:
            line_number: Line number
            character: Character position (default: 0)
            
        Returns:
            State text, or None if there was an error or timeout
        """
        result, timed_out, error = run_with_timeout(
            lambda: self.file_client.get_goal(line=line_number, character=character),
            timeout=self.timeout_seconds,
        )
        
        if timed_out:
            logger.warning(f"get_goal timed out after {self.timeout_seconds}s at line {line_number}")
            print(f"[get_state_at_line] get_goal timed out after {self.timeout_seconds}s at line {line_number}")
            raise StateFetchAbort(f"get_goal timed out after {self.timeout_seconds}s at line {line_number}", is_timeout=True)

        
        if error is not None:
            return ""
        
        if result is None:
            return ""
        
        state_text = extract_state_text(result)
        return state_text
    
    def get_state(self) -> Optional[str]:
        """
        Get the state at the end of the current code.
        
        Returns:
            State text, or None if there was an error
        """
        # Calculate the line number at the end of the code
        lines = self.current_code.rstrip().split('\n')
        end_line = len(lines)
        
        return self._get_state_at_line(end_line, character=0)
    
    def update(self, tactic: str) -> tuple[Optional[str], str]:
        """
        Apply a tactic to the code and return the updated state and code.
        
        This method:
        1. Finds the correct indent for the tactic based on the hypothesis stack
        2. Adds the tactic at the end of the file with the correct indent
        3. Evaluates the updated state
        4. Updates the hypothesis stack if needed
        5. Handles case split tactics (induction/cases) by splitting them into parts
        
        Args:
            tactic: The tactic to apply
            
        Returns:
            Tuple of (state, updated_code):
            - state: The state after applying the tactic, or None if extraction failed
            - updated_code: The updated code string
        """
        # Track if we had pending case branches at the start (to detect case splits)
        had_pending_case_branches_at_start = len(self.pending_case_branches) > 0
        protected_entry: Optional[HypothesisEntry] = None
        
        # Handle standalone closing bracket } - this is a closing bracket from the splitter
        # Just insert it at the current indent (the bracket should already be popped if complete)
        if tactic.strip() == "}":
            indent_level = len(self.hypothesis_stack) if self.hypothesis_stack else 0
            # The closing bracket should be at the same indent as the opening bracket
            # If there's a bracket on the stack, use its indent; otherwise use current
            if self.hypothesis_stack:
                top_entry = self.hypothesis_stack[-1]
                if top_entry.is_bracket:
                    indent_level = top_entry.indent_level
            # Insert the closing bracket
            bracket_indent = "  " * indent_level
            closing_line = bracket_indent + "}\n"
            lines = self.current_code.split('\n')
            end_line = len(lines) - 1 if lines else 0
            end_char = len(lines[-1]) if lines else 0
            closing_change = lc.DocumentContentChange(
                text=closing_line,
                start=[end_line, end_char],
                end=[end_line, end_char],
            )
            self.file_client.update_file(changes=[closing_change])
            self.current_code = self.current_code + closing_line
            # Get state after closing
            if self.hypothesis_stack:
                top_entry = self.hypothesis_stack[-1]
                state_after = self._get_state_at_indent(top_entry.indent_level + 1)
            else:
                state_after = self.get_state()
            return state_after, self.current_code
        
        indent_level = (self.hypothesis_stack[-1].indent_level + 1) if self.hypothesis_stack else 0
        
        # Check if we have pending case branches to process
        if self.pending_case_branches and (not tactic.strip() or tactic.strip().startswith('|')):
            # Process the next pending case branch
            case_line, case_indent = self.pending_case_branches.pop()
            tactic = case_line
            indent_level = case_indent
        
        # Check if this is a case split tactic (induction/cases with | branches)
        is_case_split = self._is_case_split_tactic(tactic)
        if is_case_split:
            main_line, case_branches = self._split_case_split_tactic(tactic)
            # Store case branches for later processing
            # Case branches should be at the same indent level as the main line
            for case_branch in reversed(case_branches):
                self.pending_case_branches.append((case_branch, indent_level))
            tactic = main_line
        
        # Check if this is a bullet point (·) - these are case markers
        is_case_tactic = tactic.strip().startswith('·') or tactic.strip().startswith('. ') or tactic.strip() == '.'
        if is_case_tactic:
            # Bullet points are at the current indent level
            # They mark the start of a case, so we add them to the stack
            # This ensures that subsequent tactics are indented one level deeper
            # Get state at the current indent level to extract the case name
            case_name = None
            case_state = self._get_state_at_indent(indent_level)
            if case_state:
                case_name = self._extract_case_name_from_state(case_state)
            
            # Always add bullet point to stack, even if we can't extract a case name
            # Use a generic name if extraction failed
            if not case_name:
                case_name = f"bullet_{indent_level}"
            
            self.hypothesis_stack.append(HypothesisEntry(case_name, indent_level, True))
        
        # Handle case branch lines (starting with |)
        if tactic.strip().startswith('|'):
            # Extract case name from the branch line (e.g., "| zero =>" -> "zero")
            branch_line = tactic.strip()
            # Pattern: | name => or | name args => or | name : type =>
            match = re.match(r'\|\s*([a-zA-Z_][a-zA-Z0-9_\']*)(?:\s|$|:|=)', branch_line)
            if match:
                case_name = match.group(1)
                self.hypothesis_stack.append(HypothesisEntry(case_name, indent_level, True, True))
        
        # Check if this is a bracket/try situation (line ending with {)
        is_bracket_tactic = '\n'.join(tactic.split('\n')[:-1]).rstrip().endswith('{')
        if is_bracket_tactic:
            bracket_name = f"bracket_{indent_level}"
            self.hypothesis_stack.append(HypothesisEntry(bracket_name, indent_level, False, False, True))
            protected_entry = self.hypothesis_stack[-1]
        
        # Extract hypothesis name from "have" tactic
        hypothesis_name = self._extract_hypothesis_name_from_have(tactic)
        if hypothesis_name:
            # Check if hypothesis is completed in the same line
            is_completed_in_line = self._is_hypothesis_completed_in_same_line(tactic)
            # If the have tactic is in the same line as case tactic, add one to the indent level
            case_offset = 1 if is_case_tactic else 0
            if not is_completed_in_line:
                self.hypothesis_stack.append(HypothesisEntry(hypothesis_name, indent_level + case_offset, False))
        
        # Calculate the position at the end of the current code
        lines = self.current_code.split('\n')
        last_line = lines[-1] if lines else ""
        
        # Position is 0-indexed for line and character
        end_line = len(lines) - 1 if lines else 0
        end_char = len(last_line)
        
        indent = "  " * indent_level
        # Handle multi-line tactics by indenting each line
        tactic_lines = tactic.split('\n')
        indented_tactic_lines = [indent + line if line.strip() else line for line in tactic_lines]
        if is_bracket_tactic:
            indented_tactic_lines = indented_tactic_lines[:-1]
        indented_tactic = '\n'.join(indented_tactic_lines)
        
        text_to_insert = indented_tactic + "\n" if self.current_code.rstrip() else indented_tactic
        change = lc.DocumentContentChange(
            text=text_to_insert,
            start=[end_line, end_char],
            end=[end_line, end_char],
        )
        
        # Update the file through LSP client
        self.file_client.update_file(changes=[change])
        
        # Update internal code representation
        self.current_code = self.current_code + text_to_insert
        
        # Pop hypotheses until we reach one that cannot be popped
        # We check by verifying if the state at the proof level shows "no goals"
        popped_hypotheses = []
        
        # Track if we popped a case that has pending branches
        popped_case_with_pending = False
        popped_case_indent = None
        
        force_pop = (text_to_insert.strip().startswith('|') and not text_to_insert.strip().split('\n')[0].endswith('=>'))
        # Now check each hypothesis
        while len(self.hypothesis_stack) > 0:

            top_entry = self.hypothesis_stack[-1]
            if protected_entry is top_entry:
                break
            # Check if the hypothesis is resolved by checking the state at its proof level
            # Pass the tactic if it's the one we just added (for same-line completion check)
            check_tactic = tactic if (hypothesis_name == top_entry.name and not top_entry.is_case and not top_entry.is_bracket) else None

            # Check if hypothesis is resolved
            if force_pop or self._check_hypothesis_resolved_at_indent(
                top_entry.indent_level,
                top_entry.name,
                top_entry.is_case,
                is_case_split=top_entry.is_case_split,
                tactic=check_tactic,
            ):
                force_pop = False
                popped_entry = self.hypothesis_stack.pop()
                popped_hypotheses.append((popped_entry.name, popped_entry.indent_level, popped_entry.is_case))
                
                # If this is a bracket, add the closing } at the same indent
                if popped_entry.is_bracket:
                    bracket_indent = "  " * popped_entry.indent_level
                    closing_line = bracket_indent + "}\n"
                    # Insert the closing bracket
                    lines = self.current_code.split('\n')
                    end_line = len(lines) - 1 if lines else 0
                    end_char = len(lines[-1]) if lines else 0
                    closing_change = lc.DocumentContentChange(
                        text=closing_line,
                        start=[end_line, end_char],
                        end=[end_line, end_char],
                    )
                    self.file_client.update_file(changes=[closing_change])
                    self.current_code = self.current_code + closing_line
                
                # Track if we popped a case that has pending branches
                if popped_entry.is_case_split and self.pending_case_branches:
                    popped_case_with_pending = True
                    popped_case_indent = popped_entry.indent_level
                    # Break here to handle the next case branch before continuing to pop
                    break
                if popped_entry.is_case_split:
                    if not self.pending_case_branches:
                        force_pop = True
                    else:
                        tactic, case_indent = self.pending_case_branches[-1]
                        if case_indent < popped_entry.indent_level:
                            force_pop = True
            else:
                break
        
        # If we forcefully popped the main goal, so that the hypothesis stack is empty, 
        # finish now with "no goals"
        if len(self.hypothesis_stack) == 0:
            return "no goals", self.current_code

        top_entry = self.hypothesis_stack[-1]
        top_name = top_entry.name
        hyp_indent_level = top_entry.indent_level
        is_case = top_entry.is_case
        state_after = self._get_state_at_indent(hyp_indent_level + 1)
        # If we just applied a case split main line and have pending case branches,
        # automatically apply the first case branch
        if is_case_split and self.pending_case_branches:
            return self.update("")
        
        # Check if we completed a case and should move to the next pending case branch
        # This happens in two scenarios:
        # 1. We just popped a case that has pending branches (handled above)
        # 2. The current case (still on stack) is completed
        if popped_case_with_pending and self.pending_case_branches:
            return self.update("")
        
        # Also check if current case (still on stack) is completed
        # This check happens after popping, so if a case is still on the stack and has pending branches,
        # we need to check if it's completed to move to the next case
        if self.pending_case_branches and top_entry.is_case_split:
            # Check if current case is completed
            # Check state at the case's proof level (indent + 1)
            case_proof_state = self._get_state_at_indent(hyp_indent_level + 1)
            case_completed = False
            if case_proof_state:
                if case_proof_state.strip() == "no goals":
                    case_completed = True
            
            if case_completed:
                return self.update("")
        
        # If we have no more pending case branches but there's still a case on the stack,
        # check if the case is actually completed by checking the state
        # This handles the case where the last case branch completes
        # But we should NOT pop bullet points (·) immediately - they need to be proven first
        if not self.pending_case_branches and is_case:
            # Determine whether the current case is resolved
            case_proof_state = self._get_state_at_indent(hyp_indent_level + 1)
            case_completed = False
            # if had_pending_case_branches_at_start:
            #     case_completed = True
            if case_proof_state and case_proof_state.strip() == "no goals":
                case_completed = True
            if case_completed:
                finished_case_split = False
                popped_entry = None
                if len(self.hypothesis_stack) > 1:
                    popped_entry = self.hypothesis_stack.pop()
                    if had_pending_case_branches_at_start:
                        finished_case_split = True
                if self.hypothesis_stack:
                    top_entry = self.hypothesis_stack[-1]
                    top_name = top_entry.name
                    hyp_indent_level = top_entry.indent_level
                    is_case = top_entry.is_case
                    state_after = self._get_state_at_indent(hyp_indent_level + 1)
                # After popping the last case, check if the parent hypothesis should be popped
                # This should ONLY happen for case splits (induction/cases with | branches),
                # NOT for other case-generating tactics like constructor with · bullets
                # We detect case splits by checking if we finished processing all cases from a case split
                if finished_case_split and popped_entry and len(self.hypothesis_stack) > 1:
                    parent_entry = self.hypothesis_stack[-1]
                    if not parent_entry.is_case and parent_entry.indent_level == popped_entry.indent_level - 1:
                        # This is the hypothesis that contained the induction/cases
                        # Since all cases are done, the hypothesis is resolved
                        parent_proof_state = self._get_state_at_indent(parent_entry.indent_level + 1)
                        
                        # For induction/cases, once all cases are done, consider it resolved
                        # even if state doesn't show "no goals" (this is a Lean quirk)
                        if parent_proof_state is not None:
                            # Check if we're back at the parent's goal (not still in a case)
                            current_case = self._extract_case_name_from_state(parent_proof_state)
                            if current_case is None:
                                self.hypothesis_stack.pop()
                                if self.hypothesis_stack:
                                    top_entry = self.hypothesis_stack[-1]
                                    top_name = top_entry.name
                                    hyp_indent_level = top_entry.indent_level
                                    is_case = top_entry.is_case
                                    state_after = self._get_state_at_indent(hyp_indent_level + 1)
        return state_after, self.current_code

    @classmethod
    def apply_tactic(cls, file_client: lc.SingleFileClient, tactic: str) -> tuple[Optional[str], str]:
        """
        Apply a tactic to the code and return the updated state and code.
        
        Args:
            file_client: The Lean LSP file client
            tactic: The tactic to apply
            
        Returns:
            Tuple of (state, updated_code)
        """
        applier = cls(file_client)
        state, code = applier.update(tactic)
        return state, code
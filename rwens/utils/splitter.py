"""Tactic splitter for Lean 4 proofs.

This module handles splitting proof lines into complete tactics, handling:
- Unclosed brackets (excluding { and })
- Lines that don't start with recognized tactics
- Special handling for cases/induction with pattern matching (| patterns)
"""

import re
from typing import List, Tuple, Set

from rwens.utils.lean4_tactics import ALL_TACTICS


class TacticSplitter:
    """Splits Lean 4 proof text into complete tactics."""
    
    # Tactic words that trigger cases/induction pattern matching
    PATTERN_TACTICS = ['cases', "cases'", 'induction', "induction'"]
    
    def __init__(self):
        """Initialize the tactic splitter."""
        self._tactic_words = self._build_tactic_words()
    
    @staticmethod
    def _get_indentation(line: str) -> int:
        """Get the indentation level (number of leading spaces) of a line."""
        return len(line) - len(line.lstrip())
    
    def _build_tactic_words(self) -> Set[str]:
        """Build set of all tactic words for matching."""
        tactic_words = set()
        for tactic in ALL_TACTICS:
            words = tactic.split()
            if words:
                tactic_words.add(words[0])  # Add first word
            tactic_words.add(tactic)  # Also add full tactic
        
        # Add special cases
        tactic_words.update([
            "have", "have:", "let", "set", "show", "suffices", "calc", "match"
        ])
        
        return tactic_words
    
    def _is_tactic_start(self, line: str) -> bool:
        """Check if a line starts with a recognized tactic word."""
        stripped = line.lstrip()
        if not stripped:
            return False

        # Pattern/case lines and closing braces always begin a new "tactic-like" unit.
        if stripped.startswith("|") or stripped.startswith("}"):
            return True

        # Standalone bullet points (· or .) are separate tactics (case markers).
        if stripped == "·" or stripped == ".":
            return True

        # Allow bullet-prefixed tactics like "· simp" or ". intro".
        if stripped.startswith("·") or stripped.startswith("."):
            stripped = stripped[1:].lstrip()
            if not stripped:
                return False
        
        for tactic in self._tactic_words:
            if stripped.startswith(tactic):
                next_char_idx = len(tactic)
                if next_char_idx >= len(stripped):
                    return True  # Tactic at end of line
                next_char = stripped[next_char_idx]
                if next_char in [' ', '\t', "'", ";", "["]:
                    return True
        
        return False
    
    @staticmethod
    def _is_pattern_line(line: str) -> bool:
        """Check if a line is a pattern matching line (starts with |)."""
        stripped = line.lstrip()
        return stripped.startswith('|')
    
    def _is_cases_induction(self, line: str) -> bool:
        """Check if a line starts a cases/induction/match tactic."""
        stripped = line.lstrip()
        # Allow bullet-prefixed cases/induction/match, e.g. "· cases h with"
        if stripped.startswith("·") or stripped.startswith("."):
            stripped = stripped[1:].lstrip()
        # Check for cases/induction tactics
        if any(
            stripped.startswith(tactic) and 
            (len(stripped) == len(tactic) or stripped[len(tactic)] in [' ', '\t', "'"])
            for tactic in self.PATTERN_TACTICS
        ):
            return True
        # Check for match tactics (contains "match ... with" pattern)
        # Pattern: "match" followed by expression, then "with" at end of line
        if re.search(r"match\s+.*?\s+with\s*$", stripped):
            return True
        return False

    @staticmethod
    def _line_ends_with_calc(line: str) -> bool:
        """
        Return True if the line ends with a `calc` keyword (possibly after pattern arrows),
        e.g. `| succ n ih => calc`.
        """
        stripped = line.strip()
        return bool(re.search(r"\bcalc\s*$", stripped))

    @staticmethod
    def _line_ends_with_all_goals(line: str) -> bool:
        """
        Return True if the line ends with `all_goals`.
        When true, the line plus subsequent higher-indented lines form a single tactic block
        (e.g. `all_goals` followed by `positivity`).
        """
        stripped = line.strip()
        return bool(re.search(r"\ball_goals\s*$", stripped))

    @staticmethod
    def _line_ends_with_repeat(line: str) -> bool:
        """
        Return True if the line ends with `repeat`.
        When true, the line plus subsequent higher-indented lines form a single tactic block
        (e.g. `repeat` followed by indented tactics).
        """
        stripped = line.strip()
        return bool(re.search(r"\brepeat\s*$", stripped))

    def _split_tactics_with_line_map(
        self, proof_lines: List[str]
    ) -> Tuple[List[Tuple[int, str]], List[int]]:
        """
        Split proof lines into tactics *sequentially* (in source order) and also return
        a per-line mapping indicating which tactic each line belongs to.

        This is deliberately different from `_split_tactics`'s older pattern-stack logic:
        pattern (`| ... =>`) lines are treated as their own tactic units so reconstruction
        preserves the original ordering.

        Returns:
            (tactics, line_to_tactic_id)
            - tactics: list of (first_line_id, tactic_text) in source order
            - line_to_tactic_id: list of length len(proof_lines), where each entry is the
              index of the tactic containing that line.
        """
        tactics: List[Tuple[int, str]] = []
        line_to_tactic: List[int] = [-1] * len(proof_lines)
        if not proof_lines:
            return tactics, line_to_tactic

        def _leading_spaces(s: str) -> int:
            return len(s) - len(s.lstrip(" "))

        def _scan_indented_end(start: int) -> int:
            """Scan until we leave the indented block (next line has same or less indent)."""
            base_indent = _leading_spaces(proof_lines[start])
            j = start + 1
            it = 0
            while j < len(proof_lines):
                it += 1
                if it > 1000:
                    break
                if _leading_spaces(proof_lines[j]) > base_indent:
                    j += 1
                    continue
                break
            return j

        def _scan_normal_end(start: int) -> int:
            first = proof_lines[start]
            first_stripped = first.lstrip()
            # Allow bullet-prefixed "have" as well.
            first_for_kw = first_stripped
            if first_for_kw.startswith("·") or first_for_kw.startswith("."):
                first_for_kw = first_for_kw[1:].lstrip()
            is_have_tactic = first_for_kw.startswith("have")

            # Special-case: lines ending with `{` open a brace block and should be a single
            # "tactic" line, unless we are still accumulating a multi-line `have` header.
            if first.rstrip().endswith("{") and not (is_have_tactic and ":=" not in first):
                return start + 1

            open_paren = first.count("(") - first.count(")")
            open_bracket = first.count("[") - first.count("]")
            open_angle = first.count("⟨") - first.count("⟩")
            open_brace = first.count("{") - first.count("}")
            seen_assign = (":=" in first) if is_have_tactic else True

            j = start + 1
            it = 0
            while j < len(proof_lines):
                it += 1
                if it > 1000:
                    break

                nxt = proof_lines[j]

                # For multi-line `have` headers (before `:=`), keep accumulating regardless
                # of apparent tactic starts.
                if is_have_tactic and not seen_assign:
                    open_paren += nxt.count("(") - nxt.count(")")
                    open_bracket += nxt.count("[") - nxt.count("]")
                    open_angle += nxt.count("⟨") - nxt.count("⟩")
                    open_brace += nxt.count("{") - nxt.count("}")
                    if ":=" in nxt:
                        seen_assign = True
                    j += 1
                    continue

                # Otherwise: if the next line clearly begins a new tactic unit, stop.
                if self._is_cases_induction(nxt) or self._is_pattern_line(nxt) or self._is_calc(nxt):
                    break
                if self._is_pattern_line(nxt) and self._line_ends_with_calc(nxt):
                    break
                if self._line_ends_with_all_goals(nxt):
                    break
                if self._line_ends_with_repeat(nxt):
                    break

                # If brackets are closed, then a "tactic start" begins a new tactic; otherwise
                # we keep accumulating continuation lines.
                if open_paren == 0 and open_bracket == 0 and open_angle == 0 and open_brace == 0:
                    if nxt.lstrip() and self._is_tactic_start(nxt):
                        break

                open_paren += nxt.count("(") - nxt.count(")")
                open_bracket += nxt.count("[") - nxt.count("]")
                open_angle += nxt.count("⟨") - nxt.count("⟩")
                open_brace += nxt.count("{") - nxt.count("}")
                j += 1

            return j

        i = 0
        while i < len(proof_lines):
            line = proof_lines[i]
            stripped = line.lstrip()

            # Standalone closing brace: treat as its own unit.
            if stripped == "}":
                start = i
                end = i + 1
            # cases/induction/match headers are single-line units.
            elif self._is_cases_induction(line):
                start = i
                end = i + 1
            # pattern lines are single-line units unless they end with `calc` or `all_goals`,
            # in which case they start a block.
            elif self._is_pattern_line(line) and self._line_ends_with_calc(line):
                start = i
                end = _scan_indented_end(i)
            elif self._is_pattern_line(line) and self._line_ends_with_all_goals(line):
                start = i
                end = _scan_indented_end(i)
            elif self._is_pattern_line(line) and self._line_ends_with_repeat(line):
                start = i
                end = _scan_indented_end(i)
            elif self._is_pattern_line(line):
                start = i
                end = i + 1
            # calc lines start a calc block.
            elif self._is_calc(line) or self._line_ends_with_calc(line):
                start = i
                end = _scan_indented_end(i)
            # all_goals lines (line ends with all_goals) + subsequent higher-indent lines = one tactic.
            elif self._line_ends_with_all_goals(line):
                start = i
                end = _scan_indented_end(i)
            # repeat lines (line ends with repeat) + subsequent higher-indent lines = one tactic.
            elif self._line_ends_with_repeat(line):
                start = i
                end = _scan_indented_end(i)
            else:
                start = i
                end = _scan_normal_end(i)

            if end <= start:
                end = start + 1

            current_lines = proof_lines[start:end]
            tactic_text = self._normalize_tactic_indentation(current_lines)
            tactic_id = len(tactics)
            tactics.append((start, tactic_text))
            for j in range(start, end):
                line_to_tactic[j] = tactic_id

            i = end

        return tactics, line_to_tactic
    
    @staticmethod
    def _is_calc(line: str) -> bool:
        """Check if a line starts a calc tactic."""
        stripped = line.lstrip()
        # Allow bullet-prefixed calc blocks like "· calc" or ". calc"
        if stripped.startswith("·") or stripped.startswith("."):
            stripped = stripped[1:].lstrip()
        return stripped.startswith("calc") and (
            len(stripped) == 4 or stripped[4] in [" ", "\t"]
        )
    
    @staticmethod
    def _preprocess_proof_lines(proof_lines: List[str]) -> List[str]:
        """Preprocess proof lines by removing comments and blank lines."""
        cleaned_lines = []
        for line in proof_lines:
            stripped = line.strip()
            
            # Skip blank lines
            if not stripped:
                continue
            
            # Skip comment-only lines
            if stripped.startswith('--'):
                continue
            
            # Remove inline comments
            comment_idx = line.find('--')
            if comment_idx >= 0:
                before_comment = line[:comment_idx]
                line = before_comment.rstrip()
                if not line.strip():
                    continue
            
            cleaned_lines.append(line)
        
        return cleaned_lines
    
    @staticmethod
    def _normalize_tactic_indentation(tactic_lines: List[str]) -> str:
        """Normalize indentation by removing only the first line's indentation from all lines."""
        if not tactic_lines:
            return ""
        
        first_line = tactic_lines[0]
        first_indent = len(first_line) - len(first_line.lstrip())
        
        stripped_lines = []
        for tactic_line in tactic_lines:
            line_leading = len(tactic_line) - len(tactic_line.lstrip())
            if line_leading <= first_indent:
                stripped_lines.append(tactic_line.lstrip())
            else:
                stripped_lines.append(tactic_line[first_indent:])
        
        return '\n'.join(stripped_lines)
    
    def _handle_cases_induction(
        self,
        proof_lines: List[str],
        i: int,
        line_indent: int,
        pattern_stack: List,
        tactics: List[Tuple[int, str]]
    ) -> int:
        """Handle cases/induction tactic and pattern matching.
        
        Returns:
            Updated index i
        """
        line_stripped = proof_lines[i].lstrip()
        
        # Close any patterns at same or higher indentation level
        while pattern_stack and pattern_stack[-1][0] > line_indent:
            indent, first_line_id, tactic_text = pattern_stack.pop()
            tactics.append((first_line_id, tactic_text))
        
        # Push new pattern to stack
        first_line_id = i
        tactic_text = line_stripped
        pattern_stack.append([line_indent, first_line_id, tactic_text])
        
        return i + 1
    
    def _handle_pattern_line(
        self,
        proof_lines: List[str],
        i: int,
        line_indent: int,
        pattern_stack: List,
        tactics: List[Tuple[int, str]]
    ) -> int:
        """Handle pattern matching line (|).
        
        Returns:
            Updated index i, or None if not handled
        """
        line_stripped = proof_lines[i].lstrip()
        
        # Close patterns at higher indentation
        while pattern_stack and pattern_stack[-1][0] > line_indent:
            indent, first_line_id, tactic_text = pattern_stack.pop()
            tactics.append((first_line_id, tactic_text))
        
        if pattern_stack:
            # Add pattern line to current pattern on stack
            pattern_stack[-1][-1] += '\n' + (pattern_stack[-1][0] - line_indent) * ' ' + line_stripped
            return i + 1
        
        return None  # Not handled
    
    def _handle_indented_block(
        self,
        proof_lines: List[str],
        i: int,
        line_indent: int,
        tactics: List[Tuple[int, str]]
    ) -> int:
        """Handle a block of form: first line + subsequent lines with strictly greater indent.
        
        Used for calc, all_goals, and similar constructs.
        
        Returns:
            Updated index i
        """
        line = proof_lines[i]
        first_line_id = i
        current_tactic_lines = [line]
        base_indent = line_indent

        i += 1
        it = 0
        while i < len(proof_lines):
            it += 1
            if it > 100:
                break
            next_line = proof_lines[i]
            next_indent = self._get_indentation(next_line)
            if next_indent > base_indent:
                current_tactic_lines.append(next_line)
                i += 1
                continue
            break

        tactic_text = self._normalize_tactic_indentation(current_tactic_lines)
        tactics.append((first_line_id, tactic_text))
        return i
    
    def _handle_normal_tactic(
        self,
        proof_lines: List[str],
        i: int,
        tactics: List[Tuple[int, str]]
    ) -> int:
        """Handle normal tactic with bracket balancing.
        
        Returns:
            Updated index i
        """
        line = proof_lines[i].lstrip().lstrip('}')
        stripped_line = line.lstrip()
        indent = len(line) - len(stripped_line)
        stripped_line = stripped_line.lstrip('}')
        line = indent * ' ' + stripped_line
        first_line_id = i
        current_tactic_lines = [line]
        
        # Check if this is a "have" tactic - need special handling
        is_have_tactic = stripped_line.startswith('have')
        
        # Track brackets (excluding braces - handled separately)
        open_paren = line.count('(') - line.count(')')
        open_bracket = line.count('[') - line.count(']')
        open_angle = line.count('⟨') - line.count('⟩')
        open_brace = line.count('{') - line.count('}')
        
            # Handle brace blocks - lines ending with {
        if line.rstrip().endswith('{') and not (is_have_tactic and ":=" not in line):
            # This is a bracket opening - add it as a tactic, then add closing } as separate tactic
            tactic_text = self._normalize_tactic_indentation([line])
            tactics.append((first_line_id, tactic_text + "\n}"))
            i += 1
            return i
        
        if line.strip() == "}" or line.strip() == "":
            # Handle closing brace - this should be matched with opening
            i += 1
            return i
        
        i += 1
        
        # Continue accumulating lines until tactic is complete
        normal_iterations = 0
        while i < len(proof_lines):
            normal_iterations += 1
            if normal_iterations > 100:
                break
            
            next_line = proof_lines[i]
            next_line_stripped = next_line.lstrip()
            
            # Check if this starts a new cases/induction - end current tactic
            if self._is_cases_induction(next_line):
                break
            
            # For "have" tactics, check if we've seen ":=" (the proof starts)
            if is_have_tactic:
                # Accumulate all lines so far and check if ":=" is in the accumulated text
                accumulated_text = '\n'.join(current_tactic_lines)
                if ':=' in accumulated_text:
                    # We've found :=, so the have statement is complete
                    # Check if brackets are closed before breaking
                    if open_paren == 0 and open_bracket == 0 and open_angle == 0 and open_brace == 0:
                        # All brackets closed and := found, check if next line is a continuation
                        if next_line_stripped and not self._is_tactic_start(next_line):
                            # Continuation line (part of the proof after :=), add it
                            current_tactic_lines.append(next_line)
                            open_paren += next_line.count('(') - next_line.count(')')
                            open_bracket += next_line.count('[') - next_line.count(']')
                            open_angle += next_line.count('⟨') - next_line.count('⟩')
                            open_brace += next_line.count('{') - next_line.count('}')
                            i += 1
                            continue
                        else:
                            break
                    # Brackets not closed, continue accumulating
                    current_tactic_lines.append(next_line)
                    open_paren += next_line.count('(') - next_line.count(')')
                    open_bracket += next_line.count('[') - next_line.count(']')
                    open_angle += next_line.count('⟨') - next_line.count('⟩')
                    open_brace += next_line.count('{') - next_line.count('}')
                    i += 1
                    continue
                else:
                    # Haven't found := yet, continue accumulating
                    current_tactic_lines.append(next_line)
                    open_paren += next_line.count('(') - next_line.count(')')
                    open_bracket += next_line.count('[') - next_line.count(']')
                    open_angle += next_line.count('⟨') - next_line.count('⟩')
                    open_brace += next_line.count('{') - next_line.count('}')
                    i += 1
                    continue
            
            # Check if brackets are closed
            if open_paren == 0 and open_bracket == 0 and open_angle == 0 and open_brace == 0:
                # All brackets closed, check if next line is a continuation
                if next_line_stripped and not self._is_tactic_start(next_line):
                    # Continuation line, add it
                    current_tactic_lines.append(next_line)
                    open_paren += next_line.count('(') - next_line.count(')')
                    open_bracket += next_line.count('[') - next_line.count(']')
                    open_angle += next_line.count('⟨') - next_line.count('⟩')
                    open_brace += next_line.count('{') - next_line.count('}')
                    i += 1
                    continue
                else:
                    break
            
            # Brackets not closed, continue accumulating
            current_tactic_lines.append(next_line)
            open_paren += next_line.count('(') - next_line.count(')')
            open_bracket += next_line.count('[') - next_line.count(']')
            open_angle += next_line.count('⟨') - next_line.count('⟩')
            open_brace += next_line.count('{') - next_line.count('}')
            i += 1
        
        # Build tactic text
        if current_tactic_lines:
            tactic_text = self._normalize_tactic_indentation(current_tactic_lines)
            tactics.append((first_line_id, tactic_text))
        
        return i
    
    def _split_tactics(self, proof_lines: List[str]) -> List[Tuple[int, str]]:
        """Split proof lines into complete tactics.
        
        Args:
            proof_lines: List of proof lines (after ":=" by, preprocessed)
        
        Returns:
            List of (first_line_id, tactic_text) tuples, sorted by first_line_id
        """
        tactics = []  # List of (first_line_id, tactic_text) tuples
        pattern_stack = []  # Stack for cases/induction pattern matching
        
        if not proof_lines:
            return tactics
        
        i = 0
        iteration_count = 0
        while i < len(proof_lines):
            iteration_count += 1
            if iteration_count > 1000:
                break
            
            line = proof_lines[i]
            line_stripped = line.lstrip()
            # Skip standalone closing brackets (these are closing brackets we added)
            if line_stripped == "}":
                i += 1
                continue
            line_indent = self._get_indentation(line)
            
            # Handle cases/induction
            if self._is_cases_induction(line):
                i = self._handle_cases_induction(
                    proof_lines, i, line_indent, pattern_stack, tactics
                )
                continue
            
            # Handle pattern lines
            if self._is_pattern_line(line):
                new_i = self._handle_pattern_line(
                    proof_lines, i, line_indent, pattern_stack, tactics
                )
                if new_i is not None:
                    i = new_i
                    continue
            
            # Handle calc tactics (indented block)
            if self._is_calc(line):
                i = self._handle_indented_block(proof_lines, i, line_indent, tactics)
                continue
            
            # Handle all_goals blocks (line ends with all_goals + higher-indent lines)
            if self._line_ends_with_all_goals(line):
                i = self._handle_indented_block(proof_lines, i, line_indent, tactics)
                continue

            # Handle repeat blocks (line ends with repeat + higher-indent lines)
            if self._line_ends_with_repeat(line):
                i = self._handle_indented_block(proof_lines, i, line_indent, tactics)
                continue
            
            # Handle normal tactics
            i = self._handle_normal_tactic(proof_lines, i, tactics)
        
        # Add any remaining tactics from the stack
        while pattern_stack:
            indent, first_line_id, tactic_text = pattern_stack.pop()
            tactics.append((first_line_id, tactic_text))
        
        # Sort by first_line_id
        tactics.sort(key=lambda x: x[0])
        
        return tactics
    
    @staticmethod
    def _find_proof_lines(full_lean_content: str) -> Tuple[int, List[str]]:
        """Find the first ':= by' line after the first 'theorem' and return all lines after it.
        
        Returns:
            Tuple of (proof_start_line_index, proof_lines)
        """
        # Find the first "theorem" keyword
        theorem_pattern = r'\btheorem\s+'
        theorem_match = re.search(theorem_pattern, full_lean_content)
        
        if not theorem_match:
            return -1, []
        
        theorem_start_pos = theorem_match.start()
        
        # Find the line number where theorem starts
        lines = full_lean_content.split('\n')
        theorem_line_idx = full_lean_content[:theorem_start_pos].count('\n')
        
        # Search for ':= by' only after the theorem keyword
        for i, line in enumerate(lines[theorem_line_idx:], start=theorem_line_idx):
            if ':= by' in line:
                # Return the index of the line with ':= by' and all lines after it
                proof_lines = lines[i+1:]
                # Preprocess to remove comments and blank lines
                proof_lines = TacticSplitter._preprocess_proof_lines(proof_lines)
                return i, proof_lines
        
        return -1, []
    
    @classmethod
    def split_proof_into_tactics(cls, proof_text: str) -> List[Tuple[int, str]]:
        """Split proof text into tactics.
        
        Given the full proof text (including the theorem statement), extracts the proof
        lines after ':=' by' and splits them into complete tactics.
        
        Args:
            proof_text: Full proof text (can include imports, theorem statement, etc.)
        
        Returns:
            List of (line_id, tactic_text) tuples where:
            - line_id is the index of the first line of the tactic (0-indexed within proof lines)
            - tactic_text is the complete tactic text with relative indentation preserved
            Results are sorted by line_id.
        """
        # Find proof lines
        proof_start_idx, proof_lines = cls._find_proof_lines(proof_text)
        
        if proof_start_idx == -1 or not proof_lines:
            return []
        
        # Create instance and split into tactics
        splitter = cls()
        tactics = splitter._split_tactics(proof_lines)
        
        return tactics

    @classmethod
    def split_proof_into_tactics_with_line_map(
        cls, proof_text: str
    ) -> Tuple[List[Tuple[int, str]], List[int]]:
        """
        Like `split_proof_into_tactics`, but also returns a per-line mapping indicating
        which tactic each (preprocessed) proof line belongs to.

        Returns:
            (tactics, line_to_tactic_id)
        """
        proof_start_idx, proof_lines = cls._find_proof_lines(proof_text)
        if proof_start_idx == -1 or not proof_lines:
            return [], []
        splitter = cls()
        return splitter._split_tactics_with_line_map(proof_lines)


def main():
    """Main function to test tactic splitting on test files."""
    from pathlib import Path
    
    test_dir = Path("tests/tactic_splitter")
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
    
    # Get all test files
    test_files = sorted(test_dir.glob("test_*.lean"))
    
    if not test_files:
        print(f"No test files found in {test_dir}")
        return
    
    print(f"Found {len(test_files)} test files\n")
    print("=" * 80)
    
    for test_file in test_files:
        print(f"\n{'=' * 80}")
        print(f"Processing: {test_file.name}")
        print('=' * 80)
        
        try:
            # Read file content
            content = test_file.read_text(encoding='utf-8')
            
            if not content.strip():
                print("File is empty, skipping...")
                continue
            
            # Find proof lines
            proof_start_idx, proof_lines = TacticSplitter._find_proof_lines(content)
            
            if proof_start_idx == -1:
                print("No ':= by' found in file, skipping...")
                continue
            
            if not proof_lines:
                print("No proof lines found after ':= by', skipping...")
                continue
            
            print(f"\nProof lines ({len(proof_lines)} lines):")
            print("-" * 80)
            for i, line in enumerate(proof_lines):
                print(f"{i:3d}: {line}")
            
            # Split into tactics
            splitter = TacticSplitter()
            tactic_results = splitter._split_tactics(proof_lines)
            
            print(f"\n\nRecovered {len(tactic_results)} tactics:")
            print("-" * 80)
            
            for first_line_id, tactic_text in tactic_results:
                print(f"Tactic starting at line {first_line_id}:")
                print("-" * 40)
                print(tactic_text)
                print("-" * 40)
            
        except Exception as e:
            print(f"Error processing {test_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 80}")
    print("Testing complete!")


if __name__ == "__main__":
    main()

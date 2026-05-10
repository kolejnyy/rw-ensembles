"""
Auxiliary functions for dataset processing.
"""

import re
from typing import Optional, Tuple

_DECL_START_RE = re.compile(r"(?m)^(theorem|lemma|def)\b")
_BY_RE = re.compile(r":=\s*by\b", re.MULTILINE | re.DOTALL)


def split_declarations_theorem_proof(full_text: str) -> Tuple[str, str, str]:
    """
    Split a Lean file into:
      - declarations: everything before the first theorem/lemma/def
      - theorem_stmt: from that keyword through the first `:= by`
      - proof_body: everything after `:= by` (leading newlines stripped)
    """
    normalized = full_text.replace("\r\n", "\n")
    m_decl = _DECL_START_RE.search(normalized)
    if m_decl is None:
        raise ValueError("Could not find theorem/lemma/def in file")

    decls = normalized[: m_decl.start()].rstrip("\n") + "\n"
    rest = normalized[m_decl.start() :]

    m_by = _BY_RE.search(rest)
    if m_by is None:
        raise ValueError("Could not find ':= by' in theorem statement")

    theorem_stmt = rest[: m_by.end()].rstrip("\n") + "\n"
    proof_body = rest[m_by.end() :].lstrip("\n")
    return decls, theorem_stmt, proof_body


def remove_comments(code: str) -> str:
    """
    Remove comments from Lean code.
    
    Removes:
    - Single-line comments starting with "--"
    - Multi-line comment blocks starting with "/-" and ending with "-/"
    
    When a comment is removed, the entire line(s) containing only the comment are also removed.
    
    Args:
        code: The Lean code string
        
    Returns:
        Code with comments and comment-only lines removed
    """
    # Normalize line endings
    code = code.replace("\r\n", "\n")
    
    # First, remove multi-line comments (/- ... -/)
    # This regex handles comments across multiple lines
    # We'll handle line removal separately
    pattern = r'/-.*?-/'
    code = re.sub(pattern, '', code, flags=re.DOTALL)
    
    # Now process line by line to remove single-line comments and empty lines
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Find the first "--" that's not inside a string
        comment_pos = line.find('--')
        if comment_pos != -1:
            # Check if it's inside a string (simple heuristic: count quotes before it)
            before_comment = line[:comment_pos]
            # Count unescaped double quotes
            quote_count = before_comment.count('"') - before_comment.count('\\"')
            # If odd number of quotes, we're inside a string, so don't remove
            if quote_count % 2 == 0:
                # Remove comment part
                line = line[:comment_pos].rstrip()
            # If in a string, keep the line as-is
        
        # Only add line if it has non-whitespace content
        # This removes lines that became empty after comment removal
        if line.strip():
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_state_text(result) -> str:
    """
    Extract state text from Lean LSP goal result.
    
    Args:
        result: The result from leanclient's get_goal method
        
    Returns:
        Formatted state text
    """
    if result is None:
        return ""

    goals = result.get("goals")
    rendered = result.get("rendered", "")

    if isinstance(goals, list) and len(goals) == 0:
        return rendered

    if isinstance(goals, list) and len(goals) > 0:
        # Join multiple goals with blank lines between them.
        return "\n\n".join(goals)

    # Fallback: prefer rendered if present.
    return rendered


def split_into_lines_with_tactics(code: str) -> list[tuple[str, str]]:
    """
    Split code into (prefix, next_tactic) pairs, handling multi-line tactics.
    
    A tactic can span multiple lines if it contains unclosed brackets.
    Brackets are tracked: `(`, `[`, and `{`. If a line ends with unclosed
    brackets, the tactic continues on the next line(s) until all brackets are closed.
    
    For each tactic in the proof, returns a tuple where:
    - prefix: all code up to (but not including) the current tactic
    - next_tactic: the current tactic (may span multiple lines)
    
    Args:
        code: The full proof code
        
    Returns:
        List of (prefix, next_tactic) tuples
    """
    lines = code.split('\n')
    result = []
    
    i = 0
    while i < len(lines):
        # Start of a new tactic
        tactic_start = i
        tactic_lines = []
        
        # Track bracket balance
        paren_count = 0  # (
        bracket_count = 0  # [
        brace_count = 0  # {
        
        # Accumulate lines until brackets are balanced
        while i < len(lines):
            line = lines[i]
            tactic_lines.append(line)
            
            # Count brackets in this line
            for char in line:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            
            # If all brackets are balanced, this is the end of the tactic
            if paren_count == 0 and bracket_count == 0 and brace_count == 0:
                i += 1
                break
            
            i += 1
        
        # Combine all lines of this tactic
        next_tactic = '\n'.join(tactic_lines)
        
        # Prefix is everything before this tactic
        prefix = '\n'.join(lines[:tactic_start])
        
        result.append((prefix, next_tactic))
    
    return result


def should_discard_file(full_lean_content: str) -> bool:
    """Check if file should be discarded due to multiple theorems or lemmas before main theorem.
    
    Returns True if file should be discarded (has multiple theorems or lemmas before main theorem).
    """
    
    # Count "theorem" declarations (whole word, not in comments/strings)
    # Look for "theorem" followed by whitespace or newline (not part of another word)
    theorem_pattern = r'\btheorem\s+'
    theorem_matches = list(re.finditer(theorem_pattern, full_lean_content))
    
    # Count "lemma" declarations
    lemma_pattern = r'\blemma\s+'
    lemma_matches = list(re.finditer(lemma_pattern, full_lean_content))

    # Count "def" declarations
    def_pattern = r'\bdef\s+'
    def_matches = list(re.finditer(def_pattern, full_lean_content))
    
    # Discard if:
    # 1. More than one theorem declaration
    # 2. Any lemma or def declarations before the main theorem
    if len(theorem_matches) > 1 or len(theorem_matches) == 0:
        return True
    
    if len(lemma_matches) > 0 or len(def_matches) > 0:
        return True

    return False
"""
Utilities for parsing and analyzing Lean proof states.

This module provides functions to extract and classify variables and hypotheses
from Lean proof states for canonicalization purposes.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ========== Renaming ==========
# Patterns and helpers for variable/hypothesis name normalization, subscript/superscript.
# Shared with VariableRenamer (have/intro, known names).
_TEXT_CHAR_CLASS = "a-zA-Z0-9_₀₁₂₃₄₅₆₇₈₉"
_RENAMING_PREFIXES = ["ih", "a", "n", "f", "x", "h", "S", "P", "z"]
_SUBSCRIPT_TO_DIGIT = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}
_HAVE_PATTERN = re.compile(
    r"(?m)^(?P<prefix>\s*(?:[·.]\s*)?)have(?P<space>\s*)(?P<name>[^\s:]+)?(?P<rest>\s*:\s*)"
)
_INTRO_PATTERN = re.compile(
    r"(?m)^(?P<prefix>\s*(?:[·.]\s*)?)intro\s+(?P<names>[^;\n]+)"
)


def _is_subscript_char(ch: str) -> bool:
    return ch in _SUBSCRIPT_TO_DIGIT


def split_multi_variable_line(line: str) -> List[Tuple[str, str]]:
    """
    Split a line that may contain multiple variables of the same type.
    
    For example, "x y : ℝ" becomes [("x", "ℝ"), ("y", "ℝ")].
    
    Args:
        line: A line of the form "var1 var2 ... : type"
    
    Returns:
        List of (name, type) tuples
    """
    if " : " not in line and " :\n" not in line:
        return []
    
    parts = line.split(" : ", 1)
    parts_newline = line.split(" :\n", 1)

    # Find the first occurrence of " : " or " :\n"
    if len(parts[0]) > len(parts_newline[0]):
        parts = parts_newline

    if len(parts) != 2:
        return []
    
    names_part = parts[0].strip()
    type_part = parts[1].strip()
    
    # Split variable names by whitespace
    names = names_part.split()
    
    return [(name, type_part) for name in names if name]


def prev_shadowed(name: str) -> str:
    """
    Get the previous shadowed name of a name.
    """
    if "✝" not in name:
        return name

    if name.endswith("✝"):
        return name.split("✝")[0]

    # Otherwise, the name is <name>✝<superscript number>
    # so we need to convert the superscript number to a regular number
    # and then return the name with dagger and superscript number one lower

    def _superscript_to_number(superscript: str) -> str:
        """Convert a superscript number to a regular number."""
        return str(int(superscript.replace("⁰", "0").replace("¹", "1").replace("²", "2").replace("³", "3").replace("⁴", "4").replace("⁵", "5").replace("⁶", "6").replace("⁷", "7").replace("⁸", "8").replace("⁹", "9")))

    def _number_to_superscript(number: str) -> str:
        """Convert a regular number to a superscript number."""
        return str(int(number)).replace("0", "⁰").replace("1", "¹").replace("2", "²").replace("3", "³").replace("4", "⁴").replace("5", "⁵").replace("6", "⁶").replace("7", "⁷").replace("8", "⁸").replace("9", "⁹")

    superscript = _superscript_to_number(name.split("✝")[1])
    if int(superscript) == 1:
        return name.split("✝")[0] + "✝"
    return name.split("✝")[0] + "✝" + _number_to_superscript(str(int(superscript) - 1))


def normalize_superscript_numbers(name: str) -> str:
    """
    Convert superscript numbers in a name to regular numbers.

    Handles shadowed hypotheses like 'hM✝¹' which should become 'hM✝1'.
    Superscript digits: ⁰¹²³⁴⁵⁶⁷⁸⁹ → 0123456789

    Args:
        name: The name string that may contain superscript numbers

    Returns:
        Name with superscript numbers converted to regular numbers
    """
    superscript_to_normal = str.maketrans({
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    })
    return name.translate(superscript_to_normal)


def build_intro_tactic(names: List[str], var_mapping: Dict[str, str]) -> str:
    """
    Build an 'intro' tactic string that introduces names using the given mapping.
    Names not in the mapping use 'shadowed_<base>' for the base name.
    """
    intro_tactic = "intro "
    for name in names:
        if name in var_mapping:
            intro_tactic += var_mapping[name] + " "
        else:
            base = name.split("✝")[0]
            intro_tactic += var_mapping.get(base, "shadowed_" + base) + " "
    return intro_tactic.rstrip()


def replace_known_names(tactic: str, var_mapping: Dict[str, str]) -> str:
    """
    Replace variable/hypothesis names in tactic text with their mapping, using
    word-boundary matching (so substrings inside longer identifiers are not replaced).
    """
    if not var_mapping:
        return tactic
    keys = sorted(var_mapping.keys(), key=len, reverse=True)
    pattern = re.compile(
        rf"(?<![{_TEXT_CHAR_CLASS}])({'|'.join(re.escape(k) for k in keys)})(?![{_TEXT_CHAR_CLASS}])"
    )

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return var_mapping.get(key, key)

    return pattern.sub(repl, tactic)


def detect_intro_prefix(name: str) -> Optional[str]:
    """
    Detect the renaming prefix for an intro name (e.g. 'h', 'ih', 'x').
    Returns None if the name does not match any known prefix plus optional subscript digits.
    """
    for p in sorted(_RENAMING_PREFIXES, key=len, reverse=True):
        if not name.startswith(p):
            continue
        suffix = name[len(p):]
        if not suffix or all(
            _is_subscript_char(ch) or ch.isdigit() for ch in suffix
        ):
            return p
    return None


def split_prefixed_subscript(value: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Split a value like 'h₁' or 'x' into (prefix, number) using _RENAMING_PREFIXES
    and _SUBSCRIPT_TO_DIGIT. Returns (None, None) if no matching prefix.
    """
    for p in sorted(_RENAMING_PREFIXES, key=len, reverse=True):
        if not value.startswith(p):
            continue
        suffix = value[len(p):]
        if not suffix:
            return p, None
        digits: List[str] = []
        for ch in suffix:
            d = _SUBSCRIPT_TO_DIGIT.get(ch)
            if d is None:
                break
            digits.append(d)
        if digits:
            return p, int("".join(digits))
        return p, None
    return None, None


def rename_have_hypotheses(
    tactic: str, used_numbers: Dict[str, Set[int]]
) -> str:
    """
    Rename 'have' hypothesis names in tactic lines to the next allocated h-subscript
    name, using _HAVE_PATTERN. Mutates used_numbers.
    """
    def repl(match: re.Match[str]) -> str:
        prefix = match.group("prefix") or ""
        space = match.group("space") or " "
        rest = match.group("rest") or ""
        if rest and not rest.startswith(" "):
            rest = " " + rest
        new_name = get_next_prefixed_name("h", used_numbers)
        return f"{prefix}have{space}{new_name}{rest}"

    return _HAVE_PATTERN.sub(repl, tactic)


def rename_intro_line(tactic: str, used_numbers: Dict[str, Set[int]]) -> str:
    """
    Rename 'intro' names in tactic lines to the next allocated prefix-based names,
    and replace those names in the rest of the tactic (e.g. "at h" after ";"). Mutates used_numbers.
    """
    intro_mappings: Dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        prefix = match.group("prefix") or ""
        names_blob = match.group("names") or ""
        names = names_blob.split()
        renamed = []
        for name in names:
            pkey = detect_intro_prefix(name)
            if pkey is None:
                pkey = "h"
            new_name = get_next_prefixed_name(pkey, used_numbers)
            intro_mappings[name] = new_name
            renamed.append(new_name)
        suffix = " ".join(renamed)
        return f"{prefix}intro {suffix}" if suffix else match.group(0)

    result = _INTRO_PATTERN.sub(repl, tactic)
    if intro_mappings:
        result = replace_known_names(result, intro_mappings)
    return result


# ========== State ==========
# Parsing Lean proof state: declarations, variables, hypotheses, definitions.


def extract_all_declarations(state: str) -> List[List[Tuple[str, str, str]]]:
    """
    Extract all variable and hypothesis declarations from a Lean state.
    
    Variables and hypotheses are at the beginning of new lines (no indentation)
    and are followed by " : ". Multi-line definitions are supported by collecting
    all indented continuation lines after a declaration.
    
    If the state contains multiple cases separated by blank lines (\\n\\n), they
    are processed separately, with each case returning its own list of declarations.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        List of lists, where each inner list contains (name, type, full_declaration)
        tuples for each declaration in that case. Always returns a list of lists,
        even if there's only one case (for compatibility).
    """
    # Split state by double newlines to handle multiple cases/goals
    state_parts = [part.strip() for part in state.split("\n\n") if part.strip()]
    
    # If no parts found, return empty list of lists
    if not state_parts:
        return []
    
    all_declarations = []
    for state_part in state_parts:
        declarations = _extract_declarations_from_part(state_part)
        all_declarations.append(declarations)
    
    return all_declarations


def _extract_declarations_from_part(state_part: str) -> List[Tuple[str, str, str]]:
    """
    Extract declarations from a single state part (one case/goal).
    
    Args:
        state_part: A single case/goal state as a string
    
    Returns:
        List of (name, type, full_declaration) tuples for each declaration
    """
    declarations = []
    lines = state_part.split("\n")
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        #print("(====)",stripped)
        # Skip empty lines and the goal line (starts with ⊢)
        if not stripped or stripped.startswith("⊢"):
            i += 1
            continue
        
        # Skip "case ..." lines (no indentation, starts with "case ")
        if (not line.startswith(" ") and not line.startswith("\t") and 
            stripped.startswith("case ")):
            i += 1
            continue
        
        # Check if line starts without indentation (beginning of line)
        if not line.startswith(" ") and not line.startswith("\t") and (" : " in line or line.endswith(" :")):
            # This is the start of a declaration
            declaration_lines = [stripped]
            type_start_idx = stripped.find(" : ")
            if type_start_idx == -1:
                type_start_idx = line.find(" :")
            
            # Collect all subsequent indented lines as part of the definition
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                
                # Stop if we hit the goal line
                if next_stripped.startswith("⊢"):
                    break
                
                # Stop if we hit an unindented line (next declaration)
                if next_stripped and not next_line.startswith(" ") and not next_line.startswith("\t"):
                    break
                
                # Collect indented continuation lines
                if next_stripped:
                    declaration_lines.append(next_stripped)
                
                i += 1
            
            # Combine all lines into the full declaration
            full_declaration = "\n".join(declaration_lines)
            #print("(----)",full_declaration)
            
            # Extract the part before " : " for names
            first_line_before_colon = declaration_lines[0][:type_start_idx].strip()
            
            # Extract the type part (everything after " : " on first line, plus all continuation lines)
            type_part = declaration_lines[0][min(type_start_idx + 3, len(declaration_lines[0])-1):].strip()
            if len(declaration_lines) > 1:
                # If there are continuation lines, join them with newlines
                type_part += " \n " + " \n ".join(declaration_lines[1:])
            
            # Split line in case of multiple variables
            var_type_pairs = split_multi_variable_line("\n".join(declaration_lines))
            #print("(var_type_pairs)",var_type_pairs)
            for name, _ in var_type_pairs:
                # Include all declarations (including shadowed ones with ✝)
                # Shadowed declarations will be handled separately in renaming
                
                # For multi-line definitions, we need to reconstruct the type
                # by taking everything after the name declaration
                if len(var_type_pairs) == 1:
                    # Single variable, use the full type part we extracted
                    declarations.append((name, type_part, full_declaration))
                else:
                    # Multiple variables on same line - they all share the same type
                    # Extract type from the original line
                    _, original_type = var_type_pairs[0]
                    if len(declaration_lines) > 1:
                        # Multi-line, append continuation lines
                        full_type = original_type + " \n " + " \n ".join(declaration_lines[1:])
                    else:
                        full_type = original_type
                    declarations.append((name, full_type, full_declaration))
            
            continue
        
        i += 1
    
    return declarations


def extract_recognized_names(state: str) -> List[Set[str]]:
    """
    Extract all recognized names (variables and hypotheses) from a state.
    
    This builds a set of all names that can be referenced in type declarations,
    separately for each case.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        List of sets, where each set contains all recognized names for that case
    """
    declarations_list = extract_all_declarations(state)
    return [{name for name, _, _ in declarations} for declarations in declarations_list]


def is_name_mentioned(name: str, text: str) -> bool:
    """
    Check if a name is mentioned in the given text.
    
    Uses word boundaries to avoid partial matches (e.g., "x" in "exists").
    
    Args:
        name: The name to search for
        text: The text to search in
    
    Returns:
        True if the name appears in the text as a standalone word
    """
    # Use word boundaries to match whole words only
    # In Lean, we need to be careful with special characters
    # Pattern matches word boundaries around the name
    pattern = r'\b' + re.escape(name) + r'\b'
    return bool(re.search(pattern, text))


def classify_declarations(state: str) -> Tuple[List[List[Tuple[str, str, str]]], List[List[Tuple[str, str, str]]]]:
    """
    Classify declarations into variables and hypotheses.
    
    A declaration is a hypothesis if:
    1. Its type mentions another recognized name, OR
    2. Its type is exactly "True" or "False"
    Otherwise, it is a variable.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        Tuple of (variables_list, hypotheses_list) where each is a list of lists,
        with each inner list containing (name, type, full_line) tuples for one case
    """
    declarations_list = extract_all_declarations(state)
    recognized_names_list = extract_recognized_names(state)
    
    variables_list = []
    hypotheses_list = []
    
    # Process each case separately
    for declarations, recognized_names in zip(declarations_list, recognized_names_list):
        variables = []
        hypotheses = []
        
        # First pass: collect all declarations with their types
        # Skip shadowed declarations (contain ✝) - they won't be renamed
        for name, type_decl, full_line in declarations:
            # Skip shadowed declarations for classification (they won't be renamed)
            if "✝" in name:
                continue
            
            # Remove this name from recognized names to avoid self-reference
            other_names = recognized_names - {name}
            
            # Check if type is exactly "True" or "False" (stripped, no newlines)
            type_stripped = type_decl.strip()
            is_true_or_false = type_stripped == "True" or type_stripped == "False"
            
            # Check if type mentions any other recognized name
            mentions_other_name = any(is_name_mentioned(other_name, type_decl) 
                                      for other_name in other_names)
            
            is_hypothesis = is_true_or_false or mentions_other_name
            indicators = [
                " = ",
                " > ",
                " < ",
                " ≠ ",
                " ≤ ",
                " ≥ ",
                " ↔ ",
            ]
            if any(indicator in type_decl for indicator in indicators):
                is_hypothesis = True
            
            if is_hypothesis:
                hypotheses.append((name, type_decl, full_line))
            else:
                variables.append((name, type_decl, full_line))
        
        variables_list.append(variables)
        hypotheses_list.append(hypotheses)
    
    return variables_list, hypotheses_list


def get_variable_names(state: str) -> List[List[str]]:
    """
    Get a list of all variable names in the state, separately for each case.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        List of lists, where each inner list contains variable names for that case
    """
    variables_list, _ = classify_declarations(state)
    return [[name for name, _, _ in variables] for variables in variables_list]


def get_hypothesis_names(state: str) -> List[List[str]]:
    """
    Get a list of all hypothesis names in the state, separately for each case.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        List of lists, where each inner list contains hypothesis names for that case
    """
    _, hypotheses_list = classify_declarations(state)
    return [[name for name, _, _ in hypotheses] for hypotheses in hypotheses_list]


def hyps_from_goal(goal: str) -> List[str]:
    """Hypothesis names for the first case of the given goal state, or [] if none."""
    cases = get_hypothesis_names(goal)
    return list(cases[0]) if cases else []


def get_all_names(state: str) -> List[List[str]]:
    """
    Get a list of all names (variables and hypotheses) in the state, separately for each case.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        List of lists, where each inner list contains all names for that case
    """
    variables_list, hypotheses_list = classify_declarations(state)
    return [
        [name for name, _, _ in variables] + [name for name, _, _ in hypotheses]
        for variables, hypotheses in zip(variables_list, hypotheses_list)
    ]


def get_variable_definitions(state: str) -> List[Dict[str, str]]:
    """
    Get a dictionary mapping variable names to their type definitions, separately for each case.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        List of dictionaries, where each dictionary maps variable names to their types for that case
    """
    variables_list, _ = classify_declarations(state)
    return [{name: type_decl for name, type_decl, _ in variables} for variables in variables_list]


def get_hypothesis_definitions(state: str) -> List[Dict[str, str]]:
    """
    Get a dictionary mapping hypothesis names to their type definitions, separately for each case.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        List of dictionaries, where each dictionary maps hypothesis names to their types for that case
    """
    _, hypotheses_list = classify_declarations(state)
    return [{name: type_decl for name, type_decl, _ in hypotheses} for hypotheses in hypotheses_list]


def get_all_definitions(state: str) -> List[Dict[str, str]]:
    """
    Get a dictionary mapping all names (variables and hypotheses) to their type definitions,
    separately for each case.
    
    Args:
        state: The Lean proof state as a string
    
    Returns:
        List of dictionaries, where each dictionary maps all names to their types for that case
    """
    variables_list, hypotheses_list = classify_declarations(state)
    result_list = []
    for variables, hypotheses in zip(variables_list, hypotheses_list):
        result = {name: type_decl for name, type_decl, _ in variables}
        result.update({name: type_decl for name, type_decl, _ in hypotheses})
        result_list.append(result)
    return result_list


# ========== Renaming (number allocation, renaming maps) ==========


def get_next_number(used_numbers: Set[int], start: int = 0) -> int:
    """Find the first unused number starting from start."""
    num = start
    while num in used_numbers:
        num += 1
    return num


def number_to_subscript(num: int) -> str:
    """Convert a number to subscript Unicode characters."""
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    return ''.join(subscript_map.get(digit, digit) for digit in str(num))


def format_name_with_number(prefix: str, num: int) -> str:
    """Format a name with a subscript number (one digit if < 10, otherwise multiple digits)."""
    subscript = number_to_subscript(num)
    return f"{prefix}{subscript}"


def get_next_prefixed_name(prefix: str, used_numbers: Dict[str, Set[int]]) -> str:
    """
    Allocate the next name for a prefix (e.g. 'h' -> 'h₁') and record it in used_numbers.
    Mutates used_numbers[prefix]. Returns the new name.
    """
    nums = used_numbers.setdefault(prefix, set())
    num = get_next_number(nums)
    nums.add(num)
    return format_name_with_number(prefix, num)


def create_renaming_map(variables: List[tuple], hypotheses: List[tuple]) -> Dict[str, str]:
    """
    Create a renaming mapping for variables and hypotheses in a single case.
    
    Args:
        variables: List of (name, type_decl, full_line) tuples, sorted lexicographically
        hypotheses: List of (name, type_decl, full_line) tuples, sorted lexicographically
    
    Returns:
        Dictionary mapping old names to new names
    """
    renaming_map = {}
    used_numbers: Dict[str, Set[int]] = {
        'a': set(),
        'n': set(),
        'f': set(),
        'x': set(),
        'ih': set(),
        'h': set(),
        'S': set(),
        'P': set(),
        'z': set(),
    }
    
    # Process variables
    for name, type_decl, _ in variables:
        type_stripped = type_decl.strip()
        
        # Check if type contains -> (function type) - check this first
        # Function types take precedence over ℕ types
        is_function = '→' in type_stripped
        
        # Check if type is ℕ (look for \N or ℕ or Nat)
        # Only check this if it's not a function type
        is_nat = bool(re.search(r'ℕ', type_stripped)) if not is_function else False
        is_int = bool(re.search(r'ℤ', type_stripped)) if not is_function else False
        is_complex = bool(re.search(r'ℂ', type_stripped)) if not is_function else False
        is_set = type_stripped.startswith('Set') or type_stripped.startswith('Finset')
        is_polynomial = type_stripped.startswith('Polynomial ')
        
        if is_set:
            prefix = 'S'
        elif is_polynomial:
            prefix = 'P'
        elif is_function:
            prefix = 'f'
        elif is_nat:
            prefix = 'a'
        elif is_int:
            prefix = 'n'
        elif is_complex:
            prefix = 'z'
        else:
            prefix = 'x'
        
        num = get_next_number(used_numbers[prefix])
        used_numbers[prefix].add(num)
        new_name = format_name_with_number(prefix, num)
        renaming_map[name] = new_name
    
    # Process hypotheses
    for name, type_decl, _ in hypotheses:
        if name.startswith('ih'):
            prefix = 'ih'
        else:
            prefix = 'h'
        
        num = get_next_number(used_numbers[prefix])
        used_numbers[prefix].add(num)
        new_name = format_name_with_number(prefix, num)
        renaming_map[name] = new_name
    
    return renaming_map


# ========== Rewriting ==========
# rwcnc: goal manipulation (hypothesis type replacement), Try this diagnostic parsing,
# deduplicate by state, top-k by shortest state. Used by RewritingCanonicalizationModule.


REWRITES_IMPORT = "import rwens.lean.Rewrites"


def ensure_rewrites_import(content: str) -> str:
    """Ensure Lean content has rwens.lean.Rewrites import (for rwcnc tactic)."""
    if REWRITES_IMPORT in content or "rwens.lean.Rewrites" in content:
        return content
    if "import Mathlib\n" in content:
        return content.replace(
            "import Mathlib\n", "import Mathlib\n" + REWRITES_IMPORT + "\n", 1
        )
    return REWRITES_IMPORT + "\n" + content


def get_diagnostic_message(d: Any) -> str:
    """Extract message string from LSP diagnostic (dict with 'message' or string)."""
    if isinstance(d, dict):
        return (d.get("message") or str(d)).strip()
    return str(d).strip()


# Optional complexity/depth from rwcnc "complexity: X->Y, depth: A->B" (Y, B)
_COMPLEXITY_DEPTH_RE = re.compile(
    r"complexity:\s*\d+->(\d+).*depth:\s*\d+->(\d+)", re.IGNORECASE
)


def _parse_complexity_depth(line: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse 'complexity: X->Y, depth: A->B' to (Y, B). Returns (None, None) if no match."""
    m = _COMPLEXITY_DEPTH_RE.search(line)
    if not m:
        return (None, None)
    try:
        return (int(m.group(1)), int(m.group(2)))
    except (ValueError, IndexError):
        return (None, None)


def parse_try_this_diagnostic(
    msg: str,
) -> List[Tuple[str, str, Optional[int], Optional[int]]]:
    """
    Parse one or more "Try this: rw [...]" / "-[rwcnc]- TYPE" from a diagnostic string.
    Handles the current rwcnc output: first -[rwcnc]- is the state (type_str), optional
    second -[rwcnc]- is "complexity: X->Y, depth: A->B"; Y and B are returned as compl/depth.
    Returns list of (tactic_str, type_str, new_complexity, new_depth).
    """
    out: List[Tuple[str, str, Optional[int], Optional[int]]] = []
    lines = msg.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Try this: rw [" in line:
            try_this_idx = line.find("Try this: ")
            if try_this_idx != -1:
                tactic_str = line[try_this_idx + len("Try this: ") :].strip()
                type_str = ""
                compl: Optional[int] = None
                depth: Optional[int] = None
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("-[rwcnc]-"):
                    candidate = lines[i + 1].strip()[len("-[rwcnc]-") :].strip()
                    if not candidate.startswith("complexity:"):
                        type_str = candidate
                    i += 1
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith("-[rwcnc]-"):
                        next_content = lines[i + 1].strip()[len("-[rwcnc]-") :].strip()
                        if next_content.startswith("complexity:"):
                            compl, depth = _parse_complexity_depth(next_content)
                            i += 1
                out.append((tactic_str, type_str, compl, depth))
        i += 1
    return out


def collect_try_this_from_diagnostics(
    diagnostics: List[Any],
) -> List[Tuple[str, str, Optional[int], Optional[int]]]:
    """From LSP diagnostics list, collect (tactic_str, type_str, compl, depth) from messages containing 'Try this:'."""
    raw: List[Tuple[str, str, Optional[int], Optional[int]]] = []
    for d in diagnostics or []:
        s = get_diagnostic_message(d)
        if "Try this:" in s:
            raw.extend(parse_try_this_diagnostic(s))
    return raw


def deduplicate_by_state(
    suggestions: List[Tuple[str, str, Optional[int], Optional[int]]],
) -> List[Tuple[str, str, Optional[int], Optional[int]]]:
    """Keep one suggestion per resulting type (state), with shortest tactic; preserve compl/depth."""
    by_type: Dict[str, Tuple[str, str, Optional[int], Optional[int]]] = {}
    for item in suggestions:
        tactic, type_str, compl, depth = item
        type_str = type_str.strip()
        if type_str not in by_type or len(tactic) < len(by_type[type_str][0]):
            by_type[type_str] = (tactic, type_str, compl, depth)
    return list(by_type.values())


def top_k_shortest(
    suggestions: List[Tuple[str, str]], k: int = 5
) -> List[Tuple[str, str]]:
    """Sort by length of resulting state (type string), then tactic for ties; return first k."""
    sorted_suggestions = sorted(suggestions, key=lambda x: (len(x[1]), x[0]))
    return sorted_suggestions[:k]


def get_hypothesis_type(goal: str, hyp_name: str) -> Optional[str]:
    """Return the current type of hypothesis hyp_name in goal, or None if not found."""
    for line in goal.split("\n"):
        stripped = line.strip()
        if stripped.startswith(hyp_name + " : "):
            return stripped[len(hyp_name) + 3 :].strip()
    return None


def replace_hypothesis_in_goal(goal: str, hyp_name: str, new_type: str) -> str:
    """Replace the line 'hyp_name : old_type' in goal with 'hyp_name : new_type'."""
    lines = goal.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("⊢"):
            out.append(line)
            continue
        if stripped.startswith(hyp_name + " : "):
            prefix = line[: line.index(stripped)]
            out.append(prefix + hyp_name + " : " + new_type)
            continue
        out.append(line)
    return "\n".join(out)


def get_goal_expression(goal: str) -> str:
    """Return the goal expression (the part after ⊢). From first ⊢ line to end of state."""
    lines = goal.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("⊢"):
            rest = stripped[1:].strip()  # after ⊢
            remaining_lines = [rest] + [ln.strip() for ln in lines[i + 1 :] if ln.strip()]
            return " ".join(remaining_lines).strip()
    return ""


def replace_goal_in_goal(goal: str, new_goal_expr: str) -> str:
    """Replace the ⊢ part (from first line starting with ⊢ to end) with ⊢ new_goal_expr."""
    lines = goal.split("\n")
    out = []
    seen_goal = False
    for line in lines:
        stripped = line.strip()
        if not seen_goal and stripped.startswith("⊢"):
            seen_goal = True
            prefix = line[: line.index(stripped)]
            out.append(prefix + "⊢ " + new_goal_expr.strip())
            continue
        if seen_goal:
            continue
        out.append(line)
    return "\n".join(out)


def build_states_for_goal(
    goal: str,
    top_rewrites: List[Tuple[str, str]],
    top_k: int = 10,
) -> List[Tuple[str, str, str]]:
    """
    Build list of (state_str, rw_tactic_or_None, rw_type_or_None) for goal rewrites.
    First element is (original_goal, None, None); then up to top_k states with goal replaced.
    """
    states: List[Tuple[str, str, str]] = [(goal, None, None)]  # type: ignore[assignment]
    for tactic_str, type_str in top_rewrites[:top_k]:
        aug_goal = replace_goal_in_goal(goal, type_str.strip())
        states.append((aug_goal, tactic_str, type_str.strip()))
    return states


def build_states_for_hypothesis(
    goal: str,
    hyp_name: str,
    top_rewrites: List[Tuple[str, str]],
    top_k: int = 10,
) -> List[Tuple[str, str, str]]:
    """
    Build list of (state_str, rw_tactic_or_None, rw_type_or_None).
    First element is (original_goal, None, None); then up to top_k augmented states.
    """
    states: List[Tuple[str, str, str]] = [(goal, None, None)]  # type: ignore[assignment]
    for tactic_str, type_str in top_rewrites[:top_k]:
        aug_goal = replace_hypothesis_in_goal(goal, hyp_name, type_str.strip())
        states.append((aug_goal, tactic_str, type_str.strip()))
    return states


# Key used in best_type_per_hyp for the goal (rewritten goal expression).
GOAL_REWRITE_KEY = "_goal_"


def build_fully_augmented_goal_from_best_types(
    goal: str,
    hyps: List[str],
    best_type_per_hyp: Dict[str, str],
) -> str:
    """Build goal with optional goal rewrite and each hypothesis replaced by its given type."""
    current = goal
    if GOAL_REWRITE_KEY in best_type_per_hyp:
        current = replace_goal_in_goal(current, best_type_per_hyp[GOAL_REWRITE_KEY])
    for hyp in hyps:
        if hyp in best_type_per_hyp:
            current = replace_hypothesis_in_goal(
                current, hyp, best_type_per_hyp[hyp]
            )
    return current


# ========== Tactic ==========
# Tactic-related helpers (e.g. parsing/formatting tactics). None in this module.


# ========== Other ==========
# Miscellaneous helpers that do not fit the above categories. None in this module.

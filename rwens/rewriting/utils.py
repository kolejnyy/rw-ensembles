"""
Utilities for parsing Lean proof states and supporting the rwcnc rewriting pipeline.

Includes declaration extraction (shared with ``rwens.utils.state``), hypothesis listing
for rewriting, diagnostic parsing, and goal augmentation helpers.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def split_multi_variable_line(line: str) -> List[Tuple[str, str]]:
    """
    Split a line that may contain multiple variables of the same type.

    For example, "x y : ℝ" becomes [("x", "ℝ"), ("y", "ℝ")].
    """
    if " : " not in line and " :\n" not in line:
        return []

    parts = line.split(" : ", 1)
    parts_newline = line.split(" :\n", 1)

    if len(parts[0]) > len(parts_newline[0]):
        parts = parts_newline

    if len(parts) != 2:
        return []

    names_part = parts[0].strip()
    type_part = parts[1].strip()

    names = names_part.split()

    return [(name, type_part) for name in names if name]


def extract_all_declarations(state: str) -> List[List[Tuple[str, str, str]]]:
    """
    Extract all variable and hypothesis declarations from a Lean state.

    Variables and hypotheses are at the beginning of new lines (no indentation)
    and are followed by " : ". Multi-line definitions are supported by collecting
    all indented continuation lines after a declaration.

    If the state contains multiple cases separated by blank lines (\\n\\n), they
    are processed separately, with each case returning its own list of declarations.
    """
    state_parts = [part.strip() for part in state.split("\n\n") if part.strip()]

    if not state_parts:
        return []

    all_declarations = []
    for state_part in state_parts:
        declarations = _extract_declarations_from_part(state_part)
        all_declarations.append(declarations)

    return all_declarations


def _extract_declarations_from_part(state_part: str) -> List[Tuple[str, str, str]]:
    """Extract declarations from a single state part (one case/goal)."""
    declarations = []
    lines = state_part.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("⊢"):
            i += 1
            continue

        if (
            not line.startswith(" ")
            and not line.startswith("\t")
            and stripped.startswith("case ")
        ):
            i += 1
            continue

        if not line.startswith(" ") and not line.startswith("\t") and (
            " : " in line or line.endswith(" :")
        ):
            declaration_lines = [stripped]
            type_start_idx = stripped.find(" : ")
            if type_start_idx == -1:
                type_start_idx = line.find(" :")

            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()

                if next_stripped.startswith("⊢"):
                    break

                if next_stripped and not next_line.startswith(" ") and not next_line.startswith("\t"):
                    break

                if next_stripped:
                    declaration_lines.append(next_stripped)

                i += 1

            full_declaration = "\n".join(declaration_lines)

            first_line_before_colon = declaration_lines[0][:type_start_idx].strip()

            type_part = declaration_lines[0][
                min(type_start_idx + 3, len(declaration_lines[0]) - 1) :
            ].strip()
            if len(declaration_lines) > 1:
                type_part += " \n " + " \n ".join(declaration_lines[1:])

            var_type_pairs = split_multi_variable_line("\n".join(declaration_lines))
            for name, _ in var_type_pairs:
                if len(var_type_pairs) == 1:
                    declarations.append((name, type_part, full_declaration))
                else:
                    _, original_type = var_type_pairs[0]
                    if len(declaration_lines) > 1:
                        full_type = original_type + " \n " + " \n ".join(declaration_lines[1:])
                    else:
                        full_type = original_type
                    declarations.append((name, full_type, full_declaration))

            continue

        i += 1

    return declarations


def extract_recognized_names(state: str) -> List[Set[str]]:
    """For each case, the set of declared names (variables and hypotheses)."""
    declarations_list = extract_all_declarations(state)
    return [{name for name, _, _ in declarations} for declarations in declarations_list]


def is_name_mentioned(name: str, text: str) -> bool:
    """True if ``name`` appears in ``text`` as a standalone token."""
    pattern = r"\b" + re.escape(name) + r"\b"
    return bool(re.search(pattern, text))


def classify_declarations(
    state: str,
) -> Tuple[List[List[Tuple[str, str, str]]], List[List[Tuple[str, str, str]]]]:
    """
    Classify declarations into variables and hypotheses (per case).

    A declaration is a hypothesis if its type mentions another recognized name,
    or contains certain relational markers, or is exactly True/False.
    """
    declarations_list = extract_all_declarations(state)
    recognized_names_list = extract_recognized_names(state)

    variables_list = []
    hypotheses_list = []

    for declarations, recognized_names in zip(declarations_list, recognized_names_list):
        variables = []
        hypotheses = []

        for name, type_decl, full_line in declarations:
            if "✝" in name:
                continue

            other_names = recognized_names - {name}

            type_stripped = type_decl.strip()
            is_true_or_false = type_stripped == "True" or type_stripped == "False"

            mentions_other_name = any(
                is_name_mentioned(other_name, type_decl) for other_name in other_names
            )

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


def get_hypothesis_names(state: str) -> List[List[str]]:
    """Hypothesis names per case."""
    _, hypotheses_list = classify_declarations(state)
    return [[name for name, _, _ in hypotheses] for hypotheses in hypotheses_list]


def hyps_from_goal(goal: str) -> List[str]:
    """Hypothesis names for the first case of the given goal state, or [] if none."""
    cases = get_hypothesis_names(goal)
    return list(cases[0]) if cases else []


# ========== Rewriting (rwcnc helpers) ==========

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
    """From LSP diagnostics list, collect (tactic_str, type_str, compl, depth)."""
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
            rest = stripped[1:].strip()
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

"""
Convert Lean proof state strings into full problem statements.

StateProblemConverter performs state -> problem transformation and verifies
via round-trip (problem -> extract state -> compare). Holds a persistent
Lean session (sfc) for efficiency.

Used by CanonicalLLMProver: each candidate state from rewriting is turned into
a problem statement so a single-pass LLM can be run on it for energy scoring.
"""

import re
from pathlib import Path
from typing import Optional

from rwens.dataset.utils import split_declarations_theorem_proof


class StateProblemConverter:
    """
    Converts Lean goal states to problem statements and verifies via round-trip.

    Holds a persistent Lean session (sfc) so Mathlib and other imports stay
    loaded. Converts state -> problem, writes to Lean, extracts state again,
    and returns the problem only if states match.
    """

    # Pairs (from_str, to_str) to normalize notation in states. May grow over time.
    REPLACEMENTS: list[tuple[str, str]] = [
        ("π", "Real.pi"),
        ("√", "Real.sqrt "),  # √x -> Real.sqrt x; √(e) -> Real.sqrt (e)
    ]

    @staticmethod
    def _replace_logb_method(text: str) -> str:
        """Convert <exp1>.logb <exp2> to Real.logb <exp1> <exp2>."""
        return re.sub(
            r"(\([^()]*\)|[a-zA-Z0-9_']+)\s*\.logb\s+",
            r"Real.logb \1 ",
            text,
        )

    @staticmethod
    def _apply_replacements(text: str) -> str:
        text = StateProblemConverter._replace_logb_method(text)
        for from_str, to_str in StateProblemConverter.REPLACEMENTS:
            text = text.replace(from_str, to_str)
        return text

    @staticmethod
    def state_to_problem_pure(
        imports: str,
        state_str: str,
        theorem_name: Optional[str] = None,
    ) -> str:
        """
        Convert state to problem string (no round-trip verification).
        Raises ValueError if state has no goal.

        Usable without an instance when round-trip verification is not needed.
        """
        name = theorem_name or "anon"
        lines = state_str.strip().split("\n")
        hyps: list[str] = []
        goal_parts: list[str] = []
        imports = imports.replace("import rwens.lean.Rewrites\n", "")
        in_goal = False
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                if in_goal:
                    goal_parts.append("")
                continue
            if stripped.startswith("⊢"):
                in_goal = True
                goal_parts.append(stripped[1:].strip())
            elif in_goal:
                goal_parts.append(stripped)
            else:
                hyps.append(stripped)
        goal = "\n".join(goal_parts).strip()
        if not goal:
            raise ValueError("state_str has no goal (no line starting with ⊢)")
        goal = StateProblemConverter._apply_replacements(goal)
        for i, h in enumerate(hyps):
            hyps[i] = StateProblemConverter._apply_replacements(h)
        hyp_part = " ".join(f"({h})" for h in hyps)
        theorem_part = f"theorem {name} {hyp_part} : {goal} := by\n"
        if hyp_part == "":
            theorem_part = f"theorem {name} : {goal} := by\n"
        return imports.rstrip("\n") + "\n" + theorem_part

    @staticmethod
    def extract_theorem_name(theorem_statement: str) -> Optional[str]:
        """
        Extract the theorem/lemma/def name from a statement.

        E.g. "theorem mathd_123 (n : Nat) : n + 0 = n := by" -> "mathd_123"
        """
        m = re.match(
            r"^(?:theorem|lemma|def)\s+([a-zA-Z0-9_']+)",
            theorem_statement.strip(),
        )
        return m.group(1) if m else None

    def __init__(
        self,
        project_root: str | Path,
        timeout_seconds: float = 120.0,
    ) -> None:
        # Lazy import avoids pulling rewriting internals when only helpers are needed.
        from rwens.utils.plain_lean_session import PlainLeanSession

        self._module = PlainLeanSession(
            project_root=str(project_root),
            timeout_seconds=timeout_seconds,
        )

    def convert(
        self,
        imports: str,
        state_str: str,
        theorem_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convert state to problem, verify round-trip, return problem or None.

        Returns None if state contains ✝/sorry, round-trip fails, or states
        don't match.
        """
        if "✝" in state_str or "sorry" in state_str:
            return None
        try:
            problem = self.state_to_problem_pure(imports, state_str, theorem_name)
        except ValueError:
            return None
        try:
            decls, theorem_stmt, _ = split_declarations_theorem_proof(problem)
        except ValueError:
            return None
        self._module.reset(decls, theorem_stmt)
        round_trip_state = self._module.get_current_state()
        if round_trip_state is None or not round_trip_state.strip():
            return None
        if state_str.strip() != round_trip_state.strip():
            return None
        return problem

    def convert_no_compile(
        self,
        imports: str,
        state_str: str,
        theorem_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Convert state to problem without compiling or round-trip validation.

        Returns None only for clearly invalid states (contains ✝/sorry or no goal).
        """
        if "✝" in state_str or "sorry" in state_str:
            return None
        try:
            return self.state_to_problem_pure(imports, state_str, theorem_name)
        except ValueError:
            return None

    def close(self) -> None:
        """Clean up the Lean session."""
        self._module.close()


# Convenience alias for external callers
extract_theorem_name = StateProblemConverter.extract_theorem_name

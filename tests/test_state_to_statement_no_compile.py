#!/usr/bin/env python3
"""
Smoke tests for StateProblemConverter.convert_no_compile on 5 examples.
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from rwens.dataset.utils import split_declarations_theorem_proof
from rwens.utils.state_to_statement import StateProblemConverter


def main() -> int:
    imports = "import Mathlib\n"
    converter = StateProblemConverter(project_root=str(_project_root), timeout_seconds=30.0)

    # 5 representative states (goal-only, hypotheses, multiline goal, unicode replacement, logb method).
    examples: list[tuple[str, str]] = [
        (
            "ex_goal_only",
            "⊢ 1 + 1 = 2",
        ),
        (
            "ex_with_hyps",
            "n : Nat\nh : n = 3\n⊢ n + 1 = 4",
        ),
        (
            "ex_multiline_goal",
            "α : Type\ninst : Add α\n⊢\n  True\n  ∧ True",
        ),
        (
            "ex_unicode_replacement",
            "x : Real\n⊢ √x = Real.sqrt x",
        ),
        (
            "ex_logb_method",
            "x : Real\ny : Real\n⊢ x.logb y = Real.logb x y",
        ),
        (
            "ex_state_headers",
            "2 goals\ncase h.main\nx : Real\n⊢ π = π",
        ),
        (
            "ex_sqrt_with_parens",
            "x : Real\n⊢ √(x + 1) = Real.sqrt (x + 1)",
        ),
    ]

    failures = 0
    try:
        for name, state in examples:
            out = converter.convert_no_compile(imports, state, theorem_name=name)
            if out is None:
                failures += 1
                print(f"[FAIL] {name}: returned None")
                continue
            try:
                _decls, theorem_stmt, _proof = split_declarations_theorem_proof(out)
            except ValueError as e:
                failures += 1
                print(f"[FAIL] {name}: produced unparsable theorem ({e})")
                continue
            if f"theorem {name}" not in theorem_stmt:
                failures += 1
                print(f"[FAIL] {name}: theorem name mismatch in output")
                continue
            if name == "ex_state_headers" and "2 goals" in theorem_stmt:
                failures += 1
                print(f"[FAIL] {name}: leaked state header into theorem statement")
                continue
            print(f"[OK] {name}")
    finally:
        converter.close()

    print(f"Done: {len(examples) - failures}/{len(examples)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


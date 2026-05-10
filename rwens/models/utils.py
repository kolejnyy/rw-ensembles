"""
Auxiliary utilities for model-related code (e.g. merging proof into statement).
"""

from __future__ import annotations

import re

from rwens.dataset.utils import split_declarations_theorem_proof

# For ProofNet-style problems where statement ends with ":=" (no " by")
_DECL_START_RE = re.compile(r"(?m)^(theorem|lemma|def)\b")


def merge_proof_into_statement(problem_statement: str, proof_body: str) -> str:
    """Append the model's proof body (everything after the theorem statement) to the problem statement.
    When the problem ends with ':=' (ProofNet-style, no ' by'), appends ' by\\n' before the proof body.
    """
    raw = proof_body.rstrip("\n\r").lstrip("\n\r")
    try:
        decls, theorem_stmt, _ = split_declarations_theorem_proof(problem_statement)
    except ValueError:
        # Problem may end with ":=" (no " by"), e.g. ProofNet formal_statement
        normalized = problem_statement.replace("\r\n", "\n")
        if normalized.rstrip().endswith(":="):
            m = _DECL_START_RE.search(normalized)
            if m:
                decls = normalized[: m.start()].rstrip("\n") + "\n"
                theorem_stmt = normalized[m.start() :].rstrip("\n")
                body = raw if raw.endswith("\n") else raw + "\n"
                return decls + theorem_stmt + " by\n" + body
        return problem_statement.rstrip("\n") + "\n\n" + raw
    body = raw if raw.endswith("\n") else raw + "\n"
    return decls + theorem_stmt + body

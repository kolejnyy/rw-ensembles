"""
RWData-style equivalence **certificate**: one Lean file proving the original from the variant.

Builds the ``rwens.lean`` layout: preamble, variant with ``sorry``, auxiliary lemmas with
proof bodies, then the original theorem with composed ``have`` / ``exact`` bridge steps.
"""

from __future__ import annotations

from rwens.utils.lean_proof_text import (
    normalize_proof_indent,
    theorem_header_through_by,
    theorem_with_sorry,
)
from rwens.equivalence.types import AuxiliaryGoalsBundle, AuxiliaryVerificationResult
from rwens.utils.statement_parser import (
    StatementParser,
    all_binder_name_tokens,
    parse_theorem_through_by,
    variable_binder_name_tokens,
)


def compose_rwdata_bridge_proof_lines(
    original_theorem: str,
    variant_theorem: str,
    *,
    variant_name_override: str | None = None,
) -> list[str]:
    """
    RWData-style ``have aug_i …``, ``have aug_goal …``, ``exact goal_to_goal …`` lines
    (each line includes leading two spaces).
    """
    orig_tokens = all_binder_name_tokens(original_theorem)
    orig_var_tokens = variable_binder_name_tokens(original_theorem)
    var_toks = variable_binder_name_tokens(variant_theorem)
    hyps = StatementParser.hypotheses(variant_theorem)
    var_parsed = parse_theorem_through_by(variant_theorem)
    variant_name = variant_name_override or var_parsed.name
    vc = var_parsed.conclusion.strip()
    ot = " ".join(orig_tokens)
    lines: list[str] = []
    for i, h in enumerate(hyps):
        typ = h.type_str.strip()
        lines.append(f"  have aug_{i} : {typ} := cond_{i + 1} {ot}")
    aug_refs = " ".join(f"aug_{j}" for j in range(len(hyps)))
    vt = " ".join(var_toks)
    lines.append(f"  have aug_goal : {vc} := {variant_name} {vt} {aug_refs}")
    gt_args = " ".join(orig_var_tokens + ["aug_goal"])
    lines.append(f"  exact goal_to_goal {gt_args}")
    return lines


def build_equivalence_certificate_lean(
    preamble: str,
    original_theorem: str,
    variant_theorem: str,
    bundle: AuxiliaryGoalsBundle,
    auxiliary_results: list[AuxiliaryVerificationResult],
    *,
    bridge_variant_theorem: str | None = None,
) -> str:
    """
    Build a full Lean file: header, augmented theorem with ``sorry``, auxiliary theorems
    with stored proof bodies, then the original theorem with the composed RWData bridge.

    Requires every entry in ``auxiliary_results`` to be successful and to include
    :attr:`AuxiliaryVerificationResult.verified_proof_body`.
    """
    if len(bundle.goals) != len(auxiliary_results):
        raise ValueError("bundle.goals and auxiliary_results length mismatch.")
    for r in auxiliary_results:
        if not r.success or r.verified_proof_body is None:
            raise ValueError(
                "Certificate needs all auxiliary goals succeeded with a stored proof body."
            )

    parts: list[str] = [preamble.rstrip() + "\n\n"]
    parts.append(theorem_with_sorry(variant_theorem))
    parts.append("\n")

    for goal, res in zip(bundle.goals, auxiliary_results):
        header = theorem_header_through_by(goal.theorem_statement)
        body = normalize_proof_indent(res.verified_proof_body or "")
        parts.append(header)
        parts.append(body)
        parts.append("\n")

    orig_parsed = parse_theorem_through_by(original_theorem)
    orig_head = orig_parsed.raw_through_by.rstrip()
    if not orig_head.endswith("by"):
        raise ValueError("Original theorem must end with := by")
    bridge_src = bridge_variant_theorem or variant_theorem
    displayed_variant_name = parse_theorem_through_by(variant_theorem).name
    bridge = "\n".join(
        compose_rwdata_bridge_proof_lines(
            original_theorem,
            bridge_src,
            variant_name_override=displayed_variant_name,
        )
    )
    parts.append(orig_head + "\n" + bridge + "\n")
    return "".join(parts)

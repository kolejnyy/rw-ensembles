"""
Metrics utilities for evaluating formal theorem provers.

This module provides functions for calculating evaluation metrics such as PASS@k
and interpreting Lean diagnostics for proof success.
"""

import logging
import math
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Benign diagnostic substrings that should not cause verification to fail.
# Also used when recomputing PASS@k: failures with only benign errors count as success.
BENIGN_DIAGNOSTIC_SUBSTRINGS = [
    "unused variable",
    "has been deprecated",
    "no goals to be solved",
    "set_option linter.unnecessarySeqFocus false",
    "set_option linter.unusedTactic false",
    "set_option linter.unreachableTactic false",
    "Try this:",
    "tactic does nothing",  # linter.unusedTactic; proof may still be correct
]

# Extra benign messages for RWData **certificate** files that intentionally leave the
# variant theorem at ``sorry`` (handled by :func:`equivalence_certificate_diagnostics_acceptable`).
EQUIVALENCE_CERTIFICATE_EXTRA_BENIGN_SUBSTRINGS = (
    "declaration uses 'sorry'",
    "uses 'sorry'",
)

# Diagnostic phrases indicating an incorrectly applied tactic when state is unchanged
# or that the proof did not succeed (unsolved goals, omega/linarith failure).
TACTIC_FAILURE_PHRASES = (
    "invalid field",
    "type mismatch",
    "unknown identifier",
    "failed to find a contradiction",
    "simp made no progress",
    "unknown constant",
    "tactic 'rewrite' failed",
    "tactic 'apply' failed",
    "could not prove",
    "linarith failed",
    "unsolved goals",
    "uses 'sorry'",  # declaration uses 'sorry' = not a real proof
    "unexpected token",  # parse/syntax error, e.g. "unexpected token 'have'; expected command"
    "Lean diagnostics indicated failure"
)


def get_tactic_failure_diagnostic(diagnostics_result: Any) -> Optional[Any]:
    """
    Return the first diagnostic that contains a tactic failure phrase, or None.

    Used when state_before == state_after to detect incorrectly applied tactics
    (e.g. omega, linarith, rewrite) that produced an error.
    """
    diagnostics = getattr(diagnostics_result, "diagnostics", None) or []
    for d in diagnostics:
        d_lower = str(d).lower()
        if any(phrase in d_lower for phrase in TACTIC_FAILURE_PHRASES):
            return d
    return None


def _diagnostic_text_for_matching(d: Any) -> str:
    if isinstance(d, dict):
        m = d.get("message")
        if m is not None:
            return str(m)
    return str(d)


def equivalence_certificate_diagnostics_acceptable(diagnostics_result: Any) -> bool:
    """
    Whether Lean diagnostics are acceptable for a composed RWData certificate file.

    The certificate deliberately contains ``sorry`` on the variant theorem; Lean then
    emits ``declaration uses 'sorry'``. Together with :data:`BENIGN_DIAGNOSTIC_SUBSTRINGS`
    (unused variables, etc.), those are treated as **accepted** for end-to-end
    equivalence status. Any other diagnostic fails acceptance.
    """
    diagnostics = getattr(diagnostics_result, "diagnostics", None) or []
    if not diagnostics:
        return getattr(diagnostics_result, "success", False)

    def _msg_ok(msg_lower: str) -> bool:
        if any(b.lower() in msg_lower for b in BENIGN_DIAGNOSTIC_SUBSTRINGS):
            return True
        if any(b.lower() in msg_lower for b in EQUIVALENCE_CERTIFICATE_EXTRA_BENIGN_SUBSTRINGS):
            return True
        return False

    for d in diagnostics:
        text = _diagnostic_text_for_matching(d).lower()
        if not _msg_ok(text):
            return False
    return True


def diagnostics_indicate_success(diagnostics_result: Any) -> bool:
    """
    Return whether a Lean diagnostics result should be treated as proof success.

    True iff diagnostics_result.success is True and there are no non-benign
    diagnostics (e.g. unused variable and linter-option messages are ignored).

    Args:
        diagnostics_result: Object with .success (bool) and .diagnostics (list),
            e.g. the return value of leanclient SingleFileClient.get_diagnostics().

    Returns:
        True if the proof should be considered successful, False otherwise.
    """
    if not getattr(diagnostics_result, "success", False):
        return False
    diagnostics = getattr(diagnostics_result, "diagnostics", None) or []
    for d in diagnostics:
        d_str = str(d).lower() if d else ""
        if any(benign.lower() in d_str for benign in BENIGN_DIAGNOSTIC_SUBSTRINGS):
            continue
        return False
    return True


def pass_at_k(successes: List[bool], k: int) -> float:
    """
    Calculate PASS@k metric: the probability that at least one of k randomly selected attempts succeeds.
    
    The formula is: pass_at_k = 1 - C(n-c, k) / C(n, k)
    where:
    - n = total number of attempts
    - c = number of correct/successful attempts
    - C(n, k) = binomial coefficient (n choose k)
    
    Args:
        successes: List of success booleans for each attempt
        k: Number of attempts to consider
        
    Returns:
        PASS@k value as a float between 0 and 1
    """
    n = len(successes)
    if n == 0:
        return 0.0
    
    # Count successful attempts
    c = sum(1 for s in successes if s)
    
    # If k > n, we can't select k attempts from n, so return 0
    if k > n:
        return 0.0
    
    # If k == 0, return 0 (no attempts to consider)
    if k == 0:
        return 0.0
    
    # If all attempts succeeded, PASS@k = 1
    if c == n:
        return 1.0
    
    # If no attempts succeeded, PASS@k = 0
    if c == 0:
        return 0.0
    
    # Calculate: 1 - C(n-c, k) / C(n, k)
    # This is the probability that at least one of k randomly selected attempts succeeds
    try:
        # Number of ways to choose k attempts that all fail
        ways_all_fail = math.comb(n - c, k) if k <= n - c else 0
        
        # Total number of ways to choose k attempts
        total_ways = math.comb(n, k)
        
        # Probability that at least one succeeds = 1 - probability that all fail
        pass_at_k = 1.0 - (ways_all_fail / total_ways)
        
        return pass_at_k
    except (ValueError, OverflowError):
        # Handle edge cases (e.g., very large numbers)
        # Fallback: if we can't compute, return 0
        return 0.0

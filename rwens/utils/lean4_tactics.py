"""List of Lean 4 tactics for programmatic use."""

# Basic tactics
BASIC_TACTICS = [
    "intro", "intros", "intro!", "revert", "assumption", "exact", "apply", 
    "refine", "constructor", "split", "cases", "induction", "generalize", 
    "specialize", "lemma", "let", "set", "apply_fun", "def", "theorem"
]

# Rewriting tactics
REWRITING_TACTICS = [
    "rw", "rwa", "erw", "simp", "simp only", "simp?", "dsimp", "unfold", 
    "delta", "beta", "zeta", "eta", "rw_mod_cast", "simp_rw"
]

# Equality and congruence tactics
EQUality_TACTICS = [
    "refl", "symm", "trans", "congr", "congr'", "ext", "funext", "propext"
]

# Arithmetic tactics
ARITHMETIC_TACTICS = [
    "norm_num", "ring", "ring_nf", "linarith", "nlinarith", "omega", 
    "field_simp", "cancel_denoms", "linear_combination", "congr!",
    "compute_degree", "compute_degree!"
]

# Logic tactics
LOGIC_TACTICS = [
    "left", "right", "exfalso", "by_contra", "by_contra!", "contrapose", 
    "push_neg", "tauto", "finish", "trivial", "split_ands", "rfl", "lhs",
    "conv", "contrapose!", "exists", "absurd", "rhs", "transitivity"
]

# Set theory and quantifiers
QUANTIFIER_TACTICS = [
    "use", "obtain", "rcases", "rintro", "refine'"
]

# Tactical combinators
COMBINATOR_TACTICS = [
    "<;>", "all_goals", "any_goals", "focus", "rotate_left", 
    "rotate_right", "swap", "clear", "revert_all"
]

# Decision procedures
DECISION_TACTICS = [
    "decide", "trivial", "done", "split_ifs"
]

# Simplification tactics
SIMPLIFICATION_TACTICS = [
    "simp_all", "simp at *", "simp only at", "dsimp at", "norm_cast", 
    "push_cast", "pull_cast", "simpa", "replace", "replace:"
]

# Advanced tactics
ADVANCED_TACTICS = [
    "convert", "convert_to", "change", "show", "have", "let", "set", 
    "suffices", "wlog", "by_cases", "classical", "em", "by_contradiction",
    "exact_mod_cast", "specialize", "repeat", "clear_value",
    "reduce_mod_char", "solve_by_elim", "simp_arith"
]

# Error handling
ERROR_TACTICS = [
    "try", "try_this", "sorry", "admit", "guard_target", "guard_hyp"
]

# Term construction
TERM_TACTICS = [
    "exact", "refine", "apply", "fapply", "eapply", "mapply", "rapply",
    "and_intros", "beta_reduce", "conv_rhs", "conv_lhs", "bound"
]

# Pattern matching
PATTERN_TACTICS = [
    "match", "cases", "cases'", "induction","induction'", "generalize",
    "fin_cases", "mod_cases", "case"
]

# Syntax constructors and separators
SYNTAX_CONSTRUCTORS = [
    "·",  # Bullet point (cdot) for goal focusing
    "|",  # Pipe for pattern matching and case separation
    "{",  # Opening brace for goal focusing
    "}",  # Closing brace for goal focusing
    ".",  # Period for goal focusing
]

# Mathlib-specific tactics
MATHLIB_TACTICS = [
    "norm_num", "ring", "linarith", "nlinarith", "omega", "field_simp", 
    "positivity", "mono", "gcongr", "trans", "apply_rfl", "abel", 
    "noncomm_ring", "norm_cast", "rify", "qify", "zify", "norm_num1",
    "nth_rw", "contradiction", "subst", "replace", "interval_cases",
    "native_decide", "nth_rewrite", "set_option"
]

# Aesop tactics
AESOP_TACTICS = [
    "aesop", "aesop?", "aesop_cat", "aesop_graph"
]

# Domain-specific tactics
DOMAIN_TACTICS = [
    "continuity", "continuity'", "measurability", "measurability'", "tidy"
]

# All tactics combined (deduplicated)
ALL_TACTICS = sorted(set(
    BASIC_TACTICS + 
    REWRITING_TACTICS + 
    EQUality_TACTICS + 
    ARITHMETIC_TACTICS + 
    LOGIC_TACTICS + 
    QUANTIFIER_TACTICS + 
    COMBINATOR_TACTICS + 
    DECISION_TACTICS + 
    SIMPLIFICATION_TACTICS + 
    ADVANCED_TACTICS + 
    ERROR_TACTICS + 
    TERM_TACTICS + 
    PATTERN_TACTICS + 
    MATHLIB_TACTICS + 
    AESOP_TACTICS + 
    DOMAIN_TACTICS + 
    SYNTAX_CONSTRUCTORS
))

# Most commonly used tactics (based on frequency in proofs)
COMMON_TACTICS = [
    "simp", "rw", "apply", "intro", "exact", "have", "use", "rcases", 
    "norm_num", "ring", "linarith", "field_simp", "constructor", "split", 
    "cases", "induction", "by_contra", "assumption", "refl", "ring_nf", 
    "nlinarith", "simp only", "dsimp", "ext", "congr", "left", "right", 
    "tauto", "finish", "trivial", "decide", "omega", "positivity", "mono"
]

if __name__ == "__main__":
    print(f"Total tactics: {len(ALL_TACTICS)}")
    print(f"\nCommon tactics ({len(COMMON_TACTICS)}):")
    for tactic in COMMON_TACTICS:
        print(f"  - {tactic}")
    
    print(f"\nAll tactics:")
    for tactic in ALL_TACTICS:
        print(f"  - {tactic}")

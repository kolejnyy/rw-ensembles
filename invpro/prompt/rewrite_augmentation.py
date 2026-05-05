"""
Prompt formatter for generating equivalent theorem rewrites.
"""

from invpro.prompt.base import PromptFormatter


class RewriteAugmentationPromptFormatter(PromptFormatter):
    """
    Format prompts for generating equivalent Lean theorem statements.

    The formatter uses a principled template:
    - strict output format requirements
    - explicit invariants (semantic equivalence, no renaming)
    - few-shot examples sampled from the existing miniF2F 50x20 rewrite dataset
    """

    FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
        (
            "theorem amc12_2000_p5 (x p : ℝ) (h₀ : x < 2) (h₁ : abs (x - 2) = p) : x - p = 2 - 2 * p := by",
            "theorem amc12_2000_p5_v4 (x p : ℝ) (h₀ : 2 > x) (h₁ : p = abs (2 - x)) :\n"
            "  x = 2 - p := by",
        ),
        (
            "theorem mathd_numbertheory_780 (m x : ℤ) (h₀ : 0 ≤ x) (h₁ : 10 ≤ m ∧ m ≤ 99) (h₂ : 6 * x % m = 1)\n"
            "  (h₃ : (x - 6 ^ 2) % m = 0) : m = 43 := by",
            "theorem mathd_numbertheory_780_v4 (m x : ℤ) (h₀ : x ≥ 0) (h₁ : m ≥ 10 ∧ 99 ≥ m) (h₂ : (6 * x) % m = 1)\n"
            "  (h₃ : m ∣ (x - 36)) : m = 43 := by",
        ),
        (
            "theorem induction_divisibility_9div10tonm1 (n : ℕ) (h₀ : 0 < n) : 9 ∣ 10 ^ n - 1 := by",
            "theorem induction_divisibility_9div10tonm1_v8 (n : ℕ) (h₀ : 0 < n) : 10 ^ n ≡ 1 [MOD 9] := by",
        ),
        (
            "theorem imo_1966_p4 (n : ℕ) (x : ℝ) (h₀ : ∀ k : ℕ, 0 < k → ∀ m : ℤ, x ≠ m * Real.pi / 2 ^ k)\n"
            "  (h₁ : 0 < n) :\n"
            "  (∑ k in Finset.Icc 1 n, 1 / Real.sin (2 ^ k * x)) = 1 / Real.tan x - 1 / Real.tan (2 ^ n * x) := by",
            "theorem imo_1966_p4_v10 (n : ℕ) (x : ℝ)\n"
            "  (h₀ : ∀ k, k > 0 → ∀ m : ℤ, x ≠ m * Real.pi / 2 ^ k) (h₁ : n > 0) :\n"
            "  (∑ k in Finset.Icc 1 n, (Real.sin (2 ^ k * x))⁻¹) = (Real.cos x / Real.sin x) - (Real.cos (2 ^ n * x) / Real.sin (2 ^ n * x)) := by",
        ),
    ]

    def format(
        self,
        formal_statement: str,
        variants_min: int = 5,
        variants_max: int = 15,
        theorem_name: str | None = None,
        **kwargs,
    ) -> str:
        examples = self._format_few_shot_examples()
        theorem_label = theorem_name or "<input_theorem>"
        return (
            "You are an expert Lean4 mathematician.\n\n"
            "Task:\n"
            f"Generate {variants_min}-{variants_max} alternative theorem statements that are semantically equivalent "
            "to the provided Lean theorem. Produce as many high-quality distinct rewrites as possible, up to the maximum.\n\n"
            "Hard constraints:\n"
            "1) Keep the same variables and hypotheses names exactly as in the input.\n"
            "2) Keep the same theorem name stem and append suffixes `_v2`, `_v3`, ... in increasing order.\n"
            "3) Do not include proofs; every theorem must end exactly at `:= by` with nothing after "
            "(`sorry`, tactics, or extra lines).\n"
            "4) Each rewrite must remain mathematically equivalent to the original theorem.\n"
            "5) Avoid trivial whitespace-only edits. Prefer algebraic/logical rewrites.\n"
            "6) DO NOT substitute hypothesis values/terms into other hypotheses or the goal (e.g. if `b = 30`, do not replace `b` with `30`).\n"
            "7) Do not weaken/strengthen the theorem or add junk conjuncts/disjuncts (forbidden: `∧ True`, `↔ False`, extra unrelated assumptions/goals).\n\n"
            "Diversity requirement:\n"
            "- Include several SIMPLE rewrites that only use symmetry/commutativity style transformations.\n"
            "- Examples of SIMPLE rewrites:\n"
            "  * `a = b` -> `b = a`\n"
            "  * `a < b` -> `b > a`\n"
            "  * `2 * x = y` -> `y = x * 2`\n"
            "  * `u + v = c` -> `v + u = c`\n"
            "  * `x ^ 3` -> `x * x * x`\n"
            "  * `x ^ 2` -> `x * x`\n"
            "  * `a * b` -> `b * a`\n"
            "- Across the whole set, rewrite every hypothesis and the goal at least once when possible.\n"
            "- Also include non-trivial but equivalent reformulations (not only simple ones).\n\n"
            "Coverage checklist (must satisfy across produced variants):\n"
            "- At least 3 variants must rewrite power notation (`^ 2` or `^ 3`) into repeated multiplication.\n"
            "- At least 3 variants must use explicit symmetry/commutativity in products or equalities.\n"
            "- At least 2 variants should rewrite division-style equalities into product-style equalities when safe.\n"
            "  Example pattern: `a / b = c` -> `a = c * b` or `a = b * c`.\n"
            "- At least 2 variants should alter parenthesization/ordering of multiplicative expressions.\n"
            "- At least 2 variants should convert '∣' divisibility statements into [MOD] or [ZMOD] expressions, and vice versa.\n"
            "- Do not keep one key hypothesis text unchanged in almost all variants.\n\n"
            "Allowed transformations include:\n"
            "- equivalent rearrangements of equalities/inequalities\n"
            "- equivalent predicate forms (`a ∣ b` vs congruence/mod-0 forms when appropriate)\n"
            "- syntactic rewrites using commutativity/associativity/distributivity\n"
            "- rewriting `0 ≤ x` as `x ≥ 0`, rewriting `¬Even x` as `Odd x` when appropriate\n"
            "- rewriting equations into equivalent multiplied/divided forms without changing assumptions\n"
            "- replacing expressions with provably equivalent forms\n\n"
            "Few-shot examples (from existing miniF2F 50x20 rewrites):\n"
            f"{examples}\n\n"
            "Now produce rewrites for this theorem:\n"
            f"-- theorem id: {theorem_label}\n"
            "```lean4\n"
            f"{formal_statement.strip()}\n"
            "```\n\n"
            "Output format:\n"
            "- Output ONLY Lean theorem declarations.\n"
            "- Separate each theorem block with one blank line.\n"
            "- Start from suffix `_v2`.\n"
            "- Do not use markdown code fences.\n"
            "- Never print `sorry` or a proof body after `:= by`.\n"
        )

    def _format_few_shot_examples(self) -> str:
        chunks: list[str] = []
        for idx, (orig, rewrite) in enumerate(self.FEW_SHOT_EXAMPLES, start=1):
            chunks.append(
                f"Example {idx}:\n"
                "Input:\n"
                "```lean4\n"
                f"{orig}\n"
                "```\n"
                "One valid rewrite:\n"
                "```lean4\n"
                f"{rewrite}\n"
                "```"
            )
        return "\n\n".join(chunks)

    def format_answer(self, answer: str) -> str:
        return answer.strip()

    def extract_answer(self, response: str, **kwargs) -> str:
        return response.strip()

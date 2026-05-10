"""Tests for rwens.utils.statement_parser.StatementParser."""

from __future__ import annotations

from rwens.utils.statement_parser import (
    BinderRole,
    StatementParser,
    parse_theorem_through_by,
)


def test_parse_theorem_roundtrip() -> None:
    stmt = (
        "theorem foo (x : ℝ) (h : x > 0) : x + 1 > 1 := by\n"
    )
    p = parse_theorem_through_by(stmt)
    assert p.name == "foo"
    assert "(x : ℝ)" in p.binder_segment
    assert "(h : x > 0)" in p.binder_segment
    assert p.conclusion.strip() == "x + 1 > 1"


def test_hypotheses_excludes_variables() -> None:
    stmt = (
        "theorem bar (x y : ℝ) (h₀ : 0 < x) (h₁ : x + y = 0) : x = -y := by\n"
    )
    hyps = StatementParser.hypotheses(stmt)
    assert {h.name for h in hyps} == {"h₀", "h₁"}
    vars_ = StatementParser.variable_binders(stmt)
    assert {v.name for v in vars_} == {"x y"}


def test_hypotheses_single_prop_param() -> None:
    stmt = "theorem t (P : Prop) (hP : P) : P := by\n"
    hyps = StatementParser.hypotheses(stmt)
    # ``P : Prop`` is treated as a parameter (Prop is a known type head); ``hP : P`` is a hypothesis.
    assert {h.name for h in hyps} == {"hP"}


def test_hypotheses_complex_type_still_variable() -> None:
    stmt = "theorem t (n : ℕ) (h : n + 1 > 0) : True := by\n"
    hyps = StatementParser.hypotheses(stmt)
    assert [h.name for h in hyps] == ["h"]
    assert StatementParser.variable_binders(stmt)[0].name == "n"


def test_colon_inside_binder_not_splitting_conclusion() -> None:
    stmt = (
        "theorem nested (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 0) : f 0 = 0 := by\n"
    )
    p = parse_theorem_through_by(stmt)
    assert "∀ x : ℝ" in p.binder_segment
    assert p.conclusion.strip() == "f 0 = 0"
    hyps = StatementParser.hypotheses(stmt)
    assert any("∀" in h.type_str for h in hyps)


def test_numeric_arrow_type_is_variable_not_hypothesis() -> None:
    stmt = "theorem tf (f : ℝ -> ℝ) (h : f 0 = 0) : True := by\n"
    vars_ = StatementParser.variable_binders(stmt)
    hyps = StatementParser.hypotheses(stmt)
    assert any(v.name == "f" for v in vars_)
    assert all(h.name != "f" for h in hyps)


def test_numeric_arrow_unicode_and_nnreal_are_variable() -> None:
    stmt = "theorem tg (g : NNReal → ℕ+) (h : True) : True := by\n"
    vars_ = StatementParser.variable_binders(stmt)
    assert any(v.name == "g" for v in vars_)


def test_binary_curried_numeric_function_is_variable_like_imos_1985_p6() -> None:
    stmt = (
        "theorem imo_1985_p6 (f : ℕ → NNReal → ℝ) (h₀ : ∀ x, f 1 x = x) "
        "(h₁ : ∀ x n, f (n + 1) x = f n x * (f n x + 1 / n)) : True := by\n"
    )
    vars_ = StatementParser.variable_binders(stmt)
    hyps = StatementParser.hypotheses(stmt)
    assert any(v.name == "f" for v in vars_)
    assert {h.name for h in hyps} == {"h₀", "h₁"}


def test_triple_curried_numeric_function_is_variable() -> None:
    stmt = "theorem t (f : ℕ → ℝ → NNReal) (h : True) : True := by\n"
    vars_ = StatementParser.variable_binders(stmt)
    hyps = StatementParser.hypotheses(stmt)
    assert any(v.name == "f" for v in vars_)
    assert all(h.name != "f" for h in hyps)


def test_plain_nnreal_binder_is_variable() -> None:
    stmt = "theorem th (a b : NNReal) (h : a = b) : True := by\n"
    vars_ = StatementParser.variable_binders(stmt)
    hyps = StatementParser.hypotheses(stmt)
    assert any(v.name == "a b" for v in vars_)
    assert all(h.name != "a b" for h in hyps)


def test_goal_with_forall_colon_parses_separator_correctly() -> None:
    stmt = (
        "theorem numbertheory_notequiv2i2jasqbsqdiv8 :\n"
        "    ¬∀ a b : ℤ, (∃ i j, a = 2 * i ∧ b = 2 * j) ↔ ∃ k, a ^ 2 + b ^ 2 = 8 * k := by\n"
    )
    p = parse_theorem_through_by(stmt)
    assert p.name == "numbertheory_notequiv2i2jasqbsqdiv8"
    assert p.binder_segment == ""
    assert p.conclusion.startswith("¬∀ a b : ℤ")
    assert StatementParser.hypotheses(stmt) == []

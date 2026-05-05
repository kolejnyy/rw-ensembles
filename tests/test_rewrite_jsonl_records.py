"""Tests for rewrite dataset JSONL helpers."""

from __future__ import annotations

from invpro.dataset.rewriting.jsonl_records import (
    make_rewrite_dataset_record,
    normalize_formal_statement,
    sanitize_header_remove_aesop,
)


def test_sanitize_header_remove_aesop() -> None:
    h = "import Mathlib\nimport Aesop\n\nopen Nat\n"
    assert "Aesop" not in sanitize_header_remove_aesop(h)
    assert "import Mathlib" in sanitize_header_remove_aesop(h)


def test_truncate_theorem_after_by_strips_sorry() -> None:
    raw = "theorem mathd (x : Nat) : x = x := by\n  sorry\n"
    assert "sorry" not in normalize_formal_statement(raw)
    assert normalize_formal_statement(raw).rstrip().endswith(":= by")


def test_make_rewrite_dataset_record_shape() -> None:
    rec = make_rewrite_dataset_record(
        name="foo_v2",
        original_name="foo",
        split="test",
        informal_prefix="/-- hi -/",
        formal_statement="theorem foo_v2 := by\n",
        header="import Mathlib\n",
        variant="v2",
        dataset_name="minif2f",
        original_formal_statement="theorem foo := by\n",
        variable_map=None,
        hypothesis_map=None,
        goal="⊢ True",
        certificate=None,
    )
    assert rec["name"] == "foo_v2"
    assert rec["original_name"] == "foo"
    assert rec["variant"] == "v2"
    assert "version" not in rec
    assert rec["dataset_name"] == "minif2f"
    assert rec["variable_map"] is None
    assert rec["goal"] == "⊢ True"
    assert "certificate" not in rec

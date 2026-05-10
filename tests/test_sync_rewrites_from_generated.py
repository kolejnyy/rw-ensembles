"""Tests for sync_rewrites_from_generated."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwens.dataset.rewriting.sync_rewrites_from_generated import (
    build_records_for_problem,
    merge_rewrites_jsonl,
)


def test_build_records_for_problem_minimal(tmp_path: Path) -> None:
    prob = "foo"
    gen = tmp_path / "g"
    gen.write_text(
        "theorem foo (x : Nat) : x = x := by\n\n"
        "theorem foo_v2 (x : Nat) : x = x := by\n",
        encoding="utf-8",
    )
    bench = {
        "formal_statement": "theorem foo (x : Nat) : x = x := by\n",
        "informal_prefix": "",
        "header": "import Mathlib\n",
        "goal": "⊢ True",
    }
    recs = build_records_for_problem(
        problem_name=prob,
        generated_file=gen,
        benchmark_row=bench,
        dataset_name="minif2f",
        split="test",
        old_by_variant={},
    )
    assert len(recs) == 2
    assert recs[0]["variant"] == "original"
    assert recs[1]["variant"] == "v2"


def test_original_formal_statement_comes_from_generated_file(tmp_path: Path) -> None:
    """Seed original_formal_statement must match first theorem in file, not benchmark JSONL."""
    prob = "foo"
    gen = tmp_path / "g"
    gen.write_text(
        "theorem foo (x : Nat) : x = 0 := by\n\n"
        "theorem foo_v2 (x : Nat) : x = x := by\n",
        encoding="utf-8",
    )
    bench = {
        "formal_statement": "theorem foo (x : Nat) : x = 999 := by\n",
        "informal_prefix": "",
        "header": "import Mathlib\n",
        "goal": None,
    }
    recs = build_records_for_problem(
        problem_name=prob,
        generated_file=gen,
        benchmark_row=bench,
        dataset_name="minif2f",
        split="test",
        old_by_variant={},
    )
    assert "999" not in recs[0]["original_formal_statement"]
    assert "x=0" in recs[0]["original_formal_statement"].replace(" ", "")
    assert recs[1]["original_formal_statement"] == recs[0]["original_formal_statement"]


def test_merge_preserves_order_and_replaces_block(tmp_path: Path) -> None:
    generated_root = tmp_path / "generated" / "minif2f" / "test"
    for name, body in [
        ("p1", "theorem p1 : True := by\n\ntheorem p1_v2 : True := by\n"),
        ("p2", "theorem p2 : True := by\n\ntheorem p2_v2 : True := by\n"),
    ]:
        p = generated_root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")

    bench = {
        ("test", "p1"): {
            "formal_statement": "theorem p1 : True := by\n",
            "informal_prefix": "",
            "header": "import Mathlib\n",
            "goal": None,
        },
        ("test", "p2"): {
            "formal_statement": "theorem p2 : True := by\n",
            "informal_prefix": "",
            "header": "import Mathlib\n",
            "goal": None,
        },
    }
    rows = [
        {"original_name": "p1", "variant": "original", "name": "p1"},
        {"original_name": "p1", "variant": "v2", "name": "p1_v2"},
        {"original_name": "x", "variant": "original", "name": "x"},
        {"original_name": "p2", "variant": "original", "name": "p2"},
        {"original_name": "p2", "variant": "v2", "name": "p2_v2"},
    ]
    out, _logs = merge_rewrites_jsonl(
        rows=rows,
        problems={"p2"},
        generated_root=tmp_path / "generated",
        benchmark_index=bench,
    )
    assert [r.get("original_name") for r in out] == ["p1", "p1", "x", "p2", "p2"]
    assert out[3]["variant"] == "original" and out[4]["variant"] == "v2"


def test_merge_requires_problem_in_jsonl() -> None:
    with pytest.raises(ValueError, match="No rows found"):
        merge_rewrites_jsonl(
            rows=[{"original_name": "other", "name": "other"}],
            problems={"missing"},
            generated_root=Path("/nonexistent"),
            benchmark_index={},
        )

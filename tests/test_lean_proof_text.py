"""Tests for rwens.utils.lean_proof_text."""

from __future__ import annotations

from rwens.utils.lean_proof_text import normalize_proof_indent


def test_normalize_proof_indent_preserves_nested_blocks() -> None:
    body = """  norm_num
  have h : True := by
    trivial
  omega
"""
    out = normalize_proof_indent(body)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines[0].startswith("  norm_num")
    assert lines[1].startswith("  have h")
    assert lines[2].startswith("    trivial")
    assert lines[3].startswith("  omega")


def test_normalize_proof_indent_single_level() -> None:
    out = normalize_proof_indent("linarith\n")
    assert out == "  linarith\n"

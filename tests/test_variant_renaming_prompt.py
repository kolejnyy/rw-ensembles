"""Tests for variant renaming JSON prompt parsing."""

from __future__ import annotations

import pytest

from invpro.prompt.variant_renaming import parse_variant_renaming_response


def test_parse_variant_renaming_plain_json() -> None:
    raw = """
    {"original": "theorem t (x : Nat) : x = x := by",
     "renamed": "theorem t (y : Nat) : y = y := by",
     "variable_map": {"x": "y"},
     "hypothesis_map": {}}
    """
    o = parse_variant_renaming_response(raw)
    assert o["original"].startswith("theorem t")
    assert "y" in o["renamed"]
    assert o["variable_map"] == {"x": "y"}
    assert o["hypothesis_map"] == {}


def test_parse_variant_renaming_fenced_json() -> None:
    raw = """Here:
```json
{"original": "theorem t := by",
 "renamed": "theorem t := by",
 "variable_map": {},
 "hypothesis_map": {}}
```
"""
    o = parse_variant_renaming_response(raw)
    assert o["variable_map"] == {}
    assert o["hypothesis_map"] == {}


def test_parse_variant_renaming_rejects_missing_key() -> None:
    with pytest.raises(ValueError, match="missing key"):
        parse_variant_renaming_response('{"original": "x", "renamed": "y"}')

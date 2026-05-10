"""Rewrite dataset library: JSONL record helpers and theorem-block parsing utilities."""

from rwens.dataset.rewriting.jsonl_records import (
    empty_maps_to_none,
    make_rewrite_dataset_record,
    normalize_formal_statement,
    sanitize_header_remove_aesop,
    truncate_theorem_after_by,
)
from rwens.dataset.rewriting.parse_generated_variants import (
    extract_theorem_blocks,
    filter_variant_blocks,
    preprocess_model_output,
    theorem_declared_name,
    variant_tag_from_declared_name,
)

__all__ = [
    "empty_maps_to_none",
    "extract_theorem_blocks",
    "filter_variant_blocks",
    "make_rewrite_dataset_record",
    "normalize_formal_statement",
    "sanitize_header_remove_aesop",
    "truncate_theorem_after_by",
    "preprocess_model_output",
    "theorem_declared_name",
    "variant_tag_from_declared_name",
]

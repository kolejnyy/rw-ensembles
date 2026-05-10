# Rewriting Canonicalization Config

This document describes configuration arguments for `RewritingCanonicalization` used by:

- `CanonicalLLMProver`
- `EnsembleLLMProver`

The module class is `RewritingCanonicalizationModule` (`rwens/rewriting/module.py`), and YAML is parsed through `rwens/rewriting/conf/*`.

## YAML shape

```yaml
rewriting:
  class: RewritingCanonicalization
  parameters:
    project_root: .
    timeout_seconds: 90.0
    top_rewrites: 10
    filter_rewrite_namespaces: null
    namespace_blacklist: null

    sampling:
      max_per_step: 10
      depth: 2
      only_simplifying_rewrites: false
      reverse_order: false
      use_explicit_comm: false
      use_combined: false
      num_combined: 20

    reranking:
      type: shortest_states

    energy:
      type: theorem_surprise
```

## Parameters

### Core

- `project_root` (required, `str`)
  - Lean project root.

- `timeout_seconds` (default: `90.0`)
  - Timeout budget for Lean-side state fetching / tactic application.

- `top_rewrites` (default: `10`)
  - Number of reranked rewrite options retained per target (`hyp` or goal).

- `filter_rewrite_namespaces` (default: `null`)
  - Optional namespace allowlist for rewrites from rwcnc cache/diagnostics.

- `namespace_blacklist` (default: `null`)
  - Optional namespace denylist. A rewrite is dropped if any extracted namespace
    from tactic/premise appears in this list.
  - Applied both on fresh rwcnc output and on cached rewrite entries.

### Sampling (`parameters.sampling`)

- `max_per_step` (default: `10`)
  - rwcnc branching factor per target before reranking.

- `depth` (default: `2`)
  - rwcnc search depth.

- `only_simplifying_rewrites` (default: `false`)
  - If true, rwcnc runs in simplifying-only mode.

- `reverse_order` (default: `false`)
  - Optional rwcnc reverse-order mode.

- `use_explicit_comm` (default: `false`)
  - Adds explicit commutativity/symmetry rewrite attempts (e.g. `eq_comm`, `add_comm`, `mul_comm`, `and_comm`, `or_comm`, etc.) for each processed target.

- `use_combined` (default: `false`)
  - Chooses rewrite candidate construction mode:
    - `false`: single-target augmented states (plus `try ... at *` exploratory states).
    - `true`: random multi-target combined states sampled across hyps+goal.

- `num_combined` (default: `20`)
  - Used only when `use_combined: true`.
  - Number of random combined states sampled before dedup.
  - Sampling per target uses weighted choice by rerank index:
    - `orig`: weight `1`
    - `aug_i` (1-indexed): weight `1/(i+1)` (so `aug_1=1/2`, `aug_2=1/3`, ...).

### Reranking (`parameters.reranking`)

- `type` (required in practice)
  - Selects heuristic used by `get_reranking_heuristic(...)`.
  - Common value: `shortest_states`.

### Energy (`parameters.energy`)

- `type`
  - `none`: no energy scoring (reranking-only behavior).
  - `confidence`: confidence-based energy.
  - `theorem_surprise`: energy injected by prover (used in current ensemble/canonical LLM flows).

- `confidence_aggregation` (optional, for `confidence`)
  - Aggregation strategy for confidence-based energy when applicable.

## Candidate construction behavior

After rwcnc + reranking (+ optional explicit comm):

- `use_combined: false` (default)
  - Build one candidate per rewritten target (single-hyp/goal modifications).
  - Add exploratory `try ... at *` candidates (`ring_nf`, `simp`, `norm_num`, `field_simp`).
  - Fully-augmented best-per-site candidate is not used.

- `use_combined: true`
  - For each target, sample from `[orig, rewrite_1, rewrite_2, ...]` with the weighted scheme above.
  - Compose sampled target rewrites into a combined state.
  - Repeat `num_combined` times, then dedup by resulting state.
  - Also adds exploratory `try ... at *` candidates.

## Namespace filtering order

When collecting rwcnc rewrites, filtering is applied in this order:

1. Hardcoded tactic exclusion for `propext` rewrites.
2. `namespace_blacklist` (drop if any namespace is blocked).
3. `filter_rewrite_namespaces` allowlist (if set, all extracted namespaces must be allowed).

## Backward compatibility notes

- Sampling keys can be provided in `parameters.sampling`.
- For compatibility, some keys are also read from top-level `parameters` when not present in `sampling`.
- Prefer putting sampling-related keys under `sampling` in new configs.
- Namespace filters are top-level under `parameters` (not under `sampling`).

## Minimal examples

### Explicit comm only (default single-target mode)

```yaml
sampling:
  max_per_step: 15
  depth: 2
  only_simplifying_rewrites: true
  use_explicit_comm: true
```

### Explicit comm + combined mode

```yaml
sampling:
  max_per_step: 15
  depth: 2
  only_simplifying_rewrites: true
  use_explicit_comm: true
  use_combined: true
  num_combined: 20
```


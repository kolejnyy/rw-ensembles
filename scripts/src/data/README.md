# Rewrite + Certificate Pipeline (Data Scripts)

This folder contains CLI entrypoints for the end-to-end rewrite pipeline:

1. prepare queue
2. generate rewrites with OpenAI
3. manual accept/reject review
4. background equivalence checker (with optional LLM prover fallback)
5. apply manually corrected certificates
6. build a final confirmed JSONL dataset with certificates

Optional: copy per-problem proof folders from an existing `results/...` run into another directory, restricted to theorem names listed in `confirmed_rewrites_dataset.jsonl` (section 7 below).

All commands below assume repo root:

```bash
cd /path/to/repo
```

---

## Directory Layout

For a run `RUN_ID`, outputs are under:

`data/rewritings_pipeline/RUN_ID/`

Typical structure:

```text
RUN_ID/
  generated/<dataset>/<split>/<problem>                     # model output file (original + variants)
  accepted/<dataset>/<split>/<problem>/<variant>.lean       # manually accepted variant files
  discarded/<dataset>/<split>/<problem>/<variant>.lean      # manually rejected variants
  unverified/<dataset>/<split>/<problem>/<variant>.certificate.unverified.txt
  corrected/<dataset>/<split>/<problem>/<variant>.certificate.unverified.txt  # manual fixes live here
  equivalence_certificates/<dataset>/<split>/<problem>/<variant>.certificate.lean
  logs/
    openai_rewrite_calls.jsonl
    manual_review_decisions.jsonl
    equivalence_watcher_results.jsonl
    equivalence_watcher_state.json
    corrected_certificates_applied.jsonl
  rewrites_dataset.jsonl
  confirmed_rewrites_dataset.jsonl
```

`rewrites_dataset.jsonl` is the aggregated per-variant dataset from generation.  
`confirmed_rewrites_dataset.jsonl` is the filtered final dataset with `certificate` attached.

---

## 1) Create Queue

Script: `prepare_rewrite_queue.py`  
Module: `invpro.dataset.rewriting.prepare_queue`

Reads:
- benchmark dataset JSONL (default `data/minif2f.jsonl`)

Writes:
- queue JSONL (default under `data/rewritings_pipeline/queues/`)
- optional prompt text files

Method summary:
- filters by split
- optionally skips problems that already have rewrite files
- writes one queue record per selected theorem (problem metadata + prompt + target variant count bounds)

Example:

```bash
python scripts/src/data/prepare_rewrite_queue.py \
  --dataset-path data/minif2f.jsonl \
  --dataset-name minif2f \
  --split test \
  --output-jsonl data/rewritings_pipeline/queues/minif2f_test_queue_batch01.jsonl \
  --limit 20 \
  --variants-min 5 \
  --variants-max 15
```

---

## 2) Run OpenAI Augmentations

Script: `generate_rewrites_openai.py`  
Module: `invpro.dataset.rewriting.generate_openai`

Reads:
- queue JSONL
- prompt formatter config (optional)

Writes:
- generated theorem files in `generated/...`
- run-level `rewrites_dataset.jsonl` (aggregated rows)
- OpenAI call logs in `logs/openai_rewrite_calls.jsonl`

Method summary:
- rewrite pass produces `_v2`, `_v3`, ...
- optional renaming pass appends `_renamed` variants (does not replace base variants)
- JSONL rows include `original_formal_statement`, `variant`, optional maps (`variable_map`, `hypothesis_map`)

Example (real execution):

```bash
python scripts/src/data/generate_rewrites_openai.py \
  --queue-jsonl data/rewritings_pipeline/queues/minif2f_test_queue_batch01.jsonl \
  --output-root data/rewritings_pipeline \
  --run-id minif2f_test_gpt54_7 \
  --model gpt-5.4 \
  --execute
```

---

## 3) Start Manual Verification (Accept/Reject)

Script: `manual_review_rewrites.py`  
Module: `invpro.dataset.rewriting.manual_review`

Reads:
- either generated files (`generated/...`) or `--rewrites-jsonl`

Writes:
- accepted files in `accepted/...`
- rejected files in `discarded/...`
- decisions in `logs/manual_review_decisions.jsonl`

Method summary:
- interactive terminal review with `[a]ccept / [r]eject / [s]kip / [q]uit`
- supports resume from prior decisions
- terminal screen is cleared before each comparison for readability

Recommended (JSONL mode):

```bash
python scripts/src/data/manual_review_rewrites.py \
  --output-root data/rewritings_pipeline \
  --run-id minif2f_aug_test_split \
  --rewrites-jsonl data/rewritings_pipeline/minif2f_aug_test_split/rewrites_dataset.jsonl \
  --dataset-name minif2f \
  --split test \
  --copy-original-on-accept
```

---

## 4) Start Background Checker (LLM Prover ON)

Script: `watch_accepted_equivalence.py`  
Module: `invpro.dataset.rewriting.watch_accepted_equivalence`

Reads:
- accepted variants in `accepted/...`
- benchmark JSONL (`--dataset-jsonl`, typically `data/minif2f.jsonl`)
- run rewrites JSONL (`--rewrites-jsonl`) to recover non-renamed companion variants for `_renamed` rows

Writes:
- verified certificates in `equivalence_certificates/.../*.certificate.lean`
- unresolved debug drafts in `unverified/.../*.certificate.unverified.txt`
- status log `logs/equivalence_watcher_results.jsonl`
- processed state `logs/equivalence_watcher_state.json`

Method summary:
- polls `accepted/` for new files
- runs `EquivalenceChecker` per variant
- for renamed variants, uses paired base variant for bridge obligations
- if not accepted, emits unverified text artifact with failed auxiliary goals replaced by `sorry`

Example (continuous, with vLLM fallback):

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/src/data/watch_accepted_equivalence.py \
  --output-root data/rewritings_pipeline \
  --run-id minif2f_aug_test_split \
  --rewrites-jsonl data/rewritings_pipeline/minif2f_aug_test_split/rewrites_dataset.jsonl \
  --dataset-jsonl data/minif2f.jsonl \
  --prover-config configs/models/deepseek-single-pass-baseline-vllm.yaml \
  --llm-attempts 8 \
  --watch-seconds 5
```

Single pass:

```bash
python scripts/src/data/watch_accepted_equivalence.py \
  --output-root data/rewritings_pipeline \
  --run-id minif2f_test_gpt54_7 \
  --rewrites-jsonl data/rewritings_pipeline/minif2f_test_gpt54_7/rewrites_dataset.jsonl \
  --dataset-jsonl data/minif2f.jsonl \
  --once
```

---

## 5) Merge Manually Corrected Certificates

Script: `apply_corrected_certificates.py`  
Module: `invpro.dataset.rewriting.apply_corrected_certificates`

Reads:
- corrected files from `<run_root>/corrected/**`

Writes:
- verified moved certificates to `equivalence_certificates/**`
- merge/apply log `logs/corrected_certificates_applied.jsonl`

Method summary:
- each corrected file is Lean-verified
- only verified files are moved into final certificate folder
- failed ones stay in `corrected/` and are logged with error

Example:

```bash
python scripts/src/data/apply_corrected_certificates.py \
  --output-root data/rewritings_pipeline \
  --run-id minif2f_aug_test_split \
  --project-root .
```

Dry-run verify only:

```bash
python scripts/src/data/apply_corrected_certificates.py \
  --output-root data/rewritings_pipeline \
  --run-id minif2f_test_gpt54_7 \
  --project-root . \
  --dry-run
```

---

## 6) Build Confirmed Dataset

Script: `build_confirmed_rewrites_dataset.py`  
Module: `invpro.dataset.rewriting.build_confirmed_rewrites_dataset`

Reads:
- `rewrites_dataset.jsonl`
- manual review log
- watcher results log
- corrected apply log (optional)
- `equivalence_certificates/**`

Writes:
- `confirmed_rewrites_dataset.jsonl`

Method summary:
- joins rewrite rows with available certificate files
- by default requires manual decision `accept`
- by default excludes watcher-unconfirmed rows unless they were manually corrected and applied
- embeds full certificate text into each kept row (`certificate`)

Example:

```bash
python scripts/src/data/build_confirmed_rewrites_dataset.py \
  --output-root data/rewritings_pipeline \
  --run-id minif2f_aug_test_split
```

Optional flags:
- `--include-unaccepted-review`
- `--include-unconfirmed-watcher`

---

## 7) Copy confirmed proofs into a results directory

Script: `copy_confirmed_results_proofs.py`

Use this after you have a `confirmed_rewrites_dataset.jsonl` and a full `results/...` tree from proof generation (e.g. single-pass runs with per-theorem folders containing `attempts.jsonl`). The script walks the confirmed JSONL, collects each distinct `name` field (one row per augmented theorem), and copies the matching subdirectory from `--source-dir` to `--dest-dir`. That lets you materialize a **smaller** results tree that only contains problems present in the confirmed dataset (for example copying from `minif2f-aug-test` into a shorter path like `minif2f-aug`).

Reads:
- `confirmed_rewrites_dataset.jsonl` (via `--confirmed-jsonl`)

Writes:
- one folder per `name` under `--dest-dir`, full tree copy (typically `attempts.jsonl` plus any other files in that problem folder)

Method summary:
- unique `name` values are taken in first-seen order
- if `--dest-dir/<name>` already exists, it is replaced
- missing source folders are reported and skipped; use `--dry-run` to preview

Example:

```bash
python scripts/src/data/copy_confirmed_results_proofs.py \
  --confirmed-jsonl data/rewritings_pipeline/minif2f_aug_test_split/confirmed_rewrites_dataset.jsonl \
  --source-dir results/minif2f-aug-test/test/deepseek-single-pass \
  --dest-dir results/minif2f-aug/test/deepseek-single-pass
```

---

## Recommended End-to-End Order

1. `prepare_rewrite_queue.py`
2. `generate_rewrites_openai.py --execute`
3. `manual_review_rewrites.py`
4. `watch_accepted_equivalence.py` (continuous or periodic `--once`)
5. Manually fix files from `unverified/`, place in `corrected/`
6. `apply_corrected_certificates.py`
7. `build_confirmed_rewrites_dataset.py`
8. (optional) `copy_confirmed_results_proofs.py` — copy only confirmed theorem result folders into another `results/...` layout


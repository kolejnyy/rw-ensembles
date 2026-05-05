# Invariant Formal Provers

A research project on proof-invariant formal provers in Lean4.

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Codebase overview

The codebase is organized into several key components:

- **Configuration System** (`invpro/*/conf/`): YAML-based configuration for prompt formatters, LLM models, and formal provers. Supports hierarchical, nested configurations with type-safe factories.

- **Formal Provers** (`invpro/models/`): Implements `FormalProver` interface with step-by-step proving capabilities. `NaiveStepByStepProver` uses LLMs to generate tactics iteratively, based on the current state.

- **LLM Models** (`invpro/models/llm/`): Wrappers for language models (e.g., `QwenCoderS2T`) that generate tactics from proof states. Supports checkpoint loading and quantization.

- **Tactic Application** (`invpro/utils/applier.py`): Manages indentation tracking and state updates after applying tactics to Lean proofs.

- **Evaluation** (`scripts/src/eval/test_prover.py`, `invpro/utils/metrics.py`): Scripts and utilities for evaluating provers on datasets (e.g. miniF2F, ProofNet).

## Testing (Slurm)

This repository supports a two-stage Slurm workflow:
1. generate solution attempts
2. re-verify those attempts problem-by-problem (Slurm array, typically one at a time)

The main entry points are `scripts/run_testing.sh` and `scripts/run_verification.sh`.

### 1) End-to-end run (generation + reverification)

Edit `scripts/run_testing.sh`:
- set `REPO_ROOT`
- set `ORCH_CONFIG` to a test config YAML (for example `configs/testing/adhoc.yaml`)
- set `ORCH_EXPERIMENT_ID` (results/log namespace)

Then run:

```bash
bash scripts/run_testing.sh
```

What this does:
- submits a driver job via `scripts/bash/testing_pipeline/slurm/run_orchestrate_pipeline.sh`
- driver submits generation (`scripts/bash/testing_pipeline/slurm/generate_solutions_gpu.sh`)
- after generation finishes, submits reverification array (`scripts/bash/testing_pipeline/slurm/verify_one_problem_array_adhoc.sh`) with one concurrent task by default
- merges per-problem reverification outputs into a CSV

Expected outputs:
- generated attempts: `results/<dataset_name>/<split>/<experiment_id>/<problem>/attempts.jsonl`
- per-problem reverification JSONs: `results/<dataset_name>/<split>/<experiment_id>/.reverification_by_problem/*.json`
- merged metrics: `results/<dataset_name>/<split>/<experiment_id>/reverified_pass_at_k.csv`
- Slurm logs: `.slurm/pipeline/<experiment_id>/`

### 2) Reverification-only for an existing solutions folder

Use this when you already have `attempts.jsonl` files and only want verification.

```bash
bash scripts/run_verification.sh results/minif2f/valid/<experiment_id>
```

This wraps `scripts/src/testing_pipeline/submit_verification_array.py`, which:
- enumerates problem folders containing `attempts.jsonl` (lexicographic order)
- skips already processed problems by default (`.reverification_by_problem/<problem>.json` exists); use `--no-skip-existing` to force full reruns
- submits `scripts/bash/testing_pipeline/slurm/verify_one_problem_array_adhoc.sh` as one or more Slurm arrays (splits large runs so each job stays under typical `MaxArraySize` limits, default `--max-array-tasks 1000`)
- by default **chains** those chunk jobs with `afterok` dependencies so only one chunk runs at a time (use `--parallel-chunks` on the Python script if you want every chunk submitted at once)
- writes logs under `.slurm/pipeline/<solutions_folder_name>/` by default

Common optional flags:

```bash
bash scripts/run_verification.sh results/minif2f/valid/<experiment_id> \
  --max-concurrent 4 \
  --max-array-tasks 1000 \
  --log-tag <custom_log_tag>
```

Forward `--parallel-chunks` through the wrapper (all extra args go to `submit_verification_array.py`) if you want every chunk job queued immediately instead of waiting for the previous chunk to finish.
Use `--no-skip-existing` if you intentionally want to recompute problems that already have reverification JSON outputs.

### Config notes

The orchestration scripts read a testing YAML in the same style as `scripts/src/eval/generate_solutions.py`, e.g.:

```yaml
prover: configs/models/deepseek-single-pass-baseline-vllm.yaml
dataset_path: data/minif2f.jsonl
split: valid
num_attempts: 8
limit: 4
start_from: 0
```

`dataset_name` is optional; if omitted, it defaults to the stem of `dataset_path`.

## Scripts layout

- Top-level callable scripts:
  - `scripts/run_testing.sh`
  - `scripts/run_verification.sh`
  - `scripts/switch_lean_version.sh`
- Supplementary bash scripts:
  - `scripts/bash/testing_pipeline/slurm/` (active Slurm pipeline helpers)
  - `scripts/bash/legacy_slurm/` (older ad-hoc Slurm scripts)
  - `scripts/bash/utils/` (misc shell utilities)
- Supplementary python scripts:
  - `scripts/src/testing_pipeline/` (active generation/reverification pipeline logic)
  - `scripts/src/eval/` (evaluation/generation/compare scripts)
  - `scripts/src/analysis/` (analysis/report scripts)
  - `scripts/src/debug/` (debug/smoke-test scripts)
  - `scripts/src/data/` (dataset generation/prep scripts)
  - `scripts/src/utils/` (small utility scripts)

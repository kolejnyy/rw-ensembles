#!/usr/bin/env bash
#
# Ensemble evaluation orchestrator (stages 1–5):
#   1) export variant JSONL (optional)
#   2) generate solutions with SinglePassProver (generate_solutions.py)
#   3) recompose proofs to original theorems (recompose_ensemble_attempts.py)
#   4) Lean verification (submit_verification_array.py → CPU Slurm array)
#   5) collect / interpret results (ensemble_eval_analysis.py → .analysis/)
#
# Run from the repository root. See docs/ensemble_eval_orchestrator_MVP.md
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SKIP_STAGE1=0
USE_SLURM=0
VARIANT_DATASET=""
SOURCE_DATASET="${REPO_ROOT}/data/minif2f.jsonl"
EXPORT_EXPERIMENT=""
EXPORT_SPLIT="valid"
EXPORT_LIMIT=""
ENSEMBLE_CONFIG=""
EXPORT_DATA_DIR="${REPO_ROOT}/data"
NUM_ATTEMPTS=16
BATCH_SIZE=8
SPLIT="valid"
RUN_ID=""
SINGLE_PASS_PROVER="configs/models/deepseek-single-pass-baseline-vllm.yaml"
RUN_VERIFY=1
VERIFY_MAX_CONCURRENT=5
VERIFY_WAIT=1
RUN_ANALYSIS=1

usage() {
  sed -n '1,80p' "$0" | sed -n '/^# /s/^# //p'
  cat <<EOF
Usage:
  $0 --variant-dataset PATH --run-id ID [options]

Required:
  --variant-dataset PATH   Required with --skip-stage1: prebuilt variant JSONL.
                           If stage 1 runs, output is data/<export-experiment>.jsonl (see below).

  --run-id ID              experiment_id for results: results/<stem>/<split>/<ID>/

Stage 1 (export) — omit with --skip-stage1 if the JSONL already exists:
  --source-dataset PATH    default: data/minif2f.jsonl
  --export-experiment NAME output stem: data/<NAME>.jsonl (required if not skipping stage 1)
  --export-split SPLIT     default: valid
  --export-limit N         optional limit on source problems
  --ensemble-config PATH   EnsembleLLMProver YAML (e.g. ensemble-vllm-prover-...)

Generation:
  --num-attempts N         default: 16
  --batch-size N           default: 8
  --split SPLIT            must match rows in variant JSONL (default: valid)
  --single-pass-prover PATH default: configs/models/deepseek-single-pass-baseline-vllm.yaml

Slurm:
  --slurm                  submit stage 1 (if run) and stage 2 via sbatch; run stage 3 locally after stage 2

Verification (stage 4, after recompose):
  --no-verify              skip Lean re-verification
  --verify-max-concurrent N  Slurm array throttle (default: 5); see submit_verification_array.py
  --no-verify-wait         submit verification jobs but do not poll until done
  --verify-wait            poll squeue until verification finishes (default on)

Analysis (stage 5, after verification or if .reverification_by_problem already exists):
  --no-analysis            skip ensemble_eval_analysis.py (JSON/CSV/LaTeX under <split>/.analysis/)

Examples:
  # Prebuilt variant JSONL, local GPU (no Slurm):
  $0 --variant-dataset data/minif2f-ensemble-naive-valid-l5.jsonl \\
     --run-id naive-l5-eval --num-attempts 8 --batch-size 4

  # Same with Slurm for generation (h100):
  $0 --variant-dataset data/minif2f-ensemble-naive-valid-l5.jsonl \\
     --run-id naive-l5-eval --num-attempts 8 --batch-size 4 --slurm
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --skip-stage1) SKIP_STAGE1=1; shift ;;
    --slurm) USE_SLURM=1; shift ;;
    --variant-dataset) VARIANT_DATASET="$2"; shift 2 ;;
    --source-dataset) SOURCE_DATASET="$2"; shift 2 ;;
    --export-experiment) EXPORT_EXPERIMENT="$2"; shift 2 ;;
    --export-split) EXPORT_SPLIT="$2"; shift 2 ;;
    --export-limit) EXPORT_LIMIT="$2"; shift 2 ;;
    --ensemble-config) ENSEMBLE_CONFIG="$2"; shift 2 ;;
    --export-data-dir) EXPORT_DATA_DIR="$2"; shift 2 ;;
    --num-attempts) NUM_ATTEMPTS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --single-pass-prover) SINGLE_PASS_PROVER="$2"; shift 2 ;;
    --no-verify) RUN_VERIFY=0; shift ;;
    --verify-max-concurrent) VERIFY_MAX_CONCURRENT="$2"; shift 2 ;;
    --no-verify-wait) VERIFY_WAIT=0; shift ;;
    --verify-wait) VERIFY_WAIT=1; shift ;;
    --no-analysis) RUN_ANALYSIS=0; shift ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  echo "ERROR: --run-id is required." >&2
  usage
  exit 2
fi

if [[ "${SKIP_STAGE1}" -eq 1 ]]; then
  if [[ -z "${VARIANT_DATASET}" ]]; then
    echo "ERROR: with --skip-stage1, --variant-dataset is required." >&2
    exit 2
  fi
  VARIANT_ABS="$(realpath "${VARIANT_DATASET}")"
  if [[ ! -f "${VARIANT_ABS}" ]]; then
    echo "ERROR: variant dataset not found: ${VARIANT_ABS}" >&2
    exit 2
  fi
  DATASET_STEM="$(basename "${VARIANT_ABS}" .jsonl)"
else
  if [[ -z "${EXPORT_EXPERIMENT}" || -z "${ENSEMBLE_CONFIG}" ]]; then
    echo "ERROR: without --skip-stage1, --export-experiment and --ensemble-config are required." >&2
    exit 2
  fi
fi

ORCH_DIR="${REPO_ROOT}/results/_orchestrator/${RUN_ID}"
mkdir -p "${ORCH_DIR}"
GEN_CFG="${ORCH_DIR}/generate_config.yaml"

write_generate_config() {
  local vpath="$1"
  cat > "${GEN_CFG}" <<EOF
# Auto-generated by ensemble_eval_orchestrator.sh — do not commit
prover: ${SINGLE_PASS_PROVER}
dataset_path: ${vpath}
split: ${SPLIT}
num_attempts: ${NUM_ATTEMPTS}
batch_size: ${BATCH_SIZE}
experiment_id: ${RUN_ID}
dataset_name: ${DATASET_STEM}
EOF
  echo "Wrote ${GEN_CFG}"
}

run_export_local() {
  local exp="$1"
  local lim=()
  [[ -n "${EXPORT_LIMIT}" ]] && lim=(--limit "${EXPORT_LIMIT}")
  local out_path
  out_path="$(realpath "${EXPORT_DATA_DIR}")/${exp}.jsonl"
  conda run -n invpro python scripts/src/data/export_ensemble_inference_dataset.py \
    --dataset "$(realpath "${SOURCE_DATASET}")" \
    --experiment "${exp}" \
    --split "${EXPORT_SPLIT}" \
    --prover-config "$(realpath "${ENSEMBLE_CONFIG}")" \
    --output-path "${out_path}" \
    "${lim[@]}"
}

run_export_slurm() {
  local exp="$1"
  export REPO_ROOT
  export EXPORT_SOURCE_DATASET="$(realpath "${SOURCE_DATASET}")"
  export EXPORT_EXPERIMENT="${exp}"
  export EXPORT_SPLIT="${EXPORT_SPLIT}"
  export ENSEMBLE_CONFIG="$(realpath "${ENSEMBLE_CONFIG}")"
  export OUTPUT_PATH="$(realpath "${EXPORT_DATA_DIR}")/${exp}.jsonl"
  if [[ -n "${EXPORT_LIMIT}" ]]; then
    export EXPORT_LIMIT
  else
    unset EXPORT_LIMIT || true
  fi
  sbatch --wait "${REPO_ROOT}/scripts/bash/slurm/ensemble_stage1_export.slurm"
}

run_generate_local() {
  conda run -n invpro python scripts/src/eval/generate_solutions.py \
    --config "${GEN_CFG}" \
    --experiment-id "${RUN_ID}"
}

run_generate_slurm() {
  export REPO_ROOT
  export GENERATE_CONFIG="${GEN_CFG}"
  export EXPERIMENT_ID="${RUN_ID}"
  sbatch --wait "${REPO_ROOT}/scripts/bash/slurm/ensemble_stage2_generate.slurm"
}

run_recompose() {
  local res_root="${REPO_ROOT}/results/${DATASET_STEM}/${SPLIT}/${RUN_ID}"
  # --in-place updates attempts.jsonl (default without this only writes attempts_recomposed.jsonl).
  conda run -n invpro python scripts/src/eval/recompose_ensemble_attempts.py \
    --dataset "${VARIANT_ABS}" \
    --results-root "${res_root}" \
    --in-place \
    --backup
}

run_verify() {
  local sol="${REPO_ROOT}/results/${DATASET_STEM}/${SPLIT}/${RUN_ID}"
  mkdir -p "${REPO_ROOT}/.slurm/pipeline"
  local wait_args=()
  if [[ "${VERIFY_WAIT}" -eq 1 ]]; then
    wait_args=(--wait)
  fi
  conda run -n invpro python scripts/src/testing_pipeline/submit_verification_array.py "${sol}" \
    --max-concurrent "${VERIFY_MAX_CONCURRENT}" \
    --log-tag "ensemble-${DATASET_STEM}-${RUN_ID}-verify" \
    "${wait_args[@]}"
}

run_analysis() {
  local split_root="${REPO_ROOT}/results/${DATASET_STEM}/${SPLIT}/${RUN_ID}"
  local rev_sub="${split_root}/.reverification_by_problem"
  if [[ ! -d "${rev_sub}" ]]; then
    echo "Stage 5: skip analysis (no ${rev_sub})."
    return 0
  fi
  echo "Stage 5: ensemble evaluation analysis (seed vs ens-1+7, budgets 8/16/32/64) ..."
  conda run -n invpro python scripts/src/analysis/ensemble_eval_analysis.py \
    --reverification-dir "${split_root}"
}

# --- Stage 1 ---
if [[ "${SKIP_STAGE1}" -eq 0 ]]; then
  if [[ "${USE_SLURM}" -eq 1 ]]; then
    run_export_slurm "${EXPORT_EXPERIMENT}"
  else
    run_export_local "${EXPORT_EXPERIMENT}"
  fi
  VARIANT_ABS="$(realpath "${EXPORT_DATA_DIR}/${EXPORT_EXPERIMENT}.jsonl")"
  DATASET_STEM="$(basename "${VARIANT_ABS}" .jsonl)"
fi

if [[ ! -f "${VARIANT_ABS}" ]]; then
  echo "ERROR: variant dataset not found: ${VARIANT_ABS}" >&2
  exit 2
fi

write_generate_config "${VARIANT_ABS}"

# --- Stage 2 ---
if [[ "${USE_SLURM}" -eq 1 ]]; then
  run_generate_slurm
else
  run_generate_local
fi

# --- Stage 3 ---
run_recompose

# --- Stage 4 ---
if [[ "${RUN_VERIFY}" -eq 1 ]]; then
  echo "Stage 4: submitting Lean verification (CPU Slurm array) ..."
  run_verify
  echo "Stage 4: per-problem JSON under ${REPO_ROOT}/results/${DATASET_STEM}/${SPLIT}/${RUN_ID}/.reverification_by_problem/"
else
  echo "Stage 4: skipped (--no-verify)."
fi

# --- Stage 5 ---
if [[ "${RUN_ANALYSIS}" -eq 1 ]]; then
  run_analysis
else
  echo "Stage 5: skipped (--no-analysis)."
fi

echo "Done. Results: ${REPO_ROOT}/results/${DATASET_STEM}/${SPLIT}/${RUN_ID}"
echo "Stage 3: attempts.jsonl updated in place with recomposed proofs (backup: attempts.jsonl.bak)."

#!/bin/bash
# Stage 1: export ensemble variant JSONL. Loads EnsembleLLMProver (Lean + vLLM theorem_surprise) — requires a GPU.
#SBATCH --job-name=ens-export
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=.slurm/ensemble/%x_%j.out
#SBATCH --error=.slurm/ensemble/%x_%j.err

set -euo pipefail
mkdir -p .slurm/ensemble
cd "${REPO_ROOT:?}"

LIMIT_ARGS=()
if [[ -n "${EXPORT_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${EXPORT_LIMIT}")
fi

OUTPUT_PATH="${OUTPUT_PATH:?}"

conda run -n invpro python scripts/src/data/export_ensemble_inference_dataset.py \
  --dataset "${EXPORT_SOURCE_DATASET:?}" \
  --experiment "${EXPORT_EXPERIMENT:?}" \
  --split "${EXPORT_SPLIT:?}" \
  --prover-config "${ENSEMBLE_CONFIG:?}" \
  --output-path "${OUTPUT_PATH}" \
  "${LIMIT_ARGS[@]}"

#!/bin/bash
#SBATCH --job-name=cpu-orch-gen-rev
#SBATCH --time=167:59:59
#SBATCH --qos=long
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=.slurm/pipeline/%x_%j.out
#SBATCH --error=.slurm/pipeline/%x_%j.err
#SBATCH --nodelist=noether

set -euo pipefail

RWENS_PYTHON="${RWENS_PYTHON:-/path/to/conda/env/bin/python}"
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "REPO_ROOT must be set to the rwens repository root." >&2
  exit 1
fi
cd "${REPO_ROOT}"

if [[ -z "${ORCH_CONFIG:-}" ]]; then
  echo "ORCH_CONFIG must be set." >&2
  exit 1
fi

args=(--config "${ORCH_CONFIG}")
if [[ -n "${ORCH_EXPERIMENT_ID:-}" ]]; then
  args+=(--experiment-id "${ORCH_EXPERIMENT_ID}")
fi
if [[ "${ORCH_SKIP_GEN:-0}" == "1" ]]; then
  args+=(--skip-generation)
fi
if [[ "${ORCH_NO_MERGE:-0}" == "1" ]]; then
  args+=(--no-merge)
fi

"${RWENS_PYTHON}" scripts/src/testing_pipeline/orchestrate_generate_and_reverify_slurm.py "${args[@]}"

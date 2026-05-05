#!/bin/bash
#SBATCH --job-name=gen-sol
#SBATCH --time=139:59:59
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=.slurm/pipeline/%x_%j.out
#SBATCH --error=.slurm/pipeline/%x_%j.err

set -euo pipefail
INVPRO_PYTHON="${INVPRO_PYTHON:-/path/to/conda/env/bin/python}"
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "REPO_ROOT must be set to the invpro repository root." >&2
  exit 1
fi
cd "${REPO_ROOT}"
if [[ -z "${GEN_CONFIG:-}" ]]; then
  echo "GEN_CONFIG must be set." >&2
  exit 1
fi
gen_extra=()
if [[ -n "${GEN_EXPERIMENT_ID:-}" ]]; then
  gen_extra+=(--experiment-id "${GEN_EXPERIMENT_ID}")
fi
"${INVPRO_PYTHON}" scripts/src/eval/generate_solutions.py --config "${GEN_CONFIG}" "${gen_extra[@]}"

#!/bin/bash
#SBATCH --job-name=cpu-reverify-1p
#SBATCH --time=167:59:59
#SBATCH --qos=long
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=.slurm/pipeline/%x_%A_%a.out
#SBATCH --error=.slurm/pipeline/%x_%A_%a.err

set -euo pipefail
if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "REPO_ROOT must be set to the rwens repository root." >&2
  exit 1
fi
cd "${REPO_ROOT}"
if [[ -z "${SOLUTIONS_DIR:-}" ]]; then
  echo "SOLUTIONS_DIR must be set to the solutions base directory." >&2
  exit 1
fi
: "${SLURM_ARRAY_TASK_ID:?Expected SLURM array task id}"
# When verification is split across multiple array jobs (Slurm MaxArraySize), add this offset.
OFFSET="${VERIFY_PROBLEM_INDEX_OFFSET:-0}"
PROBLEM_INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))
WORKERS="${VERIFY_NUM_WORKERS:-4}"
TIMEOUT="${VERIFY_TIMEOUT_SECONDS:-120}"
KVALS="${VERIFY_K_VALUES:-1,2,4,8,16,32,64,128}"
extra=()
if [[ -n "${VERIFY_MAX_ATTEMPTS:-}" ]]; then
  extra+=(--max-attempts-per-problem "${VERIFY_MAX_ATTEMPTS}")
fi
if [[ -n "${VERIFY_PREAMBLE:-}" ]]; then
  extra+=(--preamble "${VERIFY_PREAMBLE}")
fi
if [[ -n "${VERIFY_PROBLEM_LIST_FILE:-}" ]]; then
  extra+=(--problem-list-file "${VERIFY_PROBLEM_LIST_FILE}")
fi
"${RWENS_PYTHON}" scripts/src/testing_pipeline/verify_solutions_one_problem_mp.py "${SOLUTIONS_DIR}" \
  --problem-index "${PROBLEM_INDEX}" \
  --num-workers "${WORKERS}" \
  --timeout-seconds "${TIMEOUT}" \
  --k-values "${KVALS}" \
  "${extra[@]}"

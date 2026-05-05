#!/bin/bash
# Stage 2: SinglePassProver generation on variant dataset (GPU, h100).
#SBATCH --job-name=ens-gen
#SBATCH --time=167:59:59
#SBATCH --qos=long
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=.slurm/ensemble/%x_%j.out
#SBATCH --error=.slurm/ensemble/%x_%j.err
#SBATCH --partition=h100

set -euo pipefail
mkdir -p .slurm/ensemble
cd "${REPO_ROOT:?}"

conda run -n invpro python scripts/src/eval/generate_solutions.py \
  --config "${GENERATE_CONFIG:?}" \
  --experiment-id "${EXPERIMENT_ID:?}"

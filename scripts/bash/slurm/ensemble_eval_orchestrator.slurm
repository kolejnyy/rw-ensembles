#!/bin/bash
# Full ensemble eval pipeline driver: runs ensemble_eval_orchestrator.sh with --slurm so
# stages 1–2 submit GPU jobs; stages 3–5 run on this allocation (recompose, verify submit, analysis).
#
# Submit from anywhere:
#   sbatch /path/to/rwens/scripts/bash/slurm/ensemble_eval_orchestrator.slurm
# Or from repo root:
#   sbatch scripts/bash/slurm/ensemble_eval_orchestrator.slurm
#
# Override repo location if needed:
#   export REPO_ROOT=/path/to/rwens
#   sbatch scripts/bash/slurm/ensemble_eval_orchestrator.slurm
#
#SBATCH --job-name=ens-orch
#SBATCH --qos=long
#SBATCH --time=168:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=.slurm/ensemble/%x_%j.out
#SBATCH --error=.slurm/ensemble/%x_%j.err

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  cd "${REPO_ROOT}" || exit 1
else
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${_here}/../../.." && pwd)"
  cd "${REPO_ROOT}" || exit 1
  export REPO_ROOT
fi

mkdir -p .slurm/ensemble

# Same configuration as interactive:
#   bash scripts/bash/ensemble_eval_orchestrator.sh --source-dataset ... --slurm
bash scripts/bash/ensemble_eval_orchestrator.sh \
  --source-dataset data/minif2f.jsonl \
  --export-experiment ens_data/minif2f-ensemble-naive-valid-l5 \
  --export-split valid \
  --export-limit 5 \
  --ensemble-config configs/models/ensemble-vllm-prover-15-2-theorem-surprise.yaml \
  --run-id naive-l5-test \
  --num-attempts 8 \
  --batch-size 8 \
  --slurm

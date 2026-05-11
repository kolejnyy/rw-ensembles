export REPO_ROOT=/path/to/repo
cd $REPO_ROOT
export ORCH_CONFIG=configs/testing/miniF2F/minif2f-valid-deepseek-noncot.yaml
export ORCH_EXPERIMENT_ID=deepseek-miniF2F-valid-noncot

mkdir -p ".slurm/pipeline/${ORCH_EXPERIMENT_ID}"
sbatch \
  --output=".slurm/pipeline/${ORCH_EXPERIMENT_ID}/%x_%j.out" \
  --error=".slurm/pipeline/${ORCH_EXPERIMENT_ID}/%x_%j.err" \
  --export=ALL,REPO_ROOT,ORCH_CONFIG,ORCH_EXPERIMENT_ID \
  scripts/bash/testing_pipeline/slurm/run_orchestrate_pipeline.sh
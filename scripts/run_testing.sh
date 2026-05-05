export REPO_ROOT=/path/to/repo
cd $REPO_ROOT
export ORCH_CONFIG=configs/testing/minif2f-deepseek-vllm-valid.yaml
export ORCH_EXPERIMENT_ID=deepseek-vllm/minif2f-valid

mkdir -p ".slurm/pipeline/${ORCH_EXPERIMENT_ID}"
sbatch \
  --output=".slurm/pipeline/${ORCH_EXPERIMENT_ID}/%x_%j.out" \
  --error=".slurm/pipeline/${ORCH_EXPERIMENT_ID}/%x_%j.err" \
  --export=ALL,REPO_ROOT,ORCH_CONFIG,ORCH_EXPERIMENT_ID \
  scripts/bash/testing_pipeline/slurm/run_orchestrate_pipeline.sh
# What are the Right Symmetries for Formal Theorem Proving?

Code implementing rewriting ensembles, introduced in the paper "What are the Right Symmetries for Formal Theorem Proving?".

## Installation

Start by creating a Python 3.10 env
```bash
conda create -n rwens python=3.10
conda activate rwens
```
Then, install torch appropriately for your device. The project was originally run with torch `2.9.1+cu128`.
```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining requirements
```bash
pip install -r requirements.txt
```

Install the package in development mode:

```bash
pip install -e .
```

## Running proof generation

Run this from the repository root on a machine with a CUDA GPU (interactive shell or an allocated GPU session). The example below uses the miniF2F valid split config bundled in this repo.

```bash
conda activate rwens
python scripts/src/eval/generate_solutions.py --config configs/testing/miniF2F/minif2f-valid-deepseek-noncot.yaml
```

The YAML points at a prover model config (`prover:`), dataset (`dataset_path`, `split`), sampling settings (`num_attempts`, `batch_size`), and an output namespace (`experiment_id`). With `minif2f-valid-deepseek-noncot.yaml`, outputs are written under:

`results/minif2f/valid/deepseek-miniF2F-valid-noncot/`

Each selected problem gets its own subdirectory containing `attempts.jsonl`. Adjust `limit`, `experiment_id`, or other fields in that YAML to change how many problems run and where results go.

While the model loads you may see tokenizer messages about `rope_scaling` field types; they do not stop vLLM from starting once weights are loaded and CUDA graph capture finishes.

## Verifying proofs on Slurm

After generation has produced `attempts.jsonl` under each problem folder, submit reverification as Slurm array jobs from the repo root:

```bash
bash scripts/run_verification.sh results/minif2f/valid/deepseek-miniF2F-valid-noncot
```

The wrapper `cd`s into `REPO_ROOT` and runs `scripts/src/testing_pipeline/submit_verification_array.py`. By default `REPO_ROOT` is `/slurm-storage/krzole/rw-ens` and `RWENS_PYTHON` is `/slurm-storage/krzole/.conda/envs/rwens/bin/python`. On another host or layout, set them before invoking the script:

```bash
export REPO_ROOT=/path/to/rw-ens
export RWENS_PYTHON=/path/to/conda/envs/rwens/bin/python
bash scripts/run_verification.sh results/minif2f/valid/deepseek-miniF2F-valid-noncot
```

Slurm prints the submitted job id(s). Logs are typically under `.slurm/pipeline/<folder_name>/` (for example `.slurm/pipeline/deepseek-miniF2F-valid-noncot/` when the argument is the path shown above). Problems that already have reverification outputs are skipped unless you pass extra flags through to `submit_verification_array.py` (for example `--no-skip-existing`). Any arguments after the solutions directory are forwarded to that Python entrypoint.

You can increase the number of files processed simoultaneously by specifying the `max-concurrent` parameter:
```bash
bash scripts/run_verification.sh results/minif2f/valid/deepseek-miniF2F-valid-noncot --max-concurrent 4
```

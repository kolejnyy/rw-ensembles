#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/repo}"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <solutions_dir> [extra args for submit_verification_array.py]" >&2
  echo "Example: $0 results/minif2f/valid/my_run_001 --max-concurrent 1" >&2
  exit 1
fi

SOLUTIONS_DIR="$1"
shift

INVPRO_PYTHON="${INVPRO_PYTHON:-/path/to/conda/env/bin/python}"
"${INVPRO_PYTHON}" scripts/src/testing_pipeline/submit_verification_array.py "${SOLUTIONS_DIR}" "$@"

#!/bin/bash
set -euo pipefail

cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <solutions_dir> [extra args for submit_verification_array.py]" >&2
  echo "Example: $0 results/minif2f/valid/my_run_001 --max-concurrent 1" >&2
  exit 1
fi

SOLUTIONS_DIR="$1"
shift

"${RWENS_PYTHON}" scripts/src/testing_pipeline/submit_verification_array.py "${SOLUTIONS_DIR}" "$@"

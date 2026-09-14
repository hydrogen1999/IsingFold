#!/bin/bash
# Run an inclusive, optionally strided training-grid shard directly on Apollo (no Slurm).

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 STAGE FIRST_INDEX LAST_INDEX" >&2
  exit 2
fi
: "${ISINGFOLD_WORK_ROOT:?set ISINGFOLD_WORK_ROOT to the staged EmbedBench root}"

stage="$1"
first="$2"
last="$3"
python="${ISINGFOLD_PYTHON:-$HOME/isingfold/.venv/bin/python}"
grid_config="${ISINGFOLD_GRID_CONFIG:-configs/training_grid_v1.json}"
grid_step="${ISINGFOLD_GRID_STEP:-1}"
execution_site="${ISINGFOLD_EXECUTION_SITE:-apollo}"
if [[ ! "${grid_step}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ISINGFOLD_GRID_STEP must be a positive integer" >&2
  exit 2
fi
export PYTHONPATH="${ISINGFOLD_WORK_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ISINGFOLD_WORK_ROOT}"
for ((index = first; index <= last; index += grid_step)); do
  "${python}" scripts/run_training_grid.py \
    --grid "${grid_config}" \
    --stage "${stage}" \
    --index "${index}" \
    --root "${ISINGFOLD_WORK_ROOT}" \
    --python "${python}" \
    --execution-site "${execution_site}" \
    --device cuda
done

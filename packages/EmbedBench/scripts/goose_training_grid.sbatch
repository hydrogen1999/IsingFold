#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# FIRST/LAST are inclusive and STEP selects a registered strided site shard.

set -euo pipefail

: "${ISINGFOLD_WORK_ROOT:?set ISINGFOLD_WORK_ROOT to the staged EmbedBench root}"
: "${ISINGFOLD_GRID_STAGE:?set ISINGFOLD_GRID_STAGE to a registered stage name}"
: "${ISINGFOLD_GRID_SIZE:?set ISINGFOLD_GRID_SIZE to the registered stage size}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"
: "${SLURM_ARRAY_TASK_COUNT:?submit this script as a Slurm array}"

if [[ ! "${ISINGFOLD_GRID_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ISINGFOLD_GRID_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] ||
   [[ ! "${SLURM_ARRAY_TASK_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Slurm array task ID/count must be non-negative/positive integers" >&2
  exit 2
fi

GRID_SIZE_VALUE=$((10#${ISINGFOLD_GRID_SIZE}))
ARRAY_TASK_ID_VALUE=$((10#${SLURM_ARRAY_TASK_ID}))
ARRAY_TASK_COUNT_VALUE=$((10#${SLURM_ARRAY_TASK_COUNT}))
if (( ARRAY_TASK_COUNT_VALUE != 2 )); then
  echo "Goose accounting policy requires exactly two array tasks" >&2
  exit 2
fi
if (( ARRAY_TASK_ID_VALUE >= ARRAY_TASK_COUNT_VALUE )); then
  echo "submit the Goose array with task IDs 0-1" >&2
  exit 2
fi

ISINGFOLD_GRID_FIRST_RAW="${ISINGFOLD_GRID_FIRST:-0}"
ISINGFOLD_GRID_LAST_RAW="${ISINGFOLD_GRID_LAST:-}"
ISINGFOLD_GRID_STEP_RAW="${ISINGFOLD_GRID_STEP:-1}"
ISINGFOLD_GRID_WORKERS_PER_TASK_RAW="${ISINGFOLD_GRID_WORKERS_PER_TASK:-1}"
if [[ ! "${ISINGFOLD_GRID_FIRST_RAW}" =~ ^[0-9]+$ ]] ||
   [[ -n "${ISINGFOLD_GRID_LAST_RAW}" && ! "${ISINGFOLD_GRID_LAST_RAW}" =~ ^[0-9]+$ ]] ||
   [[ ! "${ISINGFOLD_GRID_STEP_RAW}" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "${ISINGFOLD_GRID_WORKERS_PER_TASK_RAW}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ISINGFOLD_GRID_FIRST/LAST/STEP/WORKERS_PER_TASK must be non-negative/non-negative/positive/positive integers" >&2
  exit 2
fi
GRID_FIRST_VALUE=$((10#${ISINGFOLD_GRID_FIRST_RAW}))
GRID_STEP_VALUE=$((10#${ISINGFOLD_GRID_STEP_RAW}))
GRID_WORKERS_PER_TASK_VALUE=$((10#${ISINGFOLD_GRID_WORKERS_PER_TASK_RAW}))
if [[ -n "${ISINGFOLD_GRID_LAST_RAW}" ]]; then
  GRID_LAST_VALUE=$((10#${ISINGFOLD_GRID_LAST_RAW}))
else
  GRID_LAST_VALUE=$((GRID_SIZE_VALUE - 1))
fi
if (( GRID_FIRST_VALUE > GRID_LAST_VALUE || GRID_LAST_VALUE >= GRID_SIZE_VALUE )); then
  echo "requested grid range must satisfy 0 <= FIRST <= LAST < ISINGFOLD_GRID_SIZE" >&2
  exit 2
fi

ISINGFOLD_PYTHON="${ISINGFOLD_PYTHON:-$HOME/isingfold/.venv/bin/python}"
ISINGFOLD_GRID_CONFIG_VALUE="${ISINGFOLD_GRID_CONFIG:-configs/training_grid_v1.json}"
ISINGFOLD_EXECUTION_SITE_VALUE="${ISINGFOLD_EXECUTION_SITE:-goose}"
export PYTHONPATH="${ISINGFOLD_WORK_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
SLURM_CPUS_PER_TASK_VALUE="${SLURM_CPUS_PER_TASK:-8}"
if [[ ! "${SLURM_CPUS_PER_TASK_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLURM_CPUS_PER_TASK must be a positive integer" >&2
  exit 2
fi
OMP_THREADS_VALUE=$((10#${SLURM_CPUS_PER_TASK_VALUE} / GRID_WORKERS_PER_TASK_VALUE))
if (( OMP_THREADS_VALUE < 1 )); then
  OMP_THREADS_VALUE=1
fi
export OMP_NUM_THREADS="${OMP_THREADS_VALUE}"

cd "${ISINGFOLD_WORK_ROOT}"

run_worker() {
  local worker_id=$1
  local lane=$((ARRAY_TASK_ID_VALUE + worker_id * ARRAY_TASK_COUNT_VALUE))
  local worker_stride=$((ARRAY_TASK_COUNT_VALUE * GRID_WORKERS_PER_TASK_VALUE * GRID_STEP_VALUE))
  local grid_index
  for (( grid_index=GRID_FIRST_VALUE+lane*GRID_STEP_VALUE; grid_index<=GRID_LAST_VALUE; grid_index+=worker_stride )); do
    "${ISINGFOLD_PYTHON}" scripts/run_training_grid.py \
      --grid "${ISINGFOLD_GRID_CONFIG_VALUE}" \
      --stage "${ISINGFOLD_GRID_STAGE}" \
      --index "${grid_index}" \
      --root "${ISINGFOLD_WORK_ROOT}" \
      --python "${ISINGFOLD_PYTHON}" \
      --execution-site "${ISINGFOLD_EXECUTION_SITE_VALUE}" \
      --device cuda
  done
}

worker_pids=()
for (( WORKER_ID=0; WORKER_ID<GRID_WORKERS_PER_TASK_VALUE; WORKER_ID++ )); do
  run_worker "${WORKER_ID}" &
  worker_pids+=("$!")
done

worker_failure=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "${worker_pid}"; then
    worker_failure=1
  fi
done
exit "${worker_failure}"

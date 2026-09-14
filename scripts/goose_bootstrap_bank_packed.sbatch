#!/usr/bin/env bash
# Packed fallback when Goose rejects a large Slurm bootstrap array.
# One allocation executes immutable bootstrap shards through round-robin workers.
#SBATCH --job-name=isingfold-bootstrap-packed
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=isingfold-bootstrap-packed-%j.out
#SBATCH --error=isingfold-bootstrap-packed-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_bootstrap_bank_packed.sbatch requires one non-array Slurm job" >&2
  exit 69
fi

: "${ISINGFOLD_GRID:?set ISINGFOLD_GRID}"
: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
: "${ISINGFOLD_BOOTSTRAP_PRESET:?set ISINGFOLD_BOOTSTRAP_PRESET}"
: "${ISINGFOLD_BOOTSTRAP_BANK_PLAN:?set ISINGFOLD_BOOTSTRAP_BANK_PLAN}"
: "${ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256:?set ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256}"
: "${ISINGFOLD_BOOTSTRAP_BANK:?set ISINGFOLD_BOOTSTRAP_BANK}"
: "${ISINGFOLD_BOOTSTRAP_BANK_LOG_ROOT:?set ISINGFOLD_BOOTSTRAP_BANK_LOG_ROOT}"
: "${ISINGFOLD_BOOTSTRAP_BANK_SHARD_COUNT:?set ISINGFOLD_BOOTSTRAP_BANK_SHARD_COUNT}"

workers=${ISINGFOLD_BOOTSTRAP_BANK_WORKERS:-16}
if [[ "$ISINGFOLD_BOOTSTRAP_PRESET" != "representation-validation" \
   && "$ISINGFOLD_BOOTSTRAP_PRESET" != "validation" \
   && "$ISINGFOLD_BOOTSTRAP_PRESET" != "final-test" ]]; then
  echo "ISINGFOLD_BOOTSTRAP_PRESET must be representation-validation, validation or final-test" >&2
  exit 64
fi
if [[ ! "$ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 64
fi
if [[ ! "$ISINGFOLD_BOOTSTRAP_BANK_SHARD_COUNT" =~ ^[1-9][0-9]*$ \
   || ! "$workers" =~ ^[1-9][0-9]*$ \
   || ! "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ \
   || "$workers" -gt "$ISINGFOLD_BOOTSTRAP_BANK_SHARD_COUNT" \
   || "$workers" -gt "$SLURM_CPUS_PER_TASK" ]]; then
  echo "workers must be positive and no greater than shard count or allocation CPUs" >&2
  exit 64
fi

threads=1
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime \
  "$script_dir" cpu "$ISINGFOLD_BOOTSTRAP_BANK" "$ISINGFOLD_BOOTSTRAP_BANK_LOG_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" cpu

export ISINGFOLD_BOOTSTRAP_BANK_WORKERS=$workers
/opt/slurm/bin/srun --exact --nodes=1 --ntasks="$workers" --cpus-per-task=1 \
  --cpu-bind=cores /bin/bash -c '
    set -euo pipefail
    container_python=$1
    shift
    workers=${ISINGFOLD_BOOTSTRAP_BANK_WORKERS:?}
    index=${SLURM_PROCID:?}
    while (( index < ISINGFOLD_BOOTSTRAP_BANK_SHARD_COUNT )); do
      "$@" "$container_python" \
        -I -m isingfold.rl.cli generate-bootstrap-bank-shard \
        --grid "$ISINGFOLD_GRID" \
        --corpus "$ISINGFOLD_CORPUS" \
        --config "$ISINGFOLD_COMPLETE_CONFIG" \
        --protocol-preset "$ISINGFOLD_BOOTSTRAP_PRESET" \
        --plan "$ISINGFOLD_BOOTSTRAP_BANK_PLAN" \
        --expected-plan-sha256 "$ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256" \
        --bank "$ISINGFOLD_BOOTSTRAP_BANK" \
        --shard-index "$index" \
        --shard-count "$ISINGFOLD_BOOTSTRAP_BANK_SHARD_COUNT" \
        >"$ISINGFOLD_BOOTSTRAP_BANK_LOG_ROOT/shard-$index.out" \
        2>"$ISINGFOLD_BOOTSTRAP_BANK_LOG_ROOT/shard-$index.err"
      index=$((index + workers))
    done
  ' isingfold-packed-bootstrap-worker "$container_python" "${container_exec[@]}"

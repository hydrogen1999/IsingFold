#!/usr/bin/env bash
# Goose must run this file through a Slurm array; login-node execution is rejected.
#SBATCH --job-name=isingfold-init-bank
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=isingfold-init-bank-%A_%a.out
#SBATCH --error=isingfold-init-bank-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_initializer_bank.sbatch must run as a Slurm array job" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
: "${ISINGFOLD_INITIALIZER_BANK_PLAN:?set ISINGFOLD_INITIALIZER_BANK_PLAN}"
: "${ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256:?set ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256}"
: "${ISINGFOLD_INITIALIZER_BANK:?set ISINGFOLD_INITIALIZER_BANK}"
: "${ISINGFOLD_INITIALIZER_BANK_SHARD_COUNT:?set ISINGFOLD_INITIALIZER_BANK_SHARD_COUNT}"

if [[ ! "$ISINGFOLD_INITIALIZER_BANK_SHARD_COUNT" =~ ^[1-9][0-9]*$ \
   || "$SLURM_ARRAY_TASK_ID" -ge "$ISINGFOLD_INITIALIZER_BANK_SHARD_COUNT" ]]; then
  echo "Slurm array index must be below ISINGFOLD_INITIALIZER_BANK_SHARD_COUNT" >&2
  exit 64
fi

threads=${ISINGFOLD_THREADS:-4}
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" cpu "$ISINGFOLD_INITIALIZER_BANK"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" cpu

/opt/slurm/bin/srun --cpus-per-task="$threads" \
  "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli generate-initializer-bank-shard \
  --corpus "$ISINGFOLD_CORPUS" \
  --config "$ISINGFOLD_COMPLETE_CONFIG" \
  --plan "$ISINGFOLD_INITIALIZER_BANK_PLAN" \
  --expected-plan-sha256 "$ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256" \
  --bank "$ISINGFOLD_INITIALIZER_BANK" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --shard-count "$ISINGFOLD_INITIALIZER_BANK_SHARD_COUNT"

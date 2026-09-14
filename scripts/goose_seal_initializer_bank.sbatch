#!/usr/bin/env bash
# Seal a completed initializer bank inside one authenticated Goose allocation.
# This is deliberately a non-array job and may depend on the generating job.
#SBATCH --job-name=isingfold-seal-init
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=isingfold-seal-init-%j.out
#SBATCH --error=isingfold-seal-init-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_seal_initializer_bank.sbatch requires one non-array Slurm job" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
: "${ISINGFOLD_INITIALIZER_BANK_PLAN:?set ISINGFOLD_INITIALIZER_BANK_PLAN}"
: "${ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256:?set ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256}"
: "${ISINGFOLD_INITIALIZER_BANK:?set ISINGFOLD_INITIALIZER_BANK}"

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" cpu "$ISINGFOLD_INITIALIZER_BANK"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" cpu

seal_args=(
  --corpus "$ISINGFOLD_CORPUS"
  --config "$ISINGFOLD_COMPLETE_CONFIG"
  --plan "$ISINGFOLD_INITIALIZER_BANK_PLAN"
  --expected-plan-sha256 "$ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256"
  --bank "$ISINGFOLD_INITIALIZER_BANK"
)
if [[ -n "${ISINGFOLD_QUBIT_CAP:-}" ]]; then
  if [[ ! "$ISINGFOLD_QUBIT_CAP" =~ ^[1-9][0-9]*$ ]]; then
    echo "ISINGFOLD_QUBIT_CAP must be a positive integer" >&2
    exit 64
  fi
  seal_args+=(--qubit-cap "$ISINGFOLD_QUBIT_CAP")
fi

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli seal-initializer-bank \
  "${seal_args[@]}"

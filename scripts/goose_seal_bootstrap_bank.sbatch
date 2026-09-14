#!/usr/bin/env bash
# Seal one completed validation or final-test bootstrap bank on a Goose compute node.
#SBATCH --job-name=isingfold-seal-bootstrap
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=isingfold-seal-bootstrap-%j.out
#SBATCH --error=isingfold-seal-bootstrap-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_seal_bootstrap_bank.sbatch requires one non-array Slurm job" >&2
  exit 69
fi

: "${ISINGFOLD_GRID:?set ISINGFOLD_GRID}"
: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
: "${ISINGFOLD_BOOTSTRAP_PRESET:?set ISINGFOLD_BOOTSTRAP_PRESET}"
: "${ISINGFOLD_BOOTSTRAP_BANK_PLAN:?set ISINGFOLD_BOOTSTRAP_BANK_PLAN}"
: "${ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256:?set ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256}"
: "${ISINGFOLD_BOOTSTRAP_BANK:?set ISINGFOLD_BOOTSTRAP_BANK}"
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

threads=${ISINGFOLD_THREADS:-4}
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" cpu "$ISINGFOLD_BOOTSTRAP_BANK"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" cpu

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli seal-bootstrap-bank \
  --grid "$ISINGFOLD_GRID" \
  --corpus "$ISINGFOLD_CORPUS" \
  --config "$ISINGFOLD_COMPLETE_CONFIG" \
  --protocol-preset "$ISINGFOLD_BOOTSTRAP_PRESET" \
  --plan "$ISINGFOLD_BOOTSTRAP_BANK_PLAN" \
  --expected-plan-sha256 "$ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256" \
  --bank "$ISINGFOLD_BOOTSTRAP_BANK"

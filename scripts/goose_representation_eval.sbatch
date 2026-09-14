#!/usr/bin/env bash
# Goose must execute this file through sbatch as a nine-cell array.
#SBATCH --job-name=isingfold-rep-eval
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=isingfold-rep-eval-%A_%a.out
#SBATCH --error=isingfold-rep-eval-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_representation_eval.sbatch must run as a Slurm array job" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${ISINGFOLD_RUN_ROOT:?set ISINGFOLD_RUN_ROOT}"
: "${ISINGFOLD_REP_EVALUATIONS_ROOT:?set ISINGFOLD_REP_EVALUATIONS_ROOT}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
: "${ISINGFOLD_VALIDATION_BOOTSTRAP_BANK:?set ISINGFOLD_VALIDATION_BOOTSTRAP_BANK}"
: "${EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256:?set EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256}"
: "${EXPECTED_VALIDATION_BOOTSTRAP_MANIFEST_SHA256:?set EXPECTED_VALIDATION_BOOTSTRAP_MANIFEST_SHA256}"
: "${ISINGFOLD_GRID_STEP:?set ISINGFOLD_GRID_STEP}"

grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
first=${ISINGFOLD_GRID_FIRST:-0}
step=$ISINGFOLD_GRID_STEP
threads=${ISINGFOLD_THREADS:-4}

if [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ || ! "$first" =~ ^[0-9]+$ || ! "$step" =~ ^[1-9][0-9]*$ ]]; then
  echo "array index, ISINGFOLD_GRID_FIRST and ISINGFOLD_GRID_STEP must be valid non-negative bounds" >&2
  exit 64
fi
if [[ "$step" != "3" ]]; then
  echo "the registered three-seed grid requires ISINGFOLD_GRID_STEP=3" >&2
  exit 64
fi
index=$((first + SLURM_ARRAY_TASK_ID * step))
if (( index > 8 )); then
  echo "computed representation evaluation index is outside 0-8" >&2
  exit 64
fi

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" gpu "$ISINGFOLD_REP_EVALUATIONS_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli evaluate-representation-cell \
  --grid "$grid" \
  --index "$index" \
  --corpus "$ISINGFOLD_CORPUS" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
  --run-root "$ISINGFOLD_RUN_ROOT" \
  --complete-config "$ISINGFOLD_COMPLETE_CONFIG" \
  --bootstrap-bank "$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK" \
  --expected-bootstrap-plan-sha256 "$EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256" \
  --expected-bootstrap-manifest-sha256 "$EXPECTED_VALIDATION_BOOTSTRAP_MANIFEST_SHA256" \
  --evaluations-root "$ISINGFOLD_REP_EVALUATIONS_ROOT" \
  --device cuda \
  --threads "$threads"

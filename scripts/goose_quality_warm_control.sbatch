#!/usr/bin/env bash
# Goose must execute this file through sbatch --array=0-2.
#SBATCH --job-name=isingfold-qwarm-control
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=isingfold-qwarm-%A_%a.out
#SBATCH --error=isingfold-qwarm-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_quality_warm_control.sbatch must run as a Slurm array job" >&2
  exit 69
fi
if [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[0-2]$ ]]; then
  echo "submit exactly --array=0-2" >&2
  exit 64
fi

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${ISINGFOLD_QUALITY_LABELS:?set ISINGFOLD_QUALITY_LABELS}"
: "${ISINGFOLD_REPRESENTATION_RUN_ROOT:?set ISINGFOLD_REPRESENTATION_RUN_ROOT}"
: "${ISINGFOLD_RUN_ROOT:?set ISINGFOLD_RUN_ROOT}"
: "${EXPECTED_QUALITY_WARM_CONTROL_CONFIG_SHA256:?set EXPECTED_QUALITY_WARM_CONTROL_CONFIG_SHA256}"
: "${EXPECTED_ISINGFOLD_GRID_SHA256:?set EXPECTED_ISINGFOLD_GRID_SHA256}"
: "${ISINGFOLD_SELECTION_RECEIPT:?set ISINGFOLD_SELECTION_RECEIPT}"
: "${EXPECTED_ISINGFOLD_SELECTION_RECEIPT_SHA256:?set EXPECTED_ISINGFOLD_SELECTION_RECEIPT_SHA256}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${QUALITY_PREFLIGHT_RECEIPT:?set QUALITY_PREFLIGHT_RECEIPT}"
: "${EXPECTED_QUALITY_PREFLIGHT_SHA256:?set EXPECTED_QUALITY_PREFLIGHT_SHA256}"
: "${QUALITY_INITIALIZER_BANK:?set QUALITY_INITIALIZER_BANK}"
: "${EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256:?set EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256}"
: "${QUALITY_COMPLETE_CONFIG:?set QUALITY_COMPLETE_CONFIG}"

config=${ISINGFOLD_QUALITY_WARM_CONTROL_CONFIG:-configs/quality_warm_control_hybrid_v1.json}
grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
threads=${ISINGFOLD_THREADS:-4}

source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" gpu "$ISINGFOLD_RUN_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli warm-quality-control-cell \
  --config "$config" \
  --expected-config-sha256 "$EXPECTED_QUALITY_WARM_CONTROL_CONFIG_SHA256" \
  --grid "$grid" \
  --expected-grid-sha256 "$EXPECTED_ISINGFOLD_GRID_SHA256" \
  --corpus "$ISINGFOLD_CORPUS" \
  --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --quality-labels "$ISINGFOLD_QUALITY_LABELS" \
  --quality-initializer-bank "$QUALITY_INITIALIZER_BANK" \
  --expected-quality-initializer-bank-manifest-sha256 "$EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256" \
  --quality-complete-config "$QUALITY_COMPLETE_CONFIG" \
  --quality-preflight-receipt "$QUALITY_PREFLIGHT_RECEIPT" \
  --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256" \
  --representation-selection-receipt "$ISINGFOLD_SELECTION_RECEIPT" \
  --expected-representation-selection-sha256 "$EXPECTED_ISINGFOLD_SELECTION_RECEIPT_SHA256" \
  --representation-run-root "$ISINGFOLD_REPRESENTATION_RUN_ROOT" \
  --run-root "$ISINGFOLD_RUN_ROOT" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --device cuda \
  --threads "$threads"

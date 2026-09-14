#!/usr/bin/env bash
# Goose must execute this file through sbatch. Direct login-node execution is rejected.
#SBATCH --job-name=isingfold-rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=isingfold-%A_%a.out
#SBATCH --error=isingfold-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_rl_grid.sbatch must run as a Slurm array job; never on the login node" >&2
  exit 69
fi

: "${ISINGFOLD_GRID_STAGE:?set ISINGFOLD_GRID_STAGE}"
: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${ISINGFOLD_QUALITY_LABELS:?set ISINGFOLD_QUALITY_LABELS}"
: "${ISINGFOLD_RUN_ROOT:?set ISINGFOLD_RUN_ROOT}"
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
case "$ISINGFOLD_GRID_STAGE" in
  representation) maximum_index=8 ;;
  rl_value) maximum_index=17 ;;
  *)
    echo "ISINGFOLD_GRID_STAGE must be representation or rl_value" >&2
    exit 64
    ;;
esac
index=$((first + SLURM_ARRAY_TASK_ID * step))
if (( index > maximum_index )); then
  echo "computed $ISINGFOLD_GRID_STAGE grid index is outside 0-$maximum_index" >&2
  exit 64
fi

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" gpu "$ISINGFOLD_RUN_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

selection=()
if [[ "$ISINGFOLD_GRID_STAGE" == "representation" ]]; then
  if [[ -z "${ISINGFOLD_GATE_RECEIPT:-}" ]]; then
    echo "representation requires ISINGFOLD_GATE_RECEIPT" >&2
    exit 64
  fi
  : "${EXPECTED_ISINGFOLD_GATE_RECEIPT_SHA256:?set EXPECTED_ISINGFOLD_GATE_RECEIPT_SHA256}"
  if [[ ! "$EXPECTED_ISINGFOLD_GATE_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "EXPECTED_ISINGFOLD_GATE_RECEIPT_SHA256 must be a lowercase SHA-256" >&2
    exit 64
  fi
  selection=(
    --gate-receipt "$ISINGFOLD_GATE_RECEIPT"
    --expected-gate-receipt-sha256 "$EXPECTED_ISINGFOLD_GATE_RECEIPT_SHA256"
  )
else
  : "${ISINGFOLD_INITIALIZER_BANK_ROOT:?set ISINGFOLD_INITIALIZER_BANK_ROOT}"
  : "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
  if [[ -n "${ISINGFOLD_SELECTION_RECEIPT:-}" ]]; then
    selection=(--selection-receipt "$ISINGFOLD_SELECTION_RECEIPT")
  elif [[ -n "${ISINGFOLD_SELECTED_SIMPLER:-}" && "${ISINGFOLD_DIAGNOSTIC_MANUAL_SELECTION:-0}" == "1" ]]; then
    selection=(--selected-simpler "$ISINGFOLD_SELECTED_SIMPLER" --diagnostic-manual-selection)
  else
    echo "rl_value requires ISINGFOLD_SELECTION_RECEIPT (manual choice is diagnostic-only)" >&2
    exit 64
  fi
fi

initializer_bank_args=()
if [[ "$ISINGFOLD_GRID_STAGE" == "rl_value" ]]; then
  seed_index=$((index % 3))
  initializer_bank="$ISINGFOLD_INITIALIZER_BANK_ROOT/seed-$seed_index"
  bank_pin_name="ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_${seed_index}"
  initializer_bank_sha256=${!bank_pin_name:-}
  if [[ ! "$initializer_bank_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$bank_pin_name must be the externally recorded lowercase SHA-256" >&2
    exit 64
  fi
  initializer_bank_args=(
    --initializer-bank "$initializer_bank"
    --expected-initializer-bank-manifest-sha256 "$initializer_bank_sha256"
    --complete-config "$ISINGFOLD_COMPLETE_CONFIG"
  )
fi

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli grid-cell \
  --grid "$grid" \
  --stage "$ISINGFOLD_GRID_STAGE" \
  --index "$index" \
  --corpus "$ISINGFOLD_CORPUS" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
  --quality-labels "$ISINGFOLD_QUALITY_LABELS" \
  --quality-initializer-bank "$QUALITY_INITIALIZER_BANK" \
  --expected-quality-initializer-bank-manifest-sha256 "$EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256" \
  --quality-complete-config "$QUALITY_COMPLETE_CONFIG" \
  --quality-preflight-receipt "$QUALITY_PREFLIGHT_RECEIPT" \
  --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256" \
  --run-root "$ISINGFOLD_RUN_ROOT" \
  --device cuda \
  --threads "$threads" \
  "${initializer_bank_args[@]}" \
  "${selection[@]}"

#!/usr/bin/env bash
# Apollo has no Slurm. This launcher runs assigned cells directly and sequentially.
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 STAGE CORPUS SELECTOR_BUNDLE QUALITY_LABELS RUN_ROOT FIRST [LAST]" >&2
  exit 64
fi

stage=$1
corpus=$2
selector_bundle=$3
quality_labels=$4
run_root=$5
first=$6
last=${7:-$first}
grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
: "${ISINGFOLD_GRID_STEP:?set ISINGFOLD_GRID_STEP}"
step=$ISINGFOLD_GRID_STEP
device=${ISINGFOLD_DEVICE:-cuda}
threads=${ISINGFOLD_THREADS:-1}
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

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" gpu

if [[ "$stage" != "representation" && "$stage" != "rl_value" ]]; then
  echo "STAGE must be representation or rl_value" >&2
  exit 64
fi
if [[ ! "$first" =~ ^[0-9]+$ || ! "$last" =~ ^[0-9]+$ || ! "$step" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIRST, LAST and ISINGFOLD_GRID_STEP must be non-negative integer bounds" >&2
  exit 64
fi
if [[ "$step" != "3" ]]; then
  echo "the registered three-seed grid requires ISINGFOLD_GRID_STEP=3" >&2
  exit 64
fi

selection=()
if [[ "$stage" == "representation" ]]; then
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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
index=$first
while (( index <= last )); do
  initializer_bank_args=()
  if [[ "$stage" == "rl_value" ]]; then
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
  "$python_bin" -I -m isingfold.rl.cli grid-cell \
    --grid "$grid" \
    --stage "$stage" \
    --index "$index" \
    --corpus "$corpus" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --selector "$selector_bundle" \
    --quality-labels "$quality_labels" \
    --quality-initializer-bank "$QUALITY_INITIALIZER_BANK" \
    --expected-quality-initializer-bank-manifest-sha256 "$EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256" \
    --quality-complete-config "$QUALITY_COMPLETE_CONFIG" \
    --quality-preflight-receipt "$QUALITY_PREFLIGHT_RECEIPT" \
    --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256" \
    --run-root "$run_root" \
    --device "$device" \
    --threads "$threads" \
    "${initializer_bank_args[@]}" \
    "${selection[@]}"
  index=$((index + step))
done

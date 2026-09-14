#!/usr/bin/env bash
# Apollo has no Slurm. Run assigned selected-seed confirmation cells directly.
set -euo pipefail

if [[ $# -lt 8 || $# -gt 9 ]]; then
  echo "usage: $0 CORPUS SELECTOR QUALITY_LABELS RUN_ROOT FREEZE FREEZE_SHA FIRST LAST [STEP]" >&2
  exit 64
fi

corpus=$1
selector_bundle=$2
quality_labels=$3
run_root=$4
freeze=$5
freeze_sha=$6
first=$7
last=$8
step=${9:-1}
grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
device=${ISINGFOLD_DEVICE:-cuda}
threads=${ISINGFOLD_THREADS:-1}
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${QUALITY_PREFLIGHT_RECEIPT:?set QUALITY_PREFLIGHT_RECEIPT}"
: "${EXPECTED_QUALITY_PREFLIGHT_SHA256:?set EXPECTED_QUALITY_PREFLIGHT_SHA256}"
: "${ISINGFOLD_INITIALIZER_BANK_ROOT:?set ISINGFOLD_INITIALIZER_BANK_ROOT}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" gpu

if [[ ! "$first" =~ ^[0-2]$ || ! "$last" =~ ^[0-2]$ || ! "$step" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIRST and LAST must be in [0,2]; STEP must be positive" >&2
  exit 64
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
index=$first
while (( index <= last )); do
  initializer_bank="$ISINGFOLD_INITIALIZER_BANK_ROOT/seed-$index"
  bank_pin_name="ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_${index}"
  initializer_bank_sha256=${!bank_pin_name:-}
  if [[ ! "$initializer_bank_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "set $bank_pin_name to the out-of-band manifest SHA-256" >&2
    exit 64
  fi
  "$python_bin" -I -m isingfold.rl.cli complete-system-train-cell \
    --grid "$grid" \
    --index "$index" \
    --corpus "$corpus" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --selector "$selector_bundle" \
    --quality-labels "$quality_labels" \
    --quality-preflight-receipt "$QUALITY_PREFLIGHT_RECEIPT" \
    --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256" \
    --run-root "$run_root" \
    --initializer-bank "$initializer_bank" \
    --expected-initializer-bank-manifest-sha256 "$initializer_bank_sha256" \
    --complete-config "$ISINGFOLD_COMPLETE_CONFIG" \
    --rl-value-selection-receipt "$freeze" \
    --expected-selection-sha256 "$freeze_sha" \
    --device "$device" \
    --threads "$threads"
  index=$((index + step))
done

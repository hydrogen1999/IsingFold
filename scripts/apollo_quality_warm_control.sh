#!/usr/bin/env bash
# Apollo has no Slurm. Run one or more registered control indices directly.
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 CORPUS SELECTOR QUALITY_LABELS REPRESENTATION_RUN_ROOT RUN_ROOT FIRST [LAST]" >&2
  exit 64
fi

corpus=$1
selector=$2
quality_labels=$3
representation_run_root=$4
run_root=$5
first=$6
last=${7:-$first}
config=${ISINGFOLD_QUALITY_WARM_CONTROL_CONFIG:-configs/quality_warm_control_hybrid_v1.json}
grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
device=${ISINGFOLD_DEVICE:-cuda}
threads=${ISINGFOLD_THREADS:-1}

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

if [[ ! "$first" =~ ^[0-2]$ || ! "$last" =~ ^[0-2]$ || "$first" -gt "$last" ]]; then
  echo "FIRST and LAST must satisfy 0 <= FIRST <= LAST <= 2" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" gpu

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
for ((index = first; index <= last; index++)); do
  "$python_bin" -I -m isingfold.rl.cli warm-quality-control-cell \
    --config "$config" \
    --expected-config-sha256 "$EXPECTED_QUALITY_WARM_CONTROL_CONFIG_SHA256" \
    --grid "$grid" \
    --expected-grid-sha256 "$EXPECTED_ISINGFOLD_GRID_SHA256" \
    --corpus "$corpus" \
    --selector "$selector" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --quality-labels "$quality_labels" \
    --quality-initializer-bank "$QUALITY_INITIALIZER_BANK" \
    --expected-quality-initializer-bank-manifest-sha256 "$EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256" \
    --quality-complete-config "$QUALITY_COMPLETE_CONFIG" \
    --quality-preflight-receipt "$QUALITY_PREFLIGHT_RECEIPT" \
    --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256" \
    --representation-selection-receipt "$ISINGFOLD_SELECTION_RECEIPT" \
    --expected-representation-selection-sha256 "$EXPECTED_ISINGFOLD_SELECTION_RECEIPT_SHA256" \
    --representation-run-root "$representation_run_root" \
    --run-root "$run_root" \
    --index "$index" \
    --device "$device" \
    --threads "$threads"
done

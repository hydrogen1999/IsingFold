#!/usr/bin/env bash
# Apollo has no Slurm. Run assigned RL-value validation cells directly.
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 CORPUS SELECTOR_BUNDLE RUN_ROOT REPRESENTATION_SELECTION_RECEIPT EVALUATIONS_ROOT FIRST [LAST]" >&2
  exit 64
fi

corpus=$1
selector_bundle=$2
run_root=$3
representation_selection=$4
evaluations_root=$5
first=$6
last=${7:-$first}
grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
step=${ISINGFOLD_GRID_STEP:-1}
device=${ISINGFOLD_DEVICE:-cuda}
threads=${ISINGFOLD_THREADS:-1}
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
: "${ISINGFOLD_VALIDATION_BOOTSTRAP_BANK:?set ISINGFOLD_VALIDATION_BOOTSTRAP_BANK}"
: "${EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256:?set EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256}"
: "${EXPECTED_VALIDATION_BOOTSTRAP_MANIFEST_SHA256:?set EXPECTED_VALIDATION_BOOTSTRAP_MANIFEST_SHA256}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" gpu

if [[ ! "$first" =~ ^[0-9]+$ || ! "$last" =~ ^[0-9]+$ || ! "$step" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIRST, LAST and ISINGFOLD_GRID_STEP must be non-negative integer bounds" >&2
  exit 64
fi
if (( first > last || last > 17 )); then
  echo "RL-value evaluation bounds must satisfy 0 <= FIRST <= LAST <= 17" >&2
  exit 64
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
index=$first
while (( index <= last )); do
  "$python_bin" -I -m isingfold.rl.cli evaluate-rl-value-cell \
    --grid "$grid" \
    --index "$index" \
    --corpus "$corpus" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --selector "$selector_bundle" \
    --run-root "$run_root" \
    --complete-config "$ISINGFOLD_COMPLETE_CONFIG" \
    --bootstrap-bank "$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK" \
    --expected-bootstrap-plan-sha256 "$EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256" \
    --expected-bootstrap-manifest-sha256 "$EXPECTED_VALIDATION_BOOTSTRAP_MANIFEST_SHA256" \
    --representation-selection-receipt "$representation_selection" \
    --evaluations-root "$evaluations_root" \
    --device "$device" \
    --threads "$threads"
  index=$((index + step))
done

#!/usr/bin/env bash
# Apollo has no Slurm. Run assigned representation evaluations directly.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 CORPUS SELECTOR_BUNDLE RUN_ROOT EVALUATIONS_ROOT FIRST [LAST]" >&2
  exit 64
fi

corpus=$1
selector_bundle=$2
run_root=$3
evaluations_root=$4
first=$5
last=${6:-$first}
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
if [[ "$step" != "3" ]]; then
  echo "the registered three-seed grid requires ISINGFOLD_GRID_STEP=3" >&2
  exit 64
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
index=$first
while (( index <= last )); do
  "$python_bin" -I -m isingfold.rl.cli evaluate-representation-cell \
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
    --evaluations-root "$evaluations_root" \
    --device "$device" \
    --threads "$threads"
  index=$((index + step))
done

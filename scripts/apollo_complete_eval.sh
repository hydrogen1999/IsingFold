#!/usr/bin/env bash
# Apollo has no Slurm. Run assigned whole-system confirmation cells directly.
set -euo pipefail

if [[ $# -lt 9 || $# -gt 10 ]]; then
  echo "usage: $0 CORPUS SELECTOR RUN_ROOT CONFIG OUT_ROOT FREEZE FREEZE_SHA FIRST LAST [STEP]" >&2
  exit 64
fi

corpus=$1
selector_bundle=$2
run_root=$3
config=$4
out_root=$5
freeze=$6
freeze_sha=$7
first=$8
last=$9
step=${10:-1}
grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
device=${ISINGFOLD_DEVICE:-cuda}
threads=${ISINGFOLD_THREADS:-1}
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT:?set ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT}"
: "${EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256:?set EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256}"
: "${EXTERNAL_TUNING_SELECTION:?set EXTERNAL_TUNING_SELECTION}"
: "${EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256:?set EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256}"
: "${ISINGFOLD_FINAL_TEST_BOOTSTRAP_BANK:?set ISINGFOLD_FINAL_TEST_BOOTSTRAP_BANK}"
: "${EXPECTED_FINAL_TEST_BOOTSTRAP_PLAN_SHA256:?set EXPECTED_FINAL_TEST_BOOTSTRAP_PLAN_SHA256}"
: "${EXPECTED_FINAL_TEST_BOOTSTRAP_MANIFEST_SHA256:?set EXPECTED_FINAL_TEST_BOOTSTRAP_MANIFEST_SHA256}"
external_config=${ISINGFOLD_EXTERNAL_COMPLETE_CONFIG:-configs/external_minorminer_complete_v1.json}
tuning_registry=${ISINGFOLD_EXTERNAL_TUNING_REGISTRY:-configs/external_minorminer_tuning_hybrid_v1.json}

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
  "$python_bin" -I -m isingfold.rl.cli evaluate-complete-system-cell \
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
    --config "$config" \
    --bootstrap-bank "$ISINGFOLD_FINAL_TEST_BOOTSTRAP_BANK" \
    --expected-bootstrap-plan-sha256 "$EXPECTED_FINAL_TEST_BOOTSTRAP_PLAN_SHA256" \
    --expected-bootstrap-manifest-sha256 "$EXPECTED_FINAL_TEST_BOOTSTRAP_MANIFEST_SHA256" \
    --out "$out_root/seed-$index" \
    --rl-value-selection-receipt "$freeze" \
    --expected-selection-sha256 "$freeze_sha" \
    --device "$device" \
    --threads "$threads"
  "$python_bin" -I -m isingfold.rl.cli evaluate-external-complete-system \
    --grid "$grid" \
    --index "$index" \
    --corpus "$corpus" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --selector "$selector_bundle" \
    --config "$external_config" \
    --learned-config "$config" \
    --tuning-registry "$tuning_registry" \
    --expected-tuning-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
    --external-tuning-selection "$EXTERNAL_TUNING_SELECTION" \
    --expected-external-tuning-selection-sha256 "$EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256" \
    --out "$ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT/seed-$index" \
    --device "$device" \
    --threads "$threads"
  index=$((index + step))
done

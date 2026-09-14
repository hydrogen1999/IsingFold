#!/usr/bin/env bash
# Apollo has no Slurm. Run one stock arm for each matching learned seed index directly.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 CORPUS SELECTOR_BUNDLE OUT_ROOT FIRST LAST [STEP]" >&2
  exit 64
fi

corpus=$1
selector_bundle=$2
out_root=$3
first=$4
last=$5
step=${6:-1}
grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
external_config=${ISINGFOLD_EXTERNAL_COMPLETE_CONFIG:-configs/external_minorminer_complete_v1.json}
learned_config=${ISINGFOLD_COMPLETE_CONFIG:-configs/complete_system_lac_hybrid_cache_v1.json}
tuning_registry=${ISINGFOLD_EXTERNAL_TUNING_REGISTRY:-configs/external_minorminer_tuning_hybrid_v1.json}
device=${ISINGFOLD_DEVICE:-cuda}
threads=${ISINGFOLD_THREADS:-1}
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256:?set EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256}"
: "${EXTERNAL_TUNING_SELECTION:?set EXTERNAL_TUNING_SELECTION}"
: "${EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256:?set EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" gpu

if [[ ! "$first" =~ ^[0-2]$ || ! "$last" =~ ^[0-2]$ || ! "$step" =~ ^[1-9][0-9]*$ ]]; then
  echo "FIRST and LAST must be in [0,2]; STEP must be positive" >&2
  exit 64
fi

index=$first
while (( index <= last )); do
  "$python_bin" -I -m isingfold.rl.cli evaluate-external-complete-system \
    --grid "$grid" \
    --index "$index" \
    --corpus "$corpus" \
    --selector "$selector_bundle" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --config "$external_config" \
    --learned-config "$learned_config" \
    --tuning-registry "$tuning_registry" \
    --expected-tuning-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
    --external-tuning-selection "$EXTERNAL_TUNING_SELECTION" \
    --expected-external-tuning-selection-sha256 "$EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256" \
    --device "$device" \
    --threads "$threads" \
    --out "$out_root/seed-$index"
  index=$((index + step))
done

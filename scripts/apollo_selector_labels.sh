#!/usr/bin/env bash
# Build the authenticated train-only selector labels directly on Apollo.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CORPUS SELECTOR_LABELS_OUT" >&2
  exit 64
fi
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "apollo_selector_labels.sh rejects Slurm environments" >&2
  exit 69
fi

corpus=$1
selector_labels=$2
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"

sample_seed=${ISINGFOLD_SELECTOR_LABEL_SEED:-503}
split_seed=${ISINGFOLD_SELECTOR_SPLIT_SEED:-601}
calibration_fraction=${ISINGFOLD_SELECTOR_CALIBRATION_FRACTION:-0.2}
if [[ ! "$sample_seed" =~ ^[0-9]+$ || ! "$split_seed" =~ ^[0-9]+$ ]]; then
  echo "selector label and split seeds must be non-negative integers" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" cpu

"$python_bin" -I -m isingfold.rl.cli label-selector-data \
  --corpus "$corpus" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --seed "$sample_seed" \
  --split-seed "$split_seed" \
  --calibration-fraction "$calibration_fraction" \
  --out "$selector_labels"

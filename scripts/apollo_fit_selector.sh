#!/usr/bin/env bash
# Train and freeze IF-Q3-S0 directly on one Apollo GPU.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CORPUS SELECTOR_LABELS SELECTOR_BUNDLE_OUT" >&2
  exit 64
fi
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "apollo_fit_selector.sh rejects Slurm environments" >&2
  exit 69
fi

corpus=$1
selector_labels=$2
selector_bundle=$3
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"

seed=${ISINGFOLD_SELECTOR_TRAINING_SEED:-701}
epochs=${ISINGFOLD_SELECTOR_EPOCHS:-100}
graph_minibatch=${ISINGFOLD_SELECTOR_GRAPH_MINIBATCH:-32}
threads=${ISINGFOLD_THREADS:-4}
for value in "$seed" "$epochs" "$graph_minibatch" "$threads"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "selector seed, epochs, minibatch and threads must be integers" >&2
    exit 64
  fi
done
if (( epochs == 0 || graph_minibatch == 0 || threads == 0 )); then
  echo "selector epochs, minibatch and threads must be positive" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" gpu

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
"$python_bin" -I -m isingfold.rl.cli fit-selector \
  --corpus "$corpus" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --selector-labels "$selector_labels" \
  --seed "$seed" \
  --epochs "$epochs" \
  --graph-minibatch "$graph_minibatch" \
  --device cuda \
  --threads "$threads" \
  --out "$selector_bundle"

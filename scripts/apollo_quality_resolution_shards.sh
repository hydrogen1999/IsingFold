#!/usr/bin/env bash
# Execute or independently verify a deterministic resolution-shard range on Apollo.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 {run|verify} STAGE_INDEX SHARD_COUNT FIRST [LAST]" >&2
  exit 64
fi
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "apollo_quality_resolution_shards.sh rejects Slurm environments" >&2
  exit 69
fi
mode=$1
stage_index=$2
shard_count=$3
first=$4
last=${5:-$first}
: "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?set ISINGFOLD_EXECUTION_RUNTIME_SHA256}"
: "${ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT:?set ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT}"
: "${ISINGFOLD_QUALITY_RESOLUTION_VERIFICATION_ROOT:?set ISINGFOLD_QUALITY_RESOLUTION_VERIFICATION_ROOT}"
for value in "$stage_index" "$shard_count" "$first" "$last"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "stage and shard indices must be non-negative integers" >&2
    exit 64
  fi
done
if [[ "$mode" != "run" && "$mode" != "verify" ]] \
   || (( shard_count == 0 || first > last || last >= shard_count )); then
  echo "require run|verify and 0 <= FIRST <= LAST < SHARD_COUNT" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
source "$script_dir/quality_protocol_common.sh"
quality_initialize_protocol_arguments
quality_resolution_plan_args
require_quality_accelerator_control "$ISINGFOLD_QUALITY_ACCELERATOR" "$python_bin"
runtime_validation_json=$(
  "$script_dir/verify_runtime_source.sh" "$python_bin" "$ISINGFOLD_QUALITY_ACCELERATOR"
)
quality_bind_verified_runtime_identity "$runtime_validation_json" "$python_bin"
quality_prepare_output_root ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT
quality_prepare_output_root ISINGFOLD_QUALITY_RESOLUTION_VERIFICATION_ROOT

if [[ "$mode" == "verify" ]]; then
  : "${ISINGFOLD_QUALITY_RESOLUTION_VERIFIER_IDENTITY:?set ISINGFOLD_QUALITY_RESOLUTION_VERIFIER_IDENTITY}"
  : "${EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256:?set EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256}"
  quality_require_digest EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256
  : "${ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE:?set ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE}"
  quality_require_regular_pin_file ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE
fi

for ((index=first; index<=last; index++)); do
  printf -v shard_name 'stage-%02d-shard-%05d-of-%05d' \
    "$stage_index" "$index" "$shard_count"
  common_args=(
    "${quality_plan_args[@]}" --stage-index "$stage_index" --shard-index "$index"
    "${quality_public_args[@]}" "${quality_compute_args[@]}"
    "${quality_initializer_args[@]}"
    --execution-runtime-sha256 "$ISINGFOLD_EXECUTION_RUNTIME_SHA256"
    "${quality_optional_qubit_args[@]}"
  )
  if [[ "$mode" == "run" ]]; then
    "$python_bin" -I -m isingfold.rl.cli run-quality-resolution-shard \
      "${common_args[@]}" --out "$ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT/$shard_name"
  else
    delta_root="$ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT/$shard_name"
    delta_manifest="$delta_root/manifest.json"
    if [[ ! -f "$delta_manifest" || -L "$delta_manifest" ]]; then
      echo "missing regular delta manifest: $delta_manifest" >&2
      exit 78
    fi
    quality_lookup_manifest_pin \
      "$delta_root" "$ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE"
    EXPECTED_QUALITY_RESOLUTION_DELTA_MANIFEST_SHA256=$quality_lookup_sha256
    "$python_bin" -I -m isingfold.rl.cli verify-quality-resolution-shard \
      "${common_args[@]}" --delta-root "$delta_root" \
      --expected-delta-manifest-sha256 \
      "$EXPECTED_QUALITY_RESOLUTION_DELTA_MANIFEST_SHA256" \
      --verifier-identity "$ISINGFOLD_QUALITY_RESOLUTION_VERIFIER_IDENTITY" \
      --expected-verifier-identity-sha256 \
      "$EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256" \
      --out "$ISINGFOLD_QUALITY_RESOLUTION_VERIFICATION_ROOT/$shard_name.json"
  fi
done

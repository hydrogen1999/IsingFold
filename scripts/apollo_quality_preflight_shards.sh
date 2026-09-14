#!/usr/bin/env bash
# Replay a deterministic quality-preflight shard range directly on Apollo.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SHARD_COUNT FIRST [LAST]" >&2
  exit 64
fi
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "apollo_quality_preflight_shards.sh rejects Slurm environments" >&2
  exit 69
fi
shard_count=$1
first=$2
last=${3:-$first}
: "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?set ISINGFOLD_EXECUTION_RUNTIME_SHA256}"
: "${ISINGFOLD_QUALITY_PREFLIGHT_PLAN:?set ISINGFOLD_QUALITY_PREFLIGHT_PLAN}"
: "${EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256:?set EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256}"
: "${ISINGFOLD_QUALITY_SHARD_ROOT:?set ISINGFOLD_QUALITY_SHARD_ROOT}"
: "${ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT:?set ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT}"
for value in "$shard_count" "$first" "$last"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "shard indices must be non-negative integers" >&2
    exit 64
  fi
done
if (( shard_count == 0 || first > last || last >= shard_count )); then
  echo "require 0 <= FIRST <= LAST < SHARD_COUNT" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
source "$script_dir/quality_protocol_common.sh"
quality_initialize_protocol_arguments
quality_resolution_receipt_args
quality_require_digest EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256
require_quality_accelerator_control "$ISINGFOLD_QUALITY_ACCELERATOR" "$python_bin"
runtime_validation_json=$(
  "$script_dir/verify_runtime_source.sh" "$python_bin" "$ISINGFOLD_QUALITY_ACCELERATOR"
)
quality_bind_verified_runtime_identity "$runtime_validation_json" "$python_bin"
quality_prepare_output_root ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT

for ((index=first; index<=last; index++)); do
  printf -v shard_name 'shard-%05d-of-%05d' "$index" "$shard_count"
  "$python_bin" -I -m isingfold.rl.cli run-quality-preflight-shard \
    --plan "$ISINGFOLD_QUALITY_PREFLIGHT_PLAN" \
    --expected-plan-sha256 "$EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256" \
    "${quality_study_args[@]}" --shard-index "$index" \
    "${quality_public_args[@]}" --quality-shard-root "$ISINGFOLD_QUALITY_SHARD_ROOT" \
    "${quality_initializer_args[@]}" \
    --execution-runtime-sha256 "$ISINGFOLD_EXECUTION_RUNTIME_SHA256" \
    "${quality_compute_args[@]}" "${quality_optional_qubit_args[@]}" \
    --out "$ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT/$shard_name"
done

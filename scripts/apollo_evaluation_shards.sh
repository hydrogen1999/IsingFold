#!/usr/bin/env bash
# Apollo has no Slurm. Run a sealed shard range directly on the assigned host.
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 WORKFLOW PLAN PLAN_SHA256 SHARD_COUNT OUT_ROOT -- WORKFLOW_ARGS..." >&2
  exit 64
fi

workflow=$1
plan=$2
plan_sha256=$3
shard_count=$4
out_root=$5
shift 5
if [[ ${1:-} == "--" ]]; then
  shift
fi
if [[ ! "$shard_count" =~ ^[1-9][0-9]*$ || $# -eq 0 ]]; then
  echo "SHARD_COUNT must be positive and WORKFLOW_ARGS must be supplied" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_cuda_cli_arguments "$@"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" gpu
environment_lock=$ISINGFOLD_ENV_LOCK
environment_lock_sha256=$ISINGFOLD_ENV_LOCK_SHA256
workflow_args=("$@")

for ((shard_index = 0; shard_index < shard_count; shard_index++)); do
  "$python_bin" -I -m isingfold.rl.cli run-evaluation-shard \
    --workflow "$workflow" \
    --plan "$plan" \
    --expected-plan-sha256 "$plan_sha256" \
    --shard-index "$shard_index" \
    --out "$out_root/shard-$shard_index" \
    --cluster apollo \
    --execution-mode pinned-venv \
    --environment-lock "$environment_lock" \
    --expected-environment-lock-sha256 "$environment_lock_sha256" \
    "${workflow_args[@]}"
done

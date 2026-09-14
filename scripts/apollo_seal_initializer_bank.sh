#!/usr/bin/env bash
# Seal one completed initializer bank directly on Apollo and emit its manifest pin.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CORPUS COMPLETE_CONFIG PLAN PLAN_SHA256 BANK" >&2
  exit 64
fi
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "apollo_seal_initializer_bank.sh rejects Slurm environments" >&2
  exit 69
fi

corpus=$1
config=$2
plan=$3
plan_sha256=$4
bank=$5
if [[ ! "$plan_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "PLAN_SHA256 must be one lowercase SHA-256 digest" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" cpu

"$python_bin" -I -m isingfold.rl.cli seal-initializer-bank \
  --corpus "$corpus" \
  --config "$config" \
  --plan "$plan" \
  --expected-plan-sha256 "$plan_sha256" \
  --bank "$bank"

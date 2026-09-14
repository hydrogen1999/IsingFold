#!/usr/bin/env bash
# Apollo has no Slurm. Generate assigned target-free K=2 bootstrap shards directly.
set -euo pipefail

if [[ $# -lt 10 || $# -gt 11 ]]; then
  echo "usage: $0 GRID CORPUS CONFIG PRESET PLAN PLAN_SHA256 BANK SHARD_COUNT FIRST LAST [STEP]" >&2
  exit 64
fi

grid=$1
corpus=$2
config=$3
preset=$4
plan=$5
plan_sha256=$6
bank=$7
shard_count=$8
first=$9
last=${10}
step=${11:-1}

if [[ "$preset" != "representation-validation" \
   && "$preset" != "validation" \
   && "$preset" != "final-test" ]]; then
  echo "PRESET must be representation-validation, validation or final-test" >&2
  exit 64
fi
if [[ ! "$plan_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "PLAN_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 64
fi
if [[ ! "$shard_count" =~ ^[1-9][0-9]*$ \
   || ! "$first" =~ ^[0-9]+$ \
   || ! "$last" =~ ^[0-9]+$ \
   || ! "$step" =~ ^[1-9][0-9]*$ \
   || "$first" -ge "$shard_count" \
   || "$last" -ge "$shard_count" \
   || "$first" -gt "$last" ]]; then
  echo "SHARD_COUNT/STEP must be positive and 0 <= FIRST <= LAST < SHARD_COUNT" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" cpu

index=$first
while (( index <= last )); do
  "$python_bin" -I -m isingfold.rl.cli generate-bootstrap-bank-shard \
    --grid "$grid" \
    --corpus "$corpus" \
    --config "$config" \
    --protocol-preset "$preset" \
    --plan "$plan" \
    --expected-plan-sha256 "$plan_sha256" \
    --bank "$bank" \
    --shard-index "$index" \
    --shard-count "$shard_count"
  index=$((index + step))
done

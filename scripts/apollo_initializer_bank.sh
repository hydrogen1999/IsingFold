#!/usr/bin/env bash
# Apollo has no Slurm. Generate assigned initializer-bank shards directly.
set -euo pipefail

if [[ $# -lt 8 || $# -gt 9 ]]; then
  echo "usage: $0 CORPUS CONFIG PLAN PLAN_SHA256 BANK SHARD_COUNT FIRST LAST [STEP]" >&2
  exit 64
fi

corpus=$1
config=$2
plan=$3
plan_sha256=$4
bank=$5
shard_count=$6
first=$7
last=$8
step=${9:-1}

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
  "$python_bin" -I -m isingfold.rl.cli generate-initializer-bank-shard \
    --corpus "$corpus" \
    --config "$config" \
    --plan "$plan" \
    --expected-plan-sha256 "$plan_sha256" \
    --bank "$bank" \
    --shard-index "$index" \
    --shard-count "$shard_count"
  index=$((index + step))
done

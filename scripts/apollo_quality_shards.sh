#!/usr/bin/env bash
# Apollo has no Slurm. Run one deterministic shard-index sequence per direct process.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 CORPUS SELECTOR_BUNDLE SHARD_ROOT SHARD_COUNT FIRST [LAST]" >&2
  exit 64
fi

corpus=$1
selector_bundle=$2
shard_root=$3
shard_count=$4
first=$5
last=${6:-$first}
step=${ISINGFOLD_SHARD_STEP:-1}
concurrency=${ISINGFOLD_SHARD_CONCURRENCY:-1}
# Publication quality-v8 is defined over every authenticated train lineage.
# A positive subset is diagnostic-only and cannot satisfy the sealed protocol.
instances=${ISINGFOLD_QUALITY_LINEAGES:-0}
tasks_per_lineage=${ISINGFOLD_QUALITY_TASKS_PER_LINEAGE:-1}
states_per_lineage=${ISINGFOLD_QUALITY_STATES_PER_LINEAGE:-4}
actions=${ISINGFOLD_QUALITY_ACTIONS:-8}
seed=${ISINGFOLD_QUALITY_SEED:-907}
: "${ISINGFOLD_QUALITY_ACCELERATOR:?freeze ISINGFOLD_QUALITY_ACCELERATOR to cpu or gpu}"
quality_accelerator=$ISINGFOLD_QUALITY_ACCELERATOR
if [[ "$quality_accelerator" == "gpu" ]]; then
  device=cuda
else
  device=cpu
fi
if [[ -n "${ISINGFOLD_DEVICE:-}" && "$ISINGFOLD_DEVICE" != "$device" ]]; then
  echo "ISINGFOLD_DEVICE conflicts with the frozen quality accelerator" >&2
  exit 78
fi
export ISINGFOLD_DEVICE=$device
threads=${ISINGFOLD_THREADS:-1}
gpu_id_spec=${ISINGFOLD_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}
: "${ISINGFOLD_QUALITY_RESOLUTION_PLAN:?set ISINGFOLD_QUALITY_RESOLUTION_PLAN}"
: "${EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256:?set EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256}"
: "${ISINGFOLD_QUALITY_RESOLUTION_RECEIPT:?set ISINGFOLD_QUALITY_RESOLUTION_RECEIPT}"
: "${EXPECTED_QUALITY_RESOLUTION_RECEIPT_SHA256:?set EXPECTED_QUALITY_RESOLUTION_RECEIPT_SHA256}"
: "${ISINGFOLD_QUALITY_CAPACITY_SELECTION:?set ISINGFOLD_QUALITY_CAPACITY_SELECTION}"
: "${EXPECTED_QUALITY_CAPACITY_SELECTION_SHA256:?set EXPECTED_QUALITY_CAPACITY_SELECTION_SHA256}"
: "${ISINGFOLD_QUALITY_CAPACITY_BUDGET:?set ISINGFOLD_QUALITY_CAPACITY_BUDGET}"
: "${EXPECTED_QUALITY_CAPACITY_BUDGET_SHA256:?set EXPECTED_QUALITY_CAPACITY_BUDGET_SHA256}"
: "${ISINGFOLD_QUALITY_CAPACITY_CANARY:?set ISINGFOLD_QUALITY_CAPACITY_CANARY}"
: "${EXPECTED_QUALITY_CAPACITY_CANARY_SHA256:?set EXPECTED_QUALITY_CAPACITY_CANARY_SHA256}"
: "${ISINGFOLD_QUALITY_INITIALIZER_BANK:?set ISINGFOLD_QUALITY_INITIALIZER_BANK}"
: "${EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256:?set EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
require_quality_accelerator_control "$quality_accelerator" "$python_bin"
"$script_dir/verify_runtime_source.sh" "$python_bin" "$quality_accelerator"

for value in "$shard_count" "$first" "$last" "$step" "$concurrency"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "SHARD_COUNT, FIRST, LAST, STEP and CONCURRENCY must be integers" >&2
    exit 64
  fi
done
if (( shard_count == 0 || step == 0 || concurrency == 0 || first > last || last >= shard_count )); then
  echo "require 0 <= FIRST <= LAST < SHARD_COUNT and positive STEP/CONCURRENCY" >&2
  exit 64
fi

gpu_mode=0
if [[ "$device" == "cuda" || "$device" == "auto" ]]; then
  gpu_mode=1
fi
gpu_ids=()
if (( gpu_mode )); then
  IFS=',' read -r -a gpu_ids <<< "$gpu_id_spec"
  for gpu_id in "${gpu_ids[@]}"; do
    if [[ -z "$gpu_id" || "$gpu_id" =~ [[:space:]] ]]; then
      echo "ISINGFOLD_GPU_IDS must be a comma-separated list without empty or spaced IDs" >&2
      exit 64
    fi
  done
  if (( ${#gpu_ids[@]} < concurrency )); then
    echo "CUDA/auto mode requires at least CONCURRENCY distinct ISINGFOLD_GPU_IDS" >&2
    exit 64
  fi
  declare -A seen_gpu_ids=()
  for gpu_id in "${gpu_ids[@]:0:concurrency}"; do
    if [[ -n "${seen_gpu_ids[$gpu_id]:-}" ]]; then
      echo "the first CONCURRENCY entries in ISINGFOLD_GPU_IDS must be distinct" >&2
      exit 64
    fi
    seen_gpu_ids[$gpu_id]=1
  done
fi

mkdir -p "$shard_root"

run_shard() {
  local index=$1
  local gpu_id=$2
  local shard_name
  printf -v shard_name 'shard-%05d-of-%05d' "$index" "$shard_count"
  local -a command=("$python_bin" -I -m isingfold.rl.cli label-quality \
    --corpus "$corpus" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --selector "$selector_bundle" \
    --initializer-bank "$ISINGFOLD_QUALITY_INITIALIZER_BANK" \
    --expected-initializer-bank-manifest-sha256 \
      "$EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256" \
    --complete-config "$ISINGFOLD_COMPLETE_CONFIG" \
    --resolution-plan "$ISINGFOLD_QUALITY_RESOLUTION_PLAN" \
    --expected-resolution-plan-sha256 "$EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256" \
    --resolution-receipt "$ISINGFOLD_QUALITY_RESOLUTION_RECEIPT" \
    --expected-resolution-receipt-sha256 "$EXPECTED_QUALITY_RESOLUTION_RECEIPT_SHA256" \
    --capacity-selection "$ISINGFOLD_QUALITY_CAPACITY_SELECTION" \
    --expected-capacity-selection-sha256 "$EXPECTED_QUALITY_CAPACITY_SELECTION_SHA256" \
    --capacity-budget "$ISINGFOLD_QUALITY_CAPACITY_BUDGET" \
    --expected-capacity-budget-sha256 "$EXPECTED_QUALITY_CAPACITY_BUDGET_SHA256" \
    --capacity-canary "$ISINGFOLD_QUALITY_CAPACITY_CANARY" \
    --expected-capacity-canary-sha256 "$EXPECTED_QUALITY_CAPACITY_CANARY_SHA256" \
    --instances "$instances" \
    --tasks-per-lineage "$tasks_per_lineage" \
    --states-per-lineage "$states_per_lineage" \
    --actions "$actions" \
    --seed "$seed" \
    --device "$device" \
    --threads "$threads" \
    --shard-index "$index" \
    --shard-count "$shard_count" \
    --out "$shard_root/$shard_name")
  if (( gpu_mode )); then
    CUDA_VISIBLE_DEVICES="$gpu_id" "${command[@]}"
  else
    "${command[@]}"
  fi
}

child_pids=()
wait_batch() {
  local status=0
  local pid
  for pid in "${child_pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  child_pids=()
  return "$status"
}

terminate_children() {
  local pid
  for pid in "${child_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

index=$first
while (( index <= last )); do
  slot=${#child_pids[@]}
  gpu_id=""
  if (( gpu_mode )); then
    gpu_id=${gpu_ids[$slot]}
  fi
  run_shard "$index" "$gpu_id" &
  child_pids+=("$!")
  if (( ${#child_pids[@]} >= concurrency )); then
    wait_batch
  fi
  index=$((index + step))
done
wait_batch
trap - INT TERM

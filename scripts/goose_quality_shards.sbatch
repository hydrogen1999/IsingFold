#!/usr/bin/env bash
# Goose must run this through sbatch. Two array allocations each process a shard sequence.
#SBATCH --job-name=isingfold-quality
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=isingfold-quality-%A_%a.out
#SBATCH --error=isingfold-quality-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_quality_shards.sbatch must run as a Slurm array job, never on the login node" >&2
  exit 69
fi
if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != "2" || ! "$SLURM_ARRAY_TASK_ID" =~ ^[01]$ ]]; then
  echo "quality sharding requires exactly the registered two-element Goose array 0-1" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${ISINGFOLD_QUALITY_SHARD_ROOT:?set ISINGFOLD_QUALITY_SHARD_ROOT}"
: "${ISINGFOLD_QUALITY_SHARD_COUNT:?set ISINGFOLD_QUALITY_SHARD_COUNT}"
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
: "${ISINGFOLD_QUALITY_ACCELERATOR:?freeze ISINGFOLD_QUALITY_ACCELERATOR to cpu or gpu}"

shard_count=$ISINGFOLD_QUALITY_SHARD_COUNT
first_base=${ISINGFOLD_SHARD_FIRST:-0}
step=${ISINGFOLD_SHARD_STEP:-2}
for value in "$shard_count" "$first_base" "$step"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "shard count and FIRST/STEP must be integers" >&2
    exit 64
  fi
done
if [[ "$step" != "$SLURM_ARRAY_TASK_COUNT" ]]; then
  echo "ISINGFOLD_SHARD_STEP must equal the registered Slurm array size" >&2
  exit 64
fi
first=$((first_base + SLURM_ARRAY_TASK_ID))
last=${ISINGFOLD_SHARD_LAST:-$((shard_count - 1))}
# Publication quality-v8 is defined over every authenticated train lineage.
# A positive subset is diagnostic-only and cannot satisfy the sealed protocol.
instances=${ISINGFOLD_QUALITY_LINEAGES:-0}
tasks_per_lineage=${ISINGFOLD_QUALITY_TASKS_PER_LINEAGE:-1}
states_per_lineage=${ISINGFOLD_QUALITY_STATES_PER_LINEAGE:-4}
actions=${ISINGFOLD_QUALITY_ACTIONS:-8}
seed=${ISINGFOLD_QUALITY_SEED:-907}
threads=${ISINGFOLD_THREADS:-4}
quality_accelerator=$ISINGFOLD_QUALITY_ACCELERATOR
if [[ "$quality_accelerator" == "gpu" ]]; then
  quality_device=cuda
else
  quality_device=cpu
fi

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_quality_accelerator_control "$quality_accelerator" /usr/bin/python3
require_goose_publication_runtime \
  "$script_dir" "$quality_accelerator" "$ISINGFOLD_QUALITY_SHARD_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" "$quality_accelerator"

if [[ ! "$last" =~ ^[0-9]+$ ]]; then
  echo "LAST must be an integer" >&2
  exit 64
fi
if (( shard_count == 0 || step == 0 || first > last || last >= shard_count )); then
  echo "require 0 <= FIRST <= LAST < SHARD_COUNT and STEP > 0" >&2
  exit 64
fi

mkdir -p "$ISINGFOLD_QUALITY_SHARD_ROOT"
index=$first
while (( index <= last )); do
  printf -v shard_name 'shard-%05d-of-%05d' "$index" "$shard_count"
  /opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
    -I -m isingfold.rl.cli label-quality \
    --corpus "$ISINGFOLD_CORPUS" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
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
    --device "$quality_device" \
    --threads "$threads" \
    --shard-index "$index" \
    --shard-count "$shard_count" \
    --out "$ISINGFOLD_QUALITY_SHARD_ROOT/$shard_name"
  index=$((index + step))
done

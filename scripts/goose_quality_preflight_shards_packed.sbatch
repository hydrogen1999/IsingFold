#!/usr/bin/env bash
# Replay the registered 64-shard quality-preflight census in two GPU-owning jobs.
#SBATCH --job-name=isingfold-quality-preflight-packed
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=isingfold-quality-preflight-packed-%A_%a.out
#SBATCH --error=isingfold-quality-preflight-packed-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_quality_preflight_shards_packed.sbatch must run through Slurm, never on the login node" >&2
  exit 69
fi
if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != "2" \
   || ! "$SLURM_ARRAY_TASK_ID" =~ ^[01]$ ]]; then
  echo "packed quality preflight requires exactly the registered array 0-1" >&2
  exit 69
fi

registered_shard_count=64
: "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?set ISINGFOLD_EXECUTION_RUNTIME_SHA256}"
: "${ISINGFOLD_QUALITY_PREFLIGHT_PLAN:?set ISINGFOLD_QUALITY_PREFLIGHT_PLAN}"
: "${EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256:?set EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256}"
: "${ISINGFOLD_QUALITY_SHARD_ROOT:?set ISINGFOLD_QUALITY_SHARD_ROOT}"
: "${ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT:?set ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT}"

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
source "$script_dir/quality_protocol_common.sh"
quality_initialize_protocol_arguments
quality_resolution_receipt_args
quality_require_digest EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256
require_quality_accelerator_control "$ISINGFOLD_QUALITY_ACCELERATOR" /usr/bin/python3
threads=$quality_threads
if [[ "$ISINGFOLD_QUALITY_ACCELERATOR" != "gpu" ]]; then
  echo "packed preflight workers require the registered GPU accelerator" >&2
  exit 64
fi
require_goose_publication_runtime \
  "$script_dir" gpu "$ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT"
runtime_validation_json=$(
  /opt/slurm/bin/srun "${container_exec[@]}" \
    bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu
)
quality_bind_verified_runtime_identity "$runtime_validation_json" /usr/bin/python3

processed=0
index=$SLURM_ARRAY_TASK_ID
while (( index < registered_shard_count )); do
  printf -v shard_name 'shard-%05d-of-00064' "$index"
  output="$ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT/$shard_name"
  if [[ -e "$output" || -L "$output" ]]; then
    echo "refusing to overwrite quality-preflight replay: $output" >&2
    exit 78
  fi
  /opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
    -I -m isingfold.rl.cli run-quality-preflight-shard \
    --plan "$ISINGFOLD_QUALITY_PREFLIGHT_PLAN" \
    --expected-plan-sha256 "$EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256" \
    "${quality_study_args[@]}" --shard-index "$index" \
    "${quality_public_args[@]}" \
    --quality-shard-root "$ISINGFOLD_QUALITY_SHARD_ROOT" \
    "${quality_initializer_args[@]}" \
    --execution-runtime-sha256 "$ISINGFOLD_EXECUTION_RUNTIME_SHA256" \
    "${quality_compute_args[@]}" \
    "${quality_optional_qubit_args[@]}" \
    --out "$output"
  processed=$((processed + 1))
  index=$((index + SLURM_ARRAY_TASK_COUNT))
done

if [[ "$processed" != "32" ]]; then
  echo "packed preflight worker did not cover exactly 32 of 64 shards" >&2
  exit 70
fi

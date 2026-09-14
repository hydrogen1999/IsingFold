#!/usr/bin/env bash
# Replay all 64 quality-preflight shards in one Goose Slurm array.
#SBATCH --job-name=isingfold-quality-preflight
#SBATCH --array=0-63
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=isingfold-quality-preflight-%A_%a.out
#SBATCH --error=isingfold-quality-preflight-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_quality_preflight_shards.sbatch must run through Slurm, never on the login node" >&2
  exit 69
fi
if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != "64" \
   || ! "$SLURM_ARRAY_TASK_ID" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ ]]; then
  echo "quality preflight requires the registered array 0-63" >&2
  exit 69
fi
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
require_goose_publication_runtime \
  "$script_dir" gpu "$ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT"
runtime_validation_json=$(
  /opt/slurm/bin/srun "${container_exec[@]}" \
    bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu
)
quality_bind_verified_runtime_identity "$runtime_validation_json" /usr/bin/python3

index=$SLURM_ARRAY_TASK_ID
printf -v shard_name 'shard-%05d-of-00064' "$index"
/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli run-quality-preflight-shard \
  --plan "$ISINGFOLD_QUALITY_PREFLIGHT_PLAN" \
  --expected-plan-sha256 "$EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256" \
  "${quality_study_args[@]}" --shard-index "$index" \
  "${quality_public_args[@]}" --quality-shard-root "$ISINGFOLD_QUALITY_SHARD_ROOT" \
  "${quality_initializer_args[@]}" \
  --execution-runtime-sha256 "$ISINGFOLD_EXECUTION_RUNTIME_SHA256" \
  "${quality_compute_args[@]}" "${quality_optional_qubit_args[@]}" \
  --out "$ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_ROOT/$shard_name"

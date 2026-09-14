#!/usr/bin/env bash
# Submit with: sbatch --array=0-$((SHARD_COUNT-1)) scripts/goose_evaluation_shards.sbatch ...
# Goose execution outside a Slurm array is rejected before any artifact is opened.
#SBATCH --job-name=isingfold-eval-shard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=isingfold-eval-shard-%A_%a.out
#SBATCH --error=isingfold-eval-shard-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_evaluation_shards.sbatch must run as a Slurm array job" >&2
  exit 69
fi
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
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= shard_count )); then
  echo "Slurm array index is outside the sealed shard census" >&2
  exit 64
fi

threads=${ISINGFOLD_THREADS:-${SLURM_CPUS_PER_TASK:-}}
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
workflow_args=("$@")
require_cuda_cli_arguments "${workflow_args[@]}"
require_goose_publication_runtime "$script_dir" gpu "$out_root"
for argument in "${workflow_args[@]}"; do
  if [[ "$argument" == "--threads" || "$argument" == --threads=* ]]; then
    echo "WORKFLOW_ARGS must not override helper-validated ISINGFOLD_THREADS" >&2
    exit 64
  fi
done

/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

/opt/slurm/bin/srun "${container_exec[@]}" \
  "$container_python" -I -m isingfold.rl.cli run-evaluation-shard \
  --workflow "$workflow" \
  --plan "$plan" \
  --expected-plan-sha256 "$plan_sha256" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --out "$out_root/shard-$SLURM_ARRAY_TASK_ID" \
  --cluster goose \
  --slurm-partition "$SLURM_JOB_PARTITION" \
  --execution-mode apptainer \
  --container-runtime "$pinned_container_runtime" \
  --expected-container-runtime-sha256 "$apptainer_sha256" \
  --container-image "$pinned_container_image" \
  --expected-container-image-sha256 "$container_image_sha256" \
  "${workflow_args[@]}" \
  --threads "$threads"

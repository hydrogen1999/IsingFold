#!/usr/bin/env bash
# Run one serial post-bank quality-protocol transition on a Goose compute node.
#SBATCH --job-name=isingfold-quality-control
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=isingfold-quality-control-%j.out
#SBATCH --error=isingfold-quality-control-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_quality_control.sbatch requires one Slurm compute job, never on the login node" >&2
  exit 69
fi
: "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?set ISINGFOLD_EXECUTION_RUNTIME_SHA256}"
: "${ISINGFOLD_QUALITY_WORK_ROOT:?set ISINGFOLD_QUALITY_WORK_ROOT}"

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
source "$script_dir/quality_protocol_common.sh"
quality_initialize_protocol_arguments
quality_require_output_under_root ISINGFOLD_QUALITY_OUT ISINGFOLD_QUALITY_WORK_ROOT
if [[ "${ISINGFOLD_QUALITY_OPERATION:-}" == "run-capacity" ]]; then
  : "${ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY:?set ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY}"
  quality_require_output_under_root \
    ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY ISINGFOLD_QUALITY_WORK_ROOT
  quality_prepare_output_root ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY
fi
require_quality_accelerator_control "$ISINGFOLD_QUALITY_ACCELERATOR" /usr/bin/python3
threads=$quality_threads
require_goose_publication_runtime "$script_dir" gpu "$ISINGFOLD_QUALITY_WORK_ROOT"
runtime_validation_json=$(
  /opt/slurm/bin/srun "${container_exec[@]}" \
    bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu
)
quality_bind_verified_runtime_identity "$runtime_validation_json" /usr/bin/python3

source "$script_dir/quality_control_command.sh"
quality_cli=(
  /opt/slurm/bin/srun "${container_exec[@]}"
  "$container_python" -I -m isingfold.rl.cli
)
run_quality_control_operation

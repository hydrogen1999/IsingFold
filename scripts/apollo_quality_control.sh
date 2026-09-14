#!/usr/bin/env bash
# Run one serial post-bank quality-protocol transition directly on Apollo.
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: configure ISINGFOLD_QUALITY_OPERATION and run $0 without arguments" >&2
  exit 64
fi
if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "apollo_quality_control.sh rejects Slurm environments" >&2
  exit 69
fi
: "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?set ISINGFOLD_EXECUTION_RUNTIME_SHA256}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
source "$script_dir/quality_protocol_common.sh"
quality_initialize_protocol_arguments
if [[ "${ISINGFOLD_QUALITY_OPERATION:-}" == "run-capacity" ]]; then
  : "${ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY:?set ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY}"
  quality_prepare_output_root ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY
fi
require_quality_accelerator_control "$ISINGFOLD_QUALITY_ACCELERATOR" "$python_bin"
runtime_accelerator=$ISINGFOLD_QUALITY_ACCELERATOR
if [[ "${ISINGFOLD_QUALITY_OPERATION:-}" == "selector-parity" ]]; then
  runtime_accelerator=gpu
fi
runtime_validation_json=$(
  "$script_dir/verify_runtime_source.sh" "$python_bin" "$runtime_accelerator"
)
quality_bind_verified_runtime_identity "$runtime_validation_json" "$python_bin"

source "$script_dir/quality_control_command.sh"
quality_cli=(
  "$python_bin" -I -m isingfold.rl.cli
)
run_quality_control_operation

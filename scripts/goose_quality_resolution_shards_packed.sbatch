#!/usr/bin/env bash
# Run the registered 64-shard quality-resolution census in two GPU-owning jobs.
#SBATCH --job-name=isingfold-quality-resolution-packed
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=isingfold-quality-resolution-packed-%A_%a.out
#SBATCH --error=isingfold-quality-resolution-packed-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_quality_resolution_shards_packed.sbatch must run through Slurm, never on the login node" >&2
  exit 69
fi
if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != "2" \
   || ! "$SLURM_ARRAY_TASK_ID" =~ ^[01]$ ]]; then
  echo "packed quality resolution requires exactly the registered array 0-1" >&2
  exit 69
fi

registered_shard_count=64
: "${ISINGFOLD_QUALITY_RESOLUTION_MODE:?set ISINGFOLD_QUALITY_RESOLUTION_MODE to run or verify}"
: "${ISINGFOLD_QUALITY_RESOLUTION_STAGE_INDEX:?set ISINGFOLD_QUALITY_RESOLUTION_STAGE_INDEX}"
: "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?set ISINGFOLD_EXECUTION_RUNTIME_SHA256}"
: "${ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT:?set ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT}"
if [[ "$ISINGFOLD_QUALITY_RESOLUTION_MODE" != "run" \
   && "$ISINGFOLD_QUALITY_RESOLUTION_MODE" != "verify" ]] \
   || [[ ! "$ISINGFOLD_QUALITY_RESOLUTION_STAGE_INDEX" =~ ^[0-9]+$ ]]; then
  echo "resolution mode/stage is invalid" >&2
  exit 64
fi

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
source "$script_dir/quality_protocol_common.sh"
quality_initialize_protocol_arguments
quality_resolution_plan_args
require_quality_accelerator_control "$ISINGFOLD_QUALITY_ACCELERATOR" /usr/bin/python3
threads=$quality_threads
if [[ "$ISINGFOLD_QUALITY_ACCELERATOR" != "gpu" ]]; then
  echo "packed resolution workers require the registered GPU accelerator" >&2
  exit 64
fi

write_roots=("$ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT")
if [[ "$ISINGFOLD_QUALITY_RESOLUTION_MODE" == "verify" ]]; then
  : "${ISINGFOLD_QUALITY_RESOLUTION_VERIFICATION_ROOT:?set ISINGFOLD_QUALITY_RESOLUTION_VERIFICATION_ROOT}"
  : "${ISINGFOLD_QUALITY_RESOLUTION_VERIFIER_IDENTITY:?set ISINGFOLD_QUALITY_RESOLUTION_VERIFIER_IDENTITY}"
  : "${EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256:?set EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256}"
  : "${ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE:?set ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE}"
  : "${EXPECTED_QUALITY_RESOLUTION_DELTA_PIN_FILE_SHA256:?set EXPECTED_QUALITY_RESOLUTION_DELTA_PIN_FILE_SHA256}"
  quality_require_digest EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256
  quality_require_digest EXPECTED_QUALITY_RESOLUTION_DELTA_PIN_FILE_SHA256
  quality_require_regular_pin_file ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE
  observed_delta_pin_file_sha256=$(
    publication_file_sha256 "$ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE"
  )
  if [[ "$observed_delta_pin_file_sha256" \
     != "$EXPECTED_QUALITY_RESOLUTION_DELTA_PIN_FILE_SHA256" ]]; then
    echo "quality-resolution delta pin registry differs from its external SHA-256" >&2
    exit 78
  fi
  if [[ "$ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT" != /* \
     || ! -d "$ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT" \
     || -L "$ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT" ]]; then
    echo "verification requires an absolute non-symlink delta root" >&2
    exit 78
  fi
  write_roots=("$ISINGFOLD_QUALITY_RESOLUTION_VERIFICATION_ROOT")
fi

require_goose_publication_runtime "$script_dir" gpu "${write_roots[@]}"
runtime_validation_json=$(
  /opt/slurm/bin/srun "${container_exec[@]}" \
    bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu
)
quality_bind_verified_runtime_identity "$runtime_validation_json" /usr/bin/python3

processed=0
index=$SLURM_ARRAY_TASK_ID
while (( index < registered_shard_count )); do
  printf -v shard_name 'stage-%02d-shard-%05d-of-00064' \
    "$ISINGFOLD_QUALITY_RESOLUTION_STAGE_INDEX" "$index"
  common_args=(
    "${quality_plan_args[@]}"
    --stage-index "$ISINGFOLD_QUALITY_RESOLUTION_STAGE_INDEX"
    --shard-index "$index"
    "${quality_public_args[@]}"
    "${quality_compute_args[@]}"
    "${quality_initializer_args[@]}"
    --execution-runtime-sha256 "$ISINGFOLD_EXECUTION_RUNTIME_SHA256"
    "${quality_optional_qubit_args[@]}"
  )
  if [[ "$ISINGFOLD_QUALITY_RESOLUTION_MODE" == "run" ]]; then
    output="$ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT/$shard_name"
    if [[ -e "$output" || -L "$output" ]]; then
      echo "refusing to overwrite quality-resolution output: $output" >&2
      exit 78
    fi
    /opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
      -I -m isingfold.rl.cli run-quality-resolution-shard \
      "${common_args[@]}" --out "$output"
  else
    delta_root="$ISINGFOLD_QUALITY_RESOLUTION_DELTA_ROOT/$shard_name"
    delta_manifest="$delta_root/manifest.json"
    if [[ ! -f "$delta_manifest" || -L "$delta_manifest" ]]; then
      echo "missing regular delta manifest: $delta_manifest" >&2
      exit 78
    fi
    quality_lookup_manifest_pin \
      "$delta_root" "$ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE"
    EXPECTED_QUALITY_RESOLUTION_DELTA_MANIFEST_SHA256=$quality_lookup_sha256
    output="$ISINGFOLD_QUALITY_RESOLUTION_VERIFICATION_ROOT/$shard_name.json"
    if [[ -e "$output" || -L "$output" ]]; then
      echo "refusing to overwrite quality-resolution verification: $output" >&2
      exit 78
    fi
    /opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
      -I -m isingfold.rl.cli verify-quality-resolution-shard \
      "${common_args[@]}" --delta-root "$delta_root" \
      --expected-delta-manifest-sha256 \
      "$EXPECTED_QUALITY_RESOLUTION_DELTA_MANIFEST_SHA256" \
      --verifier-identity "$ISINGFOLD_QUALITY_RESOLUTION_VERIFIER_IDENTITY" \
      --expected-verifier-identity-sha256 \
      "$EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256" \
      --out "$output"
  fi
  processed=$((processed + 1))
  index=$((index + SLURM_ARRAY_TASK_COUNT))
done

if [[ "$processed" != "32" ]]; then
  echo "packed resolution worker did not cover exactly 32 of 64 shards" >&2
  exit 70
fi

#!/usr/bin/env bash
# Generate a pinned canary or production subset as a bounded Goose Slurm array.
# Never execute this file with bash or on a login node. Example submission:
#   # Production only after a same-shard Apollo/Goose parity attestation exists.
#   export ISINGFOLD_CORPUS_MODE=production
#   export ISINGFOLD_CORPUS_SHARD_FIRST=1
#   export ISINGFOLD_CORPUS_SHARD_LAST=63
#   export ISINGFOLD_CORPUS_SHARD_STEP=2
#   /opt/slurm/bin/sbatch --array=0-31%16 --export=ALL scripts/goose_isingfold_corpus.sbatch
#
# Freeze the plan before submission. Generate preflight v2 in the frozen runtime,
# record its file, record, identity-map, source, and provenance commitments outside
# SHARD_ROOT, and replay the full preflight before submission. Export every
# ISINGFOLD_CORPUS_* control named below. After this array succeeds, submit a
# separate dependent Slurm job that runs the same pinned container and the CLI
# `merge --plan ... --expected-plan-sha256 ... --shard-root ... --out ...` command.
# Do not merge on the Goose login node. Existing shard/claim/log/receipt paths fail.
# Stage a container only after `generation-provenance` is byte-identical to Apollo:
# Python 3.12.3 alone is insufficient; every recorded dependency version and source
# module hash must match. Export that shared digest as EXPECTED_PROVENANCE_SHA256.
# External dependencies must come from the hashed wheel lock, never editable,
# VCS, or direct-URL installs.
# Canary-only mode omits the parity attestation but requires FIRST=LAST, STEP=1,
# a one-task Slurm array, and SHARD_ROOT=CANARY_OUTPUT_ROOT/shards.
#SBATCH --job-name=isingfold-corpus
#SBATCH --array=0-31%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=isingfold-corpus-%A_%a.out
#SBATCH --error=isingfold-corpus-%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_PARTITION:-}" \
   || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_ARRAY_TASK_COUNT:-}" \
   || -z "${SLURM_ARRAY_TASK_MIN:-}" || -z "${SLURM_ARRAY_TASK_MAX:-}" \
   || -z "${SLURM_ARRAY_TASK_STEP:-}" || -z "${SLURMD_NODENAME:-}" ]]; then
  echo "goose_isingfold_corpus.sbatch requires a Slurm array allocation on a compute node" >&2
  exit 69
fi
: "${ISINGFOLD_CORPUS_SOURCE_ROOT:?set an absolute staged EmbedBench root}"
: "${ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256:?pin the runtime helper SHA-256}"
if [[ "$ISINGFOLD_CORPUS_SOURCE_ROOT" != /* \
   || ! -d "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
   || -L "$ISINGFOLD_CORPUS_SOURCE_ROOT" ]]; then
  echo "ISINGFOLD_CORPUS_SOURCE_ROOT must identify the staged EmbedBench directory" >&2
  exit 78
fi

# Slurm normally runs a private spool copy, so BASH_SOURCE[0] is not beside the
# staged helper. Anchor every source asset at the externally supplied source root,
# then prove that the submitted launcher and pinned staged launcher are identical.
script_dir="${ISINGFOLD_CORPUS_SOURCE_ROOT%/}/scripts"
runtime_helper="$script_dir/verify_isingfold_corpus_runtime.sh"
staged_launcher="$script_dir/goose_isingfold_corpus.sbatch"
if [[ ! -f "$runtime_helper" || -L "$runtime_helper" ]]; then
  echo "staged corpus runtime helper is missing or is a symbolic link" >&2
  exit 78
fi
if [[ ! "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "the external runtime helper SHA-256 commitment is invalid" >&2
  exit 78
fi
if command -v sha256sum >/dev/null 2>&1; then
  observed_runtime_helper_sha256=$(sha256sum -- "$runtime_helper" | awk '{print $1}')
else
  echo "sha256sum is required before loading the runtime helper" >&2
  exit 78
fi
if [[ "$observed_runtime_helper_sha256" != "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" ]]; then
  echo "runtime helper differs from its external SHA-256 commitment" >&2
  exit 78
fi
# shellcheck source=./scripts/verify_isingfold_corpus_runtime.sh
source "$runtime_helper"
corpus_require_canonical_directory \
  "$ISINGFOLD_CORPUS_SOURCE_ROOT" "ISINGFOLD_CORPUS_SOURCE_ROOT"
corpus_require_canonical_file "$runtime_helper" "staged corpus runtime helper"
corpus_require_canonical_file "$staged_launcher" "staged Goose corpus launcher"
submitted_launcher=${BASH_SOURCE[0]}
if [[ ! -f "$submitted_launcher" || -L "$submitted_launcher" ]]; then
  corpus_error "Slurm's submitted launcher copy must be a regular non-symbolic-link file" 78
fi
submitted_launcher_sha256=$(corpus_file_sha256 "$submitted_launcher")
staged_launcher_sha256=$(corpus_file_sha256 "$staged_launcher")
if [[ "$submitted_launcher_sha256" != "$staged_launcher_sha256" ]]; then
  corpus_error "submitted Slurm script bytes differ from the pinned staged launcher" 78
fi
if [[ ! -x /opt/slurm/bin/srun ]]; then
  corpus_error "Goose corpus generation requires /opt/slurm/bin/srun" 69
fi

: "${ISINGFOLD_CORPUS_EXPECTED_CLUSTER_NAME:?pin the Goose SLURM_CLUSTER_NAME}"
: "${ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256:?pin the staged source inventory SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256:?pin the shared Apollo/Goose generation provenance digest}"
: "${ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256:?pin runtime-lock.json SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256:?pin the runtime validator SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256:?pin the cross-site installation inventory digest}"
: "${ISINGFOLD_CORPUS_PLAN_ROOT:?set the absolute read-only directory containing PLAN}"
: "${ISINGFOLD_CORPUS_PLAN:?set the absolute prospective plan path}"
: "${ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256:?supply the independently recorded plan SHA-256}"
: "${ISINGFOLD_CORPUS_PREFLIGHT:?set the absolute prospective preflight path}"
: "${ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256:?pin the prospective preflight file SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST:?pin the preflight record digest}"
: "${ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST:?pin the prospective identity map digest}"
: "${ISINGFOLD_CORPUS_OUTPUT_BASE:?set an existing dedicated output base}"
: "${ISINGFOLD_CORPUS_SHARD_ROOT:?set the shared shard root under OUTPUT_BASE}"
: "${ISINGFOLD_CORPUS_SHARD_COUNT:?set the scientific plan shard count}"
: "${ISINGFOLD_CORPUS_SHARD_FIRST:?set the first inclusive scientific shard index}"
: "${ISINGFOLD_CORPUS_SHARD_LAST:?set the last inclusive scientific shard index}"
: "${ISINGFOLD_CORPUS_SHARD_STEP:?set the positive scientific shard stride}"
: "${ISINGFOLD_CORPUS_THREADS:?set threads per array task, at most CPUs per task}"
: "${ISINGFOLD_CORPUS_MODE:?set canary-only or production}"
: "${ISINGFOLD_CORPUS_APPTAINER:?set the absolute Apptainer executable}"
: "${ISINGFOLD_CORPUS_EXPECTED_APPTAINER_SHA256:?pin the Apptainer executable SHA-256}"
: "${ISINGFOLD_CORPUS_CONTAINER_IMAGE:?set the absolute immutable container image}"
: "${ISINGFOLD_CORPUS_EXPECTED_CONTAINER_SHA256:?pin the container image SHA-256}"
: "${ISINGFOLD_CORPUS_CONTAINER_PYTHON:?set the absolute Python path inside the image}"

runtime_lock="$ISINGFOLD_CORPUS_SOURCE_ROOT/runtime/isingfold-corpus/runtime-lock.json"
runtime_validator="$ISINGFOLD_CORPUS_SOURCE_ROOT/scripts/validate_isingfold_corpus_image.py"
corpus_require_file_commitment \
  "$runtime_lock" "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256" "runtime lock"
corpus_require_file_commitment \
  "$runtime_validator" "$ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256" "runtime validator"
corpus_require_sha256 \
  "$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256" \
  "ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256"

if [[ "${SLURM_CLUSTER_NAME:-}" != "$ISINGFOLD_CORPUS_EXPECTED_CLUSTER_NAME" ]]; then
  corpus_error "Slurm cluster identity differs from ISINGFOLD_CORPUS_EXPECTED_CLUSTER_NAME" 69
fi
corpus_require_canonical_directory \
  "$ISINGFOLD_CORPUS_PLAN_ROOT" "ISINGFOLD_CORPUS_PLAN_ROOT"
corpus_require_canonical_file "$ISINGFOLD_CORPUS_PLAN" "ISINGFOLD_CORPUS_PLAN"
corpus_require_file_commitment \
  "$ISINGFOLD_CORPUS_PREFLIGHT" \
  "$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256" \
  "prospective preflight"
case "$ISINGFOLD_CORPUS_PLAN/" in
  "$ISINGFOLD_CORPUS_PLAN_ROOT/"*) ;;
  *) corpus_error "ISINGFOLD_CORPUS_PLAN must be inside ISINGFOLD_CORPUS_PLAN_ROOT" 78 ;;
esac
case "$ISINGFOLD_CORPUS_PREFLIGHT/" in
  "$ISINGFOLD_CORPUS_PLAN_ROOT/"*) ;;
  *) corpus_error "ISINGFOLD_CORPUS_PREFLIGHT must be inside ISINGFOLD_CORPUS_PLAN_ROOT" 78 ;;
esac
corpus_require_disjoint_roots \
  "$ISINGFOLD_CORPUS_SOURCE_ROOT" "ISINGFOLD_CORPUS_SOURCE_ROOT" \
  "$ISINGFOLD_CORPUS_OUTPUT_BASE" "ISINGFOLD_CORPUS_OUTPUT_BASE"
corpus_require_disjoint_roots \
  "$ISINGFOLD_CORPUS_PLAN_ROOT" "ISINGFOLD_CORPUS_PLAN_ROOT" \
  "$ISINGFOLD_CORPUS_OUTPUT_BASE" "ISINGFOLD_CORPUS_OUTPUT_BASE"

for integer_value in \
  "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" \
  "$SLURM_ARRAY_TASK_MIN" "$SLURM_ARRAY_TASK_MAX" "$SLURM_ARRAY_TASK_STEP" \
  "${SLURM_CPUS_PER_TASK:-}" "$ISINGFOLD_CORPUS_SHARD_COUNT" \
  "$ISINGFOLD_CORPUS_SHARD_FIRST" "$ISINGFOLD_CORPUS_SHARD_LAST" \
  "$ISINGFOLD_CORPUS_SHARD_STEP" "$ISINGFOLD_CORPUS_THREADS"; do
  if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
    corpus_error "Slurm array controls, SHARD_COUNT, and THREADS must be decimal integers" 64
  fi
done

shard_count=$((10#$ISINGFOLD_CORPUS_SHARD_COUNT))
first=$((10#$ISINGFOLD_CORPUS_SHARD_FIRST))
last=$((10#$ISINGFOLD_CORPUS_SHARD_LAST))
step=$((10#$ISINGFOLD_CORPUS_SHARD_STEP))
cpus_per_task=$((10#$SLURM_CPUS_PER_TASK))
threads=$((10#$ISINGFOLD_CORPUS_THREADS))
index=$(corpus_map_shard_subset \
  "$ISINGFOLD_CORPUS_SHARD_COUNT" \
  "$ISINGFOLD_CORPUS_SHARD_FIRST" \
  "$ISINGFOLD_CORPUS_SHARD_LAST" \
  "$ISINGFOLD_CORPUS_SHARD_STEP" \
  "$SLURM_ARRAY_TASK_ID" \
  "$SLURM_ARRAY_TASK_COUNT" \
  "$SLURM_ARRAY_TASK_MIN" \
  "$SLURM_ARRAY_TASK_MAX" \
  "$SLURM_ARRAY_TASK_STEP")
if (( threads == 0 || threads > cpus_per_task )); then
  corpus_error "THREADS must lie in [1, SLURM_CPUS_PER_TASK]" 64
fi
if (( threads != 1 )); then
  corpus_error "the pinned corpus runtime requires THREADS=1" 64
fi

case "$ISINGFOLD_CORPUS_MODE" in
  canary-only)
    : "${ISINGFOLD_CORPUS_CANARY_OUTPUT_ROOT:?set the dedicated Goose canary root}"
    corpus_validate_launch_mode \
      "$ISINGFOLD_CORPUS_MODE" "$first" "$last" "$step" \
      "$SLURM_ARRAY_TASK_COUNT" "$ISINGFOLD_CORPUS_SHARD_ROOT" \
      "$ISINGFOLD_CORPUS_CANARY_OUTPUT_ROOT"
    if (( first != last || step != 1 || 10#$SLURM_ARRAY_TASK_COUNT != 1 )); then
      corpus_error "canary-only mode requires exactly one scientific and Slurm-array shard" 64
    fi
    if [[ "$ISINGFOLD_CORPUS_SHARD_ROOT" != "$ISINGFOLD_CORPUS_CANARY_OUTPUT_ROOT/shards" ]]; then
      corpus_error "canary-only SHARD_ROOT must be CANARY_OUTPUT_ROOT/shards" 78
    fi
    parity_attestation_sha256=not-applicable-canary-only
    ;;
  production)
    corpus_validate_launch_mode \
      "$ISINGFOLD_CORPUS_MODE" "$first" "$last" "$step" \
      "$SLURM_ARRAY_TASK_COUNT" "$ISINGFOLD_CORPUS_SHARD_ROOT" -
    : "${ISINGFOLD_CORPUS_PARITY_ROOT:?set the read-only parity-attestation root}"
    : "${ISINGFOLD_CORPUS_PARITY_ATTESTATION:?set the canary parity attestation path}"
    : "${ISINGFOLD_CORPUS_EXPECTED_PARITY_ATTESTATION_SHA256:?pin the parity attestation SHA-256}"
    corpus_require_canonical_directory \
      "$ISINGFOLD_CORPUS_PARITY_ROOT" "ISINGFOLD_CORPUS_PARITY_ROOT"
    corpus_require_file_commitment \
      "$ISINGFOLD_CORPUS_PARITY_ATTESTATION" \
      "$ISINGFOLD_CORPUS_EXPECTED_PARITY_ATTESTATION_SHA256" \
      "canary parity attestation"
    case "$ISINGFOLD_CORPUS_PARITY_ATTESTATION/" in
      "$ISINGFOLD_CORPUS_PARITY_ROOT/"*) ;;
      *) corpus_error "parity attestation must be inside PARITY_ROOT" 78 ;;
    esac
    corpus_require_disjoint_roots \
      "$ISINGFOLD_CORPUS_PARITY_ROOT" "ISINGFOLD_CORPUS_PARITY_ROOT" \
      "$ISINGFOLD_CORPUS_OUTPUT_BASE" "ISINGFOLD_CORPUS_OUTPUT_BASE"
    parity_attestation_sha256=$ISINGFOLD_CORPUS_EXPECTED_PARITY_ATTESTATION_SHA256
    ;;
  *) corpus_validate_launch_mode "$ISINGFOLD_CORPUS_MODE" 0 0 1 1 /invalid - ;;
esac

corpus_require_sha256 \
  "$ISINGFOLD_CORPUS_EXPECTED_APPTAINER_SHA256" \
  "ISINGFOLD_CORPUS_EXPECTED_APPTAINER_SHA256"
corpus_require_sha256 \
  "$ISINGFOLD_CORPUS_EXPECTED_CONTAINER_SHA256" \
  "ISINGFOLD_CORPUS_EXPECTED_CONTAINER_SHA256"
corpus_require_canonical_file "$ISINGFOLD_CORPUS_APPTAINER" "ISINGFOLD_CORPUS_APPTAINER"
if [[ ! -x "$ISINGFOLD_CORPUS_APPTAINER" ]]; then
  corpus_error "ISINGFOLD_CORPUS_APPTAINER must be executable" 78
fi
corpus_require_canonical_file \
  "$ISINGFOLD_CORPUS_CONTAINER_IMAGE" "ISINGFOLD_CORPUS_CONTAINER_IMAGE"
if [[ "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" != /* ]]; then
  corpus_error "ISINGFOLD_CORPUS_CONTAINER_PYTHON must be absolute" 78
fi
for bind_path in \
  "$ISINGFOLD_CORPUS_SOURCE_ROOT" "$ISINGFOLD_CORPUS_PLAN_ROOT" \
  "$ISINGFOLD_CORPUS_OUTPUT_BASE" "$ISINGFOLD_CORPUS_SHARD_ROOT"; do
  if [[ "$bind_path" == *:* || "$bind_path" == *,* ]]; then
    corpus_error "Apptainer bind paths cannot contain colon or comma" 78
  fi
done
if [[ "$ISINGFOLD_CORPUS_MODE" == "production" \
   && ( "$ISINGFOLD_CORPUS_PARITY_ROOT" == *:* \
     || "$ISINGFOLD_CORPUS_PARITY_ROOT" == *,* ) ]]; then
  corpus_error "Apptainer parity root cannot contain colon or comma" 78
fi
for shadow_root in \
  "$ISINGFOLD_CORPUS_SOURCE_ROOT" "$ISINGFOLD_CORPUS_PLAN_ROOT" \
  "$ISINGFOLD_CORPUS_OUTPUT_BASE"; do
  case "$ISINGFOLD_CORPUS_CONTAINER_PYTHON/" in
    "$shadow_root/"*)
      corpus_error "a host bind root would shadow ISINGFOLD_CORPUS_CONTAINER_PYTHON" 78
      ;;
  esac
done

observed_apptainer_sha256=$(corpus_file_sha256 "$ISINGFOLD_CORPUS_APPTAINER")
observed_container_sha256=$(corpus_file_sha256 "$ISINGFOLD_CORPUS_CONTAINER_IMAGE")
if [[ "$observed_apptainer_sha256" != "$ISINGFOLD_CORPUS_EXPECTED_APPTAINER_SHA256" ]]; then
  corpus_error "Apptainer executable differs from its external commitment" 78
fi
if [[ "$observed_container_sha256" != "$ISINGFOLD_CORPUS_EXPECTED_CONTAINER_SHA256" ]]; then
  corpus_error "container image differs from its external commitment" 78
fi
observed_plan_sha256=$(corpus_file_sha256 "$ISINGFOLD_CORPUS_PLAN")
if [[ "$observed_plan_sha256" != "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" ]]; then
  corpus_error "plan bytes differ from the external plan commitment" 78
fi

if [[ "$ISINGFOLD_CORPUS_MODE" == "canary-only" ]]; then
  corpus_prepare_output_root \
    "$ISINGFOLD_CORPUS_OUTPUT_BASE" "$ISINGFOLD_CORPUS_CANARY_OUTPUT_ROOT"
  corpus_prepare_output_root \
    "$ISINGFOLD_CORPUS_CANARY_OUTPUT_ROOT" "$ISINGFOLD_CORPUS_SHARD_ROOT"
else
  corpus_prepare_output_root \
    "$ISINGFOLD_CORPUS_OUTPUT_BASE" "$ISINGFOLD_CORPUS_SHARD_ROOT"
fi

claims_root="$ISINGFOLD_CORPUS_SHARD_ROOT/.claims"
logs_root="$ISINGFOLD_CORPUS_SHARD_ROOT/logs"
receipts_root="$ISINGFOLD_CORPUS_SHARD_ROOT/receipts"
corpus_prepare_output_root "$ISINGFOLD_CORPUS_SHARD_ROOT" "$claims_root"
corpus_prepare_output_root "$ISINGFOLD_CORPUS_SHARD_ROOT" "$logs_root"
corpus_prepare_output_root "$ISINGFOLD_CORPUS_SHARD_ROOT" "$receipts_root"
printf -v shard_name 'shard-%04d-of-%04d' "$index" "$shard_count"
output="$ISINGFOLD_CORPUS_SHARD_ROOT/$shard_name"
claim="$claims_root/$shard_name"
log="$logs_root/$shard_name.log"
receipt="$receipts_root/$shard_name.json"
for target in "$output" "$claim" "$log" "$receipt"; do
  if [[ -e "$target" || -L "$target" ]]; then
    corpus_error "refusing to overwrite existing path: $target" 73
  fi
done
if ! mkdir -- "$claim"; then
  corpus_error "another task already owns claim $claim" 73
fi
if ! (set -o noclobber; : > "$log"); then
  corpus_error "refusing to overwrite log $log" 73
fi
receipt_partial="$claim/receipt.partial.json"

declare -a binds=(
  "$ISINGFOLD_CORPUS_SOURCE_ROOT:$ISINGFOLD_CORPUS_SOURCE_ROOT:ro"
  "$ISINGFOLD_CORPUS_OUTPUT_BASE:$ISINGFOLD_CORPUS_OUTPUT_BASE:ro"
  "$ISINGFOLD_CORPUS_SHARD_ROOT:$ISINGFOLD_CORPUS_SHARD_ROOT:rw"
)
if [[ "$ISINGFOLD_CORPUS_PLAN_ROOT" != "$ISINGFOLD_CORPUS_SOURCE_ROOT" ]]; then
  binds+=("$ISINGFOLD_CORPUS_PLAN_ROOT:$ISINGFOLD_CORPUS_PLAN_ROOT:ro")
fi
if [[ "$ISINGFOLD_CORPUS_MODE" == "production" \
   && "$ISINGFOLD_CORPUS_PARITY_ROOT" != "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
   && "$ISINGFOLD_CORPUS_PARITY_ROOT" != "$ISINGFOLD_CORPUS_PLAN_ROOT" ]]; then
  binds+=("$ISINGFOLD_CORPUS_PARITY_ROOT:$ISINGFOLD_CORPUS_PARITY_ROOT:ro")
fi
bind_spec=$(IFS=,; echo "${binds[*]}")
declare -a container_exec=(
  "$ISINGFOLD_CORPUS_APPTAINER" exec
  --cleanenv
  --containall
  --no-home
  --no-mount "bind-paths,hostfs"
  --bind "$bind_spec"
  --pwd "$ISINGFOLD_CORPUS_SOURCE_ROOT"
  --env "PYTHONPATH=$ISINGFOLD_CORPUS_SOURCE_ROOT/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONHASHSEED=0"
  --env "OMP_NUM_THREADS=$threads"
  --env "OPENBLAS_NUM_THREADS=$threads"
  --env "MKL_NUM_THREADS=$threads"
  --env "NUMEXPR_NUM_THREADS=$threads"
  "$ISINGFOLD_CORPUS_CONTAINER_IMAGE"
)

# Do not let host Python or loader state leak through a supposedly clean image.
unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE
while IFS='=' read -r variable_name _; do
  case "$variable_name" in
    APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*)
      unset "$variable_name"
      ;;
  esac
done < <(env)

{
  echo "event=start"
  echo "utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "site=goose"
  echo "mode=$ISINGFOLD_CORPUS_MODE"
  echo "cluster=$SLURM_CLUSTER_NAME"
  echo "node=$SLURMD_NODENAME"
  echo "job_id=$SLURM_JOB_ID"
  echo "array_task_id=$SLURM_ARRAY_TASK_ID"
  echo "subset_first=$first"
  echo "subset_last=$last"
  echo "subset_step=$step"
  echo "shard=$shard_name"
  echo "source_root=$ISINGFOLD_CORPUS_SOURCE_ROOT"
  echo "plan=$ISINGFOLD_CORPUS_PLAN"
  echo "preflight=$ISINGFOLD_CORPUS_PREFLIGHT"
  echo "container_image=$ISINGFOLD_CORPUS_CONTAINER_IMAGE"
  echo "container_python=$ISINGFOLD_CORPUS_CONTAINER_PYTHON"
  echo "threads=$threads"
  echo "launcher_sha256=$staged_launcher_sha256"
  echo "plan_sha256=$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256"
  echo "preflight_sha256=$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256"
  echo "preflight_record_digest=$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST"
  echo "prospective_identity_map_digest=$ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST"
  echo "source_sha256=$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256"
  echo "generation_provenance_sha256=$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256"
  echo "installation_sha256=$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256"
  echo "runtime_lock_sha256=$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256"
  echo "runtime_validator_sha256=$ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256"
  echo "container_sha256=$ISINGFOLD_CORPUS_EXPECTED_CONTAINER_SHA256"
  echo "apptainer_sha256=$ISINGFOLD_CORPUS_EXPECTED_APPTAINER_SHA256"
  echo "runtime_helper_sha256=$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256"
  echo "parity_attestation_sha256=$parity_attestation_sha256"
} >> "$log"

if ! /opt/slurm/bin/srun --ntasks=1 --cpus-per-task="$threads" \
  "${container_exec[@]}" bash "$script_dir/verify_isingfold_corpus_runtime.sh" \
    validate-source-plan \
    "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" \
    "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
    "$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" \
    "$ISINGFOLD_CORPUS_PLAN" \
    "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" \
    "$shard_count" >> "$log" 2>&1; then
  echo "event=runtime-validation-failure" >> "$log"
  printf '%s\n' "runtime-validation-failure" > "$claim/FAILED"
  exit 78
fi
if ! /opt/slurm/bin/srun --ntasks=1 --cpus-per-task="$threads" \
  "${container_exec[@]}" "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" \
    "$runtime_validator" \
    --runtime-lock "$runtime_lock" \
    --source-root "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
    --expected-installation-sha256 "$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256" \
    --expected-provenance-sha256 "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" \
    >> "$log" 2>&1; then
  echo "event=installed-runtime-validation-failure" >> "$log"
  printf '%s\n' "installed-runtime-validation-failure" > "$claim/FAILED"
  exit 78
fi
if ! /opt/slurm/bin/srun --ntasks=1 --cpus-per-task="$threads" \
  "${container_exec[@]}" bash "$script_dir/verify_isingfold_corpus_runtime.sh" \
    validate-preflight \
    "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" \
    "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
    "$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" \
    "$ISINGFOLD_CORPUS_PLAN" \
    "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" \
    "$ISINGFOLD_CORPUS_PREFLIGHT" \
    "$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256" \
    "$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST" \
    "$ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST" \
    "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" \
    10 >> "$log" 2>&1; then
  echo "event=prospective-preflight-replay-failure" >> "$log"
  printf '%s\n' "prospective-preflight-replay-failure" > "$claim/FAILED"
  exit 78
fi
if [[ "$ISINGFOLD_CORPUS_MODE" == "production" ]]; then
  if ! /opt/slurm/bin/srun --ntasks=1 --cpus-per-task="$threads" \
    "${container_exec[@]}" "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" \
      -m embedbench.isingfold_cross_site_canary verify \
      --attestation "$ISINGFOLD_CORPUS_PARITY_ATTESTATION" \
      --expected-attestation-sha256 \
        "$ISINGFOLD_CORPUS_EXPECTED_PARITY_ATTESTATION_SHA256" \
      --plan "$ISINGFOLD_CORPUS_PLAN" \
      --plan-sha256 "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" \
      --source-sha256 "$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" \
      --generation-provenance-sha256 \
        "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" \
      --installation-sha256 "$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256" \
      >> "$log" 2>&1; then
    echo "event=canary-parity-validation-failure" >> "$log"
    printf '%s\n' "canary-parity-validation-failure" > "$claim/FAILED"
    exit 78
  fi
fi
if ! /opt/slurm/bin/srun --ntasks=1 --cpus-per-task="$threads" \
  "${container_exec[@]}" bash "$script_dir/verify_isingfold_corpus_runtime.sh" \
    validate-provenance \
    "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" \
    "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
    "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" "$threads" \
    >> "$log" 2>&1; then
  echo "event=provenance-validation-failure" >> "$log"
  printf '%s\n' "provenance-validation-failure" > "$claim/FAILED"
  exit 78
fi
if ! /opt/slurm/bin/srun --ntasks=1 --cpus-per-task="$threads" \
  "${container_exec[@]}" bash "$script_dir/verify_isingfold_corpus_runtime.sh" \
    validate-externals \
    "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
    "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" \
    >> "$log" 2>&1; then
  echo "event=external-install-validation-failure" >> "$log"
  printf '%s\n' "external-install-validation-failure" > "$claim/FAILED"
  exit 78
fi
if ! /opt/slurm/bin/srun --ntasks=1 --cpus-per-task="$threads" \
  "${container_exec[@]}" "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" -m pip check \
  >> "$log" 2>&1; then
  echo "event=environment-validation-failure" >> "$log"
  printf '%s\n' "environment-validation-failure" > "$claim/FAILED"
  exit 78
fi

if /opt/slurm/bin/srun --ntasks=1 --cpus-per-task="$threads" \
  "${container_exec[@]}" "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" \
    -m embedbench.isingfold_corpus_cli generate-shard \
    --plan "$ISINGFOLD_CORPUS_PLAN" \
    --expected-plan-sha256 "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" \
    --expected-source-sha256 "$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" \
    --preflight "$ISINGFOLD_CORPUS_PREFLIGHT" \
    --expected-preflight-sha256 "$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256" \
    --expected-preflight-record-digest \
      "$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST" \
    --expected-identity-map-digest "$ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST" \
    --expected-generation-provenance-digest \
      "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" \
    --index "$index" \
    --out "$output" \
    > "$receipt_partial" 2>> "$log"; then
  generation_status=0
else
  generation_status=$?
fi
if (( generation_status != 0 )); then
  {
    echo "event=failure"
    echo "utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "exit_status=$generation_status"
  } >> "$log"
  printf '%s\n' "$generation_status" > "$claim/FAILED"
  exit "$generation_status"
fi

if ! /opt/slurm/bin/srun --ntasks=1 --cpus-per-task=1 \
  "${container_exec[@]}" bash "$script_dir/verify_isingfold_corpus_runtime.sh" \
    validate-receipt \
    "$ISINGFOLD_CORPUS_CONTAINER_PYTHON" \
    "$receipt_partial" "$index" "$shard_count" "$output" \
    >> "$log" 2>&1; then
  echo "event=invalid-receipt" >> "$log"
  printf '%s\n' "invalid-receipt" > "$claim/FAILED"
  exit 78
fi
if ! ln -- "$receipt_partial" "$receipt"; then
  echo "event=receipt-collision" >> "$log"
  printf '%s\n' "receipt-collision" > "$claim/FAILED"
  exit 73
fi
rm -- "$receipt_partial"
{
  echo "event=success"
  echo "utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} >> "$log"
printf '%s\n' "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" > "$claim/SUCCESS"
echo "completed $shard_name on Goose under Slurm job $SLURM_JOB_ID"

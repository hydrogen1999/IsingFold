#!/usr/bin/env bash
# Build the corpus-generation SIF on a Goose compute node.
# Submit this file with /opt/slurm/bin/sbatch.  Never execute it with bash.
# The job emits IMAGE, IMAGE.sha256, and IMAGE.validation.json without replacing
# any existing artifact.
#SBATCH --job-name=isingfold-runtime-build
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=isingfold-runtime-build-%j.out
#SBATCH --error=isingfold-runtime-build-%j.err
#SBATCH --open-mode=append

set -euo pipefail
umask 077

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_PARTITION:-}" \
   || -z "${SLURMD_NODENAME:-}" ]]; then
  echo "goose_build_isingfold_corpus_runtime.sbatch requires a Goose Slurm allocation" >&2
  exit 69
fi
if [[ ! -x /opt/slurm/bin/srun ]]; then
  echo "runtime build requires /opt/slurm/bin/srun" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS_EXPECTED_CLUSTER_NAME:?pin the Goose SLURM_CLUSTER_NAME}"
: "${ISINGFOLD_CORPUS_SOURCE_ROOT:?set the absolute staged EmbedBench root}"
: "${ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256:?pin the staged source inventory SHA-256}"
: "${ISINGFOLD_CORPUS_RUNTIME_DEFINITION:?set the absolute Apptainer definition path}"
: "${ISINGFOLD_CORPUS_EXPECTED_DEFINITION_SHA256:?pin the definition SHA-256}"
: "${ISINGFOLD_CORPUS_REQUIREMENTS_LOCK:?set the absolute pip lock path}"
: "${ISINGFOLD_CORPUS_EXPECTED_REQUIREMENTS_SHA256:?pin the pip lock SHA-256}"
: "${ISINGFOLD_CORPUS_RUNTIME_LOCK:?set the absolute runtime-lock.json path}"
: "${ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256:?pin runtime-lock.json SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256:?pin the image validator SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_BUILD_LAUNCHER_SHA256:?pin this launcher SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256:?pin the shared runtime helper SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256:?pin the Apollo provenance digest}"
: "${ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256:?pin the cross-site installation inventory digest}"
: "${ISINGFOLD_CORPUS_APPTAINER:?set the absolute Apptainer executable}"
: "${ISINGFOLD_CORPUS_EXPECTED_APPTAINER_SHA256:?pin the Apptainer executable SHA-256}"
: "${ISINGFOLD_CORPUS_IMAGE_OUTPUT_ROOT:?set an existing dedicated output directory}"
: "${ISINGFOLD_CORPUS_CONTAINER_IMAGE:?set the new absolute .sif output path}"

submitted_launcher=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")
if [[ "$ISINGFOLD_CORPUS_SOURCE_ROOT" != /* \
   || ! -d "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
   || -L "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
   || "$(cd -- "$ISINGFOLD_CORPUS_SOURCE_ROOT" && pwd -P)" != "$ISINGFOLD_CORPUS_SOURCE_ROOT" ]]; then
  echo "ISINGFOLD_CORPUS_SOURCE_ROOT must be a canonical absolute directory" >&2
  exit 78
fi
script_dir="$ISINGFOLD_CORPUS_SOURCE_ROOT/scripts"
runtime_helper="$script_dir/verify_isingfold_corpus_runtime.sh"
if [[ ! "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" =~ ^[0-9a-f]{64}$ \
   || ! -f "$runtime_helper" || -L "$runtime_helper" ]]; then
  echo "the shared runtime helper or its external SHA-256 commitment is invalid" >&2
  exit 78
fi
if command -v sha256sum >/dev/null 2>&1; then
  observed_runtime_helper_sha256=$(sha256sum -- "$runtime_helper" | awk '{print $1}')
else
  echo "sha256sum is required before loading the shared runtime helper" >&2
  exit 78
fi
if [[ "$observed_runtime_helper_sha256" != "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" ]]; then
  echo "shared runtime helper differs from its external commitment" >&2
  exit 78
fi
# shellcheck source=./scripts/verify_isingfold_corpus_runtime.sh
source "$runtime_helper"

if [[ "${SLURM_CLUSTER_NAME:-}" != "$ISINGFOLD_CORPUS_EXPECTED_CLUSTER_NAME" ]]; then
  corpus_error "Slurm cluster identity differs from its commitment" 69
fi
corpus_require_canonical_directory "$ISINGFOLD_CORPUS_SOURCE_ROOT" "source root"
corpus_require_canonical_directory "$ISINGFOLD_CORPUS_IMAGE_OUTPUT_ROOT" "image output root"
corpus_require_canonical_file "$ISINGFOLD_CORPUS_RUNTIME_DEFINITION" "runtime definition"
corpus_require_canonical_file "$ISINGFOLD_CORPUS_REQUIREMENTS_LOCK" "requirements lock"
corpus_require_canonical_file "$ISINGFOLD_CORPUS_RUNTIME_LOCK" "runtime lock"
corpus_require_canonical_file "$ISINGFOLD_CORPUS_APPTAINER" "Apptainer executable"
if [[ ! -x "$ISINGFOLD_CORPUS_APPTAINER" ]]; then
  corpus_error "ISINGFOLD_CORPUS_APPTAINER must be executable" 78
fi

expected_definition="$ISINGFOLD_CORPUS_SOURCE_ROOT/runtime/isingfold-corpus/isingfold-corpus.def"
expected_requirements="$ISINGFOLD_CORPUS_SOURCE_ROOT/runtime/isingfold-corpus/requirements-linux-x86_64-py312.lock"
expected_runtime_lock="$ISINGFOLD_CORPUS_SOURCE_ROOT/runtime/isingfold-corpus/runtime-lock.json"
expected_validator="$ISINGFOLD_CORPUS_SOURCE_ROOT/scripts/validate_isingfold_corpus_image.py"
expected_build_launcher="$ISINGFOLD_CORPUS_SOURCE_ROOT/scripts/goose_build_isingfold_corpus_runtime.sbatch"
if [[ "$ISINGFOLD_CORPUS_RUNTIME_DEFINITION" != "$expected_definition" \
   || "$ISINGFOLD_CORPUS_REQUIREMENTS_LOCK" != "$expected_requirements" \
   || "$ISINGFOLD_CORPUS_RUNTIME_LOCK" != "$expected_runtime_lock" ]]; then
  corpus_error "runtime assets must be the canonical files inside the staged source root" 78
fi
corpus_require_canonical_file "$expected_validator" "image validator"

for digest_name in \
  ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_DEFINITION_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_REQUIREMENTS_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_BUILD_LAUNCHER_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256 \
  ISINGFOLD_CORPUS_EXPECTED_APPTAINER_SHA256; do
  corpus_require_sha256 "${!digest_name}" "$digest_name"
done

check_file_digest() {
  local path=$1
  local expected=$2
  local name=$3
  local observed
  observed=$(corpus_file_sha256 "$path")
  if [[ "$observed" != "$expected" ]]; then
    corpus_error "$name bytes differ from their external commitment" 78
  fi
}
check_file_digest \
  "$ISINGFOLD_CORPUS_RUNTIME_DEFINITION" \
  "$ISINGFOLD_CORPUS_EXPECTED_DEFINITION_SHA256" "runtime definition"
check_file_digest \
  "$ISINGFOLD_CORPUS_REQUIREMENTS_LOCK" \
  "$ISINGFOLD_CORPUS_EXPECTED_REQUIREMENTS_SHA256" "requirements lock"
check_file_digest \
  "$ISINGFOLD_CORPUS_RUNTIME_LOCK" \
  "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256" "runtime lock"
check_file_digest \
  "$expected_validator" \
  "$ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256" "image validator"
check_file_digest \
  "$expected_build_launcher" \
  "$ISINGFOLD_CORPUS_EXPECTED_BUILD_LAUNCHER_SHA256" "build launcher"
check_file_digest \
  "$submitted_launcher" \
  "$ISINGFOLD_CORPUS_EXPECTED_BUILD_LAUNCHER_SHA256" "submitted Slurm launcher"
check_file_digest \
  "$runtime_helper" \
  "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" "shared runtime helper"
check_file_digest \
  "$ISINGFOLD_CORPUS_APPTAINER" \
  "$ISINGFOLD_CORPUS_EXPECTED_APPTAINER_SHA256" "Apptainer executable"

observed_source_sha256=$(
  "$script_dir/verify_isingfold_corpus_runtime.sh" source-sha256 \
    "$ISINGFOLD_CORPUS_SOURCE_ROOT" /usr/bin/python3
)
if [[ "$observed_source_sha256" != "$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" ]]; then
  corpus_error "staged source differs from its external commitment" 78
fi

if [[ "$ISINGFOLD_CORPUS_CONTAINER_IMAGE" != /* \
   || "${ISINGFOLD_CORPUS_CONTAINER_IMAGE##*.}" != "sif" \
   || "$(dirname -- "$ISINGFOLD_CORPUS_CONTAINER_IMAGE")" != "$ISINGFOLD_CORPUS_IMAGE_OUTPUT_ROOT" ]]; then
  corpus_error "container image must be a direct .sif child of IMAGE_OUTPUT_ROOT" 78
fi
validation_receipt="$ISINGFOLD_CORPUS_CONTAINER_IMAGE.validation.json"
image_digest_receipt="$ISINGFOLD_CORPUS_CONTAINER_IMAGE.sha256"
image_basename=${ISINGFOLD_CORPUS_CONTAINER_IMAGE##*/}
build_claim="$ISINGFOLD_CORPUS_IMAGE_OUTPUT_ROOT/.${image_basename}.build-claim"
partial_image="$ISINGFOLD_CORPUS_IMAGE_OUTPUT_ROOT/.${ISINGFOLD_CORPUS_CONTAINER_IMAGE##*/}.partial.$SLURM_JOB_ID"
partial_validation="$partial_image.validation.json"
partial_digest_receipt="$build_claim/image.sha256.partial"
for target in \
  "$ISINGFOLD_CORPUS_CONTAINER_IMAGE" "$validation_receipt" "$image_digest_receipt" \
  "$build_claim" "$partial_image" "$partial_validation"; do
  if [[ -e "$target" || -L "$target" ]]; then
    corpus_error "refusing to overwrite existing runtime artifact: $target" 73
  fi
done
if ! mkdir -- "$build_claim"; then
  corpus_error "another build already owns the deterministic runtime claim" 73
fi
{
  echo "job_id=$SLURM_JOB_ID"
  echo "node=$SLURMD_NODENAME"
  echo "source_sha256=$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256"
  echo "definition_sha256=$ISINGFOLD_CORPUS_EXPECTED_DEFINITION_SHA256"
  echo "installation_sha256=$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256"
} > "$build_claim/STARTED"

unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE
while IFS='=' read -r variable_name _; do
  case "$variable_name" in
    APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*) unset "$variable_name" ;;
  esac
done < <(env)

cd -- "$ISINGFOLD_CORPUS_SOURCE_ROOT"
/opt/slurm/bin/srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  "$ISINGFOLD_CORPUS_APPTAINER" build --fakeroot \
  "$partial_image" "$ISINGFOLD_CORPUS_RUNTIME_DEFINITION"

declare -a runtime_exec=(
  "$ISINGFOLD_CORPUS_APPTAINER" exec
  --cleanenv
  --containall
  --no-home
  --no-mount "bind-paths,hostfs"
  --bind "$ISINGFOLD_CORPUS_SOURCE_ROOT:$ISINGFOLD_CORPUS_SOURCE_ROOT:ro"
  --pwd "$ISINGFOLD_CORPUS_SOURCE_ROOT"
  --env "PYTHONPATH=$ISINGFOLD_CORPUS_SOURCE_ROOT/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONHASHSEED=0"
  --env "MKL_NUM_THREADS=1"
  --env "NUMEXPR_NUM_THREADS=1"
  --env "OMP_NUM_THREADS=1"
  --env "OPENBLAS_NUM_THREADS=1"
  "$partial_image"
)
/opt/slurm/bin/srun --ntasks=1 --cpus-per-task=1 \
  "${runtime_exec[@]}" \
  /opt/isingfold/venv/bin/python \
  "$ISINGFOLD_CORPUS_SOURCE_ROOT/scripts/validate_isingfold_corpus_image.py" \
  --runtime-lock "$ISINGFOLD_CORPUS_RUNTIME_LOCK" \
  --source-root "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
  --expected-installation-sha256 "$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256" \
  --expected-provenance-sha256 "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" \
  > "$partial_validation"
/opt/slurm/bin/srun --ntasks=1 --cpus-per-task=1 \
  "$ISINGFOLD_CORPUS_APPTAINER" inspect "$partial_image" >/dev/null

image_sha256=$(corpus_file_sha256 "$partial_image")
if ! (set -o noclobber; printf '%s  %s\n' \
  "$image_sha256" "$ISINGFOLD_CORPUS_CONTAINER_IMAGE" > "$partial_digest_receipt"); then
  corpus_error "refusing to replace the staged image digest receipt" 73
fi
if ! ln -- "$partial_image" "$ISINGFOLD_CORPUS_CONTAINER_IMAGE"; then
  corpus_error "refusing to replace an existing container image" 73
fi
if ! ln -- "$partial_validation" "$validation_receipt"; then
  corpus_error "refusing to replace an existing image validation receipt" 73
fi
# The digest receipt is the commit marker and is deliberately published last.
if ! ln -- "$partial_digest_receipt" "$image_digest_receipt"; then
  corpus_error "refusing to replace an existing image digest receipt" 73
fi
rm -- "$partial_image" "$partial_validation" "$partial_digest_receipt"
{
  echo "image_sha256=$image_sha256"
  echo "job_id=$SLURM_JOB_ID"
} > "$build_claim/SUCCESS"
echo "built $ISINGFOLD_CORPUS_CONTAINER_IMAGE"
echo "container_sha256=$image_sha256"

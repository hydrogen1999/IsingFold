#!/usr/bin/env bash
# Build the one canonical native wheel, wheelhouse, and training SIF on Goose.
#SBATCH --job-name=isingfold-runtime-build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=isingfold-runtime-build-%j.out
#SBATCH --error=isingfold-runtime-build-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_PARTITION:-}" \
   || -z "${SLURMD_NODENAME:-}" ]]; then
  echo "Goose runtime construction requires an active Slurm compute-node allocation" >&2
  exit 69
fi

srun_bin=/opt/slurm/bin/srun
apptainer_bin=/usr/bin/apptainer
if [[ ! -x "$srun_bin" || ! -x "$apptainer_bin" || ! -x /usr/bin/python3 ]]; then
  echo "Goose site binaries are absent at the registered absolute paths" >&2
  exit 69
fi
: "${ISINGFOLD_RUNTIME_EXPECTED_SRUN_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_SRUN_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_APPTAINER_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_APPTAINER_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_HOST_PYTHON_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_HOST_PYTHON_SHA256}"
: "${ISINGFOLD_RUNTIME_SOURCE_ROOT:?set ISINGFOLD_RUNTIME_SOURCE_ROOT}"
: "${ISINGFOLD_RUNTIME_BUILD_OUTPUT_ROOT:?set ISINGFOLD_RUNTIME_BUILD_OUTPUT_ROOT}"
: "${ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_WHEEL_BUILDER_DEF_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_WHEEL_BUILDER_DEF_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_TRAINING_DEF_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_TRAINING_DEF_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_LOCK_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_LOCK_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_RUNTIME_REQUIREMENTS_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_RUNTIME_REQUIREMENTS_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_BUILD_REQUIREMENTS_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_BUILD_REQUIREMENTS_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_VERIFIER_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_VERIFIER_SHA256}"

if [[ "$ISINGFOLD_RUNTIME_SOURCE_ROOT" != /* \
   || -L "$ISINGFOLD_RUNTIME_SOURCE_ROOT" \
   || ! -d "$ISINGFOLD_RUNTIME_SOURCE_ROOT" ]]; then
  echo "ISINGFOLD_RUNTIME_SOURCE_ROOT must be an absolute physical directory" >&2
  exit 78
fi
source_root=$(cd -- "$ISINGFOLD_RUNTIME_SOURCE_ROOT" && pwd -P)
if [[ "$source_root" != "$ISINGFOLD_RUNTIME_SOURCE_ROOT" ]]; then
  echo "ISINGFOLD_RUNTIME_SOURCE_ROOT must use its canonical physical path" >&2
  exit 78
fi
script_dir="$source_root/scripts"
runtime_root="$source_root/runtime/isingfold-training"
wheel_builder_def="$runtime_root/isingfold-wheel-builder.def"
training_def="$runtime_root/isingfold-training.def"
runtime_lock="$runtime_root/runtime-lock.json"
runtime_requirements="$runtime_root/requirements-linux-x86_64-py312-cu128.lock"
build_requirements="$runtime_root/requirements-build-linux-x86_64-py312.lock"
verifier="$script_dir/verify_isingfold_runtime.py"

unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
while IFS='=' read -r variable_name _; do
  case "$variable_name" in
    APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*)
      unset "$variable_name"
      ;;
  esac
done < <(env)

sha256_file() {
  /usr/bin/sha256sum -- "$1" | /usr/bin/awk '{print $1}'
}
require_digest() {
  local label=$1
  local path=$2
  local expected=$3
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$label commitment must be one lowercase SHA-256 digest" >&2
    exit 78
  fi
  local observed
  observed=$(sha256_file "$path")
  if [[ "$observed" != "$expected" ]]; then
    echo "$label differs from its out-of-band commitment" >&2
    echo "expected: $expected" >&2
    echo "observed: $observed" >&2
    exit 78
  fi
}

require_digest "srun" "$srun_bin" "$ISINGFOLD_RUNTIME_EXPECTED_SRUN_SHA256"
require_digest "Apptainer" "$apptainer_bin" "$ISINGFOLD_RUNTIME_EXPECTED_APPTAINER_SHA256"
require_digest "host Python" /usr/bin/python3 \
  "$ISINGFOLD_RUNTIME_EXPECTED_HOST_PYTHON_SHA256"
require_digest "wheel-builder definition" "$wheel_builder_def" \
  "$ISINGFOLD_RUNTIME_EXPECTED_WHEEL_BUILDER_DEF_SHA256"
require_digest "training definition" "$training_def" \
  "$ISINGFOLD_RUNTIME_EXPECTED_TRAINING_DEF_SHA256"
require_digest "runtime lock" "$runtime_lock" "$ISINGFOLD_RUNTIME_EXPECTED_LOCK_SHA256"
require_digest "runtime requirements" "$runtime_requirements" \
  "$ISINGFOLD_RUNTIME_EXPECTED_RUNTIME_REQUIREMENTS_SHA256"
require_digest "build requirements" "$build_requirements" \
  "$ISINGFOLD_RUNTIME_EXPECTED_BUILD_REQUIREMENTS_SHA256"
require_digest "runtime verifier" "$verifier" "$ISINGFOLD_RUNTIME_EXPECTED_VERIFIER_SHA256"

if [[ "$ISINGFOLD_RUNTIME_BUILD_OUTPUT_ROOT" != /* \
   || "$ISINGFOLD_RUNTIME_BUILD_OUTPUT_ROOT" == "/" \
   || -e "$ISINGFOLD_RUNTIME_BUILD_OUTPUT_ROOT" ]]; then
  echo "ISINGFOLD_RUNTIME_BUILD_OUTPUT_ROOT must be an absent absolute path" >&2
  exit 78
fi
output_parent=$(cd -- "$(dirname -- "$ISINGFOLD_RUNTIME_BUILD_OUTPUT_ROOT")" && pwd -P)
output_root="$output_parent/$(basename -- "$ISINGFOLD_RUNTIME_BUILD_OUTPUT_ROOT")"
mkdir -- "$output_root"
printf '%s\n' "runtime construction has not completed" > "$output_root/INCOMPLETE"
wheelhouse="$output_root/wheelhouse"
mkdir -- "$wheelhouse"
builder_image="$output_root/isingfold-wheel-builder.sif"
training_image="$output_root/isingfold-training.sif"
validation="$output_root/runtime-validation.json"

source_contract_sha256=$(
  /usr/bin/python3 -I "$verifier" source-inventory --source-root "$source_root" \
    | /usr/bin/python3 -I -c \
      'import json,sys; print(json.load(sys.stdin)["source_sha256"])'
)
if [[ "$source_contract_sha256" != "$ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" ]]; then
  echo "staged IsingFold source differs from ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" >&2
  exit 78
fi
cd -- "$source_root"

# The read-only bind is visible during %post and prevents the builder from
# mutating the scientific/operational source it is attesting.
"$srun_bin" "$apptainer_bin" build \
  --fakeroot \
  --bind "$source_root:/mnt/isingfold-source:ro" \
  "$builder_image" "$wheel_builder_def"

"$srun_bin" "$apptainer_bin" exec \
  --cleanenv --containall --no-home --no-mount bind-paths,hostfs \
  --bind "$wheelhouse:/mnt/isingfold-export:rw" \
  "$builder_image" /bin/sh -eu -c \
  'cp /opt/isingfold/wheelhouse/*.whl /mnt/isingfold-export/ &&
   cp /opt/isingfold/wheelhouse/wheelhouse.manifest.json /mnt/isingfold-export/ &&
   cp /opt/isingfold/wheelhouse/build-receipt.json /mnt/isingfold-export/'

manifest="$wheelhouse/wheelhouse.manifest.json"
build_receipt="$wheelhouse/build-receipt.json"
manifest_sha256=$(sha256_file "$manifest")
build_receipt_sha256=$(sha256_file "$build_receipt")
"$srun_bin" /usr/bin/python3 -I "$verifier" verify-wheelhouse \
  --wheelhouse "$wheelhouse" \
  --manifest "$manifest" \
  --expected-manifest-file-sha256 "$manifest_sha256"
"$srun_bin" /usr/bin/python3 -I "$verifier" verify-receipt \
  --path "$build_receipt" \
  --expected-file-sha256 "$build_receipt_sha256" \
  --schema isingfold.runtime-build-receipt \
  --schema-version 1

mapfile -t receipt_values < <(
  /usr/bin/python3 -I - "$build_receipt" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(receipt["source_inventory"]["source_sha256"])
print(receipt["project_wheel"]["wheel_filename"])
print(receipt["project_wheel"]["wheel_sha256"])
print(receipt["project_wheel"]["native_extension_sha256"])
print(receipt["installed_inventory"]["installation_sha256"])
print(receipt["build_environment"]["python_executable_sha256"])
PY
)
if [[ ${#receipt_values[@]} -ne 6 ]]; then
  echo "build receipt did not yield the six required runtime pins" >&2
  exit 78
fi
receipt_source_sha256=${receipt_values[0]}
project_wheel_name=${receipt_values[1]}
project_wheel_sha256=${receipt_values[2]}
native_extension_sha256=${receipt_values[3]}
installation_sha256=${receipt_values[4]}
python_executable_sha256=${receipt_values[5]}
if [[ "$receipt_source_sha256" != "$ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" \
   || "$project_wheel_name" == */* || "$project_wheel_name" != *.whl ]]; then
  echo "build receipt source or project-wheel name is invalid" >&2
  exit 78
fi
for digest in "$project_wheel_sha256" "$native_extension_sha256" \
  "$installation_sha256" "$python_executable_sha256"; do
  if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
    echo "build receipt contains an invalid runtime digest" >&2
    exit 78
  fi
done

"$srun_bin" "$apptainer_bin" build \
  --fakeroot \
  --bind "$wheelhouse:/mnt/isingfold-wheelhouse:ro" \
  --build-arg "WHEELHOUSE_MANIFEST_SHA256=$manifest_sha256" \
  --build-arg "BUILD_RECEIPT_SHA256=$build_receipt_sha256" \
  "$training_image" "$training_def"
training_image_sha256=$(sha256_file "$training_image")

pinned_image=/opt/isingfold-runtime-pins/environment.sif
"$srun_bin" "$apptainer_bin" exec \
  --cleanenv --containall --no-home --no-mount bind-paths,hostfs --nv \
  --bind "$source_root:$source_root:ro,$output_root:$output_root:rw,$training_image:$pinned_image:ro" \
  --pwd "$source_root" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONHASHSEED=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --env MKL_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  "$training_image" \
  /opt/isingfold/venv/bin/python -I "$verifier" verify-runtime \
  --runtime-lock "$runtime_lock" \
  --build-receipt /opt/isingfold/runtime/build-receipt.json \
  --expected-build-receipt-sha256 "$build_receipt_sha256" \
  --source-root "$source_root" \
  --expected-source-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" \
  --wheelhouse-manifest /opt/isingfold/runtime/wheelhouse.manifest.json \
  --expected-wheelhouse-manifest-sha256 "$manifest_sha256" \
  --project-wheel "/opt/isingfold/runtime/$project_wheel_name" \
  --expected-project-wheel-sha256 "$project_wheel_sha256" \
  --expected-native-extension-sha256 "$native_extension_sha256" \
  --expected-installation-sha256 "$installation_sha256" \
  --expected-python-executable-sha256 "$python_executable_sha256" \
  --execution-mode apptainer \
  --accelerator gpu \
  --image "$pinned_image" \
  --expected-image-sha256 "$training_image_sha256" \
  --out "$validation"

rm -- "$output_root/INCOMPLETE"
printf '%s\n' \
  "ISINGFOLD_RUNTIME_BUILD_RECEIPT=$build_receipt" \
  "ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256=$build_receipt_sha256" \
  "ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256=$ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" \
  "ISINGFOLD_RUNTIME_WHEELHOUSE_MANIFEST=$manifest" \
  "ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256=$manifest_sha256" \
  "ISINGFOLD_RUNTIME_PROJECT_WHEEL=$wheelhouse/$project_wheel_name" \
  "ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256=$project_wheel_sha256" \
  "ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256=$native_extension_sha256" \
  "ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256=$installation_sha256" \
  "ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256=$python_executable_sha256" \
  "ISINGFOLD_CONTAINER_IMAGE=$training_image" \
  "ISINGFOLD_CONTAINER_IMAGE_SHA256=$training_image_sha256" \
  "ISINGFOLD_RUNTIME_VALIDATION_RECEIPT=$validation"

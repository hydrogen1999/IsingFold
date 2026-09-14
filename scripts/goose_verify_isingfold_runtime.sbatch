#!/usr/bin/env bash
# Revalidate an already built IsingFold SIF on a Goose GPU compute node.
#SBATCH --job-name=isingfold-runtime-verify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=isingfold-runtime-verify-%j.out
#SBATCH --error=isingfold-runtime-verify-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_PARTITION:-}" \
   || -z "${SLURMD_NODENAME:-}" ]]; then
  echo "Goose runtime verification requires an active Slurm compute-node allocation" >&2
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
: "${ISINGFOLD_CONTAINER_IMAGE:?set ISINGFOLD_CONTAINER_IMAGE}"
: "${ISINGFOLD_CONTAINER_IMAGE_SHA256:?set ISINGFOLD_CONTAINER_IMAGE_SHA256}"
: "${ISINGFOLD_RUNTIME_BUILD_RECEIPT:?set ISINGFOLD_RUNTIME_BUILD_RECEIPT}"
: "${ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256}"
: "${ISINGFOLD_RUNTIME_VALIDATION_OUTPUT_ROOT:?set ISINGFOLD_RUNTIME_VALIDATION_OUTPUT_ROOT}"

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source_root=$(cd -- "$script_dir/.." && pwd -P)
verifier="$script_dir/verify_isingfold_runtime.py"
runtime_lock="$source_root/runtime/isingfold-training/runtime-lock.json"

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
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ \
     || "$(sha256_file "$path")" != "$expected" ]]; then
    echo "$label differs from its out-of-band SHA-256 commitment" >&2
    exit 78
  fi
}

require_digest "srun" "$srun_bin" "$ISINGFOLD_RUNTIME_EXPECTED_SRUN_SHA256"
require_digest "Apptainer" "$apptainer_bin" "$ISINGFOLD_RUNTIME_EXPECTED_APPTAINER_SHA256"
require_digest "host Python" /usr/bin/python3 \
  "$ISINGFOLD_RUNTIME_EXPECTED_HOST_PYTHON_SHA256"
if [[ "$ISINGFOLD_CONTAINER_IMAGE" != /* || ! -f "$ISINGFOLD_CONTAINER_IMAGE" \
   || -L "$ISINGFOLD_CONTAINER_IMAGE" ]]; then
  echo "ISINGFOLD_CONTAINER_IMAGE must be an absolute regular non-symbolic-link file" >&2
  exit 78
fi
require_digest "training SIF" "$ISINGFOLD_CONTAINER_IMAGE" \
  "$ISINGFOLD_CONTAINER_IMAGE_SHA256"
require_digest "build receipt" "$ISINGFOLD_RUNTIME_BUILD_RECEIPT" \
  "$ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256"
/usr/bin/python3 -I "$verifier" verify-receipt \
  --path "$ISINGFOLD_RUNTIME_BUILD_RECEIPT" \
  --expected-file-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256" \
  --schema isingfold.runtime-build-receipt \
  --schema-version 1

project_wheel_name=$(
  /usr/bin/python3 -I - "$ISINGFOLD_RUNTIME_BUILD_RECEIPT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(receipt["project_wheel"]["wheel_filename"])
PY
)
if [[ "$project_wheel_name" == */* || "$project_wheel_name" != *.whl ]]; then
  echo "build receipt contains an invalid project wheel filename" >&2
  exit 78
fi

if [[ "$ISINGFOLD_RUNTIME_VALIDATION_OUTPUT_ROOT" != /* \
   || "$ISINGFOLD_RUNTIME_VALIDATION_OUTPUT_ROOT" == "/" \
   || -e "$ISINGFOLD_RUNTIME_VALIDATION_OUTPUT_ROOT" ]]; then
  echo "ISINGFOLD_RUNTIME_VALIDATION_OUTPUT_ROOT must be an absent absolute path" >&2
  exit 78
fi
output_parent=$(cd -- "$(dirname -- "$ISINGFOLD_RUNTIME_VALIDATION_OUTPUT_ROOT")" && pwd -P)
output_root="$output_parent/$(basename -- "$ISINGFOLD_RUNTIME_VALIDATION_OUTPUT_ROOT")"
mkdir -- "$output_root"
printf '%s\n' "runtime verification has not completed" > "$output_root/INCOMPLETE"
validation="$output_root/runtime-validation.json"
pinned_image=/opt/isingfold-runtime-pins/environment.sif

"$srun_bin" "$apptainer_bin" exec \
  --cleanenv --containall --no-home --no-mount bind-paths,hostfs --nv \
  --bind "$source_root:$source_root:ro,$output_root:$output_root:rw,$ISINGFOLD_CONTAINER_IMAGE:$pinned_image:ro" \
  --pwd "$source_root" \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONHASHSEED=0 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --env MKL_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  "$ISINGFOLD_CONTAINER_IMAGE" \
  /opt/isingfold/venv/bin/python -I "$verifier" verify-runtime \
  --runtime-lock "$runtime_lock" \
  --build-receipt /opt/isingfold/runtime/build-receipt.json \
  --expected-build-receipt-sha256 \
  "$ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256" \
  --source-root "$source_root" \
  --expected-source-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" \
  --wheelhouse-manifest /opt/isingfold/runtime/wheelhouse.manifest.json \
  --expected-wheelhouse-manifest-sha256 \
  "$ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256" \
  --project-wheel "/opt/isingfold/runtime/$project_wheel_name" \
  --expected-project-wheel-sha256 \
  "$ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256" \
  --expected-native-extension-sha256 \
  "$ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256" \
  --expected-installation-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256" \
  --expected-python-executable-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256" \
  --execution-mode apptainer \
  --accelerator gpu \
  --image "$pinned_image" \
  --expected-image-sha256 "$ISINGFOLD_CONTAINER_IMAGE_SHA256" \
  --out "$validation"

rm -- "$output_root/INCOMPLETE"
printf 'Goose CUDA runtime validation receipt: %s\n' "$validation"

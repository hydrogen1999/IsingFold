#!/usr/bin/env bash
# Authenticate the complete paper runtime and run a real CUDA smoke for GPU stages.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: verify_runtime_source.sh PYTHON ACCELERATOR" >&2
  exit 64
fi

python_bin=$1
accelerator=$2
if [[ "$accelerator" != "cpu" && "$accelerator" != "gpu" ]]; then
  echo "ACCELERATOR must be cpu or gpu" >&2
  exit 64
fi
if [[ "$accelerator" == "gpu" && -n "${ISINGFOLD_DEVICE:-}" \
   && "$ISINGFOLD_DEVICE" != "cuda" ]]; then
  echo "GPU paper stages reject ISINGFOLD_DEVICE values other than cuda" >&2
  exit 78
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/.." && pwd -P)
verifier="$script_dir/verify_isingfold_runtime.py"
runtime_lock="$source_root/runtime/isingfold-training/runtime-lock.json"

: "${ISINGFOLD_RUNTIME_BUILD_RECEIPT:?set ISINGFOLD_RUNTIME_BUILD_RECEIPT}"
: "${ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256}"
: "${ISINGFOLD_RUNTIME_WHEELHOUSE_MANIFEST:?set ISINGFOLD_RUNTIME_WHEELHOUSE_MANIFEST}"
: "${ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256}"
: "${ISINGFOLD_RUNTIME_PROJECT_WHEEL:?set ISINGFOLD_RUNTIME_PROJECT_WHEEL}"
: "${ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256}"
: "${ISINGFOLD_RUNTIME_EXECUTION_MODE:?set ISINGFOLD_RUNTIME_EXECUTION_MODE}"

arguments=(
  "$verifier" verify-runtime
  --runtime-lock "$runtime_lock"
  --build-receipt "$ISINGFOLD_RUNTIME_BUILD_RECEIPT"
  --expected-build-receipt-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256"
  --source-root "$source_root"
  --expected-source-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256"
  --wheelhouse-manifest "$ISINGFOLD_RUNTIME_WHEELHOUSE_MANIFEST"
  --expected-wheelhouse-manifest-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256"
  --project-wheel "$ISINGFOLD_RUNTIME_PROJECT_WHEEL"
  --expected-project-wheel-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256"
  --expected-native-extension-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256"
  --expected-installation-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256"
  --expected-python-executable-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256"
  --execution-mode "$ISINGFOLD_RUNTIME_EXECUTION_MODE"
  --accelerator "$accelerator"
)

case "$ISINGFOLD_RUNTIME_EXECUTION_MODE" in
  apptainer)
    : "${ISINGFOLD_RUNTIME_IMAGE:?set ISINGFOLD_RUNTIME_IMAGE}"
    : "${ISINGFOLD_RUNTIME_EXPECTED_IMAGE_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_IMAGE_SHA256}"
    arguments+=(
      --image "$ISINGFOLD_RUNTIME_IMAGE"
      --expected-image-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_IMAGE_SHA256"
    )
    ;;
  pinned-venv)
    canonical_environment_lock="$source_root/runtime/isingfold-training/requirements-linux-x86_64-py312-cu128.lock"
    if [[ "${ISINGFOLD_ENV_LOCK:-}" != "$canonical_environment_lock" ]]; then
      echo "pinned-venv execution requires ISINGFOLD_ENV_LOCK to name the canonical runtime lock" >&2
      exit 78
    fi
    if [[ -n "${ISINGFOLD_RUNTIME_WHEELHOUSE:-}" ]]; then
      arguments+=(--wheelhouse "$ISINGFOLD_RUNTIME_WHEELHOUSE")
    fi
    ;;
  *)
    echo "ISINGFOLD_RUNTIME_EXECUTION_MODE must be apptainer or pinned-venv" >&2
    exit 78
    ;;
esac

exec "$python_bin" -I "${arguments[@]}"

#!/usr/bin/env bash
# Materialize Apollo's clean venv from the exact wheelhouse built on Goose.
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Apollo runtime construction rejects Slurm environments" >&2
  exit 69
fi
if [[ $# -ne 2 ]]; then
  echo "usage: $0 WHEELHOUSE NEW_OUTPUT_ROOT" >&2
  exit 64
fi

wheelhouse=$1
output_root=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/.." && pwd -P)
runtime_root="$source_root/runtime/isingfold-training"
verifier="$script_dir/verify_isingfold_runtime.py"
runtime_lock="$runtime_root/runtime-lock.json"
runtime_requirements="$runtime_root/requirements-linux-x86_64-py312-cu128.lock"

unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1

: "${ISINGFOLD_BASE_PYTHON:?set ISINGFOLD_BASE_PYTHON to the Apollo CPython 3.12.3 executable}"
: "${EXPECTED_BASE_PYTHON_SHA256:?set EXPECTED_BASE_PYTHON_SHA256 out of band}"
: "${ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256}"
: "${ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256:?set ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256}"

if [[ "$ISINGFOLD_BASE_PYTHON" != /* || ! -f "$ISINGFOLD_BASE_PYTHON" \
   || ! -x "$ISINGFOLD_BASE_PYTHON" ]]; then
  echo "ISINGFOLD_BASE_PYTHON must be an absolute executable" >&2
  exit 78
fi
if [[ ! "$EXPECTED_BASE_PYTHON_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_BASE_PYTHON_SHA256 must be one lowercase SHA-256 digest" >&2
  exit 78
fi
observed_base_python_sha256=$(
  "$ISINGFOLD_BASE_PYTHON" -I - "$ISINGFOLD_BASE_PYTHON" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
digest = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)
if [[ "$observed_base_python_sha256" != "$EXPECTED_BASE_PYTHON_SHA256" ]]; then
  echo "Apollo base Python differs from EXPECTED_BASE_PYTHON_SHA256" >&2
  exit 78
fi
if ! "$ISINGFOLD_BASE_PYTHON" -I - <<'PY'
import platform
import sys

expected = ("3.12.3", "Linux", "x86_64")
observed = (platform.python_version(), platform.system(), platform.machine())
if observed != expected:
    raise SystemExit(f"expected Python/Linux/machine {expected!r}, observed {observed!r}")
if sys.version_info[:3] != (3, 12, 3):
    raise SystemExit("base interpreter is not exactly CPython 3.12.3")
if platform.python_implementation() != "CPython":
    raise SystemExit("base interpreter implementation is not CPython")
PY
then
  echo "Apollo base interpreter does not satisfy the runtime lock" >&2
  exit 78
fi

if [[ "$wheelhouse" != /* || ! -d "$wheelhouse" || -L "$wheelhouse" ]]; then
  echo "WHEELHOUSE must be an absolute existing non-symbolic-link directory" >&2
  exit 78
fi
wheelhouse=$(cd -- "$wheelhouse" && pwd -P)
if [[ "$output_root" != /* || "$output_root" == "/" || -e "$output_root" ]]; then
  echo "NEW_OUTPUT_ROOT must be an absent absolute path" >&2
  exit 78
fi
output_parent=$(cd -- "$(dirname -- "$output_root")" && pwd -P)
output_root="$output_parent/$(basename -- "$output_root")"

manifest="$wheelhouse/wheelhouse.manifest.json"
build_receipt="$wheelhouse/build-receipt.json"
source_contract_sha256=$(
  "$ISINGFOLD_BASE_PYTHON" -I "$verifier" \
    source-inventory --source-root "$source_root" \
    | "$ISINGFOLD_BASE_PYTHON" -I -c \
      'import json,sys; print(json.load(sys.stdin)["source_sha256"])'
)
if [[ "$source_contract_sha256" != "$ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" ]]; then
  echo "Apollo source differs from ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" >&2
  exit 78
fi
"$ISINGFOLD_BASE_PYTHON" -I "$verifier" verify-wheelhouse \
  --wheelhouse "$wheelhouse" \
  --manifest "$manifest" \
  --expected-manifest-file-sha256 \
  "$ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256"
"$ISINGFOLD_BASE_PYTHON" -I "$verifier" verify-receipt \
  --path "$build_receipt" \
  --expected-file-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256" \
  --schema isingfold.runtime-build-receipt \
  --schema-version 1
project_wheel=$(
  "$ISINGFOLD_BASE_PYTHON" -I "$verifier" project-wheel-path \
    --wheelhouse "$wheelhouse" \
    --manifest "$manifest" \
    --expected-manifest-file-sha256 \
    "$ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256"
)

mkdir -- "$output_root"
printf '%s\n' "runtime construction has not completed" > "$output_root/INCOMPLETE"

venv_root="$output_root/venv"
"$ISINGFOLD_BASE_PYTHON" -I -m venv "$venv_root"
python_bin="$venv_root/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PIP_NO_INPUT=1
export PIP_NO_INDEX=1
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export PYTHONSAFEPATH=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$python_bin" -I -m pip install \
  --no-index \
  --find-links "$wheelhouse" \
  --require-hashes \
  -r "$runtime_requirements"
"$python_bin" -I -m pip install --no-index --no-deps "$project_wheel"
"$python_bin" -I -m pip check

validation="$output_root/runtime-validation.json"
"$python_bin" -I "$verifier" verify-runtime \
  --runtime-lock "$runtime_lock" \
  --build-receipt "$build_receipt" \
  --expected-build-receipt-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256" \
  --source-root "$source_root" \
  --expected-source-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" \
  --wheelhouse-manifest "$manifest" \
  --expected-wheelhouse-manifest-sha256 \
  "$ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256" \
  --project-wheel "$project_wheel" \
  --expected-project-wheel-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256" \
  --expected-native-extension-sha256 \
  "$ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256" \
  --expected-installation-sha256 "$ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256" \
  --expected-python-executable-sha256 "$EXPECTED_BASE_PYTHON_SHA256" \
  --execution-mode pinned-venv \
  --accelerator gpu \
  --wheelhouse "$wheelhouse" \
  --out "$validation"

rm -- "$output_root/INCOMPLETE"
printf 'Apollo runtime ready: %s\nValidation receipt: %s\n' "$python_bin" "$validation"

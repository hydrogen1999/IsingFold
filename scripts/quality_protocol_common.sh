#!/usr/bin/env bash
# Shared argument construction for the immutable post-bank quality protocol.

quality_require_digest() {
  local name=$1
  local value=${!name:-}
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be one lowercase SHA-256 digest" >&2
    return 64
  fi
}

quality_bind_verified_runtime_identity() {
  if [[ $# -ne 2 ]]; then
    echo "runtime binding requires VALIDATION_JSON PYTHON" >&2
    return 64
  fi
  local validation_json=$1
  local validation_python=$2
  quality_verified_runtime_sha256=
  : "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?set ISINGFOLD_EXECUTION_RUNTIME_SHA256}"
  quality_require_digest ISINGFOLD_EXECUTION_RUNTIME_SHA256
  if [[ "$validation_python" != /* || ! -f "$validation_python" \
     || ! -x "$validation_python" ]]; then
    echo "runtime binding requires an absolute executable Python" >&2
    return 78
  fi

  local observed
  if ! observed=$(
    "$validation_python" -I -c '
import json
import re
import sys


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


raw = sys.stdin.read()
document = json.loads(
    raw,
    parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON token: {token}")
    ),
    object_pairs_hook=unique_object,
)
canonical = json.dumps(
    document,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
)
if raw != canonical + "\n":
    raise ValueError("runtime validation receipt is not canonical JSON")
if (
    not isinstance(document, dict)
    or document.get("schema") != "isingfold.runtime-validation-receipt"
    or document.get("schema_version") != 1
):
    raise ValueError("runtime validation receipt schema differs")
digest = document.get("checkpoint_runtime_registry_sha256")
if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise ValueError("runtime validation receipt has no checkpoint registry SHA-256")
print(digest)
' <<< "$validation_json"
  ); then
    echo "runtime validation output is not strict canonical JSON" >&2
    return 78
  fi
  if [[ "$observed" != "$ISINGFOLD_EXECUTION_RUNTIME_SHA256" ]]; then
    echo "ISINGFOLD_EXECUTION_RUNTIME_SHA256 differs from the verified runtime" >&2
    return 78
  fi
  quality_verified_runtime_sha256=$observed
}

quality_require_regular_pin_file() {
  local name=$1
  local value=${!name:-}
  if [[ -z "$value" || "$value" != /* || ! -f "$value" || -L "$value" ]]; then
    echo "$name must be an absolute non-symlink regular file" >&2
    return 78
  fi
}

quality_require_absent_output() {
  local name=$1
  local value=${!name:-}
  if [[ -z "$value" || "$value" != /* || -e "$value" || -L "$value" ]]; then
    echo "$name must be an absent absolute path" >&2
    return 78
  fi
  if [[ ! -d "$(dirname -- "$value")" ]]; then
    echo "$name parent directory must already exist" >&2
    return 78
  fi
}

quality_require_output_under_root() {
  local output_name=$1
  local root_name=$2
  local output=${!output_name:-}
  local root=${!root_name:-}
  if [[ -z "$output" || "$output" != /* ]]; then
    echo "$output_name must be an absolute path" >&2
    return 78
  fi
  if [[ -z "$root" || "$root" != /* || ! -d "$root" || -L "$root" ]]; then
    echo "$root_name must be an absolute non-symlink directory" >&2
    return 78
  fi
  local physical_root
  local physical_parent
  physical_root=$(cd -- "$root" && pwd -P)
  physical_parent=$(cd -- "$(dirname -- "$output")" && pwd -P)
  case "$physical_parent/" in
    "$physical_root/"*) ;;
    *)
      echo "$output_name must be contained by $root_name" >&2
      return 78
      ;;
  esac
}

quality_prepare_output_root() {
  local name=$1
  local value=${!name:-}
  if [[ -z "$value" || "$value" != /* || "$value" == "/" || -L "$value" ]]; then
    echo "$name must be a dedicated absolute non-symlink directory" >&2
    return 78
  fi
  local parent
  local physical_parent
  parent=$(dirname -- "$value")
  if [[ ! -d "$parent" || -L "$parent" ]]; then
    echo "$name parent must be an existing canonical directory" >&2
    return 78
  fi
  physical_parent=$(cd -- "$parent" && pwd -P)
  if [[ "$physical_parent" != "$parent" ]]; then
    echo "$name parent must be an existing canonical directory" >&2
    return 78
  fi
  if [[ -e "$value" && ! -d "$value" ]]; then
    echo "$name must contain directories only" >&2
    return 78
  fi
  mkdir -p -- "$value"
  local physical
  physical=$(cd -- "$value" && pwd -P)
  if [[ "$physical" != "$value" ]]; then
    echo "$name must be a canonical physical path" >&2
    return 78
  fi
}

quality_initialize_protocol_arguments() {
  : "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
  : "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
  : "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
  : "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
  : "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
  : "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
  : "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
  : "${ISINGFOLD_QUALITY_INITIALIZER_BANK:?set ISINGFOLD_QUALITY_INITIALIZER_BANK}"
  : "${EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256:?set EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256}"
  : "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
  : "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?set ISINGFOLD_EXECUTION_RUNTIME_SHA256}"
  : "${ISINGFOLD_QUALITY_ACCELERATOR:?set ISINGFOLD_QUALITY_ACCELERATOR to cpu or gpu}"

  quality_require_digest EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256
  quality_require_digest EXPECTED_QUALITY_ATTESTATION_DIGEST
  quality_require_digest EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256
  quality_require_digest ISINGFOLD_EXECUTION_RUNTIME_SHA256
  if [[ "$ISINGFOLD_QUALITY_ACCELERATOR" == "gpu" ]]; then
    quality_device=cuda
  elif [[ "$ISINGFOLD_QUALITY_ACCELERATOR" == "cpu" ]]; then
    quality_device=cpu
  else
    echo "ISINGFOLD_QUALITY_ACCELERATOR must be cpu or gpu" >&2
    return 64
  fi
  quality_threads=${ISINGFOLD_THREADS:-1}
  if [[ ! "$quality_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "ISINGFOLD_THREADS must be a positive integer" >&2
    return 64
  fi

  quality_public_args=(
    --corpus "$ISINGFOLD_CORPUS"
    --selector "$ISINGFOLD_SELECTOR_BUNDLE"
    --quality-attestation "$QUALITY_ATTESTATION"
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST"
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID"
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT"
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256"
  )
  quality_initializer_args=(
    --initializer-bank "$ISINGFOLD_QUALITY_INITIALIZER_BANK"
    --expected-initializer-bank-manifest-sha256
    "$EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256"
    --complete-config "$ISINGFOLD_COMPLETE_CONFIG"
  )
  quality_compute_args=(--device "$quality_device" --threads "$quality_threads")
  quality_optional_qubit_args=()
  if [[ -n "${ISINGFOLD_QUBIT_CAP:-}" ]]; then
    if [[ ! "$ISINGFOLD_QUBIT_CAP" =~ ^[1-9][0-9]*$ ]]; then
      echo "ISINGFOLD_QUBIT_CAP must be a positive integer" >&2
      return 64
    fi
    quality_optional_qubit_args=(--qubit-cap "$ISINGFOLD_QUBIT_CAP")
  fi
}

quality_resolution_plan_args() {
  : "${ISINGFOLD_QUALITY_RESOLUTION_PLAN:?set ISINGFOLD_QUALITY_RESOLUTION_PLAN}"
  : "${EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256:?set EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256}"
  quality_require_digest EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256
  quality_plan_args=(
    --plan "$ISINGFOLD_QUALITY_RESOLUTION_PLAN"
    --expected-plan-sha256 "$EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256"
  )
}

quality_resolution_receipt_args() {
  : "${ISINGFOLD_QUALITY_RESOLUTION_RECEIPT:?set ISINGFOLD_QUALITY_RESOLUTION_RECEIPT}"
  : "${EXPECTED_QUALITY_RESOLUTION_RECEIPT_SHA256:?set EXPECTED_QUALITY_RESOLUTION_RECEIPT_SHA256}"
  quality_require_digest EXPECTED_QUALITY_RESOLUTION_RECEIPT_SHA256
  quality_study_args=(
    --resolution-receipt "$ISINGFOLD_QUALITY_RESOLUTION_RECEIPT"
    --expected-resolution-receipt-sha256
    "$EXPECTED_QUALITY_RESOLUTION_RECEIPT_SHA256"
  )
}

quality_capacity_artifact_args() {
  : "${ISINGFOLD_QUALITY_CAPACITY_SELECTION:?set ISINGFOLD_QUALITY_CAPACITY_SELECTION}"
  : "${EXPECTED_QUALITY_CAPACITY_SELECTION_SHA256:?set EXPECTED_QUALITY_CAPACITY_SELECTION_SHA256}"
  : "${ISINGFOLD_QUALITY_CAPACITY_BUDGET:?set ISINGFOLD_QUALITY_CAPACITY_BUDGET}"
  : "${EXPECTED_QUALITY_CAPACITY_BUDGET_SHA256:?set EXPECTED_QUALITY_CAPACITY_BUDGET_SHA256}"
  quality_require_digest EXPECTED_QUALITY_CAPACITY_SELECTION_SHA256
  quality_require_digest EXPECTED_QUALITY_CAPACITY_BUDGET_SHA256
  quality_capacity_args=(
    --capacity-selection "$ISINGFOLD_QUALITY_CAPACITY_SELECTION"
    --expected-capacity-selection-sha256
    "$EXPECTED_QUALITY_CAPACITY_SELECTION_SHA256"
    --capacity-budget "$ISINGFOLD_QUALITY_CAPACITY_BUDGET"
    --expected-capacity-budget-sha256 "$EXPECTED_QUALITY_CAPACITY_BUDGET_SHA256"
  )
}

quality_read_json_pin_lines() {
  local path=$1
  quality_json_pin_args=()
  local line
  local count=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -z "$line" ]]; then
      echo "resolution pin file cannot contain blank lines" >&2
      return 64
    fi
    quality_json_pin_args+=(--pin "$line")
    count=$((count + 1))
  done < "$path"
  if (( count == 0 )); then
    echo "resolution pin file cannot be empty" >&2
    return 64
  fi
}

quality_read_manifest_pin_lines() {
  local path=$1
  quality_artifact_paths=()
  quality_artifact_sha256s=()
  local artifact
  local digest
  local remainder
  local count=0
  while IFS=$'\t' read -r artifact digest remainder || [[ -n "$artifact$digest$remainder" ]]; do
    if [[ -z "$artifact" || "$artifact" != /* || -n "$remainder" \
       || ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
      echo "manifest pin file requires PATH<TAB>LOWERCASE_SHA256 per line" >&2
      return 64
    fi
    quality_artifact_paths+=("$artifact")
    quality_artifact_sha256s+=("$digest")
    count=$((count + 1))
  done < "$path"
  if (( count == 0 )); then
    echo "manifest pin file cannot be empty" >&2
    return 64
  fi
}

quality_lookup_manifest_pin() {
  local expected_path=$1
  local pin_file=$2
  quality_read_manifest_pin_lines "$pin_file"
  quality_lookup_sha256=
  local index
  for ((index=0; index<${#quality_artifact_paths[@]}; index++)); do
    if [[ "${quality_artifact_paths[$index]}" == "$expected_path" ]]; then
      if [[ -n "$quality_lookup_sha256" ]]; then
        echo "manifest pin file contains duplicate path: $expected_path" >&2
        return 64
      fi
      quality_lookup_sha256=${quality_artifact_sha256s[$index]}
    fi
  done
  if [[ -z "$quality_lookup_sha256" ]]; then
    echo "manifest pin file has no entry for: $expected_path" >&2
    return 78
  fi
}

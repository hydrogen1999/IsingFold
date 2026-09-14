#!/usr/bin/env bash
# Shared fail-closed runtime validation for Apollo and Goose publication jobs.

publication_file_sha256() {
  local path=$1
  if [[ -x /usr/bin/sha256sum ]]; then
    /usr/bin/sha256sum -- "$path" | /usr/bin/awk '{print $1}'
  elif [[ -x /usr/bin/shasum ]]; then
    /usr/bin/shasum -a 256 -- "$path" | /usr/bin/awk '{print $1}'
  else
    echo "no SHA-256 utility is available" >&2
    return 78
  fi
}

require_quality_accelerator_control() {
  if [[ $# -ne 2 ]]; then
    echo "quality accelerator validation requires ACCELERATOR PYTHON" >&2
    return 64
  fi
  local accelerator=$1
  local validation_python=$2
  if [[ "$accelerator" == "gpu" ]]; then
    return 0
  fi
  if [[ "$accelerator" != "cpu" ]]; then
    echo "ISINGFOLD_QUALITY_ACCELERATOR must be cpu or gpu" >&2
    return 78
  fi

  local parity_receipt=${ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT:-}
  local parity_sha256=${ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT_SHA256:-}
  if [[ -z "$parity_receipt" || "$parity_receipt" != /* \
     || ! -f "$parity_receipt" || -L "$parity_receipt" ]]; then
    echo "CPU quality inference requires an absolute selector-parity receipt" >&2
    return 78
  fi
  if [[ ! "$parity_sha256" =~ ^[0-9a-f]{64}$ \
     || "$(publication_file_sha256 "$parity_receipt")" != "$parity_sha256" ]]; then
    echo "CPU quality selector-parity receipt differs from its external pin" >&2
    return 78
  fi
  if ! "$validation_python" -I - "$parity_receipt" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


document = json.loads(
    path.read_text(encoding="utf-8"),
    parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON token: {token}")
    ),
    object_pairs_hook=unique_object,
)
embedded = document.pop("record_digest", None)
canonical = json.dumps(
    document,
    allow_nan=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
valid = (
    document.get("schema") == "isingfold.quality-selector-device-parity"
    and document.get("schema_version") == 1
    and document.get("all_equal") is True
    and document.get("mismatch_count") == 0
    and isinstance(embedded, str)
    and hashlib.sha256(canonical).hexdigest() == embedded
)
if not valid:
    raise SystemExit("selector parity receipt does not certify exact CPU/CUDA parity")
PY
  then
    echo "CPU quality selector-parity receipt is not an exact all-equal certificate" >&2
    return 78
  fi
}

require_cuda_cli_arguments() {
  local device_count=0
  local argument
  while (( $# > 0 )); do
    argument=$1
    shift
    case "$argument" in
      --device)
        if (( $# == 0 )) || [[ "$1" != "cuda" ]]; then
          echo "paper model evaluation requires --device cuda" >&2
          return 78
        fi
        device_count=$((device_count + 1))
        shift
        ;;
      --device=*)
        if [[ "$argument" != "--device=cuda" ]]; then
          echo "paper model evaluation requires --device cuda" >&2
          return 78
        fi
        device_count=$((device_count + 1))
        ;;
    esac
  done
  if (( device_count != 1 )); then
    echo "paper model evaluation requires exactly one --device cuda argument" >&2
    return 78
  fi
}

require_apollo_publication_runtime() {
  if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Apollo publication runtime rejects Slurm environments" >&2
    return 69
  fi

  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE
  export PYTHONNOUSERSITE=1
  export PYTHONSAFEPATH=1
  export PIP_CONFIG_FILE=/dev/null
  export PIP_NO_INDEX=1
  export PIP_NO_INPUT=1

  local configured_python=${ISINGFOLD_PYTHON:-}
  if [[ -z "$configured_python" || "$configured_python" != /* \
     || ! -f "$configured_python" || ! -x "$configured_python" ]]; then
    echo "Apollo publication runs require ISINGFOLD_PYTHON to be an absolute executable" >&2
    return 78
  fi

  local python_parent
  local venv_root
  python_parent=$(cd -- "$(dirname -- "$configured_python")" && pwd -P)
  venv_root=$(cd -- "$python_parent/.." && pwd -P)
  if [[ ! -f "$venv_root/pyvenv.cfg" ]]; then
    echo "ISINGFOLD_PYTHON must belong to a virtualenv containing $venv_root/pyvenv.cfg" >&2
    return 78
  fi

  local environment_lock=${ISINGFOLD_ENV_LOCK:-}
  local expected_lock_sha256=${ISINGFOLD_ENV_LOCK_SHA256:-}
  if [[ -z "$environment_lock" || ! -f "$environment_lock" \
     || ! -r "$environment_lock" ]]; then
    echo "Apollo publication runs require a readable ISINGFOLD_ENV_LOCK file" >&2
    return 78
  fi
  if [[ ! "$expected_lock_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ISINGFOLD_ENV_LOCK_SHA256 must be a 64-hex lowercase SHA-256 digest" >&2
    return 78
  fi

  python_bin=$configured_python
  local actual_lock_sha256
  if ! actual_lock_sha256=$(
    "$python_bin" -I - "$environment_lock" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
  ); then
    echo "failed to compute ISINGFOLD_ENV_LOCK SHA-256" >&2
    return 78
  fi
  if [[ "$actual_lock_sha256" != "$expected_lock_sha256" ]]; then
    echo "ISINGFOLD_ENV_LOCK digest mismatch" >&2
    echo "expected: $expected_lock_sha256" >&2
    echo "actual:   $actual_lock_sha256" >&2
    return 78
  fi

  if ! "$python_bin" -I -m pip check; then
    echo "pinned virtualenv pip check failed" >&2
    return 78
  fi
}

require_goose_publication_runtime() {
  if (( $# < 3 )); then
    echo "Goose publication runtime requires SCRIPTS_DIR ACCELERATOR OUTPUT_ROOT..." >&2
    return 64
  fi
  local requested_script_dir=$1
  local accelerator=$2
  shift 2
  if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_PARTITION:-}" ]]; then
    echo "Goose publication runtime requires a Slurm allocation and partition" >&2
    return 69
  fi
  if [[ ! "${SLURM_CPUS_PER_TASK:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Goose publication runtime requires a positive SLURM_CPUS_PER_TASK" >&2
    return 69
  fi
  if [[ -n "${threads:-}" ]]; then
    if [[ ! "$threads" =~ ^[1-9][0-9]*$ \
       || "$threads" -gt "$SLURM_CPUS_PER_TASK" ]]; then
      echo "ISINGFOLD_THREADS must be positive and no greater than SLURM_CPUS_PER_TASK" >&2
      return 78
    fi
  fi
  if [[ "$accelerator" != "cpu" && "$accelerator" != "gpu" ]]; then
    echo "Goose publication accelerator must be cpu or gpu" >&2
    return 78
  fi
  if [[ -z "$requested_script_dir" || "$requested_script_dir" != /* \
     || ! -d "$requested_script_dir" ]]; then
    echo "Goose publication runtime requires an absolute scripts directory" >&2
    return 78
  fi

  apptainer_bin=${ISINGFOLD_APPTAINER:-}
  apptainer_sha256=${ISINGFOLD_APPTAINER_SHA256:-}
  container_image=${ISINGFOLD_CONTAINER_IMAGE:-}
  container_image_sha256=${ISINGFOLD_CONTAINER_IMAGE_SHA256:-}
  container_python=${ISINGFOLD_CONTAINER_PYTHON:-}
  if [[ -z "$apptainer_bin" || "$apptainer_bin" != /* \
     || ! -f "$apptainer_bin" || ! -x "$apptainer_bin" ]]; then
    echo "Goose publication runs require ISINGFOLD_APPTAINER as an absolute executable" >&2
    return 78
  fi
  if [[ ! "$apptainer_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ISINGFOLD_APPTAINER_SHA256 must be a 64-hex lowercase SHA-256 digest" >&2
    return 78
  fi
  if [[ -z "$container_image" || "$container_image" != /* \
     || ! -f "$container_image" || ! -r "$container_image" ]]; then
    echo "Goose publication runs require ISINGFOLD_CONTAINER_IMAGE as an absolute readable file" >&2
    return 78
  fi
  if [[ ! "$container_image_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ISINGFOLD_CONTAINER_IMAGE_SHA256 must be a 64-hex lowercase SHA-256 digest" >&2
    return 78
  fi
  if [[ "$container_python" != "/opt/isingfold/venv/bin/python" ]]; then
    echo "ISINGFOLD_CONTAINER_PYTHON must be /opt/isingfold/venv/bin/python" >&2
    return 78
  fi

  local observed_runtime_sha256
  local observed_image_sha256
  if ! observed_runtime_sha256=$(publication_file_sha256 "$apptainer_bin") \
     || ! observed_image_sha256=$(publication_file_sha256 "$container_image"); then
    echo "failed to authenticate the Goose publication runtime" >&2
    return 78
  fi
  if [[ "$observed_runtime_sha256" != "$apptainer_sha256" ]]; then
    echo "Apptainer runtime differs from ISINGFOLD_APPTAINER_SHA256" >&2
    return 78
  fi
  if [[ "$observed_image_sha256" != "$container_image_sha256" ]]; then
    echo "Apptainer image differs from ISINGFOLD_CONTAINER_IMAGE_SHA256" >&2
    return 78
  fi

  local runtime_build_receipt=${ISINGFOLD_RUNTIME_BUILD_RECEIPT:-}
  local expected_build_receipt=${ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256:-}
  local runtime_manifest=${ISINGFOLD_RUNTIME_WHEELHOUSE_MANIFEST:-}
  local expected_manifest=${ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256:-}
  local runtime_project_wheel=${ISINGFOLD_RUNTIME_PROJECT_WHEEL:-}
  local expected_project_wheel=${ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256:-}
  local expected_native=${ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256:-}
  local expected_installation=${ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256:-}
  local expected_python=${ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256:-}
  local expected_source=${ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256:-}
  local runtime_artifact
  for runtime_artifact in \
    "$runtime_build_receipt" "$runtime_manifest" "$runtime_project_wheel"; do
    if [[ -z "$runtime_artifact" || "$runtime_artifact" != /* \
       || ! -f "$runtime_artifact" || -L "$runtime_artifact" ]]; then
      echo "Goose publication runtime requires absolute regular attestation artifacts" >&2
      return 78
    fi
  done
  local runtime_digest
  for runtime_digest in \
    "$expected_build_receipt" "$expected_manifest" "$expected_project_wheel" \
    "$expected_native" "$expected_installation" "$expected_python" \
    "$expected_source"; do
    if [[ ! "$runtime_digest" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Goose publication runtime requires all external SHA-256 pins" >&2
      return 78
    fi
  done
  if [[ "$(publication_file_sha256 "$runtime_build_receipt")" \
       != "$expected_build_receipt" \
     || "$(publication_file_sha256 "$runtime_manifest")" != "$expected_manifest" \
     || "$(publication_file_sha256 "$runtime_project_wheel")" \
       != "$expected_project_wheel" ]]; then
    echo "Goose runtime attestation artifact bytes differ from external pins" >&2
    return 78
  fi
  local project_wheel_name
  project_wheel_name=$(basename -- "$runtime_project_wheel")
  if [[ "$project_wheel_name" != *.whl ]]; then
    echo "ISINGFOLD_RUNTIME_PROJECT_WHEEL must name a wheel" >&2
    return 78
  fi

  local scripts_root
  local source_root
  local requested_output_base=${ISINGFOLD_GOOSE_OUTPUT_BASE:-}
  local output_base
  scripts_root=$(cd -- "$requested_script_dir" && pwd -P)
  source_root=$(cd -- "$scripts_root/.." && pwd -P)
  if [[ -z "$requested_output_base" || "$requested_output_base" != /* \
     || "$requested_output_base" == "/" ]]; then
    echo "ISINGFOLD_GOOSE_OUTPUT_BASE must be an absolute dedicated output directory" >&2
    return 78
  fi
  if [[ ! -d "$requested_output_base" || ! -w "$requested_output_base" ]]; then
    echo "ISINGFOLD_GOOSE_OUTPUT_BASE must exist and be writable before sbatch" >&2
    return 78
  fi
  output_base=$(cd -- "$requested_output_base" && pwd -P)
  if [[ "$requested_output_base" != "$output_base" ]]; then
    echo "ISINGFOLD_GOOSE_OUTPUT_BASE must be a canonical physical path" >&2
    return 78
  fi
  case "$source_root/" in
    "$output_base/"*)
      echo "ISINGFOLD_GOOSE_OUTPUT_BASE cannot contain the source root" >&2
      return 78
      ;;
  esac
  case "$output_base/" in
    "$source_root/runs/"*) ;;
    "$source_root/"*)
      echo "an in-repository ISINGFOLD_GOOSE_OUTPUT_BASE must be under runs/" >&2
      return 78
      ;;
  esac
  case "$container_python/" in
    "$source_root/"*|"$output_base/"*)
      echo "host bind roots cannot shadow ISINGFOLD_CONTAINER_PYTHON" >&2
      return 78
      ;;
  esac

  pinned_container_runtime=/opt/isingfold-publication-pins/apptainer
  pinned_container_image=/opt/isingfold-publication-pins/environment.sif
  local -a binds
  binds=(
    "$source_root:$source_root:ro"
    "$output_base:$output_base:ro"
    "$apptainer_bin:$pinned_container_runtime:ro"
    "$container_image:$pinned_container_image:ro"
  )
  local requested_write_root
  local physical_write_root
  local relative_write_root
  local path_cursor
  local path_segment
  local -a write_segments
  for requested_write_root in "$@"; do
    if [[ -z "$requested_write_root" ]]; then
      echo "Goose publication output roots must be nonempty" >&2
      return 78
    fi
    if [[ "$requested_write_root" != /* ]]; then
      requested_write_root="$source_root/$requested_write_root"
    fi
    requested_write_root=${requested_write_root%/}
    case "$requested_write_root/" in
      "$output_base/"*) ;;
      *)
        echo "Goose publication output roots must be inside ISINGFOLD_GOOSE_OUTPUT_BASE" >&2
        return 78
        ;;
    esac
    if [[ "$requested_write_root" == "$output_base" ]]; then
      echo "a Goose publication output root must be narrower than its read-only base" >&2
      return 78
    fi
    relative_write_root=${requested_write_root#"$output_base"/}
    path_cursor=$output_base
    IFS=/ read -r -a write_segments <<< "$relative_write_root"
    for path_segment in "${write_segments[@]}"; do
      if [[ -z "$path_segment" || "$path_segment" == "." \
         || "$path_segment" == ".." ]]; then
        echo "Goose publication output roots must use canonical path components" >&2
        return 78
      fi
      path_cursor="$path_cursor/$path_segment"
      if [[ -L "$path_cursor" ]]; then
        echo "Goose publication output roots cannot traverse symbolic links" >&2
        return 78
      fi
      if [[ -e "$path_cursor" && ! -d "$path_cursor" ]]; then
        echo "Goose publication output roots can contain directories only" >&2
        return 78
      fi
    done
    mkdir -p -- "$requested_write_root"
    physical_write_root=$(cd -- "$requested_write_root" && pwd -P)
    if [[ "$requested_write_root" != "$physical_write_root" ]]; then
      echo "Goose publication output roots must resolve to their canonical paths" >&2
      return 78
    fi
    case "$physical_write_root/" in
      "$output_base/"*) ;;
      *)
        echo "Goose publication output root escaped its approved base" >&2
        return 78
        ;;
    esac
    binds+=("$physical_write_root:$physical_write_root:rw")
  done

  local requested_read_root
  local physical_read_root
  local input_roots=${ISINGFOLD_GOOSE_INPUT_ROOTS:-}
  if [[ -n "$input_roots" ]]; then
    local -a split_input_roots
    IFS=: read -r -a split_input_roots <<< "$input_roots"
    for requested_read_root in "${split_input_roots[@]}"; do
      if [[ -z "$requested_read_root" || "$requested_read_root" != /* ]]; then
        echo "ISINGFOLD_GOOSE_INPUT_ROOTS entries must be absolute existing paths" >&2
        return 78
      fi
      case "$container_python/" in
        "$requested_read_root/"*)
          echo "Goose input roots cannot shadow ISINGFOLD_CONTAINER_PYTHON" >&2
          return 78
          ;;
      esac
      if [[ ! -e "$requested_read_root" ]]; then
        echo "ISINGFOLD_GOOSE_INPUT_ROOTS entries must be absolute existing paths" >&2
        return 78
      fi
      physical_read_root=$(cd -- "$(dirname -- "$requested_read_root")" && pwd -P)/$(basename -- "$requested_read_root")
      case "$container_python/" in
        "$physical_read_root/"*)
          echo "Goose input roots cannot shadow ISINGFOLD_CONTAINER_PYTHON" >&2
          return 78
          ;;
      esac
      binds+=("$physical_read_root:$physical_read_root:ro")
    done
  fi

  local -a slurm_environment
  slurm_environment=(
    --env "SLURM_JOB_ID=$SLURM_JOB_ID"
    --env "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
  )
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    slurm_environment+=(--env "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID")
  fi
  if [[ -n "${SLURMD_NODENAME:-}" ]]; then
    slurm_environment+=(--env "SLURMD_NODENAME=$SLURMD_NODENAME")
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    slurm_environment+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  fi
  local -a runtime_environment
  runtime_environment=(
    --env "ISINGFOLD_RUNTIME_BUILD_RECEIPT=/opt/isingfold/runtime/build-receipt.json"
    --env "ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256=$expected_build_receipt"
    --env "ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256=$expected_source"
    --env "ISINGFOLD_RUNTIME_WHEELHOUSE_MANIFEST=/opt/isingfold/runtime/wheelhouse.manifest.json"
    --env "ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256=$expected_manifest"
    --env "ISINGFOLD_RUNTIME_PROJECT_WHEEL=/opt/isingfold/runtime/$project_wheel_name"
    --env "ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256=$expected_project_wheel"
    --env "ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256=$expected_native"
    --env "ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256=$expected_installation"
    --env "ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256=$expected_python"
    --env "ISINGFOLD_RUNTIME_EXECUTION_MODE=apptainer"
    --env "ISINGFOLD_RUNTIME_IMAGE=$pinned_container_image"
    --env "ISINGFOLD_RUNTIME_EXPECTED_IMAGE_SHA256=$container_image_sha256"
  )
  if [[ "$accelerator" == "gpu" ]]; then
    runtime_environment+=(--env "ISINGFOLD_DEVICE=cuda")
  fi

  unset LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH PYTHONUSERBASE
  local variable_name
  while IFS='=' read -r variable_name _; do
    case "$variable_name" in
      APPTAINER_*|APPTAINERENV_*|SINGULARITY_*|SINGULARITYENV_*)
        unset "$variable_name"
        ;;
    esac
  done < <(env)

  local bind_spec
  bind_spec=$(IFS=,; echo "${binds[*]}")
  container_exec=(
    "$apptainer_bin" exec --cleanenv --containall --no-home
    --no-mount bind-paths,hostfs
  )
  if [[ "$accelerator" == "gpu" ]]; then
    container_exec+=(--nv)
  fi
  container_exec+=(
    --bind "$bind_spec"
    --pwd "$source_root"
    "${slurm_environment[@]}"
    "${runtime_environment[@]}"
    "$container_image"
  )
}

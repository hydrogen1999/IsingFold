#!/usr/bin/env bash
# Read-only launch guard shared by the Apollo and Goose corpus launchers.
#
# Before staging, record these independent commitments outside the run tree:
#   scripts/verify_isingfold_corpus_runtime.sh file-sha256 PLAN.json
#   scripts/verify_isingfold_corpus_runtime.sh source-sha256 EMBEDBENCH_ROOT PYTHON
#   scripts/verify_isingfold_corpus_runtime.sh generation-provenance EMBEDBENCH_ROOT PYTHON 1
# The launchers require those recorded values; they never trust digests embedded in
# the plan or generated output.

set -euo pipefail

corpus_error() {
  local message=$1
  local status=${2:-78}
  echo "IsingFold corpus launch guard: $message" >&2
  return "$status"
}

corpus_require_sha256() {
  local value=$1
  local name=$2
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    corpus_error "$name must be one lowercase SHA-256 digest" 64
    return
  fi
}

corpus_require_pinned_threads() {
  local value=$1
  if [[ "$value" != "1" ]]; then
    corpus_error "the corpus runtime contract requires exactly one worker thread" 64
    return
  fi
}

corpus_file_sha256() {
  local path=$1
  if [[ ! -f "$path" || -L "$path" ]]; then
    corpus_error "cannot hash non-regular or symbolic-link file: $path" 78
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -- "$path" | awk '{print $1}'
  else
    corpus_error "no SHA-256 command is available" 78
  fi
}

corpus_executable_sha256() {
  local path=$1
  if [[ "$path" != /* || ! -f "$path" || ! -x "$path" ]]; then
    corpus_error "cannot hash a missing or non-executable program: $path" 78
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -- "$path" | awk '{print $1}'
  else
    corpus_error "no SHA-256 command is available" 78
  fi
}

corpus_require_canonical_directory() {
  local path=$1
  local name=$2
  local physical
  if [[ "$path" != /* || "$path" == "/" || ! -d "$path" || -L "$path" ]]; then
    corpus_error "$name must be an absolute, existing, non-root directory" 78
    return
  fi
  physical=$(cd -- "$path" && pwd -P)
  if [[ "${path%/}" != "$physical" ]]; then
    corpus_error "$name must be a canonical physical path without symbolic links" 78
    return
  fi
}

corpus_require_canonical_file() {
  local path=$1
  local name=$2
  local physical
  if [[ "$path" != /* || ! -f "$path" || -L "$path" ]]; then
    corpus_error "$name must be an absolute regular non-symbolic-link file" 78
    return
  fi
  physical=$(cd -- "$(dirname -- "$path")" && pwd -P)/$(basename -- "$path")
  if [[ "$path" != "$physical" ]]; then
    corpus_error "$name must use a canonical physical parent path" 78
    return
  fi
}

corpus_require_file_commitment() {
  local file_name=$1
  local expected_sha256=$2
  local name=$3
  local observed_sha256
  corpus_require_canonical_file "$file_name" "$name" || return
  corpus_require_sha256 "$expected_sha256" "$name SHA-256" || return
  observed_sha256=$(corpus_file_sha256 "$file_name") || return
  if [[ "$observed_sha256" != "$expected_sha256" ]]; then
    corpus_error "$name bytes differ from their external commitment" 78
    return
  fi
}

corpus_require_disjoint_roots() {
  local first=$1
  local first_name=$2
  local second=$3
  local second_name=$4
  case "$first/" in
    "$second/"*)
      corpus_error "$first_name cannot be inside $second_name" 78
      return
      ;;
  esac
  case "$second/" in
    "$first/"*)
      corpus_error "$second_name cannot be inside $first_name" 78
      return
      ;;
  esac
}

corpus_prepare_output_root() {
  local output_base=$1
  local output_root=$2
  local relative
  local cursor
  local segment
  local physical
  local -a segments

  corpus_require_canonical_directory \
    "$output_base" "ISINGFOLD_CORPUS_OUTPUT_BASE" || return
  if [[ "$output_root" != /* || "$output_root" == "$output_base" ]]; then
    corpus_error "ISINGFOLD_CORPUS_SHARD_ROOT must be an absolute strict child of ISINGFOLD_CORPUS_OUTPUT_BASE" 78
    return
  fi
  case "$output_root/" in
    "$output_base/"*) ;;
    *)
      corpus_error "ISINGFOLD_CORPUS_SHARD_ROOT escaped ISINGFOLD_CORPUS_OUTPUT_BASE" 78
      return
      ;;
  esac

  relative=${output_root#"$output_base"/}
  cursor=$output_base
  IFS=/ read -r -a segments <<< "$relative"
  for segment in "${segments[@]}"; do
    if [[ -z "$segment" || "$segment" == "." || "$segment" == ".." ]]; then
      corpus_error "ISINGFOLD_CORPUS_SHARD_ROOT contains a non-canonical component" 78
      return
    fi
    cursor="$cursor/$segment"
    if [[ -L "$cursor" ]]; then
      corpus_error "ISINGFOLD_CORPUS_SHARD_ROOT cannot traverse symbolic links" 78
      return
    fi
    if [[ -e "$cursor" && ! -d "$cursor" ]]; then
      corpus_error "ISINGFOLD_CORPUS_SHARD_ROOT can contain directories only" 78
      return
    fi
  done
  mkdir -p -- "$output_root"
  physical=$(cd -- "$output_root" && pwd -P)
  if [[ "$physical" != "$output_root" ]]; then
    corpus_error "ISINGFOLD_CORPUS_SHARD_ROOT did not resolve to its declared path" 78
    return
  fi
}

corpus_map_shard_subset() {
  local shard_count_raw=$1
  local first_raw=$2
  local last_raw=$3
  local shard_step_raw=$4
  local array_task_id_raw=$5
  local array_count_raw=$6
  local array_min_raw=$7
  local array_max_raw=$8
  local array_step_raw=$9
  local raw
  local shard_count
  local first
  local last
  local shard_step
  local array_task_id
  local array_count
  local array_min
  local array_max
  local array_step
  local expected_array_tasks
  local index

  for raw in \
    "$shard_count_raw" "$first_raw" "$last_raw" "$shard_step_raw" \
    "$array_task_id_raw" "$array_count_raw" "$array_min_raw" \
    "$array_max_raw" "$array_step_raw"; do
    if [[ ! "$raw" =~ ^[0-9]+$ ]]; then
      corpus_error "shard subset and Slurm array controls must be decimal integers" 64
      return
    fi
  done
  shard_count=$((10#$shard_count_raw))
  first=$((10#$first_raw))
  last=$((10#$last_raw))
  shard_step=$((10#$shard_step_raw))
  array_task_id=$((10#$array_task_id_raw))
  array_count=$((10#$array_count_raw))
  array_min=$((10#$array_min_raw))
  array_max=$((10#$array_max_raw))
  array_step=$((10#$array_step_raw))

  if (( shard_count == 0 || shard_step == 0 || first > last || last >= shard_count )); then
    corpus_error "require 0 <= SHARD_FIRST <= SHARD_LAST < SHARD_COUNT and positive STEP" 64
    return
  fi
  if (( (last - first) % shard_step != 0 )); then
    corpus_error "SHARD_LAST must be reachable exactly from SHARD_FIRST by SHARD_STEP" 64
    return
  fi
  expected_array_tasks=$(((last - first) / shard_step + 1))
  if (( array_count != expected_array_tasks || array_min != 0 \
     || array_max != expected_array_tasks - 1 || array_step != 1 \
     || array_task_id >= array_count )); then
    corpus_error "Slurm array must be exactly 0..(subset-size-1) with scheduling step 1" 64
    return
  fi
  index=$((first + array_task_id * shard_step))
  if (( index > last || index >= shard_count )); then
    corpus_error "mapped scientific shard index lies outside the registered subset" 64
    return
  fi
  printf '%s\n' "$index"
}

corpus_validate_launch_mode() {
  local mode=$1
  local first=$2
  local last=$3
  local step=$4
  local task_count=$5
  local shard_root=$6
  local canary_root=$7
  case "$mode" in
    canary-only)
      if [[ ! "$first" =~ ^[0-9]+$ || ! "$last" =~ ^[0-9]+$ \
         || "$step" != "1" || "$task_count" != "1" || "$first" != "$last" ]]; then
        corpus_error "canary-only mode requires exactly one shard and one task" 64
        return
      fi
      if [[ "$canary_root" != /* || "$shard_root" != "$canary_root/shards" ]]; then
        corpus_error "canary-only mode requires SHARD_ROOT=CANARY_OUTPUT_ROOT/shards" 78
        return
      fi
      ;;
    production) ;;
    *)
      corpus_error "ISINGFOLD_CORPUS_MODE must be canary-only or production" 64
      return
      ;;
  esac
}

corpus_source_sha256() {
  local source_root=$1
  local python_bin=$2
  corpus_require_canonical_directory "$source_root" "EmbedBench source root" || return
  if [[ "$python_bin" != /* || ! -f "$python_bin" || ! -x "$python_bin" ]]; then
    corpus_error "source hashing requires an absolute executable Python" 78
    return
  fi
  "$python_bin" - "$source_root" <<'PY'
import hashlib
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    root / "pyproject.toml",
    root / "scripts" / "apollo_isingfold_corpus.sh",
    root / "scripts" / "goose_isingfold_corpus.sbatch",
    root / "scripts" / "verify_isingfold_corpus_runtime.sh",
]
src = root / "src"
if not src.is_dir():
    raise SystemExit("EmbedBench source root has no src directory")

members = list(required)
for current, directories, filenames in os.walk(src, followlinks=False):
    current_path = Path(current)
    kept = []
    for name in sorted(directories):
        candidate = current_path / name
        if candidate.is_symlink():
            raise SystemExit(f"source inventory contains directory symlink: {candidate}")
        if (
            name != "__pycache__"
            and not name.endswith(".egg-info")
            and not name.endswith(".dist-info")
        ):
            kept.append(name)
    directories[:] = kept
    for name in sorted(filenames):
        if name.endswith((".pyc", ".pyo")) or name == ".DS_Store":
            continue
        members.append(current_path / name)

framed = hashlib.sha256()
framed.update(b"embedbench-isingfold-corpus-source-v1\0")
for path in sorted(set(members), key=lambda item: item.relative_to(root).as_posix()):
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"source inventory member is not a regular file: {path}")
    relative = path.relative_to(root).as_posix().encode("utf-8")
    payload = path.read_bytes()
    framed.update(len(relative).to_bytes(8, "big"))
    framed.update(relative)
    framed.update(len(payload).to_bytes(8, "big"))
    framed.update(payload)
print(framed.hexdigest())
PY
}

corpus_generation_provenance() {
  local source_root=$1
  local python_bin=$2
  local thread_count=$3
  corpus_require_canonical_directory "$source_root" "EmbedBench source root" || return
  corpus_require_pinned_threads "$thread_count" || return
  if [[ "$python_bin" != /* || ! -f "$python_bin" || ! -x "$python_bin" ]]; then
    corpus_error "generation provenance requires an absolute executable Python" 78
    return
  fi
  env -u PYTHONHOME -u PYTHONUSERBASE -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 PYTHONPATH="$source_root/src" \
    MKL_NUM_THREADS="$thread_count" NUMEXPR_NUM_THREADS="$thread_count" \
    OMP_NUM_THREADS="$thread_count" OPENBLAS_NUM_THREADS="$thread_count" \
    "$python_bin" - "$source_root" <<'PY'
import json
import sys
from pathlib import Path

from embedbench.candidate_bank import content_digest
from embedbench.isingfold_corpus_shard import _generation_provenance

root = Path(sys.argv[1])
import embedbench.isingfold_corpus_shard as shard_module

expected = (root / "src" / "embedbench" / "isingfold_corpus_shard.py").resolve()
actual = Path(shard_module.__file__).resolve()
if actual != expected:
    raise SystemExit(f"imported corpus generator from {actual}, expected {expected}")
provenance = _generation_provenance()
document = {
    "provenance": provenance,
    "provenance_digest": content_digest(provenance),
}
print(json.dumps(document, allow_nan=False, separators=(",", ":"), sort_keys=True))
PY
}

corpus_generation_provenance_sha256() {
  local source_root=$1
  local python_bin=$2
  local thread_count=$3
  local document
  corpus_require_pinned_threads "$thread_count" || return
  document=$(
    corpus_generation_provenance "$source_root" "$python_bin" "$thread_count"
  ) || return
  printf '%s\n' "$document" \
    | "$python_bin" -c 'import json, sys; print(json.load(sys.stdin)["provenance_digest"])'
}

corpus_validate_external_installs() {
  local source_root=$1
  local python_bin=$2
  corpus_require_canonical_directory "$source_root" "EmbedBench source root" || return
  if [[ "$python_bin" != /* || ! -f "$python_bin" || ! -x "$python_bin" ]]; then
    corpus_error "external-install validation requires an absolute executable Python" 78
    return
  fi
  env -u PYTHONHOME -u PYTHONUSERBASE -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 PYTHONPATH="$source_root/src" \
    "$python_bin" - "$source_root" <<'PY'
import ast
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
generator_source = root / "src" / "embedbench" / "isingfold_corpus_shard.py"
tree = ast.parse(generator_source.read_text(encoding="utf-8"), filename=str(generator_source))
dependency_nodes = [
    node.value
    for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_DEPENDENCIES" for target in node.targets)
]
if len(dependency_nodes) != 1:
    raise SystemExit("generator source must contain one literal _DEPENDENCIES assignment")
dependencies = ast.literal_eval(dependency_nodes[0])
if (
    not isinstance(dependencies, tuple)
    or not dependencies
    or any(not isinstance(name, str) or not name for name in dependencies)
):
    raise SystemExit("generator _DEPENDENCIES must be one non-empty tuple of names")


def normalized(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


external = {normalized(name) for name in dependencies if name != "embedbench"}
violations = []
for name in sorted(external):
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        violations.append(f"{name}: missing distribution")
        continue
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is not None:
        document = json.loads(direct_url)
        violations.append(f"{name}: direct-URL install")
        if document.get("vcs_info") is not None:
            violations.append(f"{name}: VCS direct_url install")
        directory = document.get("dir_info")
        if isinstance(directory, dict) and directory.get("editable") is True:
            violations.append(f"{name}: editable direct_url install")
    for entry in distribution.files or ():
        filename = str(entry).lower()
        if filename.endswith(".egg-link") or "__editable__" in filename:
            violations.append(f"{name}: editable-install metadata {entry}")

freeze = subprocess.run(
    [sys.executable, "-m", "pip", "freeze", "--all"],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
).stdout.splitlines()
for line in freeze:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    if stripped.startswith("-e ") or stripped.startswith("--editable "):
        location = stripped.split(maxsplit=1)[1]
        if location.startswith(("git+", "hg+", "svn+", "bzr+")):
            violations.append(f"editable VCS requirement: {location}")
            continue
        if location.startswith("file://"):
            location = location[7:]
        try:
            editable_root = Path(location).resolve()
        except OSError:
            violations.append(f"unresolvable editable requirement: {location}")
        else:
            if editable_root != root:
                violations.append(f"external editable requirement: {editable_root}")
        continue
    if " @ " in stripped:
        project, location = stripped.split(" @ ", 1)
        if normalized(project) in external and location.startswith(
            ("git+", "hg+", "svn+", "bzr+")
        ):
            violations.append(f"{normalized(project)}: VCS requirement")

if violations:
    unique = "\n".join(f"  - {item}" for item in sorted(set(violations)))
    raise SystemExit(
        "production corpus generation forbids editable, VCS, or direct-URL "
        "external installs:\n" + unique
    )
PY
}

corpus_validate_generation_provenance() {
  local python_bin=$1
  local source_root=$2
  local expected_provenance_sha256=$3
  local thread_count=$4
  local observed
  corpus_require_sha256 \
    "$expected_provenance_sha256" \
    "ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" || return
  corpus_require_pinned_threads "$thread_count" || return
  observed=$(
    corpus_generation_provenance_sha256 "$source_root" "$python_bin" "$thread_count"
  ) || return
  if [[ "$observed" != "$expected_provenance_sha256" ]]; then
    echo "expected generation provenance: $expected_provenance_sha256" >&2
    echo "observed generation provenance: $observed" >&2
    corpus_error "Python/dependency/source provenance differs from the cross-site commitment" 78
    return
  fi
}

corpus_validate_source_and_plan() {
  local python_bin=$1
  local source_root=$2
  local expected_source_sha256=$3
  local plan=$4
  local expected_plan_sha256=$5
  local shard_count=$6
  local observed_source_sha256
  local observed_plan_sha256

  corpus_require_sha256 \
    "$expected_source_sha256" "ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" || return
  corpus_require_sha256 \
    "$expected_plan_sha256" "ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" || return
  corpus_require_canonical_directory \
    "$source_root" "ISINGFOLD_CORPUS_SOURCE_ROOT" || return
  corpus_require_canonical_file "$plan" "ISINGFOLD_CORPUS_PLAN" || return
  if [[ ! "$shard_count" =~ ^[1-9][0-9]*$ ]]; then
    corpus_error "ISINGFOLD_CORPUS_SHARD_COUNT must be a positive integer" 64
    return
  fi

  observed_plan_sha256=$(corpus_file_sha256 "$plan") || return
  if [[ "$observed_plan_sha256" != "$expected_plan_sha256" ]]; then
    corpus_error "plan bytes differ from the external plan commitment" 78
    return
  fi
  observed_source_sha256=$(corpus_source_sha256 "$source_root" "$python_bin") || return
  if [[ "$observed_source_sha256" != "$expected_source_sha256" ]]; then
    echo "expected source SHA-256: $expected_source_sha256" >&2
    echo "observed source SHA-256: $observed_source_sha256" >&2
    corpus_error "staged source differs from the external source commitment" 78
    return
  fi

  env -u PYTHONHOME -u PYTHONUSERBASE -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 PYTHONPATH="$source_root/src" \
    "$python_bin" - "$source_root" "$plan" "$expected_plan_sha256" "$shard_count" <<'PY'
import sys
from pathlib import Path

from embedbench.isingfold_corpus_plan import read_corpus_plan
import embedbench.isingfold_corpus_cli as cli

root = Path(sys.argv[1])
expected_module = (root / "src" / "embedbench" / "isingfold_corpus_cli.py").resolve()
actual_module = Path(cli.__file__).resolve()
if actual_module != expected_module:
    raise SystemExit(
        f"configured Python imported corpus CLI from {actual_module}, expected {expected_module}"
    )
plan = read_corpus_plan(Path(sys.argv[2]), expected_sha256=sys.argv[3])
expected_count = int(sys.argv[4])
if plan.shard_count != expected_count:
    raise SystemExit(
        f"plan shard_count={plan.shard_count}, launcher shard_count={expected_count}"
    )
PY
}

corpus_validate_apollo_runtime() {
  local python_bin=$1
  local expected_python_sha256=$2
  local environment_lock=$3
  local expected_environment_sha256=$4
  local source_root=$5
  local expected_source_sha256=$6
  local plan=$7
  local expected_plan_sha256=$8
  local shard_count=$9
  local expected_provenance_sha256=${10}
  local thread_count=${11}
  local runtime_lock=${12}
  local expected_runtime_lock_sha256=${13}
  local runtime_validator=${14}
  local expected_validator_sha256=${15}
  local expected_installation_sha256=${16}
  local python_parent
  local virtualenv_root
  local observed

  if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    corpus_error "Apollo execution rejects every Slurm job or array environment" 69
    return
  fi
  if [[ "$python_bin" != /* || ! -f "$python_bin" || ! -x "$python_bin" ]]; then
    corpus_error "ISINGFOLD_CORPUS_PYTHON must be an absolute executable" 78
    return
  fi
  corpus_require_sha256 \
    "$expected_python_sha256" "ISINGFOLD_CORPUS_EXPECTED_PYTHON_SHA256" || return
  observed=$(corpus_executable_sha256 "$python_bin") || return
  if [[ "$observed" != "$expected_python_sha256" ]]; then
    corpus_error "Python executable differs from ISINGFOLD_CORPUS_EXPECTED_PYTHON_SHA256" 78
    return
  fi
  python_parent=$(cd -- "$(dirname -- "$python_bin")" && pwd -P)
  virtualenv_root=$(cd -- "$python_parent/.." && pwd -P)
  if [[ ! -f "$virtualenv_root/pyvenv.cfg" || -L "$virtualenv_root/pyvenv.cfg" ]]; then
    corpus_error "ISINGFOLD_CORPUS_PYTHON must belong to a virtual environment" 78
    return
  fi
  corpus_require_canonical_file "$environment_lock" "ISINGFOLD_CORPUS_ENV_LOCK" || return
  corpus_require_sha256 \
    "$expected_environment_sha256" "ISINGFOLD_CORPUS_EXPECTED_ENV_LOCK_SHA256" || return
  observed=$(corpus_file_sha256 "$environment_lock") || return
  if [[ "$observed" != "$expected_environment_sha256" ]]; then
    corpus_error "environment lock differs from its external commitment" 78
    return
  fi

  corpus_validate_source_and_plan \
    "$python_bin" "$source_root" "$expected_source_sha256" \
    "$plan" "$expected_plan_sha256" "$shard_count" || return
  corpus_validate_generation_provenance \
    "$python_bin" "$source_root" "$expected_provenance_sha256" \
    "$thread_count" || return
  corpus_validate_external_installs "$source_root" "$python_bin" || return
  env -u PYTHONHOME -u PYTHONUSERBASE -u PYTHONPATH \
    PYTHONNOUSERSITE=1 "$python_bin" -m pip check || return
  corpus_require_file_commitment \
    "$runtime_lock" "$expected_runtime_lock_sha256" "runtime lock" || return
  corpus_require_file_commitment \
    "$runtime_validator" "$expected_validator_sha256" "runtime validator" || return
  corpus_require_sha256 \
    "$expected_installation_sha256" \
    "ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256" || return
  env -u PYTHONHOME -u PYTHONUSERBASE -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 PYTHONPATH="$source_root/src" \
    MKL_NUM_THREADS="$thread_count" NUMEXPR_NUM_THREADS="$thread_count" \
    OMP_NUM_THREADS="$thread_count" OPENBLAS_NUM_THREADS="$thread_count" \
    "$python_bin" "$runtime_validator" \
      --runtime-lock "$runtime_lock" \
      --source-root "$source_root" \
      --expected-installation-sha256 "$expected_installation_sha256" \
      --expected-provenance-sha256 "$expected_provenance_sha256" \
      --expected-python-executable "$python_bin" \
      >/dev/null
}

corpus_validate_preflight() {
  local python_bin=$1
  local source_root=$2
  local expected_source_sha256=$3
  local plan=$4
  local expected_plan_sha256=$5
  local preflight=$6
  local expected_preflight_sha256=$7
  local expected_preflight_record_digest=$8
  local expected_identity_map_digest=$9
  local expected_provenance_sha256=${10}
  local workers=${11}

  corpus_require_canonical_directory \
    "$source_root" "ISINGFOLD_CORPUS_SOURCE_ROOT" || return
  corpus_require_canonical_file "$plan" "ISINGFOLD_CORPUS_PLAN" || return
  corpus_require_file_commitment \
    "$preflight" "$expected_preflight_sha256" "prospective preflight" || return
  corpus_require_sha256 \
    "$expected_source_sha256" "ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" || return
  corpus_require_sha256 \
    "$expected_plan_sha256" "ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" || return
  corpus_require_sha256 \
    "$expected_preflight_record_digest" \
    "ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST" || return
  corpus_require_sha256 \
    "$expected_identity_map_digest" \
    "ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST" || return
  corpus_require_sha256 \
    "$expected_provenance_sha256" \
    "ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" || return
  if [[ ! "$workers" =~ ^[1-9][0-9]*$ || 10#$workers -gt 64 ]]; then
    corpus_error "preflight replay workers must be an integer in [1, 64]" 64
    return
  fi
  env -u PYTHONHOME -u PYTHONUSERBASE -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 PYTHONPATH="$source_root/src" \
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$python_bin" -m embedbench.isingfold_corpus_cli verify-preflight \
      --plan "$plan" \
      --expected-plan-sha256 "$expected_plan_sha256" \
      --expected-source-sha256 "$expected_source_sha256" \
      --preflight "$preflight" \
      --expected-preflight-sha256 "$expected_preflight_sha256" \
      --expected-preflight-record-digest "$expected_preflight_record_digest" \
      --expected-identity-map-digest "$expected_identity_map_digest" \
      --expected-generation-provenance-digest "$expected_provenance_sha256" \
      --workers "$workers" \
      >/dev/null
}

corpus_validate_receipt() {
  local python_bin=$1
  local receipt=$2
  local shard_index=$3
  local shard_count=$4
  local output_directory=$5
  "$python_bin" - "$receipt" "$shard_index" "$shard_count" "$output_directory" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as stream:
    receipt = json.load(stream, parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON value: {token}")
    ))
if not isinstance(receipt, dict):
    raise SystemExit("corpus CLI receipt is not a JSON object")
expected = {
    "shard_index": int(sys.argv[2]),
    "shard_count": int(sys.argv[3]),
    "output_directory": str(Path(sys.argv[4]).resolve()),
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(f"corpus CLI receipt {key}={receipt.get(key)!r}, expected {value!r}")
if not isinstance(receipt.get("lineage_count"), int) or receipt["lineage_count"] <= 0:
    raise SystemExit("corpus CLI receipt has no positive lineage_count")
PY
}

corpus_helper_main() {
  if [[ $# -lt 1 ]]; then
    echo "usage: $0 file-sha256 FILE | executable-sha256 PROGRAM | source-sha256 ROOT PYTHON | generation-provenance ROOT PYTHON THREADS | provenance-sha256 ROOT PYTHON THREADS | validate-externals ROOT PYTHON | map-subset SHARDS FIRST LAST STEP TASK COUNT MIN MAX ARRAY_STEP | validate-mode MODE FIRST LAST STEP TASKS SHARD_ROOT CANARY_ROOT_OR_DASH | validate-source-plan PYTHON ROOT SOURCE_SHA PLAN PLAN_SHA SHARDS | validate-provenance PYTHON ROOT PROVENANCE_SHA THREADS | validate-preflight PYTHON ROOT SOURCE_SHA PLAN PLAN_SHA PREFLIGHT PREFLIGHT_SHA PREFLIGHT_RECORD IDENTITY_MAP PROVENANCE WORKERS | validate-receipt PYTHON RECEIPT INDEX SHARDS OUT" >&2
    return 64
  fi
  local command=$1
  shift
  case "$command" in
    file-sha256)
      if [[ $# -ne 1 ]]; then
        corpus_error "file-sha256 requires FILE" 64
        return
      fi
      corpus_file_sha256 "$1"
      ;;
    executable-sha256)
      if [[ $# -ne 1 ]]; then
        corpus_error "executable-sha256 requires PROGRAM" 64
        return
      fi
      corpus_executable_sha256 "$1"
      ;;
    source-sha256)
      if [[ $# -ne 2 ]]; then
        corpus_error "source-sha256 requires EMBEDBENCH_ROOT PYTHON" 64
        return
      fi
      corpus_source_sha256 "$1" "$2"
      ;;
    generation-provenance)
      if [[ $# -ne 3 ]]; then
        corpus_error "generation-provenance requires EMBEDBENCH_ROOT PYTHON THREADS" 64
        return
      fi
      corpus_generation_provenance "$1" "$2" "$3"
      ;;
    provenance-sha256)
      if [[ $# -ne 3 ]]; then
        corpus_error "provenance-sha256 requires EMBEDBENCH_ROOT PYTHON THREADS" 64
        return
      fi
      corpus_generation_provenance_sha256 "$1" "$2" "$3"
      ;;
    validate-externals)
      if [[ $# -ne 2 ]]; then
        corpus_error "validate-externals requires EMBEDBENCH_ROOT PYTHON" 64
        return
      fi
      corpus_validate_external_installs "$1" "$2"
      ;;
    map-subset)
      if [[ $# -ne 9 ]]; then
        corpus_error "map-subset requires SHARDS FIRST LAST STEP TASK COUNT MIN MAX ARRAY_STEP" 64
        return
      fi
      corpus_map_shard_subset "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"
      ;;
    validate-mode)
      if [[ $# -ne 7 ]]; then
        corpus_error "validate-mode requires MODE FIRST LAST STEP TASKS SHARD_ROOT CANARY_ROOT_OR_DASH" 64
        return
      fi
      corpus_validate_launch_mode "$1" "$2" "$3" "$4" "$5" "$6" "$7"
      ;;
    validate-source-plan)
      if [[ $# -ne 6 ]]; then
        corpus_error "validate-source-plan requires PYTHON ROOT SOURCE_SHA PLAN PLAN_SHA SHARDS" 64
        return
      fi
      corpus_validate_source_and_plan "$1" "$2" "$3" "$4" "$5" "$6"
      ;;
    validate-provenance)
      if [[ $# -ne 4 ]]; then
        corpus_error "validate-provenance requires PYTHON ROOT PROVENANCE_SHA THREADS" 64
        return
      fi
      corpus_validate_generation_provenance "$1" "$2" "$3" "$4"
      ;;
    validate-preflight)
      if [[ $# -ne 11 ]]; then
        corpus_error "validate-preflight requires PYTHON ROOT SOURCE_SHA PLAN PLAN_SHA PREFLIGHT PREFLIGHT_SHA PREFLIGHT_RECORD IDENTITY_MAP PROVENANCE WORKERS" 64
        return
      fi
      corpus_validate_preflight \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}"
      ;;
    validate-receipt)
      if [[ $# -ne 5 ]]; then
        corpus_error "validate-receipt requires PYTHON RECEIPT INDEX SHARDS OUT" 64
        return
      fi
      corpus_validate_receipt "$1" "$2" "$3" "$4" "$5"
      ;;
    *)
      corpus_error "unknown helper command: $command" 64
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  corpus_helper_main "$@"
fi

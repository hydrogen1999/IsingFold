#!/usr/bin/env bash
# Generate a pinned IsingFold production-corpus shard range directly on Apollo.
# Apollo has no Slurm; this launcher rejects a Slurm environment.
#
# Workflow (the plan digest must be recorded somewhere outside the run tree):
#   1. PYTHON -m embedbench.isingfold_corpus_cli plan --shard-count 64 --out PLAN
#   2. Record plan, source, and generation-provenance commitments with the verifier.
#   3. In the frozen runtime, publish preflight v2 once, record its file, record,
#      and identity-map digests externally, and replay all 3,082 identities.
#   4. First use MODE=canary-only, FIRST=LAST, STEP=1, CONCURRENCY=1, and a
#      dedicated CANARY_OUTPUT_ROOT. Generate that same shard on Goose.
#   5. Compare the copied read-only canaries with `python -m
#      embedbench.isingfold_cross_site_canary publish` and record its file SHA.
#   6. Only then use MODE=production with the pinned parity attestation. For a
#      two-site split, Apollo may use FIRST=0 LAST=62 STEP=2 while Goose uses odds.
#   7. After every shard exists, run the CLI `merge` command against SHARD_ROOT.
#
# Required controls: SOURCE_ROOT, PYTHON, EXPECTED_PYTHON_SHA256, ENV_LOCK,
# EXPECTED_ENV_LOCK_SHA256, EXPECTED_RUNTIME_HELPER_SHA256,
# EXPECTED_RUNTIME_LOCK_SHA256, EXPECTED_VALIDATOR_SHA256,
# EXPECTED_INSTALLATION_SHA256, EXPECTED_SOURCE_SHA256,
# EXPECTED_PROVENANCE_SHA256, PLAN, EXPECTED_PLAN_SHA256, PREFLIGHT,
# EXPECTED_PREFLIGHT_SHA256, EXPECTED_PREFLIGHT_RECORD_DIGEST,
# EXPECTED_IDENTITY_MAP_DIGEST, OUTPUT_BASE, SHARD_ROOT,
# SHARD_COUNT, SHARD_FIRST, SHARD_LAST, SHARD_STEP, CONCURRENCY, and THREADS (all
# prefixed ISINGFOLD_CORPUS_). Existing shard, claim, log, or receipt paths are
# fatal.
# External generator dependencies, including minorminer, must be immutable wheel
# installs; editable, VCS, and direct-URL installs are rejected before any shard
# starts.

set -euo pipefail
umask 077

if [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "apollo_isingfold_corpus.sh must run directly; Apollo has no Slurm" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS_SOURCE_ROOT:?set an absolute staged EmbedBench root}"
: "${ISINGFOLD_CORPUS_PYTHON:?set an absolute virtualenv Python}"
: "${ISINGFOLD_CORPUS_EXPECTED_PYTHON_SHA256:?pin the Python executable SHA-256}"
: "${ISINGFOLD_CORPUS_ENV_LOCK:?set an absolute environment lock path}"
: "${ISINGFOLD_CORPUS_EXPECTED_ENV_LOCK_SHA256:?pin the environment lock SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256:?pin the runtime helper SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256:?pin runtime-lock.json SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256:?pin the runtime validator SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256:?pin the cross-site installation inventory digest}"
: "${ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256:?pin the staged source inventory SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256:?pin the cross-site generation provenance digest}"
: "${ISINGFOLD_CORPUS_PLAN:?set the absolute prospective plan path}"
: "${ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256:?supply the independently recorded plan SHA-256}"
: "${ISINGFOLD_CORPUS_PREFLIGHT:?set the absolute prospective preflight path}"
: "${ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256:?pin the prospective preflight file SHA-256}"
: "${ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST:?pin the preflight record digest}"
: "${ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST:?pin the prospective identity map digest}"
: "${ISINGFOLD_CORPUS_OUTPUT_BASE:?set an existing dedicated output base}"
: "${ISINGFOLD_CORPUS_SHARD_ROOT:?set a new or resumable shard root under OUTPUT_BASE}"
: "${ISINGFOLD_CORPUS_SHARD_COUNT:?set the plan shard count}"
: "${ISINGFOLD_CORPUS_SHARD_FIRST:?set the first inclusive shard index}"
: "${ISINGFOLD_CORPUS_SHARD_LAST:?set the last inclusive shard index}"
: "${ISINGFOLD_CORPUS_SHARD_STEP:?set a positive shard stride}"
: "${ISINGFOLD_CORPUS_CONCURRENCY:?set the bounded Apollo worker count}"
: "${ISINGFOLD_CORPUS_THREADS:?set threads per worker}"
: "${ISINGFOLD_CORPUS_MODE:?set canary-only or production}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root_from_launcher=$(cd -- "$script_dir/.." && pwd -P)
runtime_helper="$script_dir/verify_isingfold_corpus_runtime.sh"
runtime_lock="$source_root_from_launcher/runtime/isingfold-corpus/runtime-lock.json"
runtime_validator="$script_dir/validate_isingfold_corpus_image.py"
if [[ ! "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" =~ ^[0-9a-f]{64}$ \
   || ! -f "$runtime_helper" || -L "$runtime_helper" ]]; then
  echo "the runtime helper or its external SHA-256 commitment is invalid" >&2
  exit 78
fi
if command -v sha256sum >/dev/null 2>&1; then
  observed_runtime_helper_sha256=$(sha256sum -- "$runtime_helper" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  observed_runtime_helper_sha256=$(shasum -a 256 -- "$runtime_helper" | awk '{print $1}')
else
  echo "a SHA-256 command is required before loading the runtime helper" >&2
  exit 78
fi
if [[ "$observed_runtime_helper_sha256" != "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" ]]; then
  echo "runtime helper differs from its external SHA-256 commitment" >&2
  exit 78
fi
# shellcheck source=./scripts/verify_isingfold_corpus_runtime.sh
source "$runtime_helper"

if [[ "$ISINGFOLD_CORPUS_SOURCE_ROOT" != "$source_root_from_launcher" ]]; then
  corpus_error "launcher is not inside ISINGFOLD_CORPUS_SOURCE_ROOT" 78
fi
corpus_require_disjoint_roots \
  "$ISINGFOLD_CORPUS_SOURCE_ROOT" "ISINGFOLD_CORPUS_SOURCE_ROOT" \
  "$ISINGFOLD_CORPUS_OUTPUT_BASE" "ISINGFOLD_CORPUS_OUTPUT_BASE"

for integer_value in \
  "$ISINGFOLD_CORPUS_SHARD_COUNT" \
  "$ISINGFOLD_CORPUS_SHARD_FIRST" \
  "$ISINGFOLD_CORPUS_SHARD_LAST" \
  "$ISINGFOLD_CORPUS_SHARD_STEP" \
  "$ISINGFOLD_CORPUS_CONCURRENCY" \
  "$ISINGFOLD_CORPUS_THREADS"; do
  if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
    corpus_error "shard controls, CONCURRENCY, and THREADS must be decimal integers" 64
  fi
done

shard_count=$((10#$ISINGFOLD_CORPUS_SHARD_COUNT))
first=$((10#$ISINGFOLD_CORPUS_SHARD_FIRST))
last=$((10#$ISINGFOLD_CORPUS_SHARD_LAST))
step=$((10#$ISINGFOLD_CORPUS_SHARD_STEP))
concurrency=$((10#$ISINGFOLD_CORPUS_CONCURRENCY))
threads=$((10#$ISINGFOLD_CORPUS_THREADS))
if (( shard_count == 0 || step == 0 || concurrency == 0 || threads == 0 )); then
  corpus_error "SHARD_COUNT, SHARD_STEP, CONCURRENCY, and THREADS must be positive" 64
fi
if (( threads != 1 )); then
  corpus_error "the pinned corpus runtime requires THREADS=1" 64
fi
if (( first > last || last >= shard_count )); then
  corpus_error "require 0 <= SHARD_FIRST <= SHARD_LAST < SHARD_COUNT" 64
fi

case "$ISINGFOLD_CORPUS_MODE" in
  canary-only)
    : "${ISINGFOLD_CORPUS_CANARY_OUTPUT_ROOT:?set the dedicated Apollo canary root}"
    corpus_validate_launch_mode \
      "$ISINGFOLD_CORPUS_MODE" "$first" "$last" "$step" "$concurrency" \
      "$ISINGFOLD_CORPUS_SHARD_ROOT" "$ISINGFOLD_CORPUS_CANARY_OUTPUT_ROOT"
    if (( first != last || step != 1 || concurrency != 1 )); then
      corpus_error "canary-only mode requires one shard, SHARD_STEP=1, and CONCURRENCY=1" 64
    fi
    if [[ "$ISINGFOLD_CORPUS_SHARD_ROOT" != "$ISINGFOLD_CORPUS_CANARY_OUTPUT_ROOT/shards" ]]; then
      corpus_error "canary-only SHARD_ROOT must be CANARY_OUTPUT_ROOT/shards" 78
    fi
    parity_attestation_sha256=not-applicable-canary-only
    ;;
  production)
    corpus_validate_launch_mode \
      "$ISINGFOLD_CORPUS_MODE" "$first" "$last" "$step" "$concurrency" \
      "$ISINGFOLD_CORPUS_SHARD_ROOT" -
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

corpus_validate_apollo_runtime \
  "$ISINGFOLD_CORPUS_PYTHON" \
  "$ISINGFOLD_CORPUS_EXPECTED_PYTHON_SHA256" \
  "$ISINGFOLD_CORPUS_ENV_LOCK" \
  "$ISINGFOLD_CORPUS_EXPECTED_ENV_LOCK_SHA256" \
  "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
  "$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" \
  "$ISINGFOLD_CORPUS_PLAN" \
  "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" \
  "$shard_count" \
  "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" \
  "$threads" \
  "$runtime_lock" \
  "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256" \
  "$runtime_validator" \
  "$ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256" \
  "$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256"

corpus_validate_preflight \
  "$ISINGFOLD_CORPUS_PYTHON" \
  "$ISINGFOLD_CORPUS_SOURCE_ROOT" \
  "$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" \
  "$ISINGFOLD_CORPUS_PLAN" \
  "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" \
  "$ISINGFOLD_CORPUS_PREFLIGHT" \
  "$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256" \
  "$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST" \
  "$ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST" \
  "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" \
  10

if [[ "$ISINGFOLD_CORPUS_MODE" == "production" ]]; then
  env -u PYTHONHOME -u PYTHONUSERBASE -u PYTHONPATH \
    PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 \
    PYTHONPATH="$ISINGFOLD_CORPUS_SOURCE_ROOT/src" \
    MKL_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads" \
    OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" \
    "$ISINGFOLD_CORPUS_PYTHON" -m embedbench.isingfold_cross_site_canary verify \
      --attestation "$ISINGFOLD_CORPUS_PARITY_ATTESTATION" \
      --expected-attestation-sha256 \
        "$ISINGFOLD_CORPUS_EXPECTED_PARITY_ATTESTATION_SHA256" \
      --plan "$ISINGFOLD_CORPUS_PLAN" \
      --plan-sha256 "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" \
      --source-sha256 "$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256" \
      --generation-provenance-sha256 \
        "$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256" \
      --installation-sha256 "$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256" \
      >/dev/null
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

declare -a selected_indices=()
for ((index = first; index <= last; index += step)); do
  selected_indices+=("$index")
done

shard_name_for() {
  local index=$1
  printf 'shard-%04d-of-%04d' "$index" "$shard_count"
}

# Refuse the entire range before starting if any requested identity is occupied.
for index in "${selected_indices[@]}"; do
  shard_name=$(shard_name_for "$index")
  for target in \
    "$ISINGFOLD_CORPUS_SHARD_ROOT/$shard_name" \
    "$claims_root/$shard_name" \
    "$logs_root/$shard_name.log" \
    "$receipts_root/$shard_name.json"; do
    if [[ -e "$target" || -L "$target" ]]; then
      corpus_error "refusing to overwrite existing path: $target" 73
    fi
  done
done

run_shard() {
  local index=$1
  local shard_name
  local output
  local claim
  local log
  local receipt
  local receipt_partial
  local status
  shard_name=$(shard_name_for "$index")
  output="$ISINGFOLD_CORPUS_SHARD_ROOT/$shard_name"
  claim="$claims_root/$shard_name"
  log="$logs_root/$shard_name.log"
  receipt="$receipts_root/$shard_name.json"
  receipt_partial="$claim/receipt.partial.json"

  if ! mkdir -- "$claim"; then
    echo "refusing duplicate claim for $shard_name" >&2
    return 73
  fi
  if ! (set -o noclobber; : > "$log"); then
    echo "refusing to overwrite log $log" >&2
    return 73
  fi
  {
    echo "event=start"
    echo "utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "site=apollo"
    echo "mode=$ISINGFOLD_CORPUS_MODE"
    echo "host=$(hostname)"
    echo "shard=$shard_name"
    echo "source_root=$ISINGFOLD_CORPUS_SOURCE_ROOT"
    echo "plan=$ISINGFOLD_CORPUS_PLAN"
    echo "preflight=$ISINGFOLD_CORPUS_PREFLIGHT"
    echo "python=$ISINGFOLD_CORPUS_PYTHON"
    echo "environment_lock=$ISINGFOLD_CORPUS_ENV_LOCK"
    echo "threads=$threads"
    echo "launcher_concurrency=$concurrency"
    echo "plan_sha256=$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256"
    echo "preflight_sha256=$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256"
    echo "preflight_record_digest=$ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST"
    echo "prospective_identity_map_digest=$ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST"
    echo "source_sha256=$ISINGFOLD_CORPUS_EXPECTED_SOURCE_SHA256"
    echo "generation_provenance_sha256=$ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256"
    echo "installation_sha256=$ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256"
    echo "runtime_lock_sha256=$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256"
    echo "runtime_validator_sha256=$ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256"
    echo "environment_lock_sha256=$ISINGFOLD_CORPUS_EXPECTED_ENV_LOCK_SHA256"
    echo "python_sha256=$ISINGFOLD_CORPUS_EXPECTED_PYTHON_SHA256"
    echo "runtime_helper_sha256=$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256"
    echo "parity_attestation_sha256=$parity_attestation_sha256"
  } >> "$log"

  if env -u PYTHONHOME -u PYTHONUSERBASE -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH="$ISINGFOLD_CORPUS_SOURCE_ROOT/src" \
    OMP_NUM_THREADS="$threads" \
    OPENBLAS_NUM_THREADS="$threads" \
    MKL_NUM_THREADS="$threads" \
    NUMEXPR_NUM_THREADS="$threads" \
    "$ISINGFOLD_CORPUS_PYTHON" -m embedbench.isingfold_corpus_cli generate-shard \
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
    status=0
  else
    status=$?
  fi

  if (( status != 0 )); then
    {
      echo "event=failure"
      echo "utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "exit_status=$status"
    } >> "$log"
    printf '%s\n' "$status" > "$claim/FAILED"
    return "$status"
  fi
  if ! corpus_validate_receipt \
    "$ISINGFOLD_CORPUS_PYTHON" "$receipt_partial" \
    "$index" "$shard_count" "$output" 2>> "$log"; then
    echo "event=invalid-receipt" >> "$log"
    printf '%s\n' "invalid-receipt" > "$claim/FAILED"
    return 78
  fi
  if ! ln -- "$receipt_partial" "$receipt"; then
    echo "refusing to overwrite receipt $receipt" >> "$log"
    printf '%s\n' "receipt-collision" > "$claim/FAILED"
    return 73
  fi
  rm -- "$receipt_partial"
  {
    echo "event=success"
    echo "utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } >> "$log"
  printf '%s\n' "$ISINGFOLD_CORPUS_EXPECTED_PLAN_SHA256" > "$claim/SUCCESS"
}

declare -a child_pids=()
terminate_children() {
  local pid
  for pid in "${child_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

wait_batch() {
  local failed=0
  local pid
  for pid in "${child_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  child_pids=()
  return "$failed"
}

overall_failure=0
for index in "${selected_indices[@]}"; do
  run_shard "$index" &
  child_pids+=("$!")
  if (( ${#child_pids[@]} >= concurrency )); then
    if ! wait_batch; then
      overall_failure=1
    fi
  fi
done
if ! wait_batch; then
  overall_failure=1
fi
trap - INT TERM

if (( overall_failure != 0 )); then
  echo "one or more Apollo corpus shards failed; claims and logs were retained" >&2
  exit 1
fi
echo "completed ${#selected_indices[@]} authenticated corpus shard(s) on Apollo"

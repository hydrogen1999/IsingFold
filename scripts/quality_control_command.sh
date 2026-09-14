#!/usr/bin/env bash
# Dispatch exactly one serial quality-protocol transition through quality_cli[].

run_quality_control_operation() {
  if (( ${#quality_cli[@]} == 0 )); then
    echo "quality_cli command prefix is empty" >&2
    return 64
  fi
  local operation=${ISINGFOLD_QUALITY_OPERATION:-}
  : "${ISINGFOLD_QUALITY_OUT:?set ISINGFOLD_QUALITY_OUT}"
  quality_require_absent_output ISINGFOLD_QUALITY_OUT

  case "$operation" in
    selector-parity)
      : "${ISINGFOLD_QUALITY_PROTOCOL_CONFIG:?set ISINGFOLD_QUALITY_PROTOCOL_CONFIG}"
      : "${EXPECTED_QUALITY_PROTOCOL_CONFIG_SHA256:?set EXPECTED_QUALITY_PROTOCOL_CONFIG_SHA256}"
      quality_require_digest EXPECTED_QUALITY_PROTOCOL_CONFIG_SHA256
      "${quality_cli[@]}" audit-quality-selector-device-parity \
        --corpus "$ISINGFOLD_CORPUS" \
        --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
        --quality-protocol-config "$ISINGFOLD_QUALITY_PROTOCOL_CONFIG" \
        --expected-quality-protocol-config-sha256 \
        "$EXPECTED_QUALITY_PROTOCOL_CONFIG_SHA256" \
        "${quality_initializer_args[@]}" \
        --cpu-device cpu --accelerator-device cuda \
        --execution-runtime-sha256 "$ISINGFOLD_EXECUTION_RUNTIME_SHA256" \
        --threads "$quality_threads" \
        "${quality_optional_qubit_args[@]}" --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    plan-resolution)
      : "${ISINGFOLD_QUALITY_PROTOCOL_CONFIG:?set ISINGFOLD_QUALITY_PROTOCOL_CONFIG}"
      : "${EXPECTED_QUALITY_PROTOCOL_CONFIG_SHA256:?set EXPECTED_QUALITY_PROTOCOL_CONFIG_SHA256}"
      : "${ISINGFOLD_GRID_CONFIG:?set ISINGFOLD_GRID_CONFIG}"
      : "${EXPECTED_ISINGFOLD_GRID_CONFIG_SHA256:?set EXPECTED_ISINGFOLD_GRID_CONFIG_SHA256}"
      : "${ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT:?set ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT}"
      : "${ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT_SHA256:?set ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT_SHA256}"
      quality_require_digest EXPECTED_QUALITY_PROTOCOL_CONFIG_SHA256
      quality_require_digest EXPECTED_ISINGFOLD_GRID_CONFIG_SHA256
      quality_require_digest ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT_SHA256
      "${quality_cli[@]}" plan-quality-resolution \
        --config "$ISINGFOLD_QUALITY_PROTOCOL_CONFIG" \
        --expected-config-sha256 "$EXPECTED_QUALITY_PROTOCOL_CONFIG_SHA256" \
        --grid "$ISINGFOLD_GRID_CONFIG" \
        --expected-grid-sha256 "$EXPECTED_ISINGFOLD_GRID_CONFIG_SHA256" \
        "${quality_public_args[@]}" \
        --selector-device-parity "$ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT" \
        --expected-selector-device-parity-sha256 \
        "$ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT_SHA256" \
        "${quality_compute_args[@]}" "${quality_initializer_args[@]}" \
        "${quality_optional_qubit_args[@]}" --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    publish-resolution-verifier)
      quality_resolution_plan_args
      : "${ISINGFOLD_RESOLUTION_VERIFIER_RUNTIME_SHA256:?set ISINGFOLD_RESOLUTION_VERIFIER_RUNTIME_SHA256}"
      : "${ISINGFOLD_RESOLUTION_ATTESTOR_ID:?set ISINGFOLD_RESOLUTION_ATTESTOR_ID}"
      quality_require_digest ISINGFOLD_RESOLUTION_VERIFIER_RUNTIME_SHA256
      if [[ -z "${quality_verified_runtime_sha256:-}" ]]; then
        echo "publish-resolution-verifier requires a verified execution runtime" >&2
        return 78
      fi
      quality_require_digest quality_verified_runtime_sha256
      if [[ "$ISINGFOLD_RESOLUTION_VERIFIER_RUNTIME_SHA256" \
         != "$quality_verified_runtime_sha256" ]]; then
        echo "verifier runtime must equal the verified execution runtime" >&2
        return 78
      fi
      "${quality_cli[@]}" publish-quality-resolution-verifier-identity \
        "${quality_plan_args[@]}" \
        --verification-runtime-sha256 "$ISINGFOLD_RESOLUTION_VERIFIER_RUNTIME_SHA256" \
        --attestor-id "$ISINGFOLD_RESOLUTION_ATTESTOR_ID" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    publish-resolution-pins)
      quality_resolution_plan_args
      : "${ISINGFOLD_QUALITY_RESOLUTION_COMPLETED_STAGE:?set ISINGFOLD_QUALITY_RESOLUTION_COMPLETED_STAGE}"
      : "${ISINGFOLD_QUALITY_RESOLUTION_PIN_FILE:?set ISINGFOLD_QUALITY_RESOLUTION_PIN_FILE}"
      if [[ ! "$ISINGFOLD_QUALITY_RESOLUTION_COMPLETED_STAGE" =~ ^[0-9]+$ ]]; then
        echo "ISINGFOLD_QUALITY_RESOLUTION_COMPLETED_STAGE must be non-negative" >&2
        return 64
      fi
      quality_require_regular_pin_file ISINGFOLD_QUALITY_RESOLUTION_PIN_FILE
      quality_read_json_pin_lines "$ISINGFOLD_QUALITY_RESOLUTION_PIN_FILE"
      "${quality_cli[@]}" publish-quality-resolution-shard-pins \
        "${quality_plan_args[@]}" \
        --completed-stage-index "$ISINGFOLD_QUALITY_RESOLUTION_COMPLETED_STAGE" \
        "${quality_json_pin_args[@]}" --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    merge-resolution)
      quality_resolution_plan_args
      : "${ISINGFOLD_QUALITY_RESOLUTION_PIN_REGISTRY:?set ISINGFOLD_QUALITY_RESOLUTION_PIN_REGISTRY}"
      : "${EXPECTED_QUALITY_RESOLUTION_PIN_REGISTRY_SHA256:?set EXPECTED_QUALITY_RESOLUTION_PIN_REGISTRY_SHA256}"
      quality_require_digest EXPECTED_QUALITY_RESOLUTION_PIN_REGISTRY_SHA256
      "${quality_cli[@]}" merge-quality-resolution \
        "${quality_plan_args[@]}" \
        --pin-registry "$ISINGFOLD_QUALITY_RESOLUTION_PIN_REGISTRY" \
        --expected-pin-registry-sha256 \
        "$EXPECTED_QUALITY_RESOLUTION_PIN_REGISTRY_SHA256" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    publish-capacity-budget)
      : "${ISINGFOLD_CAPACITY_BUDGET_ID:?set ISINGFOLD_CAPACITY_BUDGET_ID}"
      : "${ISINGFOLD_CAPACITY_MAX_ARTIFACT_BYTES:?set ISINGFOLD_CAPACITY_MAX_ARTIFACT_BYTES}"
      : "${ISINGFOLD_CAPACITY_MAX_CPU_SECONDS:?set ISINGFOLD_CAPACITY_MAX_CPU_SECONDS}"
      : "${ISINGFOLD_CAPACITY_MAX_ELAPSED_SECONDS:?set ISINGFOLD_CAPACITY_MAX_ELAPSED_SECONDS}"
      : "${ISINGFOLD_CAPACITY_AVAILABLE_WORKERS:?set ISINGFOLD_CAPACITY_AVAILABLE_WORKERS}"
      : "${ISINGFOLD_CAPACITY_MIN_SCRATCH_FREE_BYTES:?set ISINGFOLD_CAPACITY_MIN_SCRATCH_FREE_BYTES}"
      "${quality_cli[@]}" publish-quality-capacity-budget \
        --budget-id "$ISINGFOLD_CAPACITY_BUDGET_ID" \
        --maximum-artifact-bytes "$ISINGFOLD_CAPACITY_MAX_ARTIFACT_BYTES" \
        --maximum-cpu-seconds "$ISINGFOLD_CAPACITY_MAX_CPU_SECONDS" \
        --maximum-elapsed-seconds "$ISINGFOLD_CAPACITY_MAX_ELAPSED_SECONDS" \
        --available-workers "$ISINGFOLD_CAPACITY_AVAILABLE_WORKERS" \
        --minimum-scratch-free-bytes "$ISINGFOLD_CAPACITY_MIN_SCRATCH_FREE_BYTES" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    plan-capacity)
      quality_resolution_plan_args
      quality_resolution_receipt_args
      "${quality_cli[@]}" plan-quality-capacity-canary \
        "${quality_plan_args[@]}" "${quality_study_args[@]}" \
        "${quality_public_args[@]}" "${quality_compute_args[@]}" \
        "${quality_initializer_args[@]}" "${quality_optional_qubit_args[@]}" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    run-capacity)
      quality_resolution_plan_args
      quality_resolution_receipt_args
      quality_capacity_artifact_args
      : "${ISINGFOLD_CAPACITY_HOST_CLASS:?set ISINGFOLD_CAPACITY_HOST_CLASS}"
      : "${ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY:?set ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY}"
      "${quality_cli[@]}" quality-capacity-canary \
        "${quality_plan_args[@]}" "${quality_study_args[@]}" \
        "${quality_public_args[@]}" "${quality_compute_args[@]}" \
        "${quality_initializer_args[@]}" \
        --execution-runtime-sha256 "$ISINGFOLD_EXECUTION_RUNTIME_SHA256" \
        --host-class "$ISINGFOLD_CAPACITY_HOST_CLASS" \
        "${quality_optional_qubit_args[@]}" "${quality_capacity_args[@]}" \
        --scratch-directory "$ISINGFOLD_CAPACITY_SCRATCH_DIRECTORY" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    verify-capacity)
      quality_resolution_plan_args
      quality_resolution_receipt_args
      quality_capacity_artifact_args
      : "${ISINGFOLD_QUALITY_CAPACITY_CANARY:?set ISINGFOLD_QUALITY_CAPACITY_CANARY}"
      : "${EXPECTED_QUALITY_CAPACITY_CANARY_SHA256:?set EXPECTED_QUALITY_CAPACITY_CANARY_SHA256}"
      quality_require_digest EXPECTED_QUALITY_CAPACITY_CANARY_SHA256
      "${quality_cli[@]}" verify-quality-capacity-canary \
        "${quality_plan_args[@]}" "${quality_study_args[@]}" \
        "${quality_capacity_args[@]}" \
        --capacity-canary "$ISINGFOLD_QUALITY_CAPACITY_CANARY" \
        --expected-capacity-canary-sha256 "$EXPECTED_QUALITY_CAPACITY_CANARY_SHA256" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    plan-preflight)
      quality_resolution_plan_args
      quality_resolution_receipt_args
      : "${ISINGFOLD_QUALITY_SHARD_ROOT:?set ISINGFOLD_QUALITY_SHARD_ROOT}"
      : "${ISINGFOLD_QUALITY_SHARD_PIN_FILE:?set ISINGFOLD_QUALITY_SHARD_PIN_FILE}"
      quality_require_regular_pin_file ISINGFOLD_QUALITY_SHARD_PIN_FILE
      quality_read_manifest_pin_lines "$ISINGFOLD_QUALITY_SHARD_PIN_FILE"
      quality_shard_pin_args=()
      for digest in "${quality_artifact_sha256s[@]}"; do
        quality_shard_pin_args+=(--expected-quality-shard-manifest-sha256 "$digest")
      done
      "${quality_cli[@]}" plan-quality-preflight-shards \
        --resolution-plan "$ISINGFOLD_QUALITY_RESOLUTION_PLAN" \
        --expected-resolution-plan-sha256 "$EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256" \
        "${quality_study_args[@]}" "${quality_public_args[@]}" \
        --quality-shard-root "$ISINGFOLD_QUALITY_SHARD_ROOT" \
        "${quality_shard_pin_args[@]}" --shard-count "${#quality_artifact_paths[@]}" \
        "${quality_compute_args[@]}" "${quality_optional_qubit_args[@]}" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    merge-preflight)
      quality_resolution_plan_args
      : "${ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_PIN_FILE:?set ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_PIN_FILE}"
      quality_require_regular_pin_file ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_PIN_FILE
      quality_read_manifest_pin_lines "$ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_PIN_FILE"
      quality_replay_args=()
      for ((index=0; index<${#quality_artifact_paths[@]}; index++)); do
        quality_replay_args+=(
          --replay-shard "${quality_artifact_paths[$index]}"
          --expected-replay-shard-manifest-sha256 "${quality_artifact_sha256s[$index]}"
        )
      done
      "${quality_cli[@]}" merge-quality-preflight-shards \
        "${quality_plan_args[@]}" "${quality_replay_args[@]}" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    merge-quality)
      : "${ISINGFOLD_QUALITY_SHARD_PIN_FILE:?set ISINGFOLD_QUALITY_SHARD_PIN_FILE}"
      : "${ISINGFOLD_QUALITY_REPLAY_BUNDLE:?set ISINGFOLD_QUALITY_REPLAY_BUNDLE}"
      : "${EXPECTED_QUALITY_REPLAY_BUNDLE_SHA256:?set EXPECTED_QUALITY_REPLAY_BUNDLE_SHA256}"
      quality_require_regular_pin_file ISINGFOLD_QUALITY_SHARD_PIN_FILE
      quality_require_digest EXPECTED_QUALITY_REPLAY_BUNDLE_SHA256
      quality_read_manifest_pin_lines "$ISINGFOLD_QUALITY_SHARD_PIN_FILE"
      quality_merge_shard_args=()
      for artifact in "${quality_artifact_paths[@]}"; do
        quality_merge_shard_args+=(--shard "$artifact")
      done
      "${quality_cli[@]}" merge-quality-labels \
        "${quality_public_args[@]}" "${quality_initializer_args[@]}" \
        "${quality_compute_args[@]}" "${quality_optional_qubit_args[@]}" \
        "${quality_merge_shard_args[@]}" \
        --trusted-replay-bundle "$ISINGFOLD_QUALITY_REPLAY_BUNDLE" \
        --expected-trusted-replay-bundle-sha256 \
        "$EXPECTED_QUALITY_REPLAY_BUNDLE_SHA256" --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    global-preflight)
      : "${ISINGFOLD_QUALITY_LABELS:?set ISINGFOLD_QUALITY_LABELS}"
      quality_workers=${ISINGFOLD_QUALITY_PREFLIGHT_WORKERS:-1}
      if [[ ! "$quality_workers" =~ ^[1-9][0-9]*$ ]]; then
        echo "ISINGFOLD_QUALITY_PREFLIGHT_WORKERS must be positive" >&2
        return 64
      fi
      "${quality_cli[@]}" quality-preflight \
        "${quality_public_args[@]}" "${quality_initializer_args[@]}" \
        "${quality_compute_args[@]}" "${quality_optional_qubit_args[@]}" \
        --quality-labels "$ISINGFOLD_QUALITY_LABELS" \
        --min-resolved-rows 128 --min-resolved-lineages 128 \
        --workers "$quality_workers" --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    verify-binding)
      quality_resolution_plan_args
      quality_resolution_receipt_args
      : "${ISINGFOLD_QUALITY_MANIFEST:?set ISINGFOLD_QUALITY_MANIFEST}"
      : "${EXPECTED_QUALITY_MANIFEST_SHA256:?set EXPECTED_QUALITY_MANIFEST_SHA256}"
      quality_require_digest EXPECTED_QUALITY_MANIFEST_SHA256
      "${quality_cli[@]}" verify-quality-resolution-binding \
        "${quality_plan_args[@]}" "${quality_study_args[@]}" \
        --quality-manifest "$ISINGFOLD_QUALITY_MANIFEST" \
        --expected-quality-manifest-sha256 "$EXPECTED_QUALITY_MANIFEST_SHA256" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    verify-readiness)
      quality_resolution_plan_args
      quality_resolution_receipt_args
      : "${ISINGFOLD_QUALITY_BINDING:?set ISINGFOLD_QUALITY_BINDING}"
      : "${EXPECTED_QUALITY_BINDING_SHA256:?set EXPECTED_QUALITY_BINDING_SHA256}"
      : "${ISINGFOLD_QUALITY_MANIFEST:?set ISINGFOLD_QUALITY_MANIFEST}"
      : "${EXPECTED_QUALITY_MANIFEST_SHA256:?set EXPECTED_QUALITY_MANIFEST_SHA256}"
      : "${ISINGFOLD_QUALITY_PREFLIGHT_RECEIPT:?set ISINGFOLD_QUALITY_PREFLIGHT_RECEIPT}"
      : "${EXPECTED_QUALITY_PREFLIGHT_SHA256:?set EXPECTED_QUALITY_PREFLIGHT_SHA256}"
      quality_require_digest EXPECTED_QUALITY_BINDING_SHA256
      quality_require_digest EXPECTED_QUALITY_MANIFEST_SHA256
      quality_require_digest EXPECTED_QUALITY_PREFLIGHT_SHA256
      "${quality_cli[@]}" verify-quality-training-input-readiness \
        "${quality_plan_args[@]}" "${quality_study_args[@]}" \
        --binding "$ISINGFOLD_QUALITY_BINDING" \
        --expected-binding-sha256 "$EXPECTED_QUALITY_BINDING_SHA256" \
        --quality-manifest "$ISINGFOLD_QUALITY_MANIFEST" \
        --expected-quality-manifest-sha256 "$EXPECTED_QUALITY_MANIFEST_SHA256" \
        --quality-preflight "$ISINGFOLD_QUALITY_PREFLIGHT_RECEIPT" \
        --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    gates)
      quality_resolution_plan_args
      : "${ISINGFOLD_GRID_CONFIG:?set ISINGFOLD_GRID_CONFIG}"
      : "${ISINGFOLD_QUALITY_LABELS:?set ISINGFOLD_QUALITY_LABELS}"
      : "${ISINGFOLD_QUALITY_PREFLIGHT_RECEIPT:?set ISINGFOLD_QUALITY_PREFLIGHT_RECEIPT}"
      : "${EXPECTED_QUALITY_PREFLIGHT_SHA256:?set EXPECTED_QUALITY_PREFLIGHT_SHA256}"
      : "${ISINGFOLD_SELECTOR_LABELS:?set ISINGFOLD_SELECTOR_LABELS}"
      : "${ISINGFOLD_EXACT_CONFORMANCE_CORPUS:?set ISINGFOLD_EXACT_CONFORMANCE_CORPUS}"
      : "${EXPECTED_EXACT_CONFORMANCE_SHA256:?set EXPECTED_EXACT_CONFORMANCE_SHA256}"
      quality_require_digest EXPECTED_QUALITY_PREFLIGHT_SHA256
      quality_require_digest EXPECTED_EXACT_CONFORMANCE_SHA256
      quality_gate_seed=${ISINGFOLD_GATE_SEED:-1009}
      quality_gate_instances=${ISINGFOLD_GATE_INSTANCES:-6}
      if [[ ! "$quality_gate_seed" =~ ^[0-9]+$ \
         || ! "$quality_gate_instances" =~ ^[1-9][0-9]*$ ]]; then
        echo "gate seed/instance count is invalid" >&2
        return 64
      fi
      "${quality_cli[@]}" gates \
        "${quality_public_args[@]}" "${quality_compute_args[@]}" \
        "${quality_optional_qubit_args[@]}" \
        --seed "$quality_gate_seed" --reward-reads 256 \
        --grid "$ISINGFOLD_GRID_CONFIG" "${quality_plan_args[@]}" \
        "${quality_initializer_args[@]}" --quality-labels "$ISINGFOLD_QUALITY_LABELS" \
        --quality-preflight-receipt "$ISINGFOLD_QUALITY_PREFLIGHT_RECEIPT" \
        --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256" \
        --selector-labels "$ISINGFOLD_SELECTOR_LABELS" \
        --exact-conformance-corpus "$ISINGFOLD_EXACT_CONFORMANCE_CORPUS" \
        --expected-exact-conformance-sha256 "$EXPECTED_EXACT_CONFORMANCE_SHA256" \
        --partition validation --instances "$quality_gate_instances" \
        --out "$ISINGFOLD_QUALITY_OUT"
      ;;
    *)
      echo "unknown ISINGFOLD_QUALITY_OPERATION: $operation" >&2
      return 64
      ;;
  esac
}

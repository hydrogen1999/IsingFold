#!/usr/bin/env bash
# Goose must run whole-system confirmation through a Slurm array.
#SBATCH --job-name=isingfold-complete-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=isingfold-complete-eval-%A_%a.out
#SBATCH --error=isingfold-complete-eval-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_complete_eval.sbatch must run as a Slurm array job" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${ISINGFOLD_RUN_ROOT:?set ISINGFOLD_RUN_ROOT}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"
: "${ISINGFOLD_COMPLETE_OUT_ROOT:?set ISINGFOLD_COMPLETE_OUT_ROOT}"
: "${ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT:?set ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT}"
: "${ISINGFOLD_RL_FREEZE:?set ISINGFOLD_RL_FREEZE}"
: "${ISINGFOLD_RL_FREEZE_SHA256:?set ISINGFOLD_RL_FREEZE_SHA256}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256:?set EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256}"
: "${EXTERNAL_TUNING_SELECTION:?set EXTERNAL_TUNING_SELECTION}"
: "${EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256:?set EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256}"
: "${ISINGFOLD_FINAL_TEST_BOOTSTRAP_BANK:?set ISINGFOLD_FINAL_TEST_BOOTSTRAP_BANK}"
: "${EXPECTED_FINAL_TEST_BOOTSTRAP_PLAN_SHA256:?set EXPECTED_FINAL_TEST_BOOTSTRAP_PLAN_SHA256}"
: "${EXPECTED_FINAL_TEST_BOOTSTRAP_MANIFEST_SHA256:?set EXPECTED_FINAL_TEST_BOOTSTRAP_MANIFEST_SHA256}"

grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
external_config=${ISINGFOLD_EXTERNAL_COMPLETE_CONFIG:-configs/external_minorminer_complete_v1.json}
tuning_registry=${ISINGFOLD_EXTERNAL_TUNING_REGISTRY:-configs/external_minorminer_tuning_hybrid_v1.json}
threads=${ISINGFOLD_THREADS:-4}
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime \
  "$script_dir" \
  gpu \
  "$ISINGFOLD_COMPLETE_OUT_ROOT" \
  "$ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli evaluate-complete-system-cell \
  --grid "$grid" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --corpus "$ISINGFOLD_CORPUS" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
  --run-root "$ISINGFOLD_RUN_ROOT" \
  --config "$ISINGFOLD_COMPLETE_CONFIG" \
  --bootstrap-bank "$ISINGFOLD_FINAL_TEST_BOOTSTRAP_BANK" \
  --expected-bootstrap-plan-sha256 "$EXPECTED_FINAL_TEST_BOOTSTRAP_PLAN_SHA256" \
  --expected-bootstrap-manifest-sha256 "$EXPECTED_FINAL_TEST_BOOTSTRAP_MANIFEST_SHA256" \
  --out "$ISINGFOLD_COMPLETE_OUT_ROOT/seed-$SLURM_ARRAY_TASK_ID" \
  --rl-value-selection-receipt "$ISINGFOLD_RL_FREEZE" \
  --expected-selection-sha256 "$ISINGFOLD_RL_FREEZE_SHA256" \
  --device cuda \
  --threads "$threads"

# Run the matching stock row inside the same allocation. The paired aggregator requires
# exact hostname/device/thread/determinism identity and reports both total wall clocks.
/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli evaluate-external-complete-system \
  --grid "$grid" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --corpus "$ISINGFOLD_CORPUS" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
  --config "$external_config" \
  --learned-config "$ISINGFOLD_COMPLETE_CONFIG" \
  --tuning-registry "$tuning_registry" \
  --expected-tuning-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
  --external-tuning-selection "$EXTERNAL_TUNING_SELECTION" \
  --expected-external-tuning-selection-sha256 "$EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256" \
  --out "$ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT/seed-$SLURM_ARRAY_TASK_ID" \
  --device cuda \
  --threads "$threads"

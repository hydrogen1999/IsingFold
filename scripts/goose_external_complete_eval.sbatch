#!/usr/bin/env bash
# Goose must run the frozen stock complete-system arm inside a Slurm allocation.
#SBATCH --job-name=isingfold-stock-complete
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=isingfold-stock-complete-%A_%a.out
#SBATCH --error=isingfold-stock-complete-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_external_complete_eval.sbatch must run as a Slurm array" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT:?set ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256:?set EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256}"
: "${EXTERNAL_TUNING_SELECTION:?set EXTERNAL_TUNING_SELECTION}"
: "${EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256:?set EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256}"

grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
external_config=${ISINGFOLD_EXTERNAL_COMPLETE_CONFIG:-configs/external_minorminer_complete_v1.json}
learned_config=${ISINGFOLD_COMPLETE_CONFIG:-configs/complete_system_lac_hybrid_cache_v1.json}
tuning_registry=${ISINGFOLD_EXTERNAL_TUNING_REGISTRY:-configs/external_minorminer_tuning_hybrid_v1.json}
threads=${ISINGFOLD_THREADS:-4}
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" gpu "$ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli evaluate-external-complete-system \
  --grid "$grid" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --corpus "$ISINGFOLD_CORPUS" \
  --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --config "$external_config" \
  --learned-config "$learned_config" \
  --tuning-registry "$tuning_registry" \
  --expected-tuning-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
  --external-tuning-selection "$EXTERNAL_TUNING_SELECTION" \
  --expected-external-tuning-selection-sha256 "$EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256" \
  --device cuda \
  --threads "$threads" \
  --out "$ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT/seed-$SLURM_ARRAY_TASK_ID"

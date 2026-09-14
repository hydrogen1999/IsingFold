#!/usr/bin/env bash
# Goose must run each tuning seed's seven candidates sequentially on one Slurm node.
#SBATCH --job-name=isingfold-stock-tuning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-2
#SBATCH --output=isingfold-stock-tuning-%A_%a.out
#SBATCH --error=isingfold-stock-tuning-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_external_tuning.sbatch must run as a Slurm array" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${ISINGFOLD_EXTERNAL_TUNING_OUT_ROOT:?set ISINGFOLD_EXTERNAL_TUNING_OUT_ROOT}"
: "${EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256:?set EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"

registry=${ISINGFOLD_EXTERNAL_TUNING_REGISTRY:-configs/external_minorminer_tuning_hybrid_v1.json}
grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
external_config=${ISINGFOLD_EXTERNAL_COMPLETE_CONFIG:-configs/external_minorminer_complete_v1.json}
learned_config=${ISINGFOLD_COMPLETE_CONFIG:-configs/complete_system_lac_hybrid_cache_v1.json}
threads=1
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" gpu "$ISINGFOLD_EXTERNAL_TUNING_OUT_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

for candidate_index in {0..6}; do
  /opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
    -I -m isingfold.rl.cli evaluate-external-tuning-cell \
    --registry "$registry" \
    --expected-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
    --grid "$grid" \
    --index "$candidate_index" \
    --tuning-seed-index "$SLURM_ARRAY_TASK_ID" \
    --corpus "$ISINGFOLD_CORPUS" \
    --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
    --quality-attestation "$QUALITY_ATTESTATION" \
    --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
    --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
    --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
    --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
    --external-config "$external_config" \
    --learned-config "$learned_config" \
    --device cuda \
    --threads "$threads" \
    --out "$ISINGFOLD_EXTERNAL_TUNING_OUT_ROOT"
done

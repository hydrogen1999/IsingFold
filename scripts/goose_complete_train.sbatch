#!/usr/bin/env bash
# Goose must run this three-seed confirmation through a Slurm array.
#SBATCH --job-name=isingfold-complete-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=isingfold-complete-train-%A_%a.out
#SBATCH --error=isingfold-complete-train-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_complete_train.sbatch must run as a Slurm array job" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${ISINGFOLD_QUALITY_LABELS:?set ISINGFOLD_QUALITY_LABELS}"
: "${ISINGFOLD_RUN_ROOT:?set ISINGFOLD_RUN_ROOT}"
: "${ISINGFOLD_RL_FREEZE:?set ISINGFOLD_RL_FREEZE}"
: "${ISINGFOLD_RL_FREEZE_SHA256:?set ISINGFOLD_RL_FREEZE_SHA256}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${QUALITY_PREFLIGHT_RECEIPT:?set QUALITY_PREFLIGHT_RECEIPT}"
: "${EXPECTED_QUALITY_PREFLIGHT_SHA256:?set EXPECTED_QUALITY_PREFLIGHT_SHA256}"
: "${ISINGFOLD_INITIALIZER_BANK_ROOT:?set ISINGFOLD_INITIALIZER_BANK_ROOT}"
: "${ISINGFOLD_COMPLETE_CONFIG:?set ISINGFOLD_COMPLETE_CONFIG}"

grid=${ISINGFOLD_GRID_CONFIG:-configs/rl_grid_hybrid_v1.json}
threads=${ISINGFOLD_THREADS:-4}
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" gpu "$ISINGFOLD_RUN_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

initializer_bank="$ISINGFOLD_INITIALIZER_BANK_ROOT/seed-$SLURM_ARRAY_TASK_ID"
bank_pin_name="ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_${SLURM_ARRAY_TASK_ID}"
initializer_bank_sha256=${!bank_pin_name:-}
if [[ ! "$initializer_bank_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "set $bank_pin_name to the out-of-band manifest SHA-256" >&2
  exit 64
fi

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli complete-system-train-cell \
  --grid "$grid" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --corpus "$ISINGFOLD_CORPUS" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --selector "$ISINGFOLD_SELECTOR_BUNDLE" \
  --quality-labels "$ISINGFOLD_QUALITY_LABELS" \
  --quality-preflight-receipt "$QUALITY_PREFLIGHT_RECEIPT" \
  --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256" \
  --run-root "$ISINGFOLD_RUN_ROOT" \
  --initializer-bank "$initializer_bank" \
  --expected-initializer-bank-manifest-sha256 "$initializer_bank_sha256" \
  --complete-config "$ISINGFOLD_COMPLETE_CONFIG" \
  --rl-value-selection-receipt "$ISINGFOLD_RL_FREEZE" \
  --expected-selection-sha256 "$ISINGFOLD_RL_FREEZE_SHA256" \
  --device cuda \
  --threads "$threads"

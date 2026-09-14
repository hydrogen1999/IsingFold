#!/usr/bin/env bash
# Goose runs every final-strength shard inside a scheduled allocation.
#SBATCH --job-name=isingfold-final-strength
#SBATCH --array=0-127%16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=isingfold-final-strength-%A_%a.out
#SBATCH --error=isingfold-final-strength-%A_%a.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_final_strength_audit.sbatch must run as a Slurm array" >&2
  exit 69
fi
if [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || (( SLURM_ARRAY_TASK_ID > 127 )); then
  echo "final-strength shard index must be in 0-127" >&2
  exit 64
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_FINAL_STRENGTH_PLAN:?set ISINGFOLD_FINAL_STRENGTH_PLAN}"
: "${EXPECTED_FINAL_STRENGTH_PLAN_SHA256:?set EXPECTED_FINAL_STRENGTH_PLAN_SHA256}"
: "${ISINGFOLD_FINAL_STRENGTH_EXECUTION:?set ISINGFOLD_FINAL_STRENGTH_EXECUTION}"
: "${EXPECTED_FINAL_STRENGTH_EXECUTION_SHA256:?set EXPECTED_FINAL_STRENGTH_EXECUTION_SHA256}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${LEARNED_COMPLETE_0:?set LEARNED_COMPLETE_0}"
: "${LEARNED_COMPLETE_1:?set LEARNED_COMPLETE_1}"
: "${LEARNED_COMPLETE_2:?set LEARNED_COMPLETE_2}"
: "${EXPECTED_LEARNED_REPORT_SHA256_0:?set EXPECTED_LEARNED_REPORT_SHA256_0}"
: "${EXPECTED_LEARNED_REPORT_SHA256_1:?set EXPECTED_LEARNED_REPORT_SHA256_1}"
: "${EXPECTED_LEARNED_REPORT_SHA256_2:?set EXPECTED_LEARNED_REPORT_SHA256_2}"
: "${STOCK_COMPLETE_0:?set STOCK_COMPLETE_0}"
: "${STOCK_COMPLETE_1:?set STOCK_COMPLETE_1}"
: "${STOCK_COMPLETE_2:?set STOCK_COMPLETE_2}"
: "${EXPECTED_STOCK_REPORT_SHA256_0:?set EXPECTED_STOCK_REPORT_SHA256_0}"
: "${EXPECTED_STOCK_REPORT_SHA256_1:?set EXPECTED_STOCK_REPORT_SHA256_1}"
: "${EXPECTED_STOCK_REPORT_SHA256_2:?set EXPECTED_STOCK_REPORT_SHA256_2}"
: "${ISINGFOLD_FINAL_STRENGTH_SHARD_ROOT:?set ISINGFOLD_FINAL_STRENGTH_SHARD_ROOT}"

threads=${ISINGFOLD_THREADS:-4}
script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" cpu "$ISINGFOLD_FINAL_STRENGTH_SHARD_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" cpu
mkdir -p "$ISINGFOLD_FINAL_STRENGTH_SHARD_ROOT"
printf -v shard_name 'shard-%03d-of-128' "$SLURM_ARRAY_TASK_ID"

/opt/slurm/bin/srun --cpus-per-task="$threads" \
  "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli run-final-strength-audit-shard \
  --plan "$ISINGFOLD_FINAL_STRENGTH_PLAN" \
  --expected-plan-sha256 "$EXPECTED_FINAL_STRENGTH_PLAN_SHA256" \
  --execution-manifest "$ISINGFOLD_FINAL_STRENGTH_EXECUTION" \
  --expected-execution-manifest-sha256 "$EXPECTED_FINAL_STRENGTH_EXECUTION_SHA256" \
  --corpus "$ISINGFOLD_CORPUS" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --learned-evaluation "$LEARNED_COMPLETE_0" \
  --expected-learned-report-sha256 "$EXPECTED_LEARNED_REPORT_SHA256_0" \
  --learned-evaluation "$LEARNED_COMPLETE_1" \
  --expected-learned-report-sha256 "$EXPECTED_LEARNED_REPORT_SHA256_1" \
  --learned-evaluation "$LEARNED_COMPLETE_2" \
  --expected-learned-report-sha256 "$EXPECTED_LEARNED_REPORT_SHA256_2" \
  --external-evaluation "$STOCK_COMPLETE_0" \
  --expected-external-report-sha256 "$EXPECTED_STOCK_REPORT_SHA256_0" \
  --external-evaluation "$STOCK_COMPLETE_1" \
  --expected-external-report-sha256 "$EXPECTED_STOCK_REPORT_SHA256_1" \
  --external-evaluation "$STOCK_COMPLETE_2" \
  --expected-external-report-sha256 "$EXPECTED_STOCK_REPORT_SHA256_2" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --out "$ISINGFOLD_FINAL_STRENGTH_SHARD_ROOT/$shard_name"

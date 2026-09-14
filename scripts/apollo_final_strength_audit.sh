#!/usr/bin/env bash
# Apollo runs final-strength shards directly. Use FIRST/LAST/STEP for disjoint workers.
set -euo pipefail

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

first=${ISINGFOLD_SHARD_FIRST:-0}
last=${ISINGFOLD_SHARD_LAST:-127}
step=${ISINGFOLD_SHARD_STEP:-1}
for value in "$first" "$last" "$step"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "FIRST, LAST, and STEP must be nonnegative integers" >&2
    exit 64
  fi
done
if (( first > last || last > 127 || step == 0 )); then
  echo "require 0 <= FIRST <= LAST <= 127 and STEP > 0" >&2
  exit 64
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/publication_runtime.sh"
require_apollo_publication_runtime
"$script_dir/verify_runtime_source.sh" "$python_bin" cpu
mkdir -p "$ISINGFOLD_FINAL_STRENGTH_SHARD_ROOT"

common=(
  --plan "$ISINGFOLD_FINAL_STRENGTH_PLAN"
  --expected-plan-sha256 "$EXPECTED_FINAL_STRENGTH_PLAN_SHA256"
  --execution-manifest "$ISINGFOLD_FINAL_STRENGTH_EXECUTION"
  --expected-execution-manifest-sha256 "$EXPECTED_FINAL_STRENGTH_EXECUTION_SHA256"
  --corpus "$ISINGFOLD_CORPUS"
  --quality-attestation "$QUALITY_ATTESTATION"
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST"
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID"
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT"
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256"
  --learned-evaluation "$LEARNED_COMPLETE_0"
  --expected-learned-report-sha256 "$EXPECTED_LEARNED_REPORT_SHA256_0"
  --learned-evaluation "$LEARNED_COMPLETE_1"
  --expected-learned-report-sha256 "$EXPECTED_LEARNED_REPORT_SHA256_1"
  --learned-evaluation "$LEARNED_COMPLETE_2"
  --expected-learned-report-sha256 "$EXPECTED_LEARNED_REPORT_SHA256_2"
  --external-evaluation "$STOCK_COMPLETE_0"
  --expected-external-report-sha256 "$EXPECTED_STOCK_REPORT_SHA256_0"
  --external-evaluation "$STOCK_COMPLETE_1"
  --expected-external-report-sha256 "$EXPECTED_STOCK_REPORT_SHA256_1"
  --external-evaluation "$STOCK_COMPLETE_2"
  --expected-external-report-sha256 "$EXPECTED_STOCK_REPORT_SHA256_2"
)

index=$first
while (( index <= last )); do
  printf -v shard_name 'shard-%03d-of-128' "$index"
  "$python_bin" -I -m isingfold.rl.cli run-final-strength-audit-shard \
    "${common[@]}" \
    --shard-index "$index" \
    --out "$ISINGFOLD_FINAL_STRENGTH_SHARD_ROOT/$shard_name"
  index=$((index + step))
done

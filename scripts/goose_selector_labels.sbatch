#!/usr/bin/env bash
# Build the authenticated train-only strength-selector labels on a Goose compute node.
#SBATCH --job-name=isingfold-selector-labels
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=isingfold-selector-labels-%j.out
#SBATCH --error=isingfold-selector-labels-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_selector_labels.sbatch requires one non-array Slurm job" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_LABEL_WORK_ROOT:?set ISINGFOLD_SELECTOR_LABEL_WORK_ROOT}"
: "${ISINGFOLD_SELECTOR_LABELS:?set ISINGFOLD_SELECTOR_LABELS}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"

sample_seed=${ISINGFOLD_SELECTOR_LABEL_SEED:-503}
split_seed=${ISINGFOLD_SELECTOR_SPLIT_SEED:-601}
calibration_fraction=${ISINGFOLD_SELECTOR_CALIBRATION_FRACTION:-0.2}
if [[ ! "$sample_seed" =~ ^[0-9]+$ || ! "$split_seed" =~ ^[0-9]+$ ]]; then
  echo "selector label and split seeds must be non-negative integers" >&2
  exit 64
fi
if [[ "$ISINGFOLD_SELECTOR_LABELS" != /* \
   || -e "$ISINGFOLD_SELECTOR_LABELS" || -L "$ISINGFOLD_SELECTOR_LABELS" ]]; then
  echo "ISINGFOLD_SELECTOR_LABELS must be an absent absolute artifact path" >&2
  exit 78
fi

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" cpu "$ISINGFOLD_SELECTOR_LABEL_WORK_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" cpu

work_root=$(cd -- "$ISINGFOLD_SELECTOR_LABEL_WORK_ROOT" && pwd -P)
artifact_parent=$(cd -- "$(dirname -- "$ISINGFOLD_SELECTOR_LABELS")" && pwd -P)
if [[ "$artifact_parent" != "$work_root" ]]; then
  echo "ISINGFOLD_SELECTOR_LABELS must be a direct child of its writable work root" >&2
  exit 78
fi

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli label-selector-data \
  --corpus "$ISINGFOLD_CORPUS" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --seed "$sample_seed" \
  --split-seed "$split_seed" \
  --calibration-fraction "$calibration_fraction" \
  --out "$ISINGFOLD_SELECTOR_LABELS"

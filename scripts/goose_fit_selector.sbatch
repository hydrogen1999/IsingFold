#!/usr/bin/env bash
# Train and freeze IF-Q3-S0 on one authenticated Goose GPU allocation.
#SBATCH --job-name=isingfold-fit-selector
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=isingfold-fit-selector-%j.out
#SBATCH --error=isingfold-fit-selector-%j.err
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "goose_fit_selector.sbatch requires one non-array Slurm job" >&2
  exit 69
fi

: "${ISINGFOLD_CORPUS:?set ISINGFOLD_CORPUS}"
: "${ISINGFOLD_SELECTOR_LABELS:?set ISINGFOLD_SELECTOR_LABELS}"
: "${ISINGFOLD_SELECTOR_BUNDLE_WORK_ROOT:?set ISINGFOLD_SELECTOR_BUNDLE_WORK_ROOT}"
: "${ISINGFOLD_SELECTOR_BUNDLE:?set ISINGFOLD_SELECTOR_BUNDLE}"
: "${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"
: "${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set EXPECTED_QUALITY_ATTESTATION_DIGEST}"
: "${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"
: "${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"
: "${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"

seed=${ISINGFOLD_SELECTOR_TRAINING_SEED:-701}
epochs=${ISINGFOLD_SELECTOR_EPOCHS:-100}
graph_minibatch=${ISINGFOLD_SELECTOR_GRAPH_MINIBATCH:-32}
threads=${ISINGFOLD_THREADS:-4}
for value in "$seed" "$epochs" "$graph_minibatch" "$threads"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "selector seed, epochs, minibatch and threads must be integers" >&2
    exit 64
  fi
done
if (( epochs == 0 || graph_minibatch == 0 || threads == 0 )); then
  echo "selector epochs, minibatch and threads must be positive" >&2
  exit 64
fi
if [[ "$ISINGFOLD_SELECTOR_BUNDLE" != /* \
   || -e "$ISINGFOLD_SELECTOR_BUNDLE" || -L "$ISINGFOLD_SELECTOR_BUNDLE" ]]; then
  echo "ISINGFOLD_SELECTOR_BUNDLE must be an absent absolute artifact path" >&2
  exit 78
fi

script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}
source "$script_dir/publication_runtime.sh"
require_goose_publication_runtime "$script_dir" gpu "$ISINGFOLD_SELECTOR_BUNDLE_WORK_ROOT"
/opt/slurm/bin/srun "${container_exec[@]}" \
  bash "$script_dir/verify_runtime_source.sh" "$container_python" gpu

work_root=$(cd -- "$ISINGFOLD_SELECTOR_BUNDLE_WORK_ROOT" && pwd -P)
artifact_parent=$(cd -- "$(dirname -- "$ISINGFOLD_SELECTOR_BUNDLE")" && pwd -P)
if [[ "$artifact_parent" != "$work_root" ]]; then
  echo "ISINGFOLD_SELECTOR_BUNDLE must be a direct child of its writable work root" >&2
  exit 78
fi

/opt/slurm/bin/srun "${container_exec[@]}" "$container_python" \
  -I -m isingfold.rl.cli fit-selector \
  --corpus "$ISINGFOLD_CORPUS" \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --selector-labels "$ISINGFOLD_SELECTOR_LABELS" \
  --seed "$seed" \
  --epochs "$epochs" \
  --graph-minibatch "$graph_minibatch" \
  --device cuda \
  --threads "$threads" \
  --out "$ISINGFOLD_SELECTOR_BUNDLE"

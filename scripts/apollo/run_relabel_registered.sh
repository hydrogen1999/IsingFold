# Task 6 of docs/plans/2026-09-15-plan.md: labels under the registered schedule, with the
# corrected compiler, on the modern corpora. 160 training and 80 held-out lineages per host,
# two states each, up to eight candidates per state, 256-read labels, 512-read assessment.
# The cache carries provenance, so a later trainer cannot use it against a different corpus,
# schedule or compiler without saying so.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/relabel
for H in pegasus6 zephyr4; do
  CACHE=runs/relabel/${H}_registered.pkl
  if [ -s "$CACHE" ]; then echo "cache present for $H"; continue; fi
  python -u probes/train_quality.py --corpus runs/modern/$H/corpus --family if-mlp \
    --train-lineages 160 --eval-lineages 80 --states 2 --max-candidates 8 \
    --reads 256 --fresh-reads 512 --epochs 1 --loss bce --rank-weight 0.0 \
    --inner-fraction 0.0 --qubit-cap 248 --cache "$CACHE" > runs/relabel/${H}.log 2>&1 &
  sleep 3
done
wait
echo "RELABEL FINISHED"

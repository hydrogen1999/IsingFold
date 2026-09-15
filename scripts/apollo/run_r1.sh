# R1 from the fifth audit: with correct labels and one shared dataset, can the head fit at all,
# and does the loss it is fitted with decide the answer?
#
# The arms differ only in the loss. They read the same cached labels, so a difference between
# them is the loss and not the data. Train regret is the capacity number: how much utility the
# argmax over the head gives up against the best labelled candidate, on the states it was fitted
# on. A head that cannot drive that down has not been shown to lack capacity until the
# optimisation has been given a fair run, which is why the step count and learning rate are
# raised and the gradient norm is logged.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/r1
CACHE=runs/r1/labels.pkl

# One arm first, alone, so it builds the cache the others reuse.
if [ ! -s "$CACHE" ]; then
  python -u probes/train_quality.py --corpus runs/v1/corpus --family if-mlp     --train-lineages 200 --eval-lineages 60 --states 2 --max-candidates 8     --reads 256 --fresh-reads 512 --epochs 800 --learning-rate 1e-3     --loss bce --rank-weight 0.0 --inner-fraction 0.0 --cache "$CACHE"     --out runs/r1/bce.json > runs/r1/bce.log 2>&1
fi
for ARM in joint consistent; do
  LOG=runs/r1/$ARM.log
  if [ -s "$LOG" ] && grep -q "TRAIN QUALITY DONE" "$LOG"; then echo "skip $ARM"; continue; fi
  rm -f "$LOG"
  python -u probes/train_quality.py --corpus runs/v1/corpus --family if-mlp     --train-lineages 200 --eval-lineages 60 --states 2 --max-candidates 8     --reads 256 --fresh-reads 512 --epochs 800 --learning-rate 1e-3     --loss $ARM --rank-weight 1.0 --inner-fraction 0.0 --cache "$CACHE"     --out runs/r1/$ARM.json > "$LOG" 2>&1 &
  sleep 3
done
wait
echo "R1 FINISHED"

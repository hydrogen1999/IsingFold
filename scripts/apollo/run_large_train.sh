# Direction 5, the closure of the prediction direction: the corrected scorer on 1680 training
# lineages against the same held-out protocol, three seeds, chained behind the large labels.
# If held-out gain stays where 160 lineages put it, prediction is closed for good, not for now.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=4
until grep -q "LARGE LABELS FINISHED" runs/large_labels.out 2>/dev/null; do sleep 600; done
for SEED in 0 1 2; do
  LOG=runs/large/train_seed${SEED}.log
  if [ -s "$LOG" ] && grep -q "SUCCESSOR SCORER DONE" "$LOG"; then continue; fi
  python -u probes/train_successor.py --cache runs/large/pegasus6_registered.pkl \
    --epochs 400 --width 64 --learning-rate 1e-3 --clip 10.0 --rank-weight 1.0 \
    --inner-fraction 0.0 --qubit-cap 248 --seed "$SEED" \
    --out runs/large/train_seed${SEED}.json > "$LOG" 2>&1 &
  sleep 3
done
wait
echo "LARGE TRAIN FINISHED"

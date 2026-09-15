# Where is the bottleneck? Everything else has been ruled out one at a time: capacity fits the
# labels exactly on seen states, three losses transfer identically badly, the labels were wrong
# and were fixed, the teacher was wrong and was fixed, and the representation is worth about
# +0.008 which is below the spread between seeds. The one quantity nobody has varied is how much
# data the thing is fitted on.
#
# 389 states from 200 lineages is roughly three thousand labelled candidates for a function over
# program graphs. If the curve is still climbing at 200, the bottleneck is data and every other
# effect measured here is noise around it. If it is flat from 50 onward, data is not the answer
# and the representation question becomes the live one again.
#
# The held-out lineages never change and the training subsets are nested, so the curve measures
# data rather than a different test set.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/curve
for N in 25 50 100 200; do
  for SEED in 0 1; do
    LOG=runs/curve/n${N}_seed${SEED}.log
    if [ -s "$LOG" ] && grep -q "SUCCESSOR SCORER DONE" "$LOG"; then echo "skip $N/$SEED"; continue; fi
    rm -f "$LOG"
    python -u probes/train_successor.py --cache runs/r1/labels.pkl --epochs 400 --width 64 \
      --learning-rate 1e-3 --clip 10.0 --rank-weight 1.0 --inner-fraction 0.0 \
      --train-subset "$N" --seed "$SEED" --out runs/curve/n${N}_seed${SEED}.json \
      > "$LOG" 2>&1 &
    sleep 2
  done
done
wait
echo "CURVE FINISHED"

# The full representation ablation on the hard corpus, chained behind the label collection so
# nothing idles. Four arms from the design's table, two seeds each, all reading one cache:
#
#   F0     the successor program alone
#   Fpos   plus the hardware's own coordinates                      (section 7.1)
#   Fphys  plus the energy margin of each chain's cheapest cut       (section 8.2)
#   Fall   both
#
# Same labels, same states, same references, same width, same steps. What differs between arms is
# what the model is shown, which is the only way to attribute a difference to the representation.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
CACHE=runs/hard/labels.pkl
mkdir -p runs/hard_abl

waited=0
while [ ! -s "$CACHE" ]; do
  sleep 60
  waited=$((waited + 60))
  if [ "$waited" -gt 7200 ]; then echo "labels never appeared"; exit 1; fi
done
echo "labels ready after ${waited}s of waiting"
# The writer may still be flushing; a partial pickle would fail every arm at once.
sleep 30

run () {
  name=$1; seed=$2; shift 2
  LOG=runs/hard_abl/${name}_seed${seed}.log
  if [ -s "$LOG" ] && grep -q "SUCCESSOR SCORER DONE" "$LOG"; then echo "skip $name/$seed"; return; fi
  rm -f "$LOG"
  python -u probes/train_successor.py --cache "$CACHE" --epochs 400 --width 64 \
    --learning-rate 1e-3 --clip 10.0 --rank-weight 1.0 --inner-fraction 0.0 \
    --seed "$seed" --out runs/hard_abl/${name}_seed${seed}.json "$@" \
    > "$LOG" 2>&1 &
  sleep 3
}

for SEED in 0 1; do
  run F0    "$SEED"
  run Fpos  "$SEED" --coords
  run Fphys "$SEED" --physics
  run Fall  "$SEED" --coords --physics
done
wait
echo "HARD ABLATION FINISHED"

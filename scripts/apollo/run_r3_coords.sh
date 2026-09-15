# The meeting design, section 7.1: give the model the hardware's own indices instead of an
# arbitrary label. Chimera (i,j,u,k), Pegasus (u,w,k,z), Zephyr (u,w,k,j,z), taken from the
# generator's converter and normalised, with a flag for a family that has no native mapping.
#
# Paired against runs/r2b, which is the same scorer, the same labels, the same three seeds and
# the same clip, without the coordinates. Nothing else differs, so a gap between them is the
# representation.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/r3
for SEED in 0 1 2; do
  LOG=runs/r3/coords_seed$SEED.log
  if [ -s "$LOG" ] && grep -q "SUCCESSOR SCORER DONE" "$LOG"; then echo "skip $SEED"; continue; fi
  rm -f "$LOG"
  python -u probes/train_successor.py --cache runs/r1/labels.pkl --epochs 400 --width 64 \
    --learning-rate 1e-3 --clip 10.0 --rank-weight 1.0 --inner-fraction 0.0 --coords \
    --seed $SEED --out runs/r3/coords_seed$SEED.json > "$LOG" 2>&1 &
  sleep 3
done
wait
echo "R3 COORDS FINISHED"

# R2 again with the two things the first run left unsettled: one seed, and an optimiser clipped
# at 1.0 while its raw gradient norm reached 4200. Three seeds at a looser clip say whether
# +0.0362 is the representation or the draw.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/r2b
for SEED in 0 1 2; do
  LOG=runs/r2b/seed$SEED.log
  if [ -s "$LOG" ] && grep -q "SUCCESSOR SCORER DONE" "$LOG"; then echo "skip $SEED"; continue; fi
  rm -f "$LOG"
  python -u probes/train_successor.py --cache runs/r1/labels.pkl --epochs 400 --width 64     --learning-rate 1e-3 --clip 10.0 --rank-weight 1.0 --inner-fraction 0.0 --seed $SEED     --out runs/r2b/seed$SEED.json > "$LOG" 2>&1 &
  sleep 3
done
wait
echo "R2B FINISHED"

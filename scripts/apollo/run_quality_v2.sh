# The quality experiment re-run after a fourth audit found two label bugs in the first version.
#
#   * COMMIT was resolved as workspace plus new_chains. COMMIT carries no new_chains; the
#     environment returns the archive entry it names. Every commit at a state after a change was
#     labelled with the wrong embedding.
#   * The control called "incumbent" was the current workspace, not the protected entry, so the
#     baseline changed identity halfway through the episode.
#
# The statistics are also repaired: the inner split is by lineage, arms are joined on the state
# they belong to rather than truncated to a common length, the bootstrap resamples lineages, the
# fresh-read seeds come from a fixed table instead of Python\x27s salted hash, and ranks average
# their ties.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/quality_v2
for FAM in if-dual if-mlp; do
  LOG=runs/quality_v2/$FAM.log
  if [ -s "$LOG" ] && grep -q "TRAIN QUALITY DONE" "$LOG"; then echo "skip $FAM"; continue; fi
  rm -f "$LOG"
  python -u probes/train_quality.py --corpus runs/v1/corpus --family $FAM     --train-lineages 200 --eval-lineages 60 --states 2 --max-candidates 8     --reads 256 --fresh-reads 512 --epochs 400 --learning-rate 3e-4     --weight-decay 1e-4 --inner-fraction 0.25 --patience 30 --seed 0 > "$LOG" 2>&1 &
  sleep 5
done
wait
echo "QUALITY V2 FINISHED"

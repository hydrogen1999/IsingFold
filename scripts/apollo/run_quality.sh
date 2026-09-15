# Step 1 of the V4 plan: learn commit quality directly from labelled candidate pools, then test
# whether it transfers to lineages the model never trained on.
#
# A third audit established that the signal exists within a state and that a freshly initialised
# network fits it to zero regret on the states it was fitted on. What that cannot show is whether
# the fit is a property of embeddings or of those particular states. Training and held-out states
# here come from disjoint lineages, every pick is re-measured on independent reads, and the
# ceiling is the oracle over the same pool measured the same way.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/quality
skipped=0
started=0
for FAM in if-dual if-mlp; do
  LOG=runs/quality/$FAM.log
  # A finished run is one whose log reaches the end marker. Treating the existence of a file as
  # completion is how the previous launcher reported success for a run that had crashed, and how
  # a second invocation of this script printed a completion marker for work it never did.
  if [ -s "$LOG" ] && grep -q "TRAIN QUALITY DONE" "$LOG"; then
    echo "skip $FAM: already complete"; skipped=$((skipped + 1)); continue
  fi
  rm -f "$LOG"
  python -u probes/train_quality.py --corpus runs/v1/corpus --family $FAM \
    --train-lineages 200 --eval-lineages 60 --states 2 --max-candidates 8 \
    --reads 256 --fresh-reads 512 --epochs 200 --learning-rate 3e-4 --seed 0 \
    > "$LOG" 2>&1 &
  started=$((started + 1))
  sleep 5
done
wait
if [ "$started" -gt 0 ]; then
  echo "QUALITY RUN FINISHED, $started trained, $skipped skipped"
else
  echo "QUALITY NOTHING TO DO, $skipped already complete"
fi

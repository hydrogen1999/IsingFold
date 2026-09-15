# The arm the corrected run cannot supply: can the network fit correct labels at all?
#
# With the label bugs fixed and early stopping on, the head sits near zero on the states it was
# fitted on as well as on new ones, and the stopping rule fires at epoch 45. That conflates two
# different diagnoses. This arm removes the stopping rule and the weight decay so the fit is
# unconstrained: if it reaches the ceiling on fitted states, the failure is transfer; if it does
# not, correct labels are harder to fit than the wrong ones were, which is its own finding.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/quality_cap
for FAM in if-mlp if-dual; do
  LOG=runs/quality_cap/$FAM.log
  if [ -s "$LOG" ] && grep -q "TRAIN QUALITY DONE" "$LOG"; then echo "skip $FAM"; continue; fi
  rm -f "$LOG"
  python -u probes/train_quality.py --corpus runs/v1/corpus --family $FAM     --train-lineages 200 --eval-lineages 60 --states 2 --max-candidates 8     --reads 256 --fresh-reads 512 --epochs 400 --learning-rate 3e-4     --weight-decay 0.0 --inner-fraction 0.0 --seed 0 > "$LOG" 2>&1 &
  sleep 5
done
wait
echo "QUALITY CAP FINISHED"

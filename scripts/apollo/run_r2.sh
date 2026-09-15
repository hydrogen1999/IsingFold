# R2 of the fifth audit: a representation ablation, nothing else changed.
#
# The same labels the action head was trained on, read from the same cache, the same states, the
# same references and the same evaluation. What changes is what the model is shown. The action
# head reads the environment observation and reached the oracle exactly on the states it was
# fitted on, regret -0.0000 and rank correlation +0.714, while transferring -0.029 against a
# random pick on new lineages where the ceiling is +0.112. If the successor encoder transfers,
# the difference is the representation.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/r2
LOG=runs/r2/successor.log
if [ -s "$LOG" ] && grep -q "SUCCESSOR SCORER DONE" "$LOG"; then echo "skip: already complete"; exit 0; fi
rm -f "$LOG"
python -u probes/train_successor.py --cache runs/r1/labels.pkl --epochs 400 --width 64   --learning-rate 1e-3 --rank-weight 1.0 --inner-fraction 0.0 --seed 0   --out runs/r2/successor.json > "$LOG" 2>&1
echo "R2 FINISHED"

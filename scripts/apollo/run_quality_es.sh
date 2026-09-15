set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/quality_es
for FAM in if-dual if-mlp; do
  LOG=runs/quality_es/$FAM.log
  if [ -s "$LOG" ] && grep -q "TRAIN QUALITY DONE" "$LOG"; then echo "skip $FAM"; continue; fi
  rm -f "$LOG"
  python -u probes/train_quality.py --corpus runs/v1/corpus --family $FAM     --train-lineages 200 --eval-lineages 60 --states 2 --max-candidates 8     --reads 256 --fresh-reads 512 --epochs 400 --learning-rate 3e-4     --weight-decay 1e-4 --inner-fraction 0.25 --patience 30 --seed 0 > "$LOG" 2>&1 &
  sleep 5
done
wait
echo "QUALITY ES FINISHED"

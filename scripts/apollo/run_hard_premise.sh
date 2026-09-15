# The two measurements the easy corpus could not support, on data where chains are long enough
# for chain integrity to bind: does spending qubits buy quality, and does a scorer that reads the
# compiled program transfer?
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/hard
CACHE=runs/hard/labels.pkl
if [ ! -s "$CACHE" ]; then
  python -u probes/train_quality.py --corpus runs/hard/corpus --family if-mlp \
    --train-lineages 200 --eval-lineages 60 --states 2 --max-candidates 8 \
    --reads 256 --fresh-reads 512 --epochs 1 --loss bce --rank-weight 0.0 \
    --inner-fraction 0.0 --cache "$CACHE" > runs/hard/labels.log 2>&1
fi
python -u probes/frontier_premise.py --cache "$CACHE" --split both \
  > runs/hard/premise.log 2>&1
grep -v "^#" runs/hard/premise.log | tail -30
echo "HARD PREMISE DONE"

# Direction 5 of the relaunch: labels at ten times the lineage count, registered schedule,
# corrected compiler, Pegasus 6, for the data-scale test of the quality surrogate. 2000
# training and 360 held-out lineages, two states each, up to eight candidates.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/large
CACHE=runs/large/pegasus6_registered.pkl
if [ -s "$CACHE" ]; then echo "cache present"; exit 0; fi
python -u probes/train_quality.py --corpus runs/large/pegasus6/corpus --family if-mlp \
  --train-lineages 2000 --eval-lineages 360 --states 2 --max-candidates 8 \
  --reads 256 --fresh-reads 512 --epochs 1 --loss bce --rank-weight 0.0 \
  --inner-fraction 0.0 --qubit-cap 248 --cache "$CACHE" > runs/large/labels.log 2>&1
echo "LARGE LABELS FINISHED"

# Label collection for the diverse corpus, run on goose so apollo keeps the eight representation
# arms. Eight cores here with four already busy, so two workers and two threads each.
#
# The split matters more than the count. Every held-out measurement in this project so far has
# been other coefficient draws of the same shape; this corpus records a structural family per
# lineage, so a split can hold out whole families and ask whether anything transfers across
# structure. That question has never been asked here.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/diverse
CACHE=runs/diverse/labels.pkl
if [ -s "$CACHE" ]; then echo "cache already present"; exit 0; fi
python -u probes/train_quality.py --corpus runs/diverse/corpus --family if-mlp \
  --train-lineages 220 --eval-lineages 70 --states 2 --max-candidates 8 \
  --reads 256 --fresh-reads 512 --epochs 1 --loss bce --rank-weight 0.0 \
  --inner-fraction 0.0 --cache "$CACHE" > runs/diverse/labels.log 2>&1
echo "DIVERSE LABELS DONE"

set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
SEED=${SEED:-0}
OUT=runs/v3/seed$SEED
mkdir -p $OUT/frozen
for F in if-core if-dual if-mlp; do cp -n $OUT/bestof_$F/policy.pt $OUT/frozen/$F.pt || true; done
CK=if-core=$OUT/frozen/if-core.pt,if-dual=$OUT/frozen/if-dual.pt,if-mlp=$OUT/frozen/if-mlp.pt

python -u probes/registered_bar.py --corpus runs/v1/corpus --split test --lineages 90 --k 8 \
  --initializer minorminer --checkpoints "$CK" > $OUT/registered_bar.log 2>&1
echo "BAR DONE"
python -u probes/rank_actions.py --corpus runs/v1/corpus --split validation --lineages 40 \
  --max-candidates 10 --checkpoints "$CK" > $OUT/rank_actions.log 2>&1
echo "RANK DONE"
python -u probes/frontier2.py --corpus runs/v1/corpus --split test --lineages 60 \
  --ladder 2,4,8,12,16,24,40 --rounds 1,2 --draws 8 --checkpoints "$CK" \
  > $OUT/frontier2.log 2>&1
echo "V3 EVAL DONE"

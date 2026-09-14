# Track B, fourth configuration, and the first that answers the criticism runs/external_baseline.log
# made of the previous three. Everything about the model and the training algorithm is unchanged
# from bestof2; the one change is the floor. The episode is now protected by stock minorminer
# instead of the self-contained router, so the policy is learning to improve on the standard
# tool rather than to replace it. The comparison that decides this is the policy's best-of-K
# against minorminer given the same wall clock, which is the comparison the previous
# configuration lost by 0.058 to 0.065.
set -u
cd ~/isingfold_ladder3
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$HOME/isingfold_ladder3/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/bestof3
for F in if-core if-dual if-mlp; do
  OUT=runs/bestof3_${F}_s0
  LOG=runs/bestof3/${F}_s0.log
  if [ -s "$LOG" ]; then echo "skipping $F: $LOG is not empty"; continue; fi
  python -u probes/train_bestof.py --corpus runs/corpus_c4x10 --family $F --out $OUT \
    --rounds 60 --window 10 --lineages 24 --k 8 --epochs 2 --learning-rate 3e-4 --seed 0 \
    --initializer minorminer --mm-tries 10 \
    --eval-every 5 --eval-instances 40 > $LOG 2>&1 &
  sleep 5
done
wait
echo "BESTOF3 DONE"

# Track B, third configuration. Two changes over ladder6, both aimed at the variance L-163
# measured: a batch now holds eight lineages four times each instead of thirty-two lineages
# once, and each lineage's own mean advantage is subtracted inside the batch, so the gradient
# compares actions on one problem instead of comparing problems.
set -u
cd ~/isingfold_ladder3
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$HOME/isingfold_ladder3/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=3
mkdir -p runs/ladder8
for F in if-core if-dual if-mlp; do
  for S in 0 1; do
    python -u probes/train_ladder.py --corpus runs/corpus_c4x10 --out runs/ladder8_${F}_s${S} \
      --families $F --updates 200 --eval-every 10 --eval-instances 40 --episodes 32 --episodes-per-lineage 4 \
      --within-lineage-baseline --ppo-epochs 4 --minibatch 64 \
      --learning-rate 1e-3 --kl-target 0.02 --kl-guard 0.04 --reward-reads 128 \
      --qubit-cap 120 --seeds 1 --seed-start $S --device cuda \
      > runs/ladder8/${F}_s${S}.log 2>&1 &
  done
done
wait
echo done > runs/ladder8/DONE

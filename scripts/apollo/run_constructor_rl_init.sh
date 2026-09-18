# RL on the curriculum hosts, initialised from the imitation prioritiser trained on the
# Pegasus 6 records. Chained on the model file. Same corpora, deadline and evaluation as the
# from-scratch run. This is the legacy35/local-MLP feature ablation; the primary
# contextual constructor is launched by run_independent_constructor.sh.
# The old imitation checkpoint has no lineage provenance; these explicit legacy
# ablations must not be reported as verified held-out validation.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
until [ -s runs/imitation/pegasus6.pt ]; do sleep 300; done
mkdir -p runs/rl
python -u probes/train_constructor_rl.py --corpus runs/small/pegasus3/corpus \
  --actor local --features legacy --value-baseline loo --allow-unverified-init \
  --init runs/imitation/pegasus6.pt --out runs/rl/pegasus3_init.pt --iterations 300 \
  --instances-per-iteration 4 --episodes-per-instance 2 --deadline 30 --eval-every 10 --seed 0 \
  > runs/rl/pegasus3_init.log 2>&1 &
python -u probes/train_constructor_rl.py --corpus runs/small/zephyr2/corpus \
  --actor local --features legacy --value-baseline loo --allow-unverified-init \
  --init runs/imitation/pegasus6.pt --out runs/rl/zephyr2_init.pt --iterations 300 \
  --instances-per-iteration 4 --episodes-per-instance 2 --deadline 30 --eval-every 10 --seed 0 \
  > runs/rl/zephyr2_init.log 2>&1 &
wait
echo "CONSTRUCTOR RL INIT FINISHED"

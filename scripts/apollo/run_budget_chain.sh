# The budget-conditioned chain: retrain the prioritiser on the existing imitation records with
# the budget features, then the RL runs on the curriculum hosts (from scratch and initialised)
# and the deployment of the imitation policy on the full corpora at 600 s.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=4
mkdir -p runs/imitation runs/rl
for H in pegasus6 zephyr4; do
  [ -s runs/imitation/${H}_budget.pt ] && continue
  python -u probes/train_prioritiser.py --dump runs/imitation/${H}_hint.pkl \
    --corpus runs/fill/$H/corpus --epochs 30 --width 64 --out runs/imitation/${H}_budget.pt \
    > runs/imitation/${H}_budget_train.log 2>&1 &
done
wait
export OMP_NUM_THREADS=2
python -u probes/train_constructor_rl.py --corpus runs/small/pegasus3/corpus \
  --out runs/rl/pegasus3_scratch.pt --iterations 300 --instances-per-iteration 4 \
  --episodes-per-instance 2 --deadline 30 --eval-every 10 --seed 0 > runs/rl/pegasus3_scratch.log 2>&1 &
python -u probes/train_constructor_rl.py --corpus runs/small/pegasus3/corpus \
  --init runs/imitation/pegasus6_budget.pt --out runs/rl/pegasus3_init.pt --iterations 300 \
  --instances-per-iteration 4 --episodes-per-instance 2 --deadline 30 --eval-every 10 --seed 0 \
  > runs/rl/pegasus3_init.log 2>&1 &
python -u probes/train_constructor_rl.py --corpus runs/small/zephyr2/corpus \
  --init runs/imitation/pegasus6_budget.pt --out runs/rl/zephyr2_init.pt --iterations 300 \
  --instances-per-iteration 4 --episodes-per-instance 2 --deadline 30 --eval-every 10 --seed 0 \
  > runs/rl/zephyr2_init.log 2>&1 &
for H in pegasus6 zephyr4; do
  python -u probes/train_constructor_rl.py --corpus runs/fill/$H/corpus --init runs/imitation/${H}_budget.pt \
    --out runs/rl/${H}_imitation_deploy.pt --iterations 0 --holdout-fraction 1.0 --deadline 600 \
    > runs/rl/${H}_imitation_deploy.log 2>&1 &
done
wait
echo "BUDGET CHAIN FINISHED"

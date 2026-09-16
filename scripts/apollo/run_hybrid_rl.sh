# RL of the hybrid on the curriculum hosts: the policy places roots, minorminer completes
# within ten seconds during training, sixty at evaluation, reward is success. Hard cells only
# for training (70 percent fill is solved by minorminer alone and gives no gradient).
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/hybrid
python -u probes/train_hybrid_rl.py --corpus runs/small/pegasus3/corpus --init runs/imitation/pegasus6_budget.pt \
  --out runs/hybrid/rl_pegasus3_init.pt --iterations 100 --hard-only --seed 0 > runs/hybrid/rl_pegasus3_init.log 2>&1 &
python -u probes/train_hybrid_rl.py --corpus runs/small/pegasus3/corpus \
  --out runs/hybrid/rl_pegasus3_scratch.pt --iterations 100 --hard-only --seed 0 > runs/hybrid/rl_pegasus3_scratch.log 2>&1 &
python -u probes/train_hybrid_rl.py --corpus runs/small/zephyr2/corpus --init runs/imitation/pegasus6_budget.pt \
  --out runs/hybrid/rl_zephyr2_init.pt --iterations 100 --hard-only --seed 0 > runs/hybrid/rl_zephyr2_init.log 2>&1 &
wait
echo "HYBRID RL FINISHED"

# The hybrid with the fast layout sampler: hundreds of layouts a minute in training, and at
# evaluation as many layouts as the deadline allows, each completed by the router. Same
# corpora, same baseline (the router alone until the deadline), same cells.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/hybrid
python -u probes/train_hybrid_rl.py --corpus runs/small/pegasus3/corpus --init runs/imitation/pegasus6_budget.pt --fast \
  --out runs/hybrid/fast_pegasus3_init.pt --iterations 300 --episodes-per-instance 8 --hard-only --seed 0 \
  > runs/hybrid/fast_pegasus3_init.log 2>&1 &
python -u probes/train_hybrid_rl.py --corpus runs/small/pegasus3/corpus --fast \
  --out runs/hybrid/fast_pegasus3_scratch.pt --iterations 300 --episodes-per-instance 8 --hard-only --seed 0 \
  > runs/hybrid/fast_pegasus3_scratch.log 2>&1 &
python -u probes/train_hybrid_rl.py --corpus runs/small/zephyr2/corpus --init runs/imitation/pegasus6_budget.pt --fast \
  --out runs/hybrid/fast_zephyr2_init.pt --iterations 300 --episodes-per-instance 8 --hard-only --seed 0 \
  > runs/hybrid/fast_zephyr2_init.log 2>&1 &
wait
echo "HYBRID FAST FINISHED"

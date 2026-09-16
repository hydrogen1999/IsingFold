# Hybrid runs after the review of 7044446: the prioritiser is retrained on the existing
# imitation records with the coefficient features, then five runs, three on validity and two
# on quality, all evaluated with the fair protocol (both arms search and select by
# measurement under the same deadline, budget and read cap).
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=4
mkdir -p runs/imitation runs/hybrid
for H in pegasus6 zephyr4; do
  python -u probes/train_prioritiser.py --dump runs/imitation/${H}_hint.pkl \
    --corpus runs/fill/$H/corpus --epochs 30 --width 64 --out runs/imitation/${H}_v3.pt \
    > runs/imitation/${H}_v3_train.log 2>&1 &
done
wait
export OMP_NUM_THREADS=2
COMMON="--fast --iterations 300 --episodes-per-instance 8 --eval-every 20 --seed 0"
python -u probes/train_hybrid_rl.py --corpus runs/small/pegasus3/corpus --init runs/imitation/pegasus6_v3.pt $COMMON --hard-only \
  --out runs/hybrid/v3_valid_pegasus3_init.pt > runs/hybrid/v3_valid_pegasus3_init.log 2>&1 &
python -u probes/train_hybrid_rl.py --corpus runs/small/pegasus3/corpus $COMMON --hard-only \
  --out runs/hybrid/v3_valid_pegasus3_scratch.pt > runs/hybrid/v3_valid_pegasus3_scratch.log 2>&1 &
python -u probes/train_hybrid_rl.py --corpus runs/small/zephyr2/corpus --init runs/imitation/pegasus6_v3.pt $COMMON --hard-only \
  --out runs/hybrid/v3_valid_zephyr2_init.pt > runs/hybrid/v3_valid_zephyr2_init.log 2>&1 &
python -u probes/train_hybrid_rl.py --corpus runs/small/pegasus3/corpus --init runs/imitation/pegasus6_v3.pt $COMMON --objective quality \
  --out runs/hybrid/v3_quality_pegasus3_init.pt > runs/hybrid/v3_quality_pegasus3_init.log 2>&1 &
python -u probes/train_hybrid_rl.py --corpus runs/small/zephyr2/corpus --init runs/imitation/pegasus6_v3.pt $COMMON --objective quality \
  --out runs/hybrid/v3_quality_zephyr2_init.pt > runs/hybrid/v3_quality_zephyr2_init.log 2>&1 &
wait
echo "HYBRID V3 FINISHED"

# The imitation prioritiser as a deployed policy on the full fill corpora: sample episodes
# until a 300 s deadline, return on the first valid one, every instance (holdout fraction 1,
# no training). This is the number to read against the anytime baseline's 300 s column.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/rl
for H in pegasus6 zephyr4; do
  python -u probes/train_constructor_rl.py --corpus runs/fill/$H/corpus --init runs/imitation/$H.pt \
    --out runs/rl/${H}_imitation_deploy.pt --iterations 0 --holdout-fraction 1.0 --deadline 300 \
    > runs/rl/${H}_imitation_deploy.log 2>&1 &
  sleep 2
done
wait
echo "IMITATION DEPLOY FINISHED"

set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1
until grep -q "FILL FINISHED" runs/fill.out 2>/dev/null; do sleep 60; done
for H in pegasus6 zephyr4; do
  python -u probes/fill_budget.py --corpus runs/fill/$H/corpus > runs/fill/${H}_budget.log 2>&1 &
  sleep 2
done
wait
echo "FILL BUDGET FINISHED"

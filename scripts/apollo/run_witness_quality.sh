set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
until grep -q "FILL FINISHED" runs/fill.out 2>/dev/null; do sleep 60; done
for H in pegasus6 zephyr4; do
  [ -s runs/fill/$H/corpus/instances.jsonl ] || { echo "no corpus for $H"; continue; }
  python -u probes/witness_quality.py --corpus runs/fill/$H/corpus > runs/fill/${H}_witness.log 2>&1 &
  sleep 3
done
wait
echo "WITNESS FINISHED"

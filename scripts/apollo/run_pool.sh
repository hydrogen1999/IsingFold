set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
for H in pegasus6 zephyr4; do
  python -u probes/pool_ceiling.py --corpus runs/modern/$H/corpus --lineages 90 --k 8 --alpha 3.0     --qubit-cap 248 > runs/modern/$H/pool.log 2>&1 &
  sleep 3
done
wait
echo "POOL FINISHED"

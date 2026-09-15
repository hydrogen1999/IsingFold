set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/fill
python -u probes/gen_fill_corpus.py --out runs/fill/pegasus6/corpus --host pegasus --host-size 6 > runs/fill/pegasus6.log 2>&1 &
python -u probes/gen_fill_corpus.py --out runs/fill/zephyr4/corpus --host zephyr --host-size 4 > runs/fill/zephyr4.log 2>&1 &
wait
echo "FILL FINISHED"

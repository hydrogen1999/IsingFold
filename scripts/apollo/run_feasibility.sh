set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1
mkdir -p runs/feasibility
python -u probes/feasibility_sweep.py --host pegasus --host-size 6 > runs/feasibility/pegasus6.log 2>&1 &
python -u probes/feasibility_sweep.py --host zephyr --host-size 4 > runs/feasibility/zephyr4.log 2>&1 &
wait
echo "FEASIBILITY FINISHED"

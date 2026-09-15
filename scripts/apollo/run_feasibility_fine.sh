# Second pass of the feasibility sweep: the clique threshold lies between 56 and 64 on both
# hosts, so this walks that gap one variable at a time; dense and scalefree are still at
# full validity at 64, so their range is extended until they break.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1
mkdir -p runs/feasibility
for H in "pegasus 6" "zephyr 4"; do
  set -- $H
  python -u probes/feasibility_sweep.py --host $1 --host-size $2 --families clique \
    --sizes 57,58,59,60,61,62,63 --instances 6 --draws 4 > runs/feasibility/$1$2_clique_fine.log 2>&1 &
  python -u probes/feasibility_sweep.py --host $1 --host-size $2 --families dense,scalefree \
    --sizes 72,80,96,112,128,160,192 --instances 6 --draws 4 > runs/feasibility/$1$2_sparse_far.log 2>&1 &
done
wait
echo "FEASIBILITY FINE FINISHED"

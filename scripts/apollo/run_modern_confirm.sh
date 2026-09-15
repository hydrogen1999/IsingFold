# The same sweep with repeats=1, because repeats=2 took the better of two draws for every grown
# arm while the starting embedding was measured once. All three arms carried that inflation
# equally, so the comparison between them stands, but every "against the start" column was
# inflated by a maximum over two noisy blocks and none of those numbers can be quoted.
#
# One draw per arm, one measurement each, nothing maximised.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
for CFG in "pegasus 6" "zephyr 4"; do
  set -- $CFG
  DIR=runs/modern/${1}${2}
  LOG=$DIR/sweep_confirm.log
  if [ -s "$LOG" ] && grep -q "BUDGET SWEEP DONE" "$LOG"; then echo "skip $1"; continue; fi
  rm -f "$LOG"
  python -u probes/budget_sweep.py --corpus "$DIR/corpus" --lineages 90 --repeats 1 \
    --alphas 3.0,2.2,1.6 --qubit-cap 248 --reads 512 --seed 7 > "$LOG" 2>&1 &
  sleep 3
done
wait
echo "MODERN CONFIRM FINISHED"

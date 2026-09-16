# The qubit-spend sweep under the registered schedule: the premise that more contacts raise
# solve probability, measured under the objective rather than under the auto schedule that
# cancels compression. Same corpora, lineages, alphas and seed as the earlier unbiased
# confirmation, one draw per arm, so the two tables differ only in the schedule.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1
for CFG in "pegasus 6" "zephyr 4"; do
  set -- $CFG
  DIR=runs/modern/${1}${2}
  LOG=$DIR/sweep_registered.log
  if [ -s "$LOG" ] && grep -q "BUDGET SWEEP DONE" "$LOG"; then echo "skip $1"; continue; fi
  python -u probes/budget_sweep.py --corpus "$DIR/corpus" --lineages 90 --repeats 1 \
    --alphas 3.0,2.2,1.6 --qubit-cap 248 --reads 512 --seed 7 > "$LOG" 2>&1 &
  sleep 3
done
wait
echo "SPEND REGISTERED FINISHED"

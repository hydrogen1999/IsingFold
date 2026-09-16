# Tasks 7 and 8 of docs/plans/2026-09-15-plan.md, chained behind the relabel.
#
# Task 7: on identical labels (registered schedule, provenance-checked), identical picks and
# identical assessment seeds, the old encoding (strength ratio times mean|J|) against the
# corrected one (the evaluated program), three model seeds each, per host. The paired
# difference between the two encodings is the number; each arm's own held-out gain is
# printed beside it.
#
# Task 8: label reliability on the held-out split, two independent 512-read blocks per
# candidate, 32 lineages per host.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
until grep -q "RELABEL FINISHED" runs/relabel.out 2>/dev/null; do sleep 120; done
mkdir -p runs/corrected
run () {
  host=$1; arm=$2; seed=$3; shift 3
  LOG=runs/corrected/${host}_${arm}_seed${seed}.log
  if [ -s "$LOG" ] && grep -q "SUCCESSOR SCORER DONE" "$LOG"; then echo "skip $LOG"; return; fi
  python -u probes/train_successor.py --cache runs/relabel/${host}_registered.pkl \
    --epochs 400 --width 64 --learning-rate 1e-3 --clip 10.0 --rank-weight 1.0 \
    --inner-fraction 0.0 --qubit-cap 248 --seed "$seed" \
    --out runs/corrected/${host}_${arm}_seed${seed}.json "$@" > "$LOG" 2>&1 &
  sleep 3
}
for H in pegasus6 zephyr4; do
  [ -s runs/relabel/${H}_registered.pkl ] || { echo "no cache for $H"; continue; }
  for SEED in 0 1 2; do
    run $H corrected $SEED
    run $H legacy    $SEED --legacy-compile
  done
  python -u probes/label_reliability.py --cache runs/relabel/${H}_registered.pkl \
    --lineages 32 --qubit-cap 248 > runs/corrected/${H}_reliability.log 2>&1 &
  sleep 3
done
wait
echo "CORRECTED VS OLD FINISHED"

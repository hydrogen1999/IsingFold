# Task 12b, chained behind the hinted replays. Train the shortlist prioritiser on one host's
# imitation records and measure what matters on the other host: unhinted validity of the
# construction replay with the learned scorer as the generator's preference. The trainer's own
# held-out (a quarter of the instances) reports teacher agreement; the cross-host replay is the
# number that goes on the board, against the unhinted heuristic order's near-zero.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=4
until grep -q "WITNESS REPLAY DONE" runs/fill/pegasus6_replay_hint.log 2>/dev/null \
   && grep -q "WITNESS REPLAY DONE" runs/fill/zephyr4_replay_hint.log 2>/dev/null; do sleep 300; done
mkdir -p runs/imitation
for H in pegasus6 zephyr4; do
  if [ ! -s runs/imitation/${H}.pt ]; then
    python -u probes/train_prioritiser.py --dump runs/imitation/${H}_hint.pkl \
      --corpus runs/fill/$H/corpus --epochs 30 --width 64 --out runs/imitation/${H}.pt \
      > runs/imitation/${H}_train.log 2>&1
  fi
done
python -u probes/witness_replay.py --corpus runs/fill/zephyr4/corpus --scorer runs/imitation/pegasus6.pt \
  > runs/fill/zephyr4_replay_scored_by_pegasus.log 2>&1 &
python -u probes/witness_replay.py --corpus runs/fill/pegasus6/corpus --scorer runs/imitation/zephyr4.pt \
  > runs/fill/pegasus6_replay_scored_by_zephyr.log 2>&1 &
wait
echo "PRIORITISER FINISHED"

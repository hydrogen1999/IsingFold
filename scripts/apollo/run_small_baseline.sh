# The anytime baseline on the curriculum hosts, same corpora (same seed) as the RL run, at
# deadlines that match a policy episode on a small host.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1
mkdir -p runs/small
for HS in "pegasus 3" "zephyr 2"; do
  set -- $HS
  [ -s runs/small/$1$2/corpus/instances.jsonl ] || \
    python -u probes/gen_fill_corpus.py --out runs/small/$1$2/corpus --host $1 --host-size $2 \
      --fills 0.70,0.80,0.85,0.90,0.95 --alphas 3.0,2.0 --per-cell 12 --seed 20260916 \
      > runs/small/$1$2_generate.log 2>&1
  python -u probes/anytime_baseline.py --corpus runs/small/$1$2/corpus --deadlines 10,30,60,120 \
    > runs/small/$1$2_anytime.log 2>&1 &
done
wait
echo "SMALL BASELINE FINISHED"

# Does a valid embedding at the embeddability threshold solve anything? A small corpus per host
# at the last sizes where minorminer still succeeds on most draws, with easy sizes of the same
# families alongside for contrast, then the minorminer pool ceiling at K=4 on each. If the
# threshold cells sit near zero solve probability, a validity win there is a win on problems
# the annealer cannot solve, and nothing should be built for that regime.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/threshold
run () {
  host=$1; size=$2; cap=$3; vars=$4
  D=runs/threshold/$host$size
  mkdir -p $D
  if [ ! -s $D/corpus/instances.jsonl ]; then
    python -u probes/gen_hard_corpus.py --out $D/corpus --host $host --host-size $size \
      --variables $vars --families clique,dense,scalefree --instances 54 \
      --qubit-cap $cap --mm-tries 10 --strength-reads 128 > $D/generate.log 2>&1
  fi
  python -u probes/pool_ceiling.py --corpus $D/corpus --lineages 60 --k 4 --alpha 3.0 \
    --qubit-cap $cap > $D/pool.log 2>&1
}
run pegasus 6 680 "24,60,128,176" &
sleep 5
run zephyr 4 576 "24,58,136,184" &
wait
echo "THRESHOLD QUALITY FINISHED"

# The budget sweep on the hardware that matters.
#
# Everything measured so far ran on Chimera 4, and Chimera is the one topology where the two
# strategic ways of spending a qubit do not exist. Its mean degree is 5.5, so a chain almost never
# has two mutually adjacent members to close a cycle on: growing for redundancy moves the bridge
# fraction from 1.000 to 0.975, which is nothing. On Pegasus 6 the same growth reaches 0.806 and
# on Zephyr 4 it reaches 0.731, and contacts per logical edge nearly double rather than rising a
# tenth. The same clique-16 also costs 21 to 23 qubits there against 41 on Chimera, so there is
# far more room left to spend.
#
# So the finding that spending qubits is harmful was measured where spending them well is
# impossible. This runs the same sweep where it is possible.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/modern
for CFG in "pegasus 6" "zephyr 4"; do
  set -- $CFG
  HOST=$1; SIZE=$2
  DIR=runs/modern/${HOST}${SIZE}
  mkdir -p "$DIR"
  if [ ! -s "$DIR/corpus/manifest.json" ]; then
    python -u probes/gen_hard_corpus.py --out "$DIR/corpus" --host "$HOST" --host-size "$SIZE" \
      --variables 16,18,20 --families clique,dense,scalefree,modular --instances 240 \
      --alpha 0.9 --clause-length 5 --weights 0.4,1.0,2.5 --qubit-cap 248 --mm-tries 20 \
      --seed 20260915 > "$DIR/generate.log" 2>&1
  fi
  grep -oE '\{"kept.*seconds": [0-9]+' "$DIR/generate.log" | head -1
done
for CFG in "pegasus 6" "zephyr 4"; do
  set -- $CFG
  DIR=runs/modern/${1}${2}
  python -u probes/budget_sweep.py --corpus "$DIR/corpus" --lineages 60 --repeats 2 \
    --alphas 3.0,2.2,1.6,1.2 --qubit-cap 248 --reads 512 \
    > "$DIR/sweep.log" 2>&1 &
  sleep 3
done
wait
echo "MODERN SWEEP FINISHED"

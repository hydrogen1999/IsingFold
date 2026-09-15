# A corpus where chains are long enough for the objective to matter.
#
# The old corpus is four independent problems in a trenchcoat: 4.48 components, largest holding
# eleven of a nominal sixteen variables, three spins coupled to nothing, chains one or two qubits
# long, minorminer done in two milliseconds. On data like that chain integrity is almost never
# the binding constraint, and measured over its own candidates spending more qubits correlates
# with worse quality at -0.203, because more qubits there just means a longer chain nobody needed.
#
# Cliques of sixteen on Chimera 4 need chains of four and about forty-four qubits of the hundred
# and twenty allowed, so chain breaks are on the critical path and there is room left to spend
# more qubits if spending them helps. Whether it helps is the thing to measure, not to assume.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/hard
if [ ! -s runs/hard/corpus/manifest.json ]; then
  python -u probes/gen_hard_corpus.py --out runs/hard/corpus --host chimera --host-size 4 \
    --kind clique --variables 16 --instances 600 --alpha 0.9 --clause-length 5 \
    --weights 0.4,1.0,2.5 --qubit-cap 120 --mm-tries 20 --seed 20260915 \
    > runs/hard/generate.log 2>&1
fi
tail -3 runs/hard/generate.log
python -u probes/corpus_report.py --corpus runs/hard/corpus --lineages 48 --headroom-draws 6 \
  > runs/hard/corpus_report.log 2>&1
grep -v "^#" runs/hard/corpus_report.log | tail -14
echo "HARD CORPUS READY"

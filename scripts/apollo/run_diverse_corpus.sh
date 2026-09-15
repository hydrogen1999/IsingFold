# A corpus that varies the thing which decides embedding difficulty, instead of drawing one shape
# six hundred times.
#
# The previous hard corpus was 600 draws of clique-16 on Chimera 4. That tests transfer between
# coefficient draws and calls it generalisation, and it cannot show a resource-quality relation
# that only appears between shapes, because within a clique every embedding costs about the same.
#
# Seven families across three sizes, twenty-one cells filled round-robin. They differ in how
# demand for connectivity is distributed: uniform and high (clique, dense), uniform and low
# (sparse), concentrated in hubs (scalefree), split between blocks (modular), laid out in a plane
# (lattice), or split across two sides (bipartite). The family is recorded in each lineage, so a
# split can hold out whole families and test structural transfer rather than coefficient transfer.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
mkdir -p runs/diverse
if [ ! -s runs/diverse/corpus/manifest.json ]; then
  python -u probes/gen_hard_corpus.py --out runs/diverse/corpus --host chimera --host-size 4 \
    --variables 14,16,18 --instances 630 --alpha 0.9 --clause-length 5 \
    --weights 0.4,1.0,2.5 --qubit-cap 120 --mm-tries 20 --seed 20260915 \
    > runs/diverse/generate.log 2>&1
fi
grep -oE '\{"kept.*\}' runs/diverse/generate.log | head -1
python -u probes/corpus_report.py --corpus runs/diverse/corpus --lineages 63 --headroom-draws 6 \
  > runs/diverse/corpus_report.log 2>&1
grep -v "^#" runs/diverse/corpus_report.log | tail -13
echo "DIVERSE CORPUS READY"

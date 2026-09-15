# Two checks before the threshold regime is trusted. First, whether the threshold is an artefact
# of minorminer's budget: the same sizes at five times the tries. Second, for cliques, whether
# the deterministic clique embedder that ships with minorminer already solves what the random
# one cannot, in which case the clique threshold is not a place a learned embedder can win.
# Alongside, the dense and scalefree thresholds are located to the nearest eight variables.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1
for H in "pegasus 6" "zephyr 4"; do
  set -- $H
  python -u probes/feasibility_sweep.py --host $1 --host-size $2 --families clique \
    --sizes 59,60,61,62,63 --instances 6 --draws 4 --tries 50 > runs/feasibility/$1$2_clique_tries50.log 2>&1 &
  python -u probes/feasibility_sweep.py --host $1 --host-size $2 --families dense \
    --sizes 136,144,152 --instances 6 --draws 4 > runs/feasibility/$1$2_dense_fine.log 2>&1 &
  python -u probes/feasibility_sweep.py --host $1 --host-size $2 --families scalefree \
    --sizes 176,184,200,208 --instances 6 --draws 4 > runs/feasibility/$1$2_scalefree_fine.log 2>&1 &
done
python - > runs/feasibility/busclique.log 2>&1 <<'PY'
import sys, os, time
sys.path.insert(0, "probes")
from gen_hard_corpus import host_graph
from minorminer.busclique import find_clique_embedding
for name, size in (("pegasus", 6), ("zephyr", 4)):
    host = host_graph(name, size)
    for n in (56, 58, 60, 61, 62, 63, 64, 66, 68, 72):
        t0 = time.time()
        try:
            emb = find_clique_embedding(n, host)
            ok = len(emb) == n
            used = sum(len(c) for c in emb.values()) if ok else 0
        except Exception as e:
            ok, used = False, 0
        print("%s%d clique %d  deterministic %s  qubits %d  %.1fs" % (name, size, n, "yes" if ok else "no", used, time.time() - t0), flush=True)
print("BUSCLIQUE DONE")
PY
wait
echo "FEASIBILITY CHECKS FINISHED"

"""Where does minorminer start failing on the current hardware?

Every comparison so far sits in the regime where minorminer finds a valid embedding on every
draw, and there a learned embedder has no measured lever: quality cannot be predicted, and
eight independent draws beat every proposed pool at matched measurements. Near the
embeddability threshold the score changes. When a fraction of draws fail, best-of-K is a
best over the draws that exist, and an embedder with a higher validity rate wins before any
quality question is asked. This sweeps problem size per family and records the validity rate
and wall time of minorminer, so the threshold regime can be located before anything is built
for it.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from gen_hard_corpus import host_graph, logical_graph
from _initializers import minorminer_initializer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--host-size", type=int, required=True)
    ap.add_argument("--families", default="clique,dense,scalefree")
    ap.add_argument("--sizes", default="16,20,24,28,32,36,40,48,56,64")
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    host = host_graph(a.host, a.host_size)
    mm = minorminer_initializer(a.tries)
    rng = np.random.default_rng(a.seed)
    print(json.dumps({"host": "%s%d" % (a.host, a.host_size), "qubits": host.number_of_nodes(),
                      "tries": a.tries, "draws": a.draws, "instances": a.instances}), flush=True)
    print("  %-10s %5s %8s %8s %9s %s" % ("family", "n", "valid", "seconds", "qubits", ""))
    for family in a.families.split(","):
        for n in (int(x) for x in a.sizes.split(",")):
            valid, secs, used = 0, [], []
            total = 0
            for i in range(a.instances):
                g = logical_graph(n, family, rng)
                for j in range(a.draws):
                    total += 1
                    t0 = time.time()
                    ch = mm(g, host, 1000 * n + 17 * i + j)
                    secs.append(time.time() - t0)
                    if ch is not None:
                        valid += 1
                        used.append(sum(len(c) for c in ch.values()))
            rate = valid / max(1, total)
            print("  %-10s %5d %8.2f %8.1f %9s %s"
                  % (family, n, rate, float(np.mean(secs)),
                     ("%.0f" % np.mean(used)) if used else "-",
                     "<- threshold regime" if 0.1 <= rate <= 0.9 else ""), flush=True)
            if valid == 0:
                break
    print("FEASIBILITY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

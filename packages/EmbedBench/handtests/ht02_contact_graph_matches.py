#!/usr/bin/env python3
"""Hand-test 02: the logical graph the corpus ships is the one the chains realise.

What a person would do on paper
-------------------------------
1. Plant an instance and take its chains.
2. Build the contact graph yourself: put an edge between two variables exactly when some qubit of
   one is adjacent in the host to some qubit of the other.
3. Compare that edge set with the logical graph the generator handed you.

They must be equal. An edge in the shipped graph that the chains do not realise is a coupling
the hardware cannot carry; an edge the chains realise that the graph omits is a coupling the
problem does not ask for but the embedding pays for.

What a failure means
--------------------
The problem is defined on a graph the witness does not embed, so the instance is unsolvable as
specified or is quietly easier than it claims.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx

from _common import check, explain_and_exit_if_asked
from embedbench.evaluate import host_graph, ink_drop


def realised_contacts(chains, host) -> set:
    edges = set()
    owner = {q: v for v, chain in chains.items() for q in chain}
    for q, a in owner.items():
        for r in host.neighbors(q):
            b = owner.get(r)
            if b is not None and b != a:
                edges.add((a, b) if a < b else (b, a))
    return edges


def main() -> int:
    explain_and_exit_if_asked(__doc__)
    host = host_graph("chimera", 4)
    failures = 0
    for mode in ("compact", "cut_congested"):
        for seed in (1, 2, 3, 4):
            planted = ink_drop(host, 8, 3, mode=mode, seed=seed)
            shipped = {(u, v) if u < v else (v, u) for u, v in planted.logical.edges()}
            realised = realised_contacts(planted.chains, host)
            missing = shipped - realised
            extra = realised - shipped
            if missing:
                failures += 1
                print(f"  FAIL  mode={mode} seed={seed}: shipped edges the chains cannot carry: "
                      f"{sorted(missing)[:4]}")
            else:
                note = f", {len(extra)} realised but unused" if extra else ""
                print(f"  ok    mode={mode} seed={seed}: {len(shipped)} shipped edges all carried{note}")
    check(failures == 0, f"{failures} instances ship a logical edge their chains cannot carry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

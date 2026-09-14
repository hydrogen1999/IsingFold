#!/usr/bin/env python3
"""Hand-test 05: a large host does not silently produce a problem with nothing in it.

What a person would do on paper
-------------------------------
1. Ask for 16 variables of chain size 4 on a real-sized host, Pegasus 16.
2. Draw where the droplets landed. On a host with thousands of qubits, free growth settles them
   far apart, and two chains that never touch cannot be coupled.
3. Count the edges of the resulting logical graph. A graph with two edges over sixteen variables
   is not a problem: any sampler solves it instantly and every embedding scores 1.0.

The generator confines planting to a connected window a few times the size of what it plants,
which is how a real problem occupies a real processor. This test checks the confinement is on
and that it works: it compares edge counts with the window off and on.

What a failure means
--------------------
A corpus that looks healthy by every summary statistic and contains nothing to solve. This is the
worst failure mode here because it does not announce itself: solve probability saturates at one
and every method ties.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import check, explain_and_exit_if_asked
from embedbench.evaluate import (EvalConfig, _host_window, _instance_seed, host_graph,
                                  ink_drop, instances)

HOST = "pegasus"
SIZE = 16
N_VARS = 16
CHAIN = 4
SEEDS = 3


def planted_edges(window_multiple: float, host) -> list:
    """The logical graph the planting produces, before any Ising problem is put on it."""
    counts = []
    for i in range(SEEDS):
        gseed = _instance_seed(12345, "compact", i)
        blocked = None
        if window_multiple > 0:
            room = int(window_multiple * N_VARS * CHAIN)
            if room < host.number_of_nodes():
                blocked = set(host.nodes()) - _host_window(host, room, gseed)
        counts.append(ink_drop(host, N_VARS, CHAIN, mode="compact", seed=gseed,
                               blocked=blocked).logical.number_of_edges())
    return counts


def usable_instances(window_multiple: float) -> int:
    """How many of those survive to become an Ising problem the corpus can ship."""
    cfg = EvalConfig(topology=HOST, size=SIZE, source="inkdrop", n_vars=N_VARS, chain_size=CHAIN,
                     n=SEEDS, modes="compact", window_multiple=window_multiple, seed=12345)
    return sum(1 for _ in instances(cfg))


def main() -> int:
    explain_and_exit_if_asked(__doc__)
    host = host_graph(HOST, SIZE)
    print(f"  host {HOST} {SIZE}: {host.number_of_nodes()} qubits, planting {N_VARS} variables "
          f"of chain size {CHAIN}")
    free = planted_edges(0.0, host)
    confined = planted_edges(4.0, host)
    print(f"  window off: logical edges {free}, {usable_instances(0.0)} of {SEEDS} usable")
    print(f"  window on : logical edges {confined}, {usable_instances(4.0)} of {SEEDS} usable")
    check(sum(confined) > sum(free),
          "confinement did not increase the number of couplings, so it is not doing its job")
    check(min(confined) >= 8,
          f"even confined, an instance has only {min(confined)} logical edges on {N_VARS} variables")
    check(usable_instances(4.0) == SEEDS,
          "confinement still loses instances to planting failure")
    print(f"  ok    confinement raises the mean from {sum(free)/len(free):.1f} to "
          f"{sum(confined)/len(confined):.1f} edges per instance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

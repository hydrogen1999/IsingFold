#!/usr/bin/env python3
"""Hand-test 01: the planted chains really are a minor embedding.

What a person would do on paper
-------------------------------
1. Ask the generator to plant 8 variables of chain size 3 on a Chimera 4 host.
2. Print the chains. There should be 8 of them.
3. For each chain, trace its qubits in the host graph and confirm you can walk between any two
   of them without leaving the chain: it is connected.
4. Check no qubit appears in two chains.
5. For each edge of the logical graph, find at least one host coupler with one end in each of
   the two chains. That coupler is what will carry the logical coupling.

If all five hold, the planted object is a witness: an embedding that exists by construction, so
any embedder that fails to find one on this instance has failed, not the instance.

What a failure means
--------------------
The corpus ships a "witness" that is not one. Every label derived from it, including any claim
about an optimal qubit count, is void.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import chains_are_valid_embedding, check, explain_and_exit_if_asked
from embedbench.evaluate import host_graph, ink_drop


def main() -> int:
    explain_and_exit_if_asked(__doc__)
    host = host_graph("chimera", 4)
    failures = 0
    for mode in ("compact", "elongated", "cut_congested", "near_capacity"):
        for seed in (1, 2, 3):
            planted = ink_drop(host, 8, 3, mode=mode, seed=seed)
            reasons = chains_are_valid_embedding(planted.chains, planted.logical, host)
            if reasons:
                failures += 1
                print(f"  FAIL  mode={mode} seed={seed}")
                for reason in reasons[:5]:
                    print(f"        {reason}")
            else:
                print(f"  ok    mode={mode} seed={seed}: {len(planted.chains)} chains, "
                      f"{planted.qubits_used} qubits, longest {planted.max_chain_length}, "
                      f"{planted.logical.number_of_edges()} logical edges")
    check(failures == 0, f"{failures} planted embeddings are not embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

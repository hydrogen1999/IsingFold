"""A Context for a whole modern host.

The registered terminal reserve encodes a COMMIT-only support of at most 32 features per qubit
of cap, sized for the 120-qubit Chimera experiments. Pegasus 6 has 680 qubits and Zephyr 4 has
576, so a Context at those caps needs the reserve's feature work raised in step; nothing else
in the registry depends on the cap.
"""
from dataclasses import replace

from isingfold.rl.contracts import RESERVE, Context


def host_context(qubit_cap: int, **kw) -> Context:
    work = max(RESERVE.feature_work, 32 * (qubit_cap + 64))
    return Context(qubit_cap=qubit_cap, reserve=replace(RESERVE, feature_work=work), **kw)

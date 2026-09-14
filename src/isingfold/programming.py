"""The coefficient-programming map Pi(phi). Implements eq. (eq:prog) of architecture/SPEC.tex.

    h^phys_iq = h_i / |C_i|            for q in C_i
    J^phys_qr = J_ij / |rho_ij|        for (q,r) in rho_ij
    a_q       = |h^phys_iq| + sum_r |J^phys_qr|

This rule is fixed because a_q is *not* determined by phi alone. A logical coupling
realised by three couplers can be programmed as (J,0,0) or (J/3,J/3,J/3), and the two give
different a_q and so different d_i(S). Any other rule is admissible but must be declared
with the result it produced.

a_q is the *exterior* drive on a qubit. The chain's own couplers E^ch_i carry the chain
strength F, which is what eq. (1) compares the drive against, so they contribute nothing
to a_q. Folding them in would make the quantity compare F with itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from isingfold.embedding import Coupler, Embedding, LogicalProblem, Node, Qubit


@dataclass(frozen=True)
class PhysicalCoefficients:
    """The output of Pi(phi)."""

    h_phys: dict[Qubit, float]
    j_phys: dict[Coupler, float]
    load: dict[Qubit, float]
    """a_q, the per-site exterior load."""

    def chain_load(self, embedding: Embedding, i: Node) -> float:
        """v_i(C_i), the total exterior drive the chain carries."""
        return sum(self.load[q] for q in embedding.chains[i])


def program(embedding: Embedding, problem: LogicalProblem) -> PhysicalCoefficients:
    """Apply Pi(phi) to one embedding of one logical problem."""
    h_phys: dict[Qubit, float] = {}
    for i, chain in embedding.chains.items():
        share = problem.h.get(i, 0.0) / len(chain)
        for q in chain:
            h_phys[q] = share

    j_phys: dict[Coupler, float] = {}
    for edge, couplers in embedding.contacts.items():
        if not couplers:
            # An unrealised logical edge. The state is legal during search (S_relaxed) and
            # illegal at the end (S_valid); either way it programs no coupling, and
            # silently dropping it here is the honest reading of eq. (2).
            continue
        share = problem.j[edge] / len(couplers)
        for c in couplers:
            j_phys[c] = share

    load: dict[Qubit, float] = {q: abs(v) for q, v in h_phys.items()}
    for (q, r), w in j_phys.items():
        load[q] = load.get(q, 0.0) + abs(w)
        load[r] = load.get(r, 0.0) + abs(w)

    return PhysicalCoefficients(h_phys=h_phys, j_phys=j_phys, load=load)

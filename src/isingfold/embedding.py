"""The objects of Section 1 of architecture/SPEC.tex: a logical problem and an embedding.

An embedding is phi = ({(C_i, E^ch_i)}, rho): a chain per logical variable, the chain
couplers actually programmed, and the couplers realising each logical edge.

E^ch_i is kept separate from E_H[C_i] on purpose. The specification is explicit that
c_i(S) counts *programmed* couplers, and that the two sets coincide here only because this
implementation programs every internal coupler rather than a spanning tree. Making that a
default rather than an assumption is what lets intra-chain redundancy have any effect, and
what lets a spanning-tree variant be tested later without touching eq. (1).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import networkx as nx

Qubit = int
Node = int
Coupler = tuple[Qubit, Qubit]
LogicalEdge = tuple[Node, Node]


def _key(a: int, b: int) -> tuple[int, int]:
    """Undirected pairs are stored one way round so lookups never depend on orientation."""
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True)
class LogicalProblem:
    """G_L = (V_L, E_L, h, J)."""

    graph: nx.Graph
    h: Mapping[Node, float]
    j: Mapping[LogicalEdge, float]

    @classmethod
    def from_dicts(
        cls, h: Mapping[Node, float], j: Mapping[LogicalEdge, float]
    ) -> LogicalProblem:
        g = nx.Graph()
        g.add_nodes_from(h)
        norm: dict[LogicalEdge, float] = {}
        for (u, v), w in j.items():
            if u == v:
                raise ValueError(f"self-coupling on node {u} is not a logical edge")
            norm[_key(u, v)] = float(w)
            g.add_edge(u, v)
        return cls(graph=g, h={k: float(v) for k, v in h.items()}, j=norm)

    @property
    def n(self) -> int:
        return self.graph.number_of_nodes()

    def coupling(self, u: Node, v: Node) -> float:
        return self.j[_key(u, v)]


@dataclass(frozen=True)
class Embedding:
    """phi = ({(C_i, E^ch_i)}, rho)."""

    chains: Mapping[Node, frozenset[Qubit]]
    chain_edges: Mapping[Node, frozenset[Coupler]]
    contacts: Mapping[LogicalEdge, frozenset[Coupler]]
    host: nx.Graph = field(repr=False)

    @classmethod
    def from_chains(
        cls,
        chains: Mapping[Node, Iterable[Qubit]],
        host: nx.Graph,
        problem: LogicalProblem,
        chain_edges: Mapping[Node, Iterable[Coupler]] | None = None,
    ) -> Embedding:
        """Derive E^ch and rho from raw chains, as a minorminer result gives them.

        By default every host coupler induced on C_i is programmed. Pass `chain_edges` to
        program a subset, for instance a spanning tree.
        """
        cs = {i: frozenset(qs) for i, qs in chains.items()}
        for i, c in cs.items():
            if not c:
                raise ValueError(f"chain of variable {i} is empty")
            missing = c - set(host.nodes)
            if missing:
                raise ValueError(f"chain {i} uses qubits absent from the host: {sorted(missing)}")

        if chain_edges is None:
            ce = {
                i: frozenset(_key(*e) for e in host.subgraph(c).edges())
                for i, c in cs.items()
            }
        else:
            ce = {i: frozenset(_key(*e) for e in es) for i, es in chain_edges.items()}
            for i, es in ce.items():
                for a, b in es:
                    if a not in cs[i] or b not in cs[i]:
                        raise ValueError(f"E^ch_{i} carries {(a, b)}, not internal to the chain")
                    if not host.has_edge(a, b):
                        raise ValueError(f"E^ch_{i} carries {(a, b)}, absent from the host")

        owner: dict[Qubit, Node] = {q: i for i, c in cs.items() for q in c}
        contacts: dict[LogicalEdge, frozenset[Coupler]] = {}
        for u, v in problem.graph.edges():
            if u not in cs or v not in cs:
                continue
            found = {
                _key(q, r)
                for q in cs[u]
                for r in host.neighbors(q)
                if owner.get(r) == v
            }
            contacts[_key(u, v)] = frozenset(found)
        return cls(chains=cs, chain_edges=ce, contacts=contacts, host=host)

    def chain_graph(self, i: Node) -> nx.Graph:
        """G^ch_i = (C_i, E^ch_i), the graph eq. (1) cuts."""
        g = nx.Graph()
        g.add_nodes_from(self.chains[i])
        g.add_edges_from(self.chain_edges[i])
        return g

    def r_prog(self, i: Node) -> int:
        """Programmed intra-chain redundancy, |E^ch_i| - |C_i| + 1."""
        return len(self.chain_edges[i]) - len(self.chains[i]) + 1

    def r_avail(self, i: Node) -> int:
        """What the hardware would allow, |E_H[C_i]| - |C_i| + 1."""
        induced = self.host.subgraph(self.chains[i]).number_of_edges()
        return induced - len(self.chains[i]) + 1

    @property
    def qubits_used(self) -> int:
        """Q."""
        return sum(len(c) for c in self.chains.values())

    @property
    def max_chain_length(self) -> int:
        """L_max."""
        return max(len(c) for c in self.chains.values())

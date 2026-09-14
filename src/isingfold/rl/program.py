"""Physical programs at the four registered strengths, autoscaling and exact faithfulness.

Spec: Rev2 section 2.2-2.3 (baseline programming policy, autoscaling, precision) and
MODEL_SPEC Eq. (3d) and (17). The compiler is deterministic and independent of any learned
score; the checker verifies coefficient *sums*, so no enumeration over logical assignments
is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

import networkx as nx

from isingfold.embedding import LogicalProblem

Node = Hashable
Qubit = Hashable
Coupler = tuple[Qubit, Qubit]


def _key(a: Qubit, b: Qubit) -> Coupler:
    return (a, b) if str(a) <= str(b) else (b, a)


def instance_scale(problem: LogicalProblem, epsilon: float = 1e-6) -> float:
    """S_I of MODEL_SPEC Eq. (17): the RMS coefficient scale of the logical instance."""

    n = problem.n
    m = len(problem.j)
    total = sum(v * v for v in problem.h.values()) + sum(v * v for v in problem.j.values())
    return max(epsilon, math.sqrt(total / max(1, n + m)))


def strength_registry(problem: LogicalProblem, ratios: Iterable[float], epsilon: float = 1e-6) -> tuple[float, ...]:
    """The four registered strengths ``F_j = r_j S_I``, frozen before any learning."""

    scale = instance_scale(problem, epsilon)
    return tuple(float(r) * scale for r in ratios)


@dataclass(frozen=True)
class Program:
    """One compiled physical program ``Pi_{phi,F}``: what a sampler actually receives."""

    strength: float
    strength_index: int
    scale: float
    """``a_{phi,F}``, the shrink-only global rescaling of Rev2 Eq. (scale)."""
    h_phys: Mapping[Qubit, float]
    j_phys: Mapping[Coupler, float]
    chain_edges: Mapping[Node, frozenset[Coupler]]
    contact_counts: Mapping[tuple[Node, Node], int]
    offset: float
    """The state-independent ``-F sum_i |E_H[C_i]|`` term of Rev2 Eq. (program-identity)."""

    @property
    def qubits(self) -> int:
        return len(self.h_phys)

    def max_field(self) -> float:
        return max((abs(v) for v in self.h_phys.values()), default=0.0)

    def max_coupling(self) -> float:
        return max((abs(v) for v in self.j_phys.values()), default=0.0)


def compile_program(
    chains: Mapping[Node, frozenset[Qubit]],
    host: nx.Graph,
    problem: LogicalProblem,
    strength: float,
    strength_index: int,
    field_limit: float = 4.0,
    coupler_limit: float = 2.0,
) -> Program:
    """Divide fields over branch sets, couplings over realised contacts, ferromagnet inside.

    Rev2 section 2.2. The embedding must be disjoint; a relaxed overlapping workspace has no
    consistent physical Hamiltonian and its coefficient channels are missing by contract.
    """

    # Every loop below walks a frozenset of labels, and the insertion order it produces becomes
    # the variable order of the compiled program and then of the sampler. A frozenset of strings
    # iterates in per-process hash order, so the same embedding compiled in two processes gave
    # two different variable orders and, with the seed pinned and the digest identical, two
    # different measured utilities. Sorting by the label's text fixes a canonical order that does
    # not depend on the process, and changes nothing else: the coefficients are identical, only
    # the order in which they are written down.
    order = sorted(chains, key=str)

    owner: dict[Qubit, Node] = {}
    for i in order:
        for q in sorted(chains[i], key=str):
            if q in owner:
                raise ValueError("cannot compile an overlapping workspace")
            owner[q] = i

    h_pre: dict[Qubit, float] = {}
    for i in order:
        chain = chains[i]
        share = problem.h.get(i, 0.0) / len(chain)
        for q in sorted(chain, key=str):
            h_pre[q] = share

    chain_edges: dict[Node, frozenset[Coupler]] = {}
    j_pre: dict[Coupler, float] = {}
    for i in order:
        chain = chains[i]
        edges = frozenset(_key(a, b) for a, b in host.subgraph(chain).edges())
        chain_edges[i] = edges
        for e in sorted(edges, key=str):
            j_pre[e] = j_pre.get(e, 0.0) - strength

    contact_counts: dict[tuple[Node, Node], int] = {}
    for (u, v), coupling in problem.j.items():
        if u not in chains or v not in chains:
            continue
        contacts = [
            _key(q, r)
            for q in sorted(chains[u], key=str)
            for r in host.neighbors(q)
            if owner.get(r) == v
        ]
        contact_counts[(u, v) if str(u) <= str(v) else (v, u)] = len(contacts)
        if not contacts:
            continue
        share = coupling / len(contacts)
        for e in contacts:
            j_pre[e] = j_pre.get(e, 0.0) + share

    scale = _autoscale(h_pre, j_pre, field_limit, coupler_limit)
    h_phys = {q: scale * v for q, v in h_pre.items()}
    j_phys = {e: scale * v for e, v in j_pre.items()}
    # ``h_phys`` and ``j_phys`` are the actual post-autoscale coefficients.  Keep the
    # aligned-chain constant in those same units; storing the pre-scale constant next to
    # post-scale coefficients makes the program identity internally inconsistent.
    offset = -scale * strength * sum(len(e) for e in chain_edges.values())
    return Program(
        strength=strength,
        strength_index=strength_index,
        scale=scale,
        h_phys=h_phys,
        j_phys=j_phys,
        chain_edges=chain_edges,
        contact_counts=contact_counts,
        offset=offset,
    )


def _autoscale(
    h_pre: Mapping[Qubit, float],
    j_pre: Mapping[Coupler, float],
    field_limit: float,
    coupler_limit: float,
) -> float:
    """Rev2 Eq. (scale): shrink only, never amplify, and never a second time."""

    worst_h = max((abs(v) for v in h_pre.values()), default=0.0)
    worst_j = max((abs(v) for v in j_pre.values()), default=0.0)
    denom = max(1.0, worst_h / field_limit if field_limit > 0 else 0.0,
                worst_j / coupler_limit if coupler_limit > 0 else 0.0)
    return 1.0 / denom


def compile_registry(
    chains: Mapping[Node, frozenset[Qubit]],
    host: nx.Graph,
    problem: LogicalProblem,
    strengths: Iterable[float],
    field_limit: float = 4.0,
    coupler_limit: float = 2.0,
) -> tuple[Program, ...]:
    return tuple(
        compile_program(chains, host, problem, f, j, field_limit, coupler_limit)
        for j, f in enumerate(strengths)
    )


@dataclass(frozen=True)
class FaithfulnessReport:
    ok: bool
    field_error: float
    coupling_error: float
    undeclared_terms: int
    disconnected_chains: tuple[Node, ...]
    range_ok: bool
    support_errors: tuple[str, ...] = ()
    internal_error: float = 0.0
    offset_error: float = 0.0
    finite_ok: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "field_error": self.field_error,
            "coupling_error": self.coupling_error,
            "undeclared_terms": self.undeclared_terms,
            "disconnected_chains": [str(c) for c in self.disconnected_chains],
            "range_ok": self.range_ok,
            "support_errors": list(self.support_errors),
            "internal_error": self.internal_error,
            "offset_error": self.offset_error,
            "finite_ok": self.finite_ok,
        }


def check_faithfulness(
    program: Program,
    chains: Mapping[Node, frozenset[Qubit]],
    host: nx.Graph,
    problem: LogicalProblem,
    field_limit: float = 4.0,
    coupler_limit: float = 2.0,
    tolerance: float = 1e-9,
) -> FaithfulnessReport:
    """MODEL_SPEC Eq. (3d): programmed sums reproduce ``a_j h_i`` and ``a_j J_ik`` exactly.

    Also checks that the programmed internal edges connect every chain, that no undeclared
    inter-chain coupling was introduced, and that the actual coefficients fit the range.
    """

    owner = {q: i for i, chain in chains.items() for q in chain}
    a = program.scale
    support_errors: list[str] = []
    claimed = set(owner)
    if set(program.h_phys) != claimed:
        support_errors.append("physical field support differs from claimed qubits")
    if set(program.chain_edges) != set(chains):
        support_errors.append("chain-edge records do not cover the logical domain")

    expected_chain_edges: dict[Node, frozenset[Coupler]] = {}
    expected_h: dict[Qubit, float] = {}
    expected_j: dict[Coupler, float] = {}
    for node, chain in chains.items():
        expected_chain_edges[node] = frozenset(_key(q, r) for q, r in host.subgraph(chain).edges())
        if program.chain_edges.get(node, frozenset()) != expected_chain_edges[node]:
            support_errors.append(f"programmed chain edges differ for {node!r}")
        for qubit in chain:
            expected_h[qubit] = a * problem.h.get(node, 0.0) / len(chain)
        for edge in expected_chain_edges[node]:
            expected_j[edge] = -a * program.strength

    expected_contacts: dict[tuple[Node, Node], int] = {}
    for (left, right), coupling in problem.j.items():
        pair = (left, right) if str(left) <= str(right) else (right, left)
        contacts = {
            _key(q, r)
            for q in chains.get(left, frozenset())
            for r in chains.get(right, frozenset())
            if q != r and host.has_edge(q, r)
        }
        expected_contacts[pair] = len(contacts)
        if contacts:
            share = a * coupling / len(contacts)
            for edge in contacts:
                expected_j[edge] = expected_j.get(edge, 0.0) + share
    if dict(program.contact_counts) != expected_contacts:
        support_errors.append("contact-count receipt differs from active hardware contacts")

    for edge in program.j_phys:
        q, r = edge
        if edge != _key(q, r) or q == r or not host.has_edge(q, r):
            support_errors.append(f"physical coupling {edge!r} is not an active canonical edge")
        if q not in claimed or r not in claimed:
            support_errors.append(f"physical coupling {edge!r} leaves claimed support")
    unexpected_fields = set(program.h_phys) - set(expected_h)
    unexpected_couplers = set(program.j_phys) - set(expected_j)
    missing_couplers = set(expected_j) - set(program.j_phys)
    undeclared = len(unexpected_fields) + len(unexpected_couplers)
    if missing_couplers:
        support_errors.append("required physical couplers are missing")

    field_error = max(
        (abs(program.h_phys.get(q, 0.0) - value) for q, value in expected_h.items()),
        default=0.0,
    )
    coupling_error = max(
        (abs(program.j_phys.get(edge, 0.0) - value) for edge, value in expected_j.items()),
        default=0.0,
    )
    internal_error = max(
        (
            abs(program.j_phys.get(edge, 0.0) + a * program.strength)
            for edges in expected_chain_edges.values()
            for edge in edges
        ),
        default=0.0,
    )
    expected_offset = -a * program.strength * sum(
        len(edges) for edges in expected_chain_edges.values()
    )
    offset_error = abs(program.offset - expected_offset)
    disconnected = tuple(
        node
        for node, chain in chains.items()
        if _chain_is_broken(chain, program.chain_edges.get(node, frozenset()))
    )
    numeric_values = [
        program.strength,
        program.scale,
        program.offset,
        *program.h_phys.values(),
        *program.j_phys.values(),
    ]
    finite_ok = all(math.isfinite(float(value)) for value in numeric_values)
    range_ok = (
        finite_ok
        and 0.0 < program.scale <= 1.0 + tolerance
        and program.strength > 0.0
        and program.max_field() <= field_limit + tolerance
        and program.max_coupling() <= coupler_limit + tolerance
    )
    ok = (
        field_error <= tolerance
        and coupling_error <= tolerance
        and internal_error <= tolerance
        and offset_error <= tolerance
        and undeclared == 0
        and not disconnected
        and range_ok
        and finite_ok
        and not support_errors
    )
    return FaithfulnessReport(
        ok=ok,
        field_error=field_error,
        coupling_error=coupling_error,
        undeclared_terms=undeclared,
        disconnected_chains=disconnected,
        range_ok=range_ok,
        support_errors=tuple(dict.fromkeys(support_errors)),
        internal_error=internal_error,
        offset_error=offset_error,
        finite_ok=finite_ok,
    )


def _chain_is_broken(chain: frozenset[Qubit], edges: frozenset[Coupler]) -> bool:
    if len(chain) <= 1:
        return False
    g = nx.Graph()
    g.add_nodes_from(chain)
    g.add_edges_from(edges)
    return not nx.is_connected(g)


def program_features(
    program: Program,
    chains: Mapping[Node, frozenset[Qubit]],
    problem: LogicalProblem,
) -> dict[str, float]:
    """``Pi_phi`` for the frozen strength selector: deployable program facts only.

    Contains no ground energy, no planted witness, no success count and no evaluator key
    (Rev2 Eq. strength-selector).
    """

    lengths = [len(c) for c in chains.values()]
    n_chains = max(1, len(lengths))
    couplings = [abs(v) for v in problem.j.values()] or [0.0]
    contacts = list(program.contact_counts.values()) or [0]
    return {
        "strength": program.strength,
        "scale": program.scale,
        "qubits": float(sum(lengths)),
        "max_chain": float(max(lengths, default=0)),
        "mean_chain": float(sum(lengths) / n_chains),
        "single_qubit_fraction": float(sum(1 for length in lengths if length == 1) / n_chains),
        "max_field": program.max_field(),
        "max_coupling": program.max_coupling(),
        "mean_contacts": float(sum(contacts) / max(1, len(contacts))),
        "single_contact_fraction": float(sum(1 for c in contacts if c == 1) / max(1, len(contacts))),
        "strength_over_jmax": program.strength / max(1e-12, max(couplings)),
        "chain_edges": float(sum(len(e) for e in program.chain_edges.values())),
    }

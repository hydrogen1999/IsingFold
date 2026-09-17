"""The four meanings of validity, checked exactly and independently of any learned score.

Spec: MODEL_SPEC section 2.3. ``p_search`` admits a live workspace (overlap allowed inside
the envelope); ``p_embed`` is the graph predicate; ``p_return`` additionally requires all
four programs to compile faithfully. The validator reconstructs everything from the
serialised assignment: it never trusts a cached Boolean from the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import weakref

import networkx as nx

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context, OverlapProfile, WorkVector
from isingfold.rl.program import (
    FaithfulnessReport,
    Program,
    check_faithfulness,
    compile_registry,
    strength_registry,
)

Node = Hashable
Qubit = Hashable


@dataclass(frozen=True)
class ValidationReceipt:
    """An auditable record an observer can re-check without trusting the model."""

    valid: bool
    reasons: tuple[str, ...]
    qubits: int
    max_chain: int
    unrealized_demands: int
    programs: tuple[FaithfulnessReport, ...] = ()
    strengths: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "qubits": self.qubits,
            "max_chain": self.max_chain,
            "unrealized_demands": self.unrealized_demands,
            "strengths": list(self.strengths),
            "programs": [p.as_dict() for p in self.programs],
        }


# Per-host memo of chain connectivity and of chain-pair contacts. A search state validates
# every chain on every candidate's dry run; between candidates only the affected chains
# change, so the answers for the others are the same. Keyed by the host graph object (never
# mutated once built) and the exact frozen chains; bounded so a long run cannot grow it
# without limit. Results are identical to the direct computation by construction.
_CONNECTED_MEMO: "weakref.WeakKeyDictionary[nx.Graph, dict]" = weakref.WeakKeyDictionary()
_CONTACT_MEMO: "weakref.WeakKeyDictionary[nx.Graph, dict]" = weakref.WeakKeyDictionary()
_MEMO_LIMIT = 200_000


def _memo(table: "weakref.WeakKeyDictionary", host: nx.Graph) -> dict:
    try:
        cache = table.get(host)
    except TypeError:
        return {}
    if cache is None:
        cache = {}
        try:
            table[host] = cache
        except TypeError:
            pass
    if len(cache) > _MEMO_LIMIT:
        cache.clear()
    return cache


def chain_is_connected(chain: frozenset[Qubit], host: nx.Graph) -> bool:
    """``nx.is_connected(host.subgraph(chain))`` for a nonempty chain inside the host,
    by a breadth-first walk over the adjacency dict, memoised per host and chain."""
    chain = frozenset(chain)
    if len(chain) <= 1:
        return True
    cache = _memo(_CONNECTED_MEMO, host)
    hit = cache.get(chain)
    if hit is not None:
        return hit
    adj = host._adj
    start = next(iter(chain))
    seen = {start}
    stack = [start]
    while stack:
        q = stack.pop()
        for r in adj[q]:
            if r in chain and r not in seen:
                seen.add(r)
                stack.append(r)
    hit = len(seen) == len(chain)
    cache[chain] = hit
    return hit


def chains_touch(cu: frozenset[Qubit], cv: frozenset[Qubit], host: nx.Graph) -> bool:
    """Whether some qubit of ``cu`` and some *other* qubit of ``cv`` share a host edge,
    memoised per host and chain pair."""
    cu, cv = frozenset(cu), frozenset(cv)
    cache = _memo(_CONTACT_MEMO, host)
    key = (cu, cv) if len(cu) <= len(cv) else (cv, cu)
    hit = cache.get(key)
    if hit is not None:
        return hit
    small, big = key
    adj = host._adj
    hit = any(r != q and r in big for q in small if q in adj for r in adj[q])
    cache[key] = hit
    return hit


def unrealized_demands(
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    host: nx.Graph,
) -> int:
    """``U(phi)`` of MODEL_SPEC Eq. (3b). Sharing a qubit is never a contact."""

    count = 0
    for u, v in logical.edges():
        cu, cv = chains.get(u, frozenset()), chains.get(v, frozenset())
        if not cu or not cv:
            count += 1
            continue
        if not chains_touch(cu, cv, host):
            count += 1
    return count


def occupancy(chains: Mapping[Node, frozenset[Qubit]]) -> dict[Qubit, int]:
    """``o(q) = |O(q)|``: every logical claimant of a hardware vertex, not just the first."""

    out: dict[Qubit, int] = {}
    for chain in chains.values():
        for q in chain:
            out[q] = out.get(q, 0) + 1
    return out


def excess_occupancy(chains: Mapping[Node, frozenset[Qubit]]) -> int:
    return sum(o - 1 for o in occupancy(chains).values() if o > 1)


def chains_are_connected(
    chains: Mapping[Node, frozenset[Qubit]], host: nx.Graph
) -> tuple[Node, ...]:
    bad = []
    adj = host._adj
    for i, chain in chains.items():
        if not chain:
            continue
        if any(q not in adj for q in chain):
            bad.append(i)
            continue
        if len(chain) > 1 and not chain_is_connected(chain, host):
            bad.append(i)
    return tuple(bad)


def p_search(
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    host: nx.Graph,
    qubit_cap: int,
    overlap: OverlapProfile,
    remaining: WorkVector | None = None,
    allow_empty: bool = False,
) -> ValidationReceipt:
    """Search-state admissibility. Neither ``U = 0`` nor disjointness is required here."""

    reasons: list[str] = []
    if set(chains) != set(logical.nodes()):
        reasons.append("every logical id needs exactly one branch-set record")
    empties = [i for i, c in chains.items() if not c]
    if empties and not allow_empty:
        reasons.append(f"{len(empties)} empty chains outside construction")
    disconnected = chains_are_connected(chains, host)
    if disconnected:
        reasons.append(f"disconnected or off-host chains: {[str(i) for i in disconnected][:4]}")

    occ = occupancy(chains)
    if occ and max(occ.values()) > overlap.max_occupancy:
        reasons.append(f"occupancy {max(occ.values())} exceeds envelope {overlap.max_occupancy}")
    excess = sum(o - 1 for o in occ.values() if o > 1)
    if excess > overlap.excess_cap(qubit_cap):
        reasons.append(f"claim debt {excess} exceeds {overlap.excess_cap(qubit_cap)}")
    unique = len(occ)
    if unique > qubit_cap:
        reasons.append(f"unique qubits {unique} exceed cap {qubit_cap}")
    if remaining is not None and not remaining.is_nonnegative:
        reasons.append("a binding budget coordinate is negative")

    lengths = [len(c) for c in chains.values()] or [0]
    return ValidationReceipt(
        valid=not reasons,
        reasons=tuple(reasons),
        qubits=unique,
        max_chain=max(lengths),
        unrealized_demands=unrealized_demands(chains, logical, host),
    )


def p_embed(
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    host: nx.Graph,
    qubit_cap: int,
) -> ValidationReceipt:
    """MODEL_SPEC Eq. (3c): complete, connected, disjoint, all demands realised, in cap."""

    reasons: list[str] = []
    if set(chains) != set(logical.nodes()):
        reasons.append("branch sets do not cover the logical domain")
    if any(not c for c in chains.values()):
        reasons.append("an empty branch set")
    disconnected = chains_are_connected(chains, host)
    if disconnected:
        reasons.append(f"disconnected or off-host chains: {[str(i) for i in disconnected][:4]}")
    occ = occupancy(chains)
    if occ and max(occ.values()) > 1:
        reasons.append("branch sets are not pairwise disjoint")
    u = unrealized_demands(chains, logical, host)
    if u:
        reasons.append(f"{u} logical demands without a physical contact")
    if len(occ) > qubit_cap:
        reasons.append(f"unique qubits {len(occ)} exceed cap {qubit_cap}")
    lengths = [len(c) for c in chains.values()] or [0]
    return ValidationReceipt(
        valid=not reasons,
        reasons=tuple(reasons),
        qubits=len(occ),
        max_chain=max(lengths),
        unrealized_demands=u,
    )


def p_return(
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    host: nx.Graph,
    problem: LogicalProblem,
    ctx: Context,
) -> tuple[ValidationReceipt, tuple[Program, ...]]:
    """Return admissibility: ``p_embed`` plus four faithful, in-range compiled programs."""

    receipt = p_embed(chains, logical, host, ctx.qubit_cap)
    problem_nodes = set(problem.graph.nodes())
    logical_nodes = set(logical.nodes())
    problem_edges = {
        frozenset((left, right)) for left, right in problem.graph.edges()
    }
    logical_edges = {frozenset((left, right)) for left, right in logical.edges()}
    support_reasons = list(receipt.reasons)
    if problem_nodes != logical_nodes:
        support_reasons.append("logical graph and problem node domains disagree")
    if problem_edges != logical_edges:
        support_reasons.append("logical graph and problem coupling support disagree")
    if support_reasons != list(receipt.reasons):
        receipt = ValidationReceipt(
            valid=False,
            reasons=tuple(support_reasons),
            qubits=receipt.qubits,
            max_chain=receipt.max_chain,
            unrealized_demands=receipt.unrealized_demands,
        )
    if not receipt.valid:
        return receipt, ()

    strengths = strength_registry(problem, ctx.strength_ratios, ctx.epsilon_strength)
    programs = compile_registry(
        chains, host, problem, strengths, ctx.field_limit, ctx.coupler_limit
    )
    reports = tuple(
        check_faithfulness(p, chains, host, problem, ctx.field_limit, ctx.coupler_limit)
        for p in programs
    )
    reasons = list(receipt.reasons)
    for j, report in enumerate(reports):
        if not report.ok:
            reasons.append(f"program {j} unfaithful: {report.as_dict()}")
    ok = not reasons
    return (
        ValidationReceipt(
            valid=ok,
            reasons=tuple(reasons),
            qubits=receipt.qubits,
            max_chain=receipt.max_chain,
            unrealized_demands=receipt.unrealized_demands,
            programs=reports,
            strengths=strengths,
        ),
        programs,
    )

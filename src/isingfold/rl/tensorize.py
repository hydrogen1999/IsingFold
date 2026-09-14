"""Exact semantic tensors for IF-Core, following MODEL_SPEC Appendix A.

Eight core object types carry 120 named scalar slots, stored as value/knownness pairs, plus
an eight-way opcode and the explicitly specified chain-factor, route and archive
descriptors. Unknown values are zero *after* transformation with knownness zero; an observed
zero has knownness one. Nothing here may read a witness, a ground energy, an evaluator count
or a random key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence

import networkx as nx
import numpy as np

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Candidate, Context, Opcode, WorkVector
from isingfold.rl.program import Program, compile_program

Node = Hashable
Qubit = Hashable

N_LOGICAL, N_HARDWARE = 20, 18
N_LEDGE, N_HEDGE, N_CLAIM, N_CONFLICT = 6, 10, 6, 4
N_ACTION, N_GLOBAL = 24, 32
N_FACTOR, N_ROUTE, N_ARCHIVE = 8, 4, 5
DISTANCE_RADIUS = 6

MISSING = float("nan")
EDGE_USE_CHAIN, EDGE_USE_LOGICAL = 0, 1
ROLE_OLD, ROLE_NEW, ROLE_ARCHIVE = 0, 1, 2


def _identity_key(value: Hashable) -> tuple[str, str, str]:
    """Total ordering for heterogeneous graph IDs without conflating ``1`` and ``"1"``."""

    kind = type(value)
    return kind.__module__, kind.__qualname__, repr(value)


def t_count(x: float) -> float:
    """Nonnegative counts, ages, distances and work: ``log(1+x)``."""

    return math.log1p(max(0.0, float(x)))


def t_signed(x: float) -> float:
    """Signed count or work differences: ``sgn(x) log(1+|x|)``."""

    return math.copysign(math.log1p(abs(float(x))), x)


def t_coef(x: float, scale: float = 1.0) -> float:
    """Coefficients and coefficient-derived loads, with a recorded per-unit scale."""

    s = max(1e-12, scale)
    return math.copysign(math.log1p(abs(float(x)) / s), x)


def pack(rows: Sequence[Sequence[float]], n_slots: int) -> np.ndarray:
    """Value/knownness packing: ``[u_1..u_D, m_1..m_D]``; NaN marks a missing quantity."""

    if not rows:
        return np.zeros((0, 2 * n_slots), dtype=np.float32)
    values = np.asarray(rows, dtype=np.float64)
    if values.shape[1] != n_slots:
        raise ValueError(f"expected {n_slots} slots, got {values.shape[1]}")
    known = np.isfinite(values)
    values = np.where(known, values, 0.0)
    return np.concatenate([values, known.astype(np.float64)], axis=1).astype(np.float32)


@dataclass
class Observation:
    """Concatenated tensors and index arrays for one decision state."""

    logical: np.ndarray
    hardware: np.ndarray
    logical_edges: np.ndarray
    hardware_edges: np.ndarray
    claims: np.ndarray
    conflicts: np.ndarray
    globals_: np.ndarray
    actions: np.ndarray
    factors: np.ndarray
    routes: np.ndarray
    archive: np.ndarray
    edge_use_hardware: np.ndarray
    edge_use_logical: np.ndarray
    index_logical_edges: np.ndarray
    index_hardware_edges: np.ndarray
    index_claims: np.ndarray
    index_conflict_claimants: np.ndarray
    index_conflict_host: np.ndarray
    index_factor_action: np.ndarray
    index_factor_logical: np.ndarray
    index_factor_membership: np.ndarray
    index_factor_archive: np.ndarray
    index_route_action: np.ndarray
    index_route_logical: np.ndarray
    index_route_positions: np.ndarray
    index_action_archive: np.ndarray
    index_action_conflicts: np.ndarray
    index_edge_use_factor: np.ndarray
    index_edge_use_hardware: np.ndarray
    index_edge_use_logical: np.ndarray
    index_factor_realized_edge_use: np.ndarray
    factor_roles: np.ndarray
    route_roles: np.ndarray
    edge_use_roles: np.ndarray
    legal_mask: np.ndarray
    real_action_mask: np.ndarray
    qubit_ids: tuple[Qubit, ...]
    logical_ids: tuple[Node, ...]

    @property
    def n_actions(self) -> int:
        return int(self.actions.shape[0])

    def counts(self) -> dict[str, int]:
        return {
            "logical": int(self.logical.shape[0]),
            "hardware": int(self.hardware.shape[0]),
            "claims": int(self.claims.shape[0]),
            "conflicts": int(self.conflicts.shape[0]),
            "actions": int(self.actions.shape[0]),
            "factors": int(self.factors.shape[0]),
            "routes": int(self.routes.shape[0]),
            "archive": int(self.archive.shape[0]),
            "edge_uses": int(self.edge_use_roles.shape[0]),
        }


@dataclass
class Ages:
    """Decision ages, counted in completed selected decisions (MODEL_SPEC section 2.4)."""

    chain: dict[Node, int] = field(default_factory=dict)
    claim: dict[tuple[Node, Qubit], int] = field(default_factory=dict)
    conflict: dict[Qubit, int] = field(default_factory=dict)
    demand: dict[tuple[Node, Node], int] = field(default_factory=dict)
    occupancy: dict[Qubit, int] = field(default_factory=dict)


def _pair(u: Node, v: Node) -> tuple[Node, Node]:
    return (u, v) if _identity_key(u) <= _identity_key(v) else (v, u)


def _edge_value(mapping: Mapping, u: Hashable, v: Hashable, default: float = 0.0) -> float:
    """Orientation-independent lookup that also works for heterogeneous IDs.

    Some legacy records were canonicalised by a domain-specific ordering while the v1.1
    tensor contract uses typed identities.  Trying both stored orientations is therefore
    necessary at this boundary; the returned semantic edge is still canonical internally.
    """

    if (u, v) in mapping:
        return float(mapping[(u, v)])
    if (v, u) in mapping:
        return float(mapping[(v, u)])
    key = _pair(u, v)
    return float(mapping.get(key, default))


def _has_edge_key(mapping: Mapping, u: Hashable, v: Hashable) -> bool:
    return (u, v) in mapping or (v, u) in mapping or _pair(u, v) in mapping


def _program_edge_value(mapping: Mapping, u: Hashable, v: Hashable, default: float = 0.0) -> float:
    if (u, v) in mapping:
        return float(mapping[(u, v)])
    if (v, u) in mapping:
        return float(mapping[(v, u)])
    return float(default)


def _bfs_distance(
    host: nx.Graph,
    sources: set[Qubit],
    targets: set[Qubit],
    radius: int = DISTANCE_RADIUS,
) -> float:
    """Capped breadth-first distance on active hardware; unreachable within the cap is missing."""

    if not sources or not targets:
        return MISSING
    if sources & targets:
        return 0.0
    frontier = set(sources)
    seen = set(sources)
    for d in range(1, radius + 1):
        nxt: set[Qubit] = set()
        for q in frontier:
            for r in host.neighbors(q):
                if r in seen:
                    continue
                if r in targets:
                    return float(d)
                seen.add(r)
                nxt.add(r)
        if not nxt:
            return MISSING
        frontier = nxt
    return MISSING


def _owner_sets(
    chains: Mapping[Node, frozenset[Qubit]],
) -> dict[Qubit, list[Node]]:
    owners: dict[Qubit, list[Node]] = {}
    for logical_id, chain in chains.items():
        for qubit in chain:
            owners.setdefault(qubit, []).append(logical_id)
    return owners


def _contact_count(
    host: nx.Graph,
    left: frozenset[Qubit],
    right: frozenset[Qubit],
) -> int:
    return sum(1 for q in left for r in right if q != r and host.has_edge(q, r))


def _phase_program(
    *,
    logical: nx.Graph,
    host: nx.Graph,
    problem: LogicalProblem,
    chains: Mapping[Node, frozenset[Qubit]],
    strength: float,
    ctx: Context,
) -> Program | None:
    """Compile a phase overlay only when it is a complete, disjoint embedding.

    Overlapping or incomplete proposal phases remain structurally observable, but their
    coefficient channels are missing.  This prevents current-program coefficients from
    leaking into a different candidate or archive phase.
    """

    if set(chains) != set(logical.nodes()) or any(not chains.get(i) for i in logical.nodes()):
        return None
    claimed: set[Qubit] = set()
    for chain in chains.values():
        if not set(chain) <= set(host.nodes()) or claimed & set(chain):
            return None
        if len(chain) > 1 and not nx.is_connected(host.subgraph(chain)):
            return None
        claimed.update(chain)
    if len(claimed) > ctx.qubit_cap:
        return None
    if any(_contact_count(host, chains[u], chains[v]) == 0 for u, v in logical.edges()):
        return None
    try:
        return compile_program(
            chains,
            host,
            problem,
            strength,
            0,
            ctx.field_limit,
            ctx.coupler_limit,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def build_observation(
    *,
    ctx: Context,
    logical: nx.Graph,
    host: nx.Graph,
    problem: LogicalProblem,
    chains: Mapping[Node, frozenset[Qubit]],
    candidates: Sequence[Candidate],
    legal_mask: Sequence[bool],
    archive: Sequence,
    remaining: WorkVector,
    ages: Ages,
    strengths: Sequence[float],
    program: Program | None,
    mode_is_improvement: bool,
    restarts_left: int,
    workspace_valid: bool,
    protected_available: bool,
    instance_scale: float,
    coef_scale: float = 1.0,
) -> Observation:
    """Assemble every tensor and relation for one decision."""

    logical_ids = tuple(sorted(logical.nodes(), key=_identity_key))
    l_index = {v: k for k, v in enumerate(logical_ids)}
    qubit_ids = tuple(sorted(host.nodes(), key=_identity_key))
    q_index = {q: k for k, q in enumerate(qubit_ids)}

    occ: dict[Qubit, int] = {}
    for chain in chains.values():
        for q in chain:
            occ[q] = occ.get(q, 0) + 1
    owner_sets = _owner_sets(chains)
    # The same OLD, NEW, or ARCHIVE phase is referenced by every factor and every
    # physical edge-use row belonging to that phase.  Retain the mapping/program
    # objects alongside cached derived sets so object ids cannot be recycled while this
    # observation is being built.
    phase_owner_cache: dict[
        int,
        tuple[Mapping[Node, frozenset[Qubit]], dict[Qubit, list[Node]]],
    ] = {id(chains): (chains, owner_sets)}
    phase_chain_support_cache: dict[
        int,
        tuple[Program, frozenset[frozenset[Qubit]]],
    ] = {}

    def phase_owner_sets(
        phase_chains: Mapping[Node, frozenset[Qubit]],
    ) -> dict[Qubit, list[Node]]:
        cached = phase_owner_cache.get(id(phase_chains))
        if cached is None or cached[0] is not phase_chains:
            owners = _owner_sets(phase_chains)
            phase_owner_cache[id(phase_chains)] = (phase_chains, owners)
            return owners
        return cached[1]

    def program_chain_support(phase_program: Program) -> frozenset[frozenset[Qubit]]:
        cached = phase_chain_support_cache.get(id(phase_program))
        if cached is None or cached[0] is not phase_program:
            support = frozenset(
                frozenset(edge)
                for edge_set in phase_program.chain_edges.values()
                for edge in edge_set
            )
            phase_chain_support_cache[id(phase_program)] = (phase_program, support)
            return support
        return cached[1]

    used = set(occ)
    contact: dict[tuple[Node, Node], int] = {}
    for u, v in logical.edges():
        cu, cv = chains.get(u, frozenset()), chains.get(v, frozenset())
        contact[_pair(u, v)] = sum(1 for a in cu for b in cv if a != b and host.has_edge(a, b))

    # --- A.2 logical nodes -------------------------------------------------------
    l_rows: list[list[float]] = []
    for v in logical_ids:
        chain = chains.get(v, frozenset())
        js = [_edge_value(problem.j, v, u) for u in logical.neighbors(v)]
        absj = [abs(x) for x in js]
        deg = sum(1 for x in absj if x > 0)
        realized = [contact.get(_pair(v, u), 0) > 0 for u in logical.neighbors(v)]
        internal = host.subgraph(chain).number_of_edges() if chain else 0
        boundary = sum(1 for q in chain for r in host.neighbors(q) if r not in chain)
        free_nbrs = len({r for q in chain for r in host.neighbors(q) if r not in used})
        contested = sum(1 for q in chain if occ.get(q, 0) > 1)
        unreal = [u for u in logical.neighbors(v) if contact.get(_pair(v, u), 0) == 0]
        dist = MISSING
        if unreal and chain:
            targets = set().union(*[set(chains.get(u, frozenset())) for u in unreal]) or set()
            dist = _bfs_distance(host, set(chain), targets)
        connected = (
            1.0 if chain and (len(chain) == 1 or nx.is_connected(host.subgraph(chain))) else 0.0
        )
        l_rows.append(
            [
                t_coef(problem.h.get(v, 0.0), coef_scale),
                t_coef(abs(problem.h.get(v, 0.0)), coef_scale),
                t_count(deg),
                t_coef(sum(absj), coef_scale),
                t_coef(max(absj, default=0.0), coef_scale),
                t_coef(sum(js), coef_scale),
                (sum(1 for x in js if x > 0) / len(js)) if js else 0.0,
                1.0 if chain else 0.0,
                t_count(len(chain)),
                t_count(len(program.chain_edges.get(v, ()))) if program is not None else MISSING,
                t_count(internal),
                (sum(realized) / len(realized)) if realized else 0.0,
                t_count(contested),
                t_count(max((occ.get(q, 0) for q in chain), default=0)),
                t_count(boundary),
                t_count(free_nbrs),
                t_count(ages.chain.get(v, 0)),
                t_count(0 if ctx.tabu_tenure == 0 else ctx.tabu_tenure),
                dist if dist is MISSING else t_count(dist),
                connected,
            ]
        )

    # --- A.3 hardware nodes ------------------------------------------------------
    h_rows: list[list[float]] = []
    all_free = {q for q in qubit_ids if occ.get(q, 0) == 0}
    for q in qubit_ids:
        nbrs = list(host.neighbors(q))
        free = [r for r in nbrs if occ.get(r, 0) == 0]
        occupied = [r for r in nbrs if occ.get(r, 0) > 0]
        claimants_nbr = len({i for r in nbrs for i in owner_sets.get(r, ())})
        h_prog = program.h_phys.get(q, 0.0) if program is not None else None
        j_inc = (
            max(
                (abs(w) for (a, b), w in program.j_phys.items() if q in (a, b)),
                default=0.0,
            )
            if program is not None
            else None
        )
        nearest_free = 0.0 if occ.get(q, 0) == 0 else _bfs_distance(host, {q}, all_free)
        two_hop = len((set(nbrs) | {r2 for r in nbrs for r2 in host.neighbors(r)}) - {q})
        h_rows.append(
            [
                1.0,
                t_count(len(nbrs)),
                MISSING,
                t_count(occ.get(q, 0)),
                1.0 if occ.get(q, 0) > 1 else 0.0,
                (len(free) / len(nbrs)) if nbrs else 0.0,
                (len(occupied) / len(nbrs)) if nbrs else 0.0,
                MISSING,
                t_count(claimants_nbr),
                t_coef(h_prog, coef_scale) if h_prog is not None else MISSING,
                t_coef(abs(h_prog), coef_scale) if h_prog is not None else MISSING,
                t_coef(j_inc, coef_scale) if j_inc is not None else MISSING,
                MISSING,
                MISSING,
                MISSING,
                t_count(ages.occupancy.get(q, 0)),
                nearest_free if nearest_free is MISSING else t_count(nearest_free),
                t_count(two_hop),
            ]
        )

    # --- A.4 logical and hardware edges -----------------------------------------
    le_rows: list[list[float]] = []
    le_index: list[tuple[int, int]] = []
    le_oriented: dict[tuple[Node, Node], int] = {}
    for u, v in logical.edges():
        w = _edge_value(problem.j, u, v)
        c = contact.get(_pair(u, v), 0)
        cu, cv = chains.get(u, frozenset()), chains.get(v, frozenset())
        dist = _bfs_distance(host, set(cu), set(cv)) if cu and cv else MISSING
        row = [
            t_coef(w, coef_scale),
            t_coef(abs(w), coef_scale),
            1.0 if c > 0 else 0.0,
            t_count(c),
            dist if dist is MISSING else t_count(dist),
            t_count(_edge_value(ages.demand, u, v, 0.0)) if c == 0 else 0.0,
        ]
        le_rows.append(row)
        le_index.append((l_index[u], l_index[v]))
        le_oriented[(u, v)] = len(le_rows) - 1
        le_rows.append(row)
        le_index.append((l_index[v], l_index[u]))
        le_oriented[(v, u)] = len(le_rows) - 1

    he_rows: list[list[float]] = []
    he_index: list[tuple[int, int]] = []
    he_undirected: dict[tuple[Qubit, Qubit], int] = {}
    programmed_chain_edges = (
        {frozenset(edge) for edges in program.chain_edges.values() for edge in edges}
        if program is not None
        else set()
    )
    for a, b in host.edges():
        oa, ob = owner_sets.get(a, []), owner_sets.get(b, [])
        shared = len(set(oa) & set(ob))
        pairs = len({_pair(x, y) for x in oa for y in ob if x != y})
        jp = _program_edge_value(program.j_phys, a, b) if program is not None else None
        is_chain = frozenset((a, b)) in programmed_chain_edges
        is_logical = program is not None and any(
            left != right and _has_edge_key(problem.j, left, right) for left in oa for right in ob
        )
        denom = len(oa) + len(ob)
        contested_share = (
            (sum(1 for i in oa if occ.get(a, 0) > 1) + sum(1 for i in ob if occ.get(b, 0) > 1))
            / denom
            if denom
            else 0.0
        )
        row = [
            1.0,
            t_count(shared),
            t_count(pairs),
            (1.0 if is_chain else 0.0) if program is not None else MISSING,
            (1.0 if is_logical else 0.0) if program is not None else MISSING,
            t_coef(jp, coef_scale) if jp is not None else MISSING,
            t_coef(abs(jp), coef_scale) if jp is not None else MISSING,
            MISSING,
            MISSING,
            contested_share,
        ]
        he_rows.append(row)
        he_index.append((q_index[a], q_index[b]))
        he_undirected[_pair(a, b)] = len(he_rows) - 1
        he_rows.append(row)
        he_index.append((q_index[b], q_index[a]))

    # --- A.5 claims and conflicts ------------------------------------------------
    claim_rows: list[list[float]] = []
    claim_index: list[tuple[int, int]] = []
    for i, chain in chains.items():
        sub = host.subgraph(chain)
        articulation = (
            set(nx.articulation_points(sub)) if len(chain) > 2 and nx.is_connected(sub) else set()
        )
        for q in sorted(chain, key=str):
            inside = sum(1 for r in host.neighbors(q) if r in chain)
            on_boundary = any(r not in chain for r in host.neighbors(q))
            realized_here = sum(
                1
                for u in logical.neighbors(i)
                if any(host.has_edge(q, r) for r in chains.get(u, frozenset()))
            )
            claim_rows.append(
                [
                    t_count(ages.claim.get((i, q), 0)),
                    1.0 if occ.get(q, 0) > 1 else 0.0,
                    t_count(inside),
                    1.0 if on_boundary else 0.0,
                    1.0 if q in articulation else 0.0,
                    t_count(realized_here),
                ]
            )
            claim_index.append((l_index[i], q_index[q]))

    conflict_rows: list[list[float]] = []
    conflict_host: list[tuple[int, int]] = []
    conflict_claimants: list[tuple[int, int]] = []
    conflict_by_qubit: dict[Qubit, int] = {}
    for k, (q, owners) in enumerate(
        sorted(
            ((q, o) for q, o in owner_sets.items() if len(o) > 1),
            key=lambda item: _identity_key(item[0]),
        )
    ):
        conflict_by_qubit[q] = k
        nbr_claimants = len({i for r in host.neighbors(q) for i in owner_sets.get(r, ())})
        free_nbr = sum(1 for r in host.neighbors(q) if occ.get(r, 0) == 0)
        conflict_rows.append(
            [
                t_count(len(owners) - 1),
                t_count(ages.conflict.get(q, 0)),
                t_count(nbr_claimants),
                t_count(free_nbr),
            ]
        )
        conflict_host.append((k, q_index[q]))
        for i in owners:
            conflict_claimants.append((k, l_index[i]))

    # --- A.6 global context ------------------------------------------------------
    fr = remaining.fractions_of(ctx.caps)
    deterministic_fields = (
        "route_expansions",
        "materializations",
        "compiler_calls",
        "validator_calls",
        "cut_edge_visits",
        "restart_work",
        "feature_work",
    )
    deterministic_cap = sum(getattr(ctx.caps, name) for name in deterministic_fields)
    deterministic_remaining = sum(getattr(remaining, name) for name in deterministic_fields)
    aggregate_deterministic_fraction = (
        deterministic_remaining / deterministic_cap if deterministic_cap > 0 else MISSING
    )
    reserve_coordinates = [
        min(1.0, getattr(remaining, name) / required)
        for name in remaining.as_dict()
        if (required := getattr(ctx.reserve, name)) > 0
    ]
    terminal_reserve_fraction = min(reserve_coordinates) if reserve_coordinates else MISSING
    excess = sum(o - 1 for o in occ.values() if o > 1)
    max_mag = max(
        [abs(v) for v in problem.h.values()] + [abs(v) for v in problem.j.values()] or [0.0]
    )
    g = [
        t_count(len(logical_ids)),
        t_count(logical.number_of_edges()),
        t_count(host.number_of_nodes()),
        t_count(host.number_of_edges()),
        t_count(ctx.qubit_cap),
        t_count(len(occ)),
        t_count(excess),
        t_count(sum(1 for e in logical.edges() if contact.get(_pair(*e), 0) == 0)),
        fr["decisions"] if fr["decisions"] is not None else MISSING,
        aggregate_deterministic_fraction,
        t_count(restarts_left),
        t_count(len(archive)),
        1.0 if workspace_valid else 0.0,
        1.0 if protected_available else 0.0,
        t_coef(instance_scale, coef_scale),
        t_coef(max_mag, coef_scale),
        *[t_coef(f, coef_scale) for f in strengths],
        t_coef(ctx.field_limit, coef_scale),
        t_coef(ctx.coupler_limit, coef_scale),
        program.scale if program is not None else MISSING,
        1.0 if mode_is_improvement else 0.0,
        fr["route_expansions"] if fr["route_expansions"] is not None else MISSING,
        fr["materializations"] if fr["materializations"] is not None else MISSING,
        fr["compiler_calls"] if fr["compiler_calls"] is not None else MISSING,
        fr["validator_calls"] if fr["validator_calls"] is not None else MISSING,
        fr["cut_edge_visits"] if fr["cut_edge_visits"] is not None else MISSING,
        fr["restart_work"] if fr["restart_work"] is not None else MISSING,
        terminal_reserve_fraction,
        fr["feature_work"] if fr["feature_work"] is not None else MISSING,
    ]
    if len(g) != N_GLOBAL:
        raise AssertionError(f"global schema must have {N_GLOBAL} slots, built {len(g)}")

    # --- A.7 actions and A.8 factors / routes / archive --------------------------
    a_rows: list[list[float]] = []
    opcodes: list[int] = []
    factor_rows: list[list[float]] = []
    factor_roles: list[int] = []
    f_action: list[tuple[int, int]] = []
    f_logical: list[tuple[int, int]] = []
    f_membership: list[tuple[int, int]] = []
    f_archive: list[tuple[int, int]] = []
    edge_use_h_rows: list[list[float]] = []
    edge_use_l_rows: list[list[float]] = []
    edge_use_roles: list[int] = []
    edge_use_factor: list[int] = []
    edge_use_hardware: list[int] = []
    edge_use_logical: list[int] = []
    factor_realized_edge_use: list[tuple[int, int]] = []
    route_rows: list[list[float]] = []
    route_roles: list[int] = []
    r_action: list[tuple[int, int]] = []
    r_logical: list[tuple[int, int]] = []
    r_positions: list[tuple[int, int]] = []
    a_archive: list[tuple[int, int]] = []
    a_conflicts: list[tuple[int, int]] = []

    def phase_logical_row(
        left: Node,
        right: Node,
        phase_chains: Mapping[Node, frozenset[Qubit]],
        *,
        current_memory: bool,
    ) -> list[float]:
        coupling = _edge_value(problem.j, left, right)
        left_chain = phase_chains.get(left, frozenset())
        right_chain = phase_chains.get(right, frozenset())
        count = _contact_count(host, left_chain, right_chain)
        distance = (
            _bfs_distance(host, set(left_chain), set(right_chain))
            if left_chain and right_chain
            else MISSING
        )
        if count > 0:
            demand_age = 0.0
        elif current_memory:
            demand_age = t_count(_edge_value(ages.demand, left, right, 0.0))
        else:
            demand_age = MISSING
        return [
            t_coef(coupling, coef_scale),
            t_coef(abs(coupling), coef_scale),
            1.0 if count > 0 else 0.0,
            t_count(count),
            distance if distance is MISSING else t_count(distance),
            demand_age,
        ]

    def phase_hardware_row(
        left: Qubit,
        right: Qubit,
        phase_chains: Mapping[Node, frozenset[Qubit]],
        phase_program: Program | None,
    ) -> list[float]:
        phase_owners = phase_owner_sets(phase_chains)
        owners_left = phase_owners.get(left, [])
        owners_right = phase_owners.get(right, [])
        shared = len(set(owners_left) & set(owners_right))
        logical_pairs = len({_pair(i, j) for i in owners_left for j in owners_right if i != j})
        denominator = len(owners_left) + len(owners_right)
        contested_fraction = (
            (
                sum(1 for _ in owners_left if len(owners_left) > 1)
                + sum(1 for _ in owners_right if len(owners_right) > 1)
            )
            / denominator
            if denominator
            else 0.0
        )
        if phase_program is None:
            chain_flag = logical_flag = coupling = MISSING
        else:
            chain_support = program_chain_support(phase_program)
            chain_flag = 1.0 if frozenset((left, right)) in chain_support else 0.0
            logical_flag = (
                1.0
                if any(
                    i != j and _has_edge_key(problem.j, i, j)
                    for i in owners_left
                    for j in owners_right
                )
                else 0.0
            )
            coupling = _program_edge_value(phase_program.j_phys, left, right, 0.0)
        return [
            1.0,
            t_count(shared),
            t_count(logical_pairs),
            chain_flag,
            logical_flag,
            t_coef(coupling, coef_scale) if math.isfinite(coupling) else MISSING,
            t_coef(abs(coupling), coef_scale) if math.isfinite(coupling) else MISSING,
            MISSING,
            MISSING,
            contested_fraction,
        ]

    def add_factor(
        role: int,
        chain: frozenset[Qubit],
        owner: Node,
        changed: bool,
        action_id: int | None,
        archive_id: int | None,
        phase_chains: Mapping[Node, frozenset[Qubit]],
        phase_program: Program | None,
    ) -> int:
        fid = len(factor_rows)
        sub = host.subgraph(chain) if chain else nx.Graph()
        internal = sub.number_of_edges() if chain else 0
        boundary = sum(1 for q in chain for r in host.neighbors(q) if r not in chain)
        phase_owners = phase_owner_sets(phase_chains)
        contested = sum(1 for q in chain if len(phase_owners.get(q, ())) > 1)
        connected = 1.0 if chain and (len(chain) == 1 or nx.is_connected(sub)) else 0.0
        programmed_internal = (
            len(phase_program.chain_edges.get(owner, ())) if phase_program is not None else None
        )
        factor_rows.append(
            [
                1.0 if role == ROLE_NEW else 0.0,
                t_count(len(chain)),
                t_count(internal),
                t_count(programmed_internal) if programmed_internal is not None else MISSING,
                t_count(boundary),
                t_count(contested),
                connected,
                1.0 if changed else 0.0,
            ]
        )
        factor_roles.append(role)
        if action_id is not None:
            f_action.append((fid, action_id))
        f_logical.append((fid, l_index[owner]))
        for q in chain:
            f_membership.append((fid, q_index[q]))
        if archive_id is not None:
            f_archive.append((fid, archive_id))

        # Explicit edge-use records retain the physical/logical association and phase.
        for left, right in host.subgraph(chain).edges():
            use_id = len(edge_use_roles)
            edge_use_roles.append(EDGE_USE_CHAIN)
            edge_use_factor.append(fid)
            edge_use_hardware.append(he_undirected[_pair(left, right)])
            edge_use_logical.append(-1)
            edge_use_h_rows.append(phase_hardware_row(left, right, phase_chains, phase_program))
            edge_use_l_rows.append([MISSING] * N_LEDGE)

        for neighbor in logical.neighbors(owner):
            logical_row = le_oriented[(owner, neighbor)]
            first_use: int | None = None
            for left in chain:
                for right in phase_chains.get(neighbor, frozenset()):
                    if left == right or not host.has_edge(left, right):
                        continue
                    use_id = len(edge_use_roles)
                    if first_use is None:
                        first_use = use_id
                    edge_use_roles.append(EDGE_USE_LOGICAL)
                    edge_use_factor.append(fid)
                    edge_use_hardware.append(he_undirected[_pair(left, right)])
                    edge_use_logical.append(logical_row)
                    edge_use_h_rows.append(
                        phase_hardware_row(left, right, phase_chains, phase_program)
                    )
                    edge_use_l_rows.append(
                        phase_logical_row(
                            owner,
                            neighbor,
                            phase_chains,
                            current_memory=role == ROLE_OLD,
                        )
                    )
            if first_use is not None:
                factor_realized_edge_use.append((fid, first_use))
        return fid

    arch_rows: list[list[float]] = []
    for k, entry in enumerate(archive):
        arch_rows.append(
            [
                t_count(entry.age),
                1.0 if entry.protected else 0.0,
                1.0 if entry.admissible else 0.0,
                t_count(entry.qubits),
                t_count(entry.max_chain),
            ]
        )
        archived_chains = {i: frozenset(c) for i, c in entry.chains.items()}
        archived_program = (
            _phase_program(
                logical=logical,
                host=host,
                problem=problem,
                chains=archived_chains,
                strength=float(strengths[0]),
                ctx=ctx,
            )
            if entry.admissible
            else None
        )
        for owner, chain in archived_chains.items():
            add_factor(
                ROLE_ARCHIVE,
                chain,
                owner,
                False,
                None,
                k,
                archived_chains,
                archived_program,
            )

    q_used_now = len(occ)
    for aid, cand in enumerate(candidates):
        successor = dict(chains)
        successor.update(cand.new_chains)
        successor = {i: frozenset(c) for i, c in successor.items()}
        successor_program = (
            _phase_program(
                logical=logical,
                host=host,
                problem=problem,
                chains=successor,
                strength=float(strengths[0]),
                ctx=ctx,
            )
            if cand.changes_workspace
            else None
        )
        s_occ: dict[Qubit, int] = {}
        for c in successor.values():
            for q in c:
                s_occ[q] = s_occ.get(q, 0) + 1
        s_excess = sum(o - 1 for o in s_occ.values() if o > 1)
        s_contact_missing = sum(
            1
            for u, v in logical.edges()
            if not any(
                a != b and host.has_edge(a, b)
                for a in successor.get(u, frozenset())
                for b in successor.get(v, frozenset())
            )
        )
        old_claims = sum(len(c) for c in cand.old_chains.values())
        new_claims = sum(len(c) for c in cand.new_chains.values())
        added = sum(
            len(cand.new_chains[i] - cand.old_chains.get(i, frozenset())) for i in cand.new_chains
        )
        removed = sum(
            len(cand.old_chains.get(i, frozenset()) - cand.new_chains[i]) for i in cand.new_chains
        )
        lengths_new = [len(c) for c in successor.values()] or [0]
        old_boundary = sum(
            sum(
                1
                for q in cand.old_chains.get(i, frozenset())
                for r in host.neighbors(q)
                if r not in cand.old_chains.get(i, frozenset())
            )
            for i in cand.new_chains
        )
        new_boundary = sum(
            sum(
                1
                for q in cand.new_chains[i]
                for r in host.neighbors(q)
                if r not in cand.new_chains[i]
            )
            for i in cand.new_chains
        )
        old_free = sum(
            len(
                {
                    r
                    for q in cand.old_chains.get(i, frozenset())
                    for r in host.neighbors(q)
                    if r not in used
                }
            )
            for i in cand.new_chains
        )
        new_free = sum(
            len({r for q in cand.new_chains[i] for r in host.neighbors(q) if r not in s_occ})
            for i in cand.new_chains
        )
        route_lengths = [len(p) for p, _ in cand.routes] or [0]
        entry = (
            archive[cand.archive_ref]
            if cand.archive_ref is not None and cand.archive_ref < len(archive)
            else None
        )
        successor_structural = (
            set(successor) == set(logical.nodes())
            and all(set(chain) <= set(host.nodes()) for chain in successor.values())
            and all(
                (not chain and not mode_is_improvement)
                or bool(chain)
                and (len(chain) == 1 or nx.is_connected(host.subgraph(chain)))
                for chain in successor.values()
            )
            and len(s_occ) <= ctx.qubit_cap
            and max(s_occ.values(), default=0) <= ctx.overlap.max_occupancy
            and s_excess <= ctx.overlap.excess_cap(ctx.qubit_cap)
        )
        terminal_action = cand.opcode in (Opcode.COMMIT, Opcode.STOP)
        proposal_work = sum(cand.proposal_work.as_dict().values())
        application_work = sum(cand.work.as_dict().values())
        a_rows.append(
            [
                t_count(len(cand.affected)),
                t_count(old_claims),
                t_count(new_claims),
                t_count(added),
                t_count(removed),
                t_signed(len(s_occ) - q_used_now),
                t_signed(max(lengths_new) - max([len(c) for c in chains.values()] or [0])),
                t_signed(s_excess - excess),
                t_signed(sum(1 for o in s_occ.values() if o > 1) - len(conflict_rows)),
                t_signed(
                    s_contact_missing
                    - sum(1 for e in logical.edges() if contact.get(_pair(*e), 0) == 0)
                ),
                t_count(max(s_occ.values(), default=0)),
                1.0 if successor_structural else 0.0,
                t_count(max(0, ctx.qubit_cap - len(s_occ))),
                MISSING if terminal_action else t_count(proposal_work),
                t_count(application_work),
                MISSING if terminal_action else t_count(max(route_lengths)),
                MISSING
                if terminal_action
                else t_count(sum(route_lengths) / max(1, len(route_lengths))),
                MISSING if terminal_action else t_signed(new_boundary - old_boundary),
                MISSING if terminal_action else t_signed(new_free - old_free),
                MISSING
                if terminal_action
                else t_count(max(0, restarts_left - (1 if cand.opcode is Opcode.RESTART else 0))),
                t_count(entry.age) if entry is not None else MISSING,
                t_count(entry.qubits) if entry is not None else MISSING,
                1.0 if entry is not None and cand.changes_workspace else 0.0,
                1.0 if cand.changes_workspace else 0.0,
            ]
        )
        opcodes.append(cand.opcode.index)
        if cand.archive_ref is not None:
            a_archive.append((aid, cand.archive_ref))
        touched_qubits = {
            q
            for i in cand.affected
            for q in (
                set(cand.old_chains.get(i, frozenset())) | set(cand.new_chains.get(i, frozenset()))
            )
        }
        touched_qubits.update(q for path, _owner in cand.routes for q in path)
        for q in sorted(touched_qubits & set(conflict_by_qubit), key=_identity_key):
            a_conflicts.append((aid, conflict_by_qubit[q]))
        for i in cand.affected:
            old_chain = frozenset(cand.old_chains.get(i, frozenset()))
            new_chain = frozenset(cand.new_chains[i])
            changed = new_chain != old_chain
            add_factor(
                ROLE_OLD,
                old_chain,
                i,
                False,
                aid,
                None,
                chains,
                program,
            )
            add_factor(
                ROLE_NEW,
                new_chain,
                i,
                changed,
                aid,
                None,
                successor,
                successor_program,
            )
        for path, owner in cand.routes:
            route_id = len(route_roles)
            route_roles.append(ROLE_NEW)
            L = len(path)
            for k, q in enumerate(path):
                route_rows.append(
                    [
                        k / max(L - 1, 1),
                        1.0 if k == 0 or L == 1 else 0.0,
                        1.0 if k == L - 1 or L == 1 else 0.0,
                        1.0 if q in successor.get(owner, frozenset()) else 0.0,
                    ]
                )
                r_positions.append((route_id, q_index[q]))
            r_action.append((route_id, aid))
            r_logical.append((route_id, l_index[owner]))

    opcode_onehot = np.zeros((len(a_rows), len(Opcode)), dtype=np.float32)
    for k, idx in enumerate(opcodes):
        opcode_onehot[k, idx] = 1.0
    actions = (
        np.concatenate([pack(a_rows, N_ACTION), opcode_onehot], axis=1)
        if a_rows
        else np.zeros((0, 2 * N_ACTION + len(Opcode)), dtype=np.float32)
    )

    return Observation(
        logical=pack(l_rows, N_LOGICAL),
        hardware=pack(h_rows, N_HARDWARE),
        logical_edges=pack(le_rows, N_LEDGE),
        hardware_edges=pack(he_rows, N_HEDGE),
        claims=pack(claim_rows, N_CLAIM),
        conflicts=pack(conflict_rows, N_CONFLICT),
        globals_=pack([g], N_GLOBAL),
        actions=actions,
        factors=pack(factor_rows, N_FACTOR),
        routes=pack(route_rows, N_ROUTE),
        archive=pack(arch_rows, N_ARCHIVE),
        edge_use_hardware=pack(edge_use_h_rows, N_HEDGE),
        edge_use_logical=pack(edge_use_l_rows, N_LEDGE),
        index_logical_edges=np.asarray(le_index, dtype=np.int64).T.reshape(2, -1),
        index_hardware_edges=np.asarray(he_index, dtype=np.int64).T.reshape(2, -1),
        index_claims=np.asarray(claim_index, dtype=np.int64).T.reshape(2, -1),
        index_conflict_claimants=np.asarray(conflict_claimants, dtype=np.int64).T.reshape(2, -1),
        index_conflict_host=np.asarray(conflict_host, dtype=np.int64).T.reshape(2, -1),
        index_factor_action=np.asarray(f_action, dtype=np.int64).T.reshape(2, -1),
        index_factor_logical=np.asarray(f_logical, dtype=np.int64).T.reshape(2, -1),
        index_factor_membership=np.asarray(f_membership, dtype=np.int64).T.reshape(2, -1),
        index_factor_archive=np.asarray(f_archive, dtype=np.int64).T.reshape(2, -1),
        index_route_action=np.asarray(r_action, dtype=np.int64).T.reshape(2, -1),
        index_route_logical=np.asarray(r_logical, dtype=np.int64).T.reshape(2, -1),
        index_route_positions=np.asarray(r_positions, dtype=np.int64).T.reshape(2, -1),
        index_action_archive=np.asarray(a_archive, dtype=np.int64).T.reshape(2, -1),
        index_action_conflicts=np.asarray(a_conflicts, dtype=np.int64).T.reshape(2, -1),
        index_edge_use_factor=np.asarray(edge_use_factor, dtype=np.int64),
        index_edge_use_hardware=np.asarray(edge_use_hardware, dtype=np.int64),
        index_edge_use_logical=np.asarray(edge_use_logical, dtype=np.int64),
        index_factor_realized_edge_use=np.asarray(
            factor_realized_edge_use, dtype=np.int64
        ).T.reshape(2, -1),
        factor_roles=np.asarray(factor_roles, dtype=np.int64),
        route_roles=np.asarray(route_roles, dtype=np.int64),
        edge_use_roles=np.asarray(edge_use_roles, dtype=np.int64),
        legal_mask=np.asarray(list(legal_mask), dtype=bool),
        real_action_mask=np.ones(len(a_rows), dtype=bool),
        qubit_ids=qubit_ids,
        logical_ids=logical_ids,
    )

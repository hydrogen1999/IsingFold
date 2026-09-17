"""Outcome-blind candidate batches with fixed quotas and charged work.

Spec: Rev2 section 3.5 and MODEL_SPEC section 2.4. Every candidate is fully materialised and
bound before scoring, every attempt is charged whether or not it is selected, and identical
successors are deduplicated so provenance alone cannot multiply an action's policy mass. No
quality outcome, witness or ground-state reference may influence this module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence

import networkx as nx

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    ArchiveEntry,
    Candidate,
    Context,
    Mode,
    Opcode,
    RestartCacheSlot,
    WorkVector,
    chain_key,
    stable_digest,
)
from isingfold.rl.router import rebuild_group, route_between_sets, route_variable

Node = Hashable
Qubit = Hashable

Initializer = Callable[[nx.Graph, nx.Graph, int], dict[Node, frozenset[Qubit]] | None]

AUTHENTICATED_RESTART_CACHE_V1 = "authenticated-restart-cache-v1"
LEGACY_ONLINE_INITIALIZER_RESTARTS_V1 = "legacy-online-initializer-restarts-v1"
_IMPROVEMENT_RESTART_PROTOCOLS = frozenset(
    {
        AUTHENTICATED_RESTART_CACHE_V1,
        LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    }
)

PROPOSAL_VERSION = "if-proposal-v4-authenticated-restart-cache"


def bound_successor_key(
    chains: Mapping[Node, frozenset[Qubit]],
    work: WorkVector,
    *,
    restart: bool,
    restart_cache_after_digest: str | None = None,
) -> str:
    """Identity of transition semantics, excluding provenance and heuristic family."""

    return stable_digest(
        {
            "chains": chain_key(chains),
            "work": work.as_dict(),
            "restart_resets_workspace_memory": restart,
            "restart_token_delta": -1 if restart else 0,
            "restart_cache_after_digest": restart_cache_after_digest,
        }
    )


def restart_cache_digest(slots: Sequence[RestartCacheSlot]) -> str:
    """Bind ordered cache evidence and consumption without exposing it as a feature."""

    return stable_digest(
        [
            {
                "slot_index": slot.slot_index,
                "status": slot.status,
                "chains": None if slot.chains is None else chain_key(slot.chains),
                "snapshot_record_digest": slot.snapshot_record_digest,
                "attempt_receipt_root": slot.attempt_receipt_root,
                "consumed": slot.consumed,
            }
            for slot in slots
        ]
    )


@dataclass
class WorkMeter:
    """Deterministic work boundaries: the generator stops instead of overspending."""

    route_expansions: int
    materializations: int
    restart_work: int = 10**9
    used_expansions: int = 0
    used_materializations: int = 0
    used_restart_work: int = 0

    def remaining_expansions(self) -> int:
        return max(0, self.route_expansions - self.used_expansions)

    def remaining_materializations(self) -> int:
        return max(0, self.materializations - self.used_materializations)

    def can_charge(self, expansions: int = 0, *, restart_work: int = 0) -> bool:
        """Whether one more materialisation with the stated exact cost fits.

        A depleted route coordinate must not block a zero-route archive restore or reset.
        """

        return (
            expansions >= 0
            and restart_work >= 0
            and self.used_expansions + expansions <= self.route_expansions
            and self.used_materializations + 1 <= self.materializations
            and self.used_restart_work + restart_work <= self.restart_work
        )

    def exhausted(self) -> bool:
        """No further proposal of any family can be materialised."""

        return self.used_materializations >= self.materializations

    def charge(self, expansions: int = 0, *, restart_work: int = 0) -> WorkVector:
        if not self.can_charge(expansions, restart_work=restart_work):
            raise ValueError("proposal charge exceeds the common work allowance")
        self.used_expansions += expansions
        self.used_materializations += 1
        self.used_restart_work += restart_work
        return WorkVector(
            route_expansions=expansions,
            materializations=1,
            restart_work=restart_work,
        )

    def can_restart(self) -> bool:
        return self.can_charge(0, restart_work=1)


@dataclass(frozen=True)
class ProposalBatch:
    candidates: tuple[Candidate, ...]
    work: WorkVector
    counts: Mapping[str, int]
    rejected: int


def strongest_neighbours(problem: LogicalProblem, logical: nx.Graph, v: Node, k: int) -> list[Node]:
    """The k logical neighbours with the largest ``|J|``: where a broken chain costs most."""

    nbrs = list(logical.neighbors(v))
    nbrs.sort(key=lambda u: (-abs(problem.coupling(v, u)), str(u)))
    return nbrs[:k]


def _pair(u: Node, v: Node) -> tuple[Node, Node]:
    return (u, v) if str(u) <= str(v) else (v, u)


@dataclass(frozen=True)
class RepairSeed:
    """A repair group plus the exact defect that makes the opcode applicable."""

    group: tuple[Node, ...]
    target_conflict: Qubit | None = None
    target_demand: tuple[Node, Node] | None = None


def repair_seeds(
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    host: nx.Graph,
    group_sizes: Sequence[int],
) -> list[RepairSeed]:
    """Groups that contain every claimant of a contested qubit, or both ends of a demand."""

    groups: list[RepairSeed] = []
    allowed = sorted(set(group_sizes))
    nodes = sorted(chains, key=str)

    def complete_group(required: Sequence[Node]) -> tuple[Node, ...] | None:
        unique = list(dict.fromkeys(required))
        target_size = next((size for size in allowed if size >= len(unique)), None)
        if target_size is None or target_size > len(nodes):
            return None
        extras = [node for node in nodes if node not in unique]
        return tuple(unique + extras[: target_size - len(unique)])

    claimants: dict[Qubit, list[Node]] = {}
    for i, chain in chains.items():
        for q in chain:
            claimants.setdefault(q, []).append(i)
    for q, owners in sorted(claimants.items(), key=lambda item: str(item[0])):
        if len(owners) > 1:
            group = complete_group(sorted(owners, key=str))
            if group is not None:
                groups.append(RepairSeed(group, target_conflict=q))
    for u, v in logical.edges():
        cu, cv = chains.get(u, frozenset()), chains.get(v, frozenset())
        if cu and cv and not any(host.has_edge(a, b) for a in cu for b in cv if a != b):
            group = complete_group((u, v))
            if group is not None:
                groups.append(RepairSeed(group, target_demand=_pair(u, v)))
    return groups


def conflict_groups(
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    host: nx.Graph,
    size: int,
) -> list[tuple[Node, ...]]:
    """Compatibility view of repair groups; claimant sets are never truncated."""

    return [seed.group for seed in repair_seeds(chains, logical, host, (size,))]



def _collapse_identical_successors(
    items: "Sequence[tuple[Candidate, int]]",
) -> "list[tuple[Candidate, int]]":
    """Keep one entry per distinct successor assignment, the one that costs least work.

    Two restores that rebuild the same chains are the same move to the environment and to the
    policy; they differ only in the bookkeeping of how the archive was replayed. Letting both
    hold a slot spends the action pool on a choice that does not exist.
    """

    best: dict[frozenset, tuple[Candidate, int]] = {}
    order: list[frozenset] = []
    for cand, used in items:
        key = (cand.opcode,
               frozenset((node, frozenset(chain))
                         for node, chain in cand.new_chains.items()))
        prior = best.get(key)
        if prior is None:
            best[key] = (cand, used)
            order.append(key)
        elif used < prior[1]:
            best[key] = (cand, used)
    return [best[key] for key in order]

class ProposalGenerator:
    """Builds one candidate batch per decision under a fixed family allocation."""

    def __init__(
        self,
        ctx: Context,
        logical: nx.Graph,
        host: nx.Graph,
        problem: LogicalProblem,
        initializer: Initializer | None = None,
        mode: Mode = Mode.IMPROVEMENT,
        improvement_restart_protocol: str = AUTHENTICATED_RESTART_CACHE_V1,
        overfill: float = 4.0,
        jitter_scale: float = 0.15,
        allow_satisfied_growth: bool = False,
    ) -> None:
        if improvement_restart_protocol not in _IMPROVEMENT_RESTART_PROTOCOLS:
            raise ValueError("unknown improvement restart protocol")
        if type(allow_satisfied_growth) is not bool:
            raise ValueError("allow_satisfied_growth must be Boolean")
        self.ctx = ctx
        self.logical = logical
        self.host = host
        self.problem = problem
        self.initializer = initializer
        self.mode = mode
        self.improvement_restart_protocol = improvement_restart_protocol
        self.overfill = overfill
        self.jitter_scale = jitter_scale
        # The independent constructor can spend spare capacity to improve a valid
        # embedding's physical program. Older proposal registries retain their support.
        self.allow_satisfied_growth = allow_satisfied_growth
        # An optional prioritiser over (variable, qubit) pairs, higher first, consulted by
        # the construction families before their budget truncates the offer. The registered
        # cap of 64 state-changing candidates applies to what a decision sees, not to how
        # the generator ranks what it considers. A teacher sets it to prefer a witness; at
        # deployment a learned scorer can stand in the same place. None means the family's
        # own deterministic order.
        self.prefer = None
        self._pref_cache: dict = {}

    def _pref(self, variable, qubit) -> float:
        if self.prefer is None:
            return 0.0
        key = (variable, qubit)
        cache = self._pref_cache
        if key not in cache:
            cache[key] = float(self.prefer(variable, qubit))
        return cache[key]

    # -- families -------------------------------------------------------------------

    def _place(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        budget: int,
        meter: WorkMeter,
    ) -> list[tuple[Candidate, int]]:
        """Bind singleton placements for empty construction chains.

        Free roots come first, but legal overlap roots remain proposal candidates under the
        registered overlap profile.  The exact mask, not this heuristic ordering, decides
        admissibility.
        """

        out: list[tuple[Candidate, int]] = []
        occupied = _occupancy_excluding(chains, set())
        empty = sorted((i for i, chain in chains.items() if not chain), key=str)
        if not empty:
            return out

        # Roots are offered per variable, and the variables with placed logical neighbours
        # come first: their roots are the free qubits adjacent to those neighbours' chains,
        # ordered by how many placed neighbours they touch. On a planted instance every chain
        # of the witness touches the chains of its logical neighbours, so a witness root is
        # always among these. The earlier rule offered the lexicographically first free
        # qubits to the lexicographically first empty variable, which on a large host placed
        # everything in one corner and could not replay a witness at all.
        def adjacent_roots(variable):
            touches: dict = {}
            for nb in self.logical.neighbors(variable):
                chain = chains.get(nb)
                if not chain:
                    continue
                for q in chain:
                    for r in self.host.neighbors(q):
                        if occupied.get(r, 0) == 0:
                            touches[r] = touches.get(r, 0) + 1
            return sorted(touches, key=lambda r: (-self._pref(variable, r), -touches[r], str(r)))

        def placed_neighbours(variable):
            return sum(1 for nb in self.logical.neighbors(variable) if chains.get(nb))

        def best_pref(vr):
            return max((self._pref(vr[0], r) for r in vr[1]), default=0.0)

        attached = [(v, adjacent_roots(v)) for v in empty]
        attached = [(v, roots) for v, roots in attached if roots]
        # The most constrained variables first: those with the most placed neighbours have
        # the fewest qubits adjacent to all of them, so a few roots each cover their
        # options; on a host of degree fifteen a single root per variable does not.
        attached.sort(key=lambda vr: (-best_pref(vr), -placed_neighbours(vr[0]),
                                      -self.logical.degree(vr[0]), str(vr[0])))
        if budget > 64:
            # Wide support: every frontier variable is offered, with as many of its adjacent
            # roots as the budget allows (at most twelve each), instead of eight variables
            # with a shortlist. On a planted instance the witness root of every frontier
            # variable is adjacent to a placed neighbour, so it is in this offer whenever
            # its variable is; the registered 64-candidate shortlist covered eight variables
            # of a frontier of fifty and blocked the witness walk within twenty steps.
            per_variable = max(1, min(12, budget // max(1, len(attached))))
        else:
            per_variable = max(1, budget // 8)
        attached = [(v, roots[:per_variable]) for v, roots in attached[: max(1, budget // per_variable)]]
        if not attached:
            # Nothing placed yet: a spread of free roots for the highest-degree empty
            # variable, every k-th free qubit, so the first placement is not confined to one
            # corner. A teacher that needs a specific first root starts the environment from
            # a one-variable partial embedding instead.
            first = sorted(empty, key=lambda v: (-self.logical.degree(v), str(v)))[0]
            free = [q for q in sorted(self.host.nodes(), key=str) if occupied.get(q, 0) == 0]
            stride = max(1, len(free) // max(1, budget))
            spread = free[::stride]
            if self.prefer is not None:
                spread = sorted(free, key=lambda q: (-self._pref(first, q), str(q)))[: len(spread)]
            attached = [(first, spread)]

        # Round-robin across the attached variables so no single variable eats the quota.
        cursors = [0] * len(attached)
        progressed = True
        while progressed:
            progressed = False
            for i, (variable, roots) in enumerate(attached):
                if cursors[i] >= len(roots):
                    continue
                if len(out) >= budget or not meter.can_charge(1):
                    return out
                root = roots[cursors[i]]
                cursors[i] += 1
                progressed = True
                proposal_work = meter.charge(1)
                out.append(
                    (
                        self._make(
                            Opcode.PLACE,
                            chains,
                            {variable: frozenset({root})},
                            routes=(((root,), variable),),
                            proposal_work=proposal_work,
                            provenance=f"place:{variable}:{root}",
                        ),
                        1,
                    )
                )
        return out

    def _grow(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        budget: int,
        meter: WorkMeter,
    ) -> list[tuple[Candidate, int]]:
        """Bind one placed chain grown by one adjacent free qubit, as a REWRITE_ONE.

        Construction needs it: a ROUTE may only add qubits that realise a demand between two
        placed chains, so a chain's further qubits, which unplaced neighbours will need to
        touch, could otherwise never appear before those neighbours are placed, and they
        cannot be placed before the qubits exist. Chains with the most unplaced logical
        neighbours come first, rotated with the state so no chain holds the budget.
        With ``allow_satisfied_growth``, already satisfied chains remain eligible:
        allocating another adjacent free qubit is a policy-controlled quality refinement.
        """
        out: list[tuple[Candidate, int]] = []
        occupied = _occupancy_excluding(chains, set())
        placed = [v for v, c in chains.items() if c]
        if not placed:
            return out

        def need(v):
            """Unplaced logical neighbours, plus demands to placed neighbours still unmet:
            both are reasons this chain will have to reach further. The first version
            counted only unplaced neighbours, so once every variable was placed no chain
            could grow and the last demands, which need a longer chain, stayed unmet."""
            n = 0
            for u in self.logical.neighbors(v):
                other = chains.get(u)
                if not other:
                    n += 1
                elif not any(
                    q != r and self.host.has_edge(q, r) for q in chains[v] for r in other
                ):
                    n += 1
            return n

        if not self.allow_satisfied_growth:
            placed = [v for v in placed if need(v) > 0]
        if not placed:
            return out
        placed.sort(key=lambda v: (-need(v), str(v)))
        start = sum(len(c) for c in chains.values()) % len(placed)
        placed = placed[start:] + placed[:start]
        if self.prefer is not None:
            def chain_pref(v):
                return max((self._pref(v, r) for q in chains[v] for r in self.host.neighbors(q)
                            if occupied.get(r, 0) == 0), default=0.0)
            placed.sort(key=lambda v: -chain_pref(v))
        for v in placed:
            chain = chains[v]
            free = sorted(
                {r for q in chain for r in self.host.neighbors(q) if occupied.get(r, 0) == 0},
                key=lambda r: (-self._pref(v, r), str(r)),
            )
            scan = sum(self.host.degree(q) for q in chain)
            for r in free:
                if len(out) >= budget or not meter.can_charge(scan):
                    return out
                anchor = next(q for q in sorted(chain, key=str) if self.host.has_edge(q, r))
                proposal_work = meter.charge(scan)
                out.append(
                    (
                        self._make(
                            Opcode.REWRITE_ONE,
                            chains,
                            {v: frozenset(set(chain) | {r})},
                            routes=(((anchor, r), v),),
                            proposal_work=proposal_work,
                            provenance=f"grow:{v}:{r}",
                        ),
                        scan,
                    )
                )
        return out

    def _shrink(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        budget: int,
        meter: WorkMeter,
    ) -> list[tuple[Candidate, int]]:
        """Bind one placed chain with one qubit removed, as a REWRITE_ONE.

        The only action that returns a qubit to the host. A qubit may go if the chain stays
        connected and every demand the chain realises now is still realised without it.
        Without this, every route and growth is irreversible, and a policy that misplaces
        fills the host and ends with nothing legal but STOP, which is how the first
        imitation-initialised episodes ended: zero free qubits at 83 percent of demands.
        """
        out: list[tuple[Candidate, int]] = []
        placed = sorted((v for v, c in chains.items() if len(c) >= 2), key=str)
        if not placed:
            return out
        start = sum(len(c) for c in chains.values()) % len(placed)
        placed = placed[start:] + placed[:start]
        for v in placed:
            chain = chains[v]
            others = [(u, chains[u]) for u in self.logical.neighbors(v) if chains.get(u)]
            met_now = [
                u for u, oc in others
                if any(self.host.has_edge(q, r) for q in chain for r in oc)
            ]
            for q in sorted(chain, key=str):
                rest = chain - {q}
                if not nx.is_connected(self.host.subgraph(rest)):
                    continue
                if any(
                    not any(self.host.has_edge(a, r) for a in rest for r in chains[u])
                    for u in met_now
                ):
                    continue
                if len(out) >= budget or not meter.can_charge(1):
                    return out
                proposal_work = meter.charge(1)
                out.append(
                    (
                        self._make(
                            Opcode.REWRITE_ONE,
                            chains,
                            {v: frozenset(rest)},
                            proposal_work=proposal_work,
                            provenance=f"shrink:{v}:{q}",
                        ),
                        1,
                    )
                )
        return out

    def _route_demands(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        budget: int,
        meter: WorkMeter,
    ) -> list[tuple[Candidate, int]]:
        """Bind connected endpoint supersets that realise one selected demand."""

        out: list[tuple[Candidate, int]] = []
        occupied = _occupancy_excluding(chains, set())
        unmet = []
        for left, right in sorted(
            self.logical.edges(), key=lambda edge: (str(edge[0]), str(edge[1]))
        ):
            left_chain = chains.get(left, frozenset())
            right_chain = chains.get(right, frozenset())
            if not left_chain or not right_chain:
                continue
            if any(q != r and self.host.has_edge(q, r) for q in left_chain for r in right_chain):
                continue
            unmet.append((left, right))
        if not unmet:
            return out
        # Rotate the starting demand with the state, so a demand whose offers are all useless
        # does not occupy the whole family budget at every decision; the rotation is a
        # function of the state, never of a clock or a random draw.
        start = sum(len(c) for c in chains.values()) % len(unmet)
        unmet = unmet[start:] + unmet[:start]
        if self.prefer is not None:
            def demand_pref(lr):
                l, r = lr
                return max(
                    [self._pref(l, q) for c in chains[l] for q in self.host.neighbors(c)
                     if occupied.get(q, 0) == 0]
                    + [self._pref(r, q) for c in chains[r] for q in self.host.neighbors(c)
                       if occupied.get(q, 0) == 0],
                    default=0.0,
                )
            unmet.sort(key=lambda lr: -demand_pref(lr))

        def distance_to(target_chain, limit=8):
            """Host BFS distance from every qubit within ``limit`` of the target chain."""
            dist = {q: 0 for q in target_chain}
            frontier = list(target_chain)
            for d in range(1, limit + 1):
                nxt = []
                for q in frontier:
                    for r in self.host.neighbors(q):
                        if r not in dist:
                            dist[r] = d
                            nxt.append(r)
                frontier = nxt
                if not frontier:
                    break
            return dist

        for left, right in unmet:
            pair = _pair(left, right)
            left_chain = chains.get(left, frozenset())
            right_chain = chains.get(right, frozenset())
            for owner, target in ((left, right), (right, left)):
                if len(out) >= budget or not meter.can_charge(1):
                    return out
                # In construction the route runs on the residual host: free qubits and the
                # two chains it joins. The router walks through occupied qubits on purpose,
                # since temporary overlap is what repair works on in improvement mode; in
                # construction an overlapping route leaves a workspace the return check
                # rejects, so no COMMIT is ever offered, which is how policy episodes
                # with every demand met still ended without a valid embedding.
                if self.mode is Mode.CONSTRUCTION:
                    keep = {q for q, o in occupied.items() if o == 0} | set(chains[owner]) | set(chains[target])
                    keep |= {q for q in self.host.nodes() if q not in occupied}
                    route_host = self.host.subgraph(keep)
                else:
                    route_host = self.host
                attempt = route_between_sets(
                    route_host,
                    chains[owner],
                    chains[target],
                    expansion_budget=min(4_000, meter.remaining_expansions()),
                )
                proposal_work = meter.charge(attempt.expansions)
                if attempt.path is None:
                    continue
                path = attempt.path
                # The target endpoint remains owned by the other logical variable.  Every
                # vertex in the stored route segment is owned by ``owner`` in the successor.
                owned_segment = path[:-1]
                if not owned_segment:
                    continue
                replacement = frozenset(set(chains[owner]) | set(owned_segment))
                out.append(
                    (
                        self._make(
                            Opcode.ROUTE,
                            chains,
                            {owner: replacement},
                            routes=((tuple(owned_segment), owner),),
                            proposal_work=proposal_work,
                            target_demand=pair,
                            provenance=f"route:{left}:{right}:{owner}",
                        ),
                        attempt.expansions,
                    )
                )
            # One-qubit bridges, after the router path: a free qubit adjacent to both
            # chains. The router returns one shortest path per demand,
            # and when several bridges exist that path is one arbitrary choice; offering the
            # bridges themselves lets a teacher's chain be reached exactly. Bounded by the
            # family budget like everything else.
            bridges = sorted(
                (
                    r
                    for q in left_chain
                    for r in self.host.neighbors(q)
                    if occupied.get(r, 0) == 0
                    and any(self.host.has_edge(r, t) for t in right_chain)
                ),
                key=str,
            )
            # Each bridge is offered to either owner: which chain absorbs the qubit is a
            # decision, and a teacher's chain may hold it on either side.
            scan = sum(self.host.degree(q) for q in left_chain)
            pairs = sorted(((r, owner) for r in bridges for owner in (left, right)),
                           key=lambda ro: (-self._pref(ro[1], ro[0]), str(ro[0]), str(ro[1])))
            for r, owner in pairs:
                if True:
                    if len(out) >= budget or not meter.can_charge(scan):
                        return out
                    anchor = next(
                        q for q in sorted(chains[owner], key=str) if self.host.has_edge(q, r)
                    )
                    proposal_work = meter.charge(scan)
                    out.append(
                        (
                            self._make(
                                Opcode.ROUTE,
                                chains,
                                {owner: frozenset(set(chains[owner]) | {r})},
                                routes=(((anchor, r), owner),),
                                proposal_work=proposal_work,
                                target_demand=pair,
                                provenance=f"bridge:{left}:{right}:{owner}:{r}",
                            ),
                            scan,
                        )
                    )
            if not bridges:
                # No single qubit joins the two chains: both grow by one free qubit and meet,
                # x adjacent to the left chain, y adjacent to the right chain, x adjacent to y.
                # A ROUTE must realise its demand, so the two additions travel together; a
                # teacher whose two chains each carry one more qubit is reached this way.
                left_free = sorted(
                    {r for q in left_chain for r in self.host.neighbors(q) if occupied.get(r, 0) == 0},
                    key=str,
                )
                right_free = {
                    r for q in right_chain for r in self.host.neighbors(q) if occupied.get(r, 0) == 0
                }
                meets = sorted(
                    (
                        (x, y)
                        for x in left_free
                        for y in sorted(self.host.neighbors(x), key=str)
                        if y in right_free and y != x
                    ),
                    key=lambda xy: (-(self._pref(left, xy[0]) + self._pref(right, xy[1])),
                                    str(xy[0]), str(xy[1])),
                )
                scan = sum(self.host.degree(q) for q in left_chain) + sum(
                    self.host.degree(q) for q in right_chain
                )
                for x, y in meets[:4]:
                    if len(out) >= budget or not meter.can_charge(scan):
                        return out
                    ax = next(q for q in sorted(left_chain, key=str) if self.host.has_edge(q, x))
                    ay = next(q for q in sorted(right_chain, key=str) if self.host.has_edge(q, y))
                    proposal_work = meter.charge(scan)
                    out.append(
                        (
                            self._make(
                                Opcode.ROUTE,
                                chains,
                                {
                                    left: frozenset(set(left_chain) | {x}),
                                    right: frozenset(set(right_chain) | {y}),
                                },
                                routes=(((ax, x), left), ((ay, y), right)),
                                proposal_work=proposal_work,
                                target_demand=pair,
                                provenance=f"meet:{left}:{right}:{x}:{y}",
                            ),
                            scan,
                        )
                    )
        return out

    def _single(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        rng: random.Random,
        budget: int,
        meter: WorkMeter,
    ) -> list[tuple[Candidate, int]]:
        out: list[tuple[Candidate, int]] = []
        variables = list(chains)
        rng.shuffle(variables)
        for v in variables:
            if len(out) >= budget or not meter.can_charge(1):
                break
            for rank in (0, 1, 2):
                if len(out) >= budget or not meter.can_charge(1):
                    break
                neighbours = [chains[u] for u in self.logical.neighbors(v) if chains.get(u)]
                attempt = route_variable(
                    self.host,
                    neighbours,
                    _occupancy_excluding(chains, {v}),
                    overfill=self.overfill,
                    rng=rng,
                    jitter_scale=self.jitter_scale,
                    root_rank=rank,
                    expansion_budget=min(4_000, meter.remaining_expansions()),
                )
                proposal_work = meter.charge(attempt.expansions)
                result = attempt.result
                if result is None or result.chain == chains[v]:
                    continue
                cand = self._make(
                    Opcode.REWRITE_ONE,
                    chains,
                    {v: result.chain},
                    routes=tuple(
                        (owned, v)
                        for path in result.paths
                        if (owned := tuple(q for q in path if q in result.chain))
                    ),
                    proposal_work=proposal_work,
                    provenance=f"single:{v}:r{rank}",
                )
                out.append((cand, result.expansions))
        return out

    def _repair(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        rng: random.Random,
        budget: int,
        meter: WorkMeter,
    ) -> list[tuple[Candidate, int]]:
        out: list[tuple[Candidate, int]] = []
        seeds = repair_seeds(chains, self.logical, self.host, self.ctx.group_sizes)
        rng.shuffle(seeds)
        for seed in seeds:
            if len(out) >= budget or not meter.can_charge(1):
                break
            rebuilt = rebuild_group(
                self.host,
                chains,
                self.logical,
                seed.group,
                rng=rng,
                overfill=self.overfill,
                jitter_scale=self.jitter_scale,
                root_rank=rng.randrange(2),
                expansion_budget=min(4_000, meter.remaining_expansions()),
            )
            proposal_work = meter.charge(rebuilt.expansions)
            if rebuilt.placed is None:
                continue
            placed = rebuilt.placed
            expansions = rebuilt.expansions
            if all(placed[i] == chains.get(i) for i in placed):
                continue
            out.append(
                (
                    self._make(
                        Opcode.REPAIR_GROUP,
                        chains,
                        placed,
                        proposal_work=proposal_work,
                        target_demand=seed.target_demand,
                        target_conflict=seed.target_conflict,
                        provenance="repair",
                    ),
                    expansions,
                )
            )
        return out

    def _group(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        rng: random.Random,
        size: int,
        budget: int,
        meter: WorkMeter,
        opcode: Opcode = Opcode.REWRITE_GROUP,
        seeds: Sequence[tuple[Node, ...]] | None = None,
    ) -> list[tuple[Candidate, int]]:
        out: list[tuple[Candidate, int]] = []
        if size > len(chains):
            return out
        pool: list[tuple[Node, ...]]
        if seeds is not None:
            pool = [g for g in seeds if g]
        else:
            variables = list(chains)
            rng.shuffle(variables)
            pool = []
            for v in variables:
                group = [v] + strongest_neighbours(self.problem, self.logical, v, size - 1)
                if len(group) < size:
                    extra = [u for u in variables if u not in group][: size - len(group)]
                    group += extra
                pool.append(tuple(group[:size]))
        for group in pool:
            if len(out) >= budget or not meter.can_charge(1):
                break
            group = tuple(dict.fromkeys(group))[:size]
            if len(group) < min(size, 2):
                continue
            rebuilt = rebuild_group(
                self.host,
                chains,
                self.logical,
                group,
                rng=rng,
                overfill=self.overfill,
                jitter_scale=self.jitter_scale,
                root_rank=rng.randrange(2),
                expansion_budget=min(4_000, meter.remaining_expansions()),
            )
            proposal_work = meter.charge(rebuilt.expansions)
            if rebuilt.placed is None:
                continue
            placed = rebuilt.placed
            expansions = rebuilt.expansions
            if all(placed[i] == chains.get(i) for i in placed):
                continue
            out.append(
                (
                    self._make(
                        opcode,
                        chains,
                        placed,
                        proposal_work=proposal_work,
                        provenance=f"{opcode.value.lower()}:{size}",
                    ),
                    expansions,
                )
            )
        return out

    def _restart(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        rng: random.Random,
        budget: int,
        meter: WorkMeter,
        restart_cache: Sequence[RestartCacheSlot] | None = None,
    ) -> list[tuple[Candidate, int]]:
        if budget <= 0:
            return []
        if self.mode is Mode.CONSTRUCTION:
            if not any(chains.values()) or not meter.can_charge(0):
                return []
            proposal_work = meter.charge(0)
            return [
                (
                    self._make(
                        Opcode.RESTART,
                        chains,
                        {node: frozenset() for node in self.logical.nodes()},
                        proposal_work=proposal_work,
                        provenance="restart:empty",
                    ),
                    0,
                )
            ]
        if restart_cache is not None:
            out: list[tuple[Candidate, int]] = []
            seen_embeddings: set[str] = set()
            for slot in restart_cache:
                if len(out) >= budget or not meter.can_charge(0):
                    break
                if slot.status != "SUCCESS" or slot.consumed or slot.chains is None:
                    continue
                proposal_work = meter.charge(0)
                embedding_key = chain_key(slot.chains)
                if embedding_key in seen_embeddings:
                    continue
                seen_embeddings.add(embedding_key)
                after = tuple(
                    item.consume() if item.slot_index == slot.slot_index else item
                    for item in restart_cache
                )
                after_digest = restart_cache_digest(after)
                out.append(
                    (
                        self._make(
                            Opcode.RESTART,
                            chains,
                            dict(slot.chains),
                            proposal_work=proposal_work,
                            restart_cache_slot=slot.slot_index,
                            restart_cache_after_digest=after_digest,
                            provenance=f"restart-cache:{slot.slot_index}",
                        ),
                        0,
                    )
                )
            return out
        if self.improvement_restart_protocol != LEGACY_ONLINE_INITIALIZER_RESTARTS_V1:
            raise RuntimeError(
                "registered improvement proposals require an authenticated restart cache"
            )
        if self.initializer is None:
            return []
        out: list[tuple[Candidate, int]] = []
        for _ in range(budget):
            if not meter.can_restart():
                break
            proposal_work = meter.charge(0, restart_work=1)
            seed = rng.randrange(2**31)
            fresh = self.initializer(self.logical, self.host, seed)
            if fresh is None:
                continue
            out.append(
                (
                    self._make(
                        Opcode.RESTART,
                        chains,
                        dict(fresh),
                        proposal_work=proposal_work,
                        provenance=f"legacy-online-restart-v1:{seed}",
                    ),
                    0,
                )
            )
        return out

    def _restore(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        archive: Sequence[ArchiveEntry],
        budget: int,
        meter: WorkMeter,
    ) -> list[tuple[Candidate, int]]:
        """Bind partial restoration of archived chains as ordinary group rewrites.

        The archive reference is retained so exact masking can verify every NEW chain.  A
        restore neither consumes a selected-restart token nor receives free materialisation.
        """

        if budget <= 0:
            return []
        nodes = sorted(self.logical.nodes(), key=str)
        eligible = [node for node in nodes if chains.get(node)]
        sizes = [size for size in self.ctx.group_sizes if size <= len(eligible)]
        out: list[tuple[Candidate, int]] = []
        seen_groups: set[tuple[int, tuple[Node, ...]]] = set()
        for archive_ref, entry in enumerate(archive):
            if (
                not entry.admissible
                or set(entry.chains) != set(self.logical.nodes())
                or any(not entry.chains.get(node) for node in nodes)
            ):
                continue
            changed = [
                node
                for node in eligible
                if frozenset(chains[node]) != frozenset(entry.chains[node])
            ]
            for size in sizes:
                for seed in changed:
                    if len(out) >= budget or not meter.can_charge(0):
                        return out
                    # Prefer other changed chains, then strongly interacting neighbours,
                    # then stable logical order.  This is outcome-blind and reproducible.
                    ordered = [seed]
                    ordered.extend(node for node in changed if node != seed)
                    ordered.extend(strongest_neighbours(self.problem, self.logical, seed, size - 1))
                    ordered.extend(eligible)
                    selected = tuple(dict.fromkeys(ordered))[:size]
                    group = tuple(sorted(selected, key=str))
                    if len(group) != size:
                        continue
                    group_key = (archive_ref, group)
                    if group_key in seen_groups:
                        continue
                    seen_groups.add(group_key)
                    proposal_work = meter.charge(0)
                    replacements = {node: frozenset(entry.chains[node]) for node in group}
                    out.append(
                        (
                            self._make(
                                Opcode.REWRITE_GROUP,
                                chains,
                                replacements,
                                proposal_work=proposal_work,
                                archive_ref=archive_ref,
                                provenance=f"restore:{archive_ref}:k{size}",
                            ),
                            0,
                        )
                    )
        return out

    def _restart_restore(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        archive: Sequence[ArchiveEntry],
        rng: random.Random,
        budget: int,
        meter: WorkMeter,
        *,
        allow_restart: bool,
        restart_cache: Sequence[RestartCacheSlot] | None,
    ) -> list[tuple[Candidate, int]]:
        """Fill the shared pool, reserving cached restart actions before restores."""

        if allow_restart and restart_cache is not None:
            restarted = self._restart(
                chains,
                rng,
                budget,
                meter,
                restart_cache=restart_cache,
            )
            remaining = max(0, budget - len(restarted))
            return restarted + self._restore(chains, archive, remaining, meter)
        # The legacy branch used to hand the whole pool to restore and give restart whatever
        # was left, which was nothing: four raw restores can all rebuild the same physical
        # assignment while differing in payload and work, so they survive the payload-key
        # dedup downstream and occupy every slot. One rewrite was enough to leave a state with
        # restarts_left = 2 and no legal RESTART in it. An external audit reproduced this.
        #
        # Identical successors are collapsed to the cheapest first, so a slot buys a distinct
        # state rather than a different way of reaching the same one, and the escape family
        # keeps a reserved share of the pool whenever it is allowed and has allowance left.
        restored = _collapse_identical_successors(self._restore(chains, archive, budget, meter))
        if allow_restart and budget > 0:
            reserved = max(1, budget // 2)
            restored = restored[: max(0, budget - reserved)]
        remaining = max(0, budget - len(restored))
        restarted = self._restart(chains, rng, remaining, meter) if allow_restart else []
        return restored + restarted

    # -- assembly -------------------------------------------------------------------

    def _make(
        self,
        opcode: Opcode,
        chains: Mapping[Node, frozenset[Qubit]],
        new_chains: Mapping[Node, frozenset[Qubit]],
        *,
        routes: tuple[tuple[tuple[Qubit, ...], Node], ...] = (),
        proposal_work: WorkVector,
        provenance: str = "",
        target_demand: tuple[Node, Node] | None = None,
        target_conflict: Qubit | None = None,
        archive_ref: int | None = None,
        restart_cache_slot: int | None = None,
        restart_cache_after_digest: str | None = None,
    ) -> Candidate:
        affected = tuple(sorted(new_chains, key=str))
        successor = dict(chains)
        successor.update(new_chains)
        work = WorkVector(
            decisions=1,
            route_expansions=0,
            materializations=0,
            # Admission checks the selected realised workspace at all four registered
            # strengths.  This is an application bound, not proposal work.
            compiler_calls=4,
            validator_calls=1,
            feature_work=8 * len(affected),
        )
        return Candidate(
            opcode=opcode,
            affected=affected,
            old_chains={i: chains.get(i, frozenset()) for i in affected},
            new_chains={i: frozenset(c) for i, c in new_chains.items()},
            work=work,
            payload_key=bound_successor_key(
                successor,
                work,
                restart=opcode is Opcode.RESTART,
                restart_cache_after_digest=restart_cache_after_digest,
            ),
            proposal_work=proposal_work,
            routes=routes,
            archive_ref=archive_ref,
            target_demand=target_demand,
            target_conflict=target_conflict,
            restart_cache_slot=restart_cache_slot,
            restart_cache_after_digest=restart_cache_after_digest,
            provenance=provenance,
        )

    def generate(
        self,
        chains: Mapping[Node, frozenset[Qubit]],
        rng: random.Random,
        *,
        restarts_left: int = 0,
        archive: Sequence[ArchiveEntry] = (),
        restart_cache: Sequence[RestartCacheSlot] | None = None,
        allowance: WorkVector | None = None,
    ) -> ProposalBatch:
        """Fill from a fixed family order until the common work allowance is exhausted.

        The low-level generator retains one versioned compatibility mode for tests and
        non-publication diagnostics. Registered Profile-I environments always construct it
        with ``AUTHENTICATED_RESTART_CACHE_V1``.
        """

        if (
            self.mode is Mode.IMPROVEMENT
            and restarts_left > 0
            and int(self.ctx.quotas.get("restart", 0)) > 0
            and restart_cache is None
            and self.improvement_restart_protocol
            != LEGACY_ONLINE_INITIALIZER_RESTARTS_V1
        ):
            raise RuntimeError(
                "registered improvement proposals require an authenticated restart cache"
            )

        quotas = dict(
            self.ctx.construction_quotas if self.mode is Mode.CONSTRUCTION else self.ctx.quotas
        )
        # One state per proposal round: the preference of a (variable, qubit) pair is asked
        # by several families and is the same for all of them.
        self._pref_cache = {}
        common_allowance = allowance or self.ctx.caps
        meter = WorkMeter(
            route_expansions=common_allowance.route_expansions,
            materializations=common_allowance.materializations,
            restart_work=common_allowance.restart_work,
        )
        families: list[tuple[str, Callable[[int], list[tuple[Candidate, int]]]]] = []
        if self.mode is Mode.CONSTRUCTION:
            empty = any(not chain for chain in chains.values())
            families.append(("place", lambda budget: self._place(chains, budget, meter)))
            families.append(("route", lambda budget: self._route_demands(chains, budget, meter)))
            families.append(("grow", lambda budget: self._grow(chains, budget, meter)))
            families.append(("shrink", lambda budget: self._shrink(chains, budget, meter)))
            # Ordinary rewrites cannot empty a chain, so defer them until every variable
            # has been placed.  Repair remains useful once a selected defect is bound.
            if not empty:
                families.append(
                    ("rewrite", lambda budget: self._single(chains, rng, budget, meter))
                )
                families.append(("repair", lambda budget: self._repair(chains, rng, budget, meter)))
        else:
            families.append(("single", lambda budget: self._single(chains, rng, budget, meter)))
            for size in self.ctx.group_sizes:
                key = f"group{size}"
                families.append(
                    (
                        key,
                        lambda budget, size=size: self._group(chains, rng, size, budget, meter),
                    )
                )
            families.append(("repair", lambda budget: self._repair(chains, rng, budget, meter)))
        if restarts_left > 0 or archive:
            families.append(
                (
                    "restart",
                    lambda budget: self._restart_restore(
                        chains,
                        archive,
                        rng,
                        budget,
                        meter,
                        allow_restart=restarts_left > 0,
                        restart_cache=restart_cache,
                    ),
                )
            )

        seen: set[str] = set()
        kept: list[Candidate] = []
        counts: dict[str, int] = {name: 0 for name in quotas}
        counts["restore"] = 0

        def retain(name: str, items: Sequence[tuple[Candidate, int]]) -> None:
            for cand, _used in items:
                if len(kept) >= self.ctx.max_state_changing:
                    return
                if cand.payload_key in seen:
                    continue
                seen.add(cand.payload_key)
                kept.append(cand)
                count_name = "restore" if cand.archive_ref is not None else name
                counts[count_name] = counts.get(count_name, 0) + 1

        # First honour every registered allocation in the documented family order.
        active_families = [(name, build) for name, build in families if quotas.get(name, 0) > 0]
        for name, build in active_families:
            if len(kept) >= self.ctx.max_state_changing or meter.exhausted():
                break
            budget = min(quotas[name], self.ctx.max_state_changing - len(kept))
            retain(name, build(budget))

        # Deterministically reassign quota shortfall.  Each family gets the entire remaining
        # row allowance once, in the same fixed order.  A family routine exhausts its finite
        # candidate pool or the shared meter; a second pass would only replay deterministic
        # PLACE/ROUTE alternatives and charge duplicates.
        if len(kept) < self.ctx.max_state_changing and not meter.exhausted():
            for name, build in active_families:
                # Restarts and restores share a hard four-attempt pool.  Other families may
                # absorb quota shortfall; this safety/escape family may not exceed its
                # registered pool merely because an unrelated family was inapplicable.
                if name == "restart":
                    continue
                if len(kept) >= self.ctx.max_state_changing or meter.exhausted():
                    break
                retain(name, build(self.ctx.max_state_changing - len(kept)))
        # Generation work only: the environment charges preparation feature work and the
        # per-application validation separately, so nothing is counted twice (Eq. 3i).
        work = WorkVector(
            route_expansions=meter.used_expansions,
            materializations=meter.used_materializations,
            restart_work=meter.used_restart_work,
        )
        return ProposalBatch(
            tuple(kept),
            work,
            counts,
            meter.used_materializations - len(kept),
        )


def _occupancy_excluding(
    chains: Mapping[Node, frozenset[Qubit]], drop: set[Node]
) -> dict[Qubit, int]:
    occ: dict[Qubit, int] = {}
    for i, c in chains.items():
        if i in drop:
            continue
        for q in c:
            occ[q] = occ.get(q, 0) + 1
    return occ


def router_initializer(
    overfill: float = 4.0, jitter_scale: float = 0.1, attempts: int = 8
) -> Initializer:
    """A self-contained greedy initializer: place variables in degree order with the router.

    Useful where minorminer is unavailable and as a second registered initializer; its
    witness is a feasibility result, never a quality label.
    """

    def _init(logical: nx.Graph, host: nx.Graph, seed: int):
        rng = random.Random(seed)
        order = sorted(logical.nodes(), key=lambda v: (-logical.degree(v), str(v)))
        for attempt in range(attempts):
            rebuilt = rebuild_group(
                host,
                {v: frozenset() for v in logical.nodes()},
                logical,
                order,
                rng=rng,
                overfill=overfill,
                jitter_scale=jitter_scale,
                root_rank=0,
                order=order,
                exclusive=True,
                expansion_budget=50_000,
            )
            if rebuilt.placed is None:
                continue
            placed = rebuilt.placed
            occ: dict = {}
            for c in placed.values():
                for q in c:
                    occ[q] = occ.get(q, 0) + 1
            if occ and max(occ.values()) == 1 and all(placed.values()):
                missing = [
                    (u, v)
                    for u, v in logical.edges()
                    if not any(a != b and host.has_edge(a, b) for a in placed[u] for b in placed[v])
                ]
                if not missing:
                    return {v: frozenset(c) for v, c in placed.items()}
        return None

    return _init


def minorminer_initializer(tries: int = 10) -> Initializer:
    """The registered improvement initializer. Its witness is not a quality label."""

    def _init(logical: nx.Graph, host: nx.Graph, seed: int):
        try:
            import minorminer
        except ImportError:  # pragma: no cover - optional baseline dependency
            return None
        emb = minorminer.find_embedding(
            list(logical.edges()), list(host.edges()), random_seed=seed % (2**31), tries=tries
        )
        if not emb or set(emb) != set(logical.nodes()):
            return None
        return {v: frozenset(c) for v, c in emb.items()}

    return _init

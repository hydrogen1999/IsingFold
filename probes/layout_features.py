"""Versioned layout observations with residual capacity and coupling-aware routing.

The original 35 candidate channels are retained exactly. Thirty additional channels
describe the current free host, not a planted solution or a promised completion. Their
values lie in [-1, 1]; signed channels are deliberately not gauge invariant. This is a
layout representation ablation, not a certificate of feasibility or annealing quality.

For every candidate anchor q, branch capacity is the size of the connected component
reachable through a free neighbour after removing q. Cyclic neighbours can lead to the
same component: their capacities must not be added. Native-coordinate ``directions''
mean increases/decreases of topology indices, not Euclidean compass directions.

Residual components and all branch capacities are computed once per membership state
with an iterative low-link DFS. Distances to each requested placed neighbour chain are
computed lazily by a multi-source BFS and reused by all candidate roots in that state.
"""
from bisect import bisect_right
from collections import deque

import numpy as np

from candidate_features import FeatureContext, WIDTH as BASE_WIDTH

FEATURE_VERSION = "layout-v2"
FEATURE_NAMES = (
    "component_host_fraction", "component_free_fraction", "free_degree_fraction",
    "articulation", "smallest_branch_host_fraction", "largest_branch_host_fraction",
    "fragmentation_after_anchor", "dead_end_neighbour_fraction",
    *tuple(f"coord_{axis}_{direction}_branch_fraction"
           for axis in range(5) for direction in ("negative", "positive")),
    "coupling_weighted_distance", "coupling_weighted_contact",
    "coupling_weighted_unreachable", "signed_field",
    "signed_coupling_proximity", "signed_coupling_contact",
    "placed_coupling_share", "strongest_coupling_distance",
    "occupancy", "unplaced_variable_fraction", "unmet_edge_fraction",
    "remaining_budget_host_fraction",
)
WIDTH = BASE_WIDTH + len(FEATURE_NAMES)
FEATURE_INDEX = {name: BASE_WIDTH + i for i, name in enumerate(FEATURE_NAMES)}
FEATURE_SLICES = {
    "legacy": slice(0, BASE_WIDTH), "capacity": slice(BASE_WIDTH, BASE_WIDTH + 8),
    "directions": slice(BASE_WIDTH + 8, BASE_WIDTH + 18),
    "couplings": slice(BASE_WIDTH + 18, BASE_WIDTH + 26),
    "global": slice(BASE_WIDTH + 26, WIDTH),
}


def _residual_capacities(adjacency):
    """Return per-node component size and per-directed-edge onward capacity.

    Removing a DFS node separates precisely its children whose low-link is at least
    the node's discovery time, plus a possible remainder. Binary search identifies
    the relevant child for non-tree descendant edges. No per-candidate graph copies
    or recursive DFS are needed; memory is linear in the residual graph size.
    """
    tin, low, parent, children, size, end = {}, {}, {}, {}, {}, {}
    component_sizes = {}
    clock = 0
    for root in adjacency:
        if root in tin:
            continue
        tin[root] = low[root] = clock
        clock += 1
        parent[root], children[root] = None, []
        nodes = [root]
        stack = [(root, iter(adjacency[root]))]
        while stack:
            q, neighbours = stack[-1]
            try:
                r = next(neighbours)
            except StopIteration:
                stack.pop()
                size[q] = 1 + sum(size[c] for c in children[q])
                end[q] = clock - 1
                if parent[q] is not None:
                    low[parent[q]] = min(low[parent[q]], low[q])
                continue
            if r == parent[q] or r == q:
                continue
            if r in tin:
                low[q] = min(low[q], tin[r])
                continue
            parent[r], children[r] = q, []
            children[q].append(r)
            tin[r] = low[r] = clock
            clock += 1
            nodes.append(r)
            stack.append((r, iter(adjacency[r])))
        component_sizes.update({q: len(nodes) for q in nodes})

    branches, articulation = {}, {}
    for q, neighbours in adjacency.items():
        separated = {c for c in children[q] if low[c] >= tin[q]}
        remainder = component_sizes[q] - 1 - sum(size[c] for c in separated)
        articulation[q] = len(separated) + int(remainder > 0) > 1
        child_times = [tin[c] for c in children[q]]
        branches[q] = {}
        for r in neighbours:
            capacity = remainder
            if tin[q] < tin[r] <= end[q]:
                i = bisect_right(child_times, tin[r]) - 1
                child = children[q][i]
                if child in separated:
                    capacity = size[child]
            branches[q][r] = capacity
    return component_sizes, branches, articulation


class LayoutFeatureContext(FeatureContext):
    """An all-free-root policy's state context; never reads a task's witness."""

    feature_version = FEATURE_VERSION
    width = WIDTH

    def __init__(self, task, budget=None):
        super().__init__(task, budget)
        if self.host.is_directed() or self.host.is_multigraph():
            raise ValueError("layout features require a simple undirected host")
        problem = getattr(task, "problem", None)
        self._h = dict(getattr(problem, "h", {}) or {})
        self._j = {}
        for (u, v), value in dict(getattr(problem, "j", {}) or {}).items():
            key = frozenset((u, v))
            if key in self._j:
                raise ValueError("logical couplings must have one entry per undirected edge")
            self._j[key] = float(value)
        if not all(np.isfinite(float(x)) for x in (*self._h.values(), *self._j.values())):
            raise ValueError("layout coefficients must be finite")
        self._host_size = max(1, self.host.number_of_nodes())
        self._degree_scale = max(1, max(self.host_degree.values(), default=0))
        self._layout_state_key = None
        self._distances = {}

    def _ensure_layout_state(self, chains):
        allowed, placed = self.state(chains)
        if self._layout_state_key == self._state_key:
            return
        self._layout_state_key = self._state_key
        self._free = set(allowed)
        self._adjacency = {q: tuple(r for r in self.host.neighbors(q)
                                    if r in self._free and r != q) for q in self._free}
        self._components, self._branches, self._articulation = _residual_capacities(self._adjacency)
        self._distances = {}
        self._layout_placed = placed
        occupied = {q for chain in placed.values() for q in chain}
        owners = {q: v for v, chain in placed.items() for q in chain}
        contacts = {frozenset((owners[q], owners[r])) for q in occupied
                    for r in self.host.neighbors(q) if r in owners and owners[q] != owners[r]}
        unmet = sum(frozenset((u, v)) not in contacts for u, v in self.logical.edges())
        self._global = [len(occupied) / self._host_size,
                        sum(v not in placed for v in self.logical) / max(1, len(self.logical)),
                        unmet / max(1, self.logical.number_of_edges()),
                        np.clip((self.budget - len(occupied)) / self._host_size, -1.0, 1.0)]

    def _distance_to_chain(self, variable):
        """Free-only paths may touch their target chain at the final edge."""
        if variable not in self._distances:
            sources = {r for q in self._layout_placed[variable]
                       for r in self.host.neighbors(q) if r in self._free}
            distances = {r: 1 for r in sources}
            queue = deque(sources)
            while queue:
                q = queue.popleft()
                for r in self._adjacency[q]:
                    if r not in distances:
                        distances[r] = distances[q] + 1
                        queue.append(r)
            self._distances[variable] = distances
        return self._distances[variable]

    def _capacity_features(self, anchor):
        if anchor not in self._free:
            return [0.0] * 18
        branches = self._branches[anchor]
        capacities = list(branches.values())
        component = self._components[anchor]
        largest = max(capacities, default=0)
        values = [component / self._host_size, component / max(1, len(self._free)),
                  len(branches) / self._degree_scale, float(self._articulation[anchor]),
                  min(capacities, default=0) / self._host_size, largest / self._host_size,
                  1.0 - largest / (component - 1) if component > 1 else 0.0,
                  sum(len(self._adjacency[r]) <= 1 for r in branches) / max(1, len(branches))]
        directions = [0.0] * 10
        if anchor in self.coords:
            c0 = self.coords[anchor]
            for neighbour, capacity in branches.items():
                if neighbour not in self.coords:
                    continue
                for axis, (a, b) in enumerate(zip(c0[:5], self.coords[neighbour][:5])):
                    if abs(b - a) <= 1e-12:
                        continue
                    index = 2 * axis + int(b > a)
                    directions[index] = max(directions[index], capacity / self._host_size)
        return values + directions

    def _coupling_features(self, variable, anchor):
        field = float(self._h.get(variable, 0.0)) / self.h_scale
        if variable not in self.logical:
            return [0.0] * 8
        neighbours = list(self.logical.neighbors(variable))
        coefficients = {u: self._j.get(frozenset((variable, u)), 0.0) for u in neighbours}
        total = sum(abs(x) for x in coefficients.values())
        placed = {u: j for u, j in coefficients.items() if u in self._layout_placed and j != 0.0}
        weight = sum(abs(x) for x in placed.values())
        if weight == 0.0:
            return [0.0, 0.0, 0.0, field, 0.0, 0.0, 0.0, 0.0]
        distance, contact, unreachable, signed_proximity, signed_contact = 0.0, 0.0, 0.0, 0.0, 0.0
        strongest = max(abs(j) for j in placed.values())
        strongest_distances = []
        for u, j in placed.items():
            d = self._distance_to_chain(u).get(anchor)
            normalized = d / (d + 1.0) if d is not None else 1.0
            distance += abs(j) * normalized
            contact += abs(j) * (d == 1)
            unreachable += abs(j) * (d is None)
            signed_proximity += j / (d + 1.0) if d is not None else 0.0
            signed_contact += j * (d == 1)
            if abs(j) == strongest:
                strongest_distances.append(normalized)
        return [distance / weight, contact / weight, unreachable / weight, field,
                signed_proximity / weight, signed_contact / weight,
                weight / total, float(np.mean(strongest_distances))]

    def pair(self, variable, qubits, chains, opcode="PLACE"):
        """Return legacy35 + layout30 float32 channels; extras describe PLACE only.

        The current budget/occupancy is supplied, not a hypothetical planted budget.
        All new features are observations. They never mask a root or penalize qubit use.
        """
        qubits = tuple(qubits)
        legacy = super().pair(variable, qubits, chains, opcode)
        if opcode != "PLACE" or not qubits:
            return np.concatenate((legacy, np.zeros(len(FEATURE_NAMES), dtype=np.float32)))
        self._ensure_layout_state(chains)
        anchor = sorted(qubits, key=str)[0]
        extra = self._capacity_features(anchor) + self._coupling_features(variable, anchor) + self._global
        return np.concatenate((legacy, np.asarray(extra, dtype=np.float32)))

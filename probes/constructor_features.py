"""Outcome-blind observations for a constructor's fully materialized actions.

Every action exposes its actual successor, including the archived embedding a COMMIT
selects. Features never consume witness chains, ground energy, evaluator labels, or a
minorminer completion. Connectivity/contact/load summaries are observations, not a
certificate of downstream quality. This finite summary is not claimed to be Markov.

The legacy 35 channels are followed by opcode/subtype indicators, local before/after
capacity and coupling features, global before/after summaries and their differences,
action edits, and explicit work/horizon context. Local capacity temporarily releases
the observed chain while keeping every other chain as an obstacle. At most eight
deterministically chosen chain anchors are pooled; global summaries inspect all chains.
"""
from collections import OrderedDict

import networkx as nx
import numpy as np

from candidate_features import FeatureContext, WIDTH as LEGACY_WIDTH
from layout_features import LayoutFeatureContext, WIDTH as LAYOUT_WIDTH
from isingfold.rl.contracts import OPCODES, WORK_FIELDS

FEATURE_VERSION = "constructor-v1"
SUBTYPES = ("grow", "shrink", "rewrite", "restore")
SUMMARY_NAMES = (
    "occupied_fraction", "membership_fraction", "placed_fraction", "realized_edge_fraction",
    "realized_coupling_fraction", "overlap_fraction", "connected_chain_fraction",
    "outside_host_fraction", "max_chain_fraction", "mean_chain_fraction",
    "internal_edge_fraction", "bridge_fraction", "leaf_fraction", "cycle_fraction",
    "contact_edge_fraction", "weighted_contact_multiplicity", "contact_load_concentration",
    "largest_free_component_fraction", "free_component_count_fraction", "budget_left_fraction",
    "structurally_complete", "within_budget", "signed_realized_coupling", "field_load_concentration",
)
_BLOCKS = (
    ("legacy", LEGACY_WIDTH), ("opcode", len(OPCODES)), ("subtype", len(SUBTYPES)),
    ("local_before", LAYOUT_WIDTH - LEGACY_WIDTH), ("local_after", LAYOUT_WIDTH - LEGACY_WIDTH),
    ("before", len(SUMMARY_NAMES)), ("after", len(SUMMARY_NAMES)), ("delta", len(SUMMARY_NAMES)),
    ("edits", 8), ("remaining_work", len(WORK_FIELDS)), ("usable_work", len(WORK_FIELDS)),
    ("action_work", len(WORK_FIELDS)), ("proposal_work", len(WORK_FIELDS)), ("context", 7),
)
FEATURE_SLICES, WIDTH = {}, 0
for _name, _size in _BLOCKS:
    FEATURE_SLICES[_name] = slice(WIDTH, WIDTH + _size)
    WIDTH += _size
SUMMARY_INDEX = {name: i for i, name in enumerate(SUMMARY_NAMES)}


def _key(chains):
    return frozenset((v, frozenset(c)) for v, c in chains.items() if c)


def _frozen(chains):
    return {v: frozenset(c) for v, c in chains.items() if c}


def _order(node):
    return type(node).__module__, type(node).__qualname__, repr(node)


def _opcode(candidate):
    return getattr(candidate.opcode, "value", candidate.opcode)


class ConstructorFeatureContext:
    feature_version = FEATURE_VERSION
    width = WIDTH

    def __init__(self, task, budget=None):
        self.task = task
        self.legacy = FeatureContext(task, budget)
        self.budget = self.legacy.budget
        self.host, self.logical = task.host, task.logical
        self._host_nodes = set(self.host)
        # adjacency lists once; the per-candidate summary walks them instead of building
        # networkx subgraph views, whose filter layer dominated the step cost on large hosts
        self._neighbours = {q: tuple(self.host[q]) for q in self.host}
        self._m, self._n = max(1, len(self.host)), max(1, len(self.logical))
        self._host_edges = max(1, self.host.number_of_edges())
        problem = getattr(task, "problem", None)
        self._h = dict(getattr(problem, "h", {}) or {})
        self._j = {frozenset((u, v)): float(j)
                   for (u, v), j in dict(getattr(problem, "j", {}) or {}).items()}
        if not all(np.isfinite(float(x)) for x in (*self._h.values(), *self._j.values())):
            raise ValueError("constructor coefficients must be finite")
        self._coupling_mass = sum(abs(j) for j in self._j.values())
        self._state_key = None
        self._summary_cache = {}
        self._local_cache = OrderedDict()
        self._chain_graph_cache = OrderedDict()

    def _chain_graph_summary(self, chain):
        """Reuse unchanged chain topology across a candidate batch and later states."""
        chain = frozenset(chain)
        if chain not in self._chain_graph_cache:
            graph = self.host.subgraph(chain)
            components = nx.number_connected_components(graph)
            self._chain_graph_cache[chain] = (
                int(bool(chain) and set(chain) <= self._host_nodes and components == 1),
                graph.number_of_edges(), sum(1 for _ in nx.bridges(graph)),
                sum(d == 1 for _, d in graph.degree()),
                graph.number_of_edges() - len(graph) + components,
            )
            if len(self._chain_graph_cache) > 2048:
                self._chain_graph_cache.popitem(last=False)
        self._chain_graph_cache.move_to_end(chain)
        return self._chain_graph_cache[chain]

    def _successor(self, candidate, current, state):
        opcode = _opcode(candidate)
        if opcode == "COMMIT":
            ref = getattr(candidate, "archive_ref", None)
            archive = getattr(state, "archive", None)
            if (isinstance(ref, bool) or not isinstance(ref, (int, np.integer))
                    or archive is None or not 0 <= ref < len(archive)):
                raise ValueError("COMMIT observation requires a valid state.archive reference")
            return _frozen(archive[ref].chains)
        if opcode == "STOP":
            return dict(current)
        successor = dict(current)
        successor.update({v: frozenset(c) for v, c in candidate.new_chains.items()})
        return _frozen(successor)

    def _summary(self, chains):
        signature = _key(chains)
        if signature in self._summary_cache:
            return self._summary_cache[signature]
        occupied = set().union(*chains.values()) if chains else set()
        memberships = sum(len(c) for c in chains.values())
        valid_nodes = occupied <= self._host_nodes
        internal = bridges = leaves = cycles = connected = 0
        owners = {}
        for v, chain in chains.items():
            for q in chain:
                owners.setdefault(q, set()).add(v)
            chain_connected, chain_internal, chain_bridges, chain_leaves, chain_cycles = self._chain_graph_summary(chain)
            connected += chain_connected
            internal += chain_internal
            bridges += chain_bridges
            leaves += chain_leaves
            cycles += chain_cycles
        contacts = {}
        # only edges incident to occupied qubits can be contacts; both directions of an
        # edge collapse in the frozenset keys, so this equals the walk over host.edges()
        for q, q_owners in owners.items():
            for r in self._neighbours.get(q, ()):
                r_owners = owners.get(r)
                if not r_owners:
                    continue
                for v in q_owners:
                    for u in r_owners:
                        if v != u and self.logical.has_edge(v, u):
                            contacts.setdefault(frozenset((u, v)), set()).add(frozenset((q, r)))
        realized = len(contacts)
        mass = sum(abs(self._j.get(edge, 0.0)) for edge in contacts)
        signed = sum(self._j.get(edge, 0.0) for edge in contacts)
        weighted_multiplicity = sum(abs(self._j.get(edge, 0.0)) * len(pairs) / (1 + len(pairs))
                                    for edge, pairs in contacts.items())
        contact_loads = {}
        for edge, pairs in contacts.items():
            load = abs(self._j.get(edge, 0.0)) / len(pairs)
            for pair in pairs:
                for q in pair:
                    contact_loads[q] = contact_loads.get(q, 0.0) + load
        # A declared equal-contact/equal-field proxy, not the compiler's physical program.
        field_loads = {}
        for v, chain in chains.items():
            load = abs(float(self._h.get(v, 0.0))) / len(chain)
            for q in chain:
                field_loads[q] = field_loads.get(q, 0.0) + load
        free_components = self._free_component_sizes(occupied)
        within = len(occupied) <= self.budget
        complete = (set(chains) == set(self.logical) and connected == len(self.logical)
                    and memberships == len(occupied) and valid_nodes
                    and realized == self.logical.number_of_edges())
        row = np.asarray([
            len(occupied) / self._m, memberships / self._m, len(chains) / self._n,
            realized / max(1, self.logical.number_of_edges()), mass / max(self._coupling_mass, 1e-30),
            (memberships - len(occupied)) / self._m, connected / self._n,
            len(occupied - self._host_nodes) / self._m,
            max(map(len, chains.values()), default=0) / self._m,
            (memberships / max(1, len(chains))) / self._m, internal / self._host_edges,
            bridges / max(1, internal), leaves / max(1, memberships), cycles / self._host_edges,
            sum(len(c) for c in contacts.values()) / self._host_edges,
            weighted_multiplicity / max(self._coupling_mass, 1e-30),
            max(contact_loads.values(), default=0.0) / max(sum(contact_loads.values()), 1e-30),
            max(free_components, default=0) / self._m, len(free_components) / self._m,
            (self.budget - len(occupied)) / self._m, float(complete), float(within),
            signed / max(self._coupling_mass, 1e-30),
            max(field_loads.values(), default=0.0) / max(sum(field_loads.values()), 1e-30),
        ], dtype=np.float32)
        self._summary_cache[signature] = np.clip(row, -1.0, 1.0)
        return self._summary_cache[signature]

    def _free_component_sizes(self, occupied):
        """Sizes of the connected components of the host minus ``occupied``: a breadth-first
        walk over precomputed adjacency lists, equal to networkx's connected_components of
        the induced subgraph."""
        seen = set(occupied) & self._host_nodes
        sizes = []
        for start in self._neighbours:
            if start in seen:
                continue
            seen.add(start)
            frontier, size = [start], 1
            while frontier:
                node = frontier.pop()
                for nxt in self._neighbours[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        size += 1
                        frontier.append(nxt)
            sizes.append(size)
        return sizes

    def _local(self, chains, variables, preferred):
        pairs = []
        for v in sorted(set(variables), key=_order):
            chain = chains.get(v, ())
            if not chain:
                continue
            anchors = sorted(set(preferred.get(v, ())) & set(chain), key=_order)
            if not anchors:
                anchors = sorted(chain, key=_order)
            pairs.extend((v, q) for q in (anchors[:1] if len(anchors) == 1 else (anchors[0], anchors[-1])))
        rows = []
        for v, q in pairs[:8]:
            released = {u: c for u, c in chains.items() if u != v}
            signature = _key(released)
            if signature not in self._local_cache:
                self._local_cache[signature] = LayoutFeatureContext(self.task, self.budget)
                if len(self._local_cache) > 16:
                    self._local_cache.popitem(last=False)
            local = self._local_cache[signature]
            self._local_cache.move_to_end(signature)
            rows.append(local.pair(v, [q], released, "PLACE")[LEGACY_WIDTH:])
        return np.mean(rows, axis=0) if rows else np.zeros(LAYOUT_WIDTH - LEGACY_WIDTH, dtype=np.float32)

    def observe(self, candidate, chains, *, state=None, ctx=None, steps_left=0, max_steps=1):
        if (not np.isfinite(max_steps) or max_steps <= 0 or not np.isfinite(steps_left)
                or steps_left < 0):
            raise ValueError("step horizon must be finite, max_steps positive, steps_left nonnegative")
        opcode = _opcode(candidate)
        if opcode not in OPCODES:
            raise ValueError("unknown constructor opcode")
        current = _frozen(chains)
        signature = _key(current)
        if signature != self._state_key:
            self._state_key = signature
            self._summary_cache.clear()
            self._local_cache.clear()
        successor = self._successor(candidate, current, state)
        affected = tuple(getattr(candidate, "affected", ()))
        variables = affected if affected else tuple(set(current) | set(successor))
        added = {v: set(successor.get(v, ())) - set(current.get(v, ())) for v in variables}
        removed = {v: set(current.get(v, ())) - set(successor.get(v, ())) for v in variables}
        new_count, old_count = sum(map(len, added.values())), sum(map(len, removed.values()))
        occupied = set().union(*current.values()) if current else set()
        after_occupied = set().union(*successor.values()) if successor else set()
        base_chains = successor if opcode == "COMMIT" else current
        legacy = self.legacy.candidate((opcode, affected, tuple(set().union(*added.values()) if added else ())), base_chains)
        subtype = [0.0] * len(SUBTYPES)
        if opcode.startswith("REWRITE") or opcode == "REPAIR_GROUP":
            if getattr(candidate, "archive_ref", None) is not None:
                subtype[3] = 1.0
            elif new_count and not old_count:
                subtype[0] = 1.0
            elif old_count and not new_count:
                subtype[1] = 1.0
            else:
                subtype[2] = 1.0
        before, after = self._summary(current), self._summary(successor)
        edits = [new_count / self._m, old_count / self._m,
                 len(after_occupied - occupied) / self._m, len(occupied - after_occupied) / self._m,
                 len(affected) / self._n,
                 sum(len(path) for path, _ in getattr(candidate, "routes", ())) / self._m,
                 float(getattr(candidate, "target_demand", None) is not None),
                 float(getattr(candidate, "archive_ref", None) is not None)]

        def work_row(work, subtract=None):
            result = []
            for name in WORK_FIELDS:
                amount = float(getattr(work, name, 0.0)) - float(getattr(subtract, name, 0.0))
                cap = float(getattr(getattr(ctx, "caps", None), name, 0.0))
                result.append(np.clip(amount / cap, -1.0, 1.0) if cap > 0 else amount / (1 + abs(amount)))
            return result

        remaining = getattr(state, "remaining", None)
        horizon = np.clip(steps_left / max_steps, 0.0, 1.0)
        restarts = getattr(state, "restarts_left", 0)
        allowance = max(1, getattr(ctx, "restart_allowance", 2))
        context = [horizon, 1.0 - horizon, np.clip(restarts / allowance, 0.0, 1.0),
                   min(len(getattr(state, "archive", ())) / max(1, getattr(ctx, "max_commit", 8)), 1.0),
                   float(state is not None), float(ctx is not None), float(opcode in ("STOP", "COMMIT"))]
        row = np.concatenate((legacy, [float(opcode == op) for op in OPCODES], subtype,
                              self._local(current, variables, removed), self._local(successor, variables, added),
                              before, after, after - before, edits, work_row(remaining),
                              work_row(remaining, getattr(ctx, "reserve", None)),
                              work_row(getattr(candidate, "work", None)),
                              work_row(getattr(candidate, "proposal_work", None)), context)).astype(np.float32)
        if row.shape != (WIDTH,) or not np.isfinite(row).all():
            raise ValueError("constructor observation is nonfinite or has an invalid feature width")
        return row

"""Compact public-coefficient observations for quality-directed construction.

The first 20 channels are exactly ``Features(task, local_channels=True)``. Twelve
additional channels describe the before/after states of edited chains and their
logical neighbors. COMMIT instead describes every chain in the selected archive;
STOP describes the unchanged workspace. No witness, outcome, evaluator, or external
embedding solver is read. The features do not modify reward or penalize resources.

These are structural proxies, NOT chain-break probabilities or a physical simulator.
The load proxy splits |h_v| uniformly across chain v, and |J_uv| uniformly across
physical contacts between u and v. Its concentration is
``sum(load_q**2) / sum(load_q)**2``. The bridge proxy
is twice the greatest smaller-side load fraction over bridges in a chain. Cycle
redundancy is its cycle rank divided by its number of internal edges. The signed
channel is signed realized coupling mass, not an estimate of frustration.

For each observed chain v, contact coverage is sum(|J_vu| * 1[k_vu>0]) /
sum(|J_vu|), and redundancy replaces the indicator by 1-1/k_vu (zero if
k_vu=0). The signed channel uses J_vu in the coverage numerator. Chains are
pooled with fixed public weights |h_v|+sum_u|J_vu|; if every weight in the
scope is zero the pool is uniform. Absent chains contribute zeros. Thus all
extras are bounded, and before/after always use the same scope and weights.

Extra-channel cost depends on edited chains and their logical neighborhood, not a
scan of the full hardware. Bounded caches reuse unchanged chain and contact work.
The local20 prefix retains its existing cost. Host topology and problem coefficients
are immutable for the lifetime of a feature context.
"""
from collections import OrderedDict

import numpy as np

from constructor_tiny_gate import Features
from isingfold.rl.contracts import OPCODES


FEATURE_VERSION = "constructor-local-physics-v1"
BASE_WIDTH = len(OPCODES) + 12
SUMMARY_NAMES = (
    "weighted_realized_contacts", "weighted_contact_redundancy",
    "coefficient_load_concentration", "bridge_load_bottleneck_proxy",
    "cycle_redundancy", "signed_realized_coupling",
)
SUMMARY_INDEX = {name: i for i, name in enumerate(SUMMARY_NAMES)}
FEATURE_NAMES = tuple(f"{block}_{name}" for block in ("before", "after")
                      for name in SUMMARY_NAMES)
FEATURE_SLICES = {"local": slice(0, BASE_WIDTH),
                  "before": slice(BASE_WIDTH, BASE_WIDTH + len(SUMMARY_NAMES)),
                  "after": slice(BASE_WIDTH + len(SUMMARY_NAMES), BASE_WIDTH + 2 * len(SUMMARY_NAMES))}
WIDTH = BASE_WIDTH + len(FEATURE_NAMES)


def _cached(cache, key, compute, limit=4096):
    if key not in cache:
        cache[key] = compute()
        if len(cache) > limit:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


class PhysicsFeatures(Features):
    """Local20 plus six coefficient/topology summaries before and after an action."""

    feature_version = FEATURE_VERSION
    width = WIDTH

    def __init__(self, task):
        super().__init__(task, local_channels=True)
        problem = getattr(task, "problem", None)
        self._h = {v: float(h) for v, h in (getattr(problem, "h", {}) or {}).items()}
        self._j = {frozenset((u, v)): float(j)
                   for (u, v), j in (getattr(problem, "j", {}) or {}).items()}
        if not all(np.isfinite(x) for x in (*self._h.values(), *self._j.values())):
            raise ValueError("physics features require finite public coefficients")
        # A shared scale protects summation/squaring against overflow and preserves
        # all ratios, including the relative importance of fields versus couplings.
        scale = max((abs(x) for x in (*self._h.values(), *self._j.values())), default=0.)
        if scale:
            self._h = {v: h / scale for v, h in self._h.items()}
            self._j = {edge: j / scale for edge, j in self._j.items()}
        self._chain_mass = {v: abs(self._h.get(v, 0.)) + sum(
            abs(self._j.get(frozenset((v, u)), 0.)) for u in neighbors)
            for v, neighbors in self._logical_neighbours.items()}
        self._topology_cache = OrderedDict()
        self._contacts_cache = OrderedDict()
        self._physics_cache = OrderedDict()

    def _contacts(self, a, b):
        """Oriented physical contacts, so each first endpoint belongs to chain a."""
        return _cached(self._contacts_cache, (a, b), lambda: tuple(
            (q, r) for q in a for r in self._neighbours.get(q, ())
            if r in b and r != q))

    def _topology(self, chain):
        def compute():
            adjacent = {q: tuple(r for r in self._neighbours.get(q, ()) if r in chain)
                        for q in chain}
            order, parent, depth, low, component, bridges = [], {}, {}, {}, {}, []
            roots = []
            # Iterative DFS avoids recursion limits on long chains. Bridge children
            # and reverse DFS order later give all bridge-side load sums in O(|C|).
            for root in chain:
                if root in depth:
                    continue
                roots.append(root)
                parent[root] = None
                depth[root] = low[root] = len(order)
                component[root] = root
                order.append(root)
                stack = [(root, iter(adjacent[root]))]
                while stack:
                    q, it = stack[-1]
                    r = next(it, None)
                    if r is None:
                        stack.pop()
                        p = parent[q]
                        if p is not None:
                            low[p] = min(low[p], low[q])
                            if low[q] > depth[p]:
                                bridges.append(q)
                        continue
                    if r == parent[q]:
                        continue
                    if r in depth:
                        low[q] = min(low[q], depth[r])
                    else:
                        parent[r] = q
                        depth[r] = low[r] = len(order)
                        component[r] = root
                        order.append(r)
                        stack.append((r, iter(adjacent[r])))
            edges = sum(map(len, adjacent.values())) // 2
            cycle = max(0, edges - len(chain) + len(roots)) / max(1, edges)
            return order, parent, component, bridges, cycle
        return _cached(self._topology_cache, chain, compute)

    def _chain_summary(self, v, chains):
        chain = frozenset(chains.get(v, ()))
        neighbors = tuple((u, frozenset(chains.get(u, ())))
                          for u in self._logical_neighbours[v])
        key = (v, chain, frozenset(neighbors))

        def compute():
            if not chain:
                return np.zeros(6, dtype=np.float64)
            loads = {q: abs(self._h.get(v, 0.)) / len(chain) for q in chain}
            total_j = realized = redundancy = signed = 0.
            for u, other in neighbors:
                j = self._j.get(frozenset((v, u)), 0.)
                weight = abs(j)
                total_j += weight
                contacts = self._contacts(chain, other)
                count = len(contacts)
                if count:
                    realized += weight
                    redundancy += weight * (1. - 1. / count)
                    signed += j
                    for q, _ in contacts:
                        loads[q] += weight / count
            total_load = sum(loads.values())
            concentration = sum((load / total_load) ** 2 for load in loads.values()) if total_load else 0.
            order, parent, component, bridges, cycle = self._topology(chain)
            subtree = dict(loads)
            for q in reversed(order):
                p = parent[q]
                if p is not None:
                    subtree[p] += subtree[q]
            bottleneck = max((2. * min(subtree[q], max(0., subtree[component[q]] - subtree[q]))
                              / total_load for q in bridges), default=0.) if total_load else 0.
            return np.array([realized / total_j if total_j else 0.,
                             redundancy / total_j if total_j else 0., concentration,
                             bottleneck, cycle, signed / total_j if total_j else 0.])
        return _cached(self._physics_cache, key, compute)

    def _physics_summary(self, chains, scope):
        if not scope:
            return np.zeros(6, dtype=np.float32)
        mass = sum(self._chain_mass[v] for v in scope)
        weights = {v: self._chain_mass[v] / mass if mass else 1. / len(scope) for v in scope}
        row = sum((weights[v] * self._chain_summary(v, chains) for v in scope), start=np.zeros(6))
        return np.clip(row, -1., 1.).astype(np.float32)

    def observe(self, candidate, chains, *, state=None, ctx=None, steps_left=1, max_steps=1):
        opcode = getattr(candidate.opcode, "value", candidate.opcode)
        if opcode == "COMMIT":
            ref = getattr(candidate, "archive_ref", None)
            archive = getattr(state, "archive", None)
            if (isinstance(ref, bool) or not isinstance(ref, (int, np.integer))
                    or archive is None or not 0 <= ref < len(archive)):
                raise ValueError("COMMIT physics features require a valid archive reference")
            after = dict(archive[ref].chains)
            scope = set(self._logical_neighbours)
        elif opcode == "STOP":
            after = dict(chains)
            scope = set(self._logical_neighbours)
        else:
            after = dict(chains)
            after.update(candidate.new_chains)
            changed = {v for v in candidate.new_chains
                       if frozenset(after.get(v, ())) != frozenset(chains.get(v, ()))}
            scope = changed | {u for v in changed for u in self._logical_neighbours[v]}
        prefix = super().observe(candidate, chains, state=state, ctx=ctx,
                                 steps_left=steps_left, max_steps=max_steps)
        return np.concatenate([prefix, self._physics_summary(chains, scope),
                               self._physics_summary(after, scope)])

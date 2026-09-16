"""Features of a construction candidate, computed from the chains alone.

A prioritiser ranks (variable, qubit) pairs before the generator truncates its offer, and at
deployment it is called for every pair the generator considers, so its input has to be cheap
and local: what the qubit's neighbourhood looks like on the residual host, where the variable
stands in the logical graph, and how the two relate. Nothing here needs the full observation.
"""
import numpy as np

from space_features import residual_graph
from successor_scorer import native_coordinates, parse_host_name

OPCODES = ("PLACE", "ROUTE", "REWRITE_ONE", "COMMIT", "OTHER")
WIDTH = 5 + 4 + 5 + 5 + 5 + 6 + 2 + 3


class FeatureContext:
    """Per-instance constants and a per-state cache, so a step's candidates share work."""

    def __init__(self, task, budget=None):
        self.task = task
        self.budget = float(budget) if budget else float(task.host.number_of_nodes())
        self.host = task.host
        self.logical = task.logical
        try:
            fam, size = parse_host_name(task.name.split("-")[0])
            self.coords = native_coordinates(self.host, fam, size) or {}
        except Exception:
            # A host outside the three families, or a toy: no coordinates, which the
            # feature vector reports as zeros rather than as an invented position.
            self.coords = {}
        self.degree = {v: self.logical.degree(v) for v in self.logical.nodes()}
        self.max_degree = max(self.degree.values()) if self.degree else 1
        self.host_degree = {q: self.host.degree(q) for q in self.host.nodes()}
        # the instance's coefficients, per variable: field magnitude, total and largest
        # incident coupling, so the actor can tell a heavily coupled variable from a light one
        prob = getattr(task, "problem", None)
        h = dict(getattr(prob, "h", {}) or {}) if prob is not None else {}
        j = dict(getattr(prob, "j", {}) or {}) if prob is not None else {}
        self.h_abs = {v: abs(float(h.get(v, 0.0))) for v in self.logical.nodes()}
        self.j_sum = {v: 0.0 for v in self.logical.nodes()}
        self.j_max = {v: 0.0 for v in self.logical.nodes()}
        for (u, w), val in j.items():
            for x in (u, w):
                if x in self.j_sum:
                    self.j_sum[x] += abs(float(val))
                    self.j_max[x] = max(self.j_max[x], abs(float(val)))
        self.h_scale = max(self.h_abs.values(), default=1.0) or 1.0
        self.j_scale = max(self.j_sum.values(), default=1.0) or 1.0
        self._state_key = None
        self._allowed = None
        self._placed = None

    def state(self, chains):
        key = tuple(sorted((str(v), len(c)) for v, c in chains.items() if c))
        if key != self._state_key:
            self._state_key = key
            self._allowed = residual_graph(self.host, chains)
            self._placed = {v: set(c) for v, c in chains.items() if c}
        return self._allowed, self._placed

    def pair(self, variable, qubits, chains, opcode="PLACE"):
        """Feature vector for one candidate: variable, qubits it adds, current chains."""
        allowed, placed = self.state(chains)
        q0 = sorted(qubits, key=str)[0] if qubits else None
        f = [1.0 if opcode == o else 0.0 for o in OPCODES[:4]]
        f.append(1.0 if opcode not in OPCODES[:4] else 0.0)
        # variable
        nbrs = list(self.logical.neighbors(variable)) if variable in self.logical else []
        placed_nb = [u for u in nbrs if u in placed]
        f += [self.degree.get(variable, 0) / self.max_degree,
              len(placed_nb) / max(1, len(nbrs)),
              (len(nbrs) - len(placed_nb)) / max(1, self.max_degree),
              len(placed.get(variable, ())) / 6.0]
        if q0 is None:
            f += [0.0] * (WIDTH - 3 - len(f))
            f += [self.h_abs.get(variable, 0.0) / self.h_scale if variable is not None else 0.0,
                  self.j_sum.get(variable, 0.0) / self.j_scale if variable is not None else 0.0,
                  self.j_max.get(variable, 0.0) / max(1e-9, self.j_scale) if variable is not None else 0.0]
            return np.asarray(f[:WIDTH], dtype=np.float32)
        # qubit on the residual host, by neighbour counts. The first version used BFS free
        # volumes to radius two and the free component; called for hundreds of pairs a step
        # it made an episode on 144 qubits take forty seconds, and the counts carry the same
        # local information the prioritiser has shown it needs.
        reduced = allowed - set(qubits)
        nb1 = [r for r in self.host.neighbors(q0) if r in reduced]
        second = set()
        dead = 0
        for r in nb1:
            further = [t for t in self.host.neighbors(r) if t in reduced and t != q0]
            second.update(further)
            if not further:
                dead += 1
        f += [len(nb1) / 16.0, len(second) / 64.0, 0.0, dead / 16.0,
              self.host_degree.get(q0, 0) / 16.0]
        # pair: contact with placed neighbours' chains, and with any placed chain
        touches = 0
        others = 0
        for r in self.host.neighbors(q0):
            for u, chain in placed.items():
                if r in chain:
                    if u in placed_nb or u == variable:
                        touches += 1
                    else:
                        others += 1
        f += [touches / max(1, len(placed_nb)), min(touches, 8) / 8.0, min(others, 8) / 8.0,
              len(qubits) / 4.0, 1.0 if variable in placed else 0.0]
        # coordinates of the qubit and its offset from placed neighbours' chains
        c0 = self.coords[q0] if q0 in self.coords else [0.0] * 5
        c0 = list(c0)[:5] + [0.0] * (5 - len(list(c0)[:5]))
        if placed_nb:
            pts = [self.coords[q] for u in placed_nb for q in placed[u] if q in self.coords]
            cen = np.mean([list(p)[:5] + [0.0] * (5 - len(list(p)[:5])) for p in pts], axis=0) if pts else np.zeros(5)
            off = float(np.abs(np.asarray(c0[:4]) - np.asarray(cen[:4])).sum())
        else:
            off = 0.0
        f += c0[:5] + [off / 4.0]
        # budget: what this candidate spends against what is left
        used = sum(len(c) for c in placed.values())
        n_vars = max(1, self.logical.number_of_nodes())
        f += [(used + len(qubits)) / self.budget, (self.budget - used - len(qubits)) / n_vars]
        # coefficients of the variable
        f += [self.h_abs.get(variable, 0.0) / self.h_scale, self.j_sum.get(variable, 0.0) / self.j_scale,
              self.j_max.get(variable, 0.0) / max(1e-9, self.j_scale)]
        f = f[:WIDTH] + [0.0] * (WIDTH - len(f))
        return np.asarray(f, dtype=np.float32)

    def candidate(self, cand_tuple, chains):
        opcode, affected, added = cand_tuple
        variable = affected[0] if affected else None
        if opcode == "ROUTE" and len(affected) == 2:
            a = self.pair(affected[0], [q for q in added if q not in chains.get(affected[1], ())], chains, opcode)
            b = self.pair(affected[1], [q for q in added if q not in chains.get(affected[0], ())], chains, opcode)
            return (a + b) / 2.0
        return self.pair(variable, list(added), chains, opcode)


FRONTIER_WIDTH = 4


def frontier_features(host, logical, chains, variable, qubit, radius=3, cap=512):
    """What a candidate qubit opens or closes: the free space reachable from it within two
    and three steps on the residual host, how many other chains border that space (the
    corridors it competes for), and how many unmet logical demands of the variable it would
    serve. These are the quantities a policy needs to avoid taking the only corridor of
    another chain while gaining one contact for its own."""
    occupied = {q for c in chains.values() for q in c}
    owner = {q: v for v, c in chains.items() for q in c}
    seen = {qubit}
    frontier = [qubit]
    reach2 = 0
    competitors = set()
    depth = 0
    while frontier and depth < radius:
        depth += 1
        nxt = []
        for q in frontier:
            for r in host.neighbors(q):
                if r in seen:
                    continue
                if r in occupied:
                    if owner[r] != variable:
                        competitors.add(owner[r])
                    continue
                seen.add(r)
                nxt.append(r)
                if len(seen) > cap:
                    break
        frontier = nxt
        if depth == 2:
            reach2 = len(seen) - 1
    reach3 = len(seen) - 1
    served = 0
    for u in logical.neighbors(variable):
        cu = chains.get(u, ())
        if cu and not any(host.has_edge(a, b) for a in chains.get(variable, ()) for b in cu):
            if any(host.has_edge(qubit, b) for b in cu):
                served += 1
    return [min(reach2, 64) / 64.0, min(reach3, 256) / 256.0, min(len(competitors), 8) / 8.0,
            min(served, 4) / 4.0]

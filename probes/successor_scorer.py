"""Score an embedding from the program it compiles to, not from the state it was reached through.

The head the specification carries reads the environment's observation: the current state, the
factors that changed, a pooled view of the candidate. On labels from a fixed candidate pool it
fits perfectly, reaching the oracle with zero regret and a rank correlation of 0.71 on the states
it was fitted on, and transfers nothing to new lineages, -0.029 against a random pick where the
ceiling is +0.112. Capacity is not the problem and neither is the loss or the teacher; what it
learns is a map from those particular states to their answers.

A fifth external audit proposed the change this file implements. Success probability is a
property of one object: the physical program a chosen embedding compiles to at the strength it
will be run at. It does not depend on how that embedding was reached, on what else was in the
candidate pool, or on how much search budget is left. So the scorer is given that object and
nothing else:

    qubit messages over the physical graph
      -> pooling into the chain each qubit belongs to
      -> messages between chains along realised logical couplings
      -> one number

Node inputs are the programmed field on the qubit, its degree inside its own chain and outside
it, and the size of its chain. Edge inputs are the programmed coupling, its sign and magnitude,
and whether it is a chain edge holding one variable together or a contact carrying a logical
coupling. Chain inputs add the logical field and the number of contacts the variable realises.

Nothing here can see the state, the history or the alternatives, which is the point: two
different searches that arrive at the same program must get the same score.
"""
from __future__ import annotations

COORD_WIDTH = 5
"""Widest native tuple in the registered families, Zephyr's five; shorter ones are padded."""

import torch
import torch.nn as nn



def parse_host_name(name):
    """Split a corpus host name such as "chimera4" into its family and size."""
    letters = "".join(c for c in name if c.isalpha())
    digits = "".join(c for c in name if c.isdigit())
    return (letters, int(digits)) if letters and digits else (None, None)


def native_coordinates(host, family_hint=None, size_hint=None):
    """Per-qubit topology coordinates, from the generator's own converter.

    The meeting design asks for the hardware's own indices rather than an arbitrary integer
    label: Chimera (i, j, u, k), Pegasus (u, w, k, z), Zephyr (u, w, k, j, z). They are topology
    indices, not a Euclidean position, so they are handed over as normalised offsets and an
    orientation flag and nothing is inferred from a relabelled id. A host whose family is not one
    of the three gets zeros and a flag saying so, rather than invented coordinates.
    """
    # A corpus stores the host as a plain graph with string labels and no generator attributes,
    # so the family and size are taken from the instance's declared host name when the graph
    # itself does not carry them. The labels are the generator's own linear indices as text, so
    # the converter still applies; nothing is inferred from an arbitrary relabelling.
    family = host.graph.get("family") or family_hint
    size = host.graph.get("rows") or size_hint
    if family is None or size is None:
        return None
    try:
        import dwave_networkx as dnx
    except Exception:
        return None

    def linear(q):
        try:
            return int(q)
        except (TypeError, ValueError):
            return None

    if family == "chimera":
        rows = cols = int(size)
        tile = host.graph.get("tile", 4)
        conv = dnx.chimera_coordinates(rows, cols, tile)
        out = {}
        for q in host.nodes():
            n = linear(q)
            if n is None:
                return None
            i, j, u, k = conv.linear_to_chimera(n)
            out[q] = [i / max(1, rows - 1), j / max(1, cols - 1), float(u),
                      k / max(1, tile - 1)]
        return out
    if family == "pegasus":
        m = int(size)
        conv = dnx.pegasus_coordinates(m)
        out = {}
        for q in host.nodes():
            n = linear(q)
            if n is None:
                return None
            u, w, k, z = conv.linear_to_pegasus(n)
            out[q] = [float(u), w / max(1, m), k / 11.0, z / max(1, m)]
        return out
    if family == "zephyr":
        m = int(size)
        t = host.graph.get("tile", 4)
        conv = dnx.zephyr_coordinates(m, t)
        out = {}
        for q in host.nodes():
            n = linear(q)
            if n is None:
                return None
            u, w, k, j, z = conv.linear_to_zephyr(n)
            out[q] = [float(u), w / max(1, 2 * m), k / max(1, t - 1), float(j),
                      z / max(1, m)]
        return out
    return None


def program_graph(program, chains, problem, device=None, coords=None):
    """Turn one compiled program into the tensors the scorer reads.

    Qubit labels are sorted so the encoding of a program does not depend on dictionary order,
    for the same reason the compiler sorts them: a frozenset of labels iterates in per-process
    hash order and the result would differ between runs.
    """
    owner = {q: v for v, chain in chains.items() for q in chain}
    qubits = sorted(program.h_phys, key=str)
    index = {q: i for i, q in enumerate(qubits)}
    nodes_per_chain = {v: len(c) for v, c in chains.items()}

    inside = {q: 0 for q in qubits}
    outside = {q: 0 for q in qubits}
    edges, feats = [], []
    for (u, w), j in program.j_phys.items():
        if u not in index or w not in index:
            continue
        same = owner.get(u) is not None and owner.get(u) == owner.get(w)
        if same:
            inside[u] += 1
            inside[w] += 1
        else:
            outside[u] += 1
            outside[w] += 1
        for a, b in ((u, w), (w, u)):
            edges.append((index[a], index[b]))
            feats.append([float(j), abs(float(j)), 1.0 if j < 0 else 0.0,
                          1.0 if same else 0.0])

    # Four structural channels, then the hardware's own coordinates when the family provides
    # them. A missing-coordinate flag rides along so a host without a native mapping is a
    # declared absence rather than a row of plausible-looking zeros.
    node_feat = []
    for q in qubits:
        row = [float(program.h_phys[q]), float(inside[q]), float(outside[q]),
               float(nodes_per_chain.get(owner.get(q), 1))]
        if coords is not None:
            c = coords.get(q)
            if c is None:
                row += [0.0] * COORD_WIDTH + [0.0]
            else:
                row += list(c) + [0.0] * (COORD_WIDTH - len(c)) + [1.0]
        node_feat.append(row)

    logical = sorted(chains, key=str)
    lindex = {v: i for i, v in enumerate(logical)}
    membership = [lindex[owner[q]] for q in qubits]
    contacts = {v: 0 for v in logical}
    for (x, y), n in program.contact_counts.items():
        if x in contacts:
            contacts[x] += int(n)
        if y in contacts:
            contacts[y] += int(n)
    chain_feat = [[float(problem.h.get(v, 0.0)), float(len(chains[v])),
                   float(contacts.get(v, 0)), float(len(program.chain_edges.get(v, ())))]
                  for v in logical]

    lo_edges, lo_feats = [], []
    for (x, y), coupling in problem.j.items():
        if x not in lindex or y not in lindex:
            continue
        n = program.contact_counts.get((x, y), program.contact_counts.get((y, x), 0))
        for a, b in ((x, y), (y, x)):
            lo_edges.append((lindex[a], lindex[b]))
            lo_feats.append([float(coupling), abs(float(coupling)), float(n),
                             1.0 if n <= 1 else 0.0])

    t = lambda v, shape: (torch.tensor(v, dtype=torch.float32, device=device) if v
                          else torch.zeros(shape, dtype=torch.float32, device=device))
    return {
        "node": t(node_feat, (0, 4 + (COORD_WIDTH + 1 if coords is not None else 0))),
        "edge_index": (torch.tensor(edges, dtype=torch.long, device=device).t()
                       if edges else torch.zeros((2, 0), dtype=torch.long, device=device)),
        "edge": t(feats, (0, 4)),
        "membership": torch.tensor(membership, dtype=torch.long, device=device) if membership
        else torch.zeros((0,), dtype=torch.long, device=device),
        "chain": t(chain_feat, (0, 4)),
        "chain_edge_index": (torch.tensor(lo_edges, dtype=torch.long, device=device).t()
                             if lo_edges else torch.zeros((2, 0), dtype=torch.long,
                                                          device=device)),
        "chain_edge": t(lo_feats, (0, 4)),
        "n_chains": len(logical),
        "scale": float(program.scale),
        "strength": float(program.strength),
    }


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    return nn.Sequential(*layers[:-1])


class SuccessorScorer(nn.Module):
    """Qubits, then chains, then one number. Small on purpose: the question is whether this
    representation transfers, and a large model would confound that with capacity."""

    def __init__(self, width: int = 64, qubit_rounds: int = 2, chain_rounds: int = 2,
                 node_dim: int = 4):
        super().__init__()
        self.qubit_in = _mlp([node_dim, width, width])
        self.qubit_msg = nn.ModuleList(_mlp([2 * width + 4, width, width])
                                       for _ in range(qubit_rounds))
        self.qubit_upd = nn.ModuleList(_mlp([2 * width, width, width])
                                       for _ in range(qubit_rounds))
        self.chain_in = _mlp([width + 4, width, width])
        self.chain_msg = nn.ModuleList(_mlp([2 * width + 4, width, width])
                                       for _ in range(chain_rounds))
        self.chain_upd = nn.ModuleList(_mlp([2 * width, width, width])
                                       for _ in range(chain_rounds))
        self.readout = _mlp([2 * width + 2, width, width // 2, 1])

    def forward(self, g):
        x = self.qubit_in(g["node"])
        ei = g["edge_index"]
        for msg, upd in zip(self.qubit_msg, self.qubit_upd):
            if ei.numel():
                m = msg(torch.cat([x[ei[0]], x[ei[1]], g["edge"]], dim=-1))
                agg = torch.zeros_like(x).index_add_(0, ei[1], m)
            else:
                agg = torch.zeros_like(x)
            x = x + upd(torch.cat([x, agg], dim=-1))

        n_chains = max(1, g["n_chains"])
        pooled = torch.zeros((n_chains, x.shape[-1]), dtype=x.dtype, device=x.device)
        if g["membership"].numel():
            pooled = pooled.index_add_(0, g["membership"], x)
        c = self.chain_in(torch.cat([pooled, g["chain"]], dim=-1))
        ce = g["chain_edge_index"]
        for msg, upd in zip(self.chain_msg, self.chain_upd):
            if ce.numel():
                m = msg(torch.cat([c[ce[0]], c[ce[1]], g["chain_edge"]], dim=-1))
                agg = torch.zeros_like(c).index_add_(0, ce[1], m)
            else:
                agg = torch.zeros_like(c)
            c = c + upd(torch.cat([c, agg], dim=-1))

        summary = torch.cat([c.mean(dim=0), c.amax(dim=0),
                             torch.tensor([g["scale"], g["strength"]], dtype=c.dtype,
                                          device=c.device)], dim=-1)
        return self.readout(summary).squeeze(-1)

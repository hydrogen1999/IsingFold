"""First decision model on certified decision samples: f_theta(S_t, q) trained pairwise.

Input is one record of `dataset.DecisionRecord`: the window graph, the cores of the in-play
variables, the frozen context, the focus variable, and the candidate qubits. Every window
node gets structural features computed from the record alone (no minorminer, no labels);
a small message-passing network over the window adjacency produces node states; a candidate
qubit's score reads its state together with a pooled window state and global features.

Training signal: for every pair of candidate actions whose certified values differ, the
better one must score higher (logistic pairwise loss). Evaluation: top-1 accuracy, meaning
the arg-max candidate is one of the certified best actions, plus pairwise accuracy.

Torch is imported lazily so the feature code stays importable without it.
"""
from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np

NODE_FEATS = 14
GLOBAL_FEATS = 6
MAX_HOP = 6.0


def adj_full(rec: dict, q: int) -> list[int]:
    """Neighbours of a window qubit in the full host, reconstructed from the frozen chains
    the record carries: window edges plus adjacency to frozen qubits is what the encoder
    needs; records store frozen chains, so we test host adjacency through the stored
    `frozen_adjacency` if present, else through window edges only."""
    fa = rec.get("frozen_adjacency")
    if fa is not None:
        return list(fa.get(str(q), fa.get(q, [])))
    return []


def _bfs(adj: dict[int, list[int]], sources: set[int], passable: set[int]) -> dict[int, int]:
    """Hop distance from `sources` moving only through `passable` nodes (sources included)."""
    dist = {s: 0 for s in sources}
    dq = deque(sources)
    while dq:
        x = dq.popleft()
        for n in adj[x]:
            if n not in dist and (n in passable):
                dist[n] = dist[x] + 1
                dq.append(n)
    return dist


@dataclass
class Encoded:
    nodes: list[int]
    x: np.ndarray            # [n, NODE_FEATS]
    adj: np.ndarray          # [n, n] normalised adjacency
    g: np.ndarray            # [GLOBAL_FEATS]
    cand_idx: np.ndarray     # [a] indices into nodes
    values: list[tuple[int, int, int]]
    best_mask: np.ndarray    # [a] 1 if the action attains the max value
    greedy_idx: int
    instance_id: str
    topology: str
    resource_idx: int = -1
    difficulty: str = "base"
    nontrivial: bool = True
    """False when only one action has a feasible completion (second-best value is zero)."""


def encode(rec: dict) -> Encoded:
    nodes = list(rec["window_nodes"])
    idx = {q: i for i, q in enumerate(nodes)}
    adj = {q: [] for q in nodes}
    for a, b in rec["window_edges"]:
        adj[a].append(b); adj[b].append(a)
    focus = rec["focus"]
    cores = {int(v): set(c) for v, c in rec["cores"].items()}
    logical = {int(v): set() for v in rec["in_play"]}
    for a, b in rec["logical_edges"]:
        logical[a].add(b); logical[b].add(a)
    focus_core = cores.get(focus, set())
    nb_vars = logical[focus]
    nb_cores = {v: cores.get(v, set()) for v in nb_vars if cores.get(v)}
    other_cores = {v: cores.get(v, set()) for v in rec["in_play"] if v != focus and v not in nb_vars and cores.get(v)}
    occupied = set().union(*cores.values()) if cores else set()
    free = set(nodes) - occupied
    maxdeg = max(1, max(len(adj[q]) for q in nodes))
    # distances
    d_focus = _bfs(adj, focus_core, free | focus_core) if focus_core else {}
    d_nb = {}
    for v, c in nb_cores.items():
        d_nb[v] = _bfs(adj, c, free | c)
    touched = {v for v, c in nb_cores.items() if any(n in c for q in focus_core for n in adj[q])}
    # frozen requirements of the focus: window qubits adjacent to each frozen neighbour's chain
    frozen = {int(u): set(c) for u, c in rec.get("frozen", {}).items()}
    req_sets = []
    for v, u in rec.get("frozen_edges", []):
        if v == focus and u in frozen:
            hit = {q for q in nodes if any(t in frozen[u] for t in adj_full(rec, q))}
            req_sets.append(hit)
    unmet = [h for h in req_sets if not (h & focus_core)]
    d_req = _bfs(adj, set().union(*unmet), free | set().union(*unmet)) if unmet else {}
    fg = nx.Graph(); fg.add_nodes_from(free)
    fg.add_edges_from((a, b) for a, b in rec["window_edges"] if a in free and b in free)
    art = set(nx.articulation_points(fg)) if fg.number_of_nodes() else set()
    x = np.zeros((len(nodes), NODE_FEATS), dtype=np.float32)
    for i, q in enumerate(nodes):
        nbs = adj[q]
        x[i, 0] = q in focus_core
        x[i, 1] = any(q in c for c in nb_cores.values())
        x[i, 2] = any(q in c for c in other_cores.values())
        x[i, 3] = q in free
        x[i, 4] = len(nbs) / maxdeg
        x[i, 5] = sum(1 for n in nbs if n in free) / max(1, len(nbs))
        x[i, 6] = min(d_focus.get(q, MAX_HOP), MAX_HOP) / MAX_HOP if focus_core else 1.0
        dn = [min(d.get(q, MAX_HOP), MAX_HOP) for v, d in d_nb.items() if v not in touched]
        x[i, 7] = (min(dn) / MAX_HOP) if dn else 1.0
        x[i, 8] = sum(1 for v, c in nb_cores.items() if v not in touched and any(n in c for n in nbs)) / max(1, len(nb_vars))
        x[i, 9] = q in art
        x[i, 10] = any(n in focus_core for n in nbs)
        x[i, 11] = sum(1 for n in nbs if n in occupied) / max(1, len(nbs))
        x[i, 12] = sum(1 for h in unmet if q in h) / max(1, len(req_sets)) if req_sets else 0.0
        x[i, 13] = (min(d_req.get(q, MAX_HOP), MAX_HOP) / MAX_HOP) if unmet else 0.0
    A = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for a, b in rec["window_edges"]:
        A[idx[a], idx[b]] = 1.0; A[idx[b], idx[a]] = 1.0
    deg = A.sum(1, keepdims=True) + 1.0
    A_norm = (A + np.eye(len(nodes), dtype=np.float32)) / deg
    q_cap = -rec["witness_outcome"][1]
    used = sum(len(c) for c in cores.values())
    g = np.array([
        len(rec["in_play"]) / 4.0,
        len(rec["logical_edges"]) / 6.0,
        len(focus_core) / rec["l_cap"],
        len([v for v in nb_vars if v not in touched]) / max(1, len(nb_vars)) if nb_vars else 0.0,
        (q_cap - used) / (rec["l_cap"] * len(rec["in_play"])),
        rec["l_cap"] / 4.0,
    ], dtype=np.float32)
    cand = np.array([idx[q] for _, q in rec["actions"]], dtype=np.int64)
    vals = [tuple(v) for v in rec["values"]]
    best = max(vals)
    best_mask = np.array([1.0 if v == best else 0.0 for v in vals], dtype=np.float32)
    greedy_idx = rec["actions"].index(rec["greedy_action"])
    resource_idx = rec["actions"].index(rec["resource_action"]) if "resource_action" in rec else -1
    second = sorted((v[0] for v in vals), reverse=True)[1] if len(vals) > 1 else 0
    return Encoded(nodes, x, A_norm, g, cand, vals, best_mask, greedy_idx, rec["instance_id"], rec["topology"], resource_idx,
                   rec.get("difficulty", "base"), bool(second > 0))


def load_records(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as fh:
            out += [json.loads(l) for l in fh]
    return out


def split_by_instance(encs: list[Encoded], test_frac: float, seed: int):
    ids = sorted({e.instance_id for e in encs})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_frac))
    test = set(ids[:n_test])
    return [e for e in encs if e.instance_id not in test], [e for e in encs if e.instance_id in test]


# ---------------------------------------------------------------- torch part

def build_model(hidden: int = 64, layers: int = 3):
    import torch
    import torch.nn as nn

    class MPNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(NODE_FEATS, hidden), nn.ReLU())
            self.msg = nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.read = nn.Sequential(nn.Linear(2 * hidden + GLOBAL_FEATS, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x, adj, g, cand):
            h = self.inp(x)
            for m, ln in zip(self.msg, self.norm):
                agg = adj @ h
                h = ln(h + m(torch.cat([h, agg], -1)))
            pooled = h.mean(0, keepdim=True).expand(len(cand), -1)
            hc = h[cand]
            return self.read(torch.cat([hc, pooled, g.expand(len(cand), -1)], -1)).squeeze(-1)

    return MPNN()


def build_linear():
    """Same candidate features, a linear scorer: the 'is the network needed' control."""
    import torch.nn as nn

    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(NODE_FEATS + GLOBAL_FEATS, 1)

        def forward(self, x, adj, g, cand):
            import torch
            return self.w(torch.cat([x[cand], g.expand(len(cand), -1)], -1)).squeeze(-1)

    return Lin()


def _better(vi, vj, tol: float) -> bool:
    """vi beats vj: for integer (structural) labels any lexicographic difference; for float
    first components (p_solve) the first component must exceed by at least `tol`."""
    if tol > 0:
        return vi[0] - vj[0] >= tol
    return vi > vj


def pair_loss(scores, values, tol: float = 0.0):
    """Logistic loss over all ordered pairs with a certified difference."""
    import torch
    vals = [tuple(v) for v in values]
    li, lj = [], []
    for i in range(len(vals)):
        for j in range(len(vals)):
            if _better(vals[i], vals[j], tol):
                li.append(i); lj.append(j)
    if not li:
        return None
    d = scores[li] - scores[lj]
    return torch.nn.functional.softplus(-d).mean()


def evaluate(model, encs, device="cpu", tol: float = 0.0) -> dict:
    import torch
    model.eval()
    top1 = pair_ok = pair_n = 0
    greedy = rnd = res = 0.0
    with torch.no_grad():
        for e in encs:
            s = model(torch.from_numpy(e.x).to(device), torch.from_numpy(e.adj).to(device),
                      torch.from_numpy(e.g).to(device), torch.from_numpy(e.cand_idx).to(device)).cpu().numpy()
            k = int(np.argmax(s))
            top1 += e.best_mask[k]
            greedy += e.best_mask[e.greedy_idx]
            res += e.best_mask[e.resource_idx] if e.resource_idx >= 0 else 0.0
            rnd += e.best_mask.mean()
            for i in range(len(e.values)):
                for j in range(len(e.values)):
                    if _better(e.values[i], e.values[j], tol):
                        pair_n += 1
                        pair_ok += s[i] > s[j]
    n = max(1, len(encs))
    return {"n": len(encs), "top1": float(top1 / n), "pair_acc": float(pair_ok / max(1, pair_n)),
            "greedy_top1": float(greedy / n), "random_top1": float(rnd / n), "resource_top1": float(res / n)}


def train(model, train_encs, val_encs, *, epochs=20, lr=1e-3, seed=0, device="cpu", log=print, tol: float = 0.0):
    import torch
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = None
    for ep in range(epochs):
        model.train()
        order = list(range(len(train_encs)))
        rng.shuffle(order)
        tot = cnt = 0
        for b0 in range(0, len(order), 32):
            opt.zero_grad()
            loss = 0.0
            m = 0
            for k in order[b0:b0 + 32]:
                e = train_encs[k]
                s = model(torch.from_numpy(e.x).to(device), torch.from_numpy(e.adj).to(device),
                          torch.from_numpy(e.g).to(device), torch.from_numpy(e.cand_idx).to(device))
                l = pair_loss(s, e.values, tol)
                if l is not None:
                    loss = loss + l; m += 1
            if m:
                (loss / m).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach()); cnt += m
        ev = evaluate(model, val_encs, device, tol)
        log(f"epoch {ep+1:3d} loss {tot/max(1,cnt):.4f} val top1 {ev['top1']:.3f} pair {ev['pair_acc']:.3f} (greedy {ev['greedy_top1']:.3f} random {ev['random_top1']:.3f})")
        if best is None or ev["top1"] > best[0]:
            best = (ev["top1"], ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return best[1]


# ---------------------------------------------------------------- deployment helpers

def save_model(model, path, meta: dict | None = None):
    import torch
    torch.save({"state": model.state_dict(), "meta": meta or {}}, path)


def load_model(path, hidden: int = 64, layers: int = 3, heads: int = 4):
    import torch
    ck = torch.load(path, map_location="cpu")
    meta = ck.get("meta", {})
    m = build_arch(
        meta.get("arch", "mpnn"),
        meta.get("hidden", hidden),
        meta.get("layers", layers),
        meta.get("heads", heads),
    )
    m.load_state_dict(ck["state"])
    m.eval()
    return m


class Scorer:
    """Score the candidate actions of a record-shaped state dict with a trained model."""

    def __init__(self, model):
        self.model = model

    def __call__(self, rec: dict) -> list[float]:
        import torch
        e = encode(rec)
        with torch.no_grad():
            s = self.model(torch.from_numpy(e.x), torch.from_numpy(e.adj), torch.from_numpy(e.g),
                           torch.from_numpy(e.cand_idx))
        return [float(v) for v in s]


# ---------------------------------------------------------------- architecture variants (2026-09-09)

def hop_matrix(adj_norm: np.ndarray, max_hop: int = 4) -> np.ndarray:
    """Integer hop distances (capped) from the normalised adjacency; used as an attention bias."""
    n = adj_norm.shape[0]
    A = (adj_norm > 0).astype(np.int32)
    np.fill_diagonal(A, 0)
    dist = np.full((n, n), max_hop + 1, dtype=np.int64)
    np.fill_diagonal(dist, 0)
    reach = np.eye(n, dtype=np.int32)
    cur = np.eye(n, dtype=np.int32)
    for h in range(1, max_hop + 1):
        cur = ((cur @ A) > 0).astype(np.int32)
        new = (cur > 0) & (reach == 0)
        dist[new] = h
        reach = reach | cur
    return dist


def build_arch(arch: str = "mpnn", hidden: int = 64, layers: int = 3, heads: int = 4):
    """mpnn: the residual mean-aggregation network of L-78. gin: sum aggregation with an MLP
    update and a learnable epsilon (Xu et al. 2019). gatv2: dynamic attention over the
    adjacency (Brody et al. 2022), multi-head. gps: GraphGPS-style layers, a local GIN
    update plus global multi-head attention with a learned bias per hop distance (up to 4),
    which lets a candidate qubit attend to frozen-neighbour contacts anywhere in the window.
    All take the same node features, global features and candidate indices."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if arch == "mpnn":
        return build_model(hidden, layers)

    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.read = nn.Sequential(nn.Linear(2 * hidden + GLOBAL_FEATS, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, h, g, cand):
            pooled = h.mean(0, keepdim=True).expand(len(cand), -1)
            return self.read(torch.cat([h[cand], pooled, g.expand(len(cand), -1)], -1)).squeeze(-1)

    class GIN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(NODE_FEATS, hidden)
            self.eps = nn.Parameter(torch.zeros(layers))
            self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.out = Readout()

        def forward(self, x, adj, g, cand):
            A = (adj > 0).float(); A = A - torch.diag(torch.diag(A))  # raw adjacency, sum aggregation
            h = F.relu(self.inp(x))
            for k in range(layers):
                h = self.norm[k](h + self.mlps[k]((1 + self.eps[k]) * h + A @ h))
            return self.out(h, g, cand)

    class GATv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(NODE_FEATS, hidden)
            self.W = nn.ModuleList([nn.Linear(2 * hidden, hidden * heads) for _ in range(layers)])
            self.a = nn.ModuleList([nn.Linear(hidden, 1, bias=False) for _ in range(layers)])
            self.proj = nn.ModuleList([nn.Linear(hidden * heads, hidden) for _ in range(layers)])
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.out = Readout()

        def forward(self, x, adj, g, cand):
            n = x.shape[0]
            A = (adj > 0).float()  # includes self loops
            h = F.relu(self.inp(x))
            for k in range(layers):
                hi = h.unsqueeze(1).expand(n, n, hidden); hj = h.unsqueeze(0).expand(n, n, hidden)
                z = F.leaky_relu(self.W[k](torch.cat([hi, hj], -1)), 0.2).view(n, n, heads, hidden)
                e = self.a[k](z).squeeze(-1)  # n, n, heads
                e = e.masked_fill(A.unsqueeze(-1) == 0, float("-inf"))
                att = torch.softmax(e, dim=1)  # over neighbours j
                msg = torch.einsum("ijh,ijhd->ihd", att, z).reshape(n, heads * hidden)
                h = self.norm[k](h + self.proj[k](msg))
            return self.out(h, g, cand)

    class GPS(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(NODE_FEATS, hidden)
            self.local = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
            self.attn = nn.ModuleList([nn.MultiheadAttention(hidden, heads, batch_first=True) for _ in range(layers)])
            self.bias = nn.Parameter(torch.zeros(layers, heads, 6))  # hop 0..4 and "farther"
            self.ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden)) for _ in range(layers)])
            self.n1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.n2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.n3 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.out = Readout()

        def forward(self, x, adj, g, cand):
            n = x.shape[0]
            A = (adj > 0).float(); A = A - torch.diag(torch.diag(A))
            hops = torch.from_numpy(hop_matrix(adj.detach().cpu().numpy())).to(x.device)  # n, n in 0..5
            h = F.relu(self.inp(x))
            for k in range(layers):
                h = self.n1[k](h + self.local[k](h + A @ h))
                bias = self.bias[k][:, hops]  # heads, n, n
                a, _ = self.attn[k](h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), attn_mask=(-bias).reshape(heads, n, n), need_weights=False)
                h = self.n2[k](h + a.squeeze(0))
                h = self.n3[k](h + self.ff[k](h))
            return self.out(h, g, cand)

    return {"gin": GIN, "gatv2": GATv2, "gps": GPS}[arch]()

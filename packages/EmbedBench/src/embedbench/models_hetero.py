"""Candidate-conditioned heterogeneous scorer on the programmed Ising (design note
docs/design_hetero.md).

Graph per candidate: qubit nodes (window qubits and the qubits of the frozen chains adjacent
to the window) and, for Quality V2, every variable and logical coupling in the full problem.
The legacy V1 path retains its local variable graph. Edge types are dense tensors: hardware
couplers (window edges, window-to-frozen adjacency, and local frozen-chain paths), signed
logical couplings, and membership. Programmed coefficients follow the
programmer: h/|chain| on each qubit of a chain, J/#contacts on each contact coupler, chain
couplers flagged. Message passing alternates qubit-qubit (edge-gated), qubit-to-variable
(mean and max over the chain), variable-to-qubit (broadcast), variable-variable (J-gated).
Readout: the focus variable's state, the candidate's pooled qubit states, and a
cross-candidate attention over the candidate set; a scalar per candidate.

Loss: listwise softmax cross-entropy with soft targets exp(p/T) over the candidate set and a
per-candidate reliability weight (1 for second-stage estimates, `w1` for screening ones);
pairwise logistic available as an ablation.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field

import numpy as np

from embedbench.hamiltonian_context import (
    HAMILTONIAN_CONTEXT_DIMENSION,
    encode_hamiltonian_context,
)

QF = 10  # qubit features (last: in the current/original chain of the focus)
VF = 7   # variable features
EF = 5   # coupler edge features: exists, chain coupler, contact, J_phys (signed, per contact), assumed (frozen-chain path, not a recorded coupler)


@dataclass
class HeteroEncoded:
    xq: np.ndarray          # [nq, QF] shared
    xv: np.ndarray          # [nv, VF]
    Eqq: np.ndarray         # [nq, nq, EF] shared part (frozen chain couplers, non-candidate contacts)
    Avv: np.ndarray         # [nv, nv] signed logical coupling transform
    Mvq: np.ndarray         # [nv, nq] membership for frozen variables (focus row empty)
    cand_masks: np.ndarray  # [m, nq] candidate qubits
    cand_contacts: list     # per candidate: list of (qi, qj, J_phys) contact couplers to neighbour chains
    focus_index: int
    focus_h: float
    p: list
    stage: list
    best_index: int
    resource_index: int
    original_index: int
    source: str
    instance_id: str
    topology: str
    hamiltonian_context: np.ndarray = field(
        default_factory=lambda: np.zeros(HAMILTONIAN_CONTEXT_DIMENSION, dtype=np.float32)
    )


@dataclass(frozen=True)
class HeteroModelInput:
    """Deployable heterogeneous graph input with no targets or policy indices."""

    qubit_features: np.ndarray
    variable_features: np.ndarray
    coupler_features: np.ndarray
    logical_adjacency: np.ndarray
    membership: np.ndarray
    candidate_masks: np.ndarray
    candidate_contacts: tuple[tuple[tuple[int, int, float], ...], ...]
    focus_index: int
    focus_h: float
    hamiltonian_context: np.ndarray


def encode_hetero_input(
    rec: dict,
    *,
    _legacy_original_index: int | None = None,
    require_hamiltonian_context: bool = False,
) -> HeteroModelInput:
    """Encode only information available when a candidate set is scored.

    The private legacy index is used exclusively by ``encode_hetero`` to preserve the
    original V1 scorer's current-chain feature. Quality V2 never supplies it, so its input
    is independent of result labels and candidate-policy indices.
    """

    if require_hamiltonian_context or "problem" in rec:
        hamiltonian_context = encode_hamiltonian_context(rec).values
    else:
        hamiltonian_context = np.zeros(
            HAMILTONIAN_CONTEXT_DIMENSION, dtype=np.float32
        )
    win = list(rec["window_nodes"])
    frozen = {int(v): sorted(c) for v, c in rec["frozen"].items()}
    full_logical_graph = require_hamiltonian_context
    focus_variable = int(rec.get("focus", 0))
    if full_logical_graph:
        raw_all_chains = rec.get("all_chains")
        if not isinstance(raw_all_chains, dict):
            raise ValueError("full heterogeneous encoding requires all_chains")
        all_chains = {int(v): sorted(c) for v, c in raw_all_chains.items()}
        if focus_variable in all_chains or not set(frozen) <= set(all_chains):
            raise ValueError("all_chains must cover local frozen chains and exclude focus")
        problem = rec["problem"]
        logical_h = {int(v): float(value) for v, value in problem["h"].items()}
        if set(logical_h) != set(all_chains) | {focus_variable}:
            raise ValueError("problem.h and all_chains disagree on logical variables")
        logical_edges: list[tuple[int, int, float]] = []
        for raw_left, raw_right, raw_coupling in problem["J"]:
            coupling = float(raw_coupling)
            if coupling != 0.0:
                logical_edges.append((int(raw_left), int(raw_right), coupling))
        full_focus_couplings = {
            right if left == focus_variable else left: coupling
            for left, right, coupling in logical_edges
            if focus_variable in (left, right)
        }
        focus_neighbours = [int(neighbour) for neighbour in rec["neighbours"]]
        if len(focus_neighbours) != len(set(focus_neighbours)):
            raise ValueError("record neighbours must be unique")
        # The planted witness graph can contain structural embedding constraints whose
        # programmed Ising coefficient cancels to zero.  Such neighbours are intentionally
        # retained by quality_chain so every candidate satisfies the witness minor, while
        # problem.J contains only the nonzero Hamiltonian.  Every programmed focus coupling
        # must therefore be present, but the structural neighbour set may be a strict
        # superset with a recorded edge_J of zero.
        if not set(full_focus_couplings) <= set(focus_neighbours):
            raise ValueError("record neighbours omit a coupling from full problem.J")
        edge_values = [float(value) for value in rec.get("edge_J", [])]
        if len(edge_values) != len(focus_neighbours) or any(
            not math.isclose(
                value,
                full_focus_couplings.get(neighbour, 0.0),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for neighbour, value in zip(focus_neighbours, edge_values, strict=True)
        ):
            raise ValueError("record edge_J disagrees with full problem.J")
        J = dict(zip(focus_neighbours, edge_values, strict=True))
    else:
        all_chains = frozen
        focus_neighbours = list(rec["neighbours"])
        J = dict(zip(focus_neighbours, rec.get("edge_J", [0.0] * len(focus_neighbours))))
        logical_h = {
            variable: value
            for variable, value in zip(
                focus_neighbours,
                rec.get("neighbour_h", [0.0] * len(focus_neighbours)),
            )
        }
        logical_h[focus_variable] = float(rec.get("focus_h", 0.0))
        logical_edges = [
            (focus_variable, neighbour, float(J[neighbour]))
            for neighbour in focus_neighbours
        ]
    fq = [q for c in frozen.values() for q in c]
    qubits = sorted(set(win) | set(fq))
    qi = {q: i for i, q in enumerate(qubits)}
    nq = len(qubits)
    variables = ["focus"] + sorted(all_chains)
    vi = {v: i for i, v in enumerate(variables)}
    nv = len(variables)
    # couplers
    E = np.zeros((nq, nq, EF), dtype=np.float32)
    def add(a, b, chain=0.0, contact=0.0, jphys=0.0, assumed=0.0):
        for u, w in ((a, b), (b, a)):
            E[u, w, 0] = 1.0; E[u, w, 1] = max(E[u, w, 1], chain); E[u, w, 2] = max(E[u, w, 2], contact); E[u, w, 3] = jphys; E[u, w, 4] = max(E[u, w, 4], assumed)
    for a, b in rec["window_edges"]:
        add(qi[a], qi[b])
    for q, ts in rec.get("frozen_adjacency", {}).items():
        for t in ts:
            if int(q) in qi and t in qi: add(qi[int(q)], qi[t])
    owner = {}
    for v, c in frozen.items():
        for k in range(len(c) - 1):
            if E[qi[c[k]], qi[c[k + 1]], 0] == 0:
                add(qi[c[k]], qi[c[k + 1]], chain=1.0, assumed=1.0)  # the record has no intra-chain couplers of frozen chains
            else:
                add(qi[c[k]], qi[c[k + 1]], chain=1.0)
        for q in c: owner[q] = v
    # membership and variable features
    M = np.zeros((nv, nq), dtype=np.float32)
    xv = np.zeros((nv, VF), dtype=np.float32)
    focus_h = float(logical_h[focus_variable])
    ndeg = dict(zip(focus_neighbours, rec.get("neighbour_degree", [1] * len(focus_neighbours))))
    logical_degree = {variable: 0 for variable in all_chains}
    logical_degree[focus_variable] = 0
    for left, right, _ in logical_edges:
        logical_degree[left] += 1
        logical_degree[right] += 1

    def signed_log(value: float) -> float:
        return math.copysign(math.log1p(abs(value)), value) if value else 0.0

    for v, c in all_chains.items():
        if v in frozen:
            for q in frozen[v]:
                M[vi[v], qi[q]] = 1.0
        if full_logical_graph:
            field_value = float(logical_h[v])
            xv[vi[v]] = [
                0.0,
                1.0 if v in J else 0.0,
                signed_log(J.get(v, 0.0)),
                len(c) / 4.0,
                logical_degree[v] / max(1, len(all_chains)),
                signed_log(field_value),
                math.log1p(abs(field_value)),
            ]
        else:
            xv[vi[v]] = [
                0.0,
                1.0 if v in J else 0.0,
                abs(J.get(v, 0.0)),
                len(c) / 4.0,
                ndeg.get(v, 1) / 6.0,
                0.0,
                0.0,
            ]
    if full_logical_graph:
        xv[vi["focus"]] = [
            1.0,
            0.0,
            0.0,
            0.0,
            logical_degree[focus_variable] / max(1, len(all_chains)),
            signed_log(focus_h),
            math.log1p(abs(focus_h)),
        ]
    else:
        xv[vi["focus"]] = [
            1.0,
            0.0,
            0.0,
            0.0,
            len(focus_neighbours) / 6.0,
            abs(focus_h),
            1.0,
        ]
    Avv = np.zeros((nv, nv), dtype=np.float32)
    for left, right, coupling in logical_edges:
        left_index = vi["focus"] if left == focus_variable else vi[left]
        right_index = vi["focus"] if right == focus_variable else vi[right]
        encoded_coupling = signed_log(coupling) if full_logical_graph else abs(coupling)
        Avv[left_index, right_index] = Avv[right_index, left_index] = encoded_coupling
    # qubit features (shared): in window, frozen-owned, degree, free-neighbour share, touches-required, h_phys of owner
    xq = np.zeros((nq, QF), dtype=np.float32)
    deg = E[:, :, 0].sum(1)
    current = (
        set(rec["candidates"][_legacy_original_index])
        if _legacy_original_index is not None and _legacy_original_index >= 0
        else set()
    )
    nbr_h = logical_h
    for q in qubits:
        i = qi[q]
        own = owner.get(q)
        nb = [w for w in range(nq) if E[i, w, 0] > 0]
        free_nb = sum(1 for w in nb if qubits[w] not in owner)
        touches = sum(1 for w in nb if owner.get(qubits[w]) in J)
        h_phys = (nbr_h.get(own, 0.0) / max(1, len(frozen.get(own, [1])))) if own is not None else 0.0  # signed h/|chain| of the owner
        xq[i] = [q in set(win), own is not None, deg[i] / 8.0, free_nb / max(1, len(nb)), touches / max(1, len(focus_neighbours)),
                 (J.get(own, 0.0) if own is not None else 0.0), h_phys, 0.0, 0.0, q in current]
    # candidates
    cands = rec["candidates"]
    masks = np.zeros((len(cands), nq), dtype=np.float32)
    contacts = []
    for k, c in enumerate(cands):
        for q in c: masks[k, qi[q]] = 1.0
        cs = set(c)
        cc = []
        for v in focus_neighbours:
            if v not in frozen: continue
            pairs = [(qi[a], qi[b]) for a in c for b in frozen[v] if E[qi[a], qi[b], 0] > 0]
            for a, b in pairs:
                cc.append((a, b, J.get(v, 0.0) / max(1, len(pairs))))
        contacts.append(cc)
    return HeteroModelInput(
        qubit_features=xq,
        variable_features=xv,
        coupler_features=E,
        logical_adjacency=Avv,
        membership=M,
        candidate_masks=masks,
        candidate_contacts=tuple(tuple(contact) for contact in contacts),
        focus_index=vi["focus"],
        focus_h=focus_h,
        hamiltonian_context=hamiltonian_context,
    )


def encode_hetero(rec: dict) -> HeteroEncoded:
    """Legacy V1 encoding, including the labels consumed outside its scorer."""

    original_index = int(rec["original_index"])
    model_input = encode_hetero_input(rec, _legacy_original_index=original_index)
    candidate_count = model_input.candidate_masks.shape[0]
    return HeteroEncoded(
        model_input.qubit_features,
        model_input.variable_features,
        model_input.coupler_features,
        model_input.logical_adjacency,
        model_input.membership,
        model_input.candidate_masks,
        [list(contacts) for contacts in model_input.candidate_contacts],
        model_input.focus_index,
        model_input.focus_h,
        list(rec["p_solve"]),
        list(rec.get("stage", [1] * candidate_count)),
        rec["best_index"],
        rec["resource_index"],
        original_index,
        rec["source"],
        rec["instance_id"],
        rec["topology"],
        model_input.hamiltonian_context,
    )


def _model_input_from_encoded(
    encoded: HeteroEncoded,
    *,
    drop_legacy_current_chain: bool,
) -> HeteroModelInput:
    qubit_features = encoded.xq
    if drop_legacy_current_chain:
        qubit_features = encoded.xq.copy()
        qubit_features[:, -1] = 0.0
    return HeteroModelInput(
        qubit_features=qubit_features,
        variable_features=encoded.xv,
        coupler_features=encoded.Eqq,
        logical_adjacency=encoded.Avv,
        membership=encoded.Mvq,
        candidate_masks=encoded.cand_masks,
        candidate_contacts=tuple(
            tuple((int(left), int(right), float(coupling)) for left, right, coupling in contacts)
            for contacts in encoded.cand_contacts
        ),
        focus_index=int(encoded.focus_index),
        focus_h=float(encoded.focus_h),
        hamiltonian_context=encoded.hamiltonian_context,
    )


def model_input_from_hetero(encoded: HeteroEncoded) -> HeteroModelInput:
    """Remove all training metadata and the legacy policy-index feature."""

    if not isinstance(encoded, HeteroEncoded):
        raise TypeError("encoded must be a HeteroEncoded training value")
    return _model_input_from_encoded(encoded, drop_legacy_current_chain=True)


def build_hetero(hidden: int = 64, layers: int = 3, heads: int = 4):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Hetero(nn.Module):
        def __init__(self):
            super().__init__()
            self.iq = nn.Linear(QF, hidden); self.iv = nn.Linear(VF, hidden)
            self.eg = nn.ModuleList([nn.Linear(EF, hidden) for _ in range(layers)])        # edge gate
            self.qq = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            self.qv = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])  # mean,max over chain
            self.vq = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            self.vv = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
            self.uq = nn.ModuleList([nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
            self.uv = nn.ModuleList([nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
            self.nq = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)]); self.nv = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
            self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.read = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward_many(self, xq, xv, E, Avv, M):
            """Message-pass a batch of candidate-conditioned graphs in parallel."""
            if xq.ndim != 3 or xv.ndim != 3 or E.ndim != 4 or Avv.ndim != 3 or M.ndim != 3:
                raise ValueError("batched heterogeneous tensors have inconsistent ranks")
            hq = F.relu(self.iq(xq)); hv = F.relu(self.iv(xv))
            mask_qq = E[:, :, :, 0]
            for k in range(len(self.qq)):
                gate = torch.sigmoid(self.eg[k](E)) * mask_qq.unsqueeze(-1)
                messages = self.qq[k](hq).unsqueeze(1)
                agg_qq = (gate * messages).sum(2) / (mask_qq.sum(2, keepdim=True) + 1.0)
                msg_vq = torch.bmm(M.transpose(1, 2), self.vq[k](hv))
                hq = self.nq[k](hq + self.uq[k](torch.cat([hq, agg_qq, msg_vq], -1)))
                cnt = M.sum(2, keepdim=True).clamp(min=1.0)
                mean_q = torch.bmm(M, hq) / cnt
                membership = M > 0
                candidates = hq.unsqueeze(1).expand(-1, M.shape[1], -1, -1)
                masked = candidates.masked_fill(~membership.unsqueeze(-1), float("-inf"))
                max_q = masked.amax(2)
                max_q = torch.where(
                    membership.any(2, keepdim=True),
                    max_q,
                    torch.zeros_like(max_q),
                )
                agg_qv = self.qv[k](torch.cat([mean_q, max_q], -1))
                agg_vv = torch.bmm(Avv, self.vv[k](hv)) / (
                    Avv.abs().sum(2, keepdim=True) + 1.0
                )
                hv = self.nv[k](hv + self.uv[k](torch.cat([hv, agg_qv, agg_vv], -1)))
            return hq, hv

        def forward_one(self, xq, xv, E, Avv, M):
            """Compatibility wrapper for one candidate-conditioned graph."""
            hq, hv = self.forward_many(
                xq.unsqueeze(0),
                xv.unsqueeze(0),
                E.unsqueeze(0),
                Avv.unsqueeze(0),
                M.unsqueeze(0),
            )
            return hq[0], hv[0]

        def candidate_representations(self, e: HeteroModelInput):
            if not isinstance(e, HeteroModelInput):
                raise TypeError("heterogeneous candidate encoder requires HeteroModelInput")
            device = self.iq.weight.device
            xq = torch.from_numpy(e.qubit_features).to(device)
            xv = torch.from_numpy(e.variable_features).to(device)
            E0 = torch.from_numpy(e.coupler_features).to(device)
            Avv = torch.from_numpy(e.logical_adjacency).to(device)
            M0 = torch.from_numpy(e.membership).to(device)
            masks = torch.from_numpy(e.candidate_masks).to(device)
            candidate_count = masks.shape[0]
            cq = masks > 0
            xq_batch = xq.unsqueeze(0).expand(candidate_count, -1, -1).clone()
            xv_batch = xv.unsqueeze(0).expand(candidate_count, -1, -1)
            E = E0.unsqueeze(0).expand(candidate_count, -1, -1, -1).clone()
            Avv_batch = Avv.unsqueeze(0).expand(candidate_count, -1, -1)
            M = M0.unsqueeze(0).expand(candidate_count, -1, -1).clone()
            M[:, e.focus_index, :] = masks
            focus_fields = masks * (
                float(e.focus_h) / masks.sum(1, keepdim=True).clamp(min=1.0)
            )
            xq_batch[:, :, 7] = focus_fields

            candidate_outer = cq.unsqueeze(2) & cq.unsqueeze(1)
            sub = E[:, :, :, 0] * candidate_outer.to(E.dtype)
            E[:, :, :, 1] = torch.maximum(E[:, :, :, 1], sub)
            contact_candidate = []
            contact_left = []
            contact_right = []
            contact_coupling = []
            for candidate_index, contacts in enumerate(e.candidate_contacts):
                for left, right, coupling in contacts:
                    contact_candidate.append(candidate_index)
                    contact_left.append(left)
                    contact_right.append(right)
                    contact_coupling.append(coupling)
            if contact_candidate:
                candidates_index = torch.as_tensor(contact_candidate, device=device)
                left_index = torch.as_tensor(contact_left, device=device)
                right_index = torch.as_tensor(contact_right, device=device)
                coupling = torch.as_tensor(contact_coupling, dtype=E.dtype, device=device)
                E[candidates_index, left_index, right_index, 2] = 1.0
                E[candidates_index, right_index, left_index, 2] = 1.0
                E[candidates_index, left_index, right_index, 3] = coupling
                E[candidates_index, right_index, left_index, 3] = coupling

            hq, hv = self.forward_many(xq_batch, xv_batch, E, Avv_batch, M)
            candidate_max = hq.masked_fill(~cq.unsqueeze(-1), float("-inf")).amax(1)
            candidate_mean = torch.bmm(masks.unsqueeze(1), hq).squeeze(1) / masks.sum(
                1, keepdim=True
            ).clamp(min=1.0)
            H = torch.cat(
                [hv[:, e.focus_index, :], candidate_mean, candidate_max], dim=-1
            ).unsqueeze(0)
            ctx, _ = self.cross(H[:, :, :hidden], H[:, :, :hidden], H[:, :, :hidden], need_weights=False)
            return torch.cat([H.squeeze(0), ctx.squeeze(0)], -1)

        def forward(self, e: HeteroEncoded):
            if not isinstance(e, HeteroEncoded):
                raise TypeError("legacy heterogeneous scorer requires HeteroEncoded")
            model_input = _model_input_from_encoded(
                e,
                drop_legacy_current_chain=False,
            )
            return self.read(self.candidate_representations(model_input)).squeeze(-1)

    return Hetero()


def build_hetero_candidate_encoder(hidden: int = 64, layers: int = 3, heads: int = 4):
    """Return the heterogeneous backbone before its scalar V1 readout."""

    import torch.nn as nn

    class HeteroCandidateEncoder(nn.Module):
        output_dim = 4 * hidden

        def __init__(self) -> None:
            super().__init__()
            self.encoder = build_hetero(hidden, layers, heads)
            self.encoder.read = nn.Identity()

        def forward(self, model_input: HeteroModelInput):
            if not isinstance(model_input, HeteroModelInput):
                raise TypeError("heterogeneous candidate encoder requires HeteroModelInput")
            return self.encoder.candidate_representations(model_input)

    return HeteroCandidateEncoder()


def listwise_loss(scores, p, stage, T: float = 0.05, w1: float = 0.3):
    import torch
    p_t = torch.tensor(p, dtype=torch.float32, device=scores.device)
    w = torch.tensor(
        [1.0 if s == 2 else w1 for s in stage],
        dtype=torch.float32,
        device=scores.device,
    )
    target = torch.softmax(p_t / T, 0) * w
    target = target / target.sum().clamp(min=1e-8)
    logp = torch.log_softmax(scores, 0)
    return -(target * logp).sum()


class HeteroScorer:
    """Deployment wrapper: record dict -> scores, same contract as models_chain.ChainScorer."""

    def __init__(self, model, *, deploy_view: bool = False):
        self.model = model
        self.deploy_view = deploy_view

    def __call__(self, rec: dict) -> list[float]:
        import torch
        from embedbench.models_chain import _prepare_scorer_record

        prepared = _prepare_scorer_record(rec, self.deploy_view)
        with torch.no_grad():
            return [float(v) for v in self.model(encode_hetero(prepared))]


def load_scorer(path):
    """Load a checkpoint saved by scripts/train_chain.py of any architecture and return a
    record-level scorer (mpnn/gin/gatv2/gps -> ChainScorer, hetero -> HeteroScorer)."""
    from collections.abc import Mapping

    import torch

    from embedbench.models_chain import (
        ChainScorer,
        checkpoint_preprocessing,
        load_chain_model,
    )

    ck = torch.load(path, map_location="cpu")
    meta = ck.get("meta", {})
    if not isinstance(meta, Mapping):
        raise ValueError("checkpoint metadata must be an object")
    arch = meta.get("arch", "mpnn")
    if arch == "hetero":
        flags = checkpoint_preprocessing(meta)
        if flags["neighbour_feats"]:
            raise ValueError("neighbour-feats preprocessing is not applicable to hetero")
        if flags["no_length_feats"]:
            raise ValueError("no-length-feats preprocessing is not applicable to hetero")
        model = build_hetero(
            meta.get("hidden", 64),
            meta.get("layers", 3),
            meta.get("heads", 4),
        )
        model.load_state_dict(ck["state"])
        model.eval()
        return HeteroScorer(model, deploy_view=flags["deploy_view"])
    return ChainScorer(load_chain_model(path))

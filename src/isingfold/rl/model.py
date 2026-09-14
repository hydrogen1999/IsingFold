"""IF-Core: a typed graph actor-critic over fully materialised rewrites.

Spec: MODEL_SPEC sections 3-4. Three dual local blocks Eq. (6), two ownership/conflict
fusion blocks Eq. (7), phase-aware chain factors Eq. (8)-(11), an ordered route encoder
Eq. (12), an archive summary Eq. (13), the action-set summary Eq. (14), a bounded action-quality
prior, a masked categorical actor Eq. (15), and two state critics Eq. (16). Pooling is segmented
per observation: a candidate in one observation can never enter another observation's denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from isingfold.rl.contracts import OPCODES
from isingfold.rl.tensorize import (
    N_ACTION,
    N_ARCHIVE,
    N_CLAIM,
    N_CONFLICT,
    N_FACTOR,
    N_GLOBAL,
    N_HARDWARE,
    N_HEDGE,
    N_LEDGE,
    N_LOGICAL,
    N_ROUTE,
    Observation,
)

WIDTH = 128
EDGE_WIDTH = 32
PADDED_ACTIONS = 73
ACTION_QUALITY_HEAD = "bounded-qmu-action-head-v2"
ACTION_QUALITY_POLICY_COUPLING_SCHEMA = "isingfold.action-quality-policy-coupling-v1"
QUALITY_POLICY_PRIOR_DEFAULT = "bounded-centered-v1"
QUALITY_POLICY_PRIOR_MODES = (
    "auxiliary-only-v1",
    "bounded-centered-v1",
    "clipped-logit-v1",
    "bounded-centered-detached-v1",
    "raw-logit-diagnostic-v1",
)


def quality_policy_prior_contract(mode: str) -> dict[str, object]:
    """Return the immutable transform and gradient semantics for one registered mode."""

    contracts: dict[str, dict[str, object]] = {
        "auxiliary-only-v1": {
            "transform": "zero",
            "output_interval": [0.0, 0.0],
            "policy_gradient_to_quality_head": False,
            "publication_eligible": True,
        },
        "bounded-centered-v1": {
            "transform": "two-sigmoid-minus-one",
            "output_interval": [-1.0, 1.0],
            "policy_gradient_to_quality_head": True,
            "publication_eligible": True,
        },
        "clipped-logit-v1": {
            "transform": "one-half-times-clamp-minus-two-plus-two",
            "output_interval": [-1.0, 1.0],
            "policy_gradient_to_quality_head": True,
            "publication_eligible": True,
        },
        "bounded-centered-detached-v1": {
            "transform": "detach-two-sigmoid-minus-one",
            "output_interval": [-1.0, 1.0],
            "policy_gradient_to_quality_head": False,
            "publication_eligible": True,
        },
        "raw-logit-diagnostic-v1": {
            "transform": "identity",
            "output_interval": None,
            "policy_gradient_to_quality_head": True,
            "publication_eligible": False,
        },
    }
    if not isinstance(mode, str) or mode not in contracts:
        raise ValueError(
            f"unknown quality-prior mode {mode!r}; expected one of {QUALITY_POLICY_PRIOR_MODES}"
        )
    return {
        "schema": ACTION_QUALITY_POLICY_COUPLING_SCHEMA,
        "mode": mode,
        **contracts[mode],
    }


def mlp(sizes: Sequence[int], layer_norm: bool = False) -> nn.Module:
    layers: list[nn.Module] = []
    for k in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[k], sizes[k + 1]))
        if k < len(sizes) - 2:
            layers.append(nn.SiLU())
    if layer_norm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


def projector(input_width: int, output_width: int = WIDTH) -> nn.Sequential:
    """The exact Appendix-A projector: Linear, SiLU, then LayerNorm."""

    return nn.Sequential(
        nn.Linear(input_width, output_width),
        nn.SiLU(),
        nn.LayerNorm(output_width),
    )


def _segment(src: Tensor, index: Tensor, size: int) -> Tensor:
    """Mean and max over segments; an empty segment returns zeros, never ``-inf``."""

    width = src.shape[-1]
    if size == 0:
        return src.new_zeros((0, 2 * width))
    total = src.new_zeros((size, width))
    count = src.new_zeros((size, 1))
    if src.numel():
        total.index_add_(0, index, src)
        count.index_add_(0, index, src.new_ones((src.shape[0], 1)))
    mean = total / count.clamp(min=1.0)
    maximum = src.new_full((size, width), float("-inf"))
    if src.numel():
        maximum = maximum.scatter_reduce(
            0, index.unsqueeze(-1).expand(-1, width), src, reduce="amax", include_self=True
        )
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return torch.cat([mean, maximum], dim=-1)


class LocalBlock(nn.Module):
    """One dual local message-passing block, Eq. (6)."""

    def __init__(self) -> None:
        super().__init__()
        self.message = mlp([2 * WIDTH + EDGE_WIDTH, WIDTH, WIDTH])
        self.update = mlp([WIDTH + 2 * WIDTH, 2 * WIDTH, WIDTH])
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, z: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            aggregated = z.new_zeros((z.shape[0], 2 * WIDTH))
        else:
            src, dst = edge_index[0], edge_index[1]
            m = self.message(torch.cat([z[src], z[dst], edge_attr], dim=-1))
            aggregated = _segment(m, dst, z.shape[0])
        return self.norm(z + self.update(torch.cat([z, aggregated], dim=-1)))


class RelationBlock(nn.Module):
    """One typed relation inside a fusion block, with the degree gate of Eq. (7)."""

    def __init__(self) -> None:
        super().__init__()
        self.message = mlp([2 * WIDTH + EDGE_WIDTH, WIDTH, WIDTH])
        self.update = mlp([WIDTH + 2 * WIDTH, 2 * WIDTH, WIDTH])

    def forward(self, z_dst: Tensor, z_src: Tensor, index: Tensor, edge_attr: Tensor) -> Tensor:
        if index.numel() == 0:
            return torch.zeros_like(z_dst)
        src, dst = index[0], index[1]
        m = self.message(torch.cat([z_src[src], z_dst[dst], edge_attr], dim=-1))
        aggregated = _segment(m, dst, z_dst.shape[0])
        degree = z_dst.new_zeros((z_dst.shape[0], 1))
        degree.index_add_(0, dst, z_dst.new_ones((dst.shape[0], 1)))
        return (degree > 0).float() * self.update(torch.cat([z_dst, aggregated], dim=-1))


class FusionBlock(nn.Module):
    """Six directional ownership/conflict relations, synchronous update, Eq. (7)."""

    RELATIONS = ("L2H", "H2L", "L2X", "X2L", "H2X", "X2H")

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict({r: RelationBlock() for r in self.RELATIONS})
        self.norm_l = nn.LayerNorm(WIDTH)
        self.norm_h = nn.LayerNorm(WIDTH)
        self.norm_x = nn.LayerNorm(WIDTH)

    def forward(
        self,
        zl: Tensor,
        zh: Tensor,
        zx: Tensor,
        claims: Tensor,
        claim_attr: Tensor,
        conflict_claimants: Tensor,
        conflict_host: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        zero_x = zl.new_zeros(
            (conflict_claimants.shape[1] if conflict_claimants.numel() else 0, EDGE_WIDTH)
        )
        zero_h = zl.new_zeros((conflict_host.shape[1] if conflict_host.numel() else 0, EDGE_WIDTH))
        # Tensor relations are stored as (logical, hardware), (conflict, logical), and
        # (conflict, hardware), respectively.  RelationBlock expects (source, destination).
        h2l = torch.flip(claims, dims=[0]) if claims.numel() else claims
        l2x = (
            torch.flip(conflict_claimants, dims=[0])
            if conflict_claimants.numel()
            else conflict_claimants
        )
        h2x = torch.flip(conflict_host, dims=[0]) if conflict_host.numel() else conflict_host

        uh = self.blocks["L2H"](zh, zl, claims, claim_attr)
        ul = self.blocks["H2L"](zl, zh, h2l, claim_attr)
        ul = ul + self.blocks["X2L"](zl, zx, conflict_claimants, zero_x)
        ux = self.blocks["L2X"](zx, zl, l2x, zero_x)
        ux = ux + self.blocks["H2X"](zx, zh, h2x, zero_h)
        uh = uh + self.blocks["X2H"](zh, zx, conflict_host, zero_h)
        return self.norm_l(zl + ul), self.norm_h(zh + uh), self.norm_x(zx + ux)


class RouteEncoder(nn.Module):
    """Two gated convolution blocks over ordered positions, Eq. (12)."""

    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(EDGE_WIDTH + WIDTH, WIDTH)
        self.pos = mlp([2 * N_ROUTE, EDGE_WIDTH])
        self.conv = nn.ModuleList([nn.Conv1d(WIDTH, 2 * WIDTH, 3, padding=1) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(WIDTH) for _ in range(2)])
        self.pool = nn.Linear(4 * WIDTH, WIDTH)
        self.bind = mlp([3 * WIDTH, WIDTH, WIDTH])
        self.phase = nn.Embedding(3, WIDTH)

    def forward(
        self,
        route_features: Tensor,
        route_qubits: Tensor,
        zh: Tensor,
        route_owner: Tensor,
        route_roles: Tensor,
        zl: Tensor,
        n_routes: int,
        route_of_position: Tensor,
        max_route_length: int,
    ) -> Tensor:
        if n_routes == 0 or route_features.shape[0] == 0:
            return zl.new_zeros((0, WIDTH))
        if (
            isinstance(max_route_length, bool)
            or not isinstance(max_route_length, int)
            or max_route_length <= 0
        ):
            raise ValueError("max_route_length must be a positive host integer")
        x = self.project(torch.cat([self.pos(route_features), zh[route_qubits]], dim=-1))
        lengths = torch.zeros(n_routes, dtype=torch.long, device=x.device)
        lengths.index_add_(0, route_of_position, torch.ones_like(route_of_position))
        padded = x.new_zeros((n_routes, max_route_length, WIDTH))
        mask = x.new_zeros((n_routes, max_route_length, 1))
        # Compute each token's position within its route entirely on-device.  The
        # former scalar loop called ``.item()`` once per token, forcing a CUDA
        # synchronization for every ordered route position.  Stable sorting preserves
        # the input order inside each route even when route tokens are interleaved.
        order = torch.argsort(route_of_position, stable=True)
        starts = torch.cumsum(lengths, dim=0) - lengths
        sorted_positions = torch.arange(
            route_of_position.shape[0], dtype=torch.long, device=x.device
        ) - torch.repeat_interleave(starts, lengths)
        positions = torch.empty_like(route_of_position)
        positions[order] = sorted_positions
        padded[route_of_position, positions] = x
        mask[route_of_position, positions] = 1.0
        seq = padded * mask
        for conv, norm in zip(self.conv, self.norm, strict=True):
            u, v = conv(seq.transpose(1, 2)).chunk(2, dim=1)
            gated = torch.tanh(u) * torch.sigmoid(v)
            seq = norm(seq + gated.transpose(1, 2)) * mask
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean = seq.sum(dim=1) / counts
        maximum = seq.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        first = seq[:, 0, :]
        last_index = (counts.squeeze(-1).long() - 1).clamp(min=0)
        last = seq[torch.arange(n_routes, device=x.device), last_index]
        r0 = self.pool(torch.cat([mean, maximum, first, last], dim=-1))
        return self.bind(torch.cat([r0, zl[route_owner], self.phase(route_roles)], dim=-1))


@dataclass
class ModelOutput:
    masked_log_probs: Tensor
    utility_value: Tensor
    failure_logit: Tensor
    failure_value: Tensor
    action_count: Tensor
    action_quality_value: Tensor | None = None
    action_quality_logit: Tensor | None = None


class IFCore(nn.Module):
    """The reference implementation IF-Core-v1: width 128, three local, two fusion blocks."""

    def __init__(
        self,
        width: int = WIDTH,
        local_blocks: int = 3,
        fusion_blocks: int = 2,
        improvement_mode: bool = True,
        quality_prior_mode: str = QUALITY_POLICY_PRIOR_DEFAULT,
    ) -> None:
        super().__init__()
        if width != WIDTH:  # the spec fixes the pilot width
            raise ValueError("IF-Core-v1 fixes hidden width 128")
        if type(improvement_mode) is not bool:
            raise ValueError("improvement_mode must be Boolean")
        if (
            isinstance(local_blocks, bool)
            or not isinstance(local_blocks, int)
            or local_blocks < 0
            or isinstance(fusion_blocks, bool)
            or not isinstance(fusion_blocks, int)
            or fusion_blocks < 0
        ):
            raise ValueError("local and fusion block counts must be nonnegative integers")
        self.improvement_mode = improvement_mode
        self.model_family = "IF-Core"
        self.action_quality_head = ACTION_QUALITY_HEAD
        self.quality_prior_mode = quality_prior_mode
        self.action_quality_policy_coupling = quality_policy_prior_contract(quality_prior_mode)
        self.proj_logical = projector(2 * N_LOGICAL)
        self.proj_hardware = projector(2 * N_HARDWARE)
        self.proj_conflict = projector(2 * N_CONFLICT)
        self.proj_ledge = mlp([2 * N_LEDGE, EDGE_WIDTH])
        self.proj_hedge = mlp([2 * N_HEDGE, EDGE_WIDTH])
        self.proj_claim = mlp([2 * N_CLAIM, EDGE_WIDTH])
        self.proj_factor = projector(2 * N_FACTOR)
        self.proj_archive = projector(2 * N_ARCHIVE)
        self.proj_action = projector(2 * N_ACTION + len(OPCODES))
        self.proj_global = mlp([2 * N_GLOBAL, 64])
        self.role = nn.Embedding(3, WIDTH)

        self.local_l = nn.ModuleList([LocalBlock() for _ in range(local_blocks)])
        self.local_h = nn.ModuleList([LocalBlock() for _ in range(local_blocks)])
        self.fusion = nn.ModuleList([FusionBlock() for _ in range(fusion_blocks)])

        self.context_hedge = mlp([2 * WIDTH + EDGE_WIDTH, 64, EDGE_WIDTH])
        self.context_ledge = mlp([2 * WIDTH + EDGE_WIDTH, 64, EDGE_WIDTH])
        self.edge_use = mlp([2 * EDGE_WIDTH + 2, 64, EDGE_WIDTH])
        self.factor = mlp([5 * WIDTH, 2 * WIDTH, WIDTH])
        self.pair = mlp([4 * WIDTH, 2 * WIDTH, WIDTH])
        self.routes = RouteEncoder()
        self.archive_entry = mlp([3 * WIDTH, 2 * WIDTH, WIDTH])
        self.archive_summary = mlp([2 * WIDTH, WIDTH])
        self.action = mlp([WIDTH + 2 * WIDTH + 2 * WIDTH + 2 * WIDTH + WIDTH, 2 * WIDTH, WIDTH])
        self.action_set = mlp([2 * WIDTH, WIDTH])
        self.state = mlp([6 * WIDTH + WIDTH + 64, 2 * WIDTH, WIDTH])
        self.actor = mlp([4 * WIDTH, 2 * WIDTH, WIDTH, 1])
        self.action_quality = mlp([4 * WIDTH, 2 * WIDTH, WIDTH, 1])
        self.utility = mlp([2 * WIDTH, WIDTH, WIDTH // 2, 1])
        self.failure = mlp([2 * WIDTH, WIDTH, WIDTH // 2, 1])
        if self.improvement_mode:
            self.failure.requires_grad_(False)

    def quality_policy_prior(self, action_quality_logit: Tensor) -> Tensor:
        """Transform warm-start Q logits into the registered bounded policy feature."""

        if self.quality_prior_mode == "auxiliary-only-v1":
            return torch.zeros_like(action_quality_logit)
        if self.quality_prior_mode == "bounded-centered-v1":
            return 2.0 * torch.sigmoid(action_quality_logit) - 1.0
        if self.quality_prior_mode == "clipped-logit-v1":
            return 0.5 * torch.clamp(action_quality_logit, min=-2.0, max=2.0)
        if self.quality_prior_mode == "bounded-centered-detached-v1":
            return (2.0 * torch.sigmoid(action_quality_logit) - 1.0).detach()
        if self.quality_prior_mode == "raw-logit-diagnostic-v1":
            return action_quality_logit
        raise RuntimeError("model carries an unregistered quality-prior mode")

    # -- helpers ---------------------------------------------------------------------

    @staticmethod
    def _t(array: np.ndarray, device: torch.device, dtype=torch.float32) -> Tensor:
        return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device)

    def encode(self, obs: Observation, device: torch.device) -> dict[str, Tensor]:
        zl = self.proj_logical(self._t(obs.logical, device))
        zh = self.proj_hardware(self._t(obs.hardware, device))
        zx = self.proj_conflict(self._t(obs.conflicts, device))
        el = self.proj_ledge(self._t(obs.logical_edges, device))
        eh = self.proj_hedge(self._t(obs.hardware_edges, device))
        claim_attr = self.proj_claim(self._t(obs.claims, device))
        il = self._t(obs.index_logical_edges, device, torch.long)
        ih = self._t(obs.index_hardware_edges, device, torch.long)
        ic = self._t(obs.index_claims, device, torch.long)
        icc = self._t(obs.index_conflict_claimants, device, torch.long)
        ich = self._t(obs.index_conflict_host, device, torch.long)

        for block_l, block_h in zip(self.local_l, self.local_h, strict=True):
            zl = block_l(zl, il, el)
            zh = block_h(zh, ih, eh)
        for fusion in self.fusion:
            zl, zh, zx = fusion(zl, zh, zx, ic, claim_attr, icc, ich)
        return {"zl": zl, "zh": zh, "zx": zx, "claim_attr": claim_attr}

    @staticmethod
    def _validate_graph_indices(obs: Observation) -> None:
        """Reject malformed local graph relations before disjoint-union batching.

        Offsetting an invalid local index could otherwise make it point into the next
        observation.  The single-observation path also uses this check so batching cannot
        make a previously invalid tensor silently meaningful.
        """

        limits = {
            "index_logical_edges": (obs.logical.shape[0], obs.logical.shape[0]),
            "index_hardware_edges": (obs.hardware.shape[0], obs.hardware.shape[0]),
            "index_claims": (obs.logical.shape[0], obs.hardware.shape[0]),
            "index_conflict_claimants": (obs.conflicts.shape[0], obs.logical.shape[0]),
            "index_conflict_host": (obs.conflicts.shape[0], obs.hardware.shape[0]),
        }
        for name, row_limits in limits.items():
            relation = np.asarray(getattr(obs, name))
            if relation.ndim != 2 or relation.shape[0] != len(row_limits):
                raise ValueError(f"{name} must have shape ({len(row_limits)}, N)")
            for row, limit in enumerate(row_limits):
                values = relation[row]
                if values.size and (np.any(values < 0) or np.any(values >= limit)):
                    raise ValueError(f"{name} contains an out-of-range local index")

    def _encode_batch(
        self, observations: Sequence[Observation], device: torch.device
    ) -> list[dict[str, Tensor]]:
        """Encode a graph-disjoint observation union in one set of trunk kernels.

        Every relation endpoint is shifted by the corresponding local node offset.  The
        segment reducers therefore retain exactly the same neighbourhoods as independent
        encoding, while projectors and message-passing layers operate on a materially larger
        device batch.
        """

        counts_l = [int(obs.logical.shape[0]) for obs in observations]
        counts_h = [int(obs.hardware.shape[0]) for obs in observations]
        counts_x = [int(obs.conflicts.shape[0]) for obs in observations]
        counts_c = [int(obs.claims.shape[0]) for obs in observations]

        offsets_l = np.cumsum([0, *counts_l[:-1]], dtype=np.int64)
        offsets_h = np.cumsum([0, *counts_h[:-1]], dtype=np.int64)
        offsets_x = np.cumsum([0, *counts_x[:-1]], dtype=np.int64)

        def rows(name: str) -> np.ndarray:
            return np.concatenate([np.asarray(getattr(obs, name)) for obs in observations], axis=0)

        def relation(name: str, offsets: Sequence[int]) -> np.ndarray:
            shifted: list[np.ndarray] = []
            for obs, local_offsets in zip(observations, offsets, strict=True):
                item = np.asarray(getattr(obs, name), dtype=np.int64).copy()
                for row, offset in enumerate(local_offsets):
                    item[row] += int(offset)
                shifted.append(item)
            return np.concatenate(shifted, axis=1)

        zl = self.proj_logical(self._t(rows("logical"), device))
        zh = self.proj_hardware(self._t(rows("hardware"), device))
        zx = self.proj_conflict(self._t(rows("conflicts"), device))
        el = self.proj_ledge(self._t(rows("logical_edges"), device))
        eh = self.proj_hedge(self._t(rows("hardware_edges"), device))
        claim_attr = self.proj_claim(self._t(rows("claims"), device))
        il = self._t(
            relation("index_logical_edges", [(o, o) for o in offsets_l]), device, torch.long
        )
        ih = self._t(
            relation("index_hardware_edges", [(o, o) for o in offsets_h]),
            device,
            torch.long,
        )
        ic = self._t(
            relation("index_claims", list(zip(offsets_l, offsets_h, strict=True))),
            device,
            torch.long,
        )
        icc = self._t(
            relation(
                "index_conflict_claimants",
                list(zip(offsets_x, offsets_l, strict=True)),
            ),
            device,
            torch.long,
        )
        ich = self._t(
            relation("index_conflict_host", list(zip(offsets_x, offsets_h, strict=True))),
            device,
            torch.long,
        )

        for block_l, block_h in zip(self.local_l, self.local_h, strict=True):
            zl = block_l(zl, il, el)
            zh = block_h(zh, ih, eh)
        for fusion in self.fusion:
            zl, zh, zx = fusion(zl, zh, zx, ic, claim_attr, icc, ich)

        encoded: list[dict[str, Tensor]] = []
        starts_l = np.cumsum([0, *counts_l], dtype=np.int64)
        starts_h = np.cumsum([0, *counts_h], dtype=np.int64)
        starts_x = np.cumsum([0, *counts_x], dtype=np.int64)
        starts_c = np.cumsum([0, *counts_c], dtype=np.int64)
        for index in range(len(observations)):
            encoded.append(
                {
                    "zl": zl[int(starts_l[index]) : int(starts_l[index + 1])],
                    "zh": zh[int(starts_h[index]) : int(starts_h[index + 1])],
                    "zx": zx[int(starts_x[index]) : int(starts_x[index + 1])],
                    "claim_attr": claim_attr[int(starts_c[index]) : int(starts_c[index + 1])],
                }
            )
        return encoded

    def forward_single(self, obs: Observation, device: torch.device | None = None) -> ModelOutput:
        device = device or next(self.parameters()).device
        self._validate_graph_indices(obs)
        enc = self.encode(obs, device)
        return self._decode_single(obs, enc, device)

    def _decode_single(
        self, obs: Observation, enc: dict[str, Tensor], device: torch.device
    ) -> ModelOutput:
        n_actions = obs.actions.shape[0]
        if obs.legal_mask.shape != (n_actions,) or obs.real_action_mask.shape != (n_actions,):
            raise ValueError("action, real-row, and legal masks must have identical lengths")
        if n_actions > PADDED_ACTIONS:
            raise ValueError(f"IF-Core supports at most {PADDED_ACTIONS} action rows")
        legal_numpy = np.logical_and(obs.legal_mask, obs.real_action_mask)
        if not legal_numpy.any():
            raise ValueError("IF-Core requires a nonempty legal support")

        zl, zh, zx = enc["zl"], enc["zh"], enc["zx"]

        # chain factors, Eq. (10) with the OLD/NEW/ARCHIVE role embedding
        factors = self._t(obs.factors, device)
        n_factors = factors.shape[0]
        if n_factors:
            d = self.proj_factor(factors) + self.role(self._t(obs.factor_roles, device, torch.long))
            fm = self._t(obs.index_factor_membership, device, torch.long)
            membership = (
                _segment(zh[fm[1]], fm[0], n_factors)
                if fm.numel()
                else zh.new_zeros((n_factors, 2 * WIDTH))
            )
            fl = self._t(obs.index_factor_logical, device, torch.long)
            owner = zl.new_zeros((n_factors, WIDTH))
            if fl.numel():
                owner[fl[0]] = zl[fl[1]]

            n_uses = int(obs.edge_use_roles.shape[0])
            if n_uses:
                use_factor = self._t(obs.index_edge_use_factor, device, torch.long)
                use_hardware = self._t(obs.index_edge_use_hardware, device, torch.long)
                use_logical_numpy = np.asarray(obs.index_edge_use_logical, dtype=np.int64)
                use_logical = self._t(use_logical_numpy, device, torch.long)
                use_roles = self._t(obs.edge_use_roles, device, torch.long)
                ih = self._t(obs.index_hardware_edges, device, torch.long)
                he_endpoints = ih[:, use_hardware]
                raw_h = self.proj_hedge(self._t(obs.edge_use_hardware, device))
                bar_h = self.context_hedge(
                    torch.cat(
                        [
                            zh[he_endpoints[0]] + zh[he_endpoints[1]],
                            torch.abs(zh[he_endpoints[0]] - zh[he_endpoints[1]]),
                            raw_h,
                        ],
                        dim=-1,
                    )
                )
                bar_l = zl.new_zeros((n_uses, EDGE_WIDTH))
                logical_use_numpy = use_logical_numpy >= 0
                if logical_use_numpy.any():
                    logical_use = self._t(logical_use_numpy, device, torch.bool)
                    il = self._t(obs.index_logical_edges, device, torch.long)
                    logical_rows = use_logical[logical_use]
                    le_endpoints = il[:, logical_rows]
                    raw_l = self.proj_ledge(self._t(obs.edge_use_logical, device)[logical_use])
                    bar_l[logical_use] = self.context_ledge(
                        torch.cat(
                            [
                                zl[le_endpoints[0]] + zl[le_endpoints[1]],
                                torch.abs(zl[le_endpoints[0]] - zl[le_endpoints[1]]),
                                raw_l,
                            ],
                            dim=-1,
                        )
                    )
                use_onehot = torch.nn.functional.one_hot(use_roles, num_classes=2).to(zl.dtype)
                use_vectors = self.edge_use(torch.cat([bar_h, bar_l, use_onehot], dim=-1))
                use = _segment(use_vectors, use_factor, n_factors)
                realized_relation = self._t(obs.index_factor_realized_edge_use, device, torch.long)
                realized = (
                    _segment(bar_l[realized_relation[1]], realized_relation[0], n_factors)
                    if realized_relation.numel()
                    else zl.new_zeros((n_factors, 2 * EDGE_WIDTH))
                )
            else:
                use = zl.new_zeros((n_factors, 2 * EDGE_WIDTH))
                realized = zl.new_zeros((n_factors, 2 * EDGE_WIDTH))
            b = self.factor(torch.cat([d, owner, membership, use, realized], dim=-1))
        else:
            b = zl.new_zeros((0, WIDTH))

        fa = self._t(obs.index_factor_action, device, torch.long)
        roles = self._t(obs.factor_roles, device, torch.long)
        joint = zl.new_zeros((max(n_actions, 1), 2 * WIDTH))
        if fa.numel() and n_actions:
            relation_factors = fa[0]
            old_factor_ids = relation_factors[roles[relation_factors] == 0]
            new_factor_ids = relation_factors[roles[relation_factors] == 1]
            # pair OLD and NEW by (action, logical owner), Eq. (11)
            fl = self._t(obs.index_factor_logical, device, torch.long)
            owner_of = torch.zeros(n_factors, dtype=torch.long, device=device)
            if fl.numel():
                owner_of[fl[0]] = fl[1]
            action_of = torch.full((n_factors,), -1, dtype=torch.long, device=device)
            action_of[fa[0]] = fa[1]
            key = action_of * (int(zl.shape[0]) + 1) + owner_of
            if old_factor_ids.numel() and new_factor_ids.numel():
                # Match OLD/NEW factors entirely on device, then project every pair in
                # one matrix operation.  The former Python dictionary/loop launched one
                # tiny MLP per (action, owner), which serialized CUDA execution.
                new_keys = key[new_factor_ids]
                new_order = torch.argsort(new_keys, stable=True)
                sorted_new_keys = new_keys[new_order]
                old_keys = key[old_factor_ids]
                positions = torch.searchsorted(sorted_new_keys, old_keys, right=True) - 1
                safe_positions = positions.clamp(min=0)
                matched = (positions >= 0) & (sorted_new_keys[safe_positions] == old_keys)
                matched_old = old_factor_ids[matched]
                if matched_old.numel():
                    matched_new = new_factor_ids[new_order[safe_positions[matched]]]
                    pair_vectors = self.pair(
                        torch.cat(
                            [
                                b[matched_old],
                                b[matched_new],
                                b[matched_new] - b[matched_old],
                                zl[owner_of[matched_old]],
                            ],
                            dim=-1,
                        )
                    )
                    joint = _segment(
                        pair_vectors,
                        action_of[matched_old],
                        n_actions,
                    )
        joint = joint[:n_actions] if n_actions else joint.new_zeros((0, 2 * WIDTH))

        # routes, Eq. (12)
        n_routes = int(obs.route_roles.shape[0])
        route_summary = zl.new_zeros((max(n_actions, 1), 2 * WIDTH))
        if n_routes and obs.routes.shape[0]:
            rp = self._t(obs.index_route_positions, device, torch.long)
            ra = self._t(obs.index_route_action, device, torch.long)
            route_logical = np.asarray(obs.index_route_logical, dtype=np.int64)
            route_owner_numpy = np.full(n_routes, -1, dtype=np.int64)
            route_owner_numpy[route_logical[0]] = route_logical[1]
            if np.any(route_owner_numpy < 0):
                raise ValueError("every real route needs exactly one logical owner")
            route_owner = self._t(route_owner_numpy, device, torch.long)
            vectors = self.routes(
                self._t(obs.routes, device),
                rp[1],
                zh,
                route_owner,
                self._t(obs.route_roles, device, torch.long),
                zl,
                n_routes,
                rp[0],
                int(
                    np.bincount(
                        np.asarray(obs.index_route_positions[0], dtype=np.int64),
                        minlength=n_routes,
                    ).max()
                ),
            )
            route_summary = _segment(vectors, ra[1], max(n_actions, 1))
        route_summary = (
            route_summary[:n_actions] if n_actions else route_summary.new_zeros((0, 2 * WIDTH))
        )

        # touched conflicts and archive tokens
        conflict_summary = zl.new_zeros((n_actions, 2 * WIDTH))
        if zx.shape[0] and n_actions and obs.index_action_conflicts.size:
            action_conflicts = self._t(obs.index_action_conflicts, device, torch.long)
            conflict_summary = _segment(zx[action_conflicts[1]], action_conflicts[0], n_actions)

        archive_rows = self._t(obs.archive, device)
        n_entries = archive_rows.shape[0]
        if n_entries:
            fz = self._t(obs.index_factor_archive, device, torch.long)
            per_entry = (
                _segment(b[fz[0]], fz[1], n_entries)
                if fz.numel()
                else b.new_zeros((n_entries, 2 * WIDTH))
            )
            iota_k = self.archive_entry(
                torch.cat([self.proj_archive(archive_rows), per_entry], dim=-1)
            )
            iota = self.archive_summary(
                torch.cat([iota_k.mean(dim=0), iota_k.max(dim=0).values], dim=-1)
            )
        else:
            iota_k = zl.new_zeros((0, WIDTH))
            iota = zl.new_zeros(WIDTH)

        action_archive = zl.new_zeros((n_actions, WIDTH))
        if n_entries and obs.index_action_archive.size:
            ia = self._t(obs.index_action_archive, device, torch.long)
            action_archive[ia[0]] = iota_k[ia[1]]

        v_a = zl.new_zeros((n_actions, WIDTH))
        if n_actions:
            v_a = self.action(
                torch.cat(
                    [
                        self.proj_action(self._t(obs.actions, device)),
                        joint,
                        route_summary,
                        conflict_summary,
                        action_archive,
                    ],
                    dim=-1,
                )
            )

        legal = self._t(legal_numpy.astype(np.float32), device).bool()
        legal_actions = v_a[legal]
        c_k = self.action_set(
            torch.cat([legal_actions.mean(dim=0), legal_actions.max(dim=0).values], dim=-1)
        )

        pools = [zl, zh, zx]
        pooled = []
        for z in pools:
            if z.shape[0]:
                pooled.extend([z.mean(dim=0), z.max(dim=0).values])
            else:
                pooled.extend([z.new_zeros(WIDTH), z.new_zeros(WIDTH)])
        g = self.proj_global(self._t(obs.globals_, device)).squeeze(0)
        z_s = self.state(torch.cat([*pooled, iota, g], dim=-1))

        actor_input = torch.cat(
            [
                v_a,
                z_s.unsqueeze(0).expand(n_actions, -1),
                c_k.unsqueeze(0).expand(n_actions, -1),
                v_a * z_s.unsqueeze(0),
            ],
            dim=-1,
        )
        action_quality_logit = self.action_quality(actor_input).squeeze(-1)
        action_quality_value = torch.sigmoid(action_quality_logit)
        logits = self.actor(actor_input).squeeze(-1) + self.quality_policy_prior(
            action_quality_logit
        )
        log_probs = logits - torch.logsumexp(logits[legal], dim=0)
        log_probs = torch.where(legal, log_probs, torch.full_like(log_probs, float("-inf")))

        critic_in = torch.cat([z_s, c_k], dim=-1)
        utility = self.utility(critic_in).squeeze(-1)
        if self.improvement_mode:
            failure_logit = torch.zeros_like(utility)
            failure_value = torch.zeros_like(utility)
        else:
            failure_logit = self.failure(critic_in).squeeze(-1)
            failure_value = torch.sigmoid(failure_logit)
        return ModelOutput(
            masked_log_probs=log_probs,
            utility_value=utility,
            failure_logit=failure_logit,
            failure_value=failure_value,
            action_count=torch.tensor(
                int(np.count_nonzero(obs.real_action_mask)), dtype=torch.long, device=device
            ),
            action_quality_value=action_quality_value,
            action_quality_logit=action_quality_logit,
        )

    def forward(
        self,
        observation_batch: Sequence[Observation],
        candidate_batch: Sequence[np.ndarray] | None = None,
        legal_mask: Sequence[np.ndarray] | None = None,
        device: torch.device | None = None,
    ) -> ModelOutput:
        """Public padded batch API from MODEL_SPEC section 7.1.

        ``Observation`` owns all relational candidate payloads.  The optional candidate and
        legal arrays make replay call sites state their stored support explicitly; they must
        have one row per observation and are installed without regenerating candidates.
        Graph trunks are encoded as a disjoint union, then sliced before observation-local
        factor, route, archive, action and state pooling.  No pool or categorical denominator
        can cross an observation boundary.
        """

        observations = list(observation_batch)
        if not observations:
            raise ValueError("observation_batch must be nonempty")
        if candidate_batch is not None and len(candidate_batch) != len(observations):
            raise ValueError("candidate_batch length differs from observation_batch")
        if legal_mask is not None and len(legal_mask) != len(observations):
            raise ValueError("legal_mask length differs from observation_batch")

        prepared: list[Observation] = []
        for index, observation in enumerate(observations):
            actions = (
                np.asarray(candidate_batch[index], dtype=np.float32)
                if candidate_batch is not None
                else observation.actions
            )
            mask = (
                np.asarray(legal_mask[index], dtype=bool)
                if legal_mask is not None
                else observation.legal_mask
            )
            if actions.shape != observation.actions.shape:
                raise ValueError("candidate tensor shape differs from its stored support")
            prepared.append(replace(observation, actions=actions, legal_mask=mask))

        for observation in prepared:
            self._validate_graph_indices(observation)
        target_device = device or next(self.parameters()).device
        encoded = self._encode_batch(prepared, target_device)
        outputs = [
            self._decode_single(observation, enc, target_device)
            for observation, enc in zip(prepared, encoded, strict=True)
        ]
        target_device = outputs[0].masked_log_probs.device
        padded = torch.full(
            (len(outputs), PADDED_ACTIONS),
            float("-inf"),
            dtype=outputs[0].masked_log_probs.dtype,
            device=target_device,
        )
        for row, output in enumerate(outputs):
            count = output.masked_log_probs.shape[0]
            padded[row, :count] = output.masked_log_probs
        quality_values = torch.full_like(padded, float("nan"))
        quality_logits = torch.full_like(padded, float("nan"))
        for row, output in enumerate(outputs):
            if output.action_quality_value is None or output.action_quality_logit is None:
                raise RuntimeError("IF-Core omitted its registered action-quality head")
            count = output.masked_log_probs.shape[0]
            quality_values[row, :count] = output.action_quality_value
            quality_logits[row, :count] = output.action_quality_logit
        return ModelOutput(
            masked_log_probs=padded,
            utility_value=torch.stack([output.utility_value for output in outputs]),
            failure_logit=torch.stack([output.failure_logit for output in outputs]),
            failure_value=torch.stack([output.failure_value for output in outputs]),
            action_count=torch.stack([output.action_count for output in outputs]),
            action_quality_value=quality_values,
            action_quality_logit=quality_logits,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class IFDual(IFCore):
    """Three separate local MPNNs and phase-aware factors, without cross-graph fusion."""

    def __init__(
        self,
        *,
        improvement_mode: bool = True,
        quality_prior_mode: str = QUALITY_POLICY_PRIOR_DEFAULT,
    ) -> None:
        super().__init__(
            local_blocks=3,
            fusion_blocks=0,
            improvement_mode=improvement_mode,
            quality_prior_mode=quality_prior_mode,
        )
        self.model_family = "IF-Dual"


class IFMLP(IFCore):
    """No graph message passing: shared projectors, exact descriptors and relational pooling."""

    def __init__(
        self,
        *,
        improvement_mode: bool = True,
        quality_prior_mode: str = QUALITY_POLICY_PRIOR_DEFAULT,
    ) -> None:
        super().__init__(
            local_blocks=0,
            fusion_blocks=0,
            improvement_mode=improvement_mode,
            quality_prior_mode=quality_prior_mode,
        )
        self.model_family = "IF-MLP"


MODEL_FAMILIES = ("if-mlp", "if-dual", "if-core")


def build_model(
    family: str,
    *,
    improvement_mode: bool = True,
    quality_prior_mode: str = QUALITY_POLICY_PRIOR_DEFAULT,
) -> IFCore:
    """Build one registered representation-screen family, failing on unknown aliases."""

    normalized = family.strip().lower()
    constructors = {
        "if-mlp": IFMLP,
        "if-dual": IFDual,
        "if-core": IFCore,
    }
    try:
        constructor = constructors[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown model family {family!r}; expected one of {MODEL_FAMILIES}"
        ) from exc
    return constructor(
        improvement_mode=improvement_mode,
        quality_prior_mode=quality_prior_mode,
    )


# Public name used by the architecture document.  ``IFCore`` remains the compatibility
# spelling used by the initial PPO/evaluation implementation.
GraphActorCritic = IFCore

"""Independent graph selector for the four IF-Q3-S0 terminal programs.

The deployable selector is deliberately separate from the PPO actor.  It evaluates the
logical graph, the compiled physical program and the logical-to-hardware ownership relation
at one registered strength.  The same parameters are reused for all four strengths and are
frozen before PPO collection.  Ground energies, witnesses, evaluator keys and success counts
are labels or provenance only; none is representable by :class:`StrengthGraphInput`.

``predict`` and feature-only ``fit_selector`` remain as an explicitly legacy compatibility
path for old aggregate-only corpora.  New training and deployment use ``forward`` /
``select_embedding`` with graph inputs.  The compatibility path never fabricates a graph.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as torch_f

WIDTH = 128
EDGE_WIDTH = 32
N_LOGICAL = 20
N_HARDWARE = 18
N_LOGICAL_EDGE = 6
N_HARDWARE_EDGE = 10
N_CLAIM = 6
N_GLOBAL = 32
N_STRENGTHS = 4
NODE_PROJECTOR_ARCHITECTURE = ("Linear", "SiLU", "LayerNorm")

# Zero-based scalar slots excluded from the selector.  Both the value and knownness channel
# are masked inside the model, so a corrupt/pre-v1 tensorizer cannot expose search memory.
MASKED_LOGICAL_SLOTS = (16, 17)
MASKED_HARDWARE_SLOTS = (15,)
MASKED_LOGICAL_EDGE_SLOTS = (5,)
MASKED_CLAIM_SLOTS = (0,)
ALLOWED_GLOBAL_SLOTS = tuple(range(0, 8)) + (12,) + tuple(range(14, 23))

FEATURE_ORDER: tuple[str, ...] = (
    "strength",
    "scale",
    "qubits",
    "max_chain",
    "mean_chain",
    "single_qubit_fraction",
    "max_field",
    "max_coupling",
    "mean_contacts",
    "single_contact_fraction",
    "strength_over_jmax",
    "chain_edges",
)
"""Legacy deployable aggregate allowlist, retained only for corpus migration."""


ArrayLike = np.ndarray | Tensor


def _numpy(array: ArrayLike) -> np.ndarray:
    if isinstance(array, Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _validate_feature_matrix(name: str, array: ArrayLike, width: int) -> None:
    value = _numpy(array)
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} feature width must be {width}, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} features must be finite after value/knownness packing")
    scalar_count = width // 2
    knownness = value[:, scalar_count:]
    if not np.logical_or(knownness == 0.0, knownness == 1.0).all():
        raise ValueError(f"{name} knownness channels must be binary")
    if not np.all(value[:, :scalar_count][knownness == 0.0] == 0.0):
        raise ValueError(f"{name} missing values must be zero after transformation")


def _validate_index(
    name: str,
    index: ArrayLike,
    source_count: int,
    destination_count: int,
) -> None:
    value = _numpy(index)
    if value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(f"{name} must have shape (2, E)")
    if value.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer endpoints")
    if value.size == 0:
        return
    if value[0].min() < 0 or value[0].max() >= source_count:
        raise ValueError(f"{name} source endpoint is out of range")
    if value[1].min() < 0 or value[1].max() >= destination_count:
        label = "claim endpoint" if name == "index_claims" else f"{name} destination endpoint"
        raise ValueError(f"{label} is out of range")


@dataclass(frozen=True)
class StrengthGraphInput:
    """One fully compiled, deployable program graph at one registered strength.

    Scalar matrices use Appendix A value/knownness packing.  Edge indices are directed;
    claims are stored as ``(logical, hardware)``.  No generic metadata mapping is accepted,
    which makes the neural input allowlist explicit and mechanically inspectable.
    """

    logical: ArrayLike
    hardware: ArrayLike
    logical_edges: ArrayLike
    hardware_edges: ArrayLike
    claims: ArrayLike
    globals_: ArrayLike
    index_logical_edges: ArrayLike
    index_hardware_edges: ArrayLike
    index_claims: ArrayLike
    strength_index: int
    strength: float
    scale: float
    normalizer_digest: str = "selector-normalizer-unit-v1"

    def validate(self) -> StrengthGraphInput:
        _validate_feature_matrix("logical", self.logical, 2 * N_LOGICAL)
        _validate_feature_matrix("hardware", self.hardware, 2 * N_HARDWARE)
        _validate_feature_matrix("logical edge", self.logical_edges, 2 * N_LOGICAL_EDGE)
        _validate_feature_matrix("hardware edge", self.hardware_edges, 2 * N_HARDWARE_EDGE)
        _validate_feature_matrix("claim", self.claims, 2 * N_CLAIM)
        global_array = _numpy(self.globals_)
        if global_array.shape != (2 * N_GLOBAL,) or not np.isfinite(global_array).all():
            raise ValueError("global feature width must be 64 and all channels must be finite")
        global_knownness = global_array[N_GLOBAL:]
        if not np.logical_or(global_knownness == 0.0, global_knownness == 1.0).all():
            raise ValueError("global knownness channels must be binary")
        if not np.all(global_array[:N_GLOBAL][global_knownness == 0.0] == 0.0):
            raise ValueError("global missing values must be zero after transformation")
        logical_count = int(_numpy(self.logical).shape[0])
        hardware_count = int(_numpy(self.hardware).shape[0])
        if logical_count == 0 or hardware_count == 0:
            raise ValueError("selector graphs require real logical and hardware nodes")
        _validate_index(
            "index_logical_edges",
            self.index_logical_edges,
            logical_count,
            logical_count,
        )
        _validate_index(
            "index_hardware_edges",
            self.index_hardware_edges,
            hardware_count,
            hardware_count,
        )
        _validate_index("index_claims", self.index_claims, logical_count, hardware_count)
        if int(_numpy(self.logical_edges).shape[0]) != int(
            _numpy(self.index_logical_edges).shape[1]
        ):
            raise ValueError("logical edge feature/index counts differ")
        if int(_numpy(self.hardware_edges).shape[0]) != int(
            _numpy(self.index_hardware_edges).shape[1]
        ):
            raise ValueError("hardware edge feature/index counts differ")
        if int(_numpy(self.claims).shape[0]) != int(_numpy(self.index_claims).shape[1]):
            raise ValueError("claim feature/index counts differ")
        if (
            isinstance(self.strength_index, bool)
            or not isinstance(self.strength_index, int)
            or not 0 <= self.strength_index < N_STRENGTHS
        ):
            raise ValueError("strength_index must be an integer in [0, 4)")
        if not math.isfinite(float(self.strength)) or float(self.strength) <= 0.0:
            raise ValueError("strength must be positive and finite")
        if not math.isfinite(float(self.scale)) or not 0.0 < float(self.scale) <= 1.0:
            raise ValueError("program scale must be finite and in (0, 1]")
        if not isinstance(self.normalizer_digest, str) or not self.normalizer_digest:
            raise ValueError("normalizer_digest must be a non-empty registered identifier")
        return self


def feature_vector(features: Mapping[str, float], strength_index: int) -> np.ndarray:
    """Legacy aggregate features plus strength one-hot, with an exact allowlist."""

    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise ValueError(f"missing deployable strength features: {missing}")
    if not 0 <= strength_index < N_STRENGTHS:
        raise ValueError("strength_index must be in [0, 4)")
    base = [float(features[name]) for name in FEATURE_ORDER]
    if not np.isfinite(base).all():
        raise ValueError("deployable strength features must be finite")
    onehot = [1.0 if k == strength_index else 0.0 for k in range(N_STRENGTHS)]
    return np.asarray(base + onehot, dtype=np.float64)


def _mlp(widths: Sequence[int], *, layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2:
            layers.append(nn.SiLU())
    if layer_norm:
        layers.append(nn.LayerNorm(widths[-1]))
    return nn.Sequential(*layers)


def _node_projector(input_width: int) -> nn.Sequential:
    """Appendix-A node projector, including its activation before normalization."""

    return nn.Sequential(
        nn.Linear(input_width, WIDTH),
        nn.SiLU(),
        nn.LayerNorm(WIDTH),
    )


def _segment_mean_max(messages: Tensor, destinations: Tensor, count: int) -> Tensor:
    if messages.numel() == 0:
        return messages.new_zeros((count, 2 * WIDTH))
    sums = messages.new_zeros((count, WIDTH))
    sums.index_add_(0, destinations, messages)
    degrees = messages.new_zeros((count, 1))
    degrees.index_add_(0, destinations, messages.new_ones((destinations.shape[0], 1)))
    means = sums / degrees.clamp_min(1.0)
    maxima = messages.new_full((count, WIDTH), float("-inf"))
    maxima.scatter_reduce_(
        0,
        destinations[:, None].expand(-1, WIDTH),
        messages,
        reduce="amax",
        include_self=True,
    )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return torch.cat((means, maxima), dim=-1)


def _pool_mean_max(nodes: Tensor) -> Tensor:
    if nodes.shape[0] == 0:
        return nodes.new_zeros(2 * WIDTH)
    return torch.cat((nodes.mean(dim=0), nodes.max(dim=0).values), dim=-1)


class _LocalBlock(nn.Module):
    """One Eq. (6) local message-passing block."""

    def __init__(self) -> None:
        super().__init__()
        self.message = _mlp((2 * WIDTH + EDGE_WIDTH, WIDTH, WIDTH))
        self.update = _mlp((3 * WIDTH, 2 * WIDTH, WIDTH))
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_attributes: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            aggregate = nodes.new_zeros((nodes.shape[0], 2 * WIDTH))
        else:
            source, destination = edge_index
            messages = self.message(
                torch.cat((nodes[source], nodes[destination], edge_attributes), dim=-1)
            )
            aggregate = _segment_mean_max(messages, destination, nodes.shape[0])
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), dim=-1)))


class _OwnershipRelation(nn.Module):
    """One direction of Eq. (7), including its no-neighbour degree gate."""

    def __init__(self) -> None:
        super().__init__()
        self.message = _mlp((2 * WIDTH + EDGE_WIDTH, WIDTH, WIDTH))
        self.update = _mlp((3 * WIDTH, 2 * WIDTH, WIDTH))

    def forward(
        self,
        destination_nodes: Tensor,
        source_nodes: Tensor,
        edge_index: Tensor,
        edge_attributes: Tensor,
    ) -> Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(destination_nodes)
        source, destination = edge_index
        messages = self.message(
            torch.cat(
                (source_nodes[source], destination_nodes[destination], edge_attributes),
                dim=-1,
            )
        )
        aggregate = _segment_mean_max(messages, destination, destination_nodes.shape[0])
        degree = destination_nodes.new_zeros((destination_nodes.shape[0], 1))
        degree.index_add_(
            0,
            destination,
            destination_nodes.new_ones((destination.shape[0], 1)),
        )
        update = self.update(torch.cat((destination_nodes, aggregate), dim=-1))
        return (degree > 0).to(update.dtype) * update


class _OwnershipFusionBlock(nn.Module):
    """Two-direction ownership-only fusion for the independent selector."""

    def __init__(self) -> None:
        super().__init__()
        self.logical_to_hardware = _OwnershipRelation()
        self.hardware_to_logical = _OwnershipRelation()
        self.logical_norm = nn.LayerNorm(WIDTH)
        self.hardware_norm = nn.LayerNorm(WIDTH)

    def forward(
        self,
        logical: Tensor,
        hardware: Tensor,
        claims: Tensor,
        claim_attributes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Claims are stored logical -> hardware.  Both directions read the same pre-block
        # node states and are therefore synchronous.
        hardware_update = self.logical_to_hardware(
            hardware,
            logical,
            claims,
            claim_attributes,
        )
        reverse = torch.flip(claims, dims=(0,)) if claims.numel() else claims
        logical_update = self.hardware_to_logical(
            logical,
            hardware,
            reverse,
            claim_attributes,
        )
        return (
            self.logical_norm(logical + logical_update),
            self.hardware_norm(hardware + hardware_update),
        )


def _mask_packed(tensor: Tensor, scalar_count: int, masked_slots: Sequence[int]) -> Tensor:
    if not masked_slots:
        return tensor
    result = tensor.clone()
    columns = tuple(masked_slots) + tuple(scalar_count + slot for slot in masked_slots)
    result[..., list(columns)] = 0.0
    return result


class StrengthSelectorModel(nn.Module):
    """IF-Q3-S0 graph program predictor, independently trained then frozen.

    The trunk is three separate logical/hardware local blocks followed by two ownership-only
    fusion blocks.  Four program strengths are four calls through these shared weights.  The
    exact response head is ``582 -> 128 -> 64 -> 1``; ``forward`` applies sigmoid.
    """

    FORMAT = "isingfold.strength-selector.graph.v2"
    VERSION = "if-q3-s0-graph-selector-2"

    def __init__(
        self,
        *,
        strength_transform_scale: float = 1.0,
        coefficient_transform_scale: float = 1.0,
        normalizer_digest: str = "selector-normalizer-unit-v1",
    ) -> None:
        super().__init__()
        if not math.isfinite(strength_transform_scale) or strength_transform_scale <= 0.0:
            raise ValueError("strength_transform_scale must be positive and finite")
        if not math.isfinite(coefficient_transform_scale) or coefficient_transform_scale <= 0.0:
            raise ValueError("coefficient_transform_scale must be positive and finite")
        if not isinstance(normalizer_digest, str) or not normalizer_digest:
            raise ValueError("normalizer_digest must be a non-empty registered identifier")
        self.width = WIDTH
        self.version = self.VERSION
        self.normalizer_digest = normalizer_digest

        self.logical_projector = _node_projector(2 * N_LOGICAL)
        self.hardware_projector = _node_projector(2 * N_HARDWARE)
        self.logical_edge_projector = _mlp((2 * N_LOGICAL_EDGE, EDGE_WIDTH))
        self.hardware_edge_projector = _mlp((2 * N_HARDWARE_EDGE, EDGE_WIDTH))
        self.claim_projector = _mlp((2 * N_CLAIM, EDGE_WIDTH))
        self.global_projector = _mlp((2 * N_GLOBAL, 64))

        self.local_logical = nn.ModuleList(_LocalBlock() for _ in range(3))
        self.local_hardware = nn.ModuleList(_LocalBlock() for _ in range(3))
        self.ownership_fusion = nn.ModuleList(_OwnershipFusionBlock() for _ in range(2))
        self.head = _mlp((582, 128, 64, 1))

        self.register_buffer(
            "strength_transform_scale",
            torch.tensor(float(strength_transform_scale), dtype=torch.float64),
        )
        self.register_buffer(
            "coefficient_transform_scale",
            torch.tensor(float(coefficient_transform_scale), dtype=torch.float64),
        )
        self.register_buffer("calibration_temperature", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("calibration_bias", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("_frozen", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_graph_fitted", torch.tensor(False, dtype=torch.bool))

        # Migration-only aggregate selector.  It is never consulted by graph forward calls.
        legacy_width = len(FEATURE_ORDER) + N_STRENGTHS
        self.register_buffer("legacy_weights", torch.zeros(legacy_width, dtype=torch.float64))
        self.register_buffer("legacy_bias", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("legacy_mean", torch.zeros(legacy_width, dtype=torch.float64))
        self.register_buffer("legacy_std", torch.ones(legacy_width, dtype=torch.float64))
        self.register_buffer("legacy_available", torch.tensor(False, dtype=torch.bool))

    @classmethod
    def untrained(cls) -> StrengthSelectorModel:
        """Return a trainable graph selector with no aggregate compatibility fit."""

        return cls()

    @property
    def frozen(self) -> bool:
        return bool(self._frozen.item())

    @property
    def deployment_ready(self) -> bool:
        """Whether this artifact was trained with graph records rather than legacy summaries."""

        return (
            self.frozen
            and bool(self._graph_fitted.item())
            and not bool(self.legacy_available.item())
        )

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _tensor(self, value: ArrayLike, *, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(_numpy(value)), dtype=dtype, device=self._device()
        )

    def _raw_logit(self, item: StrengthGraphInput) -> Tensor:
        item.validate()
        if item.normalizer_digest != self.normalizer_digest:
            raise ValueError("program graph normalizer does not match the frozen selector artifact")
        logical = self._tensor(item.logical)
        hardware = self._tensor(item.hardware)
        logical_edges = self._tensor(item.logical_edges)
        hardware_edges = self._tensor(item.hardware_edges)
        claims_features = self._tensor(item.claims)
        globals_ = self._tensor(item.globals_)
        logical_index = self._tensor(item.index_logical_edges, dtype=torch.long)
        hardware_index = self._tensor(item.index_hardware_edges, dtype=torch.long)
        claims_index = self._tensor(item.index_claims, dtype=torch.long)

        logical = _mask_packed(logical, N_LOGICAL, MASKED_LOGICAL_SLOTS)
        hardware = _mask_packed(hardware, N_HARDWARE, MASKED_HARDWARE_SLOTS)
        logical_edges = _mask_packed(
            logical_edges,
            N_LOGICAL_EDGE,
            MASKED_LOGICAL_EDGE_SLOTS,
        )
        claims_features = _mask_packed(claims_features, N_CLAIM, MASKED_CLAIM_SLOTS)
        masked_global = tuple(slot for slot in range(N_GLOBAL) if slot not in ALLOWED_GLOBAL_SLOTS)
        globals_ = _mask_packed(globals_, N_GLOBAL, masked_global)

        logical = self.logical_projector(logical)
        hardware = self.hardware_projector(hardware)
        logical_edge_attributes = self.logical_edge_projector(logical_edges)
        hardware_edge_attributes = self.hardware_edge_projector(hardware_edges)
        claim_attributes = self.claim_projector(claims_features)
        for logical_block, hardware_block in zip(
            self.local_logical,
            self.local_hardware,
            strict=True,
        ):
            logical = logical_block(logical, logical_index, logical_edge_attributes)
            hardware = hardware_block(hardware, hardware_index, hardware_edge_attributes)
        for fusion in self.ownership_fusion:
            logical, hardware = fusion(logical, hardware, claims_index, claim_attributes)

        graph_context = torch.cat(
            (
                _pool_mean_max(logical),
                _pool_mean_max(hardware),
                self.global_projector(globals_),
            ),
            dim=-1,
        )
        onehot = torch_f.one_hot(
            torch.tensor(item.strength_index, device=self._device()),
            num_classes=N_STRENGTHS,
        ).to(dtype=graph_context.dtype)
        transformed_strength = math.copysign(
            math.log1p(abs(float(item.strength)) / float(self.strength_transform_scale.item())),
            float(item.strength),
        )
        condition = graph_context.new_tensor((transformed_strength, float(item.scale)))
        response_input = torch.cat((graph_context, onehot, condition), dim=-1)
        if response_input.shape[0] != 582:
            raise AssertionError(f"selector head expected width 582, got {response_input.shape[0]}")
        return self.head(response_input).squeeze(-1)

    def raw_logits(self, inputs: Sequence[StrengthGraphInput]) -> Tensor:
        """Uncalibrated shared-network logits for graph selector pretraining."""

        if not inputs:
            return next(self.parameters()).new_zeros((0,))
        return torch.stack(tuple(self._raw_logit(item) for item in inputs))

    def logits(self, inputs: Sequence[StrengthGraphInput]) -> Tensor:
        """Calibrated logits; calibration parameters are frozen artifact state."""

        raw = self.raw_logits(inputs)
        temperature = self.calibration_temperature.to(dtype=raw.dtype).clamp_min(1e-6)
        bias = self.calibration_bias.to(dtype=raw.dtype)
        return raw / temperature + bias

    def forward(self, inputs: Sequence[StrengthGraphInput]) -> Tensor:
        return torch.sigmoid(self.logits(inputs))

    def select_embedding(self, inputs: Sequence[StrengthGraphInput]) -> int:
        """Select among exactly four compiled programs; ``argmax`` breaks ties by index."""

        if len(inputs) != N_STRENGTHS:
            raise ValueError("IF-Q3-S0 selection requires exactly four program graphs")
        if tuple(item.strength_index for item in inputs) != tuple(range(N_STRENGTHS)):
            raise ValueError("program graphs must be ordered by strength index 0, 1, 2, 3")
        with torch.no_grad():
            return int(torch.argmax(self(inputs)).item())

    def predict(self, features: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Run the aggregate-only migration model for historical corpora.

        This method intentionally cannot claim graph-selector readiness.  New callers must
        tensorize complete program graphs and call :meth:`forward` or
        :meth:`select_embedding`.
        """

        if not bool(self.legacy_available.item()):
            raise RuntimeError("no legacy aggregate fit; use complete graph program inputs")
        if len(features) != N_STRENGTHS:
            raise ValueError("IF-Q3-S0 requires exactly four feature records")
        rows = np.stack([feature_vector(value, index) for index, value in enumerate(features)])
        mean = self.legacy_mean.detach().cpu().numpy()
        std = self.legacy_std.detach().cpu().numpy()
        weights = self.legacy_weights.detach().cpu().numpy()
        standardized = (rows - mean) / np.where(std > 1e-9, std, 1.0)
        logits = standardized @ weights + float(self.legacy_bias.item())
        # Stable sigmoid avoids overflow while retaining exact tie behavior.
        return np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-np.minimum(logits, 709.0))),
            np.exp(np.maximum(logits, -745.0)) / (1.0 + np.exp(np.maximum(logits, -745.0))),
        )

    def select(
        self,
        programs: Sequence[object],
        features: Sequence[Mapping[str, float]],
    ) -> int:
        """Compatibility dispatch while the environment migrates to ``select_embedding``."""

        if len(programs) == N_STRENGTHS and all(
            isinstance(program, StrengthGraphInput) for program in programs
        ):
            return self.select_embedding(programs)  # type: ignore[arg-type]
        scores = self.predict(features)
        return int(np.argmax(scores))

    def fit_calibration(
        self,
        inputs: Sequence[StrengthGraphInput],
        hits: Sequence[int],
        reads: Sequence[int],
        *,
        max_iter: int = 64,
    ) -> None:
        """Fit scalar temperature/bias on a reserved training-side calibration partition."""

        if self.frozen:
            raise RuntimeError("cannot calibrate a frozen selector")
        if not inputs:
            raise ValueError("calibration requires count-complete graph inputs")
        with torch.no_grad():
            raw = self.raw_logits(inputs).detach().to(dtype=torch.float64)
        target = _count_targets(hits, reads, device=raw.device, dtype=raw.dtype)
        log_temperature = torch.zeros((), dtype=raw.dtype, device=raw.device, requires_grad=True)
        bias = torch.zeros((), dtype=raw.dtype, device=raw.device, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            (log_temperature, bias),
            lr=0.25,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            calibrated = raw / log_temperature.exp().clamp_min(1e-6) + bias
            loss = torch_f.binary_cross_entropy_with_logits(calibrated, target)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.calibration_temperature.copy_(log_temperature.exp().cpu())
            self.calibration_bias.copy_(bias.cpu())

    def freeze(self) -> StrengthSelectorModel:
        """Freeze trunk, head, normalizers and calibration before PPO."""

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._frozen.fill_(True)
        self.eval()
        return self

    def artifact_digest(self) -> str:
        """Content digest covering architecture, preprocessing, calibration and weights."""

        metadata = {
            "format": self.FORMAT,
            "version": self.version,
            "width": self.width,
            "node_projector": NODE_PROJECTOR_ARCHITECTURE,
            "local_blocks": len(self.local_logical),
            "fusion_blocks": len(self.ownership_fusion),
            "masked_logical": MASKED_LOGICAL_SLOTS,
            "masked_hardware": MASKED_HARDWARE_SLOTS,
            "masked_logical_edge": MASKED_LOGICAL_EDGE_SLOTS,
            "masked_claim": MASKED_CLAIM_SLOTS,
            "allowed_global": ALLOWED_GLOBAL_SLOTS,
            "normalizer_digest": self.normalizer_digest,
            "coefficient_transform_scale": float(self.coefficient_transform_scale.item()),
        }
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for name, value in sorted(self.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically persist a frozen selector and its verified content digest."""

        if not self.frozen:
            raise RuntimeError("refusing to publish an unfrozen selector")
        if self.version != self.VERSION:
            raise RuntimeError("refusing to publish an unsupported selector architecture version")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT,
            "version": self.version,
            "normalizer_digest": self.normalizer_digest,
            "state_dict": {name: value.detach().cpu() for name, value in self.state_dict().items()},
            "digest": self.artifact_digest(),
            "frozen": True,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary_name)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
    ) -> StrengthSelectorModel:
        """Load graph artifacts, with read-only support for historical NumPy selectors."""

        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except (RuntimeError, ValueError, TypeError):
            data = np.load(path, allow_pickle=False)
            model = cls()
            model._install_legacy(
                weights=data["weights"],
                bias=float(data["bias"][0]),
                mean=data["mean"],
                std=data["std"],
            )
            model.version = str(data["version"][0]) if "version" in data else "legacy-loaded"
            return model.freeze()
        if not isinstance(payload, dict) or payload.get("format") != cls.FORMAT:
            raise ValueError("unsupported strength-selector artifact format")
        if payload.get("version") != cls.VERSION:
            raise ValueError("unsupported strength-selector artifact version")
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("strength-selector artifact has no state_dict")
        transform_scale = float(state["strength_transform_scale"].item())
        coefficient_scale = float(state["coefficient_transform_scale"].item())
        normalizer_digest = payload.get("normalizer_digest")
        if not isinstance(normalizer_digest, str) or not normalizer_digest:
            raise ValueError("strength-selector artifact has no normalizer digest")
        model = cls(
            strength_transform_scale=transform_scale,
            coefficient_transform_scale=coefficient_scale,
            normalizer_digest=normalizer_digest,
        )
        model.load_state_dict(state, strict=True)
        if payload.get("frozen") is not True:
            raise ValueError("refusing to load a published selector that is not frozen")
        model.freeze()
        expected = payload.get("digest")
        if not isinstance(expected, str) or model.artifact_digest() != expected:
            raise ValueError("strength-selector artifact digest mismatch")
        return model

    def _install_legacy(
        self,
        *,
        weights: np.ndarray,
        bias: float,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        expected = len(FEATURE_ORDER) + N_STRENGTHS
        arrays = tuple(np.asarray(value, dtype=np.float64) for value in (weights, mean, std))
        if any(value.shape != (expected,) for value in arrays):
            raise ValueError("legacy selector arrays have an incompatible feature width")
        if not all(np.isfinite(value).all() for value in arrays) or not math.isfinite(bias):
            raise ValueError("legacy selector contains non-finite parameters")
        with torch.no_grad():
            self.legacy_weights.copy_(torch.from_numpy(arrays[0]))
            self.legacy_bias.fill_(bias)
            self.legacy_mean.copy_(torch.from_numpy(arrays[1]))
            self.legacy_std.copy_(torch.from_numpy(arrays[2]))
            self.legacy_available.fill_(True)


@dataclass(frozen=True)
class StrengthRecord:
    """Four count-complete program labels, optionally with their deployable graph inputs."""

    features: tuple[Mapping[str, float], ...]
    hits: tuple[int, ...]
    reads: tuple[int, ...]
    lineage: str = ""
    graph_inputs: tuple[StrengthGraphInput, ...] | None = None
    task_id: str = ""
    source_record_id: str = ""
    selector_partition: str = ""

    def __post_init__(self) -> None:
        if not (len(self.features) == len(self.hits) == len(self.reads) == N_STRENGTHS):
            raise ValueError("IF-Q3-S0 requires exactly four complete program/count records")
        for index, (hits, reads) in enumerate(zip(self.hits, self.reads, strict=True)):
            if isinstance(hits, bool) or isinstance(reads, bool):
                raise ValueError(f"count pair {index} must contain integers")
            if not isinstance(hits, int) or not isinstance(reads, int):
                raise ValueError(f"count pair {index} must contain integers")
            if reads <= 0 or not 0 <= hits <= reads:
                raise ValueError(f"invalid count pair {index}: ({hits}, {reads})")
        for index, features in enumerate(self.features):
            feature_vector(features, index)
        if self.graph_inputs is not None:
            if len(self.graph_inputs) != N_STRENGTHS:
                raise ValueError("graph selector records require exactly four program graphs")
            for index, graph in enumerate(self.graph_inputs):
                graph.validate()
                if graph.strength_index != index:
                    raise ValueError("program graphs must follow registry strength order")
        for value, name in (
            (self.task_id, "task_id"),
            (self.source_record_id, "source_record_id"),
        ):
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        if self.selector_partition not in {
            "",
            "train",
            "calibration",
            "audit_val",
            "audit_test",
        }:
            raise ValueError("selector_partition is not registered")


def _count_targets(
    hits: Sequence[int] | Tensor,
    reads: Sequence[int] | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    hits_tensor = torch.as_tensor(hits, device=device)
    reads_tensor = torch.as_tensor(reads, device=device)
    if hits_tensor.shape != reads_tensor.shape:
        raise ValueError("hits and reads must have identical shapes")
    if hits_tensor.dtype == torch.bool or reads_tensor.dtype == torch.bool:
        raise ValueError("hits and reads must be integer counts")
    if hits_tensor.is_floating_point() or reads_tensor.is_floating_point():
        raise ValueError("hits and reads must be integer counts")
    if (
        torch.any(reads_tensor <= 0)
        or torch.any(hits_tensor < 0)
        or torch.any(hits_tensor > reads_tensor)
    ):
        raise ValueError("count pairs require 0 <= hits <= reads and reads > 0")
    return hits_tensor.to(dtype=dtype) / reads_tensor.to(dtype=dtype)


def binomial_bce_with_logits(
    logits: Tensor,
    hits: Sequence[int] | Tensor,
    reads: Sequence[int] | Tensor,
) -> Tensor:
    """Stable normalized Eq. (18), with equal weight per program-strength pair."""

    targets = _count_targets(hits, reads, device=logits.device, dtype=logits.dtype)
    if targets.shape != logits.shape:
        raise ValueError("one count pair is required for each selector logit")
    return torch_f.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def _fit_legacy_selector(
    records: Sequence[StrengthRecord],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> StrengthSelectorModel:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for record in records:
        for index, (features, hits, reads) in enumerate(
            zip(record.features, record.hits, record.reads, strict=True)
        ):
            rows.append(feature_vector(features, index))
            targets.append(hits / reads)
    if not rows:
        raise ValueError("no count-complete records to fit the selector")
    x = np.stack(rows)
    y = np.asarray(targets)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    standardized = (x - mean) / std
    weights = np.zeros(standardized.shape[1])
    bias = 0.0
    for _ in range(epochs):
        logits = standardized @ weights + bias
        probabilities = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-np.minimum(logits, 709.0))),
            np.exp(np.maximum(logits, -745.0)) / (1.0 + np.exp(np.maximum(logits, -745.0))),
        )
        weights -= learning_rate * (
            standardized.T @ (probabilities - y) / standardized.shape[0] + l2 * weights
        )
        bias -= learning_rate * float((probabilities - y).mean())
    model = StrengthSelectorModel()
    model._install_legacy(weights=weights, bias=bias, mean=mean, std=std)
    model.version = "if-q3-s0-legacy-aggregate-selector-1"
    return model.freeze()


def fit_selector(
    records: Sequence[StrengthRecord],
    *,
    calibration_records: Sequence[StrengthRecord] = (),
    epochs: int = 400,
    learning_rate: float = 0.1,
    graph_learning_rate: float = 3e-4,
    l2: float = 1e-3,
    seed: int = 0,
    strength_transform_scale: float = 1.0,
    coefficient_transform_scale: float = 1.0,
    normalizer_digest: str = "selector-normalizer-unit-v1",
    device: str | torch.device | None = None,
    graph_minibatch: int = 32,
    gradient_norm: float = 1.0,
    history: list[dict[str, object]] | None = None,
) -> StrengthSelectorModel:
    """Fit Eq. (18), choosing graph mode only for count-complete graph records.

    Aggregate-only historical records use a labelled compatibility artifact.  Mixing record
    modes is rejected.  Calibration lineages must be disjoint from fitting lineages.
    """

    if not records:
        raise ValueError("no count-complete records to fit the selector")
    if any(record.selector_partition not in {"", "train"} for record in records):
        raise ValueError("selector fitting refuses calibration or audit records")
    if any(record.selector_partition not in {"", "calibration"} for record in calibration_records):
        raise ValueError("selector calibration refuses fitting or audit records")
    if isinstance(graph_minibatch, bool) or graph_minibatch <= 0:
        raise ValueError("graph minibatch must be a positive number of records")
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise ValueError("selector gradient norm must be positive and finite")
    target_device = torch.device("cpu" if device is None else device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA selector fitting was requested but CUDA is unavailable")
    if target_device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS selector fitting was requested but MPS is unavailable")
    graph_mode = [record.graph_inputs is not None for record in records]
    if any(graph_mode) and not all(graph_mode):
        raise ValueError("cannot mix graph-complete and aggregate-only selector records")
    if not all(graph_mode):
        if calibration_records:
            raise ValueError("legacy aggregate selector does not support graph calibration")
        return _fit_legacy_selector(
            records,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        ).to(target_device)

    if any(record.graph_inputs is None for record in calibration_records):
        raise ValueError("calibration records must include complete graph inputs")
    train_lineages = {record.lineage for record in records if record.lineage}
    calibration_lineages = {record.lineage for record in calibration_records if record.lineage}
    overlap = train_lineages & calibration_lineages
    if overlap:
        raise ValueError(f"selector fit/calibration lineage overlap: {sorted(overlap)[:8]}")

    torch.manual_seed(seed)
    model = StrengthSelectorModel(
        strength_transform_scale=strength_transform_scale,
        coefficient_transform_scale=coefficient_transform_scale,
        normalizer_digest=normalizer_digest,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=graph_learning_rate,
        weight_decay=l2,
    )
    model.train()
    for epoch in range(epochs):
        epoch_seed = int.from_bytes(
            hashlib.sha256(
                json.dumps(
                    {
                        "domain": "if-q3-s0-graph-record-order",
                        "seed": seed,
                        "epoch": epoch,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        order = np.random.default_rng(epoch_seed).permutation(len(records))
        weighted_loss = 0.0
        pairs_seen = 0
        optimizer_steps = 0
        max_preclip_gradient_norm = 0.0
        for start in range(0, len(order), graph_minibatch):
            batch_indices = order[start : start + graph_minibatch]
            batch_records = tuple(records[int(index)] for index in batch_indices)
            batch_inputs = tuple(
                graph for record in batch_records for graph in record.graph_inputs or ()
            )
            batch_hits = tuple(hit for record in batch_records for hit in record.hits)
            batch_reads = tuple(read for record in batch_records for read in record.reads)
            if not batch_inputs or not (len(batch_inputs) == len(batch_hits) == len(batch_reads)):
                raise RuntimeError("selector minibatch lost a graph/count program pair")
            optimizer.zero_grad(set_to_none=True)
            logits = model.raw_logits(batch_inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite selector logits")
            loss = binomial_bce_with_logits(logits, batch_hits, batch_reads)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite selector binomial loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError("non-finite selector gradient")
            observed_gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_norm
            )
            if not torch.isfinite(observed_gradient_norm):
                raise RuntimeError("non-finite selector gradient norm")
            optimizer.step()
            pair_count = len(batch_inputs)
            weighted_loss += float(loss.detach().item()) * pair_count
            pairs_seen += pair_count
            optimizer_steps += 1
            max_preclip_gradient_norm = max(
                max_preclip_gradient_norm, float(observed_gradient_norm.detach().item())
            )
        expected_pairs = len(records) * N_STRENGTHS
        if pairs_seen != expected_pairs or len(order) != len(records):
            raise RuntimeError("selector epoch did not consume every program/count pair once")
        if history is not None:
            history.append(
                {
                    "epoch": epoch + 1,
                    "graph_minibatch_records": graph_minibatch,
                    "records_seen": len(records),
                    "program_pairs_seen": pairs_seen,
                    "optimizer_steps": optimizer_steps,
                    "mean_binomial_bce": weighted_loss / pairs_seen,
                    "max_observed_preclip_gradient_norm": max_preclip_gradient_norm,
                    "record_order_digest": hashlib.sha256(
                        json.dumps(
                            [int(index) for index in order],
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
    model._graph_fitted.fill_(True)

    if calibration_records:
        calibration_inputs = tuple(
            graph for record in calibration_records for graph in record.graph_inputs or ()
        )
        calibration_hits = tuple(hit for record in calibration_records for hit in record.hits)
        calibration_reads = tuple(read for record in calibration_records for read in record.reads)
        model.fit_calibration(calibration_inputs, calibration_hits, calibration_reads)
    return model.freeze()


def selector_regret(
    model: StrengthSelectorModel,
    records: Sequence[StrengthRecord],
) -> dict[str, float]:
    """Training-side descriptive metric; not the sealed post-freeze audit protocol."""

    if not records:
        raise ValueError("selector regret requires at least one record")
    chosen: list[float] = []
    oracle: list[float] = []
    fixed: list[float] = []
    uniform_random_expected: list[float] = []
    for record in records:
        rates = np.asarray(
            [hits / reads for hits, reads in zip(record.hits, record.reads, strict=True)]
        )
        if record.graph_inputs is not None:
            pick = model.select_embedding(record.graph_inputs)
        else:
            pick = model.select((), record.features)
        chosen.append(float(rates[pick]))
        oracle.append(float(rates.max()))
        fixed.append(float(rates[1]))
        uniform_random_expected.append(float(rates.mean()))
    chosen_array = np.asarray(chosen)
    oracle_array = np.asarray(oracle)
    return {
        "n": float(len(records)),
        "selected_mean": float(chosen_array.mean()),
        "oracle_mean": float(oracle_array.mean()),
        "fixed_f2_mean": float(np.mean(fixed)),
        "random_mean": float(np.mean(uniform_random_expected)),
        "regret_vs_oracle": float((oracle_array - chosen_array).mean()),
        "regret_fixed_vs_oracle": float((oracle_array - np.asarray(fixed)).mean()),
        "top1_rate": float(np.mean(chosen_array >= oracle_array - 1e-12)),
    }

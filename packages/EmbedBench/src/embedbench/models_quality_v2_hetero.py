"""Label-free heterogeneous backbone for the Quality Value V2 heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from embedbench.hamiltonian_context import HAMILTONIAN_CONTEXT_DIMENSION
from embedbench.models_hetero import HeteroModelInput, build_hetero_candidate_encoder
from embedbench.models_quality_v2 import (
    ROBUSTNESS_NAMES,
    QualityV2Config,
    QualityV2Output,
)


class HeteroQualityValueV2(nn.Module):
    """Attach quality and mechanism heads to the heterogeneous graph encoder.

    ``forward`` accepts only ``HeteroModelInput``. Training labels, label-fidelity stages,
    and baseline/oracle indices remain outside the neural boundary.
    """

    def __init__(self, config: QualityV2Config) -> None:
        super().__init__()
        if config.arch != "hetero":
            raise ValueError("heterogeneous Quality V2 requires arch='hetero'")
        if config.hidden <= 0 or config.layers <= 0 or config.heads <= 0:
            raise ValueError("hidden, layers, and heads must be positive")
        if config.hidden % config.heads:
            raise ValueError("hidden must be divisible by heads for the heterogeneous backbone")
        if config.neighbour_feats:
            raise ValueError("neighbour_feats is not applicable to the heterogeneous backbone")
        if config.robustness_dim != len(ROBUSTNESS_NAMES):
            raise ValueError("robustness_dim must match the registered robustness target schema")

        self.config = config
        self.graph_encoder = build_hetero_candidate_encoder(
            config.hidden,
            config.layers,
            config.heads,
        )
        self.task_adapter = nn.Sequential(
            nn.Linear(self.graph_encoder.output_dim, config.hidden),
            nn.ReLU(),
            nn.LayerNorm(config.hidden),
        )
        self.problem_adapter = nn.Sequential(
            nn.Linear(HAMILTONIAN_CONTEXT_DIMENSION, config.hidden),
            nn.ReLU(),
            nn.LayerNorm(config.hidden),
        )
        self.candidate_context = nn.Sequential(
            nn.Linear(3 * config.hidden, config.hidden),
            nn.ReLU(),
            nn.LayerNorm(config.hidden),
        )
        self.quality_head = nn.Linear(config.hidden, 2)
        self.capacity_head = nn.Linear(config.hidden, 1)
        self.robustness_head = nn.Linear(config.hidden, config.robustness_dim)
        self.terminal_qubits_head = (
            nn.Linear(config.hidden, 1) if config.predict_terminal_qubits else None
        )

    def forward(self, model_input: HeteroModelInput) -> QualityV2Output:
        if not isinstance(model_input, HeteroModelInput):
            raise TypeError("HeteroQualityValueV2.forward requires HeteroModelInput")
        problem_context = torch.tensor(
            model_input.hamiltonian_context,
            dtype=self.problem_adapter[0].weight.dtype,
            device=self.problem_adapter[0].weight.device,
        )
        if problem_context.shape != (HAMILTONIAN_CONTEXT_DIMENSION,) or not bool(
            torch.isfinite(problem_context).all()
        ):
            raise ValueError("hamiltonian_context must be one finite registered feature vector")
        representation = self.graph_encoder(model_input)
        local = self.task_adapter(representation)
        state = local.mean(dim=0, keepdim=True).expand_as(local)
        problem = self.problem_adapter(problem_context).unsqueeze(0).expand_as(local)
        conditioned = self.candidate_context(torch.cat((local, state, problem), dim=-1))
        quality = self.quality_head(conditioned)
        terminal = (
            F.softplus(self.terminal_qubits_head(conditioned).squeeze(-1))
            if self.terminal_qubits_head is not None
            else None
        )
        return QualityV2Output(
            quality_logit=quality[:, 0],
            quality_log_concentration=quality[:, 1],
            future_capacity=torch.sigmoid(self.capacity_head(conditioned).squeeze(-1)),
            robustness=F.softplus(self.robustness_head(conditioned)),
            terminal_qubits=terminal,
        )


def build_quality_v2_hetero_model(
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    *,
    predict_terminal_qubits: bool = False,
    robustness_dim: int = len(ROBUSTNESS_NAMES),
) -> HeteroQualityValueV2:
    """Build a reconstructable heterogeneous V2 model with no feasibility head."""

    return HeteroQualityValueV2(
        QualityV2Config(
            arch="hetero",
            hidden=hidden,
            layers=layers,
            heads=heads,
            neighbour_feats=False,
            predict_terminal_qubits=predict_terminal_qubits,
            robustness_dim=robustness_dim,
        )
    )

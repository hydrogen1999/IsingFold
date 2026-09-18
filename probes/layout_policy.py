"""Contextual layout actor--critic with explicit, reproducible root support.

This is a candidate-set network, not a graph neural network. It pools the features
of all currently offered roots, then scores each root in that shared context. Its
value head sees the pooled state before an action is sampled. The sampler always
starts from an empty placement and never reads a witness or an initial embedding.

The legacy shortlist remains an ablation. ``all_free`` gives each unoccupied host
node support; it does not guarantee that every offered root admits completion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


DEFAULT_FEATURE_WIDTH = 35


def next_variable(task, chains):
    """Retain the original logical-variable order to isolate the root policy."""
    remaining = set(task.logical) - {v for v, chain in chains.items() if chain}
    if not remaining:
        return None
    return max(remaining, key=lambda v: (
        sum(bool(chains.get(u)) for u in task.logical.neighbors(v)),
        task.logical.degree(v), str(v),
    ))


def root_support(task, v, chains, strategy="all_free", rng=None):
    """Return an immutable ordered support without consulting a witness.

    ``legacy`` reproduces fast_layout's 48 touching / 24 spread shortlist,
    including its random single-root fallback when an RNG is provided. The
    ``all_free`` support is deterministic and independent of the RNG.
    """
    if strategy not in {"all_free", "legacy"}:
        raise ValueError("root support must be 'all_free' or 'legacy'")
    if v not in task.logical:
        raise ValueError("variable is not in the logical graph")
    occupied = {q for chain in chains.values() for q in chain}
    free = [q for q in sorted(task.host, key=str) if q not in occupied]
    if strategy == "all_free" or not free:
        return tuple(free)
    placed_nb = [u for u in task.logical.neighbors(v) if chains.get(u)]
    if placed_nb:
        touches = {}
        for u in placed_nb:
            for q in chains[u]:
                for r in task.host.neighbors(q):
                    if r not in occupied:
                        touches[r] = touches.get(r, 0) + 1
        roots = sorted(touches, key=lambda r: (-touches[r], str(r)))[:48]
    else:
        roots = free[::max(1, len(free) // 24)][:24]
    if not roots:
        # Match legacy's insertion-order free list for its random fallback.
        fallback = [q for q in task.host if q not in occupied]
        roots = [fallback[int(rng.integers(len(fallback)))]] if rng is not None else free[:1]
    return tuple(roots)


class LayoutActorCritic(nn.Module):
    """Permutation-equivariant candidate actor and invariant state-value baseline."""

    def __init__(self, width=64, in_dim=DEFAULT_FEATURE_WIDTH):
        super().__init__()
        if any(not np.isfinite(x) or x < 1 or float(x) != int(x) for x in (width, in_dim)):
            raise ValueError("width and in_dim must be finite positive integers")
        self.width, self.in_dim = int(width), int(in_dim)
        width, in_dim = self.width, self.in_dim
        self.encoder = nn.Sequential(nn.Linear(in_dim, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        # Mean and max retain both average competition and scarce/extreme options.
        # The count keeps the value informed when a support duplicates statistics.
        self.actor = nn.Sequential(nn.Linear(3 * width + 1, width), nn.SiLU(),
                                   nn.Linear(width, 1))
        self.critic = nn.Sequential(nn.Linear(2 * width + 1, width), nn.SiLU(),
                                    nn.Linear(width, 1))

    def _outputs(self, feats):
        if feats.ndim != 2 or feats.shape[0] < 1 or feats.shape[1] != self.in_dim:
            raise ValueError("features must be a nonempty [candidates, in_dim] tensor")
        encoded = self.encoder(feats)
        count = encoded.new_tensor([np.log1p(encoded.shape[0])])
        context = torch.cat((encoded.mean(0), encoded.max(0).values, count))
        logits = self.actor(torch.cat((encoded, context.expand(len(encoded), -1)), dim=1))
        return logits.squeeze(-1), self.critic(context).squeeze(-1)

    def forward(self, feats):
        return self._outputs(feats)[0]

    def distribution_value(self, feats, temperature=1.0):
        _check_temperature(temperature)
        logits, value = self._outputs(feats)
        return torch.distributions.Categorical(logits=logits / temperature), value


@dataclass(frozen=True)
class LayoutDecision:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor | None
    support_size: int
    roots: tuple
    action: int
    variable: Any
    features: torch.Tensor | None = None
    state_chains: dict | None = None


def _check_temperature(temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def sample_layout_v2(task, model, fc, temperature, rng, train=True,
                     support="all_free", capture_teacher=False):
    """Build roots from empty; retain differentiable rollout statistics for RL.

    Sampling uses only the supplied NumPy RNG. Validation cannot consume PyTorch's
    training random stream. ``capture_teacher`` records a state/features snapshot;
    it does not read or incorporate a teacher. Independent local Prioritiser models
    are supported, with ``value=None``, for architecture-controlled ablations.

    The declared feature-context budget is a hard root-count cap: a budget smaller
    than the number of variables returns a partial placement, never an over-budget
    layout. Feature contexts without a budget retain the legacy full-host limit.
    """
    _check_temperature(temperature)
    if support not in {"all_free", "legacy"}:
        raise ValueError("root support must be 'all_free' or 'legacy'")
    budget = float(getattr(fc, "budget", task.host.number_of_nodes()))
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("layout qubit budget must be finite and nonnegative")
    root_limit = min(task.logical.number_of_nodes(), task.host.number_of_nodes(), int(budget))
    chains, decisions = {}, []
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    while len(chains) < root_limit:
        v = next_variable(task, chains)
        if v is None:
            break
        roots = root_support(task, v, chains, support, rng)
        if not roots:
            break
        feats = torch.as_tensor(np.stack([fc.pair(v, [r], chains, "PLACE") for r in roots]),
                                dtype=dtype, device=device)
        if hasattr(model, "distribution_value"):
            dist, value = model.distribution_value(feats, temperature)
        else:
            dist = torch.distributions.Categorical(logits=model(feats) / temperature)
            value = None
        if train:
            probabilities = dist.probs.detach().cpu().numpy().astype(np.float64)
            probabilities /= probabilities.sum()
            action = int(rng.choice(len(roots), p=probabilities))
        else:
            action = int(dist.logits.argmax())
        decisions.append(LayoutDecision(
            log_prob=dist.log_prob(torch.tensor(action, device=device)),
            entropy=dist.entropy(), value=value, support_size=len(roots),
            roots=roots, action=action, variable=v,
            features=feats if capture_teacher else None,
            state_chains=dict(chains) if capture_teacher else None,
        ))
        chains[v] = frozenset({roots[action]})
    return chains, decisions


def witness_set_loss(logits, roots, witness_chain, *, training=False):
    """Training-only feasible-root set supervision on teacher-consistent prefixes.

    Rewards probability mass on any offered root belonging to a witness chain,
    without privileging a particular qubit or inserting absent roots. This is a
    feasibility teacher, not a quality, advantage or continuation-Q target. The
    caller must build prefixes inside disjoint witness chains; an arbitrary policy
    prefix may already block completion of that witness. No available teacher root
    returns None rather than manufacturing a target. Validation use is rejected.
    """
    if not training:
        raise ValueError("witness supervision is training-only")
    if logits.ndim != 1 or len(logits) != len(roots):
        raise ValueError("logits must match the ordered root support")
    witness_nodes = set(witness_chain)
    selected = [i for i, q in enumerate(roots) if q in witness_nodes]
    if not selected:
        return None
    indices = torch.as_tensor(selected, dtype=torch.long, device=logits.device)
    return torch.logsumexp(logits, 0) - torch.logsumexp(logits[indices], 0)

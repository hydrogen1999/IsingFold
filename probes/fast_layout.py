"""A layout sampler outside the environment: one root per variable in milliseconds.

The environment's PLACE decision costs tens of milliseconds of proposal generation and
validation per step; a layout of a hundred variables takes seconds, and the router that
completes it takes less than one. This places variables sequentially on the residual host
with the same features and the same prioritiser network, so the policy can propose
hundreds of layouts within a deadline and training sees hundreds of episodes a minute.

Order: the variable with the most placed logical neighbours first (ties by degree). Roots:
the free qubits adjacent to the placed neighbours' chains, or a spread of free qubits when
nothing is placed yet. The policy's scores over those roots give a softmax to sample from
or an argmax; log-probabilities are kept for REINFORCE.
"""
import numpy as np
import torch

from candidate_features import FeatureContext


def sample_layout(task, model, fc, temperature, rng, train=True, max_roots=48, spread=24):
    host, logical = task.host, task.logical
    chains = {}
    occupied = set()
    logps = []
    remaining = set(logical.nodes())
    degree = dict(logical.degree())
    while remaining:
        placed_nb = {v: sum(1 for u in logical.neighbors(v) if u in chains) for v in remaining}
        v = max(remaining, key=lambda x: (placed_nb[x], degree[x], str(x)))
        if placed_nb[v] > 0:
            touches = {}
            for u in logical.neighbors(v):
                for q in chains.get(u, ()):
                    for r in host.neighbors(q):
                        if r not in occupied:
                            touches[r] = touches.get(r, 0) + 1
            roots = sorted(touches, key=lambda r: (-touches[r], str(r)))[:max_roots]
        else:
            free = [q for q in sorted(host.nodes(), key=str) if q not in occupied]
            stride = max(1, len(free) // spread)
            roots = free[::stride][:spread]
        if not roots:
            free = [q for q in host.nodes() if q not in occupied]
            if not free:
                break
            roots = [free[rng.integers(0, len(free))]]
        feats = np.stack([fc.pair(v, [r], chains, "PLACE") for r in roots])
        scores = model(torch.as_tensor(feats)) / temperature
        if train:
            dist = torch.distributions.Categorical(logits=scores)
            j = int(dist.sample())
            logps.append(dist.log_prob(torch.tensor(j)))
        else:
            j = int(torch.argmax(scores))
        r = roots[j]
        chains[v] = frozenset({r})
        occupied.add(r)
        remaining.discard(v)
    return chains, logps

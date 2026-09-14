"""One deterministic, label-free native repair primitive.

This is the method-side boundary used by external data and evaluation tools.
It knows only graphs, an incumbent embedding, one repair neighborhood, and
hard search caps.  Dataset identities, quality labels, splits, and policies
remain outside IsingFold.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from lac_minorminer.components import (
    DefaultVertexSelector,
    GreedyAcceptancePolicy,
    NativeCandidateProvider,
    NoRouteCostProvider,
    StagnationControlPolicy,
)
from lac_minorminer.orchestrator import SearchOrchestrator
from lac_minorminer.scoring import ResourceScorer
from lac_minorminer.session import SearchSession
from lac_minorminer.validation import validate_embedding


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        relation = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {relation} integer")
    return value


def repair_once(
    logical: Any,
    host: Any,
    incumbent: Mapping[Hashable, Sequence[Hashable]],
    *,
    neighborhood: Sequence[Hashable],
    random_seed: int,
    max_candidates: int = 8,
    max_transitions: int = 200,
) -> dict[str, object]:
    """Reset ``neighborhood`` and repair from a fresh incumbent session once.

    The fixed classical orchestrator uses one try and no wall-clock timeout.
    A successful result contains a complete, independently validated embedding;
    an exhausted search returns ``chains=None`` with structural diagnostics.
    """

    if not isinstance(incumbent, Mapping):
        raise TypeError("incumbent must map logical variables to chains")
    random_seed = _integer(random_seed, "random_seed", minimum=0)
    max_candidates = _integer(max_candidates, "max_candidates", minimum=1)
    max_transitions = _integer(max_transitions, "max_transitions", minimum=0)

    probe = SearchSession(
        logical,
        host,
        random_seed=random_seed,
        max_candidates=max_candidates,
    )
    source = probe.normalized_source
    target = probe.normalized_target
    if set(incumbent) != set(source.labels):
        raise ValueError("incumbent must contain exactly one chain per logical variable")
    copied = {logical_id: tuple(incumbent[logical_id]) for logical_id in source.labels}
    report = validate_embedding(logical, host, copied)
    if not report.valid:
        raise ValueError(f"incumbent is not a valid embedding: {'; '.join(report.errors)}")

    selected = tuple(neighborhood)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("neighborhood must contain unique logical variables")
    try:
        logical_indices = tuple(source.index(logical_id) for logical_id in selected)
    except ValueError as error:
        raise ValueError("neighborhood contains an unknown logical variable") from error

    compact_chains = tuple(
        tuple(target.index(qubit) for qubit in copied[logical_id]) for logical_id in source.labels
    )
    session = SearchSession.from_chains(
        logical,
        host,
        compact_chains,
        random_seed=random_seed,
        max_candidates=max_candidates,
    )
    session.perturb(logical_indices)
    orchestrator = SearchOrchestrator(
        selector=DefaultVertexSelector(),
        route_cost_provider=NoRouteCostProvider(),
        candidate_provider=NativeCandidateProvider(),
        scorer=ResourceScorer(),
        acceptance_policy=GreedyAcceptancePolicy(),
        control_policy=StagnationControlPolicy(3),
        max_local_repairs=0,
    )
    diagnostics = orchestrator.run(
        session,
        random.Random(random_seed),
        random_seed=random_seed,
        tries=1,
        max_transitions=max_transitions,
        timeout=None,
    )
    best = diagnostics.best_incumbent
    if not diagnostics.success or best is None:
        return {
            "chains": None,
            "transitions": diagnostics.transitions,
            "reason": diagnostics.termination_reason.value,
        }

    chains = {
        logical_id: [
            target.labels[target_id] for target_id in sorted(best.chains[index])
        ]
        for index, logical_id in enumerate(source.labels)
    }
    verified = validate_embedding(logical, host, chains)
    if not verified.valid:
        raise RuntimeError(
            "native repair reported success but independent validation failed: "
            + "; ".join(verified.errors)
        )
    return {
        "chains": chains,
        "transitions": diagnostics.transitions,
        "reason": "success",
    }

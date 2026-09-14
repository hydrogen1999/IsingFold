"""Replaceable operation-level contracts and classical defaults."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ._options import integer_option
from .session import SearchSession


class VertexSelector(Protocol):
    def select(self, snapshot: Any, eligible: Sequence[int], rng: random.Random) -> int: ...


class RouteCostProvider(Protocol):
    def costs(self, snapshot: Any, logical_id: int, target_size: int) -> Sequence[float] | None: ...


class CandidateProvider(Protocol):
    def propose(
        self, session: SearchSession, logical_id: int, target_costs: Sequence[float] | None
    ) -> Any: ...


class CandidateScorer(Protocol):
    def score(self, snapshot: Any, candidates: Any) -> Sequence[float]: ...


class AcceptancePolicy(Protocol):
    def choose(self, scores: Sequence[float], candidates: Any) -> int | None: ...


class ControlDecision(Enum):
    CONTINUE = "continue"
    RESTART = "restart"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ControlContext:
    attempt: int
    transitions: int
    repeated_state_visits: int
    accepted: bool
    has_candidates: bool


class ControlPolicy(Protocol):
    def decide(self, context: ControlContext) -> ControlDecision: ...


class RepairSelector(Protocol):
    def select(self, snapshot: Any, rng: random.Random) -> Sequence[int]: ...


class DefaultVertexSelector:
    def select(self, snapshot: Any, eligible: Sequence[int], rng: random.Random) -> int:
        if not eligible:
            raise ValueError("the selector requires at least one eligible variable")
        best_priority = max(
            (snapshot.conflicts[node], snapshot.missing_incident_edges[node]) for node in eligible
        )
        tied = sorted(
            node
            for node in eligible
            if (snapshot.conflicts[node], snapshot.missing_incident_edges[node]) == best_priority
        )
        return rng.choice(tied)


class NoRouteCostProvider:
    def costs(self, snapshot: Any, logical_id: int, target_size: int) -> None:
        return None


class NativeCandidateProvider:
    def propose(
        self,
        session: SearchSession,
        logical_id: int,
        target_costs: Sequence[float] | None,
    ) -> Any:
        return session.propose(logical_id, target_costs)


class WideCandidateProvider:
    """Request a bounded, applicable batch beyond the default resource prefix."""

    def __init__(self, scoring_candidates: int) -> None:
        self._scoring_candidates = integer_option(
            "scoring_candidates", scoring_candidates, minimum=1
        )

    @property
    def scoring_candidates(self) -> int:
        return self._scoring_candidates

    def propose(
        self,
        session: SearchSession,
        logical_id: int,
        target_costs: Sequence[float] | None,
    ) -> Any:
        return session.propose_applicable(
            logical_id,
            scoring_candidates=self._scoring_candidates,
            target_costs=target_costs,
        )


class GreedyAcceptancePolicy:
    def choose(self, scores: Sequence[float], candidates: Any) -> int | None:
        if not scores:
            return None
        return min(range(len(scores)), key=lambda index: (scores[index], index))


class StagnationControlPolicy:
    def __init__(self, max_repeated_visits: int = 3) -> None:
        if max_repeated_visits < 1:
            raise ValueError("max_repeated_visits must be positive")
        self._max_repeated_visits = max_repeated_visits

    def decide(self, context: ControlContext) -> ControlDecision:
        if context.repeated_state_visits >= self._max_repeated_visits:
            return ControlDecision.RESTART
        return ControlDecision.CONTINUE


class ViolationNeighborhoodSelector:
    """Select a bounded high-pressure neighborhood for local destroy-and-repair."""

    def __init__(self, max_size: int = 4) -> None:
        if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
            raise ValueError("max_size must be a positive integer")
        self._max_size = max_size

    def select(self, snapshot: Any, rng: random.Random) -> tuple[int, ...]:
        logicals = list(range(len(snapshot.chains)))
        if not logicals:
            raise ValueError("cannot select a repair neighborhood from an empty source graph")
        rng.shuffle(logicals)
        logicals.sort(
            key=lambda logical: (
                snapshot.conflicts[logical],
                snapshot.missing_incident_edges[logical],
            ),
            reverse=True,
        )
        size = min(self._max_size, max(1, len(logicals) // 2))
        return tuple(logicals[:size])

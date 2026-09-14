"""Python orchestration across batched native search operations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from . import _core
from ._policy_validation import (
    validated_choice,
    validated_costs,
    validated_repair_neighborhood,
    validated_scores,
    validated_selection,
)
from ._options import integer_option, seed_option, timeout_option
from .components import (
    DefaultVertexSelector,
    GreedyAcceptancePolicy,
    NativeCandidateProvider,
    NoRouteCostProvider,
    StagnationControlPolicy,
)
from .scoring import ResourceScorer
from .session import SearchSession


@dataclass(frozen=True, slots=True)
class TransitionOutcome:
    logical_id: int
    candidate_index: int | None
    candidate_chains: tuple[tuple[int, ...], ...]
    scores: tuple[float, ...]
    accepted: bool
    before: Any
    after: Any
    policy_mode: str | None = None
    policy_reason: str | None = None


class SearchOrchestrator:
    def __init__(
        self,
        *,
        selector=None,
        route_cost_provider=None,
        candidate_provider=None,
        scorer=None,
        acceptance_policy=None,
        control_policy=None,
        repair_selector=None,
        max_local_repairs: int = 0,
    ) -> None:
        self.selector = selector if selector is not None else DefaultVertexSelector()
        self.route_cost_provider = (
            route_cost_provider if route_cost_provider is not None else NoRouteCostProvider()
        )
        self.candidate_provider = (
            candidate_provider if candidate_provider is not None else NativeCandidateProvider()
        )
        self.scorer = scorer if scorer is not None else ResourceScorer()
        self.acceptance_policy = (
            acceptance_policy if acceptance_policy is not None else GreedyAcceptancePolicy()
        )
        policy_validator = getattr(self.scorer, "validate_acceptance_policy", None)
        if policy_validator is not None:
            if not callable(policy_validator):
                raise TypeError("scorer acceptance-policy validator must be callable")
            policy_validator(self.acceptance_policy)
        self.control_policy = (
            control_policy if control_policy is not None else StagnationControlPolicy()
        )
        self.max_local_repairs = integer_option("max_local_repairs", max_local_repairs, minimum=0)
        if repair_selector is None and self.max_local_repairs:
            raise ValueError("max_local_repairs requires a repair_selector")
        if repair_selector is not None and not self.max_local_repairs:
            raise ValueError("repair_selector requires positive max_local_repairs")
        self.repair_selector = repair_selector

    @staticmethod
    def _discard_safely(session: SearchSession, batch: Any) -> None:
        try:
            session.discard(batch)
        except BaseException:
            pass

    def _policy_audit(self) -> tuple[str | None, str | None]:
        audit = getattr(self.scorer, "policy_audit", None)
        if audit is None:
            return None, None
        if not callable(audit):
            raise TypeError("scorer policy-audit provider must be callable")
        value = audit()
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not value[0]
            or (value[1] is not None and not isinstance(value[1], str))
        ):
            raise ValueError("scorer policy audit must return (mode, reason)")
        return value

    def transition(
        self,
        session: SearchSession,
        rng: random.Random,
        *,
        allow_valid_refinement: bool = False,
    ) -> TransitionOutcome:
        if not isinstance(allow_valid_refinement, bool):
            raise ValueError("allow_valid_refinement must be a boolean")
        before = session.snapshot()
        eligible = session.eligible_variables()
        if not eligible and allow_valid_refinement and before.valid:
            eligible = list(range(len(before.chains)))
        if not eligible:
            raise ValueError("no logical variable is eligible for repair")
        logical = validated_selection(self.selector.select(before, eligible, rng), eligible)
        costs = validated_costs(
            self.route_cost_provider.costs(before, logical, len(session.normalized_target.labels)),
            len(session.normalized_target.labels),
        )
        batch = self.candidate_provider.propose(session, logical, costs)
        if not isinstance(batch, _core.CandidateBatch) or batch.logical != logical:
            self._discard_safely(session, batch)
            raise ValueError("candidate provider returned a foreign or mismatched batch")

        chains = tuple(tuple(candidate.chain) for candidate in batch.candidates)
        if not chains:
            session.discard(batch)
            return TransitionOutcome(logical, None, chains, (), False, before, session.snapshot())

        try:
            scores = validated_scores(self.scorer.score(before, batch), len(chains))
            choice = validated_choice(self.acceptance_policy.choose(scores, batch), len(chains))
            policy_mode, policy_reason = self._policy_audit()
        except BaseException:
            self._discard_safely(session, batch)
            raise
        if choice is None:
            session.discard(batch)
            accepted = False
        else:
            session.apply(batch, choice)
            accepted = True
        return TransitionOutcome(
            logical,
            choice,
            chains,
            tuple(scores),
            accepted,
            before,
            session.snapshot(),
            policy_mode,
            policy_reason,
        )

    def repair(self, session: SearchSession, snapshot: Any, rng: random.Random) -> tuple[int, ...]:
        if self.repair_selector is None:
            raise ValueError("no local repair selector is configured")
        logical_ids = validated_repair_neighborhood(
            self.repair_selector.select(snapshot, rng), len(snapshot.chains)
        )
        session.perturb(logical_ids)
        return logical_ids

    def run(
        self,
        session: SearchSession,
        rng: random.Random,
        *,
        random_seed: int,
        tries: int,
        max_transitions: int,
        timeout: float | None,
        anytime: bool = False,
        terminal_scorer=None,
    ):
        from ._search_loop import run_search

        random_seed = seed_option(random_seed, allow_none=False)
        tries = integer_option("tries", tries, minimum=1)
        max_transitions = integer_option("max_transitions", max_transitions, minimum=0)
        timeout = timeout_option(timeout)
        return run_search(
            self,
            session,
            rng,
            random_seed=random_seed,
            tries=tries,
            max_transitions=max_transitions,
            timeout=timeout,
            anytime=anytime,
            terminal_scorer=terminal_scorer,
        )

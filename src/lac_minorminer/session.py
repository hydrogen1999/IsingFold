"""Label-aware Python facade over the native integer-ID search session."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from . import _core
from ._graph_input import NormalizedGraph, normalize_graph
from ._options import integer_option, seed_option
from ._policy_validation import validated_costs
from .diagnostics import SearchWorkCounters


def _native_work_limits(work_cap: SearchWorkCounters | None):
    limits = _core.NativeWorkLimits()
    if work_cap is None:
        return limits
    if not isinstance(work_cap, SearchWorkCounters):
        raise TypeError("work_cap must be a SearchWorkCounters instance or None")
    for name, value in work_cap.as_dict().items():
        setattr(limits, name, value)
    return limits


class SearchSession:
    def __init__(
        self,
        source: Any,
        target: Any,
        *,
        random_seed: int,
        max_candidates: int,
        work_cap: SearchWorkCounters | None = None,
    ) -> None:
        random_seed = seed_option(random_seed, allow_none=False)
        max_candidates = integer_option("max_candidates", max_candidates, minimum=1)
        self._source = normalize_graph(source)
        self._target = normalize_graph(target)
        self._max_candidates = max_candidates
        self._work_cap = work_cap
        native_source = _core.Graph(len(self._source.labels), self._source.edges)
        native_target = _core.Graph(len(self._target.labels), self._target.edges)
        self._native = _core.SearchSession(
            native_source,
            native_target,
            random_seed=random_seed,
            max_candidates=max_candidates,
            work_limits=_native_work_limits(work_cap),
        )

    @classmethod
    def from_chains(
        cls,
        source: Any,
        target: Any,
        chains: Iterable[Iterable[int]],
        *,
        random_seed: int,
        max_candidates: int,
        work_cap: SearchWorkCounters | None = None,
    ) -> SearchSession:
        """Restore a semantic search state for reproducible counterfactual shards."""

        random_seed = seed_option(random_seed, allow_none=False)
        max_candidates = integer_option("max_candidates", max_candidates, minimum=1)
        normalized_source = normalize_graph(source)
        normalized_target = normalize_graph(target)
        restored_chains = [list(chain) for chain in chains]
        if len(restored_chains) != len(normalized_source.labels):
            raise ValueError("chains must contain exactly one chain per logical node")
        branch = object.__new__(cls)
        branch._source = normalized_source
        branch._target = normalized_target
        branch._max_candidates = max_candidates
        branch._work_cap = work_cap
        native_source = _core.Graph(len(normalized_source.labels), normalized_source.edges)
        native_target = _core.Graph(len(normalized_target.labels), normalized_target.edges)
        branch._native = _core.SearchSession.from_chains(
            native_source,
            native_target,
            restored_chains,
            random_seed=random_seed,
            max_candidates=max_candidates,
            work_limits=_native_work_limits(work_cap),
        )
        return branch

    @property
    def source_labels(self) -> tuple[Any, ...]:
        return self._source.labels

    @property
    def target_labels(self) -> tuple[Any, ...]:
        return self._target.labels

    @property
    def normalized_source(self) -> NormalizedGraph:
        return self._source

    @property
    def normalized_target(self) -> NormalizedGraph:
        return self._target

    @property
    def max_candidates(self) -> int:
        """Maximum size of the native decision batch."""

        return self._max_candidates

    @property
    def work_cap(self) -> SearchWorkCounters | None:
        """Prospective native work ceiling, or ``None`` for the public unbounded mode."""

        return self._work_cap

    @property
    def session_id(self) -> int:
        """Opaque native identity used to reject cross-session policy inputs."""

        return int(self._native.session_id)

    def snapshot(self):
        return self._native.snapshot()

    def work_counters(self) -> SearchWorkCounters:
        """Return an immutable exact snapshot of all nine native work coordinates."""

        native = self._native.work_counters()
        return SearchWorkCounters(
            decisions=int(native.decisions),
            route_expansions=int(native.route_expansions),
            materializations=int(native.materializations),
            compiler_calls=int(native.compiler_calls),
            validator_calls=int(native.validator_calls),
            cut_edge_visits=int(native.cut_edge_visits),
            restart_work=int(native.restart_work),
            evaluator_reads=int(native.evaluator_reads),
            feature_work=int(native.feature_work),
        )

    def fork(self, *, random_seed: int | None = None) -> SearchSession:
        """Copy state into a branch, optionally reseeding native continuation randomness."""

        branch = object.__new__(SearchSession)
        branch._source = self._source
        branch._target = self._target
        branch._max_candidates = self._max_candidates
        branch._work_cap = self._work_cap
        branch._native = (
            self._native.fork()
            if random_seed is None
            else self._native.fork_with_seed(seed_option(random_seed, allow_none=False))
        )
        return branch

    def eligible_variables(self) -> list[int]:
        return self._native.eligible_variables()

    def propose(self, logical_id: int, target_costs: Sequence[float] | None = None):
        if target_costs is None:
            costs = []
        else:
            costs = validated_costs(target_costs, len(self._target.labels))
            assert costs is not None
        return self._native.propose(logical_id, costs)

    def propose_applicable(
        self,
        logical_id: int,
        scoring_candidates: int,
        target_costs: Sequence[float] | None = None,
    ):
        """Return up to ``scoring_candidates`` choices in an applicable one-shot batch."""

        scoring_candidates = integer_option(
            "scoring_candidates", scoring_candidates, minimum=self._max_candidates
        )
        if target_costs is None:
            costs = []
        else:
            costs = validated_costs(target_costs, len(self._target.labels))
            assert costs is not None
        return self._native.propose_applicable(logical_id, scoring_candidates, costs)

    def propose_with_audit(
        self,
        logical_id: int,
        audit_candidates: int,
        target_costs: Sequence[float] | None = None,
    ):
        audit_candidates = integer_option(
            "audit_candidates", audit_candidates, minimum=self._max_candidates
        )
        if target_costs is None:
            costs = []
        else:
            costs = validated_costs(target_costs, len(self._target.labels))
            assert costs is not None
        return self._native.propose_with_audit(logical_id, audit_candidates, costs)

    def materialize(self, logical_id: int, chains: Iterable[Iterable[int]]):
        return self._native.materialize(logical_id, [list(chain) for chain in chains])

    def candidate_is_valid(self, batch, candidate_index: int) -> bool:
        """Validate a hypothetical candidate natively without consuming its handle."""

        return bool(self._native.candidate_is_valid(batch, candidate_index))

    def apply(self, batch, candidate_index: int) -> None:
        self._native.apply(batch, candidate_index)

    def discard(self, batch) -> None:
        self._native.discard(batch)

    def perturb(self, logical_ids: Iterable[int]) -> None:
        """Reset one repair neighborhood while preserving all other chains."""

        self._native.perturb(list(logical_ids))

    def restart(self) -> None:
        self._native.restart()

    def embedding(self) -> dict[Any, list[Any]]:
        snapshot = self.snapshot()
        return {
            logical: [self._target.labels[target] for target in snapshot.chains[index]]
            for index, logical in enumerate(self._source.labels)
        }

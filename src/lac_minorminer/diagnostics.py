"""Stable diagnostic records for bounded embedding searches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


WORK_COUNTER_SCHEMA = "lac-minorminer.native-work"
WORK_COUNTER_VERSION = 3
WORK_COUNTER_FIELDS = (
    "decisions",
    "route_expansions",
    "materializations",
    "compiler_calls",
    "validator_calls",
    "cut_edge_visits",
    "restart_work",
    "evaluator_reads",
    "feature_work",
)


@dataclass(frozen=True, slots=True)
class SearchWorkCounters:
    """Exact v3 semantic-operation counters for one prospectively bounded search lineage.

    These values count registered algorithm events, not estimated CPU instructions.  The
    native schema defines route expansions as settled Dijkstra vertices (or enumerated
    singleton roots), materializations as complete candidate successors before truncation or
    deduplication, validator calls as top-level embedding predicates, and feature work as
    registered scalar predicates or ranks.  For the structural Chimera fallback, lane-component
    vertex settlements are route expansions, complete L-chain objects are materializations, and
    topology, compatibility, and bounded clique-search predicates are feature work.  Restart work
    excludes the initial top-level construction because the complete-system runner charges that
    invocation.

    This initializer has no Ising compiler, cut generator, or outcome evaluator.  Consequently
    ``compiler_calls``, ``cut_edge_visits``, and ``evaluator_reads`` are provably zero here.
    """

    decisions: int = 0
    route_expansions: int = 0
    materializations: int = 0
    compiler_calls: int = 0
    validator_calls: int = 0
    cut_edge_visits: int = 0
    restart_work: int = 0
    evaluator_reads: int = 0
    feature_work: int = 0

    def __post_init__(self) -> None:
        for name in WORK_COUNTER_FIELDS:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 2**64
            ):
                raise ValueError(
                    f"work counter {name} must be a nonnegative integer below 2**64"
                )

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in WORK_COUNTER_FIELDS}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SearchWorkCounters:
        if set(payload) != set(WORK_COUNTER_FIELDS):
            raise ValueError("work counter payload has missing or unknown coordinates")
        return cls(**{name: payload[name] for name in WORK_COUNTER_FIELDS})

    def plus(self, **increments: int) -> SearchWorkCounters:
        if not set(increments) <= set(WORK_COUNTER_FIELDS):
            raise ValueError("unknown work counter increment")
        values = self.as_dict()
        for name, amount in increments.items():
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise ValueError("work counter increments must be nonnegative integers")
            values[name] += amount
        return SearchWorkCounters(**values)


class TerminationReason(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    TRANSITION_LIMIT = "transition_limit"
    EMPTY_TARGET = "empty_target"
    NO_CANDIDATE = "no_candidate"
    TRIES_EXHAUSTED = "tries_exhausted"
    CONTROL_STOP = "control_stop"
    WORK_BUDGET_EXHAUSTED = "work_budget_exhausted"


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    attempt: int
    transition: int
    logical_id: int
    candidate_index: int | None
    accepted: bool
    before_generation: int
    after_generation: int
    max_occupancy: int
    total_excess_occupancy: int
    missing_source_edges: int
    selected_chain: tuple[int, ...] | None
    policy_mode: str | None = None
    policy_reason: str | None = None


@dataclass(frozen=True, slots=True)
class IncumbentRecord:
    attempt: int
    transition: int
    generation: int
    terminal_value: float | None
    chains: tuple[tuple[int, ...], ...]
    used_target_nodes: int
    longest_chain: int
    total_chain_length: int


@dataclass(frozen=True, slots=True)
class RepairRecord:
    attempt: int
    repair: int
    after_transition: int
    logical_ids: tuple[int, ...]
    before_generation: int
    after_generation: int
    before_max_occupancy: int
    after_max_occupancy: int
    before_total_excess_occupancy: int
    after_total_excess_occupancy: int
    before_missing_source_edges: int
    after_missing_source_edges: int


@dataclass(frozen=True, slots=True)
class SearchDiagnostics:
    success: bool
    termination_reason: TerminationReason
    random_seed: int
    attempts: int
    transitions: int
    proposals: int
    applied: int
    discarded: int
    elapsed_seconds: float
    package_version: str
    backend: str
    trace: tuple[TransitionRecord, ...]
    incumbents: tuple[IncumbentRecord, ...] = ()
    best_incumbent_index: int | None = None
    anytime: bool = False
    repair_trace: tuple[RepairRecord, ...] = ()
    work: SearchWorkCounters = SearchWorkCounters()
    work_counter_schema: str = WORK_COUNTER_SCHEMA
    work_counter_version: int = WORK_COUNTER_VERSION
    work_cap: SearchWorkCounters | None = None
    work_budget_exhausted_coordinate: str | None = None
    search_profile: str = "v0"
    profile_detail: str = "v0"
    structural_fallback_invoked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.work, SearchWorkCounters):
            raise TypeError("search work must use SearchWorkCounters")
        if not isinstance(self.search_profile, str) or not self.search_profile:
            raise ValueError("search profile must be a nonempty string")
        if not isinstance(self.profile_detail, str) or not self.profile_detail:
            raise ValueError("search profile detail must be a nonempty string")
        if not isinstance(self.structural_fallback_invoked, bool):
            raise ValueError("structural fallback invocation flag must be Boolean")
        if (
            self.work_counter_schema != WORK_COUNTER_SCHEMA
            or self.work_counter_version != WORK_COUNTER_VERSION
        ):
            raise ValueError("unsupported search work-counter schema")
        if self.work.decisions != self.transitions:
            raise ValueError("decision work must equal the consumed transition count")
        if any(
            getattr(self.work, name) != 0
            for name in ("compiler_calls", "cut_edge_visits", "evaluator_reads")
        ):
            raise ValueError("native search cannot claim compiler, cut, or evaluator work")
        if self.work_cap is not None:
            if not isinstance(self.work_cap, SearchWorkCounters):
                raise TypeError("search work cap must use SearchWorkCounters")
            if any(
                getattr(self.work, name) > getattr(self.work_cap, name)
                for name in WORK_COUNTER_FIELDS
            ):
                raise ValueError("search work exceeds its prospective native cap")
        exhausted = self.termination_reason is TerminationReason.WORK_BUDGET_EXHAUSTED
        if exhausted:
            if (
                self.work_cap is None
                or self.work_budget_exhausted_coordinate not in WORK_COUNTER_FIELDS
            ):
                raise ValueError("budget termination requires a cap and exhausted coordinate")
        elif self.work_budget_exhausted_coordinate is not None:
            raise ValueError("non-budget termination cannot claim an exhausted coordinate")

    @property
    def best_incumbent(self) -> IncumbentRecord | None:
        if self.best_incumbent_index is None:
            return None
        return self.incumbents[self.best_incumbent_index]

    @property
    def repairs(self) -> int:
        return len(self.repair_trace)

    def structural_dict(self) -> dict[str, Any]:
        trace = []
        for record in self.trace:
            serialized = asdict(record)
            if record.policy_mode is None:
                serialized.pop("policy_mode")
                serialized.pop("policy_reason")
            trace.append(serialized)
        result = {
            "success": self.success,
            "termination_reason": self.termination_reason.value,
            "random_seed": self.random_seed,
            "attempts": self.attempts,
            "transitions": self.transitions,
            "proposals": self.proposals,
            "applied": self.applied,
            "discarded": self.discarded,
            "package_version": self.package_version,
            "backend": self.backend,
            "trace": trace,
            "incumbents": [asdict(record) for record in self.incumbents],
            "best_incumbent_index": self.best_incumbent_index,
            "anytime": self.anytime,
            "work": self.work.as_dict(),
            "work_counter_schema": self.work_counter_schema,
            "work_counter_version": self.work_counter_version,
            "work_cap": None if self.work_cap is None else self.work_cap.as_dict(),
            "work_budget_exhausted_coordinate": self.work_budget_exhausted_coordinate,
            "search_profile": self.search_profile,
            "profile_detail": self.profile_detail,
            "structural_fallback_invoked": self.structural_fallback_invoked,
        }
        if self.repair_trace:
            result["repair_trace"] = [asdict(record) for record in self.repair_trace]
        return result

    def to_dict(self) -> dict[str, Any]:
        result = self.structural_dict()
        result["elapsed_seconds"] = self.elapsed_seconds
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchDiagnostics":
        if not {
            "work_cap",
            "work_budget_exhausted_coordinate",
            "search_profile",
            "profile_detail",
            "structural_fallback_invoked",
        } <= set(payload):
            raise ValueError("search diagnostics omitted profile or prospective work-budget fields")
        if (
            payload.get("work_counter_schema") != WORK_COUNTER_SCHEMA
            or payload.get("work_counter_version") != WORK_COUNTER_VERSION
        ):
            raise ValueError("unsupported or missing search work-counter schema")
        raw_work = payload.get("work")
        if not isinstance(raw_work, dict):
            raise ValueError("search diagnostics work counters must be an object")
        raw_cap = payload.get("work_cap")
        if raw_cap is not None and not isinstance(raw_cap, dict):
            raise ValueError("search diagnostics work cap must be an object or null")
        trace = []
        for raw_record in payload["trace"]:
            record = dict(raw_record)
            if record["selected_chain"] is not None:
                record["selected_chain"] = tuple(record["selected_chain"])
            trace.append(TransitionRecord(**record))
        incumbents = []
        for raw_record in payload.get("incumbents", []):
            record = dict(raw_record)
            record["chains"] = tuple(tuple(chain) for chain in record["chains"])
            incumbents.append(IncumbentRecord(**record))
        repair_trace = []
        for raw_record in payload.get("repair_trace", []):
            record = dict(raw_record)
            record["logical_ids"] = tuple(record["logical_ids"])
            repair_trace.append(RepairRecord(**record))
        return cls(
            success=payload["success"],
            termination_reason=TerminationReason(payload["termination_reason"]),
            random_seed=payload["random_seed"],
            attempts=payload["attempts"],
            transitions=payload["transitions"],
            proposals=payload["proposals"],
            applied=payload["applied"],
            discarded=payload["discarded"],
            elapsed_seconds=payload["elapsed_seconds"],
            package_version=payload["package_version"],
            backend=payload["backend"],
            trace=tuple(trace),
            incumbents=tuple(incumbents),
            best_incumbent_index=payload.get("best_incumbent_index"),
            anytime=payload.get("anytime", False),
            repair_trace=tuple(repair_trace),
            work=SearchWorkCounters.from_dict(raw_work),
            work_counter_schema=payload["work_counter_schema"],
            work_counter_version=payload["work_counter_version"],
            work_cap=None if raw_cap is None else SearchWorkCounters.from_dict(raw_cap),
            work_budget_exhausted_coordinate=payload.get(
                "work_budget_exhausted_coordinate"
            ),
            search_profile=payload["search_profile"],
            profile_detail=payload["profile_detail"],
            structural_fallback_invoked=payload["structural_fallback_invoked"],
        )

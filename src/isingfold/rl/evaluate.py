"""Independent final evaluation, raw receipts and lineage-clustered comparisons.

The policy environment is run in deployment mode.  Only after it has returned a physical
program does this module open the evaluator boundary and obtain a fresh, deterministic read
block.  Publication comparisons first require exact instance/repetition pairs, then average
those repetitions inside each independent base lineage.

Spec: Architecture Rev2, ``Experiments that decide whether the architecture works``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from isingfold.rl.contracts import (
    WORK_FIELDS,
    Candidate,
    Context,
    InitFailureRecord,
    Mode,
    Opcode,
    TerminalRecord,
    WorkVector,
    stable_digest,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, StrengthSelector, task_initializer
from isingfold.rl.evaluator import sample_program
from isingfold.rl.program import Program
from isingfold.rl.proposal import Initializer, LEGACY_ONLINE_INITIALIZER_RESTARTS_V1
from isingfold.rl.tensorize import (
    EDGE_USE_LOGICAL,
    N_ACTION,
    N_ARCHIVE,
    N_FACTOR,
    N_LEDGE,
    ROLE_ARCHIVE,
    ROLE_NEW,
    ROLE_OLD,
)
from isingfold.rl.validate import occupancy

EVALUATION_RECEIPT_VERSION = "isingfold-evaluation-receipt-v3"
TASK_TERMINAL = "TASK_TERMINAL"
INITIALIZATION_FAILURE = "INITIALIZATION_FAILURE"


class EvaluationProtocolError(RuntimeError):
    """A missing, ambiguous or internally inconsistent benchmark record."""


def repair_target_satisfied(candidate: Candidate, env: EmbeddingEnv) -> bool:
    """Return whether a selected repair eliminated its registered defect.

    Repair proposals have exactly one target: either a contested hardware qubit or a
    missing logical-edge contact.  Telemetry must use both forms, matching the action
    grammar, rather than silently treating demand-directed repairs as failures.
    """

    if candidate.opcode is not Opcode.REPAIR_GROUP or env.state is None:
        return False
    if candidate.target_conflict is not None:
        return occupancy(env.state.chains).get(candidate.target_conflict, 0) <= 1
    if candidate.target_demand is None:
        return False
    left, right = candidate.target_demand
    left_chain = env.state.chains.get(left, frozenset())
    right_chain = env.state.chains.get(right, frozenset())
    return any(
        left_qubit != right_qubit and env.task.host.has_edge(left_qubit, right_qubit)
        for left_qubit in left_chain
        for right_qubit in right_chain
    )


def _is_digest(value: str | None) -> bool:
    if value is None or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class EpisodeOutcome:
    """One raw controller attempt, before any clustering or aggregation.

    The first nine fields retain the original lightweight API.  Receipts emitted by
    :func:`run_controller` also populate all provenance, program, count and work fields.
    Initializer failures are system outcomes (``population_eligible=False``), not fabricated
    zero-utility policy episodes.
    """

    instance: str
    lineage: str
    returned_valid: bool
    utility: float | None
    qubits: int | None
    max_chain: int | None
    decisions: int
    selected_strength: float | None
    reason: str
    repetition: int = 0
    episode_seed: int | None = None
    evaluator_seed: int | None = None
    program_digest: str | None = None
    selected_strength_index: int | None = None
    evaluator_hits: int | None = None
    evaluator_reads: int | None = None
    work: WorkVector | None = None
    validation_digest: str | None = None
    broken_chain_fraction: float | None = None
    mean_energy_residual: float | None = None
    online_seconds: float | None = field(default=None, compare=False)
    evaluator_seconds: float | None = field(default=None, compare=False)
    returned_embedding: Mapping[str, frozenset] | None = field(
        default=None, compare=False, repr=False
    )
    """The embedding the episode actually returned, for callers that feed one episode's result
    into the next. Excluded from equality and from the receipt: it is a convenience handle on
    state the receipt already pins through `program_digest`, never a new claim."""
    controller_calls: int | None = None
    controller_seconds: float | None = field(default=None, compare=False)
    outcome_kind: str = TASK_TERMINAL
    population_eligible: bool = True
    chain_sizes: tuple[int, ...] | None = None
    overlap_events: int | None = None
    overlap_decisions: int | None = None
    repair_attempts: int | None = None
    repair_successes: int | None = None
    candidate_states: int | None = None
    legal_actions_total: int | None = None
    legal_opcode_types_total: int | None = None

    @property
    def pair_key(self) -> tuple[str, str, int]:
        """The exact matching key; lineage alone is only the later bootstrap cluster."""

        return (self.lineage or self.instance, self.instance, self.repetition)

    @property
    def fixed_program_r99(self) -> int | None:
        """Read-count diagnostic for this fixed program only, never a population R99."""

        if self.evaluator_hits is None or self.evaluator_reads is None:
            return None
        if self.evaluator_hits <= 0:
            return None
        if self.evaluator_hits >= self.evaluator_reads:
            return 1
        probability = self.evaluator_hits / self.evaluator_reads
        return int(math.ceil(math.log(0.01) / math.log1p(-probability)))

    def validate_receipt(self, *, require_complete: bool = False) -> None:
        """Fail closed on impossible counts, labels or provenance."""

        if not isinstance(self.instance, str) or not isinstance(self.lineage, str):
            raise EvaluationProtocolError("instance and base lineage must be strings")
        if (
            not self.instance
            or not self.lineage
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            raise EvaluationProtocolError("instance and base lineage must be nonempty")
        if type(self.repetition) is not int or self.repetition < 0:
            raise EvaluationProtocolError("repetition must be a nonnegative integer")
        if type(self.decisions) is not int or self.decisions < 0:
            raise EvaluationProtocolError("decision count must be a nonnegative integer")
        if type(self.returned_valid) is not bool or type(self.population_eligible) is not bool:
            raise EvaluationProtocolError("validity and population flags must be Boolean")
        if self.outcome_kind not in (TASK_TERMINAL, INITIALIZATION_FAILURE):
            raise EvaluationProtocolError(f"unknown outcome kind {self.outcome_kind!r}")
        for name, value in (
            ("episode_seed", self.episode_seed),
            ("evaluator_seed", self.evaluator_seed),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise EvaluationProtocolError(f"{name} must be a nonnegative integer")
        if self.work is not None and any(
            type(getattr(self.work, name)) is not int for name in WORK_FIELDS
        ):
            raise EvaluationProtocolError("work coordinates must be integers")
        if self.work is not None and not self.work.is_nonnegative:
            raise EvaluationProtocolError("work coordinates cannot be negative")
        for name, value in (
            ("online_seconds", self.online_seconds),
            ("evaluator_seconds", self.evaluator_seconds),
            ("controller_seconds", self.controller_seconds),
        ):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0.0
            ):
                raise EvaluationProtocolError(f"{name} must be finite and nonnegative")
        if self.controller_calls is not None and (
            type(self.controller_calls) is not int or self.controller_calls < 0
        ):
            raise EvaluationProtocolError("controller calls must be a nonnegative integer")
        if (self.controller_calls is None) != (self.controller_seconds is None):
            raise EvaluationProtocolError(
                "controller calls and controller time must appear together"
            )
        if self.controller_calls is not None and self.controller_calls != self.decisions:
            raise EvaluationProtocolError(
                "controller call count must equal the recorded decision count"
            )
        if self.controller_calls == 0 and self.controller_seconds != 0.0:
            raise EvaluationProtocolError("zero controller calls require zero controller time")
        if (
            self.controller_seconds is not None
            and self.online_seconds is not None
            and self.controller_seconds
            > self.online_seconds + max(1e-12, 1e-9 * self.online_seconds)
        ):
            raise EvaluationProtocolError("controller time cannot exceed online search time")
        mechanism_values = (
            self.overlap_events,
            self.overlap_decisions,
            self.repair_attempts,
            self.repair_successes,
            self.candidate_states,
            self.legal_actions_total,
            self.legal_opcode_types_total,
        )
        if any(value is not None for value in mechanism_values):
            if any(type(value) is not int or value < 0 for value in mechanism_values):
                raise EvaluationProtocolError(
                    "mechanism telemetry must be a complete nonnegative integer block"
                )
            if self.repair_successes > self.repair_attempts:
                raise EvaluationProtocolError("repair successes cannot exceed repair attempts")
            if self.overlap_events > self.overlap_decisions:
                raise EvaluationProtocolError("overlap events cannot exceed overlap decisions")
            if self.controller_calls is not None and self.candidate_states != self.controller_calls:
                raise EvaluationProtocolError(
                    "candidate-state count must equal controller call count"
                )
            if self.legal_actions_total < self.candidate_states:
                raise EvaluationProtocolError(
                    "every candidate state must expose at least one legal action"
                )
            if not (
                self.candidate_states
                <= self.legal_opcode_types_total
                <= len(Opcode) * self.candidate_states
            ):
                raise EvaluationProtocolError(
                    "legal opcode-type total is inconsistent with candidate states"
                )
        if self.chain_sizes is not None and (
            not isinstance(self.chain_sizes, tuple)
            or not self.chain_sizes
            or any(type(value) is not int or value <= 0 for value in self.chain_sizes)
        ):
            raise EvaluationProtocolError(
                "chain sizes must be a nonempty tuple of positive integers"
            )
        if self.broken_chain_fraction is not None and (
            isinstance(self.broken_chain_fraction, bool)
            or not math.isfinite(self.broken_chain_fraction)
            or not 0.0 <= self.broken_chain_fraction <= 1.0
        ):
            raise EvaluationProtocolError("broken-chain fraction must be finite and in [0,1]")
        if self.mean_energy_residual is not None and (
            isinstance(self.mean_energy_residual, bool)
            or not math.isfinite(self.mean_energy_residual)
            or self.mean_energy_residual < 0.0
        ):
            raise EvaluationProtocolError("mean energy residual must be finite and nonnegative")
        if self.program_digest is not None and not _is_digest(self.program_digest):
            raise EvaluationProtocolError("program digest is not a SHA-256 value")
        if self.validation_digest is not None and not _is_digest(self.validation_digest):
            raise EvaluationProtocolError("validation digest is not a SHA-256 value")
        if require_complete:
            if (
                self.episode_seed is None
                or self.work is None
                or not _is_digest(self.validation_digest)
            ):
                raise EvaluationProtocolError("receipt is missing seed, work or validation digest")
            if self.online_seconds is None:
                raise EvaluationProtocolError("complete receipt is missing online search time")

        if self.outcome_kind == INITIALIZATION_FAILURE:
            if self.population_eligible:
                raise EvaluationProtocolError("initializer failure cannot be a conditional episode")
            if self.utility is not None or self.returned_valid:
                raise EvaluationProtocolError("initializer failure cannot carry a utility label")
            if any(
                value is not None
                for value in (
                    self.qubits,
                    self.max_chain,
                    self.evaluator_seed,
                    self.program_digest,
                    self.selected_strength,
                    self.selected_strength_index,
                    self.evaluator_hits,
                    self.evaluator_reads,
                    self.evaluator_seconds,
                    self.broken_chain_fraction,
                    self.mean_energy_residual,
                    self.chain_sizes,
                )
            ) or (self.work is not None and self.work.evaluator_reads != 0):
                raise EvaluationProtocolError(
                    "initializer failure cannot carry valid-program or evaluator telemetry"
                )
            return
        if not self.population_eligible:
            raise EvaluationProtocolError("a task terminal must be population eligible")
        if (
            self.utility is None
            or isinstance(self.utility, bool)
            or not math.isfinite(self.utility)
            or not 0.0 <= self.utility <= 1.0
        ):
            raise EvaluationProtocolError("task utility must be a finite probability in [0,1]")

        if not self.returned_valid:
            if self.utility != 0.0:
                raise EvaluationProtocolError("ordinary no-valid terminal must retain utility zero")
            if any(
                item is not None
                for item in (
                    self.qubits,
                    self.max_chain,
                    self.evaluator_seed,
                    self.program_digest,
                    self.selected_strength,
                    self.selected_strength_index,
                    self.evaluator_hits,
                    self.evaluator_reads,
                    self.evaluator_seconds,
                    self.broken_chain_fraction,
                    self.mean_energy_residual,
                    self.chain_sizes,
                )
            ) or (self.work is not None and self.work.evaluator_reads != 0):
                raise EvaluationProtocolError(
                    "no-valid terminal cannot claim a sampled program or evaluator telemetry"
                )
            return

        if (
            type(self.qubits) is not int
            or self.qubits <= 0
            or type(self.max_chain) is not int
            or self.max_chain <= 0
        ):
            raise EvaluationProtocolError("valid return needs positive qubit and chain counts")
        if self.chain_sizes is not None and (
            sum(self.chain_sizes) != self.qubits or max(self.chain_sizes) != self.max_chain
        ):
            raise EvaluationProtocolError(
                "terminal chain-size distribution disagrees with qubit or maximum-chain counts"
            )
        if (
            self.selected_strength is None
            or isinstance(self.selected_strength, bool)
            or not math.isfinite(self.selected_strength)
        ):
            raise EvaluationProtocolError("valid return needs a finite selected strength")
        if self.selected_strength_index is not None and (
            type(self.selected_strength_index) is not int or self.selected_strength_index < 0
        ):
            raise EvaluationProtocolError("selected strength index cannot be negative")
        if require_complete and (
            not _is_digest(self.program_digest)
            or self.selected_strength_index is None
            or self.evaluator_seed is None
        ):
            raise EvaluationProtocolError("valid receipt lacks program or evaluator provenance")
        if (self.evaluator_hits is None) != (self.evaluator_reads is None):
            raise EvaluationProtocolError("evaluator hits and reads must appear together")
        if self.evaluator_hits is not None and self.evaluator_reads is not None:
            if (
                type(self.evaluator_hits) is not int
                or type(self.evaluator_reads) is not int
                or self.evaluator_reads <= 0
                or not 0 <= self.evaluator_hits <= self.evaluator_reads
            ):
                raise EvaluationProtocolError("invalid evaluator count block")
            expected = self.evaluator_hits / self.evaluator_reads
            if not math.isclose(self.utility, expected, rel_tol=0.0, abs_tol=1e-15):
                raise EvaluationProtocolError("utility does not equal the stored hit/read count")
        elif require_complete:
            raise EvaluationProtocolError("valid receipt is missing evaluator counts")
        if require_complete and (
            self.evaluator_seconds is None
            or self.broken_chain_fraction is None
            or self.mean_energy_residual is None
        ):
            raise EvaluationProtocolError(
                "valid receipt is missing evaluator latency or sample diagnostics"
            )
        if (
            self.work is not None
            and self.evaluator_reads is not None
            and self.work.evaluator_reads < self.evaluator_reads
        ):
            raise EvaluationProtocolError("work receipt omits final evaluator reads")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": EVALUATION_RECEIPT_VERSION,
            "instance": self.instance,
            "lineage": self.lineage,
            "repetition": self.repetition,
            "episode_seed": self.episode_seed,
            "evaluator_seed": self.evaluator_seed,
            "outcome_kind": self.outcome_kind,
            "population_eligible": self.population_eligible,
            "returned_valid": self.returned_valid,
            "reason": self.reason,
            "utility": self.utility,
            "qubits": self.qubits,
            "max_chain": self.max_chain,
            "decisions": self.decisions,
            "selected_strength": self.selected_strength,
            "selected_strength_index": self.selected_strength_index,
            "program_digest": self.program_digest,
            "validation_digest": self.validation_digest,
            "evaluator_hits": self.evaluator_hits,
            "evaluator_reads": self.evaluator_reads,
            "broken_chain_fraction": self.broken_chain_fraction,
            "mean_energy_residual": self.mean_energy_residual,
            "work": None if self.work is None else self.work.as_dict(),
            "online_seconds": self.online_seconds,
            "evaluator_seconds": self.evaluator_seconds,
            "controller_calls": self.controller_calls,
            "controller_seconds": self.controller_seconds,
            "chain_sizes": None if self.chain_sizes is None else list(self.chain_sizes),
            "overlap_events": self.overlap_events,
            "overlap_decisions": self.overlap_decisions,
            "repair_attempts": self.repair_attempts,
            "repair_successes": self.repair_successes,
            "candidate_states": self.candidate_states,
            "legal_actions_total": self.legal_actions_total,
            "legal_opcode_types_total": self.legal_opcode_types_total,
            "fixed_program_r99": self.fixed_program_r99,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        require_complete: bool = True,
    ) -> EpisodeOutcome:
        expected = set(cls("x", "x", False, 0.0, None, None, 0, None, "x").as_dict())
        if set(payload) != expected:
            raise EvaluationProtocolError(
                f"receipt schema keys differ: missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        if payload["schema"] != EVALUATION_RECEIPT_VERSION:
            raise EvaluationProtocolError(f"unsupported receipt schema {payload['schema']!r}")

        def required_string(name: str) -> str:
            value = payload[name]
            if not isinstance(value, str):
                raise EvaluationProtocolError(f"receipt field {name!r} must be a string")
            return value

        def required_bool(name: str) -> bool:
            value = payload[name]
            if type(value) is not bool:
                raise EvaluationProtocolError(f"receipt field {name!r} must be Boolean")
            return value

        def optional_int(name: str) -> int | None:
            value = payload[name]
            if value is None:
                return None
            if type(value) is not int:
                raise EvaluationProtocolError(f"receipt field {name!r} must be an integer or null")
            return value

        def required_int(name: str) -> int:
            value = optional_int(name)
            if value is None:
                raise EvaluationProtocolError(f"receipt field {name!r} cannot be null")
            return value

        def optional_float(name: str) -> float | None:
            value = payload[name]
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvaluationProtocolError(f"receipt field {name!r} must be numeric or null")
            converted = float(value)
            if not math.isfinite(converted):
                raise EvaluationProtocolError(f"receipt field {name!r} must be finite")
            return converted

        work_payload = payload["work"]
        work: WorkVector | None
        if work_payload is None:
            work = None
        elif isinstance(work_payload, dict) and set(work_payload) == set(WORK_FIELDS):
            if any(type(work_payload[name]) is not int for name in WORK_FIELDS):
                raise EvaluationProtocolError("work coordinates must be integers")
            work = WorkVector(**{name: work_payload[name] for name in WORK_FIELDS})
            if not work.is_nonnegative:
                raise EvaluationProtocolError("work coordinates cannot be negative")
        else:
            raise EvaluationProtocolError("work receipt has the wrong fields")
        program_value = payload["program_digest"]
        validation_value = payload["validation_digest"]
        chain_sizes_value = payload["chain_sizes"]
        if chain_sizes_value is not None and (
            not isinstance(chain_sizes_value, list)
            or any(type(value) is not int for value in chain_sizes_value)
        ):
            raise EvaluationProtocolError(
                "receipt field 'chain_sizes' must be an integer array or null"
            )
        if program_value is not None and not isinstance(program_value, str):
            raise EvaluationProtocolError("receipt field 'program_digest' must be a string or null")
        if validation_value is not None and not isinstance(validation_value, str):
            raise EvaluationProtocolError(
                "receipt field 'validation_digest' must be a string or null"
            )
        outcome = cls(
            instance=required_string("instance"),
            lineage=required_string("lineage"),
            returned_valid=required_bool("returned_valid"),
            utility=optional_float("utility"),
            qubits=optional_int("qubits"),
            max_chain=optional_int("max_chain"),
            decisions=required_int("decisions"),
            selected_strength=optional_float("selected_strength"),
            reason=required_string("reason"),
            repetition=required_int("repetition"),
            episode_seed=optional_int("episode_seed"),
            evaluator_seed=optional_int("evaluator_seed"),
            program_digest=program_value,
            selected_strength_index=optional_int("selected_strength_index"),
            evaluator_hits=optional_int("evaluator_hits"),
            evaluator_reads=optional_int("evaluator_reads"),
            work=work,
            validation_digest=validation_value,
            broken_chain_fraction=optional_float("broken_chain_fraction"),
            mean_energy_residual=optional_float("mean_energy_residual"),
            online_seconds=optional_float("online_seconds"),
            evaluator_seconds=optional_float("evaluator_seconds"),
            controller_calls=optional_int("controller_calls"),
            controller_seconds=optional_float("controller_seconds"),
            outcome_kind=required_string("outcome_kind"),
            population_eligible=required_bool("population_eligible"),
            chain_sizes=(None if chain_sizes_value is None else tuple(chain_sizes_value)),
            overlap_events=optional_int("overlap_events"),
            overlap_decisions=optional_int("overlap_decisions"),
            repair_attempts=optional_int("repair_attempts"),
            repair_successes=optional_int("repair_successes"),
            candidate_states=optional_int("candidate_states"),
            legal_actions_total=optional_int("legal_actions_total"),
            legal_opcode_types_total=optional_int("legal_opcode_types_total"),
        )
        outcome.validate_receipt(require_complete=require_complete)
        recorded_r99 = optional_int("fixed_program_r99")
        if recorded_r99 is not None and recorded_r99 <= 0:
            raise EvaluationProtocolError("fixed-program R99 must be positive or null")
        if recorded_r99 != outcome.fixed_program_r99:
            raise EvaluationProtocolError("fixed-program R99 differs from evaluator counts")
        return outcome


Controller = Callable[[object, np.random.Generator], int]

RESOURCE_FIRST_CONTROLLER_VERSION = "classical-resource-first-v1"
QUALITY_AWARE_CONTROLLER_VERSION = "classical-quality-aware-v1"
QUALITY_AWARE_WEIGHTS = MappingProxyType(
    {
        "weighted_logical_contacts": 4.0,
        "logical_contact_multiplicity": 0.75,
        "programmed_chain_cycle_rank": 0.5,
        "missing_logical_contacts": -20.0,
        "overlap_conflicts": -12.0,
        "overlap_excess": -8.0,
        "unique_qubits": -0.12,
        "longest_chain": -0.35,
        "application_work": -1e-4,
    }
)


def baseline_method_metadata() -> dict[str, dict[str, object]]:
    """Versioned information contracts for the built-in same-support controls.

    These are action *selectors*, not independent proposal systems.  Every selector receives
    the already charged, fully materialised batch and exact mask produced by ``EmbeddingEnv``.
    The metadata is deliberately explicit that no final evaluator feedback is available.
    Stock minorminer is therefore not listed here: it is an external-system baseline whose
    native search cannot honestly be represented as selection over this action batch.
    """

    shared = {
        "support_contract": "environment-materialised-masked-actions",
        "online_evaluator_feedback": False,
        "evaluator_oracle": False,
        "controller_timing_scope": (
            "provided-action-selection-callable-only; excludes environment preparation"
        ),
    }
    return {
        "return_initial": {
            **shared,
            "method_id": "return-authenticated-initial-v1",
            "selection_rule": "first-legal-protected-commit",
            "tie_break": "support-order",
        },
        "random_masked": {
            **shared,
            "method_id": "uniform-random-masked-v1",
            "selection_rule": "uniform-over-legal-actions",
            "tie_break": "registered-episode-rng",
        },
        "classical_resource_first": {
            **shared,
            "method_id": RESOURCE_FIRST_CONTROLLER_VERSION,
            "selection_rule": "feasibility-then-resource-lexicographic",
            "state_score_order": [
                "missing-logical-contact-delta",
                "overlap-conflict-delta",
                "overlap-excess-delta",
                "unique-qubit-delta",
                "longest-chain-delta",
                "application-work",
            ],
            "move_acceptance": "strict-defect-fix-or-nonworsening-strict-resource-gain",
            "tie_break": "uniform-random-exact-ties-registered-episode-rng",
        },
        "classical_quality_aware": {
            **shared,
            "method_id": QUALITY_AWARE_CONTROLLER_VERSION,
            "selection_rule": "deployment-structural-programming-proxy",
            "proxy_weights": dict(QUALITY_AWARE_WEIGHTS),
            "move_acceptance": "strict-positive-local-proxy-delta",
            "tie_break": "stable-payload-key",
        },
    }


def _legal_indices(decision: object) -> list[int]:
    candidates = getattr(decision, "candidates", ())
    mask = getattr(decision, "legal_mask", ())
    if len(candidates) != len(mask):
        raise EvaluationProtocolError("controller received inconsistent candidates and mask")
    legal = [index for index, admissible in enumerate(mask) if admissible]
    if not legal:
        raise EvaluationProtocolError("controller received an all-false action support")
    return legal


def _packed_value(
    matrix: np.ndarray,
    row: int,
    slot: int,
    width: int,
    *,
    default: float = 0.0,
) -> float:
    """Read one public value/knownness slot, failing closed on malformed tensors."""

    values = np.asarray(matrix)
    if values.ndim != 2 or not 0 <= row < values.shape[0] or values.shape[1] < 2 * width:
        raise EvaluationProtocolError("controller received a malformed deployment tensor")
    if not 0 <= slot < width:
        raise AssertionError("controller feature slot lies outside its registered schema")
    known = float(values[row, width + slot])
    value = float(values[row, slot])
    if not math.isfinite(known) or not math.isfinite(value):
        raise EvaluationProtocolError("controller received a nonfinite deployment tensor")
    return value if known > 0.5 else default


def _inverse_count(value: float) -> float:
    return math.expm1(max(0.0, value))


def _inverse_signed(value: float) -> float:
    return math.copysign(math.expm1(abs(value)), value)


def _action_value(decision: object, index: int, slot: int) -> float:
    return _packed_value(decision.observation.actions, index, slot, N_ACTION)


def _archive_value(decision: object, archive_index: int, slot: int) -> float:
    return _packed_value(decision.observation.archive, archive_index, slot, N_ARCHIVE)


def _choose_minimum(
    decision: object,
    indices: Sequence[int],
    score: Callable[[int], tuple[float, ...]],
    rng: np.random.Generator,
    *,
    random_ties: bool,
) -> int:
    if not indices:
        raise EvaluationProtocolError("classical controller has no candidate to select")
    scored = [(score(index), index) for index in indices]
    best = min(item[0] for item in scored)
    tied = [index for value, index in scored if value == best]
    if random_ties and len(tied) > 1:
        return int(rng.choice(np.asarray(tied, dtype=np.int64)))
    return min(tied, key=lambda index: (decision.candidates[index].payload_key, index))


def resource_first_controller(decision: object, rng: np.random.Generator) -> int:
    """Classical feasibility/resource greedy control over the exact masked support.

    A state-changing action is taken only when it removes a structural defect or strictly
    reduces ``(unique qubits, longest chain)`` without worsening overlap/contact defects.
    Otherwise the controller commits the least-resource admissible archive entry.  Exact
    score ties are sampled with the registered episode RNG; all non-tie behaviour is
    deterministic.  This controller never reads ``exact_state`` or an evaluator outcome.
    """

    legal = _legal_indices(decision)
    state_actions = [index for index in legal if decision.candidates[index].changes_workspace]
    commits = [index for index in legal if decision.candidates[index].opcode.value == "COMMIT"]

    def state_score(index: int) -> tuple[float, ...]:
        # Slots are Appendix A.7: missing contacts, conflicts, excess, Q, Lmax and work.
        return (
            _inverse_signed(_action_value(decision, index, 9)),
            _inverse_signed(_action_value(decision, index, 8)),
            _inverse_signed(_action_value(decision, index, 7)),
            _inverse_signed(_action_value(decision, index, 5)),
            _inverse_signed(_action_value(decision, index, 6)),
            _inverse_count(_action_value(decision, index, 14)),
        )

    improving: list[int] = []
    for index in state_actions:
        missing, conflicts, excess, qubits, longest, _work = state_score(index)
        defects_do_not_worsen = missing <= 0.0 and conflicts <= 0.0 and excess <= 0.0
        fixes_defect = missing < 0.0 or conflicts < 0.0 or excess < 0.0
        saves_resource = qubits < 0.0 or (qubits == 0.0 and longest < 0.0)
        if fixes_defect or (defects_do_not_worsen and saves_resource):
            improving.append(index)
    if improving:
        return _choose_minimum(decision, improving, state_score, rng, random_ties=True)

    if commits:

        def commit_score(index: int) -> tuple[float, ...]:
            archive_index = decision.candidates[index].archive_ref
            if archive_index is None:
                raise EvaluationProtocolError("COMMIT candidate omitted its archive reference")
            return (
                _inverse_count(_archive_value(decision, archive_index, 3)),
                _inverse_count(_archive_value(decision, archive_index, 4)),
                _inverse_count(_archive_value(decision, archive_index, 0)),
                _inverse_count(_action_value(decision, index, 14)),
            )

        return _choose_minimum(decision, commits, commit_score, rng, random_ties=True)

    # Construction-mode fallback: progress by the same lexicographic score; STOP remains a
    # final fallback and is never disguised as a successful resource choice.
    return _choose_minimum(decision, legal, state_score, rng, random_ties=True)


def _factor_ids(observation: object, action_index: int, role: int) -> list[int]:
    links = np.asarray(observation.index_factor_action, dtype=np.int64)
    roles = np.asarray(observation.factor_roles, dtype=np.int64)
    if links.size == 0:
        return []
    if links.ndim != 2 or links.shape[0] != 2:
        raise EvaluationProtocolError("action-factor incidence has the wrong shape")
    return [
        int(factor)
        for factor, action in links.T
        if int(action) == action_index and roles[int(factor)] == role
    ]


def _archive_factor_ids(observation: object, archive_index: int) -> list[int]:
    links = np.asarray(observation.index_factor_archive, dtype=np.int64)
    roles = np.asarray(observation.factor_roles, dtype=np.int64)
    if links.size == 0:
        return []
    if links.ndim != 2 or links.shape[0] != 2:
        raise EvaluationProtocolError("archive-factor incidence has the wrong shape")
    return [
        int(factor)
        for factor, archive in links.T
        if int(archive) == archive_index and roles[int(factor)] == ROLE_ARCHIVE
    ]


def _programming_statistics(
    observation: object,
    factors: Sequence[int],
) -> tuple[float, float, float]:
    """Return weighted contacts, contact multiplicity and intra-chain cycle rank."""

    factor_set = set(factors)
    if not factor_set:
        return 0.0, 0.0, 0.0
    contact_weight = 0.0
    contacts = 0.0
    use_factors = np.asarray(observation.index_edge_use_factor, dtype=np.int64)
    use_roles = np.asarray(observation.edge_use_roles, dtype=np.int64)
    if use_factors.shape != use_roles.shape:
        raise EvaluationProtocolError("edge-use factor and role arrays have different shapes")
    for use_index, (factor, role) in enumerate(zip(use_factors, use_roles, strict=True)):
        if int(factor) not in factor_set or int(role) != EDGE_USE_LOGICAL:
            continue
        contacts += 1.0
        scaled_abs_j = _packed_value(
            observation.edge_use_logical,
            use_index,
            1,
            N_LEDGE,
        )
        contact_weight += _inverse_count(scaled_abs_j)

    cycle_rank = 0.0
    for factor in factor_set:
        chain_size = _inverse_count(_packed_value(observation.factors, factor, 1, N_FACTOR))
        internal_edges = _inverse_count(_packed_value(observation.factors, factor, 3, N_FACTOR))
        cycle_rank += max(0.0, internal_edges - max(0.0, chain_size - 1.0))
    return contact_weight, contacts, cycle_rank


def _quality_delta(decision: object, index: int) -> float:
    observation = decision.observation
    old_stats = _programming_statistics(
        observation,
        _factor_ids(observation, index, ROLE_OLD),
    )
    new_stats = _programming_statistics(
        observation,
        _factor_ids(observation, index, ROLE_NEW),
    )
    weighted_contacts = new_stats[0] - old_stats[0]
    contacts = new_stats[1] - old_stats[1]
    cycle_rank = new_stats[2] - old_stats[2]
    delta_missing = _inverse_signed(_action_value(decision, index, 9))
    delta_conflicts = _inverse_signed(_action_value(decision, index, 8))
    delta_excess = _inverse_signed(_action_value(decision, index, 7))
    delta_qubits = _inverse_signed(_action_value(decision, index, 5))
    delta_longest = _inverse_signed(_action_value(decision, index, 6))
    work = _inverse_count(_action_value(decision, index, 14))
    # Frozen, monotone pilot proxy.  Contact evidence dominates modest resource growth while
    # hard validity and budget constraints remain solely owned by the environment.
    return (
        QUALITY_AWARE_WEIGHTS["weighted_logical_contacts"] * weighted_contacts
        + QUALITY_AWARE_WEIGHTS["logical_contact_multiplicity"] * contacts
        + QUALITY_AWARE_WEIGHTS["programmed_chain_cycle_rank"] * cycle_rank
        + QUALITY_AWARE_WEIGHTS["missing_logical_contacts"] * delta_missing
        + QUALITY_AWARE_WEIGHTS["overlap_conflicts"] * delta_conflicts
        + QUALITY_AWARE_WEIGHTS["overlap_excess"] * delta_excess
        + QUALITY_AWARE_WEIGHTS["unique_qubits"] * delta_qubits
        + QUALITY_AWARE_WEIGHTS["longest_chain"] * delta_longest
        + QUALITY_AWARE_WEIGHTS["application_work"] * work
    )


def _archive_quality(decision: object, index: int) -> float:
    archive_index = decision.candidates[index].archive_ref
    if archive_index is None:
        raise EvaluationProtocolError("COMMIT candidate omitted its archive reference")
    weighted_contacts, contacts, cycle_rank = _programming_statistics(
        decision.observation,
        _archive_factor_ids(decision.observation, archive_index),
    )
    qubits = _inverse_count(_archive_value(decision, archive_index, 3))
    longest = _inverse_count(_archive_value(decision, archive_index, 4))
    return (
        QUALITY_AWARE_WEIGHTS["weighted_logical_contacts"] * weighted_contacts
        + QUALITY_AWARE_WEIGHTS["logical_contact_multiplicity"] * contacts
        + QUALITY_AWARE_WEIGHTS["programmed_chain_cycle_rank"] * cycle_rank
        + QUALITY_AWARE_WEIGHTS["unique_qubits"] * qubits
        + QUALITY_AWARE_WEIGHTS["longest_chain"] * longest
    )


def quality_aware_controller(decision: object, rng: np.random.Generator) -> int:
    """Deployment-only structural/programming LNS scorer on the shared action batch.

    The frozen score values weighted logical-contact multiplicity and programmed intra-chain
    cycle redundancy, penalises unresolved structural defects, and applies small resource/work
    costs.  It never samples a candidate, reads a target energy, or observes reward.  A move is
    made only for a strictly positive local proxy delta; otherwise the best archive is committed.
    """

    del rng  # Every tie is resolved by stable payload identity for this deterministic arm.
    legal = _legal_indices(decision)
    state_actions = [index for index in legal if decision.candidates[index].changes_workspace]
    state_scores = {index: _quality_delta(decision, index) for index in state_actions}
    positive = [index for index in state_actions if state_scores[index] > 1e-12]
    if positive:
        best_value = max(state_scores[index] for index in positive)
        tied = [index for index in positive if state_scores[index] == best_value]
        return min(tied, key=lambda index: (decision.candidates[index].payload_key, index))

    commits = [index for index in legal if decision.candidates[index].opcode.value == "COMMIT"]
    if commits:
        archive_scores = {index: _archive_quality(decision, index) for index in commits}
        best_value = max(archive_scores.values())
        tied = [index for index in commits if archive_scores[index] == best_value]
        return min(tied, key=lambda index: (decision.candidates[index].payload_key, index))

    # Only construction mode can lack a commit. Prefer the highest proxy delta, with STOP as
    # an ordinary masked candidate rather than a fabricated valid return.
    if state_actions:
        best_value = max(state_scores.values())
        tied = [index for index in state_actions if state_scores[index] == best_value]
        return min(tied, key=lambda index: (decision.candidates[index].payload_key, index))
    return min(legal, key=lambda index: (decision.candidates[index].payload_key, index))


def random_masked_controller(decision, rng: np.random.Generator) -> int:
    legal = _legal_indices(decision)
    return int(rng.choice(legal))


def first_commit_controller(decision, rng: np.random.Generator) -> int:
    """The return-initial baseline: commit the protected initializer immediately."""

    for index, candidate in enumerate(decision.candidates):
        if candidate.opcode.value == "COMMIT" and decision.legal_mask[index]:
            return index
    return random_masked_controller(decision, rng)


def torch_controller(model, device=None, greedy: bool = False) -> Controller:
    """Deployment rule of Algorithm 2: categorical sampling at temperature one."""

    import torch

    def _controller(decision, rng: np.random.Generator) -> int:
        with torch.no_grad():
            out = model.forward_single(decision.observation, device)
        logp = out.masked_log_probs.detach().cpu().numpy()
        finite = np.isfinite(logp)
        if not finite.any():
            raise EvaluationProtocolError("model produced an all-false/nonfinite support")
        probs = np.zeros_like(logp)
        probs[finite] = np.exp(logp[finite])
        total = float(probs.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise EvaluationProtocolError("model produced invalid categorical probabilities")
        probs /= total
        return int(np.argmax(probs)) if greedy else int(rng.choice(len(probs), p=probs))

    return _controller


def _derived_seed(base_seed: int, domain: str, task: EmbeddingTask, repetition: int) -> int:
    payload = json.dumps(
        {
            "base_seed": base_seed,
            "domain": domain,
            "instance": task.name,
            "lineage": task.lineage or task.name,
            "repetition": repetition,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31)


def _typed_identity(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def program_digest(
    program: Program,
    chains: Mapping[object, Sequence[object]] | None = None,
) -> str:
    """Digest the sampled physical program and, when supplied, its decoder ownership."""

    def pair(left: object, right: object) -> tuple[str, str]:
        a, b = _typed_identity(left), _typed_identity(right)
        return (a, b) if a <= b else (b, a)

    payload = {
        "strength": program.strength,
        "strength_index": program.strength_index,
        "scale": program.scale,
        "offset": program.offset,
        "h_phys": sorted(
            ((_typed_identity(qubit), value) for qubit, value in program.h_phys.items())
        ),
        "j_phys": sorted(((*pair(a, b), value) for (a, b), value in program.j_phys.items())),
        "chain_edges": sorted(
            (
                _typed_identity(owner),
                sorted(pair(a, b) for a, b in edges),
            )
            for owner, edges in program.chain_edges.items()
        ),
        "contact_counts": sorted(
            ((*pair(left, right), count) for (left, right), count in program.contact_counts.items())
        ),
        "decoder_chains": (
            None
            if chains is None
            else sorted(
                (
                    _typed_identity(owner),
                    sorted(_typed_identity(qubit) for qubit in chain),
                )
                for owner, chain in chains.items()
            )
        ),
    }
    return stable_digest(payload)


def run_controller(
    tasks: Sequence[EmbeddingTask],
    ctx: Context,
    controller: Controller,
    *,
    initializer: Initializer,
    selector: StrengthSelector,
    reward_reads: int | None = None,
    seed: int = 0,
    repetitions: int = 1,
    max_steps: int = 64,
) -> list[EpisodeOutcome]:
    """Evaluate one controller with fresh external reads and complete raw receipts.

    Ground energy is touched only after deployment-mode search has returned a fixed program.
    Episode and evaluator RNG streams are domain-separated and depend on the exact pair key,
    so independently run arms use identical registered seeds without depending on task order.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if max_steps < ctx.caps.decisions:
        raise ValueError("max_steps cannot truncate the registered environment decision budget")
    reads = ctx.audit_reads if reward_reads is None else reward_reads
    if isinstance(reads, bool) or not isinstance(reads, int) or reads <= 0:
        raise ValueError("reward_reads must be a positive integer")
    if reads > ctx.caps.evaluator_reads:
        raise ValueError("final evaluator reads exceed the registered evaluator-read cap")
    identities = [(task.lineage or task.name, task.name) for task in tasks]
    if len(identities) != len(set(identities)):
        raise EvaluationProtocolError(
            "evaluation population contains duplicate instance identities"
        )
    for task in tasks:
        if task.ground_energy is None or not math.isfinite(task.ground_energy):
            raise EvaluationProtocolError(
                f"final IF-Q3 evaluation needs a certified finite target for {task.name!r}"
            )

    outcomes: list[EpisodeOutcome] = []
    for repetition in range(repetitions):
        for task in tasks:
            episode_seed = _derived_seed(seed, "policy", task, repetition)
            evaluator_seed = _derived_seed(seed, "final-evaluator", task, repetition)
            env = EmbeddingEnv(
                task,
                ctx,
                mode=Mode.IMPROVEMENT,
                initializer=task_initializer(task, initializer),
                selector=selector,
                # Deployment performs no in-environment reward sample.  Keep the registered
                # N_est reserve here; the independent audit block below is charged exactly once.
                reward_reads=ctx.n_est_reads,
                improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
                seed=episode_seed,
            )
            rng = np.random.default_rng(episode_seed)
            started = time.perf_counter()
            overlap_active = False
            overlap_events = 0
            overlap_decisions = 0
            repair_attempts = 0
            repair_successes = 0
            candidate_states = 0
            legal_actions_total = 0
            legal_opcode_types_total = 0
            result = env.reset(episode_seed)
            if isinstance(result, InitFailureRecord):
                outcomes.append(
                    EpisodeOutcome(
                        instance=task.name,
                        lineage=task.lineage or task.name,
                        returned_valid=False,
                        utility=None,
                        qubits=None,
                        max_chain=None,
                        decisions=0,
                        selected_strength=None,
                        reason=result.reason,
                        repetition=repetition,
                        episode_seed=episode_seed,
                        work=result.work,
                        validation_digest=stable_digest(
                            {"kind": INITIALIZATION_FAILURE, "reason": result.reason}
                        ),
                        online_seconds=time.perf_counter() - started,
                        controller_calls=0,
                        controller_seconds=0.0,
                        outcome_kind=INITIALIZATION_FAILURE,
                        population_eligible=False,
                        overlap_events=0,
                        overlap_decisions=0,
                        repair_attempts=0,
                        repair_successes=0,
                        candidate_states=0,
                        legal_actions_total=0,
                        legal_opcode_types_total=0,
                    )
                )
                continue
            steps = 0
            controller_calls = 0
            controller_seconds = 0.0
            while not isinstance(result, TerminalRecord) and steps < max_steps:
                candidate_states += 1
                legal_candidates = [
                    candidate
                    for candidate, legal in zip(result.candidates, result.legal_mask, strict=True)
                    if legal
                ]
                legal_actions_total += len(legal_candidates)
                legal_opcode_types_total += len(
                    {candidate.opcode for candidate in legal_candidates}
                )
                current_overlap = bool(
                    env.state is not None
                    and any(value > 1 for value in occupancy(env.state.chains).values())
                )
                if current_overlap:
                    overlap_decisions += 1
                    if not overlap_active:
                        overlap_events += 1
                overlap_active = current_overlap
                controller_started = time.perf_counter()
                chosen = int(controller(result, rng))
                controller_seconds += time.perf_counter() - controller_started
                controller_calls += 1
                selected = result.candidates[chosen]
                step = env.step(result, chosen, evaluate_training_reward=False)
                if selected.opcode is Opcode.REPAIR_GROUP:
                    repair_attempts += 1
                    if repair_target_satisfied(selected, env):
                        repair_successes += 1
                result = step.next_decision_or_terminal
                steps += 1
            if not isinstance(result, TerminalRecord):
                raise EvaluationProtocolError(
                    f"live task {task.name!r} reached the external max_steps cut; "
                    "collector truncation is disabled"
                )
            online_seconds = time.perf_counter() - started
            validation_digest = stable_digest(result.validation_receipt)
            if not result.returned_valid:
                outcome = EpisodeOutcome(
                    instance=task.name,
                    lineage=task.lineage or task.name,
                    returned_valid=False,
                    utility=0.0,
                    qubits=None,
                    max_chain=None,
                    decisions=steps,
                    selected_strength=None,
                    reason=result.terminal_reason.value,
                    repetition=repetition,
                    episode_seed=episode_seed,
                    work=result.cumulative_work,
                    validation_digest=validation_digest,
                    online_seconds=online_seconds,
                    controller_calls=controller_calls,
                    controller_seconds=controller_seconds,
                    overlap_events=overlap_events,
                    overlap_decisions=overlap_decisions,
                    repair_attempts=repair_attempts,
                    repair_successes=repair_successes,
                    candidate_states=candidate_states,
                    legal_actions_total=legal_actions_total,
                    legal_opcode_types_total=legal_opcode_types_total,
                )
                outcome.validate_receipt(require_complete=True)
                outcomes.append(outcome)
                continue

            if not isinstance(result.selected_program, Program) or result.embedding is None:
                raise EvaluationProtocolError(
                    "valid terminal omitted its exact program or embedding"
                )
            if result.selected_index != result.selected_program.strength_index:
                raise EvaluationProtocolError(
                    "terminal selector index does not match selected program"
                )
            evaluator_started = time.perf_counter()
            block = sample_program(
                result.selected_program,
                result.embedding,
                task.problem,
                task.ground_energy,
                num_reads=reads,
                seed=evaluator_seed,
                num_sweeps=ctx.num_sweeps,
                beta_range=ctx.beta_range,
            )
            evaluator_seconds = time.perf_counter() - evaluator_started
            if block.strength_index != result.selected_index or block.reads != reads:
                raise EvaluationProtocolError(
                    "evaluator returned counts for the wrong strength or read budget"
                )
            total_work = result.cumulative_work + WorkVector(evaluator_reads=block.reads)
            if not total_work.fits_in(ctx.caps):
                raise EvaluationProtocolError(
                    "final evaluator block exceeds the registered work cap"
                )
            chains = result.embedding
            outcome = EpisodeOutcome(
                returned_embedding={n: frozenset(c) for n, c in chains.items()},
                instance=task.name,
                lineage=task.lineage or task.name,
                returned_valid=True,
                utility=block.rate,
                qubits=sum(len(chain) for chain in chains.values()),
                max_chain=max(len(chain) for chain in chains.values()),
                decisions=steps,
                selected_strength=result.selected_strength,
                reason=result.terminal_reason.value,
                repetition=repetition,
                episode_seed=episode_seed,
                evaluator_seed=evaluator_seed,
                program_digest=program_digest(result.selected_program, result.embedding),
                selected_strength_index=result.selected_index,
                evaluator_hits=block.hits,
                evaluator_reads=block.reads,
                work=total_work,
                validation_digest=validation_digest,
                broken_chain_fraction=block.broken_fraction,
                mean_energy_residual=block.mean_residual,
                online_seconds=online_seconds,
                evaluator_seconds=evaluator_seconds,
                controller_calls=controller_calls,
                controller_seconds=controller_seconds,
                chain_sizes=tuple(sorted(len(chain) for chain in chains.values())),
                overlap_events=overlap_events,
                overlap_decisions=overlap_decisions,
                repair_attempts=repair_attempts,
                repair_successes=repair_successes,
                candidate_states=candidate_states,
                legal_actions_total=legal_actions_total,
                legal_opcode_types_total=legal_opcode_types_total,
            )
            outcome.validate_receipt(require_complete=True)
            outcomes.append(outcome)
    return outcomes


def _analysis_outcome(outcome: EpisodeOutcome) -> None:
    """The subset of receipt validity needed by legacy in-memory comparisons."""

    if not outcome.instance:
        raise EvaluationProtocolError("outcome has an empty instance identifier")
    if isinstance(outcome.repetition, bool) or outcome.repetition < 0:
        raise EvaluationProtocolError("outcome has an invalid repetition")
    if not outcome.population_eligible:
        if outcome.outcome_kind != INITIALIZATION_FAILURE:
            raise EvaluationProtocolError("ineligible outcome is not an initializer failure")
        return
    if outcome.utility is None or not math.isfinite(outcome.utility):
        raise EvaluationProtocolError("eligible outcome has no finite utility")
    if not 0.0 <= outcome.utility <= 1.0:
        raise EvaluationProtocolError("utility lies outside [0,1]")
    if not outcome.returned_valid and outcome.utility != 0.0:
        raise EvaluationProtocolError("ordinary failures must remain in utility as zero")


def _index_pairs(outcomes: Sequence[EpisodeOutcome]) -> dict[tuple[str, str, int], EpisodeOutcome]:
    indexed: dict[tuple[str, str, int], EpisodeOutcome] = {}
    for outcome in outcomes:
        _analysis_outcome(outcome)
        if outcome.pair_key in indexed:
            raise EvaluationProtocolError(f"duplicate pair key {outcome.pair_key!r}")
        indexed[outcome.pair_key] = outcome
    return indexed


@dataclass(frozen=True)
class PairedEndpoint:
    n_lineages: int
    delta_utility: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    delta_feasibility: float
    feasibility_ci_low: float
    noninferior: bool
    mean_reference: float
    mean_treatment: float
    qubit_ratio: float | None
    r99_reference: None = None
    r99_treatment: None = None
    n_pairs: int = 0
    ties: int = 0
    feasibility_ci_high: float = 0.0
    n_joint_valid_pairs: int = 0
    conditional_delta_utility: float | None = None
    delta_decisions: float | None = None
    work_delta: Mapping[str, float] | None = None

    def as_dict(self) -> dict[str, object]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        if self.work_delta is not None:
            result["work_delta"] = dict(self.work_delta)
        return result


def _lineage_weights(lineages: Sequence[str], supplied: Mapping[str, float] | None) -> np.ndarray:
    if supplied is None:
        return np.full(len(lineages), 1.0 / len(lineages), dtype=float)
    missing = set(lineages) - set(supplied)
    if missing:
        raise EvaluationProtocolError(f"missing preregistered lineage weights: {sorted(missing)}")
    weights = np.asarray([supplied[lineage] for lineage in lineages], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise EvaluationProtocolError("lineage weights must be finite, nonnegative and nonzero")
    return weights / weights.sum()


def _bootstrap_draws(
    values: np.ndarray, weights: np.ndarray, *, count: int, rng: np.random.Generator
) -> np.ndarray:
    draws = np.empty(count, dtype=float)
    for index in range(count):
        sampled = rng.choice(len(values), size=len(values), replace=True, p=weights)
        draws[index] = float(values[sampled].mean())
    return draws


def paired_endpoint(
    treatment: Sequence[EpisodeOutcome],
    reference: Sequence[EpisodeOutcome],
    *,
    margin: float = 0.02,
    bootstrap: int = 5_000,
    seed: int = 0,
    strict_pairs: bool = False,
    require_same_seeds: bool = True,
    lineage_weights: Mapping[str, float] | None = None,
) -> PairedEndpoint:
    """Failure-aware utility difference with a paired lineage-cluster bootstrap.

    ``strict_pairs=True`` is mandatory for publication comparisons.  The permissive default
    exists only for the historical lightweight API; :func:`compare_arms` always enables the
    strict protocol.
    """

    if not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
        raise ValueError("margin must be finite and in [0,1]")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap <= 0:
        raise ValueError("bootstrap must be a positive integer")
    arm_t, arm_r = _index_pairs(treatment), _index_pairs(reference)
    keys_t, keys_r = set(arm_t), set(arm_r)
    if strict_pairs and keys_t != keys_r:
        missing_t = sorted(keys_r - keys_t)
        missing_r = sorted(keys_t - keys_r)
        raise EvaluationProtocolError(
            "missing paired receipts: "
            f"treatment_missing={missing_t[:5]}, reference_missing={missing_r[:5]}"
        )
    paired_keys = sorted(keys_t & keys_r)
    if not paired_keys:
        raise EvaluationProtocolError("no shared instance/repetition pair between the two arms")

    eligible_pairs: list[tuple[EpisodeOutcome, EpisodeOutcome]] = []
    for key in paired_keys:
        row_t, row_r = arm_t[key], arm_r[key]
        if row_t.population_eligible != row_r.population_eligible:
            raise EvaluationProtocolError(f"conditional-population mismatch at pair {key!r}")
        if require_same_seeds:
            if row_t.episode_seed != row_r.episode_seed:
                # Legacy records have ``None`` in both arms and remain comparable.
                raise EvaluationProtocolError(f"policy seed mismatch at pair {key!r}")
            if row_t.returned_valid and row_r.returned_valid:
                if row_t.evaluator_seed != row_r.evaluator_seed:
                    raise EvaluationProtocolError(f"evaluator seed mismatch at pair {key!r}")
        if row_t.population_eligible:
            eligible_pairs.append((row_t, row_r))
    if not eligible_pairs:
        raise EvaluationProtocolError("all matched attempts are initializer system failures")

    by_lineage: dict[str, list[tuple[EpisodeOutcome, EpisodeOutcome]]] = {}
    for pair in eligible_pairs:
        by_lineage.setdefault(pair[0].lineage or pair[0].instance, []).append(pair)
    lineages = sorted(by_lineage)
    weights = _lineage_weights(lineages, lineage_weights)
    treatment_means = np.asarray(
        [np.mean([float(row_t.utility) for row_t, _ in by_lineage[key]]) for key in lineages]
    )
    reference_means = np.asarray(
        [np.mean([float(row_r.utility) for _, row_r in by_lineage[key]]) for key in lineages]
    )
    delta_utility = treatment_means - reference_means
    feasibility_delta = np.asarray(
        [
            np.mean([float(row_t.returned_valid) for row_t, _ in by_lineage[key]])
            - np.mean([float(row_r.returned_valid) for _, row_r in by_lineage[key]])
            for key in lineages
        ]
    )
    rng = np.random.default_rng(seed)
    utility_draws = _bootstrap_draws(delta_utility, weights, count=bootstrap, rng=rng)
    feasibility_draws = _bootstrap_draws(feasibility_delta, weights, count=bootstrap, rng=rng)

    joint_by_lineage: dict[str, list[tuple[EpisodeOutcome, EpisodeOutcome]]] = {}
    for row_t, row_r in eligible_pairs:
        if row_t.returned_valid and row_r.returned_valid:
            joint_by_lineage.setdefault(row_t.lineage or row_t.instance, []).append((row_t, row_r))
    conditional_delta: float | None = None
    qubit_ratio: float | None = None
    if joint_by_lineage:
        joint_lineages = sorted(joint_by_lineage)
        if lineage_weights is None:
            joint_weights = np.full(len(joint_lineages), 1.0 / len(joint_lineages))
        else:
            joint_weights = _lineage_weights(joint_lineages, lineage_weights)
        joint_t_u = np.asarray(
            [
                np.mean([float(row_t.utility) for row_t, _ in joint_by_lineage[key]])
                for key in joint_lineages
            ]
        )
        joint_r_u = np.asarray(
            [
                np.mean([float(row_r.utility) for _, row_r in joint_by_lineage[key]])
                for key in joint_lineages
            ]
        )
        conditional_delta = float(np.dot(joint_weights, joint_t_u - joint_r_u))
        joint_t_q = np.asarray(
            [
                np.mean([float(row_t.qubits) for row_t, _ in joint_by_lineage[key]])
                for key in joint_lineages
            ]
        )
        joint_r_q = np.asarray(
            [
                np.mean([float(row_r.qubits) for _, row_r in joint_by_lineage[key]])
                for key in joint_lineages
            ]
        )
        reference_qubits = float(np.dot(joint_weights, joint_r_q))
        if reference_qubits > 0.0:
            qubit_ratio = float(np.dot(joint_weights, joint_t_q) / reference_qubits)

    decision_delta = np.asarray(
        [
            np.mean([row_t.decisions - row_r.decisions for row_t, row_r in by_lineage[key]])
            for key in lineages
        ],
        dtype=float,
    )
    work_delta: dict[str, float] | None = None
    if all(row_t.work is not None and row_r.work is not None for row_t, row_r in eligible_pairs):
        work_delta = {}
        for field_name in WORK_FIELDS:
            values = np.asarray(
                [
                    np.mean(
                        [
                            getattr(row_t.work, field_name) - getattr(row_r.work, field_name)
                            for row_t, row_r in by_lineage[key]
                        ]
                    )
                    for key in lineages
                ]
            )
            work_delta[field_name] = float(np.dot(weights, values))

    feasibility_low = float(np.percentile(feasibility_draws, 5.0))
    return PairedEndpoint(
        n_lineages=len(lineages),
        delta_utility=float(np.dot(weights, delta_utility)),
        ci_low=float(np.percentile(utility_draws, 2.5)),
        ci_high=float(np.percentile(utility_draws, 97.5)),
        wins=int(np.sum(delta_utility > 0.0)),
        losses=int(np.sum(delta_utility < 0.0)),
        delta_feasibility=float(np.dot(weights, feasibility_delta)),
        feasibility_ci_low=feasibility_low,
        noninferior=bool(feasibility_low > -margin),
        mean_reference=float(np.dot(weights, reference_means)),
        mean_treatment=float(np.dot(weights, treatment_means)),
        qubit_ratio=qubit_ratio,
        n_pairs=len(eligible_pairs),
        ties=int(np.sum(delta_utility == 0.0)),
        feasibility_ci_high=float(np.percentile(feasibility_draws, 95.0)),
        n_joint_valid_pairs=sum(len(rows) for rows in joint_by_lineage.values()),
        conditional_delta_utility=conditional_delta,
        delta_decisions=float(np.dot(weights, decision_delta)),
        work_delta=None if work_delta is None else MappingProxyType(work_delta),
    )


def _mean_or_none(values: Sequence[float | int]) -> float | None:
    return float(np.mean(values)) if values else None


def _observed(values: Sequence[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def _complete_mean(values: Sequence[float | None], expected: int) -> float | None:
    observed = _observed(values)
    return float(np.mean(observed)) if expected > 0 and len(observed) == expected else None


def _complete_total(values: Sequence[float | None], expected: int) -> float | None:
    observed = _observed(values)
    return float(sum(observed)) if expected > 0 and len(observed) == expected else None


def secondary_metrics(outcomes: Sequence[EpisodeOutcome]) -> dict[str, object]:
    """Descriptive metrics with explicit denominators and no invented empty-population zeros."""

    for outcome in outcomes:
        _analysis_outcome(outcome)
    episodes = [outcome for outcome in outcomes if outcome.population_eligible]
    valid = [outcome for outcome in episodes if outcome.returned_valid]
    system = [outcome for outcome in outcomes if not outcome.population_eligible]
    online_values = [outcome.online_seconds for outcome in outcomes]
    evaluator_values = [outcome.evaluator_seconds for outcome in valid]
    controller_values = [outcome.controller_seconds for outcome in outcomes]
    controller_calls = [outcome.controller_calls for outcome in outcomes]
    broken_values = [outcome.broken_chain_fraction for outcome in valid]
    residual_values = [outcome.mean_energy_residual for outcome in valid]
    chain_rows = [outcome.chain_sizes for outcome in valid if outcome.chain_sizes is not None]
    chain_values = [size for row in chain_rows for size in row]
    mechanism_fields = (
        "overlap_events",
        "overlap_decisions",
        "repair_attempts",
        "repair_successes",
        "candidate_states",
        "legal_actions_total",
        "legal_opcode_types_total",
    )
    mechanism_complete = bool(episodes) and all(
        all(getattr(outcome, name) is not None for name in mechanism_fields) for outcome in episodes
    )
    mechanism_totals = {
        name: (
            sum(int(getattr(outcome, name)) for outcome in episodes) if mechanism_complete else None
        )
        for name in mechanism_fields
    }
    end_to_end_values: list[float | None] = []
    for outcome in outcomes:
        if outcome.online_seconds is None:
            end_to_end_values.append(None)
        elif outcome.returned_valid:
            end_to_end_values.append(
                None
                if outcome.evaluator_seconds is None
                else outcome.online_seconds + outcome.evaluator_seconds
            )
        else:
            end_to_end_values.append(outcome.online_seconds)
    online_total = _complete_total(online_values, len(outcomes))
    controller_total = _complete_total(controller_values, len(outcomes))
    r99_values = [
        outcome.fixed_program_r99 for outcome in valid if outcome.fixed_program_r99 is not None
    ]
    count_complete = [
        outcome
        for outcome in valid
        if outcome.evaluator_hits is not None and outcome.evaluator_reads is not None
    ]
    result: dict[str, object] = {
        "attempts": len(outcomes),
        "episodes": len(episodes),
        "initialization_failures": len(system),
        "initialization_failure_rate": (len(system) / len(outcomes) if outcomes else None),
        "valid_returns": len(valid),
        "valid_return_rate": (
            float(np.mean([outcome.returned_valid for outcome in episodes])) if episodes else None
        ),
        "utility_mean": _mean_or_none([float(outcome.utility) for outcome in episodes]),
        "conditional_utility": _mean_or_none([float(outcome.utility) for outcome in valid]),
        "qubits_mean": _mean_or_none([int(outcome.qubits) for outcome in valid]),
        "max_chain_mean": _mean_or_none([int(outcome.max_chain) for outcome in valid]),
        "decisions_mean": _mean_or_none([outcome.decisions for outcome in episodes]),
        "evaluator_hits_total": (
            sum(int(outcome.evaluator_hits) for outcome in valid)
            if valid and all(outcome.evaluator_hits is not None for outcome in valid)
            else None
        ),
        "evaluator_reads_total": (
            sum(int(outcome.evaluator_reads) for outcome in valid)
            if valid and all(outcome.evaluator_reads is not None for outcome in valid)
            else None
        ),
        "online_seconds_observed": len(_observed(online_values)),
        "online_seconds_mean": _complete_mean(online_values, len(outcomes)),
        "online_seconds_total": online_total,
        "evaluator_seconds_observed_valid": len(_observed(evaluator_values)),
        "evaluator_seconds_mean_valid": _complete_mean(evaluator_values, len(valid)),
        "evaluator_seconds_total_valid": _complete_total(evaluator_values, len(valid)),
        "end_to_end_seconds_observed": len(_observed(end_to_end_values)),
        "end_to_end_seconds_mean": _complete_mean(end_to_end_values, len(outcomes)),
        "end_to_end_seconds_total": _complete_total(end_to_end_values, len(outcomes)),
        "controller_timing_scope": (
            "provided-action-selection-callable-only; excludes environment preparation"
            if outcomes and len(_observed(controller_values)) == len(outcomes)
            else None
        ),
        "controller_seconds_observed": len(_observed(controller_values)),
        "controller_seconds_mean": _complete_mean(controller_values, len(outcomes)),
        "controller_seconds_total": controller_total,
        "controller_calls_total": (
            sum(int(value) for value in controller_calls)
            if outcomes and all(type(value) is int and value >= 0 for value in controller_calls)
            else None
        ),
        "controller_share_of_online_seconds": (
            controller_total / online_total
            if controller_total is not None and online_total is not None and online_total > 0.0
            else None
        ),
        "broken_chain_fraction_observed_valid": len(_observed(broken_values)),
        "broken_chain_fraction_mean_valid": _complete_mean(broken_values, len(valid)),
        "mean_energy_residual_observed_valid": len(_observed(residual_values)),
        "mean_energy_residual_mean_valid": _complete_mean(residual_values, len(valid)),
        "chain_sizes_observed_valid": len(chain_rows),
        "chain_size_count": (
            len(chain_values) if valid and len(chain_rows) == len(valid) else None
        ),
        "chain_size_distribution": (
            {
                "min": float(np.min(chain_values)),
                "q25": float(np.quantile(chain_values, 0.25)),
                "median": float(np.median(chain_values)),
                "q75": float(np.quantile(chain_values, 0.75)),
                "max": float(np.max(chain_values)),
                "mean": float(np.mean(chain_values)),
            }
            if valid and len(chain_rows) == len(valid) and chain_values
            else None
        ),
        "mechanism_telemetry_observed_episodes": (len(episodes) if mechanism_complete else 0),
        "overlap_events_total": mechanism_totals["overlap_events"],
        "overlap_decisions_total": mechanism_totals["overlap_decisions"],
        "repair_attempts_total": mechanism_totals["repair_attempts"],
        "repair_successes_total": mechanism_totals["repair_successes"],
        "repair_success_rate": (
            mechanism_totals["repair_successes"] / mechanism_totals["repair_attempts"]
            if mechanism_complete and mechanism_totals["repair_attempts"] > 0
            else None
        ),
        "candidate_states_total": mechanism_totals["candidate_states"],
        "legal_actions_total": mechanism_totals["legal_actions_total"],
        "legal_opcode_types_total": mechanism_totals["legal_opcode_types_total"],
        "legal_actions_mean_per_candidate_state": (
            mechanism_totals["legal_actions_total"] / mechanism_totals["candidate_states"]
            if mechanism_complete and mechanism_totals["candidate_states"] > 0
            else None
        ),
        "legal_opcode_types_mean_per_candidate_state": (
            mechanism_totals["legal_opcode_types_total"] / mechanism_totals["candidate_states"]
            if mechanism_complete and mechanism_totals["candidate_states"] > 0
            else None
        ),
        "fixed_program_r99_scope": (
            "per-returned-fixed-program point-estimate read count; not population TTS"
        ),
        "fixed_program_r99_valid_program_count": len(valid),
        "fixed_program_r99_count_complete_valid": len(count_complete),
        "fixed_program_r99_defined_count": len(r99_values),
        "fixed_program_r99_zero_hit_count": sum(
            outcome.evaluator_hits == 0 for outcome in count_complete
        ),
        "fixed_program_r99_mean_defined": _mean_or_none(r99_values),
        "fixed_program_r99_median_defined": (float(np.median(r99_values)) if r99_values else None),
    }
    complete_work = [outcome.work for outcome in outcomes if outcome.work is not None]
    for field_name in WORK_FIELDS:
        result[f"work_{field_name}_mean"] = (
            _mean_or_none([getattr(work, field_name) for work in complete_work])
            if len(complete_work) == len(outcomes) and outcomes
            else None
        )
    return result


class ComparisonFamily(str, Enum):
    """Attribution rows registered by Architecture Rev2 section 6.1."""

    PROPOSAL_HEADROOM = "proposal_headroom"
    CLASSICAL_SCORING = "classical_scoring"
    SUPERVISED_REPRESENTATION = "supervised_representation"
    PPO_INCREMENT = "ppo_increment"
    PPO_WARM_START = "ppo_warm_start"
    REPRESENTATION = "representation"
    CUT_MECHANISM = "cut_mechanism"
    ACTION_GRAMMAR = "action_grammar"
    OVERLAP = "overlap"
    STRENGTH_SELECTION = "strength_selection"
    LEARNED_VS_CLASSICAL = "learned_vs_classical"
    EXTERNAL_SYSTEM = "external_system"


@dataclass(frozen=True)
class EvaluationArm:
    """Named baseline, learned method or ablation plus immutable protocol metadata."""

    name: str
    role: str
    outcomes: tuple[EpisodeOutcome, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name or not self.role:
            raise ValueError("evaluation arm name and role must be nonempty")
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ComparisonSpec:
    """Preregistered direction, matching requirements and uncertainty settings."""

    name: str
    treatment: str
    reference: str
    family: ComparisonFamily
    matched_metadata: tuple[str, ...] = ()
    margin: float = 0.02
    bootstrap: int = 5_000
    seed: int = 0
    lineage_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.name or self.treatment == self.reference:
            raise ValueError("comparison needs a name and two distinct arms")
        object.__setattr__(self, "family", ComparisonFamily(self.family))
        object.__setattr__(self, "matched_metadata", tuple(self.matched_metadata))
        if self.lineage_weights is not None:
            object.__setattr__(
                self,
                "lineage_weights",
                MappingProxyType(dict(self.lineage_weights)),
            )


@dataclass(frozen=True)
class ComparisonResult:
    spec: ComparisonSpec
    endpoint: PairedEndpoint
    treatment_metrics: Mapping[str, object]
    reference_metrics: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "comparison": self.spec.name,
            "family": self.spec.family.value,
            "treatment": self.spec.treatment,
            "reference": self.spec.reference,
            "endpoint": self.endpoint.as_dict(),
            "treatment_metrics": dict(self.treatment_metrics),
            "reference_metrics": dict(self.reference_metrics),
        }


def compare_arms(arms: Mapping[str, EvaluationArm], spec: ComparisonSpec) -> ComparisonResult:
    """Run one strict, matched baseline/ablation comparison."""

    try:
        treatment = arms[spec.treatment]
        reference = arms[spec.reference]
    except KeyError as exc:
        raise EvaluationProtocolError(f"comparison references missing arm {exc.args[0]!r}") from exc
    if treatment.name != spec.treatment or reference.name != spec.reference:
        raise EvaluationProtocolError("arm mapping key does not match its declared name")
    for key in spec.matched_metadata:
        if key not in treatment.metadata or key not in reference.metadata:
            raise EvaluationProtocolError(f"matched metadata field {key!r} is absent")
        if treatment.metadata[key] != reference.metadata[key]:
            raise EvaluationProtocolError(
                f"matched metadata field {key!r} differs: "
                f"{treatment.metadata[key]!r} != {reference.metadata[key]!r}"
            )
    for outcome in (*treatment.outcomes, *reference.outcomes):
        outcome.validate_receipt(require_complete=True)
    endpoint = paired_endpoint(
        treatment.outcomes,
        reference.outcomes,
        margin=spec.margin,
        bootstrap=spec.bootstrap,
        seed=spec.seed,
        strict_pairs=True,
        require_same_seeds=True,
        lineage_weights=spec.lineage_weights,
    )
    return ComparisonResult(
        spec=spec,
        endpoint=endpoint,
        treatment_metrics=MappingProxyType(secondary_metrics(treatment.outcomes)),
        reference_metrics=MappingProxyType(secondary_metrics(reference.outcomes)),
    )


def comparison_matrix(
    arms: Mapping[str, EvaluationArm], specs: Sequence[ComparisonSpec]
) -> dict[str, ComparisonResult]:
    """Evaluate a preregistered set of anchor, attribution and external comparisons."""

    results: dict[str, ComparisonResult] = {}
    for spec in specs:
        if spec.name in results:
            raise EvaluationProtocolError(f"duplicate comparison name {spec.name!r}")
        results[spec.name] = compare_arms(arms, spec)
    return results


def write_episode_receipts(
    path: str | Path,
    outcomes: Sequence[EpisodeOutcome],
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write canonical evaluator-only JSONL and return its SHA-256 digest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not outcomes:
        raise EvaluationProtocolError("cannot write an empty receipt batch")
    indexed = _index_pairs(outcomes)
    if len(indexed) != len(outcomes):  # Defensive; _index_pairs already rejects duplicates.
        raise EvaluationProtocolError("receipt batch contains duplicate pairs")
    for outcome in outcomes:
        outcome.validate_receipt(require_complete=True)
    rows = sorted(outcomes, key=lambda outcome: outcome.pair_key)
    content = b"".join(
        (
            json.dumps(
                outcome.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for outcome in rows
    )
    digest = hashlib.sha256(content).hexdigest()
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(destination) from None
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _strict_json(line: str, line_number: int) -> Mapping[str, object]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationProtocolError(
                    f"duplicate JSON key {key!r} on receipt line {line_number}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            line,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvaluationProtocolError(f"nonfinite JSON scalar {value!r} on line {line_number}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise EvaluationProtocolError(f"invalid receipt JSON on line {line_number}") from exc
    if not isinstance(payload, dict):
        raise EvaluationProtocolError(f"receipt line {line_number} is not an object")
    return payload


def read_episode_receipts(
    path: str | Path, *, expected_sha256: str | None = None
) -> list[EpisodeOutcome]:
    """Authenticate and strictly parse evaluator-only JSONL receipts."""

    content = Path(path).read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise EvaluationProtocolError(
            f"receipt SHA-256 mismatch: expected {expected_sha256}, observed {actual}"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationProtocolError("receipt file is not valid UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise EvaluationProtocolError("receipt file is empty")
    if any(not line.strip() for line in lines):
        raise EvaluationProtocolError("receipt file contains a blank record")
    outcomes = [
        EpisodeOutcome.from_dict(_strict_json(line, number))
        for number, line in enumerate(lines, start=1)
    ]
    if not outcomes:
        raise EvaluationProtocolError("receipt file contains no records")
    _index_pairs(outcomes)
    return outcomes

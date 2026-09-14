"""Immutable, replayable counterfactual candidate groups for quality learning.

The bank separates candidate generation from policy selection.  Every policy
sees the same unlabeled alternatives; quality labels are retained only by the
replay engine for acceptance and audit reporting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
CANDIDATE_FEATURE_NAMES_V1 = frozenset({"max_chain", "total_qubits"})
EVALUATION_PROTOCOL_FIELDS_V1 = frozenset(
    {
        "decoder",
        "noise_model",
        "reference_energy",
        "sampler",
        "sampler_version",
        "schedule",
        "seed_derivation",
    }
)
Split = Literal["train", "val", "test"]
QualityObjective = Literal["solve_probability_then_residual-v1", "residual_mean-v1"]
Matrix = tuple[tuple[float, ...], ...]


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def _frozen_value(value: Any) -> Any:
    """Return a recursively immutable canonical representation."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("frozen JSON object keys must be strings")
        return tuple((key, _frozen_value(value[key])) for key in sorted(value))
    if isinstance(value, (tuple, list)):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("frozen JSON numbers must be finite")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported frozen JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return one deterministic UTF-8 JSON encoding."""

    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_seed(root_seed: int, domain: str, *parts: Any) -> int:
    """Derive an unambiguous, domain-separated 63-bit seed."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root_seed must be an integer")
    if not isinstance(domain, str) or not domain:
        raise ValueError("domain must be a non-empty string")
    payload = {"domain": domain, "parts": [root_seed, *parts], "version": SCHEMA_VERSION}
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:8], "big") & (
        2**63 - 1
    )


def assign_split(split_unit_id: str) -> Split:
    """Assign a logical case to the canonical 70/10/20 train/val/test split."""

    if not isinstance(split_unit_id, str) or not split_unit_id:
        raise ValueError("split_unit_id must be a non-empty string")
    bucket = int(
        content_digest({"salt": "isingfold-candidate-bank-v1", "id": split_unit_id})[:8], 16
    )
    bucket %= 10_000
    if bucket < 7_000:
        return "train"
    if bucket < 8_000:
        return "val"
    return "test"


def _require_schema(value: Mapping[str, Any], name: str) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{name} requires schema_version {SCHEMA_VERSION}")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} schema fields differ: missing={missing}, unknown={unknown}")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _matrix(value: Sequence[Sequence[float]] | None, name: str) -> Matrix | None:
    if value is None:
        return None
    return tuple(tuple(_finite(item, name) for item in row) for row in value)


@dataclass(frozen=True, slots=True)
class QualityOutcome:
    """Mean quality at one chain strength, never an optimum over audit data."""

    p_solve: float | None
    residual_mean: float | None

    def __post_init__(self) -> None:
        if self.p_solve is None and self.residual_mean is None:
            raise ValueError("a quality outcome needs p_solve or residual_mean")
        if self.p_solve is not None:
            p_solve = _finite(self.p_solve, "p_solve outcome")
            if not 0.0 <= p_solve <= 1.0:
                raise ValueError("p_solve outcome must lie in [0, 1]")
            object.__setattr__(self, "p_solve", p_solve)
        if self.residual_mean is not None:
            object.__setattr__(
                self,
                "residual_mean",
                _finite(self.residual_mean, "residual outcome"),
            )

    def ranking_key(self, objective: QualityObjective) -> tuple[float, ...]:
        """Return the quality key for one explicitly registered objective."""

        if objective == "solve_probability_then_residual-v1":
            if self.p_solve is None:
                raise ValueError("solve-probability objective requires p_solve labels")
            residual_tiebreak = 0.0 if self.residual_mean is None else -self.residual_mean
            return self.p_solve, residual_tiebreak
        if objective == "residual_mean-v1":
            if self.residual_mean is None:
                raise ValueError("residual objective requires residual_mean labels")
            return (-self.residual_mean,)
        raise ValueError(f"unknown quality objective {objective!r}")

    def better_than(
        self,
        other: QualityOutcome,
        *,
        objective: QualityObjective,
        margin: float = 0.0,
    ) -> bool:
        """Return whether this outcome conservatively improves on ``other``."""

        if (self.p_solve is None) != (other.p_solve is None):
            raise ValueError("cannot compare outcomes with different label schemas")
        if (self.residual_mean is None) != (other.residual_mean is None):
            raise ValueError("cannot compare outcomes with different label schemas")
        if objective == "solve_probability_then_residual-v1":
            if self.p_solve is None or other.p_solve is None:
                raise ValueError("solve-probability objective requires p_solve labels")
            delta = self.p_solve - other.p_solve
            if delta > margin:
                return True
            if delta != 0.0 or margin > 0.0:
                return False
            if self.residual_mean is None:
                return False
        elif objective != "residual_mean-v1":
            raise ValueError(f"unknown quality objective {objective!r}")
        if self.residual_mean is None or other.residual_mean is None:
            raise ValueError("residual objective requires residual_mean labels")
        return other.residual_mean - self.residual_mean > margin


@dataclass(frozen=True, slots=True)
class EvaluationCurve:
    partition: str
    strengths: tuple[float, ...]
    seeds: tuple[int, ...]
    p_solve: Matrix | None
    residual_mean: Matrix | None
    reads: int
    sweeps: int
    objective: QualityObjective
    evaluation_protocol: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        if self.partition not in {"decision", "audit"}:
            raise ValueError("partition must be 'decision' or 'audit'")
        strengths = tuple(_finite(value, "strength") for value in self.strengths)
        if not strengths or len(set(strengths)) != len(strengths):
            raise ValueError("strengths must be a non-empty unique grid")
        if any(value < 0.0 for value in strengths):
            raise ValueError("strengths must be non-negative")
        seeds = tuple(self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be a non-empty unique grid")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("evaluation seeds must be non-negative integers")
        p_solve = _matrix(self.p_solve, "p_solve label")
        residual = _matrix(self.residual_mean, "residual label")
        if p_solve is None and residual is None:
            raise ValueError("an evaluation curve needs p_solve or residual labels")
        if self.objective not in {
            "solve_probability_then_residual-v1",
            "residual_mean-v1",
        }:
            raise ValueError("unknown quality objective")
        if self.objective == "solve_probability_then_residual-v1" and p_solve is None:
            raise ValueError("solve-probability objective requires p_solve labels")
        if self.objective == "residual_mean-v1" and residual is None:
            raise ValueError("residual objective requires residual_mean labels")
        evaluation_protocol = _normalise_pairs(self.evaluation_protocol)
        if {name for name, _ in evaluation_protocol} != EVALUATION_PROTOCOL_FIELDS_V1:
            raise ValueError(
                "evaluation_protocol must record decoder, sampler/version, schedule, "
                "seed derivation, reference energy, and noise model"
            )
        for name, matrix in (("p_solve", p_solve), ("residual_mean", residual)):
            if matrix is None:
                continue
            if len(matrix) != len(strengths) or any(len(row) != len(seeds) for row in matrix):
                raise ValueError(f"{name} shape must match the strength/seed grid")
        if p_solve is not None and any(not 0.0 <= value <= 1.0 for row in p_solve for value in row):
            raise ValueError("p_solve labels must lie in [0, 1]")
        object.__setattr__(self, "strengths", strengths)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "p_solve", p_solve)
        object.__setattr__(self, "residual_mean", residual)
        object.__setattr__(self, "evaluation_protocol", evaluation_protocol)
        object.__setattr__(self, "reads", _positive_int(self.reads, "reads"))
        object.__setattr__(self, "sweeps", _positive_int(self.sweeps, "sweeps"))

    def label_schema(self) -> tuple[bool, bool]:
        return self.p_solve is not None, self.residual_mean is not None

    def outcome_at_index(self, index: int) -> QualityOutcome:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("strength index must be an integer")
        if not 0 <= index < len(self.strengths):
            raise IndexError("strength index is outside the evaluation grid")
        return QualityOutcome(
            p_solve=(
                None
                if self.p_solve is None
                else math.fsum(self.p_solve[index]) / len(self.p_solve[index])
            ),
            residual_mean=(
                None
                if self.residual_mean is None
                else math.fsum(self.residual_mean[index]) / len(self.residual_mean[index])
            ),
        )

    def outcome_at_strength(self, strength: float) -> QualityOutcome:
        selected = _finite(strength, "strength")
        try:
            index = self.strengths.index(selected)
        except ValueError as error:
            raise ValueError("strength is outside the evaluation grid") from error
        return self.outcome_at_index(index)

    def best_index(self) -> int:
        """Select a strength on this curve; callers must never invoke this on audit."""

        if self.partition != "decision":
            raise ValueError("chain strength may only be selected on the decision partition")
        return max(
            range(len(self.strengths)),
            key=lambda index: (
                self.outcome_at_index(index).ranking_key(self.objective),
                -self.strengths[index],
            ),
        )

    def best_strength(self) -> float:
        return self.strengths[self.best_index()]

    def best_outcome(self) -> QualityOutcome:
        return self.outcome_at_index(self.best_index())

    def grid_signature(self) -> tuple[Any, ...]:
        return (
            self.strengths,
            self.seeds,
            self.reads,
            self.sweeps,
            self.objective,
            self.evaluation_protocol,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**_canonical_value(asdict(self)), "schema_version": SCHEMA_VERSION}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationCurve:
        _require_schema(value, "evaluation curve")
        _require_exact_keys(
            value,
            {
                "partition",
                "evaluation_protocol",
                "objective",
                "p_solve",
                "reads",
                "residual_mean",
                "schema_version",
                "seeds",
                "strengths",
                "sweeps",
            },
            "evaluation curve",
        )
        return cls(
            partition=value["partition"],
            strengths=tuple(value["strengths"]),
            seeds=tuple(value["seeds"]),
            p_solve=None if value.get("p_solve") is None else tuple(map(tuple, value["p_solve"])),
            residual_mean=(
                None
                if value.get("residual_mean") is None
                else tuple(map(tuple, value["residual_mean"]))
            ),
            reads=value["reads"],
            sweeps=value["sweeps"],
            objective=value["objective"],
            evaluation_protocol=tuple(map(tuple, value["evaluation_protocol"])),
        )


def _normalise_chains(chains: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    occupied: set[int] = set()
    for raw_chain in chains:
        chain = tuple(sorted(raw_chain))
        if not chain:
            raise ValueError("candidate chains must be non-empty")
        if any(isinstance(node, bool) or not isinstance(node, int) for node in chain):
            raise ValueError("candidate qubits must be integer IDs")
        if len(set(chain)) != len(chain):
            raise ValueError("a candidate chain repeats a qubit")
        if occupied.intersection(chain):
            raise ValueError("candidate chains overlap")
        occupied.update(chain)
        result.append(chain)
    if not result:
        raise ValueError("a candidate embedding must contain at least one chain")
    return tuple(result)


def _normalise_features(
    features: Sequence[tuple[str, float]],
    *,
    total_qubits: int,
    max_chain: int,
) -> tuple[tuple[str, float], ...]:
    result = tuple(sorted((name, _finite(value, f"feature {name!r}")) for name, value in features))
    if any(not isinstance(name, str) or not name for name, _ in result):
        raise ValueError("feature names must be non-empty strings")
    if len({name for name, _ in result}) != len(result):
        raise ValueError("feature names must be unique")
    names = {name for name, _ in result}
    if names != CANDIDATE_FEATURE_NAMES_V1:
        raise ValueError(
            "features must exactly match candidate feature schema v1: "
            f"{sorted(CANDIDATE_FEATURE_NAMES_V1)}"
        )
    expected = {"max_chain": float(max_chain), "total_qubits": float(total_qubits)}
    if dict(result) != expected:
        raise ValueError("candidate resource features must be derived from chains")
    return result


def derive_candidate_id(group_id: str, chains: Sequence[Sequence[int]]) -> str:
    """Derive an opaque identifier solely from pre-label candidate content."""

    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string")
    digest = content_digest(
        {
            "chains": _normalise_chains(chains),
            "domain": "isingfold-candidate-v1",
            "group_id": group_id,
        }
    )
    return f"candidate-{digest}"


def _candidate_payload(record: CandidateRecord) -> dict[str, Any]:
    return {
        "candidate_id": record.candidate_id,
        "chains": record.chains,
        "decision": record.decision.to_dict(),
        "audit": record.audit.to_dict(),
        "feature_schema_version": record.feature_schema_version,
        "features": record.features,
        "group_id": record.group_id,
        "max_chain": record.max_chain,
        "schema_version": SCHEMA_VERSION,
        "total_qubits": record.total_qubits,
    }


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    group_id: str
    candidate_id: str
    chains: tuple[tuple[int, ...], ...]
    total_qubits: int
    max_chain: int
    decision: EvaluationCurve
    audit: EvaluationCurve
    feature_schema_version: int
    features: tuple[tuple[str, float], ...]
    record_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("group_id must be a non-empty string")
        chains = _normalise_chains(self.chains)
        expected_id = derive_candidate_id(self.group_id, chains)
        if self.candidate_id != expected_id:
            raise ValueError("candidate_id must be derived from pre-label candidate content")
        if self.decision is None or not isinstance(self.decision, EvaluationCurve):
            raise ValueError("decision labels are required")
        if self.audit is None or not isinstance(self.audit, EvaluationCurve):
            raise ValueError("audit labels are required")
        if self.decision.partition != "decision":
            raise ValueError("decision curve has the wrong partition")
        if self.audit.partition != "audit":
            raise ValueError("audit curve has the wrong partition")
        if self.decision.strengths != self.audit.strengths:
            raise ValueError("decision and audit strength grids must match")
        if self.decision.objective != self.audit.objective:
            raise ValueError("decision and audit quality objectives must match")
        if self.decision.evaluation_protocol != self.audit.evaluation_protocol:
            raise ValueError("decision and audit evaluation protocols must match")
        if set(self.decision.seeds).intersection(self.audit.seeds):
            raise ValueError("decision and audit seed sets must be disjoint")
        if self.decision.label_schema() != self.audit.label_schema():
            raise ValueError("decision and audit label schemas must match")
        total_qubits = sum(len(chain) for chain in chains)
        max_chain = max(map(len, chains))
        if self.total_qubits != total_qubits or self.max_chain != max_chain:
            raise ValueError("stored candidate resource metrics do not match its chains")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unknown candidate feature schema {self.feature_schema_version!r}")
        object.__setattr__(self, "chains", chains)
        object.__setattr__(
            self,
            "features",
            _normalise_features(
                self.features,
                total_qubits=total_qubits,
                max_chain=max_chain,
            ),
        )
        if not isinstance(self.record_digest, str) or len(self.record_digest) != 64:
            raise ValueError("candidate record_digest must be a SHA-256 digest")

    @classmethod
    def create(
        cls,
        *,
        group_id: str,
        candidate_id: str | None = None,
        chains: Sequence[Sequence[int]],
        decision: EvaluationCurve | None,
        audit: EvaluationCurve | None,
        features: Sequence[tuple[str, float]] | None = None,
    ) -> CandidateRecord:
        if decision is None:
            raise ValueError("decision labels are required")
        if audit is None:
            raise ValueError("audit labels are required")
        normalised = _normalise_chains(chains)
        total_qubits = sum(map(len, normalised))
        max_chain = max(map(len, normalised))
        expected_id = derive_candidate_id(group_id, normalised)
        if candidate_id is not None and candidate_id != expected_id:
            raise ValueError("candidate_id must be derived from pre-label candidate content")
        resource_features = (
            ("max_chain", float(max_chain)),
            ("total_qubits", float(total_qubits)),
        )
        selected_features = resource_features if features is None else features
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "group_id", group_id)
        object.__setattr__(provisional, "candidate_id", expected_id)
        object.__setattr__(provisional, "chains", normalised)
        object.__setattr__(provisional, "total_qubits", total_qubits)
        object.__setattr__(provisional, "max_chain", max_chain)
        object.__setattr__(provisional, "decision", decision)
        object.__setattr__(provisional, "audit", audit)
        object.__setattr__(provisional, "feature_schema_version", FEATURE_SCHEMA_VERSION)
        object.__setattr__(
            provisional,
            "features",
            _normalise_features(
                selected_features,
                total_qubits=total_qubits,
                max_chain=max_chain,
            ),
        )
        object.__setattr__(provisional, "record_digest", "0" * 64)
        digest = content_digest(_candidate_payload(provisional))
        object.__setattr__(provisional, "record_digest", digest)
        cls.__post_init__(provisional)
        return provisional

    def to_dict(self) -> dict[str, Any]:
        return {
            **_candidate_payload(self),
            "record_digest": self.record_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateRecord:
        _require_schema(value, "candidate record")
        _require_exact_keys(
            value,
            {
                "audit",
                "candidate_id",
                "chains",
                "decision",
                "feature_schema_version",
                "features",
                "group_id",
                "max_chain",
                "record_digest",
                "schema_version",
                "total_qubits",
            },
            "candidate record",
        )
        result = cls(
            group_id=value["group_id"],
            candidate_id=value["candidate_id"],
            chains=tuple(map(tuple, value["chains"])),
            total_qubits=value["total_qubits"],
            max_chain=value["max_chain"],
            decision=EvaluationCurve.from_dict(value["decision"]),
            audit=EvaluationCurve.from_dict(value["audit"]),
            feature_schema_version=value["feature_schema_version"],
            features=tuple(map(tuple, value.get("features", ()))),
            record_digest=value["record_digest"],
        )
        if result.record_digest != content_digest(_candidate_payload(result)):
            raise ValueError("candidate record digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class RepairAttempt:
    attempt_id: str
    repair_seed: int
    neighborhood: tuple[int, ...]
    status: str
    candidate_id: str | None
    slot: int = 0
    transitions: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("attempt_id must be a non-empty string")
        if isinstance(self.repair_seed, bool) or not isinstance(self.repair_seed, int):
            raise ValueError("repair_seed must be an integer")
        neighborhood = tuple(self.neighborhood)
        if not neighborhood or any(
            isinstance(node, bool) or not isinstance(node, int) for node in neighborhood
        ):
            raise ValueError("repair neighborhood must contain integer logical IDs")
        allowed = {"valid", "duplicate", "no_change", "repair_failed"}
        if self.status not in allowed:
            raise ValueError(f"unknown repair attempt status {self.status!r}")
        if self.status in {"valid", "duplicate"} and not self.candidate_id:
            raise ValueError("a valid or duplicate repair attempt must identify its candidate")
        if self.status in {"no_change", "repair_failed"} and self.candidate_id is not None:
            raise ValueError("an unsuccessful repair attempt cannot identify a candidate")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("repair-attempt reason must be a non-empty string or None")
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 0:
            raise ValueError("attempt slot must be a non-negative integer")
        if (
            isinstance(self.transitions, bool)
            or not isinstance(self.transitions, int)
            or self.transitions < 0
        ):
            raise ValueError("attempt transitions must be a non-negative integer")
        object.__setattr__(self, "neighborhood", neighborhood)

    def to_dict(self) -> dict[str, Any]:
        return {**_canonical_value(asdict(self)), "schema_version": SCHEMA_VERSION}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RepairAttempt:
        _require_schema(value, "repair attempt")
        _require_exact_keys(
            value,
            {
                "attempt_id",
                "candidate_id",
                "neighborhood",
                "reason",
                "repair_seed",
                "schema_version",
                "slot",
                "status",
                "transitions",
            },
            "repair attempt",
        )
        return cls(
            attempt_id=value["attempt_id"],
            repair_seed=value["repair_seed"],
            neighborhood=tuple(value["neighborhood"]),
            status=value["status"],
            candidate_id=value.get("candidate_id"),
            slot=value.get("slot", 0),
            transitions=value.get("transitions", 0),
            reason=value.get("reason"),
        )


def _structural_embedding_payload(
    group_id: str,
    chains: Sequence[Sequence[int]],
) -> dict[str, Any]:
    canonical = _normalise_chains(chains)
    return {
        "candidate_id": derive_candidate_id(group_id, canonical),
        "chains": canonical,
        "max_chain": max(map(len, canonical)),
        "schema_version": SCHEMA_VERSION,
        "total_qubits": sum(map(len, canonical)),
    }


def generation_payload(
    *,
    group_id: str,
    instance_id: str,
    instance_record_digest: str,
    split_unit_id: str,
    group_seed: int,
    protocol: str | Sequence[tuple[str, Any]],
    incumbent_chains: Sequence[Sequence[int]],
    candidate_chains: Sequence[Sequence[Sequence[int]]],
    attempts: Sequence[RepairAttempt],
    rejection_reason: str | None,
) -> dict[str, Any]:
    """Build the immutable, pre-label payload committed by a sealed group."""

    candidates = sorted(
        (_structural_embedding_payload(group_id, chains) for chains in candidate_chains),
        key=lambda item: item["candidate_id"],
    )
    ordered_attempts = sorted(attempts, key=lambda item: item.slot)
    return {
        "attempts": [attempt.to_dict() for attempt in ordered_attempts],
        "candidates": candidates,
        "group_id": group_id,
        "group_seed": group_seed,
        "incumbent": _structural_embedding_payload(group_id, incumbent_chains),
        "instance_id": instance_id,
        "instance_record_digest": instance_record_digest,
        "protocol": _normalise_pairs(protocol),
        "rejection_reason": rejection_reason,
        "schema_version": SCHEMA_VERSION,
        "split": assign_split(split_unit_id),
        "split_unit_id": split_unit_id,
    }


def derive_generation_digest(**kwargs: Any) -> str:
    """Return the SHA-256 commitment for a complete pre-label group draft."""

    return content_digest(generation_payload(**kwargs))


def _normalise_pairs(value: str | Sequence[tuple[str, Any]]) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, str):
        return (("name", value),)
    result = tuple(sorted((name, _frozen_value(item)) for name, item in value))
    if any(not isinstance(name, str) or not name for name, _ in result):
        raise ValueError("protocol names must be non-empty strings")
    if len({name for name, _ in result}) != len(result):
        raise ValueError("protocol names must be unique")
    return result


def derive_group_id(
    instance_id: str,
    incumbent_chains: Sequence[Sequence[int]],
    protocol: str | Sequence[tuple[str, Any]],
    group_seed: int,
) -> str:
    """Derive a group identifier before any quality labels are observed."""

    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance_id must be a non-empty string")
    if isinstance(group_seed, bool) or not isinstance(group_seed, int):
        raise ValueError("group_seed must be an integer")
    digest = content_digest(
        {
            "domain": "isingfold-candidate-group-v1",
            "group_seed": group_seed,
            "incumbent_chains": _normalise_chains(incumbent_chains),
            "instance_id": instance_id,
            "protocol": _normalise_pairs(protocol),
        }
    )
    return f"group-{digest}"


def derive_attempt_id(group_id: str, slot: int) -> str:
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string")
    if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
        raise ValueError("attempt slot must be a non-negative integer")
    digest = content_digest(
        {
            "domain": "isingfold-repair-attempt-v1",
            "group_id": group_id,
            "slot": slot,
        }
    )
    return f"attempt-{digest}"


def _group_payload(group: CandidateGroup) -> dict[str, Any]:
    return {
        "attempts": [attempt.to_dict() for attempt in group.attempts],
        "candidates": [candidate.to_dict() for candidate in group.candidates],
        "group_id": group.group_id,
        "group_seed": group.group_seed,
        "generation_digest": group.generation_digest,
        "incumbent": group.incumbent.to_dict(),
        "instance_id": group.instance_id,
        "instance_record_digest": group.instance_record_digest,
        "protocol": group.protocol,
        "schema_version": SCHEMA_VERSION,
        "split": group.split,
        "split_unit_id": group.split_unit_id,
    }


@dataclass(frozen=True, slots=True)
class CandidateGroup:
    group_id: str
    instance_id: str
    instance_record_digest: str
    split_unit_id: str
    split: Split
    group_seed: int
    protocol: tuple[tuple[str, Any], ...]
    incumbent: CandidateRecord
    candidates: tuple[CandidateRecord, ...]
    attempts: tuple[RepairAttempt, ...]
    generation_digest: str
    record_digest: str

    @classmethod
    def create(
        cls,
        *,
        group_id: str,
        instance_id: str,
        instance_record_digest: str,
        split_unit_id: str,
        split: Split,
        group_seed: int,
        protocol: str | Sequence[tuple[str, Any]],
        incumbent: CandidateRecord,
        candidates: Sequence[CandidateRecord],
        attempts: Sequence[RepairAttempt],
        generation_digest: str,
    ) -> CandidateGroup:
        expected_group_id = derive_group_id(
            instance_id,
            incumbent.chains,
            protocol,
            group_seed,
        )
        if group_id != expected_group_id:
            raise ValueError("group_id must be derived from pre-label group content")
        provisional = cls(
            group_id=group_id,
            instance_id=instance_id,
            instance_record_digest=instance_record_digest,
            split_unit_id=split_unit_id,
            split=split,
            group_seed=group_seed,
            protocol=_normalise_pairs(protocol),
            incumbent=incumbent,
            candidates=tuple(sorted(candidates, key=lambda item: item.candidate_id)),
            attempts=tuple(sorted(attempts, key=lambda item: item.slot)),
            generation_digest=generation_digest,
            record_digest="0" * 64,
        )
        result = cls(
            group_id=provisional.group_id,
            instance_id=provisional.instance_id,
            instance_record_digest=provisional.instance_record_digest,
            split_unit_id=provisional.split_unit_id,
            split=provisional.split,
            group_seed=provisional.group_seed,
            protocol=provisional.protocol,
            incumbent=provisional.incumbent,
            candidates=provisional.candidates,
            attempts=provisional.attempts,
            generation_digest=provisional.generation_digest,
            record_digest=content_digest(_group_payload(provisional)),
        )
        validate_group(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {**_group_payload(self), "record_digest": self.record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateGroup:
        _require_schema(value, "candidate group")
        _require_exact_keys(
            value,
            {
                "attempts",
                "candidates",
                "generation_digest",
                "group_id",
                "group_seed",
                "incumbent",
                "instance_id",
                "instance_record_digest",
                "protocol",
                "record_digest",
                "schema_version",
                "split",
                "split_unit_id",
            },
            "candidate group",
        )
        result = cls(
            group_id=value["group_id"],
            instance_id=value["instance_id"],
            instance_record_digest=value["instance_record_digest"],
            split_unit_id=value["split_unit_id"],
            split=value["split"],
            group_seed=value["group_seed"],
            protocol=tuple(map(tuple, value["protocol"])),
            incumbent=CandidateRecord.from_dict(value["incumbent"]),
            candidates=tuple(CandidateRecord.from_dict(item) for item in value["candidates"]),
            attempts=tuple(RepairAttempt.from_dict(item) for item in value.get("attempts", ())),
            generation_digest=value["generation_digest"],
            record_digest=value["record_digest"],
        )
        validate_group(result)
        return result


def _validate_candidate_digest(candidate: CandidateRecord, group_id: str) -> None:
    if candidate.group_id != group_id:
        raise ValueError("candidate record belongs to a different group")
    expected = content_digest(_candidate_payload(candidate))
    if candidate.record_digest != expected:
        raise ValueError(f"candidate {candidate.candidate_id!r} digest mismatch")


def validate_group(group: CandidateGroup) -> None:
    if not isinstance(group.group_id, str) or not group.group_id:
        raise ValueError("group_id must be a non-empty string")
    if not isinstance(group.instance_id, str) or not group.instance_id:
        raise ValueError("instance_id must be a non-empty string")
    if not isinstance(group.instance_record_digest, str) or len(group.instance_record_digest) != 64:
        raise ValueError("instance_record_digest must be a SHA-256 digest")
    if not isinstance(group.split_unit_id, str) or not group.split_unit_id:
        raise ValueError("split_unit_id must be a non-empty string")
    if group.split not in {"train", "val", "test"}:
        raise ValueError("unknown candidate-group split")
    if group.split != assign_split(group.split_unit_id):
        raise ValueError("candidate-group split must be derived from split_unit_id")
    if isinstance(group.group_seed, bool) or not isinstance(group.group_seed, int):
        raise ValueError("group_seed must be an integer")
    if group.group_id != derive_group_id(
        group.instance_id,
        group.incumbent.chains,
        group.protocol,
        group.group_seed,
    ):
        raise ValueError("group_id must be derived from pre-label group content")
    if len(group.candidates) < 2:
        raise ValueError("a candidate group requires at least two valid unique alternatives")
    if group.candidates != tuple(sorted(group.candidates, key=lambda item: item.candidate_id)):
        raise ValueError("candidate records must be in canonical candidate_id order")
    if group.attempts != tuple(sorted(group.attempts, key=lambda item: item.slot)):
        raise ValueError("repair attempts must be in canonical slot order")
    attempt_slots = dict(group.protocol).get("attempt_slots")
    if isinstance(attempt_slots, bool) or not isinstance(attempt_slots, int) or attempt_slots <= 0:
        raise ValueError("candidate-group protocol requires a positive attempt_slots")
    if len(group.attempts) != attempt_slots:
        raise ValueError("candidate group must retain every fixed repair-attempt slot")
    if tuple(attempt.slot for attempt in group.attempts) != tuple(range(attempt_slots)):
        raise ValueError("repair attempt slots must be complete and contiguous")
    records = (group.incumbent, *group.candidates)
    for record in records:
        _validate_candidate_digest(record, group.group_id)
    all_ids = [candidate.candidate_id for candidate in records]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("candidate_id values must be unique within a group")
    ids = [candidate.candidate_id for candidate in group.candidates]
    embeddings = [candidate.chains for candidate in group.candidates]
    if len(embeddings) != len(set(embeddings)):
        raise ValueError("candidate embeddings must be unique within a group")
    if group.incumbent.chains in set(embeddings):
        raise ValueError("candidate embedding must differ from the incumbent")
    for record in records:
        if record.decision.strengths != record.audit.strengths:
            raise ValueError("decision and audit strength grids must match")
        if set(record.decision.seeds).intersection(record.audit.seeds):
            raise ValueError("decision and audit seed sets must be disjoint")
    decision_grid = group.incumbent.decision.grid_signature()
    audit_grid = group.incumbent.audit.grid_signature()
    if any(candidate.decision.grid_signature() != decision_grid for candidate in group.candidates):
        raise ValueError("decision evaluation grid mismatch within candidate group")
    if any(candidate.audit.grid_signature() != audit_grid for candidate in group.candidates):
        raise ValueError("audit evaluation grid mismatch within candidate group")
    label_schema = group.incumbent.decision.label_schema()
    if group.incumbent.audit.label_schema() != label_schema or any(
        candidate.decision.label_schema() != label_schema
        or candidate.audit.label_schema() != label_schema
        for candidate in group.candidates
    ):
        raise ValueError("quality label schema mismatch within candidate group")
    known_ids = set(ids)
    attempt_ids = [attempt.attempt_id for attempt in group.attempts]
    slots = [attempt.slot for attempt in group.attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("repair attempt_id values must be unique within a group")
    if len(slots) != len(set(slots)):
        raise ValueError("repair attempt slots must be unique within a group")
    if any(
        attempt.attempt_id != derive_attempt_id(group.group_id, attempt.slot)
        for attempt in group.attempts
    ):
        raise ValueError("repair attempt_id must be derived from group_id and slot")
    for attempt in group.attempts:
        if attempt.status in {"valid", "duplicate"} and attempt.candidate_id not in known_ids:
            raise ValueError("repair attempt references an unknown candidate_id")
    valid_ids = [attempt.candidate_id for attempt in group.attempts if attempt.status == "valid"]
    if len(valid_ids) != len(set(valid_ids)) or set(valid_ids) != known_ids:
        raise ValueError("each candidate requires exactly one valid repair-attempt provenance")
    expected_generation_digest = derive_generation_digest(
        group_id=group.group_id,
        instance_id=group.instance_id,
        instance_record_digest=group.instance_record_digest,
        split_unit_id=group.split_unit_id,
        group_seed=group.group_seed,
        protocol=group.protocol,
        incumbent_chains=group.incumbent.chains,
        candidate_chains=tuple(candidate.chains for candidate in group.candidates),
        attempts=group.attempts,
        rejection_reason=None,
    )
    if group.generation_digest != expected_generation_digest:
        raise ValueError("candidate group does not match its pre-label generation digest")
    expected = content_digest(_group_payload(group))
    if group.record_digest != expected:
        raise ValueError("candidate-group record digest mismatch")


def _canonical_edges(
    edges: Sequence[Sequence[int]], nodes: tuple[int, ...], name: str
) -> tuple[tuple[int, int], ...]:
    allowed = set(nodes)
    result: list[tuple[int, int]] = []
    for raw in edges:
        if len(raw) != 2:
            raise ValueError(f"{name} edges must contain two endpoints")
        first, second = raw
        if any(isinstance(node, bool) or not isinstance(node, int) for node in (first, second)):
            raise ValueError(f"{name} edge endpoints must be integers")
        if first == second or first not in allowed or second not in allowed:
            raise ValueError(f"invalid {name} edge")
        result.append((min(first, second), max(first, second)))
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate {name} edge")
    return tuple(sorted(result))


def _canonical_h(
    h: Sequence[tuple[int, float]], logical_nodes: tuple[int, ...]
) -> tuple[tuple[int, float], ...]:
    if any(isinstance(node, bool) or not isinstance(node, int) for node, _ in h):
        raise ValueError("h endpoints must be integer logical node IDs")
    values = tuple(sorted((node, _finite(value, "h coefficient")) for node, value in h))
    if len(values) != len(logical_nodes) or {node for node, _ in values} != set(logical_nodes):
        raise ValueError("h must contain exactly one coefficient per logical node")
    return values


def _canonical_j(
    j: Sequence[tuple[int, int, float]], logical_nodes: tuple[int, ...]
) -> tuple[tuple[int, int, float], ...]:
    allowed = set(logical_nodes)
    values: list[tuple[int, int, float]] = []
    for first, second, raw_value in j:
        if any(isinstance(node, bool) or not isinstance(node, int) for node in (first, second)):
            raise ValueError("J endpoints must be integers")
        if first == second or first not in allowed or second not in allowed:
            raise ValueError("invalid J endpoint")
        values.append((min(first, second), max(first, second), _finite(raw_value, "J coefficient")))
    result = tuple(sorted(values))
    if len({(first, second) for first, second, _ in result}) != len(result):
        raise ValueError("duplicate J edge")
    return result


def derive_split_unit_id(
    *,
    logical_nodes: Sequence[int],
    logical_edges: Sequence[Sequence[int]],
    h: Sequence[tuple[int, float]],
    j: Sequence[tuple[int, int, float]],
) -> str:
    """Fingerprint the logical Ising problem before topology variants exist."""

    nodes = tuple(sorted(logical_nodes))
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(isinstance(node, bool) or not isinstance(node, int) for node in nodes)
    ):
        raise ValueError("logical_nodes must be non-empty unique integer IDs")
    edges = _canonical_edges(logical_edges, nodes, "logical")
    h_values = _canonical_h(h, nodes)
    j_values = _canonical_j(j, nodes)
    if {(first, second) for first, second, _ in j_values} != set(edges):
        raise ValueError("J coefficients must correspond exactly to logical_edges")
    digest = content_digest(
        {
            "domain": "isingfold-logical-problem-v1",
            "h": h_values,
            "j": j_values,
            "logical_edges": edges,
            "logical_nodes": nodes,
        }
    )
    return f"logical-{digest}"


def derive_instance_id(
    *,
    split_unit_id: str,
    host_nodes: Sequence[int],
    host_edges: Sequence[Sequence[int]],
) -> str:
    """Fingerprint one topology/defect realization of a logical problem."""

    if not isinstance(split_unit_id, str) or not split_unit_id:
        raise ValueError("split_unit_id must be a non-empty string")
    nodes = tuple(sorted(host_nodes))
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or any(isinstance(node, bool) or not isinstance(node, int) for node in nodes)
    ):
        raise ValueError("host_nodes must be non-empty unique integer IDs")
    edges = _canonical_edges(host_edges, nodes, "host")
    digest = content_digest(
        {
            "domain": "isingfold-instance-v1",
            "host_edges": edges,
            "host_nodes": nodes,
            "split_unit_id": split_unit_id,
        }
    )
    return f"instance-{digest}"


def _instance_payload(instance: InstanceRecord) -> dict[str, Any]:
    return {
        "family": instance.family,
        "h": instance.h,
        "host_edges": instance.host_edges,
        "host_nodes": instance.host_nodes,
        "instance_id": instance.instance_id,
        "j": instance.j,
        "logical_edges": instance.logical_edges,
        "logical_nodes": instance.logical_nodes,
        "metadata": instance.metadata,
        "schema_version": SCHEMA_VERSION,
        "split_unit_id": instance.split_unit_id,
        "topology": instance.topology,
    }


@dataclass(frozen=True, slots=True)
class InstanceRecord:
    split_unit_id: str
    instance_id: str
    family: str
    topology: str
    logical_nodes: tuple[int, ...]
    logical_edges: tuple[tuple[int, int], ...]
    host_nodes: tuple[int, ...]
    host_edges: tuple[tuple[int, int], ...]
    h: tuple[tuple[int, float], ...]
    j: tuple[tuple[int, int, float], ...]
    metadata: tuple[tuple[str, Any], ...]
    record_digest: str

    @classmethod
    def create(
        cls,
        *,
        split_unit_id: str | None = None,
        instance_id: str | None = None,
        family: str,
        topology: str,
        logical_nodes: Sequence[int],
        logical_edges: Sequence[Sequence[int]],
        host_nodes: Sequence[int],
        host_edges: Sequence[Sequence[int]],
        h: Sequence[tuple[int, float]],
        j: Sequence[tuple[int, int, float]],
        metadata: Sequence[tuple[str, Any]] = (),
    ) -> InstanceRecord:
        logical = tuple(sorted(logical_nodes))
        host = tuple(sorted(host_nodes))
        if not logical or len(logical) != len(set(logical)):
            raise ValueError("logical_nodes must be non-empty and unique")
        if not host or len(host) != len(set(host)):
            raise ValueError("host_nodes must be non-empty and unique")
        logical_edges_value = _canonical_edges(logical_edges, logical, "logical")
        host_edges_value = _canonical_edges(host_edges, host, "host")
        h_values = _canonical_h(h, logical)
        j_values = _canonical_j(j, logical)
        expected_split_unit_id = derive_split_unit_id(
            logical_nodes=logical,
            logical_edges=logical_edges_value,
            h=h_values,
            j=j_values,
        )
        if split_unit_id is not None and split_unit_id != expected_split_unit_id:
            raise ValueError("split_unit_id must be derived from the logical problem")
        expected_instance_id = derive_instance_id(
            split_unit_id=expected_split_unit_id,
            host_nodes=host,
            host_edges=host_edges_value,
        )
        if instance_id is not None and instance_id != expected_instance_id:
            raise ValueError("instance_id must be derived from pre-label instance content")
        provisional = cls(
            split_unit_id=expected_split_unit_id,
            instance_id=expected_instance_id,
            family=family,
            topology=topology,
            logical_nodes=logical,
            logical_edges=logical_edges_value,
            host_nodes=host,
            host_edges=host_edges_value,
            h=h_values,
            j=j_values,
            metadata=_normalise_pairs(metadata),
            record_digest="0" * 64,
        )
        for name in ("split_unit_id", "instance_id", "family", "topology"):
            if not isinstance(getattr(provisional, name), str) or not getattr(provisional, name):
                raise ValueError(f"{name} must be a non-empty string")
        if {(first, second) for first, second, _ in j_values} != set(logical_edges_value):
            raise ValueError("J coefficients must correspond exactly to logical_edges")
        return cls(
            **{
                field: getattr(provisional, field)
                for field in provisional.__dataclass_fields__
                if field != "record_digest"
            },
            record_digest=content_digest(_instance_payload(provisional)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**_instance_payload(self), "record_digest": self.record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InstanceRecord:
        _require_schema(value, "instance record")
        _require_exact_keys(
            value,
            {
                "family",
                "h",
                "host_edges",
                "host_nodes",
                "instance_id",
                "j",
                "logical_edges",
                "logical_nodes",
                "metadata",
                "record_digest",
                "schema_version",
                "split_unit_id",
                "topology",
            },
            "instance record",
        )
        result = cls.create(
            split_unit_id=value["split_unit_id"],
            instance_id=value["instance_id"],
            family=value["family"],
            topology=value["topology"],
            logical_nodes=tuple(value["logical_nodes"]),
            logical_edges=tuple(map(tuple, value["logical_edges"])),
            host_nodes=tuple(value["host_nodes"]),
            host_edges=tuple(map(tuple, value["host_edges"])),
            h=tuple(map(tuple, value["h"])),
            j=tuple(map(tuple, value["j"])),
            metadata=tuple(map(tuple, value.get("metadata", ()))),
        )
        if result.record_digest != value["record_digest"]:
            raise ValueError("instance record digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class BankManifest:
    jsonl_sha256: str
    instance_count: int
    group_count: int
    schema_version: int = SCHEMA_VERSION
    record_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"manifest requires schema_version {SCHEMA_VERSION}")
        if not isinstance(self.jsonl_sha256, str) or len(self.jsonl_sha256) != 64:
            raise ValueError("jsonl_sha256 must be a SHA-256 digest")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in (self.instance_count, self.group_count)
        ):
            raise ValueError("manifest counts must be non-negative")
        if not isinstance(self.record_digest, str) or len(self.record_digest) != 64:
            raise ValueError("manifest record_digest must be a SHA-256 digest")


def _manifest_payload(manifest: BankManifest) -> dict[str, Any]:
    return {
        "group_count": manifest.group_count,
        "instance_count": manifest.instance_count,
        "jsonl_sha256": manifest.jsonl_sha256,
        "schema_version": manifest.schema_version,
    }


def _validate_instance_digest(instance: InstanceRecord) -> None:
    if instance.record_digest != content_digest(_instance_payload(instance)):
        raise ValueError(f"instance {instance.instance_id!r} digest mismatch")


def validate_instance(instance: InstanceRecord) -> None:
    """Revalidate semantics and content identities, including replaced records."""

    expected_split = derive_split_unit_id(
        logical_nodes=instance.logical_nodes,
        logical_edges=instance.logical_edges,
        h=instance.h,
        j=instance.j,
    )
    if instance.split_unit_id != expected_split:
        raise ValueError("instance split_unit_id does not match its logical problem")
    expected_instance = derive_instance_id(
        split_unit_id=instance.split_unit_id,
        host_nodes=instance.host_nodes,
        host_edges=instance.host_edges,
    )
    if instance.instance_id != expected_instance:
        raise ValueError("instance_id does not match its problem and host graph")
    if not isinstance(instance.family, str) or not instance.family:
        raise ValueError("family must be a non-empty string")
    if not isinstance(instance.topology, str) or not instance.topology:
        raise ValueError("topology must be a non-empty string")
    _validate_instance_digest(instance)


def _validate_embedding(candidate: CandidateRecord, instance: InstanceRecord) -> None:
    if len(candidate.chains) != len(instance.logical_nodes):
        raise ValueError("candidate chain count must match logical node count")
    host_nodes = set(instance.host_nodes)
    host_edges = set(instance.host_edges)
    for chain in candidate.chains:
        if not set(chain).issubset(host_nodes):
            raise ValueError("candidate embedding contains a qubit outside the host graph")
        if len(chain) == 1:
            continue
        seen = {chain[0]}
        frontier = [chain[0]]
        allowed = set(chain)
        while frontier:
            first = frontier.pop()
            for edge in host_edges:
                if first not in edge:
                    continue
                second = edge[1] if edge[0] == first else edge[0]
                if second in allowed and second not in seen:
                    seen.add(second)
                    frontier.append(second)
        if seen != allowed:
            raise ValueError("candidate chain is disconnected in the host graph")
    logical_index = {node: index for index, node in enumerate(instance.logical_nodes)}
    for first, second in instance.logical_edges:
        first_chain = candidate.chains[logical_index[first]]
        second_chain = candidate.chains[logical_index[second]]
        if not any(
            (min(left, right), max(left, right)) in host_edges
            for left in first_chain
            for right in second_chain
        ):
            raise ValueError("candidate embedding does not realize every logical edge")


def validate_group_against_instance(group: CandidateGroup, instance: InstanceRecord) -> None:
    validate_instance(instance)
    validate_group(group)
    if group.instance_id != instance.instance_id:
        raise ValueError("candidate group references the wrong instance")
    if group.instance_record_digest != instance.record_digest:
        raise ValueError("candidate group references the wrong instance record digest")
    if group.split_unit_id != instance.split_unit_id:
        raise ValueError("candidate group and instance split_unit_id mismatch")
    if group.split != assign_split(instance.split_unit_id):
        raise ValueError("candidate group split does not match its logical split unit")
    logical_nodes = set(instance.logical_nodes)
    for attempt in group.attempts:
        if not set(attempt.neighborhood).issubset(logical_nodes):
            raise ValueError("repair-attempt neighborhood contains an unknown logical node")
    for candidate in (group.incumbent, *group.candidates):
        _validate_embedding(candidate, instance)


def write_bank(
    path: str | os.PathLike[str],
    *,
    instances: Sequence[InstanceRecord],
    groups: Sequence[CandidateGroup],
) -> BankManifest:
    """Write a deterministic combined JSONL and return its detached manifest."""

    instance_rows = tuple(sorted(instances, key=lambda item: item.instance_id))
    group_rows = tuple(sorted(groups, key=lambda item: item.group_id))
    if len({item.instance_id for item in instance_rows}) != len(instance_rows):
        raise ValueError("duplicate instance_id in candidate bank")
    if len({item.group_id for item in group_rows}) != len(group_rows):
        raise ValueError("duplicate group_id in candidate bank")
    by_id = {item.instance_id: item for item in instance_rows}
    for instance in instance_rows:
        validate_instance(instance)
    for group in group_rows:
        if group.instance_id not in by_id:
            raise ValueError(f"candidate group references unknown instance {group.instance_id!r}")
        instance = by_id[group.instance_id]
        validate_group_against_instance(group, instance)
    rows = [
        *({"kind": "instance", "record": item.to_dict()} for item in instance_rows),
        *({"kind": "group", "record": item.to_dict()} for item in group_rows),
    ]
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    provisional = BankManifest(
        jsonl_sha256=hashlib.sha256(raw).hexdigest(),
        instance_count=len(instance_rows),
        group_count=len(group_rows),
        record_digest="0" * 64,
    )
    return BankManifest(
        jsonl_sha256=provisional.jsonl_sha256,
        instance_count=provisional.instance_count,
        group_count=provisional.group_count,
        record_digest=content_digest(_manifest_payload(provisional)),
    )


def _strict_object(raw: bytes, name: str) -> dict[str, Any]:
    def reject_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite number {token}")

    value = json.loads(raw, object_pairs_hook=reject_pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def read_bank(
    path: str | os.PathLike[str], manifest: BankManifest
) -> tuple[tuple[InstanceRecord, ...], tuple[CandidateGroup, ...]]:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"manifest requires schema_version {SCHEMA_VERSION}")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.jsonl_sha256:
        raise ValueError("candidate-bank checksum mismatch")
    if manifest.record_digest != content_digest(_manifest_payload(manifest)):
        raise ValueError("candidate-bank manifest digest mismatch")
    instances: list[InstanceRecord] = []
    groups: list[CandidateGroup] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        row = _strict_object(line, f"candidate-bank line {index}")
        if row.get("kind") == "instance":
            instance = InstanceRecord.from_dict(row["record"])
            validate_instance(instance)
            instances.append(instance)
        elif row.get("kind") == "group":
            group = CandidateGroup.from_dict(row["record"])
            validate_group(group)
            groups.append(group)
        else:
            raise ValueError(f"candidate-bank line {index} has unknown kind")
    if (len(instances), len(groups)) != (manifest.instance_count, manifest.group_count):
        raise ValueError("candidate-bank manifest count mismatch")
    by_id = {instance.instance_id: instance for instance in instances}
    if len(by_id) != len(instances) or len({group.group_id for group in groups}) != len(groups):
        raise ValueError("candidate-bank contains duplicate record IDs")
    if any(group.instance_id not in by_id for group in groups):
        raise ValueError("candidate-bank group references a missing instance")
    for group in groups:
        instance = by_id[group.instance_id]
        validate_group_against_instance(group, instance)
    return tuple(instances), tuple(groups)


@dataclass(frozen=True, slots=True)
class UnlabeledCandidateView:
    candidate_id: str
    chains: tuple[tuple[int, ...], ...]
    total_qubits: int
    max_chain: int
    features: tuple[tuple[str, float], ...]

    @classmethod
    def from_record(cls, record: CandidateRecord) -> UnlabeledCandidateView:
        return cls(
            candidate_id=record.candidate_id,
            chains=record.chains,
            total_qubits=record.total_qubits,
            max_chain=record.max_chain,
            features=record.features,
        )


@dataclass(frozen=True, slots=True)
class UnlabeledAttemptView:
    """Label-free generation context for one visible repair attempt."""

    slot: int
    neighborhood: tuple[int, ...]
    status: str
    candidate_id: str | None
    transitions: int

    @classmethod
    def from_record(cls, record: RepairAttempt) -> UnlabeledAttemptView:
        return cls(
            slot=record.slot,
            neighborhood=record.neighborhood,
            status=record.status,
            candidate_id=record.candidate_id,
            transitions=record.transitions,
        )


@dataclass(frozen=True, slots=True)
class UnlabeledGroupView:
    """Complete deployable policy input with no labels or split identifiers."""

    family: str
    topology: str
    logical_nodes: tuple[int, ...]
    logical_edges: tuple[tuple[int, int], ...]
    h: tuple[tuple[int, float], ...]
    j: tuple[tuple[int, int, float], ...]
    host_nodes: tuple[int, ...]
    host_edges: tuple[tuple[int, int], ...]
    incumbent: UnlabeledCandidateView
    attempts: tuple[UnlabeledAttemptView, ...]
    candidates: tuple[UnlabeledCandidateView, ...]

    @classmethod
    def from_records(
        cls,
        *,
        instance: InstanceRecord,
        group: CandidateGroup,
        candidates: Sequence[CandidateRecord],
    ) -> UnlabeledGroupView:
        # Copy an explicit allowlist instead of wrapping either source record.  In
        # particular, free-form metadata/protocol fields and identity/split fields
        # must never cross the deployable policy boundary.
        visible_candidates = tuple(
            UnlabeledCandidateView.from_record(candidate) for candidate in candidates
        )
        visible_ids = {candidate.candidate_id for candidate in visible_candidates}
        visible_attempts = tuple(
            UnlabeledAttemptView.from_record(attempt)
            for attempt in group.attempts
            if attempt.candidate_id is None or attempt.candidate_id in visible_ids
        )
        return cls(
            family=instance.family,
            topology=instance.topology,
            logical_nodes=instance.logical_nodes,
            logical_edges=instance.logical_edges,
            h=instance.h,
            j=instance.j,
            host_nodes=instance.host_nodes,
            host_edges=instance.host_edges,
            incumbent=UnlabeledCandidateView.from_record(group.incumbent),
            attempts=visible_attempts,
            candidates=visible_candidates,
        )


Selector = Callable[[UnlabeledGroupView, random.Random], str]


@dataclass(frozen=True, slots=True)
class ReplayPolicy:
    name: str
    selector: Selector | None
    implementation_id: str
    deployable: bool = True
    oracle_partition: Literal["decision", "audit"] | None = None
    trust_boundary: Literal["in-process-attested", "oracle"] = "in-process-attested"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("replay policy name must be non-empty")
        if not isinstance(self.implementation_id, str) or not self.implementation_id:
            raise ValueError("policy implementation_id must be a non-empty versioned identifier")
        if self.oracle_partition not in {None, "decision", "audit"}:
            raise ValueError("oracle_partition must be 'decision', 'audit', or None")
        if self.oracle_partition is None and not callable(self.selector):
            raise ValueError("a deployable replay policy needs a selector")
        if self.oracle_partition is not None and self.deployable:
            raise ValueError("an oracle replay policy cannot be deployable")
        expected_boundary = "oracle" if self.oracle_partition is not None else "in-process-attested"
        if self.trust_boundary != expected_boundary:
            raise ValueError(f"policy trust_boundary must be {expected_boundary!r}")


@dataclass(frozen=True, slots=True)
class AcceptanceProtocol:
    """Charged post-selection verification used by every replayed policy."""

    name: str = "one-shot-decision-improvement-v1"
    margin: float = 0.0
    decision_evaluations: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("acceptance protocol name must be non-empty")
        margin = _finite(self.margin, "acceptance margin")
        if margin < 0.0:
            raise ValueError("acceptance margin must be non-negative")
        if self.decision_evaluations != 1:
            raise ValueError("candidate replay requires exactly one charged decision evaluation")
        object.__setattr__(self, "margin", margin)


DEFAULT_ACCEPTANCE = AcceptanceProtocol()


@dataclass(frozen=True, slots=True)
class ReplayResult:
    group_id: str
    split: Split
    policy_name: str
    policy_implementation_id: str
    policy_deployable: bool
    policy_trust_boundary: str
    replay_seed: int
    q_cap: int | None
    eligible_candidate_ids: tuple[str, ...]
    selected_candidate_id: str
    selected_strength: float
    decision_accepted: bool
    final_candidate_id: str
    final_strength: float
    audit_quality: QualityOutcome
    acceptance_protocol: str
    acceptance_margin: float
    decision_evaluations: int

    @property
    def audit_outcome(self) -> float:
        """Backward-compatible primary audit value; use audit_quality for both metrics."""

        if self.audit_quality.p_solve is not None:
            return self.audit_quality.p_solve
        assert self.audit_quality.residual_mean is not None
        return -self.audit_quality.residual_mean


def resource_policy() -> ReplayPolicy:
    def select(group: UnlabeledGroupView, rng: random.Random) -> str:
        candidates = group.candidates
        # Every bank candidate is a feasible, vertex-disjoint embedding, so the
        # first two native ResourceScorer keys are tied at (1, 0). At this sealed,
        # materialized-embedding boundary route_cost is also tied; the remaining
        # key, used_target_nodes, is exactly total_qubits here.
        best_total_qubits = min(item.total_qubits for item in candidates)
        tied = sorted(
            (item for item in candidates if item.total_qubits == best_total_qubits),
            key=lambda item: item.candidate_id,
        )
        return rng.choice(tied).candidate_id

    return ReplayPolicy(
        "resource",
        select,
        implementation_id="embedbench.native-resource-total-qubits-random-tie-v1",
    )


def max_chain_first_policy() -> ReplayPolicy:
    """Return the former max-chain-first resource heuristic as an explicit ablation."""

    def select(group: UnlabeledGroupView, rng: random.Random) -> str:
        candidates = group.candidates
        best_resources = min((item.max_chain, item.total_qubits) for item in candidates)
        tied = sorted(
            (item for item in candidates if (item.max_chain, item.total_qubits) == best_resources),
            key=lambda item: item.candidate_id,
        )
        return rng.choice(tied).candidate_id

    return ReplayPolicy(
        "max-chain-first",
        select,
        implementation_id="embedbench.max-chain-total-qubits-random-tie-v1",
    )


def random_policy() -> ReplayPolicy:
    def select(group: UnlabeledGroupView, rng: random.Random) -> str:
        candidates = group.candidates
        return rng.choice(sorted(candidates, key=lambda item: item.candidate_id)).candidate_id

    return ReplayPolicy("random", select, implementation_id="embedbench.random-v1")


def fit_oracle_policy() -> ReplayPolicy:
    return ReplayPolicy(
        "fit-oracle",
        None,
        implementation_id="embedbench.fit-oracle-v1",
        deployable=False,
        oracle_partition="decision",
        trust_boundary="oracle",
    )


def audit_oracle_policy() -> ReplayPolicy:
    return ReplayPolicy(
        "audit-oracle",
        None,
        implementation_id="embedbench.audit-oracle-post-acceptance-v1",
        deployable=False,
        oracle_partition="audit",
        trust_boundary="oracle",
    )


def replay_policy(
    groups: Sequence[CandidateGroup],
    policy: ReplayPolicy,
    *,
    seed: int,
    instances: Sequence[InstanceRecord],
    q_cap: int | None = None,
    acceptance: AcceptanceProtocol = DEFAULT_ACCEPTANCE,
) -> tuple[ReplayResult, ...]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("replay seed must be a non-negative integer")
    if q_cap is not None and (isinstance(q_cap, bool) or not isinstance(q_cap, int) or q_cap <= 0):
        raise ValueError("q_cap must be a positive integer")

    def decision_choice(candidate: CandidateRecord) -> tuple[float, QualityOutcome]:
        strength = candidate.decision.best_strength()
        return strength, candidate.decision.outcome_at_strength(strength)

    def audit_at_decision_choice(candidate: CandidateRecord) -> QualityOutcome:
        strength, _ = decision_choice(candidate)
        return candidate.audit.outcome_at_strength(strength)

    def oracle_key(candidate: CandidateRecord, partition: str) -> tuple[Any, ...]:
        outcome = (
            decision_choice(candidate)[1]
            if partition == "decision"
            else audit_at_decision_choice(candidate)
        )
        return (
            *(-value for value in outcome.ranking_key(candidate.decision.objective)),
            candidate.candidate_id,
        )

    instance_rows = tuple(instances)
    by_instance = {instance.instance_id: instance for instance in instance_rows}
    if len(by_instance) != len(instance_rows):
        raise ValueError("replay instances contain duplicate instance_id values")
    for instance in instance_rows:
        validate_instance(instance)

    results: list[ReplayResult] = []
    for group in groups:
        instance = by_instance.get(group.instance_id)
        if instance is None:
            raise ValueError(f"replay group {group.group_id!r} has no supplied instance")
        validate_group_against_instance(group, instance)
        if q_cap is not None and group.incumbent.total_qubits > q_cap:
            raise ValueError(f"incumbent in group {group.group_id!r} exceeds q_cap")
        eligible = tuple(
            candidate
            for candidate in group.candidates
            if q_cap is None or candidate.total_qubits <= q_cap
        )
        if not eligible:
            raise ValueError(f"no candidate in group {group.group_id!r} satisfies q_cap")
        if policy.oracle_partition is None:
            view = UnlabeledGroupView.from_records(
                instance=instance,
                group=group,
                candidates=eligible,
            )
            rng = random.Random(stable_seed(seed, "candidate-replay", policy.name, group.group_id))
            assert policy.selector is not None
            selected_id = policy.selector(view, rng)
            by_id = {candidate.candidate_id: candidate for candidate in eligible}
            if selected_id not in by_id:
                raise ValueError("replay policy selected a candidate outside its unlabeled view")
            selected = by_id[selected_id]
        else:
            label = policy.oracle_partition
            if label == "audit":
                _, incumbent_decision = decision_choice(group.incumbent)

                def post_acceptance_audit(
                    candidate: CandidateRecord,
                    incumbent_decision: QualityOutcome = incumbent_decision,
                    incumbent: CandidateRecord = group.incumbent,
                ) -> tuple[Any, ...]:
                    _, candidate_decision = decision_choice(candidate)
                    final = (
                        candidate
                        if candidate_decision.better_than(
                            incumbent_decision,
                            objective=candidate.decision.objective,
                            margin=acceptance.margin,
                        )
                        else incumbent
                    )
                    return oracle_key(final, "audit")

                selected = min(eligible, key=post_acceptance_audit)
            else:
                selected = min(eligible, key=lambda candidate: oracle_key(candidate, label))
        selected_strength, selected_decision = decision_choice(selected)
        _, incumbent_decision = decision_choice(group.incumbent)
        accepted = selected_decision.better_than(
            incumbent_decision,
            objective=selected.decision.objective,
            margin=acceptance.margin,
        )
        final = selected if accepted else group.incumbent
        final_strength, _ = decision_choice(final)
        results.append(
            ReplayResult(
                group_id=group.group_id,
                split=group.split,
                policy_name=policy.name,
                policy_implementation_id=policy.implementation_id,
                policy_deployable=policy.deployable,
                policy_trust_boundary=policy.trust_boundary,
                replay_seed=seed,
                q_cap=q_cap,
                eligible_candidate_ids=tuple(candidate.candidate_id for candidate in eligible),
                selected_candidate_id=selected.candidate_id,
                selected_strength=selected_strength,
                decision_accepted=accepted,
                final_candidate_id=final.candidate_id,
                final_strength=final_strength,
                audit_quality=final.audit.outcome_at_strength(final_strength),
                acceptance_protocol=acceptance.name,
                acceptance_margin=acceptance.margin,
                decision_evaluations=acceptance.decision_evaluations,
            )
        )
    return tuple(results)


def filter_groups_by_split(
    groups: Sequence[CandidateGroup],
    *,
    split: Split,
    instances: Sequence[InstanceRecord],
) -> tuple[CandidateGroup, ...]:
    if split not in {"train", "val", "test"}:
        raise ValueError("unknown split")
    instance_rows = tuple(instances)
    by_instance = {instance.instance_id: instance for instance in instance_rows}
    if len(by_instance) != len(instance_rows):
        raise ValueError("split filtering received duplicate instance_id values")
    assignments: dict[str, Split] = {}
    for group in groups:
        instance = by_instance.get(group.instance_id)
        if instance is None:
            raise ValueError(f"candidate group {group.group_id!r} has no supplied instance")
        validate_group_against_instance(group, instance)
        previous = assignments.setdefault(group.split_unit_id, group.split)
        if previous != group.split:
            raise ValueError(
                f"split leakage: split unit {group.split_unit_id!r} appears in multiple splits"
            )
    return tuple(group for group in groups if group.split == split)

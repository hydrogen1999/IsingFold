"""Prospective, outcome-blind production plan for the IsingFold corpus.

The plan fixes scientific coverage before Minorminer, simulated annealing, exact
energies, or learned models are run.  Execution sharding is recorded for replay but
never enters a lineage identifier or a generated instance's scientific identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_corpus_shard import LineageRequest, lineage_shard
from embedbench.isingfold_design import (
    PRODUCTION_TEST_FLOOR,
    PRODUCTION_TRAIN_FLOOR,
    PRODUCTION_VALIDATION_FLOOR,
    DifficultyRule,
    MeasurementBudget,
)

_SCHEMA = "embedbench.isingfold-corpus-generation-plan"
_VERSION = 2
_HEX = frozenset("0123456789abcdef")
_PARTITION_COUNTS = {
    "train": PRODUCTION_TRAIN_FLOOR,
    "val": PRODUCTION_VALIDATION_FLOOR,
    "test": PRODUCTION_TEST_FLOOR,
}
_TOPOLOGY_SIZE_IID = {"chimera": 4, "pegasus": 3, "zephyr": 2}
_TOPOLOGY_SIZE_OOD = {"chimera": 5, "pegasus": 4, "zephyr": 3}
_APPLICATION_FAMILIES = ("graphcut", "portfolio", "jobshop")
# Fixed before proposal or quality evaluation.  The release-v4 prospective census
# needs at most slot 597, including the sparse Chimera OOD quotient panel.  The next
# power of two leaves deterministic headroom without changing any accepted row.
_PRODUCTION_PROSPECTIVE_SEARCH_SLOTS = 1024


def _exact_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an exact JSON object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return dict(value)


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class CandidateScreeningProtocol:
    """Non-authoritative curves used only to exercise CandidateBank compatibility."""

    attempt_slots: int = 16
    incumbent_search_slots: int = 8
    max_split_search: int = _PRODUCTION_PROSPECTIVE_SEARCH_SLOTS
    strengths: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    reads: int = 8
    sweeps: int = 50
    role: str = "compatibility-screening-not-training-target"

    def __post_init__(self) -> None:
        for name in (
            "attempt_slots",
            "incumbent_search_slots",
            "max_split_search",
            "reads",
            "sweeps",
        ):
            _positive_int(getattr(self, name), name)
        if self.attempt_slots < 2:
            raise ValueError("attempt_slots must permit at least two alternatives")
        if type(self.strengths) is not tuple or not self.strengths:
            raise ValueError("strengths must be a nonempty tuple")
        checked: list[float] = []
        for value in self.strengths:
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError("strengths must be positive finite binary64 values")
            checked.append(value)
        if len(checked) != len(set(checked)):
            raise ValueError("strengths must be unique")
        if self.role != "compatibility-screening-not-training-target":
            raise ValueError("CandidateBank screening curves cannot be training targets")

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "strengths": list(self.strengths)}

    @classmethod
    def from_dict(cls, value: object) -> CandidateScreeningProtocol:
        document = _exact_mapping(
            value,
            {
                "attempt_slots",
                "incumbent_search_slots",
                "max_split_search",
                "reads",
                "role",
                "strengths",
                "sweeps",
            },
            "candidate screening protocol",
        )
        strengths = document["strengths"]
        if type(strengths) is not list:
            raise ValueError("screening strengths must be a JSON array")
        return cls(**{**document, "strengths": tuple(strengths)})


@dataclass(frozen=True, slots=True)
class PlannedLineage:
    request: LineageRequest
    difficulty_intent: Literal["easy", "hard"]
    shift_axes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, LineageRequest):
            raise TypeError("planned lineage request must be a LineageRequest")
        if self.difficulty_intent not in {"easy", "hard"}:
            raise ValueError("difficulty_intent must be easy or hard")
        if type(self.shift_axes) is not tuple:
            raise TypeError("shift_axes must be an immutable tuple")
        if self.shift_axes != tuple(sorted(set(self.shift_axes))):
            raise ValueError("shift_axes must be sorted and unique")
        request = self.request
        if request.distribution_regime == "ood":
            if request.partition != "test" or not self.shift_axes:
                raise ValueError("OOD lineages must be test-only and name a real shift")
            if self.difficulty_intent != "hard":
                raise ValueError("the registered OOD panel is a hard stress panel")
            required = {"fault-severity", "host-size", "logical-size"}
            if not required <= set(self.shift_axes):
                raise ValueError(
                    "OOD shift axes do not establish the registered distribution shift"
                )
            if request.host_size != _TOPOLOGY_SIZE_OOD[request.topology]:
                raise ValueError("OOD host size does not differ from the IID support")
            if request.n_variables not in {14, 16}:
                raise ValueError("OOD logical size must lie outside the IID support")
            if request.qubit_fraction < 0.04 or request.coupler_fraction < 0.06:
                raise ValueError("OOD fault severity must lie outside the IID support")
            if request.origin == "synthetic-ink-drop" and (
                "ink-mode" not in self.shift_axes
                or request.ink_drop_mode
                not in {
                    "cut_congested",
                    "near_capacity",
                }
            ):
                raise ValueError("synthetic OOD requires a held-out ink mode")
        else:
            if self.shift_axes:
                raise ValueError("IID lineages cannot claim distribution-shift axes")
            if request.host_size != _TOPOLOGY_SIZE_IID[request.topology]:
                raise ValueError("IID lineage uses an out-of-support host size")
            if request.n_variables not in {6, 8, 10, 12}:
                raise ValueError("IID logical size is outside the registered support")
            if request.qubit_fraction > 0.02 or request.coupler_fraction > 0.03:
                raise ValueError("IID fault severity exceeds the registered support")
            if request.origin == "synthetic-ink-drop" and request.ink_drop_mode not in {
                "compact",
                "elongated",
            }:
                raise ValueError("IID synthetic lineage uses a held-out ink mode")
        if self.difficulty_intent == "easy":
            if request.n_variables not in {6, 8}:
                raise ValueError("easy intent must use the small logical-size panel")
            if request.qubit_fraction != 0.0 or request.coupler_fraction != 0.0:
                raise ValueError("easy intent must use a pristine host")
        elif request.distribution_regime == "iid":
            if request.n_variables not in {10, 12}:
                raise ValueError("hard IID intent must use the large IID logical-size panel")
            if request.qubit_fraction <= 0.0 or request.coupler_fraction <= 0.0:
                raise ValueError("hard IID intent must include realized host faults")

    def to_dict(self) -> dict[str, object]:
        return {
            "difficulty_intent": self.difficulty_intent,
            "request": asdict(self.request),
            "shift_axes": list(self.shift_axes),
        }

    @classmethod
    def from_dict(cls, value: object) -> PlannedLineage:
        document = _exact_mapping(
            value,
            {"difficulty_intent", "request", "shift_axes"},
            "planned lineage",
        )
        request = _exact_mapping(
            document["request"],
            {
                "application_family",
                "coupler_fraction",
                "distribution_regime",
                "host_size",
                "ink_chain_size",
                "ink_drop_mode",
                "lineage_id",
                "n_variables",
                "origin",
                "partition",
                "qubit_fraction",
                "topology",
            },
            "lineage request",
        )
        axes = document["shift_axes"]
        if type(axes) is not list or any(type(item) is not str or not item for item in axes):
            raise ValueError("shift_axes must be a JSON array of nonempty strings")
        return cls(
            request=LineageRequest(**request),
            difficulty_intent=document["difficulty_intent"],
            shift_axes=tuple(axes),
        )


@dataclass(frozen=True, slots=True)
class ProspectiveIdentityPolicy:
    """Frozen plan-level rule that prevents repeated logical problem identities."""

    enforcement: str = "pre-generation-fail-closed"
    identity_fields: tuple[str, ...] = ("problem_sha256", "split_unit_id")
    map_entry_fields: tuple[str, ...] = (
        "lineage_id",
        "prospective_slot",
        "split_unit_id",
        "problem_sha256",
    )
    map_protocol: str = "canonical-prospective-identity-map-v1"
    requirement: str = "global-uniqueness-across-planned-lineages"

    def __post_init__(self) -> None:
        if self.enforcement != "pre-generation-fail-closed":
            raise ValueError("prospective identity policy must fail closed before generation")
        if self.identity_fields != ("problem_sha256", "split_unit_id"):
            raise ValueError("prospective identity fields differ from the frozen policy")
        if self.map_entry_fields != (
            "lineage_id",
            "prospective_slot",
            "split_unit_id",
            "problem_sha256",
        ):
            raise ValueError("prospective identity map entry fields differ from the frozen policy")
        if self.map_protocol != "canonical-prospective-identity-map-v1":
            raise ValueError("prospective identity map protocol differs from the frozen policy")
        if self.requirement != "global-uniqueness-across-planned-lineages":
            raise ValueError("prospective identity policy must require global uniqueness")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "identity_fields": list(self.identity_fields),
            "map_entry_fields": list(self.map_entry_fields),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProspectiveIdentityPolicy:
        document = _exact_mapping(
            value,
            {
                "enforcement",
                "identity_fields",
                "map_entry_fields",
                "map_protocol",
                "requirement",
            },
            "prospective identity policy",
        )
        identity_fields = document["identity_fields"]
        map_entry_fields = document["map_entry_fields"]
        if type(identity_fields) is not list or type(map_entry_fields) is not list:
            raise ValueError("prospective identity policy fields must be JSON arrays")
        return cls(
            enforcement=document["enforcement"],
            identity_fields=tuple(identity_fields),
            map_entry_fields=tuple(map_entry_fields),
            map_protocol=document["map_protocol"],
            requirement=document["requirement"],
        )


@dataclass(frozen=True, slots=True)
class ReleaseScope:
    """Exact generalization claim made by this corpus release."""

    application_family_coverage: str = "graphcut-portfolio-jobshop-in-train-val-test"
    ood_claim_scope: str = (
        "registered-topology-host-scale-fault-and-distribution-regime-shifts"
    )
    profile_id: str = "isingfold-corpus-v4"
    separate_unseen_family_profile: str = "docs/HARD_OOD_CORPUS_SPEC.md"
    unseen_application_family_ood: bool = False

    def __post_init__(self) -> None:
        if (
            self.application_family_coverage
            != "graphcut-portfolio-jobshop-in-train-val-test"
            or self.ood_claim_scope
            != "registered-topology-host-scale-fault-and-distribution-regime-shifts"
            or self.profile_id != "isingfold-corpus-v4"
            or self.separate_unseen_family_profile != "docs/HARD_OOD_CORPUS_SPEC.md"
            or self.unseen_application_family_ood is not False
        ):
            raise ValueError("release scope differs from the frozen IsingFold v4 claim")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> ReleaseScope:
        document = _exact_mapping(
            value,
            {
                "application_family_coverage",
                "ood_claim_scope",
                "profile_id",
                "separate_unseen_family_profile",
                "unseen_application_family_ood",
            },
            "release scope",
        )
        return cls(**document)


@dataclass(frozen=True, slots=True)
class CorpusGenerationPlan:
    root_seed: int
    shard_count: int
    screening: CandidateScreeningProtocol
    measurement_budget: MeasurementBudget
    difficulty_rules: tuple[DifficultyRule, ...]
    lineages: tuple[PlannedLineage, ...]
    prospective_identity_policy: ProspectiveIdentityPolicy
    release_scope: ReleaseScope
    record_digest: str

    def __post_init__(self) -> None:
        if type(self.root_seed) is not int or self.root_seed < 0:
            raise ValueError("root_seed must be a non-negative integer")
        _positive_int(self.shard_count, "shard_count")
        if not isinstance(self.screening, CandidateScreeningProtocol):
            raise TypeError("screening must be a CandidateScreeningProtocol")
        if (
            self.screening.max_split_search
            < _PRODUCTION_PROSPECTIVE_SEARCH_SLOTS
        ):
            raise ValueError(
                "production prospective eligibility requires at least "
                f"{_PRODUCTION_PROSPECTIVE_SEARCH_SLOTS} fixed split-search slots"
            )
        if not isinstance(self.measurement_budget, MeasurementBudget):
            raise TypeError("measurement_budget must be a MeasurementBudget")
        if type(self.difficulty_rules) is not tuple or any(
            not isinstance(item, DifficultyRule) for item in self.difficulty_rules
        ):
            raise TypeError("difficulty_rules must be an immutable tuple of DifficultyRule values")
        if tuple(rule.field for rule in self.difficulty_rules) != (
            "decision_difficulty",
            "embedding_difficulty",
            "sampling_difficulty",
        ):
            raise ValueError("difficulty_rules must be canonically ordered and complete")
        if {rule.metric for rule in self.difficulty_rules} != {
            "candidate_resource_spread",
            "host_fill_fraction",
            "logical_edge_density",
        }:
            raise ValueError("difficulty rules must bind the three generated structural metrics")
        if type(self.lineages) is not tuple or any(
            not isinstance(item, PlannedLineage) for item in self.lineages
        ):
            raise TypeError("lineages must be an immutable tuple of PlannedLineage values")
        if not isinstance(self.prospective_identity_policy, ProspectiveIdentityPolicy):
            raise TypeError("plan must freeze a ProspectiveIdentityPolicy")
        if not isinstance(self.release_scope, ReleaseScope):
            raise TypeError("plan must freeze its ReleaseScope")
        ids = tuple(item.request.lineage_id for item in self.lineages)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("planned lineage IDs must be sorted and unique")
        occupied_shards = {lineage_shard(lineage_id, self.shard_count) for lineage_id in ids}
        if len(occupied_shards) != self.shard_count:
            first_empty = next(
                index for index in range(self.shard_count) if index not in occupied_shards
            )
            raise ValueError(
                "every shard index must contain at least one lineage; "
                f"first empty shard index is {first_empty}"
            )
        counts = {partition: 0 for partition in _PARTITION_COUNTS}
        for item in self.lineages:
            counts[item.request.partition] += 1
        if counts != _PARTITION_COUNTS:
            raise ValueError(f"production partition counts differ: {counts}")
        for partition in _PARTITION_COUNTS:
            rows = [item for item in self.lineages if item.request.partition == partition]
            if {item.request.origin for item in rows} != {
                "application-derived",
                "synthetic-ink-drop",
            }:
                raise ValueError(f"{partition} does not contain both problem origins")
            if {item.request.topology for item in rows} != set(_TOPOLOGY_SIZE_IID):
                raise ValueError(f"{partition} does not contain every host family")
            if {item.difficulty_intent for item in rows} != {"easy", "hard"}:
                raise ValueError(f"{partition} does not contain easy and hard panels")
            faulted = [
                item
                for item in rows
                if item.request.qubit_fraction > 0.0 or item.request.coupler_fraction > 0.0
            ]
            if not faulted or len(faulted) == len(rows):
                raise ValueError(f"{partition} must contain pristine and faulted hosts")
        _sha256(self.record_digest, "plan record digest")
        if self.record_digest != content_digest(self._payload()):
            raise ValueError("corpus plan record digest mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "lineages": [item.to_dict() for item in self.lineages],
            "difficulty_rules": [rule.as_dict() for rule in self.difficulty_rules],
            "measurement_budget": asdict(self.measurement_budget),
            "prospective_identity_policy": self.prospective_identity_policy.to_dict(),
            "release_scope": self.release_scope.to_dict(),
            "root_seed": self.root_seed,
            "schema": _SCHEMA,
            "schema_version": _VERSION,
            "screening": self.screening.to_dict(),
            "shard_count": self.shard_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "record_digest": self.record_digest}

    @classmethod
    def from_dict(cls, value: object) -> CorpusGenerationPlan:
        document = _exact_mapping(
            value,
            {
                "lineages",
                "difficulty_rules",
                "measurement_budget",
                "prospective_identity_policy",
                "release_scope",
                "record_digest",
                "root_seed",
                "schema",
                "schema_version",
                "screening",
                "shard_count",
            },
            "corpus generation plan",
        )
        if document["schema"] != _SCHEMA or document["schema_version"] != _VERSION:
            raise ValueError("unsupported corpus generation plan schema")
        raw_lineages = document["lineages"]
        if type(raw_lineages) is not list:
            raise ValueError("plan lineages must be a JSON array")
        raw_rules = document["difficulty_rules"]
        if type(raw_rules) is not list:
            raise ValueError("difficulty_rules must be a JSON array")
        rules: list[DifficultyRule] = []
        for raw_rule in raw_rules:
            rule = _exact_mapping(
                raw_rule,
                {"field", "hard_if", "metric", "threshold"},
                "difficulty rule",
            )
            rules.append(DifficultyRule(**rule))
        budget = _exact_mapping(
            document["measurement_budget"],
            {
                "decision_evaluations",
                "embedding_attempts",
                "measurement_budget_id",
                "sampler_reads",
            },
            "measurement budget",
        )
        return cls(
            root_seed=document["root_seed"],
            shard_count=document["shard_count"],
            screening=CandidateScreeningProtocol.from_dict(document["screening"]),
            measurement_budget=MeasurementBudget(**budget),
            difficulty_rules=tuple(rules),
            lineages=tuple(PlannedLineage.from_dict(item) for item in raw_lineages),
            prospective_identity_policy=ProspectiveIdentityPolicy.from_dict(
                document["prospective_identity_policy"]
            ),
            release_scope=ReleaseScope.from_dict(document["release_scope"]),
            record_digest=document["record_digest"],
        )


@dataclass(frozen=True, slots=True)
class CorpusPlanWriteReceipt:
    path: Path
    sha256: str
    record_digest: str
    lineage_count: int


def _lineage(
    *,
    root_seed: int,
    partition: str,
    ordinal: int,
) -> PlannedLineage:
    topology = ("chimera", "pegasus", "zephyr")[(ordinal // 2) % 3]
    origin = ("application-derived", "synthetic-ink-drop")[ordinal % 2]
    is_ood = partition == "test" and (ordinal // 12) % 2 == 1
    difficulty: Literal["easy", "hard"] = (
        "hard" if is_ood or (ordinal // 6) % 2 == 1 else "easy"
    )
    panel = (ordinal // 2) % 2
    if is_ood:
        n_variables = (14, 16)[panel]
        qubit_fraction = (0.04, 0.06)[panel]
        coupler_fraction = (0.06, 0.08)[(ordinal // 4) % 2]
        host_size = _TOPOLOGY_SIZE_OOD[topology]
        ink_mode = ("cut_congested", "near_capacity")[panel]
        ink_chain_size = 4
        axes = (
            "fault-severity",
            "host-size",
            *(("ink-mode",) if origin == "synthetic-ink-drop" else ()),
            "logical-size",
        )
        shift_axes = tuple(sorted(axes))
    elif difficulty == "easy":
        n_variables = (6, 8)[panel]
        qubit_fraction = 0.0
        coupler_fraction = 0.0
        host_size = _TOPOLOGY_SIZE_IID[topology]
        ink_mode = ("compact", "elongated")[panel]
        ink_chain_size = {"chimera": 8, "pegasus": 3, "zephyr": 3}[topology]
        shift_axes = ()
    else:
        n_variables = (10, 12)[panel]
        qubit_fraction = (0.01, 0.02)[panel]
        coupler_fraction = (0.02, 0.03)[(ordinal // 4) % 2]
        host_size = _TOPOLOGY_SIZE_IID[topology]
        ink_mode = ("compact", "elongated")[panel]
        ink_chain_size = 4
        shift_axes = ()
    family = (
        _APPLICATION_FAMILIES[(ordinal // 2 + ordinal // 18) % len(_APPLICATION_FAMILIES)]
        if origin == "application-derived"
        else "ink-drop-quotient"
    )
    identity = content_digest(
        {
            "domain": "isingfold-production-lineage-plan-v4",
            "ordinal": ordinal,
            "partition": partition,
            "root_seed": root_seed,
        }
    )[:16]
    request = LineageRequest(
        lineage_id=f"if-prod-v4-{partition}-{ordinal:04d}-{identity}",
        application_family=family,
        origin=origin,
        partition=partition,
        distribution_regime="ood" if is_ood else "iid",
        topology=topology,
        host_size=host_size,
        n_variables=n_variables,
        qubit_fraction=qubit_fraction,
        coupler_fraction=coupler_fraction,
        ink_chain_size=ink_chain_size,
        ink_drop_mode=ink_mode,
    )
    return PlannedLineage(
        request=request,
        difficulty_intent=difficulty,
        shift_axes=shift_axes,
    )


def build_production_corpus_plan(
    *,
    root_seed: int = 260912,
    shard_count: int = 64,
    screening: CandidateScreeningProtocol | None = None,
) -> CorpusGenerationPlan:
    """Freeze the minimum powered 3,082-lineage corpus before observing outcomes."""

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    _positive_int(shard_count, "shard_count")
    selected_screening = screening or CandidateScreeningProtocol()
    measurement_budget = MeasurementBudget(
        measurement_budget_id="isingfold-candidate-screen-v1-a16-i8-r8-s50",
        decision_evaluations=selected_screening.attempt_slots,
        embedding_attempts=(
            selected_screening.attempt_slots + selected_screening.incumbent_search_slots
        ),
        sampler_reads=(
            selected_screening.reads * len(selected_screening.strengths) * 2
        ),
    )
    difficulty_rules = (
        DifficultyRule(
            field="decision_difficulty",
            metric="candidate_resource_spread",
            hard_if="greater-than-or-equal",
            threshold=0.01,
        ),
        DifficultyRule(
            field="embedding_difficulty",
            metric="host_fill_fraction",
            hard_if="greater-than-or-equal",
            threshold=0.08,
        ),
        DifficultyRule(
            field="sampling_difficulty",
            metric="logical_edge_density",
            hard_if="greater-than-or-equal",
            threshold=0.3,
        ),
    )
    lineages = tuple(
        sorted(
            (
                _lineage(root_seed=root_seed, partition=partition, ordinal=ordinal)
                for partition, count in _PARTITION_COUNTS.items()
                for ordinal in range(count)
            ),
            key=lambda item: item.request.lineage_id,
        )
    )
    prospective_identity_policy = ProspectiveIdentityPolicy()
    release_scope = ReleaseScope()
    payload = {
        "lineages": [item.to_dict() for item in lineages],
        "difficulty_rules": [rule.as_dict() for rule in difficulty_rules],
        "measurement_budget": asdict(measurement_budget),
        "prospective_identity_policy": prospective_identity_policy.to_dict(),
        "release_scope": release_scope.to_dict(),
        "root_seed": root_seed,
        "schema": _SCHEMA,
        "schema_version": _VERSION,
        "screening": selected_screening.to_dict(),
        "shard_count": shard_count,
    }
    return CorpusGenerationPlan(
        root_seed=root_seed,
        shard_count=shard_count,
        screening=selected_screening,
        measurement_budget=measurement_budget,
        difficulty_rules=difficulty_rules,
        lineages=lineages,
        prospective_identity_policy=prospective_identity_policy,
        release_scope=release_scope,
        record_digest=content_digest(payload),
    )


def write_corpus_plan(
    plan: CorpusGenerationPlan,
    path: str | os.PathLike[str],
) -> CorpusPlanWriteReceipt:
    if not isinstance(plan, CorpusGenerationPlan):
        raise TypeError("plan must be a CorpusGenerationPlan")
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"corpus plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(plan.to_dict()) + b"\n"
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return CorpusPlanWriteReceipt(
        path=destination.resolve(strict=True),
        sha256=hashlib.sha256(raw).hexdigest(),
        record_digest=plan.record_digest,
        lineage_count=len(plan.lineages),
    )


def _strict_json(raw: bytes) -> object:
    def pairs(items: Sequence[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"corpus plan repeats JSON key {key!r}")
            output[key] = value
        return output

    def constant(token: str) -> None:
        raise ValueError(f"corpus plan contains non-finite number {token}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def read_corpus_plan(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
) -> CorpusGenerationPlan:
    expected = _sha256(expected_sha256, "expected plan SHA-256")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("corpus plan must be a regular non-symlink file")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("corpus plan SHA-256 mismatch")
    document = _strict_json(raw)
    plan = CorpusGenerationPlan.from_dict(document)
    if raw != canonical_json_bytes(plan.to_dict()) + b"\n":
        raise ValueError("corpus plan is not canonical JSON")
    return plan


__all__ = [
    "CandidateScreeningProtocol",
    "CorpusGenerationPlan",
    "CorpusPlanWriteReceipt",
    "PlannedLineage",
    "ProspectiveIdentityPolicy",
    "ReleaseScope",
    "build_production_corpus_plan",
    "read_corpus_plan",
    "write_corpus_plan",
]

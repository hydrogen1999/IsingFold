"""Target-free authentication and aggregation of quality-resolution deltas.

The phase-2 workers are the only components allowed to open evaluator targets.  This module
consumes their compact, independently replayed artifacts through an out-of-band pin registry,
checks the exact cumulative ``[0, C)`` census, applies the preregistered finite-population rule,
and publishes an immutable study receipt.  It intentionally has no prepared-target loader.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality_resolution import (
    BONFERRONI_ALPHA_DENOMINATOR,
    BONFERRONI_ALPHA_NUMERATOR,
    BONFERRONI_COMPARISONS,
    FAMILYWISE_ALPHA_DENOMINATOR,
    FAMILYWISE_ALPHA_NUMERATOR,
    QualityResolutionError,
    REGISTERED_CANDIDATE_CONTINUATIONS,
    select_continuation_count,
)
from isingfold.rl.data.quality_resolution_delta import (
    QUALITY_RESOLUTION_VERIFICATION_SCHEMA,
    QUALITY_RESOLUTION_VERIFICATION_VERSION,
    QUALITY_RESOLUTION_VERIFIER_IDENTITY_SCHEMA,
    QUALITY_RESOLUTION_VERIFIER_IDENTITY_VERSION,
    QualityResolutionDeltaArtifact,
    _load_delta_records,
    _plan_identity,
    _publish_file,
    _read_pinned_json,
    _thaw,
    _verify_record,
)
from isingfold.rl.data.quality_resolution_plan import QualityResolutionPlan


QUALITY_RESOLUTION_PIN_REGISTRY_SCHEMA = "isingfold.quality-resolution-shard-pin-registry"
QUALITY_RESOLUTION_PIN_REGISTRY_VERSION = 1
QUALITY_RESOLUTION_STUDY_SCHEMA = "isingfold.quality-continuation-resolution-study"
QUALITY_RESOLUTION_STUDY_VERSION = 1

_HEX = frozenset("0123456789abcdef")
_PIN_FIELDS = {
    "delta_manifest_sha256",
    "delta_root",
    "shard_index",
    "stage_index",
    "verification_path",
    "verification_sha256",
}
_REGISTRY_FIELDS = {
    "completed_continuation_range",
    "completed_stage_index",
    "pins",
    "plan_record_digest",
    "plan_sha256",
    "record_digest",
    "schema",
    "schema_version",
}
_VERIFICATION_FIELDS = {
    "delta_manifest_record_digest",
    "delta_manifest_sha256",
    "execution_identity_digest",
    "matching_count",
    "mismatches",
    "pass",
    "plan_record_digest",
    "plan_sha256",
    "record_digest",
    "replayed_count",
    "schema",
    "schema_version",
    "shard_index",
    "stage_index",
    "verifier_identity",
    "verifier_identity_digest",
    "verifier_identity_sha256",
}
_SOURCE_FIELDS = {
    "delta_manifest_record_digest",
    "delta_manifest_sha256",
    "record_count",
    "record_set_digest",
    "records_sha256",
    "shard_index",
    "stage_index",
    "verification_record_digest",
    "verification_sha256",
}
_RESULT_FIELDS = {
    "continuations",
    "lower_population_bound",
    "passes",
    "sampled_resolved_lineages",
    "sampled_resolved_rows",
    "sampled_unresolved_lineages",
}
_STUDY_FIELDS = {
    "advance",
    "candidate_ladder",
    "completed_continuation_range",
    "confidence",
    "config",
    "context",
    "corpus",
    "ground_root",
    "implementation",
    "pin_registry",
    "plan",
    "population",
    "production_protocol",
    "publisher",
    "quality_authority",
    "record_digest",
    "results",
    "sample",
    "schema",
    "schema_version",
    "selected_continuations",
    "selector",
    "source_shards",
    "stop_reason",
    "study_id",
    "terminal",
}


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise QualityResolutionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonnegative(value: object, label: str, *, upper: int | None = None) -> int:
    if type(value) is not int or value < 0 or (upper is not None and value >= upper):
        raise QualityResolutionError(f"{label} is outside its registered range")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _relative_artifact_path(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise QualityResolutionError(f"{label} must be a nonempty relative POSIX path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or value != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise QualityResolutionError(f"{label} must stay beneath the pin-registry directory")
    return value


@dataclass(frozen=True, slots=True)
class QualityResolutionShardPin:
    """Out-of-band raw-file pins for one delta and its independent verification."""

    stage_index: int
    shard_index: int
    delta_root: str
    delta_manifest_sha256: str
    verification_path: str
    verification_sha256: str

    def __post_init__(self) -> None:
        _nonnegative(self.stage_index, "resolution pin stage index")
        _nonnegative(self.shard_index, "resolution pin shard index")
        _relative_artifact_path(self.delta_root, "resolution delta root")
        _digest(self.delta_manifest_sha256, "resolution delta-manifest pin")
        _relative_artifact_path(self.verification_path, "resolution verification path")
        _digest(self.verification_sha256, "resolution verification pin")

    def as_dict(self) -> dict[str, object]:
        return {
            "delta_manifest_sha256": self.delta_manifest_sha256,
            "delta_root": self.delta_root,
            "shard_index": self.shard_index,
            "stage_index": self.stage_index,
            "verification_path": self.verification_path,
            "verification_sha256": self.verification_sha256,
        }


@dataclass(frozen=True, slots=True)
class QualityResolutionStudyReceipt:
    """Read-only canonical resolution-study receipt."""

    record: Mapping[str, object]
    raw_sha256: str

    def __post_init__(self) -> None:
        _digest(self.raw_sha256, "quality-resolution study receipt SHA-256")
        object.__setattr__(self, "record", _freeze(self.record))

    def as_dict(self) -> dict[str, Any]:
        value = _thaw(self.record)
        if not isinstance(value, dict):  # pragma: no cover - constructor is mapping-only
            raise RuntimeError("quality-resolution receipt lost its object shape")
        return value


@dataclass(frozen=True, slots=True)
class _AuthenticatedSource:
    pin: QualityResolutionShardPin
    delta: QualityResolutionDeltaArtifact
    verification: Mapping[str, object]
    verification_sha256: str


def _expected_stage_range(plan_record: Mapping[str, object], stage_index: int) -> list[int]:
    stages = plan_record.get("stages")
    if not isinstance(stages, list):
        raise QualityResolutionError("quality-resolution plan has no stage registry")
    stage = stages[_nonnegative(stage_index, "completed stage index", upper=len(stages))]
    if not isinstance(stage, Mapping) or stage.get("stage_index") != stage_index:
        raise QualityResolutionError("quality-resolution plan stage registry is noncanonical")
    delta = stage.get("continuation_range")
    if (
        not isinstance(delta, list)
        or len(delta) != 2
        or any(type(item) is not int for item in delta)
    ):
        raise QualityResolutionError("quality-resolution plan stage range is malformed")
    return [0, int(delta[1])]


def _validate_pin_census(
    pins: Sequence[QualityResolutionShardPin],
    *,
    plan_record: Mapping[str, object],
    completed_stage_index: int,
) -> tuple[QualityResolutionShardPin, ...]:
    shards = plan_record.get("shards")
    stages = plan_record.get("stages")
    if not isinstance(shards, list) or not shards or not isinstance(stages, list):
        raise QualityResolutionError("resolution plan has no complete shard/stage registry")
    completed = _nonnegative(
        completed_stage_index, "completed stage index", upper=len(stages)
    )
    canonical = tuple(sorted(pins, key=lambda pin: (pin.stage_index, pin.shard_index)))
    observed = [(pin.stage_index, pin.shard_index) for pin in canonical]
    expected = [
        (stage_index, shard_index)
        for stage_index in range(completed + 1)
        for shard_index in range(len(shards))
    ]
    if observed != expected:
        raise QualityResolutionError(
            "resolution pin registry does not contain the complete stage/shard census"
        )
    return canonical


def publish_quality_resolution_shard_pin_registry(
    path: str | Path,
    *,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    completed_stage_index: int,
    pins: Sequence[QualityResolutionShardPin],
) -> str:
    """Seal the externally prepared raw pins for one progressive stage prefix."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    if isinstance(pins, (str, bytes)) or not isinstance(pins, Sequence):
        raise TypeError("resolution shard pins must be a sequence")
    if any(not isinstance(pin, QualityResolutionShardPin) for pin in pins):
        raise TypeError("resolution shard pins must be QualityResolutionShardPin values")
    canonical = _validate_pin_census(
        pins, plan_record=plan_record, completed_stage_index=completed_stage_index
    )
    payload: dict[str, object] = {
        "completed_continuation_range": _expected_stage_range(
            plan_record, completed_stage_index
        ),
        "completed_stage_index": completed_stage_index,
        "pins": [pin.as_dict() for pin in canonical],
        "plan_record_digest": plan_record["record_digest"],
        "plan_sha256": plan_sha256,
        "schema": QUALITY_RESOLUTION_PIN_REGISTRY_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_PIN_REGISTRY_VERSION,
    }
    registry = {**payload, "record_digest": content_digest(payload)}
    return _publish_file(Path(path), registry)


def _pin_from_dict(raw: object) -> QualityResolutionShardPin:
    if not isinstance(raw, Mapping) or set(raw) != _PIN_FIELDS:
        raise QualityResolutionError("resolution shard-pin entry schema differs")
    return QualityResolutionShardPin(
        stage_index=raw["stage_index"],  # type: ignore[arg-type]
        shard_index=raw["shard_index"],  # type: ignore[arg-type]
        delta_root=raw["delta_root"],  # type: ignore[arg-type]
        delta_manifest_sha256=raw["delta_manifest_sha256"],  # type: ignore[arg-type]
        verification_path=raw["verification_path"],  # type: ignore[arg-type]
        verification_sha256=raw["verification_sha256"],  # type: ignore[arg-type]
    )


def _load_pin_registry(
    path: str | Path,
    expected_sha256: str,
    *,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
) -> tuple[dict[str, Any], tuple[QualityResolutionShardPin, ...], str]:
    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    registry, _raw, observed_sha256 = _read_pinned_json(
        path, expected_sha256, "quality-resolution shard-pin registry"
    )
    if set(registry) != _REGISTRY_FIELDS:
        raise QualityResolutionError("quality-resolution shard-pin registry schema differs")
    _verify_record(registry, "quality-resolution shard-pin registry")
    if (
        registry.get("schema") != QUALITY_RESOLUTION_PIN_REGISTRY_SCHEMA
        or registry.get("schema_version") != QUALITY_RESOLUTION_PIN_REGISTRY_VERSION
        or registry.get("plan_sha256") != plan_sha256
        or registry.get("plan_record_digest") != plan_record["record_digest"]
    ):
        raise QualityResolutionError("shard-pin registry belongs to another resolution plan")
    raw_pins = registry.get("pins")
    if not isinstance(raw_pins, list):
        raise QualityResolutionError("shard-pin registry pins must be an array")
    pins = tuple(_pin_from_dict(raw) for raw in raw_pins)
    completed = _nonnegative(
        registry.get("completed_stage_index"),
        "completed stage index",
        upper=len(plan_record["stages"]),  # type: ignore[arg-type]
    )
    canonical = _validate_pin_census(
        pins, plan_record=plan_record, completed_stage_index=completed
    )
    if list(pins) != list(canonical):
        raise QualityResolutionError("shard-pin registry entries are not canonically ordered")
    if registry.get("completed_continuation_range") != _expected_stage_range(
        plan_record, completed
    ):
        raise QualityResolutionError("shard-pin registry cumulative range differs from plan")
    return registry, canonical, observed_sha256


def _resolve_registered_path(registry_path: Path, relative: str, label: str) -> Path:
    _relative_artifact_path(relative, label)
    base = registry_path.parent.resolve()
    candidate = registry_path.parent.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise QualityResolutionError(f"{label} is missing") from error
    if not resolved.is_relative_to(base):
        raise QualityResolutionError(f"{label} escapes the pin-registry directory")
    return candidate


def _load_verification(
    path: Path,
    expected_sha256: str,
    *,
    delta: QualityResolutionDeltaArtifact,
    plan_record: Mapping[str, object],
    plan_sha256: str,
) -> tuple[dict[str, Any], str]:
    receipt, _raw, observed_sha = _read_pinned_json(
        path, expected_sha256, "quality-resolution verification receipt"
    )
    if set(receipt) != _VERIFICATION_FIELDS:
        raise QualityResolutionError("quality-resolution verification receipt schema differs")
    _verify_record(receipt, "quality-resolution verification receipt")
    verifier = receipt.get("verifier_identity")
    if not isinstance(verifier, Mapping):
        raise QualityResolutionError("quality-resolution verifier identity is malformed")
    verifier_record_digest = _verify_record(
        verifier, "quality-resolution verifier identity"
    )
    if (
        set(verifier)
        != {
            "attestor_id",
            "continuation_runner",
            "quality_implementation_contract_digest",
            "quality_module_sha256",
            "record_digest",
            "schema",
            "schema_version",
            "verification_module_sha256",
            "verification_runtime_sha256",
        }
        or verifier.get("schema") != QUALITY_RESOLUTION_VERIFIER_IDENTITY_SCHEMA
        or verifier.get("schema_version")
        != QUALITY_RESOLUTION_VERIFIER_IDENTITY_VERSION
        or verifier.get("continuation_runner")
        != "isingfold.rl.data.quality.run_continuation"
    ):
        raise QualityResolutionError("quality-resolution verifier identity schema differs")
    verifier_sha = hashlib.sha256(canonical_json_bytes(verifier) + b"\n").hexdigest()
    if (
        verifier_record_digest
        != _digest(
            receipt.get("verifier_identity_digest"),
            "resolution verifier identity digest",
        )
        or verifier_sha
        != _digest(
            receipt.get("verifier_identity_sha256"),
            "resolution verifier identity SHA-256",
        )
    ):
        raise QualityResolutionError("quality-resolution verifier identity cannot be reproduced")
    expected_count = int(delta.manifest["record_count"])
    if (
        receipt.get("schema") != QUALITY_RESOLUTION_VERIFICATION_SCHEMA
        or receipt.get("schema_version") != QUALITY_RESOLUTION_VERIFICATION_VERSION
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("plan_record_digest") != plan_record["record_digest"]
        or receipt.get("stage_index") != delta.manifest["stage_index"]
        or receipt.get("shard_index") != delta.manifest["shard_index"]
        or receipt.get("delta_manifest_sha256") != delta.manifest_sha256
        or receipt.get("delta_manifest_record_digest") != delta.manifest["record_digest"]
        or receipt.get("execution_identity_digest")
        != delta.manifest["execution_identity_digest"]
        or receipt.get("replayed_count") != expected_count
        or receipt.get("matching_count") != expected_count
        or receipt.get("mismatches") != []
        or receipt.get("pass") is not True
    ):
        raise QualityResolutionError(
            "quality-resolution verification does not prove an exact complete replay"
        )
    return receipt, observed_sha


def _plan_implementation_contract(
    plan_record: Mapping[str, object],
) -> tuple[str, str, str]:
    implementation = plan_record.get("implementation")
    if not isinstance(implementation, Mapping):
        raise QualityResolutionError("resolution plan implementation identity is malformed")
    registry = implementation.get("registry")
    if not isinstance(registry, Mapping):
        raise QualityResolutionError("resolution plan implementation registry is malformed")
    contract = _digest(
        registry.get("quality_implementation_contract_digest"),
        "planned quality implementation-contract digest",
    )
    source = _digest(
        registry.get("quality_module_sha256"), "planned quality module SHA-256"
    )
    delta_source = _digest(
        registry.get("quality_resolution_delta_module_sha256"),
        "planned quality-resolution delta module SHA-256",
    )
    return contract, source, delta_source


def _source_signature(
    manifest: Mapping[str, object], verification: Mapping[str, object]
) -> dict[str, object]:
    execution = manifest.get("execution_identity")
    verifier = verification.get("verifier_identity")
    if not isinstance(execution, Mapping) or not isinstance(verifier, Mapping):
        raise QualityResolutionError("resolution source implementation identity is malformed")
    return {
        "context_digest": manifest.get("context_digest"),
        "execution_identity": dict(execution),
        "execution_identity_digest": manifest.get("execution_identity_digest"),
        "ground_partition_receipt_record_digest": manifest.get(
            "ground_partition_receipt_record_digest"
        ),
        "ground_partition_receipt_sha256": manifest.get(
            "ground_partition_receipt_sha256"
        ),
        "quality_authority_record_digest": manifest.get(
            "quality_authority_record_digest"
        ),
        "target_access_record_digest": manifest.get("target_access_record_digest"),
        "verifier_identity": dict(verifier),
        "verifier_identity_digest": verification.get("verifier_identity_digest"),
        "verifier_identity_sha256": verification.get("verifier_identity_sha256"),
    }


def _authenticate_sources(
    registry_path: Path,
    pins: Sequence[QualityResolutionShardPin],
    *,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
) -> tuple[tuple[_AuthenticatedSource, ...], dict[str, object]]:
    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    contract_digest, quality_source_sha, delta_source_sha = (
        _plan_implementation_contract(plan_record)
    )
    sources: list[_AuthenticatedSource] = []
    signature: dict[str, object] | None = None
    for pin in pins:
        delta_root = _resolve_registered_path(
            registry_path, pin.delta_root, "quality-resolution delta root"
        )
        delta = _load_delta_records(
            delta_root,
            pin.delta_manifest_sha256,
            plan=plan,
            expected_plan_sha256=expected_plan_sha256,
        )
        if (
            delta.manifest["stage_index"] != pin.stage_index
            or delta.manifest["shard_index"] != pin.shard_index
        ):
            raise QualityResolutionError("delta identity differs from its shard-pin entry")
        verification_path = _resolve_registered_path(
            registry_path,
            pin.verification_path,
            "quality-resolution verification receipt",
        )
        verification, verification_sha = _load_verification(
            verification_path,
            pin.verification_sha256,
            delta=delta,
            plan_record=plan_record,
            plan_sha256=plan_sha256,
        )
        current = _source_signature(delta.manifest, verification)
        execution = current["execution_identity"]
        if not isinstance(execution, Mapping):  # pragma: no cover - checked by helper
            raise RuntimeError("resolution execution identity lost its object shape")
        selector = plan_record.get("selector")
        context = plan_record.get("context")
        if not isinstance(selector, Mapping) or not isinstance(context, Mapping):
            raise QualityResolutionError("resolution plan selector/context identity is malformed")
        if (
            current["context_digest"] != context.get("digest")
            or execution.get("selector_digest") != selector.get("selector_digest")
            or execution.get("device") != selector.get("device")
            or execution.get("quality_module_sha256") != quality_source_sha
            or execution.get("delta_module_sha256") != delta_source_sha
            or execution.get("quality_implementation_contract_digest") != contract_digest
            or current["verifier_identity"].get("quality_module_sha256")
            != quality_source_sha
            or current["verifier_identity"].get(
                "quality_implementation_contract_digest"
            )
            != contract_digest
            or current["verifier_identity"].get("verification_module_sha256")
            != delta_source_sha
            or content_digest(execution) != current["execution_identity_digest"]
        ):
            raise QualityResolutionError(
                "resolution source runtime/implementation differs from the sealed plan"
            )
        for field in (
            "ground_partition_receipt_record_digest",
            "ground_partition_receipt_sha256",
            "quality_authority_record_digest",
            "target_access_record_digest",
            "verifier_identity_digest",
            "verifier_identity_sha256",
        ):
            _digest(current[field], f"resolution source {field}")
        if signature is None:
            signature = current
        elif current != signature:
            raise QualityResolutionError(
                "resolution shards mix quality authority or implementation identities"
            )
        sources.append(
            _AuthenticatedSource(
                pin=pin,
                delta=delta,
                verification=_freeze(verification),  # type: ignore[arg-type]
                verification_sha256=verification_sha,
            )
        )
    if signature is None:  # pragma: no cover - complete pin census is nonempty
        raise QualityResolutionError("resolution study has no authenticated source shard")
    return tuple(sources), signature


def _expected_stage_keys(
    plan_record: Mapping[str, object], stage_index: int
) -> list[tuple[str, int, int]]:
    sample = plan_record["sample"]
    production = plan_record["production_plan"]
    stage = plan_record["stages"][stage_index]
    if not all(isinstance(item, Mapping) for item in (sample, production, stage)):
        raise QualityResolutionError("resolution plan schedule sections are malformed")
    selected_rows = sample.get("selected_row_ids")
    rows = production.get("rows")
    delta = stage.get("continuation_range")
    if not isinstance(selected_rows, list) or not isinstance(rows, list) or not isinstance(delta, list):
        raise QualityResolutionError("resolution plan row/stage census is malformed")
    row_by_id = {
        row.get("row_id"): row for row in rows if isinstance(row, Mapping)
    }
    lower, upper = delta
    keys: list[tuple[str, int, int]] = []
    for row_id in sorted(selected_rows):
        row = row_by_id.get(row_id)
        if not isinstance(row, Mapping):
            raise QualityResolutionError("resolution sample refers to an unknown planned row")
        actions = row.get("actions")
        if not isinstance(actions, list):
            raise QualityResolutionError("planned resolution action census is malformed")
        for action in actions:
            if not isinstance(action, Mapping):
                raise QualityResolutionError("planned resolution action is malformed")
            for continuation_index in range(int(lower), int(upper)):
                keys.append((str(row_id), int(action["action_index"]), continuation_index))
    return keys


def _validate_exact_stage_coverage(
    plan_record: Mapping[str, object],
    sources: Sequence[_AuthenticatedSource],
    completed_stage_index: int,
) -> dict[int, tuple[Mapping[str, object], ...]]:
    by_stage: dict[int, list[Mapping[str, object]]] = {
        stage: [] for stage in range(completed_stage_index + 1)
    }
    for source in sources:
        by_stage[source.pin.stage_index].extend(source.delta.records)
    result: dict[int, tuple[Mapping[str, object], ...]] = {}
    for stage_index in range(completed_stage_index + 1):
        records = sorted(
            by_stage[stage_index],
            key=lambda row: (
                str(row["row_id"]),
                int(row["action_index"]),
                int(row["continuation_index"]),
            ),
        )
        observed = [
            (str(row["row_id"]), int(row["action_index"]), int(row["continuation_index"]))
            for row in records
        ]
        expected = _expected_stage_keys(plan_record, stage_index)
        if observed != expected:
            raise QualityResolutionError(
                f"resolution stage {stage_index} does not cover its exact planned delta"
            )
        result[stage_index] = tuple(records)
    return result


def _resolution_result(
    plan_record: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    continuations: int,
) -> tuple[int, int, int]:
    sample = plan_record["sample"]
    production = plan_record["production_plan"]
    if not isinstance(sample, Mapping) or not isinstance(production, Mapping):
        raise QualityResolutionError("resolution plan sample/production section is malformed")
    selected_lineages = sample.get("selected_lineages")
    selected_rows = sample.get("selected_row_ids")
    rows = production.get("rows")
    if not isinstance(selected_lineages, list) or not isinstance(selected_rows, list) or not isinstance(rows, list):
        raise QualityResolutionError("resolution plan sample census is malformed")
    row_by_id = {
        row.get("row_id"): row for row in rows if isinstance(row, Mapping)
    }
    rewards: dict[tuple[str, int], list[float]] = {}
    for record in records:
        key = (str(record["row_id"]), int(record["action_index"]))
        rewards.setdefault(key, []).append(float(record["reward"]))
    resolved_rows = 0
    resolved_lineages: set[str] = set()
    for row_id in selected_rows:
        row = row_by_id.get(row_id)
        if not isinstance(row, Mapping):
            raise QualityResolutionError("resolution sample row is absent from production plan")
        actions = row.get("actions")
        if not isinstance(actions, list) or not actions:
            raise QualityResolutionError("resolution row has no evaluated-action census")
        arm_count = len(actions)
        intervals: list[tuple[int, float, float]] = []
        for action in actions:
            if not isinstance(action, Mapping):
                raise QualityResolutionError("resolution planned action is malformed")
            action_index = int(action["action_index"])
            values = rewards.get((str(row_id), action_index), [])
            if len(values) != continuations:
                raise QualityResolutionError(
                    "resolution action does not cover the exact cumulative [0,C) prefix"
                )
            mean = math.fsum(values) / continuations
            radius = math.sqrt(math.log(2.0 * arm_count / 0.05) / (2.0 * continuations))
            intervals.append(
                (action_index, max(0.0, mean - radius), min(1.0, mean + radius))
            )
        best_lower = max(lower for _index, lower, _upper in intervals)
        plausible = [index for index, _lower, upper in intervals if upper >= best_lower]
        if len(plausible) < arm_count:
            resolved_rows += 1
            resolved_lineages.add(str(row["base_lineage"]))
    resolved_count = len(resolved_lineages)
    return resolved_rows, resolved_count, len(selected_lineages) - resolved_count


def _production_protocol(
    plan_record: Mapping[str, object],
    *,
    selected_continuations: int | None,
    implementation_contract_digest: str,
) -> dict[str, object]:
    registered = plan_record["config"]["registered"]
    selector = plan_record["selector"]
    if not isinstance(registered, Mapping) or not isinstance(selector, Mapping):
        raise QualityResolutionError("resolution plan protocol sections are malformed")
    return {
        "continuations": selected_continuations,
        "evaluated_actions": registered["evaluated_actions"],
        "partition": registered["partition"],
        "quality_implementation_contract_digest": implementation_contract_digest,
        "requested_lineages": 0,
        "reward_reads": registered["reward_reads"],
        "seed": registered["quality_seed"],
        "selector_device": selector["device"],
        "states_per_lineage_cap": registered["states_per_lineage_cap"],
        "tasks_per_lineage_cap": registered["tasks_per_lineage_cap"],
    }


def _source_census(sources: Sequence[_AuthenticatedSource]) -> list[dict[str, object]]:
    census: list[dict[str, object]] = []
    for source in sources:
        manifest = source.delta.manifest
        verification = source.verification
        census.append(
            {
                "delta_manifest_record_digest": manifest["record_digest"],
                "delta_manifest_sha256": source.delta.manifest_sha256,
                "record_count": manifest["record_count"],
                "record_set_digest": manifest["record_set_digest"],
                "records_sha256": manifest["records_sha256"],
                "shard_index": source.pin.shard_index,
                "stage_index": source.pin.stage_index,
                "verification_record_digest": verification["record_digest"],
                "verification_sha256": source.verification_sha256,
            }
        )
    return census


def merge_quality_resolution(
    plan: QualityResolutionPlan,
    *,
    expected_plan_sha256: str,
    pin_registry_path: str | Path,
    expected_pin_registry_sha256: str,
    output_path: str | Path,
) -> str:
    """Authenticate and merge one complete progressive resolution-study stage."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    registry_path = Path(pin_registry_path)
    registry, pins, registry_sha = _load_pin_registry(
        registry_path,
        expected_pin_registry_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
    )
    completed_stage = int(registry["completed_stage_index"])
    sources, signature = _authenticate_sources(
        registry_path,
        pins,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
    )
    stage_records = _validate_exact_stage_coverage(
        plan_record, sources, completed_stage
    )
    results: list[dict[str, object]] = []
    resolved_by_count: dict[int, int] = {}
    cumulative: list[Mapping[str, object]] = []
    for stage_index in range(completed_stage + 1):
        cumulative.extend(stage_records[stage_index])
        continuations = int(plan_record["stages"][stage_index]["continuation_range"][1])
        resolved_rows, resolved_lineages, unresolved_lineages = _resolution_result(
            plan_record, cumulative, continuations
        )
        resolved_by_count[continuations] = resolved_lineages
        results.append(
            {
                "continuations": continuations,
                "lower_population_bound": 0,
                "passes": False,
                "sampled_resolved_lineages": resolved_lineages,
                "sampled_resolved_rows": resolved_rows,
                "sampled_unresolved_lineages": unresolved_lineages,
            }
        )
    population_size = int(plan_record["production_plan"]["lineage_count"])
    sample_size = int(plan_record["sample"]["sample_size"])
    minimum = int(plan_record["config"]["registered"]["minimum_resolved_lineages"])
    selection = select_continuation_count(
        population_size=population_size,
        sample_size=sample_size,
        resolved_lineages_by_count=resolved_by_count,
        minimum_resolved_lineages=minimum,
    )
    for result, decision in zip(results, selection.results, strict=True):
        result["lower_population_bound"] = decision.lower_population_bound
        result["passes"] = decision.passes

    execution = signature["execution_identity"]
    if not isinstance(execution, Mapping):  # pragma: no cover - source validation checks this
        raise RuntimeError("resolution execution identity lost its object shape")
    contract_digest = _digest(
        execution.get("quality_implementation_contract_digest"),
        "quality implementation-contract digest",
    )
    quality_authority = {
        "ground_partition_receipt_record_digest": signature[
            "ground_partition_receipt_record_digest"
        ],
        "ground_partition_receipt_sha256": signature[
            "ground_partition_receipt_sha256"
        ],
        "quality_authority_record_digest": signature[
            "quality_authority_record_digest"
        ],
        "target_access_record_digest": signature["target_access_record_digest"],
    }
    payload: dict[str, object] = {
        "advance": selection.advance,
        "candidate_ladder": list(REGISTERED_CANDIDATE_CONTINUATIONS),
        "completed_continuation_range": registry["completed_continuation_range"],
        "confidence": {
            "comparisons": BONFERRONI_COMPARISONS,
            "familywise_alpha": {
                "denominator": FAMILYWISE_ALPHA_DENOMINATOR,
                "numerator": FAMILYWISE_ALPHA_NUMERATOR,
            },
            "method": "exact-hypergeometric-lower-bound-bonferroni-v1",
            "per_comparison_alpha": {
                "denominator": BONFERRONI_ALPHA_DENOMINATOR,
                "numerator": BONFERRONI_ALPHA_NUMERATOR,
            },
        },
        "config": plan_record["config"],
        "context": plan_record["context"],
        "corpus": plan_record["prepared_corpus"],
        "ground_root": plan_record["ground_root"],
        "implementation": {
            "plan": plan_record["implementation"],
            "resolution_execution": dict(execution),
            "resolution_execution_digest": signature["execution_identity_digest"],
            "verification": signature["verifier_identity"],
            "verification_digest": signature["verifier_identity_digest"],
        },
        "pin_registry": {
            "raw_sha256": registry_sha,
            "record_digest": registry["record_digest"],
        },
        "plan": {
            "raw_sha256": plan_sha256,
            "record_digest": plan_record["record_digest"],
        },
        "population": {
            "lineages": population_size,
            "production_plan_digest": plan_record["production_plan_digest"],
            "rows": plan_record["production_plan"]["row_count"],
            "tasks": plan_record["production_plan"]["task_count"],
        },
        "production_protocol": _production_protocol(
            plan_record,
            selected_continuations=selection.selected_continuations,
            implementation_contract_digest=contract_digest,
        ),
        "publisher": plan_record["publisher"],
        "quality_authority": quality_authority,
        "results": results,
        "sample": {
            "record_digest": plan_record["sample"]["record_digest"],
            "sample_size": sample_size,
            "selected_lineages": plan_record["sample"]["selected_lineages"],
            "selected_row_ids": plan_record["sample"]["selected_row_ids"],
            "selected_task_ids": plan_record["sample"]["selected_task_ids"],
        },
        "schema": QUALITY_RESOLUTION_STUDY_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_STUDY_VERSION,
        "selected_continuations": selection.selected_continuations,
        "selector": plan_record["selector"],
        "source_shards": _source_census(sources),
        "stop_reason": selection.stop_reason,
        "study_id": plan_record["study_id"],
        "terminal": selection.terminal,
    }
    receipt = {**payload, "record_digest": content_digest(payload)}
    return _publish_file(Path(output_path), receipt)


def _validate_source_census(
    receipt: Mapping[str, object], plan_record: Mapping[str, object]
) -> None:
    sources = receipt.get("source_shards")
    results = receipt.get("results")
    shards = plan_record.get("shards")
    if not isinstance(sources, list) or not isinstance(results, list) or not isinstance(shards, list):
        raise QualityResolutionError("resolution receipt source census is malformed")
    expected_keys = [
        (stage, shard)
        for stage in range(len(results))
        for shard in range(len(shards))
    ]
    observed_keys: list[tuple[int, int]] = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
            raise QualityResolutionError("resolution receipt source-shard schema differs")
        stage = _nonnegative(source["stage_index"], "source stage index")
        shard = _nonnegative(source["shard_index"], "source shard index")
        observed_keys.append((stage, shard))
        for field in (
            "delta_manifest_record_digest",
            "delta_manifest_sha256",
            "record_set_digest",
            "records_sha256",
            "verification_record_digest",
            "verification_sha256",
        ):
            _digest(source[field], f"source-shard {field}")
        _nonnegative(source["record_count"], "source record count")
    if observed_keys != expected_keys:
        raise QualityResolutionError("resolution receipt lacks a complete canonical source census")


def _validate_study_receipt(
    receipt: Mapping[str, object],
    *,
    plan_record: Mapping[str, object],
    plan_sha256: str,
) -> None:
    if set(receipt) != _STUDY_FIELDS:
        raise QualityResolutionError("quality-resolution study receipt schema fields differ")
    _verify_record(receipt, "quality-resolution study receipt")
    if (
        receipt.get("schema") != QUALITY_RESOLUTION_STUDY_SCHEMA
        or receipt.get("schema_version") != QUALITY_RESOLUTION_STUDY_VERSION
        or receipt.get("study_id") != plan_record["study_id"]
        or receipt.get("candidate_ladder") != list(REGISTERED_CANDIDATE_CONTINUATIONS)
        or receipt.get("config") != plan_record["config"]
        or receipt.get("corpus") != plan_record["prepared_corpus"]
        or receipt.get("publisher") != plan_record["publisher"]
        or receipt.get("ground_root") != plan_record["ground_root"]
        or receipt.get("selector") != plan_record["selector"]
        or receipt.get("context") != plan_record["context"]
    ):
        raise QualityResolutionError("quality-resolution study receipt differs from its plan")
    plan_identity = receipt.get("plan")
    if not isinstance(plan_identity, Mapping) or plan_identity != {
        "raw_sha256": plan_sha256,
        "record_digest": plan_record["record_digest"],
    }:
        raise QualityResolutionError("quality-resolution receipt carries another plan identity")
    population = receipt.get("population")
    expected_population = {
        "lineages": plan_record["production_plan"]["lineage_count"],
        "production_plan_digest": plan_record["production_plan_digest"],
        "rows": plan_record["production_plan"]["row_count"],
        "tasks": plan_record["production_plan"]["task_count"],
    }
    if population != expected_population:
        raise QualityResolutionError("quality-resolution receipt population differs from plan")
    sample = receipt.get("sample")
    expected_sample = {
        "record_digest": plan_record["sample"]["record_digest"],
        "sample_size": plan_record["sample"]["sample_size"],
        "selected_lineages": plan_record["sample"]["selected_lineages"],
        "selected_row_ids": plan_record["sample"]["selected_row_ids"],
        "selected_task_ids": plan_record["sample"]["selected_task_ids"],
    }
    if sample != expected_sample:
        raise QualityResolutionError("quality-resolution receipt sample differs from plan")
    expected_confidence = {
        "comparisons": BONFERRONI_COMPARISONS,
        "familywise_alpha": {
            "denominator": FAMILYWISE_ALPHA_DENOMINATOR,
            "numerator": FAMILYWISE_ALPHA_NUMERATOR,
        },
        "method": "exact-hypergeometric-lower-bound-bonferroni-v1",
        "per_comparison_alpha": {
            "denominator": BONFERRONI_ALPHA_DENOMINATOR,
            "numerator": BONFERRONI_ALPHA_NUMERATOR,
        },
    }
    if receipt.get("confidence") != expected_confidence:
        raise QualityResolutionError("quality-resolution confidence rule differs")
    results = receipt.get("results")
    if not isinstance(results, list) or not results:
        raise QualityResolutionError("quality-resolution receipt has no completed candidate")
    resolved_by_count: dict[int, int] = {}
    for index, result in enumerate(results):
        if not isinstance(result, Mapping) or set(result) != _RESULT_FIELDS:
            raise QualityResolutionError("quality-resolution result schema differs")
        continuations = result["continuations"]
        if continuations != REGISTERED_CANDIDATE_CONTINUATIONS[index]:
            raise QualityResolutionError("quality-resolution results are not a ladder prefix")
        resolved = _nonnegative(
            result["sampled_resolved_lineages"], "sampled resolved lineages"
        )
        unresolved = _nonnegative(
            result["sampled_unresolved_lineages"], "sampled unresolved lineages"
        )
        resolved_rows = _nonnegative(result["sampled_resolved_rows"], "sampled resolved rows")
        if (
            resolved + unresolved != int(plan_record["sample"]["sample_size"])
            or resolved_rows < resolved
            or type(result["passes"]) is not bool
        ):
            raise QualityResolutionError("quality-resolution result denominators are inconsistent")
        resolved_by_count[int(continuations)] = resolved
    selection = select_continuation_count(
        population_size=int(plan_record["production_plan"]["lineage_count"]),
        sample_size=int(plan_record["sample"]["sample_size"]),
        resolved_lineages_by_count=resolved_by_count,
        minimum_resolved_lineages=int(
            plan_record["config"]["registered"]["minimum_resolved_lineages"]
        ),
    )
    for result, decision in zip(results, selection.results, strict=True):
        if (
            result["lower_population_bound"] != decision.lower_population_bound
            or result["passes"] is not decision.passes
        ):
            raise QualityResolutionError("quality-resolution decision cannot be reproduced")
    if (
        receipt.get("selected_continuations") != selection.selected_continuations
        or receipt.get("advance") is not selection.advance
        or receipt.get("terminal") is not selection.terminal
        or receipt.get("stop_reason") != selection.stop_reason
        or receipt.get("completed_continuation_range")
        != [0, REGISTERED_CANDIDATE_CONTINUATIONS[len(results) - 1]]
    ):
        raise QualityResolutionError("quality-resolution progressive decision is inconsistent")
    implementation = receipt.get("implementation")
    authority = receipt.get("quality_authority")
    pin_registry = receipt.get("pin_registry")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "plan",
        "resolution_execution",
        "resolution_execution_digest",
        "verification",
        "verification_digest",
    }:
        raise QualityResolutionError("resolution receipt implementation schema differs")
    if implementation.get("plan") != plan_record["implementation"]:
        raise QualityResolutionError("resolution receipt implementation differs from plan")
    execution = implementation.get("resolution_execution")
    verifier = implementation.get("verification")
    verifier_digest = (
        _verify_record(verifier, "resolution verification identity")
        if isinstance(verifier, Mapping)
        else None
    )
    if (
        not isinstance(execution, Mapping)
        or content_digest(execution)
        != _digest(
            implementation.get("resolution_execution_digest"),
            "resolution execution identity digest",
        )
        or verifier_digest
        != _digest(
            implementation.get("verification_digest"),
            "resolution verification identity digest",
        )
    ):
        raise QualityResolutionError("resolution receipt implementation identity is invalid")
    contract_digest, quality_source_sha, delta_source_sha = (
        _plan_implementation_contract(plan_record)
    )
    if (
        execution.get("quality_implementation_contract_digest") != contract_digest
        or execution.get("quality_module_sha256") != quality_source_sha
        or execution.get("delta_module_sha256") != delta_source_sha
        or verifier.get("quality_implementation_contract_digest") != contract_digest
        or verifier.get("quality_module_sha256") != quality_source_sha
        or verifier.get("verification_module_sha256") != delta_source_sha
    ):
        raise QualityResolutionError("resolution execution was not sealed by the plan")
    if not isinstance(authority, Mapping) or set(authority) != {
        "ground_partition_receipt_record_digest",
        "ground_partition_receipt_sha256",
        "quality_authority_record_digest",
        "target_access_record_digest",
    }:
        raise QualityResolutionError("resolution receipt quality-authority schema differs")
    for field, value in authority.items():
        _digest(value, f"resolution quality-authority {field}")
    if not isinstance(pin_registry, Mapping) or set(pin_registry) != {
        "raw_sha256",
        "record_digest",
    }:
        raise QualityResolutionError("resolution receipt pin-registry identity differs")
    _digest(pin_registry["raw_sha256"], "resolution pin-registry SHA-256")
    _digest(pin_registry["record_digest"], "resolution pin-registry record digest")
    expected_protocol = _production_protocol(
        plan_record,
        selected_continuations=selection.selected_continuations,
        implementation_contract_digest=contract_digest,
    )
    if receipt.get("production_protocol") != expected_protocol:
        raise QualityResolutionError("resolution receipt production protocol differs")
    _validate_source_census(receipt, plan_record)


def load_quality_resolution_study_receipt(
    path: str | Path,
    *,
    expected_receipt_sha256: str,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
) -> QualityResolutionStudyReceipt:
    """Load a canonical study receipt only through its external raw-file pin."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    receipt, _raw, observed_sha = _read_pinned_json(
        path, expected_receipt_sha256, "quality-resolution study receipt"
    )
    _validate_study_receipt(receipt, plan_record=plan_record, plan_sha256=plan_sha256)
    return QualityResolutionStudyReceipt(receipt, observed_sha)


__all__ = [
    "QUALITY_RESOLUTION_PIN_REGISTRY_SCHEMA",
    "QUALITY_RESOLUTION_PIN_REGISTRY_VERSION",
    "QUALITY_RESOLUTION_STUDY_SCHEMA",
    "QUALITY_RESOLUTION_STUDY_VERSION",
    "QualityResolutionShardPin",
    "QualityResolutionStudyReceipt",
    "load_quality_resolution_study_receipt",
    "merge_quality_resolution",
    "publish_quality_resolution_shard_pin_registry",
]

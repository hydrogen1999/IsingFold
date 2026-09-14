"""Target-free publication binding and quality-input readiness receipts.

This module does not generate labels and does not open prepared evaluator targets.  It binds a
terminal continuation-resolution study to an unchanged quality-v8 manifest, then binds that
publication to the strict quality-preflight v3 receipt.  The final receipt is scoped to
the quality-supervision input; release gates and the experiment grid remain separate consumers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from isingfold.rl.contracts import stable_digest
from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_resolution_delta import (
    _plan_identity,
    _publish_file,
    _read_pinned_json,
    _thaw,
    _verify_record,
)
from isingfold.rl.data.quality_resolution_merge import (
    QualityResolutionStudyReceipt,
    load_quality_resolution_study_receipt,
)
from isingfold.rl.data.quality_resolution_plan import QualityResolutionPlan


QUALITY_RESOLUTION_BINDING_SCHEMA = "isingfold.quality-resolution-publication-binding"
QUALITY_RESOLUTION_BINDING_VERSION = 2
QUALITY_TRAINING_INPUT_READINESS_SCHEMA = "isingfold.quality-training-input-readiness"
QUALITY_TRAINING_INPUT_READINESS_VERSION = 2

_HEX = frozenset("0123456789abcdef")
_QUALITY_SCHEMA = "isingfold.quality-label-corpus"
_QUALITY_VERSION = 8
_QUALITY_MERGED_SCHEMA = "isingfold.quality-label-merged-corpus"
_QUALITY_MERGED_VERSION = 6
_QUALITY_PREFLIGHT_SCHEMA = "isingfold.quality-resolution-preflight"
_QUALITY_PREFLIGHT_VERSION = 3
_BASE_MANIFEST_FIELDS = {
    "context",
    "context_digest",
    "ground_partition_receipt",
    "independent_denominators",
    "initializer_bank",
    "label_protocol",
    "lineages",
    "normalizer_digest",
    "partition",
    "quality_authority",
    "quality_resolution_plan",
    "record_count",
    "record_digest",
    "records_sha256",
    "sampling_receipt",
    "schema",
    "schema_version",
    "selector_digest",
    "source_corpus_manifest_sha256",
    "target_access",
    "task_ids",
}
_LABEL_PROTOCOL_FIELDS = {
    "continuations",
    "evaluated_actions",
    "implementation_contract",
    "requested_lineages",
    "reward_reads",
    "seed",
    "states_per_lineage_cap",
    "tasks_per_lineage_cap",
}
_SAMPLING_FIELDS = {
    "available_lineages",
    "requested_lineages",
    "selected_lineages",
    "selected_task_ids",
    "states_per_lineage_cap",
    "tasks_per_lineage_cap",
}
_DENOMINATOR_FIELDS = {
    "continuation_trajectories",
    "fully_unresolved_rows",
    "lineages_with_records",
    "records_total",
    "resolved_lineages",
    "resolved_rows",
    "selected_lineages",
    "selected_tasks",
    "tasks_with_records",
    "valid_continuations",
}
_BINDING_FIELDS = {
    "checks",
    "pass",
    "plan",
    "production_protocol",
    "quality_manifest",
    "record_digest",
    "resolution_receipt",
    "schema",
    "schema_version",
    "selected_continuations",
}
_PREFLIGHT_FIELDS = {
    "advance",
    "context",
    "context_digest",
    "full_replay",
    "implementation_contract",
    "initializer_bank",
    "label_protocol_digest",
    "minimum_resolved_lineages",
    "minimum_resolved_rows",
    "normalizer_digest",
    "quality_authority",
    "quality_resolution_plan",
    "record_digest",
    "resolution",
    "schema",
    "schema_version",
    "selector_digest",
    "source_corpus_manifest_sha256",
    "source_quality_manifest_record_digest",
    "source_quality_manifest_sha256",
    "source_quality_records_sha256",
}
_FULL_REPLAY_FIELDS = {
    "continuation_replay_matches",
    "continuation_replays_executed",
    "continuation_trajectories",
    "mode",
}
_RESOLUTION_SUMMARY_FIELDS = {
    "continuation_replay_matches",
    "continuation_replay_mode",
    "continuation_replays_executed",
    "continuation_trajectories_authenticated",
    "fully_unresolved_skipped",
    "label_lineages",
    "minimum_resolved_lineages",
    "minimum_resolved_rows",
    "passes_resolution",
    "records_total",
    "records_used",
    "resolved_lineages",
    "resolved_tasks",
    "unresolved_lineages",
}
_READINESS_FIELDS = {
    "binding",
    "checks",
    "continuation_trajectories",
    "pass",
    "plan",
    "quality_manifest",
    "quality_preflight",
    "record_digest",
    "resolution_receipt",
    "resolved_lineages",
    "resolved_rows",
    "schema",
    "schema_version",
    "scope",
    "selected_continuations",
}


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise QualityResolutionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonnegative(value: object, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        qualifier = "positive" if positive else "nonnegative"
        raise QualityResolutionError(f"{label} must be a {qualifier} integer")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class QualityResolutionBinding:
    """Read-only header-level binding from the study to the quality-v8 manifest."""

    record: Mapping[str, object]
    raw_sha256: str

    def __post_init__(self) -> None:
        _digest(self.raw_sha256, "quality-resolution binding SHA-256")
        object.__setattr__(self, "record", _freeze(self.record))

    def as_dict(self) -> dict[str, Any]:
        value = _thaw(self.record)
        if not isinstance(value, dict):  # pragma: no cover
            raise RuntimeError("quality-resolution binding lost its object shape")
        return value


@dataclass(frozen=True, slots=True)
class QualityTrainingInputReadiness:
    """Read-only proof that quality supervision may be loaded by model training."""

    record: Mapping[str, object]
    raw_sha256: str

    def __post_init__(self) -> None:
        _digest(self.raw_sha256, "quality-training input-readiness SHA-256")
        object.__setattr__(self, "record", _freeze(self.record))

    def as_dict(self) -> dict[str, Any]:
        value = _thaw(self.record)
        if not isinstance(value, dict):  # pragma: no cover
            raise RuntimeError("quality-training readiness lost its object shape")
        return value


def _planned_population(plan_record: Mapping[str, object]) -> tuple[list[str], list[str]]:
    production = plan_record.get("production_plan")
    if not isinstance(production, Mapping):
        raise QualityResolutionError("resolution plan production section is malformed")
    lineages = production.get("lineages")
    if not isinstance(lineages, list):
        raise QualityResolutionError("resolution plan lineage census is malformed")
    lineage_ids: list[str] = []
    task_ids: list[str] = []
    for lineage in lineages:
        if not isinstance(lineage, Mapping):
            raise QualityResolutionError("resolution plan lineage entry is malformed")
        base = lineage.get("base_lineage")
        tasks = lineage.get("task_ids")
        if type(base) is not str or not base or not isinstance(tasks, list):
            raise QualityResolutionError("resolution plan lineage identity is malformed")
        lineage_ids.append(base)
        if any(type(task) is not str or not task for task in tasks):
            raise QualityResolutionError("resolution plan task identity is malformed")
        task_ids.extend(tasks)
    canonical_lineages = sorted(lineage_ids)
    canonical_tasks = sorted(task_ids)
    if len(canonical_lineages) != len(set(canonical_lineages)) or len(canonical_tasks) != len(
        set(canonical_tasks)
    ):
        raise QualityResolutionError("resolution plan population repeats an identity")
    return canonical_lineages, canonical_tasks


def _load_quality_manifest(
    path: str | Path,
    expected_sha256: str,
    *,
    plan_record: Mapping[str, object],
    plan_sha256: str,
    study: QualityResolutionStudyReceipt,
) -> tuple[dict[str, Any], str]:
    manifest, _raw, observed_sha = _read_pinned_json(
        path, expected_sha256, "quality-v8 manifest"
    )
    schema = manifest.get("schema")
    if schema == _QUALITY_SCHEMA:
        expected_fields = _BASE_MANIFEST_FIELDS
        expected_version = _QUALITY_VERSION
    elif schema == _QUALITY_MERGED_SCHEMA:
        expected_fields = _BASE_MANIFEST_FIELDS | {"merge_receipt"}
        expected_version = _QUALITY_MERGED_VERSION
    else:
        raise QualityResolutionError(
            "resolution binding requires a monolithic or canonically merged quality-v8 manifest"
        )
    if set(manifest) != expected_fields or manifest.get("schema_version") != expected_version:
        raise QualityResolutionError("quality-v8 manifest schema fields differ")
    _verify_record(manifest, "quality-v8 manifest")
    for field in ("quality_authority", "target_access", "ground_partition_receipt"):
        value = manifest.get(field)
        if not isinstance(value, Mapping):
            raise QualityResolutionError(f"quality-v8 manifest {field} must be an object")
        _verify_record(value, f"quality-v8 manifest {field}")
    production_plan = plan_record.get("production_plan")
    if not isinstance(production_plan, Mapping):
        raise QualityResolutionError("resolution plan production section is malformed")
    expected_plan_reference = {
        "raw_sha256": plan_sha256,
        "record_digest": plan_record["record_digest"],
    }
    if (
        not isinstance(manifest.get("initializer_bank"), Mapping)
        or manifest.get("initializer_bank") != production_plan.get("initializer_bank")
        or manifest.get("quality_resolution_plan") != expected_plan_reference
    ):
        raise QualityResolutionError(
            "quality-v8 initializer bank or resolution-plan binding differs"
        )

    study_record = study.as_dict()
    if study_record.get("terminal") is not True or study_record.get("advance") is not True:
        raise QualityResolutionError(
            "quality publication requires the first passing terminal resolution receipt"
        )
    selected = study_record.get("selected_continuations")
    _nonnegative(selected, "selected continuation count", positive=True)
    protocol = study_record.get("production_protocol")
    authority = study_record.get("quality_authority")
    if not isinstance(protocol, Mapping) or not isinstance(authority, Mapping):
        raise QualityResolutionError("terminal resolution receipt has no production authority")
    label_protocol = manifest.get("label_protocol")
    if not isinstance(label_protocol, Mapping) or set(label_protocol) != _LABEL_PROTOCOL_FIELDS:
        raise QualityResolutionError("quality-v8 label protocol schema differs")
    expected_protocol = {
        "continuations": protocol["continuations"],
        "evaluated_actions": protocol["evaluated_actions"],
        "requested_lineages": protocol["requested_lineages"],
        "reward_reads": protocol["reward_reads"],
        "seed": protocol["seed"],
        "states_per_lineage_cap": protocol["states_per_lineage_cap"],
        "tasks_per_lineage_cap": protocol["tasks_per_lineage_cap"],
    }
    observed_protocol = {
        key: label_protocol[key] for key in expected_protocol
    }
    implementation_contract = label_protocol.get("implementation_contract")
    if (
        observed_protocol != expected_protocol
        or content_digest(implementation_contract)
        != protocol.get("quality_implementation_contract_digest")
    ):
        raise QualityResolutionError(
            "quality-v8 production protocol differs from terminal resolution authorization"
        )
    expected_corpus = plan_record["prepared_corpus"]["manifest_sha256"]
    expected_selector = plan_record["selector"]
    expected_context = plan_record["context"]
    if (
        manifest.get("partition") != "train"
        or manifest.get("source_corpus_manifest_sha256") != expected_corpus
        or manifest.get("selector_digest") != expected_selector["selector_digest"]
        or manifest.get("normalizer_digest") != expected_selector["normalizer_digest"]
        or manifest.get("context") != expected_context["snapshot"]
        or manifest.get("context_digest") != expected_context["digest"]
    ):
        raise QualityResolutionError(
            "quality-v8 corpus, selector, or context differs from the resolution plan"
        )
    if (
        manifest["quality_authority"]["record_digest"]
        != authority.get("quality_authority_record_digest")
        or manifest["target_access"]["record_digest"]
        != authority.get("target_access_record_digest")
        or manifest["ground_partition_receipt"]["record_digest"]
        != authority.get("ground_partition_receipt_record_digest")
    ):
        raise QualityResolutionError(
            "quality-v8 evaluator authority differs from the resolution study"
        )

    planned_lineages, planned_tasks = _planned_population(plan_record)
    sampling = manifest.get("sampling_receipt")
    expected_sampling = {
        "available_lineages": len(planned_lineages),
        "requested_lineages": 0,
        "selected_lineages": planned_lineages,
        "selected_task_ids": planned_tasks,
        "states_per_lineage_cap": protocol["states_per_lineage_cap"],
        "tasks_per_lineage_cap": protocol["tasks_per_lineage_cap"],
    }
    if (
        not isinstance(sampling, Mapping)
        or set(sampling) != _SAMPLING_FIELDS
        or dict(sampling) != expected_sampling
    ):
        raise QualityResolutionError(
            "quality-v8 sampling is not the complete all-train production population"
        )
    manifest_lineages = manifest.get("lineages")
    manifest_tasks = manifest.get("task_ids")
    if (
        not isinstance(manifest_lineages, list)
        or any(type(value) is not str or not value for value in manifest_lineages)
        or not isinstance(manifest_tasks, list)
        or any(type(value) is not str or not value for value in manifest_tasks)
    ):
        raise QualityResolutionError("quality-v8 record-bearing population is malformed")
    if (
        manifest_lineages != sorted(set(manifest_lineages))
        or not set(manifest_lineages) <= set(planned_lineages)
        or manifest_tasks != sorted(set(manifest_tasks))
        or not set(manifest_tasks) <= set(planned_tasks)
    ):
        raise QualityResolutionError("quality-v8 record-bearing population is malformed")
    record_count = _nonnegative(manifest.get("record_count"), "quality-v8 record count", positive=True)
    _digest(manifest.get("records_sha256"), "quality-v8 records SHA-256")
    denominators = manifest.get("independent_denominators")
    if not isinstance(denominators, Mapping) or set(denominators) != _DENOMINATOR_FIELDS:
        raise QualityResolutionError("quality-v8 independent denominator schema differs")
    for field in _DENOMINATOR_FIELDS:
        _nonnegative(denominators[field], f"quality-v8 denominator {field}")
    if (
        denominators["selected_lineages"] != len(planned_lineages)
        or denominators["selected_tasks"] != len(planned_tasks)
        or denominators["records_total"] != record_count
        or denominators["resolved_rows"] + denominators["fully_unresolved_rows"]
        != record_count
        or denominators["resolved_lineages"] > denominators["lineages_with_records"]
        or denominators["lineages_with_records"] != len(manifest_lineages)
        or denominators["tasks_with_records"] != len(manifest_tasks)
        or denominators["valid_continuations"]
        > denominators["continuation_trajectories"]
    ):
        raise QualityResolutionError("quality-v8 independent denominators are inconsistent")
    return manifest, observed_sha


def _binding_payload(
    *,
    plan_record: Mapping[str, object],
    plan_sha256: str,
    study: QualityResolutionStudyReceipt,
    quality_manifest: Mapping[str, object],
    quality_manifest_sha256: str,
) -> dict[str, object]:
    study_record = study.as_dict()
    protocol = study_record["production_protocol"]
    checks = {
        "all_train_population": True,
        "authority": True,
        "context_and_selector": True,
        "production_protocol": True,
        "schema": True,
    }
    return {
        "checks": checks,
        "pass": True,
        "plan": {
            "raw_sha256": plan_sha256,
            "record_digest": plan_record["record_digest"],
        },
        "production_protocol": protocol,
        "quality_manifest": {
            "raw_sha256": quality_manifest_sha256,
            "record_digest": quality_manifest["record_digest"],
            "records_sha256": quality_manifest["records_sha256"],
            "schema": quality_manifest["schema"],
            "schema_version": quality_manifest["schema_version"],
        },
        "resolution_receipt": {
            "raw_sha256": study.raw_sha256,
            "record_digest": study_record["record_digest"],
        },
        "schema": QUALITY_RESOLUTION_BINDING_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_BINDING_VERSION,
        "selected_continuations": study_record["selected_continuations"],
    }


def verify_quality_resolution_binding(
    *,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    resolution_receipt_path: str | Path,
    expected_resolution_receipt_sha256: str,
    quality_manifest_path: str | Path,
    expected_quality_manifest_sha256: str,
    output_path: str | Path,
) -> str:
    """Publish an immutable proof that quality-v8 obeys the terminal study."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    study = load_quality_resolution_study_receipt(
        resolution_receipt_path,
        expected_receipt_sha256=expected_resolution_receipt_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
    )
    quality_manifest, quality_manifest_sha = _load_quality_manifest(
        quality_manifest_path,
        expected_quality_manifest_sha256,
        plan_record=plan_record,
        plan_sha256=plan_sha256,
        study=study,
    )
    payload = _binding_payload(
        plan_record=plan_record,
        plan_sha256=plan_sha256,
        study=study,
        quality_manifest=quality_manifest,
        quality_manifest_sha256=quality_manifest_sha,
    )
    binding = {**payload, "record_digest": content_digest(payload)}
    return _publish_file(Path(output_path), binding)


def load_quality_resolution_binding(
    path: str | Path,
    *,
    expected_binding_sha256: str,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    resolution_receipt_path: str | Path,
    expected_resolution_receipt_sha256: str,
    quality_manifest_path: str | Path,
    expected_quality_manifest_sha256: str,
) -> QualityResolutionBinding:
    """Reproduce a publication binding from all externally pinned inputs."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    binding, _raw, observed_sha = _read_pinned_json(
        path, expected_binding_sha256, "quality-resolution publication binding"
    )
    if set(binding) != _BINDING_FIELDS:
        raise QualityResolutionError("quality-resolution publication-binding schema differs")
    _verify_record(binding, "quality-resolution publication binding")
    study = load_quality_resolution_study_receipt(
        resolution_receipt_path,
        expected_receipt_sha256=expected_resolution_receipt_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
    )
    quality_manifest, quality_manifest_sha = _load_quality_manifest(
        quality_manifest_path,
        expected_quality_manifest_sha256,
        plan_record=plan_record,
        plan_sha256=plan_sha256,
        study=study,
    )
    expected_payload = _binding_payload(
        plan_record=plan_record,
        plan_sha256=plan_sha256,
        study=study,
        quality_manifest=quality_manifest,
        quality_manifest_sha256=quality_manifest_sha,
    )
    expected = {**expected_payload, "record_digest": content_digest(expected_payload)}
    if binding != expected:
        raise QualityResolutionError(
            "quality-resolution publication binding cannot be reproduced"
        )
    return QualityResolutionBinding(binding, observed_sha)


def _load_quality_preflight(
    path: str | Path,
    expected_sha256: str,
    *,
    plan_record: Mapping[str, object],
    study: QualityResolutionStudyReceipt,
    quality_manifest: Mapping[str, object],
    quality_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    receipt, _raw, observed_sha = _read_pinned_json(
        path, expected_sha256, "quality-preflight v3 receipt"
    )
    if set(receipt) != _PREFLIGHT_FIELDS:
        raise QualityResolutionError("quality-preflight receipt differs from exact v3 schema")
    _verify_record(receipt, "quality-preflight v3 receipt")
    label_protocol = quality_manifest["label_protocol"]
    if (
        receipt.get("schema") != _QUALITY_PREFLIGHT_SCHEMA
        or receipt.get("schema_version") != _QUALITY_PREFLIGHT_VERSION
        or receipt.get("advance") is not True
        or receipt.get("source_quality_manifest_sha256") != quality_manifest_sha256
        or receipt.get("source_quality_manifest_record_digest")
        != quality_manifest["record_digest"]
        or receipt.get("source_quality_records_sha256")
        != quality_manifest["records_sha256"]
        or receipt.get("source_corpus_manifest_sha256")
        != plan_record["prepared_corpus"]["manifest_sha256"]
        or receipt.get("quality_authority") != quality_manifest["quality_authority"]
        or receipt.get("initializer_bank") != quality_manifest["initializer_bank"]
        or receipt.get("quality_resolution_plan")
        != quality_manifest["quality_resolution_plan"]
        or receipt.get("selector_digest") != quality_manifest["selector_digest"]
        or receipt.get("normalizer_digest") != quality_manifest["normalizer_digest"]
        or receipt.get("context") != quality_manifest["context"]
        or receipt.get("context_digest") != quality_manifest["context_digest"]
        or receipt.get("implementation_contract")
        != label_protocol["implementation_contract"]
        or receipt.get("label_protocol_digest") != stable_digest(label_protocol)
    ):
        raise QualityResolutionError(
            "quality-preflight receipt belongs to another bound quality-v8 publication"
        )
    expected_rows = int(plan_record["grid"]["quality_resolution"]["min_resolved_rows"])
    expected_lineages = int(
        plan_record["grid"]["quality_resolution"]["min_resolved_lineages"]
    )
    if (
        receipt.get("minimum_resolved_rows") != expected_rows
        or receipt.get("minimum_resolved_lineages") != expected_lineages
    ):
        raise QualityResolutionError("quality-preflight thresholds differ from the sealed grid")
    full_replay = receipt.get("full_replay")
    resolution = receipt.get("resolution")
    denominators = quality_manifest["independent_denominators"]
    if (
        not isinstance(full_replay, Mapping)
        or set(full_replay) != _FULL_REPLAY_FIELDS
        or full_replay.get("mode") != "full-exact-continuation-and-evaluator"
        or not isinstance(resolution, Mapping)
        or set(resolution) != _RESOLUTION_SUMMARY_FIELDS
    ):
        raise QualityResolutionError("quality-preflight does not expose exact replay evidence")
    trajectories = denominators["continuation_trajectories"]
    if (
        full_replay.get("continuation_trajectories") != trajectories
        or full_replay.get("continuation_replays_executed") != trajectories
        or full_replay.get("continuation_replay_matches") != trajectories
        or resolution.get("continuation_trajectories_authenticated") != trajectories
        or resolution.get("continuation_replays_executed") != trajectories
        or resolution.get("continuation_replay_matches") != trajectories
        or resolution.get("continuation_replay_mode")
        != "full-exact-continuation-and-evaluator"
    ):
        raise QualityResolutionError(
            "quality-preflight does not prove one complete exact replay"
        )
    unresolved = resolution.get("unresolved_lineages")
    if (
        resolution.get("passes_resolution") is not True
        or resolution.get("minimum_resolved_rows") != expected_rows
        or resolution.get("minimum_resolved_lineages") != expected_lineages
        or resolution.get("records_total") != denominators["records_total"]
        or resolution.get("records_used") != denominators["resolved_rows"]
        or resolution.get("fully_unresolved_skipped")
        != denominators["fully_unresolved_rows"]
        or resolution.get("label_lineages") != denominators["lineages_with_records"]
        or resolution.get("resolved_lineages") != denominators["resolved_lineages"]
        or not isinstance(unresolved, list)
        or len(unresolved)
        != denominators["lineages_with_records"] - denominators["resolved_lineages"]
        or denominators["resolved_rows"] < expected_rows
        or denominators["resolved_lineages"] < expected_lineages
    ):
        raise QualityResolutionError(
            "quality-preflight realized resolution denominators do not pass"
        )
    selected = study.as_dict()["selected_continuations"]
    if label_protocol["continuations"] != selected:
        raise QualityResolutionError("quality-preflight uses another continuation denominator")
    return receipt, observed_sha


def _readiness_payload(
    *,
    plan_record: Mapping[str, object],
    plan_sha256: str,
    study: QualityResolutionStudyReceipt,
    binding: QualityResolutionBinding,
    quality_manifest: Mapping[str, object],
    quality_manifest_sha256: str,
    preflight: Mapping[str, object],
    preflight_sha256: str,
) -> dict[str, object]:
    denominators = quality_manifest["independent_denominators"]
    study_record = study.as_dict()
    binding_record = binding.as_dict()
    return {
        "binding": {
            "raw_sha256": binding.raw_sha256,
            "record_digest": binding_record["record_digest"],
        },
        "checks": {
            "authority_and_protocol": True,
            "complete_exact_replay": True,
            "publication_binding": True,
            "resolution_thresholds": True,
        },
        "continuation_trajectories": denominators["continuation_trajectories"],
        "pass": True,
        "plan": {
            "raw_sha256": plan_sha256,
            "record_digest": plan_record["record_digest"],
        },
        "quality_manifest": {
            "raw_sha256": quality_manifest_sha256,
            "record_digest": quality_manifest["record_digest"],
            "records_sha256": quality_manifest["records_sha256"],
        },
        "quality_preflight": {
            "raw_sha256": preflight_sha256,
            "record_digest": preflight["record_digest"],
        },
        "resolution_receipt": {
            "raw_sha256": study.raw_sha256,
            "record_digest": study_record["record_digest"],
        },
        "resolved_lineages": denominators["resolved_lineages"],
        "resolved_rows": denominators["resolved_rows"],
        "schema": QUALITY_TRAINING_INPUT_READINESS_SCHEMA,
        "schema_version": QUALITY_TRAINING_INPUT_READINESS_VERSION,
        "scope": "quality-supervision-input-only",
        "selected_continuations": study_record["selected_continuations"],
    }


def verify_quality_training_input_readiness(
    *,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    resolution_receipt_path: str | Path,
    expected_resolution_receipt_sha256: str,
    binding_path: str | Path,
    expected_binding_sha256: str,
    quality_manifest_path: str | Path,
    expected_quality_manifest_sha256: str,
    quality_preflight_path: str | Path,
    expected_quality_preflight_sha256: str,
    output_path: str | Path,
) -> str:
    """Publish a fail-closed, externally pinnable quality-input readiness result."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    study = load_quality_resolution_study_receipt(
        resolution_receipt_path,
        expected_receipt_sha256=expected_resolution_receipt_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
    )
    quality_manifest, quality_manifest_sha = _load_quality_manifest(
        quality_manifest_path,
        expected_quality_manifest_sha256,
        plan_record=plan_record,
        plan_sha256=plan_sha256,
        study=study,
    )
    binding = load_quality_resolution_binding(
        binding_path,
        expected_binding_sha256=expected_binding_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
        resolution_receipt_path=resolution_receipt_path,
        expected_resolution_receipt_sha256=expected_resolution_receipt_sha256,
        quality_manifest_path=quality_manifest_path,
        expected_quality_manifest_sha256=expected_quality_manifest_sha256,
    )
    preflight, preflight_sha = _load_quality_preflight(
        quality_preflight_path,
        expected_quality_preflight_sha256,
        plan_record=plan_record,
        study=study,
        quality_manifest=quality_manifest,
        quality_manifest_sha256=quality_manifest_sha,
    )
    payload = _readiness_payload(
        plan_record=plan_record,
        plan_sha256=plan_sha256,
        study=study,
        binding=binding,
        quality_manifest=quality_manifest,
        quality_manifest_sha256=quality_manifest_sha,
        preflight=preflight,
        preflight_sha256=preflight_sha,
    )
    readiness = {**payload, "record_digest": content_digest(payload)}
    return _publish_file(Path(output_path), readiness)


def load_quality_training_input_readiness(
    path: str | Path,
    *,
    expected_readiness_sha256: str,
    binding_path: str | Path,
    expected_binding_sha256: str,
    quality_preflight_path: str | Path,
    expected_quality_preflight_sha256: str,
) -> QualityTrainingInputReadiness:
    """Authenticate a readiness receipt and its two immediate external evidence pins."""

    readiness, _raw, observed_sha = _read_pinned_json(
        path, expected_readiness_sha256, "quality-training input-readiness receipt"
    )
    if set(readiness) != _READINESS_FIELDS:
        raise QualityResolutionError("quality-training input-readiness schema differs")
    _verify_record(readiness, "quality-training input-readiness receipt")
    binding, _binding_raw, binding_sha = _read_pinned_json(
        binding_path, expected_binding_sha256, "quality-resolution publication binding"
    )
    preflight, _preflight_raw, preflight_sha = _read_pinned_json(
        quality_preflight_path,
        expected_quality_preflight_sha256,
        "quality-preflight v3 receipt",
    )
    _verify_record(binding, "quality-resolution publication binding")
    _verify_record(preflight, "quality-preflight v3 receipt")
    if (
        set(binding) != _BINDING_FIELDS
        or binding.get("schema") != QUALITY_RESOLUTION_BINDING_SCHEMA
        or binding.get("schema_version") != QUALITY_RESOLUTION_BINDING_VERSION
        or binding.get("pass") is not True
        or set(preflight) != _PREFLIGHT_FIELDS
        or preflight.get("schema") != _QUALITY_PREFLIGHT_SCHEMA
        or preflight.get("schema_version") != _QUALITY_PREFLIGHT_VERSION
        or preflight.get("advance") is not True
    ):
        raise QualityResolutionError(
            "quality-training input-readiness source evidence does not pass"
        )
    binding_quality = binding.get("quality_manifest")
    binding_plan = binding.get("plan")
    binding_resolution = binding.get("resolution_receipt")
    full_replay = preflight.get("full_replay")
    resolution = preflight.get("resolution")
    if (
        not isinstance(binding_quality, Mapping)
        or not isinstance(binding_plan, Mapping)
        or not isinstance(binding_resolution, Mapping)
        or not isinstance(full_replay, Mapping)
        or set(full_replay) != _FULL_REPLAY_FIELDS
        or not isinstance(resolution, Mapping)
        or set(resolution) != _RESOLUTION_SUMMARY_FIELDS
    ):
        raise QualityResolutionError(
            "quality-training input-readiness source evidence is malformed"
        )
    trajectories = full_replay.get("continuation_trajectories")
    resolved_rows = resolution.get("records_used")
    resolved_lineages = resolution.get("resolved_lineages")
    minimum_rows = preflight.get("minimum_resolved_rows")
    minimum_lineages = preflight.get("minimum_resolved_lineages")
    for value, label in (
        (trajectories, "preflight continuation trajectories"),
        (resolved_rows, "preflight resolved rows"),
        (resolved_lineages, "preflight resolved lineages"),
        (minimum_rows, "preflight minimum resolved rows"),
        (minimum_lineages, "preflight minimum resolved lineages"),
    ):
        _nonnegative(value, label, positive=True)
    if (
        full_replay.get("mode") != "full-exact-continuation-and-evaluator"
        or full_replay.get("continuation_replays_executed") != trajectories
        or full_replay.get("continuation_replay_matches") != trajectories
        or resolution.get("passes_resolution") is not True
        or resolution.get("continuation_trajectories_authenticated") != trajectories
        or resolution.get("continuation_replays_executed") != trajectories
        or resolution.get("continuation_replay_matches") != trajectories
        or resolution.get("continuation_replay_mode")
        != "full-exact-continuation-and-evaluator"
        or resolution.get("minimum_resolved_rows") != minimum_rows
        or resolution.get("minimum_resolved_lineages") != minimum_lineages
        or int(resolved_rows) < int(minimum_rows)
        or int(resolved_lineages) < int(minimum_lineages)
        or preflight.get("source_quality_manifest_sha256")
        != binding_quality.get("raw_sha256")
        or preflight.get("source_quality_manifest_record_digest")
        != binding_quality.get("record_digest")
        or preflight.get("source_quality_records_sha256")
        != binding_quality.get("records_sha256")
    ):
        raise QualityResolutionError(
            "quality-training input-readiness sources are not mutually bound"
        )
    expected_payload: dict[str, object] = {
        "binding": {
            "raw_sha256": binding_sha,
            "record_digest": binding["record_digest"],
        },
        "checks": {
            "authority_and_protocol": True,
            "complete_exact_replay": True,
            "publication_binding": True,
            "resolution_thresholds": True,
        },
        "continuation_trajectories": trajectories,
        "pass": True,
        "plan": dict(binding_plan),
        "quality_manifest": {
            "raw_sha256": binding_quality["raw_sha256"],
            "record_digest": binding_quality["record_digest"],
            "records_sha256": binding_quality["records_sha256"],
        },
        "quality_preflight": {
            "raw_sha256": preflight_sha,
            "record_digest": preflight["record_digest"],
        },
        "resolution_receipt": dict(binding_resolution),
        "resolved_lineages": resolved_lineages,
        "resolved_rows": resolved_rows,
        "schema": QUALITY_TRAINING_INPUT_READINESS_SCHEMA,
        "schema_version": QUALITY_TRAINING_INPUT_READINESS_VERSION,
        "scope": "quality-supervision-input-only",
        "selected_continuations": binding["selected_continuations"],
    }
    expected = {**expected_payload, "record_digest": content_digest(expected_payload)}
    if readiness != expected:
        raise QualityResolutionError(
            "quality-training input-readiness receipt cannot be reproduced"
        )
    return QualityTrainingInputReadiness(readiness, observed_sha)


__all__ = [
    "QUALITY_RESOLUTION_BINDING_SCHEMA",
    "QUALITY_RESOLUTION_BINDING_VERSION",
    "QUALITY_TRAINING_INPUT_READINESS_SCHEMA",
    "QUALITY_TRAINING_INPUT_READINESS_VERSION",
    "QualityResolutionBinding",
    "QualityTrainingInputReadiness",
    "load_quality_resolution_binding",
    "load_quality_training_input_readiness",
    "verify_quality_resolution_binding",
    "verify_quality_training_input_readiness",
]

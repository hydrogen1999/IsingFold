"""Pure three-seed aggregation for the sealed complete-system endpoint.

The CLI authenticates files before constructing :class:`AuthenticatedCompleteSystemSeedRun`.
This module then binds every report to its typed receipt batch, rejects seed or contract mixing,
and computes the preregistered equal-seed/equal-lineage estimand.  It performs no filesystem I/O
and makes no comparison against an external system.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from isingfold.rl.checkpoint import (
    RUNTIME_DEPENDENCIES,
    RUNTIME_IMPLEMENTATION_SCHEMA,
    RUNTIME_IMPLEMENTATION_VERSION,
    RUNTIME_MODULE_SOURCES,
)
from isingfold.rl.complete_system import (
    CompletePopulationIdentity,
    CompleteSystemEvidenceRecord,
    CompleteSystemReceipt,
    FrozenComponentIdentity,
    complete_system_metrics,
)
from isingfold.rl.contracts import stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.experiment_selection import crossed_bootstrap_bounds
from isingfold.rl.evaluation_strata import preregistered_subgroups
from isingfold.rl.external import BackendIdentity

COMPLETE_SYSTEM_AGGREGATE_SCHEMA = "isingfold.complete-system-three-seed-aggregate"
COMPLETE_SYSTEM_AGGREGATE_VERSION = 2
COMPLETE_SYSTEM_EVALUATION_SCHEMA = "isingfold.complete-system-evaluation"
COMPLETE_SYSTEM_EVALUATION_VERSION = 4

REGISTERED_TRAINING_SEEDS = (1103, 2207, 3301)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 130363
TWO_SIDED_ALPHA = 0.05
TAIL_ALPHA = TWO_SIDED_ALPHA / 2.0
AGGREGATION = "equal-training-seed-then-equal-immutable-base-lineage"
PRIMARY_METRIC = "unconditional-if-q3-s0-utility"

_REPORT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "partition",
        "sealed_test_opened",
        "evaluation_protocol",
        "deployment_rule",
        "training_seed_index",
        "training_seed",
        "source_cell_id",
        "selection_binding",
        "source_corpus_manifest_sha256",
        "quality_authority",
        "target_access",
        "ground_partition_receipt",
        "complete_system_config_sha256",
        "bootstrap_bank_access",
        "same_support_contract_digest",
        "context",
        "context_digest",
        "population",
        "population_digest",
        "controller",
        "selector",
        "initializer",
        "method",
        "runtime_platform",
        "inference_device_type",
        "inference_device_name",
        "inference_threads",
        "deterministic",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "repetitions",
        "metrics",
        "artifacts",
        "record_digest",
    }
)
_PROTOCOL_KEYS = frozenset(
    {
        "partition",
        "evaluation_seed",
        "repetitions",
        "audit_reads",
        "deployment_rule",
        "population_scope",
        "aggregation",
        "primary_metric",
        "bootstrap_replicates",
        "bootstrap_seed",
        "two_sided_alpha",
        "feasibility_noninferiority_margin",
        "learned_config_digest",
        "learned_config_file_sha256",
        "external_config_digest",
        "external_config_file_sha256",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "selection_receipt_sha256",
        "selection_record_digest",
        "grid_manifest_sha256",
        "selected_model_family",
        "selected_grid_model_family",
        "selected_method",
        "all_training_seeds",
        "all_source_cell_ids",
        "all_source_checkpoint_payload_digests",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
        "source_cell_id",
        "source_checkpoint_payload_digest",
        "training_seed",
        "seed_selection_forbidden",
        "selected_validation_checkpoint_reused",
        "fresh_representation_stage_required",
        "retraining_rule",
    }
)
_SELECTION_SPECIFIC = frozenset(
    {"source_cell_id", "source_checkpoint_payload_digest", "training_seed"}
)


def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in sorted(value.items())})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: object) -> object:
    return json.loads(canonical_json_bytes(value))


def _canonical_copy(value: Mapping[str, object]) -> Mapping[str, object]:
    copied = _plain_json(value)
    assert isinstance(copied, dict)
    frozen = _freeze_json(copied)
    assert isinstance(frozen, Mapping)
    return frozen


def _receipt_file_sha256(receipts: Sequence[CompleteSystemReceipt]) -> str:
    content = b"".join(
        (
            json.dumps(
                receipt.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for receipt in sorted(receipts, key=lambda item: item.pair_key)
    )
    return hashlib.sha256(content).hexdigest()


def _evidence_file_sha256(evidence: Sequence[CompleteSystemEvidenceRecord]) -> str:
    content = b"".join(
        (
            json.dumps(
                record.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in sorted(evidence, key=lambda item: item.pair_key)
    )
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class AuthenticatedCompleteSystemSeedRun:
    """One already loaded report and its authenticated complete-receipt artifact."""

    report: Mapping[str, object]
    receipts: tuple[CompleteSystemReceipt, ...]
    evidence: tuple[CompleteSystemEvidenceRecord, ...]

    def __post_init__(self) -> None:
        report = _mapping(self.report, label="complete-system evaluation report")
        canonical_json_bytes(report)
        object.__setattr__(self, "report", _canonical_copy(report))
        receipts = tuple(self.receipts)
        if not receipts or any(not isinstance(row, CompleteSystemReceipt) for row in receipts):
            raise TypeError("complete-system seed run requires typed nonempty receipts")
        object.__setattr__(self, "receipts", receipts)
        evidence = tuple(self.evidence)
        if not evidence or any(
            not isinstance(row, CompleteSystemEvidenceRecord) for row in evidence
        ):
            raise TypeError("complete-system seed run requires typed terminal evidence")
        receipt_by_pair = {receipt.pair_key: receipt for receipt in receipts}
        evidence_by_pair = {record.pair_key: record for record in evidence}
        if len(evidence_by_pair) != len(evidence) or set(evidence_by_pair) != set(
            receipt_by_pair
        ):
            raise ValueError("complete-system evidence does not cover its receipt census")
        for key, receipt in receipt_by_pair.items():
            record = evidence_by_pair[key]
            terminal_digest = (
                None if record.terminal_evidence is None else record.terminal_evidence.digest
            )
            if (
                record.population_digest != receipt.population.digest
                or record.complete_receipt_digest != receipt.as_dict()["record_digest"]
                or terminal_digest != receipt.terminal_evidence_digest
                or receipt.outcome.returned_valid != (record.terminal_evidence is not None)
            ):
                raise ValueError("complete-system evidence link differs from its receipt")
        object.__setattr__(self, "evidence", evidence)

    @property
    def report_file_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.report) + b"\n").hexdigest()


@dataclass(frozen=True)
class _ValidatedRun:
    source: AuthenticatedCompleteSystemSeedRun
    index: int
    training_seed: int
    source_cell_id: str
    source_checkpoint_payload_digest: str
    selection_common: Mapping[str, object]
    protocol: Mapping[str, object]
    population: CompletePopulationIdentity
    config_digest: str
    context_digest: str
    policy_context_digest: str
    initializer: BackendIdentity
    selector: FrozenComponentIdentity
    controller: FrozenComponentIdentity


def _validate_protocol(value: object) -> Mapping[str, object]:
    protocol = _mapping(value, label="complete-system evaluation protocol")
    _exact_keys(protocol, _PROTOCOL_KEYS, label="complete-system evaluation protocol")
    for name in (
        "evaluation_seed",
        "repetitions",
        "audit_reads",
        "bootstrap_replicates",
        "bootstrap_seed",
    ):
        _integer(protocol[name], label=f"complete-system protocol {name}")
    if type(protocol["two_sided_alpha"]) is not float or type(
        protocol["feasibility_noninferiority_margin"]
    ) is not float:
        raise ValueError("complete-system protocol alpha and NI margin must be floats")
    pinned_fields = {
        "learned_config_digest",
        "learned_config_file_sha256",
        "external_config_digest",
        "external_config_file_sha256",
    }
    if any(not _is_digest(protocol[name]) for name in pinned_fields):
        raise ValueError("complete-system protocol config pins are not SHA-256 digests")
    expected = {
        "partition": "test",
        "evaluation_seed": 55079,
        "repetitions": 4,
        "audit_reads": 4096,
        "deployment_rule": "categorical-temperature-one",
        "population_scope": "all-policy-instances-before-initialization",
        "aggregation": AGGREGATION,
        "primary_metric": PRIMARY_METRIC,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "two_sided_alpha": TWO_SIDED_ALPHA,
        "feasibility_noninferiority_margin": 0.02,
    }
    if {key: value for key, value in protocol.items() if key not in pinned_fields} != expected:
        raise ValueError("complete-system evaluation protocol differs from frozen v1")
    return protocol


def _string_array(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{label} must be a nonempty string array")
    return tuple(value)


def _digest_array(value: object, *, label: str) -> tuple[str, ...]:
    result = _string_array(value, label=label)
    if any(not _is_digest(item) for item in result):
        raise ValueError(f"{label} contains a non-SHA-256 value")
    return result


def _validate_selection(
    value: object, *, report_index: int, report_seed: int, report_cell: str
) -> tuple[Mapping[str, object], str]:
    binding = _mapping(value, label="complete-system selection binding")
    _exact_keys(binding, _SELECTION_KEYS, label="complete-system selection binding")
    if (
        binding["schema"] != "isingfold.complete-system-selected-training"
        or type(binding["schema_version"]) is not int
        or binding["schema_version"] != 3
        or binding["seed_selection_forbidden"] is not True
        or binding["selected_validation_checkpoint_reused"] is not False
        or type(binding["fresh_representation_stage_required"]) is not bool
        or binding["fresh_representation_stage_required"]
        != (binding["selected_method"] in {"supervised-only", "ppo-warm-start"})
        or binding["retraining_rule"]
        != "fresh-selected-configuration-on-persistent-k2-cache-support"
    ):
        raise ValueError("complete-system selection binding has an unsupported contract")
    for name in (
        "selection_receipt_sha256",
        "selection_record_digest",
        "grid_manifest_sha256",
    ):
        if not _is_digest(binding[name]):
            raise ValueError(f"complete-system selection {name} is not a SHA-256 digest")
    for name in ("selected_model_family", "selected_grid_model_family", "selected_method"):
        _string(binding[name], label=f"complete-system selection {name}")
    seeds_raw = binding["all_training_seeds"]
    if (
        not isinstance(seeds_raw, Sequence)
        or isinstance(seeds_raw, (str, bytes, bytearray))
        or any(type(seed) is not int for seed in seeds_raw)
        or tuple(seeds_raw) != REGISTERED_TRAINING_SEEDS
    ):
        raise ValueError("complete-system selection does not bind the exact three training seeds")
    cells = _string_array(binding["all_source_cell_ids"], label="selected source cells")
    checkpoints = _digest_array(
        binding["all_source_checkpoint_payload_digests"],
        label="selected checkpoint payload digests",
    )
    if len(cells) != 3 or len(set(cells)) != 3 or len(checkpoints) != 3:
        raise ValueError("complete-system selection does not bind exactly three selected cells")
    if (
        binding["training_seed"] != report_seed
        or binding["source_cell_id"] != report_cell
        or binding["training_seed"] != seeds_raw[report_index]
        or binding["source_cell_id"] != cells[report_index]
        or binding["source_checkpoint_payload_digest"] != checkpoints[report_index]
    ):
        raise ValueError("complete-system report differs from its selected-seed index")
    selected_runtime_registry = _mapping(
        binding["runtime_implementation_registry"],
        label="complete-system selected runtime implementation registry",
    )
    if (
        not _is_digest(binding["runtime_implementation_digest"])
        or content_digest(selected_runtime_registry)
        != binding["runtime_implementation_digest"]
    ):
        raise ValueError("complete-system selection has an inconsistent runtime registry")
    for name in (
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
    ):
        if not _is_digest(binding[name]):
            raise ValueError(f"complete-system selection {name} is not a SHA-256 digest")
    common = {key: binding[key] for key in sorted(set(binding) - _SELECTION_SPECIFIC)}
    return MappingProxyType(common), checkpoints[report_index]


def _validate_artifacts(
    value: object,
    receipts: Sequence[CompleteSystemReceipt],
    evidence: Sequence[CompleteSystemEvidenceRecord],
) -> str:
    artifacts = _mapping(value, label="complete-system artifacts")
    if set(artifacts) != {"complete_receipts", "terminal_evidence", "outcomes"}:
        raise ValueError("complete-system artifact registry is incomplete")
    expected_paths = {
        "complete_receipts": "complete_receipts.jsonl",
        "terminal_evidence": "terminal_evidence.jsonl",
        "outcomes": "outcomes.jsonl",
    }
    for name in ("complete_receipts", "terminal_evidence", "outcomes"):
        artifact = _mapping(artifacts[name], label=f"complete-system {name} artifact")
        if set(artifact) != {"path", "sha256", "count"}:
            raise ValueError(f"complete-system {name} artifact fields differ")
        if artifact["path"] != expected_paths[name]:
            raise ValueError(f"complete-system {name} artifact path differs")
        if not _is_digest(artifact["sha256"]):
            raise ValueError(f"complete-system {name} artifact SHA-256 is invalid")
        if _integer(artifact["count"], label=f"complete-system {name} count", minimum=1) != len(
            receipts
        ):
            raise ValueError(f"complete-system {name} count differs from receipts")
    receipt_sha = _receipt_file_sha256(receipts)
    receipt_artifact = _mapping(
        artifacts["complete_receipts"], label="complete-system receipt artifact"
    )
    if receipt_artifact["sha256"] != receipt_sha:
        raise ValueError("complete-system report has the wrong receipt file identity")
    evidence_artifact = _mapping(
        artifacts["terminal_evidence"], label="complete-system evidence artifact"
    )
    if evidence_artifact["sha256"] != _evidence_file_sha256(evidence):
        raise ValueError("complete-system report has the wrong evidence file identity")
    return receipt_sha


def _validate_run(run: AuthenticatedCompleteSystemSeedRun) -> _ValidatedRun:
    report = run.report
    _exact_keys(report, _REPORT_KEYS, label="complete-system evaluation report")
    if (
        report["schema"] != COMPLETE_SYSTEM_EVALUATION_SCHEMA
        or type(report["schema_version"]) is not int
        or report["schema_version"] != COMPLETE_SYSTEM_EVALUATION_VERSION
    ):
        raise ValueError("unsupported complete-system evaluation report schema")
    recorded = report["record_digest"]
    payload = {key: value for key, value in report.items() if key != "record_digest"}
    if not _is_digest(recorded) or recorded != content_digest(payload):
        raise ValueError("complete-system evaluation report digest mismatch")
    if report["partition"] != "test" or report["sealed_test_opened"] is not True:
        raise ValueError("complete-system aggregation requires sealed test reports")
    protocol = _validate_protocol(report["evaluation_protocol"])
    if (
        report["deployment_rule"] != protocol["deployment_rule"]
        or _integer(report["repetitions"], label="complete-system report repetitions", minimum=1)
        != protocol["repetitions"]
    ):
        raise ValueError("complete-system report and evaluation protocol disagree")
    index = _integer(report["training_seed_index"], label="training-seed index")
    if index not in range(3):
        raise ValueError("complete-system training-seed index is outside zero through two")
    seed = _integer(report["training_seed"], label="training seed")
    source_cell = _string(report["source_cell_id"], label="source cell")
    selection_common, source_checkpoint = _validate_selection(
        report["selection_binding"],
        report_index=index,
        report_seed=seed,
        report_cell=source_cell,
    )
    for name in (
        "source_corpus_manifest_sha256",
        "complete_system_config_sha256",
        "context_digest",
        "population_digest",
    ):
        if not _is_digest(report[name]):
            raise ValueError(f"complete-system report {name} is not a SHA-256 digest")
    context = _mapping(report["context"], label="complete-system context")
    if stable_digest(_plain_json(context)) != report["context_digest"]:
        raise ValueError("complete-system report context digest mismatch")
    quality_authority = _mapping(
        report["quality_authority"], label="complete-system quality authority"
    )
    if not quality_authority:
        raise ValueError("complete-system quality authority cannot be empty")
    target_access = _mapping(
        report["target_access"], label="complete-system target access"
    )
    ground_partition = _mapping(
        report["ground_partition_receipt"],
        label="complete-system ground partition receipt",
    )
    for record, label in (
        (target_access, "target access"),
        (ground_partition, "ground partition"),
    ):
        recorded_digest = record.get("record_digest")
        record_payload = {
            key: value for key, value in record.items() if key != "record_digest"
        }
        if not _is_digest(recorded_digest) or recorded_digest != content_digest(
            record_payload
        ):
            raise ValueError(f"complete-system {label} digest mismatch")
    bootstrap_access = _mapping(
        report["bootstrap_bank_access"],
        label="complete-system bootstrap-bank access",
    )
    bootstrap_recorded_digest = bootstrap_access.get("record_digest")
    bootstrap_payload = {
        key: value
        for key, value in bootstrap_access.items()
        if key != "record_digest"
    }
    if (
        bootstrap_access.get("schema") != "isingfold.rl-value-bootstrap-access"
        or bootstrap_access.get("schema_version") != 1
        or bootstrap_access.get("protocol_preset") != "final-test"
        or bootstrap_access.get("opened_evaluator_targets") is not False
        or not _is_digest(bootstrap_recorded_digest)
        or bootstrap_recorded_digest != content_digest(bootstrap_payload)
        or not _is_digest(report["same_support_contract_digest"])
        or bootstrap_access.get("same_support_contract_digest")
        != report["same_support_contract_digest"]
    ):
        raise ValueError(
            "complete-system bootstrap-bank access receipt or support binding is invalid"
        )
    partition_authority = quality_authority.get("evaluation_partition")
    ground_identity = (
        partition_authority.get("ground_partition")
        if isinstance(partition_authority, Mapping)
        else None
    )
    if (
        not isinstance(partition_authority, Mapping)
        or partition_authority.get("name") != "test"
        or target_access.get("record_digest")
        != partition_authority.get("target_access_record_digest")
        or not isinstance(ground_identity, Mapping)
        or ground_partition.get("record_digest")
        != ground_identity.get("receipt_record_digest")
    ):
        raise ValueError("complete-system target receipts differ from authority")
    for name in ("population", "controller", "selector", "initializer", "method"):
        _mapping(report[name], label=f"complete-system report {name}")
    _mapping(report["runtime_platform"], label="complete-system runtime platform")
    receipts = run.receipts
    metrics = _mapping(report["metrics"], label="complete-system source metrics")
    if _plain_json(metrics) != complete_system_metrics(receipts):
        raise ValueError("complete-system report metrics differ from authenticated receipts")
    _string(report["inference_device_type"], label="complete-system inference device")
    _string(report["inference_device_name"], label="complete-system inference device name")
    _integer(report["inference_threads"], label="complete-system inference threads", minimum=1)
    if type(report["deterministic"]) is not bool:
        raise ValueError("complete-system deterministic identity must be Boolean")
    registry = _mapping(
        report["runtime_implementation_registry"],
        label="complete-system runtime implementation registry",
    )
    if (
        not _is_digest(report["runtime_implementation_digest"])
        or content_digest(registry) != report["runtime_implementation_digest"]
    ):
        raise ValueError("complete-system runtime implementation digest mismatch")
    modules = _mapping(registry.get("modules"), label="runtime source registry")
    dependencies = _mapping(
        registry.get("dependencies"), label="runtime dependency registry"
    )
    expected_modules = dict(RUNTIME_MODULE_SOURCES)
    if (
        registry.get("schema") != RUNTIME_IMPLEMENTATION_SCHEMA
        or registry.get("schema_version") != RUNTIME_IMPLEMENTATION_VERSION
        or set(modules) != set(expected_modules)
        or set(dependencies) != {"python", *RUNTIME_DEPENDENCIES}
    ):
        raise ValueError("complete-system runtime registry is incomplete")
    for name, path in expected_modules.items():
        entry = _mapping(modules[name], label=f"runtime source {name}")
        if (
            set(entry) != {"source_path", "sha256"}
            or entry["source_path"] != path
            or not _is_digest(entry["sha256"])
        ):
            raise ValueError("complete-system runtime source identity is malformed")
    if any(not isinstance(value, str) or not value for value in dependencies.values()):
        raise ValueError("complete-system runtime dependency identity is malformed")
    if (
        _plain_json(selection_common["runtime_implementation_registry"])
        != _plain_json(registry)
        or selection_common["runtime_implementation_digest"]
        != report["runtime_implementation_digest"]
    ):
        raise ValueError(
            "complete-system report runtime differs from its frozen selection registry"
        )

    reference = receipts[0]
    expected_controller_id = (
        f"isingfold-policy/{selection_common['selected_model_family']}/"
        f"{selection_common['selected_method']}/seed-{seed}"
    )
    if reference.controller.component_id != expected_controller_id:
        raise ValueError("complete-system controller differs from its selected-seed identity")
    pair_keys = [receipt.pair_key for receipt in receipts]
    expected_keys = {
        (lineage, instance, repetition)
        for lineage, instance in reference.population.expected_instances
        for repetition in range(reference.population.expected_repetitions)
    }
    if len(pair_keys) != len(set(pair_keys)) or set(pair_keys) != expected_keys:
        raise ValueError("complete-system seed receipts do not cover the sealed population")
    report_controller = dict(_mapping(report["controller"], label="report controller"))
    for receipt in receipts:
        receipt.outcome.validate_receipt(require_complete=False)
        if (
            receipt.population != reference.population
            or receipt.config_digest != reference.config_digest
            or receipt.context_digest != reference.context_digest
            or receipt.policy_context_digest != reference.policy_context_digest
            or receipt.initializer != reference.initializer
            or receipt.selector != reference.selector
            or receipt.controller != reference.controller
        ):
            raise ValueError("one complete-system seed receipt batch mixes frozen contracts")
        if receipt.outcome.population_eligible is not True:
            raise ValueError("complete-system aggregate excludes conditional-only outcomes")
        if receipt.outcome.returned_valid:
            if receipt.evaluator_seed is None or receipt.outcome.evaluator_seed is None:
                raise ValueError("valid complete-system receipt lacks evaluator provenance")
            if receipt.outcome.evaluator_reads != protocol["audit_reads"]:
                raise ValueError("valid complete-system receipt has the wrong audit-read count")
        elif (
            receipt.evaluator_seed is not None
            or receipt.outcome.evaluator_seed is not None
            or receipt.outcome.utility != 0.0
        ):
            raise ValueError(
                "invalid complete-system receipt must retain zero utility and null evaluator seed"
            )
    if (
        _plain_json(report["population"]) != reference.population.as_dict()
        or report["population_digest"] != reference.population.digest
        or report["source_corpus_manifest_sha256"] != reference.population.source_manifest_sha256
    ):
        raise ValueError("complete-system report and receipt population identities differ")
    if (
        report["context_digest"] != reference.context_digest
        or _plain_json(report["selector"]) != reference.selector.as_dict()
        or _plain_json(report["initializer"]) != reference.initializer.as_dict()
        or report_controller != reference.controller.as_dict()
    ):
        raise ValueError(
            "complete-system report context, initializer, or selector identity differs"
        )
    if (
        reference.population.expected_repetitions != protocol["repetitions"]
        or reference.population.evaluation_seed != protocol["evaluation_seed"]
        or reference.population.confirmatory_design.noninferiority_margin
        != protocol["feasibility_noninferiority_margin"]
        or reference.population.confirmatory_design.learning_partition
        != protocol["partition"]
    ):
        raise ValueError("complete-system population and evaluation protocol disagree")
    method = _mapping(report["method"], label="complete-system method")
    method_config = _mapping(method.get("config"), label="complete-system method config")
    if (
        method.get("config_digest") != reference.config_digest
        or stable_digest(_plain_json(method_config)) != reference.config_digest
        or method.get("initializer") != reference.initializer.as_dict()
        or method.get("controller") != reference.controller.as_dict()
        or method.get("selector") != reference.selector.as_dict()
        or method.get("population_digest") != reference.population.digest
        or method.get("online_evaluator_feedback") is not False
    ):
        raise ValueError("complete-system report method differs from its frozen receipts")
    if (
        reference.config_digest != protocol["learned_config_digest"]
        or report["complete_system_config_sha256"]
        != protocol["learned_config_file_sha256"]
    ):
        raise ValueError("complete-system report config differs from preregistered pins")
    _validate_artifacts(report["artifacts"], receipts, run.evidence)
    return _ValidatedRun(
        source=run,
        index=index,
        training_seed=seed,
        source_cell_id=source_cell,
        source_checkpoint_payload_digest=source_checkpoint,
        selection_common=selection_common,
        protocol=protocol,
        population=reference.population,
        config_digest=reference.config_digest,
        context_digest=reference.context_digest,
        policy_context_digest=reference.policy_context_digest,
        initializer=reference.initializer,
        selector=reference.selector,
        controller=reference.controller,
    )


def _lineage_values(run: _ValidatedRun) -> tuple[dict[str, float], dict[str, float], int]:
    utility: dict[str, list[float]] = {}
    validity: dict[str, list[float]] = {}
    failures = 0
    for receipt in run.source.receipts:
        outcome = receipt.outcome
        if outcome.returned_valid:
            if outcome.utility is None or not math.isfinite(outcome.utility):
                raise ValueError("valid complete-system outcome has no finite utility")
            value = float(outcome.utility)
        else:
            value = 0.0
            failures += 1
        utility.setdefault(outcome.lineage, []).append(value)
        validity.setdefault(outcome.lineage, []).append(float(outcome.returned_valid))
    return (
        {lineage: float(np.mean(values)) for lineage, values in utility.items()},
        {lineage: float(np.mean(values)) for lineage, values in validity.items()},
        failures,
    )


def _subgroup_lineage_values(
    run: _ValidatedRun, identities: frozenset[tuple[str, str]]
) -> tuple[dict[str, float], dict[str, float], int]:
    utility: dict[str, list[float]] = {}
    validity: dict[str, list[float]] = {}
    attempts = 0
    for receipt in run.source.receipts:
        if (receipt.lineage, receipt.instance) not in identities:
            continue
        attempts += 1
        outcome = receipt.outcome
        value = float(outcome.utility) if outcome.returned_valid else 0.0
        if not math.isfinite(value):
            raise ValueError("subgroup contains a non-finite complete-system utility")
        utility.setdefault(outcome.lineage, []).append(value)
        validity.setdefault(outcome.lineage, []).append(float(outcome.returned_valid))
    if not utility or set(utility) != set(validity):
        raise ValueError("registered subgroup has no complete-system observations")
    return (
        {lineage: float(np.mean(values)) for lineage, values in utility.items()},
        {lineage: float(np.mean(values)) for lineage, values in validity.items()},
        attempts,
    )


def _descriptive_subgroup_results(
    runs: Sequence[_ValidatedRun], population: CompletePopulationIdentity
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, object]]:
    registry = preregistered_subgroups(population.evaluation_strata)
    group_count = sum(len(groups) for groups in registry.values())
    comparisons = 2 * group_count
    if comparisons <= 0:
        raise ValueError("evaluation population has no registered subgroup comparisons")
    two_sided_alpha = TWO_SIDED_ALPHA / comparisons
    tail_alpha = two_sided_alpha / 2.0
    results: dict[str, dict[str, dict[str, object]]] = {}
    family_index = 0
    for dimension, groups in registry.items():
        dimension_results: dict[str, dict[str, object]] = {}
        for value, identities in groups.items():
            rows = [_subgroup_lineage_values(run, identities) for run in runs]
            lineages = tuple(sorted(rows[0][0]))
            if any(tuple(sorted(utility)) != lineages for utility, _, _ in rows):
                raise ValueError("training seeds disagree on registered subgroup lineages")
            utility_matrix = np.asarray(
                [[utility[lineage] for lineage in lineages] for utility, _, _ in rows],
                dtype=float,
            )
            validity_matrix = np.asarray(
                [[validity[lineage] for lineage in lineages] for _, validity, _ in rows],
                dtype=float,
            )
            utility_bounds = crossed_bootstrap_bounds(
                utility_matrix,
                replicates=BOOTSTRAP_REPLICATES,
                seed=int(
                    stable_digest(
                        {
                            "base_seed": BOOTSTRAP_SEED,
                            "dimension": dimension,
                            "value": value,
                            "endpoint": "utility",
                        }
                    )[:16],
                    16,
                ),
                alpha=tail_alpha,
            )
            validity_bounds = crossed_bootstrap_bounds(
                validity_matrix,
                replicates=BOOTSTRAP_REPLICATES,
                seed=int(
                    stable_digest(
                        {
                            "base_seed": BOOTSTRAP_SEED,
                            "dimension": dimension,
                            "value": value,
                            "endpoint": "validity",
                        }
                    )[:16],
                    16,
                ),
                alpha=tail_alpha,
            )
            dimension_results[value] = {
                "family_index": family_index,
                "identity_census_digest": stable_digest(
                    [list(identity) for identity in sorted(identities)]
                ),
                "instance_count": len(identities),
                "independent_lineages": len(lineages),
                "attempts_per_training_seed": rows[0][2],
                "unconditional_utility_mean": float(np.mean(utility_matrix)),
                "unconditional_utility_confidence_interval": {
                    "lower": utility_bounds[0],
                    "upper": utility_bounds[1],
                    "two_sided_alpha": two_sided_alpha,
                },
                "system_valid_return_rate": float(np.mean(validity_matrix)),
                "system_valid_return_rate_confidence_interval": {
                    "lower": validity_bounds[0],
                    "upper": validity_bounds[1],
                    "two_sided_alpha": two_sided_alpha,
                },
                "weighting": AGGREGATION,
                "confirmatory_claim": False,
            }
            family_index += 1
        results[dimension] = dimension_results
    inference = {
        "scope": "descriptive-not-powered",
        "registered_dimensions": list(registry),
        "independent_unit": "immutable-base-lineage",
        "weighting": AGGREGATION,
        "familywise_alpha": TWO_SIDED_ALPHA,
        "multiplicity_method": "bonferroni",
        "group_count": group_count,
        "endpoints_per_group": 2,
        "comparison_count": comparisons,
        "per_interval_two_sided_alpha": two_sided_alpha,
        "per_tail_alpha": tail_alpha,
        "bootstrap_replicates_per_interval": BOOTSTRAP_REPLICATES,
        "powered_subgroup_claims": False,
    }
    return results, inference


def _system_seed_schedule(run: _ValidatedRun) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (receipt.lineage, receipt.instance, receipt.repetition, receipt.system_seed)
        for receipt in sorted(run.source.receipts, key=lambda item: item.pair_key)
    )


def aggregate_complete_system_seeds(
    runs: Sequence[AuthenticatedCompleteSystemSeedRun],
) -> dict[str, object]:
    """Aggregate exactly the three frozen selected seeds on the complete-system test set."""

    materialized = tuple(runs)
    if len(materialized) != 3 or any(
        not isinstance(run, AuthenticatedCompleteSystemSeedRun) for run in materialized
    ):
        raise ValueError("complete-system aggregation requires exactly three authenticated runs")
    validated = tuple(
        sorted((_validate_run(run) for run in materialized), key=lambda row: row.index)
    )
    if tuple(run.index for run in validated) != (0, 1, 2):
        raise ValueError("complete-system reports must cover exactly training-seed indices 0,1,2")
    if tuple(run.training_seed for run in validated) != REGISTERED_TRAINING_SEEDS:
        raise ValueError("complete-system reports do not cover the exact three training seeds")

    reference = validated[0]
    for run in validated[1:]:
        if (
            run.selection_common != reference.selection_common
            or run.protocol != reference.protocol
            or run.population != reference.population
            or run.config_digest != reference.config_digest
            or run.context_digest != reference.context_digest
            or run.policy_context_digest != reference.policy_context_digest
            or run.initializer != reference.initializer
            or run.selector != reference.selector
            or run.source.report["complete_system_config_sha256"]
            != reference.source.report["complete_system_config_sha256"]
            or run.source.report["context"] != reference.source.report["context"]
            or run.source.report["quality_authority"]
            != reference.source.report["quality_authority"]
            or run.source.report["target_access"]
            != reference.source.report["target_access"]
            or run.source.report["ground_partition_receipt"]
            != reference.source.report["ground_partition_receipt"]
            or run.source.report["bootstrap_bank_access"]
            != reference.source.report["bootstrap_bank_access"]
            or run.source.report["same_support_contract_digest"]
            != reference.source.report["same_support_contract_digest"]
            or run.source.report["runtime_implementation_registry"]
            != reference.source.report["runtime_implementation_registry"]
            or run.source.report["runtime_implementation_digest"]
            != reference.source.report["runtime_implementation_digest"]
        ):
            raise ValueError(
                "complete-system reports mix protocol, population, config, context, "
                "initializer, or selector identities"
            )

    system_seed_schedule = _system_seed_schedule(reference)
    if any(_system_seed_schedule(run) != system_seed_schedule for run in validated[1:]):
        raise ValueError("complete-system reports use different system-seed schedules")
    evaluator_seeds_by_pair: dict[tuple[str, str, int], set[int]] = {}
    for run in validated:
        for receipt in run.source.receipts:
            if receipt.evaluator_seed is not None:
                evaluator_seeds_by_pair.setdefault(receipt.pair_key, set()).add(
                    receipt.evaluator_seed
                )
    if any(len(seeds) != 1 for seeds in evaluator_seeds_by_pair.values()):
        raise ValueError("complete-system reports use different evaluator-seed schedules")

    statistics = [_lineage_values(run) for run in validated]
    lineages = tuple(sorted(statistics[0][0]))
    if any(tuple(sorted(values)) != lineages for values, _, _ in statistics):
        raise ValueError("complete-system seeds use different immutable base lineages")
    utility_matrix = np.asarray(
        [[utility[lineage] for lineage in lineages] for utility, _, _ in statistics],
        dtype=float,
    )
    validity_matrix = np.asarray(
        [[validity[lineage] for lineage in lineages] for _, validity, _ in statistics],
        dtype=float,
    )
    lower, upper = crossed_bootstrap_bounds(
        utility_matrix,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        alpha=TAIL_ALPHA,
    )
    per_seed = []
    for run, (utility, validity, failures) in zip(validated, statistics, strict=True):
        report = run.source.report
        artifacts = _mapping(report["artifacts"], label="complete-system artifacts")
        receipt_artifact = _mapping(
            artifacts["complete_receipts"], label="complete-system receipt artifact"
        )
        per_seed.append(
            {
                "training_seed_index": run.index,
                "training_seed": run.training_seed,
                "source_cell_id": run.source_cell_id,
                "source_checkpoint_payload_digest": run.source_checkpoint_payload_digest,
                "controller": run.controller.as_dict(),
                "report_record_digest": report["record_digest"],
                "report_file_sha256": run.source.report_file_sha256,
                "artifacts": report["artifacts"],
                "complete_receipts_sha256": receipt_artifact["sha256"],
                "attempt_count": len(run.source.receipts),
                "invalid_return_count": failures,
                "pre_policy_initializer_failure_count": report["metrics"][
                    "complete_failure_taxonomy"
                ]["pre_policy_initializer_failures"],
                "policy_environment_bootstrap_failure_count": report["metrics"][
                    "complete_failure_taxonomy"
                ]["policy_environment_bootstrap_failures"],
                "post_bootstrap_failure_count": report["metrics"][
                    "complete_failure_taxonomy"
                ]["post_bootstrap_failures"],
                "invalid_evaluator_seed_defined_count": 0,
                "unconditional_utility_mean": float(
                    np.mean([utility[lineage] for lineage in lineages])
                ),
                "system_valid_return_rate": float(
                    np.mean([validity[lineage] for lineage in lineages])
                ),
            }
        )

    pair_census = [
        list(receipt.pair_key)
        for receipt in sorted(reference.source.receipts, key=lambda item: item.pair_key)
    ]
    subgroup_results, subgroup_inference = _descriptive_subgroup_results(
        validated, reference.population
    )
    selection = reference.selection_common
    payload: dict[str, Any] = {
        "schema": COMPLETE_SYSTEM_AGGREGATE_SCHEMA,
        "schema_version": COMPLETE_SYSTEM_AGGREGATE_VERSION,
        "partition": "test",
        "sealed_test_opened": True,
        "primary_metric": PRIMARY_METRIC,
        "aggregation": AGGREGATION,
        "training_seed_count": 3,
        "training_seeds": list(REGISTERED_TRAINING_SEEDS),
        "seed_selection_forbidden": True,
        "selection_identity": dict(selection),
        "matched_contract": {
            "population": reference.population.as_dict(),
            "population_digest": reference.population.digest,
            "config_digest": reference.config_digest,
            "complete_system_config_sha256": reference.source.report[
                "complete_system_config_sha256"
            ],
            "quality_authority": reference.source.report["quality_authority"],
            "target_access": reference.source.report["target_access"],
            "ground_partition_receipt": reference.source.report[
                "ground_partition_receipt"
            ],
            "bootstrap_bank_access": reference.source.report[
                "bootstrap_bank_access"
            ],
            "same_support_contract_digest": reference.source.report[
                "same_support_contract_digest"
            ],
            "context": reference.source.report["context"],
            "context_digest": reference.context_digest,
            "policy_context_digest": reference.policy_context_digest,
            "initializer": reference.initializer.as_dict(),
            "selector": reference.selector.as_dict(),
            "evaluation_protocol": dict(reference.protocol),
        },
        "independent_lineages": len(lineages),
        "attempts_per_seed": len(reference.source.receipts),
        "pair_census_digest": content_digest(pair_census),
        "system_seed_schedule_digest": content_digest(system_seed_schedule),
        "population_convention": {
            "system_denominator": "every-sealed-preinitialization-instance-repetition",
            "invalid_return_utility": 0.0,
            "invalid_evaluator_seed": None,
            "conditional_episode_filtering": False,
        },
        "invalid_evaluator_seed_defined_count": 0,
        "unconditional_utility_mean": float(np.mean(utility_matrix)),
        "system_valid_return_rate": float(np.mean(validity_matrix)),
        "unconditional_utility_confidence_interval": {
            "lower": lower,
            "upper": upper,
        },
        "bootstrap_protocol": {
            "sampling": (
                "independent-with-replacement-crossed-training-seed-and-immutable-base-lineage"
            ),
            "rng": "numpy.random.Generator(PCG64)",
            "numpy_version": np.__version__,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "two_sided_alpha": TWO_SIDED_ALPHA,
            "tail_alpha": TAIL_ALPHA,
            "quantile": "numpy.percentile-linear",
        },
        "per_seed": per_seed,
        "lineage_aggregates": [
            {
                "lineage": lineage,
                "unconditional_utility_mean": float(np.mean(utility_matrix[:, index])),
                "system_valid_return_rate": float(np.mean(validity_matrix[:, index])),
            }
            for index, lineage in enumerate(lineages)
        ],
        "subgroup_results": subgroup_results,
        "subgroup_inference": subgroup_inference,
        "comparison_scope": "learned-complete-system-only",
        "runtime_comparability": {
            "identical": all(
                run.source.report["runtime_platform"] == reference.source.report["runtime_platform"]
                and run.source.report["inference_device_type"]
                == reference.source.report["inference_device_type"]
                for run in validated
            ),
            "aggregate_latency_reported": False,
        },
    }
    canonical_payload = json.loads(canonical_json_bytes(payload))
    assert isinstance(canonical_payload, dict)
    canonical_payload["record_digest"] = content_digest(canonical_payload)
    return canonical_payload


__all__ = [
    "AuthenticatedCompleteSystemSeedRun",
    "COMPLETE_SYSTEM_AGGREGATE_SCHEMA",
    "aggregate_complete_system_seeds",
]

"""Authenticated loader for the separated IsingFold training artifacts.

The producer is the standalone CandidateBank importer.  Deployment loads only public
instances, protected initializers, splits and their receipts.  A trusted training/evaluation
process must opt in before evaluator targets are opened and joined to an environment task.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import networkx as nx

from isingfold.embedding import LogicalProblem
from isingfold.rl.data.import_embedbench import (
    CERTIFIED_REFERENCE_STATUSES,
    CORPUS_DESIGN_SCHEMA,
    CORPUS_DESIGN_SCHEMA_VERSION,
    CORPUS_DESIGN_SCHEMA_VERSION_V2,
    MINIMUM_TEST_BASE_LINEAGES,
    MINIMUM_TRAIN_BASE_LINEAGES,
    MINIMUM_VALIDATION_BASE_LINEAGES,
    MINIMUM_VALIDATION_TUNING_BASE_LINEAGES,
    PREPARED_CORPUS_DESIGN_RECEIPT_SCHEMA,
    PREPARED_PROVENANCE_SCHEMA_V2,
    PREPARED_SCHEMA_VERSION_V2,
    PREPARED_SCHEMA_VERSION_V3,
    PREPARED_SCHEMA_VERSION_V4,
    REGISTERED_CORPUS_DESIGN_VERSION,
    REGISTERED_CORPUS_DESIGN_VERSION_V2,
    SOURCE_PROVENANCE_SCHEMA_V2,
    TARGET_AUTHORITY_SCHEMA,
    TARGET_AUTHORITY_VERSION,
    VALIDATION_TUNING_FAMILYWISE_ALPHA,
    VALIDATION_TUNING_NONINFERIORITY_MARGIN,
    VALIDATION_TUNING_POWER_TARGET_ID,
    VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL,
    VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH,
    VALIDATION_TUNING_PRECISION_TARGET_ID,
    content_digest,
)
from isingfold.rl.data.quality_attestation import (
    PublisherAttestation,
    QualityAttestationPin,
    QualityEvidenceReceipt,
    load_publisher_attestation,
    validate_attested_provenance,
    validate_quality_evidence,
)
from isingfold.rl.env import EmbeddingTask, Initializer, task_initializer

_FILES_V1 = frozenset(
    {
        "policy_instances.jsonl",
        "initializers.jsonl",
        "evaluator_targets.jsonl",
        "splits.json",
        "manifest.json",
    }
)
_FILES_V2 = _FILES_V1 | {"provenance.jsonl"}
_FILES_V3 = _FILES_V2
_FILES_V4 = frozenset(
    {
        "policy_instances.jsonl",
        "initializers.jsonl",
        "provenance.jsonl",
        "splits.json",
        "manifest.json",
        "targets/train.jsonl",
        "targets/val.jsonl",
        "targets/test.jsonl",
    }
)
_STRATUM_FIELDS = (
    "application_family",
    "problem_origin",
    "host_family",
    "fault_status",
    "distribution_regime",
    "calibration_status",
    "embedding_difficulty",
    "sampling_difficulty",
    "decision_difficulty",
)
_DESIGN_CONDITION_FIELDS = {
    "base_lineage_key",
    "calibration_sha256",
    "learning_partition",
    *_STRATUM_FIELDS,
}
_MODEL_FEATURE_ALLOWLIST = [
    "family",
    "h",
    "host_edges",
    "host_nodes",
    "j",
    "logical_edges",
    "logical_nodes",
    "topology",
]


@dataclass(frozen=True)
class PreparedProvenance:
    """Authenticated non-neural provenance for one production-v2 task."""

    base_parent_lineage: str
    source_logical_lineage: str
    descendant_transform_kinds: tuple[str, ...]
    descendant_transform_sha256: str
    nominal_topology: str
    nominal_size: int
    pristine_host_sha256: str
    active_topology: str
    active_host_sha256: str
    host_artifact_sha256: str
    fault_status: str
    fault_mask_sha256: str
    calibration_status: str
    calibration_sha256: str | None
    distribution_regime: str
    distribution_stratum: str
    source_partition: str
    source_release_id: str
    source_release_manifest_sha256: str
    split_manifest_sha256: str
    group_id: str
    group_record_digest: str
    instance_record_digest: str
    source_record_digest: str
    record_digest: str


@dataclass(frozen=True)
class PreparedDesignCondition:
    """One authenticated base-lineage by realized-condition scientific stratum."""

    base_lineage_key: str
    learning_partition: str
    application_family: str
    problem_origin: str
    host_family: str
    fault_status: str
    distribution_regime: str
    calibration_status: str
    calibration_sha256: str | None
    embedding_difficulty: str
    sampling_difficulty: str
    decision_difficulty: str
    registry_row_digest: str


@dataclass(frozen=True)
class PreparedTask:
    """One Profile-I task with provenance kept outside neural input tensors."""

    task: EmbeddingTask
    task_id: str
    instance_id: str
    partition: str
    initializer_record_digest: str
    public_instance_record_digest: str
    reference_status: str | None
    certificate_digest: str | None
    evaluator_protocol_digest: str | None
    prepared_schema_version: int = 1
    corpus_scope: str = "legacy-pilot-v1"
    provenance: PreparedProvenance | None = None
    design_condition: PreparedDesignCondition | None = None
    quality_attestation_digest: str | None = None
    quality_evidence_manifest_digest: str | None = None
    quality_evidence_manifest_sha256: str | None = None
    quality_target_set_digest: str | None = None
    quality_target_count: int | None = None

    def initializer(self) -> Initializer:
        return task_initializer(self.task)


@dataclass(frozen=True)
class OpenedFileIdentity:
    """One file actually opened while authorizing a partition target view."""

    authority_root: str
    role: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class TargetAccessReceipt:
    """Auditable proof of exactly one opened evaluator-target partition."""

    partition: str
    prepared_manifest_record_digest: str
    prepared_manifest_sha256: str
    target_authority_record_digest: str
    target_path: str
    target_sha256: str
    target_set_digest: str
    target_count: int
    publisher_id: str
    publisher_attestation_digest: str
    evidence_manifest_record_digest: str
    evidence_manifest_sha256: str
    opened_files: tuple[OpenedFileIdentity, ...]
    record_digest: str

    def recompute_record_digest(self) -> str:
        """Recompute the canonical identity of this target-access capability."""

        return content_digest(self._validated_payload())

    def as_dict(self) -> dict[str, Any]:
        """Return a validated canonical record suitable for downstream receipts.

        Callers must bind this complete record, or at minimum its ``record_digest``.  The
        method intentionally revalidates every field and recomputes the digest so a forged
        dataclass instance cannot silently cross a CLI or report boundary.
        """

        payload = self._validated_payload()
        observed = content_digest(payload)
        if self.record_digest != observed:
            raise ValueError("target-access receipt record digest mismatch")
        return {**payload, "record_digest": observed}

    def _validated_payload(self) -> dict[str, Any]:
        if self.partition not in {"train", "val", "test"}:
            raise ValueError("target-access receipt partition is invalid")
        digest_fields = {
            "prepared manifest": self.prepared_manifest_record_digest,
            "prepared manifest file": self.prepared_manifest_sha256,
            "target authority": self.target_authority_record_digest,
            "target file": self.target_sha256,
            "target set": self.target_set_digest,
            "publisher attestation": self.publisher_attestation_digest,
            "evidence manifest": self.evidence_manifest_record_digest,
            "evidence manifest file": self.evidence_manifest_sha256,
        }
        for name, digest in digest_fields.items():
            if not _is_sha256(digest):
                raise ValueError(f"target-access {name} digest is invalid")
        if not isinstance(self.target_count, int) or isinstance(self.target_count, bool):
            raise ValueError("target-access target count must be an integer")
        if self.target_count <= 0:
            raise ValueError("target-access target count must be positive")
        if not isinstance(self.publisher_id, str) or not self.publisher_id:
            raise ValueError("target-access publisher ID must be nonempty text")
        expected_target_path = f"targets/{self.partition}.jsonl"
        if self.target_path != expected_target_path:
            raise ValueError("target-access target path crosses its selected partition")
        if not self.opened_files:
            raise ValueError("target-access receipt must identify opened files")
        opened: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in self.opened_files:
            if not isinstance(item, OpenedFileIdentity):
                raise ValueError("target-access opened-file identity has the wrong type")
            if item.authority_root not in {"prepared", "publisher"}:
                raise ValueError("target-access opened-file authority root is invalid")
            if not isinstance(item.role, str) or not item.role:
                raise ValueError("target-access opened-file role must be nonempty text")
            if not isinstance(item.relative_path, str) or not item.relative_path:
                raise ValueError("target-access opened-file path must be nonempty text")
            relative = Path(item.relative_path)
            if (
                relative.is_absolute()
                or "\\" in item.relative_path
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError("target-access opened-file path must be safely relative")
            if not _is_sha256(item.sha256):
                raise ValueError("target-access opened-file SHA-256 is invalid")
            identity = (item.authority_root, item.role, item.relative_path)
            if identity in seen:
                raise ValueError("target-access receipt repeats an opened-file identity")
            seen.add(identity)
            opened.append(asdict(item))
        expected_target_open = {
            ("prepared", "evaluator-targets", expected_target_path, self.target_sha256)
        }
        observed_target_open = {
            (item.authority_root, item.role, item.relative_path, item.sha256)
            for item in self.opened_files
            if item.authority_root == "prepared"
        }
        if observed_target_open != expected_target_open:
            raise ValueError("target-access receipt has the wrong prepared target opening")
        if not any(
            item.authority_root == "publisher"
            and item.role == "quality-evidence-manifest"
            and item.sha256 == self.evidence_manifest_sha256
            for item in self.opened_files
        ):
            raise ValueError("target-access receipt omits its evidence-manifest opening")
        return {
            "evidence_manifest_record_digest": self.evidence_manifest_record_digest,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "opened_files": opened,
            "partition": self.partition,
            "prepared_manifest_record_digest": self.prepared_manifest_record_digest,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "publisher_attestation_digest": self.publisher_attestation_digest,
            "publisher_id": self.publisher_id,
            "target_authority_record_digest": self.target_authority_record_digest,
            "target_count": self.target_count,
            "target_path": self.target_path,
            "target_set_digest": self.target_set_digest,
            "target_sha256": self.target_sha256,
        }


@dataclass(frozen=True)
class PreparedPartitionLoad:
    """Prepared tasks and the non-discardable receipt for any target access."""

    tasks: tuple[PreparedTask, ...]
    target_access: TargetAccessReceipt | None


def _strict_object(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite number {token}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    _check_finite(value, name)
    return value


def _check_finite(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item, name)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item, name)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    return _strict_object(path.read_bytes(), name)


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{name} line {line_number} is blank")
        rows.append(_strict_object(line, f"{name} line {line_number}"))
    return rows


def _exact_keys(record: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(record) != expected:
        raise ValueError(
            f"{name} schema differs: missing={sorted(expected - set(record))}, "
            f"unknown={sorted(set(record) - expected)}"
        )


def _verify_record(record: Mapping[str, object], name: str) -> None:
    digest = record.get("record_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{name} has an invalid record digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if digest != content_digest(payload):
        raise ValueError(f"{name} record digest mismatch")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _verify_output_checksum(
    directory: Path,
    outputs: Mapping[str, object],
    filename: str,
) -> None:
    receipt = outputs[filename]
    if not isinstance(receipt, dict) or set(receipt) != {"records", "sha256"}:
        raise ValueError(f"invalid output receipt for {filename}")
    observed = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
    if receipt["sha256"] != observed:
        raise ValueError(f"prepared output checksum mismatch for {filename}")


def _nonnegative_count(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _registered_strings(value: object, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{name} must be a sorted nonempty set of strings")
    return value


def _open_probability(value: object, name: str, *, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < upper:
        raise ValueError(f"{name} must lie strictly between zero and {upper}")
    return result


def _validate_receipt_filter(
    raw: object,
    *,
    axis_values: Mapping[str, Sequence[str]],
    name: str,
) -> dict[str, frozenset[str]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    _exact_keys(raw, set(_STRATUM_FIELDS), name)
    result: dict[str, frozenset[str]] = {}
    for field in _STRATUM_FIELDS:
        values = _registered_strings(raw[field], f"{name} {field}")
        unknown = sorted(set(values) - set(axis_values[field]))
        if unknown:
            raise ValueError(f"{name} {field} contains unregistered values: {unknown}")
        result[field] = frozenset(values)
    return result


def _matched_base_lineages(
    registry: Sequence[Mapping[str, object]],
    *,
    partition: str,
    design_filter: Mapping[str, frozenset[str]],
) -> set[str]:
    return {
        str(row["base_lineage_key"])
        for row in registry
        if row["learning_partition"] == partition
        and all(row[field] in design_filter[field] for field in _STRATUM_FIELDS)
    }


def _realized_partition_filter(
    registry: Sequence[Mapping[str, object]],
    *,
    partition: str,
) -> dict[str, frozenset[str]]:
    return {
        field: frozenset(
            str(row[field]) for row in registry if row["learning_partition"] == partition
        )
        for field in _STRATUM_FIELDS
    }


def _validate_pass_results(
    value: object,
    *,
    identity_field: str,
    label: str,
    calculated: bool,
) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"prepared corpus-design {label} results must be nonempty")
    expected = {
        "actual_base_lineages",
        identity_field,
        "minimum_base_lineages",
        "status",
    }
    if calculated:
        expected.add("calculated_minimum_base_lineages")
    identities: list[str] = []
    results: dict[str, Mapping[str, object]] = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"prepared corpus-design {label} results must be objects")
        _exact_keys(row, expected, f"prepared corpus-design {label} result")
        identity = _require_text(row[identity_field], f"prepared {label} result identity")
        identities.append(identity)
        results[identity] = row
        actual = _nonnegative_count(
            row["actual_base_lineages"], f"prepared {label} actual base lineages"
        )
        minimum = _nonnegative_count(
            row["minimum_base_lineages"],
            f"prepared {label} minimum base lineages",
            positive=True,
        )
        if actual < minimum or row["status"] != "pass":
            raise ValueError(f"prepared corpus-design {label} result is not passing")
        if calculated:
            required = _nonnegative_count(
                row["calculated_minimum_base_lineages"],
                f"prepared {label} calculated minimum",
                positive=True,
            )
            if minimum < required:
                raise ValueError(f"prepared corpus-design {label} target is underplanned")
    if identities != sorted(set(identities)):
        raise ValueError(f"prepared corpus-design {label} results are not canonical")
    return results


def _validate_embedded_stratum_quotas(
    raw: object,
    *,
    axis_values: Mapping[str, Sequence[str]],
    registry: Sequence[Mapping[str, object]],
    results: Mapping[str, Mapping[str, object]],
) -> None:
    if not isinstance(raw, list) or not raw:
        raise ValueError("prepared corpus-design stratum quotas must be nonempty")
    quota_ids: list[str] = []
    covered: set[str] = set()
    has_ood_test_quota = False
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("prepared corpus-design stratum quotas must be objects")
        _exact_keys(
            row,
            {"filter", "learning_partition", "minimum_base_lineages", "quota_id"},
            "prepared corpus-design stratum quota",
        )
        quota_id = _require_text(row["quota_id"], "prepared stratum quota identity")
        quota_ids.append(quota_id)
        partition = row["learning_partition"]
        if partition not in {"train", "val", "test"}:
            raise ValueError("prepared stratum quota has an unknown partition")
        minimum = _nonnegative_count(
            row["minimum_base_lineages"],
            "prepared stratum quota minimum",
            positive=True,
        )
        design_filter = _validate_receipt_filter(
            row["filter"],
            axis_values=axis_values,
            name="prepared stratum quota filter",
        )
        matched = _matched_base_lineages(
            registry,
            partition=partition,
            design_filter=design_filter,
        )
        if len(matched) < minimum:
            raise ValueError("prepared stratum quota is not met by unique base lineages")
        covered.update(matched)
        if partition == "test" and design_filter["distribution_regime"] == {"ood"}:
            has_ood_test_quota = True
        result = results.get(quota_id)
        if (
            result is None
            or result["minimum_base_lineages"] != minimum
            or result["actual_base_lineages"] != len(matched)
        ):
            raise ValueError("prepared stratum quota disagrees with its validation result")
    if quota_ids != sorted(set(quota_ids)) or set(quota_ids) != set(results):
        raise ValueError("prepared stratum quota registry is not canonical and complete")
    all_lineages = {str(row["base_lineage_key"]) for row in registry}
    if covered != all_lineages:
        raise ValueError("prepared stratum quotas do not cover every base lineage")
    if not has_ood_test_quota:
        raise ValueError("prepared corpus design lacks a dedicated sealed-test OOD quota")


def _validate_embedded_power_targets(
    raw: object,
    *,
    axis_values: Mapping[str, Sequence[str]],
    registry: Sequence[Mapping[str, object]],
    results: Mapping[str, Mapping[str, object]],
) -> None:
    if not isinstance(raw, list) or not raw:
        raise ValueError("prepared corpus-design power targets must be nonempty")
    expected_keys = {
        "alpha",
        "alternative",
        "assumed_discordance",
        "assumed_true_difference",
        "endpoint",
        "filter",
        "learning_partition",
        "method",
        "minimum_base_lineages",
        "noninferiority_margin",
        "power_separation",
        "target_id",
        "target_power",
    }
    target_ids: list[str] = []
    has_test_target = False
    has_primary_test_target = False
    full_test_filter = _realized_partition_filter(registry, partition="test")
    full_validation_filter = _realized_partition_filter(registry, partition="val")
    has_validation_tuning_target = False
    validation_target_count = 0
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("prepared corpus-design power targets must be objects")
        _exact_keys(row, expected_keys, "prepared corpus-design power target")
        target_id = _require_text(row["target_id"], "prepared power target identity")
        target_ids.append(target_id)
        _require_text(row["endpoint"], "prepared power target endpoint")
        partition = row["learning_partition"]
        if partition not in {"train", "val", "test"}:
            raise ValueError("prepared power target has an unknown partition")
        has_test_target |= partition == "test"
        if row["method"] != "paired-binary-normal-approximation":
            raise ValueError("prepared power target has an unsupported method")
        alternative = row["alternative"]
        if alternative not in {"one-sided-noninferiority", "two-sided-difference"}:
            raise ValueError("prepared power target has an unsupported alternative")
        alpha = _open_probability(row["alpha"], "prepared power alpha", upper=0.5)
        target_power = _open_probability(row["target_power"], "prepared target power")
        if target_power <= 0.5:
            raise ValueError("prepared target power must exceed one half")
        assumed_true_difference = row["assumed_true_difference"]
        if (
            isinstance(assumed_true_difference, bool)
            or not isinstance(assumed_true_difference, (int, float))
            or not math.isfinite(float(assumed_true_difference))
            or not -1.0 < float(assumed_true_difference) < 1.0
        ):
            raise ValueError("prepared assumed true difference must lie between -1 and 1")
        assumed_true_difference = float(assumed_true_difference)
        noninferiority_margin = row["noninferiority_margin"]
        if isinstance(noninferiority_margin, bool) or not isinstance(
            noninferiority_margin, (int, float)
        ):
            raise ValueError("prepared noninferiority margin must be numeric")
        noninferiority_margin = float(noninferiority_margin)
        if alternative == "one-sided-noninferiority":
            if not 0.0 < noninferiority_margin < 1.0:
                raise ValueError("prepared noninferiority power has an invalid margin")
            expected_separation = assumed_true_difference + noninferiority_margin
        else:
            if noninferiority_margin != 0.0:
                raise ValueError("prepared two-sided power has a nonzero NI margin")
            expected_separation = abs(assumed_true_difference)
        separation = _open_probability(
            row["power_separation"],
            "prepared power separation",
        )
        if not math.isclose(separation, expected_separation, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("prepared power target has an inconsistent boundary separation")
        discordance = _open_probability(
            row["assumed_discordance"],
            "prepared assumed discordance",
        )
        alpha_quantile = (
            1.0 - alpha
            if alternative == "one-sided-noninferiority"
            else 1.0 - alpha / 2.0
        )
        calculated = max(
            1,
            math.ceil(
                discordance
                * (
                    NormalDist().inv_cdf(alpha_quantile)
                    + NormalDist().inv_cdf(target_power)
                )
                ** 2
                / separation**2
            ),
        )
        minimum = _nonnegative_count(
            row["minimum_base_lineages"],
            "prepared power target minimum",
            positive=True,
        )
        if minimum < calculated:
            raise ValueError("prepared power target is below its registered calculation")
        design_filter = _validate_receipt_filter(
            row["filter"],
            axis_values=axis_values,
            name="prepared power target filter",
        )
        has_primary_test_target |= (
            partition == "test"
            and row["endpoint"] == "valid-return-noninferiority"
            and alternative == "one-sided-noninferiority"
            and design_filter == full_test_filter
        )
        if partition == "val":
            validation_target_count += 1
            has_validation_tuning_target |= (
                target_id == VALIDATION_TUNING_POWER_TARGET_ID
                and row["endpoint"] == "valid-return-noninferiority"
                and alternative == "one-sided-noninferiority"
                and alpha <= VALIDATION_TUNING_FAMILYWISE_ALPHA
                and noninferiority_margin == VALIDATION_TUNING_NONINFERIORITY_MARGIN
                and minimum >= MINIMUM_VALIDATION_TUNING_BASE_LINEAGES
                and design_filter == full_validation_filter
            )
        actual = len(
            _matched_base_lineages(
                registry,
                partition=partition,
                design_filter=design_filter,
            )
        )
        result = results.get(target_id)
        if (
            actual < minimum
            or result is None
            or result["minimum_base_lineages"] != minimum
            or result["actual_base_lineages"] != actual
            or result["calculated_minimum_base_lineages"] != calculated
        ):
            raise ValueError("prepared power target disagrees with its validation result")
    if target_ids != sorted(set(target_ids)) or set(target_ids) != set(results):
        raise ValueError("prepared power target registry is not canonical and complete")
    if not has_test_target:
        raise ValueError("prepared corpus design lacks a sealed-test power target")
    if not has_primary_test_target:
        raise ValueError(
            "prepared valid-return noninferiority power target does not cover the full "
            "sealed-test population"
        )
    if validation_target_count != 1 or not has_validation_tuning_target:
        raise ValueError(
            "prepared corpus design lacks exactly one typed validation-tuning power target "
            "over the full validation population"
        )


def _validate_embedded_precision_targets(
    raw: object,
    *,
    axis_values: Mapping[str, Sequence[str]],
    registry: Sequence[Mapping[str, object]],
    results: Mapping[str, Mapping[str, object]],
) -> None:
    if not isinstance(raw, list) or not raw:
        raise ValueError("prepared corpus-design precision targets must be nonempty")
    expected_keys = {
        "confidence_level",
        "endpoint",
        "filter",
        "half_width",
        "learning_partition",
        "method",
        "minimum_base_lineages",
        "outcome_bounds",
        "target_id",
        "variance_bound",
    }
    target_ids: list[str] = []
    has_test_target = False
    has_primary_test_target = False
    full_test_filter = _realized_partition_filter(registry, partition="test")
    full_validation_filter = _realized_partition_filter(registry, partition="val")
    has_validation_tuning_target = False
    validation_paired_target_count = 0
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("prepared corpus-design precision targets must be objects")
        _exact_keys(row, expected_keys, "prepared corpus-design precision target")
        target_id = _require_text(row["target_id"], "prepared precision target identity")
        target_ids.append(target_id)
        _require_text(row["endpoint"], "prepared precision target endpoint")
        partition = row["learning_partition"]
        if partition not in {"train", "val", "test"}:
            raise ValueError("prepared precision target has an unknown partition")
        has_test_target |= partition == "test"
        method = row["method"]
        if method not in {
            "bounded-mean-worst-case-normal",
            "bounded-paired-difference-worst-case-normal",
        }:
            raise ValueError("prepared precision target has an unsupported method")
        bounds = row["outcome_bounds"]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in bounds
            )
        ):
            raise ValueError("prepared precision outcome bounds are invalid")
        lower, upper = (float(value) for value in bounds)
        expected_bounds = (
            (0.0, 1.0)
            if method == "bounded-mean-worst-case-normal"
            else (-1.0, 1.0)
        )
        if (lower, upper) != expected_bounds:
            raise ValueError("prepared precision outcome bounds differ from the method")
        variance_bound = row["variance_bound"]
        if isinstance(variance_bound, bool) or not isinstance(variance_bound, (int, float)):
            raise ValueError("prepared precision variance bound must be numeric")
        variance_bound = float(variance_bound)
        if not math.isclose(
            variance_bound,
            ((upper - lower) / 2.0) ** 2,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("prepared precision variance bound is not worst-case")
        confidence = _open_probability(
            row["confidence_level"],
            "prepared precision confidence",
        )
        if confidence <= 0.5:
            raise ValueError("prepared precision confidence must exceed one half")
        half_width = _open_probability(row["half_width"], "prepared precision half width")
        z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        calculated = max(1, math.ceil(variance_bound * z_value**2 / half_width**2))
        minimum = _nonnegative_count(
            row["minimum_base_lineages"],
            "prepared precision target minimum",
            positive=True,
        )
        if minimum < calculated:
            raise ValueError("prepared precision target is below its registered calculation")
        design_filter = _validate_receipt_filter(
            row["filter"],
            axis_values=axis_values,
            name="prepared precision target filter",
        )
        has_primary_test_target |= (
            partition == "test"
            and method == "bounded-paired-difference-worst-case-normal"
            and row["endpoint"] == "learned-minus-stock-unconditional-if-q3-s0"
            and design_filter == full_test_filter
        )
        if (
            partition == "val"
            and method == "bounded-paired-difference-worst-case-normal"
            and row["endpoint"] == "learned-minus-stock-unconditional-if-q3-s0"
        ):
            validation_paired_target_count += 1
            has_validation_tuning_target |= (
                target_id == VALIDATION_TUNING_PRECISION_TARGET_ID
                and confidence >= VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL
                and half_width <= VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH
                and minimum >= MINIMUM_VALIDATION_TUNING_BASE_LINEAGES
                and design_filter == full_validation_filter
            )
        actual = len(
            _matched_base_lineages(
                registry,
                partition=partition,
                design_filter=design_filter,
            )
        )
        result = results.get(target_id)
        if (
            actual < minimum
            or result is None
            or result["minimum_base_lineages"] != minimum
            or result["actual_base_lineages"] != actual
            or result["calculated_minimum_base_lineages"] != calculated
        ):
            raise ValueError("prepared precision target disagrees with its validation result")
    if target_ids != sorted(set(target_ids)) or set(target_ids) != set(results):
        raise ValueError("prepared precision target registry is not canonical and complete")
    if not has_test_target:
        raise ValueError("prepared corpus design lacks a sealed-test precision target")
    if not has_primary_test_target:
        raise ValueError(
            "prepared paired IF-Q3-S0 precision target does not cover the full sealed-test "
            "population"
        )
    if validation_paired_target_count != 1 or not has_validation_tuning_target:
        raise ValueError(
            "prepared corpus design lacks exactly one typed validation-tuning precision target "
            "over the full validation population"
        )


def _validate_corpus_design_receipt(
    raw: object,
    source_sha256: Mapping[str, object],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("prepared corpus_design must be an object")
    receipt_version = raw.get("schema_version")
    expected_fields = {
            "axis_values",
            "condition_registry",
            "corpus_design_version",
            "difficulty_calibration",
            "independent_unit",
            "manifest_record_digest",
            "manifest_sha256",
            "minimum_partition_base_lineages",
            "partition_quotas",
            "power_targets",
            "precision_targets",
            "realized_census",
            "schema",
            "schema_version",
            "stratum_quotas",
            "validation",
    }
    if receipt_version == 2:
        expected_fields.add("stratum_artifacts")
    _exact_keys(raw, expected_fields, "prepared corpus-design receipt")
    if (
        raw["schema"] != PREPARED_CORPUS_DESIGN_RECEIPT_SCHEMA
        or receipt_version not in {1, 2}
    ):
        raise ValueError("unsupported prepared corpus-design receipt schema")
    expected_semantic_version = (
        REGISTERED_CORPUS_DESIGN_VERSION
        if receipt_version == 1
        else REGISTERED_CORPUS_DESIGN_VERSION_V2
    )
    if raw["corpus_design_version"] != expected_semantic_version:
        raise ValueError("prepared corpus-design receipt has the wrong semantic version")
    if raw["independent_unit"] != "immutable-base-lineage":
        raise ValueError("prepared corpus-design receipt has the wrong independent unit")
    axes = raw["axis_values"]
    if not isinstance(axes, dict):
        raise ValueError("prepared corpus-design axis_values must be an object")
    _exact_keys(axes, set(_STRATUM_FIELDS), "prepared corpus-design axis_values")
    axis_values = {
        field: _registered_strings(axes[field], f"prepared corpus-design axis {field}")
        for field in _STRATUM_FIELDS
    }
    if any(value not in {"none", "faulted"} for value in axis_values["fault_status"]):
        raise ValueError("prepared corpus-design fault-status axis is invalid")
    if any(
        value not in {"application-derived", "synthetic"}
        for value in axis_values["problem_origin"]
    ):
        raise ValueError("prepared corpus-design problem-origin axis is invalid")
    if any(value not in {"iid", "ood"} for value in axis_values["distribution_regime"]):
        raise ValueError("prepared corpus-design distribution-regime axis is invalid")
    if any(
        value not in {"not_applicable", "recorded"}
        for value in axis_values["calibration_status"]
    ):
        raise ValueError("prepared corpus-design calibration-status axis is invalid")
    difficulty = raw["difficulty_calibration"]
    if not isinstance(difficulty, dict):
        raise ValueError("prepared difficulty calibration receipt must be an object")
    if receipt_version == 1:
        difficulty_fields = {
            "authority_sha256",
            "budget_sha256",
            "evidence_sha256",
            "outcome_blind",
            "panel_sha256",
            "protocol_sha256",
        }
        _exact_keys(difficulty, difficulty_fields, "prepared difficulty calibration receipt")
        for field in difficulty_fields - {"outcome_blind"}:
            _require_sha256(difficulty[field], f"prepared difficulty calibration {field}")
    else:
        artifact_names = {"authority", "budget", "evidence", "origin", "panel", "protocol"}
        _exact_keys(
            difficulty,
            {*artifact_names, "outcome_blind", "publisher_id"},
            "prepared evidence-backed stratum authority",
        )
        _require_text(difficulty["publisher_id"], "prepared stratum publisher ID")
        artifacts = raw["stratum_artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError("prepared stratum artifact registry must be an object")
        _exact_keys(artifacts, artifact_names, "prepared stratum artifact registry")
        for name in artifact_names:
            descriptor = difficulty[name]
            identity = artifacts[name]
            if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
                raise ValueError("prepared stratum artifact descriptor is invalid")
            if not isinstance(identity, dict) or set(identity) != {
                "path",
                "record_digest",
                "sha256",
            }:
                raise ValueError("prepared stratum artifact identity is invalid")
            if descriptor != {"path": identity["path"], "sha256": identity["sha256"]}:
                raise ValueError("prepared stratum artifact descriptor differs from its identity")
            _require_text(identity["path"], f"prepared {name} artifact path")
            _require_sha256(identity["sha256"], f"prepared {name} artifact SHA-256")
            _require_sha256(
                identity["record_digest"], f"prepared {name} artifact record digest"
            )
    if difficulty["outcome_blind"] is not True:
        raise ValueError("prepared difficulty calibration is not outcome-blind")
    _require_sha256(raw["manifest_record_digest"], "corpus-design record digest")
    manifest_sha256 = _require_sha256(raw["manifest_sha256"], "corpus-design file digest")
    if manifest_sha256 != source_sha256.get("corpus_design_manifest"):
        raise ValueError("prepared corpus-design receipt disagrees with its source registry")

    census = raw["realized_census"]
    if not isinstance(census, dict):
        raise ValueError("prepared realized corpus census must be an object")
    _exact_keys(
        census,
        {
            "base_lineage_set_digest",
            "base_lineages_by_partition",
            "by_axis",
            "by_partition",
            "by_stratum",
            "independent_unit",
            "lineage_registry_digest",
            "total_base_lineages",
            "total_realized_conditions",
            "total_tasks",
        },
        "prepared realized corpus census",
    )
    if census["independent_unit"] != "immutable-base-lineage":
        raise ValueError("prepared corpus census has the wrong independent unit")
    _require_sha256(census["base_lineage_set_digest"], "base-lineage set digest")
    _require_sha256(census["lineage_registry_digest"], "lineage registry digest")
    total_lineages = _nonnegative_count(
        census["total_base_lineages"], "total base lineages", positive=True
    )
    total_conditions = _nonnegative_count(
        census["total_realized_conditions"], "total realized conditions", positive=True
    )
    if total_conditions < total_lineages:
        raise ValueError("prepared corpus has fewer realized conditions than base lineages")
    total_tasks = _nonnegative_count(census["total_tasks"], "total corpus tasks", positive=True)
    by_partition = census["by_partition"]
    lineages_by_partition = census["base_lineages_by_partition"]
    if not isinstance(by_partition, dict) or set(by_partition) != {"train", "val", "test"}:
        raise ValueError("prepared corpus partition census differs from the schema")
    if (
        not isinstance(lineages_by_partition, dict)
        or set(lineages_by_partition) != {"train", "val", "test"}
    ):
        raise ValueError("prepared corpus lineage partition census differs from the schema")
    all_lineages: list[str] = []
    summed_tasks = 0
    for partition in ("train", "val", "test"):
        rows = lineages_by_partition[partition]
        if (
            not isinstance(rows, list)
            or any(not isinstance(item, str) or not item for item in rows)
            or rows != sorted(set(rows))
        ):
            raise ValueError("prepared base-lineage partition rows are not canonical")
        all_lineages.extend(rows)
        counts = by_partition[partition]
        if not isinstance(counts, dict) or set(counts) != {"base_lineages", "tasks"}:
            raise ValueError("prepared per-partition census differs from the schema")
        if _nonnegative_count(
            counts["base_lineages"], "partition base lineages", positive=True
        ) != len(rows):
            raise ValueError("prepared per-partition lineage count is inconsistent")
        summed_tasks += _nonnegative_count(counts["tasks"], "partition tasks", positive=True)
    if len(all_lineages) != len(set(all_lineages)) or len(all_lineages) != total_lineages:
        raise ValueError("prepared base lineages overlap partitions or differ from their total")
    if census["base_lineage_set_digest"] != content_digest(sorted(all_lineages)):
        raise ValueError("prepared base-lineage set digest is inconsistent")
    if summed_tasks != total_tasks:
        raise ValueError("prepared per-partition task counts differ from their total")
    partition_quotas = raw["partition_quotas"]
    if not isinstance(partition_quotas, dict):
        raise ValueError("prepared corpus-design partition quotas must be an object")
    _exact_keys(
        partition_quotas,
        {"train", "val", "test"},
        "prepared corpus-design partition quotas",
    )
    for partition in ("train", "val", "test"):
        quota = _nonnegative_count(
            partition_quotas[partition],
            f"prepared {partition} partition quota",
            positive=True,
        )
        if quota != by_partition[partition]["base_lineages"]:
            raise ValueError("prepared partition quota differs from the realized census")
    raw_partition_floors = raw["minimum_partition_base_lineages"]
    if not isinstance(raw_partition_floors, dict):
        raise ValueError("prepared minimum partition base lineages must be an object")
    _exact_keys(
        raw_partition_floors,
        {"train", "val", "test"},
        "prepared minimum partition base lineages",
    )
    production_floors = {
        "test": MINIMUM_TEST_BASE_LINEAGES,
        "train": MINIMUM_TRAIN_BASE_LINEAGES,
        "val": MINIMUM_VALIDATION_BASE_LINEAGES,
    }
    partition_floors: dict[str, int] = {}
    for partition in ("train", "val", "test"):
        floor = _nonnegative_count(
            raw_partition_floors[partition],
            f"prepared {partition} minimum base lineages",
            positive=True,
        )
        if floor < production_floors[partition]:
            raise ValueError(
                f"prepared {partition} minimum is below the production independent-lineage floor"
            )
        if partition_quotas[partition] < floor:
            raise ValueError(
                f"prepared {partition} quota is below its registered independent-lineage floor"
            )
        partition_floors[partition] = floor

    by_axis = census["by_axis"]
    if not isinstance(by_axis, dict) or set(by_axis) != set(_STRATUM_FIELDS):
        raise ValueError("prepared corpus axis census differs from the schema")
    for field, counts in by_axis.items():
        if not isinstance(counts, dict) or not counts:
            raise ValueError(f"prepared {field} census must be a nonempty object")
        if any(not isinstance(value, str) or not value for value in counts):
            raise ValueError(f"prepared {field} census contains an invalid value")
        validated_counts = [
            _nonnegative_count(count, f"prepared {field} count", positive=True)
            for count in counts.values()
        ]
        if any(count > total_lineages for count in validated_counts):
            raise ValueError(f"prepared {field} census exceeds the lineage total")
        if sum(validated_counts) < total_lineages:
            raise ValueError(f"prepared {field} census omits base lineages")
    if len(by_axis["host_family"]) < 2:
        raise ValueError("prepared scientific census contains only one host family")
    if set(by_axis["problem_origin"]) != {"application-derived", "synthetic"}:
        raise ValueError(
            "prepared scientific census lacks application-derived or synthetic origins"
        )
    if "faulted" not in by_axis["fault_status"]:
        raise ValueError("prepared scientific census lacks a realized faulted condition")
    for field in (
        "embedding_difficulty",
        "sampling_difficulty",
        "decision_difficulty",
    ):
        if all(value.casefold() == "easy" for value in by_axis[field]):
            raise ValueError(f"prepared scientific census is all-easy on {field}")

    condition_registry = raw["condition_registry"]
    if not isinstance(condition_registry, list) or not condition_registry:
        raise ValueError("prepared condition registry must be a nonempty list")
    condition_sort_keys: list[tuple[object, ...]] = []
    structural_keys: set[tuple[object, ...]] = set()
    registry_lineages = {partition: set() for partition in ("train", "val", "test")}
    registry_axis_members: dict[str, dict[str, set[str]]] = {
        field: {} for field in _STRATUM_FIELDS
    }
    lineage_partition: dict[str, str] = {}
    observed_fields = (
        "application_family",
        "host_family",
        "fault_status",
        "distribution_regime",
        "calibration_status",
    )
    for row in condition_registry:
        if not isinstance(row, dict):
            raise ValueError("prepared condition registry rows must be objects")
        _exact_keys(row, _DESIGN_CONDITION_FIELDS, "prepared condition registry row")
        base = _require_text(row["base_lineage_key"], "prepared condition base lineage")
        partition = row["learning_partition"]
        if partition not in {"train", "val", "test"}:
            raise ValueError("prepared condition registry has an unknown partition")
        if lineage_partition.setdefault(base, partition) != partition:
            raise ValueError("one prepared condition lineage crosses partitions")
        registry_lineages[partition].add(base)
        for field in _STRATUM_FIELDS:
            value = _require_text(row[field], f"prepared condition {field}")
            if value not in axis_values[field]:
                raise ValueError(f"prepared condition {field} is not registered")
            registry_axis_members[field].setdefault(value, set()).add(base)
        calibration_sha256 = row["calibration_sha256"]
        if row["calibration_status"] == "not_applicable":
            if calibration_sha256 is not None:
                raise ValueError("not-applicable prepared condition carries calibration data")
        else:
            _require_sha256(calibration_sha256, "prepared condition calibration digest")
        structural_key = (
            base,
            partition,
            *(row[field] for field in observed_fields),
            calibration_sha256,
        )
        if structural_key in structural_keys:
            raise ValueError("prepared condition registry repeats one realized condition")
        structural_keys.add(structural_key)
        condition_sort_keys.append(
            (
                base,
                partition,
                *(row[field] for field in _STRATUM_FIELDS),
                "" if calibration_sha256 is None else calibration_sha256,
            )
        )
    if condition_sort_keys != sorted(condition_sort_keys):
        raise ValueError("prepared condition registry is not canonical")
    if content_digest(condition_registry) != census["lineage_registry_digest"]:
        raise ValueError("prepared condition registry digest is inconsistent")
    if len(condition_registry) != total_conditions:
        raise ValueError("prepared condition registry count is inconsistent")
    if {
        partition: sorted(values) for partition, values in registry_lineages.items()
    } != lineages_by_partition:
        raise ValueError("prepared condition registry has the wrong lineage census")
    for field, value_members in registry_axis_members.items():
        if set(value_members) != set(axis_values[field]):
            raise ValueError(f"prepared corpus-design axis {field} is not exactly realized")
        expected = {value: len(members) for value, members in sorted(value_members.items())}
        if expected != by_axis[field]:
            raise ValueError(f"prepared condition registry has the wrong {field} census")
    origins_by_family: dict[str, set[str]] = {}
    for row in condition_registry:
        origins_by_family.setdefault(row["application_family"], set()).add(
            row["problem_origin"]
        )
    if any(len(origins) != 1 for origins in origins_by_family.values()):
        raise ValueError("one prepared application family maps to multiple problem origins")

    by_stratum = census["by_stratum"]
    if not isinstance(by_stratum, list) or not by_stratum:
        raise ValueError("prepared corpus stratum census must be nonempty")
    stratum_keys = {"base_lineages", "learning_partition", "tasks", *_STRATUM_FIELDS}
    stratum_total = 0
    stratum_tasks = 0
    canonical_keys: list[tuple[str, ...]] = []
    for row in by_stratum:
        if not isinstance(row, dict):
            raise ValueError("prepared corpus stratum census rows must be objects")
        _exact_keys(row, stratum_keys, "prepared corpus stratum census row")
        partition = row["learning_partition"]
        if partition not in {"train", "val", "test"}:
            raise ValueError("prepared corpus stratum has an unknown partition")
        values = tuple(
            _require_text(row[field], f"prepared stratum {field}") for field in _STRATUM_FIELDS
        )
        canonical_keys.append((partition, *values))
        stratum_total += _nonnegative_count(
            row["base_lineages"], "prepared stratum base lineages", positive=True
        )
        stratum_tasks += _nonnegative_count(row["tasks"], "prepared stratum tasks", positive=True)
    if canonical_keys != sorted(set(canonical_keys)):
        raise ValueError("prepared corpus stratum census is not canonical")
    if stratum_total < total_lineages or stratum_tasks != total_tasks:
        raise ValueError("prepared corpus stratum census differs from its totals")

    validation = raw["validation"]
    if not isinstance(validation, dict):
        raise ValueError("prepared corpus-design validation must be an object")
    _exact_keys(
        validation,
        {
            "difficulty_diversity_status",
            "host_family_diversity_status",
            "ood_quota_status",
            "power_results",
            "power_status",
            "precision_results",
            "precision_status",
            "quota_results",
            "quota_status",
            "status",
        },
        "prepared corpus-design validation",
    )
    status_fields = {
        "difficulty_diversity_status",
        "host_family_diversity_status",
        "ood_quota_status",
        "power_status",
        "precision_status",
        "quota_status",
        "status",
    }
    if any(validation[field] != "pass" for field in status_fields):
        raise ValueError("prepared corpus-design validation is not passing")
    quota_results = _validate_pass_results(
        validation["quota_results"],
        identity_field="quota_id",
        label="quota",
        calculated=False,
    )
    power_results = _validate_pass_results(
        validation["power_results"],
        identity_field="target_id",
        label="power",
        calculated=True,
    )
    precision_results = _validate_pass_results(
        validation["precision_results"],
        identity_field="target_id",
        label="precision",
        calculated=True,
    )
    _validate_embedded_stratum_quotas(
        raw["stratum_quotas"],
        axis_values=axis_values,
        registry=condition_registry,
        results=quota_results,
    )
    _validate_embedded_power_targets(
        raw["power_targets"],
        axis_values=axis_values,
        registry=condition_registry,
        results=power_results,
    )
    _validate_embedded_precision_targets(
        raw["precision_targets"],
        axis_values=axis_values,
        registry=condition_registry,
        results=precision_results,
    )
    confirmatory_test_minima = [
        row["minimum_base_lineages"]
        for row in raw["power_targets"]
        if row["learning_partition"] == "test"
        and row["endpoint"] == "valid-return-noninferiority"
        and row["alternative"] == "one-sided-noninferiority"
    ] + [
        row["minimum_base_lineages"]
        for row in raw["precision_targets"]
        if row["learning_partition"] == "test"
        and row["endpoint"] == "learned-minus-stock-unconditional-if-q3-s0"
        and row["method"] == "bounded-paired-difference-worst-case-normal"
    ]
    confirmatory_test_minimum = max(confirmatory_test_minima)
    if partition_floors["test"] < confirmatory_test_minimum:
        raise ValueError(
            "prepared test independent-lineage floor is below a confirmatory target"
        )
    validation_tuning_minimum = max(
        next(
            row["minimum_base_lineages"]
            for row in raw["power_targets"]
            if row["target_id"] == VALIDATION_TUNING_POWER_TARGET_ID
        ),
        next(
            row["minimum_base_lineages"]
            for row in raw["precision_targets"]
            if row["target_id"] == VALIDATION_TUNING_PRECISION_TARGET_ID
        ),
    )
    if partition_floors["val"] < validation_tuning_minimum:
        raise ValueError(
            "prepared validation independent-lineage floor is below its tuning targets"
        )
    source_design_payload = {
        "axis_values": raw["axis_values"],
        "corpus_design_version": raw["corpus_design_version"],
        "difficulty_calibration": difficulty,
        "independent_unit": raw["independent_unit"],
        "lineage_registry": condition_registry,
        "minimum_partition_base_lineages": raw_partition_floors,
        "partition_quotas": partition_quotas,
        "power_targets": raw["power_targets"],
        "precision_targets": raw["precision_targets"],
        "schema": CORPUS_DESIGN_SCHEMA,
        "schema_version": (
            CORPUS_DESIGN_SCHEMA_VERSION
            if receipt_version == 1
            else CORPUS_DESIGN_SCHEMA_VERSION_V2
        ),
        "stratum_quotas": raw["stratum_quotas"],
    }
    if content_digest(source_design_payload) != raw["manifest_record_digest"]:
        raise ValueError("prepared corpus-design receipt cannot reconstruct its source digest")
    return raw


def _verified_manifest_v4(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual_files != _FILES_V4:
        raise ValueError(
            "prepared-v4 corpus file set differs: "
            f"missing={sorted(_FILES_V4 - actual_files)}, "
            f"unknown={sorted(actual_files - _FILES_V4)}"
        )
    _exact_keys(
        manifest,
        {
            "corpus_design",
            "corpus_scope",
            "counts",
            "outputs",
            "policy_model_feature_allowlist",
            "provenance_record_schema",
            "qubit_cap",
            "record_digest",
            "schema",
            "schema_version",
            "source_bank_manifest_record_digest",
            "source_sha256",
            "target_authority",
        },
        "prepared-v4 manifest",
    )
    _verify_record(manifest, "prepared-v4 manifest")
    if manifest["corpus_scope"] != "production-designed-v4":
        raise ValueError("prepared-v4 corpus has the wrong declared scope")
    if manifest["provenance_record_schema"] != PREPARED_PROVENANCE_SCHEMA_V2:
        raise ValueError("prepared-v4 corpus has the wrong provenance schema")
    if manifest["policy_model_feature_allowlist"] != _MODEL_FEATURE_ALLOWLIST:
        raise ValueError("prepared-v4 policy feature allowlist differs from the registry")
    counts = manifest["counts"]
    expected_count_fields = {
        "evaluator_targets",
        "evaluator_targets_by_partition",
        "initializers",
        "policy_instances",
        "provenance_records",
    }
    if not isinstance(counts, dict):
        raise ValueError("prepared-v4 count registry must be an object")
    _exact_keys(counts, expected_count_fields, "prepared-v4 count registry")
    partition_counts = counts["evaluator_targets_by_partition"]
    if not isinstance(partition_counts, dict):
        raise ValueError("prepared-v4 partition target counts must be an object")
    _exact_keys(partition_counts, {"train", "val", "test"}, "partition target counts")
    for name in expected_count_fields - {"evaluator_targets_by_partition"}:
        _nonnegative_count(counts[name], f"prepared-v4 {name} count", positive=True)
    for partition in ("train", "val", "test"):
        _nonnegative_count(
            partition_counts[partition], f"prepared-v4 {partition} target count", positive=True
        )
    if sum(partition_counts.values()) != counts["evaluator_targets"]:
        raise ValueError("prepared-v4 partition target counts differ from their total")

    source_sha256 = manifest["source_sha256"]
    expected_source_fields = {
        "candidate_bank_jsonl",
        "candidate_bank_manifest",
        "corpus_design_manifest",
        "evaluator_targets",
        "task_provenance",
    }
    if not isinstance(source_sha256, dict):
        raise ValueError("prepared-v4 source registry must be an object")
    _exact_keys(source_sha256, expected_source_fields, "prepared-v4 source registry")
    if any(not _is_sha256(value) for value in source_sha256.values()):
        raise ValueError("prepared-v4 source registry contains an invalid digest")
    _validate_corpus_design_receipt(manifest["corpus_design"], source_sha256)

    outputs = manifest["outputs"]
    expected_outputs = _FILES_V4 - {"manifest.json"}
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        raise ValueError("prepared-v4 output registry mismatch")
    expected_records = {
        "initializers.jsonl": counts["initializers"],
        "policy_instances.jsonl": counts["policy_instances"],
        "provenance.jsonl": counts["provenance_records"],
        "splits.json": 1,
        **{
            f"targets/{partition}.jsonl": partition_counts[partition]
            for partition in ("train", "val", "test")
        },
    }
    for filename, expected_count in expected_records.items():
        receipt = outputs[filename]
        if not isinstance(receipt, dict) or set(receipt) != {"records", "sha256"}:
            raise ValueError(f"invalid prepared-v4 output receipt for {filename}")
        if receipt["records"] != expected_count or not _is_sha256(receipt["sha256"]):
            raise ValueError(f"prepared-v4 output receipt mismatch for {filename}")

    authority = manifest["target_authority"]
    if not isinstance(authority, dict):
        raise ValueError("prepared-v4 target authority must be an object")
    _exact_keys(
        authority,
        {"partitions", "record_digest", "schema", "schema_version", "total_targets"},
        "prepared-v4 target authority",
    )
    _verify_record(authority, "prepared-v4 target authority")
    if (
        authority["schema"] != TARGET_AUTHORITY_SCHEMA
        or authority["schema_version"] != TARGET_AUTHORITY_VERSION
        or authority["total_targets"] != counts["evaluator_targets"]
    ):
        raise ValueError("prepared-v4 target authority identity is invalid")
    descriptors = authority["partitions"]
    if not isinstance(descriptors, dict):
        raise ValueError("prepared-v4 target descriptors must be an object")
    _exact_keys(descriptors, {"train", "val", "test"}, "prepared-v4 target descriptors")
    for partition in ("train", "val", "test"):
        descriptor = descriptors[partition]
        if not isinstance(descriptor, dict):
            raise ValueError("prepared-v4 target descriptor must be an object")
        _exact_keys(
            descriptor,
            {"path", "records", "sha256", "target_set_digest"},
            "prepared-v4 target descriptor",
        )
        expected_path = f"targets/{partition}.jsonl"
        if (
            descriptor["path"] != expected_path
            or descriptor["records"] != partition_counts[partition]
            or descriptor["sha256"] != outputs[expected_path]["sha256"]
        ):
            raise ValueError("prepared-v4 target descriptor differs from its output commitment")
        _require_sha256(
            descriptor["target_set_digest"], f"prepared-v4 {partition} target-set digest"
        )

    # Public loading authenticates only public artifacts. Target sidecars remain unopened
    # until one explicit partition capability is exercised by load_prepared_partition.
    for filename in {
        "policy_instances.jsonl",
        "initializers.jsonl",
        "provenance.jsonl",
        "splits.json",
    }:
        _verify_output_checksum(directory, outputs, filename)
    return manifest


def _verified_manifest(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise FileNotFoundError(f"prepared corpus does not exist: {directory}")
    manifest = _read_json(directory / "manifest.json", "prepared manifest")
    if manifest.get("schema") != "isingfold.prepared-candidate-bank":
        raise ValueError("unsupported prepared corpus schema")
    version = manifest.get("schema_version")
    if version not in {
        1,
        PREPARED_SCHEMA_VERSION_V2,
        PREPARED_SCHEMA_VERSION_V3,
        PREPARED_SCHEMA_VERSION_V4,
    }:
        raise ValueError("unsupported prepared corpus schema")
    if version == PREPARED_SCHEMA_VERSION_V4:
        return _verified_manifest_v4(directory, manifest)
    expected_files = (
        _FILES_V1
        if version == 1
        else _FILES_V2
        if version == PREPARED_SCHEMA_VERSION_V2
        else _FILES_V3
    )
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            "prepared corpus file set differs: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unknown={sorted(actual_files - expected_files)}"
        )
    manifest_keys = {
        "counts",
        "outputs",
        "policy_model_feature_allowlist",
        "qubit_cap",
        "record_digest",
        "schema",
        "schema_version",
        "source_bank_manifest_record_digest",
        "source_sha256",
    }
    if version >= PREPARED_SCHEMA_VERSION_V2:
        manifest_keys |= {"corpus_scope", "provenance_record_schema"}
    if version == PREPARED_SCHEMA_VERSION_V3:
        manifest_keys.add("corpus_design")
    _exact_keys(
        manifest,
        manifest_keys,
        "prepared manifest",
    )
    _verify_record(manifest, "prepared manifest")
    expected_output_records: dict[str, int] | None = None
    if version >= PREPARED_SCHEMA_VERSION_V2:
        expected_scope = (
            "production-provenance-v2"
            if version == PREPARED_SCHEMA_VERSION_V2
            else "production-designed-v3"
        )
        if manifest["corpus_scope"] != expected_scope:
            raise ValueError(f"prepared-v{version} corpus has the wrong declared scope")
        if manifest["provenance_record_schema"] != PREPARED_PROVENANCE_SCHEMA_V2:
            raise ValueError(f"prepared-v{version} corpus has the wrong provenance schema")
        if manifest["policy_model_feature_allowlist"] != _MODEL_FEATURE_ALLOWLIST:
            raise ValueError(
                f"prepared-v{version} policy feature allowlist differs from the registry"
            )
        counts = manifest["counts"]
        if not isinstance(counts, dict) or set(counts) != {
            "evaluator_targets",
            "initializers",
            "policy_instances",
            "provenance_records",
        }:
            raise ValueError(f"prepared-v{version} count registry differs from the schema")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts.values()
        ):
            raise ValueError(f"prepared-v{version} count registry contains an invalid count")
        expected_output_records = {
            "evaluator_targets.jsonl": counts["evaluator_targets"],
            "initializers.jsonl": counts["initializers"],
            "policy_instances.jsonl": counts["policy_instances"],
            "provenance.jsonl": counts["provenance_records"],
            "splits.json": 1,
        }
        source_sha256 = manifest["source_sha256"]
        expected_source_fields = {
            "candidate_bank_jsonl",
            "candidate_bank_manifest",
            "evaluator_targets",
            "task_provenance",
        }
        if version == PREPARED_SCHEMA_VERSION_V3:
            expected_source_fields.add("corpus_design_manifest")
        if not isinstance(source_sha256, dict) or set(source_sha256) != expected_source_fields:
            raise ValueError(f"prepared-v{version} source registry differs from the schema")
        if any(not _is_sha256(value) for value in source_sha256.values()):
            raise ValueError(f"prepared-v{version} source registry contains an invalid digest")
        if version == PREPARED_SCHEMA_VERSION_V3:
            _validate_corpus_design_receipt(manifest["corpus_design"], source_sha256)
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != expected_files - {"manifest.json"}:
        raise ValueError("prepared manifest output registry mismatch")
    if expected_output_records is not None:
        for filename, expected_records in expected_output_records.items():
            receipt = outputs[filename]
            if not isinstance(receipt, dict) or set(receipt) != {"records", "sha256"}:
                raise ValueError(f"invalid output receipt for {filename}")
            if (
                isinstance(receipt["records"], bool)
                or not isinstance(receipt["records"], int)
                or receipt["records"] < 0
            ):
                raise ValueError(f"invalid output receipt record count for {filename}")
            if receipt["records"] != expected_records:
                raise ValueError(f"prepared output receipt record count mismatch for {filename}")
            if not _is_sha256(receipt["sha256"]):
                raise ValueError(f"invalid output receipt digest for {filename}")
    verified = {"policy_instances.jsonl", "initializers.jsonl", "splits.json"}
    if version >= PREPARED_SCHEMA_VERSION_V2:
        verified.add("provenance.jsonl")
    for filename in verified:
        _verify_output_checksum(directory, outputs, filename)
    return manifest


def load_prepared_corpus_design(
    directory: str | Path,
) -> dict[str, Any]:
    """Load the authenticated, self-contained scientific design receipt.

    Diagnostic prepared-v1/v2 corpora are rejected. The returned receipt retains the
    prospectively registered quota, power, and precision targets alongside the realized
    unique-base-lineage census used to validate them.
    """

    manifest = _verified_manifest(Path(directory))
    if manifest["schema_version"] not in {
        PREPARED_SCHEMA_VERSION_V3,
        PREPARED_SCHEMA_VERSION_V4,
    }:
        raise ValueError("this operation requires corpus-design-complete prepared schema v3 or v4")
    receipt = manifest["corpus_design"]
    if not isinstance(receipt, dict):  # Already enforced by _verified_manifest.
        raise RuntimeError("verified prepared corpus-design receipt is not an object")
    return receipt


def _logical_lineage(record: Mapping[str, Any]) -> str:
    return "logical-" + content_digest(
        {
            "domain": "isingfold-logical-problem-v1",
            "h": record["h"],
            "j": record["j"],
            "logical_edges": record["logical_edges"],
            "logical_nodes": record["logical_nodes"],
        }
    )


def _load_splits(directory: Path, *, schema_version: int) -> tuple[dict[str, str], dict[str, str]]:
    record = _read_json(directory / "splits.json", "prepared splits")
    lineage_field = "lineage_to_split" if schema_version == 1 else "base_parent_lineage_to_split"
    _exact_keys(
        record,
        {
            lineage_field,
            "record_digest",
            "schema",
            "schema_version",
            "test",
            "train",
            "val",
        },
        "prepared splits",
    )
    _verify_record(record, "prepared splits")
    if record["schema"] != "isingfold.lineage-splits" or record["schema_version"] != schema_version:
        raise ValueError("unsupported split schema")
    task_partition: dict[str, str] = {}
    for partition in ("train", "val", "test"):
        values = record[partition]
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ValueError(f"split {partition!r} must contain task IDs")
        for task_id in values:
            if task_id in task_partition:
                raise ValueError("one task appears in multiple partitions")
            task_partition[task_id] = partition
    lineage_to_split = record[lineage_field]
    if not isinstance(lineage_to_split, dict) or any(
        not isinstance(lineage, str) or partition not in {"train", "val", "test"}
        for lineage, partition in lineage_to_split.items()
    ):
        raise ValueError("invalid lineage-to-split registry")
    return task_partition, dict(lineage_to_split)


def _load_provenance_v2(directory: Path, *, expected_count: int) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(directory / "provenance.jsonl", "prepared provenance")
    if len(rows) != expected_count:
        raise ValueError("prepared provenance count differs from manifest")
    expected_keys = {
        "active_topology_identity",
        "base_parent_lineage",
        "calibration_identity",
        "descendant_transform_identity",
        "distribution",
        "fault_identity",
        "group_id",
        "group_record_digest",
        "instance_id",
        "instance_record_digest",
        "nominal_topology_identity",
        "record_digest",
        "schema",
        "schema_version",
        "source_logical_lineage",
        "source_record_digest",
        "source_release_id",
        "source_release_manifest_sha256",
        "split_manifest_sha256",
        "task_id",
    }
    by_task: dict[str, dict[str, Any]] = {}
    lineage_partition: dict[str, str] = {}
    source_lineage_parent: dict[str, str] = {}
    transform_parent: dict[str, str] = {}
    release_identity: tuple[str, str] | None = None
    split_identity_by_source_partition: dict[str, str] = {}
    for row in rows:
        _exact_keys(row, expected_keys, "prepared provenance")
        _verify_record(row, "prepared provenance")
        if (
            row["schema"] != PREPARED_PROVENANCE_SCHEMA_V2
            or row["schema_version"] != PREPARED_SCHEMA_VERSION_V2
        ):
            raise ValueError("unsupported prepared provenance schema")
        source_payload = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "record_digest",
                "schema",
                "schema_version",
                "source_logical_lineage",
                "source_record_digest",
                "task_id",
            }
        }
        source_payload["schema"] = SOURCE_PROVENANCE_SCHEMA_V2
        source_payload["schema_version"] = PREPARED_SCHEMA_VERSION_V2
        if content_digest(source_payload) != row["source_record_digest"]:
            raise ValueError("source provenance record digest mismatch")
        task_id = _require_text(row["task_id"], "prepared provenance task_id")
        if task_id in by_task:
            raise ValueError("duplicate prepared provenance task_id")
        _require_text(row["group_id"], "prepared provenance group_id")
        _require_text(row["instance_id"], "prepared provenance instance_id")
        _require_sha256(row["group_record_digest"], "group record digest")
        _require_sha256(row["instance_record_digest"], "instance record digest")
        _require_sha256(row["source_record_digest"], "source provenance record digest")
        base = _require_text(row["base_parent_lineage"], "base parent lineage")
        source_lineage = _require_text(row["source_logical_lineage"], "source logical lineage")

        transform = row["descendant_transform_identity"]
        if not isinstance(transform, dict):
            raise ValueError("descendant transform identity must be an object")
        _exact_keys(
            transform,
            {"kinds", "transform_sha256"},
            "descendant transform identity",
        )
        kinds = transform["kinds"]
        registered_kinds = {
            "identity",
            "gauge",
            "relabel",
            "topology",
            "fault",
            "mechanism",
            "composed",
        }
        if (
            not isinstance(kinds, list)
            or not kinds
            or any(not isinstance(kind, str) or kind not in registered_kinds for kind in kinds)
            or kinds != sorted(set(kinds))
        ):
            raise ValueError("invalid descendant transform kinds")
        if "identity" in kinds and kinds != ["identity"]:
            raise ValueError("identity cannot be combined with another transform kind")
        transform_sha256 = _require_sha256(
            transform["transform_sha256"], "descendant transform digest"
        )

        nominal = row["nominal_topology_identity"]
        if not isinstance(nominal, dict):
            raise ValueError("nominal topology identity must be an object")
        _exact_keys(
            nominal,
            {"pristine_host_sha256", "size", "topology"},
            "nominal topology identity",
        )
        nominal_topology = _require_text(nominal["topology"], "nominal topology")
        nominal_size = nominal["size"]
        if isinstance(nominal_size, bool) or not isinstance(nominal_size, int) or nominal_size <= 0:
            raise ValueError("nominal topology size must be a positive integer")
        _require_sha256(nominal["pristine_host_sha256"], "pristine host digest")

        active = row["active_topology_identity"]
        if not isinstance(active, dict):
            raise ValueError("active topology identity must be an object")
        _exact_keys(
            active,
            {"host_artifact_sha256", "host_sha256", "topology"},
            "active topology identity",
        )
        active_topology = _require_text(active["topology"], "active topology")
        _require_sha256(active["host_sha256"], "active host digest")
        host_artifact_sha256 = _require_sha256(
            active["host_artifact_sha256"], "host artifact digest"
        )
        if active_topology != nominal_topology:
            raise ValueError("nominal and active topology identities disagree")

        fault = row["fault_identity"]
        if not isinstance(fault, dict):
            raise ValueError("fault identity must be an object")
        _exact_keys(fault, {"fault_mask_sha256", "status"}, "fault identity")
        if fault["status"] not in {"none", "faulted"}:
            raise ValueError("invalid fault status")
        fault_mask_sha256 = _require_sha256(fault["fault_mask_sha256"], "fault mask digest")
        if fault_mask_sha256 != host_artifact_sha256:
            raise ValueError("fault identity does not match the active host artifact")
        if (fault["status"] == "faulted") != ("fault" in kinds):
            raise ValueError("fault status and transform identity disagree")

        calibration = row["calibration_identity"]
        if not isinstance(calibration, dict):
            raise ValueError("calibration identity must be an object")
        _exact_keys(
            calibration,
            {"calibration_sha256", "status"},
            "calibration identity",
        )
        calibration_status = calibration["status"]
        calibration_sha256 = calibration["calibration_sha256"]
        if calibration_status == "not_applicable":
            if calibration_sha256 is not None:
                raise ValueError("not-applicable calibration carries an artifact")
        elif calibration_status == "recorded":
            calibration_sha256 = _require_sha256(calibration_sha256, "calibration artifact digest")
        else:
            raise ValueError("calibration status must be explicit")

        distribution = row["distribution"]
        if not isinstance(distribution, dict):
            raise ValueError("distribution identity must be an object")
        _exact_keys(
            distribution,
            {"learning_partition", "regime", "source_partition", "stratum"},
            "distribution identity",
        )
        partition = distribution["learning_partition"]
        if partition not in {"train", "val", "test"}:
            raise ValueError("invalid provenance learning partition")
        regime = distribution["regime"]
        if regime not in {"iid", "ood"}:
            raise ValueError("invalid distribution regime")
        _require_text(distribution["source_partition"], "source partition")
        _require_text(distribution["stratum"], "distribution stratum")
        if regime == "ood" and partition != "test":
            raise ValueError("OOD provenance must be sealed test data")

        source_release_id = _require_text(row["source_release_id"], "source release ID")
        release_manifest_sha256 = _require_sha256(
            row["source_release_manifest_sha256"], "source release manifest digest"
        )
        split_manifest_sha256 = _require_sha256(
            row["split_manifest_sha256"], "source split manifest digest"
        )
        current_release_identity = (
            source_release_id,
            release_manifest_sha256,
        )
        if release_identity is None:
            release_identity = current_release_identity
        elif release_identity != current_release_identity:
            raise ValueError("prepared provenance mixes source release identities")
        prior_split_sha256 = split_identity_by_source_partition.setdefault(
            distribution["source_partition"],
            split_manifest_sha256,
        )
        if prior_split_sha256 != split_manifest_sha256:
            raise ValueError("one source partition references multiple split manifests")

        if lineage_partition.setdefault(base, partition) != partition:
            raise ValueError("one base parent lineage appears in multiple learning partitions")
        if source_lineage_parent.setdefault(source_lineage, base) != base:
            raise ValueError("one source logical lineage maps to multiple base parents")
        if transform_parent.setdefault(transform_sha256, base) != base:
            raise ValueError("one transform identity maps to multiple base parents")
        by_task[task_id] = row
    return by_task


def _as_prepared_provenance(row: Mapping[str, Any]) -> PreparedProvenance:
    transform = row["descendant_transform_identity"]
    nominal = row["nominal_topology_identity"]
    active = row["active_topology_identity"]
    fault = row["fault_identity"]
    calibration = row["calibration_identity"]
    distribution = row["distribution"]
    return PreparedProvenance(
        base_parent_lineage=row["base_parent_lineage"],
        source_logical_lineage=row["source_logical_lineage"],
        descendant_transform_kinds=tuple(transform["kinds"]),
        descendant_transform_sha256=transform["transform_sha256"],
        nominal_topology=nominal["topology"],
        nominal_size=nominal["size"],
        pristine_host_sha256=nominal["pristine_host_sha256"],
        active_topology=active["topology"],
        active_host_sha256=active["host_sha256"],
        host_artifact_sha256=active["host_artifact_sha256"],
        fault_status=fault["status"],
        fault_mask_sha256=fault["fault_mask_sha256"],
        calibration_status=calibration["status"],
        calibration_sha256=calibration["calibration_sha256"],
        distribution_regime=distribution["regime"],
        distribution_stratum=distribution["stratum"],
        source_partition=distribution["source_partition"],
        source_release_id=row["source_release_id"],
        source_release_manifest_sha256=row["source_release_manifest_sha256"],
        split_manifest_sha256=row["split_manifest_sha256"],
        group_id=row["group_id"],
        group_record_digest=row["group_record_digest"],
        instance_record_digest=row["instance_record_digest"],
        source_record_digest=row["source_record_digest"],
        record_digest=row["record_digest"],
    )


def _verify_realized_corpus_census(
    manifest: Mapping[str, Any],
    *,
    provenance_by_task: Mapping[str, Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
) -> None:
    receipt = manifest["corpus_design"]
    census = receipt["realized_census"]
    lineage_sets = {partition: set() for partition in ("train", "val", "test")}
    task_counts = {partition: 0 for partition in ("train", "val", "test")}
    axis_members: dict[str, dict[str, set[str]]] = {
        field: {}
        for field in (
            "application_family",
            "host_family",
            "fault_status",
            "distribution_regime",
            "calibration_status",
        )
    }
    conditions: set[tuple[object, ...]] = set()
    lineage_partitions: dict[str, str] = {}
    for row in provenance_by_task.values():
        distribution = row["distribution"]
        partition = distribution["learning_partition"]
        base = row["base_parent_lineage"]
        if lineage_partitions.setdefault(base, partition) != partition:
            raise ValueError("one prepared base lineage crosses learning partitions")
        lineage_sets[partition].add(base)
        task_counts[partition] += 1
        public = policies[row["instance_id"]]
        values = {
            "application_family": public["family"],
            "host_family": row["active_topology_identity"]["topology"],
            "fault_status": row["fault_identity"]["status"],
            "distribution_regime": distribution["regime"],
            "calibration_status": row["calibration_identity"]["status"],
        }
        for field, value in values.items():
            axis_members[field].setdefault(value, set()).add(base)
        conditions.add(
            (
                base,
                partition,
                *(values[field] for field in axis_members),
                row["calibration_identity"]["calibration_sha256"],
            )
        )
    expected_lineages = {
        partition: sorted(values) for partition, values in lineage_sets.items()
    }
    if census["base_lineages_by_partition"] != expected_lineages:
        raise ValueError("prepared realized base-lineage partition census is inconsistent")
    expected_partition = {
        partition: {
            "base_lineages": len(lineage_sets[partition]),
            "tasks": task_counts[partition],
        }
        for partition in ("train", "val", "test")
    }
    if census["by_partition"] != expected_partition:
        raise ValueError("prepared realized partition census is inconsistent")
    all_lineages = set().union(*lineage_sets.values())
    if (
        census["total_base_lineages"] != len(all_lineages)
        or census["total_tasks"] != len(provenance_by_task)
        or census["total_realized_conditions"] != len(conditions)
    ):
        raise ValueError("prepared realized corpus census totals are inconsistent")
    observed_fields = (
        "application_family",
        "host_family",
        "fault_status",
        "distribution_regime",
        "calibration_status",
    )
    registered_conditions = {
        (
            row["base_lineage_key"],
            row["learning_partition"],
            *(row[field] for field in observed_fields),
            row["calibration_sha256"],
        )
        for row in receipt["condition_registry"]
    }
    if conditions != registered_conditions:
        raise ValueError("prepared realized conditions differ from their authenticated registry")
    for field, value_members in axis_members.items():
        expected = {value: len(members) for value, members in sorted(value_members.items())}
        if census["by_axis"][field] != expected:
            raise ValueError(f"prepared realized {field} census is inconsistent")


def _design_conditions_by_structural_key(
    manifest: Mapping[str, Any],
) -> dict[tuple[object, ...], PreparedDesignCondition]:
    conditions: dict[tuple[object, ...], PreparedDesignCondition] = {}
    for row in manifest["corpus_design"]["condition_registry"]:
        key = (
            row["base_lineage_key"],
            row["learning_partition"],
            row["application_family"],
            row["host_family"],
            row["fault_status"],
            row["distribution_regime"],
            row["calibration_status"],
            row["calibration_sha256"],
        )
        conditions[key] = PreparedDesignCondition(
            base_lineage_key=row["base_lineage_key"],
            learning_partition=row["learning_partition"],
            application_family=row["application_family"],
            problem_origin=row["problem_origin"],
            host_family=row["host_family"],
            fault_status=row["fault_status"],
            distribution_regime=row["distribution_regime"],
            calibration_status=row["calibration_status"],
            calibration_sha256=row["calibration_sha256"],
            embedding_difficulty=row["embedding_difficulty"],
            sampling_difficulty=row["sampling_difficulty"],
            decision_difficulty=row["decision_difficulty"],
            registry_row_digest=content_digest(row),
        )
    return conditions


def _load_prepared_tasks_and_access(
    directory: str | Path,
    *,
    include_evaluator: bool,
    partition: str | None = None,
    require_provenance: bool = True,
    require_corpus_design: bool = True,
    quality_attestation_pin: QualityAttestationPin | None = None,
    permit_v4_target_access: bool = False,
) -> tuple[list[PreparedTask], TargetAccessReceipt | None]:
    """Load authenticated Profile-I tasks, opening hidden targets only when requested.

    Corpus-design-complete schemas v3 and v4 are supported, while publication workflows
    require partition-sealed v4. Diagnostic v1/v2 callers must explicitly disable the
    corresponding requirements. Opening evaluator-only ground targets additionally requires
    an out-of-band publisher pin.
    """

    if not isinstance(require_provenance, bool):
        raise TypeError("require_provenance must be boolean")
    if not isinstance(require_corpus_design, bool):
        raise TypeError("require_corpus_design must be boolean")
    root = Path(directory)
    manifest = _verified_manifest(root)
    prepared_schema_version = manifest["schema_version"]
    if partition == "validation":
        partition = "val"
    if partition is not None and partition not in {"train", "val", "test"}:
        raise ValueError(f"unknown partition {partition!r}")
    if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4 and partition is None:
        raise ValueError("prepared-v4 access requires an explicit partition")
    if (
        prepared_schema_version == PREPARED_SCHEMA_VERSION_V4
        and include_evaluator
        and not permit_v4_target_access
    ):
        raise ValueError(
            "prepared-v4 evaluator access requires load_prepared_partition so its access "
            "receipt cannot be discarded"
        )
    if require_provenance and prepared_schema_version not in {
        PREPARED_SCHEMA_VERSION_V2,
        PREPARED_SCHEMA_VERSION_V3,
        PREPARED_SCHEMA_VERSION_V4,
    }:
        raise ValueError(
            "this operation requires provenance-complete prepared schema v2 or v3, "
            "or partition-sealed v4"
        )
    if require_corpus_design and prepared_schema_version not in {
        PREPARED_SCHEMA_VERSION_V3,
        PREPARED_SCHEMA_VERSION_V4,
    }:
        raise ValueError(
            "this operation requires corpus-design-complete prepared schema v3 or v4"
        )
    attestation: PublisherAttestation | None = None
    if include_evaluator:
        if quality_attestation_pin is None:
            raise ValueError(
                "opening evaluator targets requires an out-of-band publisher-attestation pin"
            )
        attestation = load_publisher_attestation(
            quality_attestation_pin,
            prepared_manifest_path=root / "manifest.json",
        )
        if prepared_schema_version != PREPARED_SCHEMA_VERSION_V4:
            _verify_output_checksum(root, manifest["outputs"], "evaluator_targets.jsonl")
    elif quality_attestation_pin is not None:
        raise ValueError(
            "a quality-attestation pin is accepted only when evaluator targets are opened"
        )
    policy_rows = _read_jsonl(root / "policy_instances.jsonl", "policy instances")
    initializer_rows = _read_jsonl(root / "initializers.jsonl", "initializers")
    if len(policy_rows) != manifest["counts"]["policy_instances"]:
        raise ValueError("policy-instance count differs from manifest")
    if len(initializer_rows) != manifest["counts"]["initializers"]:
        raise ValueError("initializer count differs from manifest")
    policies: dict[str, dict[str, Any]] = {}
    policy_keys = {
        "family",
        "h",
        "host_edges",
        "host_nodes",
        "instance_id",
        "j",
        "logical_edges",
        "logical_nodes",
        "record_digest",
        "schema",
        "schema_version",
        "topology",
    }
    for row in policy_rows:
        _exact_keys(row, policy_keys, "policy instance")
        _verify_record(row, "policy instance")
        if row["schema"] != "isingfold.policy-instance" or row["schema_version"] != 1:
            raise ValueError("unsupported policy-instance schema")
        instance_id = row["instance_id"]
        if not isinstance(instance_id, str) or instance_id in policies:
            raise ValueError("duplicate or invalid policy instance ID")
        policies[instance_id] = row

    initializer_keys = {
        "candidate_id",
        "chains",
        "group_id",
        "group_record_digest",
        "instance_id",
        "instance_record_digest",
        "qubit_cap",
        "record_digest",
        "schema",
        "schema_version",
        "task_id",
        "validation",
    }
    source_instance_digests: dict[str, str] = {}
    for initializer in initializer_rows:
        _exact_keys(initializer, initializer_keys, "initializer")
        _verify_record(initializer, "initializer")
        if (
            initializer["schema"] != "isingfold.profile-i-initializer"
            or initializer["schema_version"] != 1
        ):
            raise ValueError("unsupported initializer schema")
        instance_id = _require_text(initializer["instance_id"], "initializer instance ID")
        source_digest = _require_sha256(
            initializer["instance_record_digest"], "initializer source-instance digest"
        )
        prior_digest = source_instance_digests.setdefault(instance_id, source_digest)
        if prior_digest != source_digest:
            raise ValueError("one policy instance references multiple source-instance digests")

    task_partition, lineage_to_split = _load_splits(
        root,
        schema_version=prepared_schema_version,
    )
    provenance_by_task = (
        {}
        if prepared_schema_version == 1
        else _load_provenance_v2(
            root,
            expected_count=manifest["counts"]["provenance_records"],
        )
    )
    if prepared_schema_version in {
        PREPARED_SCHEMA_VERSION_V3,
        PREPARED_SCHEMA_VERSION_V4,
    }:
        _verify_realized_corpus_census(
            manifest,
            provenance_by_task=provenance_by_task,
            policies=policies,
        )
        design_conditions = _design_conditions_by_structural_key(manifest)
    else:
        design_conditions = {}
    if attestation is not None:
        validate_attested_provenance(attestation, list(provenance_by_task.values()))
    targets: dict[str, dict[str, Any]] = {}
    evidence_receipt: QualityEvidenceReceipt | None = None
    target_access: TargetAccessReceipt | None = None
    if include_evaluator:
        if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4:
            if partition is None:  # Already rejected above; narrows the type.
                raise RuntimeError("prepared-v4 target access lost its partition")
            authority = manifest["target_authority"]
            descriptor = authority["partitions"][partition]
            target_relative = descriptor["path"]
            target_path = root.joinpath(*target_relative.split("/"))
            observed_target_sha = hashlib.sha256(
                target_path.read_bytes()
            ).hexdigest()
            if observed_target_sha != descriptor["sha256"]:
                raise ValueError("prepared-v4 partition target checksum mismatch")
            expected_target_count = descriptor["records"]
        else:
            target_relative = "evaluator_targets.jsonl"
            target_path = root / target_relative
            observed_target_sha = manifest["outputs"][target_relative]["sha256"]
            expected_target_count = manifest["counts"]["evaluator_targets"]
        target_rows = _read_jsonl(target_path, "evaluator targets")
        if len(target_rows) != expected_target_count:
            raise ValueError("evaluator-target count differs from manifest")
        target_keys = {
            "certificate_digest",
            "evaluator_protocol_digest",
            "instance_id",
            "instance_record_digest",
            "record_digest",
            "reference_energy",
            "reference_status",
            "schema",
            "schema_version",
        }
        if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4:
            target_keys.add("learning_partition")
        for row in target_rows:
            _exact_keys(row, target_keys, "evaluator target")
            _verify_record(row, "evaluator target")
            expected_target_version = (
                2 if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4 else 1
            )
            if (
                row["schema"] != "isingfold.evaluator-target"
                or row["schema_version"] != expected_target_version
            ):
                raise ValueError("unsupported evaluator-target schema")
            instance_id = row["instance_id"]
            if not isinstance(instance_id, str) or instance_id in targets:
                raise ValueError("duplicate or invalid evaluator target ID")
            if row["reference_status"] not in CERTIFIED_REFERENCE_STATUSES:
                raise ValueError("evaluator target has no registered certified status")
            if (
                prepared_schema_version == PREPARED_SCHEMA_VERSION_V4
                and row["learning_partition"] != partition
            ):
                raise ValueError("evaluator target crosses its committed partition")
            _require_sha256(row["certificate_digest"], "evaluator certificate digest")
            _require_sha256(
                row["evaluator_protocol_digest"], "evaluator protocol digest"
            )
            _require_sha256(row["instance_record_digest"], "target instance record digest")
            energy = row["reference_energy"]
            if isinstance(energy, bool) or not isinstance(energy, (int, float)):
                raise ValueError("evaluator target reference energy must be finite numeric data")
            try:
                finite_energy = math.isfinite(float(energy))
            except OverflowError as exc:
                raise ValueError(
                    "evaluator target reference energy must be finite numeric data"
                ) from exc
            if not finite_energy:
                raise ValueError("evaluator target reference energy must be finite numeric data")
            targets[instance_id] = row
        if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4:
            expected_target_instances = {
                initializer["instance_id"]
                for initializer in initializer_rows
                if task_partition[initializer["task_id"]] == partition
            }
        else:
            expected_target_instances = set(policies)
        if set(targets) != expected_target_instances:
            raise ValueError(
                "evaluator targets do not exactly cover their selected policy partition"
            )
        for instance_id, target in targets.items():
            if target["instance_record_digest"] != source_instance_digests.get(instance_id):
                raise ValueError("evaluator target references the wrong source-instance digest")
        if attestation is None:  # Defensive guard: assertions are not trust boundaries.
            raise RuntimeError("quality attestation was not authenticated")
        evidence_receipt = validate_quality_evidence(
            attestation,
            evaluator_targets_path=target_path,
            targets=list(targets.values()),
            partition=(
                partition
                if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4
                else None
            ),
        )
        if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4:
            expected_target_set = manifest["target_authority"]["partitions"][partition][
                "target_set_digest"
            ]
            if evidence_receipt.target_set_digest != expected_target_set:
                raise ValueError(
                    "prepared target commitment differs from publisher quality evidence"
                )
            manifest_raw = (root / "manifest.json").read_bytes()
            opened_files = [
                OpenedFileIdentity(
                    authority_root="prepared",
                    role="evaluator-targets",
                    relative_path=target_relative,
                    sha256=observed_target_sha,
                )
            ]
            opened_files.extend(
                OpenedFileIdentity(
                    authority_root="publisher",
                    role=("quality-evidence-manifest" if index == 0 else "certificate-artifact"),
                    relative_path=path,
                    sha256=sha256,
                )
                for index, (path, sha256) in enumerate(evidence_receipt.opened_files)
            )
            opened_files = list(
                dict.fromkeys(opened_files)
            )
            access_payload = {
                "evidence_manifest_record_digest": evidence_receipt.evidence_manifest_digest,
                "evidence_manifest_sha256": evidence_receipt.evidence_manifest_sha256,
                "opened_files": [asdict(item) for item in opened_files],
                "partition": partition,
                "prepared_manifest_record_digest": manifest["record_digest"],
                "prepared_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "publisher_attestation_digest": attestation.digest,
                "publisher_id": attestation.publisher_id,
                "target_authority_record_digest": manifest["target_authority"][
                    "record_digest"
                ],
                "target_count": evidence_receipt.target_count,
                "target_path": target_relative,
                "target_set_digest": evidence_receipt.target_set_digest,
                "target_sha256": observed_target_sha,
            }
            target_access = TargetAccessReceipt(
                partition=partition,
                prepared_manifest_record_digest=manifest["record_digest"],
                prepared_manifest_sha256=access_payload["prepared_manifest_sha256"],
                target_authority_record_digest=manifest["target_authority"][
                    "record_digest"
                ],
                target_path=target_relative,
                target_sha256=observed_target_sha,
                target_set_digest=evidence_receipt.target_set_digest,
                target_count=evidence_receipt.target_count,
                publisher_id=attestation.publisher_id,
                publisher_attestation_digest=attestation.digest,
                evidence_manifest_record_digest=evidence_receipt.evidence_manifest_digest,
                evidence_manifest_sha256=evidence_receipt.evidence_manifest_sha256,
                opened_files=tuple(opened_files),
                record_digest=content_digest(access_payload),
            )

    prepared: list[PreparedTask] = []
    seen_tasks: set[str] = set()
    for initializer in initializer_rows:
        task_id = initializer["task_id"]
        instance_id = initializer["instance_id"]
        if not isinstance(task_id, str) or task_id in seen_tasks:
            raise ValueError("duplicate or invalid prepared task ID")
        if instance_id not in policies:
            raise ValueError("initializer references a missing policy instance")
        if initializer["qubit_cap"] != manifest["qubit_cap"]:
            raise ValueError("initializer qubit cap differs from prepared manifest")
        if task_id not in task_partition:
            raise ValueError("initializer task is unassigned in the split registry")
        seen_tasks.add(task_id)
        task_split = task_partition[task_id]
        public = policies[instance_id]
        provenance_row = provenance_by_task.get(task_id)
        design_condition: PreparedDesignCondition | None = None
        if prepared_schema_version >= PREPARED_SCHEMA_VERSION_V2:
            if provenance_row is None:
                raise ValueError("prepared-v2 task has no provenance record")
            if provenance_row["instance_id"] != instance_id:
                raise ValueError("prepared provenance references the wrong policy instance")
            if provenance_row["instance_record_digest"] != initializer["instance_record_digest"]:
                raise ValueError("prepared provenance has the wrong instance record digest")
            if (
                provenance_row["group_id"] != initializer["group_id"]
                or provenance_row["group_record_digest"] != initializer["group_record_digest"]
            ):
                raise ValueError("prepared provenance references the wrong initializer group")
            if provenance_row["distribution"]["learning_partition"] != task_split:
                raise ValueError("prepared provenance and task split disagree")
            if provenance_row["active_topology_identity"]["topology"] != public["topology"]:
                raise ValueError("prepared active topology and policy instance disagree")
            expected_host_sha256 = content_digest(
                {
                    "edges": public["host_edges"],
                    "nodes": public["host_nodes"],
                    "schema": "embedbench.host-graph",
                    "schema_version": 1,
                }
            )
            if provenance_row["active_topology_identity"]["host_sha256"] != expected_host_sha256:
                raise ValueError("prepared active host identity does not match policy host graph")
            lineage = provenance_row["base_parent_lineage"]
            if prepared_schema_version in {
                PREPARED_SCHEMA_VERSION_V3,
                PREPARED_SCHEMA_VERSION_V4,
            }:
                distribution = provenance_row["distribution"]
                calibration = provenance_row["calibration_identity"]
                condition_key = (
                    lineage,
                    task_split,
                    public["family"],
                    provenance_row["active_topology_identity"]["topology"],
                    provenance_row["fault_identity"]["status"],
                    distribution["regime"],
                    calibration["status"],
                    calibration["calibration_sha256"],
                )
                design_condition = design_conditions.get(condition_key)
                if design_condition is None:
                    raise ValueError("prepared task has no authenticated corpus-design condition")
        else:
            if provenance_row is not None:
                raise ValueError("legacy prepared-v1 task unexpectedly carries v2 provenance")
            lineage = _logical_lineage(public)
        if lineage_to_split.get(lineage) != task_split:
            raise ValueError("task partition disagrees with its logical lineage")
        if partition is not None and task_split != partition:
            continue
        logical_nodes = list(public["logical_nodes"])
        chains = initializer["chains"]
        if not isinstance(chains, list) or len(chains) != len(logical_nodes):
            raise ValueError("initializer chain domain differs from the logical problem")
        initial_embedding = {
            node: frozenset(chain) for node, chain in zip(logical_nodes, chains, strict=True)
        }
        logical = nx.Graph()
        logical.add_nodes_from(logical_nodes)
        logical.add_edges_from(tuple(edge) for edge in public["logical_edges"])
        host = nx.Graph()
        host.add_nodes_from(public["host_nodes"])
        host.add_edges_from(tuple(edge) for edge in public["host_edges"])
        problem = LogicalProblem.from_dicts(
            {node: float(value) for node, value in public["h"]},
            {(first, second): float(value) for first, second, value in public["j"]},
        )
        target = targets.get(instance_id)
        task = EmbeddingTask(
            name=task_id,
            logical=logical,
            host=host,
            problem=problem,
            ground_energy=None if target is None else float(target["reference_energy"]),
            lineage=lineage,
            initial_embedding=initial_embedding,
        )
        prepared.append(
            PreparedTask(
                task=task,
                task_id=task_id,
                instance_id=instance_id,
                partition=task_split,
                initializer_record_digest=initializer["record_digest"],
                public_instance_record_digest=public["record_digest"],
                reference_status=None if target is None else target["reference_status"],
                certificate_digest=None if target is None else target["certificate_digest"],
                evaluator_protocol_digest=(
                    None if target is None else target["evaluator_protocol_digest"]
                ),
                prepared_schema_version=prepared_schema_version,
                corpus_scope=(
                    "legacy-pilot-v1" if prepared_schema_version == 1 else manifest["corpus_scope"]
                ),
                provenance=(
                    None if provenance_row is None else _as_prepared_provenance(provenance_row)
                ),
                design_condition=design_condition,
                quality_attestation_digest=(
                    None if attestation is None else attestation.digest
                ),
                quality_evidence_manifest_digest=(
                    None
                    if evidence_receipt is None
                    else evidence_receipt.evidence_manifest_digest
                ),
                quality_evidence_manifest_sha256=(
                    None
                    if evidence_receipt is None
                    else evidence_receipt.evidence_manifest_sha256
                ),
                quality_target_set_digest=(
                    None if evidence_receipt is None else evidence_receipt.target_set_digest
                ),
                quality_target_count=(
                    None if evidence_receipt is None else evidence_receipt.target_count
                ),
            )
        )
    expected_tasks = set(task_partition)
    actual_tasks = seen_tasks
    if actual_tasks != expected_tasks:
        raise ValueError("split registry task coverage differs from initializer records")
    if prepared_schema_version >= PREPARED_SCHEMA_VERSION_V2:
        provenance_tasks = set(provenance_by_task)
        if provenance_tasks != expected_tasks:
            raise ValueError("prepared provenance coverage differs from task registry")
    if partition is not None and not prepared:
        raise ValueError(f"partition {partition!r} contains no prepared tasks")
    return sorted(prepared, key=lambda item: item.task_id), target_access


def load_prepared_partition(
    directory: str | Path,
    *,
    partition: str | None,
    include_evaluator: bool,
    quality_attestation_pin: QualityAttestationPin | None = None,
) -> PreparedPartitionLoad:
    """Load one prepared-v4 partition without touching either other target sidecar."""

    root = Path(directory)
    manifest = _verified_manifest(root)
    if manifest["schema_version"] != PREPARED_SCHEMA_VERSION_V4:
        raise ValueError("load_prepared_partition requires partition-sealed prepared schema v4")
    tasks, access = _load_prepared_tasks_and_access(
        root,
        include_evaluator=include_evaluator,
        partition=partition,
        require_provenance=True,
        require_corpus_design=True,
        quality_attestation_pin=quality_attestation_pin,
        permit_v4_target_access=True,
    )
    if include_evaluator and access is None:
        raise RuntimeError("prepared-v4 evaluator access produced no access receipt")
    if not include_evaluator and access is not None:
        raise RuntimeError("public prepared-v4 access unexpectedly opened targets")
    return PreparedPartitionLoad(tasks=tuple(tasks), target_access=access)


def load_prepared_partition_access(
    directory: str | Path,
    *,
    partition: str,
    quality_attestation_pin: QualityAttestationPin,
) -> tuple[tuple[PreparedTask, ...], dict[str, Any]]:
    """CLI-facing evaluator load with a canonical, non-discarded access record.

    This deliberately has no ``include_evaluator`` switch: invoking it is the explicit
    capability to open exactly one v4 target partition.  The returned record includes its
    recomputed ``record_digest`` and is ready to embed in a run, report, or seal receipt.
    """

    loaded = load_prepared_partition(
        directory,
        partition=partition,
        include_evaluator=True,
        quality_attestation_pin=quality_attestation_pin,
    )
    if loaded.target_access is None:  # Defensive guard around the v4 loader contract.
        raise RuntimeError("prepared evaluator load produced no target-access receipt")
    return loaded.tasks, loaded.target_access.as_dict()


def load_prepared_tasks(
    directory: str | Path,
    *,
    include_evaluator: bool,
    partition: str | None = None,
    require_provenance: bool = True,
    require_corpus_design: bool = True,
    quality_attestation_pin: QualityAttestationPin | None = None,
) -> list[PreparedTask]:
    """Load diagnostic v1-v3 tasks or public-only v4 tasks.

    Evaluator-bearing v4 calls must use :func:`load_prepared_partition`; otherwise the
    mandatory target-access receipt could be accidentally discarded by legacy callers.
    """

    tasks, _ = _load_prepared_tasks_and_access(
        directory,
        include_evaluator=include_evaluator,
        partition=partition,
        require_provenance=require_provenance,
        require_corpus_design=require_corpus_design,
        quality_attestation_pin=quality_attestation_pin,
        permit_v4_target_access=False,
    )
    return tasks


__all__ = [
    "OpenedFileIdentity",
    "PreparedPartitionLoad",
    "PreparedDesignCondition",
    "PreparedProvenance",
    "PreparedTask",
    "TargetAccessReceipt",
    "load_prepared_corpus_design",
    "load_prepared_partition",
    "load_prepared_partition_access",
    "load_prepared_tasks",
]

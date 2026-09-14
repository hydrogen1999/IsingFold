"""Post-freeze four-strength regret audit for the final learned and stock arms.

The primary complete-system evaluator samples only the deployed strength.  This module
implements a separate diagnostic protocol over an outcome-blind, presealed subset.  It
authenticates the terminal sidecars, recompiles every valid return, and then draws two new
domain-separated blocks at each of the four registered strengths.  Block A chooses an empirical
oracle with the fixed lowest-index tie rule; independent block B estimates oracle-minus-selected
regret.  Neither block is allowed to affect training, model selection, or the primary endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.complete_system import (
    CompletePopulationIdentity,
    FrozenComponentIdentity,
    TerminalEvidence,
    _typed_identity,
    verify_terminal_evidence,
)
from isingfold.rl.complete_system_aggregate import (
    AuthenticatedCompleteSystemSeedRun,
    aggregate_complete_system_seeds,
)
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.env import EmbeddingTask
from isingfold.rl.evaluate import EpisodeOutcome, program_digest
from isingfold.rl.evaluation_strata import EvaluationStratum, STRATUM_AXES
from isingfold.rl.evaluator import ReadBlock, sample_program
from isingfold.rl.external_pairing import (
    AuthenticatedExternalCompleteRun,
    aggregate_learned_vs_stock,
    context_snapshot,
)
from isingfold.rl.external_tuning import ExternalTuningExecutionBinding
from isingfold.rl.experiment_selection import crossed_bootstrap_bounds
from isingfold.rl.program import Program
from isingfold.rl.validate import p_return

CONFIG_SCHEMA = "isingfold.final-strength-audit-config"
CONFIG_VERSION = 1
SAMPLER_SCHEMA = "isingfold.final-strength-audit-sampler"
SAMPLER_VERSION = 1
REGISTERED_CONFIG_ID = "final-strength-four-program-split-block-v1"
REGISTERED_SAMPLER_ID = "isingfold-sa-sample-program-v1"
PRODUCTION_READS_PER_BLOCK = 4096
PRODUCTION_MINIMUM_BASE_LINEAGES = 128
PRODUCTION_BOOTSTRAP_REPLICATES = 20_000
PLAN_SCHEMA = "isingfold.final-strength-audit-plan"
PLAN_VERSION = 2
ROW_SCHEMA = "isingfold.final-strength-audit-row"
ROW_VERSION = 1
RECEIPT_SCHEMA = "isingfold.final-strength-audit"
RECEIPT_VERSION = 2
EXECUTION_MANIFEST_SCHEMA = "isingfold.final-strength-audit-execution-manifest"
EXECUTION_MANIFEST_VERSION = 2
SHARD_SCHEMA = "isingfold.final-strength-audit-shard"
SHARD_VERSION = 1
MERGE_SCHEMA = "isingfold.final-strength-audit-merge"
MERGE_VERSION = 1
REGISTERED_TRAINING_SEEDS = (1103, 2207, 3301)
ARMS = ("learned", "tuned-stock")
BLOCK_DOMAINS = {
    "A": "isingfold-final-strength-audit-oracle-choice-a-v1",
    "B": "isingfold-final-strength-audit-unbiased-estimate-b-v1",
}
AUDIT_RUNTIME_DOMAIN = "isingfold-final-strength-audit-runtime-v1"
SUBSET_DOMAIN = "isingfold-final-strength-audit-stratified-subset-v1"
AGGREGATION = "equal-training-seed-then-equal-immutable-base-lineage"
_DIGEST_LENGTH = 64
_STOCK_BINDING_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "mode",
        "registry_id",
        "registry_file_sha256",
        "registry_record_digest",
        "candidate_index",
        "candidate_id",
        "candidate",
        "candidate_digest",
        "selection_file_sha256",
        "selection_record_digest",
    }
)

Arm = Literal["learned", "tuned-stock"]
Sampler = Callable[..., ReadBlock]


class FinalStrengthAuditError(RuntimeError):
    """The sealed audit contract or one of its authenticated inputs is invalid."""


@dataclass(frozen=True)
class FinalStrengthAuditConfig:
    """Typed, preregistered production contract for the post-freeze audit."""

    registry_id: str
    reads_per_block: int
    audit_seed: int
    lineages_per_stratum_cap: int
    minimum_base_lineages: int
    shard_count: int
    bootstrap_replicates: int
    bootstrap_seed: int
    two_sided_alpha: float
    sampler_id: str
    chronology: Mapping[str, object]

    _CHRONOLOGY = {
        "plan_sealed_before_test_outcomes": True,
        "six_source_runs_authenticated_before_execution_manifest": True,
        "execution_manifest_sealed_before_audit_reads": True,
        "audit_feedback_forbidden": True,
    }

    def __post_init__(self) -> None:
        if self.registry_id != REGISTERED_CONFIG_ID:
            raise ValueError("final-strength config has an unregistered identity")
        if self.reads_per_block != PRODUCTION_READS_PER_BLOCK:
            raise ValueError("production final-strength reads_per_block must be 4096")
        if type(self.audit_seed) is not int or not 0 <= self.audit_seed < 2**31:
            raise ValueError("final-strength audit_seed must be a canonical 31-bit integer")
        if (
            type(self.lineages_per_stratum_cap) is not int
            or self.lineages_per_stratum_cap <= 0
        ):
            raise ValueError("lineages_per_stratum_cap must be positive")
        if (
            type(self.minimum_base_lineages) is not int
            or self.minimum_base_lineages < PRODUCTION_MINIMUM_BASE_LINEAGES
        ):
            raise ValueError("production final-strength audit requires at least 128 lineages")
        if type(self.shard_count) is not int or self.shard_count <= 1:
            raise ValueError("production final-strength audit requires at least two shards")
        if self.bootstrap_replicates != PRODUCTION_BOOTSTRAP_REPLICATES:
            raise ValueError("production final-strength bootstrap fixes 20000 replicates")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**31:
            raise ValueError("bootstrap_seed must be a canonical 31-bit integer")
        if type(self.two_sided_alpha) is not float or self.two_sided_alpha != 0.05:
            raise ValueError("production final-strength audit fixes two-sided alpha at 0.05")
        if self.sampler_id != REGISTERED_SAMPLER_ID:
            raise ValueError("production final-strength audit changes the registered sampler")
        chronology = _canonical_mapping(self.chronology, label="audit chronology")
        if _plain_json(chronology) != self._CHRONOLOGY:
            raise ValueError("final-strength config changes mandatory chronology flags")
        object.__setattr__(self, "chronology", chronology)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": CONFIG_SCHEMA,
            "schema_version": CONFIG_VERSION,
            "registry_id": self.registry_id,
            "scope": "post-freeze-diagnostic-only",
            "reads_per_block": self.reads_per_block,
            "audit_seed": self.audit_seed,
            "lineages_per_stratum_cap": self.lineages_per_stratum_cap,
            "minimum_base_lineages": self.minimum_base_lineages,
            "shard_count": self.shard_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "two_sided_alpha": self.two_sided_alpha,
            "sampler_id": self.sampler_id,
            "monolithic_production_execution": False,
            "chronology": _plain_json(self.chronology),
        }
        if include_digest:
            payload["record_digest"] = content_digest(payload)
        return payload

    @property
    def record_digest(self) -> str:
        return content_digest(self.as_dict(include_digest=False))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FinalStrengthAuditConfig:
        expected = {
            "schema",
            "schema_version",
            "registry_id",
            "scope",
            "reads_per_block",
            "audit_seed",
            "lineages_per_stratum_cap",
            "minimum_base_lineages",
            "shard_count",
            "bootstrap_replicates",
            "bootstrap_seed",
            "two_sided_alpha",
            "sampler_id",
            "monolithic_production_execution",
            "chronology",
            "record_digest",
        }
        if set(value) != expected:
            raise ValueError("final-strength config has missing or unknown fields")
        if (
            value["schema"] != CONFIG_SCHEMA
            or value["schema_version"] != CONFIG_VERSION
            or value["scope"] != "post-freeze-diagnostic-only"
            or value["monolithic_production_execution"] is not False
            or not isinstance(value["chronology"], Mapping)
        ):
            raise ValueError("final-strength config changes the registered protocol")
        result = cls(
            registry_id=value["registry_id"],  # type: ignore[arg-type]
            reads_per_block=value["reads_per_block"],  # type: ignore[arg-type]
            audit_seed=value["audit_seed"],  # type: ignore[arg-type]
            lineages_per_stratum_cap=value["lineages_per_stratum_cap"],  # type: ignore[arg-type]
            minimum_base_lineages=value["minimum_base_lineages"],  # type: ignore[arg-type]
            shard_count=value["shard_count"],  # type: ignore[arg-type]
            bootstrap_replicates=value["bootstrap_replicates"],  # type: ignore[arg-type]
            bootstrap_seed=value["bootstrap_seed"],  # type: ignore[arg-type]
            two_sided_alpha=value["two_sided_alpha"],  # type: ignore[arg-type]
            sampler_id=value["sampler_id"],  # type: ignore[arg-type]
            chronology=value["chronology"],
        )
        if result.as_dict() != dict(value):
            raise ValueError("final-strength config digest or canonical form is invalid")
        return result


@dataclass(frozen=True)
class AuthenticatedFinalStrengthAuditConfig:
    config: FinalStrengthAuditConfig
    file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, FinalStrengthAuditConfig):
            raise TypeError("authenticated final-strength config has the wrong type")
        _require_digest(self.file_sha256, label="final-strength config file")


@dataclass(frozen=True)
class FinalStrengthSamplerIdentity:
    """Exact registered production sampler implementation and dependency identity."""

    sampler_id: str
    callable_path: str
    module_source_path: str
    module_source_sha256: str
    runtime_implementation_digest: str
    dependency_versions: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.sampler_id != REGISTERED_SAMPLER_ID:
            raise ValueError("final-strength sampler identity is unregistered")
        if self.callable_path != "isingfold.rl.evaluator.sample_program":
            raise ValueError("final-strength sampler callable differs")
        if self.module_source_path != "isingfold/rl/evaluator.py":
            raise ValueError("final-strength sampler source path differs")
        _require_digest(self.module_source_sha256, label="sampler module source")
        _require_digest(self.runtime_implementation_digest, label="sampler runtime registry")
        dependencies = _canonical_mapping(
            self.dependency_versions, label="sampler dependencies"
        )
        if set(dependencies) != {"python", "numpy", "dimod", "dwave-samplers"} or any(
            not isinstance(value, str) or not value for value in dependencies.values()
        ):
            raise ValueError("final-strength sampler dependency identity is incomplete")
        object.__setattr__(self, "dependency_versions", dependencies)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SAMPLER_SCHEMA,
            "schema_version": SAMPLER_VERSION,
            "sampler_id": self.sampler_id,
            "callable_path": self.callable_path,
            "module_source_path": self.module_source_path,
            "module_source_sha256": self.module_source_sha256,
            "runtime_implementation_digest": self.runtime_implementation_digest,
            "dependency_versions": _plain_json(self.dependency_versions),
            "seed_width_bits": 31,
            "sampling_semantics": "dimod-simulated-annealing-explicit-seed-and-sweeps",
        }

    @property
    def digest(self) -> str:
        return content_digest(self.as_dict())


def registered_final_strength_sampler_identity() -> FinalStrengthSamplerIdentity:
    registry = runtime_implementation_registry()
    modules = registry.get("modules")
    dependencies = registry.get("dependencies")
    if not isinstance(modules, Mapping) or not isinstance(dependencies, Mapping):
        raise FinalStrengthAuditError("runtime implementation registry is malformed")
    evaluator = modules.get("isingfold.rl.evaluator")
    if not isinstance(evaluator, Mapping):
        raise FinalStrengthAuditError("runtime registry omits the final sampler source")
    return FinalStrengthSamplerIdentity(
        sampler_id=REGISTERED_SAMPLER_ID,
        callable_path="isingfold.rl.evaluator.sample_program",
        module_source_path=evaluator.get("source_path"),  # type: ignore[arg-type]
        module_source_sha256=evaluator.get("sha256"),  # type: ignore[arg-type]
        runtime_implementation_digest=content_digest(registry),
        dependency_versions={
            name: dependencies.get(name)
            for name in ("python", "numpy", "dimod", "dwave-samplers")
        },
    )


def load_final_strength_audit_config(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> AuthenticatedFinalStrengthAuditConfig:
    """Authenticate and parse the exact preregistered production audit config."""

    _require_digest(expected_sha256, label="expected final-strength config file")
    content = Path(path).read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise FinalStrengthAuditError(
            f"final-strength config SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed}"
        )
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalStrengthAuditError("final-strength config is not UTF-8 JSON") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) + b"\n" != content:
        raise FinalStrengthAuditError("final-strength config is not canonical JSON")
    try:
        config = FinalStrengthAuditConfig.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise FinalStrengthAuditError("final-strength config is invalid") from exc
    return AuthenticatedFinalStrengthAuditConfig(config, expected_sha256)


def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_digest(value: object, *, label: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value  # type: ignore[return-value]


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: object) -> object:
    return json.loads(canonical_json_bytes(value))


def _canonical_mapping(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    try:
        copied = _plain_json(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite canonical JSON") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be a JSON object")
    frozen = _freeze_json(copied)
    assert isinstance(frozen, Mapping)
    return frozen


def _stock_binding(value: Mapping[str, object]) -> Mapping[str, object]:
    raw = _canonical_mapping(value, label="stock tuning execution binding")
    if set(raw) != _STOCK_BINDING_KEYS:
        raise ValueError("stock tuning execution binding has missing or unknown fields")
    try:
        typed = ExternalTuningExecutionBinding.from_mapping(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("stock tuning execution is not in the finite registered grid") from exc
    if typed.mode != "frozen-deployment":
        raise ValueError("final audit requires the frozen tuned-stock deployment binding")
    binding = _canonical_mapping(typed.as_dict(), label="typed stock tuning execution binding")
    if _plain_json(binding) != _plain_json(raw):
        raise ValueError("stock tuning execution binding is not canonical")
    return binding


def _validate_quality_authority(value: Mapping[str, object]) -> Mapping[str, object]:
    """Validate global publisher authority separately from the test partition."""

    authority = _canonical_mapping(value, label="quality authority")
    if set(authority) != {
        "schema",
        "schema_version",
        "global",
        "evaluation_partition",
        "record_digest",
    } or (
        authority.get("schema") != "isingfold.quality-authority-binding"
        or authority.get("schema_version") != 2
    ):
        raise ValueError("quality authority has missing, unknown, or unsupported fields")
    _require_digest(authority["record_digest"], label="quality authority binding record")
    if content_digest(
        {
            name: _plain_json(item)
            for name, item in authority.items()
            if name != "record_digest"
        }
    ) != authority["record_digest"]:
        raise ValueError("quality authority binding record digest differs")
    global_authority = authority["global"]
    evaluation = authority["evaluation_partition"]
    if not isinstance(global_authority, Mapping) or set(global_authority) != {
        "schema",
        "schema_version",
        "record_digest",
        "publication_id",
        "publisher_id",
        "publisher_attestation_record_digest",
        "target_authority_record_digest",
        "ground_root",
    }:
        raise ValueError("quality authority global projection is malformed")
    if (
        global_authority.get("schema") != "isingfold.global-quality-authority"
        or global_authority.get("schema_version") != 1
    ):
        raise ValueError("quality authority global projection has an unsupported schema")
    if (
        not isinstance(global_authority["publisher_id"], str)
        or not global_authority["publisher_id"]
        or not isinstance(global_authority["publication_id"], str)
        or not global_authority["publication_id"]
    ):
        raise ValueError("quality authority publisher/publication IDs must be nonempty")
    for name in (
        "publisher_attestation_record_digest",
        "target_authority_record_digest",
        "record_digest",
    ):
        _require_digest(global_authority[name], label=f"quality authority global {name}")
    ground_root = global_authority["ground_root"]
    if not isinstance(ground_root, Mapping) or set(ground_root) != {
        "receipt_sha256",
        "record_digest",
        "verifier_identity_digest",
    }:
        raise ValueError("quality authority ground-root projection is malformed")
    for name, digest in ground_root.items():
        _require_digest(digest, label=f"quality authority ground root {name}")
    if content_digest(
        {
            name: _plain_json(item)
            for name, item in global_authority.items()
            if name != "record_digest"
        }
    ) != global_authority["record_digest"]:
        raise ValueError("quality authority global record digest differs")
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "schema",
        "schema_version",
        "record_digest",
        "name",
        "target_access_record_digest",
        "evidence_manifest_record_digest",
        "evidence_manifest_sha256",
        "target_set_digest",
        "record_digest",
        "target_count",
        "ground_partition",
    }:
        raise ValueError("quality authority evaluation-partition projection is malformed")
    if (
        evaluation.get("schema") != "isingfold.partition-quality-authority"
        or evaluation.get("schema_version") != 1
        or evaluation["name"] != "test"
    ):
        raise ValueError("final-strength quality authority must select the test partition")
    for name in (
        "target_access_record_digest",
        "evidence_manifest_record_digest",
        "evidence_manifest_sha256",
        "target_set_digest",
    ):
        _require_digest(evaluation[name], label=f"quality authority test {name}")
    if type(evaluation["target_count"]) is not int or evaluation["target_count"] <= 0:
        raise ValueError("quality authority test target count must be positive")
    ground_partition = evaluation["ground_partition"]
    if not isinstance(ground_partition, Mapping) or set(ground_partition) != {
        "receipt_record_digest",
        "receipt_sha256",
        "accepted_count",
        "instance_set_digest",
    }:
        raise ValueError("quality authority ground-partition projection is malformed")
    for name in ("receipt_record_digest", "receipt_sha256", "instance_set_digest"):
        _require_digest(
            ground_partition[name], label=f"quality authority ground partition {name}"
        )
    if ground_partition["accepted_count"] != evaluation["target_count"]:
        raise ValueError("quality authority ground partition is not fully accepted")
    if content_digest(
        {
            name: _plain_json(item)
            for name, item in evaluation.items()
            if name != "record_digest"
        }
    ) != evaluation["record_digest"]:
        raise ValueError("quality authority evaluation-partition record digest differs")
    return authority


def _validate_learned_selection(value: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the pre-test frozen learned configuration and all three source cells."""

    selection = _canonical_mapping(value, label="frozen learned selection")
    expected = {
        "schema",
        "schema_version",
        "selection_receipt_sha256",
        "selection_record_digest",
        "grid_manifest_sha256",
        "selected_model_family",
        "selected_grid_model_family",
        "selected_method",
        "training_seeds",
        "source_cell_ids",
        "source_checkpoint_payload_digests",
        "representation_selection_sha256",
        "representation_selection_record_digest",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
        "seed_selection_forbidden",
    }
    if set(selection) != expected or (
        selection.get("schema") != "isingfold.final-strength-frozen-learned-selection"
        or selection.get("schema_version") != 1
        or selection.get("seed_selection_forbidden") is not True
    ):
        raise ValueError("frozen learned selection has missing or unknown fields")
    for name in (
        "selection_receipt_sha256",
        "selection_record_digest",
        "grid_manifest_sha256",
        "representation_selection_sha256",
        "representation_selection_record_digest",
        "runtime_implementation_digest",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
    ):
        _require_digest(selection[name], label=f"learned selection {name}")
    for name in ("selected_model_family", "selected_grid_model_family", "selected_method"):
        if not isinstance(selection[name], str) or not selection[name]:
            raise ValueError(f"learned selection {name} must be nonempty")
    if tuple(selection["training_seeds"]) != REGISTERED_TRAINING_SEEDS:  # type: ignore[arg-type]
        raise ValueError("learned selection changes the registered training seeds")
    cells = selection["source_cell_ids"]
    checkpoints = selection["source_checkpoint_payload_digests"]
    if (
        not isinstance(cells, Sequence)
        or isinstance(cells, (str, bytes, bytearray))
        or len(cells) != 3
        or len(set(cells)) != 3
        or any(not isinstance(cell, str) or not cell for cell in cells)
        or not isinstance(checkpoints, Sequence)
        or isinstance(checkpoints, (str, bytes, bytearray))
        or len(checkpoints) != 3
        or any(not _is_digest(item) for item in checkpoints)
    ):
        raise ValueError("learned selection does not bind three exact source cells")
    registry = selection["runtime_implementation_registry"]
    if not isinstance(registry, Mapping) or content_digest(registry) != selection[
        "runtime_implementation_digest"
    ]:
        raise ValueError("learned selection runtime registry digest differs")
    return selection


def _validate_target_access(
    value: Mapping[str, object],
    *,
    source_manifest_sha256: str | None = None,
    quality_authority: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Validate the complete prepared-v4 test-target capability record."""

    access = _canonical_mapping(value, label="test target access")
    expected = {
        "evidence_manifest_record_digest",
        "evidence_manifest_sha256",
        "opened_files",
        "partition",
        "prepared_manifest_record_digest",
        "prepared_manifest_sha256",
        "publisher_attestation_digest",
        "publisher_id",
        "record_digest",
        "target_authority_record_digest",
        "target_count",
        "target_path",
        "target_set_digest",
        "target_sha256",
    }
    if set(access) != expected:
        raise ValueError("test target access has missing or unknown fields")
    if access["partition"] != "test" or access["target_path"] != "targets/test.jsonl":
        raise ValueError("final-strength execution requires exactly the test partition")
    if not isinstance(access["publisher_id"], str) or not access["publisher_id"]:
        raise ValueError("test target access publisher ID must be nonempty")
    if type(access["target_count"]) is not int or access["target_count"] <= 0:
        raise ValueError("test target access count must be positive")
    digest_fields = (
        "evidence_manifest_record_digest",
        "evidence_manifest_sha256",
        "prepared_manifest_record_digest",
        "prepared_manifest_sha256",
        "publisher_attestation_digest",
        "record_digest",
        "target_authority_record_digest",
        "target_set_digest",
        "target_sha256",
    )
    for name in digest_fields:
        _require_digest(access[name], label=f"test target access {name}")
    opened = access["opened_files"]
    if not isinstance(opened, Sequence) or isinstance(opened, (str, bytes, bytearray)):
        raise ValueError("test target access opened-file registry must be a sequence")
    identities: set[tuple[object, object, object]] = set()
    prepared_target_opens = 0
    evidence_manifest_opens = 0
    for row in opened:
        if not isinstance(row, Mapping) or set(row) != {
            "authority_root",
            "role",
            "relative_path",
            "sha256",
        }:
            raise ValueError("test target access opened-file row is malformed")
        if row["authority_root"] not in {"prepared", "publisher"}:
            raise ValueError("test target access opened-file authority is invalid")
        if any(
            not isinstance(row[name], str) or not row[name]
            for name in ("role", "relative_path")
        ):
            raise ValueError("test target access opened-file text is invalid")
        relative_path = str(row["relative_path"])
        path = Path(relative_path)
        if (
            path.is_absolute()
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("test target access opened-file path is unsafe")
        _require_digest(row["sha256"], label="test target access opened-file")
        identity = (row["authority_root"], row["role"], row["relative_path"])
        if identity in identities:
            raise ValueError("test target access repeats an opened-file identity")
        identities.add(identity)
        if (
            row["authority_root"] == "prepared"
            and row["role"] == "evaluator-targets"
            and row["relative_path"] == access["target_path"]
            and row["sha256"] == access["target_sha256"]
        ):
            prepared_target_opens += 1
        elif row["authority_root"] == "prepared":
            raise ValueError("test target access opens an unregistered prepared artifact")
        if (
            row["authority_root"] == "publisher"
            and row["role"] == "quality-evidence-manifest"
            and row["sha256"] == access["evidence_manifest_sha256"]
        ):
            evidence_manifest_opens += 1
    if prepared_target_opens != 1 or evidence_manifest_opens != 1:
        raise ValueError("test target access omits its exact target or evidence-manifest open")
    payload = {name: _plain_json(item) for name, item in access.items() if name != "record_digest"}
    if content_digest(payload) != access["record_digest"]:
        raise ValueError("test target access record digest differs")
    if (
        source_manifest_sha256 is not None
        and access["prepared_manifest_sha256"] != source_manifest_sha256
    ):
        raise ValueError("test target access belongs to another prepared manifest")
    if quality_authority is not None and quality_authority.get("schema") == (
        "isingfold.quality-authority-binding"
    ):
        global_authority = quality_authority.get("global")
        evaluation = quality_authority.get("evaluation_partition")
        if not isinstance(global_authority, Mapping) or not isinstance(
            evaluation, Mapping
        ):
            raise ValueError("sealed quality authority projection is malformed")
        global_bindings = {
            "publisher_id": "publisher_id",
            "publisher_attestation_record_digest": "publisher_attestation_digest",
            "target_authority_record_digest": "target_authority_record_digest",
        }
        partition_bindings = {
            "target_access_record_digest": "record_digest",
            "evidence_manifest_record_digest": "evidence_manifest_record_digest",
            "evidence_manifest_sha256": "evidence_manifest_sha256",
            "target_set_digest": "target_set_digest",
            "target_count": "target_count",
        }
        if any(
            global_authority.get(authority_name) != access[access_name]
            for authority_name, access_name in global_bindings.items()
        ) or any(
            evaluation.get(authority_name) != access[access_name]
            for authority_name, access_name in partition_bindings.items()
        ):
            raise ValueError("test target access differs from the sealed quality authority")
    return access


def _learned_binding_matches_plan(
    binding: Mapping[str, object],
    selection: Mapping[str, object],
    *,
    seed: int,
) -> bool:
    index = REGISTERED_TRAINING_SEEDS.index(seed)
    common = {
        "selection_receipt_sha256": "selection_receipt_sha256",
        "selection_record_digest": "selection_record_digest",
        "grid_manifest_sha256": "grid_manifest_sha256",
        "selected_model_family": "selected_model_family",
        "selected_grid_model_family": "selected_grid_model_family",
        "selected_method": "selected_method",
        "all_training_seeds": "training_seeds",
        "all_source_cell_ids": "source_cell_ids",
        "all_source_checkpoint_payload_digests": "source_checkpoint_payload_digests",
        "runtime_implementation_registry": "runtime_implementation_registry",
        "runtime_implementation_digest": "runtime_implementation_digest",
        "quality_preflight_receipt_sha256": "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest": "quality_preflight_record_digest",
        "seed_selection_forbidden": "seed_selection_forbidden",
    }
    return (
        all(binding.get(source) == selection.get(target) for source, target in common.items())
        and binding.get("training_seed") == seed
        and binding.get("source_cell_id") == selection["source_cell_ids"][index]  # type: ignore[index]
        and binding.get("source_checkpoint_payload_digest")
        == selection["source_checkpoint_payload_digests"][index]  # type: ignore[index]
    )


def _stratum_coordinates(row: EvaluationStratum) -> tuple[tuple[str, str], ...]:
    return tuple((name, str(getattr(row, name))) for name in (*STRATUM_AXES, "size_bin"))


def _stratum_id(coordinates: Sequence[tuple[str, str]]) -> str:
    return stable_digest(
        {"domain": "isingfold-final-strength-audit-stratum-v1", "coordinates": coordinates}
    )


@dataclass(frozen=True, order=True)
class FinalStrengthAuditKey:
    """Exact opportunity key.  The arm is part of the key, not implicit metadata."""

    lineage: str
    instance: str
    repetition: int
    training_seed: int
    arm: Arm

    def __post_init__(self) -> None:
        if not self.lineage or not self.instance:
            raise ValueError("final-strength audit key needs lineage and instance")
        if type(self.repetition) is not int or self.repetition < 0:
            raise ValueError("final-strength audit repetition must be nonnegative")
        if self.training_seed not in REGISTERED_TRAINING_SEEDS:
            raise ValueError("final-strength audit key has an unregistered training seed")
        if self.arm not in ARMS:
            raise ValueError("final-strength audit key has an unsupported arm")

    @property
    def paired_key(self) -> tuple[str, str, int, int]:
        return (self.lineage, self.instance, self.repetition, self.training_seed)

    def as_dict(self) -> dict[str, object]:
        return {
            "lineage": self.lineage,
            "instance": self.instance,
            "repetition": self.repetition,
            "training_seed": self.training_seed,
            "arm": self.arm,
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FinalStrengthAuditKey:
        if set(value) != {"lineage", "instance", "repetition", "training_seed", "arm"}:
            raise ValueError("final-strength audit key fields differ")
        return cls(
            lineage=value["lineage"],  # type: ignore[arg-type]
            instance=value["instance"],  # type: ignore[arg-type]
            repetition=value["repetition"],  # type: ignore[arg-type]
            training_seed=value["training_seed"],  # type: ignore[arg-type]
            arm=value["arm"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SelectedAuditInstance:
    lineage: str
    instance: str
    stratum_id: str
    stratum_coordinates: tuple[tuple[str, str], ...]
    stratum_record_digest: str
    selection_rank: int

    def __post_init__(self) -> None:
        if not self.lineage or not self.instance or not _is_digest(self.stratum_id):
            raise ValueError("selected audit instance identity is malformed")
        expected_names = (*STRATUM_AXES, "size_bin")
        if tuple(name for name, _ in self.stratum_coordinates) != expected_names or any(
            not value for _, value in self.stratum_coordinates
        ):
            raise ValueError("selected audit instance has malformed stratum coordinates")
        if self.stratum_id != _stratum_id(self.stratum_coordinates):
            raise ValueError("selected audit instance stratum identity is inconsistent")
        _require_digest(self.stratum_record_digest, label="evaluation stratum record")
        if type(self.selection_rank) is not int or self.selection_rank < 0:
            raise ValueError("selected audit instance rank must be nonnegative")

    @property
    def identity(self) -> tuple[str, str]:
        return (self.lineage, self.instance)

    def as_dict(self) -> dict[str, object]:
        return {
            "lineage": self.lineage,
            "instance": self.instance,
            "stratum_id": self.stratum_id,
            "stratum_coordinates": [list(item) for item in self.stratum_coordinates],
            "stratum_record_digest": self.stratum_record_digest,
            "selection_rank": self.selection_rank,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SelectedAuditInstance:
        expected = {
            "lineage",
            "instance",
            "stratum_id",
            "stratum_coordinates",
            "stratum_record_digest",
            "selection_rank",
        }
        if set(value) != expected:
            raise ValueError("selected audit instance fields differ")
        coordinates = value["stratum_coordinates"]
        if not isinstance(coordinates, list) or any(
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(part, str) for part in item)
            for item in coordinates
        ):
            raise ValueError("selected audit stratum coordinates are malformed")
        return cls(
            lineage=value["lineage"],  # type: ignore[arg-type]
            instance=value["instance"],  # type: ignore[arg-type]
            stratum_id=value["stratum_id"],  # type: ignore[arg-type]
            stratum_coordinates=tuple((item[0], item[1]) for item in coordinates),
            stratum_record_digest=value["stratum_record_digest"],  # type: ignore[arg-type]
            selection_rank=value["selection_rank"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class FinalStrengthAuditPlan:
    """Outcome-blind subset and immutable execution contract sealed before any audit read."""

    population_digest: str
    evaluation_strata_digest: str
    context_digest: str
    selector_digest: str
    learned_selection: Mapping[str, object]
    learned_selection_digest: str
    quality_authority: Mapping[str, object]
    quality_authority_digest: str
    runtime_implementation_digest: str
    sampler_identity: Mapping[str, object]
    sampler_identity_digest: str
    audit_config: Mapping[str, object]
    audit_config_record_digest: str
    audit_config_file_sha256: str | None
    publication_eligible: bool
    stock_tuning_execution: Mapping[str, object]
    stock_tuning_execution_digest: str
    reads_per_block: int
    num_sweeps: int
    audit_seed: int
    lineages_per_stratum: int
    minimum_base_lineages: int
    shard_count: int
    bootstrap_replicates: int
    bootstrap_seed: int
    two_sided_alpha: float
    population_repetitions: int
    strength_ratios: tuple[float, ...]
    selected_instances: tuple[SelectedAuditInstance, ...]
    opportunity_keys: tuple[FinalStrengthAuditKey, ...]
    source_stratum_census: tuple[tuple[str, int], ...]
    selected_stratum_census: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name in (
            "population_digest",
            "evaluation_strata_digest",
            "context_digest",
            "selector_digest",
            "learned_selection_digest",
            "quality_authority_digest",
            "runtime_implementation_digest",
            "sampler_identity_digest",
            "audit_config_record_digest",
            "stock_tuning_execution_digest",
        ):
            _require_digest(getattr(self, name), label=f"audit plan {name}")
        quality = _canonical_mapping(self.quality_authority, label="quality authority")
        learned_selection = _canonical_mapping(
            self.learned_selection, label="frozen learned selection"
        )
        sampler = _canonical_mapping(self.sampler_identity, label="sampler identity")
        config = _canonical_mapping(self.audit_config, label="audit config")
        object.__setattr__(self, "quality_authority", quality)
        object.__setattr__(self, "learned_selection", learned_selection)
        object.__setattr__(self, "sampler_identity", sampler)
        object.__setattr__(self, "audit_config", config)
        if content_digest(sampler) != self.sampler_identity_digest:
            raise ValueError("audit plan sampler identity digest is inconsistent")
        observed_config_digest = (
            config.get("record_digest")
            if config.get("schema") == CONFIG_SCHEMA
            else content_digest(config)
        )
        if observed_config_digest != self.audit_config_record_digest:
            raise ValueError("audit plan config identity digest is inconsistent")
        if type(self.publication_eligible) is not bool:
            raise ValueError("audit plan publication eligibility must be Boolean")
        if self.publication_eligible:
            if not _is_digest(self.audit_config_file_sha256):
                raise ValueError("publication audit plan lacks a config whole-file pin")
            try:
                typed_config = FinalStrengthAuditConfig.from_mapping(config)
            except (TypeError, ValueError) as exc:
                raise ValueError("publication audit plan has an invalid registered config") from exc
            if typed_config.record_digest != self.audit_config_record_digest:
                raise ValueError("publication audit plan config record digest differs")
            if (
                typed_config.reads_per_block != self.reads_per_block
                or typed_config.audit_seed != self.audit_seed
                or typed_config.lineages_per_stratum_cap != self.lineages_per_stratum
                or typed_config.minimum_base_lineages != self.minimum_base_lineages
                or typed_config.shard_count != self.shard_count
                or typed_config.bootstrap_replicates != self.bootstrap_replicates
                or typed_config.bootstrap_seed != self.bootstrap_seed
                or typed_config.two_sided_alpha != self.two_sided_alpha
            ):
                raise ValueError("publication audit plan parameters differ from its config")
            _validate_quality_authority(quality)
            _validate_learned_selection(learned_selection)
            if content_digest(learned_selection) != self.learned_selection_digest:
                raise ValueError("publication audit plan learned selection digest differs")
            if learned_selection["runtime_implementation_digest"] != (
                self.runtime_implementation_digest
            ):
                raise ValueError("learned selection runtime differs from the audit plan")
            if content_digest(quality) != self.quality_authority_digest:
                raise ValueError("publication audit plan quality authority digest differs")
            expected_sampler = registered_final_strength_sampler_identity()
            if sampler != _canonical_mapping(
                expected_sampler.as_dict(), label="registered sampler identity"
            ):
                raise ValueError("publication audit plan changes the registered sampler identity")
        elif self.audit_config_file_sha256 is not None:
            _require_digest(self.audit_config_file_sha256, label="test audit config file")
        elif content_digest(learned_selection) != self.learned_selection_digest:
            raise ValueError("test audit plan learned selection digest differs")
        binding = _stock_binding(self.stock_tuning_execution)
        object.__setattr__(self, "stock_tuning_execution", binding)
        if content_digest(binding) != self.stock_tuning_execution_digest:
            raise ValueError("audit plan stock tuning digest is inconsistent")
        for name in (
            "reads_per_block",
            "num_sweeps",
            "lineages_per_stratum",
            "minimum_base_lineages",
            "population_repetitions",
            "shard_count",
            "bootstrap_replicates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"audit plan {name} must be a positive integer")
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**31:
            raise ValueError("audit plan bootstrap seed must be in [0, 2^31)")
        if type(self.two_sided_alpha) is not float or not 0.0 < self.two_sided_alpha < 1.0:
            raise ValueError("audit plan two-sided alpha must be a float in (0,1)")
        if self.publication_eligible and (
            self.reads_per_block != PRODUCTION_READS_PER_BLOCK
            or self.minimum_base_lineages < PRODUCTION_MINIMUM_BASE_LINEAGES
            or self.bootstrap_replicates != PRODUCTION_BOOTSTRAP_REPLICATES
            or self.shard_count <= 1
        ):
            raise ValueError("publication audit plan weakens registered production minima")
        if type(self.audit_seed) is not int or not 0 <= self.audit_seed < 2**31:
            raise ValueError("audit plan seed must be in [0, 2^31)")
        ratios = tuple(_finite(value, label="strength ratio") for value in self.strength_ratios)
        if len(ratios) != 4 or any(value <= 0.0 for value in ratios) or len(set(ratios)) != 4:
            raise ValueError("audit plan must retain four distinct positive strength ratios")
        object.__setattr__(self, "strength_ratios", ratios)
        selected = tuple(self.selected_instances)
        if not selected or any(not isinstance(row, SelectedAuditInstance) for row in selected):
            raise ValueError("audit plan must select typed instances")
        if selected != tuple(sorted(selected, key=lambda row: (row.stratum_id, row.selection_rank))):
            raise ValueError("audit plan selected instances are not canonically ordered")
        identities = [row.identity for row in selected]
        if len(identities) != len(set(identities)):
            raise ValueError("audit plan selects an instance more than once")
        by_stratum: dict[str, list[SelectedAuditInstance]] = defaultdict(list)
        for row in selected:
            by_stratum[row.stratum_id].append(row)
        if any(
            len({row.lineage for row in rows}) != len(rows)
            or tuple(row.selection_rank for row in rows) != tuple(range(len(rows)))
            for rows in by_stratum.values()
        ):
            raise ValueError("audit subset repeats a lineage or changes canonical ranks")
        if len({row.lineage for row in selected}) < self.minimum_base_lineages:
            raise ValueError("audit subset does not meet its minimum base-lineage census")
        census = tuple(self.source_stratum_census)
        if (
            not census
            or census != tuple(sorted(census))
            or len({name for name, _ in census}) != len(census)
            or any(not _is_digest(name) or type(count) is not int or count <= 0 for name, count in census)
            or set(name for name, _ in census) != set(by_stratum)
        ):
            raise ValueError("audit plan source-stratum census is malformed")
        selected_census = tuple(self.selected_stratum_census)
        expected_selected_census = tuple(
            sorted((name, len(rows)) for name, rows in by_stratum.items())
        )
        if (
            selected_census != expected_selected_census
            or any(
                selected_count != min(self.lineages_per_stratum, source_count)
                for (name, selected_count), (source_name, source_count) in zip(
                    selected_census, census, strict=True
                )
                if name == source_name
            )
        ):
            raise ValueError("audit selected-stratum census differs from capped stratification")
        expected_keys = tuple(
            sorted(
                FinalStrengthAuditKey(
                    row.lineage,
                    row.instance,
                    repetition,
                    seed,
                    arm,  # type: ignore[arg-type]
                )
                for row in selected
                for repetition in range(self.population_repetitions)
                for seed in REGISTERED_TRAINING_SEEDS
                for arm in ARMS
            )
        )
        if tuple(self.opportunity_keys) != expected_keys:
            raise ValueError("audit opportunity keys do not equal the sealed Cartesian subset")

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": PLAN_SCHEMA,
            "schema_version": PLAN_VERSION,
            "scope": "post-freeze-diagnostic-only",
            "outcome_blind": True,
            "primary_blocks_reused": False,
            "feedback_forbidden": True,
            "subset_protocol": SUBSET_DOMAIN,
            "aggregation": AGGREGATION,
            "block_domains": dict(BLOCK_DOMAINS),
            "population_digest": self.population_digest,
            "evaluation_strata_digest": self.evaluation_strata_digest,
            "context_digest": self.context_digest,
            "selector_digest": self.selector_digest,
            "learned_selection": _plain_json(self.learned_selection),
            "learned_selection_digest": self.learned_selection_digest,
            "quality_authority": _plain_json(self.quality_authority),
            "quality_authority_digest": self.quality_authority_digest,
            "runtime_implementation_digest": self.runtime_implementation_digest,
            "sampler_identity": _plain_json(self.sampler_identity),
            "sampler_identity_digest": self.sampler_identity_digest,
            "audit_config": _plain_json(self.audit_config),
            "audit_config_record_digest": self.audit_config_record_digest,
            "audit_config_file_sha256": self.audit_config_file_sha256,
            "publication_eligible": self.publication_eligible,
            "stock_tuning_execution": _plain_json(self.stock_tuning_execution),
            "stock_tuning_execution_digest": self.stock_tuning_execution_digest,
            "training_seeds": list(REGISTERED_TRAINING_SEEDS),
            "arms": list(ARMS),
            "reads_per_block": self.reads_per_block,
            "num_sweeps": self.num_sweeps,
            "audit_seed": self.audit_seed,
            "lineages_per_stratum": self.lineages_per_stratum,
            "minimum_base_lineages": self.minimum_base_lineages,
            "shard_count": self.shard_count,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "two_sided_alpha": self.two_sided_alpha,
            "population_repetitions": self.population_repetitions,
            "strength_ratios": list(self.strength_ratios),
            "stratum_axes": [*STRATUM_AXES, "size_bin"],
            "source_stratum_census": [list(item) for item in self.source_stratum_census],
            "selected_stratum_census": [
                list(item) for item in self.selected_stratum_census
            ],
            "selected_instances": [row.as_dict() for row in self.selected_instances],
            "opportunity_keys": [key.as_dict() for key in self.opportunity_keys],
            "opportunity_keys_digest": stable_digest(
                [key.as_dict() for key in self.opportunity_keys]
            ),
        }
        if include_digest:
            payload["record_digest"] = content_digest(payload)
        return payload

    @property
    def record_digest(self) -> str:
        return content_digest(self.as_dict(include_digest=False))

    def validate_population(self, population: CompletePopulationIdentity) -> None:
        if population.digest != self.population_digest:
            raise FinalStrengthAuditError("audit plan population digest mismatch")
        if population.evaluation_strata_digest != self.evaluation_strata_digest:
            raise FinalStrengthAuditError("audit plan evaluation-strata digest mismatch")
        if population.expected_repetitions != self.population_repetitions:
            raise FinalStrengthAuditError("audit plan repetition census mismatch")
        stratum_by_identity = {row.identity: row for row in population.evaluation_strata}
        if set(stratum_by_identity) != set(population.expected_instances):
            raise FinalStrengthAuditError("population strata do not cover expected instances")
        source_census: dict[str, set[str]] = defaultdict(set)
        for row in population.evaluation_strata:
            source_census[_stratum_id(_stratum_coordinates(row))].add(row.lineage)
        if tuple(sorted((key, len(value)) for key, value in source_census.items())) != (
            self.source_stratum_census
        ):
            raise FinalStrengthAuditError("audit plan source-stratum census changed")
        for selected in self.selected_instances:
            observed = stratum_by_identity.get(selected.identity)
            if (
                observed is None
                or observed.record_digest != selected.stratum_record_digest
                or _stratum_coordinates(observed) != selected.stratum_coordinates
            ):
                raise FinalStrengthAuditError("audit selected-instance stratum changed")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FinalStrengthAuditPlan:
        expected = {
            "schema",
            "schema_version",
            "scope",
            "outcome_blind",
            "primary_blocks_reused",
            "feedback_forbidden",
            "subset_protocol",
            "aggregation",
            "block_domains",
            "population_digest",
            "evaluation_strata_digest",
            "context_digest",
            "selector_digest",
            "learned_selection",
            "learned_selection_digest",
            "quality_authority",
            "quality_authority_digest",
            "runtime_implementation_digest",
            "sampler_identity",
            "sampler_identity_digest",
            "audit_config",
            "audit_config_record_digest",
            "audit_config_file_sha256",
            "publication_eligible",
            "stock_tuning_execution",
            "stock_tuning_execution_digest",
            "training_seeds",
            "arms",
            "reads_per_block",
            "num_sweeps",
            "audit_seed",
            "lineages_per_stratum",
            "minimum_base_lineages",
            "shard_count",
            "bootstrap_replicates",
            "bootstrap_seed",
            "two_sided_alpha",
            "population_repetitions",
            "strength_ratios",
            "stratum_axes",
            "source_stratum_census",
            "selected_stratum_census",
            "selected_instances",
            "opportunity_keys",
            "opportunity_keys_digest",
            "record_digest",
        }
        if set(value) != expected:
            raise ValueError("final-strength audit plan has missing or unknown fields")
        fixed = {
            "schema": PLAN_SCHEMA,
            "schema_version": PLAN_VERSION,
            "scope": "post-freeze-diagnostic-only",
            "outcome_blind": True,
            "primary_blocks_reused": False,
            "feedback_forbidden": True,
            "subset_protocol": SUBSET_DOMAIN,
            "aggregation": AGGREGATION,
            "block_domains": BLOCK_DOMAINS,
            "training_seeds": list(REGISTERED_TRAINING_SEEDS),
            "arms": list(ARMS),
            "stratum_axes": [*STRATUM_AXES, "size_bin"],
        }
        if any(value[name] != expected_value for name, expected_value in fixed.items()):
            raise ValueError("final-strength audit plan changes the registered protocol")
        raw_binding = value["stock_tuning_execution"]
        raw_quality = value["quality_authority"]
        raw_learned_selection = value["learned_selection"]
        raw_sampler = value["sampler_identity"]
        raw_config = value["audit_config"]
        raw_selected = value["selected_instances"]
        raw_keys = value["opportunity_keys"]
        raw_census = value["source_stratum_census"]
        raw_selected_census = value["selected_stratum_census"]
        raw_ratios = value["strength_ratios"]
        if (
            not isinstance(raw_binding, Mapping)
            or not isinstance(raw_quality, Mapping)
            or not isinstance(raw_learned_selection, Mapping)
            or not isinstance(raw_sampler, Mapping)
            or not isinstance(raw_config, Mapping)
            or not isinstance(raw_selected, list)
            or any(not isinstance(row, Mapping) for row in raw_selected)
            or not isinstance(raw_keys, list)
            or any(not isinstance(row, Mapping) for row in raw_keys)
            or not isinstance(raw_census, list)
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or type(row[1]) is not int
                for row in raw_census
            )
            or not isinstance(raw_selected_census, list)
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or type(row[1]) is not int
                for row in raw_selected_census
            )
            or not isinstance(raw_ratios, list)
        ):
            raise ValueError("final-strength audit plan arrays are malformed")
        result = cls(
            population_digest=value["population_digest"],  # type: ignore[arg-type]
            evaluation_strata_digest=value["evaluation_strata_digest"],  # type: ignore[arg-type]
            context_digest=value["context_digest"],  # type: ignore[arg-type]
            selector_digest=value["selector_digest"],  # type: ignore[arg-type]
            learned_selection=raw_learned_selection,
            learned_selection_digest=value["learned_selection_digest"],  # type: ignore[arg-type]
            quality_authority=raw_quality,
            quality_authority_digest=value["quality_authority_digest"],  # type: ignore[arg-type]
            runtime_implementation_digest=value["runtime_implementation_digest"],  # type: ignore[arg-type]
            sampler_identity=raw_sampler,
            sampler_identity_digest=value["sampler_identity_digest"],  # type: ignore[arg-type]
            audit_config=raw_config,
            audit_config_record_digest=value["audit_config_record_digest"],  # type: ignore[arg-type]
            audit_config_file_sha256=value["audit_config_file_sha256"],  # type: ignore[arg-type]
            publication_eligible=value["publication_eligible"],  # type: ignore[arg-type]
            stock_tuning_execution=raw_binding,
            stock_tuning_execution_digest=value["stock_tuning_execution_digest"],  # type: ignore[arg-type]
            reads_per_block=value["reads_per_block"],  # type: ignore[arg-type]
            num_sweeps=value["num_sweeps"],  # type: ignore[arg-type]
            audit_seed=value["audit_seed"],  # type: ignore[arg-type]
            lineages_per_stratum=value["lineages_per_stratum"],  # type: ignore[arg-type]
            minimum_base_lineages=value["minimum_base_lineages"],  # type: ignore[arg-type]
            shard_count=value["shard_count"],  # type: ignore[arg-type]
            bootstrap_replicates=value["bootstrap_replicates"],  # type: ignore[arg-type]
            bootstrap_seed=value["bootstrap_seed"],  # type: ignore[arg-type]
            two_sided_alpha=value["two_sided_alpha"],  # type: ignore[arg-type]
            population_repetitions=value["population_repetitions"],  # type: ignore[arg-type]
            strength_ratios=tuple(raw_ratios),  # type: ignore[arg-type]
            selected_instances=tuple(
                SelectedAuditInstance.from_mapping(row) for row in raw_selected
            ),
            opportunity_keys=tuple(FinalStrengthAuditKey.from_mapping(row) for row in raw_keys),
            source_stratum_census=tuple((row[0], row[1]) for row in raw_census),
            selected_stratum_census=tuple(
                (row[0], row[1]) for row in raw_selected_census
            ),
        )
        if (
            value["opportunity_keys_digest"]
            != stable_digest([key.as_dict() for key in result.opportunity_keys])
            or value["record_digest"] != result.record_digest
            or result.as_dict() != dict(value)
        ):
            raise ValueError("final-strength audit plan digest or canonical form is invalid")
        return result


@dataclass(frozen=True)
class AuthenticatedFinalStrengthAuditPlan:
    plan: FinalStrengthAuditPlan
    file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FinalStrengthAuditPlan):
            raise TypeError("authenticated audit plan has the wrong type")
        _require_digest(self.file_sha256, label="audit plan file")


@dataclass(frozen=True)
class FinalStrengthAuditResult:
    rows: tuple[Mapping[str, object], ...]
    aggregate: Mapping[str, object]
    source_runs: tuple[Mapping[str, object], ...]
    publication_eligible: bool = False
    execution_manifest_digest: str | None = None
    sampler_identity_digest: str | None = None
    source_shards: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("final-strength audit result cannot be empty")
        object.__setattr__(self, "rows", tuple(_canonical_mapping(row, label="audit row") for row in self.rows))
        object.__setattr__(self, "aggregate", _canonical_mapping(self.aggregate, label="audit aggregate"))
        object.__setattr__(
            self,
            "source_runs",
            tuple(_canonical_mapping(row, label="audit source run") for row in self.source_runs),
        )
        if type(self.publication_eligible) is not bool:
            raise ValueError("audit result publication eligibility must be Boolean")
        for name in ("execution_manifest_digest", "sampler_identity_digest"):
            value = getattr(self, name)
            if value is not None:
                _require_digest(value, label=f"audit result {name}")
        if self.publication_eligible and (
            self.execution_manifest_digest is None or self.sampler_identity_digest is None
        ):
            raise ValueError("publication audit result lacks execution or sampler identity")
        source_shards = tuple(
            _canonical_mapping(row, label="audit source shard")
            for row in self.source_shards
        )
        object.__setattr__(self, "source_shards", source_shards)


@dataclass(frozen=True)
class FinalStrengthAuditExecutionManifest:
    """Six-source-authenticated seed/program manifest sealed before audit reads."""

    plan_file_sha256: str
    plan_record_digest: str
    publication_eligible: bool
    source_runs: tuple[Mapping[str, object], ...]
    source_report_pins_digest: str
    target_access: Mapping[str, object]
    target_access_digest: str
    quality_authority: Mapping[str, object]
    quality_authority_digest: str
    runtime_implementation_digest: str
    sampler_identity: Mapping[str, object]
    sampler_identity_digest: str
    paired_validation: Mapping[str, object]
    paired_validation_digest: str
    full_primary_evaluator_seeds: tuple[int, ...]
    opportunity_execution: tuple[Mapping[str, object], ...]
    opportunity_execution_digest: str
    chronology: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "plan_file_sha256",
            "plan_record_digest",
            "source_report_pins_digest",
            "target_access_digest",
            "quality_authority_digest",
            "runtime_implementation_digest",
            "sampler_identity_digest",
            "paired_validation_digest",
            "opportunity_execution_digest",
        ):
            _require_digest(getattr(self, name), label=f"execution manifest {name}")
        if type(self.publication_eligible) is not bool:
            raise ValueError("execution manifest publication eligibility must be Boolean")
        sources = tuple(
            _canonical_mapping(row, label="execution source run") for row in self.source_runs
        )
        if len(sources) != 6:
            raise ValueError("execution manifest must bind exactly six source runs")
        source_keys = [
            (row.get("arm"), row.get("training_seed")) for row in sources
        ]
        expected_source_keys = [
            (arm, seed) for seed in REGISTERED_TRAINING_SEEDS for arm in ARMS
        ]
        if source_keys != expected_source_keys or len(set(source_keys)) != 6:
            raise ValueError("execution source runs do not form the exact paired seed census")
        source_pins = {
            f"{row['arm']}:{row['training_seed']}": row["report_file_sha256"]
            for row in sources
        }
        if content_digest(source_pins) != self.source_report_pins_digest:
            raise ValueError("execution source report pin digest differs")
        target_access = _validate_target_access(
            self.target_access,
            quality_authority=self.quality_authority,
        )
        if target_access["record_digest"] != self.target_access_digest:
            raise ValueError("execution target-access digest differs")
        for source in sources:
            for name in (
                "report_file_sha256",
                "report_record_digest",
                "terminal_evidence_file_sha256",
                "receipt_file_sha256",
                "outcome_file_sha256",
                "artifacts_digest",
                "evaluation_protocol_digest",
                "population_digest",
                "context_digest",
                "quality_authority_digest",
                "runtime_implementation_digest",
                "runtime_identity_digest",
                "selector_digest",
                "system_seed_schedule_digest",
                "evaluator_seed_schedule_digest",
            ):
                _require_digest(source.get(name), label=f"execution source {name}")
        quality = _canonical_mapping(self.quality_authority, label="manifest quality authority")
        sampler = _canonical_mapping(self.sampler_identity, label="manifest sampler identity")
        paired = _canonical_mapping(self.paired_validation, label="paired source validation")
        chronology = _canonical_mapping(self.chronology, label="manifest chronology")
        expected_chronology = {
            "plan_sealed_before_test_outcomes": True,
            "six_source_runs_authenticated_before_execution_manifest": True,
            "execution_manifest_sealed_before_audit_reads": True,
            "audit_feedback_forbidden": True,
        }
        if _plain_json(chronology) != expected_chronology:
            raise ValueError("execution manifest changes mandatory chronology")
        if self.publication_eligible:
            if content_digest(quality) != self.quality_authority_digest:
                raise ValueError("manifest quality authority digest differs")
        elif quality.get("opaque_quality_authority_digest") != self.quality_authority_digest:
            raise ValueError("test manifest opaque quality authority digest differs")
        if content_digest(sampler) != self.sampler_identity_digest:
            raise ValueError("manifest sampler identity digest differs")
        if content_digest(paired) != self.paired_validation_digest:
            raise ValueError("manifest paired validation digest differs")
        if self.publication_eligible:
            required_paired = {
                "mode": "registered-complete-paired-validator",
                "validator": "isingfold.rl.external_pairing.aggregate_learned_vs_stock",
            }
            if any(paired.get(name) != value for name, value in required_paired.items()) or any(
                not _is_digest(paired.get(name))
                for name in (
                    "aggregate_record_digest",
                    "matched_contract_digest",
                    "pair_census_digest",
                    "learned_aggregate_record_digest",
                    "frozen_external_selection_digest",
                    "complete_learned_selection_digest",
                )
            ):
                raise ValueError("publication manifest lacks registered paired validation")
        primary = tuple(self.full_primary_evaluator_seeds)
        if (
            primary != tuple(sorted(primary))
            or len(primary) != len(set(primary))
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in primary)
        ):
            raise ValueError("manifest primary evaluator seed census is malformed")
        execution = tuple(
            _canonical_mapping(row, label="opportunity execution")
            for row in self.opportunity_execution
        )
        if not execution or content_digest(
            [_plain_json(row) for row in execution]
        ) != self.opportunity_execution_digest:
            raise ValueError("manifest opportunity execution digest differs")
        allocated: list[int] = []
        expected_execution_fields = {
            "audit_key",
            "audit_key_digest",
            "valid_return",
            "source_report_digest",
            "source_evidence_file_sha256",
            "evidence_record_digest",
            "terminal_evidence_digest",
            "program_digests",
            "selected_program_digest",
            "deployed_strength_index",
            "seed_schedule",
            "seed_schedule_digest",
            "counterbalance",
            "semantic_roles",
            "record_digest",
        }
        observed_keys: list[FinalStrengthAuditKey] = []
        for row in execution:
            if set(row) != expected_execution_fields:
                raise ValueError("opportunity execution row has missing or unknown fields")
            raw_key = row.get("audit_key")
            if not isinstance(raw_key, Mapping):
                raise ValueError("opportunity execution row has no typed audit key")
            key = FinalStrengthAuditKey.from_mapping(raw_key)
            observed_keys.append(key)
            row_payload = {name: value for name, value in row.items() if name != "record_digest"}
            if (
                row.get("audit_key_digest") != key.digest
                or row.get("record_digest") != content_digest(row_payload)
                or type(row.get("valid_return")) is not bool
                or row.get("counterbalance")
                != "hash-ranked-block-and-strength-order-v1"
                or row.get("semantic_roles")
                != "block-a-oracle-choice-block-b-independent-estimate"
            ):
                raise ValueError("opportunity execution row identity or digest differs")
            if row["valid_return"] is True:
                programs = row.get("program_digests")
                deployed = row.get("deployed_strength_index")
                if (
                    not isinstance(programs, Sequence)
                    or len(programs) != 4
                    or any(not _is_digest(item) for item in programs)
                    or type(deployed) is not int
                    or deployed not in range(4)
                    or row.get("selected_program_digest") != programs[deployed]
                    or not _is_digest(row.get("terminal_evidence_digest"))
                ):
                    raise ValueError("valid execution row has malformed program pins")
            elif any(
                row.get(name) is not None
                for name in (
                    "program_digests",
                    "selected_program_digest",
                    "deployed_strength_index",
                    "terminal_evidence_digest",
                )
            ):
                raise ValueError("invalid execution row must not invent program pins")
            schedule = row.get("seed_schedule")
            if not isinstance(schedule, Sequence) or isinstance(
                schedule, (str, bytes, bytearray)
            ) or len(schedule) != 8:
                raise ValueError("every audit opportunity must preallocate eight sampler seeds")
            expected_cells = {(block, index) for block in BLOCK_DOMAINS for index in range(4)}
            observed_cells: set[tuple[object, object]] = set()
            ranks: list[int] = []
            for item in schedule:
                if not isinstance(item, Mapping) or set(item) != {
                    "block",
                    "domain",
                    "strength_index",
                    "seed",
                    "execution_rank",
                }:
                    raise ValueError("audit seed schedule cell is malformed")
                block = item["block"]
                strength_index = item["strength_index"]
                seed = item["seed"]
                rank = item["execution_rank"]
                if (
                    block not in BLOCK_DOMAINS
                    or item["domain"] != BLOCK_DOMAINS[block]  # type: ignore[index]
                    or type(strength_index) is not int
                    or strength_index not in range(4)
                    or type(seed) is not int
                    or not 0 <= seed < 2**31
                    or type(rank) is not int
                ):
                    raise ValueError("audit seed schedule cell changes registered semantics")
                observed_cells.add((block, strength_index))
                allocated.append(seed)
                ranks.append(rank)
            if observed_cells != expected_cells or sorted(ranks) != list(range(8)):
                raise ValueError("audit seed schedule is incomplete or not counterbalanced")
            if row.get("seed_schedule_digest") != content_digest(
                [_plain_json(item) for item in schedule]
            ):
                raise ValueError("audit opportunity seed schedule digest differs")
        if len(allocated) != len(set(allocated)) or set(allocated) & set(primary):
            raise ValueError("audit seed schedule reuses a sampler or primary evaluator seed")
        if len(observed_keys) != len(set(observed_keys)):
            raise ValueError("execution manifest duplicates an audit opportunity")
        object.__setattr__(self, "source_runs", sources)
        object.__setattr__(self, "target_access", target_access)
        object.__setattr__(self, "quality_authority", quality)
        object.__setattr__(self, "sampler_identity", sampler)
        object.__setattr__(self, "paired_validation", paired)
        object.__setattr__(self, "chronology", chronology)
        object.__setattr__(self, "full_primary_evaluator_seeds", primary)
        object.__setattr__(self, "opportunity_execution", execution)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": EXECUTION_MANIFEST_SCHEMA,
            "schema_version": EXECUTION_MANIFEST_VERSION,
            "scope": "post-freeze-diagnostic-only",
            "plan_file_sha256": self.plan_file_sha256,
            "plan_record_digest": self.plan_record_digest,
            "publication_eligible": self.publication_eligible,
            "source_runs": [_plain_json(row) for row in self.source_runs],
            "source_report_pins_digest": self.source_report_pins_digest,
            "target_access": _plain_json(self.target_access),
            "target_access_digest": self.target_access_digest,
            "quality_authority": _plain_json(self.quality_authority),
            "quality_authority_digest": self.quality_authority_digest,
            "runtime_implementation_digest": self.runtime_implementation_digest,
            "sampler_identity": _plain_json(self.sampler_identity),
            "sampler_identity_digest": self.sampler_identity_digest,
            "paired_validation": _plain_json(self.paired_validation),
            "paired_validation_digest": self.paired_validation_digest,
            "full_primary_evaluator_seeds": list(self.full_primary_evaluator_seeds),
            "opportunity_execution": [
                _plain_json(row) for row in self.opportunity_execution
            ],
            "opportunity_execution_digest": self.opportunity_execution_digest,
            "chronology": _plain_json(self.chronology),
        }
        if include_digest:
            payload["record_digest"] = content_digest(payload)
        return payload

    @property
    def record_digest(self) -> str:
        return content_digest(self.as_dict(include_digest=False))

    def validate_plan(self, sealed_plan: AuthenticatedFinalStrengthAuditPlan) -> None:
        plan = sealed_plan.plan
        if (
            self.plan_file_sha256 != sealed_plan.file_sha256
            or self.plan_record_digest != plan.record_digest
            or self.publication_eligible != plan.publication_eligible
            or self.quality_authority_digest
            != plan.quality_authority_digest
            or self.quality_authority != plan.quality_authority
            or self.runtime_implementation_digest
            != plan.runtime_implementation_digest
            or self.sampler_identity_digest
            != plan.sampler_identity_digest
            or self.sampler_identity != plan.sampler_identity
        ):
            raise FinalStrengthAuditError("execution manifest differs from the sealed plan")
        for source in self.source_runs:
            if (
                source["population_digest"] != plan.population_digest
                or source["context_digest"] != plan.context_digest
                or source["quality_authority_digest"] != plan.quality_authority_digest
                or source["runtime_implementation_digest"]
                != plan.runtime_implementation_digest
                or source["selector_digest"] != plan.selector_digest
            ):
                raise FinalStrengthAuditError(
                    "execution source changes population, ground, runtime, or selector authority"
                )
            selection = source["complete_frozen_selection"]
            if not isinstance(selection, Mapping):
                raise FinalStrengthAuditError("execution source selection is malformed")
            if source["arm"] == "learned" and plan.publication_eligible:
                if not _learned_binding_matches_plan(
                    selection,
                    plan.learned_selection,
                    seed=int(source["training_seed"]),
                ):
                    raise FinalStrengthAuditError(
                        "execution learned source differs from pre-test selection"
                    )
            elif source["arm"] == "tuned-stock" and _plain_json(selection) != _plain_json(
                plan.stock_tuning_execution
            ):
                raise FinalStrengthAuditError(
                    "execution stock source differs from frozen external selection"
                )
        if (
            self.paired_validation.get("frozen_external_selection_digest")
            != plan.stock_tuning_execution_digest
            or self.paired_validation.get("complete_learned_selection_digest")
            != plan.learned_selection_digest
        ):
            raise FinalStrengthAuditError(
                "execution paired validation changes a frozen arm selection"
            )
        keys = tuple(
            FinalStrengthAuditKey.from_mapping(row["audit_key"])  # type: ignore[arg-type]
            for row in self.opportunity_execution
        )
        if keys != plan.opportunity_keys:
            raise FinalStrengthAuditError("execution manifest changes the opportunity census")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> FinalStrengthAuditExecutionManifest:
        expected = {
            "schema",
            "schema_version",
            "scope",
            "plan_file_sha256",
            "plan_record_digest",
            "publication_eligible",
            "source_runs",
            "source_report_pins_digest",
            "target_access",
            "target_access_digest",
            "quality_authority",
            "quality_authority_digest",
            "runtime_implementation_digest",
            "sampler_identity",
            "sampler_identity_digest",
            "paired_validation",
            "paired_validation_digest",
            "full_primary_evaluator_seeds",
            "opportunity_execution",
            "opportunity_execution_digest",
            "chronology",
            "record_digest",
        }
        if set(value) != expected or (
            value.get("schema") != EXECUTION_MANIFEST_SCHEMA
            or value.get("schema_version") != EXECUTION_MANIFEST_VERSION
            or value.get("scope") != "post-freeze-diagnostic-only"
        ):
            raise ValueError("execution manifest has an unsupported schema")
        source_runs = value["source_runs"]
        opportunity_execution = value["opportunity_execution"]
        primary = value["full_primary_evaluator_seeds"]
        if (
            not isinstance(source_runs, list)
            or any(not isinstance(row, Mapping) for row in source_runs)
            or not isinstance(opportunity_execution, list)
            or any(not isinstance(row, Mapping) for row in opportunity_execution)
            or not isinstance(primary, list)
            or any(type(seed) is not int for seed in primary)
            or not isinstance(value["quality_authority"], Mapping)
            or not isinstance(value["target_access"], Mapping)
            or not isinstance(value["sampler_identity"], Mapping)
            or not isinstance(value["paired_validation"], Mapping)
            or not isinstance(value["chronology"], Mapping)
        ):
            raise ValueError("execution manifest collections are malformed")
        result = cls(
            plan_file_sha256=value["plan_file_sha256"],  # type: ignore[arg-type]
            plan_record_digest=value["plan_record_digest"],  # type: ignore[arg-type]
            publication_eligible=value["publication_eligible"],  # type: ignore[arg-type]
            source_runs=tuple(source_runs),
            source_report_pins_digest=value["source_report_pins_digest"],  # type: ignore[arg-type]
            target_access=value["target_access"],
            target_access_digest=value["target_access_digest"],  # type: ignore[arg-type]
            quality_authority=value["quality_authority"],
            quality_authority_digest=value["quality_authority_digest"],  # type: ignore[arg-type]
            runtime_implementation_digest=value["runtime_implementation_digest"],  # type: ignore[arg-type]
            sampler_identity=value["sampler_identity"],
            sampler_identity_digest=value["sampler_identity_digest"],  # type: ignore[arg-type]
            paired_validation=value["paired_validation"],
            paired_validation_digest=value["paired_validation_digest"],  # type: ignore[arg-type]
            full_primary_evaluator_seeds=tuple(primary),
            opportunity_execution=tuple(opportunity_execution),
            opportunity_execution_digest=value["opportunity_execution_digest"],  # type: ignore[arg-type]
            chronology=value["chronology"],
        )
        if result.as_dict() != dict(value):
            raise ValueError("execution manifest digest or canonical form is invalid")
        return result


@dataclass(frozen=True)
class AuthenticatedFinalStrengthAuditExecutionManifest:
    manifest: FinalStrengthAuditExecutionManifest
    file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, FinalStrengthAuditExecutionManifest):
            raise TypeError("authenticated execution manifest has the wrong type")
        _require_digest(self.file_sha256, label="execution manifest file")


@dataclass(frozen=True)
class _Opportunity:
    key: FinalStrengthAuditKey
    evidence: TerminalEvidence | None
    outcome: EpisodeOutcome
    evidence_record_digest: str
    source_report_digest: str
    source_evidence_file_sha256: str
    stratum: SelectedAuditInstance


def build_final_strength_audit_plan(
    population: CompletePopulationIdentity,
    context: Context,
    selector: FrozenComponentIdentity,
    *,
    quality_authority_digest: str | None = None,
    quality_authority: Mapping[str, object] | None = None,
    learned_selection: Mapping[str, object] | None = None,
    stock_tuning_execution: Mapping[str, object],
    reads_per_block: int | None = None,
    audit_seed: int | None = None,
    lineages_per_stratum_cap: int | None = None,
    minimum_base_lineages: int | None = None,
    shard_count: int | None = None,
    audit_config: AuthenticatedFinalStrengthAuditConfig | None = None,
) -> FinalStrengthAuditPlan:
    """Build a deterministic balanced subset using only sealed, outcome-blind metadata."""

    if not isinstance(population, CompletePopulationIdentity):
        raise TypeError("final-strength plan requires a complete population identity")
    if not isinstance(context, Context) or not isinstance(selector, FrozenComponentIdentity):
        raise TypeError("final-strength plan requires typed context and selector identities")
    publication_eligible = (
        audit_config is not None
        and quality_authority is not None
        and learned_selection is not None
    )
    if audit_config is not None:
        if not isinstance(audit_config, AuthenticatedFinalStrengthAuditConfig):
            raise TypeError("audit_config must be an authenticated final-strength config")
        registered = audit_config.config
        supplied = {
            "reads_per_block": reads_per_block,
            "audit_seed": audit_seed,
            "lineages_per_stratum_cap": lineages_per_stratum_cap,
            "minimum_base_lineages": minimum_base_lineages,
            "shard_count": shard_count,
        }
        expected_parameters = {
            "reads_per_block": registered.reads_per_block,
            "audit_seed": registered.audit_seed,
            "lineages_per_stratum_cap": registered.lineages_per_stratum_cap,
            "minimum_base_lineages": registered.minimum_base_lineages,
            "shard_count": registered.shard_count,
        }
        if any(
            supplied[name] is not None and supplied[name] != expected
            for name, expected in expected_parameters.items()
        ):
            raise ValueError("plan arguments differ from the authenticated audit config")
        reads_per_block = registered.reads_per_block
        audit_seed = registered.audit_seed
        lineages_per_stratum_cap = registered.lineages_per_stratum_cap
        minimum_base_lineages = registered.minimum_base_lineages
        shard_count = registered.shard_count
        bootstrap_replicates = registered.bootstrap_replicates
        bootstrap_seed = registered.bootstrap_seed
        two_sided_alpha = registered.two_sided_alpha
        config_identity = registered.as_dict()
        config_record_digest = registered.record_digest
        config_file_sha256: str | None = audit_config.file_sha256
    else:
        if any(
            value is None
            for value in (
                reads_per_block,
                audit_seed,
                lineages_per_stratum_cap,
                minimum_base_lineages,
            )
        ):
            raise ValueError("test-only compatibility plan needs all legacy audit parameters")
        shard_count = 1 if shard_count is None else shard_count
        if type(shard_count) is not int or shard_count <= 0:
            raise ValueError("test-only shard_count must be positive")
        bootstrap_replicates = 512
        bootstrap_seed = 91307
        two_sided_alpha = 0.05
        config_identity = {
            "schema": "isingfold.final-strength-audit-test-config",
            "schema_version": 1,
            "scope": "test-only-not-publication-eligible",
            "reads_per_block": reads_per_block,
            "audit_seed": audit_seed,
            "lineages_per_stratum_cap": lineages_per_stratum_cap,
            "minimum_base_lineages": minimum_base_lineages,
            "shard_count": shard_count,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "two_sided_alpha": two_sided_alpha,
            "sampler_id": REGISTERED_SAMPLER_ID,
        }
        config_record_digest = content_digest(config_identity)
        config_file_sha256 = None
    assert reads_per_block is not None
    assert audit_seed is not None
    assert lineages_per_stratum_cap is not None
    assert minimum_base_lineages is not None
    if quality_authority is not None:
        quality_identity = _validate_quality_authority(quality_authority)
        evaluation_authority = quality_identity["evaluation_partition"]
        assert isinstance(evaluation_authority, Mapping)
        ground_partition = evaluation_authority["ground_partition"]
        assert isinstance(ground_partition, Mapping)
        population_instances = sorted(instance for _, instance in population.expected_instances)
        if (
            evaluation_authority["target_count"] != len(population_instances)
            or ground_partition["instance_set_digest"]
            != content_digest(population_instances)
        ):
            raise ValueError(
                "quality authority ground-instance census differs from the sealed population"
            )
        observed_quality_digest = content_digest(quality_identity)
        if (
            quality_authority_digest is not None
            and quality_authority_digest != observed_quality_digest
        ):
            raise ValueError("quality authority mapping differs from its supplied digest")
        quality_authority_digest = observed_quality_digest
    else:
        if quality_authority_digest is None:
            raise ValueError("plan requires quality_authority or its test-only opaque digest")
        _require_digest(quality_authority_digest, label="quality authority")
        quality_identity = {
            "schema": "isingfold.final-strength-audit-test-quality-authority",
            "schema_version": 1,
            "scope": "test-only-opaque-digest-not-publication-eligible",
            "opaque_quality_authority_digest": quality_authority_digest,
        }
    if learned_selection is not None:
        learned_selection_identity = _validate_learned_selection(learned_selection)
    else:
        learned_selection_identity = {
            "schema": "isingfold.final-strength-audit-test-learned-selection",
            "schema_version": 1,
            "scope": "test-only-selector-identity-not-publication-eligible",
            "selector_digest": content_digest(selector.as_dict()),
        }
    if type(lineages_per_stratum_cap) is not int or lineages_per_stratum_cap <= 0:
        raise ValueError("lineages_per_stratum_cap must be positive")
    if type(minimum_base_lineages) is not int or minimum_base_lineages <= 0:
        raise ValueError("minimum_base_lineages must be positive")
    if type(audit_seed) is not int or not 0 <= audit_seed < 2**31:
        raise ValueError("audit seed must be in [0, 2^31)")
    binding = _stock_binding(stock_tuning_execution)
    groups: dict[str, list[EvaluationStratum]] = defaultdict(list)
    coordinates: dict[str, tuple[tuple[str, str], ...]] = {}
    for row in population.evaluation_strata:
        coordinate = _stratum_coordinates(row)
        identity = _stratum_id(coordinate)
        groups[identity].append(row)
        coordinates[identity] = coordinate
    unique_lineages = {
        identity: len({row.lineage for row in rows}) for identity, rows in groups.items()
    }
    if min(unique_lineages.values()) <= 0:
        raise ValueError("every audit stratum must contain at least one lineage")
    selected: list[SelectedAuditInstance] = []
    for stratum_id in sorted(groups):
        quota = min(lineages_per_stratum_cap, unique_lineages[stratum_id])
        candidates = sorted(
            groups[stratum_id],
            key=lambda row: (
                stable_digest(
                    {
                        "domain": SUBSET_DOMAIN,
                        "audit_seed": audit_seed,
                        "stratum_id": stratum_id,
                        "lineage": row.lineage,
                        "instance": row.instance,
                    }
                ),
                row.lineage,
                row.instance,
            ),
        )
        used_lineages: set[str] = set()
        for candidate in candidates:
            if candidate.lineage in used_lineages:
                continue
            rank = len(used_lineages)
            selected.append(
                SelectedAuditInstance(
                    lineage=candidate.lineage,
                    instance=candidate.instance,
                    stratum_id=stratum_id,
                    stratum_coordinates=coordinates[stratum_id],
                    stratum_record_digest=candidate.record_digest,
                    selection_rank=rank,
                )
            )
            used_lineages.add(candidate.lineage)
            if len(used_lineages) == quota:
                break
        if len(used_lineages) != quota:  # pragma: no cover - guarded by census above
            raise RuntimeError("deterministic audit subset selection failed its quota")
    selected.sort(key=lambda row: (row.stratum_id, row.selection_rank))
    distinct_lineages = {row.lineage for row in selected}
    if len(distinct_lineages) < minimum_base_lineages:
        raise ValueError(
            "balanced audit subset has too few independent base lineages: "
            f"required={minimum_base_lineages}, observed={len(distinct_lineages)}"
        )
    keys = tuple(
        sorted(
            FinalStrengthAuditKey(
                row.lineage,
                row.instance,
                repetition,
                seed,
                arm,  # type: ignore[arg-type]
            )
            for row in selected
            for repetition in range(population.expected_repetitions)
            for seed in REGISTERED_TRAINING_SEEDS
            for arm in ARMS
        )
    )
    runtime_digest = content_digest(runtime_implementation_registry())
    sampler_identity = registered_final_strength_sampler_identity()
    return FinalStrengthAuditPlan(
        population_digest=population.digest,
        evaluation_strata_digest=population.evaluation_strata_digest,
        context_digest=stable_digest(context_snapshot(context)),
        selector_digest=content_digest(selector.as_dict()),
        learned_selection=learned_selection_identity,
        learned_selection_digest=content_digest(learned_selection_identity),
        quality_authority=quality_identity,
        quality_authority_digest=quality_authority_digest,
        runtime_implementation_digest=runtime_digest,
        sampler_identity=sampler_identity.as_dict(),
        sampler_identity_digest=sampler_identity.digest,
        audit_config=config_identity,
        audit_config_record_digest=config_record_digest,
        audit_config_file_sha256=config_file_sha256,
        publication_eligible=publication_eligible,
        stock_tuning_execution=binding,
        stock_tuning_execution_digest=content_digest(binding),
        reads_per_block=reads_per_block,
        num_sweeps=context.num_sweeps,
        audit_seed=audit_seed,
        lineages_per_stratum=lineages_per_stratum_cap,
        minimum_base_lineages=minimum_base_lineages,
        shard_count=shard_count,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        two_sided_alpha=two_sided_alpha,
        population_repetitions=population.expected_repetitions,
        strength_ratios=context.strength_ratios,
        selected_instances=tuple(selected),
        opportunity_keys=keys,
        source_stratum_census=tuple(sorted(unique_lineages.items())),
        selected_stratum_census=tuple(
            sorted(
                (stratum_id, min(lineages_per_stratum_cap, source_count))
                for stratum_id, source_count in unique_lineages.items()
            )
        ),
    )


def write_final_strength_audit_plan(
    path: str | os.PathLike[str], plan: FinalStrengthAuditPlan
) -> str:
    """Publish one immutable canonical plan and return its externally recordable SHA-256."""

    if not isinstance(plan, FinalStrengthAuditPlan):
        raise TypeError("plan must be FinalStrengthAuditPlan")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"final-strength audit plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(plan.as_dict()) + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(content).hexdigest()


def load_final_strength_audit_plan(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> AuthenticatedFinalStrengthAuditPlan:
    """Load the plan only after checking an out-of-band whole-file SHA-256 pin."""

    _require_digest(expected_sha256, label="expected audit plan file")
    content = Path(path).read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise FinalStrengthAuditError(
            f"final-strength audit plan SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed}"
        )
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalStrengthAuditError("final-strength audit plan is not UTF-8 JSON") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) + b"\n" != content:
        raise FinalStrengthAuditError("final-strength audit plan is not canonical JSON")
    try:
        plan = FinalStrengthAuditPlan.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise FinalStrengthAuditError("final-strength audit plan is invalid") from exc
    return AuthenticatedFinalStrengthAuditPlan(plan, expected_sha256)


def _report_digest(report: Mapping[str, object], *, label: str) -> str:
    recorded = report.get("record_digest")
    payload = {name: value for name, value in report.items() if name != "record_digest"}
    if not _is_digest(recorded) or content_digest(payload) != recorded:
        raise FinalStrengthAuditError(f"{label} report digest mismatch")
    return recorded  # type: ignore[return-value]


def _report_pin(
    report: Mapping[str, object],
    *,
    plan: FinalStrengthAuditPlan,
    arm: Arm,
    training_seed: int,
) -> tuple[str, str]:
    if (
        report.get("partition") != "test"
        or report.get("sealed_test_opened") is not True
        or report.get("training_seed") != training_seed
        or report.get("training_seed_index") != REGISTERED_TRAINING_SEEDS.index(training_seed)
        or report.get("population_digest") != plan.population_digest
        or report.get("context_digest") != plan.context_digest
        or report.get("runtime_implementation_digest") != plan.runtime_implementation_digest
    ):
        raise FinalStrengthAuditError(f"{arm} report changes population/context/runtime/seed pins")
    selector = report.get("selector")
    quality = report.get("quality_authority")
    report_context = report.get("context")
    report_population = report.get("population")
    if (
        not isinstance(selector, Mapping)
        or content_digest(selector) != plan.selector_digest
        or (
            report.get("selector_digest") is not None
            and report.get("selector_digest") != plan.selector_digest
        )
        or not isinstance(quality, Mapping)
        or content_digest(quality) != plan.quality_authority_digest
        or not isinstance(report_context, Mapping)
        or stable_digest(_plain_json(report_context)) != plan.context_digest
        or not isinstance(report_population, Mapping)
        or stable_digest(_plain_json(report_population)) != plan.population_digest
    ):
        raise FinalStrengthAuditError(
            f"{arm} report changes selector, quality authority, context, or population"
        )
    registry = report.get("runtime_implementation_registry")
    if (
        not isinstance(registry, Mapping)
        or content_digest(registry) != plan.runtime_implementation_digest
    ):
        raise FinalStrengthAuditError(f"{arm} runtime registry digest mismatch")
    if arm == "tuned-stock":
        binding = report.get("external_tuning_execution")
        if (
            not isinstance(binding, Mapping)
            or _plain_json(binding) != _plain_json(plan.stock_tuning_execution)
            or report.get("external_tuning_execution_digest")
            != plan.stock_tuning_execution_digest
        ):
            raise FinalStrengthAuditError("stock run is not the presealed tuned deployment")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FinalStrengthAuditError(f"{arm} report has no artifact registry")
    entry = artifacts.get("terminal_evidence")
    if (
        not isinstance(entry, Mapping)
        or entry.get("path") != "terminal_evidence.jsonl"
        or not _is_digest(entry.get("sha256"))
    ):
        raise FinalStrengthAuditError(f"{arm} report has invalid evidence sidecar metadata")
    return _report_digest(report, label=arm), entry["sha256"]  # type: ignore[return-value]


def _source_runtime_identity(report: Mapping[str, object]) -> Mapping[str, object]:
    keys = (
        "runtime_platform",
        "inference_device_type",
        "inference_device_name",
        "inference_threads",
        "deterministic",
    )
    identity = _canonical_mapping(
        {name: report.get(name) for name in keys}, label="source runtime"
    )
    if (
        not isinstance(identity["runtime_platform"], Mapping)
        or not identity["runtime_platform"]
        or not isinstance(identity["inference_device_type"], str)
        or not identity["inference_device_type"]
        or not isinstance(identity["inference_device_name"], str)
        or not identity["inference_device_name"]
        or type(identity["inference_threads"]) is not int
        or identity["inference_threads"] <= 0
        or type(identity["deterministic"]) is not bool
    ):
        raise FinalStrengthAuditError("source runtime identity is malformed")
    return identity


def _evidence_sidecar_sha256(records: Sequence[object]) -> str:
    try:
        ordered = sorted(records, key=lambda row: row.pair_key)  # type: ignore[attr-defined]
        content = b"".join(
            (
                json.dumps(
                    row.as_dict(),  # type: ignore[attr-defined]
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            for row in ordered
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise FinalStrengthAuditError("terminal-evidence sidecar records are malformed") from exc
    return hashlib.sha256(content).hexdigest()


def _opportunities_from_runs(
    learned_runs: Sequence[AuthenticatedCompleteSystemSeedRun],
    stock_runs: Sequence[AuthenticatedExternalCompleteRun],
    *,
    plan: FinalStrengthAuditPlan,
) -> tuple[tuple[_Opportunity, ...], tuple[Mapping[str, object], ...], CompletePopulationIdentity]:
    if len(learned_runs) != 3 or len(stock_runs) != 3:
        raise FinalStrengthAuditError("final-strength audit requires exactly three runs per arm")
    if any(not isinstance(run, AuthenticatedCompleteSystemSeedRun) for run in learned_runs):
        raise TypeError("learned audit sources must be authenticated complete-system runs")
    if any(not isinstance(run, AuthenticatedExternalCompleteRun) for run in stock_runs):
        raise TypeError("stock audit sources must be authenticated external-complete runs")
    learned_by_seed = {int(run.report.get("training_seed", -1)): run for run in learned_runs}
    stock_by_seed = {int(run.report.get("training_seed", -1)): run for run in stock_runs}
    if tuple(sorted(learned_by_seed)) != REGISTERED_TRAINING_SEEDS or tuple(
        sorted(stock_by_seed)
    ) != REGISTERED_TRAINING_SEEDS:
        raise FinalStrengthAuditError("final-strength sources change the exact three seed census")
    sources: list[Mapping[str, object]] = []
    opportunities: dict[FinalStrengthAuditKey, _Opportunity] = {}
    population: CompletePopulationIdentity | None = None
    for seed in REGISTERED_TRAINING_SEEDS:
        paired_runtime: Mapping[str, object] | None = None
        for arm, run in (
            ("learned", learned_by_seed[seed]),
            ("tuned-stock", stock_by_seed[seed]),
        ):
            report = run.report
            report_digest, evidence_file_sha = _report_pin(
                report, plan=plan, arm=arm, training_seed=seed  # type: ignore[arg-type]
            )
            runtime = _source_runtime_identity(report)
            if paired_runtime is None:
                paired_runtime = runtime
            elif _plain_json(runtime) != _plain_json(paired_runtime):
                raise FinalStrengthAuditError(
                    "learned and tuned-stock primary runs used different paired runtimes"
                )
            receipts = {receipt.pair_key: receipt for receipt in run.receipts}
            evidence_records = {record.pair_key: record for record in run.evidence}
            if len(receipts) != len(run.receipts) or set(receipts) != set(evidence_records):
                raise FinalStrengthAuditError(f"{arm} receipt/evidence keys differ")
            if _evidence_sidecar_sha256(run.evidence) != evidence_file_sha:
                raise FinalStrengthAuditError(
                    f"{arm} typed evidence differs from the report's sidecar SHA-256"
                )
            reference = run.receipts[0].population
            if population is None:
                population = reference
                plan.validate_population(population)
            elif reference != population:
                raise FinalStrengthAuditError("final-strength source populations differ")
            expected_full = {
                (lineage, instance, repetition)
                for lineage, instance in reference.expected_instances
                for repetition in range(reference.expected_repetitions)
            }
            if set(receipts) != expected_full:
                raise FinalStrengthAuditError(f"{arm} source has an incomplete population census")
            if any(
                receipt.population != reference
                or receipt.selector != run.receipts[0].selector
                or receipt.context_digest != plan.context_digest
                for receipt in run.receipts
            ):
                raise FinalStrengthAuditError(
                    f"{arm} receipt batch mixes population, selector, or context identities"
                )
            if content_digest(run.receipts[0].selector.as_dict()) != plan.selector_digest:
                raise FinalStrengthAuditError(f"{arm} receipt selector differs from the plan")
            if arm == "tuned-stock":
                expected_runtime_identity_digest = content_digest(runtime)
                if any(
                    receipt.training_seed != seed
                    or receipt.quality_authority_digest != plan.quality_authority_digest
                    or receipt.runtime_identity_digest != expected_runtime_identity_digest
                    or receipt.tuning_execution is None
                    or _plain_json(receipt.tuning_execution.as_dict())
                    != _plain_json(plan.stock_tuning_execution)
                    or receipt.tuning_execution.digest
                    != plan.stock_tuning_execution_digest
                    for receipt in run.receipts
                ):
                    raise FinalStrengthAuditError(
                        "stock receipts change tuning, seed, quality, or runtime authority"
                    )
            for key in plan.opportunity_keys:
                if key.arm != arm or key.training_seed != seed:
                    continue
                pair_key = (key.lineage, key.instance, key.repetition)
                receipt = receipts.get(pair_key)
                evidence_record = evidence_records.get(pair_key)
                if receipt is None or evidence_record is None:
                    raise FinalStrengthAuditError("a sealed audit opportunity is absent")
                if receipt.outcome.returned_valid != (
                    evidence_record.terminal_evidence is not None
                ):
                    raise FinalStrengthAuditError("evidence presence differs from valid return")
                evidence_record_payload = evidence_record.as_dict()
                evidence_record_digest = evidence_record_payload.get("record_digest")
                if not _is_digest(evidence_record_digest):
                    raise FinalStrengthAuditError("evidence sidecar row lacks a record digest")
                opportunities[key] = _Opportunity(
                    key=key,
                    evidence=evidence_record.terminal_evidence,
                    outcome=receipt.outcome,
                    evidence_record_digest=evidence_record_digest,  # type: ignore[arg-type]
                    source_report_digest=report_digest,
                    source_evidence_file_sha256=evidence_file_sha,
                    stratum=next(
                        row
                        for row in plan.selected_instances
                        if row.identity == (key.lineage, key.instance)
                    ),
                )
            sources.append(
                _canonical_mapping(
                    {
                        "arm": arm,
                        "training_seed": seed,
                        "report_record_digest": report_digest,
                        "report_file_sha256": (
                            run.report_file_sha256
                            if isinstance(run, AuthenticatedCompleteSystemSeedRun)
                            else hashlib.sha256(
                                canonical_json_bytes(report) + b"\n"
                            ).hexdigest()
                        ),
                        "terminal_evidence_file_sha256": evidence_file_sha,
                        "runtime_identity": _plain_json(runtime),
                        "runtime_identity_digest": content_digest(runtime),
                    },
                    label="audit source run",
                )
            )
    if population is None or set(opportunities) != set(plan.opportunity_keys):
        raise FinalStrengthAuditError("final-strength audit source census differs from plan")
    return (
        tuple(opportunities[key] for key in plan.opportunity_keys),
        tuple(sorted(sources, key=lambda row: (str(row["training_seed"]), str(row["arm"])))),
        population,
    )


def _recompile_verified(
    opportunity: _Opportunity,
    *,
    task: EmbeddingTask,
    context: Context,
) -> tuple[Mapping[object, frozenset[object]], tuple[Program, ...]]:
    evidence = opportunity.evidence
    if evidence is None:
        raise TypeError("cannot recompile an invalid opportunity")
    verify_terminal_evidence(evidence, task=task, context=context, outcome=opportunity.outcome)
    logical = {_typed_identity(node): node for node in task.logical.nodes()}
    host = {_typed_identity(node): node for node in task.host.nodes()}
    try:
        chains = {
            logical[owner]: frozenset(host[qubit] for qubit in qubits)
            for owner, qubits in evidence.embedding
        }
    except KeyError as exc:  # pragma: no cover - independent verifier already catches this
        raise FinalStrengthAuditError("verified evidence cannot be resolved") from exc
    validation, programs = p_return(chains, task.logical, task.host, task.problem, context)
    if not validation.valid or len(programs) != 4:
        raise FinalStrengthAuditError("verified evidence did not produce four valid programs")
    return chains, programs


def _source_artifact_sha(
    report: Mapping[str, object], name: str, *, arm: Arm
) -> str:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FinalStrengthAuditError(f"{arm} report has no artifact registry")
    entry = artifacts.get(name)
    if not isinstance(entry, Mapping) or not _is_digest(entry.get("sha256")):
        raise FinalStrengthAuditError(f"{arm} report has no authenticated {name} artifact")
    return entry["sha256"]  # type: ignore[return-value]


def _source_identity(
    run: AuthenticatedCompleteSystemSeedRun | AuthenticatedExternalCompleteRun,
    *,
    arm: Arm,
    report_file_sha256_pin: str,
    strict_publication: bool,
) -> Mapping[str, object]:
    report = run.report
    observed_report_sha = run.report_file_sha256
    if report_file_sha256_pin != observed_report_sha:
        raise FinalStrengthAuditError(
            f"{arm} source report differs from its explicit out-of-band SHA-256 pin"
        )
    report_digest = _report_digest(report, label=arm)
    artifacts = report.get("artifacts")
    protocol = report.get("evaluation_protocol")
    selector = report.get("selector")
    quality = report.get("quality_authority")
    if not all(
        isinstance(value, Mapping)
        for value in (artifacts, protocol, selector, quality)
    ):
        raise FinalStrengthAuditError(f"{arm} source report omits a required identity")
    receipt_name = "complete_receipts" if arm == "learned" else "receipts"
    if (
        not strict_publication
        and arm == "tuned-stock"
        and receipt_name not in artifacts
    ):
        receipt_name = "complete_receipts"
    receipt_sha = _source_artifact_sha(report, receipt_name, arm=arm)
    evidence_sha = _source_artifact_sha(report, "terminal_evidence", arm=arm)
    outcome_sha = _source_artifact_sha(report, "outcomes", arm=arm)
    if strict_publication and isinstance(run, AuthenticatedExternalCompleteRun) and (
        receipt_sha != run.receipt_file_sha256
        or evidence_sha != run.evidence_file_sha256
        or outcome_sha != run.outcome_file_sha256
    ):
        raise FinalStrengthAuditError("stock typed artifact pins differ from its report")
    runtime = _source_runtime_identity(report)
    system_schedule = [
        [*receipt.pair_key, getattr(receipt, "system_seed", None)]
        for receipt in sorted(run.receipts, key=lambda row: row.pair_key)
    ]
    evaluator_schedule = [
        [
            *receipt.pair_key,
            getattr(receipt, "evaluator_seed", receipt.outcome.evaluator_seed),
        ]
        for receipt in sorted(run.receipts, key=lambda row: row.pair_key)
    ]
    selection = (
        report.get("selection_binding")
        if arm == "learned"
        else report.get("external_tuning_execution")
    )
    if not isinstance(selection, Mapping):
        raise FinalStrengthAuditError(f"{arm} source lacks its complete frozen selection")
    controller = report.get("controller") if arm == "learned" else None
    if arm == "learned" and not isinstance(controller, Mapping):
        raise FinalStrengthAuditError("learned source lacks its frozen controller identity")
    payload = {
        "arm": arm,
        "training_seed_index": report.get("training_seed_index"),
        "training_seed": report.get("training_seed"),
        "report_schema": report.get("schema"),
        "report_schema_version": report.get("schema_version"),
        "report_file_sha256": report_file_sha256_pin,
        "report_record_digest": report_digest,
        "receipt_file_sha256": receipt_sha,
        "terminal_evidence_file_sha256": evidence_sha,
        "outcome_file_sha256": outcome_sha,
        "artifacts": _plain_json(artifacts),
        "artifacts_digest": content_digest(artifacts),
        "evaluation_protocol": _plain_json(protocol),
        "evaluation_protocol_digest": content_digest(protocol),
        "population_digest": report.get("population_digest"),
        "context_digest": report.get("context_digest"),
        "quality_authority_digest": content_digest(quality),
        "runtime_implementation_digest": report.get("runtime_implementation_digest"),
        "runtime_identity": _plain_json(runtime),
        "runtime_identity_digest": content_digest(runtime),
        "selector": _plain_json(selector),
        "selector_digest": content_digest(selector),
        "complete_frozen_selection": _plain_json(selection),
        "complete_frozen_selection_digest": content_digest(selection),
        "controller": None if controller is None else _plain_json(controller),
        "controller_digest": None if controller is None else content_digest(controller),
        "system_seed_schedule_digest": content_digest(system_schedule),
        "evaluator_seed_schedule_digest": content_digest(evaluator_schedule),
    }
    return _canonical_mapping(payload, label="complete final-strength source identity")


def _allocate_audit_seed(
    *,
    plan: FinalStrengthAuditPlan,
    key: FinalStrengthAuditKey,
    block: Literal["A", "B"],
    strength_index: int,
    used: set[int],
) -> int:
    nonce = 0
    while nonce < 2**20:
        payload = {
            "domain": "isingfold-final-strength-global-seed-allocation-v1",
            "block_domain": BLOCK_DOMAINS[block],
            "audit_seed": plan.audit_seed,
            "audit_key": key.as_dict(),
            "strength_index": strength_index,
            "nonce": nonce,
        }
        candidate = int.from_bytes(
            hashlib.sha256(canonical_json_bytes(payload)).digest()[:4], "big"
        ) % (2**31)
        if candidate not in used:
            used.add(candidate)
            return candidate
        nonce += 1
    raise FinalStrengthAuditError("global 31-bit audit seed schedule is exhausted")


def _counterbalanced_schedule(
    plan: FinalStrengthAuditPlan,
    key: FinalStrengthAuditKey,
    *,
    used: set[int],
) -> list[dict[str, object]]:
    cells = [(block, strength) for block in BLOCK_DOMAINS for strength in range(4)]
    execution_order = sorted(
        cells,
        key=lambda cell: stable_digest(
            {
                "domain": "isingfold-final-strength-counterbalance-v1",
                "audit_seed": plan.audit_seed,
                "audit_key": key.as_dict(),
                "block": cell[0],
                "strength_index": cell[1],
            }
        ),
    )
    ranks = {cell: rank for rank, cell in enumerate(execution_order)}
    return [
        {
            "block": block,
            "domain": BLOCK_DOMAINS[block],
            "strength_index": strength,
            "seed": _allocate_audit_seed(
                plan=plan,
                key=key,
                block=block,  # type: ignore[arg-type]
                strength_index=strength,
                used=used,
            ),
            "execution_rank": ranks[(block, strength)],
        }
        for block, strength in cells
    ]


def build_final_strength_audit_execution_manifest(
    sealed_plan: AuthenticatedFinalStrengthAuditPlan,
    learned_runs: Sequence[AuthenticatedCompleteSystemSeedRun],
    stock_runs: Sequence[AuthenticatedExternalCompleteRun],
    tasks: Sequence[EmbeddingTask],
    context: Context,
    *,
    target_access: Mapping[str, object],
    source_report_sha256_pins: Mapping[str, str],
) -> FinalStrengthAuditExecutionManifest:
    """Authenticate six frozen sources and preseal every audit program and random seed."""

    if not isinstance(sealed_plan, AuthenticatedFinalStrengthAuditPlan):
        raise TypeError("execution manifest requires an authenticated audit plan")
    plan = sealed_plan.plan
    if stable_digest(context_snapshot(context)) != plan.context_digest:
        raise FinalStrengthAuditError("execution context differs from the audit plan")
    current_runtime = content_digest(runtime_implementation_registry())
    if current_runtime != plan.runtime_implementation_digest:
        raise FinalStrengthAuditError("runtime source changed after plan sealing")
    pins = _canonical_mapping(source_report_sha256_pins, label="source report pins")
    expected_pin_keys = {
        f"{arm}:{seed}" for seed in REGISTERED_TRAINING_SEEDS for arm in ARMS
    }
    if set(pins) != expected_pin_keys or any(not _is_digest(value) for value in pins.values()):
        raise FinalStrengthAuditError("execution manifest needs exactly six report SHA-256 pins")
    opportunities, _, population = _opportunities_from_runs(
        learned_runs, stock_runs, plan=plan
    )
    population.validate_tasks(tasks)
    authenticated_target_access = _validate_target_access(
        target_access,
        source_manifest_sha256=population.source_manifest_sha256,
        quality_authority=plan.quality_authority,
    )
    if authenticated_target_access["target_count"] != len(
        population.expected_instances
    ):
        raise ValueError(
            "test target count differs from the complete-system population denominator"
        )
    task_by_identity = {(task.lineage or task.name, task.name): task for task in tasks}
    if len(task_by_identity) != len(tasks):
        raise FinalStrengthAuditError("execution task set has duplicate identities")
    source_runs: list[Mapping[str, object]] = []
    learned_by_seed = {int(run.report["training_seed"]): run for run in learned_runs}
    stock_by_seed = {int(run.report["training_seed"]): run for run in stock_runs}
    for seed in REGISTERED_TRAINING_SEEDS:
        for arm, run in (
            ("learned", learned_by_seed[seed]),
            ("tuned-stock", stock_by_seed[seed]),
        ):
            source_runs.append(
                _source_identity(
                    run,
                    arm=arm,  # type: ignore[arg-type]
                    report_file_sha256_pin=pins[f"{arm}:{seed}"],  # type: ignore[arg-type]
                    strict_publication=plan.publication_eligible,
                )
            )
            if arm == "learned" and plan.publication_eligible:
                binding = run.report.get("selection_binding")
                if not isinstance(binding, Mapping) or not _learned_binding_matches_plan(
                    binding, plan.learned_selection, seed=seed
                ):
                    raise FinalStrengthAuditError(
                        "learned report selection differs from the pre-test frozen selection"
                    )
    if plan.publication_eligible:
        try:
            paired = aggregate_learned_vs_stock(learned_runs, stock_runs)
        except (TypeError, ValueError) as exc:
            raise FinalStrengthAuditError(
                "six source runs fail the registered learned-versus-stock validator"
            ) from exc
        paired_validation = {
            "mode": "registered-complete-paired-validator",
            "validator": "isingfold.rl.external_pairing.aggregate_learned_vs_stock",
            "aggregate_record_digest": paired["record_digest"],
            "matched_contract_digest": content_digest(paired["matched_contract"]),
            "pair_census_digest": paired["pair_census_digest"],
            "learned_aggregate_record_digest": paired["learned_aggregate_record_digest"],
            "frozen_external_selection_digest": plan.stock_tuning_execution_digest,
            "complete_learned_selection_digest": plan.learned_selection_digest,
        }
    else:
        # Compatibility fixtures may predate publication-grade external reports.  They are
        # deliberately and permanently labelled ineligible rather than being promoted by a
        # dependency-injected sampler or a forged report shape.
        try:
            learned_aggregate = aggregate_complete_system_seeds(learned_runs)
            learned_digest: object = learned_aggregate["record_digest"]
        except (TypeError, ValueError):
            learned_digest = None
        paired_validation = {
            "mode": "test-only-no-publication-paired-validator",
            "validator": None,
            "aggregate_record_digest": None,
            "matched_contract_digest": None,
            "pair_census_digest": None,
            "learned_aggregate_record_digest": learned_digest,
            "frozen_external_selection_digest": plan.stock_tuning_execution_digest,
            "complete_learned_selection_digest": plan.learned_selection_digest,
        }
    primary_seeds = tuple(
        sorted(
            {
                int(getattr(receipt, "evaluator_seed", receipt.outcome.evaluator_seed))
                for run in (*learned_runs, *stock_runs)
                for receipt in run.receipts
                if getattr(receipt, "evaluator_seed", receipt.outcome.evaluator_seed)
                is not None
            }
        )
    )
    used_seeds = set(primary_seeds)
    execution_rows: list[Mapping[str, object]] = []
    for opportunity in opportunities:
        task = task_by_identity.get((opportunity.key.lineage, opportunity.key.instance))
        if task is None:
            raise FinalStrengthAuditError("manifest opportunity has no authenticated task")
        if opportunity.evidence is None:
            program_digests: list[str] | None = None
            selected_program_digest: str | None = None
            terminal_evidence_digest: str | None = None
            deployed_strength_index: int | None = None
        else:
            chains, programs = _recompile_verified(
                opportunity, task=task, context=context
            )
            program_digests = [program_digest(program, chains) for program in programs]
            selected_program_digest = opportunity.evidence.selected_program_digest
            terminal_evidence_digest = opportunity.evidence.digest
            deployed_strength_index = opportunity.evidence.selected_index
            if program_digests[opportunity.evidence.selected_index] != selected_program_digest:
                raise FinalStrengthAuditError("manifest program pins differ from terminal evidence")
        schedule = _counterbalanced_schedule(plan, opportunity.key, used=used_seeds)
        row_payload = {
            "audit_key": opportunity.key.as_dict(),
            "audit_key_digest": opportunity.key.digest,
            "valid_return": opportunity.evidence is not None,
            "source_report_digest": opportunity.source_report_digest,
            "source_evidence_file_sha256": opportunity.source_evidence_file_sha256,
            "evidence_record_digest": opportunity.evidence_record_digest,
            "terminal_evidence_digest": terminal_evidence_digest,
            "program_digests": program_digests,
            "selected_program_digest": selected_program_digest,
            "deployed_strength_index": deployed_strength_index,
            "seed_schedule": schedule,
            "seed_schedule_digest": content_digest(schedule),
            "counterbalance": "hash-ranked-block-and-strength-order-v1",
            "semantic_roles": "block-a-oracle-choice-block-b-independent-estimate",
        }
        execution_rows.append(
            _canonical_mapping(
                {**row_payload, "record_digest": content_digest(row_payload)},
                label="opportunity execution manifest row",
            )
        )
    paired_identity = _canonical_mapping(paired_validation, label="paired validation")
    execution_digest = content_digest([_plain_json(row) for row in execution_rows])
    return FinalStrengthAuditExecutionManifest(
        plan_file_sha256=sealed_plan.file_sha256,
        plan_record_digest=plan.record_digest,
        publication_eligible=plan.publication_eligible,
        source_runs=tuple(source_runs),
        source_report_pins_digest=content_digest(pins),
        target_access=authenticated_target_access,
        target_access_digest=str(authenticated_target_access["record_digest"]),
        quality_authority=plan.quality_authority,
        quality_authority_digest=plan.quality_authority_digest,
        runtime_implementation_digest=plan.runtime_implementation_digest,
        sampler_identity=plan.sampler_identity,
        sampler_identity_digest=plan.sampler_identity_digest,
        paired_validation=paired_identity,
        paired_validation_digest=content_digest(paired_identity),
        full_primary_evaluator_seeds=primary_seeds,
        opportunity_execution=tuple(execution_rows),
        opportunity_execution_digest=execution_digest,
        chronology=FinalStrengthAuditConfig._CHRONOLOGY,
    )


def write_final_strength_audit_execution_manifest(
    path: str | os.PathLike[str], manifest: FinalStrengthAuditExecutionManifest
) -> str:
    """Atomically seal a canonical six-source execution manifest."""

    if not isinstance(manifest, FinalStrengthAuditExecutionManifest):
        raise TypeError("manifest must be FinalStrengthAuditExecutionManifest")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"final-strength execution manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(manifest.as_dict()) + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(content).hexdigest()


def load_final_strength_audit_execution_manifest(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    sealed_plan: AuthenticatedFinalStrengthAuditPlan,
) -> AuthenticatedFinalStrengthAuditExecutionManifest:
    """Load an execution manifest only under whole-file and sealed-plan pins."""

    _require_digest(expected_sha256, label="expected execution manifest file")
    content = Path(path).read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise FinalStrengthAuditError(
            f"execution manifest SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalStrengthAuditError("execution manifest is not UTF-8 JSON") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) + b"\n" != content:
        raise FinalStrengthAuditError("execution manifest is not canonical JSON")
    try:
        manifest = FinalStrengthAuditExecutionManifest.from_mapping(payload)
        manifest.validate_plan(sealed_plan)
    except (TypeError, ValueError) as exc:
        raise FinalStrengthAuditError("execution manifest is invalid") from exc
    return AuthenticatedFinalStrengthAuditExecutionManifest(manifest, expected_sha256)


@dataclass(frozen=True)
class FinalStrengthAuditShardResult:
    shard_index: int
    shard_count: int
    paired_keys: tuple[tuple[str, str, int, int], ...]
    rows: tuple[Mapping[str, object], ...]
    source_runs: tuple[Mapping[str, object], ...]
    plan_record_digest: str
    execution_manifest_digest: str
    execution_manifest_file_sha256: str
    sampler_identity: Mapping[str, object]
    sampler_identity_digest: str
    registered_sampler_used: bool
    publication_eligible: bool

    def __post_init__(self) -> None:
        if (
            type(self.shard_index) is not int
            or type(self.shard_count) is not int
            or self.shard_count <= 0
            or not 0 <= self.shard_index < self.shard_count
        ):
            raise ValueError("audit shard index/count is malformed")
        paired_keys = tuple(self.paired_keys)
        if not paired_keys or paired_keys != tuple(sorted(set(paired_keys))):
            raise ValueError("audit shard paired keys are empty, duplicate, or unordered")
        rows = tuple(_canonical_mapping(row, label="audit shard row") for row in self.rows)
        if not rows:
            raise ValueError("audit shard cannot be empty")
        sources = tuple(
            _canonical_mapping(row, label="audit shard source") for row in self.source_runs
        )
        if len(sources) != 6:
            raise ValueError("audit shard must bind all six source runs")
        sampler = _canonical_mapping(self.sampler_identity, label="shard sampler identity")
        for name in (
            "plan_record_digest",
            "execution_manifest_digest",
            "execution_manifest_file_sha256",
            "sampler_identity_digest",
        ):
            _require_digest(getattr(self, name), label=f"audit shard {name}")
        if content_digest(sampler) != self.sampler_identity_digest:
            raise ValueError("audit shard sampler digest differs")
        if type(self.registered_sampler_used) is not bool or type(
            self.publication_eligible
        ) is not bool:
            raise ValueError("audit shard eligibility flags must be Boolean")
        if self.publication_eligible and not self.registered_sampler_used:
            raise ValueError("dependency-injected sampler cannot be publication eligible")
        object.__setattr__(self, "paired_keys", paired_keys)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "source_runs", sources)
        object.__setattr__(self, "sampler_identity", sampler)


@dataclass(frozen=True)
class AuthenticatedFinalStrengthAuditShard:
    receipt: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]
    receipt_file_sha256: str
    rows_file_sha256: str

    def __post_init__(self) -> None:
        receipt = _canonical_mapping(self.receipt, label="audit shard receipt")
        rows = tuple(_canonical_mapping(row, label="authenticated shard row") for row in self.rows)
        _require_digest(self.receipt_file_sha256, label="audit shard receipt file")
        _require_digest(self.rows_file_sha256, label="audit shard rows file")
        receipt_bytes = canonical_json_bytes(receipt) + b"\n"
        rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        if hashlib.sha256(receipt_bytes).hexdigest() != self.receipt_file_sha256:
            raise ValueError("authenticated shard receipt file identity differs")
        if hashlib.sha256(rows_bytes).hexdigest() != self.rows_file_sha256:
            raise ValueError("authenticated shard rows file identity differs")
        raw_entry = receipt.get("raw_rows")
        if (
            not isinstance(raw_entry, Mapping)
            or raw_entry.get("sha256") != self.rows_file_sha256
            or raw_entry.get("count") != len(rows)
        ):
            raise ValueError("authenticated shard receipt does not bind its row artifact")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "rows", rows)


def _all_paired_keys(plan: FinalStrengthAuditPlan) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(sorted({key.paired_key for key in plan.opportunity_keys}))


def _shard_paired_keys(
    plan: FinalStrengthAuditPlan, *, shard_index: int
) -> tuple[tuple[str, str, int, int], ...]:
    if type(shard_index) is not int or not 0 <= shard_index < plan.shard_count:
        raise ValueError("audit shard index is outside the sealed shard census")
    pairs = _all_paired_keys(plan)
    if plan.shard_count > len(pairs):
        raise FinalStrengthAuditError(
            "sealed shard count exceeds paired opportunities; regenerate the pre-test plan"
        )
    selected = pairs[shard_index :: plan.shard_count]
    if not selected:  # pragma: no cover - count guard above
        raise FinalStrengthAuditError("deterministic shard assignment produced an empty shard")
    return selected


def _test_sampler_identity(
    sampler: Sampler, plan: FinalStrengthAuditPlan
) -> Mapping[str, object]:
    payload = {
        "schema": "isingfold.final-strength-audit-test-sampler",
        "schema_version": 1,
        "scope": "dependency-injected-test-only-not-publication-eligible",
        "callable_module": getattr(sampler, "__module__", type(sampler).__module__),
        "callable_qualname": getattr(sampler, "__qualname__", type(sampler).__qualname__),
        "registered_sampler_identity_digest": plan.sampler_identity_digest,
        "runtime_implementation_digest": plan.runtime_implementation_digest,
    }
    return _canonical_mapping(payload, label="test sampler identity")


def _execution_source_pins(
    manifest: FinalStrengthAuditExecutionManifest,
) -> dict[str, str]:
    return {
        f"{source['arm']}:{source['training_seed']}": str(source["report_file_sha256"])
        for source in manifest.source_runs
    }


def _reauthenticate_execution_inputs(
    sealed_plan: AuthenticatedFinalStrengthAuditPlan,
    sealed_execution_manifest: AuthenticatedFinalStrengthAuditExecutionManifest,
    learned_runs: Sequence[AuthenticatedCompleteSystemSeedRun],
    stock_runs: Sequence[AuthenticatedExternalCompleteRun],
    tasks: Sequence[EmbeddingTask],
    context: Context,
) -> tuple[_Opportunity, ...]:
    if not isinstance(
        sealed_execution_manifest, AuthenticatedFinalStrengthAuditExecutionManifest
    ):
        raise TypeError("shard execution requires an authenticated execution manifest")
    manifest = sealed_execution_manifest.manifest
    manifest.validate_plan(sealed_plan)
    rebuilt = build_final_strength_audit_execution_manifest(
        sealed_plan,
        learned_runs,
        stock_runs,
        tasks,
        context,
        target_access=manifest.target_access,
        source_report_sha256_pins=_execution_source_pins(manifest),
    )
    if rebuilt.as_dict() != manifest.as_dict():
        raise FinalStrengthAuditError(
            "source/program/schedule identity drifted after execution-manifest sealing"
        )
    opportunities, _, _ = _opportunities_from_runs(
        learned_runs, stock_runs, plan=sealed_plan.plan
    )
    return opportunities


def run_final_strength_audit_shard(
    sealed_plan: AuthenticatedFinalStrengthAuditPlan,
    sealed_execution_manifest: AuthenticatedFinalStrengthAuditExecutionManifest,
    learned_runs: Sequence[AuthenticatedCompleteSystemSeedRun],
    stock_runs: Sequence[AuthenticatedExternalCompleteRun],
    tasks: Sequence[EmbeddingTask],
    context: Context,
    *,
    shard_index: int,
    sampler: Sampler = sample_program,
) -> FinalStrengthAuditShardResult:
    """Run one deterministic paired-key shard under a sealed six-source manifest."""

    plan = sealed_plan.plan
    paired_keys = _shard_paired_keys(plan, shard_index=shard_index)
    opportunities = _reauthenticate_execution_inputs(
        sealed_plan,
        sealed_execution_manifest,
        learned_runs,
        stock_runs,
        tasks,
        context,
    )
    manifest = sealed_execution_manifest.manifest
    execution_by_key = {
        FinalStrengthAuditKey.from_mapping(row["audit_key"]): row  # type: ignore[arg-type]
        for row in manifest.opportunity_execution
    }
    task_by_identity = {(task.lineage or task.name, task.name): task for task in tasks}
    selected = [row for row in opportunities if row.key.paired_key in set(paired_keys)]
    expected_keys = tuple(
        key for key in plan.opportunity_keys if key.paired_key in set(paired_keys)
    )
    if tuple(row.key for row in selected) != expected_keys:
        raise FinalStrengthAuditError("shard opportunity assignment differs from the sealed plan")
    registered_sampler_used = sampler is sample_program
    sampler_identity = (
        plan.sampler_identity
        if registered_sampler_used
        else _test_sampler_identity(sampler, plan)
    )
    sampler_digest = content_digest(sampler_identity)
    rows: list[dict[str, object]] = []
    for opportunity in selected:
        entry = execution_by_key[opportunity.key]
        task = task_by_identity.get((opportunity.key.lineage, opportunity.key.instance))
        if task is None:
            raise FinalStrengthAuditError("shard opportunity has no authenticated task")
        if opportunity.evidence is None:
            rows.append(
                _invalid_row(
                    opportunity,
                    plan,
                    execution_entry=entry,
                    sampler_identity_digest=sampler_digest,
                )
            )
        else:
            rows.append(
                _valid_row(
                    opportunity,
                    plan=plan,
                    task=task,
                    context=context,
                    sampler=sampler,
                    forbidden_seeds=set(manifest.full_primary_evaluator_seeds),
                    execution_entry=entry,
                    sampler_identity_digest=sampler_digest,
                )
            )
    return FinalStrengthAuditShardResult(
        shard_index=shard_index,
        shard_count=plan.shard_count,
        paired_keys=paired_keys,
        rows=tuple(rows),
        source_runs=manifest.source_runs,
        plan_record_digest=plan.record_digest,
        execution_manifest_digest=manifest.record_digest,
        execution_manifest_file_sha256=sealed_execution_manifest.file_sha256,
        sampler_identity=sampler_identity,
        sampler_identity_digest=sampler_digest,
        registered_sampler_used=registered_sampler_used,
        publication_eligible=(
            plan.publication_eligible
            and manifest.publication_eligible
            and registered_sampler_used
        ),
    )


def _audit_seed(
    plan: FinalStrengthAuditPlan,
    key: FinalStrengthAuditKey,
    *,
    block: Literal["A", "B"],
    strength_index: int,
    forbidden: set[int],
) -> int:
    for nonce in range(16):
        payload = {
            "domain": BLOCK_DOMAINS[block],
            "audit_seed": plan.audit_seed,
            "key": key.as_dict(),
            "strength_index": strength_index,
            "nonce": nonce,
        }
        candidate = int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:4], "big")
        candidate %= 2**31
        if candidate not in forbidden:
            return candidate
    raise FinalStrengthAuditError("could not allocate a fresh domain-separated audit seed")


def _checked_sample(
    sampler: Sampler,
    program: Program,
    chains: Mapping[object, frozenset[object]],
    task: EmbeddingTask,
    *,
    plan: FinalStrengthAuditPlan,
    seed: int,
) -> ReadBlock:
    block = sampler(
        program,
        chains,
        task.problem,
        task.ground_energy,
        num_reads=plan.reads_per_block,
        seed=seed,
        num_sweeps=plan.num_sweeps,
    )
    if not isinstance(block, ReadBlock):
        raise FinalStrengthAuditError("audit sampler returned a non-ReadBlock value")
    if (
        block.reads != plan.reads_per_block
        or block.strength_index != program.strength_index
        or not 0 <= block.hits <= block.reads
        or not math.isfinite(block.broken_fraction)
        or not 0.0 <= block.broken_fraction <= 1.0
        or not math.isfinite(block.mean_residual)
        or block.mean_residual < 0.0
    ):
        raise FinalStrengthAuditError("audit sampler returned a malformed count block")
    return block


def _block_payload(block: ReadBlock, *, seed: int, domain: str) -> dict[str, object]:
    return {
        "domain": domain,
        "seed": seed,
        "hits": block.hits,
        "reads": block.reads,
        "rate": block.rate,
        "broken_chain_fraction": block.broken_fraction,
        "mean_energy_residual": block.mean_residual,
        "strength_index": block.strength_index,
    }


def _invalid_row(
    opportunity: _Opportunity,
    plan: FinalStrengthAuditPlan,
    *,
    execution_entry: Mapping[str, object] | None = None,
    sampler_identity_digest: str | None = None,
) -> dict[str, object]:
    payload = {
        "schema": ROW_SCHEMA,
        "schema_version": ROW_VERSION,
        "audit_key": opportunity.key.as_dict(),
        "audit_key_digest": opportunity.key.digest,
        "valid_return": False,
        "return_reason": opportunity.outcome.reason,
        "evidence_record_digest": opportunity.evidence_record_digest,
        "terminal_evidence_digest": None,
        "source_report_digest": opportunity.source_report_digest,
        "source_evidence_file_sha256": opportunity.source_evidence_file_sha256,
        "stratum": opportunity.stratum.as_dict(),
        "strengths": None,
        "program_digests": None,
        "deployed_strength_index": None,
        "block_a": None,
        "block_b": None,
        "oracle_a_strength_index": None,
        "oracle_a_tie_indices": None,
        "selected_oracle_agreement": None,
        "conditional_regret": None,
        "conditional_regret_definition": "null-for-invalid-return",
        "primary_blocks_reused": False,
        "feedback_forbidden": True,
        "opportunity_sensitivity_range": [-1.0, 1.0],
        "reads_consumed": 0,
        "reads_per_block": plan.reads_per_block,
        "seed_schedule_digest": (
            None if execution_entry is None else execution_entry["seed_schedule_digest"]
        ),
        "execution_manifest_opportunity_digest": (
            None if execution_entry is None else execution_entry["record_digest"]
        ),
        "sampler_identity_digest": sampler_identity_digest,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _valid_row(
    opportunity: _Opportunity,
    *,
    plan: FinalStrengthAuditPlan,
    task: EmbeddingTask,
    context: Context,
    sampler: Sampler,
    forbidden_seeds: set[int],
    execution_entry: Mapping[str, object] | None = None,
    sampler_identity_digest: str | None = None,
) -> dict[str, object]:
    evidence = opportunity.evidence
    assert evidence is not None
    chains, programs = _recompile_verified(opportunity, task=task, context=context)
    if tuple(program.strength_index for program in programs) != (0, 1, 2, 3):
        raise FinalStrengthAuditError("recompiled programs change the four-strength ordering")
    blocks_by_role: dict[str, dict[int, dict[str, object]]] = {"A": {}, "B": {}}
    if execution_entry is None:
        schedule: list[Mapping[str, object]] = []
        for program in programs:
            for block in ("A", "B"):
                seed = _audit_seed(
                    plan,
                    opportunity.key,
                    block=block,  # type: ignore[arg-type]
                    strength_index=program.strength_index,
                    forbidden=forbidden_seeds,
                )
                forbidden_seeds.add(seed)
                schedule.append(
                    {
                        "block": block,
                        "domain": BLOCK_DOMAINS[block],
                        "strength_index": program.strength_index,
                        "seed": seed,
                        "execution_rank": len(schedule),
                    }
                )
    else:
        raw_schedule = execution_entry.get("seed_schedule")
        if not isinstance(raw_schedule, Sequence):
            raise FinalStrengthAuditError("execution manifest has no audit seed schedule")
        schedule = sorted(raw_schedule, key=lambda item: int(item["execution_rank"]))  # type: ignore[index]
    for item in schedule:
        block = str(item["block"])
        strength_index = int(item["strength_index"])
        seed = int(item["seed"])
        sampled = _checked_sample(
            sampler,
            programs[strength_index],
            chains,
            task,
            plan=plan,
            seed=seed,
        )
        blocks_by_role[block][strength_index] = _block_payload(
            sampled, seed=seed, domain=BLOCK_DOMAINS[block]
        )
    blocks_a = [blocks_by_role["A"][index] for index in range(4)]
    blocks_b = [blocks_by_role["B"][index] for index in range(4)]
    rates_a = tuple(float(block["rate"]) for block in blocks_a)
    rates_b = tuple(float(block["rate"]) for block in blocks_b)
    oracle_rate_a = max(rates_a)
    oracle_ties = tuple(index for index, rate in enumerate(rates_a) if rate == oracle_rate_a)
    oracle_index = oracle_ties[0]
    selected_index = evidence.selected_index
    conditional_regret = rates_b[oracle_index] - rates_b[selected_index]
    program_digests = tuple(program_digest(program, chains) for program in programs)
    if program_digests[selected_index] != evidence.selected_program_digest:
        raise FinalStrengthAuditError("four-program identities disagree with terminal evidence")
    payload = {
        "schema": ROW_SCHEMA,
        "schema_version": ROW_VERSION,
        "audit_key": opportunity.key.as_dict(),
        "audit_key_digest": opportunity.key.digest,
        "valid_return": True,
        "return_reason": opportunity.outcome.reason,
        "evidence_record_digest": opportunity.evidence_record_digest,
        "terminal_evidence_digest": evidence.digest,
        "source_report_digest": opportunity.source_report_digest,
        "source_evidence_file_sha256": opportunity.source_evidence_file_sha256,
        "stratum": opportunity.stratum.as_dict(),
        "strengths": [program.strength for program in programs],
        "program_digests": list(program_digests),
        "deployed_strength_index": selected_index,
        "block_a": blocks_a,
        "block_b": blocks_b,
        "oracle_a_strength_index": oracle_index,
        "oracle_a_tie_indices": list(oracle_ties),
        "selected_oracle_agreement": selected_index == oracle_index,
        "conditional_regret": conditional_regret,
        "conditional_regret_definition": (
            "independent-block-b-rate-at-block-a-oracle-minus-"
            "independent-block-b-rate-at-deployed-strength"
        ),
        "primary_blocks_reused": False,
        "feedback_forbidden": True,
        "opportunity_sensitivity_range": [conditional_regret, conditional_regret],
        "reads_consumed": 8 * plan.reads_per_block,
        "reads_per_block": plan.reads_per_block,
        "seed_schedule_digest": (
            content_digest(schedule)
            if execution_entry is None
            else execution_entry["seed_schedule_digest"]
        ),
        "execution_manifest_opportunity_digest": (
            None if execution_entry is None else execution_entry["record_digest"]
        ),
        "sampler_identity_digest": sampler_identity_digest,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    if not values or len(values) != len(weights):
        raise ValueError("weighted quantile requires aligned nonempty values and weights")
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    threshold = q * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _hierarchical_mean(
    rows: Sequence[Mapping[str, object]],
    values: Callable[[Mapping[str, object]], Sequence[float]],
) -> float | None:
    """Average observations within lineage, then lineages within each of three seeds."""

    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = row["audit_key"]
        assert isinstance(key, Mapping)
        cells[(int(key["training_seed"]), str(key["lineage"]))].extend(values(row))
    seed_means: list[float] = []
    for seed in REGISTERED_TRAINING_SEEDS:
        lineage_means = [
            float(np.mean(observations))
            for (cell_seed, _), observations in cells.items()
            if cell_seed == seed and observations
        ]
        if not lineage_means:
            return None
        seed_means.append(float(np.mean(lineage_means)))
    return float(np.mean(seed_means))


def _descriptive_crossed_bootstrap(
    rows: Sequence[Mapping[str, object]],
    values: Callable[[Mapping[str, object]], Sequence[float]],
    *,
    plan: FinalStrengthAuditPlan,
    label: str,
) -> dict[str, object] | None:
    """Cross training seeds and complete-case immutable lineages for uncertainty only."""

    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = row["audit_key"]
        assert isinstance(key, Mapping)
        cells[(int(key["training_seed"]), str(key["lineage"]))].extend(values(row))
    lineage_sets = [
        {
            lineage
            for (cell_seed, lineage), observations in cells.items()
            if cell_seed == seed and observations
        }
        for seed in REGISTERED_TRAINING_SEEDS
    ]
    common = tuple(sorted(set.intersection(*lineage_sets))) if lineage_sets else ()
    if not common:
        return None
    matrix = np.asarray(
        [
            [float(np.mean(cells[(seed, lineage)])) for lineage in common]
            for seed in REGISTERED_TRAINING_SEEDS
        ],
        dtype=float,
    )
    bootstrap_seed = int(
        stable_digest(
            {
                "domain": "isingfold-final-strength-descriptive-bootstrap-v1",
                "base_seed": plan.bootstrap_seed,
                "label": label,
            }
        )[:8],
        16,
    ) % (2**31)
    lower, upper = crossed_bootstrap_bounds(
        matrix,
        replicates=plan.bootstrap_replicates,
        seed=bootstrap_seed,
        alpha=plan.two_sided_alpha / 2.0,
    )
    return {
        "lower": lower,
        "upper": upper,
        "two_sided_alpha": plan.two_sided_alpha,
        "replicates": plan.bootstrap_replicates,
        "seed": bootstrap_seed,
        "complete_case_lineages": len(common),
        "lineage_census_digest": stable_digest(list(common)),
        "sampling": "crossed-training-seed-and-immutable-base-lineage-with-replacement",
        "scope": "descriptive-diagnostic-not-primary-superiority",
    }


def _conditional_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    plan: FinalStrengthAuditPlan,
    label: str,
) -> dict[str, object]:
    by_seed_lineage: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row["conditional_regret"]
        if value is None:
            continue
        key = row["audit_key"]
        assert isinstance(key, Mapping)
        by_seed_lineage[(int(key["training_seed"]), str(key["lineage"]))].append(float(value))
    seed_cells: dict[int, list[float]] = {seed: [] for seed in REGISTERED_TRAINING_SEEDS}
    for (seed, _), values in by_seed_lineage.items():
        seed_cells[seed].append(float(np.mean(values)))
    if any(not seed_cells[seed] for seed in REGISTERED_TRAINING_SEEDS):
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "seed_complete": False,
            "seed_lineage_cells": len(by_seed_lineage),
            "descriptive_crossed_bootstrap_95_ci": None,
        }
    values: list[float] = []
    weights: list[float] = []
    for seed in REGISTERED_TRAINING_SEEDS:
        weight = 1.0 / (len(REGISTERED_TRAINING_SEEDS) * len(seed_cells[seed]))
        values.extend(seed_cells[seed])
        weights.extend([weight] * len(seed_cells[seed]))
    return {
        "mean": float(sum(value * weight for value, weight in zip(values, weights, strict=True))),
        "median": _weighted_quantile(values, weights, 0.5),
        "p95": _weighted_quantile(values, weights, 0.95),
        "seed_complete": True,
        "seed_lineage_cells": len(values),
        "descriptive_crossed_bootstrap_95_ci": _descriptive_crossed_bootstrap(
            rows,
            lambda row: (
                ()
                if row["conditional_regret"] is None
                else (float(row["conditional_regret"]),)
            ),
            plan=plan,
            label=label,
        ),
    }


def _sensitivity(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    by_seed_lineage: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        key = row["audit_key"]
        interval = row["opportunity_sensitivity_range"]
        assert isinstance(key, Mapping) and isinstance(interval, Sequence)
        by_seed_lineage[(int(key["training_seed"]), str(key["lineage"]))].append(
            (float(interval[0]), float(interval[1]))
        )
    seed_bounds: list[tuple[float, float]] = []
    for seed in REGISTERED_TRAINING_SEEDS:
        lineage_bounds = [
            (float(np.mean([item[0] for item in values])), float(np.mean([item[1] for item in values])))
            for (cell_seed, _), values in by_seed_lineage.items()
            if cell_seed == seed
        ]
        if not lineage_bounds:
            raise FinalStrengthAuditError("sensitivity aggregate lost a registered seed")
        seed_bounds.append(
            (
                float(np.mean([item[0] for item in lineage_bounds])),
                float(np.mean([item[1] for item in lineage_bounds])),
            )
        )
    return {
        "lower": float(np.mean([item[0] for item in seed_bounds])),
        "upper": float(np.mean([item[1] for item in seed_bounds])),
    }


def _repeatability(
    rows: Sequence[Mapping[str, object]],
    *,
    plan: FinalStrengthAuditPlan,
    label: str,
) -> dict[str, object]:
    valid = [row for row in rows if row["valid_return"] is True]
    if not valid:
        return {
            "interpretation": "independent-block-sampling-repeatability-not-selector-calibration",
            "all_strength_a_minus_b_mean": None,
            "all_strength_mean_absolute_error": None,
            "all_strength_root_mean_squared_error": None,
            "oracle_selection_optimism_mean": None,
            "all_strength_a_minus_b_crossed_bootstrap_95_ci": None,
        }
    def deltas(row: Mapping[str, object]) -> tuple[float, ...]:
        blocks_a = row["block_a"]
        blocks_b = row["block_b"]
        assert isinstance(blocks_a, Sequence) and isinstance(blocks_b, Sequence)
        rates_a = [float(item["rate"]) for item in blocks_a]  # type: ignore[index]
        rates_b = [float(item["rate"]) for item in blocks_b]  # type: ignore[index]
        return tuple(a - b for a, b in zip(rates_a, rates_b, strict=True))

    def optimism(row: Mapping[str, object]) -> tuple[float]:
        blocks_a = row["block_a"]
        blocks_b = row["block_b"]
        oracle_index = row["oracle_a_strength_index"]
        assert isinstance(blocks_a, Sequence) and isinstance(blocks_b, Sequence)
        assert type(oracle_index) is int
        return (
            float(blocks_a[oracle_index]["rate"])  # type: ignore[index]
            - float(blocks_b[oracle_index]["rate"]),  # type: ignore[index]
        )

    mean_delta = _hierarchical_mean(valid, deltas)
    mean_absolute = _hierarchical_mean(
        valid, lambda row: tuple(abs(value) for value in deltas(row))
    )
    mean_squared = _hierarchical_mean(
        valid, lambda row: tuple(value * value for value in deltas(row))
    )
    return {
        "interpretation": "independent-block-sampling-repeatability-not-selector-calibration",
        "all_strength_a_minus_b_mean": mean_delta,
        "all_strength_mean_absolute_error": mean_absolute,
        "all_strength_root_mean_squared_error": (
            None if mean_squared is None else math.sqrt(mean_squared)
        ),
        "oracle_selection_optimism_mean": _hierarchical_mean(valid, optimism),
        "all_strength_a_minus_b_crossed_bootstrap_95_ci": (
            _descriptive_crossed_bootstrap(
                valid,
                deltas,
                plan=plan,
                label=f"{label}-a-minus-b-repeatability",
            )
        ),
    }


def _arm_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    plan: FinalStrengthAuditPlan,
    arm: str,
) -> dict[str, object]:
    opportunities = len(rows)
    valid = [row for row in rows if row["valid_return"] is True]
    return {
        "opportunities": opportunities,
        "valid_returns": len(valid),
        "invalid_returns": opportunities - len(valid),
        "coverage": len(valid) / opportunities,
        "equal_seed_lineage_coverage": _hierarchical_mean(
            rows, lambda row: (1.0 if row["valid_return"] is True else 0.0,)
        ),
        "coverage_crossed_bootstrap_95_ci": _descriptive_crossed_bootstrap(
            rows,
            lambda row: (1.0 if row["valid_return"] is True else 0.0,),
            plan=plan,
            label=f"{arm}-coverage",
        ),
        "conditional_regret": _conditional_summary(
            rows, plan=plan, label=f"{arm}-conditional-regret"
        ),
        "selected_oracle_agreement": _hierarchical_mean(
            valid,
            lambda row: (
                1.0 if row["selected_oracle_agreement"] is True else 0.0,
            ),
        ),
        "selected_oracle_agreement_crossed_bootstrap_95_ci": (
            _descriptive_crossed_bootstrap(
                valid,
                lambda row: (
                    1.0 if row["selected_oracle_agreement"] is True else 0.0,
                ),
                plan=plan,
                label=f"{arm}-selected-oracle-agreement",
            )
            if valid
            else None
        ),
        "block_repeatability": _repeatability(rows, plan=plan, label=arm),
        "invalid_worst_case_sensitivity": {
            "definition": "invalid-regret-in-[-1,1]-with-exact-valid-plug-in",
            **_sensitivity(rows),
        },
        "reads_consumed": sum(int(row["reads_consumed"]) for row in rows),
    }


def _paired_summary(
    rows: Sequence[Mapping[str, object]], *, plan: FinalStrengthAuditPlan
) -> dict[str, object]:
    by_key: dict[tuple[str, str, int, int], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        raw_key = row["audit_key"]
        assert isinstance(raw_key, Mapping)
        key = (
            str(raw_key["lineage"]),
            str(raw_key["instance"]),
            int(raw_key["repetition"]),
            int(raw_key["training_seed"]),
        )
        by_key[key][str(raw_key["arm"])] = row
    if any(set(pair) != set(ARMS) for pair in by_key.values()):
        raise FinalStrengthAuditError("paired audit rows have incomplete learned/stock arms")
    paired_rows: list[dict[str, object]] = []
    for key, pair in sorted(by_key.items()):
        learned = pair["learned"]
        stock = pair["tuned-stock"]
        joint = learned["valid_return"] is True and stock["valid_return"] is True
        difference = (
            float(learned["conditional_regret"]) - float(stock["conditional_regret"])
            if joint
            else None
        )
        paired_rows.append(
            {
                "audit_key": {
                    "lineage": key[0],
                    "instance": key[1],
                    "repetition": key[2],
                    "training_seed": key[3],
                },
                "conditional_regret": difference,
                "opportunity_sensitivity_range": (
                    [difference, difference] if difference is not None else [-2.0, 2.0]
                ),
            }
        )
    return {
        "paired_opportunities": len(paired_rows),
        "joint_valid": sum(row["conditional_regret"] is not None for row in paired_rows),
        "joint_valid_coverage": _hierarchical_mean(
            paired_rows,
            lambda row: (1.0 if row["conditional_regret"] is not None else 0.0,),
        ),
        "joint_valid_coverage_crossed_bootstrap_95_ci": _descriptive_crossed_bootstrap(
            paired_rows,
            lambda row: (1.0 if row["conditional_regret"] is not None else 0.0,),
            plan=plan,
            label="paired-joint-valid-coverage",
        ),
        "learned_minus_tuned_stock_conditional_regret": _conditional_summary(
            paired_rows,
            plan=plan,
            label="learned-minus-stock-conditional-regret",
        ),
        "invalid_worst_case_sensitivity": {
            "definition": "non-joint-valid-paired-regret-difference-in-[-2,2]",
            **_sensitivity(paired_rows),
        },
    }


def _descriptive_strata(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = row["audit_key"]
        stratum = row["stratum"]
        assert isinstance(key, Mapping) and isinstance(stratum, Mapping)
        grouped[(str(key["arm"]), str(stratum["stratum_id"]))].append(row)
    output: dict[str, object] = {}
    for (arm, stratum_id), group in sorted(grouped.items()):
        first = group[0]["stratum"]
        assert isinstance(first, Mapping)
        valid_values = [
            float(row["conditional_regret"])
            for row in group
            if row["conditional_regret"] is not None
        ]
        output[f"{arm}:{stratum_id}"] = {
            "arm": arm,
            "stratum_id": stratum_id,
            "coordinates": first["stratum_coordinates"],
            "opportunities": len(group),
            "valid_returns": len(valid_values),
            "conditional_regret_mean": (
                None if not valid_values else float(np.mean(valid_values))
            ),
        }
    return {
        "scope": "descriptive-only-not-powered-no-multiplicity-claims",
        "groups": output,
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, object]], plan: FinalStrengthAuditPlan
) -> dict[str, object]:
    by_arm = {
        arm: [row for row in rows if row["audit_key"]["arm"] == arm]  # type: ignore[index]
        for arm in ARMS
    }
    return {
        "aggregation": AGGREGATION,
        "opportunities": len(rows),
        "base_lineages": len({key.lineage for key in plan.opportunity_keys}),
        "training_seeds": list(REGISTERED_TRAINING_SEEDS),
        "arms": {
            arm: _arm_summary(by_arm[arm], plan=plan, arm=arm) for arm in ARMS
        },
        "paired_learned_vs_tuned_stock": _paired_summary(rows, plan=plan),
        "descriptive_strata": _descriptive_strata(rows),
    }


def run_final_strength_audit(
    sealed_plan: AuthenticatedFinalStrengthAuditPlan,
    learned_runs: Sequence[AuthenticatedCompleteSystemSeedRun],
    stock_runs: Sequence[AuthenticatedExternalCompleteRun],
    tasks: Sequence[EmbeddingTask],
    context: Context,
    *,
    sampler: Sampler = sample_program,
) -> FinalStrengthAuditResult:
    """Execute the sealed audit without consulting any primary evaluator count block."""

    if not isinstance(sealed_plan, AuthenticatedFinalStrengthAuditPlan):
        raise TypeError("audit execution requires an out-of-band authenticated plan")
    plan = sealed_plan.plan
    if plan.publication_eligible:
        raise FinalStrengthAuditError(
            "monolithic production audit is forbidden; seal the six-source execution "
            "manifest and run every deterministic paired-key shard"
        )
    if stable_digest(context_snapshot(context)) != plan.context_digest:
        raise FinalStrengthAuditError("supplied audit context differs from the sealed plan")
    if tuple(context.strength_ratios) != plan.strength_ratios or context.num_sweeps != plan.num_sweeps:
        raise FinalStrengthAuditError("audit context changes strengths or sampler sweeps")
    current_runtime = content_digest(runtime_implementation_registry())
    if current_runtime != plan.runtime_implementation_digest:
        raise FinalStrengthAuditError("audit runtime source changed after the plan was sealed")
    opportunities, source_runs, population = _opportunities_from_runs(
        learned_runs, stock_runs, plan=plan
    )
    population.validate_tasks(tasks)
    task_by_identity = {(task.lineage or task.name, task.name): task for task in tasks}
    if len(task_by_identity) != len(tasks):
        raise FinalStrengthAuditError("audit task set has duplicate lineage/instance identities")
    rows: list[dict[str, object]] = []
    forbidden_seeds = {
        opportunity.evidence.evaluator_seed
        for opportunity in opportunities
        if opportunity.evidence is not None
    }
    for opportunity in opportunities:
        task = task_by_identity.get((opportunity.key.lineage, opportunity.key.instance))
        if task is None:
            raise FinalStrengthAuditError("audit opportunity has no authenticated task")
        if opportunity.evidence is None:
            rows.append(_invalid_row(opportunity, plan))
        else:
            rows.append(
                _valid_row(
                    opportunity,
                    plan=plan,
                    task=task,
                    context=context,
                    sampler=sampler,
                    forbidden_seeds=forbidden_seeds,
                )
            )
    if tuple(FinalStrengthAuditKey.from_mapping(row["audit_key"]) for row in rows) != (
        plan.opportunity_keys
    ):
        raise FinalStrengthAuditError("final-strength raw row census changed during execution")
    aggregate = _aggregate_rows(rows, plan)
    return FinalStrengthAuditResult(tuple(rows), aggregate, source_runs)


def _write_sync(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_audit_row(
    row: Mapping[str, object],
    *,
    key: FinalStrengthAuditKey,
    plan: FinalStrengthAuditPlan,
    execution_entry: Mapping[str, object],
    sampler_identity_digest: str,
) -> None:
    payload = {name: value for name, value in row.items() if name != "record_digest"}
    if (
        row.get("schema") != ROW_SCHEMA
        or row.get("schema_version") != ROW_VERSION
        or row.get("record_digest") != content_digest(payload)
        or row.get("audit_key") != key.as_dict()
        or row.get("audit_key_digest") != key.digest
        or row.get("sampler_identity_digest") != sampler_identity_digest
        or row.get("seed_schedule_digest") != execution_entry.get("seed_schedule_digest")
        or row.get("execution_manifest_opportunity_digest")
        != execution_entry.get("record_digest")
        or row.get("source_report_digest")
        != execution_entry.get("source_report_digest")
        or row.get("source_evidence_file_sha256")
        != execution_entry.get("source_evidence_file_sha256")
        or row.get("evidence_record_digest")
        != execution_entry.get("evidence_record_digest")
        or row.get("terminal_evidence_digest")
        != execution_entry.get("terminal_evidence_digest")
        or row.get("valid_return") != execution_entry.get("valid_return")
        or row.get("reads_per_block") != plan.reads_per_block
    ):
        raise ValueError("audit row identity or source link differs from the sealed manifest")
    interval = row.get("opportunity_sensitivity_range")
    if not isinstance(interval, Sequence) or len(interval) != 2:
        raise ValueError("audit row sensitivity interval is malformed")
    if row["valid_return"] is False:
        nullable = (
            "terminal_evidence_digest",
            "strengths",
            "program_digests",
            "deployed_strength_index",
            "block_a",
            "block_b",
            "oracle_a_strength_index",
            "oracle_a_tie_indices",
            "selected_oracle_agreement",
            "conditional_regret",
        )
        if (
            any(row.get(name) is not None for name in nullable)
            or list(interval) != [-1.0, 1.0]
            or row.get("reads_consumed") != 0
        ):
            raise ValueError("invalid-return audit row changes failure-aware bounds")
        return
    blocks_a = row.get("block_a")
    blocks_b = row.get("block_b")
    programs = row.get("program_digests")
    schedule = execution_entry.get("seed_schedule")
    if (
        not isinstance(blocks_a, Sequence)
        or not isinstance(blocks_b, Sequence)
        or len(blocks_a) != 4
        or len(blocks_b) != 4
        or not isinstance(programs, Sequence)
        or list(programs) != list(execution_entry.get("program_digests", ()))
        or not isinstance(schedule, Sequence)
        or row.get("deployed_strength_index")
        != execution_entry.get("deployed_strength_index")
        or row.get("reads_consumed") != 8 * plan.reads_per_block
    ):
        raise ValueError("valid audit row changes program or read identities")
    expected_seeds = {
        (str(item["block"]), int(item["strength_index"])): int(item["seed"])
        for item in schedule  # type: ignore[union-attr]
    }
    rates: dict[str, list[float]] = {"A": [], "B": []}
    for block_name, blocks in (("A", blocks_a), ("B", blocks_b)):
        for index, block in enumerate(blocks):
            if not isinstance(block, Mapping):
                raise ValueError("audit block is not an object")
            hits = block.get("hits")
            reads = block.get("reads")
            rate = block.get("rate")
            if (
                block.get("domain") != BLOCK_DOMAINS[block_name]
                or block.get("strength_index") != index
                or block.get("seed") != expected_seeds[(block_name, index)]
                or type(hits) is not int
                or type(reads) is not int
                or reads != plan.reads_per_block
                or not 0 <= hits <= reads
                or not isinstance(rate, (int, float))
                or float(rate) != hits / reads
            ):
                raise ValueError("audit block differs from its seed/count contract")
            rates[block_name].append(float(rate))
    oracle_rate = max(rates["A"])
    ties = [index for index, rate in enumerate(rates["A"]) if rate == oracle_rate]
    oracle = ties[0]
    deployed = row["deployed_strength_index"]
    if type(deployed) is not int or deployed not in range(4):
        raise ValueError("audit deployed strength index is invalid")
    regret = rates["B"][oracle] - rates["B"][deployed]
    if (
        not -1.0 <= regret <= 1.0
        or row.get("oracle_a_strength_index") != oracle
        or list(row.get("oracle_a_tie_indices", ())) != ties
        or row.get("selected_oracle_agreement") != (deployed == oracle)
        or row.get("conditional_regret") != regret
        or list(interval) != [regret, regret]
    ):
        raise ValueError("audit row regret or split-block oracle statistic is inconsistent")


def publish_final_strength_audit_shard(
    destination: str | os.PathLike[str],
    *,
    sealed_plan: AuthenticatedFinalStrengthAuditPlan,
    sealed_execution_manifest: AuthenticatedFinalStrengthAuditExecutionManifest,
    result: FinalStrengthAuditShardResult,
) -> dict[str, object]:
    """Atomically publish one immutable paired-key shard and its complete receipt."""

    if not isinstance(result, FinalStrengthAuditShardResult):
        raise TypeError("shard publication requires a typed shard result")
    plan = sealed_plan.plan
    manifest = sealed_execution_manifest.manifest
    manifest.validate_plan(sealed_plan)
    if (
        result.shard_count != plan.shard_count
        or result.plan_record_digest != plan.record_digest
        or result.execution_manifest_digest != manifest.record_digest
        or result.execution_manifest_file_sha256 != sealed_execution_manifest.file_sha256
        or result.source_runs != manifest.source_runs
    ):
        raise ValueError("audit shard result drifts from plan, manifest, or six sources")
    expected_pairs = _shard_paired_keys(plan, shard_index=result.shard_index)
    if result.paired_keys != expected_pairs:
        raise ValueError("audit shard result changes deterministic paired-key assignment")
    expected_keys = tuple(
        key for key in plan.opportunity_keys if key.paired_key in set(expected_pairs)
    )
    entries = {
        FinalStrengthAuditKey.from_mapping(row["audit_key"]): row  # type: ignore[arg-type]
        for row in manifest.opportunity_execution
    }
    if len(result.rows) != len(expected_keys):
        raise ValueError("audit shard row census is incomplete")
    for row, key in zip(result.rows, expected_keys, strict=True):
        _validate_audit_row(
            row,
            key=key,
            plan=plan,
            execution_entry=entries[key],
            sampler_identity_digest=result.sampler_identity_digest,
        )
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"final-strength audit shard already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        raw = b"".join(canonical_json_bytes(row) + b"\n" for row in result.rows)
        _write_sync(temporary / "rows.jsonl", raw)
        schedule_rows = [
            _plain_json(entries[key]["seed_schedule"]) for key in expected_keys
        ]
        payload: dict[str, object] = {
            "schema": SHARD_SCHEMA,
            "schema_version": SHARD_VERSION,
            "scope": "post-freeze-diagnostic-only",
            "publication_eligible": result.publication_eligible,
            "shard_index": result.shard_index,
            "shard_count": result.shard_count,
            "assignment": "canonical-sorted-paired-key-round-robin-v1",
            "plan": {
                "file_sha256": sealed_plan.file_sha256,
                "record_digest": plan.record_digest,
                "config_record_digest": plan.audit_config_record_digest,
                "config_file_sha256": plan.audit_config_file_sha256,
                "quality_authority_digest": plan.quality_authority_digest,
                "learned_selection_digest": plan.learned_selection_digest,
                "stock_tuning_execution_digest": plan.stock_tuning_execution_digest,
                "runtime_implementation_digest": plan.runtime_implementation_digest,
            },
            "execution_manifest": {
                "file_sha256": sealed_execution_manifest.file_sha256,
                "record_digest": manifest.record_digest,
                "source_report_pins_digest": manifest.source_report_pins_digest,
                "target_access_digest": manifest.target_access_digest,
                "opportunity_execution_digest": manifest.opportunity_execution_digest,
            },
            "paired_keys": [list(key) for key in expected_pairs],
            "paired_keys_digest": content_digest([list(key) for key in expected_pairs]),
            "opportunity_keys_digest": content_digest(
                [key.as_dict() for key in expected_keys]
            ),
            "seed_schedule_digest": content_digest(schedule_rows),
            "sampler_identity": _plain_json(result.sampler_identity),
            "sampler_identity_digest": result.sampler_identity_digest,
            "registered_sampler_used": result.registered_sampler_used,
            "source_runs": [_plain_json(row) for row in result.source_runs],
            "source_runs_digest": content_digest(
                [_plain_json(row) for row in result.source_runs]
            ),
            "raw_rows": {
                "path": "rows.jsonl",
                "schema": ROW_SCHEMA,
                "schema_version": ROW_VERSION,
                "count": len(result.rows),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            "chronology": {
                **FinalStrengthAuditConfig._CHRONOLOGY,
                "this_shard_completed_after_manifest_sealing": True,
            },
        }
        receipt = {**payload, "record_digest": content_digest(payload)}
        _write_sync(temporary / "receipt.json", canonical_json_bytes(receipt) + b"\n")
        os.replace(temporary, target)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_final_strength_audit_shard(
    directory: str | os.PathLike[str], *, expected_receipt_sha256: str
) -> AuthenticatedFinalStrengthAuditShard:
    """Authenticate a canonical shard receipt and its exact raw-row artifact."""

    _require_digest(expected_receipt_sha256, label="expected shard receipt file")
    root = Path(directory)
    receipt_content = (root / "receipt.json").read_bytes()
    observed = hashlib.sha256(receipt_content).hexdigest()
    if observed != expected_receipt_sha256:
        raise FinalStrengthAuditError(
            f"audit shard receipt SHA-256 mismatch: expected {expected_receipt_sha256}, "
            f"observed {observed}"
        )
    try:
        receipt = json.loads(receipt_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalStrengthAuditError("audit shard receipt is not UTF-8 JSON") from exc
    if (
        not isinstance(receipt, dict)
        or canonical_json_bytes(receipt) + b"\n" != receipt_content
        or receipt.get("schema") != SHARD_SCHEMA
        or receipt.get("schema_version") != SHARD_VERSION
    ):
        raise FinalStrengthAuditError("audit shard receipt has an unsupported canonical schema")
    expected_receipt_fields = {
        "schema",
        "schema_version",
        "scope",
        "publication_eligible",
        "shard_index",
        "shard_count",
        "assignment",
        "plan",
        "execution_manifest",
        "paired_keys",
        "paired_keys_digest",
        "opportunity_keys_digest",
        "seed_schedule_digest",
        "sampler_identity",
        "sampler_identity_digest",
        "registered_sampler_used",
        "source_runs",
        "source_runs_digest",
        "raw_rows",
        "chronology",
        "record_digest",
    }
    if set(receipt) != expected_receipt_fields:
        raise FinalStrengthAuditError("audit shard receipt has missing or unknown fields")
    recorded = receipt.get("record_digest")
    if recorded != content_digest(
        {name: value for name, value in receipt.items() if name != "record_digest"}
    ):
        raise FinalStrengthAuditError("audit shard receipt digest mismatch")
    raw_entry = receipt.get("raw_rows")
    if not isinstance(raw_entry, Mapping) or raw_entry.get("path") != "rows.jsonl":
        raise FinalStrengthAuditError("audit shard receipt has no registered rows artifact")
    raw = (root / "rows.jsonl").read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    if raw_entry.get("sha256") != raw_sha:
        raise FinalStrengthAuditError("audit shard raw-row SHA-256 mismatch")
    rows: list[Mapping[str, object]] = []
    for number, line in enumerate(raw.splitlines(keepends=True), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FinalStrengthAuditError(f"audit shard row {number} is invalid JSON") from exc
        if not isinstance(row, dict) or canonical_json_bytes(row) + b"\n" != line:
            raise FinalStrengthAuditError(f"audit shard row {number} is not canonical")
        rows.append(row)
    if raw_entry.get("count") != len(rows) or not rows:
        raise FinalStrengthAuditError("audit shard row count differs from its receipt")
    return AuthenticatedFinalStrengthAuditShard(
        receipt=receipt,
        rows=tuple(rows),
        receipt_file_sha256=expected_receipt_sha256,
        rows_file_sha256=raw_sha,
    )


def merge_final_strength_audit_shards(
    sealed_plan: AuthenticatedFinalStrengthAuditPlan,
    sealed_execution_manifest: AuthenticatedFinalStrengthAuditExecutionManifest,
    shards: Sequence[AuthenticatedFinalStrengthAuditShard],
) -> FinalStrengthAuditResult:
    """Authenticate a complete nonoverlapping shard set and recompute all statistics."""

    plan = sealed_plan.plan
    manifest = sealed_execution_manifest.manifest
    manifest.validate_plan(sealed_plan)
    materialized = tuple(shards)
    if len(materialized) != plan.shard_count or any(
        not isinstance(shard, AuthenticatedFinalStrengthAuditShard)
        for shard in materialized
    ):
        raise FinalStrengthAuditError("audit merge requires the complete authenticated shard census")
    ordered = tuple(sorted(materialized, key=lambda shard: int(shard.receipt["shard_index"])))
    if [shard.receipt["shard_index"] for shard in ordered] != list(range(plan.shard_count)):
        raise FinalStrengthAuditError("audit shards are missing, duplicated, or out of range")
    entries = {
        FinalStrengthAuditKey.from_mapping(row["audit_key"]): row  # type: ignore[arg-type]
        for row in manifest.opportunity_execution
    }
    source_digest = content_digest([_plain_json(row) for row in manifest.source_runs])
    sampler_digests: set[str] = set()
    seen_keys: set[FinalStrengthAuditKey] = set()
    seen_pairs: set[tuple[str, str, int, int]] = set()
    seen_sampler_seeds: set[int] = set()
    rows_by_key: dict[FinalStrengthAuditKey, Mapping[str, object]] = {}
    all_publication_eligible = True
    for index, shard in enumerate(ordered):
        receipt = shard.receipt
        plan_pin = receipt.get("plan")
        execution_pin = receipt.get("execution_manifest")
        if (
            receipt.get("schema") != SHARD_SCHEMA
            or receipt.get("schema_version") != SHARD_VERSION
            or receipt.get("shard_count") != plan.shard_count
            or receipt.get("assignment") != "canonical-sorted-paired-key-round-robin-v1"
            or not isinstance(plan_pin, Mapping)
            or plan_pin.get("file_sha256") != sealed_plan.file_sha256
            or plan_pin.get("record_digest") != plan.record_digest
            or plan_pin.get("config_record_digest") != plan.audit_config_record_digest
            or plan_pin.get("config_file_sha256") != plan.audit_config_file_sha256
            or plan_pin.get("quality_authority_digest") != plan.quality_authority_digest
            or plan_pin.get("learned_selection_digest") != plan.learned_selection_digest
            or plan_pin.get("stock_tuning_execution_digest")
            != plan.stock_tuning_execution_digest
            or plan_pin.get("runtime_implementation_digest")
            != plan.runtime_implementation_digest
            or not isinstance(execution_pin, Mapping)
            or execution_pin.get("file_sha256") != sealed_execution_manifest.file_sha256
            or execution_pin.get("record_digest") != manifest.record_digest
            or execution_pin.get("source_report_pins_digest")
            != manifest.source_report_pins_digest
            or execution_pin.get("target_access_digest")
            != manifest.target_access_digest
            or execution_pin.get("opportunity_execution_digest")
            != manifest.opportunity_execution_digest
            or receipt.get("source_runs_digest") != source_digest
            or _plain_json(receipt.get("source_runs"))
            != [_plain_json(row) for row in manifest.source_runs]
        ):
            raise FinalStrengthAuditError(
                "audit shard mixes plan, source, ground, runtime, or selection identities"
            )
        expected_pairs = _shard_paired_keys(plan, shard_index=index)
        if (
            _plain_json(receipt.get("paired_keys"))
            != [list(key) for key in expected_pairs]
            or receipt.get("paired_keys_digest")
            != content_digest([list(key) for key in expected_pairs])
        ):
            raise FinalStrengthAuditError("audit shard changes deterministic paired-key assignment")
        overlap = set(expected_pairs) & seen_pairs
        if overlap:
            raise FinalStrengthAuditError("audit shards overlap paired keys")
        seen_pairs.update(expected_pairs)
        expected_keys = tuple(
            key for key in plan.opportunity_keys if key.paired_key in set(expected_pairs)
        )
        expected_schedule_digest = content_digest(
            [_plain_json(entries[key]["seed_schedule"]) for key in expected_keys]
        )
        if receipt.get("opportunity_keys_digest") != content_digest(
            [key.as_dict() for key in expected_keys]
        ) or receipt.get("seed_schedule_digest") != expected_schedule_digest or len(
            shard.rows
        ) != len(expected_keys):
            raise FinalStrengthAuditError("audit shard opportunity census is incomplete")
        sampler_digest = receipt.get("sampler_identity_digest")
        if not _is_digest(sampler_digest) or receipt.get("sampler_identity") is None:
            raise FinalStrengthAuditError("audit shard sampler identity is malformed")
        if content_digest(receipt["sampler_identity"]) != sampler_digest:
            raise FinalStrengthAuditError("audit shard sampler identity digest differs")
        sampler_digests.add(sampler_digest)  # type: ignore[arg-type]
        all_publication_eligible = all_publication_eligible and bool(
            receipt.get("publication_eligible") is True
        )
        for row, key in zip(shard.rows, expected_keys, strict=True):
            if key in seen_keys:
                raise FinalStrengthAuditError("audit shards duplicate an opportunity row")
            _validate_audit_row(
                row,
                key=key,
                plan=plan,
                execution_entry=entries[key],
                sampler_identity_digest=sampler_digest,  # type: ignore[arg-type]
            )
            if row["valid_return"] is True:
                for name in ("block_a", "block_b"):
                    for block in row[name]:  # type: ignore[index,union-attr]
                        seed = int(block["seed"])
                        if seed in seen_sampler_seeds:
                            raise FinalStrengthAuditError("audit shards reuse a sampler seed")
                        seen_sampler_seeds.add(seed)
            seen_keys.add(key)
            rows_by_key[key] = row
    if len(sampler_digests) != 1:
        raise FinalStrengthAuditError("audit shards mix sampler identities")
    if seen_keys != set(plan.opportunity_keys) or seen_pairs != set(_all_paired_keys(plan)):
        raise FinalStrengthAuditError("audit shard merge does not cover the complete plan")
    rows = tuple(rows_by_key[key] for key in plan.opportunity_keys)
    aggregate = _aggregate_rows(rows, plan)
    sampler_digest = next(iter(sampler_digests))
    publication_eligible = (
        plan.publication_eligible
        and manifest.publication_eligible
        and all_publication_eligible
        and sampler_digest == plan.sampler_identity_digest
    )
    source_shards = tuple(
        _canonical_mapping(
            {
                "shard_index": int(shard.receipt["shard_index"]),
                "receipt_file_sha256": shard.receipt_file_sha256,
                "receipt_record_digest": shard.receipt["record_digest"],
                "rows_file_sha256": shard.rows_file_sha256,
                "paired_keys_digest": shard.receipt["paired_keys_digest"],
                "opportunity_keys_digest": shard.receipt["opportunity_keys_digest"],
                "seed_schedule_digest": shard.receipt["seed_schedule_digest"],
                "sampler_identity_digest": shard.receipt["sampler_identity_digest"],
            },
            label="merged source shard",
        )
        for shard in ordered
    )
    return FinalStrengthAuditResult(
        rows=rows,
        aggregate=aggregate,
        source_runs=manifest.source_runs,
        publication_eligible=publication_eligible,
        execution_manifest_digest=manifest.record_digest,
        sampler_identity_digest=sampler_digest,
        source_shards=source_shards,
    )


def publish_final_strength_audit(
    destination: str | os.PathLike[str],
    *,
    sealed_plan: AuthenticatedFinalStrengthAuditPlan,
    sealed_execution_manifest: AuthenticatedFinalStrengthAuditExecutionManifest | None = None,
    result: FinalStrengthAuditResult,
) -> dict[str, object]:
    """Atomically publish canonical raw rows and the receipt that authenticates them."""

    if not isinstance(sealed_plan, AuthenticatedFinalStrengthAuditPlan) or not isinstance(
        result, FinalStrengthAuditResult
    ):
        raise TypeError("audit publication requires typed sealed plan and result")
    plan = sealed_plan.plan
    if len(result.rows) != len(plan.opportunity_keys):
        raise ValueError("audit result and sealed opportunity census differ")
    for row, key in zip(result.rows, plan.opportunity_keys, strict=True):
        payload = {name: value for name, value in row.items() if name != "record_digest"}
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("schema_version") != ROW_VERSION
            or row.get("record_digest") != content_digest(payload)
            or row.get("audit_key_digest") != key.digest
        ):
            raise ValueError("audit result contains an invalid raw row")
    recomputed_aggregate = _aggregate_rows(result.rows, plan)
    if _plain_json(result.aggregate) != _plain_json(recomputed_aggregate):
        raise ValueError("audit aggregate differs from independent raw-row recomputation")
    manifest: FinalStrengthAuditExecutionManifest | None = None
    if sealed_execution_manifest is not None:
        if not isinstance(
            sealed_execution_manifest, AuthenticatedFinalStrengthAuditExecutionManifest
        ):
            raise TypeError("sealed_execution_manifest has the wrong type")
        manifest = sealed_execution_manifest.manifest
        manifest.validate_plan(sealed_plan)
        if (
            result.execution_manifest_digest != manifest.record_digest
            or result.source_runs != manifest.source_runs
            or result.sampler_identity_digest is None
        ):
            raise ValueError("audit result source or execution-manifest link differs")
        entries = {
            FinalStrengthAuditKey.from_mapping(row["audit_key"]): row  # type: ignore[arg-type]
            for row in manifest.opportunity_execution
        }
        for row, key in zip(result.rows, plan.opportunity_keys, strict=True):
            _validate_audit_row(
                row,
                key=key,
                plan=plan,
                execution_entry=entries[key],
                sampler_identity_digest=result.sampler_identity_digest,
            )
    elif plan.publication_eligible or result.publication_eligible:
        raise ValueError("publication audit requires the sealed execution manifest")
    publication_eligible = bool(
        plan.publication_eligible
        and manifest is not None
        and manifest.publication_eligible
        and result.publication_eligible
        and result.sampler_identity_digest == plan.sampler_identity_digest
    )
    merge_receipt: dict[str, object] | None = None
    if manifest is not None:
        if len(result.source_shards) != plan.shard_count:
            raise ValueError("merged audit result lacks the complete source-shard census")
        indices = [int(row.get("shard_index", -1)) for row in result.source_shards]
        if indices != list(range(plan.shard_count)):
            raise ValueError("merged audit source-shard census is unordered or incomplete")
        source_shard_fields = {
            "shard_index",
            "receipt_file_sha256",
            "receipt_record_digest",
            "rows_file_sha256",
            "paired_keys_digest",
            "opportunity_keys_digest",
            "seed_schedule_digest",
            "sampler_identity_digest",
        }
        if any(
            set(row) != source_shard_fields
            or any(
                not _is_digest(row[name])
                for name in source_shard_fields - {"shard_index"}
            )
            or row["sampler_identity_digest"] != result.sampler_identity_digest
            for row in result.source_shards
        ):
            raise ValueError("merged audit contains a malformed source-shard link")
        merge_payload: dict[str, object] = {
            "schema": MERGE_SCHEMA,
            "schema_version": MERGE_VERSION,
            "assignment": "canonical-sorted-paired-key-round-robin-v1",
            "shard_count": plan.shard_count,
            "complete_census": True,
            "plan_record_digest": plan.record_digest,
            "execution_manifest_record_digest": manifest.record_digest,
            "source_shards": [_plain_json(row) for row in result.source_shards],
            "source_shards_digest": content_digest(
                [_plain_json(row) for row in result.source_shards]
            ),
            "merged_opportunity_keys_digest": stable_digest(
                [key.as_dict() for key in plan.opportunity_keys]
            ),
            "merged_rows_digest": content_digest(
                [_plain_json(row) for row in result.rows]
            ),
            "aggregate_digest": content_digest(recomputed_aggregate),
        }
        merge_receipt = {
            **merge_payload,
            "record_digest": content_digest(merge_payload),
        }
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"final-strength audit output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        raw = b"".join(canonical_json_bytes(row) + b"\n" for row in result.rows)
        _write_sync(temporary / "rows.jsonl", raw)
        payload: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "schema_version": RECEIPT_VERSION,
            "scope": "post-freeze-diagnostic-only-no-feedback",
            "publication_eligible": publication_eligible,
            "plan": {
                "file_sha256": sealed_plan.file_sha256,
                "record_digest": plan.record_digest,
                "population_digest": plan.population_digest,
                "opportunity_keys_digest": stable_digest(
                    [key.as_dict() for key in plan.opportunity_keys]
                ),
            },
            "protocol": {
                "outcome_blind_subset": True,
                "primary_blocks_reused": False,
                "feedback_forbidden": True,
                "strength_count": 4,
                "fresh_blocks_per_strength": 2,
                "block_domains": dict(BLOCK_DOMAINS),
                "oracle_tie_rule": "lowest-strength-index",
                "regret_estimator": "block-a-selects-oracle-block-b-estimates-difference",
                "aggregation": AGGREGATION,
            },
            "raw_rows": {
                "path": "rows.jsonl",
                "schema": ROW_SCHEMA,
                "schema_version": ROW_VERSION,
                "count": len(result.rows),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            "execution_manifest": (
                None
                if manifest is None
                else {
                    "file_sha256": sealed_execution_manifest.file_sha256,
                    "record_digest": manifest.record_digest,
                    "opportunity_execution_digest": manifest.opportunity_execution_digest,
                    "source_report_pins_digest": manifest.source_report_pins_digest,
                    "target_access_digest": manifest.target_access_digest,
                }
            ),
            "sampler_identity_digest": result.sampler_identity_digest,
            "merge": merge_receipt,
            "source_runs": [_plain_json(row) for row in result.source_runs],
            "source_runs_digest": content_digest(
                [_plain_json(row) for row in result.source_runs]
            ),
            "aggregate": _plain_json(recomputed_aggregate),
            "aggregate_digest": content_digest(recomputed_aggregate),
            "audit_runtime": {
                "domain": AUDIT_RUNTIME_DOMAIN,
                "runtime_implementation_digest": plan.runtime_implementation_digest,
            },
        }
        receipt = {**payload, "record_digest": content_digest(payload)}
        receipt_bytes = canonical_json_bytes(receipt) + b"\n"
        _write_sync(temporary / "receipt.json", receipt_bytes)
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "ARMS",
    "BLOCK_DOMAINS",
    "CONFIG_SCHEMA",
    "CONFIG_VERSION",
    "EXECUTION_MANIFEST_SCHEMA",
    "EXECUTION_MANIFEST_VERSION",
    "MERGE_SCHEMA",
    "MERGE_VERSION",
    "SHARD_SCHEMA",
    "SHARD_VERSION",
    "AuthenticatedFinalStrengthAuditConfig",
    "AuthenticatedFinalStrengthAuditExecutionManifest",
    "AuthenticatedFinalStrengthAuditPlan",
    "AuthenticatedFinalStrengthAuditShard",
    "FinalStrengthAuditConfig",
    "FinalStrengthAuditError",
    "FinalStrengthAuditExecutionManifest",
    "FinalStrengthAuditKey",
    "FinalStrengthAuditPlan",
    "FinalStrengthAuditResult",
    "FinalStrengthAuditShardResult",
    "FinalStrengthSamplerIdentity",
    "SelectedAuditInstance",
    "build_final_strength_audit_execution_manifest",
    "build_final_strength_audit_plan",
    "load_final_strength_audit_config",
    "load_final_strength_audit_execution_manifest",
    "load_final_strength_audit_plan",
    "load_final_strength_audit_shard",
    "merge_final_strength_audit_shards",
    "publish_final_strength_audit",
    "publish_final_strength_audit_shard",
    "registered_final_strength_sampler_identity",
    "run_final_strength_audit",
    "run_final_strength_audit_shard",
    "write_final_strength_audit_execution_manifest",
    "write_final_strength_audit_plan",
]

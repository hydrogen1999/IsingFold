"""Deterministic, integrity-checked scientific checkpoints for IsingFold RL.

A checkpoint binds every immutable input that affects proposal support or
likelihood replay. It is therefore more than a model-weights container.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import importlib.util
import math
import os
import platform
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from isingfold.rl.contracts import OPCODES, stable_digest

CHECKPOINT_SCHEMA = "isingfold.scientific-training-checkpoint"
CHECKPOINT_VERSION = 3
ACTION_SCHEMA: tuple[str, ...] = OPCODES
PPO_TRAINER_STATE_SCHEMA: dict[str, str] = {
    "lambda_f": "number",
    "updates_done": "integer",
}
RUNTIME_IMPLEMENTATION_SCHEMA = "isingfold.rl-runtime-implementation"
RUNTIME_IMPLEMENTATION_VERSION = 2
RUNTIME_MODULE_SOURCES: tuple[tuple[str, str], ...] = (
    ("isingfold.embedding", "isingfold/embedding.py"),
    ("isingfold.rl", "isingfold/rl/__init__.py"),
    ("isingfold.rl.capacity_control", "isingfold/rl/capacity_control.py"),
    ("isingfold.rl.checkpoint", "isingfold/rl/checkpoint.py"),
    ("isingfold.rl.cli", "isingfold/rl/cli.py"),
    ("isingfold.rl.complete_system", "isingfold/rl/complete_system.py"),
    (
        "isingfold.rl.complete_system_aggregate",
        "isingfold/rl/complete_system_aggregate.py",
    ),
    ("isingfold.rl.contracts", "isingfold/rl/contracts.py"),
    ("isingfold.rl.data", "isingfold/rl/data/__init__.py"),
    (
        "isingfold.rl.data.action_certificate",
        "isingfold/rl/data/action_certificate.py",
    ),
    ("isingfold.rl.data.generate", "isingfold/rl/data/generate.py"),
    (
        "isingfold.rl.data.ground_certificate",
        "isingfold/rl/data/ground_certificate.py",
    ),
    ("isingfold.rl.data.import_embedbench", "isingfold/rl/data/import_embedbench.py"),
    ("isingfold.rl.data.import_release_v1", "isingfold/rl/data/import_release_v1.py"),
    ("isingfold.rl.data.inkdrop", "isingfold/rl/data/inkdrop.py"),
    ("isingfold.rl.data.lineage", "isingfold/rl/data/lineage.py"),
    ("isingfold.rl.data.planting", "isingfold/rl/data/planting.py"),
    ("isingfold.rl.data.prepared", "isingfold/rl/data/prepared.py"),
    ("isingfold.rl.data.program_oracle", "isingfold/rl/data/program_oracle.py"),
    ("isingfold.rl.data.quality", "isingfold/rl/data/quality.py"),
    (
        "isingfold.rl.data.quality_attestation",
        "isingfold/rl/data/quality_attestation.py",
    ),
    ("isingfold.rl.data.selector_labels", "isingfold/rl/data/selector_labels.py"),
    ("isingfold.rl.data.structural", "isingfold/rl/data/structural.py"),
    (
        "isingfold.rl.data.structural_oracle",
        "isingfold/rl/data/structural_oracle.py",
    ),
    ("isingfold.rl.env", "isingfold/rl/env.py"),
    ("isingfold.rl.evaluate", "isingfold/rl/evaluate.py"),
    ("isingfold.rl.evaluation_strata", "isingfold/rl/evaluation_strata.py"),
    ("isingfold.rl.evaluator", "isingfold/rl/evaluator.py"),
    ("isingfold.rl.experiment_selection", "isingfold/rl/experiment_selection.py"),
    ("isingfold.rl.external", "isingfold/rl/external.py"),
    ("isingfold.rl.external_pairing", "isingfold/rl/external_pairing.py"),
    ("isingfold.rl.external_tuning", "isingfold/rl/external_tuning.py"),
    ("isingfold.rl.final_strength_audit", "isingfold/rl/final_strength_audit.py"),
    ("isingfold.rl.gates", "isingfold/rl/gates.py"),
    ("isingfold.rl.model", "isingfold/rl/model.py"),
    ("isingfold.rl.ppo", "isingfold/rl/ppo.py"),
    ("isingfold.rl.program", "isingfold/rl/program.py"),
    ("isingfold.rl.proposal", "isingfold/rl/proposal.py"),
    ("isingfold.rl.q_calibration_audit", "isingfold/rl/q_calibration_audit.py"),
    ("isingfold.rl.quality_warm_control", "isingfold/rl/quality_warm_control.py"),
    ("isingfold.rl.rollout", "isingfold/rl/rollout.py"),
    ("isingfold.rl.router", "isingfold/rl/router.py"),
    ("isingfold.rl.selector_audit", "isingfold/rl/selector_audit.py"),
    ("isingfold.rl.strength", "isingfold/rl/strength.py"),
    ("isingfold.rl.strength_tensorize", "isingfold/rl/strength_tensorize.py"),
    ("isingfold.rl.tensorize", "isingfold/rl/tensorize.py"),
    ("isingfold.rl.validate", "isingfold/rl/validate.py"),
    ("lac_minorminer", "lac_minorminer/__init__.py"),
    ("lac_minorminer._graph_input", "lac_minorminer/_graph_input.py"),
    ("lac_minorminer._options", "lac_minorminer/_options.py"),
    ("lac_minorminer._policy_validation", "lac_minorminer/_policy_validation.py"),
    ("lac_minorminer._search_loop", "lac_minorminer/_search_loop.py"),
    ("lac_minorminer._version", "lac_minorminer/_version.py"),
    ("lac_minorminer.api", "lac_minorminer/api.py"),
    ("lac_minorminer.components", "lac_minorminer/components.py"),
    ("lac_minorminer.diagnostics", "lac_minorminer/diagnostics.py"),
    ("lac_minorminer.orchestrator", "lac_minorminer/orchestrator.py"),
    ("lac_minorminer.quality_v2", "lac_minorminer/quality_v2.py"),
    ("lac_minorminer.scoring", "lac_minorminer/scoring.py"),
    ("lac_minorminer.session", "lac_minorminer/session.py"),
    ("lac_minorminer.validation", "lac_minorminer/validation.py"),
)
RUNTIME_NATIVE_MODULES: tuple[str, ...] = ("lac_minorminer._core",)
RUNTIME_DEPENDENCIES: tuple[str, ...] = (
    "dimod",
    "dwave-samplers",
    "minorminer",
    "networkx",
    "numpy",
    "torch",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointIntegrityError(CheckpointError):
    """The artifact is malformed or no longer matches its digest."""


class CheckpointCompatibilityError(CheckpointError):
    """The artifact belongs to a different immutable experiment contract."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Validated experiment identity and progress."""

    schema: str
    schema_version: int
    context_snapshot: dict[str, object]
    context_digest: str
    feature_normalizer: dict[str, object]
    feature_normalizer_digest: str
    action_schema: tuple[str, ...]
    action_schema_digest: str
    proposal_version: str
    selector_digest: str
    training_lineages: tuple[str, ...]
    training_lineages_digest: str
    counters: dict[str, int]
    trainer_state: dict[str, object]
    trainer_state_digest: str
    trainer_state_schema_digest: str
    runtime_implementation_registry: dict[str, object]
    runtime_implementation_digest: str
    model_schema_digest: str
    optimizer_schema_digest: str
    model_state_digest: str
    optimizer_state_digest: str
    rng_state_digest: str
    configuration_digest: str
    payload_digest: str


_METADATA_FIELDS = frozenset(
    {
        "context_snapshot",
        "context_digest",
        "feature_normalizer",
        "feature_normalizer_digest",
        "action_schema",
        "action_schema_digest",
        "proposal_version",
        "selector_digest",
        "training_lineages",
        "training_lineages_digest",
        "counters",
        "trainer_state_digest",
        "trainer_state_schema",
        "trainer_state_schema_digest",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "model_schema",
        "model_schema_digest",
        "optimizer_schema",
        "optimizer_schema_digest",
        "model_state_digest",
        "optimizer_state_digest",
        "rng_state_digest",
        "configuration_digest",
    }
)


def runtime_implementation_registry() -> dict[str, object]:
    """Return the canonical source and dependency identity of the RL runtime.

    Paths are logical paths relative to ``src`` rather than machine-specific
    absolute paths. Source files are hashed as bytes so even a protocol change
    that forgot to bump a manually maintained version cannot resume silently.
    """

    source_root = Path(__file__).resolve().parents[2]
    modules: dict[str, object] = {}
    for module_name, relative_path in RUNTIME_MODULE_SOURCES:
        source_path = source_root / relative_path
        try:
            source = source_path.read_bytes()
        except OSError as exc:
            raise CheckpointCompatibilityError(
                f"cannot identify runtime implementation source {relative_path!r}: {exc}"
            ) from exc
        modules[module_name] = {
            "source_path": relative_path,
            "sha256": hashlib.sha256(source).hexdigest(),
        }

    dependencies: dict[str, str] = {
        "python": f"{platform.python_implementation()} {platform.python_version()}"
    }
    for distribution in RUNTIME_DEPENDENCIES:
        try:
            dependencies[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            # Optional executors such as minorminer are still semantic state:
            # installing one between save and resume changes proposal support.
            dependencies[distribution] = "not-installed"

    native_artifacts: dict[str, object] = {}
    for module_name in RUNTIME_NATIVE_MODULES:
        spec = importlib.util.find_spec(module_name)
        origin = None if spec is None else spec.origin
        if origin is None or not Path(origin).is_file():
            raise CheckpointCompatibilityError(
                f"cannot identify required native runtime artifact {module_name!r}"
            )
        native_artifacts[module_name] = {
            "artifact_kind": "extension-module",
            "sha256": hashlib.sha256(Path(origin).read_bytes()).hexdigest(),
        }

    return _normalise_runtime_registry(
        {
            "schema": RUNTIME_IMPLEMENTATION_SCHEMA,
            "schema_version": RUNTIME_IMPLEMENTATION_VERSION,
            "modules": modules,
            "native_artifacts": native_artifacts,
            "dependencies": dependencies,
        },
        "runtime implementation registry",
    )


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    context: object,
    feature_normalizer: object,
    proposal_version: str,
    selector_digest: str,
    training_lineages: Sequence[str],
    counters: Mapping[str, int],
    trainer_state: Mapping[str, object] | None = None,
    action_schema: Sequence[str] = ACTION_SCHEMA,
    runtime_registry: Mapping[str, object] | None = None,
) -> CheckpointMetadata:
    """Atomically save all state needed for exact scientific continuation."""

    context_snapshot = _mapping_snapshot(context, "context")
    normalizer = _mapping_snapshot(feature_normalizer, "feature normalizer")
    actions = _normalise_actions(action_schema)
    proposal = _nonempty(proposal_version, "proposal version")
    selector = _sha256(selector_digest, "selector digest")
    lineages = _normalise_lineages(training_lineages)
    progress = _normalise_counters(counters)
    mutable_trainer_state = _mapping_snapshot(
        {} if trainer_state is None else trainer_state, "trainer state"
    )
    trainer_state_schema = _trainer_state_schema(mutable_trainer_state)
    implementation_registry = _normalise_runtime_registry(
        runtime_implementation_registry() if runtime_registry is None else runtime_registry,
        "runtime implementation registry",
    )

    model_state = _clone_cpu(model.state_dict())
    optimizer_state = _clone_cpu(optimizer.state_dict())
    rng_state = _capture_rng()
    model_schema = _model_schema(model_state)
    optimizer_schema = _optimizer_schema(optimizer)

    context_digest = stable_digest(context_snapshot)
    normalizer_digest = stable_digest(normalizer)
    action_digest = stable_digest(list(actions))
    lineage_digest = stable_digest(list(lineages))
    model_schema_digest = stable_digest(model_schema)
    optimizer_schema_digest = stable_digest(optimizer_schema)
    model_state_digest = _content_digest(model_state)
    optimizer_state_digest = _content_digest(optimizer_state)
    rng_state_digest = _content_digest(rng_state)
    trainer_state_digest = stable_digest(mutable_trainer_state)
    trainer_state_schema_digest = stable_digest(trainer_state_schema)
    implementation_digest = stable_digest(implementation_registry)
    configuration = {
        "context_digest": context_digest,
        "feature_normalizer_digest": normalizer_digest,
        "action_schema_digest": action_digest,
        "proposal_version": proposal,
        "selector_digest": selector,
        "training_lineages_digest": lineage_digest,
        "model_schema_digest": model_schema_digest,
        "optimizer_schema_digest": optimizer_schema_digest,
        "trainer_state_schema_digest": trainer_state_schema_digest,
        "runtime_implementation_digest": implementation_digest,
    }
    metadata: dict[str, object] = {
        "context_snapshot": context_snapshot,
        "context_digest": context_digest,
        "feature_normalizer": normalizer,
        "feature_normalizer_digest": normalizer_digest,
        "action_schema": list(actions),
        "action_schema_digest": action_digest,
        "proposal_version": proposal,
        "selector_digest": selector,
        "training_lineages": list(lineages),
        "training_lineages_digest": lineage_digest,
        "counters": progress,
        "trainer_state_digest": trainer_state_digest,
        "trainer_state_schema": trainer_state_schema,
        "trainer_state_schema_digest": trainer_state_schema_digest,
        "runtime_implementation_registry": implementation_registry,
        "runtime_implementation_digest": implementation_digest,
        "model_schema": model_schema,
        "model_schema_digest": model_schema_digest,
        "optimizer_schema": optimizer_schema,
        "optimizer_schema_digest": optimizer_schema_digest,
        "model_state_digest": model_state_digest,
        "optimizer_state_digest": optimizer_state_digest,
        "rng_state_digest": rng_state_digest,
        "configuration_digest": stable_digest(configuration),
    }
    payload: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_VERSION,
        "metadata": metadata,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "trainer_state": mutable_trainer_state,
        "rng_state": rng_state,
    }
    payload_digest = _content_digest(payload)
    payload["payload_digest"] = payload_digest
    _atomic_save(payload, Path(path))
    return _parse_metadata(metadata, payload_digest, mutable_trainer_state)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    expected_context: object,
    expected_feature_normalizer: object,
    expected_proposal_version: str,
    expected_selector_digest: str,
    expected_training_lineages: Sequence[str],
    expected_action_schema: Sequence[str] = ACTION_SCHEMA,
    expected_trainer_state_schema: Mapping[str, str] | None = None,
    expected_runtime_registry: Mapping[str, object] | None = None,
    expected_runtime_digest: str | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    """Validate compatibility before restoring model, optimizer, and RNG state."""

    try:
        raw = torch.load(Path(path), map_location=map_location, weights_only=True)
    except Exception as exc:  # exact parser exceptions vary by torch version
        raise CheckpointIntegrityError(f"cannot read checkpoint: {exc}") from exc
    payload = _mapping(raw, "checkpoint payload")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointCompatibilityError(
            f"checkpoint schema mismatch: expected {CHECKPOINT_SCHEMA!r}, "
            f"got {payload.get('schema')!r}"
        )
    if payload.get("schema_version") != CHECKPOINT_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint schema version mismatch: expected {CHECKPOINT_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    top_fields = {
        "schema",
        "schema_version",
        "metadata",
        "model_state",
        "optimizer_state",
        "trainer_state",
        "rng_state",
        "payload_digest",
    }
    if set(payload) != top_fields:
        raise CheckpointIntegrityError("checkpoint payload has missing or unknown fields")
    recorded_digest = _sha256(payload["payload_digest"], "payload digest")
    unsigned = {key: value for key, value in payload.items() if key != "payload_digest"}
    if not hmac.compare_digest(recorded_digest, _content_digest(unsigned)):
        raise CheckpointIntegrityError("payload digest mismatch")

    raw_metadata = _mapping(payload["metadata"], "checkpoint metadata")
    if set(raw_metadata) != _METADATA_FIELDS:
        raise CheckpointIntegrityError("checkpoint metadata has missing or unknown fields")
    trainer_state = _stored_snapshot(payload["trainer_state"], "trainer state")
    metadata = _parse_metadata(raw_metadata, recorded_digest, trainer_state)
    model_state = _mapping(payload["model_state"], "model state")
    optimizer_state = _mapping(payload["optimizer_state"], "optimizer state")
    rng_state = _mapping(payload["rng_state"], "RNG state")
    _verify_digest(model_state, metadata.model_state_digest, "model state")
    _verify_digest(optimizer_state, metadata.optimizer_state_digest, "optimizer state")
    _verify_digest(rng_state, metadata.rng_state_digest, "RNG state")
    if stable_digest(_model_schema(model_state)) != metadata.model_schema_digest:
        raise CheckpointIntegrityError("stored model schema digest mismatch")

    implementation_digest = _expected_runtime_implementation_digest(
        expected_runtime_registry,
        expected_runtime_digest,
    )
    expected_context_digest = stable_digest(_mapping_snapshot(expected_context, "expected context"))
    expected_normalizer_digest = stable_digest(
        _mapping_snapshot(expected_feature_normalizer, "expected feature normalizer")
    )
    expected_actions = _normalise_actions(expected_action_schema)
    expected_proposal = _nonempty(expected_proposal_version, "expected proposal version")
    expected_selector = _sha256(expected_selector_digest, "expected selector digest")
    _compatible(expected_context_digest, metadata.context_digest, "context")
    _compatible(
        expected_normalizer_digest, metadata.feature_normalizer_digest, "feature normalizer"
    )
    _compatible(
        stable_digest(list(expected_actions)), metadata.action_schema_digest, "action schema"
    )
    _compatible(expected_proposal, metadata.proposal_version, "proposal version")
    _compatible(expected_selector, metadata.selector_digest, "selector digest")
    _compatible(
        implementation_digest,
        metadata.runtime_implementation_digest,
        "runtime implementation",
    )
    expected_lineages = _normalise_lineages(expected_training_lineages)
    _compatible(
        stable_digest(list(expected_lineages)),
        metadata.training_lineages_digest,
        "training lineage",
    )
    _compatible(
        stable_digest(_model_schema(model.state_dict())),
        metadata.model_schema_digest,
        "model schema",
    )
    _compatible(
        stable_digest(_optimizer_schema(optimizer)),
        metadata.optimizer_schema_digest,
        "optimizer schema",
    )
    expected_trainer_schema = _normalise_trainer_state_schema(
        {} if expected_trainer_state_schema is None else expected_trainer_state_schema
    )
    _compatible(
        stable_digest(expected_trainer_schema),
        metadata.trainer_state_schema_digest,
        "trainer state schema",
    )
    if restore_rng:
        _validate_rng(rng_state)

    previous_model = _clone_cpu(model.state_dict())
    previous_optimizer = _clone_cpu(optimizer.state_dict())
    previous_rng = _capture_rng() if restore_rng else None
    try:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        if restore_rng:
            _restore_rng(rng_state)
    except Exception:
        model.load_state_dict(previous_model, strict=True)
        optimizer.load_state_dict(previous_optimizer)
        if previous_rng is not None:
            _restore_rng(previous_rng)
        raise
    return metadata


def _parse_metadata(
    raw: Mapping[Any, Any],
    payload_digest: str,
    trainer_state: dict[str, object],
) -> CheckpointMetadata:
    context = _stored_snapshot(raw["context_snapshot"], "stored context")
    normalizer = _stored_snapshot(raw["feature_normalizer"], "stored feature normalizer")
    actions = _normalise_actions(_sequence(raw["action_schema"], "action schema"))
    proposal = _nonempty(raw["proposal_version"], "proposal version")
    selector = _sha256(raw["selector_digest"], "selector digest")
    lineages = _normalise_lineages(_sequence(raw["training_lineages"], "lineages"))
    counters = _normalise_counters(_mapping(raw["counters"], "counters"))
    model_schema = _stored_snapshot(raw["model_schema"], "model schema")
    optimizer_schema = _stored_snapshot(raw["optimizer_schema"], "optimizer schema")
    trainer_state_schema = _stored_snapshot(
        raw["trainer_state_schema"], "trainer state schema"
    )
    implementation_registry = _stored_runtime_registry(
        raw["runtime_implementation_registry"]
    )
    if trainer_state_schema != _trainer_state_schema(trainer_state):
        raise CheckpointIntegrityError("trainer state schema differs from stored state")
    context_digest = _json_digest(raw["context_digest"], context, "context")
    normalizer_digest = _json_digest(
        raw["feature_normalizer_digest"], normalizer, "feature normalizer"
    )
    action_digest = _json_digest(raw["action_schema_digest"], list(actions), "action schema")
    lineage_digest = _json_digest(
        raw["training_lineages_digest"], list(lineages), "training lineages"
    )
    model_schema_digest = _json_digest(
        raw["model_schema_digest"], model_schema, "model schema"
    )
    optimizer_schema_digest = _json_digest(
        raw["optimizer_schema_digest"], optimizer_schema, "optimizer schema"
    )
    trainer_state_digest = _json_digest(
        raw["trainer_state_digest"], trainer_state, "trainer state"
    )
    trainer_state_schema_digest = _json_digest(
        raw["trainer_state_schema_digest"],
        trainer_state_schema,
        "trainer state schema",
    )
    implementation_digest = _json_digest(
        raw["runtime_implementation_digest"],
        implementation_registry,
        "runtime implementation",
    )
    configuration = {
        "context_digest": context_digest,
        "feature_normalizer_digest": normalizer_digest,
        "action_schema_digest": action_digest,
        "proposal_version": proposal,
        "selector_digest": selector,
        "training_lineages_digest": lineage_digest,
        "model_schema_digest": model_schema_digest,
        "optimizer_schema_digest": optimizer_schema_digest,
        "trainer_state_schema_digest": trainer_state_schema_digest,
        "runtime_implementation_digest": implementation_digest,
    }
    configuration_digest = _json_digest(
        raw["configuration_digest"], configuration, "configuration"
    )
    return CheckpointMetadata(
        schema=CHECKPOINT_SCHEMA,
        schema_version=CHECKPOINT_VERSION,
        context_snapshot=context,
        context_digest=context_digest,
        feature_normalizer=normalizer,
        feature_normalizer_digest=normalizer_digest,
        action_schema=actions,
        action_schema_digest=action_digest,
        proposal_version=proposal,
        selector_digest=selector,
        training_lineages=lineages,
        training_lineages_digest=lineage_digest,
        counters=counters,
        trainer_state=trainer_state,
        trainer_state_digest=trainer_state_digest,
        trainer_state_schema_digest=trainer_state_schema_digest,
        runtime_implementation_registry=implementation_registry,
        runtime_implementation_digest=implementation_digest,
        model_schema_digest=model_schema_digest,
        optimizer_schema_digest=optimizer_schema_digest,
        model_state_digest=_sha256(raw["model_state_digest"], "model state digest"),
        optimizer_state_digest=_sha256(
            raw["optimizer_state_digest"], "optimizer state digest"
        ),
        rng_state_digest=_sha256(raw["rng_state_digest"], "RNG state digest"),
        configuration_digest=configuration_digest,
        payload_digest=payload_digest,
    )


def _mapping_snapshot(value: object, label: str) -> dict[str, object]:
    snapshot = _json_snapshot(value, label)
    if not isinstance(snapshot, dict):
        raise ValueError(f"{label} must be a mapping or dataclass")
    return snapshot


def _stored_snapshot(value: object, label: str) -> dict[str, object]:
    try:
        return _mapping_snapshot(value, label)
    except (TypeError, ValueError) as exc:
        raise CheckpointIntegrityError(str(exc)) from exc


def _normalise_runtime_registry(value: object, label: str) -> dict[str, object]:
    registry = _mapping_snapshot(value, label)
    expected_fields = {
        "schema",
        "schema_version",
        "modules",
        "native_artifacts",
        "dependencies",
    }
    if set(registry) != expected_fields:
        raise ValueError(f"{label} has missing or unknown fields")
    if registry["schema"] != RUNTIME_IMPLEMENTATION_SCHEMA:
        raise ValueError(f"{label} schema must be {RUNTIME_IMPLEMENTATION_SCHEMA!r}")
    version = registry["schema_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != RUNTIME_IMPLEMENTATION_VERSION
    ):
        raise ValueError(
            f"{label} schema version must be {RUNTIME_IMPLEMENTATION_VERSION}"
        )

    raw_modules = registry["modules"]
    if not isinstance(raw_modules, dict):
        raise ValueError(f"{label} modules must be a mapping")
    expected_sources = dict(RUNTIME_MODULE_SOURCES)
    if set(raw_modules) != set(expected_sources):
        raise ValueError(f"{label} modules have missing or unknown entries")
    modules: dict[str, object] = {}
    for module_name in sorted(expected_sources):
        record = raw_modules[module_name]
        if not isinstance(record, dict) or set(record) != {"source_path", "sha256"}:
            raise ValueError(f"{label} module {module_name!r} is malformed")
        source_path = record["source_path"]
        if source_path != expected_sources[module_name]:
            raise ValueError(f"{label} module {module_name!r} has an unexpected source path")
        modules[module_name] = {
            "source_path": source_path,
            "sha256": _sha256(record["sha256"], f"{label} module {module_name!r} source"),
        }

    raw_dependencies = registry["dependencies"]
    if not isinstance(raw_dependencies, dict):
        raise ValueError(f"{label} dependencies must be a mapping")
    expected_dependencies = {"python", *RUNTIME_DEPENDENCIES}
    if set(raw_dependencies) != expected_dependencies:
        raise ValueError(f"{label} dependencies have missing or unknown entries")
    dependencies = {
        name: _nonempty(raw_dependencies[name], f"{label} dependency {name!r} version")
        for name in sorted(expected_dependencies)
    }
    raw_native = registry["native_artifacts"]
    if not isinstance(raw_native, dict) or set(raw_native) != set(RUNTIME_NATIVE_MODULES):
        raise ValueError(f"{label} native artifacts have missing or unknown entries")
    native_artifacts: dict[str, object] = {}
    for module_name in sorted(RUNTIME_NATIVE_MODULES):
        record = raw_native[module_name]
        if (
            not isinstance(record, dict)
            or set(record) != {"artifact_kind", "sha256"}
            or record["artifact_kind"] != "extension-module"
        ):
            raise ValueError(f"{label} native artifact {module_name!r} is malformed")
        native_artifacts[module_name] = {
            "artifact_kind": "extension-module",
            "sha256": _sha256(
                record["sha256"], f"{label} native artifact {module_name!r}"
            ),
        }
    return {
        "schema": RUNTIME_IMPLEMENTATION_SCHEMA,
        "schema_version": RUNTIME_IMPLEMENTATION_VERSION,
        "modules": modules,
        "native_artifacts": native_artifacts,
        "dependencies": dependencies,
    }


def _stored_runtime_registry(value: object) -> dict[str, object]:
    try:
        return _normalise_runtime_registry(value, "stored runtime implementation registry")
    except (TypeError, ValueError) as exc:
        raise CheckpointIntegrityError(str(exc)) from exc


def _expected_runtime_implementation_digest(
    registry: Mapping[str, object] | None,
    recorded_digest: str | None,
) -> str:
    current = _normalise_runtime_registry(
        runtime_implementation_registry() if registry is None else registry,
        "expected runtime implementation registry",
    )
    current_digest = stable_digest(current)
    if recorded_digest is not None:
        asserted_digest = _sha256(
            recorded_digest, "expected runtime implementation digest"
        )
        if not hmac.compare_digest(asserted_digest, current_digest):
            if registry is None:
                raise CheckpointCompatibilityError(
                    "runtime implementation differs from the explicitly expected digest"
                )
            raise ValueError(
                "expected runtime implementation digest differs from the expected registry"
            )
    return current_digest


def _json_snapshot(value: object, label: str) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_snapshot(getattr(value, field.name), f"{label}.{field.name}")
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return _json_snapshot(value.value, label)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{label} must contain only finite numbers")
        return 0.0 if result == 0.0 else result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_snapshot(value.tolist(), label)
    if isinstance(value, Tensor):
        return _json_snapshot(value.detach().cpu().tolist(), label)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{label} mapping keys must be strings")
        for key in sorted(value):
            result[key] = _json_snapshot(value[key], f"{label}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_snapshot(item, f"{label}[]") for item in value]
    raise TypeError(f"{label} contains unsupported value of type {type(value).__name__}")


def _normalise_actions(schema: Sequence[str]) -> tuple[str, ...]:
    if isinstance(schema, (str, bytes)):
        raise ValueError("action schema must be a sequence of opcode names")
    actions = tuple(schema)
    if not actions or any(not isinstance(action, str) or not action for action in actions):
        raise ValueError("action schema must contain nonempty strings")
    if len(set(actions)) != len(actions):
        raise ValueError("action schema contains duplicate opcodes")
    return actions


def _normalise_lineages(lineages: Sequence[str]) -> tuple[str, ...]:
    if isinstance(lineages, (str, bytes)):
        raise ValueError("training lineages must be a sequence")
    result = tuple(lineages)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError("training lineages must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("training lineages contain duplicates")
    return tuple(sorted(result))


def _normalise_counters(counters: Mapping[Any, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    if any(not isinstance(key, str) or not key for key in counters):
        raise ValueError("counter names must be nonempty strings")
    for key in sorted(counters):
        value = counters[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"counter {key!r} must be a nonnegative integer")
        result[key] = value
    return result


def _trainer_state_schema(state: Mapping[str, object]) -> dict[str, str]:
    """Return the exact top-level finite-JSON type contract for mutable trainer state."""

    result: dict[str, str] = {}
    for key in sorted(state):
        value = state[key]
        if value is None:
            kind = "null"
        elif isinstance(value, bool):
            kind = "boolean"
        elif isinstance(value, int):
            kind = "integer"
        elif isinstance(value, float):
            kind = "number"
        elif isinstance(value, str):
            kind = "string"
        elif isinstance(value, list):
            kind = "array"
        elif isinstance(value, dict):
            kind = "object"
        else:  # pragma: no cover - _mapping_snapshot rejects this first
            raise TypeError(f"unsupported trainer state value {type(value).__name__}")
        result[key] = kind
    return result


def _normalise_trainer_state_schema(schema: Mapping[str, str]) -> dict[str, str]:
    allowed = {"null", "boolean", "integer", "number", "string", "array", "object"}
    if any(not isinstance(key, str) or not key for key in schema):
        raise ValueError("trainer state schema keys must be nonempty strings")
    result = dict(sorted(schema.items()))
    if any(not isinstance(kind, str) or kind not in allowed for kind in result.values()):
        raise ValueError("trainer state schema contains an unsupported JSON type")
    return result


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_digest(recorded: object, value: object, label: str) -> str:
    try:
        digest = _sha256(recorded, f"{label} digest")
    except ValueError as exc:
        raise CheckpointIntegrityError(str(exc)) from exc
    if not hmac.compare_digest(digest, stable_digest(value)):
        raise CheckpointIntegrityError(f"{label} digest mismatch")
    return digest


def _mapping(value: object, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CheckpointIntegrityError(f"{label} must be a sequence")
    return value


def _compatible(current: str, stored: str, label: str) -> None:
    if not hmac.compare_digest(current, stored):
        raise CheckpointCompatibilityError(f"{label} mismatch")


def _clone_cpu(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {_clone_cpu(key): _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _model_schema(state: Mapping[Any, Any]) -> dict[str, object]:
    schema: dict[str, object] = {}
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise CheckpointIntegrityError("model state must map string names to tensors")
        schema[name] = {
            "dtype": str(value.dtype),
            "layout": str(value.layout),
            "shape": list(value.shape),
        }
    return schema


def _optimizer_schema(optimizer: Optimizer) -> dict[str, object]:
    state = optimizer.state_dict()
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "parameter_group_sizes": [len(group["params"]) for group in state["param_groups"]],
    }


def _capture_rng() -> dict[str, object]:
    numpy_state = np.random.get_state()
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": [int(value) for value in numpy_state[1]],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda_device_count": cuda_count,
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
    }


def _validate_rng(state: Mapping[Any, Any]) -> None:
    expected = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda_device_count",
        "torch_cuda",
    }
    if set(state) != expected or not isinstance(state["python"], tuple):
        raise CheckpointIntegrityError("RNG state is malformed")
    numpy_state = _mapping(state["numpy"], "NumPy RNG state")
    numpy_fields = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    if set(numpy_state) != numpy_fields:
        raise CheckpointIntegrityError("NumPy RNG state has missing or unknown fields")
    if numpy_state["bit_generator"] != "MT19937":
        raise CheckpointCompatibilityError("NumPy RNG bit generator mismatch")
    keys = numpy_state["keys"]
    if not isinstance(keys, list) or len(keys) != 624 or any(
        not isinstance(value, int) or not 0 <= value < 2**32 for value in keys
    ):
        raise CheckpointIntegrityError("NumPy RNG keys are malformed")
    if not isinstance(state["torch_cpu"], Tensor) or state["torch_cpu"].dtype != torch.uint8:
        raise CheckpointIntegrityError("Torch CPU RNG state is malformed")
    saved_cuda_count = state["torch_cuda_device_count"]
    cuda_states = state["torch_cuda"]
    if not isinstance(saved_cuda_count, int) or saved_cuda_count < 0:
        raise CheckpointIntegrityError("Torch CUDA device count is malformed")
    if not isinstance(cuda_states, list) or len(cuda_states) != saved_cuda_count:
        raise CheckpointIntegrityError("Torch CUDA RNG state count is malformed")
    current_cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved_cuda_count != current_cuda_count:
        raise CheckpointCompatibilityError(
            "Torch CUDA RNG device-count mismatch; disable RNG restore only for inference"
        )
    if any(not isinstance(item, Tensor) or item.dtype != torch.uint8 for item in cuda_states):
        raise CheckpointIntegrityError("Torch CUDA RNG state is malformed")


def _restore_rng(state: Mapping[Any, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda_device_count"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _verify_digest(value: object, recorded: str, label: str) -> None:
    if not hmac.compare_digest(_content_digest(value), recorded):
        raise CheckpointIntegrityError(f"{label} digest mismatch")


def _content_digest(value: object) -> str:
    hasher = hashlib.sha256()
    _update_hash(hasher, value)
    return hasher.hexdigest()


def _feed(hasher: Any, tag: bytes, data: bytes = b"") -> None:
    hasher.update(tag)
    hasher.update(len(data).to_bytes(8, "big"))
    hasher.update(data)


def _update_hash(hasher: Any, value: object) -> None:
    if value is None:
        _feed(hasher, b"none")
    elif isinstance(value, bool):
        _feed(hasher, b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _feed(hasher, b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint must contain only finite numbers")
        _feed(hasher, b"float", value.hex().encode("ascii"))
    elif isinstance(value, str):
        _feed(hasher, b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _feed(hasher, b"bytes", value)
    elif isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.layout != torch.strided:
            raise ValueError("checkpoint supports only strided tensors")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError("checkpoint must contain only finite tensors")
        descriptor = stable_digest(
            {
                "dtype": str(tensor.dtype),
                "layout": str(tensor.layout),
                "shape": list(tensor.shape),
            }
        ).encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        _feed(hasher, b"tensor-descriptor", descriptor)
        _feed(hasher, b"tensor-data", raw)
    elif isinstance(value, Mapping):
        _feed(hasher, b"mapping", str(len(value)).encode("ascii"))
        keyed = [(_content_digest(key), key, item) for key, item in value.items()]
        for _, key, item in sorted(keyed, key=lambda entry: entry[0]):
            _update_hash(hasher, key)
            _update_hash(hasher, item)
    elif isinstance(value, tuple):
        _feed(hasher, b"tuple", str(len(value)).encode("ascii"))
        for item in value:
            _update_hash(hasher, item)
    elif isinstance(value, list):
        _feed(hasher, b"list", str(len(value)).encode("ascii"))
        for item in value:
            _update_hash(hasher, item)
    else:
        raise TypeError(f"unsupported checkpoint value of type {type(value).__name__}")


def _atomic_save(payload: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):  # non-POSIX fallback
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

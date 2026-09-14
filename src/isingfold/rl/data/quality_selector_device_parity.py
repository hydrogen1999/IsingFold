"""Exact CPU/CUDA parity audit for the frozen quality strength selector.

The audit is deliberately target-free.  It replays the complete all-train quality state
schedule twice, once per selector device, and compares the selector decision for every
return-valid archive embedding visible at every scheduled state.  The two passes also
recompile and re-tensorize each case so drift outside the neural forward pass is detected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from isingfold.rl.contracts import Context
from isingfold.rl.data.exact_conformance import (
    ExactConformanceError,
    publish_new_file,
    read_regular_file,
)
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import PreparedTask
from isingfold.rl.data.quality import (
    replay_decision_with_action_envelope,
    validate_quality_initializer_bank_contract,
)
from isingfold.rl.data.quality_resolution import (
    QualityResolutionConfig,
    QualityResolutionError,
    validate_quality_resolution_config,
)
from isingfold.rl.data.quality_resolution_plan import (
    ResolutionProductionPlan,
    _validated_production,
    quality_action_provenance_fingerprint,
)
from isingfold.rl.env import StrengthSelector, chain_key
from isingfold.rl.evaluate import program_digest
from isingfold.rl.initializer_bank import InitializerSnapshotBank
from isingfold.rl.program import compile_registry, strength_registry
from isingfold.rl.strength_tensorize import build_strength_inputs


QUALITY_SELECTOR_DEVICE_PARITY_SCHEMA = "isingfold.quality-selector-device-parity"
QUALITY_SELECTOR_DEVICE_PARITY_VERSION = 1

_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_FIELDS = {
    "accelerator_device",
    "all_equal",
    "census",
    "context",
    "corpus",
    "cpu_device",
    "initializer_bank",
    "mismatch_count",
    "mismatches",
    "quality_protocol",
    "record_digest",
    "rows",
    "runtime",
    "schema",
    "schema_version",
    "selector",
}
_CORPUS_FIELDS = {
    "manifest_record_digest",
    "manifest_sha256",
    "train_census_record_digest",
    "train_lineage_count",
    "train_task_count",
}
_SELECTOR_FIELDS = {
    "fit_receipt_record_digest",
    "fit_receipt_sha256",
    "normalizer_digest",
    "normalizer_file_sha256",
    "selector_digest",
    "selector_file_sha256",
}
_QUALITY_PROTOCOL_FIELDS = {
    "config",
    "config_digest",
    "config_sha256",
    "production_plan_digest",
    "production_row_count",
}
_CONTEXT_FIELDS = {"digest", "snapshot"}
_RUNTIME_FIELDS = {
    "cuda_runtime_version",
    "deterministic_algorithms",
    "execution_runtime_sha256",
    "implementation",
    "platform",
    "python_version",
    "threads",
    "torch_version",
}
_IMPLEMENTATION_FIELDS = {
    "parity_module_sha256",
    "program_module_sha256",
    "selector_module_sha256",
    "tensorizer_module_sha256",
}
_DEVICE_FIELDS = {"capability", "index", "name", "requested", "type"}
_CENSUS_FIELDS = {
    "matching_cases",
    "mismatching_cases",
    "scheduled_quality_rows",
    "selector_cases",
    "train_lineages",
    "train_tasks",
}
_CASE_IDENTITY_FIELDS = {
    "archive_index",
    "archive_key",
    "base_lineage",
    "compiled_program_digests",
    "instance_id",
    "lineage_schedule_index",
    "quality_row_id",
    "selected_embedding_digest",
    "selector_input_digest",
    "state_fingerprint",
    "state_schedule_index",
    "support_fingerprint",
    "task_id",
}
_ROW_FIELDS = _CASE_IDENTITY_FIELDS | {"accelerator", "case_id", "cpu", "equal"}
_OUTCOME_FIELDS = {
    "compiled_program_digests",
    "selected_embedding_digest",
    "selected_strength_index",
}


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise QualityResolutionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise QualityResolutionError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise QualityResolutionError(f"{label} must be a nonnegative integer")
    return value


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QualityResolutionError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise QualityResolutionError(f"{label} contains non-finite number {token}")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except QualityResolutionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityResolutionError(f"{label} is invalid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise QualityResolutionError(f"{label} must contain one JSON object")
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise QualityResolutionError(f"{label} is not finite canonical JSON") from error
    return value


def _read_pinned_json(
    path: str | Path, expected_sha256: str, label: str
) -> tuple[dict[str, Any], bytes, str]:
    expected = _digest(expected_sha256, f"expected {label} SHA-256")
    try:
        _resolved, raw = read_regular_file(path, label)
    except (ExactConformanceError, OSError) as error:
        raise QualityResolutionError(str(error)) from error
    observed = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise QualityResolutionError(f"{label} differs from its out-of-band pin")
    return _strict_json(raw, label), raw, observed


def load_quality_selector_parity_config(
    path: str | Path, *, expected_sha256: str
) -> tuple[QualityResolutionConfig, str, str, dict[str, Any]]:
    """Authenticate the exact quality protocol config used to schedule parity cases."""

    record, _raw, observed = _read_pinned_json(
        path, expected_sha256, "quality selector-parity protocol config"
    )
    config = validate_quality_resolution_config(record)
    return config, observed, content_digest(record), record


def _selector_input_digest(inputs: Sequence[object]) -> str:
    """Hash exact tensorizer output without embedding large arrays in the receipt."""

    if len(inputs) != 4:
        raise QualityResolutionError("selector parity requires exactly four strength inputs")
    rows: list[dict[str, object]] = []
    array_fields = (
        "logical",
        "hardware",
        "logical_edges",
        "hardware_edges",
        "claims",
        "globals_",
        "index_logical_edges",
        "index_hardware_edges",
        "index_claims",
    )
    for expected_index, item in enumerate(inputs):
        arrays: dict[str, object] = {}
        for field in array_fields:
            value = np.ascontiguousarray(np.asarray(getattr(item, field)))
            arrays[field] = {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
        if getattr(item, "strength_index", None) != expected_index:
            raise QualityResolutionError("selector parity strength inputs are not ordered")
        rows.append(
            {
                "arrays": arrays,
                "normalizer_digest": getattr(item, "normalizer_digest", None),
                "scale": float(getattr(item, "scale")),
                "strength": float(getattr(item, "strength")),
                "strength_index": expected_index,
            }
        )
    return content_digest(rows)


@dataclass(frozen=True, slots=True)
class SelectorParityCase:
    """One exact terminal-selector input attached to a scheduled quality state."""

    archive_index: int
    archive_key: str
    base_lineage: str
    compiled_program_digests: tuple[str, str, str, str]
    instance_id: str
    lineage_schedule_index: int
    quality_row_id: str
    selected_embedding_digest: str
    selector_input_digest: str
    selector_inputs: tuple[object, object, object, object]
    state_fingerprint: str
    state_schedule_index: int
    support_fingerprint: str
    task_id: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.archive_index, "selector parity archive index")
        _nonnegative_int(
            self.lineage_schedule_index, "selector parity lineage schedule index"
        )
        _nonnegative_int(self.state_schedule_index, "selector parity state schedule index")
        for label, value in (
            ("archive key", self.archive_key),
            ("quality row ID", self.quality_row_id),
            ("selected embedding digest", self.selected_embedding_digest),
            ("selector input digest", self.selector_input_digest),
            ("state fingerprint", self.state_fingerprint),
            ("support fingerprint", self.support_fingerprint),
        ):
            _digest(value, f"selector parity {label}")
        for label, value in (
            ("base lineage", self.base_lineage),
            ("instance ID", self.instance_id),
            ("task ID", self.task_id),
        ):
            if type(value) is not str or not value:
                raise QualityResolutionError(f"selector parity {label} must be nonempty text")
        if len(self.compiled_program_digests) != 4 or len(self.selector_inputs) != 4:
            raise QualityResolutionError("selector parity cases require exactly four programs")
        for digest in self.compiled_program_digests:
            _digest(digest, "selector parity compiled-program digest")

    def identity_payload(self) -> dict[str, object]:
        return {
            "archive_index": self.archive_index,
            "archive_key": self.archive_key,
            "base_lineage": self.base_lineage,
            "compiled_program_digests": list(self.compiled_program_digests),
            "instance_id": self.instance_id,
            "lineage_schedule_index": self.lineage_schedule_index,
            "quality_row_id": self.quality_row_id,
            "selected_embedding_digest": self.selected_embedding_digest,
            "selector_input_digest": self.selector_input_digest,
            "state_fingerprint": self.state_fingerprint,
            "state_schedule_index": self.state_schedule_index,
            "support_fingerprint": self.support_fingerprint,
            "task_id": self.task_id,
        }

    @property
    def case_id(self) -> str:
        return content_digest(self.identity_payload())


def iter_quality_selector_parity_cases(
    prepared: Sequence[PreparedTask],
    production: ResolutionProductionPlan,
    *,
    context: Context,
    selector: StrengthSelector,
    initializer_bank: InitializerSnapshotBank,
    expected_initializer_bank_manifest_sha256: str,
    allow_test_initializer_bank: bool = False,
) -> Iterable[SelectorParityCase]:
    """Replay every scheduled state and emit every return-valid archive embedding."""

    if not isinstance(production, ResolutionProductionPlan):
        raise TypeError("selector parity requires a typed resolution production plan")
    tasks = {item.task_id: item for item in prepared}
    if len(tasks) != len(prepared):
        raise QualityResolutionError("selector parity public task census contains duplicates")
    for row in production._canonical_parts()[1]:  # noqa: SLF001 - canonical typed plan order
        item = tasks.get(row.task_id)
        if item is None:
            raise QualityResolutionError("selector parity plan refers to a missing public task")
        provenance = quality_action_provenance_fingerprint(
            item, production.source_census.prepared_manifest_sha256
        )
        replayed = replay_decision_with_action_envelope(
            item.task,
            context,
            initializer=None,
            initializer_bank=initializer_bank,
            expected_initializer_bank_manifest_sha256=(
                expected_initializer_bank_manifest_sha256
            ),
            initializer_bank_episode_index=row.initializer_bank_episode_index,
            allow_test_initializer_bank=allow_test_initializer_bank,
            selector=selector,
            prefix=row.prefix,
            seed=row.environment_seed,
            reward_reads=context.n_est_reads,
            provenance_fingerprint=provenance,
        )
        if replayed is None:
            raise QualityResolutionError("selector parity planned state is no longer replayable")
        decision, envelope = replayed
        if (
            decision.state_fingerprint != row.state_fingerprint
            or decision.support_fingerprint != row.support_fingerprint
            or envelope.record_digest != row.action_envelope_record_digest
        ):
            raise QualityResolutionError("selector parity live state differs from its plan")
        exact_state = decision.exact_state
        archive = getattr(exact_state, "archive", None)
        if not isinstance(archive, list) or not archive:
            raise QualityResolutionError("scheduled quality state has no valid archive embedding")
        for archive_index, entry in enumerate(archive):
            chains = getattr(entry, "chains", None)
            if not isinstance(chains, Mapping) or not chains:
                raise QualityResolutionError("selector parity archive entry is malformed")
            key = chain_key(chains)
            if getattr(entry, "key", None) != key or getattr(entry, "admissible", None) is not True:
                raise QualityResolutionError("selector parity archive identity is inconsistent")
            programs = compile_registry(
                chains,
                item.task.host,
                item.task.problem,
                strength_registry(
                    item.task.problem,
                    context.strength_ratios,
                    context.epsilon_strength,
                ),
                context.field_limit,
                context.coupler_limit,
            )
            inputs = build_strength_inputs(
                ctx=context,
                logical=item.task.logical,
                host=item.task.host,
                problem=item.task.problem,
                chains=chains,
                programs=programs,
                coef_scale=float(getattr(selector, "coefficient_transform_scale")),
                normalizer_digest=str(getattr(selector, "normalizer_digest")),
            )
            yield SelectorParityCase(
                archive_index=archive_index,
                archive_key=key,
                base_lineage=row.base_lineage,
                compiled_program_digests=tuple(
                    program_digest(program) for program in programs
                ),  # type: ignore[arg-type]
                instance_id=row.instance_id,
                lineage_schedule_index=row.lineage_schedule_index,
                quality_row_id=row.row_id,
                selected_embedding_digest=key,
                selector_input_digest=_selector_input_digest(inputs),
                selector_inputs=inputs,  # type: ignore[arg-type]
                state_fingerprint=row.state_fingerprint,
                state_schedule_index=row.state_schedule_index,
                support_fingerprint=row.support_fingerprint,
                task_id=row.task_id,
            )


def _outcome(case: SelectorParityCase, selected_index: object) -> dict[str, object]:
    if type(selected_index) is not int or not 0 <= selected_index < 4:
        raise QualityResolutionError("selector parity inference returned an invalid index")
    return {
        "compiled_program_digests": list(case.compiled_program_digests),
        "selected_embedding_digest": case.selected_embedding_digest,
        "selected_strength_index": selected_index,
    }


def _validate_identity_sections(
    *,
    corpus: Mapping[str, object],
    selector: Mapping[str, object],
    quality_protocol: Mapping[str, object],
    context: Mapping[str, object],
    initializer_bank: Mapping[str, object],
    runtime: Mapping[str, object],
    cpu_device: Mapping[str, object],
    accelerator_device: Mapping[str, object],
) -> None:
    sections = (
        (corpus, _CORPUS_FIELDS, "corpus"),
        (selector, _SELECTOR_FIELDS, "selector"),
        (quality_protocol, _QUALITY_PROTOCOL_FIELDS, "quality protocol"),
        (context, _CONTEXT_FIELDS, "context"),
        (runtime, _RUNTIME_FIELDS, "runtime"),
        (cpu_device, _DEVICE_FIELDS, "CPU device"),
        (accelerator_device, _DEVICE_FIELDS, "accelerator device"),
    )
    for section, expected, label in sections:
        if not isinstance(section, Mapping) or set(section) != expected:
            raise QualityResolutionError(f"selector parity {label} schema fields differ")
        try:
            canonical_json_bytes(section)
        except (TypeError, ValueError) as error:
            raise QualityResolutionError(
                f"selector parity {label} is not finite canonical JSON"
            ) from error
    for field in _CORPUS_FIELDS - {"train_lineage_count", "train_task_count"}:
        _digest(corpus[field], f"selector parity corpus {field}")
    train_lineages = _positive_int(corpus["train_lineage_count"], "train lineage count")
    train_tasks = _positive_int(corpus["train_task_count"], "train task count")
    if train_tasks < train_lineages:
        raise QualityResolutionError("selector parity train census has fewer tasks than lineages")
    for field in _SELECTOR_FIELDS:
        _digest(selector[field], f"selector parity selector {field}")
    config = quality_protocol["config"]
    if not isinstance(config, Mapping):
        raise QualityResolutionError("selector parity quality config must be an object")
    validate_quality_resolution_config(config)
    _digest(quality_protocol["config_sha256"], "selector parity config SHA-256")
    if content_digest(config) != _digest(
        quality_protocol["config_digest"], "selector parity config digest"
    ):
        raise QualityResolutionError("selector parity quality config digest mismatch")
    _digest(
        quality_protocol["production_plan_digest"],
        "selector parity production-plan digest",
    )
    _positive_int(
        quality_protocol["production_row_count"],
        "selector parity production row count",
    )
    snapshot = context["snapshot"]
    if not isinstance(snapshot, Mapping) or content_digest(snapshot) != _digest(
        context["digest"], "selector parity context digest"
    ):
        raise QualityResolutionError("selector parity context identity is inconsistent")
    try:
        validate_quality_initializer_bank_contract(initializer_bank)
    except (TypeError, ValueError) as error:
        raise QualityResolutionError(str(error)) from error
    for field in ("execution_runtime_sha256",):
        _digest(runtime[field], f"selector parity runtime {field}")
    if set(runtime["implementation"]) != _IMPLEMENTATION_FIELDS:  # type: ignore[arg-type]
        raise QualityResolutionError("selector parity implementation schema fields differ")
    for value in runtime["implementation"].values():  # type: ignore[union-attr]
        _digest(value, "selector parity implementation source digest")
    _positive_int(runtime["threads"], "selector parity runtime thread count")
    if type(runtime["deterministic_algorithms"]) is not bool:
        raise QualityResolutionError("selector parity deterministic flag must be Boolean")
    if cpu_device["requested"] != "cpu" or cpu_device["type"] != "cpu":
        raise QualityResolutionError("selector parity CPU device must resolve exactly to CPU")
    if accelerator_device["requested"] != "cuda" or accelerator_device["type"] != "cuda":
        raise QualityResolutionError("selector parity accelerator must resolve exactly to CUDA")


def build_quality_selector_device_parity(
    prepared: Sequence[PreparedTask],
    production: ResolutionProductionPlan,
    *,
    context: Context,
    selector: StrengthSelector,
    initializer_bank: InitializerSnapshotBank,
    expected_initializer_bank_manifest_sha256: str,
    cpu_device: object,
    accelerator_device: object,
    corpus: Mapping[str, object],
    selector_identity: Mapping[str, object],
    quality_protocol: Mapping[str, object],
    context_identity: Mapping[str, object],
    initializer_bank_identity: Mapping[str, object],
    runtime: Mapping[str, object],
    cpu_device_identity: Mapping[str, object],
    accelerator_device_identity: Mapping[str, object],
    allow_test_initializer_bank: bool = False,
    case_factory: Callable[[], Iterable[SelectorParityCase]] | None = None,
) -> dict[str, object]:
    """Execute two complete selector passes and return one signed canonical receipt."""

    _validate_identity_sections(
        corpus=corpus,
        selector=selector_identity,
        quality_protocol=quality_protocol,
        context=context_identity,
        initializer_bank=initializer_bank_identity,
        runtime=runtime,
        cpu_device=cpu_device_identity,
        accelerator_device=accelerator_device_identity,
    )
    if case_factory is None:
        config = validate_quality_resolution_config(quality_protocol["config"])
        production = _validated_production(
            production,
            config,
            prepared_manifest_sha256=str(corpus["manifest_sha256"]),
            prepared_train_census_record_digest=str(
                corpus["train_census_record_digest"]
            ),
        )
        production_record = production.as_dict()
        source_census = production_record["source_census"]
        if (
            production_record["record_digest"]
            != quality_protocol["production_plan_digest"]
            or production_record["row_count"] != quality_protocol["production_row_count"]
            or source_census["lineage_count"] != corpus["train_lineage_count"]
            or source_census["task_count"] != corpus["train_task_count"]
            or production_record["initializer_bank"] != initializer_bank_identity
        ):
            raise QualityResolutionError(
                "selector parity identities differ from the validated production plan"
            )
    factory = case_factory or (
        lambda: iter_quality_selector_parity_cases(
            prepared,
            production,
            context=context,
            selector=selector,
            initializer_bank=initializer_bank,
            expected_initializer_bank_manifest_sha256=(
                expected_initializer_bank_manifest_sha256
            ),
            allow_test_initializer_bank=allow_test_initializer_bank,
        )
    )
    selector.to(cpu_device).eval()  # type: ignore[attr-defined]
    cpu_rows: list[tuple[dict[str, object], dict[str, object]]] = []
    seen: set[str] = set()
    for case in factory():
        if not isinstance(case, SelectorParityCase):
            raise TypeError("selector parity case factory returned an untyped value")
        if case.case_id in seen:
            raise QualityResolutionError("selector parity case census contains duplicates")
        seen.add(case.case_id)
        selected = selector.select_embedding(case.selector_inputs)  # type: ignore[attr-defined]
        cpu_rows.append((case.identity_payload(), _outcome(case, selected)))
    if not cpu_rows:
        raise QualityResolutionError("selector parity case census is empty")

    selector.to(accelerator_device).eval()  # type: ignore[attr-defined]
    rows: list[dict[str, object]] = []
    second_count = 0
    for position, case in enumerate(factory()):
        second_count += 1
        if position >= len(cpu_rows):
            raise QualityResolutionError("selector parity accelerator replay has extra cases")
        identity, cpu = cpu_rows[position]
        if case.identity_payload() != identity:
            raise QualityResolutionError(
                "selector parity accelerator replay case differs from CPU replay"
            )
        accelerator = _outcome(
            case,
            selector.select_embedding(case.selector_inputs),  # type: ignore[attr-defined]
        )
        equal = cpu == accelerator
        rows.append(
            {
                **identity,
                "accelerator": accelerator,
                "case_id": case.case_id,
                "cpu": cpu,
                "equal": equal,
            }
        )
    if second_count != len(cpu_rows):
        raise QualityResolutionError("selector parity accelerator replay omitted cases")

    expected_order = sorted(
        rows,
        key=lambda row: (
            row["lineage_schedule_index"],
            row["state_schedule_index"],
            row["task_id"],
            row["archive_index"],
            row["case_id"],
        ),
    )
    if rows != expected_order:
        raise QualityResolutionError("selector parity cases are not in canonical plan order")
    quality_row_ids = {str(row["quality_row_id"]) for row in rows}
    production_count = int(quality_protocol["production_row_count"])
    if len(quality_row_ids) != production_count:
        raise QualityResolutionError(
            "selector parity did not cover every scheduled quality row"
        )
    mismatches = [str(row["case_id"]) for row in rows if row["equal"] is False]
    census = {
        "matching_cases": len(rows) - len(mismatches),
        "mismatching_cases": len(mismatches),
        "scheduled_quality_rows": len(quality_row_ids),
        "selector_cases": len(rows),
        "train_lineages": int(corpus["train_lineage_count"]),
        "train_tasks": int(corpus["train_task_count"]),
    }
    payload: dict[str, object] = {
        "accelerator_device": dict(accelerator_device_identity),
        "all_equal": not mismatches,
        "census": census,
        "context": dict(context_identity),
        "corpus": dict(corpus),
        "cpu_device": dict(cpu_device_identity),
        "initializer_bank": dict(initializer_bank_identity),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "quality_protocol": dict(quality_protocol),
        "rows": rows,
        "runtime": dict(runtime),
        "schema": QUALITY_SELECTOR_DEVICE_PARITY_SCHEMA,
        "schema_version": QUALITY_SELECTOR_DEVICE_PARITY_VERSION,
        "selector": dict(selector_identity),
    }
    return {**payload, "record_digest": content_digest(payload)}


def validate_quality_selector_device_parity_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Validate the complete v1 schema and all recomputable parity arithmetic."""

    if not isinstance(record, Mapping) or set(record) != _TOP_LEVEL_FIELDS:
        raise QualityResolutionError("quality selector-parity schema fields differ")
    if (
        record.get("schema") != QUALITY_SELECTOR_DEVICE_PARITY_SCHEMA
        or record.get("schema_version") != QUALITY_SELECTOR_DEVICE_PARITY_VERSION
    ):
        raise QualityResolutionError("unsupported quality selector-parity receipt")
    _validate_identity_sections(
        corpus=record["corpus"],  # type: ignore[arg-type]
        selector=record["selector"],  # type: ignore[arg-type]
        quality_protocol=record["quality_protocol"],  # type: ignore[arg-type]
        context=record["context"],  # type: ignore[arg-type]
        initializer_bank=record["initializer_bank"],  # type: ignore[arg-type]
        runtime=record["runtime"],  # type: ignore[arg-type]
        cpu_device=record["cpu_device"],  # type: ignore[arg-type]
        accelerator_device=record["accelerator_device"],  # type: ignore[arg-type]
    )
    recorded_digest = _digest(record.get("record_digest"), "selector parity record digest")
    body = {key: value for key, value in record.items() if key != "record_digest"}
    if content_digest(body) != recorded_digest:
        raise QualityResolutionError("quality selector-parity record digest mismatch")
    rows = record.get("rows")
    if not isinstance(rows, list) or not rows:
        raise QualityResolutionError("quality selector-parity rows must be nonempty")
    case_ids: list[str] = []
    quality_ids: set[str] = set()
    expected_mismatches: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise QualityResolutionError("quality selector-parity row schema fields differ")
        identity = {key: row[key] for key in _CASE_IDENTITY_FIELDS}
        for field in (
            "archive_key",
            "quality_row_id",
            "selected_embedding_digest",
            "selector_input_digest",
            "state_fingerprint",
            "support_fingerprint",
        ):
            _digest(identity[field], f"selector parity row {field}")
        _nonnegative_int(identity["archive_index"], "selector parity row archive index")
        _nonnegative_int(
            identity["lineage_schedule_index"], "selector parity row lineage index"
        )
        _nonnegative_int(identity["state_schedule_index"], "selector parity row state index")
        programs = identity["compiled_program_digests"]
        if not isinstance(programs, list) or len(programs) != 4:
            raise QualityResolutionError("selector parity row requires four program digests")
        for digest in programs:
            _digest(digest, "selector parity row program digest")
        case_id = _digest(row["case_id"], "selector parity case ID")
        if case_id != content_digest(identity):
            raise QualityResolutionError("selector parity case identity digest mismatch")
        for side in ("cpu", "accelerator"):
            outcome = row[side]
            if not isinstance(outcome, Mapping) or set(outcome) != _OUTCOME_FIELDS:
                raise QualityResolutionError("selector parity outcome schema fields differ")
            selected = outcome["selected_strength_index"]
            if type(selected) is not int or not 0 <= selected < 4:
                raise QualityResolutionError("selector parity selected index is invalid")
            if (
                outcome["selected_embedding_digest"] != identity["selected_embedding_digest"]
                or outcome["compiled_program_digests"] != programs
            ):
                raise QualityResolutionError("selector parity outcome differs from its case input")
        equal = row["cpu"] == row["accelerator"]
        if type(row["equal"]) is not bool or row["equal"] is not equal:
            raise QualityResolutionError("selector parity row equality is inconsistent")
        if case_id in case_ids:
            raise QualityResolutionError("selector parity repeats a case ID")
        case_ids.append(case_id)
        quality_ids.add(str(identity["quality_row_id"]))
        if not equal:
            expected_mismatches.append(case_id)
    expected_order = sorted(
        rows,
        key=lambda row: (
            row["lineage_schedule_index"],
            row["state_schedule_index"],
            row["task_id"],
            row["archive_index"],
            row["case_id"],
        ),
    )
    if rows != expected_order:
        raise QualityResolutionError("selector parity rows are not canonically ordered")
    mismatches = record.get("mismatches")
    if mismatches != expected_mismatches:
        raise QualityResolutionError("selector parity mismatch registry is inconsistent")
    mismatch_count = record.get("mismatch_count")
    all_equal = record.get("all_equal")
    if (
        type(mismatch_count) is not int
        or mismatch_count != len(expected_mismatches)
        or type(all_equal) is not bool
        or all_equal is not (mismatch_count == 0)
    ):
        raise QualityResolutionError("quality selector-parity result is inconsistent")
    census = record.get("census")
    if not isinstance(census, Mapping) or set(census) != _CENSUS_FIELDS:
        raise QualityResolutionError("selector parity census schema fields differ")
    expected_census = {
        "matching_cases": len(rows) - mismatch_count,
        "mismatching_cases": mismatch_count,
        "scheduled_quality_rows": len(quality_ids),
        "selector_cases": len(rows),
        "train_lineages": record["corpus"]["train_lineage_count"],  # type: ignore[index]
        "train_tasks": record["corpus"]["train_task_count"],  # type: ignore[index]
    }
    if dict(census) != expected_census:
        raise QualityResolutionError("selector parity census arithmetic is inconsistent")
    if len(quality_ids) != record["quality_protocol"]["production_row_count"]:  # type: ignore[index]
        raise QualityResolutionError("selector parity receipt omits scheduled quality rows")
    return dict(record)


def publish_quality_selector_device_parity(
    path: str | Path, record: Mapping[str, object]
) -> str:
    """Atomically publish a canonical parity receipt without replacing an artifact."""

    validated = validate_quality_selector_device_parity_record(record)
    raw = canonical_json_bytes(validated) + b"\n"
    try:
        publish_new_file(path, raw)
    except ExactConformanceError as error:
        raise QualityResolutionError(str(error)) from error
    return hashlib.sha256(raw).hexdigest()


def load_quality_selector_device_parity(
    path: str | Path, *, expected_sha256: str
) -> dict[str, object]:
    """Authenticate a canonical externally pinned selector-device parity receipt."""

    record, raw, _observed = _read_pinned_json(
        path, expected_sha256, "quality selector-parity receipt"
    )
    if raw != canonical_json_bytes(record) + b"\n":
        raise QualityResolutionError(
            "quality selector-parity receipt must be canonical JSON followed by one newline"
        )
    return validate_quality_selector_device_parity_record(record)


__all__ = [
    "QUALITY_SELECTOR_DEVICE_PARITY_SCHEMA",
    "QUALITY_SELECTOR_DEVICE_PARITY_VERSION",
    "SelectorParityCase",
    "build_quality_selector_device_parity",
    "iter_quality_selector_parity_cases",
    "load_quality_selector_device_parity",
    "load_quality_selector_parity_config",
    "publish_quality_selector_device_parity",
    "validate_quality_selector_device_parity_record",
]

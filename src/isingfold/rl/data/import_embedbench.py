"""Strict standalone CandidateBank consumer for diagnostic v1-v3 and scientific v4.

This module is intentionally implemented from the published wire contract.  It does not
import :mod:`embedbench`: data producers and the IsingFold training runtime remain separate
packages with an authenticated JSON boundary between them.

The importer verifies every content identity before materialising four physically separated
artifact families.  In particular, evaluator references and CandidateBank quality labels are
never copied into ``policy_instances.jsonl`` or ``initializers.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Any

SCHEMA_VERSION = 1
PREPARED_SCHEMA_VERSION_V2 = 2
PREPARED_SCHEMA_VERSION_V3 = 3
PREPARED_SCHEMA_VERSION_V4 = 4
SOURCE_PROVENANCE_SCHEMA_V2 = "embedbench.isingfold-task-provenance"
PREPARED_PROVENANCE_SCHEMA_V2 = "isingfold.task-provenance"
CORPUS_DESIGN_SCHEMA = "isingfold.corpus-design"
CORPUS_DESIGN_SCHEMA_VERSION = 1
REGISTERED_CORPUS_DESIGN_VERSION = "if-core-v1"
CORPUS_DESIGN_SCHEMA_VERSION_V2 = 2
REGISTERED_CORPUS_DESIGN_VERSION_V2 = "if-core-v2"
TARGET_AUTHORITY_SCHEMA = "isingfold.partitioned-target-authority"
TARGET_AUTHORITY_VERSION = 1
# Operational production floors; confirmatory power/precision can raise the test floor further.
MINIMUM_TRAIN_BASE_LINEAGES = 1024
MINIMUM_VALIDATION_BASE_LINEAGES = 512
MINIMUM_TEST_BASE_LINEAGES = 1546
MINIMUM_VALIDATION_TUNING_BASE_LINEAGES = 128
VALIDATION_TUNING_FAMILYWISE_ALPHA = 0.05
VALIDATION_TUNING_NONINFERIORITY_MARGIN = 0.02
VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL = 0.95
VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH = 0.2
VALIDATION_TUNING_POWER_TARGET_ID = "validation-valid-return-noninferiority"
VALIDATION_TUNING_PRECISION_TARGET_ID = "validation-paired-utility-precision"
PREPARED_CORPUS_DESIGN_RECEIPT_SCHEMA = "isingfold.prepared-corpus-design-receipt"
CERTIFIED_REFERENCE_STATUSES = frozenset(
    {"planted_proof", "exact_enumeration", "certified_optimal"}
)
_HEX = frozenset("0123456789abcdef")
_CURVE_PROTOCOL_FIELDS = frozenset(
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
_TRANSFORM_KINDS_V2 = frozenset(
    {"identity", "gauge", "relabel", "topology", "fault", "mechanism", "composed"}
)
_LEARNING_PARTITIONS = frozenset({"train", "val", "test"})
_DISTRIBUTION_REGIMES = frozenset({"iid", "ood"})
_FAULT_STATUSES = frozenset({"none", "faulted"})
_CALIBRATION_STATUSES = frozenset({"not_applicable", "recorded"})
_PROBLEM_ORIGINS = frozenset({"application-derived", "synthetic"})
_PARTITIONS = ("train", "val", "test")
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
_OBSERVED_STRATUM_FIELDS = (
    "application_family",
    "host_family",
    "fault_status",
    "distribution_regime",
    "calibration_status",
)
_DIFFICULTY_FIELDS = (
    "embedding_difficulty",
    "sampling_difficulty",
    "decision_difficulty",
)


def _canonical_value(value: Any) -> Any:
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


def canonical_json_bytes(value: Any) -> bytes:
    """Return the exact canonical encoding used by CandidateBank v1."""

    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    """Return the CandidateBank v1 SHA-256 content identity."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def assign_split(split_unit_id: str) -> str:
    """Recompute the canonical lineage-first 70/10/20 partition."""

    _require_nonempty_string(split_unit_id, "split_unit_id")
    bucket = (
        int(
            content_digest({"salt": "isingfold-candidate-bank-v1", "id": split_unit_id})[:8],
            16,
        )
        % 10_000
    )
    if bucket < 7_000:
        return "train"
    if bucket < 8_000:
        return "val"
    return "test"


def _strict_json(raw: bytes, name: str) -> dict[str, Any]:
    def reject_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite number {token}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    # JSON accepts an overflowing exponent (for example 1e999) as ``inf`` even when
    # parse_constant is installed.  Canonicalisation performs the recursive finite check.
    _canonical_value(value)
    return value


def _read_single_json(path: Path, name: str) -> dict[str, Any]:
    return _strict_json(path.read_bytes(), name)


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{name} line {line_number} is blank")
        rows.append(_strict_json(line, f"{name} line {line_number}"))
    return rows


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} schema fields differ: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_schema_version(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise ValueError(f"{name} requires schema_version {SCHEMA_VERSION}")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _normalise_pairs(value: Any, name: str) -> list[list[Any]]:
    if isinstance(value, str):
        return [["name", value]]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of name/value pairs")
    pairs: list[list[Any]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{name} entries must be two-item lists")
        key = _require_nonempty_string(raw[0], f"{name} name")
        pairs.append([key, _canonical_value(raw[1])])
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError(f"{name} names must be unique")
    return sorted(pairs, key=lambda pair: pair[0])


def _nodes(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    nodes = [_require_int(node, f"{name} node") for node in value]
    if len(nodes) != len(set(nodes)):
        raise ValueError(f"{name} must contain unique nodes")
    return sorted(nodes)


def _edges(value: Any, nodes: list[int], name: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} edges must be a list")
    allowed = set(nodes)
    edges: list[tuple[int, int]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{name} edges must contain two endpoints")
        first = _require_int(raw[0], f"{name} edge endpoint")
        second = _require_int(raw[1], f"{name} edge endpoint")
        if first == second or first not in allowed or second not in allowed:
            raise ValueError(f"invalid {name} edge")
        edges.append((min(first, second), max(first, second)))
    if len(edges) != len(set(edges)):
        raise ValueError(f"duplicate {name} edge")
    return [list(edge) for edge in sorted(edges)]


def _h_values(value: Any, logical_nodes: list[int]) -> list[list[Any]]:
    if not isinstance(value, list):
        raise ValueError("h must be a list")
    h: list[tuple[int, float]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("h entries must be node/value pairs")
        h.append((_require_int(raw[0], "h endpoint"), _finite(raw[1], "h coefficient")))
    if len(h) != len(logical_nodes) or {node for node, _ in h} != set(logical_nodes):
        raise ValueError("h must contain exactly one coefficient per logical node")
    return [[node, coefficient] for node, coefficient in sorted(h)]


def _j_values(value: Any, logical_nodes: list[int]) -> list[list[Any]]:
    if not isinstance(value, list):
        raise ValueError("j must be a list")
    allowed = set(logical_nodes)
    j: list[tuple[int, int, float]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("j entries must contain two endpoints and one coefficient")
        first = _require_int(raw[0], "j endpoint")
        second = _require_int(raw[1], "j endpoint")
        if first == second or first not in allowed or second not in allowed:
            raise ValueError("invalid j endpoint")
        j.append((min(first, second), max(first, second), _finite(raw[2], "j coefficient")))
    if len({(first, second) for first, second, _ in j}) != len(j):
        raise ValueError("duplicate j edge")
    return [[first, second, coefficient] for first, second, coefficient in sorted(j)]


def _validate_instance(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        raw,
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
    _require_schema_version(raw["schema_version"], "instance record")
    logical_nodes = _nodes(raw["logical_nodes"], "logical_nodes")
    host_nodes = _nodes(raw["host_nodes"], "host_nodes")
    logical_edges = _edges(raw["logical_edges"], logical_nodes, "logical")
    host_edges = _edges(raw["host_edges"], host_nodes, "host")
    h = _h_values(raw["h"], logical_nodes)
    j = _j_values(raw["j"], logical_nodes)
    if {(first, second) for first, second, _ in j} != {
        (first, second) for first, second in logical_edges
    }:
        raise ValueError("j coefficients must correspond exactly to logical_edges")
    expected_split_unit_id = "logical-" + content_digest(
        {
            "domain": "isingfold-logical-problem-v1",
            "h": h,
            "j": j,
            "logical_edges": logical_edges,
            "logical_nodes": logical_nodes,
        }
    )
    if raw["split_unit_id"] != expected_split_unit_id:
        raise ValueError("instance split_unit_id does not match its logical problem")
    expected_instance_id = "instance-" + content_digest(
        {
            "domain": "isingfold-instance-v1",
            "host_edges": host_edges,
            "host_nodes": host_nodes,
            "split_unit_id": expected_split_unit_id,
        }
    )
    if raw["instance_id"] != expected_instance_id:
        raise ValueError("instance_id does not match its problem and host graph")
    payload = {
        "family": _require_nonempty_string(raw["family"], "family"),
        "h": h,
        "host_edges": host_edges,
        "host_nodes": host_nodes,
        "instance_id": expected_instance_id,
        "j": j,
        "logical_edges": logical_edges,
        "logical_nodes": logical_nodes,
        "metadata": _normalise_pairs(raw["metadata"], "metadata"),
        "schema_version": SCHEMA_VERSION,
        "split_unit_id": expected_split_unit_id,
        "topology": _require_nonempty_string(raw["topology"], "topology"),
    }
    if raw["record_digest"] != content_digest(payload):
        raise ValueError("instance record digest mismatch")
    return {**payload, "record_digest": raw["record_digest"]}


def _validate_curve(raw: Any, expected_partition: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("evaluation curve must be an object")
    _require_exact_keys(
        raw,
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
    _require_schema_version(raw["schema_version"], "evaluation curve")
    if raw["partition"] != expected_partition:
        raise ValueError(f"evaluation curve must be the {expected_partition!r} partition")
    if not isinstance(raw["strengths"], list):
        raise ValueError("strengths must be a list")
    strengths = [_finite(value, "strength") for value in raw["strengths"]]
    if (
        not strengths
        or len(strengths) != len(set(strengths))
        or any(value < 0 for value in strengths)
    ):
        raise ValueError("strengths must be a non-empty unique non-negative grid")
    if not isinstance(raw["seeds"], list):
        raise ValueError("seeds must be a list")
    seeds = [_require_int(value, "evaluation seed") for value in raw["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty unique non-negative grid")
    objective = raw["objective"]
    if objective not in {"solve_probability_then_residual-v1", "residual_mean-v1"}:
        raise ValueError("unknown quality objective")
    protocol = _normalise_pairs(raw["evaluation_protocol"], "evaluation_protocol")
    if {name for name, _ in protocol} != _CURVE_PROTOCOL_FIELDS:
        raise ValueError("evaluation_protocol fields do not match CandidateBank v1")

    def matrix(value: Any, name: str, *, probability: bool = False) -> list[list[float]] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a matrix")
        if any(not isinstance(row, list) for row in value):
            raise ValueError(f"{name} rows must be lists")
        result = [[_finite(item, name) for item in row] for row in value]
        if len(result) != len(strengths) or any(len(row) != len(seeds) for row in result):
            raise ValueError(f"{name} shape must match the strength/seed grid")
        if probability and any(not 0.0 <= item <= 1.0 for row in result for item in row):
            raise ValueError("p_solve values must lie in [0, 1]")
        return result

    p_solve = matrix(raw["p_solve"], "p_solve", probability=True)
    residual = matrix(raw["residual_mean"], "residual_mean")
    if p_solve is None and residual is None:
        raise ValueError("an evaluation curve needs p_solve or residual labels")
    if objective == "solve_probability_then_residual-v1" and p_solve is None:
        raise ValueError("solve-probability objective requires p_solve labels")
    if objective == "residual_mean-v1" and residual is None:
        raise ValueError("residual objective requires residual labels")
    return {
        "partition": expected_partition,
        "evaluation_protocol": protocol,
        "objective": objective,
        "p_solve": p_solve,
        "reads": _require_int(raw["reads"], "reads", positive=True),
        "residual_mean": residual,
        "schema_version": 1,
        "seeds": seeds,
        "strengths": strengths,
        "sweeps": _require_int(raw["sweeps"], "sweeps", positive=True),
    }


def _normalise_chains(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidate chains must be a non-empty list")
    chains: list[list[int]] = []
    occupied: set[int] = set()
    for raw_chain in value:
        if not isinstance(raw_chain, list) or not raw_chain:
            raise ValueError("candidate chains must be non-empty")
        chain = sorted(_require_int(qubit, "candidate qubit") for qubit in raw_chain)
        if len(chain) != len(set(chain)):
            raise ValueError("a candidate chain repeats a qubit")
        if occupied.intersection(chain):
            raise ValueError("candidate chains overlap")
        occupied.update(chain)
        chains.append(chain)
    return chains


def _candidate_id(group_id: str, chains: list[list[int]]) -> str:
    return "candidate-" + content_digest(
        {"chains": chains, "domain": "isingfold-candidate-v1", "group_id": group_id}
    )


def _validate_candidate(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("candidate record must be an object")
    _require_exact_keys(
        raw,
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
    _require_schema_version(raw["schema_version"], "candidate record")
    if (
        isinstance(raw["feature_schema_version"], bool)
        or not isinstance(raw["feature_schema_version"], int)
        or raw["feature_schema_version"] != 1
    ):
        raise ValueError("candidate record requires feature schema version 1")
    group_id = _require_nonempty_string(raw["group_id"], "candidate group_id")
    chains = _normalise_chains(raw["chains"])
    candidate_id = _candidate_id(group_id, chains)
    if raw["candidate_id"] != candidate_id:
        raise ValueError("candidate_id does not match candidate content")
    total_qubits = sum(map(len, chains))
    max_chain = max(map(len, chains))
    stored_total = _require_int(raw["total_qubits"], "candidate total_qubits", positive=True)
    stored_max = _require_int(raw["max_chain"], "candidate max_chain", positive=True)
    if stored_total != total_qubits or stored_max != max_chain:
        raise ValueError("candidate resource metrics do not match chains")
    if not isinstance(raw["features"], list):
        raise ValueError("candidate features must be a list")
    features: list[list[Any]] = []
    for feature in raw["features"]:
        if not isinstance(feature, list) or len(feature) != 2:
            raise ValueError("candidate features must be name/value pairs")
        features.append(
            [
                _require_nonempty_string(feature[0], "candidate feature name"),
                _finite(feature[1], "candidate feature value"),
            ]
        )
    features.sort(key=lambda feature: feature[0])
    if len({name for name, _ in features}) != len(features):
        raise ValueError("candidate feature names must be unique")
    if features != [
        ["max_chain", float(max_chain)],
        ["total_qubits", float(total_qubits)],
    ]:
        raise ValueError("candidate features do not match chains")
    decision = _validate_curve(raw["decision"], "decision")
    audit = _validate_curve(raw["audit"], "audit")
    if decision["strengths"] != audit["strengths"]:
        raise ValueError("decision and audit strength grids must match")
    if decision["objective"] != audit["objective"]:
        raise ValueError("decision and audit objectives must match")
    if decision["evaluation_protocol"] != audit["evaluation_protocol"]:
        raise ValueError("decision and audit protocols must match")
    if set(decision["seeds"]).intersection(audit["seeds"]):
        raise ValueError("decision and audit seeds must be disjoint")
    if (decision["p_solve"] is None, decision["residual_mean"] is None) != (
        audit["p_solve"] is None,
        audit["residual_mean"] is None,
    ):
        raise ValueError("decision and audit label schemas must match")
    payload = {
        "candidate_id": candidate_id,
        "chains": chains,
        "decision": decision,
        "audit": audit,
        "feature_schema_version": 1,
        "features": features,
        "group_id": group_id,
        "max_chain": max_chain,
        "schema_version": 1,
        "total_qubits": total_qubits,
    }
    if raw["record_digest"] != content_digest(payload):
        raise ValueError("candidate record digest mismatch")
    return {**payload, "record_digest": raw["record_digest"]}


def _validate_attempt(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("repair attempt must be an object")
    _require_exact_keys(
        raw,
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
    _require_schema_version(raw["schema_version"], "repair attempt")
    status = raw["status"]
    if status not in {"valid", "duplicate", "no_change", "repair_failed"}:
        raise ValueError("unknown repair attempt status")
    candidate_id = raw["candidate_id"]
    if status in {"valid", "duplicate"}:
        _require_nonempty_string(candidate_id, "attempt candidate_id")
    elif candidate_id is not None:
        raise ValueError("unsuccessful repair attempt cannot identify a candidate")
    neighborhood = [_require_int(node, "repair neighborhood node") for node in raw["neighborhood"]]
    if not neighborhood:
        raise ValueError("repair neighborhood must be non-empty")
    reason = raw["reason"]
    if reason is not None:
        _require_nonempty_string(reason, "repair-attempt reason")
    slot = _require_int(raw["slot"], "attempt slot")
    transitions = _require_int(raw["transitions"], "attempt transitions")
    if slot < 0 or transitions < 0:
        raise ValueError("attempt slot and transitions must be non-negative")
    return {
        "attempt_id": _require_nonempty_string(raw["attempt_id"], "attempt_id"),
        "candidate_id": candidate_id,
        "neighborhood": neighborhood,
        "reason": reason,
        "repair_seed": _require_int(raw["repair_seed"], "repair seed"),
        "schema_version": 1,
        "slot": slot,
        "status": status,
        "transitions": transitions,
    }


def _structural(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "chains": candidate["chains"],
        "max_chain": candidate["max_chain"],
        "schema_version": 1,
        "total_qubits": candidate["total_qubits"],
    }


def _validate_group(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        raw,
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
    _require_schema_version(raw["schema_version"], "candidate group")
    group_id = _require_nonempty_string(raw["group_id"], "group_id")
    instance_id = _require_nonempty_string(raw["instance_id"], "group instance_id")
    instance_digest = _require_sha256(raw["instance_record_digest"], "group instance record digest")
    split_unit_id = _require_nonempty_string(raw["split_unit_id"], "group split_unit_id")
    expected_split = assign_split(split_unit_id)
    if raw["split"] != expected_split:
        raise ValueError("candidate-group split does not match split_unit_id")
    group_seed = _require_int(raw["group_seed"], "group seed")
    protocol = _normalise_pairs(raw["protocol"], "group protocol")
    protocol_map = dict(protocol)
    attempt_slots = _require_int(protocol_map.get("attempt_slots"), "attempt_slots", positive=True)
    incumbent = _validate_candidate(raw["incumbent"])
    if not isinstance(raw["candidates"], list):
        raise ValueError("candidate records must be a list")
    candidates = [_validate_candidate(value) for value in raw["candidates"]]
    if len(candidates) < 2:
        raise ValueError("candidate group requires at least two alternatives")
    if candidates != sorted(candidates, key=lambda item: item["candidate_id"]):
        raise ValueError("candidate records must be in canonical candidate_id order")
    if not isinstance(raw["attempts"], list):
        raise ValueError("repair attempts must be a list")
    attempts = [_validate_attempt(value) for value in raw["attempts"]]
    if attempts != sorted(attempts, key=lambda item: item["slot"]):
        raise ValueError("repair attempts must be in canonical slot order")
    if len(attempts) != attempt_slots or [item["slot"] for item in attempts] != list(
        range(attempt_slots)
    ):
        raise ValueError("repair attempt slots must be complete and contiguous")
    expected_group_id = "group-" + content_digest(
        {
            "domain": "isingfold-candidate-group-v1",
            "group_seed": group_seed,
            "incumbent_chains": incumbent["chains"],
            "instance_id": instance_id,
            "protocol": protocol,
        }
    )
    if group_id != expected_group_id:
        raise ValueError("group_id does not match pre-label group content")
    records = [incumbent, *candidates]
    if any(candidate["group_id"] != group_id for candidate in records):
        raise ValueError("candidate record belongs to a different group")
    candidate_ids = [candidate["candidate_id"] for candidate in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id values must be unique within a group")
    if len({canonical_json_bytes(item["chains"]) for item in records}) != len(records):
        raise ValueError("candidate embeddings must be unique within a group")
    decision_signature = (
        incumbent["decision"]["strengths"],
        incumbent["decision"]["seeds"],
        incumbent["decision"]["reads"],
        incumbent["decision"]["sweeps"],
        incumbent["decision"]["objective"],
        incumbent["decision"]["evaluation_protocol"],
    )
    audit_signature = (
        incumbent["audit"]["strengths"],
        incumbent["audit"]["seeds"],
        incumbent["audit"]["reads"],
        incumbent["audit"]["sweeps"],
        incumbent["audit"]["objective"],
        incumbent["audit"]["evaluation_protocol"],
    )
    for candidate in candidates:
        if (
            candidate["decision"]["strengths"],
            candidate["decision"]["seeds"],
            candidate["decision"]["reads"],
            candidate["decision"]["sweeps"],
            candidate["decision"]["objective"],
            candidate["decision"]["evaluation_protocol"],
        ) != decision_signature:
            raise ValueError("decision evaluation grid mismatch within candidate group")
        if (
            candidate["audit"]["strengths"],
            candidate["audit"]["seeds"],
            candidate["audit"]["reads"],
            candidate["audit"]["sweeps"],
            candidate["audit"]["objective"],
            candidate["audit"]["evaluation_protocol"],
        ) != audit_signature:
            raise ValueError("audit evaluation grid mismatch within candidate group")
    known_ids = {candidate["candidate_id"] for candidate in candidates}
    if any(
        attempt["candidate_id"] not in known_ids
        for attempt in attempts
        if attempt["status"] in {"valid", "duplicate"}
    ):
        raise ValueError("repair attempt references an unknown candidate_id")
    valid_ids = [item["candidate_id"] for item in attempts if item["status"] == "valid"]
    if len(valid_ids) != len(set(valid_ids)) or set(valid_ids) != known_ids:
        raise ValueError("each candidate requires exactly one valid repair-attempt provenance")
    for attempt in attempts:
        expected_attempt_id = "attempt-" + content_digest(
            {"domain": "isingfold-repair-attempt-v1", "group_id": group_id, "slot": attempt["slot"]}
        )
        if attempt["attempt_id"] != expected_attempt_id:
            raise ValueError("repair attempt_id does not match group and slot")
    generation_payload = {
        "attempts": attempts,
        "candidates": sorted(
            (_structural(item) for item in candidates), key=lambda x: x["candidate_id"]
        ),
        "group_id": group_id,
        "group_seed": group_seed,
        "incumbent": _structural(incumbent),
        "instance_id": instance_id,
        "instance_record_digest": instance_digest,
        "protocol": protocol,
        "rejection_reason": None,
        "schema_version": 1,
        "split": expected_split,
        "split_unit_id": split_unit_id,
    }
    generation_digest = _require_sha256(raw["generation_digest"], "generation digest")
    if generation_digest != content_digest(generation_payload):
        raise ValueError("candidate group generation digest mismatch")
    payload = {
        "attempts": attempts,
        "candidates": candidates,
        "group_id": group_id,
        "group_seed": group_seed,
        "generation_digest": generation_digest,
        "incumbent": incumbent,
        "instance_id": instance_id,
        "instance_record_digest": instance_digest,
        "protocol": protocol,
        "schema_version": 1,
        "split": expected_split,
        "split_unit_id": split_unit_id,
    }
    return {**payload, "record_digest": raw["record_digest"]}


def _validate_group_digest(group: Mapping[str, Any]) -> None:
    _require_sha256(group["record_digest"], "group record digest")
    payload = {key: value for key, value in group.items() if key != "record_digest"}
    if group["record_digest"] != content_digest(payload):
        raise ValueError("group record digest mismatch")


def _is_connected(chain: list[int], host_adjacency: Mapping[int, set[int]]) -> bool:
    allowed = set(chain)
    seen = {chain[0]}
    frontier = [chain[0]]
    while frontier:
        node = frontier.pop()
        unseen = (host_adjacency[node] & allowed) - seen
        seen.update(unseen)
        frontier.extend(unseen)
    return seen == allowed


def _validate_embedding(
    candidate: Mapping[str, Any], instance: Mapping[str, Any], qubit_cap: int
) -> dict[str, Any]:
    chains = candidate["chains"]
    if len(chains) != len(instance["logical_nodes"]):
        raise ValueError("candidate chain count must match logical node count")
    host_nodes = set(instance["host_nodes"])
    host_edges = {tuple(edge) for edge in instance["host_edges"]}
    adjacency = {node: set() for node in host_nodes}
    for first, second in host_edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    if any(not set(chain).issubset(host_nodes) for chain in chains):
        raise ValueError("candidate embedding contains a qubit outside the host graph")
    if any(not _is_connected(chain, adjacency) for chain in chains):
        raise ValueError("candidate chain is disconnected in the host graph")
    logical_index = {node: index for index, node in enumerate(instance["logical_nodes"])}
    contacts = 0
    for first, second in instance["logical_edges"]:
        left = chains[logical_index[first]]
        right = chains[logical_index[second]]
        realized = any((min(u, v), max(u, v)) in host_edges for u in left for v in right)
        if not realized:
            raise ValueError("candidate embedding does not realize every logical edge")
        contacts += 1
    total_qubits = len({qubit for chain in chains for qubit in chain})
    if total_qubits > qubit_cap:
        raise ValueError("candidate embedding exceeds the registered qubit cap")
    return {
        "connected": True,
        "disjoint": True,
        "logical_edge_contacts": contacts,
        "on_host": True,
        "realizes_logical_edges": True,
        "total_qubits": total_qubits,
        "within_qubit_cap": True,
    }


def _load_bank(
    bank_path: Path, manifest_path: Path, qubit_cap: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_single_json(manifest_path, "candidate-bank manifest")
    _require_exact_keys(
        manifest,
        {"group_count", "instance_count", "jsonl_sha256", "record_digest", "schema_version"},
        "candidate-bank manifest",
    )
    _require_schema_version(manifest["schema_version"], "candidate-bank manifest")
    for name in ("instance_count", "group_count"):
        count = _require_int(manifest[name], f"manifest {name}")
        if count < 0:
            raise ValueError("manifest counts must be non-negative")
    _require_sha256(manifest["jsonl_sha256"], "manifest jsonl_sha256")
    _require_sha256(manifest["record_digest"], "manifest record_digest")
    manifest_payload = {key: manifest[key] for key in manifest if key != "record_digest"}
    if manifest["record_digest"] != content_digest(manifest_payload):
        raise ValueError("candidate-bank manifest digest mismatch")
    bank_raw = bank_path.read_bytes()
    if hashlib.sha256(bank_raw).hexdigest() != manifest["jsonl_sha256"]:
        raise ValueError("candidate-bank checksum mismatch")
    rows = _read_jsonl(bank_path, "candidate-bank")
    instances: list[dict[str, Any]] = []
    raw_groups: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        _require_exact_keys(row, {"kind", "record"}, f"candidate-bank line {index}")
        if not isinstance(row["record"], dict):
            raise ValueError(f"candidate-bank line {index} record must be an object")
        if row["kind"] == "instance":
            instances.append(_validate_instance(row["record"]))
        elif row["kind"] == "group":
            raw_groups.append(row["record"])
        else:
            raise ValueError(f"candidate-bank line {index} has unknown kind")
    if (len(instances), len(raw_groups)) != (
        manifest["instance_count"],
        manifest["group_count"],
    ):
        raise ValueError("candidate-bank manifest count mismatch")
    by_id = {instance["instance_id"]: instance for instance in instances}
    if len(by_id) != len(instances):
        raise ValueError("candidate-bank contains duplicate instance IDs")
    # Resolve the outer reference before trusting any nested generation commitment.  This
    # produces a precise failure for a group spliced from another authenticated bank.
    for raw_group in raw_groups:
        raw_instance_id = raw_group.get("instance_id")
        instance = by_id.get(raw_instance_id)
        if instance is None:
            raise ValueError("candidate group references a missing instance")
        if raw_group.get("instance_record_digest") != instance["record_digest"]:
            raise ValueError("candidate group references the wrong instance record digest")
    groups = [_validate_group(raw_group) for raw_group in raw_groups]
    if len({group["group_id"] for group in groups}) != len(groups):
        raise ValueError("candidate-bank contains duplicate group IDs")
    for group in groups:
        instance = by_id.get(group["instance_id"])
        if instance is None:
            raise ValueError("candidate group references a missing instance")
        if group["instance_record_digest"] != instance["record_digest"]:
            raise ValueError("candidate group references the wrong instance record digest")
        if group["split_unit_id"] != instance["split_unit_id"]:
            raise ValueError("candidate group and instance split_unit_id mismatch")
        logical_nodes = set(instance["logical_nodes"])
        if any(
            not set(attempt["neighborhood"]).issubset(logical_nodes)
            for attempt in group["attempts"]
        ):
            raise ValueError("repair attempt contains an unknown logical node")
        for candidate in [group["incumbent"], *group["candidates"]]:
            _validate_embedding(candidate, instance, qubit_cap)
        _validate_group_digest(group)
    return instances, groups, manifest


def _load_targets(path: Path, instances: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected_keys = {
        "certificate_digest",
        "evaluator_protocol_digest",
        "instance_id",
        "instance_record_digest",
        "reference_energy",
        "reference_status",
        "schema",
        "schema_version",
    }
    targets: list[dict[str, Any]] = []
    for index, raw in enumerate(_read_jsonl(path, "evaluator-target sidecar"), start=1):
        _require_exact_keys(raw, expected_keys, f"evaluator target line {index}")
        _require_schema_version(raw["schema_version"], "evaluator target")
        if raw["schema"] != "embedbench.evaluator-target":
            raise ValueError("evaluator target requires embedbench.evaluator-target schema v1")
        status = raw["reference_status"]
        if status not in CERTIFIED_REFERENCE_STATUSES:
            raise ValueError("evaluator target must use a registered certified status")
        targets.append(
            {
                "certificate_digest": _require_sha256(
                    raw["certificate_digest"], "certificate digest"
                ),
                "evaluator_protocol_digest": _require_sha256(
                    raw["evaluator_protocol_digest"], "evaluator protocol digest"
                ),
                "instance_id": _require_nonempty_string(raw["instance_id"], "target instance_id"),
                "instance_record_digest": _require_sha256(
                    raw["instance_record_digest"], "target instance record digest"
                ),
                "reference_energy": _finite(raw["reference_energy"], "reference energy"),
                "reference_status": status,
                "schema": "embedbench.evaluator-target",
                "schema_version": 1,
            }
        )
    by_instance = {target["instance_id"]: target for target in targets}
    if len(by_instance) != len(targets):
        raise ValueError("duplicate evaluator target for one instance")
    expected = {instance["instance_id"]: instance for instance in instances}
    if set(by_instance) != set(expected):
        missing = sorted(set(expected) - set(by_instance))
        unknown = sorted(set(by_instance) - set(expected))
        raise ValueError(
            f"evaluator target coverage mismatch: missing={missing}, unknown={unknown}"
        )
    for instance_id, target in by_instance.items():
        if target["instance_record_digest"] != expected[instance_id]["record_digest"]:
            raise ValueError("evaluator target references the wrong instance record digest")
    return sorted(targets, key=lambda item: item["instance_id"])


def _active_host_sha256(instance: Mapping[str, Any]) -> str:
    """Recompute the EmbedBench hard/OOD identity of the active host graph."""

    return content_digest(
        {
            "edges": instance["host_edges"],
            "nodes": instance["host_nodes"],
            "schema": "embedbench.host-graph",
            "schema_version": 1,
        }
    )


def _validate_source_provenance_v2(
    raw: Mapping[str, Any],
    *,
    instance: Mapping[str, Any],
    group: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one publisher-authenticated task-provenance sidecar row."""

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
        "source_release_id",
        "source_release_manifest_sha256",
        "split_manifest_sha256",
    }
    _require_exact_keys(raw, expected_keys, "task provenance")
    if raw["schema"] != SOURCE_PROVENANCE_SCHEMA_V2:
        raise ValueError(f"task provenance requires schema {SOURCE_PROVENANCE_SCHEMA_V2!r}")
    if raw["schema_version"] != PREPARED_SCHEMA_VERSION_V2:
        raise ValueError("task provenance requires schema_version 2")
    _require_sha256(raw["record_digest"], "task-provenance record digest")
    payload = {key: raw[key] for key in raw if key != "record_digest"}
    if raw["record_digest"] != content_digest(payload):
        raise ValueError("task provenance record digest mismatch")

    group_id = _require_nonempty_string(raw["group_id"], "provenance group_id")
    instance_id = _require_nonempty_string(raw["instance_id"], "provenance instance_id")
    if group_id != group["group_id"] or instance_id != instance["instance_id"]:
        raise ValueError("task provenance references the wrong group or instance")
    if raw["group_record_digest"] != group["record_digest"]:
        raise ValueError("task provenance references the wrong group record digest")
    if raw["instance_record_digest"] != instance["record_digest"]:
        raise ValueError("task provenance references the wrong instance record digest")

    base_parent_lineage = _require_nonempty_string(
        raw["base_parent_lineage"], "base_parent_lineage"
    )
    _require_nonempty_string(raw["source_release_id"], "source_release_id")
    _require_sha256(raw["source_release_manifest_sha256"], "source_release_manifest_sha256")
    _require_sha256(raw["split_manifest_sha256"], "split_manifest_sha256")

    transform = raw["descendant_transform_identity"]
    if not isinstance(transform, dict):
        raise ValueError("descendant_transform_identity must be an object")
    _require_exact_keys(
        transform,
        {"kinds", "transform_sha256"},
        "descendant transform identity",
    )
    kinds = transform["kinds"]
    if (
        not isinstance(kinds, list)
        or not kinds
        or any(not isinstance(kind, str) or kind not in _TRANSFORM_KINDS_V2 for kind in kinds)
        or kinds != sorted(set(kinds))
    ):
        raise ValueError("descendant transform kinds must be a sorted registered set")
    if "identity" in kinds and kinds != ["identity"]:
        raise ValueError("identity cannot be combined with another descendant transform kind")
    _require_sha256(transform["transform_sha256"], "descendant transform digest")

    nominal = raw["nominal_topology_identity"]
    if not isinstance(nominal, dict):
        raise ValueError("nominal_topology_identity must be an object")
    _require_exact_keys(
        nominal,
        {"pristine_host_sha256", "size", "topology"},
        "nominal topology identity",
    )
    nominal_topology = _require_nonempty_string(nominal["topology"], "nominal topology")
    _require_int(nominal["size"], "nominal topology size", positive=True)
    _require_sha256(nominal["pristine_host_sha256"], "pristine host digest")

    active = raw["active_topology_identity"]
    if not isinstance(active, dict):
        raise ValueError("active_topology_identity must be an object")
    _require_exact_keys(
        active,
        {"host_artifact_sha256", "host_sha256", "topology"},
        "active topology identity",
    )
    active_topology = _require_nonempty_string(active["topology"], "active topology")
    active_host_sha256 = _require_sha256(active["host_sha256"], "active host digest")
    host_artifact_sha256 = _require_sha256(
        active["host_artifact_sha256"], "active host artifact digest"
    )
    if active_topology != instance["topology"] or nominal_topology != active_topology:
        raise ValueError("nominal, active, and CandidateBank topology identities disagree")
    if active_host_sha256 != _active_host_sha256(instance):
        raise ValueError("active host identity does not match the CandidateBank host graph")

    fault = raw["fault_identity"]
    if not isinstance(fault, dict):
        raise ValueError("fault_identity must be an object")
    _require_exact_keys(fault, {"fault_mask_sha256", "status"}, "fault identity")
    if fault["status"] not in _FAULT_STATUSES:
        raise ValueError("fault status must be 'none' or 'faulted'")
    if _require_sha256(fault["fault_mask_sha256"], "fault mask digest") != host_artifact_sha256:
        raise ValueError("fault identity does not match the active host artifact")
    if (fault["status"] == "faulted") != ("fault" in kinds):
        raise ValueError("fault status and descendant transform identity disagree")

    calibration = raw["calibration_identity"]
    if not isinstance(calibration, dict):
        raise ValueError("calibration_identity must be an object")
    _require_exact_keys(
        calibration,
        {"calibration_sha256", "status"},
        "calibration identity",
    )
    calibration_status = calibration["status"]
    if calibration_status not in _CALIBRATION_STATUSES:
        raise ValueError("calibration status must be explicit")
    calibration_sha256 = calibration["calibration_sha256"]
    if calibration_status == "not_applicable":
        if calibration_sha256 is not None:
            raise ValueError("not-applicable calibration cannot carry an artifact digest")
    else:
        _require_sha256(calibration_sha256, "calibration artifact digest")

    distribution = raw["distribution"]
    if not isinstance(distribution, dict):
        raise ValueError("distribution must be an object")
    _require_exact_keys(
        distribution,
        {"learning_partition", "regime", "source_partition", "stratum"},
        "distribution identity",
    )
    learning_partition = distribution["learning_partition"]
    if learning_partition not in _LEARNING_PARTITIONS:
        raise ValueError("unknown learning partition in task provenance")
    regime = distribution["regime"]
    if regime not in _DISTRIBUTION_REGIMES:
        raise ValueError("distribution regime must be explicitly 'iid' or 'ood'")
    _require_nonempty_string(distribution["source_partition"], "source partition")
    _require_nonempty_string(distribution["stratum"], "distribution stratum")
    if regime == "ood" and learning_partition != "test":
        raise ValueError("OOD provenance must be sealed test data")

    return {
        **_canonical_value(raw),
        "base_parent_lineage": base_parent_lineage,
    }


def _load_source_provenance_v2(
    path: Path,
    *,
    instances: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, "task-provenance sidecar")
    instances_by_id = {instance["instance_id"]: instance for instance in instances}
    groups_by_id = {group["group_id"]: group for group in groups}
    by_group: dict[str, dict[str, Any]] = {}
    for raw in rows:
        group_id = raw.get("group_id")
        group = groups_by_id.get(group_id)
        if group is None:
            raise ValueError("task provenance references a missing CandidateBank group")
        instance = instances_by_id[group["instance_id"]]
        row = _validate_source_provenance_v2(raw, instance=instance, group=group)
        if group_id in by_group:
            raise ValueError("duplicate task provenance for one CandidateBank group")
        by_group[group_id] = row
    if set(by_group) != set(groups_by_id):
        missing = sorted(set(groups_by_id) - set(by_group))
        unknown = sorted(set(by_group) - set(groups_by_id))
        raise ValueError(f"task-provenance coverage mismatch: missing={missing}, unknown={unknown}")

    lineage_partition: dict[str, str] = {}
    source_lineage_parent: dict[str, str] = {}
    transform_parent: dict[str, str] = {}
    release_identity: tuple[str, str] | None = None
    split_identity_by_source_partition: dict[str, str] = {}
    for group_id, row in sorted(by_group.items()):
        group = groups_by_id[group_id]
        base = row["base_parent_lineage"]
        partition = row["distribution"]["learning_partition"]
        prior_partition = lineage_partition.setdefault(base, partition)
        if prior_partition != partition:
            raise ValueError("one base parent lineage appears in multiple learning partitions")
        source_lineage = group["split_unit_id"]
        prior_parent = source_lineage_parent.setdefault(source_lineage, base)
        if prior_parent != base:
            raise ValueError("one source logical lineage maps to multiple base parents")
        transform_sha256 = row["descendant_transform_identity"]["transform_sha256"]
        prior_transform_parent = transform_parent.setdefault(transform_sha256, base)
        if prior_transform_parent != base:
            raise ValueError("one descendant transform identity maps to multiple base parents")
        current_release_identity = (
            row["source_release_id"],
            row["source_release_manifest_sha256"],
        )
        if release_identity is None:
            release_identity = current_release_identity
        elif release_identity != current_release_identity:
            raise ValueError("task-provenance sidecar mixes source release identities")
        source_partition = row["distribution"]["source_partition"]
        split_sha256 = row["split_manifest_sha256"]
        prior_split_sha256 = split_identity_by_source_partition.setdefault(
            source_partition,
            split_sha256,
        )
        if prior_split_sha256 != split_sha256:
            raise ValueError("one source partition references multiple split manifests")
    return by_group


def _registered_strings(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{name} must be a sorted nonempty set of strings")
    return list(value)


def _open_probability(value: Any, name: str, *, upper: float = 1.0) -> float:
    result = _finite(value, name)
    if not 0.0 < result < upper:
        raise ValueError(f"{name} must lie strictly between zero and {upper}")
    return result


def _validate_design_filter(
    raw: Any,
    *,
    axis_values: Mapping[str, Sequence[str]],
    name: str,
) -> dict[str, frozenset[str]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object")
    _require_exact_keys(raw, set(_STRATUM_FIELDS), name)
    result: dict[str, frozenset[str]] = {}
    for field in _STRATUM_FIELDS:
        values = _registered_strings(raw[field], f"{name} {field}")
        unknown = sorted(set(values) - set(axis_values[field]))
        if unknown:
            raise ValueError(f"{name} {field} contains unregistered values: {unknown}")
        result[field] = frozenset(values)
    return result


def _matches_design_filter(
    lineage: Mapping[str, Any],
    design_filter: Mapping[str, frozenset[str]],
) -> bool:
    return all(lineage[field] in design_filter[field] for field in _STRATUM_FIELDS)


def _validate_target_identity(
    raw: Mapping[str, Any],
    *,
    seen: set[str],
    label: str,
) -> tuple[str, str, int]:
    target_id = _require_nonempty_string(raw["target_id"], f"{label} target_id")
    if target_id in seen:
        raise ValueError(f"duplicate {label} target_id")
    seen.add(target_id)
    partition = raw["learning_partition"]
    if partition not in _PARTITIONS:
        raise ValueError(f"{label} target has an unknown learning partition")
    _require_nonempty_string(raw["endpoint"], f"{label} endpoint")
    minimum = _require_int(
        raw["minimum_base_lineages"],
        f"{label} minimum_base_lineages",
        positive=True,
    )
    return target_id, partition, minimum


def _verify_content_record(record: Mapping[str, Any], name: str) -> str:
    digest = _require_sha256(record.get("record_digest"), f"{name} record digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if digest != content_digest(payload):
        raise ValueError(f"{name} record digest mismatch")
    return digest


def _safe_relative_path(value: Any, name: str) -> str:
    text = _require_nonempty_string(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a safe relative path")
    return text


def _load_design_artifact(
    root: Path,
    descriptor: Any,
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{name} descriptor must be an object")
    _require_exact_keys(descriptor, {"path", "sha256"}, f"{name} descriptor")
    relative = _safe_relative_path(descriptor["path"], f"{name} path")
    expected_sha256 = _require_sha256(descriptor["sha256"], f"{name} SHA-256")
    resolved_root = root.resolve()
    try:
        path = resolved_root.joinpath(*PurePosixPath(relative).parts).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{name} artifact is missing") from exc
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise ValueError(f"{name} artifact escapes the corpus-design directory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} artifact checksum mismatch")
    record = _strict_json(raw, f"{name} artifact")
    if raw != canonical_json_bytes(record) + b"\n":
        raise ValueError(f"{name} artifact is not canonical JSON")
    digest = _verify_content_record(record, f"{name} artifact")
    return record, {
        "path": relative,
        "record_digest": digest,
        "sha256": expected_sha256,
    }


def _load_corpus_design(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str, Path]:
    expected = _require_sha256(
        expected_sha256,
        "expected out-of-band corpus-design digest",
    )
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ValueError("corpus-design manifest differs from its out-of-band corpus-design digest")
    manifest = _strict_json(raw, "corpus-design manifest")
    _require_exact_keys(
        manifest,
        {
            "axis_values",
            "corpus_design_version",
            "difficulty_calibration",
            "independent_unit",
            "lineage_registry",
            "minimum_partition_base_lineages",
            "partition_quotas",
            "power_targets",
            "precision_targets",
            "record_digest",
            "schema",
            "schema_version",
            "stratum_quotas",
        },
        "corpus-design manifest",
    )
    schema_version = manifest["schema_version"]
    if manifest["schema"] != CORPUS_DESIGN_SCHEMA or schema_version not in {
        CORPUS_DESIGN_SCHEMA_VERSION,
        CORPUS_DESIGN_SCHEMA_VERSION_V2,
    }:
        raise ValueError("unsupported corpus-design manifest schema")
    expected_semantic_version = (
        REGISTERED_CORPUS_DESIGN_VERSION
        if schema_version == CORPUS_DESIGN_SCHEMA_VERSION
        else REGISTERED_CORPUS_DESIGN_VERSION_V2
    )
    if manifest["corpus_design_version"] != expected_semantic_version:
        raise ValueError(
            "corpus-design semantic version differs from the registered IF-Core version"
        )
    if manifest["independent_unit"] != "immutable-base-lineage":
        raise ValueError("corpus design must use immutable base lineages as its independent unit")
    _require_sha256(manifest["record_digest"], "corpus-design record digest")
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    if manifest["record_digest"] != content_digest(payload):
        raise ValueError("corpus-design record digest mismatch")
    return manifest, observed, path.parent


def _validate_evidence_backed_strata(
    calibration: Any,
    *,
    artifact_root: Path,
    instances: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    source_provenance: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, str]]]:
    """Authenticate publisher evidence and derive all non-observed stratum labels."""

    if not isinstance(calibration, Mapping):
        raise ValueError("corpus-design difficulty_calibration must be an object")
    descriptor_names = {"authority", "budget", "evidence", "origin", "panel", "protocol"}
    _require_exact_keys(
        calibration,
        {*descriptor_names, "outcome_blind", "publisher_id"},
        "corpus-design evidence-backed stratum authority",
    )
    if calibration["outcome_blind"] is not True:
        raise ValueError("corpus-design stratum authority must be outcome-blind")
    publisher_id = _require_nonempty_string(
        calibration["publisher_id"], "stratum publisher ID"
    )
    records: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, str]] = {}
    for name in sorted(descriptor_names):
        records[name], identities[name] = _load_design_artifact(
            artifact_root,
            calibration[name],
            name=f"stratum {name}",
        )

    authority = records["authority"]
    _require_exact_keys(
        authority,
        {
            "artifacts",
            "publisher_id",
            "record_digest",
            "schema",
            "schema_version",
            "source_release_id",
            "source_release_manifest_sha256",
            "statement",
        },
        "stratum publisher authority",
    )
    if (
        authority["schema"] != "embedbench.stratum-publisher-authority"
        or authority["schema_version"] != 1
        or authority["statement"]
        != "publisher-attests-origin-and-outcome-blind-difficulty-evidence"
    ):
        raise ValueError("unsupported stratum publisher authority")
    if authority["publisher_id"] != publisher_id:
        raise ValueError("stratum authority differs from its registered publisher")
    source_release_id = _require_nonempty_string(
        authority["source_release_id"], "stratum authority source release ID"
    )
    source_release_sha = _require_sha256(
        authority["source_release_manifest_sha256"],
        "stratum authority source release manifest SHA-256",
    )
    artifact_bindings = authority["artifacts"]
    if not isinstance(artifact_bindings, Mapping):
        raise ValueError("stratum authority artifact bindings must be an object")
    expected_bindings = {
        f"{name}_{suffix}"
        for name in descriptor_names - {"authority"}
        for suffix in ("record_digest", "sha256")
    }
    _require_exact_keys(
        artifact_bindings,
        expected_bindings,
        "stratum authority artifact bindings",
    )
    for name in descriptor_names - {"authority"}:
        for suffix in ("record_digest", "sha256"):
            if artifact_bindings[f"{name}_{suffix}"] != identities[name][suffix]:
                raise ValueError(f"stratum authority does not bind the {name} artifact")

    provenance_release_identities = {
        (
            row["source_release_id"],
            row["source_release_manifest_sha256"],
        )
        for row in source_provenance.values()
    }
    if provenance_release_identities != {(source_release_id, source_release_sha)}:
        raise ValueError("stratum authority belongs to another source release")

    budget = records["budget"]
    _require_exact_keys(
        budget,
        {
            "limits",
            "measurement_budget_id",
            "record_digest",
            "schema",
            "schema_version",
        },
        "difficulty budget",
    )
    if budget["schema"] != "embedbench.difficulty-budget" or budget["schema_version"] != 1:
        raise ValueError("unsupported difficulty budget")
    _require_nonempty_string(budget["measurement_budget_id"], "measurement budget ID")
    limits = budget["limits"]
    if not isinstance(limits, Mapping):
        raise ValueError("difficulty budget limits must be an object")
    _require_exact_keys(
        limits,
        {"decision_evaluations", "embedding_attempts", "sampler_reads"},
        "difficulty budget limits",
    )
    for name, value in limits.items():
        _require_int(value, f"difficulty budget {name}", positive=True)

    protocol = records["protocol"]
    _require_exact_keys(
        protocol,
        {"outcome_blind", "record_digest", "rules", "schema", "schema_version"},
        "difficulty protocol",
    )
    if (
        protocol["schema"] != "embedbench.difficulty-protocol"
        or protocol["schema_version"] != 1
        or protocol["outcome_blind"] is not True
    ):
        raise ValueError("unsupported or outcome-aware difficulty protocol")
    raw_rules = protocol["rules"]
    if not isinstance(raw_rules, list) or len(raw_rules) != len(_DIFFICULTY_FIELDS):
        raise ValueError("difficulty protocol must define exactly three rules")
    rules: dict[str, tuple[str, str, float]] = {}
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, Mapping):
            raise ValueError("difficulty protocol rules must be objects")
        _require_exact_keys(
            raw_rule,
            {"field", "hard_if", "metric", "threshold"},
            "difficulty protocol rule",
        )
        field = _require_nonempty_string(raw_rule["field"], "difficulty field")
        if field not in _DIFFICULTY_FIELDS or field in rules:
            raise ValueError("difficulty protocol has duplicate or unknown fields")
        metric = _require_nonempty_string(raw_rule["metric"], "difficulty metric")
        comparator = raw_rule["hard_if"]
        if comparator not in {"greater-than-or-equal", "less-than-or-equal"}:
            raise ValueError("difficulty protocol has an unsupported comparator")
        rules[field] = (metric, comparator, _finite(raw_rule["threshold"], "threshold"))
    if [rule["field"] for rule in raw_rules] != sorted(_DIFFICULTY_FIELDS):
        raise ValueError("difficulty protocol rules must be canonically sorted")
    if len({value[0] for value in rules.values()}) != len(rules):
        raise ValueError("difficulty protocol metrics must be unique")

    instances_by_id = {row["instance_id"]: row for row in instances}
    source_digests: dict[str, set[str]] = {}
    source_families: dict[str, set[str]] = {}
    for group in groups:
        provenance = source_provenance[group["group_id"]]
        instance = instances_by_id[group["instance_id"]]
        base = provenance["base_parent_lineage"]
        source_digests.setdefault(base, set()).add(instance["record_digest"])
        source_families.setdefault(base, set()).add(instance["family"])
    if any(len(values) != 1 for values in source_families.values()):
        raise ValueError("one base lineage has multiple application families")
    expected_lineages = set(source_digests)

    panel = records["panel"]
    _require_exact_keys(
        panel,
        {
            "base_lineages",
            "record_digest",
            "sampling_frame",
            "schema",
            "schema_version",
            "selection_outcome_blind",
        },
        "difficulty panel",
    )
    if (
        panel["schema"] != "embedbench.difficulty-panel"
        or panel["schema_version"] != 1
        or panel["sampling_frame"] != "all-registered-lineages"
        or panel["selection_outcome_blind"] is not True
    ):
        raise ValueError("unsupported or outcome-aware difficulty panel")
    panel_lineages = panel["base_lineages"]
    if (
        not isinstance(panel_lineages, list)
        or panel_lineages != sorted(set(panel_lineages))
        or set(panel_lineages) != expected_lineages
    ):
        raise ValueError("difficulty panel does not exactly cover source base lineages")

    evidence = records["evidence"]
    _require_exact_keys(
        evidence,
        {
            "budget_record_digest",
            "measurements",
            "panel_record_digest",
            "protocol_record_digest",
            "record_digest",
            "schema",
            "schema_version",
        },
        "difficulty evidence",
    )
    if evidence["schema"] != "embedbench.difficulty-evidence" or evidence["schema_version"] != 1:
        raise ValueError("unsupported difficulty evidence")
    for name in ("budget", "panel", "protocol"):
        if evidence[f"{name}_record_digest"] != identities[name]["record_digest"]:
            raise ValueError(f"difficulty evidence does not bind its {name}")
    raw_measurements = evidence["measurements"]
    if not isinstance(raw_measurements, list):
        raise ValueError("difficulty measurements must be a list")
    derived_difficulty: dict[str, dict[str, str]] = {}
    for row in raw_measurements:
        if not isinstance(row, Mapping):
            raise ValueError("difficulty measurement rows must be objects")
        _require_exact_keys(
            row,
            {"base_lineage_key", "metrics", "record_digest"},
            "difficulty measurement row",
        )
        _verify_content_record(row, "difficulty measurement row")
        base = _require_nonempty_string(row["base_lineage_key"], "measurement base lineage")
        if base in derived_difficulty:
            raise ValueError("difficulty evidence repeats a base lineage")
        raw_metrics = row["metrics"]
        if not isinstance(raw_metrics, list):
            raise ValueError("difficulty metrics must be a canonical pair list")
        metric_values: dict[str, float] = {}
        for pair in raw_metrics:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("difficulty metric entries must be name-value pairs")
            metric = _require_nonempty_string(pair[0], "difficulty metric name")
            if metric in metric_values:
                raise ValueError("difficulty measurements repeat a metric")
            metric_values[metric] = _finite(pair[1], f"difficulty metric {metric}")
        if list(metric_values) != sorted(metric_values) or set(metric_values) != {
            value[0] for value in rules.values()
        }:
            raise ValueError("difficulty measurements differ from the frozen protocol")
        labels: dict[str, str] = {}
        for field, (metric, comparator, threshold) in rules.items():
            value = metric_values[metric]
            hard = value >= threshold if comparator == "greater-than-or-equal" else value <= threshold
            labels[field] = "hard" if hard else "easy"
        derived_difficulty[base] = labels
    if set(derived_difficulty) != expected_lineages:
        raise ValueError("difficulty evidence does not exactly cover source base lineages")

    origin = records["origin"]
    _require_exact_keys(
        origin,
        {
            "publisher_id",
            "record_digest",
            "records",
            "schema",
            "schema_version",
            "source_release_id",
            "source_release_manifest_sha256",
        },
        "origin provenance",
    )
    if origin["schema"] != "embedbench.origin-provenance" or origin["schema_version"] != 1:
        raise ValueError("unsupported origin provenance")
    if (
        origin["publisher_id"] != publisher_id
        or origin["source_release_id"] != source_release_id
        or origin["source_release_manifest_sha256"] != source_release_sha
    ):
        raise ValueError("origin provenance differs from its publisher authority")
    raw_origins = origin["records"]
    if not isinstance(raw_origins, list):
        raise ValueError("origin provenance records must be a list")
    derived_origins: dict[str, str] = {}
    for row in raw_origins:
        if not isinstance(row, Mapping):
            raise ValueError("origin provenance rows must be objects")
        _require_exact_keys(
            row,
            {
                "application_family",
                "base_lineage_key",
                "generator",
                "record_digest",
                "source_instance_record_digests",
            },
            "origin provenance row",
        )
        _verify_content_record(row, "origin provenance row")
        base = _require_nonempty_string(row["base_lineage_key"], "origin base lineage")
        if base in derived_origins or base not in expected_lineages:
            raise ValueError("origin provenance repeats or invents a base lineage")
        if row["application_family"] != next(iter(source_families[base])):
            raise ValueError("origin provenance has the wrong application family")
        digests = row["source_instance_record_digests"]
        if (
            not isinstance(digests, list)
            or digests != sorted(set(digests))
            or set(digests) != source_digests[base]
        ):
            raise ValueError("origin provenance has the wrong source-instance census")
        generator = row["generator"]
        if not isinstance(generator, Mapping):
            raise ValueError("origin generator provenance must be an object")
        _require_exact_keys(
            generator,
            {"generator_id", "implementation_sha256", "kind"},
            "origin generator provenance",
        )
        _require_nonempty_string(generator["generator_id"], "origin generator ID")
        _require_sha256(
            generator["implementation_sha256"], "origin generator implementation SHA-256"
        )
        if generator["kind"] not in {"application", "synthetic"}:
            raise ValueError("origin generator has an unsupported kind")
        derived_origins[base] = (
            "application-derived" if generator["kind"] == "application" else "synthetic"
        )
    if set(derived_origins) != expected_lineages:
        raise ValueError("origin provenance does not exactly cover source base lineages")
    return derived_difficulty, derived_origins, identities


def _validate_corpus_design(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
    artifact_root: Path,
    instances: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    source_provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    axes = manifest["axis_values"]
    if not isinstance(axes, dict):
        raise ValueError("corpus-design axis_values must be an object")
    _require_exact_keys(axes, set(_STRATUM_FIELDS), "corpus-design axis_values")
    axis_values = {
        field: _registered_strings(axes[field], f"corpus-design axis {field}")
        for field in _STRATUM_FIELDS
    }
    if any(status not in _FAULT_STATUSES for status in axis_values["fault_status"]):
        raise ValueError("corpus-design fault_status axis contains an unknown status")
    if any(origin not in _PROBLEM_ORIGINS for origin in axis_values["problem_origin"]):
        raise ValueError("corpus-design problem_origin axis contains an unknown origin")
    if any(regime not in _DISTRIBUTION_REGIMES for regime in axis_values["distribution_regime"]):
        raise ValueError("corpus-design distribution_regime axis contains an unknown regime")
    if any(
        status not in _CALIBRATION_STATUSES for status in axis_values["calibration_status"]
    ):
        raise ValueError("corpus-design calibration_status axis contains an unknown status")

    difficulty_calibration = manifest["difficulty_calibration"]
    derived_difficulty: dict[str, dict[str, str]] | None = None
    derived_origins: dict[str, str] | None = None
    stratum_artifacts: dict[str, dict[str, str]] | None = None
    if manifest["schema_version"] == CORPUS_DESIGN_SCHEMA_VERSION:
        if not isinstance(difficulty_calibration, dict):
            raise ValueError("corpus-design difficulty_calibration must be an object")
        difficulty_calibration_fields = {
            "authority_sha256",
            "budget_sha256",
            "evidence_sha256",
            "outcome_blind",
            "panel_sha256",
            "protocol_sha256",
        }
        _require_exact_keys(
            difficulty_calibration,
            difficulty_calibration_fields,
            "corpus-design difficulty calibration",
        )
        for field in difficulty_calibration_fields - {"outcome_blind"}:
            _require_sha256(
                difficulty_calibration[field],
                f"corpus-design difficulty calibration {field}",
            )
        if difficulty_calibration["outcome_blind"] is not True:
            raise ValueError(
                "corpus-design difficulty calibration must be explicitly outcome-blind"
            )
    else:
        derived_difficulty, derived_origins, stratum_artifacts = (
            _validate_evidence_backed_strata(
                difficulty_calibration,
                artifact_root=artifact_root,
                instances=instances,
                groups=groups,
                source_provenance=source_provenance,
            )
        )

    raw_registry = manifest["lineage_registry"]
    if not isinstance(raw_registry, list) or not raw_registry:
        raise ValueError("corpus-design lineage_registry must be a nonempty list")
    lineage_keys = {
        "base_lineage_key",
        "calibration_sha256",
        "learning_partition",
        *_STRATUM_FIELDS,
    }
    registry: list[dict[str, Any]] = []
    seen_structural_conditions: set[tuple[Any, ...]] = set()
    lineage_partitions: dict[str, str] = {}

    def structural_condition(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["base_lineage_key"],
            row["learning_partition"],
            *(row[field] for field in _OBSERVED_STRATUM_FIELDS),
            row["calibration_sha256"],
        )

    def registry_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["base_lineage_key"],
            row["learning_partition"],
            *(row[field] for field in _STRATUM_FIELDS),
            "" if row["calibration_sha256"] is None else row["calibration_sha256"],
        )

    for raw_lineage in raw_registry:
        if not isinstance(raw_lineage, dict):
            raise ValueError("corpus-design lineage registry rows must be objects")
        _require_exact_keys(raw_lineage, lineage_keys, "lineage registry row")
        base = _require_nonempty_string(
            raw_lineage["base_lineage_key"],
            "lineage registry base_lineage_key",
        )
        partition = raw_lineage["learning_partition"]
        if partition not in _PARTITIONS:
            raise ValueError("lineage registry row has an unknown learning partition")
        if lineage_partitions.setdefault(base, partition) != partition:
            raise ValueError("one registered base lineage appears in multiple partitions")
        lineage = dict(raw_lineage)
        for field in _STRATUM_FIELDS:
            value = _require_nonempty_string(lineage[field], f"lineage registry {field}")
            if value not in axis_values[field]:
                raise ValueError(f"lineage registry {field} contains an unregistered value")
        calibration_sha256 = lineage["calibration_sha256"]
        if lineage["calibration_status"] == "not_applicable":
            if calibration_sha256 is not None:
                raise ValueError("not-applicable lineage calibration cannot carry a digest")
        else:
            _require_sha256(calibration_sha256, "lineage calibration digest")
        if derived_difficulty is not None:
            expected_difficulty = derived_difficulty.get(base)
            if expected_difficulty is None or any(
                lineage[field] != expected_difficulty[field]
                for field in _DIFFICULTY_FIELDS
            ):
                raise ValueError(
                    "lineage registry difficulty differs from its evidence-derived difficulty"
                )
        if derived_origins is not None and lineage["problem_origin"] != derived_origins.get(base):
            raise ValueError(
                "lineage registry problem origin differs from its derived problem origin"
            )
        condition_key = structural_condition(lineage)
        if condition_key in seen_structural_conditions:
            raise ValueError("corpus-design lineage registry repeats a realized condition")
        seen_structural_conditions.add(condition_key)
        registry.append(lineage)
    if [registry_sort_key(row) for row in registry] != sorted(
        registry_sort_key(row) for row in registry
    ):
        raise ValueError("corpus-design lineage registry must be canonically sorted")
    instances_by_id = {instance["instance_id"]: instance for instance in instances}
    tasks_per_lineage: dict[str, int] = {}
    tasks_per_condition: dict[tuple[Any, ...], int] = {}
    observed_lineage_partitions: dict[str, str] = {}
    for group in groups:
        provenance = source_provenance[group["group_id"]]
        instance = instances_by_id[group["instance_id"]]
        base = provenance["base_parent_lineage"]
        current = {
            "application_family": instance["family"],
            "base_lineage_key": base,
            "fault_status": provenance["fault_identity"]["status"],
            "host_family": provenance["active_topology_identity"]["topology"],
            "learning_partition": provenance["distribution"]["learning_partition"],
            "distribution_regime": provenance["distribution"]["regime"],
            "calibration_status": provenance["calibration_identity"]["status"],
            "calibration_sha256": provenance["calibration_identity"]["calibration_sha256"],
        }
        if observed_lineage_partitions.setdefault(base, current["learning_partition"]) != current[
            "learning_partition"
        ]:
            raise ValueError("one observed base lineage appears in multiple learning partitions")
        condition_key = structural_condition(current)
        tasks_per_condition[condition_key] = tasks_per_condition.get(condition_key, 0) + 1
        tasks_per_lineage[base] = tasks_per_lineage.get(base, 0) + 1
    observed_conditions = set(tasks_per_condition)
    if observed_conditions != seen_structural_conditions:
        missing = len(seen_structural_conditions - observed_conditions)
        unknown = len(observed_conditions - seen_structural_conditions)
        raise ValueError(
            "corpus-design lineage registry coverage mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    seen_lineages = set(lineage_partitions)

    partition_quotas = manifest["partition_quotas"]
    if not isinstance(partition_quotas, dict):
        raise ValueError("corpus-design partition_quotas must be an object")
    _require_exact_keys(partition_quotas, set(_PARTITIONS), "corpus-design partition quotas")
    by_partition_lineages = {
        partition: sorted(
            {
                row["base_lineage_key"]
                for row in registry
                if row["learning_partition"] == partition
            }
        )
        for partition in _PARTITIONS
    }
    by_partition: dict[str, dict[str, int]] = {}
    for partition in _PARTITIONS:
        quota = _require_int(
            partition_quotas[partition],
            f"corpus-design {partition} partition quota",
            positive=True,
        )
        actual = len(by_partition_lineages[partition])
        if actual != quota:
            raise ValueError(
                f"corpus-design {partition} partition quota requires exactly {quota} "
                f"base lineages, observed {actual}"
            )
        by_partition[partition] = {
            "base_lineages": actual,
            "tasks": sum(tasks_per_lineage[lineage] for lineage in by_partition_lineages[partition]),
        }

    raw_partition_floors = manifest["minimum_partition_base_lineages"]
    if not isinstance(raw_partition_floors, dict):
        raise ValueError("corpus-design minimum partition base lineages must be an object")
    _require_exact_keys(
        raw_partition_floors,
        set(_PARTITIONS),
        "corpus-design minimum partition base lineages",
    )
    production_floors = {
        "test": MINIMUM_TEST_BASE_LINEAGES,
        "train": MINIMUM_TRAIN_BASE_LINEAGES,
        "val": MINIMUM_VALIDATION_BASE_LINEAGES,
    }
    partition_floors: dict[str, int] = {}
    for partition in _PARTITIONS:
        floor = _require_int(
            raw_partition_floors[partition],
            f"corpus-design {partition} minimum base lineages",
            positive=True,
        )
        if floor < production_floors[partition]:
            raise ValueError(
                f"corpus-design {partition} minimum must be at least "
                f"{production_floors[partition]} independent base lineages"
            )
        quota = partition_quotas[partition]
        if quota < floor:
            raise ValueError(
                f"corpus-design {partition} partition quota is below its registered "
                "independent-lineage floor"
            )
        partition_floors[partition] = floor

    host_families = {row["host_family"] for row in registry}
    if len(host_families) < 2:
        raise ValueError("scientific corpus design requires at least two realized host families")
    problem_origins = {row["problem_origin"] for row in registry}
    if problem_origins != {"application-derived", "synthetic"}:
        raise ValueError(
            "scientific corpus design requires application-derived and synthetic origins"
        )
    origins_by_family: dict[str, set[str]] = {}
    for row in registry:
        origins_by_family.setdefault(row["application_family"], set()).add(
            row["problem_origin"]
        )
    if any(len(origins) != 1 for origins in origins_by_family.values()):
        raise ValueError("one application family maps to multiple problem origins")
    if not any(row["fault_status"] == "faulted" for row in registry):
        raise ValueError("scientific corpus design requires a realized faulted condition")
    for field in _DIFFICULTY_FIELDS:
        if all(row[field].casefold() == "easy" for row in registry):
            raise ValueError(f"scientific corpus design is all-easy on {field}")
    ood_rows = [row for row in registry if row["distribution_regime"] == "ood"]
    if not ood_rows or any(row["learning_partition"] != "test" for row in ood_rows):
        raise ValueError("scientific corpus design requires sealed-test OOD conditions")
    for field in _STRATUM_FIELDS:
        realized_values = {row[field] for row in registry}
        if realized_values != set(axis_values[field]):
            raise ValueError(f"corpus-design axis {field} differs from its realized values")

    raw_quotas = manifest["stratum_quotas"]
    if not isinstance(raw_quotas, list) or not raw_quotas:
        raise ValueError("corpus-design stratum_quotas must be a nonempty list")
    quota_results: list[dict[str, Any]] = []
    covered_lineages: set[str] = set()
    seen_quota_ids: set[str] = set()
    has_dedicated_ood_quota = False
    for raw_quota in raw_quotas:
        if not isinstance(raw_quota, dict):
            raise ValueError("corpus-design stratum quota rows must be objects")
        _require_exact_keys(
            raw_quota,
            {"filter", "learning_partition", "minimum_base_lineages", "quota_id"},
            "corpus-design stratum quota",
        )
        quota_id = _require_nonempty_string(raw_quota["quota_id"], "stratum quota_id")
        if quota_id in seen_quota_ids:
            raise ValueError("duplicate corpus-design stratum quota_id")
        seen_quota_ids.add(quota_id)
        partition = raw_quota["learning_partition"]
        if partition not in _PARTITIONS:
            raise ValueError("stratum quota has an unknown learning partition")
        minimum = _require_int(
            raw_quota["minimum_base_lineages"],
            "stratum quota minimum_base_lineages",
            positive=True,
        )
        quota_filter = _validate_design_filter(
            raw_quota["filter"], axis_values=axis_values, name="stratum quota filter"
        )
        if partition == "test" and quota_filter["distribution_regime"] == {"ood"}:
            has_dedicated_ood_quota = True
        matched = {
            row["base_lineage_key"]
            for row in registry
            if row["learning_partition"] == partition
            and _matches_design_filter(row, quota_filter)
        }
        if len(matched) < minimum:
            raise ValueError(
                f"corpus-design stratum quota {quota_id!r} requires {minimum} base "
                f"lineages, observed {len(matched)}"
            )
        covered_lineages.update(matched)
        quota_results.append(
            {
                "actual_base_lineages": len(matched),
                "minimum_base_lineages": minimum,
                "quota_id": quota_id,
                "status": "pass",
            }
        )
    if [row["quota_id"] for row in raw_quotas] != sorted(seen_quota_ids):
        raise ValueError("corpus-design stratum quotas must be sorted by quota_id")
    if covered_lineages != seen_lineages:
        raise ValueError("corpus-design stratum quotas do not cover every registered lineage")
    if not has_dedicated_ood_quota:
        raise ValueError("corpus design requires a dedicated sealed-test OOD stratum quota")

    power_results: list[dict[str, Any]] = []
    seen_target_ids: set[str] = set()
    full_partition_filters = {
        partition: {
            field: frozenset(
                row[field]
                for row in registry
                if row["learning_partition"] == partition
            )
            for field in _STRATUM_FIELDS
        }
        for partition in _PARTITIONS
    }
    full_test_filter = full_partition_filters["test"]
    full_validation_filter = full_partition_filters["val"]
    has_primary_test_power_target = False
    has_validation_tuning_power_target = False
    validation_power_target_count = 0
    raw_power_targets = manifest["power_targets"]
    if not isinstance(raw_power_targets, list) or not raw_power_targets:
        raise ValueError("corpus-design power_targets must be a nonempty list")
    power_keys = {
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
    for raw_target in raw_power_targets:
        if not isinstance(raw_target, dict):
            raise ValueError("corpus-design power targets must be objects")
        _require_exact_keys(raw_target, power_keys, "corpus-design power target")
        target_id, partition, minimum = _validate_target_identity(
            raw_target, seen=seen_target_ids, label="power"
        )
        if raw_target["method"] != "paired-binary-normal-approximation":
            raise ValueError("unsupported corpus-design power method")
        alpha = _open_probability(raw_target["alpha"], "power alpha", upper=0.5)
        alternative = raw_target["alternative"]
        if alternative not in {"one-sided-noninferiority", "two-sided-difference"}:
            raise ValueError("unsupported corpus-design power alternative")
        target_power = _open_probability(raw_target["target_power"], "target power")
        if target_power <= 0.5:
            raise ValueError("target power must exceed one half")
        assumed_true_difference = _finite(
            raw_target["assumed_true_difference"],
            "power assumed_true_difference",
        )
        if not -1.0 < assumed_true_difference < 1.0:
            raise ValueError("power assumed_true_difference must lie between -1 and 1")
        noninferiority_margin = _finite(
            raw_target["noninferiority_margin"],
            "power noninferiority_margin",
        )
        if alternative == "one-sided-noninferiority":
            if not 0.0 < noninferiority_margin < 1.0:
                raise ValueError("noninferiority power requires a margin between zero and one")
            expected_separation = assumed_true_difference + noninferiority_margin
        else:
            if noninferiority_margin != 0.0:
                raise ValueError("two-sided difference power requires zero noninferiority margin")
            expected_separation = abs(assumed_true_difference)
        separation = _open_probability(
            raw_target["power_separation"],
            "power null-boundary separation",
        )
        if not math.isclose(separation, expected_separation, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "power separation differs from the registered alternative, true difference, "
                "and noninferiority margin"
            )
        discordance = _open_probability(
            raw_target["assumed_discordance"], "power assumed_discordance"
        )
        alpha_quantile = (
            1.0 - alpha
            if alternative == "one-sided-noninferiority"
            else 1.0 - alpha / 2.0
        )
        calculated = math.ceil(
            discordance
            * (NormalDist().inv_cdf(alpha_quantile) + NormalDist().inv_cdf(target_power)) ** 2
            / separation**2
        )
        calculated = max(1, calculated)
        if minimum < calculated:
            raise ValueError(
                f"power target {target_id!r} minimum is below its registered calculation"
            )
        target_filter = _validate_design_filter(
            raw_target["filter"], axis_values=axis_values, name="power target filter"
        )
        has_primary_test_power_target |= (
            partition == "test"
            and raw_target["endpoint"] == "valid-return-noninferiority"
            and alternative == "one-sided-noninferiority"
            and target_filter == full_test_filter
        )
        if partition == "val":
            validation_power_target_count += 1
            has_validation_tuning_power_target |= (
                target_id == VALIDATION_TUNING_POWER_TARGET_ID
                and raw_target["endpoint"] == "valid-return-noninferiority"
                and alternative == "one-sided-noninferiority"
                and alpha <= VALIDATION_TUNING_FAMILYWISE_ALPHA
                and noninferiority_margin == VALIDATION_TUNING_NONINFERIORITY_MARGIN
                and minimum >= MINIMUM_VALIDATION_TUNING_BASE_LINEAGES
                and target_filter == full_validation_filter
            )
        actual = len(
            {
                row["base_lineage_key"]
                for row in registry
                if row["learning_partition"] == partition
                and _matches_design_filter(row, target_filter)
            }
        )
        if actual < minimum:
            raise ValueError(
                f"corpus-design power target {target_id!r} requires {minimum} base lineages, "
                f"observed {actual}"
            )
        power_results.append(
            {
                "actual_base_lineages": actual,
                "calculated_minimum_base_lineages": calculated,
                "minimum_base_lineages": minimum,
                "status": "pass",
                "target_id": target_id,
            }
        )
    if [row["target_id"] for row in raw_power_targets] != sorted(seen_target_ids):
        raise ValueError("corpus-design power targets must be sorted by target_id")
    if not any(row["learning_partition"] == "test" for row in raw_power_targets):
        raise ValueError("corpus design requires a sealed-test power target")
    if not has_primary_test_power_target:
        raise ValueError(
            "corpus design requires the valid-return noninferiority power target to cover "
            "the full sealed-test population"
        )
    if validation_power_target_count != 1 or not has_validation_tuning_power_target:
        raise ValueError(
            "corpus design requires exactly one typed validation-tuning power target over "
            "the full validation population"
        )

    precision_results: list[dict[str, Any]] = []
    seen_target_ids = set()
    has_primary_test_precision_target = False
    has_validation_tuning_precision_target = False
    validation_paired_precision_target_count = 0
    raw_precision_targets = manifest["precision_targets"]
    if not isinstance(raw_precision_targets, list) or not raw_precision_targets:
        raise ValueError("corpus-design precision_targets must be a nonempty list")
    precision_keys = {
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
    for raw_target in raw_precision_targets:
        if not isinstance(raw_target, dict):
            raise ValueError("corpus-design precision targets must be objects")
        _require_exact_keys(raw_target, precision_keys, "corpus-design precision target")
        target_id, partition, minimum = _validate_target_identity(
            raw_target, seen=seen_target_ids, label="precision"
        )
        method = raw_target["method"]
        if method not in {
            "bounded-mean-worst-case-normal",
            "bounded-paired-difference-worst-case-normal",
        }:
            raise ValueError("unsupported corpus-design precision method")
        bounds = raw_target["outcome_bounds"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("precision outcome_bounds must contain lower and upper bounds")
        lower = _finite(bounds[0], "precision outcome lower bound")
        upper = _finite(bounds[1], "precision outcome upper bound")
        expected_bounds = (
            (0.0, 1.0)
            if method == "bounded-mean-worst-case-normal"
            else (-1.0, 1.0)
        )
        if (lower, upper) != expected_bounds:
            raise ValueError("precision outcome bounds differ from the registered method")
        variance_bound = _finite(raw_target["variance_bound"], "precision variance bound")
        expected_variance_bound = ((upper - lower) / 2.0) ** 2
        if not math.isclose(
            variance_bound,
            expected_variance_bound,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("precision variance bound is not the bounded worst-case value")
        confidence = _open_probability(
            raw_target["confidence_level"], "precision confidence_level"
        )
        if confidence <= 0.5:
            raise ValueError("precision confidence_level must exceed one half")
        half_width = _open_probability(raw_target["half_width"], "precision half_width")
        z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        calculated = max(1, math.ceil(variance_bound * z_value**2 / half_width**2))
        if minimum < calculated:
            raise ValueError(
                f"precision target {target_id!r} minimum is below its registered calculation"
            )
        target_filter = _validate_design_filter(
            raw_target["filter"], axis_values=axis_values, name="precision target filter"
        )
        has_primary_test_precision_target |= (
            partition == "test"
            and method == "bounded-paired-difference-worst-case-normal"
            and raw_target["endpoint"] == "learned-minus-stock-unconditional-if-q3-s0"
            and target_filter == full_test_filter
        )
        if (
            partition == "val"
            and method == "bounded-paired-difference-worst-case-normal"
            and raw_target["endpoint"] == "learned-minus-stock-unconditional-if-q3-s0"
        ):
            validation_paired_precision_target_count += 1
            has_validation_tuning_precision_target |= (
                target_id == VALIDATION_TUNING_PRECISION_TARGET_ID
                and confidence >= VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL
                and half_width <= VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH
                and minimum >= MINIMUM_VALIDATION_TUNING_BASE_LINEAGES
                and target_filter == full_validation_filter
            )
        actual = len(
            {
                row["base_lineage_key"]
                for row in registry
                if row["learning_partition"] == partition
                and _matches_design_filter(row, target_filter)
            }
        )
        if actual < minimum:
            raise ValueError(
                f"corpus-design precision target {target_id!r} requires {minimum} base "
                f"lineages, observed {actual}"
            )
        precision_results.append(
            {
                "actual_base_lineages": actual,
                "calculated_minimum_base_lineages": calculated,
                "minimum_base_lineages": minimum,
                "status": "pass",
                "target_id": target_id,
            }
        )
    if [row["target_id"] for row in raw_precision_targets] != sorted(seen_target_ids):
        raise ValueError("corpus-design precision targets must be sorted by target_id")
    if not any(row["learning_partition"] == "test" for row in raw_precision_targets):
        raise ValueError("corpus design requires a sealed-test precision target")
    if not has_primary_test_precision_target:
        raise ValueError(
            "corpus design requires the paired IF-Q3-S0 precision target to cover the full "
            "sealed-test population"
        )
    if (
        validation_paired_precision_target_count != 1
        or not has_validation_tuning_precision_target
    ):
        raise ValueError(
            "corpus design requires exactly one typed validation-tuning precision target over "
            "the full validation population"
        )
    confirmatory_test_minima = [
        row["minimum_base_lineages"]
        for row in raw_power_targets
        if row["learning_partition"] == "test"
        and row["endpoint"] == "valid-return-noninferiority"
        and row["alternative"] == "one-sided-noninferiority"
    ] + [
        row["minimum_base_lineages"]
        for row in raw_precision_targets
        if row["learning_partition"] == "test"
        and row["endpoint"] == "learned-minus-stock-unconditional-if-q3-s0"
        and row["method"] == "bounded-paired-difference-worst-case-normal"
    ]
    confirmatory_test_minimum = max(confirmatory_test_minima)
    if partition_floors["test"] < confirmatory_test_minimum:
        raise ValueError(
            "corpus-design test independent-lineage floor is below a confirmatory target"
        )
    validation_tuning_minimum = max(
        next(
            row["minimum_base_lineages"]
            for row in raw_power_targets
            if row["target_id"] == VALIDATION_TUNING_POWER_TARGET_ID
        ),
        next(
            row["minimum_base_lineages"]
            for row in raw_precision_targets
            if row["target_id"] == VALIDATION_TUNING_PRECISION_TARGET_ID
        ),
    )
    if partition_floors["val"] < validation_tuning_minimum:
        raise ValueError(
            "corpus-design validation independent-lineage floor is below its tuning targets"
        )

    stratum_members: dict[tuple[str, ...], set[str]] = {}
    stratum_tasks: dict[tuple[str, ...], int] = {}
    for row in registry:
        key = (row["learning_partition"], *(row[field] for field in _STRATUM_FIELDS))
        stratum_members.setdefault(key, set()).add(row["base_lineage_key"])
        condition_key = structural_condition(row)
        stratum_tasks[key] = stratum_tasks.get(key, 0) + tasks_per_condition[condition_key]
    by_stratum = [
        {
            **{field: key[index + 1] for index, field in enumerate(_STRATUM_FIELDS)},
            "base_lineages": len(stratum_members[key]),
            "learning_partition": key[0],
            "tasks": stratum_tasks[key],
        }
        for key in sorted(stratum_members)
    ]
    realized_census = {
        "base_lineage_set_digest": content_digest(sorted(seen_lineages)),
        "base_lineages_by_partition": by_partition_lineages,
        "by_axis": {
            field: {
                value: len(
                    {
                        row["base_lineage_key"] for row in registry if row[field] == value
                    }
                )
                for value in axis_values[field]
            }
            for field in _STRATUM_FIELDS
        },
        "by_partition": by_partition,
        "by_stratum": by_stratum,
        "independent_unit": "immutable-base-lineage",
        "lineage_registry_digest": content_digest(registry),
        "total_base_lineages": len(seen_lineages),
        "total_realized_conditions": len(registry),
        "total_tasks": sum(tasks_per_lineage.values()),
    }
    receipt = {
        "axis_values": _canonical_value(manifest["axis_values"]),
        "condition_registry": _canonical_value(registry),
        "corpus_design_version": manifest["corpus_design_version"],
        "difficulty_calibration": _canonical_value(difficulty_calibration),
        "independent_unit": "immutable-base-lineage",
        "manifest_record_digest": manifest["record_digest"],
        "manifest_sha256": manifest_sha256,
        "minimum_partition_base_lineages": _canonical_value(partition_floors),
        "partition_quotas": _canonical_value(manifest["partition_quotas"]),
        "power_targets": _canonical_value(raw_power_targets),
        "precision_targets": _canonical_value(raw_precision_targets),
        "realized_census": realized_census,
        "schema": PREPARED_CORPUS_DESIGN_RECEIPT_SCHEMA,
        "schema_version": (
            1
            if manifest["schema_version"] == CORPUS_DESIGN_SCHEMA_VERSION
            else 2
        ),
        "stratum_quotas": _canonical_value(raw_quotas),
        "validation": {
            "difficulty_diversity_status": "pass",
            "host_family_diversity_status": "pass",
            "ood_quota_status": "pass",
            "power_results": power_results,
            "power_status": "pass",
            "precision_results": precision_results,
            "precision_status": "pass",
            "quota_results": quota_results,
            "quota_status": "pass",
            "status": "pass",
        },
    }
    if stratum_artifacts is not None:
        receipt["stratum_artifacts"] = _canonical_value(stratum_artifacts)
    return receipt


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _target_set_digest(records: Sequence[Mapping[str, Any]], partition: str) -> str:
    return content_digest(
        {
            "domain": "isingfold-partition-target-set-v1",
            "partition": partition,
            "targets": [
                {
                    "instance_id": record["instance_id"],
                    "target_record_digest": record["record_digest"],
                }
                for record in sorted(records, key=lambda item: item["instance_id"])
            ],
        }
    )


def _write_and_sync(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _prepare_candidate_bank(
    bank_path: str | os.PathLike[str],
    bank_manifest_path: str | os.PathLike[str],
    evaluator_targets_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    qubit_cap: int,
    provenance_path: str | os.PathLike[str] | None,
    corpus_design_manifest_path: str | os.PathLike[str] | None = None,
    expected_corpus_design_sha256: str | None = None,
    prepared_publication_version: int | None = None,
) -> dict[str, Any]:
    """Verify CandidateBank v1 and atomically publish one prepared-corpus version.

    ``output_dir`` must not exist.  This fail-closed rule makes every prepared corpus
    immutable by convention and prevents a partial rerun from mixing provenance epochs.
    The returned object is byte-for-byte the content of ``manifest.json``.
    """

    cap = _require_int(qubit_cap, "qubit_cap", positive=True)
    bank = Path(bank_path)
    bank_manifest = Path(bank_manifest_path)
    targets_path = Path(evaluator_targets_path)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"prepared output already exists: {destination}")

    instances, groups, source_manifest = _load_bank(bank, bank_manifest, cap)
    targets = _load_targets(targets_path, instances)
    by_instance = {instance["instance_id"]: instance for instance in instances}
    source_provenance = (
        None
        if provenance_path is None
        else _load_source_provenance_v2(
            Path(provenance_path),
            instances=instances,
            groups=groups,
        )
    )
    if (corpus_design_manifest_path is None) != (expected_corpus_design_sha256 is None):
        raise ValueError(
            "corpus-design manifest and out-of-band expected digest must be supplied together"
        )
    if corpus_design_manifest_path is not None and source_provenance is None:
        raise ValueError("a corpus design requires provenance-complete CandidateBank input")
    corpus_design_receipt: dict[str, Any] | None = None
    if corpus_design_manifest_path is not None:
        if expected_corpus_design_sha256 is None:  # Narrow the optional type after the XOR check.
            raise RuntimeError("corpus-design digest presence check failed")
        design, design_sha256, design_root = _load_corpus_design(
            Path(corpus_design_manifest_path),
            expected_sha256=expected_corpus_design_sha256,
        )
        expected_design_version = (
            CORPUS_DESIGN_SCHEMA_VERSION_V2
            if prepared_publication_version == PREPARED_SCHEMA_VERSION_V4
            else CORPUS_DESIGN_SCHEMA_VERSION
        )
        if design["schema_version"] != expected_design_version:
            raise ValueError(
                f"prepared-v{prepared_publication_version or PREPARED_SCHEMA_VERSION_V3} "
                f"requires corpus-design schema v{expected_design_version}"
            )
        if source_provenance is None:  # Defensive: required by the branch guard above.
            raise RuntimeError("corpus-design validation has no source provenance")
        corpus_design_receipt = _validate_corpus_design(
            design,
            manifest_sha256=design_sha256,
            artifact_root=design_root,
            instances=instances,
            groups=groups,
            source_provenance=source_provenance,
        )
    prepared_schema_version = (
        prepared_publication_version
        if corpus_design_receipt is not None and prepared_publication_version is not None
        else PREPARED_SCHEMA_VERSION_V3
        if corpus_design_receipt is not None
        else PREPARED_SCHEMA_VERSION_V2
        if source_provenance is not None
        else 1
    )

    policy_records: list[dict[str, Any]] = []
    for instance in sorted(instances, key=lambda item: item["instance_id"]):
        policy_records.append(
            _with_digest(
                {
                    "family": instance["family"],
                    "h": instance["h"],
                    "host_edges": instance["host_edges"],
                    "host_nodes": instance["host_nodes"],
                    "instance_id": instance["instance_id"],
                    "j": instance["j"],
                    "logical_edges": instance["logical_edges"],
                    "logical_nodes": instance["logical_nodes"],
                    "schema": "isingfold.policy-instance",
                    "schema_version": 1,
                    "topology": instance["topology"],
                }
            )
        )

    initializer_records: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    split_tasks: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    lineage_to_split: dict[str, str] = {}
    instance_to_partition: dict[str, str] = {}
    for group in sorted(groups, key=lambda item: item["group_id"]):
        instance = by_instance[group["instance_id"]]
        validation = _validate_embedding(group["incumbent"], instance, cap)
        task_id = "task-" + content_digest(
            {"domain": "isingfold-profile-i-task-v1", "group_id": group["group_id"]}
        )
        initializer_records.append(
            _with_digest(
                {
                    "candidate_id": group["incumbent"]["candidate_id"],
                    "chains": group["incumbent"]["chains"],
                    "group_id": group["group_id"],
                    "group_record_digest": group["record_digest"],
                    "instance_id": instance["instance_id"],
                    "instance_record_digest": instance["record_digest"],
                    "qubit_cap": cap,
                    "schema": "isingfold.profile-i-initializer",
                    "schema_version": 1,
                    "task_id": task_id,
                    "validation": validation,
                }
            )
        )
        if source_provenance is None:
            partition = assign_split(instance["split_unit_id"])
            lineage = instance["split_unit_id"]
        else:
            provenance = source_provenance[group["group_id"]]
            partition = provenance["distribution"]["learning_partition"]
            lineage = provenance["base_parent_lineage"]
            source_payload = {
                key: value
                for key, value in provenance.items()
                if key not in {"record_digest", "schema", "schema_version"}
            }
            provenance_records.append(
                _with_digest(
                    {
                        **source_payload,
                        "schema": PREPARED_PROVENANCE_SCHEMA_V2,
                        "schema_version": PREPARED_SCHEMA_VERSION_V2,
                        "source_logical_lineage": instance["split_unit_id"],
                        "source_record_digest": provenance["record_digest"],
                        "task_id": task_id,
                    }
                )
            )
        split_tasks[partition].append(task_id)
        prior_instance_partition = instance_to_partition.setdefault(
            instance["instance_id"], partition
        )
        if prior_instance_partition != partition:
            raise ValueError("one policy instance appears in multiple learning partitions")
        prior = lineage_to_split.setdefault(lineage, partition)
        if prior != partition:
            raise ValueError("one prepared lineage appears in multiple partitions")

    target_records: list[dict[str, Any]] = []
    for target in targets:
        payload = {**target, "schema": "isingfold.evaluator-target"}
        if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4:
            payload["learning_partition"] = instance_to_partition[target["instance_id"]]
            payload["schema_version"] = 2
        target_records.append(_with_digest(payload))
    splits_payload = {
        (
            "lineage_to_split" if prepared_schema_version == 1 else "base_parent_lineage_to_split"
        ): dict(sorted(lineage_to_split.items())),
        "schema": "isingfold.lineage-splits",
        "schema_version": prepared_schema_version,
        "test": sorted(split_tasks["test"]),
        "train": sorted(split_tasks["train"]),
        "val": sorted(split_tasks["val"]),
    }
    splits_record = _with_digest(splits_payload)

    file_contents: dict[str, bytes] = {
        "policy_instances.jsonl": _jsonl_bytes(policy_records),
        "initializers.jsonl": _jsonl_bytes(initializer_records),
        "splits.json": canonical_json_bytes(splits_record) + b"\n",
    }
    partitioned_targets: dict[str, list[dict[str, Any]]] | None = None
    target_authority: dict[str, Any] | None = None
    if prepared_schema_version == PREPARED_SCHEMA_VERSION_V4:
        partitioned_targets = {
            partition: sorted(
                [
                    row
                    for row in target_records
                    if row["learning_partition"] == partition
                ],
                key=lambda item: item["instance_id"],
            )
            for partition in _PARTITIONS
        }
        partition_descriptors: dict[str, dict[str, Any]] = {}
        for partition in _PARTITIONS:
            relative = f"targets/{partition}.jsonl"
            raw = _jsonl_bytes(partitioned_targets[partition])
            file_contents[relative] = raw
            partition_descriptors[partition] = {
                "path": relative,
                "records": len(partitioned_targets[partition]),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "target_set_digest": _target_set_digest(
                    partitioned_targets[partition], partition
                ),
            }
        target_authority_payload = {
            "partitions": partition_descriptors,
            "schema": TARGET_AUTHORITY_SCHEMA,
            "schema_version": TARGET_AUTHORITY_VERSION,
            "total_targets": len(target_records),
        }
        target_authority = _with_digest(target_authority_payload)
    else:
        file_contents["evaluator_targets.jsonl"] = _jsonl_bytes(target_records)
    if prepared_schema_version >= PREPARED_SCHEMA_VERSION_V2:
        file_contents["provenance.jsonl"] = _jsonl_bytes(provenance_records)
    source_hashes = {
        "candidate_bank_jsonl": hashlib.sha256(bank.read_bytes()).hexdigest(),
        "candidate_bank_manifest": hashlib.sha256(bank_manifest.read_bytes()).hexdigest(),
        "evaluator_targets": hashlib.sha256(targets_path.read_bytes()).hexdigest(),
    }
    if provenance_path is not None:
        source_hashes["task_provenance"] = hashlib.sha256(
            Path(provenance_path).read_bytes()
        ).hexdigest()
    if corpus_design_receipt is not None:
        source_hashes["corpus_design_manifest"] = corpus_design_receipt["manifest_sha256"]
    output_receipts = {
        name: {
            "records": (
                len(policy_records)
                if name == "policy_instances.jsonl"
                else len(initializer_records)
                if name == "initializers.jsonl"
                else (
                    len(partitioned_targets[name.split("/")[1].removesuffix(".jsonl")])
                    if partitioned_targets is not None and name.startswith("targets/")
                    else len(target_records)
                )
                if name == "evaluator_targets.jsonl" or name.startswith("targets/")
                else len(provenance_records)
                if name == "provenance.jsonl"
                else 1
            ),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for name, raw in file_contents.items()
    }
    manifest_payload: dict[str, Any] = {
        "counts": {
            "evaluator_targets": len(target_records),
            "initializers": len(initializer_records),
            "policy_instances": len(policy_records),
        },
        "outputs": output_receipts,
        "policy_model_feature_allowlist": [
            "family",
            "h",
            "host_edges",
            "host_nodes",
            "j",
            "logical_edges",
            "logical_nodes",
            "topology",
        ],
        "qubit_cap": cap,
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": prepared_schema_version,
        "source_bank_manifest_record_digest": source_manifest["record_digest"],
        "source_sha256": source_hashes,
    }
    if prepared_schema_version >= PREPARED_SCHEMA_VERSION_V2:
        manifest_payload["counts"]["provenance_records"] = len(provenance_records)
        manifest_payload["provenance_record_schema"] = PREPARED_PROVENANCE_SCHEMA_V2
    if prepared_schema_version == PREPARED_SCHEMA_VERSION_V2:
        manifest_payload["corpus_scope"] = "production-provenance-v2"
    elif prepared_schema_version == PREPARED_SCHEMA_VERSION_V3:
        manifest_payload["corpus_design"] = corpus_design_receipt
        manifest_payload["corpus_scope"] = "production-designed-v3"
    elif prepared_schema_version == PREPARED_SCHEMA_VERSION_V4:
        if partitioned_targets is None or target_authority is None:
            raise RuntimeError("prepared-v4 target partitioning was not constructed")
        manifest_payload["counts"]["evaluator_targets_by_partition"] = {
            partition: len(partitioned_targets[partition]) for partition in _PARTITIONS
        }
        manifest_payload["corpus_design"] = corpus_design_receipt
        manifest_payload["corpus_scope"] = "production-designed-v4"
        manifest_payload["target_authority"] = target_authority
    prepared_manifest = _with_digest(manifest_payload)
    file_contents["manifest.json"] = canonical_json_bytes(prepared_manifest) + b"\n"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=destination.parent)
    )
    try:
        for filename, raw in file_contents.items():
            _write_and_sync(temporary / filename, raw)
        # The rename publishes the complete prepared file set as one directory-level transaction.
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return prepared_manifest


def prepare_candidate_bank(
    bank_path: str | os.PathLike[str],
    bank_manifest_path: str | os.PathLike[str],
    evaluator_targets_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    qubit_cap: int,
) -> dict[str, Any]:
    """Publish the backward-compatible five-file CandidateBank-v1 pilot corpus."""

    return _prepare_candidate_bank(
        bank_path,
        bank_manifest_path,
        evaluator_targets_path,
        output_dir,
        qubit_cap=qubit_cap,
        provenance_path=None,
    )


def prepare_candidate_bank_v2(
    bank_path: str | os.PathLike[str],
    bank_manifest_path: str | os.PathLike[str],
    evaluator_targets_path: str | os.PathLike[str],
    provenance_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    qubit_cap: int,
) -> dict[str, Any]:
    """Publish provenance-complete v2; the source sidecar is mandatory and closed-schema."""

    return _prepare_candidate_bank(
        bank_path,
        bank_manifest_path,
        evaluator_targets_path,
        output_dir,
        qubit_cap=qubit_cap,
        provenance_path=provenance_path,
    )


def prepare_candidate_bank_v3(
    bank_path: str | os.PathLike[str],
    bank_manifest_path: str | os.PathLike[str],
    evaluator_targets_path: str | os.PathLike[str],
    provenance_path: str | os.PathLike[str],
    corpus_design_manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    expected_corpus_design_sha256: str,
    qubit_cap: int,
) -> dict[str, Any]:
    """Publish scientific v3 after authenticating and realizing its frozen corpus design."""

    return _prepare_candidate_bank(
        bank_path,
        bank_manifest_path,
        evaluator_targets_path,
        output_dir,
        qubit_cap=qubit_cap,
        provenance_path=provenance_path,
        corpus_design_manifest_path=corpus_design_manifest_path,
        expected_corpus_design_sha256=expected_corpus_design_sha256,
    )


def prepare_candidate_bank_v4(
    bank_path: str | os.PathLike[str],
    bank_manifest_path: str | os.PathLike[str],
    evaluator_targets_path: str | os.PathLike[str],
    provenance_path: str | os.PathLike[str],
    corpus_design_manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    expected_corpus_design_sha256: str,
    qubit_cap: int,
) -> dict[str, Any]:
    """Publish evidence-backed v4 with physically partitioned evaluator targets."""

    return _prepare_candidate_bank(
        bank_path,
        bank_manifest_path,
        evaluator_targets_path,
        output_dir,
        qubit_cap=qubit_cap,
        provenance_path=provenance_path,
        corpus_design_manifest_path=corpus_design_manifest_path,
        expected_corpus_design_sha256=expected_corpus_design_sha256,
        prepared_publication_version=PREPARED_SCHEMA_VERSION_V4,
    )


__all__ = [
    "CERTIFIED_REFERENCE_STATUSES",
    "PREPARED_PROVENANCE_SCHEMA_V2",
    "PREPARED_SCHEMA_VERSION_V2",
    "PREPARED_SCHEMA_VERSION_V3",
    "PREPARED_SCHEMA_VERSION_V4",
    "CORPUS_DESIGN_SCHEMA",
    "CORPUS_DESIGN_SCHEMA_VERSION",
    "CORPUS_DESIGN_SCHEMA_VERSION_V2",
    "MINIMUM_TEST_BASE_LINEAGES",
    "MINIMUM_TRAIN_BASE_LINEAGES",
    "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES",
    "MINIMUM_VALIDATION_BASE_LINEAGES",
    "REGISTERED_CORPUS_DESIGN_VERSION",
    "REGISTERED_CORPUS_DESIGN_VERSION_V2",
    "TARGET_AUTHORITY_SCHEMA",
    "TARGET_AUTHORITY_VERSION",
    "VALIDATION_TUNING_FAMILYWISE_ALPHA",
    "VALIDATION_TUNING_NONINFERIORITY_MARGIN",
    "VALIDATION_TUNING_POWER_TARGET_ID",
    "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL",
    "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH",
    "VALIDATION_TUNING_PRECISION_TARGET_ID",
    "PREPARED_CORPUS_DESIGN_RECEIPT_SCHEMA",
    "SOURCE_PROVENANCE_SCHEMA_V2",
    "assign_split",
    "canonical_json_bytes",
    "content_digest",
    "prepare_candidate_bank",
    "prepare_candidate_bank_v2",
    "prepare_candidate_bank_v3",
    "prepare_candidate_bank_v4",
]

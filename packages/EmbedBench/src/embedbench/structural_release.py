"""Deterministic release packaging for certified structural decision records.

This module is deliberately a wrapper around :mod:`embedbench.structural`.  It does not
change the exact search or the ink-drop generator.  Its responsibilities are scientific
identity, truthful host/witness provenance, deterministic sharding, and fail-closed I/O.

Minorminer is not imported here.  Labels are emitted only when ``structural.py`` completed
its bounded exact search for every action; aborted samples remain attrition statistics.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from embedbench.candidate_bank import canonical_json_bytes, content_digest, stable_seed
from embedbench.exact import SearchAborted, best_completion
from embedbench.inkdrop import MODES, PlantedGenerationError, ink_drop
from embedbench.objective import compare
from embedbench.structural import (
    DecisionRecord,
    GenConfig,
    GenStats,
    decision_samples_from_witness,
    frozen_adjacency,
    frozen_requirements,
    greedy_local_choice,
    host_graph,
)

SCHEMA_VERSION = 1
PLAN_SCHEMA = "embedbench.structural-release-plan"
RECORD_SCHEMA = "embedbench.certified-structural-decision"
SHARD_MANIFEST_SCHEMA = "embedbench.structural-release-shard"
RELEASE_MANIFEST_SCHEMA = "embedbench.structural-release"
DECISIONS_FILENAME = "structural_decisions.jsonl"
MANIFEST_FILENAME = "structural_release.manifest.json"
CHECKSUM_FILENAME = "SHA256SUMS"
_EXPECTED_FILES = frozenset({DECISIONS_FILENAME, MANIFEST_FILENAME, CHECKSUM_FILENAME})
_TOPOLOGIES = frozenset({"chimera", "pegasus", "zephyr"})
_SPLITS = frozenset({"train", "val", "test"})
_DIFFICULTIES = frozenset({"easy", "hard"})
_HOST_CONDITIONS = frozenset({"pristine", "faulted_nodes"})
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class StructuralPlanRow:
    """One immutable planted-witness lineage and its exact-search budget."""

    lineage_id: str
    topology: str
    size: int
    mode: str
    split: str
    ood: bool
    difficulty: str
    replicate: int
    generator_seed: int
    host_condition: str
    removed_nodes: tuple[int, ...]
    n_vars: int
    chain_size: int
    k_in_play: int
    radius: int
    l_cap: int
    max_window_free: int
    max_nodes: int
    samples_per_instance: int
    prefix_keep_prob: float
    max_actions: int
    q_cap_slack: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> StructuralPlanRow:
        fields = set(cls.__dataclass_fields__)
        raw = _exact_mapping(value, fields, "plan row")
        removed = raw["removed_nodes"]
        if not isinstance(removed, (list, tuple)):
            raise ValueError("plan row removed_nodes must be a list")
        raw["removed_nodes"] = tuple(removed)
        row = cls(**raw)
        _validate_plan_row(row)
        return row


@dataclass(frozen=True, slots=True)
class StructuralReleasePlan:
    """Closed plan; changing any row changes ``plan_digest``."""

    root_seed: int
    rows: tuple[StructuralPlanRow, ...]
    plan_digest: str

    @classmethod
    def build(
        cls, *, root_seed: int, rows: Iterable[StructuralPlanRow]
    ) -> StructuralReleasePlan:
        _require_int(root_seed, "root_seed", minimum=0)
        materialized = tuple(rows)
        if not materialized:
            raise ValueError("release plan must contain at least one row")
        if any(not isinstance(row, StructuralPlanRow) for row in materialized):
            raise TypeError("release plan rows must be StructuralPlanRow values")
        for row in materialized:
            _validate_plan_row(row)
            semantic = _seed_semantic(
                topology=row.topology,
                size=row.size,
                mode=row.mode,
                split=row.split,
                ood=row.ood,
                difficulty=row.difficulty,
                replicate=row.replicate,
                faulted=bool(row.removed_nodes),
            )
            expected_seed = stable_seed(
                root_seed,
                "structural-release-generator",
                semantic,
            )
            if row.generator_seed != expected_seed:
                raise ValueError("generator_seed is inconsistent with root_seed and plan cell")
            if row.removed_nodes:
                ideal_nodes = sorted(int(node) for node in host_graph(row.topology, row.size).nodes)
                choice = stable_seed(
                    root_seed, "structural-release-node-fault", semantic
                )
                expected_fault = (ideal_nodes[choice % len(ideal_nodes)],)
                if row.removed_nodes != expected_fault:
                    raise ValueError(
                        "removed_nodes are inconsistent with root_seed and plan cell"
                    )
        ordered = tuple(sorted(materialized, key=lambda row: row.lineage_id))
        _require_unique((row.lineage_id for row in ordered), "lineage_id")
        payload = _plan_payload(root_seed, ordered)
        return cls(root_seed=root_seed, rows=ordered, plan_digest=content_digest(payload))

    def to_dict(self) -> dict[str, Any]:
        return {**_plan_payload(self.root_seed, self.rows), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: object) -> StructuralReleasePlan:
        raw = _exact_mapping(
            value,
            {"schema", "schema_version", "root_seed", "rows", "plan_digest"},
            "release plan",
        )
        if raw["schema"] != PLAN_SCHEMA or raw["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported structural release plan schema")
        if not isinstance(raw["rows"], list):
            raise ValueError("release plan rows must be a list")
        expected = _require_sha256(raw["plan_digest"], "plan_digest")
        encoded_payload = {
            key: item for key, item in raw.items() if key != "plan_digest"
        }
        encoded_digest = content_digest(encoded_payload)
        if encoded_digest != expected:
            raise ValueError(
                f"plan_digest mismatch: expected {expected}, encoded {encoded_digest}"
            )
        plan = cls.build(
            root_seed=raw["root_seed"],
            rows=(StructuralPlanRow.from_dict(row) for row in raw["rows"]),
        )
        if plan.plan_digest != expected:
            raise ValueError(
                f"plan_digest mismatch: expected {expected}, recomputed {plan.plan_digest}"
            )
        return plan


@dataclass(frozen=True, slots=True)
class StructuralShardReceipt:
    output_directory: Path
    decisions_path: Path
    manifest_path: Path
    checksums_path: Path
    plan_digest: str
    shard_index: int
    shard_count: int
    lineage_count: int
    record_count: int


@dataclass(frozen=True, slots=True)
class StructuralMergeReceipt:
    output_directory: Path
    decisions_path: Path
    manifest_path: Path
    checksums_path: Path
    plan_digest: str
    lineage_count: int
    record_count: int


def _exact_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return dict(value)


def _require_int(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_unique(values: Iterable[str], name: str) -> None:
    observed = tuple(values)
    if len(observed) != len(set(observed)):
        raise ValueError(f"duplicate {name}")


def _row_identity_payload(row: StructuralPlanRow) -> dict[str, Any]:
    payload = row.to_dict()
    payload.pop("lineage_id")
    return {
        "domain": "embedbench.structural-lineage",
        "schema_version": SCHEMA_VERSION,
        "row": payload,
    }


def _seed_semantic(
    *,
    topology: str,
    size: int,
    mode: str,
    split: str,
    ood: bool,
    difficulty: str,
    replicate: int,
    faulted: bool,
) -> dict[str, Any]:
    return {
        "difficulty": difficulty,
        "faulted": faulted,
        "mode": mode,
        "ood": ood,
        "replicate": replicate,
        "size": size,
        "split": split,
        "topology": topology,
    }


def _validate_plan_row(row: StructuralPlanRow) -> None:
    if row.topology not in _TOPOLOGIES:
        raise ValueError(f"unsupported topology {row.topology!r}")
    if row.mode not in MODES:
        raise ValueError(f"unsupported ink mode {row.mode!r}")
    if row.split not in _SPLITS:
        raise ValueError(f"unsupported split {row.split!r}")
    if type(row.ood) is not bool:
        raise ValueError("ood must be boolean")
    if row.ood and row.split != "test":
        raise ValueError("OOD rows are permitted only in the test split")
    if row.difficulty not in _DIFFICULTIES:
        raise ValueError(f"unsupported difficulty {row.difficulty!r}")
    if row.host_condition not in _HOST_CONDITIONS:
        raise ValueError(f"unsupported host_condition {row.host_condition!r}")
    ints = {
        "size": row.size,
        "replicate": row.replicate,
        "generator_seed": row.generator_seed,
        "n_vars": row.n_vars,
        "chain_size": row.chain_size,
        "k_in_play": row.k_in_play,
        "radius": row.radius,
        "l_cap": row.l_cap,
        "max_window_free": row.max_window_free,
        "max_nodes": row.max_nodes,
        "samples_per_instance": row.samples_per_instance,
        "max_actions": row.max_actions,
    }
    for name, value in ints.items():
        _require_int(value, name, minimum=0 if name in {"replicate", "generator_seed"} else 1)
    if row.k_in_play > row.n_vars:
        raise ValueError("k_in_play cannot exceed n_vars")
    if row.chain_size > row.l_cap:
        raise ValueError("chain_size cannot exceed l_cap")
    if type(row.prefix_keep_prob) is not float or not 0.0 <= row.prefix_keep_prob <= 1.0:
        raise ValueError("prefix_keep_prob must be a float in [0,1]")
    if row.q_cap_slack is not None:
        _require_int(row.q_cap_slack, "q_cap_slack", minimum=0)
    if not isinstance(row.removed_nodes, tuple) or any(
        type(node) is not int or node < 0 for node in row.removed_nodes
    ):
        raise ValueError("removed_nodes must be a tuple of non-negative integers")
    if tuple(sorted(set(row.removed_nodes))) != row.removed_nodes:
        raise ValueError("removed_nodes must be sorted and unique")
    if row.host_condition == "pristine" and row.removed_nodes:
        raise ValueError("pristine rows cannot declare removed_nodes")
    if row.host_condition == "faulted_nodes" and not row.removed_nodes:
        raise ValueError("faulted_nodes rows require non-empty removed_nodes")
    expected_id = "srl-" + content_digest(_row_identity_payload(row))
    if row.lineage_id != expected_id:
        raise ValueError("lineage_id does not match the immutable plan row")


def _plan_payload(root_seed: int, rows: Sequence[StructuralPlanRow]) -> dict[str, Any]:
    return {
        "root_seed": root_seed,
        "rows": [row.to_dict() for row in rows],
        "schema": PLAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
    }


def make_structural_plan_row(
    *,
    root_seed: int,
    topology: str,
    size: int,
    mode: str,
    split: str,
    ood: bool,
    difficulty: str,
    replicate: int,
    faulted: bool,
    n_vars: int,
    chain_size: int,
    k_in_play: int,
    radius: int,
    l_cap: int,
    max_window_free: int,
    max_nodes: int,
    samples_per_instance: int,
    max_actions: int,
    prefix_keep_prob: float = 0.5,
    q_cap_slack: int | None = None,
) -> StructuralPlanRow:
    """Create one row with stable seed, stable ID, and an actual fault realization.

    A faulted row stores the exact ideal-host node that will be removed.  It therefore never
    labels an ideal graph as faulted merely to fill a metadata column.
    """

    _require_int(root_seed, "root_seed", minimum=0)
    if topology not in _TOPOLOGIES:
        raise ValueError(f"unsupported topology {topology!r}")
    if type(faulted) is not bool:
        raise ValueError("faulted must be boolean")
    _require_int(size, "size")
    _require_int(n_vars, "n_vars")
    semantic = _seed_semantic(
        topology=topology,
        size=size,
        mode=mode,
        split=split,
        ood=ood,
        difficulty=difficulty,
        replicate=replicate,
        faulted=faulted,
    )
    generator_seed = stable_seed(root_seed, "structural-release-generator", semantic)
    removed_nodes: tuple[int, ...] = ()
    if faulted:
        ideal_nodes = sorted(int(node) for node in host_graph(topology, size).nodes)
        if len(ideal_nodes) <= n_vars:
            raise ValueError("host is too small to realize a node fault and all variable seeds")
        choice = stable_seed(root_seed, "structural-release-node-fault", semantic)
        removed_nodes = (ideal_nodes[choice % len(ideal_nodes)],)
    provisional = StructuralPlanRow(
        lineage_id="",
        topology=topology,
        size=size,
        mode=mode,
        split=split,
        ood=ood,
        difficulty=difficulty,
        replicate=replicate,
        generator_seed=generator_seed,
        host_condition="faulted_nodes" if faulted else "pristine",
        removed_nodes=removed_nodes,
        n_vars=n_vars,
        chain_size=chain_size,
        k_in_play=k_in_play,
        radius=radius,
        l_cap=l_cap,
        max_window_free=max_window_free,
        max_nodes=max_nodes,
        samples_per_instance=samples_per_instance,
        prefix_keep_prob=prefix_keep_prob,
        max_actions=max_actions,
        q_cap_slack=q_cap_slack,
    )
    lineage_id = "srl-" + content_digest(_row_identity_payload(provisional))
    row = StructuralPlanRow(**{**provisional.to_dict(), "lineage_id": lineage_id})
    _validate_plan_row(row)
    return row


def build_production_plan(
    *, root_seed: int, replicates_per_cell: int = 1
) -> StructuralReleasePlan:
    """Return the registered factorial structural plan.

    Test cells are genuine scale OOD cells (larger host and logical graph), not an OOD flag
    attached to an in-distribution configuration.  Both host conditions are present; the
    faulted condition removes the exact node committed in its plan row.
    """

    _require_int(replicates_per_cell, "replicates_per_cell")
    base_sizes = {
        "easy": {"chimera": 3, "pegasus": 3, "zephyr": 2},
        "hard": {"chimera": 4, "pegasus": 4, "zephyr": 3},
    }
    profile = {
        "easy": {
            "n_vars": 6,
            "chain_size": 2,
            "k_in_play": 3,
            "radius": 1,
            "l_cap": 3,
            "max_window_free": 18,
            "max_nodes": 250_000,
            "samples_per_instance": 4,
            "max_actions": 8,
            "q_cap_slack": None,
        },
        "hard": {
            "n_vars": 10,
            "chain_size": 3,
            "k_in_play": 4,
            "radius": 2,
            "l_cap": 4,
            "max_window_free": 28,
            "max_nodes": 1_000_000,
            "samples_per_instance": 6,
            "max_actions": 12,
            "q_cap_slack": 0,
        },
    }
    rows: list[StructuralPlanRow] = []
    for topology in sorted(_TOPOLOGIES):
        for mode in MODES:
            for difficulty in ("easy", "hard"):
                for split in ("train", "val", "test"):
                    for faulted in (False, True):
                        for replicate in range(replicates_per_cell):
                            values = dict(profile[difficulty])
                            size = base_sizes[difficulty][topology]
                            ood = split == "test"
                            if ood:
                                size += 1
                                values["n_vars"] += 2
                                values["max_window_free"] += 2
                            rows.append(
                                make_structural_plan_row(
                                    root_seed=root_seed,
                                    topology=topology,
                                    size=size,
                                    mode=mode,
                                    split=split,
                                    ood=ood,
                                    difficulty=difficulty,
                                    replicate=replicate,
                                    faulted=faulted,
                                    **values,
                                )
                            )
    return StructuralReleasePlan.build(root_seed=root_seed, rows=rows)


def shard_index_for(lineage_id: str, shard_count: int) -> int:
    """Assign a lineage without changing any lineage or record identity."""

    if type(lineage_id) is not str or not lineage_id:
        raise ValueError("lineage_id must be a non-empty string")
    _require_int(shard_count, "shard_count")
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "embedbench.structural-release-shard-assignment",
                "lineage_id": lineage_id,
                "schema_version": SCHEMA_VERSION,
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def _graph_payload(graph: Any) -> dict[str, Any]:
    return {
        "edges": sorted(
            [min(int(a), int(b)), max(int(a), int(b))] for a, b in graph.edges
        ),
        "nodes": sorted(int(node) for node in graph.nodes),
    }


def _host_realization(row: StructuralPlanRow) -> tuple[Any, dict[str, Any]]:
    ideal = host_graph(row.topology, row.size)
    ideal_nodes = sorted(int(node) for node in ideal.nodes)
    absent = sorted(set(row.removed_nodes) - set(ideal_nodes))
    if absent:
        raise ValueError(f"removed_nodes leave the ideal host namespace: {absent}")
    realized = ideal.copy()
    realized.remove_nodes_from(row.removed_nodes)

    ideal_payload = _graph_payload(ideal)
    realized_payload = _graph_payload(realized)
    metadata = {
        "fault_intended": bool(row.removed_nodes),
        "fault_model": "removed_nodes_v1" if row.removed_nodes else None,
        "host_condition": row.host_condition,
        "ideal_host_sha256": content_digest(ideal_payload),
        "ideal_node_count": ideal.number_of_nodes(),
        "realized_host_sha256": content_digest(realized_payload),
        "realized_node_count": realized.number_of_nodes(),
        "removed_nodes": list(row.removed_nodes),
    }
    expected_delta = len(row.removed_nodes)
    if ideal.number_of_nodes() - realized.number_of_nodes() != expected_delta:
        raise RuntimeError("declared node faults were not realized exactly")
    return realized, metadata


def _gen_config(row: StructuralPlanRow) -> GenConfig:
    return GenConfig(
        topology=row.topology,
        size=row.size,
        n_vars=row.n_vars,
        chain_size=row.chain_size,
        modes=(row.mode,),
        k_in_play=row.k_in_play,
        radius=row.radius,
        l_cap=row.l_cap,
        max_window_free=row.max_window_free,
        max_nodes=row.max_nodes,
        samples_per_instance=row.samples_per_instance,
        prefix_keep_prob=row.prefix_keep_prob,
        max_actions=row.max_actions,
        q_cap_slack=row.q_cap_slack,
    )


def _validate_decision(decision: Mapping[str, Any]) -> None:
    actions = decision.get("actions")
    values = decision.get("values")
    if not isinstance(actions, list) or not isinstance(values, list) or len(actions) != len(values):
        raise ValueError("a structural decision must preserve one outcome for every action")
    if len(actions) < 2:
        raise ValueError("a released structural decision must contain at least two actions")
    if any(
        not isinstance(action, list)
        or len(action) != 2
        or any(type(item) is not int for item in action)
        for action in actions
    ):
        raise ValueError("structural actions must be [variable,qubit] integer pairs")
    if len({tuple(action) for action in actions}) != len(actions):
        raise ValueError("a structural decision cannot repeat an action")
    for value in values:
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(item) is not int for item in value)
            or value[0] not in (0, 1)
        ):
            raise ValueError("structural outcomes must be exact integer triples")
        if (value[0] == 0 and value != [0, 0, 0]) or (
            value[0] == 1 and (value[1] >= 0 or value[2] >= 0)
        ):
            raise ValueError("structural outcome violates (feasible,-Q,-L_max) semantics")
    best_action = decision.get("best_action")
    if best_action not in actions:
        raise ValueError("best_action is not in the complete action universe")
    if values[actions.index(best_action)] != max(values):
        raise ValueError("best_action does not attain the exact maximum")
    witness_outcome = decision.get("witness_outcome")
    if (
        not isinstance(witness_outcome, list)
        or len(witness_outcome) != 3
        or any(type(item) is not int for item in witness_outcome)
        or witness_outcome[0] != 1
        or witness_outcome[1] >= 0
        or witness_outcome[2] >= 0
    ):
        raise ValueError("planted witness is not a feasible exact window completion")
    if type(decision.get("exact_nodes")) is not int or decision["exact_nodes"] <= 0:
        raise ValueError("exact_nodes must report a positive deterministic search count")
    if type(decision.get("margin")) is not int or decision["margin"] <= 0:
        raise ValueError("released decisions require a strict exact margin")
    ordered = sorted(values, reverse=True)
    if ordered[0] == ordered[1]:
        raise ValueError("released decisions require a unique best outcome")
    component = next(index for index in range(3) if ordered[0][index] != ordered[1][index])
    expected_kind = ("feasibility", "qubits", "max_chain")[component]
    expected_margin = ordered[0][component] - ordered[1][component]
    if decision.get("margin_kind") != expected_kind or decision["margin"] != expected_margin:
        raise ValueError("stored structural margin differs from the complete outcome ranking")
    greedy_action = decision.get("greedy_action")
    if greedy_action not in actions or type(decision.get("greedy_agrees")) is not bool:
        raise ValueError("stored greedy action/agreement is invalid")
    expected_agreement = values[actions.index(greedy_action)] == ordered[0]
    if decision["greedy_agrees"] != expected_agreement:
        raise ValueError("stored greedy agreement differs from exact action outcomes")


def _json_value(value: Any) -> Any:
    """Convert legacy integer-keyed record maps to an unambiguous JSON value."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is int:
                json_key = str(key)
            elif type(key) is str:
                json_key = key
            else:
                raise TypeError(f"unsupported JSON object key type {type(key).__name__}")
            if json_key in result:
                raise ValueError(f"JSON key normalization collides at {json_key!r}")
            result[json_key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in {str, int, bool, float}:
        return value
    raise TypeError(f"unsupported JSON value type {type(value).__name__}")


def _witness_payload(pe: Any) -> dict[str, Any]:
    payload = _json_value(pe.to_dict())
    payload["ink_config"] = asdict(pe.config)
    payload["validation_errors"] = pe.validate()
    if payload["validation_errors"]:
        raise RuntimeError("ink-drop returned an invalid planted witness")
    return payload


def _wrapped_record(
    *,
    row: StructuralPlanRow,
    kept_index: int,
    decision: DecisionRecord,
    witness: Mapping[str, Any],
    host_realization: Mapping[str, Any],
) -> dict[str, Any]:
    decision_payload = _json_value(asdict(decision))
    _validate_decision(decision_payload)
    instance_id = "sdi-" + content_digest(
        {
            "decision": decision_payload,
            "domain": "embedbench.structural-decision-instance",
            "kept_index": kept_index,
            "lineage_id": row.lineage_id,
            "schema_version": SCHEMA_VERSION,
        }
    )
    payload = {
        "decision": decision_payload,
        "decision_index": kept_index,
        "difficulty": row.difficulty,
        "host_realization": dict(host_realization),
        "instance_id": instance_id,
        "label_authority": {
            "algorithm": "exact_branch_and_bound_window_v1",
            "budget": {"max_nodes": row.max_nodes},
            "capacity_constraint": {"q_cap_slack": row.q_cap_slack},
            "scope": "complete_in_play_variables_inside_recorded_window",
            "status": "exact",
        },
        "lineage_id": row.lineage_id,
        "ood": row.ood,
        "plan_row": _json_value(row.to_dict()),
        "planted_witness": dict(witness),
        "schema": RECORD_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "split": row.split,
    }
    return {**payload, "record_id": "sdr-" + content_digest(payload)}


def _generate_lineage(
    row: StructuralPlanRow,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    host, realization = _host_realization(row)
    cfg = _gen_config(row)
    stats = GenStats()
    try:
        pe = ink_drop(
            host,
            row.n_vars,
            row.chain_size,
            mode=row.mode,
            seed=row.generator_seed,
        )
    except PlantedGenerationError:
        stats.generation_failures += 1
        result = {
            "failure_kind": "planted_generation_error",
            "host_realization": realization,
            "lineage_id": row.lineage_id,
            "record_count": 0,
            "stats": asdict(stats),
            "status": "generation_failed",
            "witness_sha256": None,
        }
        return [], result
    stats.instances += 1
    witness = _witness_payload(pe)
    rng = random.Random(row.generator_seed ^ 0x5DEECE66D)
    decisions = list(
        decision_samples_from_witness(pe, cfg, rng, stats, row.lineage_id)
    )
    records = [
        _wrapped_record(
            row=row,
            kept_index=index,
            decision=decision,
            witness=witness,
            host_realization=realization,
        )
        for index, decision in enumerate(decisions)
    ]
    _require_unique((record["instance_id"] for record in records), "instance_id")
    _require_unique((record["record_id"] for record in records), "record_id")
    result = {
        "failure_kind": None,
        "host_realization": realization,
        "lineage_id": row.lineage_id,
        "record_count": len(records),
        "stats": asdict(stats),
        "status": "generated",
        "witness_sha256": content_digest(witness),
    }
    return records, result


def _canonical_document(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact(raw: bytes) -> dict[str, Any]:
    return {"byte_count": len(raw), "path": DECISIONS_FILENAME, "sha256": _sha256(raw)}


def _checksum_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{_sha256(files[name])}  {name}\n" for name in sorted(files)
    ).encode("utf-8")


def _aggregate_attrition(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = tuple(GenStats.__dataclass_fields__)
    totals: dict[str, Any] = {}
    for key in keys:
        if key == "margin_kinds":
            counter: Counter[str] = Counter()
            for result in results:
                counter.update(result["stats"][key])
            totals[key] = {name: counter[name] for name in sorted(counter)}
        else:
            totals[key] = sum(int(result["stats"][key]) for result in results)
    return totals


def _label_policy() -> dict[str, Any]:
    return {
        "authority": "exact_branch_and_bound_window_v1",
        "minorminer_is_label_authority": False,
        "unknown_or_aborted_records_emitted": 0,
        "wall_clock_timeout_in_identity": False,
    }


def _manifest_payload(
    *,
    schema: str,
    plan: StructuralReleasePlan,
    records_raw: bytes,
    lineage_results: Sequence[Mapping[str, Any]],
    planned_lineage_ids: Sequence[str],
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> dict[str, Any]:
    status = Counter(str(result["status"]) for result in lineage_results)
    payload: dict[str, Any] = {
        "artifact": _artifact(records_raw),
        "attrition": _aggregate_attrition(lineage_results),
        "counts": {
            "lineages": len(planned_lineage_ids),
            "records": sum(int(result["record_count"]) for result in lineage_results),
            "status": {key: status[key] for key in sorted(status)},
        },
        "label_policy": _label_policy(),
        "lineage_results": list(lineage_results),
        "plan": plan.to_dict(),
        "plan_digest": plan.plan_digest,
        "planned_lineage_ids": list(planned_lineage_ids),
        "schema": schema,
        "schema_version": SCHEMA_VERSION,
    }
    if schema == SHARD_MANIFEST_SCHEMA:
        payload["shard_count"] = shard_count
        payload["shard_index"] = shard_index
    return payload


def _record_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(destination: Path, authenticated: Mapping[str, bytes]) -> None:
    if _path_exists(destination):
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    reserved_identity: tuple[int, int] | None = None
    published_names: list[str] = []
    try:
        files = {**authenticated, CHECKSUM_FILENAME: _checksum_bytes(authenticated)}
        for name, raw in sorted(files.items()):
            _write_new(temporary / name, raw)
        _fsync_directory(temporary)

        # mkdir is the portable no-replace publication reservation. Files are
        # hard-linked from the same-filesystem staging directory, so an observed
        # partial directory is rejectable while a concurrent/retried publisher
        # can never replace its owner.
        os.mkdir(destination, mode=0o700)
        reserved = destination.stat(follow_symlinks=False)
        reserved_identity = (reserved.st_dev, reserved.st_ino)
        for name in sorted(files):
            os.link(temporary / name, destination / name)
            published_names.append(name)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        shutil.rmtree(temporary)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        if reserved_identity is not None:
            try:
                current = destination.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == reserved_identity:
                    for name in reversed(published_names):
                        (destination / name).unlink(missing_ok=True)
                    destination.rmdir()
                    _fsync_directory(destination.parent)
            except OSError:
                # Preserve an incomplete reservation rather than risk removing
                # a path whose identity changed concurrently.
                pass
        raise


def generate_structural_shard(
    plan: StructuralReleasePlan,
    shard_index: int,
    shard_count: int,
    output_dir: str | os.PathLike[str],
) -> StructuralShardReceipt:
    """Generate and atomically publish exactly one deterministic lineage shard."""

    if not isinstance(plan, StructuralReleasePlan):
        raise TypeError("plan must be a StructuralReleasePlan")
    # Round-trip closes any hand-constructed dataclass value before computation.
    plan = StructuralReleasePlan.from_dict(plan.to_dict())
    _require_int(shard_count, "shard_count")
    _require_int(shard_index, "shard_index", minimum=0)
    if shard_index >= shard_count:
        raise ValueError("shard_index must be less than shard_count")
    selected = tuple(
        row
        for row in plan.rows
        if shard_index_for(row.lineage_id, shard_count) == shard_index
    )
    records: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for row in selected:
        generated, result = _generate_lineage(row)
        records.extend(generated)
        results.append(result)
    records.sort(key=lambda record: (record["lineage_id"], record["decision_index"]))
    results.sort(key=lambda result: result["lineage_id"])
    raw = _jsonl_bytes(records)
    lineage_ids = tuple(row.lineage_id for row in selected)
    manifest_payload = _manifest_payload(
        schema=SHARD_MANIFEST_SCHEMA,
        plan=plan,
        records_raw=raw,
        lineage_results=results,
        planned_lineage_ids=lineage_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    manifest_raw = _canonical_document(_record_document(manifest_payload))
    destination = Path(output_dir)
    _publish_directory(
        destination,
        {DECISIONS_FILENAME: raw, MANIFEST_FILENAME: manifest_raw},
    )
    root = destination.resolve(strict=True)
    return StructuralShardReceipt(
        output_directory=root,
        decisions_path=root / DECISIONS_FILENAME,
        manifest_path=root / MANIFEST_FILENAME,
        checksums_path=root / CHECKSUM_FILENAME,
        plan_digest=plan.plan_digest,
        shard_index=shard_index,
        shard_count=shard_count,
        lineage_count=len(selected),
        record_count=len(records),
    )


def _read_regular(path: Path, name: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ValueError(f"{name} is missing") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file, not a symlink or directory")
    return path.read_bytes()


def _strict_json(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite value {token}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    if raw != _canonical_document(value):
        raise ValueError(f"{name} is not canonical JSON with one terminal newline")
    return value


def _strict_jsonl(raw: bytes, name: str) -> list[dict[str, Any]]:
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"{name} lacks its terminal newline")
    lines = raw[:-1].split(b"\n") if raw else []
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"{name} line {index} is blank")
        value = _strict_json(line + b"\n", f"{name} line {index}")
        rows.append(value)
    return rows


def _checksum_table(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("SHA256SUMS is not ASCII") from error
    expected_names = sorted({DECISIONS_FILENAME, MANIFEST_FILENAME})
    lines = text.splitlines()
    if len(lines) != len(expected_names) or not text.endswith("\n"):
        raise ValueError("SHA256SUMS coverage is not exact")
    table: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("SHA256SUMS is not canonical")
        digest, filename = line[:64], line[66:]
        _require_sha256(digest, "SHA256SUMS digest")
        if filename in table:
            raise ValueError("SHA256SUMS repeats a filename")
        table[filename] = digest
    canonical_layout = "".join(
        f"{table[name]}  {name}\n" for name in sorted(table)
    ).encode("ascii")
    if sorted(table) != expected_names or raw != canonical_layout:
        raise ValueError("SHA256SUMS coverage/order is not canonical")
    return table


@dataclass(frozen=True, slots=True)
class _LoadedShard:
    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]


def _integer_map(value: object, name: str) -> dict[int, list[int]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result: dict[int, list[int]] = {}
    for key, items in value.items():
        if type(key) is not str or not key.isdigit() or str(int(key)) != key:
            raise ValueError(f"{name} keys must be canonical non-negative integers")
        if not isinstance(items, list) or any(type(item) is not int for item in items):
            raise ValueError(f"{name}[{key}] must be an integer list")
        if items != sorted(set(items)):
            raise ValueError(f"{name}[{key}] must be sorted and unique")
        result[int(key)] = items
    return result


def _edge_set(value: object, name: str) -> set[tuple[int, int]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result: set[tuple[int, int]] = set()
    for edge in value:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(node) is not int for node in edge)
            or edge[0] >= edge[1]
        ):
            raise ValueError(f"{name} entries must be canonical [u,v] edges")
        result.add((edge[0], edge[1]))
    if len(result) != len(value):
        raise ValueError(f"{name} contains duplicate edges")
    return result


def _validate_host_metadata(
    row: StructuralPlanRow, value: object
) -> dict[str, Any]:
    fields = {
        "fault_intended",
        "fault_model",
        "host_condition",
        "ideal_host_sha256",
        "ideal_node_count",
        "realized_host_sha256",
        "realized_node_count",
        "removed_nodes",
    }
    raw = _exact_mapping(value, fields, "host realization")
    _, expected = _host_realization(row)
    if raw != expected:
        raise ValueError("host realization differs from the faults committed by the plan")
    return raw


def _validate_witness(
    row: StructuralPlanRow,
    value: object,
    host_realization: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "L_max",
        "Q",
        "chains",
        "host_edges",
        "host_nodes",
        "ink_config",
        "logical_edges",
        "mode",
        "seed",
        "validation_errors",
    }
    raw = _exact_mapping(value, fields, "planted witness")
    if raw["mode"] != row.mode or raw["seed"] != row.generator_seed:
        raise ValueError("planted witness mode/seed differs from the plan")
    if raw["ink_config"] != _json_value(asdict(MODES[row.mode])):
        raise ValueError("planted witness ink configuration differs from the registered mode")
    if raw["validation_errors"] != []:
        raise ValueError("planted witness declares validation errors")
    host, expected_host_metadata = _host_realization(row)
    if dict(host_realization) != expected_host_metadata:
        raise ValueError("planted witness uses unverified host metadata")
    host_payload = _graph_payload(host)
    if raw["host_nodes"] != host_payload["nodes"] or raw["host_edges"] != host_payload["edges"]:
        raise ValueError("planted witness does not contain the exact realized host")
    if content_digest(host_payload) != host_realization["realized_host_sha256"]:
        raise ValueError("planted witness host digest differs from host realization")
    chains = _integer_map(raw["chains"], "planted witness chains")
    if set(chains) != set(range(row.n_vars)) or any(not chain for chain in chains.values()):
        raise ValueError("planted witness must contain one non-empty chain per variable")
    host_nodes = set(raw["host_nodes"])
    host_edges = {tuple(edge) for edge in raw["host_edges"]}
    adjacency: dict[int, set[int]] = {node: set() for node in host_nodes}
    for left, right in host_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    occupied: set[int] = set()
    for variable, chain_list in chains.items():
        chain = set(chain_list)
        if not chain <= host_nodes:
            raise ValueError(f"planted witness chain {variable} leaves the realized host")
        if occupied & chain:
            raise ValueError("planted witness chains overlap")
        occupied |= chain
        reached = {chain_list[0]}
        frontier = [chain_list[0]]
        while frontier:
            node = frontier.pop()
            for neighbour in adjacency[node] & chain:
                if neighbour not in reached:
                    reached.add(neighbour)
                    frontier.append(neighbour)
        if reached != chain:
            raise ValueError(f"planted witness chain {variable} is disconnected")
    if raw["Q"] != sum(len(chain) for chain in chains.values()):
        raise ValueError("planted witness Q is inconsistent with its chains")
    if raw["L_max"] != max(len(chain) for chain in chains.values()):
        raise ValueError("planted witness L_max is inconsistent with its chains")
    owner = {node: variable for variable, chain in chains.items() for node in chain}
    contacts = {
        tuple(sorted((owner[left], owner[right])))
        for left, right in host_edges
        if left in owner and right in owner and owner[left] != owner[right]
    }
    logical_edges = _edge_set(raw["logical_edges"], "planted witness logical_edges")
    if logical_edges != contacts:
        raise ValueError("planted witness logical graph is not its realized contact quotient")
    return raw


def _validate_decision_provenance(
    row: StructuralPlanRow,
    decision: Mapping[str, Any],
    witness: Mapping[str, Any],
) -> None:
    if (
        decision["instance_id"] != row.lineage_id
        or decision["mode"] != row.mode
        or decision["topology"] != row.topology
        or decision["size"] != row.size
        or decision["radius"] != row.radius
        or decision["l_cap"] != row.l_cap
    ):
        raise ValueError("decision provenance differs from its immutable plan row")
    window_nodes = decision["window_nodes"]
    if (
        not isinstance(window_nodes, list)
        or any(type(node) is not int for node in window_nodes)
        or window_nodes != sorted(set(window_nodes))
    ):
        raise ValueError("decision window_nodes must be sorted and unique")
    if not set(window_nodes) <= set(witness["host_nodes"]):
        raise ValueError("decision window leaves the planted witness host")
    host_edges = {tuple(edge) for edge in witness["host_edges"]}
    expected_window_edges = {
        edge for edge in host_edges if edge[0] in window_nodes and edge[1] in window_nodes
    }
    if _edge_set(decision["window_edges"], "decision window_edges") != expected_window_edges:
        raise ValueError("decision window_edges are not the induced realized-host window")
    in_play = decision["in_play"]
    if (
        not isinstance(in_play, list)
        or any(type(variable) is not int for variable in in_play)
        or len(in_play) != len(set(in_play))
        or not set(in_play) <= set(range(row.n_vars))
    ):
        raise ValueError("decision in_play is invalid")
    logical_edges = _edge_set(witness["logical_edges"], "witness logical_edges")
    expected_logical = {
        edge for edge in logical_edges if edge[0] in in_play and edge[1] in in_play
    }
    if _edge_set(decision["logical_edges"], "decision logical_edges") != expected_logical:
        raise ValueError("decision logical_edges are not induced by its in-play variables")
    witness_chains = _integer_map(witness["chains"], "witness chains")
    witness_q = sum(len(witness_chains[variable]) for variable in in_play)
    witness_lmax = max(len(witness_chains[variable]) for variable in in_play)
    if decision["witness_outcome"] != [1, -witness_q, -witness_lmax]:
        raise ValueError("decision witness outcome differs from its planted in-play chains")
    if row.q_cap_slack is not None:
        q_cap = witness_q + row.q_cap_slack
        if any(value[0] == 1 and -value[1] > q_cap for value in decision["values"]):
            raise ValueError("decision feasible outcome exceeds its committed qubit cap")
    cores = _integer_map(decision["cores"], "decision cores")
    if set(cores) != set(in_play):
        raise ValueError("decision cores do not cover exactly the in-play variables")
    for variable, core in cores.items():
        if not set(core) <= set(witness_chains[variable]):
            raise ValueError("decision core is not a prefix-subset of its planted chain")
    frozen = _integer_map(decision["frozen"], "decision frozen chains")
    if set(frozen) & set(in_play):
        raise ValueError("decision frozen variables overlap the in-play variables")
    for variable, chain in frozen.items():
        if chain != witness_chains[variable]:
            raise ValueError("decision does not preserve a full planted frozen chain")
    actions = decision["actions"]
    if len({tuple(action) for action in actions}) != len(actions):
        raise ValueError("decision contains duplicate actions")
    if any(
        action[0] != decision["focus"] or action[1] not in window_nodes
        for action in actions
    ):
        raise ValueError("decision action leaves its focus/window universe")


def _replay_decision_labels(
    row: StructuralPlanRow,
    decision: Mapping[str, Any],
    witness: Mapping[str, Any],
) -> None:
    """Recompute every persisted structural label from the authenticated state."""

    full_host = nx.Graph()
    full_host.add_nodes_from(witness["host_nodes"])
    full_host.add_edges_from(witness["host_edges"])
    full_logical = nx.Graph()
    full_logical.add_nodes_from(range(row.n_vars))
    full_logical.add_edges_from(witness["logical_edges"])
    window = set(decision["window_nodes"])
    window_host = full_host.subgraph(window).copy()
    in_play = list(decision["in_play"])
    in_play_set = set(in_play)
    window_logical = full_logical.subgraph(in_play).copy()
    witness_chains = {
        variable: frozenset(chain)
        for variable, chain in _integer_map(
            witness["chains"], "replay witness chains"
        ).items()
    }
    cores = {
        variable: frozenset(chain)
        for variable, chain in _integer_map(
            decision["cores"], "replay decision cores"
        ).items()
    }

    adjacent_to_window = {
        neighbour
        for qubit in window
        for neighbour in full_host.neighbors(qubit)
    }
    expected_frozen = {
        variable: chain
        for variable, chain in witness_chains.items()
        if variable not in in_play_set and chain & adjacent_to_window
    }
    recorded_frozen = {
        variable: frozenset(chain)
        for variable, chain in _integer_map(
            decision["frozen"], "replay frozen chains"
        ).items()
    }
    if recorded_frozen != expected_frozen:
        raise ValueError("structural exact replay found incomplete frozen context")

    expected_adjacency = _json_value(frozen_adjacency(full_host, expected_frozen, window))
    if decision["frozen_adjacency"] != expected_adjacency:
        raise ValueError("structural exact replay found changed frozen adjacency")
    must_hit, expected_frozen_edges = frozen_requirements(
        full_host,
        full_logical,
        in_play,
        expected_frozen,
        window,
    )

    recorded_frozen_edges = decision["frozen_edges"]
    if not isinstance(recorded_frozen_edges, list) or any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(type(node) is not int for node in edge)
        for edge in recorded_frozen_edges
    ):
        raise ValueError("structural exact replay found malformed frozen-edge context")
    recorded_edge_roles = {tuple(edge) for edge in recorded_frozen_edges}
    expected_edge_roles = {tuple(edge) for edge in expected_frozen_edges}
    if (
        len(recorded_edge_roles) != len(recorded_frozen_edges)
        or recorded_edge_roles != expected_edge_roles
    ):
        raise ValueError("structural exact replay found changed frozen-edge context")

    q_cap = None
    if row.q_cap_slack is not None:
        q_cap = sum(len(witness_chains[variable]) for variable in in_play)
        q_cap += row.q_cap_slack
    replay_stats: dict[str, int] = {}
    replayed_values: list[list[int]] = []
    try:
        for action in decision["actions"]:
            variable, qubit = action
            action_cores = dict(cores)
            if qubit in {item for chain in cores.values() for item in chain}:
                raise ValueError("structural exact replay found an occupied action qubit")
            current = cores[variable]
            if current and not any(full_host.has_edge(qubit, item) for item in current):
                raise ValueError("structural exact replay found a disconnected action")
            action_cores[variable] = current | {qubit}
            outcome, _completion = best_completion(
                window_host,
                window_logical,
                action_cores,
                l_cap=row.l_cap,
                q_cap=q_cap,
                max_nodes=row.max_nodes,
                stats=replay_stats,
                must_hit=must_hit,
            )
            replayed_values.append(list(outcome))
        witness_outcome, _completion = best_completion(
            window_host,
            window_logical,
            {variable: witness_chains[variable] for variable in in_play},
            l_cap=row.l_cap,
            q_cap=q_cap,
            max_nodes=row.max_nodes,
            stats=replay_stats,
            must_hit=must_hit,
        )
    except SearchAborted as exc:
        raise ValueError("structural exact replay exceeded its committed node budget") from exc

    if replayed_values != decision["values"]:
        raise ValueError("structural action values differ from exact replay")
    if list(witness_outcome) != decision["witness_outcome"]:
        raise ValueError("structural witness outcome differs from exact replay")
    if replay_stats.get("nodes", 0) != decision["exact_nodes"]:
        raise ValueError("structural node count differs from exact replay")

    order = sorted(
        range(len(decision["actions"])),
        key=lambda index: tuple(replayed_values[index]),
        reverse=True,
    )
    margin_kind, margin = compare(
        tuple(replayed_values[order[0]]),
        tuple(replayed_values[order[1]]),
    )
    greedy = list(
        greedy_local_choice(
            window_host,
            window_logical,
            cores,
            decision["focus"],
            [tuple(action) for action in decision["actions"]],
        )
    )
    expected = {
        "best_action": decision["actions"][order[0]],
        "greedy_action": greedy,
        "greedy_agrees": replayed_values[decision["actions"].index(greedy)]
        == replayed_values[order[0]],
        "margin": int(margin),
        "margin_kind": margin_kind,
    }
    if any(decision[field] != value for field, value in expected.items()):
        raise ValueError("structural ranking metadata differs from exact replay")


def _validate_record(record: Mapping[str, Any], rows: Mapping[str, StructuralPlanRow]) -> None:
    fields = {
        "decision",
        "decision_index",
        "difficulty",
        "host_realization",
        "instance_id",
        "label_authority",
        "lineage_id",
        "ood",
        "plan_row",
        "planted_witness",
        "record_id",
        "schema",
        "schema_version",
        "split",
    }
    raw = _exact_mapping(record, fields, "structural decision record")
    if raw["schema"] != RECORD_SCHEMA or raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported structural decision record schema")
    lineage_id = raw["lineage_id"]
    if lineage_id not in rows:
        raise ValueError("record lineage_id is absent from the release plan")
    row = rows[lineage_id]
    if raw["plan_row"] != _json_value(row.to_dict()):
        raise ValueError("record plan_row differs from the immutable plan")
    if raw["split"] != row.split or raw["ood"] != row.ood or raw["difficulty"] != row.difficulty:
        raise ValueError("record split/profile metadata differs from the plan")
    decision = _exact_mapping(
        raw["decision"], set(DecisionRecord.__dataclass_fields__), "decision"
    )
    _validate_decision(decision)
    _require_int(raw["decision_index"], "decision_index", minimum=0)
    if not isinstance(raw["instance_id"], str) or not raw["instance_id"].startswith("sdi-"):
        raise ValueError("instance_id is invalid")
    expected_instance = "sdi-" + content_digest(
        {
            "decision": decision,
            "domain": "embedbench.structural-decision-instance",
            "kept_index": raw["decision_index"],
            "lineage_id": lineage_id,
            "schema_version": SCHEMA_VERSION,
        }
    )
    if raw["instance_id"] != expected_instance:
        raise ValueError("instance_id does not match lineage and decision index")
    if raw["label_authority"] != {
        "algorithm": "exact_branch_and_bound_window_v1",
        "budget": {"max_nodes": row.max_nodes},
        "capacity_constraint": {"q_cap_slack": row.q_cap_slack},
        "scope": "complete_in_play_variables_inside_recorded_window",
        "status": "exact",
    }:
        raise ValueError("label_authority is not the registered exact-window policy")
    expected_record = "sdr-" + content_digest(
        {key: value for key, value in raw.items() if key != "record_id"}
    )
    if raw["record_id"] != expected_record:
        raise ValueError("record_id does not match canonical record content")
    realization = _validate_host_metadata(row, raw["host_realization"])
    witness = _validate_witness(row, raw["planted_witness"], realization)
    _validate_decision_provenance(row, decision, witness)
    _replay_decision_labels(row, decision, witness)


def _validate_manifest_digest(manifest: Mapping[str, Any]) -> None:
    expected = _require_sha256(manifest.get("record_digest"), "manifest record_digest")
    actual = content_digest(
        {key: value for key, value in manifest.items() if key != "record_digest"}
    )
    if expected != actual:
        raise ValueError("manifest record_digest mismatch")


def _validate_stats(value: object, *, record_count: int, status: str) -> dict[str, Any]:
    raw = _exact_mapping(value, set(GenStats.__dataclass_fields__), "lineage stats")
    for key, item in raw.items():
        if key == "margin_kinds":
            if not isinstance(item, Mapping) or any(
                type(name) is not str or type(count) is not int or count < 0
                for name, count in item.items()
            ):
                raise ValueError("lineage stats margin_kinds must be non-negative counts")
        elif type(item) is not int or item < 0:
            raise ValueError(f"lineage stats {key} must be a non-negative integer")
    accounted = (
        raw["samples_kept"]
        + raw["dropped_no_margin"]
        + raw["dropped_window_too_large"]
        + raw["dropped_aborted"]
        + raw["dropped_no_actions"]
    )
    if accounted != raw["samples_attempted"]:
        raise ValueError("lineage stats do not account for every attempted sample")
    if raw["samples_kept"] != record_count:
        raise ValueError("lineage stats samples_kept differs from record_count")
    if sum(raw["margin_kinds"].values()) != record_count:
        raise ValueError("lineage stats margin kinds differ from record_count")
    if raw["greedy_disagreements"] > record_count:
        raise ValueError("lineage stats greedy disagreements exceed kept records")
    if status == "generated" and (
        raw["instances"] != 1 or raw["generation_failures"] != 0
    ):
        raise ValueError("generated lineage has inconsistent instance/failure accounting")
    if status == "generation_failed" and (
        raw["instances"] != 0
        or raw["generation_failures"] != 1
        or raw["samples_attempted"] != 0
    ):
        raise ValueError("failed lineage has inconsistent instance/failure accounting")
    return raw


def _validate_lineage_result(
    value: object,
    *,
    row: StructuralPlanRow,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = {
        "failure_kind",
        "host_realization",
        "lineage_id",
        "record_count",
        "stats",
        "status",
        "witness_sha256",
    }
    raw = _exact_mapping(value, fields, "lineage result")
    if raw["lineage_id"] != row.lineage_id:
        raise ValueError("lineage result ID differs from the release plan")
    _require_int(raw["record_count"], "lineage record_count", minimum=0)
    if raw["record_count"] != len(records):
        raise ValueError("lineage result record_count differs from JSONL records")
    if raw["status"] not in {"generated", "generation_failed"}:
        raise ValueError("lineage result has an unknown status")
    _validate_host_metadata(row, raw["host_realization"])
    _validate_stats(raw["stats"], record_count=len(records), status=raw["status"])
    if raw["status"] == "generation_failed":
        if (
            raw["failure_kind"] != "planted_generation_error"
            or raw["witness_sha256"] is not None
            or records
        ):
            raise ValueError("failed lineage result fabricates witness/record output")
    else:
        if raw["failure_kind"] is not None:
            raise ValueError("generated lineage result declares a failure kind")
        witness_sha256 = _require_sha256(
            raw["witness_sha256"], "lineage witness_sha256"
        )
        for record in records:
            if content_digest(record["planted_witness"]) != witness_sha256:
                raise ValueError("lineage witness digest differs from a wrapped record")
            if record["host_realization"] != raw["host_realization"]:
                raise ValueError("lineage host realization differs across records")
    return raw


def _validate_shard_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: StructuralReleasePlan,
    records: Sequence[Mapping[str, Any]],
    decisions_raw: bytes,
) -> None:
    fields = {
        "artifact",
        "attrition",
        "counts",
        "label_policy",
        "lineage_results",
        "plan",
        "plan_digest",
        "planned_lineage_ids",
        "record_digest",
        "schema",
        "schema_version",
        "shard_count",
        "shard_index",
    }
    raw = _exact_mapping(manifest, fields, "shard manifest")
    _validate_manifest_digest(raw)
    if raw["schema"] != SHARD_MANIFEST_SCHEMA or raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported structural shard manifest schema")
    if raw["plan_digest"] != plan.plan_digest:
        raise ValueError("shard plan_digest differs from the requested plan")
    embedded_plan = StructuralReleasePlan.from_dict(raw["plan"])
    if embedded_plan != plan:
        raise ValueError("shard embeds a different immutable release plan")
    if raw["artifact"] != _artifact(decisions_raw):
        raise ValueError("shard artifact metadata differs from authenticated JSONL bytes")
    if raw["label_policy"] != _label_policy():
        raise ValueError("shard label policy differs from the exact-only protocol")
    _require_int(raw["shard_count"], "shard_count")
    _require_int(raw["shard_index"], "shard_index", minimum=0)
    lineage_ids = raw["planned_lineage_ids"]
    results = raw["lineage_results"]
    if (
        not isinstance(lineage_ids, list)
        or any(type(lineage_id) is not str for lineage_id in lineage_ids)
        or lineage_ids != sorted(set(lineage_ids))
    ):
        raise ValueError("shard planned_lineage_ids must be sorted and unique")
    if not isinstance(results, list) or len(results) != len(lineage_ids):
        raise ValueError("shard lineage_results coverage differs from planned lineages")
    row_map = {row.lineage_id: row for row in plan.rows}
    per_lineage: dict[str, list[Mapping[str, Any]]] = {
        lineage_id: [] for lineage_id in lineage_ids
    }
    for record in records:
        lineage_id = record["lineage_id"]
        if lineage_id not in per_lineage:
            raise ValueError("JSONL record belongs to an unplanned shard lineage")
        per_lineage[lineage_id].append(record)
    validated_results = []
    for lineage_id, result in zip(lineage_ids, results, strict=True):
        if lineage_id not in row_map:
            raise ValueError("shard lineage is absent from the release plan")
        lineage_records = sorted(
            per_lineage[lineage_id], key=lambda record: record["decision_index"]
        )
        if [record["decision_index"] for record in lineage_records] != list(
            range(len(lineage_records))
        ):
            raise ValueError("lineage decision indices are not exact consecutive coverage")
        validated_results.append(
            _validate_lineage_result(
                result, row=row_map[lineage_id], records=lineage_records
            )
        )
    status = Counter(str(result["status"]) for result in validated_results)
    expected_counts = {
        "lineages": len(lineage_ids),
        "records": len(records),
        "status": {key: status[key] for key in sorted(status)},
    }
    if raw["counts"] != expected_counts:
        raise ValueError("shard manifest counts differ from its authenticated contents")
    if raw["attrition"] != _aggregate_attrition(validated_results):
        raise ValueError("shard manifest attrition differs from lineage results")


def _load_shard(
    directory: Path, plan: StructuralReleasePlan
) -> _LoadedShard:
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        raise ValueError(f"shard directory is missing: {directory}") from None
    if not stat.S_ISDIR(metadata.st_mode) or directory.is_symlink():
        raise ValueError("shard path must be a real directory")
    observed = {entry.name for entry in directory.iterdir()}
    if observed != _EXPECTED_FILES:
        raise ValueError(
            f"shard contains missing or unexpected files: "
            f"missing={sorted(_EXPECTED_FILES - observed)}, "
            f"unexpected={sorted(observed - _EXPECTED_FILES)}"
        )
    decisions_raw = _read_regular(directory / DECISIONS_FILENAME, DECISIONS_FILENAME)
    manifest_raw = _read_regular(directory / MANIFEST_FILENAME, MANIFEST_FILENAME)
    checksum_raw = _read_regular(directory / CHECKSUM_FILENAME, CHECKSUM_FILENAME)
    table = _checksum_table(checksum_raw)
    for name, raw in ((DECISIONS_FILENAME, decisions_raw), (MANIFEST_FILENAME, manifest_raw)):
        actual = _sha256(raw)
        if table[name] != actual:
            raise ValueError(f"SHA-256 mismatch for {name}: expected {table[name]}, got {actual}")
    manifest = _strict_json(manifest_raw, MANIFEST_FILENAME)
    records = tuple(_strict_jsonl(decisions_raw, DECISIONS_FILENAME))
    row_map = {row.lineage_id: row for row in plan.rows}
    for record in records:
        _validate_record(record, row_map)
    _require_unique((record["instance_id"] for record in records), "instance_id")
    _require_unique((record["record_id"] for record in records), "record_id")
    _validate_shard_manifest(
        manifest,
        plan=plan,
        records=records,
        decisions_raw=decisions_raw,
    )
    return _LoadedShard(manifest=manifest, records=records)


def merge_structural_shards(
    plan: StructuralReleasePlan,
    shard_dirs: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
) -> StructuralMergeReceipt:
    """Verify exact shard coverage and publish a shard-count-invariant release."""

    if not isinstance(plan, StructuralReleasePlan):
        raise TypeError("plan must be a StructuralReleasePlan")
    plan = StructuralReleasePlan.from_dict(plan.to_dict())
    if (
        isinstance(shard_dirs, (str, bytes))
        or not isinstance(shard_dirs, Sequence)
        or not shard_dirs
    ):
        raise ValueError("shard_dirs must be a non-empty path sequence")
    loaded = tuple(_load_shard(Path(path), plan) for path in shard_dirs)
    counts = {item.manifest.get("shard_count") for item in loaded}
    if len(counts) != 1:
        raise ValueError("shard coverage mixes shard_count values")
    shard_count = next(iter(counts))
    _require_int(shard_count, "declared shard_count")
    indices = [item.manifest.get("shard_index") for item in loaded]
    expected_indices = list(range(shard_count))
    if len(indices) != shard_count or sorted(indices) != expected_indices:
        raise ValueError(
            f"shard-index coverage must be exact: expected={expected_indices}, "
            f"observed={sorted(indices)}"
        )
    by_index = {int(item.manifest["shard_index"]): item for item in loaded}
    all_lineage_results: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    observed_lineages: list[str] = []
    for index in expected_indices:
        item = by_index[index]
        expected_lineages = [
            row.lineage_id
            for row in plan.rows
            if shard_index_for(row.lineage_id, shard_count) == index
        ]
        if item.manifest.get("planned_lineage_ids") != expected_lineages:
            raise ValueError(f"lineage coverage differs from the plan for shard {index}")
        results = item.manifest.get("lineage_results")
        if not isinstance(results, list):
            raise ValueError("shard lineage_results must be a list")
        result_ids = [result.get("lineage_id") for result in results if isinstance(result, Mapping)]
        if result_ids != expected_lineages:
            raise ValueError(f"lineage result coverage differs from the plan for shard {index}")
        per_lineage = Counter(record["lineage_id"] for record in item.records)
        for result in results:
            if result.get("record_count") != per_lineage[result["lineage_id"]]:
                raise ValueError("lineage record_count differs from JSONL records")
        observed_lineages.extend(expected_lineages)
        all_lineage_results.extend(results)
        all_records.extend(item.records)
    _require_unique(observed_lineages, "lineage_id")
    if sorted(observed_lineages) != sorted(row.lineage_id for row in plan.rows):
        raise ValueError("release lineage coverage is not exact")
    _require_unique((record["instance_id"] for record in all_records), "instance_id")
    _require_unique((record["record_id"] for record in all_records), "record_id")
    all_records.sort(key=lambda record: (record["lineage_id"], record["decision_index"]))
    all_lineage_results.sort(key=lambda result: result["lineage_id"])
    records_raw = _jsonl_bytes(all_records)
    lineage_ids = tuple(row.lineage_id for row in plan.rows)
    manifest_payload = _manifest_payload(
        schema=RELEASE_MANIFEST_SCHEMA,
        plan=plan,
        records_raw=records_raw,
        lineage_results=all_lineage_results,
        planned_lineage_ids=lineage_ids,
    )
    manifest_raw = _canonical_document(_record_document(manifest_payload))
    destination = Path(output_dir)
    _publish_directory(
        destination,
        {DECISIONS_FILENAME: records_raw, MANIFEST_FILENAME: manifest_raw},
    )
    root = destination.resolve(strict=True)
    return StructuralMergeReceipt(
        output_directory=root,
        decisions_path=root / DECISIONS_FILENAME,
        manifest_path=root / MANIFEST_FILENAME,
        checksums_path=root / CHECKSUM_FILENAME,
        plan_digest=plan.plan_digest,
        lineage_count=len(lineage_ids),
        record_count=len(all_records),
    )


__all__ = [
    "CHECKSUM_FILENAME",
    "DECISIONS_FILENAME",
    "MANIFEST_FILENAME",
    "StructuralMergeReceipt",
    "StructuralPlanRow",
    "StructuralReleasePlan",
    "StructuralShardReceipt",
    "build_production_plan",
    "generate_structural_shard",
    "make_structural_plan_row",
    "merge_structural_shards",
    "shard_index_for",
]

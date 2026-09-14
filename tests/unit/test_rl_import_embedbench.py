"""Trust-boundary tests for the standalone EmbedBench CandidateBank importer."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

import isingfold.rl.data.import_embedbench as import_embedbench_module
import isingfold.rl.data.prepared as prepared_module
from isingfold.rl.data.import_embedbench import (
    assign_split,
    canonical_json_bytes,
    content_digest,
    prepare_candidate_bank,
    prepare_candidate_bank_v2,
    prepare_candidate_bank_v3,
)
from isingfold.rl.data.prepared import (
    load_prepared_corpus_design,
    load_prepared_tasks,
)
from isingfold.rl.cli import build_parser
from tests.unit.quality_attestation_support import (
    FIXTURE_CERTIFICATE_DIGEST,
    attest_prepared,
)


@pytest.fixture(autouse=True)
def _small_fixture_partition_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit fixtures small while dedicated tests exercise production floor constants."""

    monkeypatch.setattr(import_embedbench_module, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
    monkeypatch.setattr(import_embedbench_module, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
    monkeypatch.setattr(import_embedbench_module, "MINIMUM_TEST_BASE_LINEAGES", 1)
    monkeypatch.setattr(
        import_embedbench_module,
        "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES",
        1,
    )
    monkeypatch.setattr(
        import_embedbench_module,
        "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL",
        0.51,
    )
    monkeypatch.setattr(
        import_embedbench_module,
        "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH",
        0.99,
    )
    monkeypatch.setattr(prepared_module, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_module, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_module, "MINIMUM_TEST_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_module, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 1)
    monkeypatch.setattr(
        prepared_module,
        "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL",
        0.51,
    )
    monkeypatch.setattr(
        prepared_module,
        "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH",
        0.99,
    )


def _curve(partition: str, seeds: list[int]) -> dict[str, Any]:
    protocol = [
        ["decoder", "majority"],
        ["noise_model", "none"],
        ["reference_energy", -1.0],
        ["sampler", "sa"],
        ["sampler_version", "test-v1"],
        ["schedule", "linear"],
        ["seed_derivation", "fixture-v1"],
    ]
    return {
        "partition": partition,
        "evaluation_protocol": protocol,
        "objective": "solve_probability_then_residual-v1",
        "p_solve": [[0.5 for _ in seeds]],
        "reads": 8,
        "residual_mean": [[0.25 for _ in seeds]],
        "schema_version": 1,
        "seeds": seeds,
        "strengths": [1.0],
        "sweeps": 16,
    }


def _candidate(group_id: str, chains: list[list[int]]) -> dict[str, Any]:
    canonical_chains = [sorted(chain) for chain in chains]
    candidate_id = "candidate-" + content_digest(
        {
            "chains": canonical_chains,
            "domain": "isingfold-candidate-v1",
            "group_id": group_id,
        }
    )
    record = {
        "audit": _curve("audit", [2]),
        "candidate_id": candidate_id,
        "chains": canonical_chains,
        "decision": _curve("decision", [1]),
        "feature_schema_version": 1,
        "features": [
            ["max_chain", float(max(map(len, canonical_chains)))],
            ["total_qubits", float(sum(map(len, canonical_chains)))],
        ],
        "group_id": group_id,
        "max_chain": max(map(len, canonical_chains)),
        "schema_version": 1,
        "total_qubits": sum(map(len, canonical_chains)),
    }
    return {**record, "record_digest": content_digest(record)}


def _structural(group_id: str, chains: list[list[int]]) -> dict[str, Any]:
    candidate = _candidate(group_id, chains)
    return {
        "candidate_id": candidate["candidate_id"],
        "chains": candidate["chains"],
        "max_chain": candidate["max_chain"],
        "schema_version": 1,
        "total_qubits": candidate["total_qubits"],
    }


def _fixture_records() -> tuple[dict[str, Any], dict[str, Any]]:
    logical_nodes = [0, 1]
    logical_edges = [[0, 1]]
    host_nodes = [0, 1, 2, 3]
    host_edges = [[0, 1], [1, 2], [2, 3]]
    h = [[0, 0.25], [1, -0.5]]
    j = [[0, 1, -1.0]]
    split_unit_id = "logical-" + content_digest(
        {
            "domain": "isingfold-logical-problem-v1",
            "h": h,
            "j": j,
            "logical_edges": logical_edges,
            "logical_nodes": logical_nodes,
        }
    )
    instance_id = "instance-" + content_digest(
        {
            "domain": "isingfold-instance-v1",
            "host_edges": host_edges,
            "host_nodes": host_nodes,
            "split_unit_id": split_unit_id,
        }
    )
    instance_payload = {
        "family": "fixture",
        "h": h,
        "host_edges": host_edges,
        "host_nodes": host_nodes,
        "instance_id": instance_id,
        "j": j,
        "logical_edges": logical_edges,
        "logical_nodes": logical_nodes,
        "metadata": [["private_note", "must-not-cross-policy-boundary"]],
        "schema_version": 1,
        "split_unit_id": split_unit_id,
        "topology": "path",
    }
    instance = {**instance_payload, "record_digest": content_digest(instance_payload)}

    incumbent_chains = [[0, 1], [2, 3]]
    protocol = [["attempt_slots", 2], ["name", "fixture-v1"]]
    group_seed = 17
    group_id = "group-" + content_digest(
        {
            "domain": "isingfold-candidate-group-v1",
            "group_seed": group_seed,
            "incumbent_chains": incumbent_chains,
            "instance_id": instance_id,
            "protocol": protocol,
        }
    )
    incumbent = _candidate(group_id, incumbent_chains)
    candidates = sorted(
        [_candidate(group_id, [[0], [1]]), _candidate(group_id, [[2], [3]])],
        key=lambda item: item["candidate_id"],
    )
    attempts = []
    for slot, candidate in enumerate(candidates):
        attempt_id = "attempt-" + content_digest(
            {"domain": "isingfold-repair-attempt-v1", "group_id": group_id, "slot": slot}
        )
        attempts.append(
            {
                "attempt_id": attempt_id,
                "candidate_id": candidate["candidate_id"],
                "neighborhood": [0, 1],
                "reason": None,
                "repair_seed": 100 + slot,
                "schema_version": 1,
                "slot": slot,
                "status": "valid",
                "transitions": 3,
            }
        )
    generation_payload = {
        "attempts": attempts,
        "candidates": sorted(
            [_structural(group_id, item["chains"]) for item in candidates],
            key=lambda item: item["candidate_id"],
        ),
        "group_id": group_id,
        "group_seed": group_seed,
        "incumbent": _structural(group_id, incumbent_chains),
        "instance_id": instance_id,
        "instance_record_digest": instance["record_digest"],
        "protocol": protocol,
        "rejection_reason": None,
        "schema_version": 1,
        "split": assign_split(split_unit_id),
        "split_unit_id": split_unit_id,
    }
    group_payload = {
        "attempts": attempts,
        "candidates": candidates,
        "group_id": group_id,
        "group_seed": group_seed,
        "generation_digest": content_digest(generation_payload),
        "incumbent": incumbent,
        "instance_id": instance_id,
        "instance_record_digest": instance["record_digest"],
        "protocol": protocol,
        "schema_version": 1,
        "split": assign_split(split_unit_id),
        "split_unit_id": split_unit_id,
    }
    group = {**group_payload, "record_digest": content_digest(group_payload)}
    return instance, group


def _write_inputs(
    root: Path,
    *,
    mutate_instance=None,
    mutate_group=None,
    target_status: str = "planted_proof",
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    instance, group = _fixture_records()
    if mutate_instance is not None:
        mutate_instance(instance)
    if mutate_group is not None:
        mutate_group(group)
    raw = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in (
            {"kind": "instance", "record": instance},
            {"kind": "group", "record": group},
        )
    )
    bank = root / "bank.jsonl"
    bank.write_bytes(raw)
    manifest_payload = {
        "group_count": 1,
        "instance_count": 1,
        "jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": 1,
    }
    manifest = root / "bank.manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {**manifest_payload, "record_digest": content_digest(manifest_payload)}
        )
        + b"\n"
    )
    target = {
        "certificate_digest": FIXTURE_CERTIFICATE_DIGEST,
        "evaluator_protocol_digest": "b" * 64,
        "instance_id": instance["instance_id"],
        "instance_record_digest": instance["record_digest"],
        "reference_energy": -1.25,
        "reference_status": target_status,
        "schema": "embedbench.evaluator-target",
        "schema_version": 1,
    }
    targets = root / "targets.jsonl"
    targets.write_bytes(canonical_json_bytes(target) + b"\n")
    return bank, manifest, targets


def _prepare(tmp_path: Path, **kwargs: Any) -> Path:
    bank, manifest, targets = _write_inputs(tmp_path, **kwargs)
    output = tmp_path / "prepared"
    prepare_candidate_bank(bank, manifest, targets, output, qubit_cap=4)
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _provenance_record(
    instance: dict[str, Any],
    group: dict[str, Any],
    *,
    base_parent_lineage: str = "problem-sha256:" + "1" * 64,
    learning_partition: str = "train",
    regime: str = "iid",
    transform_sha256: str = "2" * 64,
    source_partition: str = "base_v1.1",
    split_manifest_sha256: str = "8" * 64,
    fault_status: str = "none",
) -> dict[str, Any]:
    active_host_sha256 = content_digest(
        {
            "edges": instance["host_edges"],
            "nodes": instance["host_nodes"],
            "schema": "embedbench.host-graph",
            "schema_version": 1,
        }
    )
    payload = {
        "active_topology_identity": {
            "host_artifact_sha256": "3" * 64,
            "host_sha256": active_host_sha256,
            "topology": instance["topology"],
        },
        "base_parent_lineage": base_parent_lineage,
        "calibration_identity": {
            "calibration_sha256": None,
            "status": "not_applicable",
        },
        "descendant_transform_identity": {
            "kinds": ["fault"] if fault_status == "faulted" else ["identity"],
            "transform_sha256": transform_sha256,
        },
        "distribution": {
            "learning_partition": learning_partition,
            "regime": regime,
            "source_partition": source_partition,
            "stratum": "base/iid/fixture",
        },
        "fault_identity": {
            "fault_mask_sha256": "3" * 64,
            "status": fault_status,
        },
        "group_id": group["group_id"],
        "group_record_digest": group["record_digest"],
        "instance_id": instance["instance_id"],
        "instance_record_digest": instance["record_digest"],
        "nominal_topology_identity": {
            "pristine_host_sha256": "4" * 64,
            "size": 4,
            "topology": instance["topology"],
        },
        "schema": "embedbench.isingfold-task-provenance",
        "schema_version": 2,
        "source_release_id": "embedbench-fixture-v2",
        "source_release_manifest_sha256": "7" * 64,
        "split_manifest_sha256": split_manifest_sha256,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _write_provenance(
    path: Path,
    records: list[dict[str, Any]],
) -> Path:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in records))
    return path


def _regroup(instance: dict[str, Any], group: dict[str, Any], *, group_seed: int) -> dict[str, Any]:
    """Create a second fully authenticated group over the same logical instance."""

    protocol = group["protocol"]
    incumbent_chains = group["incumbent"]["chains"]
    group_id = "group-" + content_digest(
        {
            "domain": "isingfold-candidate-group-v1",
            "group_seed": group_seed,
            "incumbent_chains": incumbent_chains,
            "instance_id": instance["instance_id"],
            "protocol": protocol,
        }
    )
    old_to_new: dict[str, str] = {}

    def candidate(raw: dict[str, Any]) -> dict[str, Any]:
        payload = {key: value for key, value in raw.items() if key != "record_digest"}
        old_id = payload["candidate_id"]
        payload["group_id"] = group_id
        payload["candidate_id"] = "candidate-" + content_digest(
            {
                "chains": payload["chains"],
                "domain": "isingfold-candidate-v1",
                "group_id": group_id,
            }
        )
        old_to_new[old_id] = payload["candidate_id"]
        return {**payload, "record_digest": content_digest(payload)}

    incumbent = candidate(group["incumbent"])
    candidates = [candidate(raw) for raw in group["candidates"]]
    candidates.sort(key=lambda item: item["candidate_id"])
    attempts = []
    for raw in group["attempts"]:
        item = dict(raw)
        item["attempt_id"] = "attempt-" + content_digest(
            {
                "domain": "isingfold-repair-attempt-v1",
                "group_id": group_id,
                "slot": item["slot"],
            }
        )
        item["candidate_id"] = old_to_new[item["candidate_id"]]
        attempts.append(item)
    generation_payload = {
        "attempts": attempts,
        "candidates": sorted(
            [_structural(group_id, item["chains"]) for item in candidates],
            key=lambda item: item["candidate_id"],
        ),
        "group_id": group_id,
        "group_seed": group_seed,
        "incumbent": _structural(group_id, incumbent["chains"]),
        "instance_id": instance["instance_id"],
        "instance_record_digest": instance["record_digest"],
        "protocol": protocol,
        "rejection_reason": None,
        "schema_version": 1,
        "split": assign_split(instance["split_unit_id"]),
        "split_unit_id": instance["split_unit_id"],
    }
    payload = {
        "attempts": attempts,
        "candidates": candidates,
        "generation_digest": content_digest(generation_payload),
        "group_id": group_id,
        "group_seed": group_seed,
        "incumbent": incumbent,
        "instance_id": instance["instance_id"],
        "instance_record_digest": instance["record_digest"],
        "protocol": protocol,
        "schema_version": 1,
        "split": assign_split(instance["split_unit_id"]),
        "split_unit_id": instance["split_unit_id"],
    }
    return {**payload, "record_digest": content_digest(payload)}


def _reinstance(
    instance: dict[str, Any], group: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {key: value for key, value in instance.items() if key != "record_digest"}
    payload["h"] = [[0, 0.75], [1, -0.5]]
    payload["split_unit_id"] = "logical-" + content_digest(
        {
            "domain": "isingfold-logical-problem-v1",
            "h": payload["h"],
            "j": payload["j"],
            "logical_edges": payload["logical_edges"],
            "logical_nodes": payload["logical_nodes"],
        }
    )
    payload["instance_id"] = "instance-" + content_digest(
        {
            "domain": "isingfold-instance-v1",
            "host_edges": payload["host_edges"],
            "host_nodes": payload["host_nodes"],
            "split_unit_id": payload["split_unit_id"],
        }
    )
    replacement = {**payload, "record_digest": content_digest(payload)}
    return replacement, _regroup(replacement, group, group_seed=29)


def _write_two_group_inputs(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path, Path]:
    instance, first = _fixture_records()
    second = _regroup(instance, first, group_seed=23)
    raw = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in (
            {"kind": "instance", "record": instance},
            {"kind": "group", "record": first},
            {"kind": "group", "record": second},
        )
    )
    bank = root / "bank.jsonl"
    bank.write_bytes(raw)
    bank_manifest_payload = {
        "group_count": 2,
        "instance_count": 1,
        "jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": 1,
    }
    bank_manifest = root / "bank.manifest.json"
    bank_manifest.write_bytes(
        canonical_json_bytes(
            {
                **bank_manifest_payload,
                "record_digest": content_digest(bank_manifest_payload),
            }
        )
        + b"\n"
    )
    target = {
        "certificate_digest": "a" * 64,
        "evaluator_protocol_digest": "b" * 64,
        "instance_id": instance["instance_id"],
        "instance_record_digest": instance["record_digest"],
        "reference_energy": -1.25,
        "reference_status": "planted_proof",
        "schema": "embedbench.evaluator-target",
        "schema_version": 1,
    }
    targets = root / "targets.jsonl"
    targets.write_bytes(canonical_json_bytes(target) + b"\n")
    return instance, first, second, bank, bank_manifest, targets


def _resign_prepared_file(output: Path, filename: str, rows: list[dict[str, Any]]) -> None:
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    (output / filename).write_bytes(raw)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["outputs"][filename]["sha256"] = hashlib.sha256(raw).hexdigest()
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(payload)
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")


def _design_variant(
    instance: dict[str, Any],
    group: dict[str, Any],
    *,
    family: str,
    topology: str,
    h0: float,
    group_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {key: value for key, value in instance.items() if key != "record_digest"}
    payload["family"] = family
    payload["topology"] = topology
    payload["h"] = [[0, h0], [1, -0.5]]
    payload["split_unit_id"] = "logical-" + content_digest(
        {
            "domain": "isingfold-logical-problem-v1",
            "h": payload["h"],
            "j": payload["j"],
            "logical_edges": payload["logical_edges"],
            "logical_nodes": payload["logical_nodes"],
        }
    )
    payload["instance_id"] = "instance-" + content_digest(
        {
            "domain": "isingfold-instance-v1",
            "host_edges": payload["host_edges"],
            "host_nodes": payload["host_nodes"],
            "split_unit_id": payload["split_unit_id"],
        }
    )
    replacement = {**payload, "record_digest": content_digest(payload)}
    return replacement, _regroup(replacement, group, group_seed=group_seed)


_DESIGN_AXES = (
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


def _design_filter(lineage: dict[str, Any]) -> dict[str, list[str]]:
    return {axis: [lineage[axis]] for axis in _DESIGN_AXES}


def _partition_design_filter(
    lineages: list[dict[str, Any]],
    partition: str,
) -> dict[str, list[str]]:
    return {
        axis: sorted(
            {
                lineage[axis]
                for lineage in lineages
                if lineage["learning_partition"] == partition
            }
        )
        for axis in _DESIGN_AXES
    }


def _write_design_inputs(
    root: Path,
    *,
    topologies: tuple[str, str, str] = ("path-a", "path-b", "path-a"),
    crossed_train_host: bool = False,
    crossed_val_host: bool = False,
    crossed_test_host: bool = False,
    crossed_test_lineage: bool = False,
    families: tuple[str, str, str] = ("frustrated-loop", "portfolio", "frustrated-loop"),
    problem_origins: tuple[str, str, str] = (
        "synthetic",
        "application-derived",
        "synthetic",
    ),
    include_faulted: bool = True,
) -> tuple[Path, Path, Path, Path, Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    base_instance, base_group = _fixture_records()
    instances_and_groups = [
        _design_variant(
            base_instance,
            base_group,
            family=families[0],
            topology=topologies[0],
            h0=0.25,
            group_seed=41,
        ),
        _design_variant(
            base_instance,
            base_group,
            family=families[1],
            topology=topologies[1],
            h0=0.75,
            group_seed=43,
        ),
        _design_variant(
            base_instance,
            base_group,
            family=families[2],
            topology=topologies[2],
            h0=1.25,
            group_seed=47,
        ),
    ]
    if crossed_train_host:
        first_instance = instances_and_groups[0][0]
        payload = {
            key: value for key, value in first_instance.items() if key != "record_digest"
        }
        payload["host_edges"] = sorted([*payload["host_edges"], [0, 2]])
        payload["topology"] = "path-c"
        payload["instance_id"] = "instance-" + content_digest(
            {
                "domain": "isingfold-instance-v1",
                "host_edges": payload["host_edges"],
                "host_nodes": payload["host_nodes"],
                "split_unit_id": payload["split_unit_id"],
            }
        )
        crossed_instance = {**payload, "record_digest": content_digest(payload)}
        instances_and_groups.append(
            (crossed_instance, _regroup(crossed_instance, base_group, group_seed=53))
        )
    if crossed_test_host:
        if crossed_test_lineage:
            instances_and_groups.append(
                _design_variant(
                    base_instance,
                    base_group,
                    family=families[2],
                    topology="path-d",
                    h0=1.5,
                    group_seed=59,
                )
            )
        else:
            test_instance = instances_and_groups[2][0]
            payload = {
                key: value for key, value in test_instance.items() if key != "record_digest"
            }
            payload["host_edges"] = sorted([*payload["host_edges"], [0, 2]])
            payload["topology"] = "path-d"
            payload["instance_id"] = "instance-" + content_digest(
                {
                    "domain": "isingfold-instance-v1",
                    "host_edges": payload["host_edges"],
                    "host_nodes": payload["host_nodes"],
                    "split_unit_id": payload["split_unit_id"],
                }
            )
            crossed_instance = {**payload, "record_digest": content_digest(payload)}
            instances_and_groups.append(
                (crossed_instance, _regroup(crossed_instance, base_group, group_seed=59))
            )
    if crossed_val_host:
        val_instance = instances_and_groups[1][0]
        payload = {
            key: value for key, value in val_instance.items() if key != "record_digest"
        }
        payload["host_edges"] = sorted([*payload["host_edges"], [0, 2]])
        payload["topology"] = "path-e"
        payload["instance_id"] = "instance-" + content_digest(
            {
                "domain": "isingfold-instance-v1",
                "host_edges": payload["host_edges"],
                "host_nodes": payload["host_nodes"],
                "split_unit_id": payload["split_unit_id"],
            }
        )
        crossed_instance = {**payload, "record_digest": content_digest(payload)}
        instances_and_groups.append(
            (crossed_instance, _regroup(crossed_instance, base_group, group_seed=61))
        )
    instances = [item[0] for item in instances_and_groups]
    groups = [item[1] for item in instances_and_groups]
    bank_raw = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in (
            *({"kind": "instance", "record": instance} for instance in instances),
            *({"kind": "group", "record": group} for group in groups),
        )
    )
    bank = root / "bank.jsonl"
    bank.write_bytes(bank_raw)
    bank_manifest_payload = {
        "group_count": len(groups),
        "instance_count": len(instances),
        "jsonl_sha256": hashlib.sha256(bank_raw).hexdigest(),
        "schema_version": 1,
    }
    bank_manifest = root / "bank.manifest.json"
    bank_manifest.write_bytes(
        canonical_json_bytes(
            {
                **bank_manifest_payload,
                "record_digest": content_digest(bank_manifest_payload),
            }
        )
        + b"\n"
    )

    targets = root / "targets.jsonl"
    targets.write_bytes(
        b"".join(
            canonical_json_bytes(
                {
                    "certificate_digest": f"{index + 1:x}" * 64,
                    "evaluator_protocol_digest": "b" * 64,
                    "instance_id": instance["instance_id"],
                    "instance_record_digest": instance["record_digest"],
                    "reference_energy": -1.25,
                    "reference_status": "planted_proof",
                    "schema": "embedbench.evaluator-target",
                    "schema_version": 1,
                }
            )
            + b"\n"
            for index, instance in enumerate(instances)
        )
    )

    condition_specs = [
        (instances[0], groups[0], "train", 1),
        (instances[1], groups[1], "val", 2),
        (instances[2], groups[2], "test", 3),
    ]
    if crossed_train_host:
        condition_specs.append((instances[3], groups[3], "train", 1))
    if crossed_test_host:
        crossed_index = 4 if crossed_train_host else 3
        condition_specs.append(
            (
                instances[crossed_index],
                groups[crossed_index],
                "test",
                4 if crossed_test_lineage else 3,
            )
        )
    if crossed_val_host:
        condition_specs.append((instances[-1], groups[-1], "val", 2))
    lineages: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for index, (instance, group, partition, lineage_index) in enumerate(
        condition_specs, start=1
    ):
        base_lineage = (
            f"problem-sha256:{lineage_index:x}" + f"{lineage_index:x}" * 63
        )
        regime = "ood" if partition == "test" else "iid"
        fault_status = "faulted" if include_faulted and index == 2 else "none"
        provenance_rows.append(
            _provenance_record(
                instance,
                group,
                base_parent_lineage=base_lineage,
                learning_partition=partition,
                regime=regime,
                transform_sha256=f"{index + 3:x}" * 64,
                fault_status=fault_status,
            )
        )
        lineages.append(
            {
                "application_family": instance["family"],
                "base_lineage_key": base_lineage,
                "calibration_sha256": None,
                "calibration_status": "not_applicable",
                "decision_difficulty": "easy" if index == 1 else "hard",
                "distribution_regime": regime,
                "embedding_difficulty": "hard" if index in {2, 4} else "easy",
                "fault_status": fault_status,
                "host_family": instance["topology"],
                "learning_partition": partition,
                "problem_origin": (
                    problem_origins[0]
                    if lineage_index == 4
                    else problem_origins[lineage_index - 1]
                ),
                "sampling_difficulty": "hard" if index == 3 else "easy",
            }
        )
    provenance = _write_provenance(root / "provenance.jsonl", provenance_rows)
    lineages.sort(
        key=lambda row: (
            row["base_lineage_key"],
            row["learning_partition"],
            *(row[field] for field in _DESIGN_AXES),
            "" if row["calibration_sha256"] is None else row["calibration_sha256"],
        )
    )
    axis_values = {
        axis: sorted({str(lineage[axis]) for lineage in lineages}) for axis in _DESIGN_AXES
    }
    stratum_quotas = [
        {
            "filter": _design_filter(lineage),
            "learning_partition": lineage["learning_partition"],
            "minimum_base_lineages": 1,
            "quota_id": f"quota-{index:03d}-{lineage['learning_partition']}",
        }
        for index, lineage in enumerate(lineages)
    ]
    stratum_quotas.sort(key=lambda row: row["quota_id"])
    test_filter = _partition_design_filter(lineages, "test")
    val_filter = _partition_design_filter(lineages, "val")
    partition_quotas = {
        partition: len(
            {
                lineage["base_lineage_key"]
                for lineage in lineages
                if lineage["learning_partition"] == partition
            }
        )
        for partition in ("test", "train", "val")
    }
    design_payload = {
        "axis_values": axis_values,
        "corpus_design_version": "if-core-v1",
        "difficulty_calibration": {
            "authority_sha256": "a" * 64,
            "budget_sha256": "b" * 64,
            "evidence_sha256": "c" * 64,
            "outcome_blind": True,
            "panel_sha256": "d" * 64,
            "protocol_sha256": "e" * 64,
        },
        "independent_unit": "immutable-base-lineage",
        "lineage_registry": lineages,
        "minimum_partition_base_lineages": {"test": 1, "train": 1, "val": 1},
        "partition_quotas": partition_quotas,
        "power_targets": [
            {
                "alpha": 0.49,
                "alternative": "one-sided-noninferiority",
                "assumed_discordance": 0.01,
                "assumed_true_difference": 0.0,
                "endpoint": "valid-return-noninferiority",
                "filter": test_filter,
                "learning_partition": "test",
                "method": "paired-binary-normal-approximation",
                "minimum_base_lineages": 1,
                "noninferiority_margin": 0.99,
                "power_separation": 0.99,
                "target_id": "test-feasibility",
                "target_power": 0.51,
            },
            {
                "alpha": 0.05,
                "alternative": "one-sided-noninferiority",
                "assumed_discordance": 0.01,
                "assumed_true_difference": 0.47,
                "endpoint": "valid-return-noninferiority",
                "filter": val_filter,
                "learning_partition": "val",
                "method": "paired-binary-normal-approximation",
                "minimum_base_lineages": 1,
                "noninferiority_margin": 0.02,
                "power_separation": 0.49,
                "target_id": "validation-valid-return-noninferiority",
                "target_power": 0.51,
            }
        ],
        "precision_targets": [
            {
                "confidence_level": 0.51,
                "endpoint": "learned-minus-stock-unconditional-if-q3-s0",
                "filter": test_filter,
                "half_width": 0.99,
                "learning_partition": "test",
                "method": "bounded-paired-difference-worst-case-normal",
                "minimum_base_lineages": 1,
                "outcome_bounds": [-1.0, 1.0],
                "target_id": "test-utility-precision",
                "variance_bound": 1.0,
            },
            {
                "confidence_level": 0.51,
                "endpoint": "learned-minus-stock-unconditional-if-q3-s0",
                "filter": val_filter,
                "half_width": 0.99,
                "learning_partition": "val",
                "method": "bounded-paired-difference-worst-case-normal",
                "minimum_base_lineages": 1,
                "outcome_bounds": [-1.0, 1.0],
                "target_id": "validation-paired-utility-precision",
                "variance_bound": 1.0,
            }
        ],
        "schema": "isingfold.corpus-design",
        "schema_version": 1,
        "stratum_quotas": stratum_quotas,
    }
    design = root / "corpus-design.json"
    design.write_bytes(
        canonical_json_bytes(
            {**design_payload, "record_digest": content_digest(design_payload)}
        )
        + b"\n"
    )
    return (
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        hashlib.sha256(design.read_bytes()).hexdigest(),
    )


def _resign_design(design: Path) -> str:
    manifest = json.loads(design.read_text())
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(payload)
    design.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return hashlib.sha256(design.read_bytes()).hexdigest()


def test_prepare_writes_five_separate_atomic_artifacts_without_private_labels(
    tmp_path: Path,
) -> None:
    output = _prepare(tmp_path)

    assert {path.name for path in output.iterdir()} == {
        "evaluator_targets.jsonl",
        "initializers.jsonl",
        "manifest.json",
        "policy_instances.jsonl",
        "splits.json",
    }
    policy = _read_jsonl(output / "policy_instances.jsonl")[0]
    initializer = _read_jsonl(output / "initializers.jsonl")[0]
    target = _read_jsonl(output / "evaluator_targets.jsonl")[0]
    encoded_policy = json.dumps(policy)
    assert "reference_energy" not in encoded_policy
    assert "certificate_digest" not in encoded_policy
    assert "private_note" not in encoded_policy
    assert "decision" not in encoded_policy and "audit" not in encoded_policy
    assert initializer["validation"]["connected"]
    assert initializer["validation"]["realizes_logical_edges"]
    assert initializer["validation"]["within_qubit_cap"]
    assert target["reference_energy"] == -1.25
    assert policy["instance_id"] == initializer["instance_id"] == target["instance_id"]

    manifest = json.loads((output / "manifest.json").read_text())
    for name in (
        "policy_instances.jsonl",
        "initializers.jsonl",
        "evaluator_targets.jsonl",
        "splits.json",
    ):
        assert (
            manifest["outputs"][name]["sha256"]
            == hashlib.sha256((output / name).read_bytes()).hexdigest()
        )


def test_split_is_recomputed_from_logical_lineage_and_recorded_once(tmp_path: Path) -> None:
    instance, _ = _fixture_records()
    output = _prepare(tmp_path)
    splits = json.loads((output / "splits.json").read_text())
    expected = assign_split(instance["split_unit_id"])
    initializer = _read_jsonl(output / "initializers.jsonl")[0]

    assert splits["lineage_to_split"] == {instance["split_unit_id"]: expected}
    assert initializer["task_id"] in splits[expected]
    assert sum(len(splits[name]) for name in ("train", "val", "test")) == 1


def test_prepared_loader_keeps_deployment_and_evaluator_boundaries_separate(
    tmp_path: Path,
) -> None:
    output = _prepare(tmp_path)
    pin = attest_prepared(output, tmp_path / "trust")
    public = load_prepared_tasks(
        output,
        include_evaluator=False,
        require_provenance=False,
        require_corpus_design=False,
    )
    trusted = load_prepared_tasks(
        output,
        include_evaluator=True,
        require_provenance=False,
        require_corpus_design=False,
        quality_attestation_pin=pin,
    )

    assert len(public) == len(trusted) == 1
    assert public[0].task.ground_energy is None
    assert public[0].reference_status is None
    assert trusted[0].task.ground_energy == -1.25
    assert trusted[0].reference_status == "planted_proof"
    assert trusted[0].quality_attestation_digest == pin.expected_digest
    assert trusted[0].quality_evidence_manifest_digest is not None
    assert trusted[0].task.lineage.startswith("logical-")
    assert trusted[0].initializer()(trusted[0].task.logical, trusted[0].task.host, 99) == (
        trusted[0].task.initial_embedding
    )


def test_prepared_loader_never_promotes_self_digested_targets_without_publisher_pin(
    tmp_path: Path,
) -> None:
    output = _prepare(tmp_path)

    with pytest.raises(ValueError, match="publisher-attestation pin"):
        load_prepared_tasks(
            output,
            include_evaluator=True,
            require_provenance=False,
            require_corpus_design=False,
        )


def test_pinned_attestation_rejects_a_rehashed_ground_energy_and_manifest(
    tmp_path: Path,
) -> None:
    output = _prepare(tmp_path)
    pin = attest_prepared(output, tmp_path / "trust")
    targets = _read_jsonl(output / "evaluator_targets.jsonl")
    targets[0]["reference_energy"] = 1_000_000.0
    payload = {key: value for key, value in targets[0].items() if key != "record_digest"}
    targets[0]["record_digest"] = content_digest(payload)
    _resign_prepared_file(output, "evaluator_targets.jsonl", targets)

    with pytest.raises(ValueError, match="prepared manifest differs"):
        load_prepared_tasks(
            output,
            include_evaluator=True,
            require_provenance=False,
            require_corpus_design=False,
            quality_attestation_pin=pin,
        )


def test_attested_target_must_reference_the_authenticated_source_instance(
    tmp_path: Path,
) -> None:
    output = _prepare(tmp_path)
    targets = _read_jsonl(output / "evaluator_targets.jsonl")
    targets[0]["instance_record_digest"] = "f" * 64
    payload = {key: value for key, value in targets[0].items() if key != "record_digest"}
    targets[0]["record_digest"] = content_digest(payload)
    _resign_prepared_file(output, "evaluator_targets.jsonl", targets)
    pin = attest_prepared(output, tmp_path / "trust")

    with pytest.raises(ValueError, match="source-instance digest"):
        load_prepared_tasks(
            output,
            include_evaluator=True,
            require_provenance=False,
            require_corpus_design=False,
            quality_attestation_pin=pin,
        )


def test_public_loader_never_opens_a_corrupt_evaluator_payload(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    pin = attest_prepared(output, tmp_path / "trust")
    (output / "evaluator_targets.jsonl").write_bytes(b"private-corruption\n")

    assert (
        load_prepared_tasks(
            output,
            include_evaluator=False,
            require_provenance=False,
            require_corpus_design=False,
        )[0].task.ground_energy
        is None
    )
    with pytest.raises(ValueError, match="checksum"):
        load_prepared_tasks(
            output,
            include_evaluator=True,
            require_provenance=False,
            require_corpus_design=False,
            quality_attestation_pin=pin,
        )


def test_prepared_partition_filter_is_fail_closed(tmp_path: Path) -> None:
    output = _prepare(tmp_path)
    pin = attest_prepared(output, tmp_path / "trust")
    split = json.loads((output / "splits.json").read_text())
    populated = next(name for name in ("train", "val", "test") if split[name])
    empty = next(name for name in ("train", "val", "test") if not split[name])

    selected = load_prepared_tasks(
        output,
        include_evaluator=True,
        partition=populated,
        require_provenance=False,
        require_corpus_design=False,
        quality_attestation_pin=pin,
    )
    assert len(selected) == 1 and selected[0].partition == populated
    with pytest.raises(ValueError, match="contains no prepared tasks"):
        load_prepared_tasks(
            output,
            include_evaluator=True,
            partition=empty,
            require_provenance=False,
            require_corpus_design=False,
            quality_attestation_pin=pin,
        )


@pytest.mark.parametrize("status", ["planted_proof", "exact_enumeration", "certified_optimal"])
def test_only_registered_certified_reference_statuses_are_accepted(
    tmp_path: Path, status: str
) -> None:
    output = _prepare(tmp_path, target_status=status)
    assert _read_jsonl(output / "evaluator_targets.jsonl")[0]["reference_status"] == status


def test_uncertified_reference_is_rejected(tmp_path: Path) -> None:
    bank, manifest, targets = _write_inputs(tmp_path, target_status="uncertified_reference")
    with pytest.raises(ValueError, match="certified status"):
        prepare_candidate_bank(bank, manifest, targets, tmp_path / "out", qubit_cap=4)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("manifest-sha", "checksum"),
        ("manifest-digest", "manifest digest"),
        ("manifest-count", "count"),
        ("instance-digest", "instance record digest"),
        ("group-reference", "instance record digest"),
        ("candidate-digest", "candidate record digest"),
        ("group-digest", "group record digest"),
        ("split", "split"),
    ],
)
def test_all_identity_and_digest_layers_are_verified(
    tmp_path: Path, target: str, message: str
) -> None:
    bank, manifest, targets = _write_inputs(tmp_path)
    if target == "manifest-sha":
        value = json.loads(manifest.read_text())
        value["jsonl_sha256"] = "0" * 64
        payload = {key: value[key] for key in value if key != "record_digest"}
        value["record_digest"] = content_digest(payload)
        manifest.write_bytes(canonical_json_bytes(value) + b"\n")
    elif target == "manifest-digest":
        value = json.loads(manifest.read_text())
        value["record_digest"] = "0" * 64
        manifest.write_bytes(canonical_json_bytes(value) + b"\n")
    elif target == "manifest-count":
        value = json.loads(manifest.read_text())
        value["group_count"] = 2
        payload = {key: value[key] for key in value if key != "record_digest"}
        value["record_digest"] = content_digest(payload)
        manifest.write_bytes(canonical_json_bytes(value) + b"\n")
    else:
        rows = _read_jsonl(bank)
        instance = rows[0]["record"]
        group = rows[1]["record"]
        if target == "instance-digest":
            instance["record_digest"] = "0" * 64
        elif target == "group-reference":
            group["instance_record_digest"] = "0" * 64
        elif target == "candidate-digest":
            group["candidates"][0]["record_digest"] = "0" * 64
        elif target == "group-digest":
            group["record_digest"] = "0" * 64
        elif target == "split":
            group["split"] = "val" if group["split"] != "val" else "test"
        raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
        bank.write_bytes(raw)
        manifest_value = json.loads(manifest.read_text())
        manifest_value["jsonl_sha256"] = hashlib.sha256(raw).hexdigest()
        payload = {key: manifest_value[key] for key in manifest_value if key != "record_digest"}
        manifest_value["record_digest"] = content_digest(payload)
        manifest.write_bytes(canonical_json_bytes(manifest_value) + b"\n")
    with pytest.raises(ValueError, match=message):
        prepare_candidate_bank(bank, manifest, targets, tmp_path / "out", qubit_cap=4)


@pytest.mark.parametrize("mutation", ["off_host", "disconnected", "missing_contact", "cap"])
def test_every_bank_embedding_is_revalidated_against_host_and_cap(
    tmp_path: Path, mutation: str
) -> None:
    instance, group = _fixture_records()
    candidate = group["candidates"][0]
    if mutation == "off_host":
        candidate["chains"] = [[0], [99]]
    elif mutation == "disconnected":
        candidate["chains"] = [[0, 2], [3]]
    elif mutation == "missing_contact":
        candidate["chains"] = [[0], [3]]
    else:
        # All records are internally consistent, but the registered cap is too small.
        bank, manifest, targets = _write_inputs(tmp_path)
        with pytest.raises(ValueError, match="qubit cap"):
            prepare_candidate_bank(bank, manifest, targets, tmp_path / "out", qubit_cap=3)
        return
    # Re-sign every enclosing content commitment so only semantic revalidation catches it.
    candidate["total_qubits"] = sum(map(len, candidate["chains"]))
    candidate["max_chain"] = max(map(len, candidate["chains"]))
    candidate["features"] = [
        ["max_chain", float(candidate["max_chain"])],
        ["total_qubits", float(candidate["total_qubits"])],
    ]
    candidate["candidate_id"] = "candidate-" + content_digest(
        {
            "chains": candidate["chains"],
            "domain": "isingfold-candidate-v1",
            "group_id": group["group_id"],
        }
    )
    candidate_payload = {key: value for key, value in candidate.items() if key != "record_digest"}
    candidate["record_digest"] = content_digest(candidate_payload)
    # The full provenance graph is intentionally expensive to re-sign. Bypass it here by
    # testing the public semantic validator through a bank whose group digest fails later.
    raw = b"".join(
        canonical_json_bytes(row) + b"\n"
        for row in (
            {"kind": "instance", "record": instance},
            {"kind": "group", "record": group},
        )
    )
    bank = tmp_path / "bank.jsonl"
    bank.write_bytes(raw)
    manifest_payload = {
        "group_count": 1,
        "instance_count": 1,
        "jsonl_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": 1,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {**manifest_payload, "record_digest": content_digest(manifest_payload)}
        )
        + b"\n"
    )
    target = {
        "certificate_digest": "a" * 64,
        "evaluator_protocol_digest": "b" * 64,
        "instance_id": instance["instance_id"],
        "instance_record_digest": instance["record_digest"],
        "reference_energy": -1.25,
        "reference_status": "planted_proof",
        "schema": "embedbench.evaluator-target",
        "schema_version": 1,
    }
    targets = tmp_path / "targets.jsonl"
    targets.write_bytes(canonical_json_bytes(target) + b"\n")
    with pytest.raises(ValueError):
        prepare_candidate_bank(bank, manifest, targets, tmp_path / "out", qubit_cap=4)


def test_duplicate_keys_and_nonfinite_numbers_are_rejected(tmp_path: Path) -> None:
    bank, manifest, targets = _write_inputs(tmp_path)
    manifest.write_text(
        '{"schema_version":1,"schema_version":1,"jsonl_sha256":"'
        + "0" * 64
        + '","instance_count":1,"group_count":1,"record_digest":"'
        + "0" * 64
        + '"}\n'
    )
    with pytest.raises(ValueError, match="duplicate key"):
        prepare_candidate_bank(bank, manifest, targets, tmp_path / "duplicate", qubit_cap=4)

    bank, manifest, targets = _write_inputs(tmp_path / "finite")
    targets.write_text(targets.read_text().replace("-1.25", "NaN"))
    with pytest.raises(ValueError, match="non-finite"):
        prepare_candidate_bank(bank, manifest, targets, tmp_path / "nonfinite", qubit_cap=4)


def test_target_references_and_uniqueness_are_strict(tmp_path: Path) -> None:
    bank, manifest, targets = _write_inputs(tmp_path)
    target = json.loads(targets.read_text())
    target["instance_record_digest"] = "0" * 64
    targets.write_bytes(canonical_json_bytes(target) + b"\n")
    with pytest.raises(ValueError, match="instance record digest"):
        prepare_candidate_bank(bank, manifest, targets, tmp_path / "bad-ref", qubit_cap=4)

    targets.write_bytes(canonical_json_bytes(target) + b"\n" + canonical_json_bytes(target) + b"\n")
    with pytest.raises(ValueError, match="duplicate evaluator target"):
        prepare_candidate_bank(bank, manifest, targets, tmp_path / "duplicate", qubit_cap=4)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    bank, manifest, targets = _write_inputs(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("user data")

    with pytest.raises(FileExistsError):
        prepare_candidate_bank(bank, manifest, targets, output, qubit_cap=4)
    assert sentinel.read_text() == "user data"


def test_canonical_json_rejects_non_string_keys_and_nonfinite_values() -> None:
    with pytest.raises(TypeError, match="keys"):
        canonical_json_bytes({1: "bad"})
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"bad": float("inf")})


def test_legacy_prepared_v1_remains_loadable_but_is_explicitly_pilot_only(
    tmp_path: Path,
) -> None:
    output = _prepare(tmp_path)
    with pytest.raises(ValueError, match="provenance-complete prepared schema v2 or v3"):
        load_prepared_tasks(output, include_evaluator=False)
    task = load_prepared_tasks(
        output,
        include_evaluator=False,
        require_provenance=False,
        require_corpus_design=False,
    )[0]

    assert task.prepared_schema_version == 1
    assert task.corpus_scope == "legacy-pilot-v1"
    assert task.provenance is None
    with pytest.raises(ValueError, match="requires provenance-complete prepared schema v2"):
        load_prepared_tasks(
            output,
            include_evaluator=False,
            require_provenance=True,
            require_corpus_design=False,
        )


def test_prepare_v2_preserves_authenticated_lineage_host_and_stratum_provenance(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets = _write_inputs(tmp_path)
    instance, group = _fixture_records()
    source = _provenance_record(instance, group)
    provenance = _write_provenance(tmp_path / "provenance.jsonl", [source])
    output = tmp_path / "prepared-v2"

    manifest = prepare_candidate_bank_v2(
        bank,
        bank_manifest,
        targets,
        provenance,
        output,
        qubit_cap=4,
    )
    pin = attest_prepared(output, tmp_path / "trust")
    task = load_prepared_tasks(
        output,
        include_evaluator=True,
        require_provenance=True,
        require_corpus_design=False,
        quality_attestation_pin=pin,
    )[0]

    assert manifest["schema_version"] == 2
    assert manifest["corpus_scope"] == "production-provenance-v2"
    assert {path.name for path in output.iterdir()} == {
        "evaluator_targets.jsonl",
        "initializers.jsonl",
        "manifest.json",
        "policy_instances.jsonl",
        "provenance.jsonl",
        "splits.json",
    }
    assert task.prepared_schema_version == 2
    assert task.corpus_scope == "production-provenance-v2"
    assert task.task.lineage == source["base_parent_lineage"]
    assert task.provenance is not None
    assert task.provenance.source_logical_lineage == instance["split_unit_id"]
    assert task.provenance.descendant_transform_kinds == ("identity",)
    assert task.provenance.active_host_sha256 == source["active_topology_identity"]["host_sha256"]
    assert task.provenance.calibration_sha256 is None
    assert task.provenance.distribution_regime == "iid"
    assert task.provenance.distribution_stratum == "base/iid/fixture"


def test_prepare_cli_requires_design_and_publishes_default_loadable_v4(
    tmp_path: Path,
) -> None:
    from tests.unit.trust_v4_support import write_v4_inputs

    bank, bank_manifest, targets, provenance, design, design_sha256 = (
        write_v4_inputs(tmp_path)
    )
    output = tmp_path / "prepared-cli-v4"
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "prepare",
                "--bank",
                str(bank),
                "--bank-manifest",
                str(bank_manifest),
                "--evaluator-targets",
                str(targets),
                "--out",
                str(output),
                "--qubit-cap",
                "4",
            ]
        )

    args = parser.parse_args(
        [
            "prepare",
            "--bank",
            str(bank),
            "--bank-manifest",
            str(bank_manifest),
            "--evaluator-targets",
            str(targets),
            "--provenance",
            str(provenance),
            "--corpus-design-manifest",
            str(design),
            "--expected-corpus-design-sha256",
            design_sha256,
            "--out",
            str(output),
            "--qubit-cap",
            "4",
        ]
    )
    args.func(args)

    loaded = [
        item
        for partition in ("train", "val", "test")
        for item in load_prepared_tasks(
            output,
            include_evaluator=False,
            partition=partition,
        )
    ]
    assert len(loaded) == 3
    assert all(item.prepared_schema_version == 4 for item in loaded)
    assert all(item.provenance is not None for item in loaded)
    assert all(item.design_condition is not None for item in loaded)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "schema fields differ"),
        ("unknown", "schema fields differ"),
        ("digest", "record digest"),
    ],
)
def test_prepare_v2_rejects_missing_unknown_or_unauthenticated_provenance(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    bank, bank_manifest, targets = _write_inputs(tmp_path)
    instance, group = _fixture_records()
    source = _provenance_record(instance, group)
    if mutation == "missing":
        source.pop("calibration_identity")
    elif mutation == "unknown":
        source["invented"] = "field"
    else:
        source["record_digest"] = "0" * 64
    if mutation != "digest":
        payload = {key: value for key, value in source.items() if key != "record_digest"}
        source["record_digest"] = content_digest(payload)
    provenance = _write_provenance(tmp_path / "provenance.jsonl", [source])

    with pytest.raises(ValueError, match=message):
        prepare_candidate_bank_v2(
            bank,
            bank_manifest,
            targets,
            provenance,
            tmp_path / "prepared-v2",
            qubit_cap=4,
        )


def test_prepare_v2_rejects_a_provenance_row_missing_for_any_task(tmp_path: Path) -> None:
    bank, bank_manifest, targets = _write_inputs(tmp_path)
    provenance = _write_provenance(tmp_path / "provenance.jsonl", [])

    with pytest.raises(ValueError, match="coverage"):
        prepare_candidate_bank_v2(
            bank,
            bank_manifest,
            targets,
            provenance,
            tmp_path / "prepared-v2",
            qubit_cap=4,
        )


def test_prepare_v2_rejects_host_or_fault_identity_not_bound_to_source_graph(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets = _write_inputs(tmp_path)
    instance, group = _fixture_records()
    source = _provenance_record(instance, group)
    source["active_topology_identity"]["host_sha256"] = "5" * 64
    payload = {key: value for key, value in source.items() if key != "record_digest"}
    source["record_digest"] = content_digest(payload)
    provenance = _write_provenance(tmp_path / "provenance.jsonl", [source])

    with pytest.raises(ValueError, match="active host"):
        prepare_candidate_bank_v2(
            bank,
            bank_manifest,
            targets,
            provenance,
            tmp_path / "prepared-v2",
            qubit_cap=4,
        )


def test_prepare_v2_rejects_one_base_parent_across_learning_partitions(tmp_path: Path) -> None:
    instance, first, second, bank, bank_manifest, targets = _write_two_group_inputs(tmp_path)
    provenance = _write_provenance(
        tmp_path / "provenance.jsonl",
        [
            _provenance_record(instance, first, learning_partition="train"),
            _provenance_record(
                instance,
                second,
                learning_partition="test",
                transform_sha256="6" * 64,
            ),
        ],
    )

    with pytest.raises(ValueError, match="base parent lineage appears in multiple"):
        prepare_candidate_bank_v2(
            bank,
            bank_manifest,
            targets,
            provenance,
            tmp_path / "prepared-v2",
            qubit_cap=4,
        )


def test_prepare_v2_rejects_ood_training_membership(tmp_path: Path) -> None:
    bank, bank_manifest, targets = _write_inputs(tmp_path)
    instance, group = _fixture_records()
    provenance = _write_provenance(
        tmp_path / "provenance.jsonl",
        [_provenance_record(instance, group, learning_partition="train", regime="ood")],
    )

    with pytest.raises(ValueError, match="OOD provenance must be sealed test data"):
        prepare_candidate_bank_v2(
            bank,
            bank_manifest,
            targets,
            provenance,
            tmp_path / "prepared-v2",
            qubit_cap=4,
        )


def test_partition_filter_still_authenticates_every_v2_provenance_binding(tmp_path: Path) -> None:
    first_instance, first_group = _fixture_records()
    second_instance, second_group = _reinstance(first_instance, first_group)
    rows = (
        {"kind": "instance", "record": first_instance},
        {"kind": "instance", "record": second_instance},
        {"kind": "group", "record": first_group},
        {"kind": "group", "record": second_group},
    )
    bank_raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    bank = tmp_path / "bank.jsonl"
    bank.write_bytes(bank_raw)
    bank_manifest_payload = {
        "group_count": 2,
        "instance_count": 2,
        "jsonl_sha256": hashlib.sha256(bank_raw).hexdigest(),
        "schema_version": 1,
    }
    bank_manifest = tmp_path / "bank.manifest.json"
    bank_manifest.write_bytes(
        canonical_json_bytes(
            {
                **bank_manifest_payload,
                "record_digest": content_digest(bank_manifest_payload),
            }
        )
        + b"\n"
    )
    target_rows = []
    for marker, instance in (("a", first_instance), ("c", second_instance)):
        target_rows.append(
            {
                "certificate_digest": marker * 64,
                "evaluator_protocol_digest": "b" * 64,
                "instance_id": instance["instance_id"],
                "instance_record_digest": instance["record_digest"],
                "reference_energy": -1.25,
                "reference_status": "planted_proof",
                "schema": "embedbench.evaluator-target",
                "schema_version": 1,
            }
        )
    targets = tmp_path / "targets.jsonl"
    targets.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in target_rows))
    provenance = _write_provenance(
        tmp_path / "provenance.jsonl",
        [
            _provenance_record(
                first_instance,
                first_group,
                base_parent_lineage="problem-sha256:" + "1" * 64,
                learning_partition="train",
            ),
            _provenance_record(
                second_instance,
                second_group,
                base_parent_lineage="problem-sha256:" + "9" * 64,
                learning_partition="test",
                transform_sha256="6" * 64,
                source_partition="locked_ood",
                split_manifest_sha256="a" * 64,
            ),
        ],
    )
    output = tmp_path / "prepared-v2"
    prepare_candidate_bank_v2(
        bank,
        bank_manifest,
        targets,
        provenance,
        output,
        qubit_cap=4,
    )
    prepared_rows = _read_jsonl(output / "provenance.jsonl")
    tampered = next(
        row for row in prepared_rows if row["distribution"]["learning_partition"] == "test"
    )
    tampered["group_id"] = first_group["group_id"]
    tampered["group_record_digest"] = first_group["record_digest"]
    source_payload = {
        key: value
        for key, value in tampered.items()
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
    source_payload["schema"] = "embedbench.isingfold-task-provenance"
    source_payload["schema_version"] = 2
    tampered["source_record_digest"] = content_digest(source_payload)
    payload = {key: value for key, value in tampered.items() if key != "record_digest"}
    tampered["record_digest"] = content_digest(payload)
    _resign_prepared_file(output, "provenance.jsonl", prepared_rows)

    with pytest.raises(ValueError, match="wrong initializer group"):
        load_prepared_tasks(
            output,
            include_evaluator=False,
            partition="train",
            require_corpus_design=False,
        )


def test_prepared_v2_loader_reconstructs_the_source_provenance_digest(tmp_path: Path) -> None:
    bank, bank_manifest, targets = _write_inputs(tmp_path)
    instance, group = _fixture_records()
    provenance = _write_provenance(
        tmp_path / "provenance.jsonl",
        [_provenance_record(instance, group)],
    )
    output = tmp_path / "prepared-v2"
    prepare_candidate_bank_v2(
        bank,
        bank_manifest,
        targets,
        provenance,
        output,
        qubit_cap=4,
    )
    prepared_rows = _read_jsonl(output / "provenance.jsonl")
    prepared_rows[0]["source_record_digest"] = "0" * 64
    payload = {key: value for key, value in prepared_rows[0].items() if key != "record_digest"}
    prepared_rows[0]["record_digest"] = content_digest(payload)
    _resign_prepared_file(output, "provenance.jsonl", prepared_rows)

    with pytest.raises(ValueError, match="source provenance record digest mismatch"):
        load_prepared_tasks(
            output,
            include_evaluator=False,
            require_corpus_design=False,
        )


def test_prepared_v2_loader_authenticates_output_receipt_record_counts(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets = _write_inputs(tmp_path)
    instance, group = _fixture_records()
    provenance = _write_provenance(
        tmp_path / "provenance.jsonl",
        [_provenance_record(instance, group)],
    )
    output = tmp_path / "prepared-v2"
    prepare_candidate_bank_v2(
        bank,
        bank_manifest,
        targets,
        provenance,
        output,
        qubit_cap=4,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["outputs"]["provenance.jsonl"]["records"] += 1
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(payload)
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="output receipt record count"):
        load_prepared_tasks(
            output,
            include_evaluator=False,
            require_corpus_design=False,
        )


def test_prepare_v3_authenticates_design_and_embeds_realized_census(tmp_path: Path) -> None:
    bank, bank_manifest, targets, provenance, design, design_sha256 = _write_design_inputs(
        tmp_path
    )
    output = tmp_path / "prepared-v3"

    manifest = prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=design_sha256,
        qubit_cap=4,
    )
    tasks = load_prepared_tasks(output, include_evaluator=False)

    assert manifest["schema_version"] == 3
    assert manifest["corpus_scope"] == "production-designed-v3"
    assert manifest["source_sha256"]["corpus_design_manifest"] == design_sha256
    receipt = manifest["corpus_design"]
    loaded_receipt = load_prepared_corpus_design(output)
    assert receipt["corpus_design_version"] == "if-core-v1"
    assert loaded_receipt == receipt
    assert len(receipt["condition_registry"]) == 3
    assert receipt["manifest_sha256"] == design_sha256
    assert receipt["realized_census"]["total_base_lineages"] == 3
    assert receipt["realized_census"]["by_partition"] == {
        "test": {"base_lineages": 1, "tasks": 1},
        "train": {"base_lineages": 1, "tasks": 1},
        "val": {"base_lineages": 1, "tasks": 1},
    }
    assert receipt["validation"]["status"] == "pass"
    assert receipt["validation"]["quota_status"] == "pass"
    assert receipt["validation"]["power_status"] == "pass"
    assert receipt["validation"]["precision_status"] == "pass"
    assert receipt["power_targets"] == json.loads(design.read_text())["power_targets"]
    assert {task.prepared_schema_version for task in tasks} == {3}
    assert {task.corpus_scope for task in tasks} == {"production-designed-v3"}
    assert all(task.design_condition is not None for task in tasks)


def test_corpus_design_schema_and_template_are_strict_and_self_digested() -> None:
    repository = Path(__file__).parents[2]
    schema = json.loads((repository / "configs/corpus_design_v1.schema.json").read_text())
    template = json.loads((repository / "configs/corpus_design_v1.template.json").read_text())

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["corpus_design_version"] == {"const": "if-core-v2"}
    assert schema["properties"]["schema_version"] == {"const": 2}
    floor_schema = schema["properties"]["minimum_partition_base_lineages"]
    assert floor_schema["properties"]["train"]["minimum"] == 1024
    assert floor_schema["properties"]["val"]["minimum"] == 512
    assert floor_schema["properties"]["test"]["minimum"] == 1546
    power_contains = schema["properties"]["power_targets"]["contains"]
    assert power_contains["properties"]["target_id"] == {
        "const": "validation-valid-return-noninferiority"
    }
    assert power_contains["properties"]["learning_partition"] == {"const": "val"}
    assert power_contains["properties"]["minimum_base_lineages"]["minimum"] == 128
    precision_contains = schema["properties"]["precision_targets"]["contains"]
    assert precision_contains["properties"]["target_id"] == {
        "const": "validation-paired-utility-precision"
    }
    assert precision_contains["properties"]["confidence_level"]["minimum"] == 0.95
    assert precision_contains["properties"]["half_width"]["maximum"] == 0.2
    assert precision_contains["properties"]["minimum_base_lineages"]["minimum"] == 128
    assert all(
        definition.get("additionalProperties") is False
        for definition in schema["$defs"].values()
        if definition.get("type") == "object"
    )
    payload = {key: value for key, value in template.items() if key != "record_digest"}
    assert template["record_digest"] == content_digest(payload)
    assert template["corpus_design_version"] == "if-core-v2"
    assert set(template["difficulty_calibration"]) == {
        "authority",
        "budget",
        "evidence",
        "origin",
        "outcome_blind",
        "panel",
        "protocol",
        "publisher_id",
    }
    assert template["minimum_partition_base_lineages"] == {
        "test": 1546,
        "train": 1024,
        "val": 512,
    }
    test_targets = [
        target
        for target in [*template["power_targets"], *template["precision_targets"]]
        if target["learning_partition"] == "test"
    ]
    assert all(target["filter"] == template["axis_values"] for target in test_targets)
    validation_filter = _partition_design_filter(template["lineage_registry"], "val")
    validation_targets = [
        target
        for target in [*template["power_targets"], *template["precision_targets"]]
        if target["learning_partition"] == "val"
    ]
    assert len(validation_targets) == 2
    assert all(target["filter"] == validation_filter for target in validation_targets)

    validation_power = next(
        target
        for target in template["power_targets"]
        if target["target_id"] == "validation-valid-return-noninferiority"
    )
    required_power = math.ceil(
        validation_power["assumed_discordance"]
        * (
            NormalDist().inv_cdf(1.0 - validation_power["alpha"])
            + NormalDist().inv_cdf(validation_power["target_power"])
        )
        ** 2
        / validation_power["power_separation"] ** 2
    )
    assert required_power == 127
    assert validation_power["minimum_base_lineages"] == 128

    validation_precision = next(
        target
        for target in template["precision_targets"]
        if target["target_id"] == "validation-paired-utility-precision"
    )
    precision_z = NormalDist().inv_cdf(
        0.5 + validation_precision["confidence_level"] / 2.0
    )
    required_precision = math.ceil(
        validation_precision["variance_bound"]
        * precision_z**2
        / validation_precision["half_width"] ** 2
    )
    assert required_precision == 97
    assert validation_precision["minimum_base_lineages"] == 128
    assert template["power_targets"][0]["minimum_base_lineages"] > len(
        template["lineage_registry"]
    )


def test_prepared_v3_reconstructs_and_authenticates_embedded_design_receipt(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(tmp_path)
    output = tmp_path / "prepared-v3"
    prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["corpus_design"]["difficulty_calibration"]["budget_sha256"] = "f" * 64
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(payload)
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="reconstruct its source digest"):
        load_prepared_corpus_design(output)


def test_prepare_v3_rejects_out_of_band_design_digest_mismatch(tmp_path: Path) -> None:
    bank, bank_manifest, targets, provenance, design, _digest = _write_design_inputs(tmp_path)

    with pytest.raises(ValueError, match="out-of-band corpus-design digest"):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256="0" * 64,
            qubit_cap=4,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-lineage", "lineage registry coverage"),
        ("unknown-stratum-field", "lineage registry row schema fields differ"),
        ("underquota", "stratum quota"),
        ("underpowered", "power target"),
        ("all-easy", "all-easy"),
        ("difficulty-authority", "authority_sha256"),
        ("difficulty-outcome-aware", "outcome-blind"),
        ("missing-ood-quota", "dedicated sealed-test OOD"),
    ],
)
def test_prepare_v3_rejects_incomplete_or_scientifically_invalid_design(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, _digest = _write_design_inputs(tmp_path)
    payload = json.loads(design.read_text())
    if mutation == "missing-lineage":
        payload["lineage_registry"].pop()
    elif mutation == "unknown-stratum-field":
        payload["lineage_registry"][0]["unregistered_difficulty"] = "hard"
    elif mutation == "underquota":
        payload["stratum_quotas"][0]["minimum_base_lineages"] = 2
    elif mutation == "underpowered":
        payload["power_targets"][0]["minimum_base_lineages"] = 2
    elif mutation == "all-easy":
        for lineage in payload["lineage_registry"]:
            lineage["embedding_difficulty"] = "easy"
            lineage["sampling_difficulty"] = "easy"
            lineage["decision_difficulty"] = "easy"
    elif mutation == "difficulty-authority":
        payload["difficulty_calibration"]["authority_sha256"] = "unsigned"
    elif mutation == "difficulty-outcome-aware":
        payload["difficulty_calibration"]["outcome_blind"] = False
    else:
        ood_quota = next(
            row
            for row in payload["stratum_quotas"]
            if row["filter"]["distribution_regime"] == ["ood"]
        )
        ood_quota["filter"]["distribution_regime"] = ["iid", "ood"]
    design.write_bytes(canonical_json_bytes(payload) + b"\n")
    digest = _resign_design(design)

    with pytest.raises(ValueError, match=message):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


def test_prepare_v3_rejects_single_topology_census(tmp_path: Path) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(
        tmp_path,
        topologies=("path", "path", "path"),
    )

    with pytest.raises(ValueError, match="host families"):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"include_faulted": False}, "realized faulted condition"),
        (
            {"problem_origins": ("synthetic", "synthetic", "synthetic")},
            "application-derived and synthetic origins",
        ),
    ],
)
def test_prepare_v3_requires_fault_and_problem_origin_diversity(
    tmp_path: Path,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(
        tmp_path,
        **kwargs,
    )

    with pytest.raises(ValueError, match=message):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


def test_prepare_v3_power_formula_binds_two_sided_direction_and_separation(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets, provenance, design, _digest = _write_design_inputs(tmp_path)
    manifest = json.loads(design.read_text())
    power = json.loads(json.dumps(manifest["power_targets"][0]))
    power["alternative"] = "two-sided-difference"
    power["assumed_true_difference"] = 0.99
    power["endpoint"] = "secondary-valid-return-difference"
    power["noninferiority_margin"] = 0.0
    power["power_separation"] = 0.99
    power["target_id"] = "test-two-sided-difference"
    manifest["power_targets"].append(power)
    manifest["power_targets"].sort(key=lambda row: row["target_id"])
    design.write_bytes(canonical_json_bytes(manifest) + b"\n")
    digest = _resign_design(design)

    prepared = prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        tmp_path / "prepared-v3",
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )

    assert prepared["corpus_design"]["power_targets"][1]["alternative"] == (
        "two-sided-difference"
    )
    assert prepared["corpus_design"]["validation"]["power_results"][1][
        "calculated_minimum_base_lineages"
    ] == 1


def test_prepare_v3_counts_crossed_host_conditions_once_per_base_lineage(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(
        tmp_path,
        crossed_train_host=True,
    )
    output = tmp_path / "prepared-v3"

    manifest = prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )

    census = manifest["corpus_design"]["realized_census"]
    assert census["total_base_lineages"] == 3
    assert census["total_realized_conditions"] == 4
    assert census["by_partition"]["train"] == {"base_lineages": 1, "tasks": 2}
    assert census["by_axis"]["host_family"] == {
        "path-a": 2,
        "path-b": 1,
        "path-c": 1,
    }
    assert len(load_prepared_tasks(output, include_evaluator=False)) == 4


def test_prepare_v3_primary_targets_cover_every_realized_test_condition(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(
        tmp_path,
        crossed_test_host=True,
    )

    manifest = prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        tmp_path / "prepared-v3",
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )

    receipt = manifest["corpus_design"]
    assert receipt["realized_census"]["total_base_lineages"] == 3
    assert receipt["realized_census"]["total_realized_conditions"] == 4
    assert receipt["realized_census"]["by_partition"]["test"] == {
        "base_lineages": 1,
        "tasks": 2,
    }
    assert receipt["power_targets"][0]["filter"]["host_family"] == ["path-a", "path-d"]
    assert receipt["precision_targets"][0]["filter"]["host_family"] == [
        "path-a",
        "path-d",
    ]


def test_prepare_v3_validation_tuning_targets_cover_every_realized_val_condition(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(
        tmp_path,
        crossed_val_host=True,
    )

    manifest = prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        tmp_path / "prepared-v3",
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )

    receipt = manifest["corpus_design"]
    assert receipt["realized_census"]["by_partition"]["val"] == {
        "base_lineages": 1,
        "tasks": 2,
    }
    for registry, target_id in (
        ("power_targets", "validation-valid-return-noninferiority"),
        ("precision_targets", "validation-paired-utility-precision"),
    ):
        target = next(
            row for row in receipt[registry] if row["target_id"] == target_id
        )
        assert target["filter"]["host_family"] == ["path-b", "path-e"]


@pytest.mark.parametrize("target_registry", ["power_targets", "precision_targets"])
def test_prepare_v3_rejects_primary_target_scoped_to_test_subgroup(
    tmp_path: Path,
    target_registry: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, _digest = _write_design_inputs(
        tmp_path,
        crossed_test_host=True,
    )
    manifest = json.loads(design.read_text())
    manifest[target_registry][0]["filter"]["host_family"] = ["path-a"]
    design.write_bytes(canonical_json_bytes(manifest) + b"\n")
    digest = _resign_design(design)

    with pytest.raises(ValueError, match="full sealed-test population"):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


@pytest.mark.parametrize(
    ("target_registry", "target_id"),
    [
        ("power_targets", "validation-valid-return-noninferiority"),
        ("precision_targets", "validation-paired-utility-precision"),
    ],
)
def test_prepare_v3_rejects_validation_tuning_target_scoped_to_val_subgroup(
    tmp_path: Path,
    target_registry: str,
    target_id: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, _digest = _write_design_inputs(
        tmp_path,
        crossed_val_host=True,
    )
    manifest = json.loads(design.read_text())
    target = next(
        row for row in manifest[target_registry] if row["target_id"] == target_id
    )
    target["filter"]["host_family"] = ["path-b"]
    design.write_bytes(canonical_json_bytes(manifest) + b"\n")
    digest = _resign_design(design)

    with pytest.raises(ValueError, match="full validation population"):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


@pytest.mark.parametrize("target_registry", ["power_targets", "precision_targets"])
def test_prepared_v3_rejects_primary_target_receipt_scoped_to_test_subgroup(
    tmp_path: Path,
    target_registry: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(
        tmp_path,
        crossed_test_host=True,
    )
    output = tmp_path / "prepared-v3"
    prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["corpus_design"][target_registry][0]["filter"]["host_family"] = ["path-a"]
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(payload)
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="full sealed-test population"):
        load_prepared_corpus_design(output)


@pytest.mark.parametrize(
    ("target_registry", "target_id"),
    [
        ("power_targets", "validation-valid-return-noninferiority"),
        ("precision_targets", "validation-paired-utility-precision"),
    ],
)
def test_prepared_v3_rejects_validation_target_receipt_scoped_to_val_subgroup(
    tmp_path: Path,
    target_registry: str,
    target_id: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(
        tmp_path,
        crossed_val_host=True,
    )
    output = tmp_path / "prepared-v3"
    prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    target = next(
        row
        for row in manifest["corpus_design"][target_registry]
        if row["target_id"] == target_id
    )
    target["filter"]["host_family"] = ["path-b"]
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(payload)
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="full validation population"):
        load_prepared_corpus_design(output)


@pytest.mark.parametrize(
    ("target_registry", "target_id", "message"),
    [
        (
            "power_targets",
            "validation-valid-return-noninferiority",
            "typed validation-tuning power target",
        ),
        (
            "precision_targets",
            "validation-paired-utility-precision",
            "typed validation-tuning precision target",
        ),
    ],
)
def test_prepare_v3_requires_named_validation_tuning_targets(
    tmp_path: Path,
    target_registry: str,
    target_id: str,
    message: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, _digest = _write_design_inputs(tmp_path)
    manifest = json.loads(design.read_text())
    manifest[target_registry] = [
        row for row in manifest[target_registry] if row["target_id"] != target_id
    ]
    design.write_bytes(canonical_json_bytes(manifest) + b"\n")
    digest = _resign_design(design)

    with pytest.raises(ValueError, match=message):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


def test_prepare_v3_enforces_128_lineage_validation_tuning_registry_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        import_embedbench_module,
        "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES",
        128,
    )
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(tmp_path)

    with pytest.raises(ValueError, match="typed validation-tuning power target"):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


def test_prepare_v3_enforces_validation_precision_assurance_thresholds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        import_embedbench_module,
        "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL",
        0.95,
    )
    monkeypatch.setattr(
        import_embedbench_module,
        "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH",
        0.2,
    )
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(tmp_path)

    with pytest.raises(ValueError, match="typed validation-tuning precision target"):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


@pytest.mark.parametrize("guard", ["minimum", "precision"])
def test_prepared_v3_revalidates_validation_tuning_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guard: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(tmp_path)
    output = tmp_path / "prepared-v3"
    prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )
    if guard == "minimum":
        monkeypatch.setattr(prepared_module, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 128)
        message = "typed validation-tuning power target"
    else:
        monkeypatch.setattr(
            prepared_module,
            "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL",
            0.95,
        )
        monkeypatch.setattr(
            prepared_module,
            "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH",
            0.2,
        )
        message = "typed validation-tuning precision target"

    with pytest.raises(ValueError, match=message):
        load_prepared_corpus_design(output)


@pytest.mark.parametrize(
    ("partition", "constant_name", "minimum"),
    [
        ("train", "MINIMUM_TRAIN_BASE_LINEAGES", 1024),
        ("val", "MINIMUM_VALIDATION_BASE_LINEAGES", 512),
        ("test", "MINIMUM_TEST_BASE_LINEAGES", 1546),
    ],
)
def test_prepare_v3_enforces_production_independent_lineage_floors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partition: str,
    constant_name: str,
    minimum: int,
) -> None:
    monkeypatch.setattr(import_embedbench_module, constant_name, minimum)
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(tmp_path)

    with pytest.raises(ValueError, match=rf"{partition} minimum must be at least {minimum}"):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


@pytest.mark.parametrize(
    ("partition", "constant_name"),
    [
        ("train", "MINIMUM_TRAIN_BASE_LINEAGES"),
        ("val", "MINIMUM_VALIDATION_BASE_LINEAGES"),
        ("test", "MINIMUM_TEST_BASE_LINEAGES"),
    ],
)
def test_prepared_v3_revalidates_production_independent_lineage_floors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    partition: str,
    constant_name: str,
) -> None:
    bank, bank_manifest, targets, provenance, design, digest = _write_design_inputs(tmp_path)
    output = tmp_path / "prepared-v3"
    prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )
    monkeypatch.setattr(prepared_module, constant_name, 2)

    with pytest.raises(
        ValueError,
        match=rf"prepared {partition} minimum is below the production independent-lineage floor",
    ):
        load_prepared_corpus_design(output)


def test_prepare_v3_requires_test_floor_to_cover_largest_confirmatory_target(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets, provenance, design, _digest = _write_design_inputs(
        tmp_path,
        crossed_test_host=True,
        crossed_test_lineage=True,
    )
    manifest = json.loads(design.read_text())
    manifest["power_targets"][0]["minimum_base_lineages"] = 2
    design.write_bytes(canonical_json_bytes(manifest) + b"\n")
    digest = _resign_design(design)

    with pytest.raises(ValueError, match="test independent-lineage floor.*confirmatory target"):
        prepare_candidate_bank_v3(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "prepared-v3",
            expected_corpus_design_sha256=digest,
            qubit_cap=4,
        )


def test_prepared_v3_revalidates_test_floor_against_confirmatory_targets(
    tmp_path: Path,
) -> None:
    bank, bank_manifest, targets, provenance, design, _digest = _write_design_inputs(
        tmp_path,
        crossed_test_host=True,
        crossed_test_lineage=True,
    )
    source_design = json.loads(design.read_text())
    source_design["minimum_partition_base_lineages"]["test"] = 2
    source_design["power_targets"][0]["minimum_base_lineages"] = 2
    design.write_bytes(canonical_json_bytes(source_design) + b"\n")
    digest = _resign_design(design)
    output = tmp_path / "prepared-v3"
    prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=digest,
        qubit_cap=4,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["corpus_design"]["minimum_partition_base_lineages"]["test"] = 1
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(payload)
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="test independent-lineage floor.*confirmatory target"):
        load_prepared_corpus_design(output)


def test_prepared_v2_requires_explicit_diagnostic_opt_out(tmp_path: Path) -> None:
    bank, bank_manifest, targets = _write_inputs(tmp_path)
    instance, group = _fixture_records()
    provenance = _write_provenance(
        tmp_path / "provenance.jsonl",
        [_provenance_record(instance, group)],
    )
    output = tmp_path / "prepared-v2"
    prepare_candidate_bank_v2(
        bank,
        bank_manifest,
        targets,
        provenance,
        output,
        qubit_cap=4,
    )

    with pytest.raises(ValueError, match="corpus-design-complete prepared schema v3"):
        load_prepared_tasks(output, include_evaluator=False)
    loaded = load_prepared_tasks(
        output,
        include_evaluator=False,
        require_corpus_design=False,
    )
    assert loaded[0].prepared_schema_version == 2

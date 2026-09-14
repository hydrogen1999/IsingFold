"""Trust-boundary tests for the offline IF-Q3-S0 selector-label dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import PreparedTask
from isingfold.rl.data.quality_attestation import QualityAttestationPin
from isingfold.rl.data.selector_labels import (
    SelectorAuditAuthorization,
    build_selector_labels,
    load_selector_metadata,
    load_selector_records,
)
from isingfold.rl.env import EmbeddingTask
from isingfold.rl.evaluator import ReadBlock


def _prepared_task(
    name: str,
    lineage: str,
    partition: str,
    *,
    partition_target_count: int | None = None,
) -> PreparedTask:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts({0: 0.25, 1: -0.5}, {(0, 1): -1.0})
    embedding = {0: frozenset((0, 1)), 1: frozenset((2, 3))}
    task = EmbeddingTask(
        name=name,
        logical=logical,
        host=host,
        problem=problem,
        ground_energy=-1.75,
        lineage=lineage,
        initial_embedding=embedding,
    )
    digest = hashlib.sha256(name.encode()).hexdigest()
    target_count = (
        {"train": 2, "val": 1, "test": 1}[partition]
        if partition_target_count is None
        else partition_target_count
    )
    partition_authority = _partition_authority(partition, target_count)
    return PreparedTask(
        task=task,
        task_id=name,
        instance_id=f"instance-{name}",
        partition=partition,
        initializer_record_digest=digest,
        public_instance_record_digest=hashlib.sha256(
            f"policy-instance-{name}".encode()
        ).hexdigest(),
        reference_status="certified_optimal",
        certificate_digest=hashlib.sha256(f"certificate-{name}".encode()).hexdigest(),
        evaluator_protocol_digest=hashlib.sha256(f"protocol-{name}".encode()).hexdigest(),
        prepared_schema_version=4,
        corpus_scope="production-designed-v4",
        quality_attestation_digest="a" * 64,
        quality_evidence_manifest_digest=partition_authority[
            "evidence_manifest_record_digest"
        ],
        quality_evidence_manifest_sha256=partition_authority[
            "evidence_manifest_sha256"
        ],
        quality_target_set_digest=partition_authority["target_set_digest"],
        quality_target_count=target_count,
    )


def _source_manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "qubit_cap": 4,
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": 4,
        "fixture": True,
    }
    record = {**payload, "record_digest": content_digest(payload)}
    path = root / "manifest.json"
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    return root


def _tasks() -> list[PreparedTask]:
    return [
        _prepared_task("task-train-a", "lineage-train-a", "train"),
        _prepared_task("task-train-b", "lineage-train-b", "train"),
        _prepared_task("task-val", "lineage-val", "val"),
        _prepared_task("task-test", "lineage-test", "test"),
    ]


def _record(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _target_access(name: str, count: int) -> dict[str, object]:
    return _record(
        {
            "partition": name,
            "target_count": count,
            "schema": "test.target-access",
            "schema_version": 1,
        }
    )


def _ground_partition_receipt(name: str, count: int) -> dict[str, object]:
    return _record(
        {
            "partition": name,
            "accepted_count": count,
            "schema": "test.ground-partition",
            "schema_version": 1,
        }
    )


def _global_authority() -> dict[str, object]:
    return _record(
        {
            "ground_root": {
                "receipt_sha256": "b" * 64,
                "record_digest": "c" * 64,
                "verifier_identity_digest": "d" * 64,
            },
            "publication_id": "publication-v4",
            "publisher_attestation_record_digest": "a" * 64,
            "publisher_id": "test-publisher",
            "schema": "isingfold.global-quality-authority",
            "schema_version": 1,
            "target_authority_record_digest": "e" * 64,
        }
    )


def _partition_authority(name: str, count: int) -> dict[str, object]:
    def digest(label: str) -> str:
        return hashlib.sha256(f"{name}:{label}".encode()).hexdigest()

    return _record(
        {
            "evidence_manifest_record_digest": digest("evidence-record"),
            "evidence_manifest_sha256": digest("evidence-file"),
            "ground_partition": {
                "accepted_count": count,
                "instance_set_digest": digest("instances"),
                "receipt_record_digest": _ground_partition_receipt(name, count)[
                    "record_digest"
                ],
                "receipt_sha256": digest("ground-file"),
            },
            "name": name,
            "schema": "isingfold.partition-quality-authority",
            "schema_version": 1,
            "target_access_record_digest": _target_access(name, count)["record_digest"],
            "target_count": count,
            "target_set_digest": digest("target-set"),
        }
    )


def _authority(*, audit_mode: bool = False, train_count: int = 2) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": _global_authority(),
        "training_partition": _partition_authority("train", train_count),
    }
    if audit_mode:
        payload["audit_partitions"] = {
            "val": _partition_authority("val", 1),
            "test": _partition_authority("test", 1),
        }
    return _record(payload)


def _pin(tmp_path: Path) -> QualityAttestationPin:
    path = tmp_path / "publisher-attestation.json"
    if not path.exists():
        path.write_text("{}\n")
    return QualityAttestationPin(
        path=path,
        expected_digest="a" * 64,
        expected_publisher_id="test-publisher",
    )


def _patch_prepared(
    monkeypatch: pytest.MonkeyPatch,
    prepared_tasks: list[PreparedTask],
) -> None:
    """Patch only the CLI access seam; the production selector builder has no loader."""

    import isingfold.rl.cli as cli

    def load_partition(corpus, *, partition, pin, role=None):
        del corpus, pin
        normalized = "val" if partition == "validation" else partition
        selected = tuple(item for item in prepared_tasks if item.partition == normalized)
        resolved_role = role or (
            "training_partition" if normalized == "train" else "evaluation_partition"
        )
        payload = {
            "schema": "isingfold.quality-authority-binding",
            "schema_version": 2,
            "global": _global_authority(),
            resolved_role: _partition_authority(normalized, len(selected)),
        }
        return (
            selected,
            _record(payload),
            _target_access(normalized, len(selected)),
            _ground_partition_receipt(normalized, len(selected)),
        )

    monkeypatch.setattr(cli, "_load_quality_partition", load_partition)
    monkeypatch.setattr(
        cli,
        "_quality_attestation_pin",
        lambda args: object(),
    )
    monkeypatch.setattr(
        cli,
        "_quality_authority_identity",
        lambda tasks, *, pin: load_partition(
            "unused",
            partition=next(iter(tasks)).partition,
            pin=pin,
        )[1],
    )


def _audit_authorization() -> SelectorAuditAuthorization:
    return SelectorAuditAuthorization(
        grid_manifest_sha256="a" * 64,
        rl_value_selection_receipt_sha256="b" * 64,
        rl_value_selection_record_digest="c" * 64,
        representation_selection_receipt_sha256="d" * 64,
        representation_selection_record_digest="e" * 64,
        selected_model_family="if-core",
        selected_method="ppo-warm-start",
    )


def _successful_evaluator(calls: list[dict[str, int]]):
    def evaluate(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del chains, problem, ground_energy
        calls.append(
            {
                "strength_index": program.strength_index,
                "reads": num_reads,
                "seed": seed,
                "sweeps": num_sweeps,
            }
        )
        return ReadBlock(
            hits=(program.strength_index + seed) % (num_reads + 1),
            reads=num_reads,
            broken_fraction=0.0,
            mean_residual=0.25,
            strength_index=program.strength_index,
        )

    return evaluate


def test_builder_uses_explicit_train_capability_and_four_fresh_program_blocks(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    evaluator_calls: list[dict[str, int]] = []
    output = tmp_path / "selector-labels"
    context = Context(qubit_cap=4, n_est_reads=8, num_sweeps=16)
    authority = _authority()

    manifest = build_selector_labels(
        source,
        output,
        prepared_tasks=_tasks()[:2],
        quality_authority=authority,
        context=context,
        sample_seed=91,
        calibration_fraction=0.5,
        evaluator=_successful_evaluator(evaluator_calls),
    )

    assert manifest["schema_version"] == 4
    assert manifest["quality_authority"] == authority
    fitted_records = sum(
        manifest["partitions"][name]["records"] for name in ("train", "calibration")
    )
    assert len(evaluator_calls) == fitted_records * 4
    assert fitted_records > 2  # not the former initializer-only corpus
    assert {call["strength_index"] for call in evaluator_calls} == {0, 1, 2, 3}
    assert {call["reads"] for call in evaluator_calls} == {8}
    assert {call["sweeps"] for call in evaluator_calls} == {16}
    assert len({call["seed"] for call in evaluator_calls}) == len(evaluator_calls)
    assert manifest["partitions"]["train"]["records"] >= 1
    assert manifest["partitions"]["calibration"]["records"] >= 1
    assert manifest["partitions"]["audit_val"]["records"] == 0
    assert manifest["partitions"]["audit_test"]["records"] == 0

    train = load_selector_records(output, partition="train")
    calibration = load_selector_records(output, partition="calibration")
    metadata = load_selector_metadata(output)
    assert len(train) == manifest["partitions"]["train"]["records"]
    assert len(calibration) == manifest["partitions"]["calibration"]["records"]
    assert sum(
        manifest["terminal_source_census"][str(slot)]["retained_records"]
        for slot in (1, 2, 3)
    ) > 0
    assert metadata.reads_per_strength == 8
    assert metadata.normalizer_digest == train[0].graph_inputs[0].normalizer_digest
    assert metadata.source_prepared_manifest_sha256 == hashlib.sha256(
        (source / "manifest.json").read_bytes()
    ).hexdigest()
    assert {record.lineage for record in train}.isdisjoint(
        record.lineage for record in calibration
    )
    for record in (*train, *calibration):
        assert record.graph_inputs is not None
        assert len(record.graph_inputs) == 4
        assert record.reads == (8, 8, 8, 8)
        assert all(graph.validate() is graph for graph in record.graph_inputs)
        assert all(np.asarray(graph.logical).dtype == np.dtype("float32") for graph in record.graph_inputs)
        assert all(
            np.asarray(graph.index_claims).dtype == np.dtype("int64")
            for graph in record.graph_inputs
        )


def test_published_model_inputs_have_an_exact_deployable_allowlist(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    output = tmp_path / "selector-labels"
    build_selector_labels(
        source,
        output,
        prepared_tasks=_tasks()[:2],
        quality_authority=_authority(),
        context=Context(qubit_cap=4, n_est_reads=8),
        sample_seed=7,
        calibration_fraction=0.5,
        evaluator=_successful_evaluator([]),
    )

    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    graph_keys = {
        "claims",
        "globals",
        "hardware",
        "hardware_edges",
        "index_claims",
        "index_hardware_edges",
        "index_logical_edges",
        "logical",
        "logical_edges",
        "normalizer_digest",
        "scale",
        "strength",
        "strength_index",
    }
    tensor_keys = {"data_base64", "dtype", "shape"}
    for row in rows:
        assert len(row["graph_inputs"]) == 4
        for graph in row["graph_inputs"]:
            assert set(graph) == graph_keys
            assert all(
                set(graph[name]) == tensor_keys
                for name in graph_keys
                if isinstance(graph[name], dict)
            )
            flattened_keys = set(graph)
            assert not flattened_keys.intersection(
                {"ground_energy", "witness", "evaluator_key", "hits", "reads", "sample_seed"}
            )


def test_terminal_program_mixture_is_frozen_and_outcome_blind(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")

    def evaluator_with_hits(hits: int):
        def evaluate(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
            del chains, problem, ground_energy, seed, num_sweeps
            return ReadBlock(
                hits,
                num_reads,
                0.0,
                0.0,
                program.strength_index,
            )

        return evaluate

    outputs = []
    for name, hits in (("zero", 0), ("perfect", 8)):
        output = tmp_path / name
        build_selector_labels(
            source,
            output,
            prepared_tasks=_tasks()[:2],
            quality_authority=_authority(),
            context=Context(qubit_cap=4, n_est_reads=8),
            sample_seed=37,
            calibration_fraction=0.5,
            evaluator=evaluator_with_hits(hits),
        )
        outputs.append(
            [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
        )

    zero, perfect = outputs
    assert len(zero) == len(perfect) > 2
    assert [row["program_id"] for row in zero] == [row["program_id"] for row in perfect]
    assert [row["terminal_source_receipt"] for row in zero] == [
        row["terminal_source_receipt"] for row in perfect
    ]
    assert all(row["terminal_source_receipt"]["outcome_access"] == "none" for row in zero)
    assert all(row["hits"] == [0, 0, 0, 0] for row in zero)
    assert all(row["hits"] == [8, 8, 8, 8] for row in perfect)
    assert {row["terminal_source_receipt"]["source_slot"] for row in zero} - {0}


def test_audit_partitions_require_two_explicit_opt_ins(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    output = tmp_path / "selector-labels"
    evaluator_calls: list[dict[str, int]] = []
    build_selector_labels(
        source,
        output,
        prepared_tasks=_tasks(),
        quality_authority=_authority(audit_mode=True),
        context=Context(qubit_cap=4, n_est_reads=8),
        sample_seed=11,
        calibration_fraction=0.5,
        audit_mode=True,
        audit_authorization=_audit_authorization(),
        evaluator=_successful_evaluator(evaluator_calls),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(evaluator_calls) == sum(
        receipt["records"] for receipt in manifest["partitions"].values()
    ) * 4

    with pytest.raises(PermissionError, match="audit_mode"):
        load_selector_records(output, partition="audit_test")
    with pytest.raises(PermissionError, match="audit_mode"):
        load_selector_records(output, partition="train")
    with pytest.raises(PermissionError, match="audit_mode"):
        load_selector_metadata(output)
    with pytest.raises(ValueError, match="explicit audit"):
        load_selector_records(output, partition="test", audit_mode=True)
    with pytest.raises(PermissionError, match="final RL-value freeze"):
        load_selector_records(output, partition="audit_test", audit_mode=True)
    audit = load_selector_records(
        output,
        partition="audit_test",
        audit_mode=True,
        audit_authorization=_audit_authorization(),
    )
    assert {record.lineage for record in audit} == {"lineage-test"}
    assert all(record.reads == (4096, 4096, 4096, 4096) for record in audit)


@pytest.mark.parametrize("failure", ["missing", "short", "exception"])
def test_missing_failed_or_incomplete_reads_abort_without_publishing(
    tmp_path: Path, failure: str
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    output = tmp_path / f"selector-labels-{failure}"

    def evaluator(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del chains, problem, ground_energy, seed, num_sweeps
        if failure == "exception":
            raise RuntimeError("sampler backend failed")
        if failure == "missing":
            return None
        return ReadBlock(
            hits=0,
            reads=num_reads - 1,
            broken_fraction=0.0,
            mean_residual=0.0,
            strength_index=program.strength_index,
        )

    with pytest.raises((RuntimeError, ValueError)):
        build_selector_labels(
            source,
            output,
            prepared_tasks=_tasks()[:2],
            quality_authority=_authority(),
            context=Context(qubit_cap=4, n_est_reads=8),
            sample_seed=3,
            calibration_fraction=0.5,
            evaluator=evaluator,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.selector-labels-*"))


def test_zero_hit_blocks_are_valid_labels_not_missing_data(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")

    def zero(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del chains, problem, ground_energy, seed, num_sweeps
        return ReadBlock(0, num_reads, 0.0, 0.0, program.strength_index)

    output = tmp_path / "selector-labels"
    build_selector_labels(
        source,
        output,
        prepared_tasks=_tasks()[:2],
        quality_authority=_authority(),
        context=Context(qubit_cap=4, n_est_reads=8),
        sample_seed=4,
        calibration_fraction=0.5,
        evaluator=zero,
    )
    records = (
        *load_selector_records(output, partition="train"),
        *load_selector_records(output, partition="calibration"),
    )
    assert all(record.hits == (0, 0, 0, 0) for record in records)


def test_checksums_record_digests_and_output_immutability_fail_closed(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    output = tmp_path / "selector-labels"
    kwargs = {
        "prepared_tasks": _tasks()[:2],
        "quality_authority": _authority(),
        "context": Context(qubit_cap=4, n_est_reads=8),
        "sample_seed": 5,
        "calibration_fraction": 0.5,
        "evaluator": _successful_evaluator([]),
    }
    build_selector_labels(source, output, **kwargs)
    with pytest.raises(FileExistsError):
        build_selector_labels(source, output, **kwargs)

    records = output / "records.jsonl"
    records.write_bytes(records.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        load_selector_records(output, partition="train")


def test_split_is_deterministic_and_rejects_lineage_leakage(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    tasks = _tasks()
    context = Context(qubit_cap=4, n_est_reads=8)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_selector_labels(
        source,
        first,
        prepared_tasks=tasks[:2],
        quality_authority=_authority(),
        context=context,
        sample_seed=17,
        split_seed=23,
        calibration_fraction=0.5,
        evaluator=_successful_evaluator([]),
    )
    build_selector_labels(
        source,
        second,
        prepared_tasks=tasks[:2],
        quality_authority=_authority(),
        context=context,
        sample_seed=17,
        split_seed=23,
        calibration_fraction=0.5,
        evaluator=_successful_evaluator([]),
    )
    assert (first / "records.jsonl").read_bytes() == (second / "records.jsonl").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()

    leaked = [
        _prepared_task("train", "same-lineage", "train"),
        _prepared_task("validation", "same-lineage", "val"),
        _prepared_task("other", "other-lineage", "train"),
        _prepared_task("test", "test-lineage", "test"),
    ]
    with pytest.raises(ValueError, match="multiple prepared partitions"):
        build_selector_labels(
            source,
            tmp_path / "leaked",
            prepared_tasks=leaked,
            quality_authority=_authority(audit_mode=True),
            context=context,
            sample_seed=1,
            audit_mode=True,
            audit_authorization=_audit_authorization(),
            evaluator=_successful_evaluator([]),
        )


def test_calibration_requires_a_distinct_training_lineage(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")

    with pytest.raises(ValueError, match="at least two"):
        build_selector_labels(
            source,
            tmp_path / "selector-labels",
            prepared_tasks=[
                _prepared_task(
                    "only",
                    "only-lineage",
                    "train",
                    partition_target_count=1,
                )
            ],
            quality_authority=_authority(train_count=1),
            context=Context(qubit_cap=4, n_est_reads=8),
            sample_seed=0,
            evaluator=_successful_evaluator([]),
        )


def test_ordinary_builder_rejects_held_out_tasks_and_audit_authorities(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    with pytest.raises(PermissionError, match="train partition only"):
        build_selector_labels(
            source,
            tmp_path / "held-out",
            prepared_tasks=_tasks(),
            quality_authority=_authority(audit_mode=True),
            context=Context(qubit_cap=4, n_est_reads=8),
            sample_seed=0,
            evaluator=_successful_evaluator([]),
        )
    with pytest.raises(ValueError, match="unknown=.*audit_partitions"):
        build_selector_labels(
            source,
            tmp_path / "audit-authority",
            prepared_tasks=_tasks()[:2],
            quality_authority=_authority(audit_mode=True),
            context=Context(qubit_cap=4, n_est_reads=8),
            sample_seed=0,
            evaluator=_successful_evaluator([]),
        )


def test_audit_builder_requires_explicit_val_and_test_authorities(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    with pytest.raises(ValueError, match="missing=.*audit_partitions"):
        build_selector_labels(
            source,
            tmp_path / "missing-audit-authority",
            prepared_tasks=_tasks(),
            quality_authority=_authority(),
            context=Context(qubit_cap=4, n_est_reads=8),
            sample_seed=0,
            audit_mode=True,
            audit_authorization=_audit_authorization(),
            evaluator=_successful_evaluator([]),
        )


def test_quality_authority_nested_records_are_independently_authenticated(
    tmp_path: Path,
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    authority = json.loads(json.dumps(_authority()))
    authority["global"]["publisher_id"] = "forged-publisher"
    outer = {key: value for key, value in authority.items() if key != "record_digest"}
    authority["record_digest"] = content_digest(outer)

    with pytest.raises(ValueError, match="global quality authority record digest mismatch"):
        build_selector_labels(
            source,
            tmp_path / "forged-authority",
            prepared_tasks=_tasks()[:2],
            quality_authority=authority,
            context=Context(qubit_cap=4, n_est_reads=8),
            sample_seed=0,
            evaluator=_successful_evaluator([]),
        )


def test_selector_builder_rejects_prepared_manifest_before_v4(tmp_path: Path) -> None:
    source = _source_manifest(tmp_path / "prepared")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    payload["schema_version"] = 3
    manifest_path.write_bytes(canonical_json_bytes(_record(payload)) + b"\n")

    with pytest.raises(ValueError, match="CandidateBank v4"):
        build_selector_labels(
            source,
            tmp_path / "labels",
            prepared_tasks=_tasks()[:2],
            quality_authority=_authority(),
            context=Context(qubit_cap=4, n_est_reads=8),
            sample_seed=0,
            evaluator=_successful_evaluator([]),
        )

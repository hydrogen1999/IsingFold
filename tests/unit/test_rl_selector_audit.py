"""Post-freeze IF-Q3-S0 selector audit contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from isingfold.rl import cli  # noqa: E402
from isingfold.rl.cli import SelectorBundle, build_parser  # noqa: E402
from isingfold.rl.checkpoint import runtime_implementation_registry  # noqa: E402
from isingfold.rl.data.import_embedbench import (  # noqa: E402
    canonical_json_bytes,
    content_digest,
)
from isingfold.rl.selector_audit import (  # noqa: E402
    audit_selector_records,
    publish_selector_audit,
)
from isingfold.rl.strength import StrengthRecord, fit_selector  # noqa: E402
from isingfold.rl.contracts import Context  # noqa: E402
from isingfold.rl.data.selector_labels import build_selector_labels  # noqa: E402
from isingfold.rl.evaluate import EVALUATION_RECEIPT_VERSION  # noqa: E402
from isingfold.rl.experiment_selection import FrozenRLValueSelection  # noqa: E402
from tests.unit.test_rl_strength_gnn import _graph, _legacy_features  # noqa: E402
from tests.unit.test_rl_selector_labels import (  # noqa: E402
    _audit_authorization,
    _authority,
    _ground_partition_receipt,
    _patch_prepared,
    _source_manifest,
    _successful_evaluator,
    _target_access,
    _tasks,
)

ROOT = Path(__file__).resolve().parents[2]


class _FrozenSelector:
    frozen = True
    deployment_ready = True

    def __call__(self, graphs):
        del graphs
        return torch.tensor((0.1, 0.8, 0.2, 0.3), dtype=torch.float32)

    def select_embedding(self, graphs):
        return int(torch.argmax(self(graphs)).item())


def _record(
    task_id: str,
    lineage: str,
    hits: tuple[int, int, int, int],
    *,
    partition: str = "audit_val",
) -> StrengthRecord:
    graphs = tuple(_graph(index) for index in range(4))
    return StrengthRecord(
        features=tuple(_legacy_features(graph.strength) for graph in graphs),
        hits=hits,
        reads=(10, 10, 10, 10),
        lineage=lineage,
        graph_inputs=graphs,
        task_id=task_id,
        source_record_id=f"selector-record-{task_id}",
        selector_partition=partition,
    )


def _selection_receipt(
    path: Path,
    *,
    selector_digest: str = "d" * 64,
    corpus_manifest_sha256: str = "e" * 64,
) -> None:
    grid = json.loads((ROOT / "configs" / "rl_grid_hybrid_v1.json").read_text())
    runtime_identity = {
        "inference_device_name": "fixture-cpu",
        "inference_threads": 1,
        "runtime_platform": {
            "system": "fixture",
            "release": "fixture",
            "machine": "fixture",
            "processor": "fixture",
            "logical_cpu_count": 8,
            "slurm_partition": None,
        },
    }
    runtime_registry = runtime_implementation_registry()
    runtime_digest = content_digest(runtime_registry)
    payload = {
        "schema": "isingfold.representation-selection",
        "schema_version": 3,
        "grid_manifest_sha256": hashlib.sha256(
            (ROOT / "configs" / "rl_grid_hybrid_v1.json").read_bytes()
        ).hexdigest(),
        "stage": "representation",
        "partition": "validation",
        "selection_rule": cli.REPRESENTATION_SELECTION_RULE,
        "feasibility_constraint": {
            "reference": grid["representation_evaluation"]["feasibility_reference"],
            "margin": grid["representation_evaluation"]["margin"],
            "one_sided_alpha": grid["representation_evaluation"]["feasibility_alpha"],
            "cluster_unit": "immutable-base-lineage",
            "training_seed_treatment": "equal-weight-crossed-resampling-with-replacement",
            "lineage_treatment": "equal-weight-crossed-resampling-with-replacement",
            "bootstrap_replicates": grid["representation_evaluation"][
                "feasibility_bootstrap"
            ],
            "bootstrap_seed": grid["representation_evaluation"][
                "selection_bootstrap_seed"
            ],
            "pass_rule": "lower_bound_strictly_greater_than_negative_margin",
            "eligible_simpler": ["if-mlp", "if-dual"],
            "if_core_pass": True,
        },
        "seed_selection_forbidden": True,
        "selected_simpler": "if-dual",
        "retained_families": ["if-dual", "if-core"],
        "gate_profile": "profile-i",
        "gate_receipt_sha256": "a" * 64,
        "gate_record_digest": "b" * 64,
        "quality_preflight_receipt_sha256": "6" * 64,
        "quality_preflight_record_digest": "7" * 64,
        "runtime_implementation_registry": runtime_registry,
        "runtime_implementation_digest": runtime_digest,
        "family_aggregates": {
            family: {
                "seed_count": 3,
                "seeds": [1103, 2207, 3301],
                "independent_lineages": 3,
                "valid_return_rate": 1.0,
                "utility_mean": {
                    "if-mlp": 0.4,
                    "if-dual": 0.5,
                    "if-core": 0.7,
                }[family],
                "online_seconds_mean": {
                    "if-mlp": 1.0,
                    "if-dual": 1.5,
                    "if-core": 2.0,
                }[family],
                "feasibility_delta_vs_return_initial": 0.0,
                "feasibility_ci_low": 0.0,
                "feasibility_ci_high": 0.0,
                "feasibility_noninferior": True,
                "per_seed": [],
            }
            for family in ("if-mlp", "if-dual", "if-core")
        },
        "protocol_identity": {
            "population": 3,
            "repetitions": grid["representation_evaluation"]["repetitions"],
            "selector_digest": selector_digest,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "deployment_rule": "categorical-temperature-one",
            "gate_profile": "profile-i",
            "gate_receipt_sha256": "a" * 64,
            "gate_record_digest": "b" * 64,
            "matched_metadata": {
                "population": f"{corpus_manifest_sha256}:validation",
                "quality_authority": {
                    "publisher_id": "fixture-publisher",
                    "record_digest": "9" * 64,
                },
                "budget": "c" * 64,
                "selector": selector_digest,
                "proposal": "proposal-fixture",
                "support": "same-conditional-generator-and-exact-mask-contract",
                "policy_seed": grid["representation_evaluation"]["evaluation_seed"],
                "evaluator_reads": grid["representation_evaluation"]["audit_reads"],
                "initializer": "authenticated-profile-i",
                "repetitions": grid["representation_evaluation"]["repetitions"],
                "inference_device_type": "cpu",
                "baseline_controller_registry": "d" * 64,
                "evaluation_receipt_schema": EVALUATION_RECEIPT_VERSION,
                "evaluation_implementation_sha256": "8" * 64,
                "quality_preflight_receipt_sha256": "6" * 64,
                "quality_preflight_record_digest": "7" * 64,
                "runtime_implementation_registry": runtime_registry,
                "runtime_implementation_digest": runtime_digest,
            },
        },
        "runtime_identity_by_training_seed": {
            str(seed): runtime_identity for seed in (1103, 2207, 3301)
        },
        "source_reports": [
            {
                "cell_id": cell["cell_id"],
                "model_family": cell["model_family"],
                "seed": cell["seed"],
                "runtime_identity": runtime_identity,
                "runtime_implementation_registry": runtime_registry,
                "runtime_implementation_digest": runtime_digest,
            }
            for cell in grid["stages"]["representation"]
        ],
    }
    path.write_bytes(canonical_json_bytes({**payload, "record_digest": content_digest(payload)}) + b"\n")


def _final_freeze_fixture(
    path: Path,
    *,
    selector_digest: str = "d" * 64,
    corpus_manifest_sha256: str = "e" * 64,
) -> FrozenRLValueSelection:
    protocol = {
        "selector_digest": selector_digest,
        "corpus_manifest_sha256": corpus_manifest_sha256,
    }
    payload = {
        "schema": "fixture.final-rl-value-freeze",
        "matched_validation_protocol": protocol,
    }
    record = {**payload, "record_digest": content_digest(payload)}
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    return FrozenRLValueSelection(
        receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        record_digest=record["record_digest"],
        grid_manifest_sha256=hashlib.sha256(
            (ROOT / "configs" / "rl_grid_hybrid_v1.json").read_bytes()
        ).hexdigest(),
        model_family="if-core",
        grid_model_family="if-core",
        method="ppo-warm-start",
        training_seeds=(1103, 2207, 3301),
        cell_ids=("cell-a", "cell-b", "cell-c"),
        checkpoint_payload_digests=("1" * 64, "2" * 64, "3" * 64),
        representation_selection_sha256="4" * 64,
        representation_selection_record_digest="5" * 64,
        runtime_implementation_registry=runtime_implementation_registry(),
        runtime_implementation_digest=content_digest(runtime_implementation_registry()),
        quality_preflight_receipt_sha256="6" * 64,
        quality_preflight_record_digest="7" * 64,
    )


def test_post_freeze_audit_reports_raw_selected_f2_random_and_oracle_diagnostics() -> None:
    rows, aggregate = audit_selector_records(
        _FrozenSelector(),
        (
            _record("task-a", "lineage-a", (1, 5, 9, 3)),
            _record("task-b", "lineage-b", (2, 8, 4, 0)),
        ),
        partition="audit_val",
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["task_id"] == "task-a"
    assert first["selected_strength_index"] == 1
    assert first["fixed_f2_strength_index"] == 1
    assert first["uniform_random_expected_rate"] == pytest.approx(0.45)
    assert first["empirical_oracle_rate"] == 0.9
    assert first["selected_regret"] == pytest.approx(0.4)
    assert first["top1"] is False
    assert first["fixed_f2_top1"] is False
    assert first["uniform_random_expected_top1_probability"] == 0.25
    assert len(first["record_digest"]) == 64
    assert aggregate["records"] == 2
    assert aggregate["lineages"] == 2
    assert aggregate["selected_mean"] == pytest.approx(0.65)
    assert aggregate["fixed_f2_mean"] == pytest.approx(0.65)
    assert aggregate["uniform_random_expected_mean"] == pytest.approx(0.4)
    assert aggregate["empirical_oracle_mean"] == pytest.approx(0.85)
    assert aggregate["top1_count"] == 1
    assert aggregate["top1_rate"] == 0.5
    assert aggregate["fixed_f2_top1_rate"] == 0.5
    assert aggregate["uniform_random_expected_top1_rate"] == 0.25


def test_audit_is_fail_closed_for_non_graph_unfrozen_or_duplicate_records() -> None:
    record = _record("task-a", "lineage-a", (1, 2, 3, 4))
    with pytest.raises(ValueError, match="frozen deployment-ready"):
        audit_selector_records(object(), (record,), partition="audit_val")
    with pytest.raises(ValueError, match="graph-complete"):
        audit_selector_records(
            _FrozenSelector(),
            (
                StrengthRecord(
                    record.features,
                    record.hits,
                    record.reads,
                    lineage="x",
                    selector_partition="audit_val",
                ),
            ),
            partition="audit_val",
        )
    with pytest.raises(ValueError, match="duplicate task"):
        audit_selector_records(
            _FrozenSelector(), (record, record), partition="audit_val"
        )
    with pytest.raises(ValueError, match="audit partition"):
        audit_selector_records(_FrozenSelector(), (record,), partition="train")


def test_selector_fitting_and_calibration_refuse_partitioned_audit_rows() -> None:
    audit = _record("task-a", "lineage-a", (1, 2, 3, 4))
    train = _record(
        "task-train",
        "lineage-train",
        (1, 2, 3, 4),
        partition="train",
    )

    with pytest.raises(ValueError, match="fitting refuses"):
        fit_selector((audit,), epochs=1)
    with pytest.raises(ValueError, match="calibration refuses"):
        fit_selector((train,), calibration_records=(audit,), epochs=1)


def test_atomic_publication_binds_raw_rows_and_refuses_overwrite(tmp_path: Path) -> None:
    rows, aggregate = audit_selector_records(
        _FrozenSelector(),
        (_record("task-a", "lineage-a", (1, 5, 9, 3)),),
        partition="audit_val",
    )
    destination = tmp_path / "selector-audit"
    receipt = publish_selector_audit(
        destination,
        rows=rows,
        aggregate=aggregate,
        provenance={"selector_digest": "a" * 64},
        access_control={
            "rl_value_selection_required": False,
            "grid_manifest_sha256": None,
            "rl_value_selection_receipt_sha256": None,
            "rl_value_selection_record_digest": None,
            "representation_selection_receipt_sha256": None,
            "representation_selection_record_digest": None,
            "selected_model_family": None,
            "selected_method": None,
        },
    )

    assert set(item.name for item in destination.iterdir()) == {"rows.jsonl", "receipt.json"}
    assert receipt == json.loads((destination / "receipt.json").read_text())
    assert receipt["raw_rows"]["count"] == 1
    assert receipt["raw_rows"]["sha256"] == hashlib.sha256(
        (destination / "rows.jsonl").read_bytes()
    ).hexdigest()
    assert receipt["aggregate"] == aggregate
    unsigned = {key: value for key, value in receipt.items() if key != "record_digest"}
    assert receipt["record_digest"] == content_digest(unsigned)
    with pytest.raises(FileExistsError):
        publish_selector_audit(
            destination,
            rows=rows,
            aggregate=aggregate,
            provenance={"selector_digest": "a" * 64},
            access_control=receipt["access_control"],
        )


def test_cli_requires_final_rl_value_freeze_before_audit_test(tmp_path: Path) -> None:
    parser = build_parser()
    common = [
        "audit-selector",
        "--corpus",
        "prepared",
        "--selector",
        "selector",
        "--selector-labels",
        "labels",
        "--out",
        "audit",
    ]
    validation = parser.parse_args([*common, "--partition", "audit_val"])
    assert validation.rl_value_selection_receipt is None

    missing = parser.parse_args([*common, "--partition", "audit_test"])
    with pytest.raises(ValueError, match="rl-value-selection-receipt"):
        missing.func(missing)

    # The earlier representation decision is not sufficient to unseal test outcomes.
    receipt = tmp_path / "selection.json"
    _selection_receipt(receipt)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *common,
                "--partition",
                "audit_test",
                "--grid",
                str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
                "--representation-selection-receipt",
                str(receipt),
            ]
        )

    authorized = parser.parse_args(
        [
            *common,
            "--partition",
            "audit_test",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--rl-value-selection-receipt",
            str(receipt),
            "--expected-rl-value-selection-sha256",
            "a" * 64,
        ]
    )
    # Parsing records happens only after this independently testable receipt check.
    assert authorized.partition == "audit_test"
    assert authorized.rl_value_selection_receipt == str(receipt)


def test_production_audit_label_generation_is_also_sealed_until_freeze() -> None:
    args = build_parser().parse_args(
        [
            "label-selector-data",
            "--corpus",
            "prepared",
            "--quality-attestation",
            "publisher-attestation.json",
            "--expected-quality-attestation-digest",
            "a" * 64,
            "--expected-quality-publisher-id",
            "test-publisher",
            "--ground-certificate-root",
            "ground-certificate-root.json",
            "--expected-ground-certificate-root-sha256",
            "b" * 64,
            "--out",
            "labels",
            "--audit-mode",
        ]
    )
    with pytest.raises(ValueError, match="RL-value-selection receipt"):
        args.func(args)


def test_direct_audit_label_builder_also_requires_freeze_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    _patch_prepared(monkeypatch, _tasks())
    output = tmp_path / "labels"

    with pytest.raises(PermissionError, match="RL-value freeze"):
        build_selector_labels(
            source,
            output,
            prepared_tasks=_tasks(),
            quality_authority=_authority(audit_mode=True),
            context=Context(qubit_cap=4),
            sample_seed=5,
            audit_mode=True,
            evaluator=_successful_evaluator([]),
        )
    assert not output.exists()


def test_audit_label_normalizer_mismatch_fails_before_sampling_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    _patch_prepared(monkeypatch, _tasks())
    calls: list[dict[str, int]] = []
    output = tmp_path / "audit-labels"

    with pytest.raises(ValueError, match="different frozen normalizer"):
        build_selector_labels(
            source,
            output,
            prepared_tasks=_tasks(),
            quality_authority=_authority(audit_mode=True),
            context=Context(qubit_cap=4),
            sample_seed=5,
            split_seed=7,
            calibration_fraction=0.5,
            audit_mode=True,
            audit_authorization=_audit_authorization(),
            expected_normalizer_digest="a" * 64,
            evaluator=_successful_evaluator(calls),
        )

    assert calls == []
    assert not output.exists()


def test_tampered_final_freeze_is_rejected_before_label_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "selection.json"
    frozen = _final_freeze_fixture(receipt)
    value = json.loads(receipt.read_text())
    value["matched_validation_protocol"]["selector_digest"] = "f" * 64
    unsigned = {key: item for key, item in value.items() if key != "record_digest"}
    value["record_digest"] = content_digest(unsigned)
    receipt.write_text(json.dumps(value))
    touched: list[bool] = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        touched.append(True)
        raise AssertionError("selector labels opened before representation freeze")

    monkeypatch.setattr(
        "isingfold.rl.data.selector_labels.load_selector_metadata", forbidden
    )
    args = build_parser().parse_args(
        [
            "audit-selector",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            "--selector-labels",
            "labels",
            "--partition",
            "audit_test",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--rl-value-selection-receipt",
            str(receipt),
            "--expected-rl-value-selection-sha256",
            frozen.receipt_sha256,
            "--out",
            "audit",
        ]
    )
    with pytest.raises(ValueError, match="externally pinned"):
        args.func(args)
    assert touched == []


def test_unrelated_final_freeze_cannot_unseal_audit_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "selection.json"
    frozen = _final_freeze_fixture(receipt, selector_digest="c" * 64)
    monkeypatch.setattr(
        "isingfold.rl.experiment_selection.load_rl_value_freeze",
        lambda **kwargs: frozen,
    )
    bundle = SelectorBundle(
        model=None,
        selector_digest="d" * 64,
        normalizer={},
        normalizer_digest="f" * 64,
        coefficient_scale=1.0,
        corpus_manifest_sha256="e" * 64,
        quality_authority=_authority(),
        target_access=_target_access("train", 2),
        ground_partition_receipt=_ground_partition_receipt("train", 2),
        root=tmp_path / "selector",
    )
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *args, **kwargs: bundle)
    opened: list[bool] = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        opened.append(True)
        raise AssertionError("audit labels opened before freeze cross-binding")

    monkeypatch.setattr(
        "isingfold.rl.data.selector_labels.load_selector_metadata", forbidden
    )
    args = build_parser().parse_args(
        [
            "audit-selector",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            "--selector-labels",
            "labels",
            "--partition",
            "audit_test",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--rl-value-selection-receipt",
            str(receipt),
            "--expected-rl-value-selection-sha256",
            frozen.receipt_sha256,
            "--out",
            str(tmp_path / "audit"),
        ]
    )
    with pytest.raises(ValueError, match="different frozen selector"):
        args.func(args)
    assert opened == []


def test_cli_audits_complete_authenticated_partitions_and_binds_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    _patch_prepared(monkeypatch, _tasks())
    fit_labels = tmp_path / "fit-labels"
    audit_labels = tmp_path / "audit-labels"
    common = {
        "prepared_tasks": _tasks()[:2],
        "quality_authority": _authority(),
        "context": Context(qubit_cap=4),
        "sample_seed": 73,
        "split_seed": 19,
        "calibration_fraction": 0.5,
        "evaluator": _successful_evaluator([]),
    }
    build_selector_labels(source, fit_labels, **common)

    bundle = tmp_path / "selector"
    fit = build_parser().parse_args(
        [
            "fit-selector",
            "--corpus",
            str(source),
            "--selector-labels",
            str(fit_labels),
            "--quality-attestation",
            str(tmp_path / "publisher-attestation.json"),
            "--expected-quality-attestation-digest",
            "a" * 64,
            "--expected-quality-publisher-id",
            "test-publisher",
            "--ground-certificate-root",
            str(tmp_path / "ground-certificate-root.json"),
            "--expected-ground-certificate-root-sha256",
            "b" * 64,
            "--out",
            str(bundle),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )
    fit.func(fit)

    selection = tmp_path / "selection.json"
    fit_receipt = json.loads((bundle / "fit_receipt.json").read_text())
    frozen = _final_freeze_fixture(
        selection,
        selector_digest=fit_receipt["selector_digest"],
        corpus_manifest_sha256=hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        "isingfold.rl.experiment_selection.load_rl_value_freeze",
        lambda **kwargs: frozen,
    )
    monkeypatch.setattr(
        "isingfold.rl.data.selector_labels.sample_program",
        _successful_evaluator([]),
    )
    build_audit_labels = build_parser().parse_args(
        [
            "label-selector-data",
            "--corpus",
            str(source),
            "--out",
            str(audit_labels),
            "--quality-attestation",
            str(tmp_path / "publisher-attestation.json"),
            "--expected-quality-attestation-digest",
            "a" * 64,
            "--expected-quality-publisher-id",
            "test-publisher",
            "--ground-certificate-root",
            str(tmp_path / "ground-certificate-root.json"),
            "--expected-ground-certificate-root-sha256",
            "b" * 64,
            "--seed",
            "79",
            "--split-seed",
            "19",
            "--calibration-fraction",
            "0.5",
            "--audit-mode",
            "--selector",
            str(bundle),
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--rl-value-selection-receipt",
            str(selection),
            "--expected-rl-value-selection-sha256",
            frozen.receipt_sha256,
        ]
    )
    build_audit_labels.func(build_audit_labels)

    validation_output = tmp_path / "audit-val"
    validation = build_parser().parse_args(
        [
            "audit-selector",
            "--corpus",
            str(source),
            "--selector",
            str(bundle),
            "--selector-labels",
            str(audit_labels),
            "--partition",
            "audit_val",
            "--out",
            str(validation_output),
        ]
    )
    validation.func(validation)
    validation_receipt = json.loads((validation_output / "receipt.json").read_text())
    label_manifest = json.loads((audit_labels / "manifest.json").read_text())
    assert validation_receipt["aggregate"]["records"] == label_manifest["partitions"][
        "audit_val"
    ]["records"]
    assert validation_receipt["access_control"]["rl_value_selection_required"] is False
    assert validation_receipt["provenance"]["audit_labels_manifest_sha256"] != validation_receipt[
        "provenance"
    ]["selector_fit_labels_manifest_sha256"]

    test_output = tmp_path / "audit-test"
    held_out = build_parser().parse_args(
        [
            "audit-selector",
            "--corpus",
            str(source),
            "--selector",
            str(bundle),
            "--selector-labels",
            str(audit_labels),
            "--partition",
            "audit_test",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--rl-value-selection-receipt",
            str(selection),
            "--expected-rl-value-selection-sha256",
            frozen.receipt_sha256,
            "--out",
            str(test_output),
        ]
    )
    held_out.func(held_out)
    test_receipt = json.loads((test_output / "receipt.json").read_text())
    assert test_receipt["aggregate"]["records"] == label_manifest["partitions"][
        "audit_test"
    ]["records"]
    assert test_receipt["access_control"]["rl_value_selection_required"] is True
    assert test_receipt["access_control"]["selected_model_family"] == "if-core"
    assert test_receipt["access_control"]["selected_method"] == "ppo-warm-start"
    assert len(test_receipt["access_control"]["rl_value_selection_receipt_sha256"]) == 64
    assert len(test_receipt["access_control"]["representation_selection_receipt_sha256"]) == 64

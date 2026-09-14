"""Prepared-v4 through sealed K=2 bank through all five validation arms."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import isingfold.rl.complete_system as complete_module
import isingfold.rl.data.import_embedbench as import_module
import isingfold.rl.data.prepared as prepared_module
import isingfold.rl.evaluate as evaluation_module
import isingfold.rl.initializer_bank as initializer_module
from isingfold.rl import cli
from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.complete_system import CompleteSystemConfig, read_complete_system_receipts
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import content_digest, prepare_candidate_bank_v4
from isingfold.rl.data.prepared import load_prepared_partition
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.experiment_selection import _GridCell, _load_evaluation
from isingfold.rl.external import SearchStatus
from isingfold.rl.validation_bootstrap_bank import (
    bootstrap_context_digest,
    bootstrap_record_from_outcome,
    bootstrap_same_support_contract_digest,
    build_bootstrap_plan,
    execute_bootstrap_row,
    publish_bootstrap_record,
    seal_bootstrap_bank,
    write_bootstrap_plan,
)
from tests.unit.test_rl_complete_system_aggregate import _evaluation_authority
from tests.unit.test_rl_initializer_bank import _FakeLACInitializer, _result
from tests.unit.test_rl_selector_labels import (
    _global_authority,
    _ground_partition_receipt,
    _target_access,
)
from tests.unit.trust_v4_support import write_v4_inputs


ROOT = Path(__file__).resolve().parents[2]
ARM_NAMES = (
    "return_initial",
    "random_masked",
    "classical_resource_first",
    "classical_quality_aware",
    "policy",
)


class _PublicationFakeLAC(_FakeLACInitializer):
    """Exact-type publication stand-in; no search is possible after sealing."""

    def __init__(self, results=None) -> None:
        super().__init__(list(results or ()))
        self._implementation_manifest = self.runtime_implementation_manifest

    def search(self, logical, host, *, seed, timeout_seconds, parameters, work_cap):
        result = super().search(
            logical,
            host,
            seed=seed,
            timeout_seconds=timeout_seconds,
            parameters=parameters,
            work_cap=work_cap,
        )
        work = result.work.as_dict()
        return dataclasses.replace(
            result,
            diagnostics={
                "backend": "lac_minorminer_cpp",
                "elapsed_seconds": result.elapsed_seconds,
                "package_version": self.identity.version,
                "random_seed": seed,
                "success": result.status is SearchStatus.EMBEDDING,
                "runtime_implementation_manifest": (
                    self.runtime_implementation_manifest
                ),
                "termination_reason": (
                    "success"
                    if result.status is SearchStatus.EMBEDDING
                    else "tries_exhausted"
                ),
                "transitions": work["decisions"],
                "trace": [],
                "incumbents": (
                    [{"generation": 0}]
                    if result.status is SearchStatus.EMBEDDING
                    else []
                ),
                "work": work,
                "work_budget_exhausted_coordinate": None,
                "work_cap": work_cap.as_dict(),
                "work_counter_schema": "lac-minorminer.native-work",
                "work_counter_version": 3,
                "search_profile": "hybrid_chimera_clique_v1",
                "profile_detail": "v0_success",
                "structural_fallback_invoked": False,
            },
        )


class _Selector:
    version = "integration-v1"

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, programs, _features):
        return min(1, len(programs) - 1)


def _allow_small_prepared_v4(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (import_module, prepared_module):
        monkeypatch.setattr(module, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_TEST_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 1)
        monkeypatch.setattr(
            module, "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL", 0.51
        )
        monkeypatch.setattr(
            module, "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH", 0.99
        )


def _record(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def test_prepared_v4_sealed_bank_runs_and_reloads_all_five_cli_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _allow_small_prepared_v4(monkeypatch)
    source = tmp_path / "source"
    bank_source, bank_manifest, targets, provenance, design, design_sha = (
        write_v4_inputs(source)
    )
    corpus = tmp_path / "prepared-v4"
    prepare_candidate_bank_v4(
        bank_source,
        bank_manifest,
        targets,
        provenance,
        design,
        corpus,
        expected_corpus_design_sha256=design_sha,
        qubit_cap=4,
    )
    public = load_prepared_partition(
        corpus, partition="val", include_evaluator=False
    )
    assert public.target_access is None
    manifest = json.loads((corpus / "manifest.json").read_text())
    splits = json.loads((corpus / "splits.json").read_text())
    population = cli._preinitialization_population_tasks(
        public.tasks, manifest=manifest, partition="val"
    )
    assert len(population) == 1

    monkeypatch.setattr(
        initializer_module, "LACMinorminerInitializerBackend", _PublicationFakeLAC
    )
    monkeypatch.setattr(
        complete_module, "LACMinorminerInitializerBackend", _PublicationFakeLAC
    )
    config_path = ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"
    config = CompleteSystemConfig.from_mapping(json.loads(config_path.read_text()))
    context = Context(qubit_cap=4)
    grid_path = ROOT / "configs" / "rl_grid_hybrid_v1.json"
    grid = json.loads(grid_path.read_text())
    protocol = grid["rl_value_evaluation"]
    backend = _PublicationFakeLAC()
    support_digest = bootstrap_same_support_contract_digest(
        partition="val",
        evaluation_seed=protocol["evaluation_seed"],
        repetitions=protocol["repetitions"],
        config_digest=config.digest,
        context_digest=bootstrap_context_digest(context),
    )
    plan = build_bootstrap_plan(
        public.tasks,
        prepared_manifest=manifest,
        prepared_split_registry=splits,
        prepared_manifest_sha256=hashlib.sha256(
            (corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        protocol_registry_sha256=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        protocol_record_digest=stable_digest(protocol),
        same_support_contract_digest=support_digest,
        partition="val",
        evaluation_seed=protocol["evaluation_seed"],
        repetitions=protocol["repetitions"],
        initializer=backend,
        runtime_implementation_manifest=backend.runtime_implementation_manifest,
        config=config,
        context=context,
    )
    bootstrap_root = tmp_path / "bootstrap-bank"
    bootstrap_root.mkdir()
    plan_sha = write_bootstrap_plan(bootstrap_root / "plan.json", plan)
    logical_nodes = tuple(population[0].logical.nodes())
    host_nodes = tuple(population[0].host.nodes())
    embedding = {
        logical: frozenset({host})
        for logical, host in zip(
            logical_nodes, host_nodes[: len(logical_nodes)], strict=True
        )
    }
    executor = _PublicationFakeLAC(
        [
            _result(embedding)
            for _ in range(
                len(plan.census) * 3 * config.max_initializer_attempts
            )
        ]
    )
    for row in plan.census:
        outcome = execute_bootstrap_row(
            plan,
            row.row_key,
            population[0],
            initializer=executor,
            config=config,
            context=context,
        )
        publish_bootstrap_record(
            bootstrap_root,
            plan,
            bootstrap_record_from_outcome(plan, row.row_key, outcome),
        )
    bootstrap_manifest_sha = seal_bootstrap_bank(bootstrap_root, plan)

    quality_authority = _evaluation_authority("val", 1)
    target_access = _target_access("val", 1)
    ground_receipt = _ground_partition_receipt("val", 1)
    target_tasks = tuple(
        dataclasses.replace(
            task,
            task=dataclasses.replace(task.task, ground_energy=-1.0),
            reference_status="exact",
            certificate_digest="a" * 64,
            evaluator_protocol_digest="b" * 64,
        )
        for task in public.tasks
    )
    selector_root = tmp_path / "selector"
    selector_root.mkdir()
    fit_payload = {"schema": "integration.selector-fit", "schema_version": 1}
    (selector_root / "fit_receipt.json").write_text(
        json.dumps(_record(fit_payload), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    (selector_root / "selector.pt").write_bytes(b"integration-selector\n")
    selector = _Selector()
    selector_digest = stable_digest({"selector": "integration"})
    bundle = SimpleNamespace(
        model=selector,
        selector_digest=selector_digest,
        corpus_manifest_sha256=hashlib.sha256(
            (corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        root=selector_root,
    )
    runtime_registry = runtime_implementation_registry()
    training_global = _global_authority()
    selection_sha = "c" * 64
    selection_record = "d" * 64
    checkpoint_digest = "e" * 64
    cell_id = "rl-000-selected-simpler-supervised-s1103"
    training_contract = {
        "environment": cli._context_snapshot(context),
        "gate_profile": None,
        "gate_receipt_sha256": None,
        "gate_record_digest": None,
        "grid_cell": cell_id,
        "grid_manifest_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "model": {"implementation": "integration.policy"},
        "quality_authority": {"global": training_global},
        "quality_preflight_receipt_sha256": "1" * 64,
        "quality_preflight_record_digest": "2" * 64,
        "seed": 1103,
        "selection_mode": "validated-receipt",
        "selection_receipt_sha256": selection_sha,
        "selection_record_digest": selection_record,
    }
    training_receipt = {
        "checkpoint_payload_digest": checkpoint_digest,
        "experiment_contract": training_contract,
        "method": "supervised-only",
        "model_family": "if-mlp",
        "phase": "rl-value",
        "runtime_implementation_digest": content_digest(runtime_registry),
        "runtime_implementation_registry": runtime_registry,
    }
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"integration-checkpoint\n")

    monkeypatch.setattr(cli, "_quality_attestation_pin", lambda _args: object())
    monkeypatch.setattr(
        cli,
        "_load_quality_partition",
        lambda *args, **kwargs: (
            target_tasks,
            quality_authority,
            target_access,
            ground_receipt,
        ),
    )
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        cli,
        "_bind_selector_quality_authority",
        lambda *args, **kwargs: quality_authority,
    )
    monkeypatch.setattr(
        cli,
        "_validate_quality_authority_binding",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        cli,
        "_load_evaluation_model",
        lambda *args, **kwargs: (selector, training_receipt),
    )
    monkeypatch.setattr(
        evaluation_module,
        "torch_controller",
        lambda *args, **kwargs: evaluation_module.first_commit_controller,
    )

    def sample_program(
        program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps
    ):
        del program, chains, problem, ground_energy, seed, num_sweeps
        return ReadBlock(num_reads, num_reads, 0.0, 0.0, 1)

    monkeypatch.setattr(complete_module, "sample_program", sample_program)
    destination = tmp_path / "five-arm-evaluation"
    args = SimpleNamespace(
        audit_reads=4096,
        bootstrap_bank=str(bootstrap_root),
        checkpoint=str(checkpoint),
        complete_config=str(config_path),
        corpus=str(corpus),
        deployment_policy_context=False,
        deterministic=True,
        device="cpu",
        expected_bootstrap_manifest_sha256=bootstrap_manifest_sha,
        expected_bootstrap_plan_sha256=plan_sha,
        greedy=False,
        grid=str(grid_path),
        margin=0.02,
        out=str(destination),
        partition="validation",
        protocol_preset="validation",
        qubit_cap=None,
        repetitions=4,
        scientific_evaluation_authorization={
            "stage": "rl-value",
            "grid_manifest_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
            "grid_cell": cell_id,
            "partition": "validation",
        },
        seed=44021,
        selector=str(selector_root),
        threads=1,
    )
    cli.cmd_evaluate(args)

    report = json.loads((destination / "report.json").read_text())
    assert set(report["complete_system_receipts"]) == set(ARM_NAMES)
    by_arm = {}
    for arm in ARM_NAMES:
        metadata = report["complete_system_receipts"][arm]
        receipts = read_complete_system_receipts(
            destination / metadata["path"], expected_sha256=metadata["sha256"]
        )
        assert len(receipts) == 4
        assert {
            receipt.bootstrap_binding["clone"]["consumer_id"]
            for receipt in receipts
        } == {f"rl-value/{cell_id}/{arm}"}
        by_arm[arm] = {
            receipt.pair_key: receipt.bootstrap_binding["clone"]
            for receipt in receipts
        }
    support_fields = (
        "bank_access_record_digest",
        "same_support_contract_digest",
        "bootstrap_record_digest",
        "bootstrap_outcome_record_digest",
        "bootstrap_payload_sha256",
        "row_key",
    )
    for pair_key in by_arm["policy"]:
        expected = {
            field: by_arm["policy"][pair_key][field] for field in support_fields
        }
        assert all(
            {field: by_arm[arm][pair_key][field] for field in support_fields}
            == expected
            for arm in ARM_NAMES
        )

    loaded = _load_evaluation(
        destination / "report.json",
        cell=_GridCell(
            cell_id=cell_id,
            grid_family="selected-simpler",
            model_family="if-mlp",
            method="supervised-only",
            seed=1103,
            registered_order=0,
        ),
        grid_digest=hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        protocol=protocol,
        representation_receipt_sha256=selection_sha,
        representation_record_digest=selection_record,
    )
    assert len(loaded.policy) == 4

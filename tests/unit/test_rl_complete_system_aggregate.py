"""Three-seed complete-system aggregation contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pytest

from tests.unit.evaluation_strata_support import evaluation_contract
from isingfold.embedding import LogicalProblem
from isingfold.rl.complete_system import (
    CompleteInitializerResult,
    CompletePopulationIdentity,
    CompleteSystemConfig,
    CompleteSystemEvidenceRecord,
    FrozenComponentIdentity,
    PartialWorkVector,
    complete_system_metrics,
    complete_system_method_metadata,
    run_complete_system,
    task_population_digest,
    write_complete_system_evidence,
    write_complete_system_outcomes,
    write_complete_system_receipts,
)
from isingfold.rl.complete_system_aggregate import (
    AuthenticatedCompleteSystemSeedRun,
    aggregate_complete_system_seeds,
)
from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.contracts import Context, Opcode, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.experiment_selection import crossed_bootstrap_bounds
from isingfold.rl.experiment_selection import FrozenRLValueSelection
from isingfold.rl.external import BackendIdentity, SearchStatus
from isingfold.rl import cli
from tests.unit.test_rl_selector_labels import (
    _global_authority,
    _ground_partition_receipt,
    _partition_authority,
    _target_access,
)

_TRAINING_SEEDS = (1103, 2207, 3301)
_CELL_IDS = tuple(f"selected-cell-{index}" for index in range(3))
_CHECKPOINT_DIGESTS = tuple(stable_digest({"checkpoint": seed}) for seed in _TRAINING_SEEDS)


def test_complete_receipt_bank_binding_validator_checks_every_row() -> None:
    access_digest = stable_digest({"bank": "access"})
    support_digest = stable_digest({"support": "persistent-k2"})
    consumer_id = "seed-1103/policy"
    clone = {
        "consumer_id": consumer_id,
        "bank_access_record_digest": access_digest,
        "same_support_contract_digest": support_digest,
    }
    receipts = [
        SimpleNamespace(bootstrap_binding={"clone": dict(clone)}),
        SimpleNamespace(bootstrap_binding={"clone": dict(clone)}),
    ]

    cli._validate_complete_receipt_bank_bindings(
        receipts,
        bank_access_record_digest=access_digest,
        same_support_contract_digest=support_digest,
        consumer_id=consumer_id,
    )

    receipts[1].bootstrap_binding["clone"]["consumer_id"] = "wrong-consumer"
    with pytest.raises(ValueError, match="externally pinned bootstrap bank"):
        cli._validate_complete_receipt_bank_bindings(
            receipts,
            bank_access_record_digest=access_digest,
            same_support_contract_digest=support_digest,
            consumer_id=consumer_id,
        )


def _evaluation_authority(
    partition: str = "test", count: int = 3
) -> dict[str, object]:
    payload = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 3,
        "global": _global_authority(),
        "evaluation_partition": _partition_authority(partition, count),
    }
    return {**payload, "record_digest": content_digest(payload)}


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _task(name: str, lineage: str, *, positive: bool) -> EmbeddingTask:
    logical = nx.path_graph(2)
    host = nx.path_graph(4)
    h = {0: 1.0 if positive else -1.0, 1: -0.5}
    j = {(0, 1): -1.0}
    problem = LogicalProblem.from_dicts(h, j)
    ground = min(
        sum(h[node] * spins[node] for node in logical)
        + sum(weight * spins[left] * spins[right] for (left, right), weight in j.items())
        for values in itertools.product((-1, 1), repeat=2)
        for spins in ({node: value for node, value in zip(logical, values, strict=True)},)
    )
    return EmbeddingTask(name, logical, host, problem, ground, lineage=lineage)


class _Initializer:
    def __init__(self, results: Sequence[CompleteInitializerResult]) -> None:
        self._results = list(results)
        self.identity = BackendIdentity(
            method_id="test-initializer-v1",
            distribution="isingfold",
            version="1.0.0",
            entrypoint="tests.complete_aggregate.initializer",
            implementation="fixed-test-double",
        )

    def search(self, logical, host, *, seed, timeout_seconds, parameters):
        del logical, host, seed, timeout_seconds, parameters
        return self._results.pop(0)


def _initializer_result(valid: bool) -> CompleteInitializerResult:
    embedding = {0: frozenset({0}), 1: frozenset({1})} if valid else None
    return CompleteInitializerResult(
        SearchStatus.EMBEDDING if valid else SearchStatus.NO_EMBEDDING,
        embedding,
        elapsed_seconds=0.0,
        work=PartialWorkVector.known(),
    )


def _commit_initial(decision, rng) -> int:
    del rng
    return next(
        index
        for index, (candidate, legal) in enumerate(
            zip(decision.candidates, decision.legal_mask, strict=True)
        )
        if legal and candidate.opcode is Opcode.COMMIT and candidate.archive_ref == 0
    )


def _component(name: str, *, seed: int | None = None) -> FrozenComponentIdentity:
    suffix = "frozen" if seed is None else str(seed)
    component_id = f"{name}/{suffix}"
    if name == "controller" and seed is not None:
        component_id = f"isingfold-policy/if-core/ppo-warm-start/seed-{seed}"
    return FrozenComponentIdentity(
        component_id=component_id,
        version="v1",
        implementation=f"tests.{name}.{suffix}",
        artifact_sha256=stable_digest({"component": name, "seed": seed}),
    )


def _config() -> CompleteSystemConfig:
    return CompleteSystemConfig(
        online_wallclock_seconds=30.0,
        max_initializer_attempts=1,
        audit_reads=4096,
        selection_rule="resource-lexicographic",
        online_evaluator_feedback=False,
        initializer_backend="isingfold",
        initializer_method_id="test-initializer-v1",
        expected_initializer_version="1.0.0",
        policy_restart_mode="disabled-no-native-replay-v1",
        initializer_parameters={"tries": 1},
    )


def _protocol() -> dict[str, object]:
    config = _config()
    external_config = {
        "schema": "isingfold.external-complete-system-config",
        "schema_version": 2,
        "backend": "stock-minorminer",
        "expected_backend_version": "0.2.22",
        "online_wallclock_seconds": 30.0,
        "max_restarts": 1,
        "audit_reads": 4096,
        "selection_rule": "resource-lexicographic",
        "online_evaluator_feedback": False,
        "minorminer_parameters": {
            "tries": 1,
            "threads": 1,
            "max_no_improvement": 10,
            "chainlength_patience": 10,
        },
    }
    return {
        "partition": "test",
        "evaluation_seed": 55079,
        "repetitions": 4,
        "audit_reads": 4096,
        "deployment_rule": "categorical-temperature-one",
        "population_scope": "all-policy-instances-before-initialization",
        "aggregation": "equal-training-seed-then-equal-immutable-base-lineage",
        "primary_metric": "unconditional-if-q3-s0-utility",
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": 130363,
        "two_sided_alpha": 0.05,
        "feasibility_noninferiority_margin": 0.02,
        "learned_config_digest": config.digest,
        "learned_config_file_sha256": stable_digest(config.as_dict()),
        "external_config_digest": stable_digest(external_config),
        "external_config_file_sha256": stable_digest(external_config),
    }


def _selection_binding(
    index: int, runtime_registry: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema": "isingfold.complete-system-selected-training",
        "schema_version": 3,
        "selection_receipt_sha256": stable_digest({"selection": "file"}),
        "selection_record_digest": stable_digest({"selection": "record"}),
        "grid_manifest_sha256": stable_digest({"grid": "v1"}),
        "selected_model_family": "if-core",
        "selected_grid_model_family": "if-core",
        "selected_method": "ppo-warm-start",
        "all_training_seeds": list(_TRAINING_SEEDS),
        "all_source_cell_ids": list(_CELL_IDS),
        "all_source_checkpoint_payload_digests": list(_CHECKPOINT_DIGESTS),
        "runtime_implementation_registry": runtime_registry,
        "runtime_implementation_digest": content_digest(runtime_registry),
        "quality_preflight_receipt_sha256": "7" * 64,
        "quality_preflight_record_digest": "8" * 64,
        "source_cell_id": _CELL_IDS[index],
        "source_checkpoint_payload_digest": _CHECKPOINT_DIGESTS[index],
        "training_seed": _TRAINING_SEEDS[index],
        "seed_selection_forbidden": True,
        "selected_validation_checkpoint_reused": False,
        "fresh_representation_stage_required": True,
        "retraining_rule": (
            "fresh-selected-configuration-on-persistent-k2-cache-support"
        ),
    }


def _report(
    root: Path,
    *,
    index: int,
    receipts,
    context: Context,
    config: CompleteSystemConfig,
) -> dict[str, object]:
    raw_path = root / f"seed-{index}-receipts.jsonl"
    raw_sha = write_complete_system_receipts(raw_path, receipts)
    reference = receipts[0]
    protocol = _protocol()
    evidence = tuple(
        CompleteSystemEvidenceRecord(
            instance=receipt.instance,
            lineage=receipt.lineage,
            repetition=receipt.repetition,
            population_digest=receipt.population.digest,
            complete_receipt_digest=receipt.as_dict()["record_digest"],
            terminal_evidence=receipt.terminal_evidence,
        )
        for receipt in receipts
    )
    evidence_content = b"".join(
        (
            json.dumps(
                record.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        for record in sorted(evidence, key=lambda item: item.pair_key)
    )
    runtime_registry = runtime_implementation_registry()
    support_digest = stable_digest({"support": "persistent-k2"})
    bootstrap_access_body = {
        "schema": "isingfold.rl-value-bootstrap-access",
        "schema_version": 1,
        "same_support_contract_digest": support_digest,
        "protocol_preset": "final-test",
        "opened_evaluator_targets": False,
    }
    bootstrap_access = {
        **bootstrap_access_body,
        "record_digest": content_digest(bootstrap_access_body),
    }
    payload: dict[str, object] = {
        "schema": "isingfold.complete-system-evaluation",
        "schema_version": 4,
        "partition": "test",
        "sealed_test_opened": True,
        "evaluation_protocol": protocol,
        "deployment_rule": protocol["deployment_rule"],
        "training_seed_index": index,
        "training_seed": _TRAINING_SEEDS[index],
        "source_cell_id": _CELL_IDS[index],
        "selection_binding": _selection_binding(index, runtime_registry),
        "source_corpus_manifest_sha256": reference.population.source_manifest_sha256,
        "quality_authority": _evaluation_authority(),
        "target_access": _target_access("test", 3),
        "ground_partition_receipt": _ground_partition_receipt("test", 3),
        "complete_system_config_sha256": stable_digest(config.as_dict()),
        "bootstrap_bank_access": bootstrap_access,
        "same_support_contract_digest": support_digest,
        "context": _jsonable(context),
        "context_digest": reference.context_digest,
        "population": reference.population.as_dict(),
        "population_digest": reference.population.digest,
        "controller": reference.controller.as_dict(),
        "selector": reference.selector.as_dict(),
        "initializer": reference.initializer.as_dict(),
        "method": complete_system_method_metadata(
            config,
            reference.initializer,
            reference.controller,
            reference.selector,
            reference.population,
        ),
        "runtime_platform": {"runtime": "fixed-test-platform"},
        "inference_device_type": "cpu",
        "inference_device_name": "test-cpu",
        "inference_threads": 1,
        "deterministic": True,
        "runtime_implementation_registry": runtime_registry,
        "runtime_implementation_digest": content_digest(runtime_registry),
        "repetitions": 4,
        "metrics": complete_system_metrics(receipts),
        "artifacts": {
            "complete_receipts": {
                "path": "complete_receipts.jsonl",
                "sha256": raw_sha,
                "count": len(receipts),
            },
            "terminal_evidence": {
                "path": "terminal_evidence.jsonl",
                "sha256": hashlib.sha256(evidence_content).hexdigest(),
                "count": len(receipts),
            },
            "outcomes": {
                "path": "outcomes.jsonl",
                "sha256": stable_digest({"outcomes": index}),
                "count": len(receipts),
            },
        },
    }
    return {**payload, "record_digest": content_digest(payload)}, evidence


def _runs(monkeypatch, tmp_path: Path):
    tasks = (
        _task("quality-one", "lineage-a", positive=True),
        _task("quality-zero-1", "lineage-b", positive=False),
        _task("quality-zero-2", "lineage-b", positive=False),
    )
    context = Context(qubit_cap=4)
    config = _config()
    selector_identity = _component("selector")
    identities = tuple(sorted((task.lineage, task.name) for task in tasks))
    strata, design = evaluation_contract(identities)
    population = CompletePopulationIdentity(
        population_id="sealed-complete-test",
        source_manifest_sha256=stable_digest({"manifest": "release-v2"}),
        task_payload_sha256=task_population_digest(tasks),
        expected_instances=identities,
        expected_repetitions=4,
        evaluation_seed=55079,
        evaluation_strata=strata,
        confirmatory_design=design,
    )

    def sample(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, chains, ground_energy, seed, num_sweeps
        hits = num_reads if problem.h[0] > 0 else 0
        return ReadBlock(hits, num_reads, 0.0, 0.0, 1)

    monkeypatch.setattr("isingfold.rl.complete_system.sample_program", sample)
    runs = []
    for index, training_seed in enumerate(_TRAINING_SEEDS):
        validity = [False, True, True] * 4 if index == 0 else [True, True, True] * 4
        initializer = _Initializer([_initializer_result(valid) for valid in validity])
        controller_identity = _component("controller", seed=training_seed)
        _, receipts = run_complete_system(
            tasks,
            context,
            initializer,
            config,
            controller=_commit_initial,
            selector=fixed_strength_selector(),
            controller_identity=controller_identity,
            selector_identity=selector_identity,
            population=population,
            seed=55079,
            repetitions=4,
        )
        report, evidence = _report(
            tmp_path,
            index=index,
            receipts=receipts,
            context=context,
            config=config,
        )
        runs.append(
            AuthenticatedCompleteSystemSeedRun(report, tuple(receipts), evidence)
        )
    return tuple(runs)


def _rehash(report: Mapping[str, object], **updates: object) -> dict[str, object]:
    payload = {key: value for key, value in report.items() if key != "record_digest"}
    payload.update(updates)
    return {**payload, "record_digest": content_digest(payload)}


def test_complete_aggregate_weights_three_seeds_then_two_lineages_equally(
    monkeypatch, tmp_path: Path
) -> None:
    runs = _runs(monkeypatch, tmp_path)

    aggregate = aggregate_complete_system_seeds(runs)

    expected_matrix = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 0.0)))
    expected_interval = crossed_bootstrap_bounds(
        expected_matrix,
        replicates=20_000,
        seed=130363,
        alpha=0.025,
    )
    assert aggregate["schema"] == "isingfold.complete-system-three-seed-aggregate"
    assert aggregate["training_seeds"] == list(_TRAINING_SEEDS)
    assert aggregate["independent_lineages"] == 2
    assert aggregate["unconditional_utility_mean"] == pytest.approx(1 / 3)
    assert aggregate["unconditional_utility_confidence_interval"] == {
        "lower": pytest.approx(expected_interval[0]),
        "upper": pytest.approx(expected_interval[1]),
    }
    assert aggregate["invalid_evaluator_seed_defined_count"] == 0
    assert all(
        receipt.evaluator_seed is None and receipt.outcome.evaluator_seed is None
        for run in runs
        for receipt in run.receipts
        if not receipt.outcome.returned_valid
    )
    assert aggregate["aggregation"] == ("equal-training-seed-then-equal-immutable-base-lineage")
    assert aggregate["bootstrap_protocol"]["replicates"] == 20_000
    assert aggregate["bootstrap_protocol"]["tail_alpha"] == 0.025
    subgroups = aggregate["subgroup_results"]
    assert subgroups["distribution"]["ood"]["independent_lineages"] == 2
    assert subgroups["topology"]["path-a"]["instance_count"] == 3
    assert subgroups["size_bin"]["000002-000003"]["independent_lineages"] == 2
    assert subgroups["size_bin"]["000004-000007"]["independent_lineages"] == 1
    assert aggregate["subgroup_inference"]["scope"] == "descriptive-not-powered"
    assert aggregate["subgroup_inference"]["multiplicity_method"] == "bonferroni"
    assert aggregate["subgroup_inference"]["independent_unit"] == (
        "immutable-base-lineage"
    )
    assert "external_stock_pairing_claimed" not in aggregate
    json.dumps(aggregate, allow_nan=False)
    assert aggregate["record_digest"] == content_digest(
        {key: value for key, value in aggregate.items() if key != "record_digest"}
    )


def test_complete_aggregate_cli_authenticates_reports_receipts_evidence_and_outcomes(
    monkeypatch, tmp_path: Path
) -> None:
    runs = _runs(monkeypatch, tmp_path)
    tasks = (
        _task("quality-one", "lineage-a", positive=True),
        _task("quality-zero-1", "lineage-b", positive=False),
        _task("quality-zero-2", "lineage-b", positive=False),
    )
    context = Context(qubit_cap=4)
    config = _config()
    config_path = tmp_path / "complete-config.json"
    config_path.write_bytes(canonical_json_bytes(config.as_dict()) + b"\n")
    source_grid_path = Path(__file__).resolve().parents[2] / "configs" / "rl_grid_hybrid_v1.json"
    grid_payload = json.loads(source_grid_path.read_text())
    grid_payload["complete_system_evaluation"].update(
        {
            "learned_config_digest": config.digest,
            "learned_config_file_sha256": cli._sha256_file(config_path),
            "external_config_digest": "e" * 64,
            "external_config_file_sha256": "f" * 64,
        }
    )
    grid_path = tmp_path / "grid.json"
    grid_path.write_bytes(canonical_json_bytes(grid_payload) + b"\n")
    grid_sha = cli._sha256_file(grid_path)
    first_binding = runs[0].report["selection_binding"]
    selection_runtime_registry = dict(
        runs[0].report["runtime_implementation_registry"]
    )
    selection = FrozenRLValueSelection(
        receipt_sha256=first_binding["selection_receipt_sha256"],
        record_digest=first_binding["selection_record_digest"],
        grid_manifest_sha256=grid_sha,
        model_family="if-core",
        grid_model_family="if-core",
        method="ppo-warm-start",
        training_seeds=_TRAINING_SEEDS,
        cell_ids=_CELL_IDS,
        checkpoint_payload_digests=_CHECKPOINT_DIGESTS,
        representation_selection_sha256=stable_digest({"representation": "file"}),
        representation_selection_record_digest=stable_digest(
            {"representation": "record"}
        ),
        runtime_implementation_registry=selection_runtime_registry,
        runtime_implementation_digest=content_digest(selection_runtime_registry),
        quality_preflight_receipt_sha256="7" * 64,
        quality_preflight_record_digest="8" * 64,
    )
    monkeypatch.setattr(
        "isingfold.rl.experiment_selection.load_rl_value_freeze",
        lambda **kwargs: selection,
    )
    authority = dict(runs[0].report["quality_authority"])
    target_access = dict(runs[0].report["target_access"])
    ground_partition_receipt = dict(
        runs[0].report["ground_partition_receipt"]
    )
    monkeypatch.setattr(cli, "_quality_attestation_pin", lambda args: object())
    monkeypatch.setattr(
        cli,
        "_load_quality_partition",
        lambda *args, **kwargs: (
            [object()],
            authority,
            target_access,
            ground_partition_receipt,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_preinitialization_population_tasks",
        lambda prepared, *, manifest, partition: list(tasks),
    )
    monkeypatch.setattr(cli, "_context", lambda args, corpus=None: context)
    support_digest = stable_digest({"support": "persistent-k2"})
    bank_access_body = {
        "schema": "isingfold.rl-value-bootstrap-access",
        "schema_version": 1,
        "same_support_contract_digest": support_digest,
        "protocol_preset": "final-test",
        "opened_evaluator_targets": False,
    }
    bank_access = {
        **bank_access_body,
        "record_digest": content_digest(bank_access_body),
    }
    bank = SimpleNamespace(
        access_receipt=SimpleNamespace(
            as_dict=lambda: bank_access,
            record_digest=bank_access["record_digest"],
        )
    )
    plan = SimpleNamespace(same_support_contract_digest=support_digest)
    monkeypatch.setattr(
        cli,
        "_load_pinned_evaluation_bootstrap_bank",
        lambda *args, **kwargs: (bank, plan),
    )
    monkeypatch.setattr(
        cli,
        "_validate_complete_receipt_bank_bindings",
        lambda *args, **kwargs: None,
    )

    corpus = tmp_path / "prepared"
    corpus.mkdir()
    manifest_path = corpus / "manifest.json"
    manifest_path.write_text("{}\n")
    expected_manifest_sha = runs[0].receipts[0].population.source_manifest_sha256
    original_sha256 = cli._sha256_file

    def controlled_sha256(path):
        if Path(path) == manifest_path:
            return expected_manifest_sha
        return original_sha256(path)

    monkeypatch.setattr(cli, "_sha256_file", controlled_sha256)
    evaluation_root = tmp_path / "evaluations"
    for index, run in enumerate(runs):
        run_root = evaluation_root / f"seed-{index}"
        run_root.mkdir(parents=True)
        receipt_sha = write_complete_system_receipts(
            run_root / "complete_receipts.jsonl", run.receipts
        )
        evidence_sha = write_complete_system_evidence(
            run_root / "terminal_evidence.jsonl", run.receipts
        )
        outcome_sha = write_complete_system_outcomes(
            run_root / "outcomes.jsonl", run.receipts
        )
        report = json.loads(canonical_json_bytes(run.report))
        report.update(
            {
                "selection_binding": cli._complete_selection_binding(selection, index),
                "evaluation_protocol": grid_payload["complete_system_evaluation"],
                "source_corpus_manifest_sha256": expected_manifest_sha,
                "complete_system_config_sha256": original_sha256(config_path),
                "bootstrap_bank_access": bank_access,
                "same_support_contract_digest": support_digest,
                "artifacts": {
                    "complete_receipts": {
                        "path": "complete_receipts.jsonl",
                        "sha256": receipt_sha,
                        "count": len(run.receipts),
                    },
                    "terminal_evidence": {
                        "path": "terminal_evidence.jsonl",
                        "sha256": evidence_sha,
                        "count": len(run.receipts),
                    },
                    "outcomes": {
                        "path": "outcomes.jsonl",
                        "sha256": outcome_sha,
                        "count": len(run.receipts),
                    },
                },
            }
        )
        report["record_digest"] = content_digest(
            {key: value for key, value in report.items() if key != "record_digest"}
        )
        (run_root / "report.json").write_bytes(canonical_json_bytes(report) + b"\n")

    output = tmp_path / "aggregate.json"
    args = cli.build_parser().parse_args(
        [
            "aggregate-complete-system",
            "--grid",
            str(grid_path),
            "--corpus",
            str(corpus),
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
            "--evaluation-root",
            str(evaluation_root),
            "--config",
            str(config_path),
            "--bootstrap-bank",
            str(tmp_path / "bootstrap-bank"),
            "--expected-bootstrap-plan-sha256",
            "c" * 64,
            "--expected-bootstrap-manifest-sha256",
            "d" * 64,
            "--rl-value-selection-receipt",
            str(tmp_path / "selection.json"),
            "--expected-selection-sha256",
            selection.receipt_sha256,
            "--out",
            str(output),
        ]
    )
    args.func(args)

    aggregate = json.loads(output.read_text())
    assert aggregate["training_seeds"] == list(_TRAINING_SEEDS)
    assert aggregate["unconditional_utility_mean"] == pytest.approx(1 / 3)
    assert aggregate["matched_contract"]["quality_authority"] == authority


def test_complete_aggregate_is_permutation_invariant_and_allows_mixed_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    runs = list(_runs(monkeypatch, tmp_path))
    report = dict(runs[2].report)
    runs[2] = AuthenticatedCompleteSystemSeedRun(
        _rehash(
            report,
            runtime_platform={"runtime": "different-hpc-platform"},
            inference_device_type="cuda",
        ),
        runs[2].receipts,
        runs[2].evidence,
    )

    forward = aggregate_complete_system_seeds(runs)
    reversed_order = aggregate_complete_system_seeds(tuple(reversed(runs)))

    assert forward == reversed_order
    assert forward["runtime_comparability"] == {
        "identical": False,
        "aggregate_latency_reported": False,
    }


def test_complete_aggregate_rejects_missing_or_duplicate_selected_seed(
    monkeypatch, tmp_path: Path
) -> None:
    runs = _runs(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="exactly three"):
        aggregate_complete_system_seeds(runs[:2])
    with pytest.raises(ValueError, match="indices|training seeds"):
        aggregate_complete_system_seeds((runs[0], runs[0], runs[2]))


def test_complete_aggregate_rejects_rehashed_report_receipt_mixing(
    monkeypatch, tmp_path: Path
) -> None:
    runs = list(_runs(monkeypatch, tmp_path))
    report = dict(runs[1].report)
    artifacts = {key: dict(value) for key, value in report["artifacts"].items()}
    artifacts["complete_receipts"]["sha256"] = stable_digest({"wrong": "receipt-file"})
    runs[1] = AuthenticatedCompleteSystemSeedRun(
        _rehash(report, artifacts=artifacts), runs[1].receipts, runs[1].evidence
    )

    with pytest.raises(ValueError, match="receipt file identity"):
        aggregate_complete_system_seeds(runs)


def test_complete_aggregate_rejects_rehashed_failure_metric_tamper(
    monkeypatch, tmp_path: Path
) -> None:
    runs = list(_runs(monkeypatch, tmp_path))
    report = dict(runs[0].report)
    metrics = _jsonable(report["metrics"])
    assert isinstance(metrics, dict)
    metrics["initialization_failures"] += 1
    runs[0] = AuthenticatedCompleteSystemSeedRun(
        _rehash(report, metrics=metrics), runs[0].receipts, runs[0].evidence
    )

    with pytest.raises(ValueError, match="metrics differ"):
        aggregate_complete_system_seeds(runs)


@pytest.mark.parametrize("field", ("selector", "initializer", "evaluation_protocol"))
def test_complete_aggregate_requires_identical_frozen_contracts(
    monkeypatch, tmp_path: Path, field: str
) -> None:
    runs = list(_runs(monkeypatch, tmp_path))
    report = dict(runs[2].report)
    changed = dict(report[field])
    changed[next(iter(changed))] = "tampered"
    runs[2] = AuthenticatedCompleteSystemSeedRun(
        _rehash(report, **{field: changed}), runs[2].receipts, runs[2].evidence
    )

    with pytest.raises(ValueError, match="protocol|initializer|selector"):
        aggregate_complete_system_seeds(runs)

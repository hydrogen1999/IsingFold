from __future__ import annotations

import hashlib
import json
import random
import tomllib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import networkx as nx

torch = pytest.importorskip("torch")

from isingfold.rl.cli import (  # noqa: E402
    GATE_RECEIPT_SCHEMA_VERSION,
    REGISTERED_MODEL_FAMILIES,
    SelectorBundle,
    _LineageEqualTaskScheduler,
    _context_snapshot,
    _experiment_contract,
    _load_grid,
    _load_gate_receipt,
    _load_exact_conformance_tasks,
    _load_quality_labels,
    _load_transfer,
    _model_identity,
    _load_selector_bundle,
    _model,
    _ppo_collection_schedule_contract,
    _ppo_update_control_contract,
    _prepare_run_dir,
    _publish_gate_receipt,
    _preinitialization_population_tasks,
    _resolve_training_policy_context,
    _resolve_device,
    _require_current_ppo_collection_schedule,
    _require_current_ppo_update_control,
    _resume_training_state,
    _save_training_state,
    _seed_runtime,
    _stratified_gate_sample,
    _validate_authenticated_k2_support_gate_result,
    _validate_support_headroom_gate_result,
    build_parser,
    cmd_complete_system_train_cell,
    cmd_evaluate,
    cmd_evaluate_external,
    cmd_train,
    cmd_warm_start,
)
from isingfold.embedding import LogicalProblem  # noqa: E402
from isingfold.rl.contracts import Context  # noqa: E402
from isingfold.rl.checkpoint import (  # noqa: E402
    PPO_TRAINER_STATE_SCHEMA,
    runtime_implementation_registry,
)
from isingfold.rl.experiment_selection import FrozenRLValueSelection  # noqa: E402
from isingfold.rl.data.selector_labels import (  # noqa: E402
    build_selector_labels,
    load_selector_metadata,
)
from isingfold.rl.env import EmbeddingTask  # noqa: E402
from isingfold.rl.gates import support_headroom_inference  # noqa: E402
from isingfold.rl.complete_system import CompleteSystemConfig  # noqa: E402
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest  # noqa: E402
from tests.unit.test_rl_selector_labels import (  # noqa: E402
    _authority,
    _ground_partition_receipt,
    _patch_prepared,
    _source_manifest,
    _successful_evaluator,
    _target_access,
    _prepared_task,
    _tasks,
)
from tests.unit.test_rl_ifcore_normative import (  # noqa: E402
    _conflict_observation,
    _phase_observation,
)
from isingfold.rl.model import IFCore  # noqa: E402
from isingfold.rl.ppo import (  # noqa: E402
    PPOConfig,
    WarmStartActionValueTarget,
    WarmStartUtilityTarget,
)

ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_CONTROL_ENTRYPOINTS = frozenset(
    {
        "apollo_build_isingfold_runtime.sh",
        "goose_build_isingfold_runtime.sbatch",
        "goose_verify_isingfold_runtime.sbatch",
    }
)


def test_release_gate_schema_rejects_pre_headroom_receipts() -> None:
    assert GATE_RECEIPT_SCHEMA_VERSION == 7


def test_release_gate_rejects_a_bare_gate2_pass_without_headroom_evidence() -> None:
    with pytest.raises(ValueError, match="Gate 2"):
        _validate_support_headroom_gate_result(
            {"pass": True},
            scalable_sampling={"selected_stratum_digests": ["a" * 64]},
            reward_reads=256,
        )


def test_release_gate_rejects_a_bare_authenticated_support_claim() -> None:
    with pytest.raises(ValueError, match="Gate 2"):
        _validate_authenticated_k2_support_gate_result(
            {"pass": True},
            plan_identity={"raw_sha256": "a" * 64, "record_digest": "b" * 64},
        )


def test_release_gate_rejects_full_support_bank_outside_plan_binding() -> None:
    bank = _full_support_bank_contract()
    plan_identity = {
        "initializer_bank_contract_record_digest": bank["record_digest"],
        "initializer_bank_manifest_sha256": bank["manifest_sha256"],
        "raw_sha256": "a" * 64,
        "record_digest": "b" * 64,
    }
    result = _full_support_gate_result(plan_identity)
    changed_bank_body = {
        key: value for key, value in result["initializer_bank"].items() if key != "record_digest"
    }
    changed_bank_body["training_seed"] = 908
    result["initializer_bank"] = {
        **changed_bank_body,
        "record_digest": content_digest(changed_bank_body),
    }
    result["record_digest"] = content_digest(
        {key: value for key, value in result.items() if key != "record_digest"}
    )

    with pytest.raises(ValueError, match="initializer bank"):
        _validate_authenticated_k2_support_gate_result(
            result,
            plan_identity=plan_identity,
        )


def test_release_gate_rejects_audited_indices_that_disagree_with_legal_count() -> None:
    bank = _full_support_bank_contract()
    plan_identity = {
        "initializer_bank_contract_record_digest": bank["record_digest"],
        "initializer_bank_manifest_sha256": bank["manifest_sha256"],
        "raw_sha256": "a" * 64,
        "record_digest": "b" * 64,
    }
    result = _full_support_gate_result(plan_identity)
    row = result["rows"][0]
    row["legal_candidate_count"] = 1
    row["audited_legal_candidate_count"] = 1
    row["record_digest"] = content_digest(
        {key: value for key, value in row.items() if key != "record_digest"}
    )
    result["legal_candidate_count"] -= 1
    result["audited_candidate_count"] -= 1
    result["record_digest"] = content_digest(
        {key: value for key, value in result.items() if key != "record_digest"}
    )

    with pytest.raises(ValueError, match="exact replay row"):
        _validate_authenticated_k2_support_gate_result(
            result,
            plan_identity=plan_identity,
        )


_TARGET_FREE_HPC_LAUNCHERS = frozenset(
    {
        "apollo_bootstrap_bank.sh",
        "apollo_initializer_bank.sh",
        "apollo_seal_initializer_bank.sh",
        "goose_bootstrap_bank.sbatch",
        "goose_bootstrap_bank_packed.sbatch",
        "goose_initializer_bank.sbatch",
        "goose_initializer_bank_packed.sbatch",
        "goose_seal_bootstrap_bank.sbatch",
        "goose_seal_initializer_bank.sbatch",
    }
)


def _scientific_hpc_launchers() -> list[Path]:
    """Return launchers that execute a scientific workload, not runtime controls."""

    candidates = sorted((ROOT / "scripts").glob("apollo_*.sh")) + sorted(
        (ROOT / "scripts").glob("goose_*.sbatch")
    )
    return [path for path in candidates if path.name not in _RUNTIME_CONTROL_ENTRYPOINTS]


def _digest(character: str) -> str:
    return character * 64


def _exact_gate_result(source_manifest_sha256: str) -> dict[str, object]:
    task_ids = [f"exact-task-{index}" for index in range(8)]
    base_lineages = [f"exact-lineage-{index}" for index in range(8)]
    exact_identity = {
        "base_lineages": base_lineages,
        "corpus_id": "if-gate1-exact-v1",
        "exact_reproduction": True,
        "file_sha256": "1" * 64,
        "max_logical_variables_per_task": 12,
        "max_tasks": 8,
        "path": "exact-conformance-v1.json",
        "record_digest": "2" * 64,
        "selected_marginal_census": {
            "selected_base_lineage_count": 8,
            "selected_task_count": 8,
        },
        "selector_implementation": {
            "implementation": (
                "isingfold.rl.data.exact_conformance.select_exact_conformance_tasks"
            ),
            "selection_seed": 0,
            "source_sha256": "3" * 64,
            "tie_domain": "exact-conformance-task-tie-v1",
            "version": "exact-conformance-selector-v1",
        },
        "source_population_census": {},
        "source_corpus_manifest_sha256": source_manifest_sha256,
        "task_count": 8,
        "task_ids": task_ids,
    }
    return {
        "instances_requested": 8,
        "eligible_instances": 8,
        "witness_valid_rate": 1.0,
        "program_faithful_rate": 1.0,
        "structural_checked": 1,
        "structural_unknown": 0,
        "structural_agreement": 1.0,
        "positive_structural_checked": 1,
        "positive_structural_agreement": 1.0,
        "negative_structural_checked": 1,
        "negative_structural_unknown": 0,
        "negative_structural_agreement": 1.0,
        "overlap_search_checked": 1,
        "overlap_search_admissible_rate": 1.0,
        "overlap_return_rejection_rate": 1.0,
        "independent_oracle": {},
        "independent_program_oracle": {},
        "exact_bounds": {
            "max_tasks": 8,
            "max_logical_variables_per_task": 12,
        },
        "registered_population": {
            "exact_task_count": True,
            "task_count": 8,
            "unique_task_id_count": 8,
            "unique_base_lineage_count": 8,
        },
        "authenticated_exact_corpus": exact_identity,
        "pass": True,
    }


def _support_gate_result(stratum_ids: list[str]) -> dict[str, object]:
    headroom = support_headroom_inference(
        [0.10] * len(stratum_ids),
        stratum_ids,
    )
    return {
        "signal_name": "frozen-broad-reference-support-audit",
        "instances_requested": len(stratum_ids),
        "eligible_instances": len(stratum_ids),
        "broad_reference_batches_per_instance": 8,
        "offline_candidate_attempts": 24,
        "offline_evaluator_calls": 26,
        "offline_evaluator_reads": 26 * 256,
        "headroom_confirmation_blocks": 2 * len(stratum_ids),
        "mean_feasibility_support_recall": 0.75,
        "feasibility_support_recall_se": 0.0,
        "mean_broad_pool_quality_advantage": 0.01,
        "broad_pool_quality_advantage_se": 0.0,
        "mean_within_support_quality_spread": 0.05,
        "within_support_quality_spread_se": 0.0,
        "mean_supported_headroom_over_initial": headroom[
            "mean_supported_headroom_over_initial"
        ],
        "supported_headroom_se": headroom["supported_headroom_se"],
        "headroom_inference": headroom,
        "thresholds": {
            "min_feasibility_support_recall": 0.5,
            "min_within_support_quality_spread": 0.02,
            "max_broad_pool_quality_advantage": 0.05,
            "min_mean_headroom_over_initial": 0.02,
            "headroom_one_sided_alpha": 0.05,
            "min_per_stratum_headroom_coverage": 0.5,
        },
        "pass": True,
        "note": "fixture",
    }


def _full_support_bank_contract() -> dict[str, object]:
    bank_body: dict[str, object] = {
        "access_receipt_record_digest": "1" * 64,
        "conditional_episode_count": 128,
        "config_digest": "2" * 64,
        "context_digest": "3" * 64,
        "episode_schedule_start": 0,
        "episode_schedule_stop_exclusive": 128,
        "manifest_record_digest": "4" * 64,
        "manifest_sha256": "5" * 64,
        "opened_evaluator_data": False,
        "partition": "train",
        "plan_record_digest": "6" * 64,
        "prepared_manifest_sha256": "7" * 64,
        "protocol": "authenticated-persistent-k2-initializer-bank-v1",
        "publication_eligible": True,
        "restart_cache_slots_per_episode": 2,
        "schema": "isingfold.quality-initializer-bank-contract",
        "schema_version": 1,
        "training_seed": 907,
    }
    return {**bank_body, "record_digest": content_digest(bank_body)}


def _full_support_gate_result(plan_identity: dict[str, str]) -> dict[str, object]:
    from isingfold.rl.contracts import OPCODES

    bank = _full_support_bank_contract()
    aggregate_opcodes = {opcode: 0 for opcode in OPCODES}
    aggregate_opcodes["REPAIR_GROUP"] = 128
    aggregate_opcodes["RESTART"] = 128
    rows: list[dict[str, object]] = []
    lineages: list[str] = []
    for index in range(128):
        lineage = f"support-lineage-{index:04d}"
        lineages.append(lineage)
        row_opcodes = {opcode: 0 for opcode in OPCODES}
        row_opcodes["REPAIR_GROUP"] = 1
        row_opcodes["RESTART"] = 1
        row_body: dict[str, object] = {
            "audited_indices": [0, 1],
            "audited_legal_candidate_count": 2,
            "base_lineage": lineage,
            "candidate_count": 2,
            "candidate_payload_root": content_digest([lineage, "payloads"]),
            "family_counts": {"REPAIR_GROUP": 1, "RESTART": 1},
            "initializer_bank_episode_index": index,
            "initializer_bootstrap_record_digest": content_digest(
                [lineage, "bootstrap"]
            ),
            "instance_id": f"support-instance-{index:04d}",
            "legal_candidate_count": 2,
            "legal_family_counts": {"REPAIR_GROUP": 1, "RESTART": 1},
            "legal_indices": [0, 1],
            "legal_opcode_counts": row_opcodes,
            "opcode_counts": row_opcodes,
            "row_id": f"support-row-{index:04d}",
            "state_fingerprint": content_digest([lineage, "state"]),
            "support_fingerprint": content_digest([lineage, "support"]),
            "task_id": f"support-task-{index:04d}",
        }
        rows.append({**row_body, "record_digest": content_digest(row_body)})
    body: dict[str, object] = {
        "audited_candidate_count": 256,
        "candidate_count": 256,
        "candidate_selection": "all-legal-materialized-candidates-no-truncation",
        "family_counts": {"REPAIR_GROUP": 128, "RESTART": 128},
        "initializer_bank": bank,
        "legal_candidate_count": 256,
        "legal_family_counts": {"REPAIR_GROUP": 128, "RESTART": 128},
        "legal_opcode_counts": aggregate_opcodes,
        "minimum_independent_lineages": 128,
        "missing_required_families": [],
        "opcode_counts": aggregate_opcodes,
        "pass": True,
        "plan_record_digest": plan_identity["record_digest"],
        "plan_sha256": plan_identity["raw_sha256"],
        "protocol": "authenticated-k2-full-legal-support-v1",
        "required_families": ["REPAIR_GROUP", "RESTART"],
        "rows": rows,
        "sampled_independent_lineages": 128,
        "sampled_lineages_digest": content_digest(sorted(lineages)),
        "schema": "isingfold.gate2-authenticated-full-support",
        "schema_version": 1,
        "target_accessed": False,
    }
    return {**body, "record_digest": content_digest(body)}


def _quality_cli_args(
    path: str = "publisher-attestation.json",
    *,
    include_ground_root: bool = True,
) -> list[str]:
    arguments = [
        "--quality-attestation",
        path,
        "--expected-quality-attestation-digest",
        "a" * 64,
        "--expected-quality-publisher-id",
        "test-publisher",
    ]
    if include_ground_root:
        arguments.extend(
            [
                "--ground-certificate-root",
                "ground-certificate-root.json",
                "--expected-ground-certificate-root-sha256",
                "b" * 64,
            ]
        )
    return arguments


def test_ppo_task_schedule_weights_base_lineages_not_variant_multiplicity() -> None:
    sparse = [
        _prepared_task("a-0", "lineage-a", "train"),
        _prepared_task("b-0", "lineage-b", "train"),
    ]
    dense = [
        *(_prepared_task(f"a-{index}", "lineage-a", "train") for index in range(7)),
        _prepared_task("b-0", "lineage-b", "train"),
    ]
    sparse_schedule = _LineageEqualTaskScheduler(sparse, seed=1701)
    dense_schedule = _LineageEqualTaskScheduler(dense, seed=1701)

    sparse_lineages = [
        sparse_schedule.select(index).task.lineage for index in range(28)
    ]
    dense_lineages = [dense_schedule.select(index).task.lineage for index in range(28)]

    assert sparse_lineages == dense_lineages
    assert sparse_lineages == ["lineage-a", "lineage-b"] * 14
    assert dense_schedule.receipt["lineage_task_counts"] == {
        "lineage-a": 7,
        "lineage-b": 1,
    }
    assert sparse_schedule.receipt["rule_digest"] == dense_schedule.receipt["rule_digest"]
    assert sparse_schedule.digest != dense_schedule.digest
    selected_a = [
        dense_schedule.select(index).task_id
        for index in range(0, 28, 2)
    ]
    assert sorted(set(selected_a)) == [f"a-{index}" for index in range(7)]
    assert all(selected_a.count(task_id) == 2 for task_id in set(selected_a))


def test_scalable_gate_sample_is_order_invariant_and_lineage_equal() -> None:
    tasks = [
        _prepared_task("a-0", "lineage-a", "val"),
        _prepared_task("a-1", "lineage-a", "val"),
        _prepared_task("b-0", "lineage-b", "val"),
        _prepared_task("c-0", "lineage-c", "val"),
    ]

    forward, forward_receipt = _stratified_gate_sample(tasks, count=3, seed=907)
    reverse, reverse_receipt = _stratified_gate_sample(
        list(reversed(tasks)), count=3, seed=907
    )

    assert [item.task_id for item in forward] == [item.task_id for item in reverse]
    assert forward_receipt == reverse_receipt
    assert len({item.task.lineage for item in forward}) == 3


def test_exact_conformance_registry_requires_external_pin_and_exact_eight_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isingfold.rl.gates import (
        EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES,
        EXACT_CONFORMANCE_MAX_TASKS,
    )

    prepared = [
        _prepared_task("exact-a", "lineage-a", "val"),
        _prepared_task("exact-b", "lineage-b", "val"),
    ]
    source_digest = "9" * 64
    payload = {
        "schema": "isingfold.exact-conformance-corpus",
        "schema_version": 1,
        "corpus_id": "if-gate1-exact-v1",
        "purpose": "bounded-independent-structural-and-program-conformance",
        "source_corpus_manifest_sha256": source_digest,
        "task_ids": ["exact-a", "exact-b"],
        "max_tasks": EXACT_CONFORMANCE_MAX_TASKS,
        "max_logical_variables_per_task": EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES,
    }
    record = {**payload, "record_digest": content_digest(payload)}
    path = tmp_path / "exact-conformance.json"
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest_payload = {
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": 4,
    }
    manifest = {**manifest_payload, "record_digest": content_digest(manifest_payload)}
    (corpus / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    for filename in (
        "initializers.jsonl",
        "policy_instances.jsonl",
        "provenance.jsonl",
        "splits.json",
    ):
        (corpus / filename).write_bytes(b"")
    manifest_sha = hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest()
    payload["source_corpus_manifest_sha256"] = manifest_sha
    record = {**payload, "record_digest": content_digest(payload)}
    path.write_bytes(canonical_json_bytes(record) + b"\n")
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "isingfold.rl.data.exact_conformance.load_prepared_partition",
        lambda *args, **kwargs: SimpleNamespace(tasks=tuple(prepared), target_access=None),
    )

    with pytest.raises(ValueError, match="exactly eight"):
        _load_exact_conformance_tasks(
            path,
            expected_sha256=file_sha,
            corpus=corpus,
            expected_corpus_manifest_sha256=manifest_sha,
        )
    with pytest.raises(ValueError, match="externally pinned"):
        _load_exact_conformance_tasks(
            path,
            expected_sha256="0" * 64,
            corpus=corpus,
            expected_corpus_manifest_sha256=manifest_sha,
        )


def test_console_parser_separates_smoke_generation_from_production_inputs() -> None:
    parser = build_parser()
    smoke = parser.parse_args(["dev-generate", "--out", "scratch"])

    assert smoke.command == "dev-generate"
    assert "generate" not in parser._subparsers._group_actions[0].choices
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--corpus", "prepared", "--out", "run", "--updates", "1"])
    production = parser.parse_args(
        [
            "train",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--out",
            "run",
            "--method",
            "ppo-from-scratch",
            "--updates",
            "1",
        ]
    )
    assert production.reward_reads == 256
    assert production.episodes == 64
    assert production.ppo_epochs == 4
    assert production.minibatch == 256
    assert production.model_family == "if-core"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--corpus",
                "prepared",
                "--selector",
                "selector",
                "--out",
                "run",
                "--architecture",
                "if-core",
                "--updates",
                "1",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--corpus",
                "prepared",
                "--selector",
                "selector",
                "--out",
                "run",
                "--updates",
                "1",
                "--complete-system-no-restart",
            ]
        )


def test_complete_system_training_dispatches_all_selected_seed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_runtime_registry = runtime_implementation_registry()
    preflight_path = tmp_path / "quality-preflight.json"
    preflight_path.write_text("fixture-quality-preflight\n")
    preflight_sha256 = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    selected = FrozenRLValueSelection(
        receipt_sha256="a" * 64,
        record_digest="b" * 64,
        grid_manifest_sha256=hashlib.sha256(
            (ROOT / "configs" / "rl_grid_hybrid_v1.json").read_bytes()
        ).hexdigest(),
        model_family="if-dual",
        grid_model_family="selected-simpler",
        method="ppo-warm-start",
        training_seeds=(1103, 2207, 3301),
        cell_ids=(
            "rl-003-selected-simpler-warm-ppo-s1103",
            "rl-004-selected-simpler-warm-ppo-s2207",
            "rl-005-selected-simpler-warm-ppo-s3301",
        ),
        checkpoint_payload_digests=("c" * 64, "d" * 64, "e" * 64),
        representation_selection_sha256="f" * 64,
        representation_selection_record_digest="1" * 64,
        runtime_implementation_registry=selection_runtime_registry,
        runtime_implementation_digest=content_digest(selection_runtime_registry),
        quality_preflight_receipt_sha256=preflight_sha256,
        quality_preflight_record_digest="3" * 64,
    )
    monkeypatch.setattr(
        "isingfold.rl.experiment_selection.load_rl_value_freeze",
        lambda **kwargs: selected,
    )
    warm_captured = []
    train_captured = []
    monkeypatch.setattr("isingfold.rl.cli.cmd_warm_start", warm_captured.append)
    monkeypatch.setattr("isingfold.rl.cli.cmd_train", train_captured.append)
    monkeypatch.setattr(
        "isingfold.rl.cli._context", lambda *args, **kwargs: Context(qubit_cap=64)
    )
    monkeypatch.setattr(
        "isingfold.rl.cli._load_selector_bundle", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "isingfold.rl.cli._load_quality_preflight_receipt",
        lambda *args, **kwargs: {"record_digest": "3" * 64},
    )
    args = build_parser().parse_args(
        [
            "complete-system-train-cell",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--corpus",
            "prepared-v2",
            "--selector",
            "selector-v2",
            *_quality_cli_args(),
            "--quality-labels",
            str(tmp_path / "quality-v4"),
            "--quality-initializer-bank",
            str(tmp_path / "quality-initializer-bank"),
            "--expected-quality-initializer-bank-manifest-sha256",
            "8" * 64,
            "--quality-complete-config",
            str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
            "--quality-preflight-receipt",
            str(preflight_path),
            "--expected-quality-preflight-sha256",
            preflight_sha256,
            "--run-root",
            str(tmp_path / "confirmation"),
            "--initializer-bank",
            str(tmp_path / "initializer-bank-seed-1"),
            "--expected-initializer-bank-manifest-sha256",
            "9" * 64,
            "--complete-config",
            str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
            "--rl-value-selection-receipt",
            "rl-freeze.json",
            "--expected-selection-sha256",
            "a" * 64,
            "--index",
            "1",
            "--device",
            "cpu",
        ]
    )

    cmd_complete_system_train_cell(args)

    warm_dispatched = warm_captured[0]
    dispatched = train_captured[0]
    assert warm_dispatched.seed == 2207
    assert warm_dispatched.resume is False
    assert warm_dispatched.quality_labels == str(tmp_path / "quality-v4")
    assert warm_dispatched.quality_initializer_bank == str(
        tmp_path / "quality-initializer-bank"
    )
    assert warm_dispatched.expected_quality_initializer_bank_manifest_sha256 == "8" * 64
    assert warm_dispatched.quality_complete_config == str(
        ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"
    )
    assert warm_dispatched.out.endswith(
        "complete_system_representation/rl-004-selected-simpler-warm-ppo-s2207"
    )
    assert dispatched.seed == 2207
    assert dispatched.model_family == "if-dual"
    assert dispatched.method == "ppo-warm-start"
    assert dispatched.complete_system_no_restart is False
    assert dispatched.deployment_initializer_bank is True
    assert dispatched.initializer_bank == str(tmp_path / "initializer-bank-seed-1")
    assert dispatched.expected_initializer_bank_manifest_sha256 == "9" * 64
    assert dispatched.complete_config == str(
        ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"
    )
    assert dispatched.complete_selection_binding["all_training_seeds"] == [1103, 2207, 3301]
    assert dispatched.complete_selection_binding["source_checkpoint_payload_digest"] == "d" * 64
    assert dispatched.resume is False
    assert dispatched.complete_selection_binding["selected_validation_checkpoint_reused"] is False
    assert dispatched.warm_start == str(Path(warm_dispatched.out) / "checkpoint.pt")
    assert dispatched.warm_start_grid_cell == "rep-004-if-dual-s2207"
    assert dispatched.warm_start_checkpoint_payload_digest is None

    Path(dispatched.out).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="fresh-only"):
        cmd_complete_system_train_cell(args)
    assert len(warm_captured) == len(train_captured) == 1


def test_no_restart_training_cannot_bypass_three_seed_selection() -> None:
    with pytest.raises(ValueError, match="complete-system-train-cell"):
        cmd_train(
            SimpleNamespace(
                method="ppo-from-scratch",
                updates=1,
                episodes=1,
                ppo_epochs=1,
                minibatch=1,
                complete_system_no_restart=True,
            )
        )


def test_direct_ppo_training_cannot_select_legacy_online_restarts() -> None:
    with pytest.raises(ValueError, match="authenticated initializer bank"):
        cmd_train(
            SimpleNamespace(
                method="ppo-from-scratch",
                updates=1,
                episodes=1,
                ppo_epochs=1,
                minibatch=1,
                complete_system_no_restart=False,
                deployment_initializer_bank=False,
            )
        )


def test_scientific_evaluation_cannot_select_legacy_online_restarts() -> None:
    with pytest.raises(ValueError, match="authenticated bootstrap bank"):
        cmd_evaluate(
            SimpleNamespace(
                partition="validation",
                bootstrap_bank=None,
                scientific_evaluation_authorization={"stage": "representation"},
            )
        )

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--corpus",
                "prepared",
                "--selector",
                "selector",
                *_quality_cli_args(),
                "--out",
                "run",
                "--method",
                "ppo-from-scratch",
                "--updates",
                "1",
                "--legacy-online-initializer-restarts",
            ]
        )


def test_no_restart_ppo_requires_an_externally_pinned_initializer_bank() -> None:
    with pytest.raises(ValueError, match="initializer bank"):
        cmd_train(
            SimpleNamespace(
                method="ppo-from-scratch",
                updates=1,
                episodes=1,
                ppo_epochs=1,
                minibatch=1,
                complete_system_no_restart=True,
                complete_selection_binding={},
                initializer_bank=None,
                expected_initializer_bank_manifest_sha256=None,
                complete_config=None,
            )
        )


def test_grid_ppo_deployment_policy_requires_an_externally_pinned_initializer_bank() -> None:
    with pytest.raises(ValueError, match="initializer bank"):
        cmd_train(
            SimpleNamespace(
                method="ppo-from-scratch",
                updates=1,
                episodes=1,
                ppo_epochs=1,
                minibatch=1,
                complete_system_no_restart=False,
                deployment_initializer_bank=True,
                complete_selection_binding=None,
                initializer_bank=None,
                expected_initializer_bank_manifest_sha256=None,
                complete_config=None,
            )
        )


def test_deployment_initializer_bank_retains_k2_restart_support() -> None:
    base = Context(qubit_cap=64)
    config = CompleteSystemConfig.from_mapping(
        json.loads((ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json").read_text())
    )

    resolved, mode = _resolve_training_policy_context(
        base,
        complete_system_no_restart=False,
        deployment_initializer_bank=True,
        complete_config=config,
    )

    assert resolved == base
    assert resolved.restart_allowance == 2
    assert resolved.quotas["restart"] > 0
    assert mode == "persistent-upfront-lac-cache-k2-v2"


def test_deployment_initializer_bank_rejects_legacy_no_restart_config() -> None:
    config = CompleteSystemConfig.from_mapping(
        json.loads((ROOT / "configs" / "complete_system_lac_hybrid_v1.json").read_text())
    )

    with pytest.raises(ValueError, match="persistent K=2 restart cache"):
        _resolve_training_policy_context(
            Context(qubit_cap=64),
            complete_system_no_restart=False,
            deployment_initializer_bank=True,
            complete_config=config,
        )


def test_preinitialization_projection_deduplicates_groups_without_replaying_incumbent() -> None:
    logical = nx.path_graph(2)
    host = nx.path_graph(3)
    problem = LogicalProblem.from_dicts({0: -1.0, 1: 1.0}, {(0, 1): -1.0})
    rows = []
    for suffix, embedding in (
        ("a", {0: frozenset({0}), 1: frozenset({1})}),
        ("b", {0: frozenset({1}), 1: frozenset({2})}),
    ):
        task = EmbeddingTask(
            f"task-{suffix}",
            logical,
            host,
            problem,
            -3.0,
            lineage="base-lineage",
            initial_embedding=embedding,
        )
        rows.append(
            SimpleNamespace(
                task=task,
                instance_id="instance-public",
                partition="test",
                prepared_schema_version=4,
                corpus_scope="production-designed-v4",
                provenance=SimpleNamespace(base_parent_lineage="base-lineage"),
                design_condition=SimpleNamespace(base_lineage_key="base-lineage"),
                reference_status="exact_proof",
                certificate_digest="a" * 64,
                evaluator_protocol_digest="b" * 64,
            )
        )

    population = _preinitialization_population_tasks(
        rows,
        manifest={
            "schema_version": 4,
            "corpus_scope": "production-designed-v4",
            "target_authority": {
                "partitions": {
                    "test": {
                        "path": "targets/test.jsonl",
                        "records": 1,
                        "sha256": "c" * 64,
                        "target_set_digest": "d" * 64,
                    }
                }
            },
        },
        partition="test",
    )

    assert len(population) == 1
    assert population[0].name == "instance-public"
    assert population[0].lineage == "base-lineage"
    assert population[0].initial_embedding is None


def test_complete_system_evaluation_parser_has_no_greedy_or_legacy_escape() -> None:
    parser = build_parser()
    common = [
        "evaluate-complete-system-cell",
        "--grid",
        "grid.json",
        "--corpus",
        "prepared-v2",
        "--selector",
        "selector-v2",
        *_quality_cli_args(),
        "--run-root",
        "runs",
        "--config",
        "complete.json",
        "--bootstrap-bank",
        "test-bootstrap-bank",
        "--expected-bootstrap-plan-sha256",
        "b" * 64,
        "--expected-bootstrap-manifest-sha256",
        "c" * 64,
        "--out",
        "evaluation",
        "--rl-value-selection-receipt",
        "freeze.json",
        "--expected-selection-sha256",
        "a" * 64,
        "--index",
        "0",
    ]
    parsed = parser.parse_args(common)

    assert not hasattr(parsed, "partition")
    assert not hasattr(parsed, "repetitions")
    assert not hasattr(parsed, "seed")
    assert not hasattr(parsed, "greedy")
    assert not hasattr(parsed, "allow_legacy_pilot")
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--greedy"])
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--allow-legacy-pilot"])
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--partition", "validation"])
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--seed", "1"])


def test_ground_certificate_preflight_parser_requires_closed_verifier_pins() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "verify-ground-certificates",
            "--corpus",
            "prepared-v3",
            *_quality_cli_args(include_ground_root=False),
            "--verifier-executable",
            "verifier",
            "--expected-verifier-executable-sha256",
            "a" * 64,
            "--verifier-source",
            "verifier.py",
            "--expected-verifier-source-sha256",
            "b" * 64,
            "--verifier-environment",
            "environment.lock",
            "--expected-verifier-environment-sha256",
            "c" * 64,
            "--expected-verifier-name",
            "isingfold-ground-verifier",
            "--expected-verifier-version",
            "1.0",
            "--verifier-execution-mode",
            "static-elf",
            "--verifier-build-attestation",
            "verifier-build-attestation.json",
            "--expected-verifier-build-attestation-sha256",
            "d" * 64,
            "--out",
            "ground-certificates",
        ]
    )

    assert parsed.func.__name__ == "cmd_verify_ground_certificates"
    assert parsed.verifier_timeout_seconds == 30.0


def test_complete_system_aggregate_parser_has_no_seed_or_protocol_overrides() -> None:
    parser = build_parser()
    common = [
        "aggregate-complete-system",
        "--grid",
        str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
        "--corpus",
        "prepared-v2",
        *_quality_cli_args(),
        "--evaluation-root",
        "complete-evaluations",
        "--config",
        "complete-config.json",
        "--bootstrap-bank",
        "final-test-bootstrap-bank",
        "--expected-bootstrap-plan-sha256",
        "c" * 64,
        "--expected-bootstrap-manifest-sha256",
        "d" * 64,
        "--rl-value-selection-receipt",
        "rl-freeze.json",
        "--expected-selection-sha256",
        "b" * 64,
        "--out",
        "complete-aggregate.json",
    ]

    parsed = parser.parse_args(common)
    assert parsed.command == "aggregate-complete-system"
    assert parsed.evaluation_root == "complete-evaluations"
    for forbidden in (
        ["--index", "0"],
        ["--seed", "1"],
        ["--partition", "validation"],
        ["--repetitions", "1"],
        ["--bootstrap-replicates", "10"],
        ["--allow-legacy-pilot"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([*common, *forbidden])


def test_selector_labelling_and_fitting_are_separate_authenticated_stages() -> None:
    parser = build_parser()
    labels = parser.parse_args(
        [
            "label-selector-data",
            "--corpus",
            "prepared",
            *_quality_cli_args(),
            "--out",
            "selector-labels",
        ]
    )
    fit = parser.parse_args(
        [
            "fit-selector",
            "--corpus",
            "prepared",
            "--selector-labels",
            "selector-labels",
            *_quality_cli_args(),
            "--out",
            "selector",
        ]
    )

    assert labels.calibration_fraction == pytest.approx(0.2)
    assert labels.audit_mode is False
    assert fit.selector_labels == "selector-labels"
    assert not hasattr(fit, "selector_reads")

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["fit-selector", "--corpus", "prepared", "--out", "selector"]
        )


def test_prepare_release_v1_wires_multiple_corpora_and_certificate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[object, ...]] = []

    def fake(*args, **kwargs):
        captured.append((*args, kwargs))
        return {"schema": "prepared"}

    monkeypatch.setattr(
        "isingfold.rl.data.import_release_v1.prepare_release_v1", fake
    )
    args = build_parser().parse_args(
        [
            "prepare-release-v1",
            "--quality-corpus",
            "quality_inkdrop.jsonl",
            "--quality-corpus",
            "quality_random.jsonl",
            "--split-manifest",
            "splits_quality_problem_v2.json",
            "--checksums",
            "SHA256SUMS",
            "--problem-references",
            "problem_references.jsonl",
            "--out",
            "prepared-release",
            "--qubit-cap",
            "160",
        ]
    )

    args.func(args)

    assert captured == [
        (
            ["quality_inkdrop.jsonl", "quality_random.jsonl"],
            "splits_quality_problem_v2.json",
            "SHA256SUMS",
            "problem_references.jsonl",
            "prepared-release",
            {"qubit_cap": 160},
        )
    ]


def test_prepare_publishes_partition_sealed_v4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[object, ...]] = []

    def fake(*args, **kwargs):
        captured.append((*args, kwargs))
        return {"schema_version": 4}

    monkeypatch.setattr("isingfold.rl.cli.prepare_candidate_bank_v4", fake)
    args = build_parser().parse_args(
        [
            "prepare",
            "--bank",
            "candidate-bank.jsonl",
            "--bank-manifest",
            "candidate-bank-manifest.json",
            "--evaluator-targets",
            "evaluator-targets.jsonl",
            "--provenance",
            "provenance.jsonl",
            "--corpus-design-manifest",
            "corpus-design.json",
            "--expected-corpus-design-sha256",
            "a" * 64,
            "--out",
            "prepared-v4",
            "--qubit-cap",
            "160",
        ]
    )

    args.func(args)

    assert captured == [
        (
            "candidate-bank.jsonl",
            "candidate-bank-manifest.json",
            "evaluator-targets.jsonl",
            "provenance.jsonl",
            "corpus-design.json",
            "prepared-v4",
            {"expected_corpus_design_sha256": "a" * 64, "qubit_cap": 160},
        )
    ]


def test_legacy_pilot_override_is_explicit_and_absent_from_scientific_grid() -> None:
    parser = build_parser()
    diagnostic = parser.parse_args(
        [
            "label-quality",
            "--corpus",
            "prepared-v1",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--out",
            "labels",
            "--allow-legacy-pilot",
        ]
    )
    assert diagnostic.allow_legacy_pilot is True

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "grid-cell",
                "--grid",
                "grid.json",
                "--stage",
                "representation",
                "--index",
                "0",
                "--corpus",
                "prepared-v1",
                "--selector",
                "selector",
                "--quality-labels",
                "labels",
                "--run-root",
                "runs",
                "--allow-legacy-pilot",
            ]
        )


def test_final_audit_reads_are_independent_from_fixed_training_reward_reads() -> None:
    parser = build_parser()
    evaluation = parser.parse_args(
        [
            "evaluate",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--checkpoint",
            "checkpoint.pt",
            "--out",
            "audit",
        ]
    )
    high_read_evaluation = parser.parse_args(
        [
            "evaluate",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--checkpoint",
            "checkpoint.pt",
            "--out",
            "audit",
            "--audit-reads",
            "8192",
        ]
    )

    assert evaluation.audit_reads == 4096
    assert high_read_evaluation.audit_reads == 8192
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--corpus",
                "prepared",
                "--selector",
                "selector",
                "--out",
                "run",
                "--updates",
                "1",
                "--reward-reads",
                "512",
            ]
        )


@pytest.mark.parametrize(
    ("command", "runner"),
    (("evaluate", cmd_evaluate), ("evaluate-external", cmd_evaluate_external)),
)
def test_generic_evaluation_cannot_open_sealed_test_before_any_artifact_access(
    command: str,
    runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[bool] = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        touched.append(True)
        raise AssertionError("sealed test artifacts were opened")

    monkeypatch.setattr("isingfold.rl.cli._load_quality_partition", forbidden)
    common = [
        command,
        "--corpus",
        "prepared",
        "--selector",
        "selector",
        *_quality_cli_args(),
        "--partition",
        "test",
        "--out",
        "diagnostic",
    ]
    if command == "evaluate":
        common.extend(("--checkpoint", "checkpoint.pt"))
    else:
        common.extend(("--config", "external.json"))
    args = build_parser().parse_args(common)

    with pytest.raises(PermissionError, match="sealed test"):
        runner(args)
    assert touched == []


def test_context_snapshot_handles_immutable_quota_maps() -> None:
    context = Context(qubit_cap=24)
    snapshot = _context_snapshot(context)

    assert snapshot["qubit_cap"] == 24
    assert snapshot["quotas"]["single"] == 16
    assert snapshot["construction_quotas"]["place"] == 24
    assert snapshot["strength_ratios"] == [0.5, 1.0, 2.0, 4.0]
    json.dumps(snapshot, allow_nan=False)


def test_runtime_seed_replays_python_numpy_and_torch() -> None:
    _seed_runtime(1701, deterministic=True, threads=1)
    first = (random.random(), float(np.random.random()), torch.rand(3))
    _seed_runtime(1701, deterministic=True, threads=1)
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])
    assert _resolve_device("cpu") == torch.device("cpu")

    _seed_runtime(1701, deterministic=False, threads=1)
    assert torch.are_deterministic_algorithms_enabled() is False
    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True


def test_registered_grid_has_nine_then_eighteen_cells() -> None:
    grid, digest = _load_grid(ROOT / "configs" / "rl_grid_hybrid_v1.json")
    representation = grid["stages"]["representation"]
    rl_value = grid["stages"]["rl_value"]

    assert grid["schema_version"] == 2
    assert grid["name"] == "if-core-v2-profile-i-hybrid-chimera-registered"
    assert len(representation) == 9
    assert len(rl_value) == 18
    assert {cell["model_family"] for cell in representation} == set(
        REGISTERED_MODEL_FAMILIES
    )
    assert {cell["method"] for cell in rl_value} == {
        "supervised-only",
        "ppo-warm-start",
        "ppo-from-scratch",
    }
    assert grid["quality_policy_prior"] == {
        "schema": "isingfold.action-quality-policy-coupling-v1",
        "mode": "bounded-centered-v1",
        "transform": "two-sigmoid-minus-one",
        "output_interval": [-1.0, 1.0],
        "policy_gradient_to_quality_head": True,
        "publication_eligible": True,
    }
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda grid: grid.pop("quality_policy_prior"), "quality-policy prior"),
        (
            lambda grid: grid["quality_policy_prior"].update(
                {"transform": "identity"}
            ),
            "implementation contract",
        ),
        (
            lambda grid: grid["quality_policy_prior"].update(
                {
                    "mode": "raw-logit-diagnostic-v1",
                    "transform": "identity",
                    "output_interval": None,
                    "policy_gradient_to_quality_head": True,
                    "publication_eligible": False,
                }
            ),
            "diagnostic quality prior",
        ),
        (
            lambda grid: grid["stages"]["representation"][0].update(
                {"quality_prior_mode": "auxiliary-only-v1"}
            ),
            "per-cell overrides",
        ),
    ],
)
def test_grid_loader_fails_closed_on_quality_policy_prior_contract(
    tmp_path: Path, mutation, message: str
) -> None:
    grid = json.loads((ROOT / "configs" / "rl_grid_hybrid_v1.json").read_text())
    mutation(grid)
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(grid))

    with pytest.raises(ValueError, match=message):
        _load_grid(path)


def test_rl_value_grid_ppo_binds_a_seed_matched_deployment_initializer_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = tmp_path / "quality-preflight.json"
    preflight.write_text("fixture\n")
    captured = []
    monkeypatch.setattr("isingfold.rl.cli._context", lambda *args: Context(qubit_cap=4))
    monkeypatch.setattr("isingfold.rl.cli._load_selector_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "isingfold.rl.cli._load_quality_preflight_receipt",
        lambda *args, **kwargs: {"record_digest": "c" * 64},
    )
    monkeypatch.setattr(
        "isingfold.rl.cli._load_representation_selection",
        lambda *args, **kwargs: ("if-dual", "d" * 64, "e" * 64),
    )
    monkeypatch.setattr(
        "isingfold.rl.cli._representation_checkpoint_payload_digest",
        lambda *args, **kwargs: "7" * 64,
    )
    monkeypatch.setattr("isingfold.rl.cli.cmd_train", captured.append)

    args = build_parser().parse_args(
        [
            "grid-cell",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--stage",
            "rl_value",
            "--index",
            "3",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--quality-labels",
            "quality",
            "--quality-initializer-bank",
            str(tmp_path / "quality-initializer-bank"),
            "--expected-quality-initializer-bank-manifest-sha256",
            "8" * 64,
            "--quality-complete-config",
            str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
            "--quality-preflight-receipt",
            str(preflight),
            "--expected-quality-preflight-sha256",
            "a" * 64,
            "--run-root",
            str(tmp_path / "runs"),
            "--selection-receipt",
            "representation-selection.json",
            "--initializer-bank",
            str(tmp_path / "initializer-banks" / "seed-0"),
            "--expected-initializer-bank-manifest-sha256",
            "f" * 64,
            "--complete-config",
            str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
        ]
    )
    args.func(args)

    dispatched = captured[0]
    assert dispatched.seed == 1103
    assert dispatched.method == "ppo-warm-start"
    assert dispatched.deployment_initializer_bank is True
    assert dispatched.initializer_bank == str(tmp_path / "initializer-banks" / "seed-0")
    assert dispatched.expected_initializer_bank_manifest_sha256 == "f" * 64
    assert dispatched.complete_config == str(
        ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"
    )
    assert dispatched.quality_initializer_bank == str(
        tmp_path / "quality-initializer-bank"
    )
    assert dispatched.expected_quality_initializer_bank_manifest_sha256 == "8" * 64
    assert dispatched.quality_complete_config == str(
        ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"
    )
    assert dispatched.quality_prior_mode == "bounded-centered-v1"
    assert dispatched.warm_start_grid_cell == "rep-003-if-dual-s1103"
    assert dispatched.warm_start_checkpoint_payload_digest == "7" * 64


def test_rl_value_grid_rejects_missing_deployment_initializer_bank() -> None:
    args = build_parser().parse_args(
        [
            "grid-cell",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--stage",
            "rl_value",
            "--index",
            "3",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--quality-labels",
            "quality",
            "--quality-preflight-receipt",
            "quality-preflight.json",
            "--expected-quality-preflight-sha256",
            "a" * 64,
            "--run-root",
            "runs",
            "--selection-receipt",
            "representation-selection.json",
        ]
    )

    with pytest.raises(ValueError, match="deployment initializer bank"):
        args.func(args)


def test_rl_value_grid_rejects_a_deployment_config_outside_its_registered_pins(
    tmp_path: Path,
) -> None:
    config = json.loads(
        (ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json").read_text()
    )
    config["online_wallclock_seconds"] = 61.0
    tampered = tmp_path / "complete-system.json"
    tampered.write_text(json.dumps(config))
    args = build_parser().parse_args(
        [
            "grid-cell",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--stage",
            "rl_value",
            "--index",
            "3",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--quality-labels",
            "quality",
            "--quality-preflight-receipt",
            "quality-preflight.json",
            "--expected-quality-preflight-sha256",
            "a" * 64,
            "--run-root",
            "runs",
            "--selection-receipt",
            "representation-selection.json",
            "--initializer-bank",
            "initializer-banks/seed-0",
            "--expected-initializer-bank-manifest-sha256",
            "f" * 64,
            "--complete-config",
            str(tampered),
        ]
    )

    with pytest.raises(ValueError, match="preregistered semantic/file pins"):
        args.func(args)


def test_representation_grid_requires_and_binds_passing_gate_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = [
        "grid-cell",
        "--grid",
        str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
        "--stage",
        "representation",
        "--index",
        "0",
        "--corpus",
        "prepared",
        "--selector",
        "selector",
        *_quality_cli_args(),
        "--quality-labels",
        "quality",
        "--quality-initializer-bank",
        "quality-initializer-bank",
        "--expected-quality-initializer-bank-manifest-sha256",
        "8" * 64,
        "--quality-complete-config",
        str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
        "--quality-preflight-receipt",
        str(tmp_path / "quality-preflight.json"),
        "--expected-quality-preflight-sha256",
        "a" * 64,
        "--run-root",
        str(tmp_path / "runs"),
    ]
    with pytest.raises(ValueError, match="gate receipt"):
        missing = build_parser().parse_args(base)
        missing.func(missing)

    captured = []
    gate_requirements = []
    gate_pins = []
    monkeypatch.setattr("isingfold.rl.cli._context", lambda *args: Context(qubit_cap=4))
    (tmp_path / "quality-preflight.json").write_text("fixture\n")
    monkeypatch.setattr("isingfold.rl.cli._load_selector_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "isingfold.rl.cli._load_quality_preflight_receipt",
        lambda *args, **kwargs: {"record_digest": "c" * 64},
    )

    def load_gate(*args, **kwargs):
        del args
        gate_requirements.append(kwargs["require_profile_c"])
        gate_pins.append(kwargs["expected_sha256"])
        return "a" * 64, "b" * 64

    monkeypatch.setattr("isingfold.rl.cli._load_gate_receipt", load_gate)
    monkeypatch.setattr("isingfold.rl.cli.cmd_warm_start", captured.append)
    unpinned = build_parser().parse_args([*base, "--gate-receipt", "gates.json"])
    with pytest.raises(ValueError, match="externally pinned gate receipt"):
        unpinned.func(unpinned)

    args = build_parser().parse_args(
        [
            *base,
            "--gate-receipt",
            "gates.json",
            "--expected-gate-receipt-sha256",
            "d" * 64,
        ]
    )
    args.func(args)

    assert captured[0].gate_receipt_sha256 == "a" * 64
    assert captured[0].gate_record_digest == "b" * 64
    assert captured[0].gate_profile == "profile-i"
    assert captured[0].quality_initializer_bank == "quality-initializer-bank"
    assert captured[0].expected_quality_initializer_bank_manifest_sha256 == "8" * 64
    assert captured[0].quality_complete_config == str(
        ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"
    )
    assert gate_requirements == [False]
    assert gate_pins == ["d" * 64]

    explicit = build_parser().parse_args(
        [
            *base,
            "--gate-receipt",
            "gates.json",
            "--expected-gate-receipt-sha256",
            "d" * 64,
            "--require-profile-c-gate",
        ]
    )
    explicit.func(explicit)
    assert captured[1].gate_profile == "profile-i+profile-c"
    assert gate_requirements == [False, True]
    assert gate_pins == ["d" * 64, "d" * 64]


def test_gate_receipt_accepts_failed_profile_c_only_for_profile_i_scope(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    selector = tmp_path / "selector"
    corpus.mkdir()
    selector.mkdir()
    (corpus / "manifest.json").write_text("{}\n")

    selector_payload = {
        "schema": "fixture-selector",
        "selector_digest": "a" * 64,
        "quality_authority": _authority(),
    }
    selector_receipt = {
        **selector_payload,
        "record_digest": content_digest(selector_payload),
    }
    selector_path = selector / "fit_receipt.json"
    selector_path.write_bytes(canonical_json_bytes(selector_receipt) + b"\n")

    mandatory_names = [
        "gate_1_exact_conformance",
        "gate_2_authenticated_k2_full_support",
        "gate_3_selector_discrimination",
        "gate_4_profile_i_signal",
    ]
    selected_gate_strata = ["f" * 64, "f" * 64]
    context = Context(qubit_cap=4)
    corpus_manifest_sha256 = hashlib.sha256(
        (corpus / "manifest.json").read_bytes()
    ).hexdigest()
    exact_gate = _exact_gate_result(corpus_manifest_sha256)
    resolution_plan_identity = {
        "initializer_bank_contract_record_digest": _full_support_bank_contract()[
            "record_digest"
        ],
        "initializer_bank_manifest_sha256": _full_support_bank_contract()[
            "manifest_sha256"
        ],
        "raw_sha256": "8" * 64,
        "record_digest": "9" * 64,
    }
    payload = {
        "schema": "isingfold.release-gates",
        "schema_version": 7,
        "source_corpus_manifest_sha256": corpus_manifest_sha256,
        "quality_authority": {
            **{
                key: value
                for key, value in _authority().items()
                if key != "training_partition" and key != "record_digest"
            },
            "evaluation_partition": _authority(audit_mode=True)["audit_partitions"][
                "val"
            ],
        },
        "source_selector_digest": "a" * 64,
        "source_selector_fit_receipt_sha256": hashlib.sha256(
            selector_path.read_bytes()
        ).hexdigest(),
        "source_selector_labels_manifest_sha256": "d" * 64,
        "source_selector_labels_manifest_record_digest": "e" * 64,
        "context": _context_snapshot(context),
        "context_digest": content_digest(_context_snapshot(context)),
        "partition": "validation",
        "seed": 17,
        "parameters": {
            "instances_requested": 2,
            "reward_reads": 256,
            "profile_episodes_per_instance": 2,
            "broad_reference_batches_per_instance": 8,
            "exact_conformance_corpus": exact_gate[
                "authenticated_exact_corpus"
            ],
            "quality_resolution_plan": resolution_plan_identity,
            "scalable_gate_sampling": {
                "selected_stratum_digests": selected_gate_strata,
            },
        },
        "gate_profile": "profile-i",
        "grid_manifest_sha256": "4" * 64,
        "quality_preflight_receipt_sha256": "5" * 64,
        "quality_preflight_record_digest": "6" * 64,
        "mandatory_gate_names": mandatory_names,
        "gates": {
            mandatory_names[0]: exact_gate,
            mandatory_names[1]: _full_support_gate_result(
                resolution_plan_identity
            ),
            **{name: {"pass": True} for name in mandatory_names[2:]},
        },
        "optional_diagnostics": {
            "profile_c_construction_readiness": {"pass": False},
            "validation_support_headroom": _support_gate_result(
                selected_gate_strata
            ),
        },
        "construction_ready": False,
        "advance": True,
    }
    payload["quality_authority"]["record_digest"] = content_digest(
        payload["quality_authority"]
    )
    payload["target_access"] = _target_access("val", 1)
    payload["ground_partition_receipt"] = _ground_partition_receipt("val", 1)
    receipt = {**payload, "record_digest": content_digest(payload)}
    receipt_path = tmp_path / "gates.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    _, profile_i_digest = _load_gate_receipt(
        receipt_path,
        expected_sha256=receipt_sha256,
        corpus=corpus,
        selector=selector,
        context=context,
        expected_grid_manifest_sha256="4" * 64,
        expected_quality_preflight_sha256="5" * 64,
        expected_quality_preflight_record_digest="6" * 64,
    )

    assert len(profile_i_digest) == 64
    assert profile_i_digest != receipt["record_digest"]

    bare_gate_payload = deepcopy(payload)
    bare_gate_payload["gates"][mandatory_names[1]] = {"pass": True}
    bare_gate_receipt = {
        **bare_gate_payload,
        "record_digest": content_digest(bare_gate_payload),
    }
    bare_gate_path = tmp_path / "bare-gate2.json"
    bare_gate_path.write_bytes(canonical_json_bytes(bare_gate_receipt) + b"\n")
    with pytest.raises(ValueError, match="Gate 2"):
        _load_gate_receipt(
            bare_gate_path,
            expected_sha256=hashlib.sha256(bare_gate_path.read_bytes()).hexdigest(),
            corpus=corpus,
            selector=selector,
            context=context,
            expected_grid_manifest_sha256="4" * 64,
            expected_quality_preflight_sha256="5" * 64,
            expected_quality_preflight_record_digest="6" * 64,
        )
    with pytest.raises(ValueError, match="Profile-C"):
        _load_gate_receipt(
            receipt_path,
            expected_sha256=receipt_sha256,
            corpus=corpus,
            selector=selector,
            context=context,
            require_profile_c=True,
            expected_grid_manifest_sha256="4" * 64,
            expected_quality_preflight_sha256="5" * 64,
            expected_quality_preflight_record_digest="6" * 64,
        )
    profile_c_payload = {
        **payload,
        "optional_diagnostics": {
            "profile_c_construction_readiness": {"pass": True},
            "validation_support_headroom": _support_gate_result(
                selected_gate_strata
            ),
        },
        "construction_ready": True,
    }
    profile_c_receipt = {
        **profile_c_payload,
        "record_digest": content_digest(profile_c_payload),
    }
    profile_c_path = tmp_path / "gates-with-profile-c.json"
    profile_c_path.write_bytes(canonical_json_bytes(profile_c_receipt) + b"\n")
    profile_c_sha256 = hashlib.sha256(profile_c_path.read_bytes()).hexdigest()
    _, combined_digest = _load_gate_receipt(
        profile_c_path,
        expected_sha256=profile_c_sha256,
        corpus=corpus,
        selector=selector,
        context=context,
        require_profile_c=True,
        expected_grid_manifest_sha256="4" * 64,
        expected_quality_preflight_sha256="5" * 64,
        expected_quality_preflight_record_digest="6" * 64,
    )

    assert len(combined_digest) == 64
    assert combined_digest != profile_i_digest

    with pytest.raises(ValueError, match="externally pinned"):
        _load_gate_receipt(
            receipt_path,
            expected_sha256="0" * 64,
            corpus=corpus,
            selector=selector,
            context=context,
            expected_grid_manifest_sha256="4" * 64,
            expected_quality_preflight_sha256="5" * 64,
            expected_quality_preflight_record_digest="6" * 64,
        )

    forged_payload = deepcopy(payload)
    forged_payload["gates"]["gate_1_exact_conformance"][
        "authenticated_exact_corpus"
    ]["task_ids"] = ["repeated-task"] * 8
    forged_receipt = {
        **forged_payload,
        "record_digest": content_digest(forged_payload),
    }
    forged_path = tmp_path / "forged-gates.json"
    forged_path.write_bytes(canonical_json_bytes(forged_receipt) + b"\n")
    with pytest.raises(ValueError, match="eight sorted unique task IDs"):
        _load_gate_receipt(
            forged_path,
            expected_sha256=hashlib.sha256(forged_path.read_bytes()).hexdigest(),
            corpus=corpus,
            selector=selector,
            context=context,
            expected_grid_manifest_sha256="4" * 64,
            expected_quality_preflight_sha256="5" * 64,
            expected_quality_preflight_record_digest="6" * 64,
        )


def test_gate_receipt_competing_publishers_have_one_complete_winner(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "gates.json"
    payload = {
        "schema": "gate-publication-fixture",
        "record_digest": "a" * 64,
    }

    def publish() -> object:
        try:
            return _publish_gate_receipt(destination, payload)
        except FileExistsError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: publish(), range(2)))

    assert sum(isinstance(outcome, Path) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    assert destination.read_bytes() == canonical_json_bytes(payload) + b"\n"
    assert list(tmp_path.glob(".gates.json.*.tmp")) == []


def test_representation_evaluation_cell_uses_one_shared_registered_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []
    monkeypatch.setattr("isingfold.rl.cli.cmd_evaluate", captured.append)
    common = [
        "--grid",
        str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
        "--corpus",
        "prepared",
        "--selector",
        "selector",
        *_quality_cli_args(),
        "--run-root",
        str(tmp_path / "runs"),
        "--evaluations-root",
        str(tmp_path / "evaluations"),
        "--complete-config",
        str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
        "--bootstrap-bank",
        "validation-bootstrap-bank",
        "--expected-bootstrap-plan-sha256",
        "f" * 64,
        "--expected-bootstrap-manifest-sha256",
        "0" * 64,
        "--device",
        "cpu",
    ]
    for index in (0, 8):
        args = build_parser().parse_args(
            ["evaluate-representation-cell", "--index", str(index), *common]
        )
        args.func(args)

    assert [item.seed for item in captured] == [33049, 33049]
    assert [item.partition for item in captured] == ["validation", "validation"]
    assert [item.repetitions for item in captured] == [4, 4]
    assert [item.audit_reads for item in captured] == [4096, 4096]
    assert [item.bootstrap_bank for item in captured] == [
        "validation-bootstrap-bank",
        "validation-bootstrap-bank",
    ]
    assert captured[0].expected_bootstrap_plan_sha256 == "f" * 64
    assert captured[0].expected_bootstrap_manifest_sha256 == "0" * 64
    assert captured[0].complete_config.endswith("complete_system_lac_hybrid_cache_v1.json")
    assert captured[0].out.endswith("rep-000-if-mlp-s1103")
    assert captured[1].out.endswith("rep-008-if-core-s3301")


def test_rl_value_evaluation_reconstructs_the_deployment_policy_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []
    monkeypatch.setattr(
        "isingfold.rl.cli._load_representation_selection",
        lambda *args, **kwargs: ("if-dual", "d" * 64, "e" * 64),
    )
    monkeypatch.setattr(
        "isingfold.rl.cli._validate_rl_value_run_for_evaluation",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr("isingfold.rl.cli.cmd_evaluate", captured.append)

    args = build_parser().parse_args(
        [
            "evaluate-rl-value-cell",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--index",
            "3",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            *_quality_cli_args(),
            "--run-root",
            str(tmp_path / "runs"),
            "--evaluations-root",
            str(tmp_path / "evaluations"),
            "--complete-config",
            str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
            "--bootstrap-bank",
            "validation-bootstrap-bank",
            "--expected-bootstrap-plan-sha256",
            "f" * 64,
            "--expected-bootstrap-manifest-sha256",
            "0" * 64,
            "--selection-receipt",
            "representation-selection.json",
            "--device",
            "cpu",
        ]
    )
    args.func(args)

    assert len(captured) == 1
    assert captured[0].deployment_policy_context is False
    assert captured[0].bootstrap_bank == "validation-bootstrap-bank"


def test_grid_loader_rejects_partial_stage(tmp_path: Path) -> None:
    grid = json.loads((ROOT / "configs" / "rl_grid_hybrid_v1.json").read_text())
    grid["stages"]["representation"].pop()
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(grid))

    with pytest.raises(ValueError, match="exactly 9"):
        _load_grid(path)


def test_grid_loader_rejects_retired_schema_v1(tmp_path: Path) -> None:
    grid = json.loads((ROOT / "configs" / "rl_grid_hybrid_v1.json").read_text())
    grid["schema_version"] = 1
    path = tmp_path / "retired-grid.json"
    path.write_text(json.dumps(grid))

    with pytest.raises(ValueError, match="unsupported staged-grid"):
        _load_grid(path)

    grid["schema_version"] = 2
    grid["name"] = "if-core-v1-profile-i-pilot"
    path.write_text(json.dumps(grid))
    with pytest.raises(ValueError, match="unsupported staged-grid"):
        _load_grid(path)


def test_grid_loader_requires_disjoint_stage_evaluation_seeds(tmp_path: Path) -> None:
    grid = json.loads((ROOT / "configs" / "rl_grid_hybrid_v1.json").read_text())
    grid["representation_evaluation"]["evaluation_seed"] = grid[
        "rl_value_evaluation"
    ]["evaluation_seed"]
    path = tmp_path / "overlapping-evaluation-seeds.json"
    path.write_text(json.dumps(grid))

    with pytest.raises(ValueError, match="disjoint evaluation seed domains"):
        _load_grid(path)


def test_rl_artifacts_bind_the_registered_batched_collection_schedule() -> None:
    from isingfold.rl.ppo import (
        COLLECTION_SCHEDULE,
        COLLECTION_SCHEDULE_SCHEMA,
        COLLECTION_SCHEDULE_SCHEMA_VERSION,
    )

    schedule = _ppo_collection_schedule_contract()
    assert schedule == {
        "schema": COLLECTION_SCHEDULE_SCHEMA,
        "schema_version": COLLECTION_SCHEDULE_SCHEMA_VERSION,
        "schedule": COLLECTION_SCHEDULE,
    }
    contract = {"hyperparameters": {"ppo_collection_schedule": schedule}}
    _require_current_ppo_collection_schedule(contract)

    with pytest.raises(ValueError, match="batched PPO collection schedule"):
        _require_current_ppo_collection_schedule({"hyperparameters": {}})
    retired = json.loads(json.dumps(contract))
    retired["hyperparameters"]["ppo_collection_schedule"]["schedule"] = (
        "retired-scalar-loop-v0"
    )
    with pytest.raises(ValueError, match="batched PPO collection schedule"):
        _require_current_ppo_collection_schedule(retired)


def test_rl_contract_records_hard_kl_transactions_and_entropy_floor() -> None:
    config = PPOConfig(episodes_per_batch=1)
    control = _ppo_update_control_contract(config)

    assert control["rule"] == "epoch-rollback-geometric-backtracking-v1"
    assert control["soft_target"] < control["hard_limit"]
    assert control["epoch_transaction"] == (
        "snapshot-model-optimizer-and-rng; commit-only-below-hard-limit"
    )
    assert control["maximum_backtracks_per_epoch"] == 3
    assert control["entropy_normalization"] == (
        "legal-categorical-entropy-divided-by-log-support-v1"
    )
    assert control["entropy_floor"] > 0.0

    hyperparameters = {
        field: getattr(config, field) for field in config.__dataclass_fields__
    }
    contract = {
        "hyperparameters": {
            **hyperparameters,
            "ppo_update_control": control,
        }
    }
    _require_current_ppo_update_control(contract)
    del contract["hyperparameters"]["entropy_floor"]
    with pytest.raises(ValueError, match="update-control fields"):
        _require_current_ppo_update_control(contract)


def test_training_state_round_trip_binds_contract_and_counters(tmp_path: Path) -> None:
    model = _model("if-mlp")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    normalizer = {
        "schema": "isingfold.feature-normalizer",
        "schema_version": 1,
        "coefficient_scale": 1.0,
    }
    bundle = SelectorBundle(
        model=None,
        selector_digest=_digest("a"),
        normalizer=normalizer,
        normalizer_digest=_digest("b"),
        coefficient_scale=1.0,
        corpus_manifest_sha256=_digest("c"),
        quality_authority=_authority(),
        target_access=_target_access("train", 2),
        ground_partition_receipt=_ground_partition_receipt("train", 2),
        root=tmp_path / "selector",
    )
    context = _experiment_contract(
        phase="rl-value",
        model_family="if-mlp",
        model=model,
        method="ppo-from-scratch",
        context=Context(qubit_cap=12),
        corpus_manifest_sha256=bundle.corpus_manifest_sha256,
        quality_authority=bundle.quality_authority,
        target_access=bundle.target_access,
        ground_partition_receipt=bundle.ground_partition_receipt,
        selector_digest=bundle.selector_digest,
        normalizer_digest=bundle.normalizer_digest,
        seed=11,
        device_type="cpu",
        deterministic=True,
        hyperparameters={"updates": 2},
    )
    assert context["model_family"] == "if-mlp"
    assert context["model"]["family"] == "if-mlp"
    assert context["model"]["implementation"].endswith(".IFMLP")
    assert len(context["model"]["implementation_source_sha256"]) == 64
    run = _prepare_run_dir(tmp_path / "run", resume=False)
    saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metadata = _save_training_state(
        run,
        model=model,
        optimizer=optimizer,
        contract=context,
        bundle=bundle,
        lineages=("lineage-b", "lineage-a"),
        counters={"updates": 1, "episodes": 64},
        phase="rl-value",
        model_family="if-mlp",
        method="ppo-from-scratch",
        complete=False,
        trainer_state={"lambda_f": 0.25, "updates_done": 1},
    )
    for parameter in model.parameters():
        parameter.data.add_(99.0)

    loaded = _resume_training_state(
        run,
        model=model,
        optimizer=optimizer,
        contract=context,
        bundle=bundle,
        lineages=("lineage-a", "lineage-b"),
        expected_trainer_state_schema=PPO_TRAINER_STATE_SCHEMA,
    )

    assert loaded.payload_digest == metadata.payload_digest
    run_record = json.loads((run / "run.json").read_text())
    assert run_record["runtime_implementation_digest"] == (
        metadata.runtime_implementation_digest
    )
    assert loaded.counters == {"episodes": 64, "updates": 1}
    assert loaded.trainer_state == {"lambda_f": 0.25, "updates_done": 1}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name])


def test_new_run_refuses_existing_directory_and_resume_needs_receipts(tmp_path: Path) -> None:
    path = tmp_path / "run"
    path.mkdir()
    with pytest.raises(FileExistsError):
        _prepare_run_dir(path, resume=False)
    with pytest.raises(ValueError, match="checkpoint.pt"):
        _prepare_run_dir(path, resume=True)


def test_warm_start_cli_fits_and_reports_actor_and_utility_losses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = tmp_path / "quality-preflight.json"
    preflight.write_text("authenticated fixture\n")
    quality_directory = tmp_path / "quality"
    quality_directory.mkdir()
    (quality_directory / "manifest.json").write_text("{}\n")
    authority = {"fixture": "train-only"}
    records = [
        (
            _phase_observation(),
            (0,),
            (0,),
            WarmStartUtilityTarget(
                0.2,
                2.0,
                1,
                2,
                1.0,
                WarmStartActionValueTarget((0,), (0.2,), (2,), (1.0,), 1),
            ),
        ),
        (
            _conflict_observation(),
            (1,),
            (0, 1),
            WarmStartUtilityTarget(
                0.8,
                6.0,
                2,
                8,
                1.0,
                WarmStartActionValueTarget(
                    (0, 1), (0.2, 0.8), (4, 4), (1.0, 1.0), 2, 0
                ),
            ),
        ),
        (
            _phase_observation(),
            (0,),
            (0,),
            WarmStartUtilityTarget(
                0.5,
                4.0,
                1,
                4,
                1.0,
                WarmStartActionValueTarget((0,), (0.5,), (4,), (1.0,), 1),
            ),
        ),
    ]
    quality_manifest = {
        "lineages": ["train-lineage"],
        "quality_authority": authority,
        "target_access": {"partition": "train"},
        "ground_partition_receipt": {"partition": "train"},
        "loader_summary": {"records_used": 1},
    }
    bundle = SimpleNamespace(
        corpus_manifest_sha256="a" * 64,
        quality_authority=authority,
        selector_digest="b" * 64,
        normalizer_digest="c" * 64,
    )
    model = IFCore(improvement_mode=True)
    contract_capture: list[dict[str, object]] = []
    state_capture: list[dict[str, object]] = []

    monkeypatch.setattr(
        "isingfold.rl.cli._context", lambda *_args, **_kwargs: Context(qubit_cap=4)
    )
    monkeypatch.setattr("isingfold.rl.cli._load_selector_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(
        "isingfold.rl.cli._quality_initializer_bank_for_manifest",
        lambda *_args, **_kwargs: (None, None),
    )
    monkeypatch.setattr("isingfold.rl.cli._quality_attestation_pin", lambda *_args: object())
    monkeypatch.setattr(
        "isingfold.rl.cli._load_quality_preflight_receipt",
        lambda *_args, **_kwargs: {"record_digest": "d" * 64},
    )
    monkeypatch.setattr(
        "isingfold.rl.cli._load_quality_labels",
        lambda *_args, **_kwargs: (records, quality_manifest),
    )
    monkeypatch.setattr("isingfold.rl.cli._model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr("isingfold.rl.cli._seed_runtime", lambda *_args, **_kwargs: None)

    def contract(**kwargs):
        contract_capture.append(kwargs)
        return {"fixture": "contract"}

    monkeypatch.setattr("isingfold.rl.cli._experiment_contract", contract)
    monkeypatch.setattr(
        "isingfold.rl.cli._save_training_state",
        lambda *_args, **kwargs: state_capture.append(kwargs),
    )
    args = SimpleNamespace(
        epochs=1,
        minibatch=2,
        corpus="prepared",
        selector="selector",
        quality_labels=str(quality_directory),
        quality_preflight_receipt=str(preflight),
        expected_quality_preflight_sha256=hashlib.sha256(preflight.read_bytes()).hexdigest(),
        min_resolved_rows=1,
        min_resolved_lineages=1,
        allow_legacy_pilot=False,
        device="cpu",
        seed=17,
        deterministic=True,
        threads=1,
        model_family="if-core",
        learning_rate=3e-4,
        weight_decay=0.0,
        out=str(tmp_path / "warm"),
        resume=False,
    )

    cmd_warm_start(args)

    history = json.loads((Path(args.out) / "history.json").read_text())
    assert history["schema"] == "isingfold.warm-start-history"
    assert history["schema_version"] == 4
    assert history["epochs"][0]["corpus_rank_loss"] >= 0.0
    assert history["epochs"][0]["corpus_action_value_loss"] > 0.0
    assert history["epochs"][0]["corpus_commit_delta_loss"] > 0.0
    assert history["epochs"][0]["corpus_utility_loss"] > 0.0
    assert history["epochs"][0]["corpus_total_loss"] >= history["epochs"][0][
        "corpus_utility_loss"
    ]
    assert history["epochs"][0]["corpus_total_loss"] == pytest.approx(
        history["epochs"][0]["corpus_rank_loss"]
        + history["epochs"][0]["corpus_action_value_loss"]
        + history["epochs"][0]["corpus_commit_delta_loss"]
        + history["epochs"][0]["corpus_utility_loss"]
    )
    assert history["epochs"][0]["utility_effective_count"] == pytest.approx(12.0)
    assert history["epochs"][0]["action_effective_count"] == pytest.approx(14.0)
    assert history["epochs"][0]["actor_ranking_rows"] == 1
    assert history["epochs"][0]["action_value_rows"] == 3
    assert history["epochs"][0]["commit_delta_rows"] == 1
    assert history["epochs"][0]["memory_minibatches"] == 2
    assert state_capture[0]["counters"]["optimizer_steps"] == 1
    hyperparameters = contract_capture[0]["hyperparameters"]
    assert hyperparameters["warm_start_loss"] == (
        "all-action-qmu-plus-commit-delta-plus-resolved-rank-plus-state-value-v4"
    )
    assert hyperparameters["warm_start_loss_profile"] == {
        "profile_id": "full-qmu-v4",
        "weights": {
            "rank": 1.0,
            "utility": 0.5,
            "action_value": 1.0,
            "commit_delta": 0.5,
        },
        "q_mu_label_supervision": True,
        "production_transfer_eligible": True,
    }
    assert hyperparameters["warm_start_rank_coefficient"] == 1.0
    assert hyperparameters["warm_start_diagnostic_binding"] is None
    assert hyperparameters["warm_start_reduction"] == (
        "exact-full-corpus-gradient-accumulation-v1"
    )
    assert hyperparameters["warm_start_action_value_target"] == (
        "bounded-frozen-continuation-qmu-v1"
    )
    assert hyperparameters["warm_start_action_value_weighting"] == (
        "equal-state-count-over-propensity-v1"
    )
    assert hyperparameters["warm_start_action_value_coefficient"] == 1.0
    assert hyperparameters["warm_start_commit_delta_target"] == (
        "protected-commit-relative-qmu-v1"
    )
    assert hyperparameters["warm_start_commit_delta_coefficient"] == 0.5
    assert hyperparameters["warm_start_utility_target"] == (
        "uniform-legal-action-then-frozen-continuation-v1"
    )
    assert hyperparameters["warm_start_utility_weighting"] == (
        "bounded-variance-effective-count-v1"
    )
    assert hyperparameters["warm_start_actor_ranking_records"] == 1
    assert hyperparameters["warm_start_utility_critic_records"] == 3
    assert hyperparameters["warm_start_action_value_records"] == 3
    assert hyperparameters["warm_start_action_value_actions"] == 4
    assert hyperparameters["warm_start_commit_delta_records"] == 1
    assert hyperparameters["warm_start_corpus_denominators"] == {
        "actor_ranking_records": 1,
        "utility_effective_count": 12.0,
        "action_value_records": 3,
        "commit_delta_records": 1,
    }


def test_transfer_rejects_retired_warm_start_loss_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = IFCore(improvement_mode=True)
    quality_authority = {"record_digest": "d" * 64}
    target_access = {"record_digest": "e" * 64}
    ground_partition_receipt = {"record_digest": "f" * 64}
    bundle = SimpleNamespace(
        selector_digest="b" * 64,
        normalizer={},
        normalizer_digest="c" * 64,
        quality_authority=quality_authority,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
    )
    record = {
        "schema": "isingfold.training-run",
        "schema_version": 4,
        "phase": "representation",
        "model_family": "if-core",
        "method": "supervised-ranking",
        "complete": True,
        "experiment_contract": {
            "phase": "representation",
            "model_family": "if-core",
            "method": "supervised-ranking",
            "model": _model_identity(model, "if-core"),
            "seed": 1103,
            "grid_cell": "repr-000-if-core-s1103",
            "grid_manifest_sha256": "d" * 64,
            "corpus_manifest_sha256": "a" * 64,
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "selector_digest": bundle.selector_digest,
            "normalizer_digest": bundle.normalizer_digest,
            "quality_preflight_receipt_sha256": "5" * 64,
            "quality_preflight_record_digest": "6" * 64,
            "hyperparameters": {
                "warm_start_loss": "subset-rank-plus-count-weighted-value-v1",
                "warm_start_utility_target": (
                    "uniform-legal-action-then-frozen-continuation-v1"
                ),
                "warm_start_utility_weighting": (
                    "bounded-variance-effective-count-v1"
                ),
                "warm_start_utility_coefficient": 0.5,
            },
        },
        "feature_normalizer": {},
        "training_lineages": ["train-lineage"],
    }
    monkeypatch.setattr("isingfold.rl.cli._strict_json", lambda _path: record)
    monkeypatch.setattr("isingfold.rl.cli._verify_record", lambda *_args: None)
    monkeypatch.setattr(
        "isingfold.rl.cli._corpus_manifest_digest", lambda _corpus: "a" * 64
    )

    with pytest.raises(ValueError, match="current actor-critic loss contract"):
        _load_transfer(
            tmp_path / "checkpoint.pt",
            model=model,
            bundle=bundle,
            corpus=tmp_path / "prepared",
            model_family="if-core",
            expected_seed=1103,
            expected_grid_cell="repr-000-if-core-s1103",
            expected_grid_manifest_sha256="d" * 64,
            expected_quality_preflight_receipt_sha256="5" * 64,
            expected_quality_preflight_record_digest="6" * 64,
        )


def test_hpc_launchers_enforce_machine_scheduler_boundary() -> None:
    apollo = (ROOT / "scripts" / "apollo_rl_grid.sh").read_text()
    goose = (ROOT / "scripts" / "goose_rl_grid.sbatch").read_text()
    apollo_eval = (ROOT / "scripts" / "apollo_representation_eval.sh").read_text()
    goose_eval = (ROOT / "scripts" / "goose_representation_eval.sbatch").read_text()

    assert "sbatch" not in apollo
    assert "grid-cell" in apollo
    assert "SLURM_JOB_ID" in goose and "SLURM_ARRAY_TASK_ID" in goose
    assert "srun" in goose
    assert "grid-cell" in goose
    assert "evaluate-representation-cell" in apollo_eval
    assert "--bootstrap-bank" in apollo_eval
    assert "--expected-bootstrap-plan-sha256" in apollo_eval
    assert "--expected-bootstrap-manifest-sha256" in apollo_eval
    assert "sbatch" not in apollo_eval
    assert "SLURM_JOB_ID" in goose_eval and "srun" in goose_eval
    assert "#SBATCH --array=0-8" in goose_eval
    assert "computed representation evaluation index is outside 0-8" in goose_eval
    assert "evaluate-representation-cell" in goose_eval
    assert "--bootstrap-bank" in goose_eval
    assert "--expected-bootstrap-plan-sha256" in goose_eval
    assert "--expected-bootstrap-manifest-sha256" in goose_eval

    apollo_complete_train = (ROOT / "scripts" / "apollo_complete_train.sh").read_text()
    goose_complete_train = (ROOT / "scripts" / "goose_complete_train.sbatch").read_text()
    apollo_complete_eval = (ROOT / "scripts" / "apollo_complete_eval.sh").read_text()
    goose_complete_eval = (ROOT / "scripts" / "goose_complete_eval.sbatch").read_text()
    assert "sbatch" not in apollo_complete_train
    assert "complete-system-train-cell" in apollo_complete_train
    assert "SLURM_JOB_ID" in goose_complete_train and "srun" in goose_complete_train
    assert "complete-system-train-cell" in goose_complete_train
    assert "sbatch" not in apollo_complete_eval
    assert "evaluate-complete-system-cell" in apollo_complete_eval
    assert "SLURM_JOB_ID" in goose_complete_eval and "srun" in goose_complete_eval
    assert "evaluate-complete-system-cell" in goose_complete_eval


def test_registered_grid_launchers_fail_closed_without_three_way_seed_stride() -> None:
    launchers = (
        ROOT / "scripts" / "apollo_rl_grid.sh",
        ROOT / "scripts" / "goose_rl_grid.sbatch",
        ROOT / "scripts" / "apollo_representation_eval.sh",
        ROOT / "scripts" / "goose_representation_eval.sbatch",
    )

    for launcher in launchers:
        text = launcher.read_text()
        assert '${ISINGFOLD_GRID_STEP:?set ISINGFOLD_GRID_STEP}' in text
        assert '"$step" != "3"' in text


def test_quality_launchers_support_deterministic_parallel_shards() -> None:
    apollo = (ROOT / "scripts" / "apollo_quality_shards.sh").read_text()
    goose = (ROOT / "scripts" / "goose_quality_shards.sbatch").read_text()

    assert "sbatch" not in apollo
    assert "ISINGFOLD_SHARD_CONCURRENCY" in apollo
    assert "ISINGFOLD_GPU_IDS" in apollo
    assert 'CUDA_VISIBLE_DEVICES="$gpu_id"' in apollo
    assert "must be distinct" in apollo
    assert "wait_batch" in apollo
    assert "--shard-index" in apollo and "--shard-count" in apollo
    assert "SLURM_JOB_ID" in goose and "SLURM_ARRAY_TASK_ID" in goose
    assert "--shard-index" in goose and "--shard-count" in goose
    for launcher in (apollo, goose):
        assert "ISINGFOLD_QUALITY_LINEAGES:-0" in launcher
        assert "--initializer-bank" in launcher
        assert "--expected-initializer-bank-manifest-sha256" in launcher
        assert "--complete-config" in launcher
        assert "--resolution-plan" in launcher
        assert "--expected-resolution-plan-sha256" in launcher
        assert "--resolution-receipt" in launcher
        assert "--expected-resolution-receipt-sha256" in launcher
        assert "--capacity-selection" in launcher
        assert "--expected-capacity-selection-sha256" in launcher
        assert "--capacity-budget" in launcher
        assert "--expected-capacity-budget-sha256" in launcher
        assert "--capacity-canary" in launcher
        assert "--expected-capacity-canary-sha256" in launcher
        assert "--continuations" not in launcher


def test_scientific_training_launchers_require_quality_resolution_preflight() -> None:
    launchers = (
        ROOT / "scripts" / "apollo_rl_grid.sh",
        ROOT / "scripts" / "goose_rl_grid.sbatch",
        ROOT / "scripts" / "apollo_complete_train.sh",
        ROOT / "scripts" / "goose_complete_train.sbatch",
    )

    for launcher in launchers:
        text = launcher.read_text()
        assert '"${QUALITY_PREFLIGHT_RECEIPT:?set QUALITY_PREFLIGHT_RECEIPT}"' in text
        assert (
            '"${EXPECTED_QUALITY_PREFLIGHT_SHA256:?set '
            'EXPECTED_QUALITY_PREFLIGHT_SHA256}"' in text
        )
        assert '--quality-preflight-receipt "$QUALITY_PREFLIGHT_RECEIPT"' in text
        assert (
            '--expected-quality-preflight-sha256 '
            '"$EXPECTED_QUALITY_PREFLIGHT_SHA256"' in text
        )


def test_grid_launchers_bind_publication_quality_labels_to_their_initializer_bank() -> None:
    launchers = (
        ROOT / "scripts" / "apollo_rl_grid.sh",
        ROOT / "scripts" / "goose_rl_grid.sbatch",
    )

    for launcher in launchers:
        text = launcher.read_text()
        assert '"${QUALITY_INITIALIZER_BANK:?set QUALITY_INITIALIZER_BANK}"' in text
        assert (
            '"${EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256:?set '
            'EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256}"' in text
        )
        assert '"${QUALITY_COMPLETE_CONFIG:?set QUALITY_COMPLETE_CONFIG}"' in text
        assert '--quality-initializer-bank "$QUALITY_INITIALIZER_BANK"' in text
        assert (
            '--expected-quality-initializer-bank-manifest-sha256 '
            '"$EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256"' in text
        )
        assert '--quality-complete-config "$QUALITY_COMPLETE_CONFIG"' in text


def test_representation_grid_launchers_require_external_gate_receipt_pin() -> None:
    launchers = (
        ROOT / "scripts" / "apollo_rl_grid.sh",
        ROOT / "scripts" / "goose_rl_grid.sbatch",
    )

    for launcher in launchers:
        text = launcher.read_text()
        assert "EXPECTED_ISINGFOLD_GATE_RECEIPT_SHA256" in text
        assert (
            '--expected-gate-receipt-sha256 '
            '"$EXPECTED_ISINGFOLD_GATE_RECEIPT_SHA256"' in text
        )


def test_every_hpc_launcher_fails_closed_on_a_shadowed_source_tree() -> None:
    launchers = _scientific_hpc_launchers()

    assert launchers
    for launcher in launchers:
        text = launcher.read_text()
        assert "verify_runtime_source.sh" in text, launcher.name
        assert " -I -m isingfold.rl.cli " in " ".join(text.split()), launcher.name

    guard = (ROOT / "scripts" / "verify_runtime_source.sh").read_text()
    verifier = (ROOT / "scripts" / "verify_isingfold_runtime.py").read_text()
    assert "verify_isingfold_runtime.py" in guard
    assert 'exec "$python_bin" -I' in guard
    assert '"isingfold.rl.cli"' in verifier
    assert "specification.origin" in verifier
    assert "outside the attested wheel" in verifier
    assert "staged scientific or launcher/helper source" in verifier


def test_every_target_opening_hpc_launcher_requires_quality_authority() -> None:
    launchers = _scientific_hpc_launchers()
    # Initializer/bootstrap-bank generation is intentionally public-only.  The generic
    # evaluation-shard wrappers forward an already explicit, caller-supplied
    # authority argument vector instead of shadowing those pins in shell state.
    launchers = [
        launcher
        for launcher in launchers
        if launcher.name not in _TARGET_FREE_HPC_LAUNCHERS
        and "evaluation_shards" not in launcher.name
    ]
    required_environment = (
        '"${QUALITY_ATTESTATION:?set QUALITY_ATTESTATION}"',
        '"${EXPECTED_QUALITY_ATTESTATION_DIGEST:?set '
        'EXPECTED_QUALITY_ATTESTATION_DIGEST}"',
        '"${EXPECTED_QUALITY_PUBLISHER_ID:?set EXPECTED_QUALITY_PUBLISHER_ID}"',
        '"${GROUND_CERTIFICATE_ROOT:?set GROUND_CERTIFICATE_ROOT}"',
        '"${EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256:?set '
        'EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256}"',
    )
    required_arguments = (
        '--quality-attestation "$QUALITY_ATTESTATION"',
        '--expected-quality-attestation-digest '
        '"$EXPECTED_QUALITY_ATTESTATION_DIGEST"',
        '--expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID"',
        '--ground-certificate-root "$GROUND_CERTIFICATE_ROOT"',
        '--expected-ground-certificate-root-sha256 '
        '"$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256"',
    )

    assert launchers
    shared_quality_contract = (
        ROOT / "scripts" / "quality_protocol_common.sh"
    ).read_text()
    for launcher in launchers:
        text = launcher.read_text()
        # The quality control launchers deliberately centralize the authority
        # contract in a source-attested helper.  Audit the effective sourced
        # program instead of requiring duplicated shell fragments in each
        # entrypoint.
        effective_text = text
        if 'source "$script_dir/quality_protocol_common.sh"' in text:
            effective_text += "\n" + shared_quality_contract
        for contract in required_environment + required_arguments:
            assert contract in effective_text, f"{launcher.name} lacks {contract}"


def test_pyproject_publishes_console_entry_point() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert config["project"]["scripts"]["isingfold-rl"] == "isingfold.rl.cli:main"


def test_selector_fit_consumes_frozen_labels_and_publishes_bound_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_manifest(tmp_path / "prepared")
    _patch_prepared(monkeypatch, _tasks())
    labels = tmp_path / "selector-labels"
    build_selector_labels(
        source,
        labels,
        prepared_tasks=_tasks()[:2],
        quality_authority=_authority(),
        context=Context(qubit_cap=4),
        sample_seed=37,
        calibration_fraction=0.5,
        evaluator=_successful_evaluator([]),
    )
    output = tmp_path / "selector-bundle"
    args = build_parser().parse_args(
        [
            "fit-selector",
            "--corpus",
            str(source),
            "--selector-labels",
            str(labels),
            *_quality_cli_args(str(tmp_path / "publisher-attestation.json")),
            "--out",
            str(output),
            "--epochs",
            "1",
            "--device",
            "cpu",
        ]
    )

    args.func(args)

    metadata = load_selector_metadata(labels)
    bundle = _load_selector_bundle(output, corpus=source)
    receipt = json.loads((output / "fit_receipt.json").read_text())
    assert bundle.selector_digest == receipt["selector_digest"]
    assert bundle.normalizer_digest == metadata.normalizer_digest
    assert receipt["source_selector_labels_manifest_sha256"] == metadata.manifest_sha256
    assert receipt["optimizer"]["device_type"] == "cpu"
    assert receipt["optimizer"]["graph_minibatch_records"] == 32
    assert receipt["optimizer"]["optimizer_steps"] == 1
    assert receipt["training_history"][0]["records_seen"] == metadata.partitions["train"][
        "records"
    ]
    assert receipt["training_history"][0]["program_pairs_seen"] == 4 * metadata.partitions[
        "train"
    ]["records"]
    assert receipt["strength_transform_scale"] == metadata.coefficient_scale
    assert receipt["coefficient_transform_scale"] == metadata.coefficient_scale


def test_quality_loader_rejects_legacy_rows_without_v4_replay_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _source_manifest(tmp_path / "prepared")
    protocol = {
        "label_version": "if-q3-s0-qmu-2",
        "continuation_policy": {"policy_id": "uniform-exact-legal-support-v1"},
    }
    monkeypatch.setattr(
        "isingfold.rl.cli._quality_implementation_contract",
        lambda context: protocol,
    )
    monkeypatch.setattr(
        "isingfold.rl.cli._decode_observation",
        lambda raw: type(
            "ObservationStub",
            (),
            {"n_actions": 2, "legal_mask": (True, True)},
        )(),
    )
    root = tmp_path / "quality"
    root.mkdir()
    rows = [
        {
            "observation": {},
            "evaluated": [{"action_index": 0}, {"action_index": 1}],
            "best_actions": [0, 1],
            "label_version": "if-q3-s0-qmu-2",
            "continuation_policy": "uniform-exact-legal-support-v1",
            "lineage": "unresolved",
        },
        {
            "observation": {},
            "evaluated": [{"action_index": 0}, {"action_index": 1}],
            "best_actions": [1],
            "label_version": "if-q3-s0-qmu-2",
            "continuation_policy": "uniform-exact-legal-support-v1",
            "lineage": "resolved",
        },
    ]
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    (root / "records.jsonl").write_bytes(raw)
    manifest_payload = {
        "schema": "isingfold.quality-label-corpus",
        "schema_version": 2,
        "source_corpus_manifest_sha256": hashlib.sha256(
            (corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "selector_digest": "a" * 64,
        "normalizer_digest": "b" * 64,
        "context": _context_snapshot(Context(qubit_cap=4)),
        "partition": "train",
        "record_count": 2,
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "lineages": ["resolved", "unresolved"],
        "label_protocol": {
            "instances": 1,
            "states_per_instance": 2,
            "evaluated_actions": 2,
            "continuations": 32,
            "reward_reads": 256,
            "seed": 7,
            "implementation_contract": protocol,
        },
    }
    manifest = {**manifest_payload, "record_digest": content_digest(manifest_payload)}
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    selector = SelectorBundle(
        model=None,
        selector_digest="a" * 64,
        normalizer={},
        normalizer_digest="b" * 64,
        coefficient_scale=1.0,
        corpus_manifest_sha256=manifest_payload["source_corpus_manifest_sha256"],
        quality_authority=_authority(),
        target_access=_target_access("train", 2),
        ground_partition_receipt=_ground_partition_receipt("train", 2),
        root=tmp_path,
    )

    with pytest.raises(ValueError, match="unsupported quality-label schema version"):
        _load_quality_labels(
            root,
            corpus=corpus,
            selector=selector,
            context=Context(qubit_cap=4),
        )

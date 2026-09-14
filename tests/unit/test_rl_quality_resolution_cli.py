"""CLI orchestration for the preregistered quality-resolution workflow."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import isingfold.rl.cli as cli
import isingfold.rl.data.prepared as prepared_module
import isingfold.rl.data.quality_resolution_delta as delta_module
import isingfold.rl.data.quality_resolution_plan as plan_module
from isingfold.rl.contracts import Context
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest


PIN = "a" * 64
OTHER_PIN = "b" * 64


def _subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return set(action.choices)


def test_quality_resolution_cli_exposes_the_complete_artifact_lifecycle() -> None:
    commands = _subcommand_names(cli.build_parser())

    assert {
        "plan-quality-resolution",
        "publish-quality-resolution-verifier-identity",
        "run-quality-resolution-shard",
        "verify-quality-resolution-shard",
        "publish-quality-resolution-shard-pins",
        "merge-quality-resolution",
        "verify-quality-resolution-binding",
        "verify-quality-training-input-readiness",
        "publish-quality-capacity-budget",
        "plan-quality-capacity-canary",
        "quality-capacity-canary",
        "verify-quality-capacity-canary",
        "plan-quality-preflight-shards",
        "run-quality-preflight-shard",
        "merge-quality-preflight-shards",
    } <= commands


def test_gate2_cli_requires_plan_and_initializer_bank_and_has_no_candidate_limit() -> None:
    parser = cli.build_parser()
    parsed = parser.parse_args(
        [
            "gates",
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector-v4",
            *_authority_arguments(),
            "--grid",
            "grid.json",
            *_plan_arguments(),
            *_bank_arguments(),
            "--quality-labels",
            "quality-v7",
            "--quality-preflight-receipt",
            "preflight.json",
            "--expected-quality-preflight-sha256",
            PIN,
            "--selector-labels",
            "selector-labels",
            "--exact-conformance-corpus",
            "exact.json",
            "--expected-exact-conformance-sha256",
            OTHER_PIN,
            "--out",
            "gates.json",
        ]
    )

    assert parsed.func is cli.cmd_gates
    assert parsed.initializer_bank == "initializer-bank"
    assert not hasattr(parsed, "candidates")

    missing_bank = [
        "gates",
        "--corpus",
        "prepared-v4",
        "--selector",
        "selector-v4",
        *_authority_arguments(),
        "--grid",
        "grid.json",
        *_plan_arguments(),
        "--quality-labels",
        "quality-v7",
        "--quality-preflight-receipt",
        "preflight.json",
        "--expected-quality-preflight-sha256",
        PIN,
        "--selector-labels",
        "selector-labels",
        "--exact-conformance-corpus",
        "exact.json",
        "--expected-exact-conformance-sha256",
        OTHER_PIN,
        "--out",
        "gates.json",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(missing_bank)


def _authority_arguments() -> list[str]:
    return [
        "--quality-attestation",
        "publisher.json",
        "--expected-quality-attestation-digest",
        PIN,
        "--expected-quality-publisher-id",
        "publisher-a",
        "--ground-certificate-root",
        "ground/root.json",
        "--expected-ground-certificate-root-sha256",
        OTHER_PIN,
    ]


def _plan_arguments() -> list[str]:
    return ["--plan", "resolution-plan.json", "--expected-plan-sha256", PIN]


def _bank_arguments() -> list[str]:
    return [
        "--initializer-bank",
        "initializer-bank",
        "--expected-initializer-bank-manifest-sha256",
        PIN,
        "--complete-config",
        "complete-system.json",
    ]


def _execution_arguments() -> list[str]:
    return [
        *_plan_arguments(),
        "--stage-index",
        "2",
        "--shard-index",
        "17",
        "--corpus",
        "prepared-v4",
        "--selector",
        "selector-v4",
        *_bank_arguments(),
        *_authority_arguments(),
        "--device",
        "cpu",
        "--execution-runtime-sha256",
        OTHER_PIN,
    ]


def test_quality_resolution_parsers_require_the_registered_external_pins() -> None:
    parser = cli.build_parser()
    verifier_identity = parser.parse_args(
        [
            "publish-quality-resolution-verifier-identity",
            *_plan_arguments(),
            "--verification-runtime-sha256",
            OTHER_PIN,
            "--attestor-id",
            "independent-verifier-a",
            "--out",
            "verifier.json",
        ]
    )
    plan = parser.parse_args(
        [
            "plan-quality-resolution",
            "--config",
            "resolution.json",
            "--expected-config-sha256",
            PIN,
            "--grid",
            "grid.json",
            "--expected-grid-sha256",
            OTHER_PIN,
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector-v4",
            *_authority_arguments(),
            "--selector-device-parity",
            "selector-parity.json",
            "--expected-selector-device-parity-sha256",
            PIN,
            "--device",
            "cpu",
            *_bank_arguments(),
            "--out",
            "resolution-plan.json",
        ]
    )
    run = parser.parse_args(
        ["run-quality-resolution-shard", *_execution_arguments(), "--out", "delta"]
    )
    verify = parser.parse_args(
        [
            "verify-quality-resolution-shard",
            *_execution_arguments(),
            "--delta-root",
            "delta",
            "--expected-delta-manifest-sha256",
            PIN,
            "--verifier-identity",
            "verifier.json",
            "--expected-verifier-identity-sha256",
            OTHER_PIN,
            "--out",
            "verification.json",
        ]
    )
    merge = parser.parse_args(
        [
            "merge-quality-resolution",
            *_plan_arguments(),
            "--pin-registry",
            "pins.json",
            "--expected-pin-registry-sha256",
            OTHER_PIN,
            "--out",
            "study.json",
        ]
    )
    binding = parser.parse_args(
        [
            "verify-quality-resolution-binding",
            *_plan_arguments(),
            "--resolution-receipt",
            "study.json",
            "--expected-resolution-receipt-sha256",
            OTHER_PIN,
            "--quality-manifest",
            "quality/manifest.json",
            "--expected-quality-manifest-sha256",
            PIN,
            "--out",
            "binding.json",
        ]
    )
    readiness = parser.parse_args(
        [
            "verify-quality-training-input-readiness",
            *_plan_arguments(),
            "--resolution-receipt",
            "study.json",
            "--expected-resolution-receipt-sha256",
            OTHER_PIN,
            "--binding",
            "binding.json",
            "--expected-binding-sha256",
            PIN,
            "--quality-manifest",
            "quality/manifest.json",
            "--expected-quality-manifest-sha256",
            OTHER_PIN,
            "--quality-preflight",
            "preflight.json",
            "--expected-quality-preflight-sha256",
            PIN,
            "--out",
            "readiness.json",
        ]
    )

    assert plan.func is cli.cmd_plan_quality_resolution
    assert verifier_identity.func is cli.cmd_publish_quality_resolution_verifier_identity
    assert not hasattr(plan, "seed") and not hasattr(plan, "instances")
    assert (run.func, run.stage_index, run.shard_index) == (
        cli.cmd_run_quality_resolution_shard,
        2,
        17,
    )
    assert verify.func is cli.cmd_verify_quality_resolution_shard
    assert merge.func is cli.cmd_merge_quality_resolution
    assert binding.func is cli.cmd_verify_quality_resolution_binding
    assert readiness.func is cli.cmd_verify_quality_training_input_readiness

    missing_plan_pin = _execution_arguments()
    index = missing_plan_pin.index("--expected-plan-sha256")
    del missing_plan_pin[index : index + 2]
    with pytest.raises(SystemExit):
        parser.parse_args(["run-quality-resolution-shard", *missing_plan_pin, "--out", "delta"])


def test_pinned_plan_loader_rejects_raw_substitution_before_semantic_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {
        "config": {"raw_sha256": "1" * 64},
        "grid": {"raw_sha256": "2" * 64},
        "prepared_corpus": {
            "manifest_sha256": "3" * 64,
            "train_census_record_digest": "4" * 64,
        },
        "publisher": {"attestation_sha256": "5" * 64},
        "ground_root": {"sha256": "6" * 64},
        "selector": {
            "selector_file_sha256": "7" * 64,
            "device_parity_sha256": "8" * 64,
        },
    }
    path = tmp_path / "plan.json"
    raw = canonical_json_bytes(record) + b"\n"
    path.write_bytes(raw)
    calls: list[dict[str, object]] = []
    sentinel = object()

    def load(*args, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(plan_module, "load_quality_resolution_plan", load)
    expected = hashlib.sha256(raw).hexdigest()

    assert cli._load_pinned_quality_resolution_plan(path, expected) is sentinel
    assert calls == [
        {
            "expected_plan_sha256": expected,
            "expected_config_sha256": "1" * 64,
            "expected_grid_sha256": "2" * 64,
            "expected_prepared_manifest_sha256": "3" * 64,
            "expected_prepared_train_census_record_digest": "4" * 64,
            "expected_publisher_attestation_sha256": "5" * 64,
            "expected_ground_root_sha256": "6" * 64,
            "expected_selector_file_sha256": "7" * 64,
            "expected_selector_device_parity_sha256": "8" * 64,
        }
    ]

    calls.clear()
    with pytest.raises(ValueError, match="out-of-band pin"):
        cli._load_pinned_quality_resolution_plan(path, "f" * 64)
    assert calls == []


def test_resolution_target_loader_opens_only_pinned_canonical_train_jsonl(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "prepared-v4"
    (corpus / "targets").mkdir(parents=True)
    payload = {"instance_id": "train-1", "schema": "fixture", "schema_version": 1}
    row = {**payload, "record_digest": content_digest(payload)}
    raw = canonical_json_bytes(row) + b"\n"
    (corpus / "targets" / "train.jsonl").write_bytes(raw)
    access = {
        "target_count": 1,
        "target_path": "targets/train.jsonl",
        "target_sha256": hashlib.sha256(raw).hexdigest(),
    }

    assert cli._load_quality_resolution_target_records(corpus, access) == (row,)

    with pytest.raises(ValueError, match="only targets/train.jsonl"):
        cli._load_quality_resolution_target_records(
            corpus, {**access, "target_path": "targets/val.jsonl"}
        )
    with pytest.raises(ValueError, match="access receipt"):
        cli._load_quality_resolution_target_records(corpus, {**access, "target_sha256": OTHER_PIN})


def test_plan_command_loads_public_train_without_evaluator_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[object, ...]] = []
    context = Context(qubit_cap=160, n_est_reads=256)
    manifest_body = {"schema": "prepared", "schema_version": 4}
    manifest = {**manifest_body, "record_digest": content_digest(manifest_body)}
    fit_body = {"schema": "selector-fit", "schema_version": 1}
    fit = {**fit_body, "record_digest": content_digest(fit_body)}
    ground = {
        "receipt_sha256": "1" * 64,
        "record_digest": "2" * 64,
        "verifier_identity_digest": "3" * 64,
    }
    global_body = {
        "ground_root": ground,
        "publisher_id": "publisher-a",
        "publisher_attestation_record_digest": "4" * 64,
        "target_authority_record_digest": "5" * 64,
    }
    global_authority = {
        **global_body,
        "record_digest": content_digest(global_body),
    }
    quality_pin = SimpleNamespace(
        path="publisher.json",
        ground_certificate_root_sha256="1" * 64,
        ground_certificate_root_record_digest="2" * 64,
        global_quality_authority=global_authority,
    )

    class Model:
        def to(self, device):
            events.append(("model-to", device.type))
            return self

        def eval(self):
            return self

    bundle = SimpleNamespace(
        model=Model(),
        root=tmp_path / "selector",
        selector_digest="6" * 64,
        normalizer_digest="7" * 64,
    )
    production = SimpleNamespace(source_census=SimpleNamespace(record_digest="8" * 64))
    inputs = SimpleNamespace(config=SimpleNamespace(quality_seed=907))
    plan_record = {
        "production_plan": {"lineage_count": 1024},
        "record_digest": "9" * 64,
        "sample": {"sample_size": 128},
    }
    sealed_plan = SimpleNamespace(as_dict=lambda: plan_record)
    captured_authority: list[dict[str, object]] = []
    captured_materialization: list[dict[str, object]] = []
    initializer_bank = object()

    def load_public(corpus, *, partition, include_evaluator):
        events.append(("prepared", partition, include_evaluator))
        return SimpleNamespace(tasks=(object(),), target_access=None)

    monkeypatch.setattr(prepared_module, "load_prepared_partition", load_public)
    monkeypatch.setattr(
        plan_module, "load_quality_resolution_planning_inputs", lambda *a, **k: inputs
    )
    monkeypatch.setattr(
        plan_module,
        "materialize_resolution_production_plan",
        lambda tasks, **kwargs: (
            captured_materialization.append(kwargs) or production
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_quality_initializer_bank",
        lambda *a, **k: (initializer_bank, PIN),
    )
    monkeypatch.setattr(
        plan_module,
        "QualityResolutionPlanAuthority",
        lambda **kwargs: captured_authority.append(kwargs) or object(),
    )
    monkeypatch.setattr(plan_module, "build_quality_resolution_plan", lambda *args: sealed_plan)
    monkeypatch.setattr(plan_module, "publish_quality_resolution_plan", lambda *args: PIN)
    monkeypatch.setattr(
        cli,
        "_load_quality_selector_device_parity",
        lambda *a, **k: ({"record_digest": "a" * 64}, "b" * 64),
    )
    monkeypatch.setattr(cli, "_quality_attestation_pin", lambda args: quality_pin)
    monkeypatch.setattr(cli, "_context", lambda *args: context)
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(cli, "_bind_selector_quality_authority", lambda *a, **k: {})
    monkeypatch.setattr(cli, "_seed_runtime", lambda *a, **k: events.append(("seed",)))
    monkeypatch.setattr(cli, "_resolve_device", lambda value: SimpleNamespace(type=value))
    monkeypatch.setattr(
        cli,
        "_strict_json",
        lambda path: fit if Path(path).name == "fit_receipt.json" else manifest,
    )
    monkeypatch.setattr(
        cli,
        "load_prepared_corpus_design",
        lambda corpus: {"manifest_sha256": "c" * 64},
    )
    monkeypatch.setattr(cli, "_sha256_file", lambda path: "d" * 64)
    monkeypatch.setattr(
        cli,
        "_quality_resolution_implementation",
        lambda ctx: {
            "planner": "quality-resolution-plan-v1",
            "quality_implementation_contract_digest": "e" * 64,
            "quality_module_sha256": "f" * 64,
            "quality_resolution_delta_module_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(
        cli,
        "_load_quality_partition",
        lambda *a, **k: pytest.fail("target-bearing loader reached during planning"),
    )
    args = SimpleNamespace(
        config="resolution.json",
        expected_config_sha256=PIN,
        grid="grid.json",
        expected_grid_sha256=OTHER_PIN,
        selector_device_parity="parity.json",
        expected_selector_device_parity_sha256=PIN,
        device="cpu",
        corpus="prepared-v4",
        selector="selector-v4",
        deterministic=True,
        threads=1,
        qubit_cap=None,
        out="plan.json",
    )

    cli.cmd_plan_quality_resolution(args)

    assert events[0] == ("prepared", "train", False)
    assert captured_authority[0]["prepared_train_census_record_digest"] == "8" * 64
    assert captured_materialization[0]["initializer_bank"] is initializer_bank
    assert captured_materialization[0][
        "expected_initializer_bank_manifest_sha256"
    ] == PIN


def test_label_quality_accepts_only_a_pinned_plan_receipt_pair_for_publication() -> None:
    parsed = cli.build_parser().parse_args(
        [
            "label-quality",
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector-v4",
            *_authority_arguments(),
            "--resolution-plan",
            "resolution-plan.json",
            "--expected-resolution-plan-sha256",
            PIN,
            "--resolution-receipt",
            "resolution-study.json",
            "--expected-resolution-receipt-sha256",
            OTHER_PIN,
            "--capacity-selection",
            "capacity-selection.json",
            "--expected-capacity-selection-sha256",
            PIN,
            "--capacity-budget",
            "capacity-budget.json",
            "--expected-capacity-budget-sha256",
            OTHER_PIN,
            "--capacity-canary",
            "capacity-canary.json",
            "--expected-capacity-canary-sha256",
            PIN,
            *_bank_arguments(),
            "--instances",
            "0",
            "--seed",
            "907",
            "--device",
            "cpu",
            "--out",
            "quality-v7",
        ]
    )

    assert parsed.resolution_plan == "resolution-plan.json"
    assert parsed.resolution_receipt == "resolution-study.json"
    assert parsed.capacity_canary == "capacity-canary.json"
    assert parsed.continuations is None


def test_resolution_execution_rejects_plan_corpus_drift_before_target_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    plan = SimpleNamespace(
        as_dict=lambda: {
            "stages": [{}],
            "shards": [{}],
            "prepared_corpus": {"manifest_sha256": PIN},
            "publisher": {},
            "ground_root": {},
            "selector": {},
            "context": {},
            "implementation": {"registry": {}},
        }
    )
    quality_pin = SimpleNamespace(
        path="publisher.json",
        global_quality_authority={"ground_root": {}},
    )
    bundle = SimpleNamespace(root=Path("selector"))

    monkeypatch.setattr(cli, "_load_pinned_quality_resolution_plan", lambda *a: plan)
    monkeypatch.setattr(cli, "_quality_attestation_pin", lambda args: quality_pin)
    monkeypatch.setattr(
        prepared_module,
        "load_prepared_partition",
        lambda *a, **k: (
            events.append((k["partition"], k["include_evaluator"]))
            or SimpleNamespace(tasks=(object(),), target_access=None)
        ),
    )
    monkeypatch.setattr(cli, "_context", lambda *a: object())
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(cli, "_bind_selector_quality_authority", lambda *a, **k: {})
    monkeypatch.setattr(cli, "_corpus_manifest_digest", lambda corpus: OTHER_PIN)
    monkeypatch.setattr(
        cli,
        "_load_quality_partition",
        lambda *a, **k: pytest.fail("target partition opened before plan/corpus rejection"),
    )
    args = SimpleNamespace(
        plan="plan.json",
        expected_plan_sha256=PIN,
        stage_index=0,
        shard_index=0,
        corpus="prepared-v4",
        selector="selector-v4",
    )

    with pytest.raises(ValueError, match="target-free authority differs"):
        cli._quality_resolution_execution_inputs(args)

    assert events == [("train", False)]


def test_quality_publication_protocol_comes_only_from_terminal_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = Context(qubit_cap=160, n_est_reads=256)
    context_snapshot = cli._context_snapshot(context)
    plan = SimpleNamespace(
        as_dict=lambda: {
            "context": {
                "snapshot": context_snapshot,
                "digest": content_digest(context_snapshot),
            },
            "prepared_corpus": {"manifest_sha256": PIN},
        }
    )
    protocol = {
        "continuations": 24,
        "evaluated_actions": 8,
        "partition": "train",
        "quality_implementation_contract_digest": content_digest({"contract": "v1"}),
        "requested_lineages": 0,
        "reward_reads": 256,
        "seed": 907,
        "selector_device": "cpu",
        "states_per_lineage_cap": 4,
        "tasks_per_lineage_cap": 1,
    }
    study = SimpleNamespace(
        as_dict=lambda: {
            "advance": True,
            "terminal": True,
            "selected_continuations": 24,
            "production_protocol": protocol,
        }
    )
    monkeypatch.setattr(cli, "_load_pinned_quality_resolution_plan", lambda *a: plan)
    monkeypatch.setattr(
        "isingfold.rl.data.quality_resolution_merge.load_quality_resolution_study_receipt",
        lambda *a, **k: study,
    )
    monkeypatch.setattr(
        "isingfold.rl.data.quality_capacity.require_passing_quality_capacity_canary",
        lambda *a, **k: {
            "full_quality_launch_authorized": True,
            "quality_implementation_contract_digest": content_digest({"contract": "v1"}),
            "selected_continuations": 24,
        },
    )
    monkeypatch.setattr(cli, "_quality_implementation_contract", lambda ctx: {"contract": "v1"})
    monkeypatch.setattr(cli, "_corpus_manifest_digest", lambda corpus: PIN)
    args = SimpleNamespace(
        resolution_plan="plan.json",
        expected_resolution_plan_sha256=PIN,
        resolution_receipt="study.json",
        expected_resolution_receipt_sha256=OTHER_PIN,
        capacity_selection="selection.json",
        expected_capacity_selection_sha256=PIN,
        capacity_budget="budget.json",
        expected_capacity_budget_sha256=OTHER_PIN,
        capacity_canary="canary.json",
        expected_capacity_canary_sha256=PIN,
        continuations=None,
        actions=8,
        instances=0,
        reward_reads=256,
        seed=907,
        device="cpu",
        states_per_lineage=4,
        tasks_per_lineage=1,
        corpus="prepared-v4",
    )

    authorization = cli._quality_resolution_label_authorization(args, context)

    assert authorization is not None
    assert authorization.continuations == 24

    args.instances = 128
    with pytest.raises(ValueError, match="sealed production protocol"):
        cli._quality_resolution_label_authorization(args, context)


def test_quality_publication_rejects_a_narrow_capacity_capability_for_another_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = Context(qubit_cap=160, n_est_reads=256)
    context_snapshot = cli._context_snapshot(context)
    contract = {"contract": "v1"}
    plan = SimpleNamespace(
        as_dict=lambda: {
            "context": {
                "snapshot": context_snapshot,
                "digest": content_digest(context_snapshot),
            },
            "prepared_corpus": {"manifest_sha256": PIN},
        }
    )
    protocol = {
        "continuations": 24,
        "evaluated_actions": 8,
        "partition": "train",
        "quality_implementation_contract_digest": content_digest(contract),
        "requested_lineages": 0,
        "reward_reads": 256,
        "seed": 907,
        "selector_device": "cpu",
        "states_per_lineage_cap": 4,
        "tasks_per_lineage_cap": 1,
    }
    study = SimpleNamespace(
        as_dict=lambda: {
            "advance": True,
            "terminal": True,
            "selected_continuations": 24,
            "production_protocol": protocol,
        }
    )
    monkeypatch.setattr(cli, "_load_pinned_quality_resolution_plan", lambda *a: plan)
    monkeypatch.setattr(
        "isingfold.rl.data.quality_resolution_merge.load_quality_resolution_study_receipt",
        lambda *a, **k: study,
    )
    monkeypatch.setattr(cli, "_quality_implementation_contract", lambda ctx: contract)
    monkeypatch.setattr(cli, "_corpus_manifest_digest", lambda corpus: PIN)
    args = SimpleNamespace(
        resolution_plan="plan.json",
        expected_resolution_plan_sha256=PIN,
        resolution_receipt="study.json",
        expected_resolution_receipt_sha256=OTHER_PIN,
        capacity_selection="selection.json",
        expected_capacity_selection_sha256=PIN,
        capacity_budget="budget.json",
        expected_capacity_budget_sha256=OTHER_PIN,
        capacity_canary="canary.json",
        expected_capacity_canary_sha256=PIN,
        continuations=None,
        actions=8,
        instances=0,
        reward_reads=256,
        seed=907,
        device="cpu",
        states_per_lineage=4,
        tasks_per_lineage=1,
        corpus="prepared-v4",
    )

    monkeypatch.setattr(
        "isingfold.rl.data.quality_capacity.require_passing_quality_capacity_canary",
        lambda *a, **k: {
            "full_quality_launch_authorized": True,
            "quality_implementation_contract_digest": content_digest(contract),
            "selected_continuations": 48,
        },
    )
    with pytest.raises(ValueError, match="capacity capability differs"):
        cli._quality_resolution_label_authorization(args, context)


def test_unresolved_quality_generation_is_sealed_as_diagnostic_only() -> None:
    receipt = cli._quality_diagnostic_receipt(
        continuations=2,
        continuation_source="legacy-default-diagnostic",
        manifest_sha256=PIN,
        manifest_record_digest=OTHER_PIN,
        records_sha256="c" * 64,
    )

    assert receipt["schema"] == "isingfold.quality-label-diagnostic-status"
    assert receipt["mode"] == "diagnostic-only"
    assert receipt["publication_eligible"] is False
    assert receipt["effective_continuations"] == 2
    assert receipt["resolution_authorization_present"] is False
    cli._verify_record(receipt, "diagnostic receipt")


def test_quality_replay_parsers_require_out_of_band_artifact_pins() -> None:
    parser = cli.build_parser()
    planner = parser.parse_args(
        [
            "plan-quality-preflight-shards",
            "--resolution-plan",
            "resolution-plan.json",
            "--expected-resolution-plan-sha256",
            PIN,
            "--resolution-receipt",
            "resolution.json",
            "--expected-resolution-receipt-sha256",
            OTHER_PIN,
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector-v4",
            *_authority_arguments(),
            "--quality-shard-root",
            "quality-shards",
            "--expected-quality-shard-manifest-sha256",
            PIN,
            "--shard-count",
            "1",
            "--device",
            "cpu",
            "--out",
            "preflight-plan.json",
        ]
    )
    worker = parser.parse_args(
        [
            "run-quality-preflight-shard",
            "--plan",
            "preflight-plan.json",
            "--expected-plan-sha256",
            PIN,
            "--shard-index",
            "0",
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector-v4",
            *_authority_arguments(),
            "--quality-shard-root",
            "quality-shards",
            "--execution-runtime-sha256",
            OTHER_PIN,
            "--device",
            "cpu",
            "--out",
            "replay-0",
        ]
    )
    merger = parser.parse_args(
        [
            "merge-quality-preflight-shards",
            "--plan",
            "preflight-plan.json",
            "--expected-plan-sha256",
            PIN,
            "--replay-shard",
            "replay-0",
            "--expected-replay-shard-manifest-sha256",
            OTHER_PIN,
            "--out",
            "bundle.json",
        ]
    )
    quality_merge = parser.parse_args(
        [
            "merge-quality-labels",
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector-v4",
            *_authority_arguments(),
            "--shard",
            "quality-0",
            "--trusted-replay-bundle",
            "bundle.json",
            "--expected-trusted-replay-bundle-sha256",
            PIN,
            "--out",
            "quality-v7",
        ]
    )
    preflight = parser.parse_args(
        [
            "quality-preflight",
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector-v4",
            *_authority_arguments(),
            "--quality-labels",
            "quality-v7",
            "--workers",
            "64",
            "--out",
            "preflight-v2.json",
        ]
    )

    assert planner.shard_count == 1
    assert worker.execution_runtime_sha256 == OTHER_PIN
    assert merger.expected_replay_shard_manifest_sha256 == [OTHER_PIN]
    assert quality_merge.expected_trusted_replay_bundle_sha256 == PIN
    assert preflight.workers == 64


def test_resolution_execution_injects_train_targets_after_all_public_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    context = Context(qubit_cap=160, n_est_reads=256)
    context_snapshot = cli._context_snapshot(context)
    contract = {"contract": "v1"}
    contract_digest = content_digest(contract)
    quality_sha = "1" * 64
    delta_sha = "2" * 64
    publisher_sha = "3" * 64
    selector_sha = "4" * 64
    fit_sha = "5" * 64
    fit_body = {"schema": "selector-fit", "schema_version": 1}
    fit = {**fit_body, "record_digest": content_digest(fit_body)}
    ground_identity = {
        "receipt_sha256": "6" * 64,
        "record_digest": "7" * 64,
        "verifier_identity_digest": "8" * 64,
    }
    global_authority = {
        "ground_root": ground_identity,
        "publisher_id": "publisher-a",
        "publisher_attestation_record_digest": "9" * 64,
        "target_authority_record_digest": "a" * 64,
    }
    plan_record = {
        "stages": [{}],
        "shards": [{}],
        "prepared_corpus": {"manifest_sha256": PIN},
        "publisher": {
            "publisher_id": "publisher-a",
            "attestation_sha256": publisher_sha,
            "attestation_record_digest": "9" * 64,
            "target_authority_record_digest": "a" * 64,
        },
        "ground_root": {
            "sha256": "6" * 64,
            "record_digest": "7" * 64,
            "verifier_identity_digest": "8" * 64,
        },
        "selector": {
            "device": "cpu",
            "selector_digest": "b" * 64,
            "normalizer_digest": "c" * 64,
            "selector_file_sha256": selector_sha,
            "fit_receipt_sha256": fit_sha,
            "fit_receipt_record_digest": fit["record_digest"],
        },
        "context": {
            "snapshot": context_snapshot,
            "digest": content_digest(context_snapshot),
        },
        "implementation": {
            "registry": {
                "quality_implementation_contract_digest": contract_digest,
                "quality_module_sha256": quality_sha,
                "quality_resolution_delta_module_sha256": delta_sha,
            }
        },
        "config": {"registered": {"quality_seed": 907}},
    }
    plan = SimpleNamespace(as_dict=lambda: plan_record)
    quality_pin = SimpleNamespace(path="publisher.json", global_quality_authority=global_authority)

    class Model:
        def to(self, device):
            events.append("selector-device")
            return self

        def eval(self):
            return self

    bundle = SimpleNamespace(
        root=Path("selector"),
        model=Model(),
        selector_digest="b" * 64,
        normalizer_digest="c" * 64,
    )
    quality_authority = {"record_digest": "d" * 64}
    target_access = {"record_digest": "e" * 64}
    ground_partition = {"record_digest": "f" * 64}
    captured_authority: list[dict[str, object]] = []

    monkeypatch.setattr(cli, "_load_pinned_quality_resolution_plan", lambda *a: plan)
    monkeypatch.setattr(cli, "_quality_attestation_pin", lambda args: quality_pin)
    monkeypatch.setattr(
        prepared_module,
        "load_prepared_partition",
        lambda *a, **k: (
            events.append("public") or SimpleNamespace(tasks=("public-task",), target_access=None)
        ),
    )
    monkeypatch.setattr(cli, "_context", lambda *a: context)
    initializer_bank = object()
    monkeypatch.setattr(
        cli,
        "_load_quality_initializer_bank",
        lambda *a, **k: (events.append("bank") or (initializer_bank, PIN)),
    )
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        cli,
        "_bind_selector_quality_authority",
        lambda *a, **k: quality_authority,
    )
    monkeypatch.setattr(cli, "_corpus_manifest_digest", lambda corpus: PIN)

    def sha(path):
        name = Path(path).name
        return {
            "publisher.json": publisher_sha,
            "selector.pt": selector_sha,
            "fit_receipt.json": fit_sha,
        }[name]

    monkeypatch.setattr(cli, "_sha256_file", sha)
    monkeypatch.setattr(cli, "_strict_json", lambda path: fit)
    monkeypatch.setattr(cli, "_resolve_device", lambda value: SimpleNamespace(type=value))
    monkeypatch.setattr(cli, "_seed_runtime", lambda *a, **k: events.append("seed"))
    monkeypatch.setattr(cli, "_quality_implementation_contract", lambda ctx: contract)
    monkeypatch.setattr(
        cli,
        "_quality_resolution_source_sha256",
        lambda module, label: quality_sha if label == "quality implementation" else delta_sha,
    )
    monkeypatch.setattr(
        cli,
        "_load_quality_partition",
        lambda *a, **k: (
            events.append("targets")
            or (("target-task",), quality_authority, target_access, ground_partition)
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_quality_resolution_target_records",
        lambda *a: events.append("target-records") or ({"target": 1},),
    )
    monkeypatch.setattr(
        delta_module,
        "QualityResolutionExecutionAuthority",
        lambda **kwargs: captured_authority.append(kwargs) or SimpleNamespace(**kwargs),
    )
    args = SimpleNamespace(
        plan="plan.json",
        expected_plan_sha256=PIN,
        stage_index=0,
        shard_index=0,
        corpus="prepared-v4",
        selector="selector-v4",
        device="cpu",
        deterministic=True,
        threads=1,
        execution_runtime_sha256=OTHER_PIN,
    )

    result = cli._quality_resolution_execution_inputs(args)

    assert events.index("public") < events.index("targets")
    assert events.index("bank") < events.index("targets")
    assert events.index("seed") < events.index("targets")
    assert result[2] == ({"target": 1},)
    assert result[6] is initializer_bank and result[7] == PIN
    assert captured_authority[0]["execution_identity"]["runtime_sha256"] == OTHER_PIN


def test_quality_capacity_parsers_separate_public_selection_from_target_execution() -> None:
    parser = cli.build_parser()
    budget = parser.parse_args(
        [
            "publish-quality-capacity-budget",
            "--budget-id",
            "quality-capacity-v1",
            "--maximum-artifact-bytes",
            "1000000",
            "--maximum-cpu-seconds",
            "10000",
            "--maximum-elapsed-seconds",
            "1000",
            "--available-workers",
            "64",
            "--minimum-scratch-free-bytes",
            "3000000",
            "--out",
            "budget.json",
        ]
    )
    common = [
        *_plan_arguments(),
        "--resolution-receipt",
        "study.json",
        "--expected-resolution-receipt-sha256",
        OTHER_PIN,
        "--corpus",
        "prepared-v4",
        "--selector",
        "selector-v4",
        *_bank_arguments(),
        *_authority_arguments(),
        "--device",
        "cpu",
    ]
    selection = parser.parse_args(
        ["plan-quality-capacity-canary", *common, "--out", "selection.json"]
    )
    canary = parser.parse_args(
        [
            "quality-capacity-canary",
            *common,
            "--execution-runtime-sha256",
            PIN,
            "--host-class",
            "apollo-cpu",
            "--capacity-selection",
            "selection.json",
            "--expected-capacity-selection-sha256",
            PIN,
            "--capacity-budget",
            "budget.json",
            "--expected-capacity-budget-sha256",
            OTHER_PIN,
            "--scratch-directory",
            "scratch",
            "--out",
            "canary.json",
        ]
    )
    verify = parser.parse_args(
        [
            "verify-quality-capacity-canary",
            *_plan_arguments(),
            "--resolution-receipt",
            "study.json",
            "--expected-resolution-receipt-sha256",
            OTHER_PIN,
            "--capacity-selection",
            "selection.json",
            "--expected-capacity-selection-sha256",
            PIN,
            "--capacity-budget",
            "budget.json",
            "--expected-capacity-budget-sha256",
            OTHER_PIN,
            "--capacity-canary",
            "canary.json",
            "--expected-capacity-canary-sha256",
            PIN,
            "--out",
            "capacity-capability.json",
        ]
    )

    assert budget.func is cli.cmd_publish_quality_capacity_budget
    assert selection.func is cli.cmd_plan_quality_capacity_canary
    assert not hasattr(selection, "execution_runtime_sha256")
    assert canary.func is cli.cmd_quality_capacity_canary
    assert verify.func is cli.cmd_verify_quality_capacity_canary


def test_capacity_verification_publishes_one_immutable_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = object()
    study = object()
    capability = {
        "capacity_canary_sha256": PIN,
        "full_quality_launch_authorized": True,
    }
    destination = tmp_path / "capacity-capability.json"
    monkeypatch.setattr(cli, "_load_pinned_quality_resolution_plan", lambda *args: plan)
    monkeypatch.setattr(cli, "_load_quality_resolution_study_from_args", lambda *args: study)
    monkeypatch.setattr(
        "isingfold.rl.data.quality_capacity.require_passing_quality_capacity_canary",
        lambda *args, **kwargs: capability,
    )
    args = SimpleNamespace(
        plan="plan.json",
        expected_plan_sha256=PIN,
        resolution_receipt="study.json",
        expected_resolution_receipt_sha256=OTHER_PIN,
        capacity_selection="selection.json",
        expected_capacity_selection_sha256=PIN,
        capacity_budget="budget.json",
        expected_capacity_budget_sha256=OTHER_PIN,
        capacity_canary="canary.json",
        expected_capacity_canary_sha256=PIN,
        out=str(destination),
    )

    cli.cmd_verify_quality_capacity_canary(args)

    expected = canonical_json_bytes(capability) + b"\n"
    assert destination.read_bytes() == expected
    with pytest.raises(FileExistsError):
        cli.cmd_verify_quality_capacity_canary(args)

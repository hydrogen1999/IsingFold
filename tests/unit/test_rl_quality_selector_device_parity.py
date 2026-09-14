from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import isingfold.rl.cli as cli
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_selector_device_parity import (
    SelectorParityCase,
    build_quality_selector_device_parity,
    load_quality_selector_device_parity,
    publish_quality_selector_device_parity,
)


PIN = "a" * 64


class _DeviceAwareSelector:
    def __init__(self, selected: dict[str, int]) -> None:
        self.selected = selected
        self.device = "unset"

    def to(self, device: object):
        self.device = str(device)
        return self

    def eval(self):
        return self

    def select_embedding(self, inputs: object) -> int:
        assert len(inputs) == 4  # type: ignore[arg-type]
        return self.selected[self.device]


def _case(*, embedding: str = "1" * 64) -> SelectorParityCase:
    return SelectorParityCase(
        archive_index=0,
        archive_key="2" * 64,
        base_lineage="lineage-0",
        compiled_program_digests=("3" * 64, "4" * 64, "5" * 64, "6" * 64),
        instance_id="instance-0",
        lineage_schedule_index=0,
        quality_row_id="7" * 64,
        selected_embedding_digest=embedding,
        selector_input_digest="a" * 64,
        selector_inputs=(object(), object(), object(), object()),
        state_fingerprint="8" * 64,
        state_schedule_index=0,
        support_fingerprint="9" * 64,
        task_id="task-0",
    )


def _identities() -> dict[str, object]:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "quality_resolution_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = {"qubit_cap": 160, "n_est_reads": 256}
    bank_body = {
        "access_receipt_record_digest": "0" * 64,
        "conditional_episode_count": 1,
        "config_digest": "1" * 64,
        "context_digest": content_digest(context),
        "episode_schedule_start": 0,
        "episode_schedule_stop_exclusive": 1,
        "manifest_record_digest": "2" * 64,
        "manifest_sha256": "3" * 64,
        "opened_evaluator_data": False,
        "partition": "train",
        "plan_record_digest": "4" * 64,
        "prepared_manifest_sha256": "2" * 64,
        "protocol": "authenticated-persistent-k2-initializer-bank-v1",
        "publication_eligible": True,
        "restart_cache_slots_per_episode": 2,
        "schema": "isingfold.quality-initializer-bank-contract",
        "schema_version": 1,
        "training_seed": 1103,
    }
    bank = {**bank_body, "record_digest": content_digest(bank_body)}
    return {
        "corpus": {
            "manifest_record_digest": "1" * 64,
            "manifest_sha256": "2" * 64,
            "train_census_record_digest": "3" * 64,
            "train_lineage_count": 1,
            "train_task_count": 1,
        },
        "selector_identity": {
            "fit_receipt_record_digest": "4" * 64,
            "fit_receipt_sha256": "5" * 64,
            "normalizer_digest": "6" * 64,
            "normalizer_file_sha256": "7" * 64,
            "selector_digest": "8" * 64,
            "selector_file_sha256": "9" * 64,
        },
        "quality_protocol": {
            "config": config,
            "config_digest": content_digest(config),
            "config_sha256": "a" * 64,
            "production_plan_digest": "b" * 64,
            "production_row_count": 1,
        },
        "context_identity": {"digest": content_digest(context), "snapshot": context},
        "initializer_bank_identity": bank,
        "runtime": {
            "cuda_runtime_version": "12.8",
            "deterministic_algorithms": True,
            "execution_runtime_sha256": "c" * 64,
            "implementation": {
                "parity_module_sha256": "d" * 64,
                "program_module_sha256": "e" * 64,
                "selector_module_sha256": "f" * 64,
                "tensorizer_module_sha256": "0" * 64,
            },
            "platform": {"machine": "fixture", "system": "Linux"},
            "python_version": "3.12.0",
            "threads": 1,
            "torch_version": "2.8.0",
        },
        "cpu_device_identity": {
            "capability": None,
            "index": None,
            "name": "fixture-cpu",
            "requested": "cpu",
            "type": "cpu",
        },
        "accelerator_device_identity": {
            "capability": [9, 0],
            "index": 0,
            "name": "fixture-gpu",
            "requested": "cuda",
            "type": "cuda",
        },
    }


def _build(case_factory, selector: _DeviceAwareSelector) -> dict[str, object]:
    return build_quality_selector_device_parity(
        (),  # The injected case factory isolates the two-pass comparator in this unit test.
        object(),  # type: ignore[arg-type]
        context=object(),  # type: ignore[arg-type]
        selector=selector,
        initializer_bank=object(),  # type: ignore[arg-type]
        expected_initializer_bank_manifest_sha256=PIN,
        cpu_device="cpu",
        accelerator_device="cuda",
        case_factory=case_factory,
        **_identities(),
    )


def test_builder_records_exact_equal_and_mismatch_branches() -> None:
    equal = _build(
        lambda: iter((_case(),)), _DeviceAwareSelector({"cpu": 2, "cuda": 2})
    )

    assert equal["schema"] == "isingfold.quality-selector-device-parity"
    assert equal["schema_version"] == 1
    assert equal["all_equal"] is True
    assert equal["mismatch_count"] == 0
    assert equal["mismatches"] == []
    assert equal["census"] == {
        "matching_cases": 1,
        "mismatching_cases": 0,
        "scheduled_quality_rows": 1,
        "selector_cases": 1,
        "train_lineages": 1,
        "train_tasks": 1,
    }
    row = equal["rows"][0]
    assert row["cpu"]["selected_strength_index"] == 2
    assert row["accelerator"]["selected_strength_index"] == 2
    assert row["equal"] is True
    assert equal["record_digest"] == content_digest(
        {key: value for key, value in equal.items() if key != "record_digest"}
    )

    mismatch = _build(
        lambda: iter((_case(),)), _DeviceAwareSelector({"cpu": 1, "cuda": 3})
    )
    assert mismatch["all_equal"] is False
    assert mismatch["mismatch_count"] == 1
    assert mismatch["mismatches"] == [mismatch["rows"][0]["case_id"]]


def test_builder_rejects_second_pass_case_or_program_drift() -> None:
    calls = 0

    def drifting_cases():
        nonlocal calls
        calls += 1
        if calls == 1:
            return iter((_case(),))
        return iter((_case(embedding="f" * 64),))

    with pytest.raises(QualityResolutionError, match="replay.*differs"):
        _build(drifting_cases, _DeviceAwareSelector({"cpu": 1, "cuda": 1}))


def test_case_census_replays_every_planned_row_without_targets(tmp_path: Path) -> None:
    from isingfold.rl.data.quality_resolution_plan import (
        materialize_resolution_production_plan,
    )
    from isingfold.rl.data.quality_selector_device_parity import (
        iter_quality_selector_parity_cases,
    )
    from tests.unit.test_rl_initializer_bank import _context
    from tests.unit.test_rl_quality_initializer_bank import _sealed_test_bank
    from tests.unit.test_rl_quality_resolution_plan import _planning_inputs

    class Selector:
        coefficient_transform_scale = 1.0
        deployment_ready = True
        normalizer_digest = "selector-normalizer-unit-v1"

        def __call__(self, *_args: object) -> int:
            return 0

        def select_embedding(self, _inputs: object) -> int:
            return 0

    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    selector = Selector()
    production = materialize_resolution_production_plan(
        [public],
        context=_context(),
        selector=selector,
        config=_planning_inputs().config,
        source_corpus_manifest_sha256=bank.plan.prepared_manifest_sha256,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=manifest_sha256,
        allow_test_initializer_bank=True,
    )
    cases = list(
        iter_quality_selector_parity_cases(
            [public],
            production,
            context=_context(),
            selector=selector,
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=manifest_sha256,
            allow_test_initializer_bank=True,
        )
    )

    assert {case.quality_row_id for case in cases} == {
        row.row_id for row in production.rows
    }
    assert all(len(case.compiled_program_digests) == 4 for case in cases)
    assert all(case.selected_embedding_digest == case.archive_key for case in cases)


def test_parity_receipt_publication_is_canonical_strict_and_no_replace(
    tmp_path: Path,
) -> None:
    receipt = _build(
        lambda: iter((_case(),)), _DeviceAwareSelector({"cpu": 0, "cuda": 0})
    )
    path = tmp_path / "parity.json"
    raw_sha256 = publish_quality_selector_device_parity(path, receipt)

    assert path.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    assert load_quality_selector_device_parity(
        path, expected_sha256=raw_sha256
    ) == receipt
    with pytest.raises(FileExistsError):
        publish_quality_selector_device_parity(path, receipt)

    altered = dict(receipt)
    altered["unknown"] = True
    body = {key: value for key, value in altered.items() if key != "record_digest"}
    altered["record_digest"] = content_digest(body)
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(canonical_json_bytes(altered) + b"\n")
    with pytest.raises(QualityResolutionError, match="schema fields"):
        load_quality_selector_device_parity(
            alternate,
            expected_sha256=hashlib.sha256(alternate.read_bytes()).hexdigest(),
        )


def test_existing_cli_loader_accepts_only_complete_receipts_and_gates_cpu(
    tmp_path: Path,
) -> None:
    equal = _build(
        lambda: iter((_case(),)), _DeviceAwareSelector({"cpu": 2, "cuda": 2})
    )
    equal_path = tmp_path / "equal.json"
    equal_sha = publish_quality_selector_device_parity(equal_path, equal)
    loaded, observed = cli._load_quality_selector_device_parity(
        equal_path, equal_sha, device="cpu"
    )
    assert loaded == equal
    assert observed == equal_sha

    mismatch = _build(
        lambda: iter((_case(),)), _DeviceAwareSelector({"cpu": 1, "cuda": 3})
    )
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_sha = publish_quality_selector_device_parity(mismatch_path, mismatch)
    assert cli._load_quality_selector_device_parity(
        mismatch_path, mismatch_sha, device="cuda"
    )[0] == mismatch
    with pytest.raises(ValueError, match="CPU quality inference"):
        cli._load_quality_selector_device_parity(
            mismatch_path, mismatch_sha, device="cpu"
        )

    minimal_body = {
        "all_equal": True,
        "mismatch_count": 0,
        "schema": "isingfold.quality-selector-device-parity",
        "schema_version": 1,
    }
    minimal = {**minimal_body, "record_digest": content_digest(minimal_body)}
    minimal_path = tmp_path / "minimal.json"
    minimal_path.write_bytes(canonical_json_bytes(minimal) + b"\n")
    with pytest.raises(ValueError, match="schema fields"):
        cli._load_quality_selector_device_parity(
            minimal_path,
            hashlib.sha256(minimal_path.read_bytes()).hexdigest(),
            device="cpu",
        )


def test_cli_exposes_parity_command_with_complete_pinned_state_schedule() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "audit-quality-selector-device-parity",
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector-v4",
            "--quality-protocol-config",
            "quality-resolution.json",
            "--expected-quality-protocol-config-sha256",
            PIN,
            "--initializer-bank",
            "initializer-bank",
            "--expected-initializer-bank-manifest-sha256",
            PIN,
            "--complete-config",
            "complete-system.json",
            "--cpu-device",
            "cpu",
            "--accelerator-device",
            "cuda",
            "--execution-runtime-sha256",
            PIN,
            "--out",
            "parity.json",
        ]
    )

    assert args.func is cli.cmd_audit_quality_selector_device_parity
    assert args.cpu_device == "cpu"
    assert args.accelerator_device == "cuda"
    assert not hasattr(args, "partition")


def test_parity_command_loads_public_train_without_evaluator_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import isingfold.rl.data.prepared as prepared_module
    import isingfold.rl.data.quality_resolution_plan as plan_module
    import isingfold.rl.data.quality_selector_device_parity as parity_module

    events: list[tuple[object, ...]] = []
    public_tasks = (object(),)
    config = SimpleNamespace(quality_seed=907)
    production = SimpleNamespace(
        as_dict=lambda: {
            "lineage_count": 1,
            "row_count": 1,
            "record_digest": "1" * 64,
            "initializer_bank": {"record_digest": "0" * 64},
            "source_census": {
                "lineage_count": 1,
                "record_digest": "2" * 64,
                "task_count": 1,
            },
        }
    )
    model = _DeviceAwareSelector({"cpu": 0, "cuda": 0})
    bundle = SimpleNamespace(
        model=model,
        root=tmp_path / "selector",
        selector_digest="3" * 64,
        normalizer_digest="4" * 64,
    )
    bundle.root.mkdir()
    for name in ("selector.pt", "normalizer.json", "fit_receipt.json"):
        (bundle.root / name).write_bytes(b"fixture")
    manifest = tmp_path / "prepared" / "manifest.json"
    manifest.parent.mkdir()
    manifest_body = {"schema": "prepared", "schema_version": 4}
    manifest.write_bytes(
        canonical_json_bytes(
            {**manifest_body, "record_digest": content_digest(manifest_body)}
        )
        + b"\n"
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        prepared_module,
        "load_prepared_partition",
        lambda corpus, *, partition, include_evaluator: (
            events.append(("prepared", partition, include_evaluator))
            or SimpleNamespace(tasks=public_tasks, target_access=None)
        ),
    )
    monkeypatch.setattr(
        parity_module,
        "load_quality_selector_parity_config",
        lambda *a, **k: (config, PIN, "5" * 64, {"study_id": "fixture"}),
    )
    monkeypatch.setattr(cli, "_context", lambda *a, **k: SimpleNamespace(qubit_cap=160))
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(cli, "_load_quality_initializer_bank", lambda *a, **k: (object(), PIN))
    monkeypatch.setattr(cli, "_seed_runtime", lambda *a, **k: events.append(("seed",)))
    monkeypatch.setattr(cli, "_resolve_device", lambda value: value)
    monkeypatch.setattr(
        plan_module,
        "materialize_resolution_production_plan",
        lambda tasks, **kwargs: production,
    )
    monkeypatch.setattr(
        parity_module,
        "build_quality_selector_device_parity",
        lambda *a, **kwargs: captured.update(
            {"prepared": a[0], "production": a[1], **kwargs}
        )
        or {
            "all_equal": True,
            "mismatch_count": 0,
            "record_digest": "6" * 64,
        },
    )
    monkeypatch.setattr(
        parity_module,
        "publish_quality_selector_device_parity",
        lambda *a, **k: "7" * 64,
    )
    monkeypatch.setattr(
        cli,
        "_strict_json",
        lambda path: (
            {"record_digest": "8" * 64, "schema_version": 4}
            if Path(path).name == "manifest.json"
            else {"record_digest": "9" * 64}
        ),
    )
    monkeypatch.setattr(cli, "_verify_record", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_runtime_platform_identity", lambda: {"system": "fixture"})
    monkeypatch.setattr(
        cli,
        "_context_snapshot",
        lambda value: {"qubit_cap": value.qubit_cap},
    )
    monkeypatch.setattr(cli, "_selector_parity_runtime_identity", lambda *a, **k: {})
    monkeypatch.setattr(cli, "_selector_parity_device_identity", lambda *a, **k: {})

    args = SimpleNamespace(
        corpus=str(manifest.parent),
        selector=str(bundle.root),
        quality_protocol_config="quality.json",
        expected_quality_protocol_config_sha256=PIN,
        initializer_bank="bank",
        expected_initializer_bank_manifest_sha256=PIN,
        complete_config="complete.json",
        cpu_device="cpu",
        accelerator_device="cuda",
        execution_runtime_sha256=PIN,
        threads=1,
        deterministic=True,
        qubit_cap=None,
        out=str(tmp_path / "parity.json"),
    )

    cli.cmd_audit_quality_selector_device_parity(args)

    assert events[0] == ("prepared", "train", False)
    assert all(event[:1] != ("targets",) for event in events)
    assert captured["prepared"] is public_tasks
    assert captured["production"] is production

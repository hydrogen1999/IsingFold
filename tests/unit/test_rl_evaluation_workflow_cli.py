"""Fail-closed orchestration tests for resumable evaluation workflows.

These tests intentionally exercise the thin CLI at its filesystem and execution seams.  The
scientific runners and authenticated artifact formats are covered by their own unit suites.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import isingfold.rl.evaluation_workflow_cli as workflow_cli


ROOT = Path(__file__).resolve().parents[2]
PIN = "a" * 64
OTHER_PIN = "b" * 64


class _FakePlan:
    def __init__(self, identity: str = "sealed-full-population") -> None:
        self.identity = identity
        self.max_lineages_per_shard = 32
        self.record_digest = PIN
        self.publication_eligible = True
        self.shards = (SimpleNamespace(shard_index=0, lineages=("lineage-a",)),)

    def as_dict(self) -> dict[str, object]:
        return {"identity": self.identity}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    workflow_cli.register_evaluation_shard_commands(subparsers)
    return parser


def _learned_arguments(command: str) -> list[str]:
    values = [
        command,
        "--workflow",
        "learned-complete-system",
        "--grid",
        "grid.json",
        "--corpus",
        "prepared-v4",
        "--selector",
        "selector",
        "--run-root",
        "runs",
        "--config",
        "complete.json",
        "--rl-value-selection-receipt",
        "rl-selection.json",
        "--expected-selection-sha256",
        PIN,
        "--index",
        "0",
    ]
    return [*values, *_phase_arguments(command)]


def _tuned_stock_arguments(command: str) -> list[str]:
    values = [
        command,
        "--workflow",
        "tuned-stock-complete-system",
        "--grid",
        "grid.json",
        "--corpus",
        "prepared-v4",
        "--selector",
        "selector",
        "--config",
        "external.json",
        "--learned-config",
        "complete.json",
        "--tuning-registry",
        "registry.json",
        "--expected-tuning-registry-sha256",
        PIN,
        "--external-tuning-selection",
        "external-selection.json",
        "--expected-external-tuning-selection-sha256",
        OTHER_PIN,
        "--index",
        "1",
    ]
    return [*values, *_phase_arguments(command)]


def _validation_tuning_arguments(command: str) -> list[str]:
    values = [
        command,
        "--workflow",
        "external-validation-tuning",
        "--grid",
        "grid.json",
        "--corpus",
        "prepared-v4",
        "--selector",
        "selector",
        "--registry",
        "registry.json",
        "--expected-registry-sha256",
        PIN,
        "--external-config",
        "external.json",
        "--learned-config",
        "complete.json",
        "--index",
        "2",
        "--tuning-seed-index",
        "0",
    ]
    return [*values, *_phase_arguments(command)]


def _phase_arguments(command: str) -> list[str]:
    authority = [
        "--cluster",
        "apollo",
        "--execution-mode",
        "pinned-venv",
        "--environment-lock",
        "requirements.lock",
        "--expected-environment-lock-sha256",
        PIN,
        "--quality-attestation",
        "publisher-attestation.json",
        "--expected-quality-attestation-digest",
        PIN,
        "--expected-quality-publisher-id",
        "publisher-a",
        "--ground-certificate-root",
        "ground/root.json",
        "--expected-ground-certificate-root-sha256",
        OTHER_PIN,
    ]
    if command == "plan-evaluation-shards":
        return [
            *authority,
            "--max-lineages-per-shard",
            "32",
            "--out",
            "plan.json",
        ]
    ground = [*authority, "--plan", "plan.json", "--expected-plan-sha256", PIN]
    if command == "run-evaluation-shard":
        return [*ground, "--shard-index", "0", "--out", "shard-0"]
    if command == "merge-evaluation-shards":
        return [
            *ground,
            "--shard",
            "shard-0",
            "--expected-shard-receipt-sha256",
            OTHER_PIN,
            "--out",
            "merged",
        ]
    raise AssertionError(f"unknown command fixture: {command}")


def _prepared(*, ground: bool) -> SimpleNamespace:
    target_access = {
        "record_digest": "c" * 64,
        "partition": "test",
    }
    authority = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": {"record_digest": "d" * 64},
        "evaluation_partition": {"name": "test", "record_digest": "e" * 64},
        "record_digest": "f" * 64,
    }
    return SimpleNamespace(
        workflow="learned-complete-system",
        population=object(),
        tasks=("full-task-a", "full-task-b"),
        context=object(),
        run_coordinates={"training_seed_index": 0},
        execution_contract={"contract_kind": "fixture"},
        target_access_receipt=target_access if ground else None,
        ground_certificate_authority=authority,
        compute_class={"record_digest": "1" * 64},
        compute_provenance={"record_digest": "2" * 64},
    )


@pytest.mark.parametrize(
    "arguments",
    [_learned_arguments, _tuned_stock_arguments, _validation_tuning_arguments],
)
@pytest.mark.parametrize(
    ("command", "function_name"),
    [
        ("plan-evaluation-shards", "cmd_plan_evaluation_shards"),
        ("run-evaluation-shard", "cmd_run_evaluation_shard"),
        ("merge-evaluation-shards", "cmd_merge_evaluation_shards"),
    ],
)
def test_commands_support_all_three_evaluation_workflows(
    arguments,
    command: str,
    function_name: str,
) -> None:
    parsed = _parser().parse_args(arguments(command))

    assert parsed.func.__name__ == function_name
    assert parsed.quality_attestation == "publisher-attestation.json"
    assert parsed.ground_certificate_root == "ground/root.json"
    assert parsed.cluster == "apollo"
    assert parsed.execution_mode == "pinned-venv"


@pytest.mark.parametrize(
    "command",
    [
        "plan-evaluation-shards",
        "run-evaluation-shard",
        "merge-evaluation-shards",
    ],
)
def test_all_phases_require_pinned_ground_root_authority(
    command: str,
) -> None:
    parser = _parser()
    complete = _learned_arguments(command)
    root_flag = complete.index("--ground-certificate-root")
    missing_root = [*complete[:root_flag], *complete[root_flag + 2 :]]

    with pytest.raises(SystemExit):
        parser.parse_args(missing_root)

    parsed = parser.parse_args(complete)
    assert parsed.ground_certificate_root == "ground/root.json"
    assert parsed.expected_ground_certificate_root_sha256 == OTHER_PIN


def test_execution_environment_pin_is_required_before_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(_learned_arguments("plan-evaluation-shards"))
    args.expected_environment_lock_sha256 = None
    monkeypatch.setattr(
        workflow_cli,
        "prepare_evaluation_workflow",
        lambda *args, **kwargs: pytest.fail("missing environment pin reached preparation"),
    )

    with pytest.raises(ValueError, match="pinned-venv|required"):
        args.func(args)


def test_goose_array_index_must_match_shard_before_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(_learned_arguments("run-evaluation-shard"))
    args.cluster = "goose"
    args.slurm_partition = "gpu"
    monkeypatch.setenv("SLURM_JOB_ID", "17")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setattr(
        workflow_cli,
        "prepare_evaluation_workflow",
        lambda *args, **kwargs: pytest.fail("array drift reached preparation"),
    )

    with pytest.raises(ValueError, match="array index"):
        args.func(args)


def test_compute_class_excludes_hostname_and_bare_metal_is_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from isingfold.rl import cli as core
    from isingfold.rl import checkpoint

    lock = tmp_path / "requirements.lock"
    lock.write_text("fixture==1\n")
    lock_sha256 = workflow_cli.hashlib.sha256(lock.read_bytes()).hexdigest()
    args = SimpleNamespace(
        cluster="apollo",
        slurm_partition=None,
        execution_mode="pinned-venv",
        environment_lock=str(lock),
        expected_environment_lock_sha256=lock_sha256,
        container_runtime=None,
        expected_container_runtime_sha256=None,
        container_image=None,
        expected_container_image_sha256=None,
        threads=1,
        deterministic=True,
    )
    monkeypatch.setattr(
        checkpoint,
        "runtime_implementation_registry",
        lambda: {"schema": "fixture", "native_artifacts": {}, "dependencies": {}},
    )
    host = {"name": "apollo-a"}

    def runtime_platform():
        return {
            "hostname": host["name"],
            "system": "Linux",
            "release": "fixture",
            "machine": "x86_64",
            "processor": "fixture-cpu",
            "logical_cpu_count": 32,
            "slurm_partition": None,
        }

    monkeypatch.setattr(core, "_runtime_platform_identity", runtime_platform)
    device = SimpleNamespace(type="cpu")
    first_class, first_provenance = workflow_cli._compute_class_and_provenance(args, device=device)
    host["name"] = "apollo-b"
    second_class, second_provenance = workflow_cli._compute_class_and_provenance(
        args, device=device
    )

    assert first_class == second_class
    assert "hostname" not in str(first_class)
    assert first_provenance["hostname"] == "apollo-a"
    assert second_provenance["hostname"] == "apollo-b"
    assert first_provenance["record_digest"] != second_provenance["record_digest"]

    args.execution_mode = "bare-metal"
    args.environment_lock = None
    args.expected_environment_lock_sha256 = None
    diagnostic_class, _ = workflow_cli._compute_class_and_provenance(args, device=device)
    assert diagnostic_class["publication_eligible"] is False


def test_plan_uses_public_only_full_population_and_caps_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(_learned_arguments("plan-evaluation-shards"))
    public = _prepared(ground=False)
    plan = _FakePlan()
    observations: dict[str, object] = {}

    def prepare(namespace, *, target_mode):
        observations["target_mode"] = target_mode
        assert namespace is args
        return public

    def build(**kwargs):
        observations["population"] = kwargs["population"]
        observations["max_lineages_per_shard"] = kwargs["max_lineages_per_shard"]
        assert kwargs["run_coordinates"] is public.run_coordinates
        assert kwargs["execution_contract"] is public.execution_contract
        assert kwargs["quality_authority"] is public.ground_certificate_authority
        assert kwargs["compute_class"] is public.compute_class
        return plan

    def write(path, value):
        observations["write"] = (path, value)
        return PIN

    monkeypatch.setattr(workflow_cli, "prepare_evaluation_workflow", prepare)
    monkeypatch.setattr(workflow_cli, "build_evaluation_plan", build)
    monkeypatch.setattr(workflow_cli, "write_evaluation_plan", write)

    args.func(args)

    assert observations == {
        "target_mode": "public",
        "population": public.population,
        "max_lineages_per_shard": 32,
        "write": ("plan.json", plan),
    }


def test_partition_preparation_keeps_plan_public_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isingfold.rl import cli as core

    args = SimpleNamespace(corpus="prepared-v4")
    opened = SimpleNamespace(tasks=("public-a", "public-b"), target_access=None)
    authority = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": {},
        "evaluation_partition": {},
        "record_digest": PIN,
    }
    observations: dict[str, object] = {}

    def public_loader(directory, *, partition, include_evaluator):
        observations["public"] = (directory, partition, include_evaluator)
        return opened

    monkeypatch.setattr(workflow_cli, "load_prepared_partition", public_loader)
    monkeypatch.setattr(core, "_quality_attestation_pin", lambda value: object())
    monkeypatch.setattr(core, "_quality_authority_identity", lambda *a, **k: authority)
    monkeypatch.setattr(
        core,
        "_validate_quality_authority_binding",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(
        core,
        "_load_quality_partition",
        lambda *a, **k: pytest.fail("public plan opened a ground partition"),
    )

    rows, target_access, authority = workflow_cli._open_evaluation_partition(
        args, partition="test", target_mode="public"
    )

    assert rows == opened.tasks
    assert target_access is None
    assert authority == {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": {},
        "evaluation_partition": {},
        "record_digest": PIN,
    }
    assert observations["public"] == ("prepared-v4", "test", False)


def test_ground_partition_preparation_binds_typed_projections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isingfold.rl import cli as core

    args = SimpleNamespace(
        corpus="prepared-v4",
        ground_certificate_root="ground/root.json",
        expected_ground_certificate_root_sha256=OTHER_PIN,
    )
    pin = object()
    access = {"record_digest": "c" * 64}
    authority = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": {"record_digest": "d" * 64},
        "evaluation_partition": {"name": "val", "record_digest": "e" * 64},
        "record_digest": "f" * 64,
    }
    observations: dict[str, object] = {}

    monkeypatch.setattr(core, "_quality_attestation_pin", lambda value: pin)

    def ground_loader(corpus, **kwargs):
        observations["ground"] = (corpus, kwargs)
        return ("ground-a", "ground-b"), authority, access, {"receipt": "ground"}

    monkeypatch.setattr(core, "_load_quality_partition", ground_loader)
    monkeypatch.setattr(
        core,
        "_validate_quality_authority_binding",
        lambda value, **kwargs: value,
    )

    rows, access, authority = workflow_cli._open_evaluation_partition(
        args, partition="val", target_mode="ground"
    )

    assert rows == ("ground-a", "ground-b")
    assert access == {"record_digest": "c" * 64}
    assert authority["evaluation_partition"]["name"] == "val"
    assert observations["ground"] == (
        "prepared-v4",
        {
            "partition": "val",
            "pin": pin,
            "role": "evaluation_partition",
        },
    )


def test_plan_rejects_more_than_32_lineages_per_shard_before_corpus_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _learned_arguments("plan-evaluation-shards")
    arguments[arguments.index("32")] = "33"
    monkeypatch.setattr(
        workflow_cli,
        "prepare_evaluation_workflow",
        lambda *args, **kwargs: pytest.fail("invalid shard cap opened the corpus"),
    )

    with pytest.raises(SystemExit):
        _parser().parse_args(arguments)


@pytest.mark.parametrize(
    ("arguments", "missing_attribute"),
    [
        (_tuned_stock_arguments, "expected_tuning_registry_sha256"),
        (
            _tuned_stock_arguments,
            "expected_external_tuning_selection_sha256",
        ),
        (_validation_tuning_arguments, "expected_registry_sha256"),
    ],
)
def test_external_workflows_reject_missing_out_of_band_pins_before_preparation(
    arguments,
    missing_attribute: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(arguments("plan-evaluation-shards"))
    setattr(args, missing_attribute, None)
    monkeypatch.setattr(
        workflow_cli,
        "prepare_evaluation_workflow",
        lambda *args, **kwargs: pytest.fail("missing external pin opened an input"),
    )

    with pytest.raises(ValueError, match="pin|required"):
        args.func(args)


def test_run_reauthenticates_full_population_before_execution_and_binds_ground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(_learned_arguments("run-evaluation-shard"))
    prepared = _prepared(ground=True)
    sealed = _FakePlan()
    observations: dict[str, object] = {}

    monkeypatch.setattr(
        workflow_cli,
        "load_evaluation_plan",
        lambda path, *, expected_sha256: (
            observations.update(plan_load=(path, expected_sha256)) or sealed
        ),
    )

    def prepare(namespace, *, target_mode):
        observations["target_mode"] = target_mode
        return prepared

    def build(**kwargs):
        observations["population"] = kwargs["population"]
        observations["max_lineages_per_shard"] = kwargs["max_lineages_per_shard"]
        return sealed

    def execute(value, plan, shard_index):
        assert value is prepared
        assert value.tasks == ("full-task-a", "full-task-b")
        observations["execution"] = (plan, shard_index)
        return ("typed-receipt",)

    def publish(destination, **kwargs):
        observations["publication"] = (destination, kwargs)
        return {"record_digest": "f" * 64}, OTHER_PIN

    monkeypatch.setattr(workflow_cli, "prepare_evaluation_workflow", prepare)
    monkeypatch.setattr(workflow_cli, "build_evaluation_plan", build)
    monkeypatch.setattr(workflow_cli, "execute_evaluation_shard", execute)
    monkeypatch.setattr(workflow_cli, "publish_evaluation_shard", publish)

    args.func(args)

    destination, publication = observations["publication"]
    assert observations["plan_load"] == ("plan.json", PIN)
    assert observations["target_mode"] == "ground"
    assert observations["population"] is prepared.population
    assert observations["max_lineages_per_shard"] == 32
    assert observations["execution"] == (sealed, 0)
    assert destination == "shard-0"
    assert publication["plan"] is sealed
    assert publication["shard_index"] == 0
    assert publication["receipts"] == ("typed-receipt",)
    assert publication["target_access_receipt"] is prepared.target_access_receipt
    assert publication["ground_certificate_authority"] is (prepared.ground_certificate_authority)
    assert publication["compute_provenance"] is prepared.compute_provenance


def test_run_rejects_plan_drift_before_selected_shard_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(_learned_arguments("run-evaluation-shard"))
    monkeypatch.setattr(
        workflow_cli,
        "load_evaluation_plan",
        lambda *args, **kwargs: _FakePlan("sealed"),
    )
    monkeypatch.setattr(
        workflow_cli,
        "prepare_evaluation_workflow",
        lambda *args, **kwargs: _prepared(ground=True),
    )
    monkeypatch.setattr(
        workflow_cli,
        "build_evaluation_plan",
        lambda **kwargs: _FakePlan("different-full-population"),
    )
    monkeypatch.setattr(
        workflow_cli,
        "execute_evaluation_shard",
        lambda *args, **kwargs: pytest.fail("plan drift reached shard execution"),
    )

    with pytest.raises(ValueError, match="plan|population|authority"):
        args.func(args)


def test_merge_reopens_ground_partition_and_replays_the_full_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(_learned_arguments("merge-evaluation-shards"))
    prepared = _prepared(ground=True)
    plan = _FakePlan()
    loaded_shard = SimpleNamespace(
        target_access_receipt=prepared.target_access_receipt,
        ground_certificate_authority=prepared.ground_certificate_authority,
    )
    merged = SimpleNamespace(
        target_access_receipt=prepared.target_access_receipt,
        ground_certificate_authority=prepared.ground_certificate_authority,
    )
    observations: dict[str, object] = {}

    monkeypatch.setattr(workflow_cli, "load_evaluation_plan", lambda *a, **k: plan)
    monkeypatch.setattr(
        workflow_cli,
        "prepare_evaluation_workflow",
        lambda namespace, *, target_mode: observations.update(target_mode=target_mode) or prepared,
    )
    monkeypatch.setattr(workflow_cli, "build_evaluation_plan", lambda **kwargs: plan)

    def load_shard(path, *, plan, expected_receipt_sha256):
        observations["shard_load"] = (path, plan, expected_receipt_sha256)
        return loaded_shard

    def merge(value, shards, *, tasks, context):
        observations["merge"] = (value, shards, tasks, context)
        return merged

    def publish(path, *, plan, merged):
        observations["publish"] = (path, plan, merged)
        return {"record_digest": "f" * 64}, OTHER_PIN

    monkeypatch.setattr(workflow_cli, "load_evaluation_shard", load_shard)
    monkeypatch.setattr(workflow_cli, "merge_evaluation_shards", merge)
    monkeypatch.setattr(workflow_cli, "publish_merged_evaluation", publish)

    args.func(args)

    assert observations["target_mode"] == "ground"
    assert observations["shard_load"] == ("shard-0", plan, OTHER_PIN)
    assert observations["merge"] == (
        plan,
        (loaded_shard,),
        prepared.tasks,
        prepared.context,
    )
    assert observations["publish"] == ("merged", plan, merged)


def test_merge_rejects_shards_bound_to_another_ground_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(_learned_arguments("merge-evaluation-shards"))
    prepared = _prepared(ground=True)
    plan = _FakePlan()
    stale_shard = SimpleNamespace(
        target_access_receipt={"record_digest": "1" * 64},
        ground_certificate_authority={
            "global": {"record_digest": "2" * 64},
            "partition": {"name": "test", "record_digest": "3" * 64},
        },
    )
    monkeypatch.setattr(workflow_cli, "load_evaluation_plan", lambda *a, **k: plan)
    monkeypatch.setattr(
        workflow_cli,
        "prepare_evaluation_workflow",
        lambda *a, **k: prepared,
    )
    monkeypatch.setattr(workflow_cli, "build_evaluation_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        workflow_cli,
        "load_evaluation_shard",
        lambda *a, **k: stale_shard,
    )
    monkeypatch.setattr(
        workflow_cli,
        "merge_evaluation_shards",
        lambda *a, **k: pytest.fail("mixed ground authority reached merge"),
    )

    with pytest.raises(ValueError, match="ground|target-access|authority"):
        args.func(args)


def test_evaluation_shard_hpc_launchers_respect_host_schedulers() -> None:
    apollo = (ROOT / "scripts" / "apollo_evaluation_shards.sh").read_text()
    goose = (ROOT / "scripts" / "goose_evaluation_shards.sbatch").read_text()
    publication_runtime = (ROOT / "scripts" / "publication_runtime.sh").read_text()

    assert "sbatch" not in apollo and "/opt/slurm/bin/srun" not in apollo
    assert "run-evaluation-shard" in apollo
    assert "--cluster apollo" in apollo
    assert "--execution-mode pinned-venv" in apollo
    assert "ISINGFOLD_ENV_LOCK_SHA256" in apollo
    assert "SLURM_JOB_ID" in goose and "SLURM_ARRAY_TASK_ID" in goose
    assert "/opt/slurm/bin/srun" in goose
    assert "run-evaluation-shard" in goose
    assert '"${container_exec[@]}"' in goose
    assert "require_goose_publication_runtime" in goose
    assert "--cluster goose" in goose
    assert "--execution-mode apptainer" in goose
    assert "ISINGFOLD_APPTAINER_SHA256" in goose + publication_runtime
    assert "ISINGFOLD_CONTAINER_IMAGE_SHA256" in goose + publication_runtime
    for text in (apollo, goose):
        assert "--workflow" in text
        assert "--expected-plan-sha256" in text
        assert "--shard-index" in text
        assert "verify_runtime_source.sh" in text

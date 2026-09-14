from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import networkx as nx

import isingfold.rl.data.quality as quality_module
import isingfold.rl.data.quality_resolution_plan as plan_module
from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import (
    PreparedDesignCondition,
    PreparedProvenance,
    PreparedTask,
)
from isingfold.rl.data.quality import (
    QUALITY_INITIALIZER_BANK_PROTOCOL,
    continuation_seed,
)
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_resolution_plan import (
    QUALITY_RESOLUTION_PLAN_SCHEMA,
    QUALITY_RESOLUTION_PLAN_VERSION,
    PlannedResolutionAction,
    PlannedResolutionLineage,
    PlannedResolutionRow,
    QualityResolutionPlanAuthority,
    ResolutionPreparedCensus,
    ResolutionProductionPlan,
    build_quality_resolution_plan,
    load_quality_resolution_plan,
    load_quality_resolution_planning_inputs,
    materialize_resolution_production_plan,
    publish_quality_resolution_plan,
)
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "quality_resolution_v1.json"
GRID = ROOT / "configs" / "rl_grid_hybrid_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _planning_inputs():
    return load_quality_resolution_planning_inputs(
        CONFIG,
        expected_config_sha256=_sha(CONFIG),
        grid_path=GRID,
        expected_grid_sha256=_sha(GRID),
    )


def _authority(
    production: ResolutionProductionPlan | None = None,
) -> QualityResolutionPlanAuthority:
    production = _production() if production is None else production
    context = quality_module._jsonable(  # noqa: SLF001 - exact production serializer fixture
        Context(qubit_cap=4, n_est_reads=256, num_sweeps=1)
    )
    assert isinstance(context, dict)
    implementation = {
        "planner": "quality-resolution-plan-v1",
        "quality_implementation_contract_digest": "0" * 64,
        "quality_module_sha256": "1" * 64,
        "quality_resolution_delta_module_sha256": "2" * 64,
    }
    return QualityResolutionPlanAuthority(
        prepared_manifest_sha256="2" * 64,
        prepared_manifest_record_digest="3" * 64,
        prepared_train_census_record_digest=production.source_census.record_digest,
        corpus_design_manifest_sha256="4" * 64,
        publisher_id="fixture-publisher",
        publisher_attestation_sha256="5" * 64,
        publisher_attestation_record_digest="6" * 64,
        target_authority_record_digest="7" * 64,
        ground_root_sha256="8" * 64,
        ground_root_record_digest="9" * 64,
        verifier_identity_digest="a" * 64,
        selector_digest="b" * 64,
        selector_file_sha256="c" * 64,
        selector_fit_receipt_sha256="d" * 64,
        selector_fit_record_digest="e" * 64,
        normalizer_digest="f" * 64,
        selector_device="cpu",
        selector_device_parity_sha256="0" * 64,
        selector_device_parity_record_digest="1" * 64,
        context=context,
        context_digest=content_digest(context),
        implementation=implementation,
        implementation_digest=content_digest(implementation),
    )


def _stratum(index: int) -> tuple[tuple[str, str | bool], ...]:
    return (
        ("application_family", ("portfolio", "graph-cut")[index % 2]),
        ("problem_origin", ("application-derived", "synthetic")[index % 2]),
        ("host_family", ("chimera", "pegasus", "zephyr")[index % 3]),
        ("fault_status", ("none", "faulted")[index % 2]),
        ("distribution_regime", ("iid", "ood")[index % 2]),
        ("calibration_status", ("not_applicable", "recorded")[index % 2]),
        ("embedding_difficulty", ("easy", "hard")[index % 2]),
        ("sampling_difficulty", ("hard", "easy")[index % 2]),
        ("decision_difficulty", ("easy", "medium", "hard")[index % 3]),
    )


def _bank_contract(count: int = 128) -> dict[str, object]:
    planning_context = quality_module._jsonable(  # noqa: SLF001
        Context(qubit_cap=4, n_est_reads=256, num_sweeps=1)
    )
    assert isinstance(planning_context, dict)
    body: dict[str, object] = {
        "access_receipt_record_digest": "a" * 64,
        "conditional_episode_count": count,
        "config_digest": "b" * 64,
        "context_digest": content_digest(planning_context),
        "episode_schedule_start": 0,
        "episode_schedule_stop_exclusive": count,
        "manifest_record_digest": "d" * 64,
        "manifest_sha256": "e" * 64,
        "opened_evaluator_data": False,
        "partition": "train",
        "plan_record_digest": "f" * 64,
        "prepared_manifest_sha256": "2" * 64,
        "protocol": QUALITY_INITIALIZER_BANK_PROTOCOL,
        "publication_eligible": True,
        "restart_cache_slots_per_episode": 2,
        "schema": "isingfold.quality-initializer-bank-contract",
        "schema_version": 1,
        "training_seed": 1103,
    }
    return {**body, "record_digest": content_digest(body)}


def _production(count: int = 128) -> ResolutionProductionPlan:
    rows: list[PlannedResolutionRow] = []
    lineages: list[PlannedResolutionLineage] = []
    for index in range(count):
        lineage = f"lineage-{index:04d}"
        task_id = f"task-{index:04d}"
        instance_id = f"instance-{index:04d}"
        state_fingerprint = hashlib.sha256(f"state-{index}".encode()).hexdigest()
        environment_seed = index + 100
        action = PlannedResolutionAction(
            action_index=0,
            payload_key=f"payload-{index}",
            opcode="COMMIT",
            selected_payload_digest=hashlib.sha256(f"payload-{index}".encode()).hexdigest(),
            applied_action_record_digest=None,
            continuation_seeds=tuple(
                continuation_seed(environment_seed, task_id, state_fingerprint, 0, offset)
                for offset in range(128)
            ),
        )
        row = PlannedResolutionRow.create(
            base_lineage=lineage,
            task_id=task_id,
            instance_id=instance_id,
            lineage_schedule_index=index,
            state_schedule_index=0,
            environment_seed=environment_seed,
            initializer_bank_episode_index=index,
            initializer_bootstrap_record_digest=hashlib.sha256(
                f"bootstrap-{index}".encode()
            ).hexdigest(),
            prefix=(),
            state_fingerprint=state_fingerprint,
            support_fingerprint=hashlib.sha256(f"support-{index}".encode()).hexdigest(),
            action_envelope_record_digest=hashlib.sha256(f"envelope-{index}".encode()).hexdigest(),
            action_provenance_fingerprint=hashlib.sha256(
                f"provenance-{index}".encode()
            ).hexdigest(),
            actions=(action,),
        )
        rows.append(row)
        lineages.append(
            PlannedResolutionLineage(
                base_lineage=lineage,
                schedule_index=index,
                task_ids=(task_id,),
                stratum=_stratum(index),
            )
        )
    schedule = {
        lineage: position
        for position, lineage in enumerate(
            sorted(
                (item.base_lineage for item in lineages),
                key=lambda lineage: (
                    plan_module._quality_protocol_seed(907, "quality-lineage-sample", lineage),
                    lineage,
                ),
            )
        )
    }
    lineages = [replace(item, schedule_index=schedule[item.base_lineage]) for item in lineages]
    rows = [replace(item, lineage_schedule_index=schedule[item.base_lineage]) for item in rows]
    census = ResolutionPreparedCensus(
        prepared_manifest_sha256="2" * 64,
        lineage_tasks=tuple((lineage.base_lineage, lineage.task_ids) for lineage in lineages),
    )
    return ResolutionProductionPlan(
        lineages=tuple(lineages),
        rows=tuple(rows),
        source_census=census,
        initializer_bank_contract=_bank_contract(count),
    )


def _public_prepared(index: int, *, partition: str = "train") -> PreparedTask:
    lineage = f"public-lineage-{index:04d}"
    task_id = f"public-task-{index:04d}"
    logical = nx.path_graph(2)
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts(
        {0: 0.25, 1: -0.5},
        {(0, 1): -1.0},
    )
    embedding = {0: frozenset((0, 1)), 1: frozenset((2, 3))}
    task = EmbeddingTask(
        name=task_id,
        logical=logical,
        host=host,
        problem=problem,
        ground_energy=None,
        lineage=lineage,
        initial_embedding=embedding,
    )
    stratum = dict(_stratum(index))
    condition = PreparedDesignCondition(
        base_lineage_key=lineage,
        learning_partition=partition,
        application_family=str(stratum["application_family"]),
        problem_origin=str(stratum["problem_origin"]),
        host_family=str(stratum["host_family"]),
        fault_status=str(stratum["fault_status"]),
        distribution_regime=str(stratum["distribution_regime"]),
        calibration_status=str(stratum["calibration_status"]),
        calibration_sha256=None,
        embedding_difficulty=str(stratum["embedding_difficulty"]),
        sampling_difficulty=str(stratum["sampling_difficulty"]),
        decision_difficulty=str(stratum["decision_difficulty"]),
        registry_row_digest=hashlib.sha256(f"condition-{index}".encode()).hexdigest(),
    )
    return PreparedTask(
        task=task,
        task_id=task_id,
        instance_id=f"public-instance-{index:04d}",
        partition=partition,
        initializer_record_digest=hashlib.sha256(f"initializer-{index}".encode()).hexdigest(),
        public_instance_record_digest=hashlib.sha256(
            f"policy-instance-{index}".encode()
        ).hexdigest(),
        reference_status=None,
        certificate_digest=None,
        evaluator_protocol_digest=None,
        prepared_schema_version=4,
        corpus_scope="production-designed-v4",
        design_condition=condition,
    )


def _public_provenance(
    item: PreparedTask,
    *,
    source_partition: str = "isingfold-corpus-v4",
) -> PreparedProvenance:
    def digest(label: str) -> str:
        return content_digest({"label": label, "task_id": item.task_id})

    return PreparedProvenance(
        base_parent_lineage=str(item.task.lineage),
        source_logical_lineage=f"source-{item.task_id}",
        descendant_transform_kinds=("identity",),
        descendant_transform_sha256=digest("transform"),
        nominal_topology="path",
        nominal_size=item.task.host.number_of_nodes(),
        pristine_host_sha256=digest("pristine-host"),
        active_topology="path",
        active_host_sha256=digest("active-host"),
        host_artifact_sha256=digest("host-artifact"),
        fault_status="none",
        fault_mask_sha256=digest("fault-mask"),
        calibration_status="not_applicable",
        calibration_sha256=None,
        distribution_regime="iid",
        distribution_stratum="quality-resolution-fixture",
        source_partition=source_partition,
        source_release_id="quality-resolution-fixture-release",
        source_release_manifest_sha256=digest("release"),
        split_manifest_sha256=digest("split"),
        group_id=f"group-{item.task_id}",
        group_record_digest=digest("group"),
        instance_record_digest=digest("instance"),
        source_record_digest=digest("source"),
        record_digest=digest("provenance"),
    )


@pytest.fixture(autouse=True)
def _smaller_unit_population(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan_module, "MINIMUM_PRODUCTION_LINEAGES", 128)


def test_planning_inputs_require_raw_external_pins_and_registered_grid_minima() -> None:
    loaded = _planning_inputs()

    assert loaded.config.study_id == "if-quality-resolution-v1"
    assert loaded.config_sha256 == _sha(CONFIG)
    assert loaded.grid_sha256 == _sha(GRID)
    assert loaded.grid_resolution_minima == {
        "min_resolved_lineages": 128,
        "min_resolved_rows": 128,
    }

    with pytest.raises(QualityResolutionError, match="config.*out-of-band pin"):
        load_quality_resolution_planning_inputs(
            CONFIG,
            expected_config_sha256="0" * 64,
            grid_path=GRID,
            expected_grid_sha256=_sha(GRID),
        )
    with pytest.raises(QualityResolutionError, match="grid.*out-of-band pin"):
        load_quality_resolution_planning_inputs(
            CONFIG,
            expected_config_sha256=_sha(CONFIG),
            grid_path=GRID,
            expected_grid_sha256="0" * 64,
        )


def test_plan_samples_only_after_full_plan_identity_and_assigns_whole_lineages() -> None:
    production = _production()
    plan = build_quality_resolution_plan(_planning_inputs(), _authority(), production)
    record = plan.as_dict()

    assert record["schema"] == QUALITY_RESOLUTION_PLAN_SCHEMA
    assert record["schema_version"] == QUALITY_RESOLUTION_PLAN_VERSION
    assert record["production_plan"]["lineage_count"] == 128
    assert record["production_plan_digest"] == production.record_digest
    assert record["sample"]["population_plan_digest"] == production.record_digest
    assert len(record["sample"]["selected_lineages"]) == 128
    assert len(record["shards"]) == 64
    assert all(len(shard["lineages"]) == 2 for shard in record["shards"])
    assert [stage["continuation_range"] for stage in record["stages"]] == [
        [0, 12],
        [12, 16],
        [16, 24],
        [24, 32],
        [32, 48],
        [48, 64],
        [64, 96],
        [96, 128],
    ]
    assert len(record["work_schedule"]) == 8 * 64
    assert len({unit["work_id"] for unit in record["work_schedule"]}) == 8 * 64

    assigned = [lineage for shard in record["shards"] for lineage in shard["lineages"]]
    assert len(assigned) == len(set(assigned)) == 128
    assert set(assigned) == set(record["sample"]["selected_lineages"])
    assert all(
        set(unit["lineages"]) == set(record["shards"][unit["shard_index"]]["lineages"])
        for unit in record["work_schedule"]
    )


def test_plan_bytes_are_order_invariant_and_contain_no_target_payload() -> None:
    production = _production()
    reversed_production = ResolutionProductionPlan(
        lineages=tuple(reversed(production.lineages)),
        rows=tuple(reversed(production.rows)),
        source_census=production.source_census,
        initializer_bank_contract=production.initializer_bank_contract,
    )

    first = build_quality_resolution_plan(_planning_inputs(), _authority(), production)
    replay = build_quality_resolution_plan(_planning_inputs(), _authority(), reversed_production)
    first_raw = canonical_json_bytes(first.as_dict()) + b"\n"
    replay_raw = canonical_json_bytes(replay.as_dict()) + b"\n"

    assert first_raw == replay_raw
    serialized = first_raw.decode("utf-8")
    for forbidden in (
        "ground_energy",
        "reference_energy",
        "target_access",
        "evaluator_targets",
        "targets/",
        "continuation_rewards",
    ):
        assert forbidden not in serialized


def test_plan_rejects_incomplete_population_or_non_train_projection() -> None:
    with pytest.raises(QualityResolutionError, match="at least 128"):
        build_quality_resolution_plan(_planning_inputs(), _authority(), _production(127))

    production = _production()
    with pytest.raises(QualityResolutionError, match="train-only"):
        replace(production.rows[0], partition="val")


def test_plan_rejects_a_projection_that_omits_an_authenticated_source_lineage() -> None:
    production = _production()
    expanded_census = ResolutionPreparedCensus(
        prepared_manifest_sha256="2" * 64,
        lineage_tasks=(
            *production.source_census.lineage_tasks,
            ("lineage-omitted", ("task-omitted",)),
        ),
    )
    subset = replace(production, source_census=expanded_census)
    authority = replace(
        _authority(production),
        prepared_train_census_record_digest=expanded_census.record_digest,
    )

    with pytest.raises(QualityResolutionError, match="exact all-train lineage census"):
        build_quality_resolution_plan(_planning_inputs(), authority, subset)


def test_plan_recursively_rejects_a_nested_target_payload() -> None:
    production = _production()
    authority = _authority(production)
    context = {**dict(authority.context), "nested": {"ground_energy": -1.0}}
    authority = replace(
        authority,
        context=context,
        context_digest=content_digest(context),
    )

    with pytest.raises(QualityResolutionError, match="forbidden evaluator field"):
        build_quality_resolution_plan(_planning_inputs(), authority, production)


def test_plan_rejects_initializer_bank_from_another_context() -> None:
    production = _production()
    bank = dict(production.initializer_bank_contract)
    bank["context_digest"] = "f" * 64
    body = {key: value for key, value in bank.items() if key != "record_digest"}
    bank["record_digest"] = content_digest(body)
    changed = replace(production, initializer_bank_contract=bank)

    with pytest.raises(QualityResolutionError, match="Context differs"):
        build_quality_resolution_plan(_planning_inputs(), _authority(changed), changed)


def test_materializer_rejects_prepared_row_not_in_bank_source_identity(
    tmp_path: Path,
) -> None:
    from tests.unit.test_rl_initializer_bank import _context as bank_context
    from tests.unit.test_rl_quality_initializer_bank import _sealed_test_bank

    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    changed = replace(public, initializer_record_digest="f" * 64)

    with pytest.raises(QualityResolutionError, match="bank source identity"):
        materialize_resolution_production_plan(
            [changed],
            context=bank_context(),
            selector=fixed_strength_selector(),
            config=_planning_inputs().config,
            source_corpus_manifest_sha256=bank.plan.prepared_manifest_sha256,
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=manifest_sha256,
            allow_test_initializer_bank=True,
        )


def test_row_rejects_wrong_or_reordered_continuation_seed_registry() -> None:
    production = _production()
    row = production.rows[0]
    action = row.actions[0]
    wrong = replace(action, continuation_seeds=tuple(reversed(action.continuation_seeds)))

    with pytest.raises(QualityResolutionError, match="continuation seed registry"):
        build_quality_resolution_plan(
            _planning_inputs(),
            _authority(),
            ResolutionProductionPlan(
                lineages=production.lineages,
                rows=(replace(row, actions=(wrong,)), *production.rows[1:]),
                source_census=production.source_census,
                initializer_bank_contract=production.initializer_bank_contract,
            ),
        )


def test_plan_publication_is_immutable_and_loader_requires_every_external_pin(
    tmp_path: Path,
) -> None:
    plan = build_quality_resolution_plan(_planning_inputs(), _authority(), _production())
    path = tmp_path / "resolution-plan.json"
    raw_sha = publish_quality_resolution_plan(path, plan)

    loaded = load_quality_resolution_plan(
        path,
        expected_plan_sha256=raw_sha,
        expected_config_sha256=_sha(CONFIG),
        expected_grid_sha256=_sha(GRID),
        expected_prepared_manifest_sha256="2" * 64,
        expected_prepared_train_census_record_digest=plan.as_dict()["prepared_corpus"][
            "train_census_record_digest"
        ],
        expected_publisher_attestation_sha256="5" * 64,
        expected_ground_root_sha256="8" * 64,
        expected_selector_file_sha256="c" * 64,
        expected_selector_device_parity_sha256="0" * 64,
    )
    assert loaded.as_dict() == plan.as_dict()

    with pytest.raises(FileExistsError):
        publish_quality_resolution_plan(path, plan)
    with pytest.raises(QualityResolutionError, match="plan.*out-of-band pin"):
        load_quality_resolution_plan(
            path,
            expected_plan_sha256="f" * 64,
            expected_config_sha256=_sha(CONFIG),
            expected_grid_sha256=_sha(GRID),
            expected_prepared_manifest_sha256="2" * 64,
            expected_prepared_train_census_record_digest=plan.as_dict()["prepared_corpus"][
                "train_census_record_digest"
            ],
            expected_publisher_attestation_sha256="5" * 64,
            expected_ground_root_sha256="8" * 64,
            expected_selector_file_sha256="c" * 64,
            expected_selector_device_parity_sha256="0" * 64,
        )
    with pytest.raises(QualityResolutionError, match="selector file pin"):
        load_quality_resolution_plan(
            path,
            expected_plan_sha256=raw_sha,
            expected_config_sha256=_sha(CONFIG),
            expected_grid_sha256=_sha(GRID),
            expected_prepared_manifest_sha256="2" * 64,
            expected_prepared_train_census_record_digest=plan.as_dict()["prepared_corpus"][
                "train_census_record_digest"
            ],
            expected_publisher_attestation_sha256="5" * 64,
            expected_ground_root_sha256="8" * 64,
            expected_selector_file_sha256="f" * 64,
            expected_selector_device_parity_sha256="0" * 64,
        )


def test_loader_rejects_a_self_consistent_alternate_plan_without_original_raw_pin(
    tmp_path: Path,
) -> None:
    plan = build_quality_resolution_plan(_planning_inputs(), _authority(), _production())
    record = plan.as_dict()
    original_path = tmp_path / "original.json"
    original_sha = publish_quality_resolution_plan(original_path, plan)

    record["publisher"]["publisher_id"] = "substituted-publisher"
    body = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = content_digest(body)
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(canonical_json_bytes(record) + b"\n")

    with pytest.raises(QualityResolutionError, match="plan.*out-of-band pin"):
        load_quality_resolution_plan(
            alternate,
            expected_plan_sha256=original_sha,
            expected_config_sha256=_sha(CONFIG),
            expected_grid_sha256=_sha(GRID),
            expected_prepared_manifest_sha256="2" * 64,
            expected_prepared_train_census_record_digest=plan.as_dict()["prepared_corpus"][
                "train_census_record_digest"
            ],
            expected_publisher_attestation_sha256="5" * 64,
            expected_ground_root_sha256="8" * 64,
            expected_selector_file_sha256="c" * 64,
            expected_selector_device_parity_sha256="0" * 64,
        )


def test_materializer_rejects_the_unpinned_legacy_initializer_path() -> None:
    with pytest.raises(QualityResolutionError, match="pinned initializer bank"):
        materialize_resolution_production_plan(
            [_public_prepared(0)],
            context=Context(qubit_cap=4, n_est_reads=256),
            selector=fixed_strength_selector(),
            config=_planning_inputs().config,
            source_corpus_manifest_sha256="2" * 64,
        )


def test_public_task_validation_keeps_source_partition_distinct_from_learning_split() -> None:
    public = _public_prepared(0, partition="train")
    public = replace(public, provenance=_public_provenance(public))

    assert plan_module._validate_public_prepared_task(public) is public.design_condition


def test_public_task_validation_rejects_wrong_provenance_base_lineage() -> None:
    public = _public_prepared(0, partition="train")
    provenance = _public_provenance(public)
    public = replace(
        public,
        provenance=replace(provenance, base_parent_lineage="different-lineage"),
    )

    with pytest.raises(QualityResolutionError, match="provenance crosses"):
        plan_module._validate_public_prepared_task(public)


def test_materializer_rejects_target_bearing_or_non_train_tasks_before_selector_use() -> None:
    class ForbiddenSelector:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("selector was reached after a target-leak violation")

    public = _public_prepared(0)
    target_bearing = replace(public, task=replace(public.task, ground_energy=-1.75))
    with pytest.raises(QualityResolutionError, match="evaluator target"):
        materialize_resolution_production_plan(
            [target_bearing],
            context=Context(qubit_cap=4, n_est_reads=256),
            selector=ForbiddenSelector(),
            config=_planning_inputs().config,
            source_corpus_manifest_sha256="2" * 64,
        )

    with pytest.raises(QualityResolutionError, match="train-only"):
        materialize_resolution_production_plan(
            [_public_prepared(0, partition="val")],
            context=Context(qubit_cap=4, n_est_reads=256),
            selector=ForbiddenSelector(),
            config=_planning_inputs().config,
            source_corpus_manifest_sha256="2" * 64,
        )

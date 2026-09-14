from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_design import (
    CorpusDesignError,
    CorpusDesignProfile,
    DifficultyRule,
    LineageFact,
    MeasurementBudget,
    TaskFact,
    build_corpus_design_v2,
)

PARTITIONS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rules() -> tuple[DifficultyRule, ...]:
    return (
        DifficultyRule(
            field="decision_difficulty",
            metric="decision_score",
            hard_if="greater-than-or-equal",
            threshold=0.5,
        ),
        DifficultyRule(
            field="embedding_difficulty",
            metric="embedding_score",
            hard_if="greater-than-or-equal",
            threshold=0.5,
        ),
        DifficultyRule(
            field="sampling_difficulty",
            metric="sampling_score",
            hard_if="greater-than-or-equal",
            threshold=0.5,
        ),
    )


def _lineage(
    partition: str,
    *,
    family: str,
    generator_kind: str,
    measurements: tuple[tuple[str, float], ...],
) -> LineageFact:
    marker = {"train": "1", "val": "2", "test": "3"}[partition]
    return LineageFact(
        base_lineage_key=f"problem-sha256:{marker * 64}",
        application_family=family,
        generator_id=f"embedbench/{family}/v1",
        generator_implementation_sha256=(str(int(marker) + 3) * 64),
        generator_kind=generator_kind,
        source_instance_record_digests=((str(int(marker) + 6) * 64),),
        measurements=measurements,
    )


def _task(
    partition: str,
    lineage: LineageFact,
    *,
    topology: str,
    faulted: bool,
    regime: str,
) -> TaskFact:
    marker = {"train": "a", "val": "b", "test": "c"}[partition]
    host_artifact = ("d" if faulted else "e") * 64
    return TaskFact(
        group_id=f"group-{partition}",
        base_parent_lineage=lineage.base_lineage_key,
        active_topology_identity={
            "host_artifact_sha256": host_artifact,
            "host_sha256": marker * 64,
            "topology": topology,
        },
        nominal_topology_identity={
            "pristine_host_sha256": "f" * 64,
            "size": 4,
            "topology": topology,
        },
        fault_identity={
            "fault_mask_sha256": host_artifact,
            "status": "faulted" if faulted else "none",
        },
        calibration_identity={
            "calibration_sha256": None,
            "status": "not_applicable",
        },
        descendant_transform_identity={
            "kinds": ["fault"] if faulted else ["identity"],
            "transform_sha256": marker * 64,
        },
        distribution={
            "learning_partition": partition,
            "regime": regime,
            "source_partition": "release-v2",
            "stratum": f"{partition}/{regime}/{topology}",
        },
    )


def _facts() -> tuple[tuple[LineageFact, ...], tuple[TaskFact, ...]]:
    lineages = (
        _lineage(
            "train",
            family="frustrated-loop",
            generator_kind="synthetic",
            measurements=(
                ("decision_score", 0.0),
                ("embedding_score", 0.0),
                ("sampling_score", 0.0),
            ),
        ),
        _lineage(
            "val",
            family="portfolio",
            generator_kind="application",
            measurements=(
                ("decision_score", 1.0),
                ("embedding_score", 1.0),
                ("sampling_score", 0.0),
            ),
        ),
        _lineage(
            "test",
            family="frustrated-loop",
            generator_kind="synthetic",
            measurements=(
                ("decision_score", 1.0),
                ("embedding_score", 0.0),
                ("sampling_score", 1.0),
            ),
        ),
    )
    tasks = (
        _task(
            "train",
            lineages[0],
            topology="pegasus",
            faulted=False,
            regime="iid",
        ),
        _task(
            "val",
            lineages[1],
            topology="zephyr",
            faulted=True,
            regime="iid",
        ),
        _task(
            "test",
            lineages[2],
            topology="pegasus",
            faulted=False,
            regime="ood",
        ),
    )
    return lineages, tasks


def _test_profile() -> CorpusDesignProfile:
    return CorpusDesignProfile.explicit_test(
        train_floor=1,
        validation_floor=1,
        test_floor=1,
        validation_tuning_minimum=1,
        reason="three-lineage unit wire fixture",
    )


def _build(root: Path, **overrides: object):
    lineages, tasks = _facts()
    kwargs: dict[str, object] = {
        "lineages": lineages,
        "tasks": tasks,
        "output_directory": root,
        "publisher_id": "embedbench-release-authority",
        "source_release_id": "embedbench-release-v2",
        "source_release_manifest_sha256": "4" * 64,
        "split_manifest_sha256": "5" * 64,
        "measurement_budget": MeasurementBudget(
            measurement_budget_id="difficulty-budget-v2",
            decision_evaluations=8,
            embedding_attempts=8,
            sampler_reads=32,
        ),
        "difficulty_rules": _rules(),
        "profile": _test_profile(),
    }
    kwargs.update(overrides)
    return build_corpus_design_v2(**kwargs)


def test_builds_canonical_design_strata_and_publication_index(tmp_path: Path) -> None:
    output = tmp_path / "design-publication"
    receipt = _build(output)

    expected = {
        "corpus_design_v2.json",
        "publication-index.json",
        *(f"strata/{name}.json" for name in (
            "authority",
            "budget",
            "evidence",
            "origin",
            "panel",
            "protocol",
        )),
    }
    observed = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    assert observed == expected
    assert receipt.test_only is True
    assert receipt.base_lineage_count == 3
    assert receipt.task_count == 3

    for relative in sorted(expected):
        path = output / relative
        document = json.loads(path.read_text())
        assert path.read_bytes() == canonical_json_bytes(document) + b"\n"
        payload = {key: value for key, value in document.items() if key != "record_digest"}
        assert document["record_digest"] == content_digest(payload)

    design = json.loads((output / "corpus_design_v2.json").read_text())
    assert design["schema"] == "isingfold.corpus-design"
    assert design["schema_version"] == 2
    assert design["corpus_design_version"] == "if-core-v2"
    assert design["minimum_partition_base_lineages"] == {
        "test": 1,
        "train": 1,
        "val": 1,
    }
    assert design["partition_quotas"] == {"test": 1, "train": 1, "val": 1}
    assert set(design["axis_values"]["host_family"]) == {"pegasus", "zephyr"}
    assert set(design["axis_values"]["problem_origin"]) == {
        "application-derived",
        "synthetic",
    }
    assert {
        target["target_id"] for target in design["power_targets"]
    } == {
        "test-valid-return-noninferiority",
        "validation-valid-return-noninferiority",
    }
    assert {
        target["target_id"] for target in design["precision_targets"]
    } == {
        "test-learned-minus-stock-if-q3-s0-precision",
        "validation-paired-utility-precision",
    }

    calibration = design["difficulty_calibration"]
    for name in ("authority", "budget", "evidence", "origin", "panel", "protocol"):
        descriptor = calibration[name]
        assert descriptor == {
            "path": f"strata/{name}.json",
            "sha256": _sha256(output / "strata" / f"{name}.json"),
        }
    authority = json.loads((output / "strata" / "authority.json").read_text())
    for name in ("budget", "evidence", "origin", "panel", "protocol"):
        artifact = json.loads((output / "strata" / f"{name}.json").read_text())
        assert authority["artifacts"][f"{name}_sha256"] == _sha256(
            output / "strata" / f"{name}.json"
        )
        assert authority["artifacts"][f"{name}_record_digest"] == artifact["record_digest"]

    publication = json.loads((output / "publication-index.json").read_text())
    assert publication["schema"] == "embedbench.isingfold-publication-index"
    assert publication["schema_version"] == 1
    assert publication["corpus_design"] == {
        "path": "corpus_design_v2.json",
        "sha256": _sha256(output / "corpus_design_v2.json"),
    }
    assert [task["group_id"] for task in publication["tasks"]] == sorted(
        task["group_id"] for task in publication["tasks"]
    )
    serialized = canonical_json_bytes({"design": design, "publication": publication})
    for forbidden in (b"reference_energy", b"p_solve", b"residual_mean", b"audit"):
        assert forbidden not in serialized


def test_default_profile_never_silently_lowers_production_floors(tmp_path: Path) -> None:
    lineages, tasks = _facts()
    with pytest.raises(CorpusDesignError, match="train.*1024"):
        build_corpus_design_v2(
            lineages=lineages,
            tasks=tasks,
            output_directory=tmp_path / "production",
            publisher_id="publisher",
            source_release_id="release",
            source_release_manifest_sha256="4" * 64,
            split_manifest_sha256="5" * 64,
            measurement_budget=MeasurementBudget("budget", 1, 1, 1),
            difficulty_rules=_rules(),
        )
    assert not (tmp_path / "production").exists()

    with pytest.raises(ValueError, match="production floors are fixed"):
        CorpusDesignProfile(
            kind="production",
            train_floor=1,
            validation_floor=1,
            test_floor=1,
            validation_tuning_minimum=1,
            reason=None,
        )


def test_test_profile_requires_an_explicit_reason() -> None:
    with pytest.raises(ValueError, match="explicit test-profile reason"):
        CorpusDesignProfile.explicit_test(
            train_floor=1,
            validation_floor=1,
            test_floor=1,
            validation_tuning_minimum=1,
            reason="",
        )


def test_difficulty_protocol_rejects_target_derived_metric_names() -> None:
    with pytest.raises(ValueError, match="outcome-blind"):
        DifficultyRule(
            field="sampling_difficulty",
            metric="p_solve_at_selected_strength",
            hard_if="less-than-or-equal",
            threshold=0.5,
        )


def test_rejects_ood_outside_sealed_test(tmp_path: Path) -> None:
    lineages, tasks = _facts()
    distribution = dict(tasks[1].distribution)
    distribution["regime"] = "ood"
    invalid_tasks = (tasks[0], replace(tasks[1], distribution=distribution), tasks[2])

    with pytest.raises(CorpusDesignError, match="OOD.*test"):
        _build(tmp_path / "invalid", tasks=invalid_tasks)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("single-host", "two host families"),
        ("single-origin", "both problem origins"),
        ("no-fault", "faulted coverage"),
        ("all-easy", "easy and hard"),
    ],
)
def test_rejects_nondiverse_or_all_easy_designs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    lineages, tasks = _facts()
    if mutation == "single-host":
        tasks = tuple(
            replace(
                task,
                active_topology_identity={
                    **task.active_topology_identity,
                    "topology": "pegasus",
                },
                nominal_topology_identity={
                    **task.nominal_topology_identity,
                    "topology": "pegasus",
                },
            )
            for task in tasks
        )
    elif mutation == "single-origin":
        lineages = tuple(replace(row, generator_kind="synthetic") for row in lineages)
    elif mutation == "no-fault":
        tasks = tuple(
            replace(
                task,
                fault_identity={
                    "fault_mask_sha256": task.active_topology_identity[
                        "host_artifact_sha256"
                    ],
                    "status": "none",
                },
                descendant_transform_identity={
                    "kinds": ["identity"],
                    "transform_sha256": task.descendant_transform_identity[
                        "transform_sha256"
                    ],
                },
            )
            for task in tasks
        )
    else:
        lineages = tuple(
            replace(
                row,
                measurements=(
                    ("decision_score", 0.0),
                    ("embedding_score", 0.0),
                    ("sampling_score", 0.0),
                ),
            )
            for row in lineages
        )
    with pytest.raises(CorpusDesignError, match=message):
        _build(tmp_path / mutation, lineages=lineages, tasks=tasks)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "owned-by-user"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _build(output)
    assert marker.read_text(encoding="utf-8") == "preserve"

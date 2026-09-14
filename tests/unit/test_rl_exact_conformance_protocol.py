from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import networkx as nx
import pytest

import isingfold.rl.data.exact_conformance as exact_module
from isingfold.embedding import LogicalProblem
from isingfold.rl.data.exact_conformance import (
    EXACT_CONFORMANCE_CORPUS_ID,
    EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES,
    ExactConformanceError,
    build_exact_conformance_registry,
    load_exact_conformance_registry,
    publish_exact_conformance_registry,
    select_exact_conformance_tasks,
)
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import PreparedDesignCondition, PreparedTask
from isingfold.rl.env import EmbeddingTask


SOURCE_SHA = "7" * 64


def _condition(index: int, *, lineage: str | None = None) -> PreparedDesignCondition:
    return PreparedDesignCondition(
        base_lineage_key=lineage or f"lineage-{index:02d}",
        learning_partition="val",
        application_family=("frustrated-loop", "portfolio", "graph-cut")[index % 3],
        problem_origin=("synthetic", "application-derived")[index % 2],
        host_family=("chimera", "pegasus", "zephyr")[index % 3],
        fault_status=("none", "faulted")[index % 2],
        distribution_regime=("iid", "ood")[index % 2],
        calibration_status=("not_applicable", "calibrated")[index % 2],
        calibration_sha256=None if index % 2 == 0 else "c" * 64,
        embedding_difficulty=("easy", "hard")[index % 2],
        sampling_difficulty=("hard", "easy")[index % 2],
        decision_difficulty=("easy", "hard", "medium")[index % 3],
        registry_row_digest=hashlib.sha256(f"condition-{index}".encode()).hexdigest(),
    )


def _task(
    index: int,
    *,
    task_id: str | None = None,
    lineage: str | None = None,
    logical_size: int = 2,
    partition: str = "val",
    initial_embedding: dict[int, frozenset[int]] | None | object = ...,  # type: ignore[assignment]
    design_condition: PreparedDesignCondition | None | object = ...,  # type: ignore[assignment]
) -> PreparedTask:
    logical = nx.path_graph(logical_size)
    host = nx.path_graph(max(2 * logical_size, 2))
    problem = LogicalProblem.from_dicts(
        {node: 0.0 for node in logical.nodes},
        {tuple(edge): -1.0 for edge in logical.edges},
    )
    if initial_embedding is ...:
        initial_embedding = {node: frozenset((2 * node, 2 * node + 1)) for node in logical.nodes}
    resolved_lineage = lineage or f"lineage-{index:02d}"
    if design_condition is ...:
        design_condition = _condition(index, lineage=resolved_lineage)
    name = task_id or f"task-{index:02d}"
    embedded = EmbeddingTask(
        name=name,
        logical=logical,
        host=host,
        problem=problem,
        ground_energy=None,
        lineage=resolved_lineage,
        initial_embedding=initial_embedding,  # type: ignore[arg-type]
    )
    return PreparedTask(
        task=embedded,
        task_id=name,
        instance_id=f"instance-{index:02d}",
        partition=partition,
        initializer_record_digest=hashlib.sha256(name.encode()).hexdigest(),
        public_instance_record_digest=hashlib.sha256(
            f"policy-instance-{index:02d}".encode()
        ).hexdigest(),
        reference_status=None,
        certificate_digest=None,
        evaluator_protocol_digest=None,
        prepared_schema_version=4,
        corpus_scope="production-designed-v4",
        design_condition=design_condition,  # type: ignore[arg-type]
    )


def _population() -> list[PreparedTask]:
    return [_task(index) for index in range(12)]


def _write_registry(path: Path, payload: dict[str, object]) -> str:
    record = {**payload, "record_digest": content_digest(payload)}
    raw = canonical_json_bytes(record) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_public_prepared_stubs(corpus: Path) -> None:
    for filename in (
        "initializers.jsonl",
        "policy_instances.jsonl",
        "provenance.jsonl",
        "splits.json",
    ):
        (corpus / filename).write_bytes(b"")


def test_selector_is_order_invariant_exact_and_lineage_independent() -> None:
    population = _population()
    duplicate_lineage = _task(
        99,
        task_id="task-00-variant",
        lineage="lineage-00",
        design_condition=replace(
            _condition(5),
            base_lineage_key="lineage-00",
            registry_row_digest="d" * 64,
        ),
    )
    population.append(duplicate_lineage)

    forward = select_exact_conformance_tasks(population, SOURCE_SHA)
    reverse = select_exact_conformance_tasks(list(reversed(population)), SOURCE_SHA)

    assert forward.task_ids == reverse.task_ids
    assert len(forward.task_ids) == 8
    assert forward.task_ids == tuple(sorted(forward.task_ids))
    assert len({item.design_condition.base_lineage_key for item in forward.tasks}) == 8
    assert forward.source_population_census == reverse.source_population_census
    assert forward.selected_marginal_census == reverse.selected_marginal_census


def test_selector_is_independent_of_evaluator_only_fields() -> None:
    population = _population()
    target_bearing = [
        replace(
            item,
            task=replace(item.task, ground_energy=-1000.0 - index),
            reference_status="certified_optimal",
            certificate_digest=f"{index + 1:x}" * 64,
            evaluator_protocol_digest=f"{index + 2:x}" * 64,
        )
        for index, item in enumerate(population)
    ]

    public = select_exact_conformance_tasks(population, SOURCE_SHA)
    evaluator = select_exact_conformance_tasks(target_bearing, SOURCE_SHA)

    assert public.task_ids == evaluator.task_ids
    assert public.greedy_task_ids == evaluator.greedy_task_ids


def test_selector_and_registry_have_a_frozen_golden_identity() -> None:
    selection = select_exact_conformance_tasks(_population(), SOURCE_SHA)
    registry = build_exact_conformance_registry(selection, SOURCE_SHA)
    raw = canonical_json_bytes(registry) + b"\n"

    assert selection.greedy_task_ids == (
        "task-08",
        "task-09",
        "task-07",
        "task-06",
        "task-05",
        "task-04",
        "task-02",
        "task-01",
    )
    assert selection.task_ids == (
        "task-01",
        "task-02",
        "task-04",
        "task-05",
        "task-06",
        "task-07",
        "task-08",
        "task-09",
    )
    assert hashlib.sha256(raw).hexdigest() == (
        "cd8c25311ebe2e7dd8d8b25ff93e8495566ede30f3318ac2e6f7c176d0664828"
    )


def test_selector_accepts_the_inclusive_twelve_variable_cap() -> None:
    population = [_task(0, logical_size=EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES)] + [
        _task(index) for index in range(1, 8)
    ]

    selection = select_exact_conformance_tasks(population, SOURCE_SHA)

    assert len(selection.tasks) == 8
    assert max(task.task.logical.number_of_nodes() for task in selection.tasks) == 12


def test_selector_accepts_a_nonempty_witness_when_no_initializer_is_present() -> None:
    population = []
    for item in _population()[:8]:
        witness = item.task.initial_embedding
        population.append(
            replace(
                item,
                task=replace(
                    item.task,
                    initial_embedding=None,
                    witness=witness,
                ),
            )
        )

    selection = select_exact_conformance_tasks(population, SOURCE_SHA)

    assert len(selection.task_ids) == 8


def test_selector_uses_sha_fallback_after_equal_coverage_loads() -> None:
    condition = _condition(0)
    population = [
        _task(
            index,
            task_id=f"equal-{index:02d}",
            lineage=f"equal-lineage-{index:02d}",
            design_condition=replace(
                condition,
                base_lineage_key=f"equal-lineage-{index:02d}",
                registry_row_digest=hashlib.sha256(f"row-{index}".encode()).hexdigest(),
            ),
        )
        for index in range(9)
    ]

    selection = select_exact_conformance_tasks(population, SOURCE_SHA)

    # The first greedy choice is entirely determined by the specified SHA-256 fallback.
    assert selection.greedy_task_ids[0] == min(
        (item.task_id for item in population),
        key=lambda task_id: hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": "exact-conformance-task-tie-v1",
                    "parts": [
                        0,
                        SOURCE_SHA,
                        selection.stratum_digests[task_id],
                        next(
                            item.design_condition.base_lineage_key
                            for item in population
                            if item.task_id == task_id
                        ),
                        task_id,
                    ],
                    "version": 1,
                }
            )
        ).hexdigest(),
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda tasks: tasks[:7], "at least 8"),
        (
            lambda tasks: [
                *tasks[:7],
                _task(99, lineage=tasks[0].task.lineage),
            ],
            "independent base lineages",
        ),
        (
            lambda tasks: [
                replace(tasks[0], design_condition=None),
                *tasks[1:8],
            ],
            "at least 8",
        ),
        (
            lambda tasks: [
                _task(
                    0,
                    logical_size=EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES + 1,
                ),
                *tasks[1:8],
            ],
            "at least 8",
        ),
        (
            lambda tasks: [
                _task(0, initial_embedding=None),
                *tasks[1:8],
            ],
            "at least 8",
        ),
    ],
    ids=(
        "fewer-than-eight",
        "fewer-than-eight-lineages",
        "missing-design-condition",
        "logical-size-above-cap",
        "missing-witness",
    ),
)
def test_selector_rejects_an_ineligible_population(mutate, match: str) -> None:
    with pytest.raises(ExactConformanceError, match=match):
        select_exact_conformance_tasks(mutate(_population()), SOURCE_SHA)


def test_selector_rejects_duplicate_task_ids_and_invalid_axis_levels() -> None:
    population = _population()
    duplicate = replace(population[1], task_id=population[0].task_id)
    with pytest.raises(ExactConformanceError, match="duplicate task IDs"):
        select_exact_conformance_tasks([population[0], duplicate, *population[2:]], SOURCE_SHA)

    invalid_condition = replace(population[0].design_condition, fault_status=3)  # type: ignore[arg-type]
    with pytest.raises(ExactConformanceError, match="invalid fault_status"):
        select_exact_conformance_tasks(
            [replace(population[0], design_condition=invalid_condition), *population[1:]],
            SOURCE_SHA,
        )


def test_selector_fails_closed_on_a_derived_tie_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exact_module, "_seed_digest", lambda *args: "0" * 64)

    with pytest.raises(ExactConformanceError, match="collision"):
        select_exact_conformance_tasks(_population(), SOURCE_SHA)


def test_registry_has_fixed_identity_and_loader_recomputes_selection(tmp_path: Path) -> None:
    population = _population()
    selection = select_exact_conformance_tasks(population, SOURCE_SHA)
    registry = build_exact_conformance_registry(selection, SOURCE_SHA)
    path = tmp_path / "exact.json"
    raw_sha = _write_registry(
        path, {key: value for key, value in registry.items() if key != "record_digest"}
    )

    selected, identity = load_exact_conformance_registry(
        path,
        expected_registry_sha256=raw_sha,
        prepared=population,
        source_corpus_manifest_sha256=SOURCE_SHA,
    )

    assert registry["corpus_id"] == EXACT_CONFORMANCE_CORPUS_ID
    assert tuple(item.task_id for item in selected) == selection.task_ids
    assert identity["exact_reproduction"] is True
    assert identity["base_lineages"] == sorted(
        item.design_condition.base_lineage_key for item in selection.tasks
    )
    assert len(set(identity["base_lineages"])) == 8

    alternate_ids = sorted(
        [
            *selection.task_ids[:-1],
            next(item.task_id for item in population if item.task_id not in selection.task_ids),
        ]
    )
    forged_payload = {
        **{key: value for key, value in registry.items() if key != "record_digest"},
        "task_ids": alternate_ids,
    }
    forged_sha = _write_registry(path, forged_payload)
    with pytest.raises(ExactConformanceError, match="deterministic selector"):
        load_exact_conformance_registry(
            path,
            expected_registry_sha256=forged_sha,
            prepared=population,
            source_corpus_manifest_sha256=SOURCE_SHA,
        )


def test_registry_reader_rejects_noncanonical_duplicate_unknown_and_symlink(
    tmp_path: Path,
) -> None:
    population = _population()
    selection = select_exact_conformance_tasks(population, SOURCE_SHA)
    registry = build_exact_conformance_registry(selection, SOURCE_SHA)
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_bytes(canonical_json_bytes(registry) + b"\n")

    pretty_path = tmp_path / "pretty.json"
    pretty_path.write_text(json.dumps(registry, indent=2) + "\n")
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"schema":"a","schema":"b"}\n')
    unknown_payload = {**registry, "unknown": True}
    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_bytes(canonical_json_bytes(unknown_payload) + b"\n")
    symlink_path = tmp_path / "link.json"
    symlink_path.symlink_to(canonical_path)

    for path, match in (
        (pretty_path, "canonical"),
        (duplicate_path, "duplicate"),
        (unknown_path, "schema fields"),
        (symlink_path, "symlink"),
    ):
        with pytest.raises(ExactConformanceError, match=match):
            load_exact_conformance_registry(
                path,
                expected_registry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                prepared=population,
                source_corpus_manifest_sha256=SOURCE_SHA,
            )


def test_registry_reader_rejects_boolean_integer_and_nonfinite_json(tmp_path: Path) -> None:
    population = _population()
    registry = build_exact_conformance_registry(
        select_exact_conformance_tasks(population, SOURCE_SHA), SOURCE_SHA
    )
    boolean_payload = {
        **{key: value for key, value in registry.items() if key != "record_digest"},
        "schema_version": True,
    }
    boolean_path = tmp_path / "boolean.json"
    boolean_sha = _write_registry(boolean_path, boolean_payload)
    with pytest.raises(ExactConformanceError, match="incompatible identity"):
        load_exact_conformance_registry(
            boolean_path,
            expected_registry_sha256=boolean_sha,
            prepared=population,
            source_corpus_manifest_sha256=SOURCE_SHA,
        )

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_bytes(b'{"value":NaN}\n')
    with pytest.raises(ExactConformanceError, match="non-finite"):
        load_exact_conformance_registry(
            nonfinite_path,
            expected_registry_sha256=hashlib.sha256(nonfinite_path.read_bytes()).hexdigest(),
            prepared=population,
            source_corpus_manifest_sha256=SOURCE_SHA,
        )


def test_publisher_is_immutable_and_emits_golden_canonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population = _population()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest_payload = {
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": 4,
    }
    manifest = {**manifest_payload, "record_digest": content_digest(manifest_payload)}
    (corpus / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    _write_public_prepared_stubs(corpus)
    manifest_sha = hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest()

    observed: list[bool] = []

    def fake_loader(directory: Path, partition: str, *, include_evaluator: bool):
        observed.append(include_evaluator)
        assert Path(directory) == corpus
        assert partition == "val"
        return type("Loaded", (), {"tasks": tuple(population), "target_access": None})()

    monkeypatch.setattr("isingfold.rl.data.exact_conformance.load_prepared_partition", fake_loader)
    out = tmp_path / "registry.json"
    receipt = publish_exact_conformance_registry(
        corpus,
        expected_corpus_manifest_sha256=manifest_sha,
        out=out,
    )

    expected = build_exact_conformance_registry(
        select_exact_conformance_tasks(population, manifest_sha), manifest_sha
    )
    assert out.read_bytes() == canonical_json_bytes(expected) + b"\n"
    assert receipt["file_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    second = tmp_path / "registry-second.json"
    publish_exact_conformance_registry(
        corpus,
        expected_corpus_manifest_sha256=manifest_sha,
        out=second,
    )
    assert second.read_bytes() == out.read_bytes()
    assert observed == [False, False]
    with pytest.raises(FileExistsError):
        publish_exact_conformance_registry(
            corpus,
            expected_corpus_manifest_sha256=manifest_sha,
            out=out,
        )
    with pytest.raises(ExactConformanceError, match="externally pinned"):
        publish_exact_conformance_registry(
            corpus,
            expected_corpus_manifest_sha256="0" * 64,
            out=tmp_path / "wrong-source.json",
        )
    symlink_out = tmp_path / "symlink-output.json"
    symlink_out.symlink_to(out)
    with pytest.raises(ExactConformanceError, match="symlink"):
        publish_exact_conformance_registry(
            corpus,
            expected_corpus_manifest_sha256=manifest_sha,
            out=symlink_out,
        )
    initializer_path = corpus / "initializers.jsonl"
    initializer_path.unlink()
    initializer_path.symlink_to(corpus / "policy_instances.jsonl")
    with pytest.raises(ExactConformanceError, match="symlink"):
        publish_exact_conformance_registry(
            corpus,
            expected_corpus_manifest_sha256=manifest_sha,
            out=tmp_path / "public-symlink.json",
        )


def test_publisher_rejects_selector_source_change_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population = _population()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest_payload = {
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": 4,
    }
    manifest = {**manifest_payload, "record_digest": content_digest(manifest_payload)}
    manifest_path = corpus / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    _write_public_prepared_stubs(corpus)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        "isingfold.rl.data.exact_conformance.load_prepared_partition",
        lambda *args, **kwargs: type(
            "Loaded", (), {"tasks": tuple(population), "target_access": None}
        )(),
    )
    identities = iter(
        (
            {"source_sha256": "a" * 64},
            {"source_sha256": "b" * 64},
        )
    )
    monkeypatch.setattr(
        exact_module,
        "selector_implementation_identity",
        lambda: next(identities),
    )
    out = tmp_path / "must-not-publish.json"

    with pytest.raises(ExactConformanceError, match="implementation changed"):
        publish_exact_conformance_registry(
            corpus,
            expected_corpus_manifest_sha256=manifest_sha,
            out=out,
        )

    assert not out.exists()


def test_competing_publishers_produce_one_complete_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    population = _population()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest_payload = {
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": 4,
    }
    manifest = {**manifest_payload, "record_digest": content_digest(manifest_payload)}
    manifest_path = corpus / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    _write_public_prepared_stubs(corpus)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def fake_loader(directory: Path, partition: str, *, include_evaluator: bool):
        assert include_evaluator is False
        return type("Loaded", (), {"tasks": tuple(population), "target_access": None})()

    monkeypatch.setattr("isingfold.rl.data.exact_conformance.load_prepared_partition", fake_loader)
    out = tmp_path / "race.json"

    def publish() -> object:
        try:
            return publish_exact_conformance_registry(
                corpus,
                expected_corpus_manifest_sha256=manifest_sha,
                out=out,
            )
        except FileExistsError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))

    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, FileExistsError) for result in results) == 1
    parsed = json.loads(out.read_text())
    assert parsed["record_digest"] == content_digest(
        {key: value for key, value in parsed.items() if key != "record_digest"}
    )
    assert list(tmp_path.glob(".race.json.*.tmp")) == []

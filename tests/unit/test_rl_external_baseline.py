"""Complete-system external baseline contracts.

Stock minorminer is intentionally exercised through an injected fake here.  The optional
package is not a CI dependency and its native search is not a same-support controller.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context, WorkVector, stable_digest
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.evaluate import (
    INITIALIZATION_FAILURE,
    EpisodeOutcome,
    read_episode_receipts,
    write_episode_receipts,
)
from isingfold.rl.external import (
    AUDIT_READS,
    BackendAvailability,
    BackendIdentity,
    BackendSearchResult,
    ExternalBaselineConfig,
    ExternalBackendError,
    SearchStatus,
    StockMinorminerBackend,
    stock_minorminer_artifact_manifest,
    external_method_metadata,
    run_external_system,
    unconditional_system_outcomes,
    write_external_attempt_receipts,
)


def _tiny_task() -> EmbeddingTask:
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    logical = nx.path_graph(3)
    rng = random.Random(7)
    h = {node: rng.choice((-1.0, 1.0)) for node in logical}
    j = {edge: rng.choice((-1.0, 1.0)) for edge in logical.edges()}
    problem = LogicalProblem.from_dicts(h, j)
    nodes = tuple(logical)
    ground = min(
        sum(h[node] * spins[node] for node in nodes)
        + sum(weight * spins[left] * spins[right] for (left, right), weight in j.items())
        for values in itertools.product((-1, 1), repeat=len(nodes))
        for spins in ({node: value for node, value in zip(nodes, values, strict=True)},)
    )
    return EmbeddingTask("external-tiny", logical, host, problem, ground, lineage="base-0")


class FakeBackend:
    def __init__(self, results: list[BackendSearchResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[int, float]] = []
        self.identity = BackendIdentity(
            method_id="fake-external-v1",
            distribution="fake-embedder",
            version="1.2.3",
            entrypoint="tests.FakeBackend",
            implementation="injected-test-double",
        )

    def search(self, logical, host, *, seed: int, timeout_seconds: float, parameters):
        del logical, host, parameters
        self.calls.append((seed, timeout_seconds))
        if not self.results:
            raise AssertionError("runner exceeded the registered restart cap")
        result = self.results.pop(0)
        if result.status is SearchStatus.ERROR:
            raise ExternalBackendError(result.detail or "fake crash")
        return result


def _config(*, restarts: int = 2) -> ExternalBaselineConfig:
    return ExternalBaselineConfig.from_mapping(
        {
            "schema": "isingfold.external-baseline-config",
            "schema_version": 1,
            "backend": "stock-minorminer",
            "expected_backend_version": "0.2.22",
            "embedding_wallclock_seconds": 30.0,
            "max_restarts": restarts,
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
    )


def test_external_config_fixes_fresh_4096_read_audit_and_one_try_restarts() -> None:
    config = _config()

    assert config.audit_reads == AUDIT_READS == 4096
    assert config.minorminer_parameters["tries"] == 1
    assert config.online_evaluator_feedback is False

    payload = config.as_dict()
    payload["audit_reads"] = 256
    with pytest.raises(ValueError, match="4096"):
        ExternalBaselineConfig.from_mapping(payload)


def test_checked_in_external_config_is_registered_and_parseable() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs" / "external_minorminer_v1.json").read_text())

    config = ExternalBaselineConfig.from_mapping(payload)

    assert config.backend == "stock-minorminer"
    assert config.expected_backend_version == "0.2.22"
    assert config.embedding_wallclock_seconds == 60.0
    payload = config.as_dict()
    payload["minorminer_parameters"]["tries"] = 2
    with pytest.raises(ValueError, match="tries=1"):
        ExternalBaselineConfig.from_mapping(payload)


def test_external_runner_selects_without_outcomes_then_samples_once(monkeypatch) -> None:
    task = _tiny_task()
    # Both are valid.  The second uses fewer qubits and must win without seeing evaluator data.
    larger = {0: frozenset({0, 3}), 1: frozenset({1, 4}), 2: frozenset({2, 5})}
    smaller = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend = FakeBackend(
        [
            BackendSearchResult(SearchStatus.EMBEDDING, larger, 0.2),
            BackendSearchResult(SearchStatus.EMBEDDING, smaller, 0.3),
        ]
    )
    sampled: list[tuple[dict[int, frozenset[int]], int, int]] = []

    def fake_sample(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, problem, ground_energy, num_sweeps
        sampled.append((dict(chains), num_reads, seed))
        return ReadBlock(1024, num_reads, 0.125, 0.25, 1)

    monkeypatch.setattr("isingfold.rl.external.sample_program", fake_sample)
    outcomes, receipts = run_external_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        selector=fixed_strength_selector(),
        seed=91,
        repetitions=1,
    )

    assert len(outcomes) == len(receipts) == 1
    assert sampled[0][0] == smaller
    assert sampled[0][1] == 4096
    assert outcomes[0].evaluator_reads == 4096
    assert outcomes[0].utility == pytest.approx(0.25)
    assert outcomes[0].population_eligible is True
    assert outcomes[0].work is not None
    assert outcomes[0].work.restart_work == 2
    # Two restart outputs are screened, then the selected one is rebuilt and revalidated.
    assert outcomes[0].work.validator_calls == 3
    assert outcomes[0].work.compiler_calls == 4
    assert receipts[0].selected_restart == 1
    assert receipts[0].selection_rule == "resource-lexicographic"
    assert receipts[0].online_evaluator_feedback is False
    assert receipts[0].restart_seeds == tuple(seed for seed, _ in backend.calls)
    assert len(set(receipts[0].restart_seeds)) == 2
    assert outcomes[0].validation_digest == receipts[0].validation_digest
    assert outcomes[0].controller_calls is None
    assert outcomes[0].controller_seconds is None


def test_search_failure_is_an_unconditional_zero_not_an_initializer_subset(monkeypatch) -> None:
    task = _tiny_task()
    backend = FakeBackend(
        [
            BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.1),
            BackendSearchResult(SearchStatus.TIMED_OUT, None, 0.2),
        ]
    )
    monkeypatch.setattr(
        "isingfold.rl.external.sample_program",
        lambda *args, **kwargs: pytest.fail("a failed system must not open the evaluator"),
    )

    outcomes, receipts = run_external_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        selector=fixed_strength_selector(),
        seed=4,
    )

    outcome = outcomes[0]
    assert outcome.returned_valid is False
    assert outcome.utility == 0.0
    assert outcome.population_eligible is True
    assert outcome.outcome_kind == "TASK_TERMINAL"
    assert outcome.reason == "EXTERNAL_NO_VALID_EMBEDDING"
    assert outcome.work is not None and outcome.work.restart_work == 2
    outcome.validate_receipt(require_complete=True)
    assert receipts[0].selected_restart is None


def test_external_raw_receipts_are_canonical_and_bind_system_outcomes(
    monkeypatch, tmp_path
) -> None:
    task = _tiny_task()
    backend = FakeBackend(
        [BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.1)]
    )
    monkeypatch.setattr(
        "isingfold.rl.external.sample_program",
        lambda *args, **kwargs: pytest.fail("a failed system must not open the evaluator"),
    )
    outcomes, details = run_external_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(restarts=1),
        selector=fixed_strength_selector(),
    )

    outcome_digest = write_episode_receipts(tmp_path / "outcomes.jsonl", outcomes)
    detail_digest = write_external_attempt_receipts(tmp_path / "attempts.jsonl", details)

    assert read_episode_receipts(
        tmp_path / "outcomes.jsonl", expected_sha256=outcome_digest
    ) == outcomes
    assert len(outcome_digest) == len(detail_digest) == 64
    assert details[0].as_dict()["record_digest"] == stable_digest(
        details[0].as_dict(include_digest=False)
    )


def test_initializer_failure_conversion_is_explicitly_unconditional() -> None:
    original = EpisodeOutcome(
        instance="i",
        lineage="L",
        returned_valid=False,
        utility=None,
        qubits=None,
        max_chain=None,
        decisions=0,
        selected_strength=None,
        reason="initializer returned no embedding",
        episode_seed=7,
        work=WorkVector(restart_work=1),
        validation_digest="a" * 64,
        online_seconds=0.1,
        outcome_kind=INITIALIZATION_FAILURE,
        population_eligible=False,
    )

    converted = unconditional_system_outcomes([original])[0]

    assert converted.utility == 0.0
    assert converted.population_eligible is True
    assert converted.outcome_kind == "TASK_TERMINAL"
    assert converted.reason.startswith("INITIALIZATION_FAILURE_ZERO:")
    assert original.utility is None and original.population_eligible is False


def test_backend_crash_is_integrity_error_not_a_zero_label() -> None:
    task = _tiny_task()
    backend = FakeBackend([BackendSearchResult(SearchStatus.ERROR, None, 0.0, "native crash")])

    with pytest.raises(ExternalBackendError, match="native crash"):
        run_external_system(
            [task],
            Context(qubit_cap=9),
            backend,
            _config(restarts=1),
            selector=fixed_strength_selector(),
        )


def test_optional_stock_backend_reports_unavailability_without_importing(monkeypatch) -> None:
    def missing(_name: str):
        raise StockMinorminerBackend.package_not_found_error("minorminer")

    monkeypatch.setattr("isingfold.rl.external.importlib_metadata.distribution", missing)

    availability = StockMinorminerBackend.probe(expected_version="0.2.22")

    assert availability.available is False
    assert availability.identity is None
    assert "pip install" in availability.reason


def test_stock_backend_identity_hashes_installed_native_and_source_artifacts() -> None:
    try:
        manifest = stock_minorminer_artifact_manifest(expected_version="0.2.22")
    except StockMinorminerBackend.package_not_found_error:
        pytest.skip("optional stock minorminer is not installed")

    assert manifest["schema"] == "isingfold.stock-minorminer-artifact-manifest"
    assert manifest["record_digest"] == stable_digest(
        {name: value for name, value in manifest.items() if name != "record_digest"}
    )
    assert any(
        Path(row["path"]).suffix in {".so", ".pyd", ".dylib", ".dll"}
        for row in manifest["files"]
    )
    availability = StockMinorminerBackend.probe(expected_version="0.2.22")
    assert availability.available is True
    assert availability.identity is not None
    assert availability.identity.implementation.endswith(
        f"artifact-record-sha256:{manifest['record_digest']}"
    )


def test_external_method_is_explicitly_not_same_support_or_an_oracle() -> None:
    metadata = external_method_metadata(_config(), BackendIdentity(
        method_id="stock-minorminer-restarts-v1",
        distribution="minorminer",
        version="0.2.22",
        entrypoint="minorminer.find_embedding",
        implementation="native-package",
    ))

    assert metadata["comparison_scope"] == "external-complete-system"
    assert metadata["same_support_causal_ablation"] is False
    assert metadata["online_evaluator_feedback"] is False
    assert metadata["evaluator_oracle"] is False
    assert metadata["controller_timing_scope"].startswith("not-applicable")
    assert metadata["unimplemented_plugin_slots"] == ["OCT", "ATOM"]


def test_cli_registers_a_separate_external_system_command() -> None:
    from isingfold.rl.cli import build_parser

    parsed = build_parser().parse_args(
        [
            "evaluate-external",
            "--config",
            "configs/external_minorminer_v1.json",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
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
            "--partition",
            "test",
            "--repetitions",
            "3",
            "--seed",
            "1201",
            "--out",
            "external-eval",
        ]
    )

    assert parsed.command == "evaluate-external"
    assert parsed.partition == "test"
    assert parsed.repetitions == 3
    assert not hasattr(parsed, "checkpoint")


def test_cli_unavailability_is_explicit_and_writes_no_fake_outcomes(
    monkeypatch, tmp_path
) -> None:
    from isingfold.rl.cli import build_parser

    root = Path(__file__).resolve().parents[2]
    destination = tmp_path / "unavailable"
    monkeypatch.setattr(
        StockMinorminerBackend,
        "probe",
        classmethod(
            lambda cls, *, expected_version: BackendAvailability(
                False,
                f"minorminer {expected_version} is not installed; pip install baseline extra",
            )
        ),
    )
    parsed = build_parser().parse_args(
        [
            "evaluate-external",
            "--config",
            str(root / "configs" / "external_minorminer_v1.json"),
            "--corpus",
            str(tmp_path / "does-not-need-to-exist"),
            "--selector",
            str(tmp_path / "does-not-need-to-exist-either"),
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
            str(destination),
        ]
    )

    parsed.func(parsed)

    report = json.loads((destination / "report.json").read_text())
    assert report["status"] == "unavailable"
    assert report["no_fallback"] is True
    assert report["artifacts"] == {}
    assert {item.name for item in destination.iterdir()} == {"report.json"}
    recorded = report.pop("record_digest")
    assert recorded == stable_digest(report)

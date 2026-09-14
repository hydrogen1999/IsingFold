"""Publication-facing external complete-system contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tests.unit.evaluation_strata_support import evaluation_contract
from isingfold.embedding import LogicalProblem
from isingfold.rl.complete_system import (
    CompletePopulationIdentity,
    CompleteSystemConfig,
    FrozenComponentIdentity,
    task_population_digest,
)
from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.external import (
    BackendIdentity,
    BackendSearchResult,
    SearchStatus,
    STOCK_MINORMINER_METHOD,
)
from isingfold.rl.external_pairing import (
    AuthenticatedExternalCompleteRun,
    aggregate_learned_vs_stock,
    ExternalCompleteSystemConfig,
    context_snapshot,
    external_complete_summary,
    load_authenticated_external_complete_run,
    read_external_complete_evidence,
    read_external_complete_outcomes,
    read_external_complete_receipts,
    runtime_identity,
    run_external_complete_system,
    write_external_complete_evidence,
    write_external_complete_outcomes,
    write_external_complete_receipts,
)
from isingfold.rl.external_pairing import (
    _confirmatory_target_result,
    _fixed_sequence_gatekeeping,
    _PairedMatrices,
    _primary_precision_target_result,
)
from isingfold.rl.external_tuning import (
    EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
    EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST,
    ExternalTuningCandidate,
    ExternalTuningExecutionBinding,
)


def _task() -> EmbeddingTask:
    logical = nx.path_graph(2)
    host = nx.path_graph(4)
    h = {0: -1.0, 1: 0.5}
    j = {(0, 1): -1.0}
    problem = LogicalProblem.from_dicts(h, j)
    ground = min(
        sum(h[node] * spins[node] for node in logical)
        + sum(weight * spins[left] * spins[right] for (left, right), weight in j.items())
        for values in itertools.product((-1, 1), repeat=2)
        for spins in ({node: value for node, value in zip(logical, values, strict=True)},)
    )
    return EmbeddingTask("instance-public", logical, host, problem, ground, lineage="base-0")


def test_fixed_sequence_never_declares_superiority_when_noninferiority_fails() -> None:
    result = _fixed_sequence_gatekeeping(
        confirmatory={"all_targets_pass": False},
        primary_precision={
            "all_targets_pass": True,
            "ungated_learned_superiority_criterion_pass": True,
        },
        familywise_alpha=0.05,
    )

    assert result["method"] == "fixed-sequence-gatekeeping-v1"
    assert result["strong_familywise_error_control"] is True
    assert result["claims"][0]["tested"] is True
    assert result["claims"][0]["passed"] is False
    assert result["claims"][1]["tested"] is False
    assert result["claims"][1]["passed"] is False
    assert result["primary_learned_superiority"] is False


def _learned_config(*, seconds: float = 30.0) -> CompleteSystemConfig:
    return CompleteSystemConfig(
        online_wallclock_seconds=seconds,
        max_initializer_attempts=1,
        audit_reads=4096,
        selection_rule="resource-lexicographic",
        online_evaluator_feedback=False,
        initializer_backend="lac-minorminer",
        initializer_method_id="lac-minorminer-hybrid-chimera-clique-v1-initializer-v4",
        expected_initializer_version="0.1.0",
        policy_restart_mode="disabled-no-native-replay-v1",
        initializer_parameters={"tries": 1, "max_transitions": 100, "max_candidates": 8},
    )


def _stock_config(*, seconds: float = 30.0) -> ExternalCompleteSystemConfig:
    return ExternalCompleteSystemConfig(
        backend="stock-minorminer",
        expected_backend_version="0.2.22",
        online_wallclock_seconds=seconds,
        max_restarts=1,
        audit_reads=4096,
        selection_rule="resource-lexicographic",
        online_evaluator_feedback=False,
        minorminer_parameters={
            "tries": 1,
            "threads": 1,
            "max_no_improvement": 10,
            "chainlength_patience": 10,
        },
    )


def _tuned_stock_default() -> ExternalTuningExecutionBinding:
    candidate = ExternalTuningCandidate.from_mapping(
        {
            "candidate_id": "stock-default-v1",
            "family": "stock-default",
            "outer_restart_policy": "single-native-call",
            "outer_restart_cap": 1,
            "native_parameters": {
                "tries": 10,
                "threads": 1,
                "max_no_improvement": 10,
                "chainlength_patience": 10,
            },
            "candidate_ranking": (
                "qubits-max_chain-embedding_digest-restart_index"
            ),
            "strength_rule": "frozen-selector-argmax-four-programs",
            "online_evaluator_feedback": False,
        }
    )
    return ExternalTuningExecutionBinding(
        mode="frozen-deployment",
        registry_id="stock-minorminer-0.2.22-validation-tuning-hybrid-v1",
        registry_file_sha256=EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
        registry_record_digest=EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST,
        candidate_index=0,
        candidate=candidate,
        selection_file_sha256="e" * 64,
        selection_record_digest="f" * 64,
    )


def _tuned_quality_p5() -> ExternalTuningExecutionBinding:
    candidate = ExternalTuningCandidate.from_mapping(
        {
            "candidate_id": "time-quality-p5-v1",
            "family": "time-saturating-quality",
            "outer_restart_policy": "until-wallclock-or-outer-cap",
            "outer_restart_cap": 64,
            "native_parameters": {
                "tries": 1,
                "threads": 1,
                "max_no_improvement": 5,
                "chainlength_patience": 5,
            },
            "candidate_ranking": (
                "frozen-selector-max-p_solve-then-qubits-max_chain-"
                "embedding_digest-restart_index"
            ),
            "strength_rule": "same-frozen-selector-argmax-four-programs",
            "online_evaluator_feedback": False,
        }
    )
    return ExternalTuningExecutionBinding(
        mode="frozen-deployment",
        registry_id="stock-minorminer-0.2.22-validation-tuning-hybrid-v1",
        registry_file_sha256=EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
        registry_record_digest=EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST,
        candidate_index=4,
        candidate=candidate,
        selection_file_sha256="e" * 64,
        selection_record_digest="f" * 64,
    )


def _selector() -> FrozenComponentIdentity:
    return FrozenComponentIdentity("selector", "v1", "tests.selector", "a" * 64)


def _population(task: EmbeddingTask, *, repetitions: int = 1, seed: int = 17):
    identities = ((task.lineage or task.name, task.name),)
    strata, design = evaluation_contract(identities)
    return CompletePopulationIdentity(
        population_id="test-population",
        source_manifest_sha256="b" * 64,
        task_payload_sha256=task_population_digest([task]),
        expected_instances=identities,
        expected_repetitions=repetitions,
        evaluation_seed=seed,
        evaluation_strata=strata,
        confirmatory_design=design,
    )


class _Backend:
    def __init__(self, result: BackendSearchResult, *, delay: float = 0.0) -> None:
        self.result = result
        self.delay = delay
        self.identity = BackendIdentity(
            method_id=STOCK_MINORMINER_METHOD,
            distribution="minorminer",
            version="0.2.22",
            entrypoint="minorminer.find_embedding",
            implementation=(
                "native-package-persistent-subprocess-hard-deadline:"
                + "artifact-record-sha256:"
                + "8" * 64
            ),
        )

    def search(self, logical, host, *, seed, timeout_seconds, parameters):
        del logical, host, seed, timeout_seconds, parameters
        if self.delay:
            time.sleep(self.delay)
        return self.result


def test_exact_external_runner_binds_population_authorities_and_fresh_seed(monkeypatch) -> None:
    task = _task()
    population = _population(task)
    context = Context(qubit_cap=4)
    backend = _Backend(
        BackendSearchResult(
            SearchStatus.EMBEDDING,
            {0: frozenset({0}), 1: frozenset({1})},
            0.01,
        )
    )
    sampled = []

    def sample(program, chains, problem, ground, *, num_reads, seed, num_sweeps):
        del program, chains, problem, ground, num_sweeps
        sampled.append((num_reads, seed))
        return ReadBlock(2048, 4096, 0.0, 0.0, 1)

    monkeypatch.setattr("isingfold.rl.external_pairing.sample_program", sample)
    outcomes, receipts = run_external_complete_system(
        [task],
        context,
        backend,
        _stock_config(),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=population,
        quality_authority={"publisher_id": "publisher", "target_set_digest": "c" * 64},
        seed=17,
        repetitions=1,
    )

    assert len(outcomes) == len(receipts) == 1
    assert outcomes[0].utility == 0.5
    assert outcomes[0].evaluator_seed == receipts[0].evaluator_seed
    assert sampled == [(4096, receipts[0].evaluator_seed)]
    assert receipts[0].population == population
    assert receipts[0].selector == _selector()
    assert receipts[0].quality_authority_digest == content_digest(
        {"publisher_id": "publisher", "target_set_digest": "c" * 64}
    )
    assert receipts[0].observed_work.route_expansions is None
    assert receipts[0].cap_compliance["route_expansions"] is None
    assert receipts[0].wallclock_cap_seconds == 30.0


def test_tuned_quality_runner_saturates_restart_cap_and_prefers_selector_quality(
    monkeypatch,
) -> None:
    task = _task()
    context = Context(qubit_cap=4)
    population = _population(task)
    config = dataclasses.replace(_stock_config(), max_restarts=64)
    binding = _tuned_quality_p5()

    class SequenceBackend:
        def __init__(self) -> None:
            self.identity = BackendIdentity(
                method_id=STOCK_MINORMINER_METHOD,
                distribution="minorminer",
                version="0.2.22",
                entrypoint="minorminer.find_embedding",
                implementation=(
                    "native-package-persistent-subprocess-hard-deadline:"
                    + "artifact-record-sha256:"
                    + "8" * 64
                ),
            )
            self.calls: list[dict[str, int]] = []

        def search(self, logical, host, *, seed, timeout_seconds, parameters):
            del logical, host, seed, timeout_seconds
            self.calls.append(dict(parameters))
            if len(self.calls) == 1:
                embedding = {0: frozenset({0}), 1: frozenset({1})}
                return BackendSearchResult(SearchStatus.EMBEDDING, embedding, 0.0)
            if len(self.calls) == 2:
                embedding = {0: frozenset({0, 1}), 1: frozenset({2, 3})}
                return BackendSearchResult(SearchStatus.EMBEDDING, embedding, 0.0)
            return BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.0)

    class FrozenQualitySelector:
        deployment_ready = True
        normalizer_digest = "selector-normalizer-unit-v1"
        coefficient_transform_scale = 1.0

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, inputs):
            assert len(inputs) == 4
            self.calls += 1
            return np.asarray(
                [0.20, 0.15, 0.10, 0.05]
                if self.calls == 1
                else [0.30, 0.40, 0.50, 0.90]
            )

        def select_embedding(self, inputs):
            del inputs
            return 0

    backend = SequenceBackend()
    selector = FrozenQualitySelector()
    evaluator_calls = 0

    def sample(program, chains, problem, ground, *, num_reads, seed, num_sweeps):
        nonlocal evaluator_calls
        del program, chains, problem, ground, seed, num_sweeps
        evaluator_calls += 1
        return ReadBlock(num_reads, num_reads, 0.0, 0.0, 3)

    monkeypatch.setattr("isingfold.rl.external_pairing.sample_program", sample)
    outcomes, receipts = run_external_complete_system(
        [task],
        context,
        backend,
        config,
        learned_config=_learned_config(),
        selector=selector,
        selector_identity=_selector(),
        population=population,
        quality_authority={"publisher": "fixture"},
        seed=population.evaluation_seed,
        repetitions=1,
        tuning_execution=binding,
    )

    assert outcomes[0].returned_valid is True
    assert evaluator_calls == 1
    assert len(backend.calls) == len(receipts[0].restarts) == 64
    assert all(call == dict(binding.candidate.native_parameters) for call in backend.calls)
    assert receipts[0].selected_restart == 1
    assert receipts[0].restarts[0].selector_max_p_solve == 0.20
    assert receipts[0].restarts[1].selector_max_p_solve == 0.90
    assert receipts[0].observed_work.restart_work == 64
    assert receipts[0].observed_work.compiler_calls == 8
    assert receipts[0].observed_work.validator_calls == 4
    assert receipts[0].tuning_execution == binding


def test_tuned_stock_default_is_one_native_call_with_ten_internal_tries(
    monkeypatch,
) -> None:
    task = _task()
    population = _population(task)

    class CaptureBackend(_Backend):
        def __init__(self) -> None:
            super().__init__(
                BackendSearchResult(
                    SearchStatus.EMBEDDING,
                    {0: frozenset({0}), 1: frozenset({1})},
                    0.0,
                )
            )
            self.parameters: list[dict[str, int]] = []

        def search(self, logical, host, *, seed, timeout_seconds, parameters):
            self.parameters.append(dict(parameters))
            return super().search(
                logical,
                host,
                seed=seed,
                timeout_seconds=timeout_seconds,
                parameters=parameters,
            )

    backend = CaptureBackend()
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: ReadBlock(4096, 4096, 0.0, 0.0, 1),
    )
    _, receipts = run_external_complete_system(
        [task],
        Context(qubit_cap=4),
        backend,
        dataclasses.replace(_stock_config(), max_restarts=64),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=population,
        quality_authority={"publisher": "fixture"},
        seed=population.evaluation_seed,
        repetitions=1,
        tuning_execution=_tuned_stock_default(),
    )

    assert len(receipts[0].restarts) == 1
    assert backend.parameters == [
        {
            "tries": 10,
            "threads": 1,
            "max_no_improvement": 10,
            "chainlength_patience": 10,
        }
    ]


def test_timeout_discards_embedding_and_has_zero_utility_with_null_evaluator(monkeypatch) -> None:
    task = _task()
    backend = _Backend(
        BackendSearchResult(
            SearchStatus.EMBEDDING,
            {0: frozenset({0}), 1: frozenset({1})},
            0.0,
        ),
        delay=0.003,
    )
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: pytest.fail("an over-time result cannot open the evaluator"),
    )
    outcomes, receipts = run_external_complete_system(
        [task],
        Context(qubit_cap=4),
        backend,
        _stock_config(seconds=0.001),
        learned_config=_learned_config(seconds=0.001),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=_population(task),
        quality_authority={"publisher_id": "publisher"},
        seed=17,
        repetitions=1,
    )

    assert outcomes[0].returned_valid is False
    assert outcomes[0].utility == 0.0
    assert outcomes[0].evaluator_seed is None
    assert receipts[0].evaluator_seed is None
    assert receipts[0].restarts[0].status == SearchStatus.TIMED_OUT.value
    assert receipts[0].wallclock_compliant is False


def test_external_receipt_rejects_nonregistered_restart_seed(monkeypatch) -> None:
    task = _task()
    backend = _Backend(BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.0))
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: pytest.fail("failed external attempt cannot be sampled"),
    )
    _, receipts = run_external_complete_system(
        [task],
        Context(qubit_cap=4),
        backend,
        _stock_config(),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=_population(task),
        quality_authority={"publisher_id": "publisher"},
        seed=17,
        repetitions=1,
    )
    restart = receipts[0].restarts[0]
    changed = dataclasses.replace(restart, seed=(restart.seed + 1) % (2**31))

    with pytest.raises(ValueError, match="restart seed schedule"):
        dataclasses.replace(receipts[0], restarts=(changed,))


def test_external_receipt_rejects_known_false_work_cap(monkeypatch) -> None:
    task = _task()
    backend = _Backend(BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.0))
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: pytest.fail("failed external attempt cannot be sampled"),
    )
    _, receipts = run_external_complete_system(
        [task],
        Context(qubit_cap=4),
        backend,
        _stock_config(),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=_population(task),
        quality_authority={"publisher_id": "publisher"},
        seed=17,
        repetitions=1,
    )
    cap = dataclasses.replace(receipts[0].work_cap, restart_work=0)
    compliance = receipts[0].observed_work.cap_compliance(cap)

    with pytest.raises(ValueError, match="known work-cap violation"):
        dataclasses.replace(receipts[0], work_cap=cap, cap_compliance=compliance)


def test_external_runner_stops_before_known_validator_cap_violation(monkeypatch) -> None:
    task = _task()
    base = Context(qubit_cap=4)
    context = dataclasses.replace(
        base,
        caps=dataclasses.replace(base.caps, validator_calls=0),
        reserve=dataclasses.replace(base.reserve, validator_calls=0),
    )
    backend = _Backend(
        BackendSearchResult(
            SearchStatus.EMBEDDING,
            {0: frozenset({0}), 1: frozenset({1})},
            0.0,
        )
    )
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.p_embed",
        lambda *args, **kwargs: pytest.fail("validator cap must be checked prospectively"),
    )
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: pytest.fail("work-capped result cannot be sampled"),
    )
    outcomes, receipts = run_external_complete_system(
        [task],
        context,
        backend,
        _stock_config(),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=_population(task),
        quality_authority={"publisher_id": "publisher"},
        seed=17,
        repetitions=1,
    )

    assert outcomes[0].reason == "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED"
    assert receipts[0].restarts[0].status == "WORK_CAP_EXHAUSTED"
    assert all(value is not False for value in receipts[0].cap_compliance.values())


def test_external_runner_reuses_one_worker_per_task_repetition(monkeypatch) -> None:
    class Session:
        startup_seconds = 0.0

        def __init__(self, owner) -> None:
            self.owner = owner

        def search(self, *, seed, timeout_seconds):
            del seed, timeout_seconds
            self.owner.searches += 1
            return BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.0)

        def close(self) -> None:
            self.owner.closes += 1

    class SessionBackend(_Backend):
        def __init__(self) -> None:
            super().__init__(BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.0))
            self.opens = 0
            self.searches = 0
            self.closes = 0

        def open_session(self, logical, host, *, timeout_seconds, parameters):
            del logical, host, timeout_seconds, parameters
            self.opens += 1
            return Session(self)

        def search(self, *args, **kwargs):
            raise AssertionError("persistent publication runner cannot use one-shot search")

    task = _task()
    backend = SessionBackend()
    stock_config = dataclasses.replace(_stock_config(), max_restarts=3)
    learned_config = dataclasses.replace(
        _learned_config(), max_initializer_attempts=3
    )
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: pytest.fail("failed attempts cannot be sampled"),
    )
    _, receipts = run_external_complete_system(
        [task],
        Context(qubit_cap=4),
        backend,
        stock_config,
        learned_config=learned_config,
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=_population(task, repetitions=2),
        quality_authority={"publisher_id": "publisher"},
        seed=17,
        repetitions=2,
    )

    assert (backend.opens, backend.searches, backend.closes) == (2, 6, 2)
    assert all(receipt.adapter_startup_seconds == 0.0 for receipt in receipts)


def test_ordinary_backend_error_is_failure_zero_not_an_integrity_exception(monkeypatch) -> None:
    task = _task()
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: pytest.fail("ordinary failure cannot open the evaluator"),
    )
    outcomes, receipts = run_external_complete_system(
        [task],
        Context(qubit_cap=4),
        _Backend(BackendSearchResult(SearchStatus.ERROR, None, 0.0, "search failed")),
        _stock_config(),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=_population(task),
        quality_authority={"publisher_id": "publisher"},
        seed=17,
        repetitions=1,
    )

    assert outcomes[0].utility == 0.0
    assert outcomes[0].evaluator_seed is None
    assert receipts[0].restarts[0].status == SearchStatus.ERROR.value


def test_external_receipt_reader_rejects_rehashed_authority_tamper(monkeypatch, tmp_path) -> None:
    task = _task()
    population = _population(task)
    context = Context(qubit_cap=4)
    backend = _Backend(BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.0))
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: pytest.fail("failed external attempt cannot be sampled"),
    )
    quality = {"publisher_id": "publisher"}
    outcomes, receipts = run_external_complete_system(
        [task],
        context,
        backend,
        _stock_config(),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=population,
        quality_authority=quality,
        seed=17,
        repetitions=1,
    )
    outcome_sha = write_external_complete_outcomes(tmp_path / "outcomes.jsonl", outcomes)
    receipt_sha = write_external_complete_receipts(tmp_path / "receipts.jsonl", receipts)
    loaded_outcomes = read_external_complete_outcomes(
        tmp_path / "outcomes.jsonl", expected_sha256=outcome_sha
    )
    loaded = read_external_complete_receipts(
        tmp_path / "receipts.jsonl",
        outcomes=loaded_outcomes,
        expected_backend=backend.identity,
        expected_selector=_selector(),
        expected_population=population,
        expected_config_digest=_stock_config().digest,
        expected_learned_config_digest=_learned_config().digest,
        expected_context_digest=stable_digest(context_snapshot(context)),
        expected_quality_authority_digest=content_digest(quality),
        expected_work_cap=context.caps,
        expected_sha256=receipt_sha,
    )
    assert loaded == receipts

    rows = (tmp_path / "receipts.jsonl").read_text().splitlines()
    import json

    row = json.loads(rows[0])
    row["context_digest"] = "d" * 64
    row["record_digest"] = stable_digest(
        {key: value for key, value in row.items() if key != "record_digest"}
    )
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tampered_sha = __import__("hashlib").sha256(tampered.read_bytes()).hexdigest()
    with pytest.raises(Exception, match="externally supplied authority"):
        read_external_complete_receipts(
            tampered,
            outcomes=loaded_outcomes,
            expected_backend=backend.identity,
            expected_selector=_selector(),
            expected_population=population,
            expected_config_digest=_stock_config().digest,
            expected_learned_config_digest=_learned_config().digest,
            expected_context_digest=stable_digest(context_snapshot(context)),
            expected_quality_authority_digest=content_digest(quality),
            expected_work_cap=context.caps,
            expected_sha256=tampered_sha,
        )


def test_external_terminal_evidence_recompiles_and_rejects_tamper(
    monkeypatch, tmp_path
) -> None:
    task = _task()
    context = Context(qubit_cap=4)
    backend = _Backend(
        BackendSearchResult(
            SearchStatus.EMBEDDING,
            {0: frozenset({0}), 1: frozenset({1})},
            0.0,
        )
    )
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: ReadBlock(4096, 4096, 0.0, 0.0, 1),
    )
    _, receipts = run_external_complete_system(
        [task],
        context,
        backend,
        _stock_config(),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=_population(task),
        quality_authority={"publisher_id": "publisher"},
        seed=17,
        repetitions=1,
    )
    path = tmp_path / "terminal_evidence.jsonl"
    digest = write_external_complete_evidence(path, receipts)
    evidence = read_external_complete_evidence(
        path,
        receipts=receipts,
        tasks=[task],
        context=context,
        expected_sha256=digest,
    )
    assert evidence[0].terminal_evidence is not None

    content = path.read_bytes().replace(b'"hits":4096', b'"hits":4095')
    path.write_bytes(content)
    with pytest.raises(Exception, match="digest|evidence"):
        read_external_complete_evidence(
            path,
            receipts=receipts,
            tasks=[task],
            context=context,
            expected_sha256=hashlib.sha256(content).hexdigest(),
        )


def test_publication_external_cli_has_no_seed_partition_repetition_or_legacy_override() -> None:
    from isingfold.rl.cli import build_parser

    parser = build_parser()
    arguments = [
        "evaluate-external-complete-system",
        "--grid",
        "grid.json",
        "--corpus",
        "prepared-v2",
        "--selector",
        "selector-v2",
        "--quality-attestation",
        "attestation.json",
        "--expected-quality-attestation-digest",
        "a" * 64,
        "--expected-quality-publisher-id",
        "publisher",
        "--ground-certificate-root",
        "ground-certificate-root.json",
        "--expected-ground-certificate-root-sha256",
        "b" * 64,
        "--config",
        "external.json",
        "--learned-config",
        "learned.json",
        "--tuning-registry",
        "external-tuning-registry.json",
        "--expected-tuning-registry-sha256",
        "c" * 64,
        "--external-tuning-selection",
        "external-tuning-selection.json",
        "--expected-external-tuning-selection-sha256",
        "d" * 64,
        "--index",
        "0",
        "--out",
        "evaluation",
    ]
    parsed = parser.parse_args(arguments)
    assert not hasattr(parsed, "seed")
    assert not hasattr(parsed, "partition")
    assert not hasattr(parsed, "repetitions")
    assert not hasattr(parsed, "allow_legacy_pilot")
    with pytest.raises(SystemExit):
        parser.parse_args([*arguments, "--seed", "7"])


def test_validation_tuning_cli_has_exact_candidate_seed_census_and_no_partition() -> None:
    from isingfold.rl.cli import build_parser

    arguments = [
        "evaluate-external-tuning-cell",
        "--registry",
        "registry.json",
        "--expected-registry-sha256",
        "a" * 64,
        "--grid",
        "grid.json",
        "--index",
        "6",
        "--tuning-seed-index",
        "2",
        "--corpus",
        "prepared-v3",
        "--selector",
        "selector-v2",
        "--quality-attestation",
        "attestation.json",
        "--expected-quality-attestation-digest",
        "b" * 64,
        "--expected-quality-publisher-id",
        "publisher",
        "--ground-certificate-root",
        "ground-certificate-root.json",
        "--expected-ground-certificate-root-sha256",
        "c" * 64,
        "--external-config",
        "external.json",
        "--learned-config",
        "learned.json",
        "--out",
        "tuning-root",
    ]
    parsed = build_parser().parse_args(arguments)
    assert parsed.index == 6
    assert parsed.tuning_seed_index == 2
    assert not hasattr(parsed, "partition")
    assert not hasattr(parsed, "seed")
    assert not hasattr(parsed, "repetitions")


def test_checked_in_external_complete_config_matches_learned_envelope() -> None:
    root = Path(__file__).resolve().parents[2]
    external = ExternalCompleteSystemConfig.from_mapping(
        json.loads((root / "configs/external_minorminer_complete_v1.json").read_text())
    )
    learned = CompleteSystemConfig.from_mapping(
        json.loads((root / "configs/complete_system_lac_hybrid_cache_v1.json").read_text())
    )
    external.validate_symmetric_envelope(learned, Context(qubit_cap=64))


def test_external_selection_is_authenticated_before_test_targets_are_loaded(
    monkeypatch, tmp_path
) -> None:
    from isingfold.rl import cli
    from isingfold.rl import external_tuning

    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        external_tuning,
        "load_external_tuning_selection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("selection-authentication-sentinel")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_quality_partition",
        lambda *args, **kwargs: pytest.fail(
            "test targets were loaded before tuning selection authentication"
        ),
    )
    parsed = cli.build_parser().parse_args(
        [
            "evaluate-external-complete-system",
            "--grid",
            str(root / "configs/rl_grid_hybrid_v1.json"),
            "--corpus",
            str(tmp_path / "unopened-test-corpus"),
            "--selector",
            str(tmp_path / "selector"),
            "--quality-attestation",
            str(tmp_path / "attestation.json"),
            "--expected-quality-attestation-digest",
            "a" * 64,
            "--expected-quality-publisher-id",
            "publisher",
            "--ground-certificate-root",
            str(tmp_path / "ground.json"),
            "--expected-ground-certificate-root-sha256",
            "b" * 64,
            "--config",
            str(root / "configs/external_minorminer_complete_v1.json"),
            "--learned-config",
            str(root / "configs/complete_system_lac_hybrid_cache_v1.json"),
            "--tuning-registry",
            str(root / "configs/external_minorminer_tuning_hybrid_v1.json"),
            "--expected-tuning-registry-sha256",
            EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
            "--external-tuning-selection",
            str(tmp_path / "selection.json"),
            "--expected-external-tuning-selection-sha256",
            "c" * 64,
            "--index",
            "0",
            "--out",
            str(tmp_path / "external-test"),
        ]
    )
    with pytest.raises(ValueError, match="selection-authentication-sentinel"):
        parsed.func(parsed)


@pytest.mark.parametrize(
    "command",
    ("evaluate-complete-system-cell", "evaluate-external-complete-system"),
)
def test_config_pins_fail_before_any_test_target_load(
    monkeypatch, tmp_path, command
) -> None:
    from isingfold.rl import cli

    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/complete_system_lac_hybrid_cache_v1.json").read_text())
    payload["online_wallclock_seconds"] = 61.0
    changed = tmp_path / "changed-learned.json"
    changed.write_bytes(canonical_json_bytes(payload) + b"\n")
    monkeypatch.setattr(
        cli,
        "_load_quality_partition",
        lambda *args, **kwargs: pytest.fail("config pins must precede test target loading"),
    )
    common = [
        command,
        "--grid",
        str(root / "configs/rl_grid_hybrid_v1.json"),
        "--corpus",
        str(tmp_path / "unopened-corpus"),
        "--selector",
        str(tmp_path / "selector"),
        "--quality-attestation",
        str(tmp_path / "attestation.json"),
        "--expected-quality-attestation-digest",
        "a" * 64,
        "--expected-quality-publisher-id",
        "publisher",
        "--ground-certificate-root",
        str(tmp_path / "ground-certificate-root.json"),
        "--expected-ground-certificate-root-sha256",
        "b" * 64,
        "--index",
        "0",
        "--out",
        str(tmp_path / f"{command}-out"),
    ]
    if command == "evaluate-complete-system-cell":
        common.extend(
            [
                "--run-root",
                str(tmp_path / "runs"),
                "--config",
                str(changed),
                "--bootstrap-bank",
                str(tmp_path / "bootstrap-bank"),
                "--expected-bootstrap-plan-sha256",
                "d" * 64,
                "--expected-bootstrap-manifest-sha256",
                "e" * 64,
                "--rl-value-selection-receipt",
                str(tmp_path / "freeze.json"),
                "--expected-selection-sha256",
                "b" * 64,
            ]
        )
    else:
        common.extend(
            [
                "--config",
                str(root / "configs/external_minorminer_complete_v1.json"),
                "--learned-config",
                str(changed),
                "--tuning-registry",
                str(root / "configs/external_minorminer_tuning_hybrid_v1.json"),
                "--expected-tuning-registry-sha256",
                EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
                "--external-tuning-selection",
                str(tmp_path / "external-selection.json"),
                "--expected-external-tuning-selection-sha256",
                "c" * 64,
            ]
        )
    args = cli.build_parser().parse_args(common)
    with pytest.raises(ValueError, match="preregistered semantic/file pins"):
        args.func(args)


def test_paired_aggregate_cli_requires_three_out_of_band_report_pins() -> None:
    from isingfold.rl.cli import build_parser

    parsed = build_parser().parse_args(
        [
            "aggregate-learned-vs-stock",
            "--grid",
            "grid.json",
            "--corpus",
            "corpus",
            "--quality-attestation",
            "attestation.json",
            "--expected-quality-attestation-digest",
            "a" * 64,
            "--expected-quality-publisher-id",
            "publisher",
            "--ground-certificate-root",
            "ground-certificate-root.json",
            "--expected-ground-certificate-root-sha256",
            "b" * 64,
            "--learned-config",
            "learned.json",
            "--external-config",
            "external.json",
            "--tuning-registry",
            "external-tuning-registry.json",
            "--expected-tuning-registry-sha256",
            "c" * 64,
            "--external-tuning-selection",
            "external-tuning-selection.json",
            "--expected-external-tuning-selection-sha256",
            "d" * 64,
            *sum(
                (
                    [
                        "--learned-evaluation",
                        f"learned-{index}",
                        "--expected-learned-report-sha256",
                        str(index) * 64,
                    ]
                    for index in range(1, 4)
                ),
                [],
            ),
            *sum(
                (
                    [
                        "--external-evaluation",
                        f"stock-{index}",
                        "--expected-external-report-sha256",
                        "a" * 64,
                    ]
                    for index in range(3)
                ),
                [],
            ),
            "--out",
            "paired.json",
        ]
    )
    assert len(parsed.learned_evaluation) == 3
    assert len(parsed.expected_learned_report_sha256) == 3
    assert len(parsed.external_evaluation) == 3
    assert len(parsed.expected_external_report_sha256) == 3
    assert not hasattr(parsed, "seed")
    assert not hasattr(parsed, "repetitions")
    assert not hasattr(parsed, "partition")


def test_external_complete_launchers_preserve_apollo_and_goose_scheduler_rules() -> None:
    root = Path(__file__).resolve().parents[2]
    apollo = (root / "scripts/apollo_external_complete_eval.sh").read_text()
    goose = (root / "scripts/goose_external_complete_eval.sbatch").read_text()
    combined_apollo = (root / "scripts/apollo_complete_eval.sh").read_text()
    combined_goose = (root / "scripts/goose_complete_eval.sbatch").read_text()

    assert "evaluate-external-complete-system" in apollo
    assert "sbatch" not in apollo and "/opt/slurm/bin/srun" not in apollo
    assert "SLURM_JOB_ID" in goose
    assert "/opt/slurm/bin/srun" in goose
    assert "evaluate-external-complete-system" in goose
    for source in (apollo, goose, combined_apollo, combined_goose):
        assert "--quality-attestation" in source
        assert "--expected-quality-attestation-digest" in source
        assert "--expected-quality-publisher-id" in source
        assert "--ground-certificate-root" in source
        assert "--expected-ground-certificate-root-sha256" in source
        assert "--tuning-registry" in source
        assert "--external-tuning-selection" in source
        assert "--expected-external-tuning-selection-sha256" in source


def test_external_tuning_launchers_keep_each_seed_census_on_one_machine() -> None:
    root = Path(__file__).resolve().parents[2]
    apollo = (root / "scripts/apollo_external_tuning.sh").read_text()
    goose = (root / "scripts/goose_external_tuning.sbatch").read_text()

    assert "sbatch" not in apollo and "/opt/slurm/bin/srun" not in apollo
    assert "SLURM_JOB_ID" in goose and "#SBATCH --array=0-2" in goose
    assert "/opt/slurm/bin/srun" in goose
    for source in (apollo, goose):
        assert "for candidate_index in {0..6}" in source
        assert "evaluate-external-tuning-cell" in source
        assert "--tuning-seed-index" in source
        assert "--expected-registry-sha256" in source
        assert "--ground-certificate-root" in source


def test_paired_aggregate_uses_equal_seed_then_equal_lineage_weighting(
    monkeypatch, tmp_path
) -> None:
    from tests.unit import test_rl_complete_system_aggregate as learned_fixtures

    learned_runs = learned_fixtures._runs(monkeypatch, tmp_path / "learned")
    tasks = (
        learned_fixtures._task("quality-one", "lineage-a", positive=True),
        learned_fixtures._task("quality-zero-1", "lineage-b", positive=False),
        learned_fixtures._task("quality-zero-2", "lineage-b", positive=False),
    )
    context = Context(qubit_cap=4)
    population = learned_runs[0].receipts[0].population
    selector_identity = learned_runs[0].receipts[0].selector
    learned_config = learned_fixtures._config()
    stock_config = _stock_config()
    backend = _Backend(
        BackendSearchResult(
            SearchStatus.EMBEDDING,
            {0: frozenset({0}), 1: frozenset({1})},
            0.0,
        )
    )
    quality = dict(learned_runs[0].report["quality_authority"])
    tuning_execution = _tuned_stock_default()

    def stock_sample(program, chains, problem, ground, *, num_reads, seed, num_sweeps):
        del program, chains, problem, ground, seed, num_sweeps
        return ReadBlock(1024, num_reads, 0.0, 0.0, 1)

    monkeypatch.setattr("isingfold.rl.external_pairing.sample_program", stock_sample)
    stock_runs = []
    runtime_registry = runtime_implementation_registry()
    for index, learned_run in enumerate(learned_runs):
        compute_identity = runtime_identity(
            runtime_platform=learned_run.report["runtime_platform"],
            inference_device_type=learned_run.report["inference_device_type"],
            inference_device_name=learned_run.report["inference_device_name"],
            inference_threads=learned_run.report["inference_threads"],
            deterministic=learned_run.report["deterministic"],
        )
        outcomes, receipts = run_external_complete_system(
            tasks,
            context,
            backend,
            stock_config,
            learned_config=learned_config,
            selector=fixed_strength_selector(),
            selector_identity=selector_identity,
            population=population,
            quality_authority=quality,
            seed=55079,
            repetitions=4,
            training_seed_index=index,
            training_seed=(1103, 2207, 3301)[index],
            compute_identity=compute_identity,
            tuning_execution=tuning_execution,
        )
        stock_root = tmp_path / f"stock-{index}"
        outcome_sha = write_external_complete_outcomes(
            stock_root / "outcomes.jsonl", outcomes
        )
        receipt_sha = write_external_complete_receipts(
            stock_root / "external_receipts.jsonl", receipts
        )
        evidence_sha = write_external_complete_evidence(
            stock_root / "terminal_evidence.jsonl", receipts
        )
        payload = {
            "schema": "isingfold.external-complete-system-evaluation",
            "schema_version": 4,
            "status": "complete",
            "partition": "test",
            "sealed_test_opened": True,
            "no_fallback": True,
            "evaluation_protocol": learned_fixtures._protocol(),
            "training_seed_index": index,
            "training_seed": (1103, 2207, 3301)[index],
            "grid_manifest_sha256": stable_digest({"grid": "v1"}),
            "source_corpus_manifest_sha256": population.source_manifest_sha256,
            "quality_authority": quality,
            "target_access": dict(learned_run.report["target_access"]),
            "ground_partition_receipt": dict(
                learned_run.report["ground_partition_receipt"]
            ),
            "context": context_snapshot(context),
            "context_digest": stable_digest(context_snapshot(context)),
            "work_cap": context.caps.as_dict(),
            "population": population.as_dict(),
            "population_digest": population.digest,
            "selector": selector_identity.as_dict(),
            "selector_digest": content_digest(selector_identity.as_dict()),
            "external_tuning_execution": tuning_execution.as_dict(),
            "external_tuning_execution_digest": tuning_execution.digest,
            "backend": backend.identity.as_dict(),
            "external_config": stock_config.as_dict(),
            "external_config_digest": stock_config.digest,
            "external_config_file_sha256": stable_digest(stock_config.as_dict()),
            "learned_config": learned_config.as_dict(),
            "learned_config_digest": learned_config.digest,
            "learned_config_file_sha256": stable_digest(learned_config.as_dict()),
            **dict(compute_identity),
            "runtime_implementation_registry": runtime_registry,
            "runtime_implementation_digest": content_digest(runtime_registry),
            "external_pairing_implementation_sha256": runtime_registry["modules"][
                "isingfold.rl.external_pairing"
            ]["sha256"],
            "method": {
                "method_id": STOCK_MINORMINER_METHOD,
                "comparison_scope": "paired-external-complete-system",
                "selection_rule": tuning_execution.candidate.candidate_ranking,
                "online_evaluator_feedback": False,
                "total_online_wallclock_seconds": 30.0,
                "final_evaluator": "one-fresh-independent-4096-read-block-after-selection",
                "native_internal_work": "unavailable-retained-as-null",
                "adapter_process_model": (
                    "one-persistent-python-worker-per-task-repetition"
                ),
                "latency_scope": (
                    "same-machine-rowwise-total-online-wallclock-includes-adapter-startup"
                ),
            },
            "summary": external_complete_summary(outcomes, receipts),
            "artifacts": {
                "receipts": {
                    "path": "external_receipts.jsonl",
                    "sha256": receipt_sha,
                    "count": len(receipts),
                },
                "terminal_evidence": {
                    "path": "terminal_evidence.jsonl",
                    "sha256": evidence_sha,
                    "count": len(receipts),
                },
                "outcomes": {
                    "path": "outcomes.jsonl",
                    "sha256": outcome_sha,
                    "count": len(outcomes),
                },
            },
        }
        report = {**payload, "record_digest": content_digest(payload)}
        report_content = canonical_json_bytes(report) + b"\n"
        (stock_root / "report.json").write_bytes(report_content)
        stock_runs.append(
            load_authenticated_external_complete_run(
                stock_root,
                expected_report_sha256=hashlib.sha256(report_content).hexdigest(),
                tasks=tasks,
                context=context,
            )
        )
    stock_run = stock_runs[0]
    with pytest.raises(TypeError):
        stock_run.report["quality_authority"]["publisher_id"] = "tampered"

    aggregate = aggregate_learned_vs_stock(learned_runs, stock_runs)

    # Equal-lineage learned utility is 1/3, while stock is 1/4.  Weighting the two
    # lineage-b instances separately would produce -1/36 and must not occur.
    assert aggregate["unconditional_utility_difference"] == pytest.approx(1 / 12)
    assert aggregate["independent_lineages"] == 2
    assert aggregate["attempts_per_training_seed"] == 12
    assert aggregate["matched_contract"]["population_digest"] == population.digest
    assert aggregate["runtime_comparability"]["total_online_envelope_symmetric"] is True
    assert aggregate["subgroup_results"]["distribution"]["ood"][
        "independent_lineages"
    ] == 2
    confirmatory = aggregate["confirmatory_feasibility_noninferiority"]
    assert confirmatory["design_digest"] == population.confirmatory_design.record_digest
    assert confirmatory["noninferiority_margin"] == 0.02
    assert confirmatory["multiplicity_method"] == "prespecified-alpha-spending-v1"
    assert confirmatory["all_targets_pass"] is False
    target = confirmatory["targets"][0]
    assert target["endpoint"] == "valid-return-noninferiority"
    assert target["alpha"] == 0.05
    assert target["filter"] == population.confirmatory_design.power_targets[0].design_filter.as_dict()
    assert target["independent_lineages"] == 2
    assert target["informative_discordant_lineages"] == 1
    assert "insufficient-informative-discordant-lineages" in target["failure_reasons"]
    assert target["passed"] is False
    assert aggregate["feasibility_noninferiority"] is False
    fixed_sequence = aggregate["confirmatory_fixed_sequence"]
    assert fixed_sequence["familywise_alpha"] == 0.05
    assert fixed_sequence["strong_familywise_error_control"] is True
    assert fixed_sequence["claims"][1]["tested"] is False
    assert fixed_sequence["claims"][1]["passed"] is False
    assert aggregate["primary_learned_superiority"] is False
    primary = aggregate["primary_paired_utility_precision"]
    assert primary["design_digest"] == population.confirmatory_design.record_digest
    assert primary["multiplicity_method"] == "single-prespecified-primary-no-adjustment"
    assert primary["all_targets_pass"] is True
    precision = primary["targets"][0]
    assert precision["endpoint"] == "learned-minus-stock-unconditional-if-q3-s0"
    assert precision["outcome_bounds"] == [-1.0, 1.0]
    assert precision["variance_bound"] == 1.0
    assert precision["independent_lineages"] == 2
    assert precision["sample_size_guard_pass"] is True
    assert precision["achieved_half_width_guard_pass"] is True

    from isingfold.rl.complete_system_aggregate import AuthenticatedCompleteSystemSeedRun

    changed_payload = {
        key: value
        for key, value in learned_runs[0].report.items()
        if key != "record_digest"
    }
    changed_payload["inference_device_name"] = "different-device"
    changed_report = {
        **changed_payload,
        "record_digest": content_digest(changed_payload),
    }
    mismatched = AuthenticatedCompleteSystemSeedRun(
        changed_report,
        learned_runs[0].receipts,
        learned_runs[0].evidence,
    )
    with pytest.raises(ValueError, match="exact compute contracts"):
        aggregate_learned_vs_stock(
            (mismatched, learned_runs[1], learned_runs[2]), stock_runs
        )

    with pytest.raises(ValueError, match="identity|sealed population census"):
        AuthenticatedExternalCompleteRun(
            report=report,
            receipts=tuple(receipts[:-1]),
            evidence=stock_runs[-1].evidence[:-1],
            receipt_file_sha256=receipt_sha,
            evidence_file_sha256=evidence_sha,
            outcome_file_sha256=outcome_sha,
        )


def test_confirmatory_gate_cannot_pass_one_lineage_or_all_zero_pairs() -> None:
    _, design = evaluation_contract((("lineage-a", "instance-a"),))
    zeros = np.zeros((3, 1), dtype=float)
    matrices = _PairedMatrices(
        lineages=("lineage-a",),
        attempts_per_training_seed=1,
        unique_pair_count=1,
        utility_difference=zeros,
        feasibility_difference=zeros,
        learned_utility=zeros,
        learned_feasibility=zeros,
        stock_utility=zeros,
        stock_feasibility=zeros,
        informative_discordant_lineages=(),
        informative_discordant_pairs=(),
        discordant_pair_occurrences=0,
        total_pair_occurrences=3,
    )

    result = _confirmatory_target_result(design.power_targets[0], design, matrices)

    assert result["bound_guard_pass"] is True
    assert result["sample_size_guard_pass"] is False
    assert result["discordance_guard_pass"] is False
    assert result["passed"] is False
    assert set(result["failure_reasons"]) >= {
        "insufficient-independent-lineages",
        "insufficient-informative-discordant-lineages",
        "insufficient-informative-discordant-pairs",
    }
    precision = _primary_precision_target_result(
        design.precision_targets[0], matrices
    )
    assert precision["achieved_half_width_guard_pass"] is True
    assert precision["sample_size_guard_pass"] is False
    assert precision["learned_superiority_ci_lower_above_zero"] is False
    assert precision["passed"] is False

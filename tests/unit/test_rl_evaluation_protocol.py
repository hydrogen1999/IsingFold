"""Publication evaluation contracts for IsingFold.

These tests deliberately exercise the evaluator-only path rather than PPO.  A benchmark
receipt must be replayable at the instance/repetition level before it can be clustered by
base lineage for inference.
"""

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Candidate, Context, Opcode, WorkVector
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluate import (
    INITIALIZATION_FAILURE,
    ComparisonFamily,
    ComparisonSpec,
    EpisodeOutcome,
    EvaluationArm,
    EvaluationProtocolError,
    compare_arms,
    paired_endpoint,
    program_digest,
    read_episode_receipts,
    repair_target_satisfied,
    run_controller,
    secondary_metrics,
    write_episode_receipts,
)
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.program import Program
from isingfold.rl.proposal import router_initializer


def _outcome(
    instance: str,
    lineage: str,
    repetition: int,
    utility: float,
    *,
    valid: bool = True,
    seed: int = 7,
    qubits: int = 4,
) -> EpisodeOutcome:
    reads = 10 if valid else None
    hits = int(round(utility * 10)) if valid else None
    return EpisodeOutcome(
        instance=instance,
        lineage=lineage,
        returned_valid=valid,
        utility=utility,
        qubits=qubits if valid else None,
        max_chain=2 if valid else None,
        decisions=3,
        selected_strength=1.0 if valid else None,
        reason="COMMIT" if valid else "BUDGET_NO_VALID",
        repetition=repetition,
        episode_seed=seed,
        evaluator_seed=seed + 1 if valid else None,
        program_digest="a" * 64 if valid else None,
        selected_strength_index=1 if valid else None,
        evaluator_hits=hits,
        evaluator_reads=reads,
        work=WorkVector(decisions=3, evaluator_reads=reads or 0),
        validation_digest="b" * 64,
        broken_chain_fraction=0.2 if valid else None,
        mean_energy_residual=0.3 if valid else None,
        online_seconds=0.5,
        evaluator_seconds=0.1 if valid else None,
        controller_calls=3,
        controller_seconds=0.05,
    )


def _tiny_task() -> EmbeddingTask:
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4))
    logical = nx.path_graph(3)
    rng = random.Random(3)
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
    return EmbeddingTask("tiny", logical, host, problem, ground, lineage="base-0")


def test_program_digest_is_canonical_and_binds_physical_coefficients() -> None:
    first = Program(
        strength=1.0,
        strength_index=0,
        scale=0.5,
        h_phys={1: 0.25, 0: -0.25},
        j_phys={(0, 1): -0.5},
        chain_edges={"x": frozenset({(0, 1)})},
        contact_counts={("x", "y"): 1},
        offset=-0.5,
    )
    reordered = Program(
        strength=1.0,
        strength_index=0,
        scale=0.5,
        h_phys={0: -0.25, 1: 0.25},
        j_phys={(0, 1): -0.5},
        chain_edges={"x": frozenset({(0, 1)})},
        contact_counts={("x", "y"): 1},
        offset=-0.5,
    )
    changed = Program(**{**reordered.__dict__, "j_phys": {(0, 1): -0.4}})

    assert program_digest(first) == program_digest(reordered)
    assert program_digest(first) != program_digest(changed)


def test_repair_telemetry_accepts_conflict_and_demand_targets() -> None:
    host = nx.path_graph(3)
    demand = Candidate(
        opcode=Opcode.REPAIR_GROUP,
        affected=("a", "b"),
        old_chains={"a": frozenset({0}), "b": frozenset({2})},
        new_chains={"a": frozenset({0}), "b": frozenset({1})},
        work=WorkVector(),
        payload_key="demand-repair",
        target_demand=("a", "b"),
    )
    environment = SimpleNamespace(
        task=SimpleNamespace(host=host),
        state=SimpleNamespace(chains=demand.new_chains),
    )
    assert repair_target_satisfied(demand, environment)
    environment.state = SimpleNamespace(chains=demand.old_chains)
    assert not repair_target_satisfied(demand, environment)

    conflict = Candidate(
        opcode=Opcode.REPAIR_GROUP,
        affected=("a", "b"),
        old_chains={"a": frozenset({1}), "b": frozenset({1})},
        new_chains={"a": frozenset({0}), "b": frozenset({1})},
        work=WorkVector(),
        payload_key="conflict-repair",
        target_conflict=1,
    )
    environment.state = SimpleNamespace(chains=conflict.new_chains)
    assert repair_target_satisfied(conflict, environment)


def test_run_controller_emits_complete_external_evaluator_receipts(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_sample(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, chains, problem, ground_energy, num_sweeps
        calls.append((num_reads, seed))
        return ReadBlock(3, num_reads, 0.125, 0.4, 1)

    monkeypatch.setattr("isingfold.rl.evaluate.sample_program", fake_sample)
    task = _tiny_task()
    outcomes = run_controller(
        [task],
        Context(qubit_cap=20),
        lambda decision, rng: next(
            index
            for index, candidate in enumerate(decision.candidates)
            if candidate.opcode.value == "COMMIT" and decision.legal_mask[index]
        ),
        initializer=router_initializer(),
        selector=fixed_strength_selector(),
        reward_reads=8,
        seed=19,
        repetitions=2,
    )

    assert len(outcomes) == 2 and len(calls) == 2
    assert {row.repetition for row in outcomes} == {0, 1}
    assert all(row.evaluator_hits == 3 and row.evaluator_reads == 8 for row in outcomes)
    assert all(row.utility == pytest.approx(3 / 8) for row in outcomes)
    assert all(row.program_digest and len(row.program_digest) == 64 for row in outcomes)
    assert all(row.episode_seed is not None and row.evaluator_seed is not None for row in outcomes)
    assert outcomes[0].episode_seed != outcomes[0].evaluator_seed
    assert outcomes[0].work is not None and outcomes[0].work.evaluator_reads == 8
    assert outcomes[0].controller_calls == outcomes[0].decisions
    assert outcomes[0].controller_seconds is not None
    assert outcomes[0].online_seconds is not None
    assert 0.0 <= outcomes[0].controller_seconds <= outcomes[0].online_seconds
    assert outcomes[0].evaluator_seconds is not None
    assert outcomes[0].broken_chain_fraction == pytest.approx(0.125)
    assert outcomes[0].mean_energy_residual == pytest.approx(0.4)
    assert outcomes[0].chain_sizes == (1, 1, 1)
    assert outcomes[0].overlap_events == 0
    assert outcomes[0].overlap_decisions == 0
    assert outcomes[0].repair_attempts == 0
    assert outcomes[0].repair_successes == 0
    assert outcomes[0].candidate_states == outcomes[0].controller_calls
    assert outcomes[0].legal_actions_total >= outcomes[0].candidate_states
    assert outcomes[0].legal_opcode_types_total >= outcomes[0].candidate_states
    outcomes[0].validate_receipt(require_complete=True)


def test_pairing_is_exact_at_instance_and_repetition_level() -> None:
    reference = [_outcome("i", "L", 0, 0.2), _outcome("i", "L", 1, 0.4)]
    treatment = [_outcome("i", "L", 0, 0.3)]

    with pytest.raises(EvaluationProtocolError, match="missing paired receipts"):
        paired_endpoint(treatment, reference, strict_pairs=True, bootstrap=32)

    with pytest.raises(EvaluationProtocolError, match="duplicate pair key"):
        paired_endpoint(reference + [reference[0]], reference, strict_pairs=True, bootstrap=32)


def test_paired_endpoint_averages_repetitions_then_bootstraps_lineages() -> None:
    reference = [
        _outcome("i0", "L0", 0, 0.1),
        _outcome("i0", "L0", 1, 0.3),
        _outcome("i1", "L1", 0, 0.8),
    ]
    treatment = [
        _outcome("i0", "L0", 0, 0.3),
        _outcome("i0", "L0", 1, 0.5),
        _outcome("i1", "L1", 0, 0.7),
    ]
    endpoint = paired_endpoint(
        treatment,
        reference,
        strict_pairs=True,
        bootstrap=256,
        seed=2,
    )

    # Per-lineage deltas are +0.2 and -0.1; repetitions are not pseudo-replicates.
    assert endpoint.n_pairs == 3
    assert endpoint.n_lineages == 2
    assert endpoint.delta_utility == pytest.approx(0.05)
    assert endpoint.wins == 1 and endpoint.losses == 1
    assert endpoint.r99_treatment is None and endpoint.r99_reference is None


def test_joint_valid_conditioning_does_not_mix_different_failure_sets() -> None:
    reference = [
        _outcome("i0", "L0", 0, 0.4, valid=True, qubits=4),
        _outcome("i1", "L1", 0, 0.0, valid=False),
    ]
    treatment = [
        _outcome("i0", "L0", 0, 0.5, valid=True, qubits=8),
        _outcome("i1", "L1", 0, 0.6, valid=True, qubits=100),
    ]
    endpoint = paired_endpoint(
        treatment,
        reference,
        strict_pairs=True,
        require_same_seeds=False,
        bootstrap=64,
    )

    assert endpoint.n_joint_valid_pairs == 1
    assert endpoint.qubit_ratio == pytest.approx(2.0)
    assert endpoint.conditional_delta_utility == pytest.approx(0.1)


def test_empty_and_no_valid_secondary_metrics_are_missing_not_fabricated_zero() -> None:
    empty = secondary_metrics([])
    failed = secondary_metrics([_outcome("i", "L", 0, 0.0, valid=False)])

    assert empty["valid_return_rate"] is None
    assert empty["utility_mean"] is None
    assert failed["valid_return_rate"] == 0.0
    assert failed["conditional_utility"] is None
    assert failed["qubits_mean"] is None
    assert failed["fixed_program_r99_defined_count"] == 0
    assert failed["fixed_program_r99_mean_defined"] is None


def test_secondary_metrics_report_complete_latency_quality_and_fixed_program_r99() -> None:
    first = _outcome("i0", "L0", 0, 0.5)
    second = _outcome("i1", "L1", 0, 0.0)
    first = EpisodeOutcome(
        **{
            **first.__dict__,
            "online_seconds": 2.0,
            "evaluator_seconds": 0.4,
            "controller_seconds": 0.5,
            "broken_chain_fraction": 0.1,
            "mean_energy_residual": 0.25,
        }
    )
    second = EpisodeOutcome(
        **{
            **second.__dict__,
            "online_seconds": 1.0,
            "evaluator_seconds": 0.2,
            "controller_seconds": 0.25,
            "broken_chain_fraction": 0.3,
            "mean_energy_residual": 0.75,
        }
    )

    metrics = secondary_metrics([first, second])

    assert first.fixed_program_r99 == 7
    assert second.fixed_program_r99 is None
    assert metrics["online_seconds_observed"] == 2
    assert metrics["online_seconds_mean"] == pytest.approx(1.5)
    assert metrics["evaluator_seconds_mean_valid"] == pytest.approx(0.3)
    assert metrics["end_to_end_seconds_mean"] == pytest.approx(1.8)
    assert metrics["controller_seconds_total"] == pytest.approx(0.75)
    assert metrics["controller_share_of_online_seconds"] == pytest.approx(0.25)
    assert metrics["broken_chain_fraction_mean_valid"] == pytest.approx(0.2)
    assert metrics["mean_energy_residual_mean_valid"] == pytest.approx(0.5)
    assert metrics["fixed_program_r99_defined_count"] == 1
    assert metrics["fixed_program_r99_zero_hit_count"] == 1
    assert metrics["fixed_program_r99_mean_defined"] == pytest.approx(7.0)


def test_secondary_metrics_publish_mechanism_denominators_and_chain_distribution() -> None:
    first = EpisodeOutcome(
        **{
            **_outcome("i0", "L0", 0, 0.5).__dict__,
            "qubits": 7,
            "max_chain": 4,
            "chain_sizes": (1, 2, 4),
            "overlap_events": 2,
            "overlap_decisions": 3,
            "repair_attempts": 2,
            "repair_successes": 1,
            "candidate_states": 3,
            "legal_actions_total": 18,
            "legal_opcode_types_total": 9,
        }
    )
    second = EpisodeOutcome(
        **{
            **_outcome("i1", "L1", 0, 0.6).__dict__,
            "qubits": 5,
            "max_chain": 3,
            "chain_sizes": (2, 3),
            "overlap_events": 0,
            "overlap_decisions": 0,
            "repair_attempts": 1,
            "repair_successes": 1,
            "candidate_states": 3,
            "legal_actions_total": 12,
            "legal_opcode_types_total": 6,
        }
    )

    metrics = secondary_metrics([first, second])

    assert metrics["chain_sizes_observed_valid"] == 2
    assert metrics["chain_size_count"] == 5
    assert metrics["chain_size_distribution"] == {
        "min": 1.0,
        "q25": 2.0,
        "median": 2.0,
        "q75": 3.0,
        "max": 4.0,
        "mean": pytest.approx(2.4),
    }
    assert metrics["overlap_events_total"] == 2
    assert metrics["overlap_decisions_total"] == 3
    assert metrics["repair_attempts_total"] == 3
    assert metrics["repair_successes_total"] == 2
    assert metrics["repair_success_rate"] == pytest.approx(2 / 3)
    assert metrics["candidate_states_total"] == 6
    assert metrics["legal_actions_mean_per_candidate_state"] == 5.0
    assert metrics["legal_opcode_types_mean_per_candidate_state"] == 2.5


def test_receipt_rejects_inconsistent_timing_and_evaluator_telemetry() -> None:
    row = _outcome("i", "L", 0, 0.5)
    with pytest.raises(EvaluationProtocolError, match="controller time"):
        EpisodeOutcome(
            **{**row.__dict__, "controller_seconds": 0.6, "online_seconds": 0.5}
        ).validate_receipt(require_complete=True)
    with pytest.raises(EvaluationProtocolError, match="call count"):
        EpisodeOutcome(**{**row.__dict__, "controller_calls": row.decisions - 1}).validate_receipt(
            require_complete=True
        )

    failed = _outcome("f", "L", 0, 0.0, valid=False)
    with pytest.raises(EvaluationProtocolError, match="no-valid terminal"):
        EpisodeOutcome(**{**failed.__dict__, "broken_chain_fraction": 0.1}).validate_receipt(
            require_complete=True
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("qubits", 3),
        ("max_chain", 2),
        ("evaluator_seed", 19),
        ("program_digest", "a" * 64),
        ("selected_strength", 1.0),
        ("selected_strength_index", 0),
        ("evaluator_hits", 0),
        ("evaluator_reads", 10),
        ("evaluator_seconds", 0.0),
        ("chain_sizes", (1, 1, 1)),
    ),
)
def test_ordinary_invalid_receipt_rejects_every_valid_only_field(field, value) -> None:
    failed = _outcome("f", "L", 0, 0.0, valid=False)

    with pytest.raises(EvaluationProtocolError, match="no-valid terminal"):
        EpisodeOutcome(**{**failed.__dict__, field: value}).validate_receipt()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("qubits", 3),
        ("max_chain", 2),
        ("evaluator_seed", 19),
        ("program_digest", "a" * 64),
        ("selected_strength", 1.0),
        ("selected_strength_index", 0),
        ("evaluator_hits", 0),
        ("evaluator_reads", 10),
        ("evaluator_seconds", 0.0),
        ("chain_sizes", (1, 1, 1)),
    ),
)
def test_initializer_failure_rejects_every_valid_only_field(field, value) -> None:
    failure = EpisodeOutcome(
        instance="f",
        lineage="L",
        returned_valid=False,
        utility=None,
        qubits=None,
        max_chain=None,
        decisions=0,
        selected_strength=None,
        reason="INITIALIZER_FAILURE",
        episode_seed=7,
        work=WorkVector(restart_work=1),
        validation_digest="b" * 64,
        online_seconds=0.1,
        controller_calls=0,
        controller_seconds=0.0,
        outcome_kind=INITIALIZATION_FAILURE,
        population_eligible=False,
    )

    with pytest.raises(EvaluationProtocolError, match="initializer failure"):
        EpisodeOutcome(**{**failure.__dict__, field: value}).validate_receipt()


def test_run_controller_rejects_an_evaluator_read_budget_mismatch(monkeypatch) -> None:
    def wrong_reads(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, chains, problem, ground_energy, seed, num_sweeps
        return ReadBlock(1, num_reads - 1, 0.0, 0.0, 1)

    monkeypatch.setattr("isingfold.rl.evaluate.sample_program", wrong_reads)
    with pytest.raises(EvaluationProtocolError, match="read budget"):
        run_controller(
            [_tiny_task()],
            Context(qubit_cap=20),
            lambda decision, rng: next(
                index
                for index, candidate in enumerate(decision.candidates)
                if candidate.opcode.value == "COMMIT" and decision.legal_mask[index]
            ),
            initializer=router_initializer(),
            selector=fixed_strength_selector(),
            reward_reads=8,
        )


def test_comparison_structure_checks_matched_protocol_metadata() -> None:
    rows_a = (_outcome("i", "L", 0, 0.6),)
    rows_b = (_outcome("i", "L", 0, 0.4),)
    treatment = EvaluationArm(
        "ppo",
        "learned",
        rows_a,
        {"population": "sealed-v1", "budget": "B1", "selector": "S1"},
    )
    reference = EvaluationArm(
        "supervised",
        "baseline",
        rows_b,
        {"population": "sealed-v1", "budget": "B1", "selector": "S1"},
    )
    spec = ComparisonSpec(
        "ppo_increment",
        treatment="ppo",
        reference="supervised",
        family=ComparisonFamily.PPO_INCREMENT,
        matched_metadata=("population", "budget", "selector"),
        bootstrap=32,
    )

    result = compare_arms({"ppo": treatment, "supervised": reference}, spec)
    assert result.endpoint.delta_utility == pytest.approx(0.2)
    assert result.spec.family is ComparisonFamily.PPO_INCREMENT

    mismatched = EvaluationArm(
        "supervised",
        "baseline",
        rows_b,
        {"population": "sealed-v1", "budget": "B2", "selector": "S1"},
    )
    with pytest.raises(EvaluationProtocolError, match="budget"):
        compare_arms({"ppo": treatment, "supervised": mismatched}, spec)


def test_receipt_jsonl_round_trip_is_strict_and_lossless(tmp_path) -> None:
    path = tmp_path / "episodes.jsonl"
    rows = [_outcome("i", "L", 0, 0.6)]

    digest = write_episode_receipts(path, rows)
    loaded = read_episode_receipts(path, expected_sha256=digest)

    assert loaded == rows
    with pytest.raises(FileExistsError):
        write_episode_receipts(path, rows)
    with pytest.raises(EvaluationProtocolError, match="SHA-256"):
        read_episode_receipts(path, expected_sha256="0" * 64)


def test_receipt_parser_rejects_coercible_but_wrong_json_scalar_types(tmp_path) -> None:
    import json

    row = _outcome("i", "L", 0, 0.6).as_dict()
    row["returned_valid"] = "false"
    path = tmp_path / "bad-bool.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(EvaluationProtocolError, match="returned_valid"):
        read_episode_receipts(path)

    row = _outcome("i", "L", 0, 0.6).as_dict()
    row["work"]["decisions"] = 3.5
    path = tmp_path / "bad-work.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(EvaluationProtocolError, match="work"):
        read_episode_receipts(path)

    row = _outcome("i", "L", 0, 0.6).as_dict()
    row["fixed_program_r99"] = 999
    path = tmp_path / "bad-r99.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(EvaluationProtocolError, match="R99"):
        read_episode_receipts(path)

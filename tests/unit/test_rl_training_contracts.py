"""Normative safety contracts that must hold before a scientific training run."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context, InitFailureRecord, Opcode, WorkVector
from isingfold.rl.data.lineage import Split, check_no_leakage
from isingfold.rl.data.inkdrop import InkDropError, ink_drop
from isingfold.rl.data.quality import (
    ActionQuality,
    QualityRecord,
    label_decision,
)
from isingfold.rl.data.structural import ContinuationDomain, exact_feasibility
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluate import EpisodeOutcome, paired_endpoint
from isingfold.rl.evaluator import sample_program
from isingfold.rl.model import IFCore
from isingfold.rl.ppo import PPOConfig, collect
from isingfold.rl.program import compile_program
from isingfold.rl.proposal import LEGACY_ONLINE_INITIALIZER_RESTARTS_V1
from isingfold.rl.rollout import RolloutBuffer, Transition
from isingfold.rl.strength import StrengthSelectorModel
from tests.unit.quality_receipt_support import fake_continuation_result


def _tiny() -> tuple[EmbeddingTask, dict[int, frozenset[int]]]:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts(
        {0: 0.25, 1: -0.5},
        {(0, 1): -1.25},
    )
    chains = {0: frozenset({0, 1}), 1: frozenset({2, 3})}
    task = EmbeddingTask("tiny", logical, host, problem, -1.5, "lineage-0")
    return task, chains


def _improvement_env() -> EmbeddingEnv:
    task, chains = _tiny()
    return EmbeddingEnv(
        task,
        Context(qubit_cap=4),
        initializer=lambda *_: chains,
        selector=fixed_strength_selector(),
        reward_reads=4,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=7,
    )


def test_context_is_deeply_immutable_and_terminal_reserve_fits_caps() -> None:
    with pytest.raises(ValueError, match="reserve"):
        Context(
            qubit_cap=4,
            caps=WorkVector(decisions=0),
            reserve=WorkVector(decisions=1),
        )

    ctx = Context(qubit_cap=4)
    with pytest.raises(TypeError):
        ctx.quotas["single"] = 999


def test_commit_returns_the_exact_selected_program_for_deployment() -> None:
    env = _improvement_env()
    decision = env.reset()
    index = next(
        i for i, candidate in enumerate(decision.candidates) if candidate.opcode is Opcode.COMMIT
    )
    terminal = env.step(
        decision,
        index,
        evaluate_training_reward=False,
    ).next_decision_or_terminal

    assert "selected_program" in {item.name for item in fields(type(terminal))}
    assert terminal.selected_program is not None
    assert terminal.selected_program.strength == terminal.selected_strength
    assert terminal.training_reward is None
    assert terminal.training_cost is None


def test_commit_uses_the_frozen_graph_strength_selector_boundary() -> None:
    task, chains = _tiny()
    selector = StrengthSelectorModel()
    selector._graph_fitted.fill_(True)
    selector.freeze()
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=4),
        initializer=lambda *_: chains,
        selector=selector,
        reward_reads=4,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=5,
    )
    decision = env.reset()
    index = next(
        i for i, candidate in enumerate(decision.candidates) if candidate.opcode is Opcode.COMMIT
    )

    terminal = env.step(
        decision,
        index,
        evaluate_training_reward=False,
    ).next_decision_or_terminal

    assert terminal.selected_program is not None
    assert terminal.selected_index in range(4)


def test_evaluator_rejects_a_short_sampler_read_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from dwave.samplers import SimulatedAnnealingSampler

    task, chains = _tiny()
    program = compile_program(chains, task.host, task.problem, 1.0, 0)
    short = SimpleNamespace(
        variables=[0, 1, 2, 3],
        record=SimpleNamespace(sample=np.ones((1, 4), dtype=np.int8)),
    )
    monkeypatch.setattr(SimulatedAnnealingSampler, "sample", lambda self, *args, **kwargs: short)

    with pytest.raises(ValueError, match="read"):
        sample_program(
            program,
            chains,
            task.problem,
            task.ground_energy,
            num_reads=4,
            seed=1,
        )


def test_quality_counterfactuals_use_distinct_recorded_rng_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import isingfold.rl.data.quality as quality

    task, chains = _tiny()
    seen: list[int] = []

    def fake_continuation(*args, **kwargs):
        seen.append(kwargs["continuation_seed"])
        return fake_continuation_result(
            continuation_seed=kwargs["continuation_seed"],
            reward=0.5,
            reward_reads=kwargs["reward_reads"],
            prefix=kwargs["prefix"],
            chains=chains,
            selected_index=1,
            num_sweeps=args[1].num_sweeps,
        )

    monkeypatch.setattr(quality, "run_continuation", fake_continuation)
    record = label_decision(
        task,
        Context(qubit_cap=4),
        initializer=lambda *_: chains,
        selector=fixed_strength_selector(),
        prefix=(),
        seed=11,
        provenance_fingerprint="f" * 64,
        evaluated_actions=2,
        continuations=3,
        reward_reads=4,
    )

    assert record is not None
    assert len(seen) == 6
    assert len(set(seen)) == 6
    assert tuple(
        seed for action in record.evaluated for seed in action.continuation_seeds
    ) == tuple(seen)


def test_quality_small_samples_remain_inconclusive_instead_of_manufacturing_a_winner() -> None:
    task, chains = _tiny()
    del task
    positive_receipts = tuple(
        fake_continuation_result(
            continuation_seed=seed,
            reward=1.0,
            reward_reads=2,
            prefix=(0,),
            chains=chains,
        ).receipt
        for seed in (101, 102)
    )
    negative_receipts = tuple(
        fake_continuation_result(
            continuation_seed=seed,
            reward=0.0,
            reward_reads=2,
            prefix=(1,),
            chains=None,
            returned_valid=False,
        ).receipt
        for seed in (201, 202)
    )
    record = QualityRecord(
        instance="tiny",
        lineage="lineage-0",
        prefix=(),
        support=(),
        evaluated=(
            ActionQuality(
                action_index=0,
                payload_key="a",
                opcode="REWRITE_ONE",
                q_mu=1.0,
                continuations=2,
                valid_returns=2,
                inclusion_probability=1.0,
                selected_payload_digest="1" * 64,
                applied_action_record_digest="2" * 64,
                continuation_seeds=(101, 102),
                continuation_rewards=(1.0, 1.0),
                continuation_valid=(True, True),
                continuation_receipts=positive_receipts,
            ),
            ActionQuality(
                action_index=1,
                payload_key="b",
                opcode="REWRITE_ONE",
                q_mu=0.0,
                continuations=2,
                valid_returns=0,
                inclusion_probability=1.0,
                selected_payload_digest="3" * 64,
                applied_action_record_digest="4" * 64,
                continuation_seeds=(201, 202),
                continuation_rewards=(0.0, 0.0),
                continuation_valid=(False, False),
                continuation_receipts=negative_receipts,
            ),
        ),
        continuation_policy="uniform-exact-legal-support-v1",
        action_envelope_record_digest="5" * 64,
    )

    # With only two bounded continuation outcomes per action, simultaneous 95% Hoeffding
    # intervals overlap even at the most extreme empirical means.
    assert record.best_actions() == (0, 1)


def test_quality_counterfactual_end_to_end_can_fork_future_rng_without_staling_support() -> None:
    task, chains = _tiny()
    record = label_decision(
        task,
        Context(qubit_cap=4),
        initializer=lambda *_: chains,
        selector=fixed_strength_selector(),
        prefix=(),
        seed=19,
        provenance_fingerprint="f" * 64,
        evaluated_actions=2,
        continuations=1,
        reward_reads=4,
    )

    assert record is not None
    assert len(record.evaluated) == 2
    assert len(record.evaluated[0].continuation_rewards) == 1


def test_quality_record_keeps_complete_bound_support(monkeypatch: pytest.MonkeyPatch) -> None:
    import isingfold.rl.data.quality as quality

    task, chains = _tiny()
    monkeypatch.setattr(
        quality,
        "run_continuation",
        lambda *args, **kwargs: fake_continuation_result(
            continuation_seed=kwargs["continuation_seed"],
            reward=0.5,
            reward_reads=kwargs["reward_reads"],
            prefix=kwargs["prefix"],
            chains=chains,
            selected_index=1,
            num_sweeps=args[1].num_sweeps,
        ),
    )
    record = label_decision(
        task,
        Context(qubit_cap=4),
        initializer=lambda *_: chains,
        selector=fixed_strength_selector(),
        prefix=(),
        seed=3,
        provenance_fingerprint="f" * 64,
        evaluated_actions=2,
        continuations=1,
        reward_reads=4,
    )

    required = {
        "old_chains",
        "new_chains",
        "routes",
        "work",
        "payload_key",
        "legal",
    }
    assert record is not None
    assert required <= set(record.support[0])
    assert record.state_fingerprint
    assert record.support_fingerprint
    assert record.context_version == "rev2-pilot-3-restart-cache"
    assert len(record.action_envelope_record_digest) == 64
    assert len(record.evaluated[0].selected_payload_digest) == 64
    if record.evaluated[0].opcode in {"COMMIT", "STOP"}:
        assert record.evaluated[0].applied_action_record_digest is None
    else:
        assert len(record.evaluated[0].applied_action_record_digest or "") == 64


def test_quality_labelling_never_silently_drops_a_registered_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import isingfold.rl.data.quality as quality

    task, chains = _tiny()
    monkeypatch.setattr(quality, "run_continuation", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="denominator"):
        label_decision(
            task,
            Context(qubit_cap=4),
            initializer=lambda *_: chains,
            selector=fixed_strength_selector(),
            prefix=(),
            seed=3,
            provenance_fingerprint="f" * 64,
            evaluated_actions=2,
            continuations=1,
            reward_reads=4,
        )


def test_quality_prefix_rejects_negative_action_indices() -> None:
    task, chains = _tiny()
    assert (
        label_decision(
            task,
            Context(qubit_cap=4),
            initializer=lambda *_: chains,
            selector=fixed_strength_selector(),
            prefix=(-1,),
            seed=3,
            provenance_fingerprint="f" * 64,
            evaluated_actions=1,
            continuations=1,
            reward_reads=4,
        )
        is None
    )


def test_collection_aborts_instead_of_returning_a_short_behavior_batch() -> None:
    class AlwaysFails:
        def reset(self):
            return InitFailureRecord("failed", WorkVector(restart_work=1))

    config = PPOConfig(episodes_per_batch=2, init_attempt_cap=2)
    with pytest.raises(RuntimeError, match="initial|fill|batch"):
        collect(lambda _seed, _episode_index: AlwaysFails(), IFCore(), config)


def test_rollout_rejects_an_incomplete_live_episode() -> None:
    buffer = RolloutBuffer()
    episode = buffer.add_episode()
    buffer.add(
        Transition(
            observation=None,
            legal_mask=np.array([True]),
            chosen_index=0,
            old_log_prob=0.0,
            old_log_probs=np.array([0.0]),
            old_utility=0.1,
            old_failure=0.0,
            reward=0.0,
            cost=0.0,
            terminated=False,
            episode=episode.index,
        )
    )
    with pytest.raises(RuntimeError, match="complete|terminal"):
        buffer.compute_targets()


def test_split_overlap_is_reported_as_leakage() -> None:
    split = Split(train=("root",), validation=(), test=("root",))
    report = check_no_leakage(split, [{"lineage_root": "root"}])
    assert not report["ok"]
    assert report["conflicts"] == {"root": ["test", "train"]}


def test_cli_partition_selection_fails_closed() -> None:
    from isingfold.rl.cli import _select_tasks

    task, _ = _tiny()
    with pytest.raises(ValueError, match="missing"):
        _select_tasks([task], {"train": [task.lineage]}, "test")
    with pytest.raises(ValueError, match="non-empty"):
        _select_tasks([task], {"train": [task.lineage], "test": []}, "test")
    with pytest.raises(ValueError, match="leakage"):
        _select_tasks(
            [task],
            {"train": [task.lineage], "validation": [task.lineage], "test": ["other"]},
            "train",
        )


def test_inkdrop_requested_cell_and_budget_are_binding() -> None:
    with pytest.raises(InkDropError, match="budget"):
        ink_drop(nx.path_graph(8), 2, chain_size=2, budget=1)
    with pytest.raises(InkDropError, match="density"):
        ink_drop(nx.path_graph(8), 2, chain_size=2, density=1.1)


def test_structural_solver_rejects_an_invalid_frozen_exterior() -> None:
    logical = nx.empty_graph(2)
    label = exact_feasibility(
        {0: frozenset({0}), 1: frozenset({0})},
        logical,
        nx.path_graph(2),
        ContinuationDomain((), frozenset({0, 1})),
    )
    assert label.feasible is False
    assert "frozen" in label.reason


def test_structural_positive_witness_is_a_complete_embedding() -> None:
    logical = nx.path_graph(2)
    label = exact_feasibility(
        {0: frozenset({0}), 1: frozenset()},
        logical,
        nx.path_graph(3),
        ContinuationDomain((1,), frozenset({1, 2}), max_chain=1),
    )
    assert label.feasible is True
    assert label.witness is not None and set(label.witness) == set(logical.nodes)


def test_paired_endpoint_summaries_exclude_unpaired_rows() -> None:
    def outcome(name: str, lineage: str, utility: float) -> EpisodeOutcome:
        return EpisodeOutcome(name, lineage, True, utility, 1, 1, 1, None, "COMMIT")

    endpoint = paired_endpoint(
        [outcome("paired", "A", 1.0), outcome("extra", "B", 0.0)],
        [outcome("paired", "A", 0.0)],
        bootstrap=32,
        seed=0,
    )
    assert endpoint.n_lineages == 1
    assert endpoint.mean_treatment == 1.0
    assert endpoint.mean_reference == 0.0
    assert endpoint.r99_treatment is None
    assert endpoint.r99_reference is None

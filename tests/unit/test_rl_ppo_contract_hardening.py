"""Fail-closed PPO mode, collection and exact-replay contracts."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    Candidate,
    Context,
    Mode,
    Opcode,
    WorkVector,
    candidate_support_key,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, fixed_strength_selector
from isingfold.rl.model import IFCore
from isingfold.rl.ppo import PPOConfig, PPOTrainer, collect
from isingfold.rl.proposal import LEGACY_ONLINE_INITIALIZER_RESTARTS_V1
from isingfold.rl.rollout import snapshot_candidates

torch = pytest.importorskip("torch")


def test_cached_restart_snapshot_preserves_complete_support_fingerprint() -> None:
    restart_cache_after_digest = "a" * 64
    restart = Candidate(
        opcode=Opcode.RESTART,
        affected=(0,),
        old_chains={0: frozenset({0})},
        new_chains={0: frozenset({1})},
        work=WorkVector(decisions=1),
        payload_key="cached-restart-successor",
        proposal_work=WorkVector(materializations=1),
        restart_cache_slot=0,
        restart_cache_after_digest=restart_cache_after_digest,
    )

    snapshotted = snapshot_candidates((restart,))

    assert snapshotted[0].restart_cache_slot == 0
    assert snapshotted[0].restart_cache_after_digest == restart_cache_after_digest
    assert candidate_support_key(snapshotted, (True,)) == candidate_support_key(
        (restart,), (True,)
    )


def _task_and_chains() -> tuple[EmbeddingTask, dict[int, frozenset[int]]]:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts({0: 0.25, 1: -0.5}, {(0, 1): -1.25})
    chains = {0: frozenset({0, 1}), 1: frozenset({2, 3})}
    return EmbeddingTask("ppo-contract", logical, host, problem, -1.5, "lineage-ppo"), chains


def _trainer_and_buffer(*, greedy: bool = False):
    task, chains = _task_and_chains()
    context = Context(qubit_cap=4)

    def factory(seed: int, _episode_index: int) -> EmbeddingEnv:
        return EmbeddingEnv(
            task,
            context,
            mode=Mode.IMPROVEMENT,
            initializer=lambda *_: chains,
            selector=fixed_strength_selector(),
            reward_reads=2,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
            seed=seed,
        )

    torch.manual_seed(3)
    model = IFCore(improvement_mode=True)
    config = PPOConfig(episodes_per_batch=1, epochs=1, minibatch=128, seed=5)
    trainer = PPOTrainer(model, config, mode=Mode.IMPROVEMENT, total_updates=1)
    buffer, _ = collect(factory, trainer.behaviour_snapshot(), config, greedy=greedy)
    assert buffer.transitions
    return trainer, buffer


@pytest.mark.parametrize(
    ("improvement_mode", "trainer_mode"),
    ((True, Mode.CONSTRUCTION), (False, Mode.IMPROVEMENT)),
)
def test_trainer_rejects_both_model_mode_mismatch_directions_before_optimizer(
    improvement_mode: bool,
    trainer_mode: Mode,
) -> None:
    model = IFCore(improvement_mode=improvement_mode)
    with pytest.raises(ValueError, match="mode"):
        PPOTrainer(model, PPOConfig(episodes_per_batch=1), mode=trainer_mode, total_updates=1)


def test_matching_construction_trainer_keeps_failure_head_trainable() -> None:
    model = IFCore(improvement_mode=False)
    trainer = PPOTrainer(
        model,
        PPOConfig(episodes_per_batch=1),
        mode=Mode.CONSTRUCTION,
        total_updates=1,
    )

    failure_parameters = list(model.failure.parameters())
    optimized = {
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    }
    assert failure_parameters
    assert all(parameter.requires_grad for parameter in failure_parameters)
    assert all(id(parameter) in optimized for parameter in failure_parameters)


def test_collection_rejects_environment_model_mode_mismatch_before_reset() -> None:
    class WrongModeEnvironment:
        mode = Mode.CONSTRUCTION

        def reset(self):
            raise AssertionError("mode must be checked before reset")

    with pytest.raises(ValueError, match="mode"):
        collect(
            lambda _seed, _episode_index: WrongModeEnvironment(),
            IFCore(improvement_mode=True),
            PPOConfig(episodes_per_batch=1),
        )


def test_greedy_rollout_is_marked_evaluation_only_and_rejected_by_ppo() -> None:
    trainer, buffer = _trainer_and_buffer(greedy=True)
    calls = 0

    def step(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    trainer.optimizer.step = step
    assert buffer.metadata["training_eligible"] is False
    with pytest.raises(RuntimeError, match="greedy|evaluation-only"):
        trainer.update(buffer)
    assert calls == 0


@pytest.mark.parametrize(
    "tamper",
    (
        "payload_key",
        "support_fingerprint",
        "state_fingerprint",
        "context_version",
        "charged_work_receipt",
        "candidate_application_work",
        "candidate_proposal_work",
        "relational_observation",
        "receipt_version",
    ),
)
def test_update_authenticates_exact_saved_support_before_optimizer(tamper: str) -> None:
    trainer, buffer = _trainer_and_buffer()
    transition = buffer.transitions[0]
    assert transition.replay_receipt is not None
    calls = 0

    def step(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    trainer.optimizer.step = step
    if tamper == "payload_key":
        transition.payload_key = "forged-selected-payload"
    elif tamper == "support_fingerprint":
        transition.support_fingerprint = "0" * 64
    elif tamper == "state_fingerprint":
        transition.state_fingerprint = "1" * 64
    elif tamper == "context_version":
        transition.context_version = "forged-context"
    elif tamper == "charged_work_receipt":
        transition.charged_work_receipt = {
            **transition.charged_work_receipt,
            "feature_work": transition.charged_work_receipt["feature_work"] + 1,
        }
    elif tamper in {"candidate_application_work", "candidate_proposal_work"}:
        candidates = list(transition.candidates)
        candidate = candidates[transition.chosen_index]
        field = "work" if tamper == "candidate_application_work" else "proposal_work"
        current = getattr(candidate, field)
        candidates[transition.chosen_index] = replace(
            candidate,
            **{field: replace(current, materializations=current.materializations + 1)},
        )
        transition.candidates = tuple(candidates)
    elif tamper == "relational_observation":
        transition.observation.index_factor_action = (
            transition.observation.index_factor_action.copy()
        )
        if transition.observation.index_factor_action.size:
            transition.observation.index_factor_action[1, 0] = (
                transition.observation.index_factor_action[1, 0] + 1
            ) % transition.observation.n_actions
        else:
            transition.observation.actions = transition.observation.actions.copy()
            transition.observation.actions[0, 0] += np.float32(1.0)
    elif tamper == "receipt_version":
        transition.replay_receipt = replace(
            transition.replay_receipt,
            schema_version=transition.replay_receipt.schema_version + 1,
        )
    else:  # pragma: no cover - keeps the parametrization exhaustive
        raise AssertionError(tamper)

    with pytest.raises(RuntimeError, match="replay|receipt|support|payload|observation|work"):
        trainer.update(buffer)
    assert calls == 0


def test_old_unversioned_rollout_cannot_be_used_for_training() -> None:
    trainer, buffer = _trainer_and_buffer()
    buffer.metadata.pop("replay_receipt_schema_version")

    with pytest.raises(RuntimeError, match="version|schema"):
        trainer.update(buffer)


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_schema",
        "wrong_schema_version",
        "wrong_schedule",
        "wrong_training_seed",
        "wrong_update_index",
        "wrong_start",
        "wrong_stop",
        "missing_episode_indices",
        "wrong_episode_indices",
    ),
)
def test_update_rejects_tampered_collection_schedule_before_optimizer(tamper: str) -> None:
    trainer, buffer = _trainer_and_buffer()
    calls = 0

    def step(*_args, **_kwargs):
        nonlocal calls
        calls += 1

    trainer.optimizer.step = step
    if tamper == "missing_schema":
        buffer.metadata.pop("collection_schedule_schema")
    elif tamper == "wrong_schema_version":
        buffer.metadata["collection_schedule_schema_version"] = 2
    elif tamper == "wrong_schedule":
        buffer.metadata["collection_schedule"] = "sequential-legacy"
    elif tamper == "wrong_training_seed":
        buffer.metadata["collection_training_seed"] = trainer.config.seed + 1
    elif tamper == "wrong_update_index":
        buffer.metadata["collection_update_index"] = trainer.updates_done + 1
    elif tamper == "wrong_start":
        buffer.metadata["episode_schedule_start"] = 1
    elif tamper == "wrong_stop":
        buffer.metadata["episode_schedule_stop_exclusive"] = 2
    elif tamper == "missing_episode_indices":
        buffer.metadata.pop("episode_schedule_indices", None)
    elif tamper == "wrong_episode_indices":
        buffer.metadata["episode_schedule_indices"] = [1]
    else:  # pragma: no cover - keeps the parametrization exhaustive
        raise AssertionError(tamper)

    with pytest.raises(RuntimeError, match="collection schedule"):
        trainer.update(buffer)
    assert calls == 0


def test_likelihood_replay_rejects_collection_schedule_before_model_forward(monkeypatch) -> None:
    trainer, buffer = _trainer_and_buffer()
    buffer.metadata["episode_schedule_indices"] = [9]

    def forbidden_forward(*_args, **_kwargs):
        raise AssertionError("schedule authentication must precede replay inference")

    monkeypatch.setattr(trainer, "_forward_many", forbidden_forward)
    with pytest.raises(RuntimeError, match="collection schedule"):
        trainer.replay_check(buffer)


def test_update_uses_saved_payloads_without_rerunning_randomized_proposals(monkeypatch) -> None:
    from isingfold.rl.proposal import ProposalGenerator

    trainer, buffer = _trainer_and_buffer()
    receipt = buffer.transitions[0].replay_receipt
    assert receipt is not None
    assert {
        "opcode",
        "affected",
        "old_chains",
        "new_chains",
        "routes",
        "proposal_work",
        "application_work_bound",
        "restart_semantics",
    } <= set(receipt.selected_candidate_payload)

    def forbidden_generate(*_args, **_kwargs):
        raise AssertionError("PPO replay must not invoke the proposal generator")

    monkeypatch.setattr(ProposalGenerator, "generate", forbidden_generate)
    logs = trainer.update(buffer)
    assert logs["transitions"] == buffer.n_transitions

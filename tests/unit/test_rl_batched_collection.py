"""Adversarial contracts for synchronous batched PPO rollout collection."""

from __future__ import annotations

import copy
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
import torch

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    Context,
    DecisionState,
    InitFailureRecord,
    Mode,
    StepResult,
    TerminalReason,
    TerminalRecord,
    WorkVector,
    candidate_support_key,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, fixed_strength_selector
from isingfold.rl.model import PADDED_ACTIONS, IFCore, ModelOutput
from isingfold.rl.ppo import (
    COLLECTION_SCHEDULE,
    COLLECTION_SCHEDULE_SCHEMA,
    COLLECTION_SCHEDULE_SCHEMA_VERSION,
    PPOConfig,
    collect,
)
from isingfold.rl.proposal import LEGACY_ONLINE_INITIALIZER_RESTARTS_V1


def _decision(*, variables: int = 2) -> DecisionState:
    logical = nx.path_graph(variables)
    host = nx.path_graph(2 * variables)
    problem = LogicalProblem.from_dicts(
        {node: 0.0 for node in logical},
        {(u, v): -1.0 for u, v in logical.edges},
    )
    chains = {node: frozenset((2 * node, 2 * node + 1)) for node in logical}
    task = EmbeddingTask(
        f"batch-{variables}",
        logical,
        host,
        problem,
        -float(logical.number_of_edges()),
        f"lineage-{variables}",
    )
    result = EmbeddingEnv(
        task,
        Context(qubit_cap=2 * variables),
        mode=Mode.IMPROVEMENT,
        initializer=lambda *_: chains,
        selector=fixed_strength_selector(),
        reward_reads=2,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=17,
    ).reset()
    assert isinstance(result, DecisionState)
    return result


def _terminal(*, reward: float = 1.0) -> TerminalRecord:
    return TerminalRecord(
        returned_valid=True,
        terminal_reason=TerminalReason.COMMIT,
        embedding={},
        selected_strength=1.0,
        selected_index=0,
        validation_receipt={"valid": True},
        cumulative_work=WorkVector(),
        training_reward=reward,
        training_cost=0.0,
    )


class _ScriptedEnvironment:
    mode = Mode.IMPROVEMENT

    def __init__(self, decisions: list[DecisionState], *, reward: float = 1.0) -> None:
        self._decisions = decisions
        self._position = 0
        self._terminal = _terminal(reward=reward)

    def reset(self) -> DecisionState | TerminalRecord:
        if not self._decisions:
            return self._terminal
        return self._decisions[0]

    def step(self, decision: DecisionState, index: int) -> StepResult:
        assert decision is self._decisions[self._position]
        assert 0 <= index < len(decision.candidates)
        self._position += 1
        terminal = self._position == len(self._decisions)
        successor = self._terminal if terminal else self._decisions[self._position]
        return StepResult(
            successor,
            self._terminal.training_reward if terminal else 0.0,
            0.0,
            terminal,
            terminal_reason=TerminalReason.COMMIT if terminal else None,
        )


class _RecordingIFCore(IFCore):
    def __init__(self) -> None:
        super().__init__(improvement_mode=True)
        self.batch_sizes: list[int] = []

    def forward_single(self, *_args, **_kwargs) -> ModelOutput:
        raise AssertionError("production collection must not invoke IFCore.forward_single")

    def forward(self, observation_batch, *args, **kwargs) -> ModelOutput:
        self.batch_sizes.append(len(observation_batch))
        return super().forward(observation_batch, *args, **kwargs)


class _DeterministicIFCore(IFCore):
    """Cheap IFCore-shaped policy used to inspect row/support plumbing."""

    def __init__(self) -> None:
        super().__init__(improvement_mode=True)
        self.batch_sizes: list[int] = []

    def forward_single(self, *_args, **_kwargs) -> ModelOutput:
        raise AssertionError("IFCore-shaped policies must use the batched interface")

    def forward(
        self,
        observation_batch,
        candidate_batch=None,
        legal_mask=None,
        device=None,
    ) -> ModelOutput:
        del candidate_batch
        observations = list(observation_batch)
        self.batch_sizes.append(len(observations))
        target_device = device or next(self.parameters()).device
        padded = torch.full(
            (len(observations), PADDED_ACTIONS),
            float("-inf"),
            device=target_device,
        )
        counts: list[int] = []
        for row, observation in enumerate(observations):
            mask = np.asarray(
                legal_mask[row] if legal_mask is not None else observation.legal_mask,
                dtype=bool,
            )
            count = int(observation.actions.shape[0])
            counts.append(count)
            scores = torch.arange(count, dtype=torch.float32, device=target_device)
            legal = torch.as_tensor(mask, dtype=torch.bool, device=target_device)
            padded[row, :count] = torch.where(
                legal,
                scores - torch.logsumexp(scores[legal], dim=0),
                torch.full_like(scores, float("-inf")),
            )
        zeros = torch.zeros(len(observations), device=target_device)
        return ModelOutput(
            masked_log_probs=padded,
            utility_value=zeros,
            failure_logit=zeros,
            failure_value=zeros,
            action_count=torch.as_tensor(counts, dtype=torch.long, device=target_device),
        )


class _ForwardSingleOnly(torch.nn.Module):
    improvement_mode = True

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward_single(self, observation, device=None) -> ModelOutput:
        self.calls += 1
        target_device = device or self.anchor.device
        legal = torch.as_tensor(observation.legal_mask, dtype=torch.bool, device=target_device)
        log_probs = torch.full(legal.shape, float("-inf"), device=target_device)
        log_probs[legal] = -torch.log(legal.sum().to(torch.float32))
        zero = self.anchor.to(target_device) * 0.0
        return ModelOutput(
            masked_log_probs=log_probs,
            utility_value=zero,
            failure_logit=zero,
            failure_value=zero,
            action_count=legal.sum(),
        )


def test_production_ifcore_collection_batches_every_active_wave() -> None:
    decision = _decision()
    model = _RecordingIFCore().eval()
    environments = [
        _ScriptedEnvironment([copy.deepcopy(decision), copy.deepcopy(decision)]) for _ in range(3)
    ]

    buffer, _ = collect(
        lambda _seed, episode_index: environments[episode_index],
        model,
        PPOConfig(episodes_per_batch=3, seed=11),
        greedy=True,
    )

    assert model.batch_sizes == [3, 3]
    assert buffer.n_episodes == 3
    assert buffer.n_transitions == 6
    assert [len(episode.transitions) for episode in buffer.episodes] == [2, 2, 2]


def test_batched_rows_bind_their_own_variable_sized_action_support() -> None:
    small = _decision(variables=2)
    large = _decision(variables=3)
    assert len(small.candidates) != len(large.candidates)
    model = _DeterministicIFCore().eval()

    buffer, _ = collect(
        lambda _seed, episode_index: _ScriptedEnvironment([[small], [large]][episode_index]),
        model,
        PPOConfig(episodes_per_batch=2, seed=13),
        greedy=True,
    )

    assert model.batch_sizes == [2]
    transitions = [buffer.transitions[episode.transitions[0]] for episode in buffer.episodes]
    assert [row.old_log_probs.shape for row in transitions] == [
        (len(small.candidates),),
        (len(large.candidates),),
    ]
    assert [row.chosen_index for row in transitions] == [
        len(small.candidates) - 1,
        len(large.candidates) - 1,
    ]
    assert [row.payload_key for row in transitions] == [
        small.candidates[-1].payload_key,
        large.candidates[-1].payload_key,
    ]


def test_per_episode_rng_is_reproducible_and_independent_of_peer_horizon() -> None:
    decision = _decision()

    def run(first_horizon: int):
        seeds: dict[int, int] = {}
        model = _DeterministicIFCore().eval()

        def factory(seed: int, episode_index: int) -> _ScriptedEnvironment:
            seeds[episode_index] = seed
            horizon = first_horizon if episode_index == 0 else 4
            return _ScriptedEnvironment([copy.deepcopy(decision) for _ in range(horizon)])

        buffer, _ = collect(
            factory,
            model,
            PPOConfig(episodes_per_batch=2, seed=97),
        )
        second = buffer.episodes[1]
        rows = [buffer.transitions[index] for index in second.transitions]
        return (
            seeds[1],
            [row.chosen_index for row in rows],
            [row.replay_receipt.record_digest for row in rows],
            buffer.metadata,
        )

    short_peer = run(1)
    long_peer = run(5)
    repeated = run(1)

    assert short_peer == repeated
    assert short_peer[:3] == long_peer[:3]
    assert short_peer[3]["collection_schedule_schema"] == COLLECTION_SCHEDULE_SCHEMA
    assert short_peer[3]["collection_schedule_schema_version"] == COLLECTION_SCHEDULE_SCHEMA_VERSION
    assert short_peer[3]["collection_schedule"] == COLLECTION_SCHEDULE


def test_scalar_double_fallback_handles_retry_singleton_and_zero_decision_terminal() -> None:
    decision = _decision()
    only_commit = next(
        index
        for index, candidate in enumerate(decision.candidates)
        if candidate.opcode.value == "COMMIT"
    )
    mask = np.zeros(len(decision.candidates), dtype=bool)
    mask[only_commit] = True
    observation = copy.deepcopy(decision.observation)
    observation.legal_mask = mask.copy()
    singleton = replace(
        decision,
        observation=observation,
        legal_mask=tuple(bool(value) for value in mask),
        support_fingerprint=candidate_support_key(decision.candidates, mask),
    )
    attempts = {0: 0, 1: 0}

    class _Failure:
        def reset(self) -> InitFailureRecord:
            return InitFailureRecord("retry", WorkVector(restart_work=1))

    def factory(_seed: int, episode_index: int):
        attempts[episode_index] += 1
        if episode_index == 0 and attempts[episode_index] == 1:
            return _Failure()
        if episode_index == 0:
            return _ScriptedEnvironment([singleton])
        return _ScriptedEnvironment([])

    model = _ForwardSingleOnly()
    buffer, stats = collect(
        factory,
        model,
        PPOConfig(episodes_per_batch=2, init_attempt_cap=2, seed=101),
    )

    assert attempts == {0: 2, 1: 1}
    assert stats["init_failures"] == 1.0
    assert model.calls == 1
    assert buffer.n_episodes == 2
    assert buffer.n_transitions == 1
    assert buffer.transitions[0].chosen_index == only_commit
    assert buffer.transitions[0].old_log_prob == pytest.approx(0.0)
    assert [len(episode.transitions) for episode in buffer.episodes] == [1, 0]


def test_initializer_attempt_cap_is_exact() -> None:
    attempts = 0

    class _Failure:
        def reset(self) -> InitFailureRecord:
            return InitFailureRecord("retry", WorkVector(restart_work=1))

    def factory(_seed: int, _episode_index: int) -> _Failure:
        nonlocal attempts
        attempts += 1
        return _Failure()

    with pytest.raises(RuntimeError, match="initializer-attempt cap"):
        collect(
            factory,
            _ForwardSingleOnly(),
            PPOConfig(episodes_per_batch=1, init_attempt_cap=3),
        )

    assert attempts == 3

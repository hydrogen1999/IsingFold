"""Tensor schema, IF-Core forward contracts, PPO bookkeeping and the endpoints."""

from __future__ import annotations

import itertools
import random

import networkx as nx
import numpy as np
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context, Opcode
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask
from isingfold.rl.evaluate import (
    EpisodeOutcome,
    first_commit_controller,
    paired_endpoint,
    run_controller,
    secondary_metrics,
)
from isingfold.rl.proposal import (
    LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    router_initializer,
)
from isingfold.rl.rollout import RolloutBuffer, Transition
from isingfold.rl.strength import StrengthRecord, fit_selector, selector_regret
from isingfold.rl.tensorize import N_ACTION, N_GLOBAL, N_LOGICAL, pack

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def task() -> EmbeddingTask:
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 6))
    logical = nx.gnm_random_graph(5, 7, seed=4)
    rng = random.Random(1)
    h = {v: rng.choice([-1.0, 1.0]) for v in logical.nodes()}
    j = {(u, v): rng.choice([-1.0, 1.0]) for u, v in logical.edges()}
    problem = LogicalProblem.from_dicts(h, j)
    nodes = sorted(logical.nodes())
    ground = min(
        sum(h[i] * s[i] for i in nodes) + sum(w * s[u] * s[v] for (u, v), w in j.items())
        for s in ({i: v for i, v in zip(nodes, c)} for c in itertools.product([-1, 1], repeat=len(nodes)))
    )
    return EmbeddingTask("tiny", logical, host, problem, ground)


@pytest.fixture(scope="module")
def decision(task):
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=40),
        initializer=router_initializer(),
        reward_reads=8,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=2,
    )
    return env.reset(2)


def test_value_knownness_packing():
    rows = [[1.0, float("nan"), 0.0]]
    packed = pack(rows, 3)
    assert packed.shape == (1, 6)
    assert packed[0, 1] == 0.0 and packed[0, 4] == 0.0  # missing: zero value, zero knownness
    assert packed[0, 2] == 0.0 and packed[0, 5] == 1.0  # observed zero: knownness one


def test_observation_schema_widths(decision):
    obs = decision.observation
    assert obs.logical.shape[1] == 2 * N_LOGICAL
    assert obs.globals_.shape == (1, 2 * N_GLOBAL)
    assert obs.actions.shape[1] == 2 * N_ACTION + len(Opcode)
    assert obs.legal_mask.shape[0] == obs.actions.shape[0]
    assert np.isfinite(obs.logical).all() and np.isfinite(obs.actions).all()


def test_actor_normalises_over_the_legal_support_only(decision):
    from isingfold.rl.model import IFCore

    torch.manual_seed(0)
    model = IFCore(improvement_mode=True)
    out = model.forward_single(decision.observation)
    logp = out.masked_log_probs.detach().numpy()
    legal = decision.observation.legal_mask
    assert np.isfinite(logp[legal]).all()
    assert np.allclose(np.exp(logp[legal]).sum(), 1.0, atol=1e-5)
    if (~legal).any():
        assert np.isneginf(logp[~legal]).all()
    assert out.failure_value.item() == 0.0  # protected improvement: conditional failure is zero


def test_singleton_support_has_probability_one(decision):
    from isingfold.rl.model import IFCore

    obs = decision.observation
    obs = type(obs)(**{**obs.__dict__})
    mask = np.zeros_like(obs.legal_mask)
    mask[0] = True
    obs.legal_mask = mask
    torch.manual_seed(0)
    out = IFCore(improvement_mode=True).forward_single(obs)
    logp = out.masked_log_probs.detach().numpy()
    assert logp[0] == pytest.approx(0.0, abs=1e-5)
    entropy = -float(np.exp(logp[0]) * logp[0])
    assert entropy == pytest.approx(0.0, abs=1e-5)


def test_candidate_permutation_permutes_probabilities(decision):
    from isingfold.rl.model import IFCore

    torch.manual_seed(0)
    model = IFCore(improvement_mode=True).eval()
    obs = decision.observation
    base = model.forward_single(obs).masked_log_probs.detach().numpy()

    order = np.arange(obs.actions.shape[0])[::-1].copy()
    inverse = np.argsort(order)
    permuted = type(obs)(**{**obs.__dict__})
    permuted.actions = obs.actions[order]
    permuted.legal_mask = obs.legal_mask[order]
    permuted.index_factor_action = obs.index_factor_action.copy()
    if permuted.index_factor_action.size:
        permuted.index_factor_action[1] = inverse[obs.index_factor_action[1]]
    permuted.index_route_action = obs.index_route_action.copy()
    if permuted.index_route_action.size:
        permuted.index_route_action[1] = inverse[obs.index_route_action[1]]
    permuted.index_action_archive = obs.index_action_archive.copy()
    if permuted.index_action_archive.size:
        permuted.index_action_archive[0] = inverse[obs.index_action_archive[0]]
    moved = model.forward_single(permuted).masked_log_probs.detach().numpy()
    assert np.allclose(np.sort(np.exp(base[np.isfinite(base)])), np.sort(np.exp(moved[np.isfinite(moved)])), atol=1e-5)


def test_gae_resets_between_episodes_and_bootstraps_zero():
    buffer = RolloutBuffer()
    for e in range(2):
        episode = buffer.add_episode()
        for t in range(3):
            buffer.add(
                Transition(
                    observation=None,
                    legal_mask=np.ones(2, dtype=bool),
                    chosen_index=0,
                    old_log_prob=0.0,
                    old_log_probs=np.zeros(2),
                    old_utility=0.1 * (t + 1),
                    old_failure=0.0,
                    reward=0.0 if t < 2 else 0.25,
                    cost=0.0,
                    terminated=t == 2,
                    episode=episode.index,
                )
            )
        episode.terminal_reward = 0.25
    buffer.compute_targets(0.95)
    assert buffer.target_utility is not None
    assert np.allclose(buffer.target_utility, 0.25)
    # the terminal transition bootstraps zero, so its advantage is reward minus its old value
    assert buffer.gae_utility[2] == pytest.approx(0.25 - 0.3)
    assert buffer.gae_utility[5] == pytest.approx(0.25 - 0.3)


def test_likelihood_replay_is_exact_before_any_update(task):
    from isingfold.rl.model import IFCore
    from isingfold.rl.ppo import PPOConfig, PPOTrainer, collect

    ctx = Context(qubit_cap=40)
    initializer = router_initializer()

    def factory(seed: int, _episode_index: int) -> EmbeddingEnv:
        return EmbeddingEnv(
            task,
            ctx,
            initializer=initializer,
            reward_reads=8,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
            seed=seed,
        )

    torch.manual_seed(0)
    model = IFCore(improvement_mode=True)
    config = PPOConfig(episodes_per_batch=2, epochs=1, minibatch=8, seed=0)
    trainer = PPOTrainer(model, config, total_updates=2)
    snapshot = trainer.behaviour_snapshot()
    buffer, stats = collect(factory, snapshot, config)
    assert buffer.n_transitions > 0
    assert stats["valid_return_rate"] == 1.0
    assert trainer.replay_check(buffer) < 1e-4
    logs = trainer.update(buffer)
    assert "kl" in logs and logs["transitions"] > 0


def test_strength_selector_fits_and_reports_regret():
    records = []
    for k in range(24):
        rates = [0.2 + 0.1 * k % 0.5, 0.6, 0.4, 0.1]
        features = tuple(
            {"strength": f, "qubits": 20.0, "max_chain": 3.0, "mean_contacts": 1.2,
             "strength_over_jmax": f, "scale": 1.0, "mean_chain": 2.0,
             "single_qubit_fraction": 0.1, "max_field": 1.0, "max_coupling": 1.0,
             "single_contact_fraction": 0.5, "chain_edges": 10.0}
            for f in (0.5, 1.0, 2.0, 4.0)
        )
        records.append(StrengthRecord(features, tuple(int(100 * r) for r in rates), (100,) * 4))
    model = fit_selector(records, epochs=200)
    report = selector_regret(model, records)
    assert report["regret_vs_oracle"] >= 0.0
    assert report["selected_mean"] <= report["oracle_mean"] + 1e-9


def test_paired_endpoint_and_feasibility_gate():
    treatment = [EpisodeOutcome(f"i{k}", f"l{k}", True, 0.6, 20, 3, 5, 1.0, "COMMIT") for k in range(8)]
    reference = [EpisodeOutcome(f"i{k}", f"l{k}", True, 0.5, 20, 3, 5, 1.0, "COMMIT") for k in range(8)]
    endpoint = paired_endpoint(treatment, reference, seed=0)
    assert endpoint.delta_utility == pytest.approx(0.1)
    assert endpoint.ci_low > 0 and endpoint.noninferior
    assert endpoint.r99_treatment is None and endpoint.r99_reference is None
    metrics = secondary_metrics(treatment)
    assert metrics["valid_return_rate"] == 1.0


def test_controllers_run_and_return_valid_embeddings(task):
    ctx = Context(qubit_cap=40)
    from isingfold.rl.env import fixed_strength_selector

    outcomes = run_controller(
        [task],
        ctx,
        first_commit_controller,
        initializer=router_initializer(),
        selector=fixed_strength_selector(),
        reward_reads=8,
        seed=3,
    )
    assert outcomes and outcomes[0].returned_valid and outcomes[0].reason == "COMMIT"

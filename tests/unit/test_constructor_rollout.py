"""Construction integration gates: all output decisions belong to the actor."""
from types import SimpleNamespace
import random

import networkx as nx
import numpy as np
import pytest
import torch

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Mode, Opcode
from isingfold.rl.env import EmbeddingEnv
from isingfold.rl.proposal import ProposalGenerator

import constructor_rollout as rollout


class PublicTask:
    """Evaluator-only witness access is an error even in training construction."""
    def __init__(self, triangle=False, hide_ground=False):
        self.logical = nx.complete_graph(3) if triangle else nx.path_graph(2)
        self.host = nx.cycle_graph(5) if triangle else nx.path_graph(5)
        self.problem = LogicalProblem.from_dicts(
            {v: 0.0 for v in self.logical}, {edge: -1.0 for edge in self.logical.edges()})
        self.name = "public-task"
        self.lineage = "toy"
        self.hide_ground = hide_ground

    @property
    def witness(self):
        raise AssertionError("constructor accessed a feasibility witness")

    @property
    def initial_embedding(self):
        raise AssertionError("constructor accessed an initial embedding")

    @property
    def ground_energy(self):
        if self.hide_ground:
            raise AssertionError("inference accessed evaluator ground energy")
        return -float(self.logical.number_of_edges())


class RowActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.supports = []

    def distribution_value(self, rows, temperature):
        self.supports.append(len(rows))
        return (torch.distributions.Categorical(logits=self.weight * rows[:, 0] / temperature),
                self.weight * 0)


def place(variable, root):
    return lambda candidate: (candidate.opcode is Opcode.PLACE
                              and candidate.new_chains == {variable: frozenset({root})})


def opcode(op):
    return lambda candidate: candidate.opcode is op


def provenance(prefix):
    return lambda candidate: candidate.provenance.startswith(prefix)


class ScriptFeatures:
    """Deterministic actor preferences, applied to the real environment's legal rows."""
    budget = 5

    def __init__(self, plan):
        self.plan = plan
        self.states = {}
        self.offered = {}

    def observe(self, candidate, chains, *, state, ctx, steps_left, max_steps):
        step = state.spent.decisions
        self.states[step] = dict(chains)
        self.offered.setdefault(step, []).append(candidate)
        wanted = self.plan[min(step, len(self.plan) - 1)](candidate)
        return np.asarray([1000.0 if wanted else -1000.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def forbid_completion_solver(monkeypatch):
    import minorminer

    def forbidden(*args, **kwargs):
        raise AssertionError("constructor called minorminer")

    monkeypatch.setattr(minorminer, "find_embedding", forbidden)


def run(plan, *, task=None, **kwargs):
    task = PublicTask() if task is None else task
    fc = ScriptFeatures(plan)
    actor = RowActor()
    options = {"train": False, "objective": "feasibility"}
    options.update(kwargs)
    result = rollout.episode(task, actor, fc, 1.0, options.pop("max_steps", 20),
                             np.random.default_rng(7), options.pop("deadline", 60.0), **options)
    return result, fc, actor


def grown_plan():
    return [place(0, 0), provenance("grow:"), place(1, 2), provenance("shrink:"),
            opcode(Opcode.COMMIT)]


def test_actor_constructs_grows_refines_and_commits_without_witness_or_solver():
    result, fc, actor = run(grown_plan(), train=True, shaping_coef=0.3)
    assert result["valid"], result["reason"]
    assert [d.opcode for d in result["decisions"]] == [
        "PLACE", "REWRITE_ONE", "PLACE", "REWRITE_ONE", "COMMIT"]
    # The fourth decision shrinks an already valid embedding instead of auto-committing.
    assert any(c.opcode is Opcode.COMMIT for c in fc.offered[3])
    assert len(result["embedding"][0]) == 1
    assert not any(fc.states[0].values())
    assert result["selected_program"] is result["terminal"].selected_program
    assert len(result["logps"]) == len(result["rewards"]) == len(result["togo"]) == 5
    assert len(result["potentials"]) == len(actor.supports) == 5
    assert result["logps"][-1].requires_grad
    assert sum(result["rewards"]) == pytest.approx(result["base_return"]) == 1
    assert result["togo"][0] == pytest.approx(result["return"])
    assert result["terminal_potential"] == 0


def test_actor_selects_route_to_realise_a_missing_demand():
    plan = [place(0, 0), place(1, 1), place(2, 2), opcode(Opcode.ROUTE), opcode(Opcode.COMMIT)]
    result, _, _ = run(plan, task=PublicTask(triangle=True))
    assert result["valid"], result["reason"]
    assert [d.opcode for d in result["decisions"]] == ["PLACE", "PLACE", "PLACE", "ROUTE", "COMMIT"]
    assert result["frac"] == 1


def test_actor_can_spend_more_qubits_after_validity_then_commit_that_embedding():
    plan = [place(0, 0), place(1, 1), provenance("grow:"),
            lambda c: c.opcode is Opcode.COMMIT and c.archive_ref == 1]
    result, fc, _ = run(plan)
    assert result["valid"] and result["steps"] == 4
    assert sum(map(len, fc.states[2].values())) == 2
    assert any(c.opcode is Opcode.COMMIT for c in fc.offered[2])
    assert result["decisions"][2].provenance.startswith("grow:")
    assert sum(map(len, result["embedding"].values())) == 3
    assert all(nx.is_connected(PublicTask().host.subgraph(chain))
               for chain in result["embedding"].values())
    assert set(result["embedding"][0]).isdisjoint(result["embedding"][1])
    assert result["allow_satisfied_growth"] is True
    assert result["context_version"].endswith("-independent-constructor-v1")


def test_satisfied_growth_is_opt_in_for_existing_proposal_callers():
    task = PublicTask()
    ctx = rollout._context(task, 5, 20, 8, {"grow": 8}, 2)
    chains = {0: frozenset({0}), 1: frozenset({1})}
    legacy = ProposalGenerator(ctx, task.logical, task.host, task.problem, mode=Mode.CONSTRUCTION)
    assert not legacy.generate(chains, random.Random(0)).candidates
    expanded = ProposalGenerator(ctx, task.logical, task.host, task.problem,
                                 mode=Mode.CONSTRUCTION, allow_satisfied_growth=True)
    candidates = expanded.generate(chains, random.Random(0)).candidates
    assert candidates and all(c.provenance.startswith("grow:") for c in candidates)
    for candidate in candidates:
        successor = {**chains, **candidate.new_chains}
        assert all(nx.is_connected(task.host.subgraph(chain)) for chain in successor.values())
        assert set(successor[0]).isdisjoint(successor[1])


def test_satisfied_growth_remains_subject_to_environment_qubit_cap():
    task = PublicTask()
    ctx = rollout._context(task, 2, 20, 8, None, 2)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None)
    env.generator.allow_satisfied_growth = True
    decision = env.reset(0)
    for wanted in (place(0, 0), place(1, 1)):
        chosen = next(i for i, c in enumerate(decision.candidates)
                      if decision.legal_mask[i] and wanted(c))
        decision = env.step(decision, chosen, evaluate_training_reward=False).next_decision_or_terminal
    grow_indices = [i for i, c in enumerate(decision.candidates) if c.provenance.startswith("grow:")]
    assert grow_indices and not any(decision.legal_mask[i] for i in grow_indices)
    assert any(c.opcode is Opcode.COMMIT and legal
               for c, legal in zip(decision.candidates, decision.legal_mask))


def test_legal_stop_is_chosen_and_not_masked():
    result, fc, _ = run([opcode(Opcode.STOP)])
    assert result["reason"] == "STOP_NO_VALID"
    assert not result["valid"] and result["steps"] == 1
    assert any(c.opcode is Opcode.PLACE for c in fc.offered[0])


def test_legal_restart_returns_to_empty_and_remains_an_actor_decision():
    plan = [place(0, 0), opcode(Opcode.RESTART), place(0, 0), provenance("grow:"),
            place(1, 2), opcode(Opcode.COMMIT)]
    result, fc, _ = run(plan)
    assert result["valid"]
    assert result["decisions"][1].opcode == "RESTART"
    assert not any(fc.states[2].values())


def test_horizon_without_commit_cannot_return_a_valid_archive():
    result, _, _ = run(grown_plan(), max_steps=3, shaping_coef=0.4, objective="quality")
    assert not result["valid"] and result["reason"] == "HORIZON"
    assert result["frac"] == 1
    assert result["embedding"] is result["terminal"] is result["selected_program"] is None
    assert result["return"] == pytest.approx(0)
    assert result["rewards"][0] > 0 and result["rewards"][-1] < 0
    assert len(result["decisions"]) == len(result["rewards"]) == 3


def test_quality_label_aligns_with_selected_commit_and_returns_to_go():
    calls = []

    def measure(task, terminal, seed, reads):
        calls.append((terminal, seed, reads))
        return 0.5

    result, _, _ = run(grown_plan(), objective="quality", train=True, measure=measure,
                       reward_reads=13, shaping_coef=0.2)
    assert len(calls) == 1 and calls[0][0] is result["terminal"] and calls[0][2] == 13
    assert result["residual"] == 0.5 and result["quality_measured"]
    assert result["base_return"] == pytest.approx(0.875)
    assert result["togo"][0] == pytest.approx(result["base_return"])
    assert result["togo"][-1] == pytest.approx(result["base_return"] - result["potentials"][-1])
    assert result["quality_coverage"] == 1


def test_missing_quality_cannot_silently_train_as_a_failed_embedding():
    with pytest.raises(ValueError, match="missing its requested residual label"):
        run(grown_plan(), objective="quality", train=True, measure=lambda *args: None)


def test_equal_quality_has_equal_reward_despite_different_qubit_counts():
    larger_plan = [place(0, 0), provenance("grow:"), place(1, 2), opcode(Opcode.COMMIT)]
    larger, _, _ = run(larger_plan, objective="quality", train=True, measure=lambda *args: 0.5)
    smaller, _, _ = run(grown_plan(), objective="quality", train=True, measure=lambda *args: 0.5)
    assert sum(map(len, larger["embedding"].values())) == 3
    assert sum(map(len, smaller["embedding"].values())) == 2
    assert larger["base_return"] == smaller["base_return"]


def test_inference_does_not_read_ground_or_measure_even_with_stochastic_actor():
    def forbidden(*args):
        raise AssertionError("inference measured quality")

    with torch.no_grad():
        result, _, _ = run(grown_plan(), task=PublicTask(hide_ground=True), objective="quality",
                           train=True, evaluate_reward=False, measure=forbidden)
    assert result["valid"] and result["unmeasured_valid"]
    assert not any(logp.requires_grad for logp in result["logps"])


def test_deadline_includes_reset(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock.now)
    reset = EmbeddingEnv.reset

    def slow_reset(self, seed):
        result = reset(self, seed)
        clock.now = 2.0
        return result

    monkeypatch.setattr(EmbeddingEnv, "reset", slow_reset)
    result, _, _ = run(grown_plan(), deadline=1.0)
    assert result["reason"] == "DEADLINE" and result["steps"] == 0
    assert result["secs"] == 2.0


def test_deadline_includes_features_without_charging_an_unselected_action(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock.now)
    observe = ScriptFeatures.observe

    def slow_features(self, *args, **kwargs):
        row = observe(self, *args, **kwargs)
        clock.now = 2.0
        return row

    monkeypatch.setattr(ScriptFeatures, "observe", slow_features)
    result, _, _ = run(grown_plan(), deadline=1.0)
    assert result["reason"] == "DEADLINE" and result["steps"] == 0
    assert result["decisions"] == result["rewards"] == []


def test_late_commit_is_rejected_and_terminal_potential_is_zero(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock.now)
    step = EmbeddingEnv.step

    def slow_commit(self, decision, chosen, **kwargs):
        result = step(self, decision, chosen, **kwargs)
        if decision.candidates[chosen].opcode is Opcode.COMMIT:
            clock.now = 2.0
        return result

    monkeypatch.setattr(EmbeddingEnv, "step", slow_commit)
    result, _, _ = run(grown_plan(), deadline=1.0, objective="quality", shaping_coef=0.3)
    assert not result["valid"] and result["reason"] == "DEADLINE"
    assert result["decisions"][-1].opcode == "COMMIT"
    assert result["terminal"] is None
    assert result["return"] == pytest.approx(0)


def test_training_measurement_time_does_not_reject_an_on_time_construction(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(rollout.time, "monotonic", lambda: clock.now)

    def delayed_measure(*args):
        clock.now = 4.0
        return 0.0

    result, _, _ = run(grown_plan(), deadline=1.0, objective="quality", train=True,
                       measure=delayed_measure)
    assert result["valid"] and result["reason"] == "COMMIT"
    assert result["secs"] == 0 and result["reward_secs"] == result["total_secs"] == 4

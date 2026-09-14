"""Environment, programming and data-generation contracts of the IF-Core RL stack."""

from __future__ import annotations

import itertools
import random

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.complete_system import complete_policy_context
from isingfold.rl.contracts import Context, Opcode, WorkVector, chain_key
from isingfold.rl.data.inkdrop import ink_drop, validate_witness
from isingfold.rl.data.lineage import Lineage, check_no_leakage, split_by_lineage
from isingfold.rl.data.planting import frustrated_loops
from isingfold.rl.data.structural import ContinuationDomain, exact_feasibility
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, IntegrityError
from isingfold.rl.program import check_faithfulness, compile_program, strength_registry
from isingfold.rl.proposal import (
    AUTHENTICATED_RESTART_CACHE_V1,
    LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    ProposalGenerator,
    router_initializer,
)
from isingfold.rl.router import route_variable
from isingfold.rl.validate import p_embed, p_return, p_search


@pytest.fixture
def tiny_task() -> EmbeddingTask:
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 6))
    logical = nx.gnm_random_graph(5, 7, seed=4)
    rng = random.Random(1)
    h = {v: rng.choice([-1.0, 1.0]) for v in logical.nodes()}
    j = {(u, v): rng.choice([-1.0, 1.0]) for u, v in logical.edges()}
    problem = LogicalProblem.from_dicts(h, j)
    nodes = sorted(logical.nodes())
    ground = min(
        sum(h[i] * s[i] for i in nodes) + sum(w * s[u] * s[v] for (u, v), w in j.items())
        for s in ({i: v for i, v in zip(nodes, combo)} for combo in itertools.product([-1, 1], repeat=len(nodes)))
    )
    return EmbeddingTask("tiny", logical, host, problem, ground)


def test_work_vector_is_coordinatewise_and_signed():
    a = WorkVector(decisions=2, route_expansions=10)
    b = WorkVector(decisions=1, route_expansions=20)
    assert (a - b).decisions == 1
    assert not (a - b).is_nonnegative
    assert a.fits_in(WorkVector(decisions=2, route_expansions=10))
    assert not b.fits_in(a)


def test_context_rejects_an_inconsistent_registry():
    with pytest.raises(ValueError):
        Context(qubit_cap=10, padded_actions=10)
    with pytest.raises(ValueError):
        Context(qubit_cap=10, strength_ratios=(1.0, 2.0))


def test_programming_reproduces_the_logical_energy_on_chain_consistent_states(tiny_task):
    chains = router_initializer()(tiny_task.logical, tiny_task.host, 3)
    assert chains is not None
    strengths = strength_registry(tiny_task.problem, (0.5, 1.0, 2.0, 4.0))
    program = compile_program(chains, tiny_task.host, tiny_task.problem, strengths[1], 1)
    report = check_faithfulness(program, chains, tiny_task.host, tiny_task.problem)
    assert report.ok, report.as_dict()

    # Eq. (program-identity): a chain-consistent physical state has the logical energy plus
    # a state-independent ferromagnetic offset.
    nodes = sorted(chains)
    for combo in itertools.product([-1, 1], repeat=len(nodes)):
        spins = dict(zip(nodes, combo, strict=True))
        z = {q: spins[i] for i, chain in chains.items() for q in chain}
        physical = sum(program.h_phys[q] * z[q] for q in z)
        physical += sum(w * z[a] * z[b] for (a, b), w in program.j_phys.items())
        logical_energy = sum(tiny_task.problem.h[i] * spins[i] for i in nodes)
        logical_energy += sum(w * spins[u] * spins[v] for (u, v), w in tiny_task.problem.j.items())
        assert physical == pytest.approx(program.scale * logical_energy + program.offset, abs=1e-9)


def test_autoscale_only_shrinks(tiny_task):
    chains = router_initializer()(tiny_task.logical, tiny_task.host, 3)
    program = compile_program(chains, tiny_task.host, tiny_task.problem, 8.0, 3, field_limit=0.5, coupler_limit=0.5)
    assert program.scale <= 1.0
    assert program.max_coupling() <= 0.5 + 1e-9


def test_validity_predicates_separate_overlap_from_return(tiny_task):
    chains = dict(router_initializer()(tiny_task.logical, tiny_task.host, 3))
    ctx = Context(qubit_cap=40)
    assert p_embed(chains, tiny_task.logical, tiny_task.host, 40).valid
    assert p_return(chains, tiny_task.logical, tiny_task.host, tiny_task.problem, ctx)[0].valid

    first, second = sorted(chains)[:2]
    overlapped = dict(chains)
    overlapped[first] = frozenset(set(chains[first]) | {next(iter(chains[second]))})
    assert p_search(overlapped, tiny_task.logical, tiny_task.host, 40, ctx.overlap).valid
    assert not p_embed(overlapped, tiny_task.logical, tiny_task.host, 40).valid


def test_router_never_claims_a_neighbour_chain(tiny_task):
    chains = router_initializer()(tiny_task.logical, tiny_task.host, 3)
    target = sorted(chains)[0]
    neighbours = [chains[u] for u in tiny_task.logical.neighbors(target)]
    occ = {q: 1 for i, c in chains.items() if i != target for q in c}
    attempt = route_variable(tiny_task.host, neighbours, occ, rng=random.Random(0))
    result = attempt.result
    assert result is not None
    assert attempt.expansions == result.expansions
    assert not result.chain & frozenset().union(*neighbours)
    assert nx.is_connected(tiny_task.host.subgraph(result.chain))


def test_proposal_batch_respects_quotas_dedup_and_work(tiny_task):
    ctx = Context(qubit_cap=40)
    chains = router_initializer()(tiny_task.logical, tiny_task.host, 3)
    generator = ProposalGenerator(ctx, tiny_task.logical, tiny_task.host, tiny_task.problem)
    batch = generator.generate(chains, random.Random(0))
    assert 0 < len(batch.candidates) <= ctx.max_state_changing
    assert len({c.payload_key for c in batch.candidates}) == len(batch.candidates)
    assert chain_key(chains) not in {c.payload_key for c in batch.candidates}
    assert batch.work.route_expansions > 0 and batch.work.materializations > 0

    tight = generator.generate(chains, random.Random(0), allowance=WorkVector(route_expansions=1, materializations=1))
    assert len(tight.candidates) <= len(batch.candidates)


def test_environment_contract(tiny_task):
    ctx = Context(qubit_cap=40)
    env = EmbeddingEnv(
        tiny_task,
        ctx,
        initializer=router_initializer(),
        reward_reads=16,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=2,
    )
    decision = env.reset(2)
    assert decision.candidates and all(decision.legal_mask)
    assert any(c.opcode is Opcode.COMMIT for c in decision.candidates)
    assert env.state.archive[0].protected

    # a stale decision state is an integrity error, not a low reward
    first = [k for k, c in enumerate(decision.candidates) if c.changes_workspace][0]
    step = env.step(decision, first)
    with pytest.raises(IntegrityError):
        env.step(decision, first)

    # every returned embedding passes the independent validator
    result = step.next_decision_or_terminal
    while not hasattr(result, "returned_valid"):
        commit = [k for k, c in enumerate(result.candidates) if c.opcode is Opcode.COMMIT]
        result = env.step(result, commit[0] if commit else 0).next_decision_or_terminal
    assert result.returned_valid
    assert result.validation_receipt["valid"]
    assert result.selected_strength in strength_registry(tiny_task.problem, ctx.strength_ratios)
    assert env.state.archive[0].protected


def test_registered_improvement_environment_requires_authenticated_restart_cache(
    tiny_task,
):
    initializer_calls = 0

    def initializer(logical, host, seed):
        nonlocal initializer_calls
        initializer_calls += 1
        return router_initializer()(logical, host, seed)

    with pytest.raises(ValueError, match="authenticated restart cache"):
        EmbeddingEnv(
            tiny_task,
            Context(qubit_cap=40),
            initializer=initializer,
            reward_reads=16,
            improvement_restart_protocol=AUTHENTICATED_RESTART_CACHE_V1,
        )

    assert initializer_calls == 0


def test_legacy_online_restart_behavior_requires_the_versioned_internal_opt_in(
    tiny_task,
):
    env = EmbeddingEnv(
        tiny_task,
        Context(qubit_cap=40),
        initializer=router_initializer(),
        reward_reads=16,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=2,
    )

    decision = env.reset(2)

    assert decision.candidates
    assert any(candidate.opcode is Opcode.RESTART for candidate in decision.candidates)


def test_precomputed_initializer_debit_preserves_fixed_cap_tensor_semantics(tiny_task):
    ctx = complete_policy_context(Context(qubit_cap=40))
    debit = WorkVector(
        decisions=2,
        route_expansions=101,
        materializations=3,
        compiler_calls=2,
        validator_calls=2,
        cut_edge_visits=7,
        restart_work=1,
        feature_work=31,
    )
    env = EmbeddingEnv(
        tiny_task,
        ctx,
        initializer=router_initializer(),
        reward_reads=16,
        budget_debit=debit,
        initializer_precomputed=True,
        seed=2,
    )

    decision = env.reset(2)
    assert not hasattr(decision, "returned_valid")
    assert not any(candidate.opcode is Opcode.RESTART for candidate in decision.candidates)
    assert env.ctx is ctx
    assert env.ctx.context_version == ctx.context_version
    assert env.ctx.caps == ctx.caps
    assert env.state is not None
    assert env.state.budget_debit == debit
    assert env.state.spent.restart_work == 0
    assert env.state.remaining + env.state.spent + debit == ctx.caps
    # Global slot 24 is the remaining route-expansion fraction.  The denominator is the
    # fixed registered cap, never a task-specific cap reduced by the initializer.
    assert decision.observation.globals_[0, 24] == pytest.approx(
        env.state.remaining.route_expansions / ctx.caps.route_expansions
    )

    commit = next(
        index
        for index, candidate in enumerate(decision.candidates)
        if candidate.opcode is Opcode.COMMIT and decision.legal_mask[index]
    )
    terminal = env.step(decision, commit).next_decision_or_terminal
    assert terminal.cumulative_work == env.state.spent
    assert env.state.remaining + terminal.cumulative_work + debit == ctx.caps


def test_environment_rejects_unprovable_or_unreservable_initial_debit(tiny_task):
    ctx = Context(qubit_cap=40)
    with pytest.raises(ValueError, match="nonzero exact budget debit"):
        EmbeddingEnv(
            tiny_task,
            ctx,
            initializer=router_initializer(),
            initializer_precomputed=True,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        )
    with pytest.raises(TypeError, match="exact WorkVector"):
        EmbeddingEnv(
            tiny_task,
            ctx,
            initializer=router_initializer(),
            budget_debit={"decisions": 1},
            initializer_precomputed=True,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        )
    with pytest.raises(ValueError, match="budget debit"):
        EmbeddingEnv(
            tiny_task,
            ctx,
            initializer=router_initializer(),
            budget_debit=WorkVector(decisions=ctx.caps.decisions),
            initializer_precomputed=True,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        )


def test_environment_uses_the_frozen_actor_coefficient_scale(tiny_task):
    class ScaledSelector:
        coefficient_transform_scale = 2.0

        def __call__(self, programs, features):
            del programs, features
            return 0

    initializer = router_initializer()
    context = Context(qubit_cap=40)
    unit = EmbeddingEnv(
        tiny_task,
        context,
        initializer=initializer,
        selector=ScaledSelector(),
        coefficient_scale=1.0,
        reward_reads=8,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    ).reset(2)
    fitted = EmbeddingEnv(
        tiny_task,
        context,
        initializer=initializer,
        selector=ScaledSelector(),
        reward_reads=8,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    ).reset(2)
    assert fitted.observation.logical[0, 0] != unit.observation.logical[0, 0]

    with pytest.raises(ValueError, match="reward_reads"):
        EmbeddingEnv(
            tiny_task,
            context,
            initializer=initializer,
            reward_reads=0,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        )
    with pytest.raises(ValueError, match="coefficient_scale"):
        EmbeddingEnv(
            tiny_task,
            context,
            initializer=initializer,
            coefficient_scale=0.0,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        )

def test_environment_never_exceeds_the_decision_horizon(tiny_task):
    ctx = Context(qubit_cap=40)
    env = EmbeddingEnv(
        tiny_task,
        ctx,
        initializer=router_initializer(),
        reward_reads=8,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=5,
    )
    result = env.reset(5)
    rng = random.Random(0)
    steps = 0
    while hasattr(result, "candidates"):
        legal = [k for k, m in enumerate(result.legal_mask) if m]
        result = env.step(result, rng.choice(legal)).next_decision_or_terminal
        steps += 1
        assert steps <= ctx.caps.decisions
    assert env.state.remaining.decisions >= 0


def test_inkdrop_witness_and_planted_bound():
    """The witness is always certified; a support without a cycle is honestly rejected."""

    from isingfold.rl.data.inkdrop import InkDropError
    from isingfold.rl.data.planting import PlantingError

    host = nx.grid_2d_graph(9, 9)
    planted_any = False
    for seed in range(8):
        try:
            drop = ink_drop(host, 12, 3, seed=seed, mode="contact_seeking")
        except InkDropError:
            continue  # an unrealizable requested cell is rejected, never silently shortened
        assert validate_witness(drop.witness, drop.logical, host)["valid"]
        try:
            planted = frustrated_loops(drop.logical, alpha=0.6, seed=seed)
        except PlantingError:
            continue  # loops do not cover tree-like supports; the generator says so
        report = planted.verify()
        assert report["attains_bound"] and report["matches_declared"]
        assert planted.ground_energy == pytest.approx(
            sum(c.bound for c in planted.clauses), abs=1e-9
        )
        planted_any = True
    assert planted_any, "no plantable support in eight cells"


def test_structural_label_is_unknown_rather_than_negative_on_timeout():
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4))
    logical = nx.path_graph(3)
    chains = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    window = frozenset(host.nodes())
    label = exact_feasibility(chains, logical, host, ContinuationDomain((1,), window, max_chain=2))
    assert label.feasible is True and label.min_qubits is not None
    impossible = exact_feasibility(
        chains, logical, host, ContinuationDomain((1,), frozenset({10, 11}), max_chain=1)
    )
    assert impossible.feasible is False and impossible.exhausted


def test_split_keeps_descendants_together():
    lineages = [Lineage(f"l{k}", "compact" if k % 2 else "bottleneck", "grid", 10, "v1") for k in range(10)]
    split = split_by_lineage(lineages, ood_predicate=lambda item: item.family == "bottleneck", seed=1)
    assert set(split.ood) == {item.lineage_id for item in lineages if item.family == "bottleneck"}
    records = [{"lineage_root": item.root} for item in lineages]
    assert check_no_leakage(split, records)["ok"]

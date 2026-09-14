"""Normative IF-Core tensor/model contracts from MODEL_SPEC Appendix A.

These tests intentionally inspect semantic tensors and pre-MLP inputs.  A shape-only
network can pass ordinary smoke tests while silently erasing the phase, edge-use, or
per-action relations that make IF-Core different from a flat candidate scorer.
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import ArchiveEntry, Candidate, Context, Opcode, WorkVector
from isingfold.rl.program import compile_program
from isingfold.rl.tensorize import Ages, N_ACTION, N_ROUTE, Observation, build_observation, t_count

torch = pytest.importorskip("torch")


def _phase_observation() -> Observation:
    logical = nx.Graph()
    logical.add_edge("u", "v")
    problem = LogicalProblem(logical, {"u": 2.0, "v": -1.0}, {("u", "v"): 1.5})
    host = nx.path_graph(5)
    chains = {"u": frozenset({0, 1}), "v": frozenset({2})}
    candidate = Candidate(
        opcode=Opcode.REWRITE_ONE,
        affected=("u",),
        old_chains={"u": chains["u"]},
        new_chains={"u": frozenset({3, 4})},
        routes=(((3, 4), "u"),),
        work=WorkVector(decisions=1, materializations=1),
        payload_key="move-u",
    )
    archive = [
        ArchiveEntry(
            chains=chains,
            protected=True,
            age=0,
            admissible=True,
            qubits=3,
            max_chain=2,
            key="seed",
        )
    ]
    ctx = Context(qubit_cap=5)
    program = compile_program(chains, host, problem, 1.0, 0)
    return build_observation(
        ctx=ctx,
        logical=logical,
        host=host,
        problem=problem,
        chains=chains,
        candidates=[candidate],
        legal_mask=[True],
        archive=archive,
        remaining=ctx.caps,
        ages=Ages(),
        strengths=(0.5, 1.0, 2.0, 4.0),
        program=program,
        mode_is_improvement=True,
        restarts_left=2,
        workspace_valid=True,
        protected_available=True,
        instance_scale=1.0,
    )


def _conflict_observation() -> Observation:
    logical = nx.Graph()
    logical.add_edge("u", "v")
    problem = LogicalProblem(logical, {"u": 1.0, "v": -1.0}, {("u", "v"): 1.0})
    host = nx.path_graph(4)
    chains = {"u": frozenset({0, 1}), "v": frozenset({1, 2})}
    rewrite = Candidate(
        Opcode.REPAIR_GROUP,
        ("u", "v"),
        {"u": chains["u"], "v": chains["v"]},
        {"u": frozenset({0}), "v": frozenset({1, 2})},
        WorkVector(decisions=1),
        "repair",
    )
    stop = Candidate(Opcode.STOP, (), {}, {}, WorkVector(decisions=1), "stop")
    ctx = Context(qubit_cap=4)
    return build_observation(
        ctx=ctx,
        logical=logical,
        host=host,
        problem=problem,
        chains=chains,
        candidates=[rewrite, stop],
        legal_mask=[True, True],
        archive=[],
        remaining=ctx.caps,
        ages=Ages(),
        strengths=(0.5, 1.0, 2.0, 4.0),
        program=None,
        mode_is_improvement=False,
        restarts_left=2,
        workspace_valid=False,
        protected_available=False,
        instance_scale=1.0,
    )


def test_phase_edge_use_records_are_explicit_and_factor_input_is_exactly_640() -> None:
    from isingfold.rl.model import IFCore

    obs = _phase_observation()
    assert obs.edge_use_roles.size > 0
    assert set(obs.edge_use_roles.tolist()) == {0, 1}  # CHAIN and LOGICAL
    assert obs.index_edge_use_factor.shape == (obs.edge_use_roles.size,)
    assert obs.index_edge_use_hardware.shape == (obs.edge_use_roles.size,)
    assert obs.index_edge_use_logical.shape == (obs.edge_use_roles.size,)
    assert np.any(obs.index_edge_use_logical < 0)  # guarded CHAIN records
    assert np.any(obs.index_edge_use_logical >= 0)
    assert obs.index_factor_realized_edge_use.shape[0] == 2

    def factor_edges(role: int) -> set[frozenset[int]]:
        factor_id = int(np.flatnonzero(obs.factor_roles == role)[-1])
        use_rows = np.flatnonzero(obs.index_edge_use_factor == factor_id)
        return {
            frozenset(
                (
                    obs.qubit_ids[obs.index_hardware_edges[0, obs.index_edge_use_hardware[row]]],
                    obs.qubit_ids[obs.index_hardware_edges[1, obs.index_edge_use_hardware[row]]],
                )
            )
            for row in use_rows
        }

    # NEW is encoded against its own successor, not against the current OLD program.
    assert factor_edges(0) == {frozenset({0, 1}), frozenset({1, 2})}
    assert factor_edges(1) == {frozenset({2, 3}), frozenset({3, 4})}

    model = IFCore(improvement_mode=True).eval()
    assert model.factor[0].in_features == 640
    captured: list[torch.Tensor] = []
    hook = model.factor.register_forward_pre_hook(
        lambda _m, args: captured.append(args[0].detach())
    )
    model.forward_single(obs)
    hook.remove()
    assert captured and captured[0].shape[1] == 640
    # Last 64 channels are the factor-specific realized-logical-edge mean/max pool.
    assert torch.count_nonzero(captured[0][:, -64:]).item() > 0


def test_factor_roles_and_route_phase_are_neural_inputs_not_constant_zero() -> None:
    from isingfold.rl.model import IFCore

    obs = _phase_observation()
    model = IFCore(improvement_mode=True).eval()
    factor_inputs: list[torch.Tensor] = []
    route_bind_inputs: list[torch.Tensor] = []
    h1 = model.factor.register_forward_pre_hook(
        lambda _m, args: factor_inputs.append(args[0].detach())
    )
    h2 = model.routes.bind.register_forward_pre_hook(
        lambda _m, args: route_bind_inputs.append(args[0].detach())
    )
    model.forward_single(obs)
    h1.remove()
    h2.remove()

    # Archive, OLD, and NEW descriptors each receive their own learned role token.
    roles = obs.factor_roles
    descriptor_tokens = factor_inputs[0][:, :128]
    representatives = [descriptor_tokens[np.flatnonzero(roles == role)[0]] for role in (0, 1, 2)]
    assert not torch.allclose(representatives[0], representatives[1])
    assert not torch.allclose(representatives[0], representatives[2])
    assert route_bind_inputs and torch.count_nonzero(route_bind_inputs[0][:, -128:]).item() > 0


def test_old_new_factor_pairs_use_one_batched_projection() -> None:
    """Action/owner pairs must not launch one tiny MLP kernel per pair."""

    from isingfold.rl.model import IFCore

    obs = _phase_observation()
    model = IFCore(improvement_mode=True).eval()
    pair_inputs: list[torch.Tensor] = []
    hook = model.pair.register_forward_pre_hook(
        lambda _module, args: pair_inputs.append(args[0].detach())
    )
    model.forward_single(obs)
    hook.remove()

    assert len(pair_inputs) == 1
    assert pair_inputs[0].shape == (1, 512)


def test_route_position_packing_has_no_per_position_scalar_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packing an ordered route never synchronizes CUDA tensors back to the host."""

    from isingfold.rl.model import RouteEncoder

    original_item = torch.Tensor.item
    item_calls = 0

    def counted_item(value: torch.Tensor, *args: object) -> object:
        nonlocal item_calls
        item_calls += 1
        return original_item(value, *args)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    encoder = RouteEncoder().eval()
    route_of_position = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    with torch.no_grad():
        output = encoder(
            torch.zeros((6, 2 * N_ROUTE)),
            torch.arange(6, dtype=torch.long),
            torch.zeros((6, 128)),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            torch.zeros((2, 128)),
            2,
            route_of_position,
            3,
        )

    assert output.shape == (2, 128)
    assert item_calls == 0


def test_tensorization_builds_phase_ownership_once_per_distinct_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factor edge rows must reuse phase ownership instead of rebuilding it per edge."""

    from isingfold.rl import tensorize

    original_owner_sets = tensorize._owner_sets
    calls = 0

    def counted_owner_sets(chains):
        nonlocal calls
        calls += 1
        return original_owner_sets(chains)

    monkeypatch.setattr(tensorize, "_owner_sets", counted_owner_sets)
    _phase_observation()

    # Current, archived, and successor phases are the only three distinct mappings.
    assert calls == 3


def test_ifcore_decode_has_no_tensor_truth_value_cuda_synchronizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode control flow is decided from host metadata, never a device scalar."""

    from isingfold.rl.model import IFCore

    original_bool = torch.Tensor.__bool__
    calls = 0

    def counted_bool(value: torch.Tensor) -> bool:
        nonlocal calls
        calls += 1
        return original_bool(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    with torch.no_grad():
        IFCore(improvement_mode=True).eval().forward_single(_phase_observation())

    assert calls == 0


def test_forward_single_encodes_the_graph_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scalar compatibility path must not duplicate the expensive GNN trunk."""

    from isingfold.rl.model import IFCore

    model = IFCore(improvement_mode=True).eval()
    original_encode = model.encode
    calls = 0

    def counted_encode(obs: Observation, device: torch.device) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        return original_encode(obs, device)

    monkeypatch.setattr(model, "encode", counted_encode)
    model.forward_single(_phase_observation())
    assert calls == 1


def test_action_slot_14_contains_exact_candidate_proposal_work() -> None:
    logical = nx.Graph([("u", "v")])
    problem = LogicalProblem(logical, {"u": 2.0, "v": -1.0}, {("u", "v"): 1.5})
    host = nx.path_graph(5)
    chains = {"u": frozenset({0, 1}), "v": frozenset({2})}
    proposal_work = WorkVector(route_expansions=7, materializations=1)
    candidate = Candidate(
        opcode=Opcode.REWRITE_ONE,
        affected=("u",),
        old_chains={"u": chains["u"]},
        new_chains={"u": frozenset({3, 4})},
        work=WorkVector(decisions=1, validator_calls=1),
        payload_key="charged-move",
        proposal_work=proposal_work,
    )
    ctx = Context(qubit_cap=5)
    charged = build_observation(
        ctx=ctx,
        logical=logical,
        host=host,
        problem=problem,
        chains=chains,
        candidates=[candidate],
        legal_mask=[True],
        archive=[],
        remaining=ctx.caps,
        ages=Ages(),
        strengths=(0.5, 1.0, 2.0, 4.0),
        program=compile_program(chains, host, problem, 1.0, 0),
        mode_is_improvement=True,
        restarts_left=2,
        workspace_valid=True,
        protected_available=False,
        instance_scale=1.0,
    )

    proposal_slot = 13
    assert charged.actions[0, proposal_slot] == pytest.approx(t_count(8))
    assert charged.actions[0, N_ACTION + proposal_slot] == 1.0
    assert charged.actions.shape[1] == 56


def test_touched_conflicts_are_pooled_per_action() -> None:
    from isingfold.rl.model import IFCore

    obs = _conflict_observation()
    assert obs.index_action_conflicts.shape == (2, 1)
    assert tuple(obs.index_action_conflicts[:, 0]) == (0, 0)

    model = IFCore(improvement_mode=False).eval()
    captured: list[torch.Tensor] = []
    hook = model.action.register_forward_pre_hook(
        lambda _m, args: captured.append(args[0].detach())
    )
    model.forward_single(obs)
    hook.remove()
    # action input = descriptor 128, chains 256, routes 256, conflicts 256, archive 128
    conflict_slice = captured[0][:, 640:896]
    assert torch.count_nonzero(conflict_slice[0]).item() > 0
    assert torch.count_nonzero(conflict_slice[1]).item() == 0


def test_only_active_topology_marks_nominal_fault_fields_missing_and_program_zeros_known() -> None:
    obs = _phase_observation()
    n_h = 18
    q0 = obs.qubit_ids.index(0)
    q4 = obs.qubit_ids.index(4)

    # No nominal graph/calibration snapshot was supplied: nominal degree and fault fraction miss.
    assert obs.hardware[q0, n_h + 2] == 0.0
    assert obs.hardware[q0, n_h + 7] == 0.0
    # Nearest-free means graph distance to *any* free active vertex, not only a free neighbor.
    assert obs.hardware[q0, n_h + 16] == 1.0
    assert obs.hardware[q0, 16] == pytest.approx(t_count(3))
    # A complete program makes unused programmable fields known zeros.
    assert obs.hardware[q4, n_h + 9] == 1.0
    assert obs.hardware[q4, 9] == 0.0
    assert obs.hardware[q4, n_h + 10] == 1.0
    assert obs.hardware[q4, 10] == 0.0

    # The unused active edge (2,3) likewise has known-zero program flags/coefficient.
    edge_row = next(
        k
        for k, (u, v) in enumerate(obs.index_hardware_edges.T)
        if {obs.qubit_ids[u], obs.qubit_ids[v]} == {2, 3}
    )
    for slot in (3, 4, 5, 6):
        assert obs.hardware_edges[edge_row, 10 + slot] == 1.0
        assert obs.hardware_edges[edge_row, slot] == 0.0


def test_global_aggregate_and_terminal_reserve_slots_follow_the_registered_ledgers() -> None:
    obs0 = _phase_observation()
    # Rebuild with deliberately nonuniform ledgers so feature_work cannot masquerade as slot 10
    # and evaluator reads cannot masquerade as terminal reserve in slot 31.
    logical = nx.Graph([("u", "v")])
    host = nx.path_graph(5)
    problem = LogicalProblem(logical, {"u": 1.0, "v": 1.0}, {("u", "v"): -1.0})
    chains = {"u": frozenset({0}), "v": frozenset({1})}
    caps = WorkVector(32, 100, 10, 20, 10, 100, 10, 5000, 1000)
    reserve = WorkVector(1, 0, 2, 12, 4, 0, 0, 256, 416)
    remaining = WorkVector(16, 50, 5, 15, 2, 50, 5, 4500, 600)
    ctx = Context(qubit_cap=5, caps=caps, reserve=reserve)
    stop = Candidate(Opcode.STOP, (), {}, {}, WorkVector(decisions=1), "stop")
    obs = build_observation(
        ctx=ctx,
        logical=logical,
        host=host,
        problem=problem,
        chains=chains,
        candidates=[stop],
        legal_mask=[True],
        archive=[],
        remaining=remaining,
        ages=Ages(),
        strengths=(0.5, 1.0, 2.0, 4.0),
        program=None,
        mode_is_improvement=False,
        restarts_left=1,
        workspace_valid=False,
        protected_available=False,
        instance_scale=1.0,
    )
    deterministic_fields = (
        "route_expansions",
        "materializations",
        "compiler_calls",
        "validator_calls",
        "cut_edge_visits",
        "restart_work",
        "feature_work",
    )
    expected_aggregate = sum(getattr(remaining, f) for f in deterministic_fields) / sum(
        getattr(caps, f) for f in deterministic_fields
    )
    assert obs.globals_[0, 9] == pytest.approx(expected_aggregate)
    assert obs.globals_[0, 30] == pytest.approx(0.5)  # validator reserve is the bottleneck
    assert obs.globals_[0, 9] != obs0.globals_[0, 9]


def test_mixed_python_node_types_have_stable_undirected_pair_semantics() -> None:
    logical = nx.Graph()
    logical.add_edge("1", 1)  # insertion orientation intentionally opposes the J record
    problem = LogicalProblem(logical, {1: 1.0, "1": -1.0}, {(1, "1"): 2.0})
    host = nx.Graph()
    host.add_edge("q", 0)
    chains = {1: frozenset(), "1": frozenset()}
    stop = Candidate(Opcode.STOP, (), {}, {}, WorkVector(decisions=1), "stop")
    ctx = Context(qubit_cap=2)
    obs = build_observation(
        ctx=ctx,
        logical=logical,
        host=host,
        problem=problem,
        chains=chains,
        candidates=[stop],
        legal_mask=[True],
        archive=[],
        remaining=ctx.caps,
        ages=Ages(demand={("1", 1): 5}),
        strengths=(0.5, 1.0, 2.0, 4.0),
        program=None,
        mode_is_improvement=False,
        restarts_left=2,
        workspace_valid=False,
        protected_available=False,
        instance_scale=1.0,
    )
    assert obs.logical_ids == (1, "1")
    assert obs.qubit_ids == (0, "q")
    assert np.all(obs.logical_edges[:, 0] != 0.0)  # J lookup did not depend on edge orientation
    assert np.all(obs.logical_edges[:, 5] == pytest.approx(t_count(5)))


def test_projectors_public_batch_padding_masks_and_frozen_improvement_failure_tower() -> None:
    from isingfold.rl.model import GraphActorCritic, IFCore

    obs = _phase_observation()
    model = GraphActorCritic(improvement_mode=True).eval()
    for projector in (
        model.proj_logical,
        model.proj_hardware,
        model.proj_conflict,
        model.proj_factor,
    ):
        assert [type(layer).__name__ for layer in projector] == ["Linear", "SiLU", "LayerNorm"]
    assert not any(parameter.requires_grad for parameter in model.failure.parameters())

    other = _conflict_observation()  # empty archive, unlike ``obs``; conflicts are nonempty
    single = model.forward_single(obs)
    out = model(
        [obs, other],
        [obs.actions, other.actions],
        [obs.legal_mask, other.legal_mask],
    )
    assert out.masked_log_probs.shape == (2, 73)
    assert out.utility_value.shape == out.failure_logit.shape == out.failure_value.shape == (2,)
    assert out.action_count.dtype == torch.int64
    assert torch.equal(out.action_count, torch.tensor([1, 2]))
    assert torch.isneginf(out.masked_log_probs[0, 1:]).all()
    assert torch.isneginf(out.masked_log_probs[1, 2:]).all()
    assert torch.equal(out.failure_value, torch.zeros(2))
    # The first observation is identical alone and in a heterogeneous batch.
    assert torch.allclose(out.masked_log_probs[0, :1], single.masked_log_probs)
    assert torch.allclose(out.utility_value[0], single.utility_value)

    invalid = replace(obs, real_action_mask=np.zeros_like(obs.real_action_mask))
    with pytest.raises(ValueError, match="nonempty legal support"):
        IFCore(improvement_mode=True).forward_single(invalid)


def test_action_permutation_moves_every_action_relation_and_exact_probabilities() -> None:
    from isingfold.rl.model import IFCore

    obs = _conflict_observation()
    model = IFCore(improvement_mode=False).eval()
    base = model.forward_single(obs).masked_log_probs.detach()
    order = np.array([1, 0])
    inverse = np.argsort(order)
    moved = replace(
        obs,
        actions=obs.actions[order],
        legal_mask=obs.legal_mask[order],
        real_action_mask=obs.real_action_mask[order],
        index_factor_action=obs.index_factor_action.copy(),
        index_route_action=obs.index_route_action.copy(),
        index_action_archive=obs.index_action_archive.copy(),
        index_action_conflicts=obs.index_action_conflicts.copy(),
    )
    for field, endpoint in (
        ("index_factor_action", 1),
        ("index_route_action", 1),
        ("index_action_archive", 0),
        ("index_action_conflicts", 0),
    ):
        relation = getattr(moved, field)
        if relation.size:
            relation[endpoint] = inverse[relation[endpoint]]
    permuted = model.forward_single(moved).masked_log_probs.detach()
    assert torch.allclose(permuted, base[torch.as_tensor(order)], atol=1e-6, rtol=1e-6)

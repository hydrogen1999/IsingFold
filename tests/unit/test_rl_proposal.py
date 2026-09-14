from __future__ import annotations

import random
from dataclasses import replace

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    ArchiveEntry,
    Candidate,
    Context,
    Mode,
    Opcode,
    RestartCacheSlot,
    WorkVector,
    candidate_support_key,
    chain_key,
    stable_digest,
)
from isingfold.rl.proposal import (
    AUTHENTICATED_RESTART_CACHE_V1,
    LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    ProposalGenerator,
    WorkMeter,
    bound_successor_key,
    strongest_neighbours,
)
from isingfold.rl import router
from isingfold.rl.router import rebuild_group, route_between_sets, route_variable


def test_strongest_neighbours_uses_numeric_logical_coupling_identity() -> None:
    problem = LogicalProblem.from_dicts(
        {2: 0.0, 3: 0.0, 10: 0.0},
        {(2, 3): 1.0, (2, 10): 9.0},
    )

    assert strongest_neighbours(problem, problem.graph, 2, 2) == [10, 3]


def test_failed_route_reports_every_expansion_before_no_common_root() -> None:
    host = nx.path_graph(6)
    claimed_component = frozenset(host.nodes())

    attempt = route_variable(
        host,
        [claimed_component],
        {qubit: 1 for qubit in claimed_component},
        expansion_budget=100,
    )

    assert attempt.result is None
    assert attempt.expansions == host.number_of_nodes()


def test_weighted_router_does_not_requeue_every_edge_into_an_unsettled_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A uniform complete graph needs one frontier insertion per vertex.

    Without a tentative-distance map, every incoming edge reinserted an unsettled
    vertex and made the Pegasus proposal loop spend most of its time in ``heapq``.
    Settled expansion accounting is unchanged by suppressing those dominated entries.
    """

    pushes = 0
    original_push = router.heapq.heappush

    def counted_push(heap: list[tuple[float, int, int]], item: tuple[float, int, int]) -> None:
        nonlocal pushes
        pushes += 1
        original_push(heap, item)

    monkeypatch.setattr(router.heapq, "heappush", counted_push)
    host = nx.complete_graph(12)
    attempt = route_variable(
        host,
        [frozenset({0})],
        {},
        expansion_budget=host.number_of_nodes(),
    )

    assert attempt.expansions == host.number_of_nodes()
    assert pushes == host.number_of_nodes()


def test_weighted_router_predecessor_matches_the_settled_shortest_distance() -> None:
    """A later, worse relaxation must not overwrite the best pending predecessor."""

    host = nx.Graph(
        [("source", "short"), ("source", "late"), ("short", "target"), ("late", "target")]
    )
    jitter = {"source": 1.0, "short": 1.0, "late": 4.0, "target": 5.0}

    distance, predecessor, expansions = router._dijkstra(
        host,
        {"source"},
        {},
        1.0,
        jitter,
        host.number_of_nodes(),
    )

    assert expansions == host.number_of_nodes()
    assert distance["target"] == pytest.approx(7.0)
    assert router._path_to(predecessor, "target", {"source"}) == (
        "source",
        "short",
        "target",
    )


def test_route_variable_computes_each_node_cost_once_per_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neighbour-specific Dijkstra passes share the same occupancy/jitter costs."""

    calls = 0
    original_cost = router.qubit_cost

    def counted_cost(occupancy: int, overfill: float) -> float:
        nonlocal calls
        calls += 1
        return original_cost(occupancy, overfill)

    monkeypatch.setattr(router, "qubit_cost", counted_cost)
    host = nx.cycle_graph(20)
    attempt = route_variable(
        host,
        [frozenset({0}), frozenset({10})],
        {},
        expansion_budget=2 * host.number_of_nodes(),
    )

    assert attempt.expansions > 0
    assert calls == host.number_of_nodes()


def test_group_rebuild_failure_preserves_work_from_earlier_members() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.empty_graph(1)

    attempt = rebuild_group(
        host,
        {0: frozenset(), 1: frozenset()},
        logical,
        (0, 1),
        rng=random.Random(0),
        order=(0, 1),
        expansion_budget=20,
    )

    assert attempt.placed is None
    assert attempt.expansions == 2


def test_route_family_charges_metered_bfs_pops_not_path_length() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.Graph([(0, 1), (1, 2), (0, 3), (0, 4), (0, 5)])
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    generator = ProposalGenerator(
        Context(qubit_cap=6, construction_quotas={"route": 1}),
        logical,
        host,
        problem,
        mode=Mode.CONSTRUCTION,
    )

    expected = route_between_sets(host, {0}, {2}, expansion_budget=100)
    assert expected.path == (0, 1, 2)
    assert expected.expansions > len(expected.path)
    batch = generator.generate(
        {0: frozenset({0}), 1: frozenset({2})},
        random.Random(0),
        allowance=WorkVector(route_expansions=100, materializations=1),
    )

    assert len(batch.candidates) == 1
    assert batch.work.route_expansions == expected.expansions
    assert batch.candidates[0].proposal_work.route_expansions == expected.expansions


def test_failed_materialization_attempts_are_charged_and_reported() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.empty_graph(2)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    context = Context(qubit_cap=2, quotas={"single": 2})
    generator = ProposalGenerator(context, logical, host, problem)

    allowance = WorkVector(route_expansions=20, materializations=20)
    batch = generator.generate(
        {0: frozenset({0}), 1: frozenset({1})},
        random.Random(0),
        allowance=allowance,
    )

    assert not batch.candidates
    assert batch.work.materializations > 0
    assert batch.work.route_expansions > 0
    assert batch.rejected == batch.work.materializations
    assert batch.work.fits_in(allowance)


def test_candidate_keeps_exact_proposal_receipt_and_support_binds_it() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(5)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    context = Context(qubit_cap=5, quotas={"single": 2})
    generator = ProposalGenerator(context, logical, host, problem)

    batch = generator.generate(
        {0: frozenset({0}), 1: frozenset({1})},
        random.Random(4),
        allowance=WorkVector(route_expansions=100, materializations=20),
    )

    assert batch.candidates
    assert all(candidate.proposal_work.materializations == 1 for candidate in batch.candidates)
    assert all(candidate.proposal_work.is_nonnegative for candidate in batch.candidates)
    retained = sum(
        sum(candidate.proposal_work.as_dict().values()) for candidate in batch.candidates
    )
    assert retained <= sum(batch.work.as_dict().values())

    first = batch.candidates[0]
    changed_receipt = replace(
        first,
        proposal_work=first.proposal_work + WorkVector(route_expansions=1),
    )
    assert candidate_support_key([first], [True]) != candidate_support_key(
        [changed_receipt], [True]
    )


def test_restart_cache_deduplicates_identical_exemplars_without_free_work() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    chains = {0: frozenset({0}), 1: frozenset({1})}
    cached = {0: frozenset({2}), 1: frozenset({3})}
    slots = tuple(
        RestartCacheSlot(
            slot_index=index,
            status="SUCCESS",
            chains=cached,
            snapshot_record_digest=stable_digest({"slot": index}),
            attempt_receipt_root=stable_digest({"attempt": index}),
        )
        for index in range(2)
    )
    generator = ProposalGenerator(
        Context(qubit_cap=4, quotas={"restart": 2}),
        logical,
        host,
        problem,
        initializer=lambda *_: chains,
    )

    batch = generator.generate(
        chains,
        random.Random(5),
        restarts_left=2,
        restart_cache=slots,
        allowance=WorkVector(materializations=2),
    )

    assert len(batch.candidates) == 1
    assert batch.candidates[0].restart_cache_slot == 0
    assert batch.work == WorkVector(materializations=2)
    assert batch.rejected == 1


def test_registered_improvement_proposals_fail_before_calling_online_initializer() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    calls = 0

    def forbidden_initializer(_logical, _host, _seed):
        nonlocal calls
        calls += 1
        return {0: frozenset({2}), 1: frozenset({3})}

    generator = ProposalGenerator(
        Context(qubit_cap=4, quotas={"restart": 2}),
        logical,
        host,
        problem,
        initializer=forbidden_initializer,
        improvement_restart_protocol=AUTHENTICATED_RESTART_CACHE_V1,
    )

    with pytest.raises(RuntimeError, match="authenticated restart cache"):
        generator.generate(
            {0: frozenset({0}), 1: frozenset({1})},
            random.Random(5),
            restarts_left=2,
            allowance=WorkVector(materializations=2, restart_work=2),
        )

    assert calls == 0


def test_legacy_online_initializer_restarts_are_explicitly_versioned() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    calls = 0

    def initializer(_logical, _host, _seed):
        nonlocal calls
        calls += 1
        return {0: frozenset({2}), 1: frozenset({3})}

    generator = ProposalGenerator(
        Context(qubit_cap=4, quotas={"restart": 2}),
        logical,
        host,
        problem,
        initializer=initializer,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    )

    batch = generator.generate(
        {0: frozenset({0}), 1: frozenset({1})},
        random.Random(5),
        restarts_left=2,
        allowance=WorkVector(materializations=2, restart_work=2),
    )

    assert calls == 2
    assert batch.work.restart_work == 2
    assert all(candidate.provenance.startswith("legacy-online-restart-v1:") for candidate in batch.candidates)


def test_quota_shortfall_refills_in_fixed_family_order_until_common_allowance() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(20)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    context = Context(qubit_cap=20, quotas={"single": 1, "group2": 1})

    class DeterministicGenerator(ProposalGenerator):
        serial = 0

        def _single(self, chains, rng, budget, meter):
            del chains, rng, budget, meter
            return []

        def _group(
            self,
            chains,
            rng,
            size,
            budget,
            meter,
            opcode=Opcode.REWRITE_GROUP,
            seeds=None,
        ):
            del rng, opcode, seeds
            if size != 2:
                return []
            out = []
            while len(out) < budget and meter.can_charge(1):
                serial = self.serial
                self.serial += 1
                receipt = meter.charge(1)
                replacement = {0: frozenset({2 + serial}), 1: frozenset({3 + serial})}
                successor = dict(chains)
                successor.update(replacement)
                work = WorkVector(decisions=1)
                out.append(
                    (
                        Candidate(
                            Opcode.REWRITE_GROUP,
                            (0, 1),
                            {0: chains[0], 1: chains[1]},
                            replacement,
                            work,
                            bound_successor_key(successor, work, restart=False),
                            proposal_work=receipt,
                            provenance=f"group2:{serial}",
                        ),
                        1,
                    )
                )
            return out

        def _repair(self, chains, rng, budget, meter):
            del chains, rng, budget, meter
            return []

    generator = DeterministicGenerator(context, logical, host, problem)
    allowance = WorkVector(route_expansions=5, materializations=5)

    first = generator.generate(
        {0: frozenset({0}), 1: frozenset({1})},
        random.Random(9),
        allowance=allowance,
    )
    second_generator = DeterministicGenerator(context, logical, host, problem)
    second = second_generator.generate(
        {0: frozenset({0}), 1: frozenset({1})},
        random.Random(9),
        allowance=allowance,
    )

    assert len(first.candidates) == 5
    assert first.work == allowance
    assert first.counts["single"] == 0
    assert first.counts["group2"] == 5
    assert [candidate.payload_key for candidate in first.candidates] == [
        candidate.payload_key for candidate in second.candidates
    ]


def test_archive_restore_is_a_paid_bound_group_rewrite_sharing_restart_quota() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.cycle_graph(6)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    current = {0: frozenset({4}), 1: frozenset({5})}
    archived = {0: frozenset({0}), 1: frozenset({1})}
    entry = ArchiveEntry(
        chains=archived,
        protected=True,
        age=3,
        admissible=True,
        qubits=2,
        max_chain=1,
        key=chain_key(archived),
    )
    initializer_calls: list[int] = []

    def initializer(_logical, _host, seed):
        initializer_calls.append(seed)
        return {0: frozenset({2}), 1: frozenset({3})}

    context = Context(qubit_cap=6, quotas={"restart": 4})
    generator = ProposalGenerator(
        context,
        logical,
        host,
        problem,
        initializer=initializer,
        mode=Mode.IMPROVEMENT,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    )
    batch = generator.generate(
        current,
        random.Random(3),
        restarts_left=2,
        archive=[entry],
        allowance=WorkVector(route_expansions=0, materializations=100, restart_work=100),
    )

    restores = [candidate for candidate in batch.candidates if candidate.archive_ref is not None]
    restarts = [candidate for candidate in batch.candidates if candidate.opcode is Opcode.RESTART]
    assert restores
    assert len(restores) + len(restarts) <= context.quotas["restart"]
    assert batch.counts["restore"] == len(restores)
    restore = restores[0]
    assert restore.opcode is Opcode.REWRITE_GROUP
    assert restore.archive_ref == 0
    assert all(restore.new_chains[node] == archived[node] for node in restore.affected)
    assert restore.proposal_work == WorkVector(materializations=1)
    assert restore.work.restart_work == 0
    assert batch.work.materializations == 4
    assert batch.work.restart_work == len(initializer_calls)


def test_work_meter_allows_zero_route_restore_at_an_exhausted_route_coordinate() -> None:
    meter = WorkMeter(route_expansions=0, materializations=1, restart_work=0)

    receipt = meter.charge(0)

    assert receipt == WorkVector(materializations=1)
    assert meter.used_expansions == 0
    with pytest.raises(ValueError, match="allowance"):
        meter.charge(0)

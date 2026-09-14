from __future__ import annotations

import pytest

from lac_minorminer import SearchSession
from lac_minorminer import _core


class GraphLike:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def is_directed(self) -> bool:
        return False


def test_native_session_exposes_read_only_batched_operations() -> None:
    source = _core.Graph(1, [])
    target = _core.Graph(3, [(0, 1), (1, 2)])
    session = _core.SearchSession(source, target, random_seed=17, max_candidates=2)

    snapshot = session.snapshot()
    assert snapshot.valid
    assert len(snapshot.chains) == 1

    batch = session.propose(0, [4.0, 1.0, 2.0])
    assert [candidate.chain for candidate in batch.candidates] == [[1], [2]]
    assert batch.candidates[0].rank.max_occupancy == 1
    with pytest.raises(AttributeError):
        batch.token = 100

    session.apply(batch, 0)
    assert session.snapshot().chains == [[1]]
    with pytest.raises(RuntimeError, match="stale|consumed"):
        session.apply(batch, 0)


def test_native_audit_batch_is_wider_linked_and_non_applicable() -> None:
    source = _core.Graph(1, [])
    target = _core.Graph(3, [(0, 1), (1, 2)])
    session = _core.SearchSession(source, target, random_seed=17, max_candidates=2)

    audited = session.propose_with_audit(0, 3, [4.0, 1.0, 2.0])

    assert len(audited.decision_batch.candidates) == 2
    assert len(audited.audit_batch.candidates) == 3
    assert [candidate.candidate_id for candidate in audited.decision_batch.candidates] == [0, 1]
    assert [candidate.candidate_id for candidate in audited.audit_batch.candidates] == [0, 1, 2]
    assert audited.audit_batch.token == 0
    with pytest.raises(RuntimeError, match="stale|foreign"):
        session.apply(audited.audit_batch, 0)

    session.apply(audited.decision_batch, 0)
    assert session.snapshot().chains == [[1]]


def test_label_aware_session_validates_and_forwards_audit_requests() -> None:
    session = SearchSession(
        GraphLike(["logical"], []),
        GraphLike(["left", "middle", "right"], [("left", "middle"), ("middle", "right")]),
        random_seed=17,
        max_candidates=2,
    )

    audited = session.propose_with_audit(0, audit_candidates=3, target_costs=[4.0, 1.0, 2.0])

    assert [candidate.chain for candidate in audited.decision_batch.candidates] == [[1], [2]]
    assert [candidate.chain for candidate in audited.audit_batch.candidates] == [[1], [2], [0]]
    session.discard(audited.decision_batch)

    with pytest.raises(ValueError, match="audit_candidates"):
        session.propose_with_audit(0, audit_candidates=0)


def test_applicable_scoring_batch_can_apply_beyond_default_resource_bound() -> None:
    session = SearchSession(
        GraphLike(["logical"], []),
        GraphLike(["left", "middle", "right"], [("left", "middle"), ("middle", "right")]),
        random_seed=17,
        max_candidates=1,
    )

    narrow = session.propose(0, target_costs=[4.0, 1.0, 2.0])
    assert [candidate.chain for candidate in narrow.candidates] == [[1]]
    session.discard(narrow)

    wide = session.propose_applicable(0, scoring_candidates=3, target_costs=[4.0, 1.0, 2.0])
    assert wide.session_id == session.session_id
    assert [candidate.candidate_id for candidate in wide.candidates] == [0, 1, 2]
    assert wide.candidates[2].chain == [0]

    session.apply(wide, 2)
    assert session.snapshot().chains == [[0]]
    with pytest.raises(RuntimeError, match="stale|consumed"):
        session.apply(wide, 2)


def test_native_candidate_validation_is_non_consuming_exact_and_session_local() -> None:
    source = GraphLike(["a", "b"], [("a", "b")])
    target = GraphLike(range(3), [(0, 1), (1, 2)])
    session = SearchSession.from_chains(
        source,
        target,
        ((0,), (2,)),
        random_seed=17,
        max_candidates=2,
    )
    batch = session.materialize(0, [[1, 2], [1]])
    overlap = next(index for index, item in enumerate(batch.candidates) if item.chain == [1, 2])
    valid = next(index for index, item in enumerate(batch.candidates) if item.chain == [1])

    assert not session.candidate_is_valid(batch, overlap)
    assert session.candidate_is_valid(batch, valid)

    other = SearchSession.from_chains(
        source,
        target,
        ((0,), (2,)),
        random_seed=17,
        max_candidates=2,
    )
    with pytest.raises(RuntimeError, match="stale|foreign"):
        other.candidate_is_valid(batch, valid)

    session.apply(batch, valid)
    assert session.snapshot().valid


def test_applicable_scoring_batch_rejects_stale_foreign_and_invalid_handles() -> None:
    graph = GraphLike(["left", "middle", "right"], [("left", "middle"), ("middle", "right")])
    session = SearchSession(GraphLike(["logical"], []), graph, random_seed=17, max_candidates=1)
    other = SearchSession(GraphLike(["logical"], []), graph, random_seed=17, max_candidates=1)

    stale = session.propose_applicable(0, scoring_candidates=3)
    session.restart()
    with pytest.raises(RuntimeError, match="stale|foreign"):
        session.apply(stale, 0)

    foreign = session.propose_applicable(0, scoring_candidates=3)
    destination = other.propose_applicable(0, scoring_candidates=3)
    with pytest.raises(RuntimeError, match="stale|foreign"):
        other.apply(foreign, 0)
    other.discard(destination)
    session.discard(foreign)

    wrong_index = session.propose_applicable(0, scoring_candidates=3)
    with pytest.raises(IndexError, match="outside"):
        session.apply(wrong_index, len(wrong_index.candidates))
    session.apply(wrong_index, 0)

    with pytest.raises(ValueError, match="scoring_candidates"):
        session.propose_applicable(0, scoring_candidates=0)


def test_native_graph_rejects_invalid_edges() -> None:
    with pytest.raises(ValueError, match="self-loop"):
        _core.Graph(1, [(0, 0)])


def test_session_fork_preserves_state_and_rng_but_branches_independently() -> None:
    source = GraphLike(["a", "b", "c"], [("a", "b"), ("b", "c")])
    target = GraphLike(range(5), [(0, 1), (1, 2), (2, 3), (3, 4)])
    session = SearchSession(source, target, random_seed=991, max_candidates=5)
    branch = session.fork()

    assert branch.snapshot().chains == session.snapshot().chains
    assert branch.snapshot().generation == session.snapshot().generation
    seeded_a = session.fork(random_seed=444)
    seeded_b = session.fork(random_seed=444)
    seeded_a.restart()
    seeded_b.restart()
    assert seeded_a.snapshot().chains == seeded_b.snapshot().chains
    session.restart()
    branch.restart()
    assert branch.snapshot().chains == session.snapshot().chains

    independent = SearchSession(
        GraphLike(["logical"], []), target, random_seed=991, max_candidates=5
    )
    independent_branch = independent.fork()
    parent_batch = independent.materialize(0, [[0]])
    branch_batch = independent_branch.materialize(0, [[4]])
    independent.apply(parent_batch, 0)
    independent_branch.apply(branch_batch, 0)
    assert independent.snapshot().chains[0] == [0]
    assert independent_branch.snapshot().chains[0] == [4]


def test_session_fork_rejects_outstanding_proposal_and_cross_branch_handle() -> None:
    # Use one isolated logical node; an empty edge iterable cannot encode it.
    session = SearchSession(
        GraphLike(["logical"], []),
        GraphLike(range(3), [(0, 1), (1, 2)]),
        random_seed=7,
        max_candidates=3,
    )
    branch = session.fork()
    batch = session.propose(0)

    with pytest.raises(RuntimeError, match="outstanding"):
        session.fork()
    with pytest.raises(RuntimeError, match="foreign|stale"):
        branch.apply(batch, 0)
    session.discard(batch)


def test_session_restore_from_chains_reconstructs_semantic_state_for_hpc_shards() -> None:
    source = GraphLike(["a", "b"], [("a", "b")])
    target = GraphLike(range(4), [(0, 1), (1, 2), (2, 3)])

    first = SearchSession.from_chains(
        source,
        target,
        ((0, 1), (2,)),
        random_seed=27101,
        max_candidates=3,
    )
    second = SearchSession.from_chains(
        source,
        target,
        ((0, 1), (2,)),
        random_seed=27101,
        max_candidates=3,
    )

    assert first.snapshot().chains == second.snapshot().chains == [[0, 1], [2]]
    assert first.snapshot().valid
    first.restart()
    second.restart()
    assert first.snapshot().chains == second.snapshot().chains

    with pytest.raises(ValueError, match="one chain"):
        SearchSession.from_chains(
            source,
            target,
            ((0,),),
            random_seed=1,
            max_candidates=3,
        )


def test_session_local_perturbation_is_deterministic_and_preserves_other_chains() -> None:
    source = GraphLike(range(5), [])
    target = GraphLike(range(8), [(node, node + 1) for node in range(7)])
    chains = ((0, 1), (2,), (3,), (4,), (5,))
    first = SearchSession.from_chains(source, target, chains, random_seed=808, max_candidates=4)
    second = SearchSession.from_chains(source, target, chains, random_seed=808, max_candidates=4)

    first.perturb([0, 2, 4])
    second.perturb([0, 2, 4])

    snapshot = first.snapshot()
    assert snapshot.chains == second.snapshot().chains
    assert snapshot.chains[1] == [2]
    assert snapshot.chains[3] == [4]
    assert all(len(snapshot.chains[logical]) == 1 for logical in (0, 2, 4))
    assert snapshot.max_occupancy == 1

    with pytest.raises(ValueError, match="duplicates"):
        first.perturb([0, 0])

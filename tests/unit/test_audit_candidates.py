from __future__ import annotations

from lac_minorminer import SearchSession, validate_embedding
from isingfold_lac_b import (
    Embedding,
    HardwareGraph,
    IsingProblem,
    ObjectiveContext,
    program_embedding,
    select_uniform_strength,
)


def _grid_edges(rows: int, columns: int) -> list[tuple[int, int]]:
    edges = []
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            if row + 1 < rows:
                edges.append((node, node + columns))
            if column + 1 < columns:
                edges.append((node, node + 1))
    return edges


def test_pretruncation_audit_exposes_a_downstream_better_route() -> None:
    source = [(0, 1), (0, 2), (0, 3)]
    target = _grid_edges(4, 4)
    session = SearchSession(source, target, random_seed=75, max_candidates=1)
    before = session.snapshot()
    assert session.target_labels != tuple(range(16))

    audited = session.propose_with_audit(0, audit_candidates=16)

    assert session.snapshot().chains == before.chains
    assert [candidate.candidate_id for candidate in audited.decision_batch.candidates] == [0]
    assert [candidate.candidate_id for candidate in audited.audit_batch.candidates[:2]] == [0, 1]
    assert audited.decision_batch.candidates[0].chain == audited.audit_batch.candidates[0].chain
    assert audited.audit_batch.candidates[0].chain == [9, 13, 14]
    assert audited.audit_batch.candidates[1].chain == [9, 12, 13, 14]

    context = ObjectiveContext(
        problem=IsingProblem(
            linear=(-2.0, -0.5, 1.0, 0.0),
            quadratic=((0, 1, -0.5), (0, 2, -0.5), (0, 3, 4.0)),
        ),
        hardware=HardwareGraph(num_nodes=16, edges=session.normalized_target.edges),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 6.0),
        q_cap=10,
    )
    downstream_y = []
    for candidate in audited.audit_batch.candidates[:2]:
        chains = [tuple(chain) for chain in before.chains]
        chains[0] = tuple(candidate.chain)
        validation = validate_embedding(
            session.normalized_source.edges,
            session.normalized_target.edges,
            {logical: list(chain) for logical, chain in enumerate(chains)},
        )
        assert validation.valid, validation.errors
        program = program_embedding(context, Embedding(tuple(chains)))
        downstream_y.append(select_uniform_strength(context, program).objective_value)

    assert downstream_y[0] == 0.5681505108531595
    assert downstream_y[1] == 0.576512827260232
    assert downstream_y[1] > downstream_y[0]

    session.discard(audited.decision_batch)
    assert session.snapshot().chains == before.chains

    applicable = session.propose_applicable(0, scoring_candidates=16)
    assert [candidate.candidate_id for candidate in applicable.candidates[:2]] == [0, 1]
    assert applicable.candidates[1].chain == audited.audit_batch.candidates[1].chain

    session.apply(applicable, 1)
    after = session.snapshot()
    assert after.chains[0] == [9, 12, 13, 14]
    assert after.valid

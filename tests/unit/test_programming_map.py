from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from isingfold_lac_b.objective import HardwareGraph, IsingProblem, ObjectiveContext
from isingfold_lac_b.programming import (
    Embedding,
    PhysicalProgram,
    audit_faithfulness,
    program_embedding,
)


def _context(*, q_cap: int = 5, with_cross_edges: bool = True) -> ObjectiveContext:
    edges = [(0, 1), (0, 2), (1, 2), (3, 4)]
    if with_cross_edges:
        edges.extend([(1, 3), (2, 4)])
    return ObjectiveContext(
        problem=IsingProblem(
            linear=(2.0, -1.0),
            quadratic=((0, 1, -3.0),),
        ),
        hardware=HardwareGraph(num_nodes=5, edges=tuple(edges)),
        beta_dev=1.5,
        j_max=1.0,
        f_domain=(0.0, 4.0),
        q_cap=q_cap,
    )


def test_default_programming_map_is_faithful_and_uses_available_chords() -> None:
    context = _context()
    embedding = Embedding(chains=((0, 1, 2), (3, 4)))

    program = program_embedding(context, embedding)

    assert program.active_nodes == (0, 1, 2, 3, 4)
    assert program.linear == pytest.approx((2 / 3, 2 / 3, 2 / 3, -0.5, -0.5))
    assert program.problem_quadratic == (
        (1, 3, -1.5),
        (2, 4, -1.5),
    )
    assert program.chain_edges_by_logical == (
        ((0, 1), (0, 2), (1, 2)),
        ((3, 4),),
    )
    assert program.total_chain_edges == 4
    assert audit_faithfulness(context.problem, program).faithful

    for logical_spins in itertools.product((-1, 1), repeat=2):
        physical_spins = program.consistent_spins(logical_spins)
        assert program.unpenalized_energy(physical_spins) == pytest.approx(
            context.problem.energy(logical_spins), abs=1e-12
        )
        assert program.energy(physical_spins, strength=1.25) == pytest.approx(
            context.problem.energy(logical_spins) - 1.25 * program.total_chain_edges,
            abs=1e-12,
        )
        assert program.induced_logical_spins(physical_spins) == logical_spins


def test_explicit_programmed_edges_can_select_a_spanning_tree() -> None:
    program = program_embedding(
        _context(),
        Embedding(
            chains=((0, 1, 2), (3, 4)),
            chain_edges=(((0, 1), (1, 2)), ((3, 4),)),
        ),
    )

    assert program.chain_edges_by_logical == (((0, 1), (1, 2)), ((3, 4),))
    assert program.total_chain_edges == 3
    assert audit_faithfulness(_context().problem, program).faithful


@pytest.mark.parametrize(
    "num_nodes, edges",
    [
        (1, ()),
        (4, ((0, 1), (1, 2), (2, 3))),
        (4, ((0, 1), (0, 2), (0, 3))),
        (4, ((0, 1), (0, 2), (0, 3), (1, 2))),
    ],
    ids=("singleton", "path", "branched", "chorded"),
)
def test_programming_is_faithful_across_registered_chain_topologies(
    num_nodes: int, edges: tuple[tuple[int, int], ...]
) -> None:
    context = ObjectiveContext(
        problem=IsingProblem(linear=(0.75,), quadratic=()),
        hardware=HardwareGraph(num_nodes=num_nodes, edges=edges),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 2.0),
        q_cap=num_nodes,
    )
    program = program_embedding(context, Embedding(chains=(tuple(range(num_nodes)),)))

    assert program.total_chain_edges == len(edges)
    assert audit_faithfulness(context.problem, program).faithful
    for logical_spin in (-1, 1):
        physical = program.consistent_spins((logical_spin,))
        assert program.unpenalized_energy(physical) == pytest.approx(
            context.problem.energy((logical_spin,))
        )


def test_broken_chain_state_reports_defects_and_has_no_induced_logical_state() -> None:
    program = program_embedding(_context(), Embedding(chains=((0, 1, 2), (3, 4))))
    spins = (1, -1, 1, -1, -1)

    assert program.broken_edge_count(spins) == 2
    assert not program.is_chain_consistent(spins)
    assert program.induced_logical_spins(spins) is None


def test_faithfulness_audit_detects_a_tampered_field_map() -> None:
    program = program_embedding(_context(), Embedding(chains=((0, 1, 2), (3, 4))))
    tampered = replace(program, linear=(program.linear[0] + 0.25, *program.linear[1:]))

    report = audit_faithfulness(_context().problem, tampered)

    assert not report.faithful
    assert any("linear" in error for error in report.errors)


def test_physical_program_digest_is_canonical_and_changes_with_programming() -> None:
    program = program_embedding(_context(), Embedding(chains=((0, 1, 2), (3, 4))))

    rebuilt = type(program)(**program.constructor_values())
    reordered = PhysicalProgram(
        active_nodes=program.active_nodes,
        chains=tuple(tuple(reversed(chain)) for chain in program.chains),
        linear=program.linear,
        problem_quadratic=tuple(reversed(program.problem_quadratic)),
        chain_edges_by_logical=tuple(
            tuple(reversed(edges)) for edges in program.chain_edges_by_logical
        ),
    )
    tampered = replace(program, linear=(program.linear[0] + 0.25, *program.linear[1:]))

    assert rebuilt.digest == program.digest
    assert reordered.digest == program.digest
    assert len(program.digest) == 64
    assert tampered.digest != program.digest


def test_physical_program_rejects_an_empty_chain_even_if_other_chains_cover_positions() -> None:
    with pytest.raises(ValueError, match="empty"):
        PhysicalProgram(
            active_nodes=(0,),
            chains=((0,), ()),
            linear=(0.0,),
            problem_quadratic=(),
            chain_edges_by_logical=((), ()),
        )


@pytest.mark.parametrize(
    "embedding, message",
    [
        (Embedding(chains=((0, 1, 2),)), "logical variables"),
        (Embedding(chains=((0, 1, 2), ())), "empty"),
        (Embedding(chains=((0, 1, 2), (2, 3))), "overlap"),
        (Embedding(chains=((0, 1, 5), (3, 4))), "range"),
        (Embedding(chains=((0, 1, 1), (3, 4))), "repeats"),
    ],
)
def test_invalid_chain_assignments_are_rejected(embedding: Embedding, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        program_embedding(_context(), embedding)


def test_disconnected_chain_and_missing_logical_coupler_are_rejected() -> None:
    disconnected = ObjectiveContext(
        problem=IsingProblem(linear=(0.0, 0.0), quadratic=((0, 1, -1.0),)),
        hardware=HardwareGraph(num_nodes=4, edges=((0, 2), (1, 3), (2, 3))),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 3.0),
        q_cap=4,
    )
    with pytest.raises(ValueError, match="connected"):
        program_embedding(disconnected, Embedding(chains=((0, 1), (2, 3))))

    with pytest.raises(ValueError, match="not realized"):
        program_embedding(
            _context(with_cross_edges=False),
            Embedding(chains=((0, 1, 2), (3, 4))),
        )


def test_qubit_cap_and_invalid_explicit_chain_edges_are_rejected() -> None:
    with pytest.raises(ValueError, match="q_cap"):
        program_embedding(_context(q_cap=4), Embedding(chains=((0, 1, 2), (3, 4))))

    invalid_edges = [
        ((((0, 1), (1, 3)), ((3, 4),)), "inside its chain"),
        ((((0, 1), (1, 2), (2, 0)), ((3, 4),)), "canonical"),
        ((((0, 1),), ((3, 4),)), "connected"),
        ((((0, 1), (1, 2), (0, 4)), ((3, 4),)), "inside its chain"),
    ]
    for chain_edges, message in invalid_edges:
        with pytest.raises(ValueError, match=message):
            program_embedding(
                _context(),
                Embedding(chains=((0, 1, 2), (3, 4)), chain_edges=chain_edges),
            )

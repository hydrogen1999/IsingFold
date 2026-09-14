from __future__ import annotations

import itertools
from dataclasses import replace

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context, OverlapProfile, WorkVector
from isingfold.rl.evaluator import logical_energy
from isingfold.rl.program import check_faithfulness, compile_registry, strength_registry
from isingfold.rl.validate import p_embed, p_return, p_search


def _tiny_problem() -> tuple[nx.Graph, nx.Graph, LogicalProblem, dict[int, frozenset[int]]]:
    logical = nx.Graph([(0, 1)])
    host = nx.Graph([(0, 1), (1, 2), (2, 3)])
    problem = LogicalProblem.from_dicts({0: 0.25, 1: -0.5}, {(0, 1): -1.25})
    chains = {0: frozenset({0, 1}), 1: frozenset({2, 3})}
    return logical, host, problem, chains


def test_search_overlap_is_not_mistaken_for_embedding_validity() -> None:
    logical, host, problem, _ = _tiny_problem()
    chains = {0: frozenset({1}), 1: frozenset({1})}
    overlap = OverlapProfile(name="O1", max_occupancy=2, excess_fraction_of_qubit_cap=0.25)

    assert p_search(chains, logical, host, 4, overlap).valid
    assert not p_embed(chains, logical, host, 4).valid
    assert not p_return(chains, logical, host, problem, Context(qubit_cap=4))[0].valid


@pytest.mark.parametrize(
    ("chains", "qubit_cap", "overlap"),
    [
        (
            {0: frozenset({0, 1}), 1: frozenset({0, 1})},
            4,
            OverlapProfile(name="O1", max_occupancy=2, excess_fraction_of_qubit_cap=0.25),
        ),
        (
            {0: frozenset({0}), 1: frozenset({0}), 2: frozenset({0})},
            10,
            OverlapProfile(name="O1", max_occupancy=2, excess_fraction_of_qubit_cap=0.10),
        ),
    ],
)
def test_search_rejects_each_overlap_envelope_violation(
    chains: dict[int, frozenset[int]], qubit_cap: int, overlap: OverlapProfile
) -> None:
    logical = nx.empty_graph(len(chains))
    host = nx.path_graph(4)

    assert not p_search(chains, logical, host, qubit_cap, overlap).valid


def test_compiled_program_obeys_aligned_energy_identity_at_all_assignments() -> None:
    _, host, problem, chains = _tiny_problem()
    strengths = strength_registry(problem, (0.5, 1.0, 2.0, 4.0))

    for program in compile_registry(chains, host, problem, strengths):
        internal_edges = sum(len(edges) for edges in program.chain_edges.values())
        expected_offset = -program.scale * program.strength * internal_edges
        assert program.offset == pytest.approx(expected_offset)

        for values in itertools.product((-1, 1), repeat=problem.n):
            logical_spins = dict(zip(sorted(problem.graph), values, strict=True))
            physical_spins = {
                qubit: logical_spins[node]
                for node, chain in chains.items()
                for qubit in chain
            }
            physical_energy = sum(
                bias * physical_spins[qubit] for qubit, bias in program.h_phys.items()
            ) + sum(
                coupling * physical_spins[u] * physical_spins[v]
                for (u, v), coupling in program.j_phys.items()
            )
            expected = program.scale * (
                logical_energy(problem, logical_spins) - program.strength * internal_edges
            )
            assert physical_energy == pytest.approx(expected)


def test_return_contract_compiles_exactly_four_faithful_programs() -> None:
    logical, host, problem, chains = _tiny_problem()
    receipt, programs = p_return(chains, logical, host, problem, Context(qubit_cap=4))

    assert receipt.valid
    assert len(programs) == 4
    assert len(receipt.programs) == 4
    assert all(report.ok for report in receipt.programs)


def test_return_rejects_a_logical_graph_that_disagrees_with_problem_support() -> None:
    logical, host, _, chains = _tiny_problem()
    problem_without_coupling = LogicalProblem.from_dicts({0: 0.25, 1: -0.5}, {})

    receipt, programs = p_return(
        chains,
        logical,
        host,
        problem_without_coupling,
        Context(qubit_cap=4),
    )

    assert not receipt.valid
    assert not programs
    assert any("problem" in reason for reason in receipt.reasons)


@pytest.mark.parametrize("mutation", ["internal_sign", "off_host", "extra_field", "offset"])
def test_faithfulness_rejects_corrupted_program_support_and_coefficients(mutation: str) -> None:
    _, host, problem, chains = _tiny_problem()
    program = compile_registry(
        chains,
        host,
        problem,
        strength_registry(problem, (1.0,)),
    )[0]
    if mutation == "internal_sign":
        edge = next(iter(program.chain_edges[0]))
        changed = dict(program.j_phys)
        changed[edge] = abs(changed[edge])
        corrupted = replace(program, j_phys=changed)
    elif mutation == "off_host":
        changed = dict(program.j_phys)
        changed[(0, 99)] = 0.5
        corrupted = replace(program, j_phys=changed)
    elif mutation == "extra_field":
        changed = dict(program.h_phys)
        changed[99] = 0.5
        corrupted = replace(program, h_phys=changed)
    else:
        corrupted = replace(program, offset=program.offset + 1.0)

    report = check_faithfulness(corrupted, chains, host, problem)

    assert not report.ok


def test_negative_binding_budget_is_a_search_contract_violation() -> None:
    logical, host, _, chains = _tiny_problem()
    remaining = WorkVector(decisions=-1)

    receipt = p_search(
        chains,
        logical,
        host,
        qubit_cap=4,
        overlap=OverlapProfile(name="O0", max_occupancy=1, excess_fraction_of_qubit_cap=0),
        remaining=remaining,
    )

    assert not receipt.valid
    assert any("negative" in reason for reason in receipt.reasons)

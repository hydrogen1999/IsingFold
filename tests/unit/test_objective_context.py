from __future__ import annotations

import math

import pytest

from isingfold_lac_b.objective import HardwareGraph, IsingProblem, ObjectiveContext


def test_problem_energy_and_context_digest_are_canonical() -> None:
    problem = IsingProblem(
        linear=(0.5, -1.0),
        quadratic=((0, 1, -2.0),),
    )
    hardware = HardwareGraph(
        num_nodes=4,
        edges=((2, 3), (0, 1), (1, 2)),
    )
    context = ObjectiveContext(
        problem=problem,
        hardware=hardware,
        beta_dev=2.0,
        j_max=1.5,
        f_domain=(0.0, 4.5),
        q_cap=4,
    )

    assert problem.energy((1, -1)) == pytest.approx(3.5)
    assert hardware.edges == ((0, 1), (1, 2), (2, 3))
    assert context.objective_id == "scaled-gibbs-exact-no-decode-v1"
    assert context.autoscaling_id == "canonical-max-v1"
    assert context.scale_divisor(0.5, chain_edge_count=1) == 1.0
    assert context.scale_divisor(3.0, chain_edge_count=1) == 2.0
    assert context.digest == ObjectiveContext(**context.constructor_values()).digest
    assert len(context.digest) == 64


def test_unused_chain_strength_does_not_rescale_all_singleton_program() -> None:
    context = ObjectiveContext(
        problem=IsingProblem(linear=(0.25,), quadratic=()),
        hardware=HardwareGraph(num_nodes=1, edges=()),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 8.0),
        q_cap=1,
    )

    assert context.scale_divisor(8.0, chain_edge_count=0) == 1.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"linear": (), "quadratic": ()}, "at least one"),
        ({"linear": (0.0,), "quadratic": ((0, 0, 1.0),)}, "self-loop"),
        ({"linear": (0.0, 0.0), "quadratic": ((1, 0, 1.0),)}, "canonical"),
        (
            {
                "linear": (0.0, 0.0),
                "quadratic": ((0, 1, 1.0), (0, 1, -1.0)),
            },
            "duplicate",
        ),
        ({"linear": (math.inf,), "quadratic": ()}, "finite"),
        ({"linear": (0.0, 0.0), "quadratic": ((0, 2, 1.0),)}, "range"),
    ],
)
def test_invalid_ising_problem_is_rejected(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IsingProblem(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_nodes": 0, "edges": ()}, "positive"),
        ({"num_nodes": 2, "edges": ((0, 0),)}, "self-loop"),
        ({"num_nodes": 2, "edges": ((1, 0),)}, "canonical"),
        ({"num_nodes": 2, "edges": ((0, 2),)}, "range"),
        ({"num_nodes": 2, "edges": ((0, 1), (0, 1))}, "duplicate"),
    ],
)
def test_invalid_hardware_graph_is_rejected(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HardwareGraph(**kwargs)


def test_invalid_objective_context_and_strength_are_rejected() -> None:
    problem = IsingProblem(linear=(0.0, 0.0), quadratic=((0, 1, -1.0),))
    hardware = HardwareGraph(num_nodes=3, edges=((0, 1), (1, 2)))

    invalid = [
        ({"beta_dev": 0.0}, "beta_dev"),
        ({"j_max": math.nan}, "j_max"),
        ({"f_domain": (-1.0, 2.0)}, "f_domain"),
        ({"f_domain": (2.0, 1.0)}, "f_domain"),
        ({"q_cap": 1}, "logical variables"),
        ({"q_cap": 4}, "hardware nodes"),
        ({"autoscaling_id": "mystery"}, "autoscaling"),
    ]
    for override, message in invalid:
        values = {
            "problem": problem,
            "hardware": hardware,
            "beta_dev": 1.0,
            "j_max": 1.0,
            "f_domain": (0.0, 3.0),
            "q_cap": 3,
        }
        values.update(override)
        with pytest.raises(ValueError, match=message):
            ObjectiveContext(**values)

    context = ObjectiveContext(
        problem=problem,
        hardware=hardware,
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 3.0),
        q_cap=3,
    )
    with pytest.raises(ValueError, match="strength"):
        context.scale_divisor(3.5, chain_edge_count=1)
    with pytest.raises(ValueError, match="chain_edge_count"):
        context.scale_divisor(1.0, chain_edge_count=-1)

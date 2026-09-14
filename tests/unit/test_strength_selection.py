from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

from isingfold_lac_b.evaluation import ExactEvaluator
from isingfold_lac_b.evaluation.strength import (
    binding_log_solve_derivative,
    select_uniform_strength,
)
from isingfold_lac_b.objective import HardwareGraph, IsingProblem, ObjectiveContext
from isingfold_lac_b.programming import Embedding, program_embedding


def _interior_context(*, f_domain: tuple[float, float] = (0.0, 8.0)) -> ObjectiveContext:
    return ObjectiveContext(
        problem=IsingProblem(linear=(-1.0, -1.0), quadratic=((0, 1, 1.0),)),
        hardware=HardwareGraph(num_nodes=3, edges=((0, 1), (0, 2))),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=f_domain,
        q_cap=3,
    )


def _single_chain_context() -> ObjectiveContext:
    return ObjectiveContext(
        problem=IsingProblem(linear=(-1.0,), quadratic=()),
        hardware=HardwareGraph(num_nodes=2, edges=((0, 1),)),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 4.0),
        q_cap=2,
    )


def test_all_singleton_program_chooses_smallest_admissible_strength() -> None:
    context = ObjectiveContext(
        problem=IsingProblem(linear=(-0.5,), quadratic=()),
        hardware=HardwareGraph(num_nodes=1, edges=()),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 5.0),
        q_cap=1,
    )
    program = program_embedding(context, Embedding(chains=((0,),)))

    selected = select_uniform_strength(context, program)

    assert selected.f_star == 0.0
    assert selected.location == "all_singleton_minimum"
    assert selected.certified_log_regret_bound == 0.0
    assert selected.objective_value == selected.evaluation.p_solve


def test_scaling_knee_is_selected_when_right_derivative_is_nonpositive() -> None:
    context = _single_chain_context()
    program = program_embedding(context, Embedding(chains=((0, 1),)))

    selected = select_uniform_strength(context, program)

    assert selected.f_star == context.j_max
    assert selected.location == "scaling_knee"
    assert selected.balance_at_selection < 0.0
    assert selected.certified_log_regret_bound == 0.0


def test_binding_branch_interior_optimum_has_a_certificate_and_matches_dense_grid() -> None:
    context = _interior_context()
    program = program_embedding(context, Embedding(chains=((0, 1), (2,))))
    evaluator = ExactEvaluator()

    selected = select_uniform_strength(
        context,
        program,
        evaluator=evaluator,
        strength_tolerance=1e-9,
        log_regret_tolerance=1e-11,
    )

    dense_best = max(
        evaluator.evaluate(context, program, strength=index * 8.0 / 4000).p_solve
        for index in range(4001)
    )
    assert selected.location == "interior_binding"
    assert selected.f_star == pytest.approx(1.2232774344576147, abs=2e-8)
    assert abs(selected.balance_at_selection) < 1e-7
    assert selected.objective_value >= dense_best - 1e-9
    assert selected.certified_log_regret_bound <= 1e-10
    assert selected.final_bracket[0] <= selected.f_star <= selected.final_bracket[1]


def test_upper_and_lower_binding_boundaries_are_detected() -> None:
    upper_context = _interior_context(f_domain=(0.0, 1.1))
    upper_program = program_embedding(upper_context, Embedding(chains=((0, 1), (2,))))
    upper = select_uniform_strength(upper_context, upper_program)

    lower_context = _interior_context(f_domain=(1.5, 8.0))
    lower_program = program_embedding(lower_context, Embedding(chains=((0, 1), (2,))))
    lower = select_uniform_strength(lower_context, lower_program)

    assert upper.f_star == 1.1
    assert upper.location == "upper_boundary_binding"
    assert upper.balance_at_selection > 0.0
    assert lower.f_star == 1.5
    assert lower.location == "lower_boundary_binding"
    assert lower.balance_at_selection < 0.0


def test_pre_knee_probability_is_monotone_and_domain_maximum_is_selected() -> None:
    base = _single_chain_context()
    context = ObjectiveContext(**{**base.constructor_values(), "f_domain": (0.1, 0.8)})
    program = program_embedding(context, Embedding(chains=((0, 1),)))
    evaluator = ExactEvaluator()

    probabilities = [
        evaluator.evaluate(context, program, strength=strength).p_solve
        for strength in (0.1, 0.3, 0.5, 0.8)
    ]
    selected = select_uniform_strength(context, program, evaluator=evaluator)

    assert probabilities == sorted(probabilities)
    assert len(set(probabilities)) == len(probabilities)
    assert selected.f_star == 0.8
    assert selected.location == "upper_boundary_pre_knee"


def test_binding_derivative_matches_a_centered_finite_difference() -> None:
    context = _interior_context()
    program = program_embedding(context, Embedding(chains=((0, 1), (2,))))
    evaluator = ExactEvaluator()
    strength = 2.0
    step = 1e-5

    at_strength = evaluator.evaluate(context, program, strength=strength)
    derivative = binding_log_solve_derivative(context, at_strength)
    finite_difference = (
        evaluator.evaluate(context, program, strength=strength + step).log_p_solve
        - evaluator.evaluate(context, program, strength=strength - step).log_p_solve
    ) / (2.0 * step)

    assert derivative == pytest.approx(finite_difference, rel=1e-8, abs=1e-10)


def test_reported_regret_bound_dominates_dense_grid_on_generated_tiny_cases() -> None:
    generator = random.Random(20260904)
    evaluator = ExactEvaluator()
    hardware = HardwareGraph(num_nodes=3, edges=((0, 1), (0, 2)))

    for _ in range(12):
        context = ObjectiveContext(
            problem=IsingProblem(
                linear=(generator.uniform(-1.0, 1.0), generator.uniform(-1.0, 1.0)),
                quadratic=((0, 1, generator.uniform(-1.0, 1.0)),),
            ),
            hardware=hardware,
            beta_dev=generator.uniform(0.25, 2.0),
            j_max=1.0,
            f_domain=(0.0, 4.0),
            q_cap=3,
        )
        program = program_embedding(context, Embedding(chains=((0, 1), (2,))))

        selected = select_uniform_strength(context, program, evaluator=evaluator)
        dense_best_log = max(
            evaluator.evaluate(context, program, strength=index / 200.0).log_p_solve
            for index in range(801)
        )

        assert dense_best_log <= (
            selected.log_objective_value + selected.certified_log_regret_bound + 1e-10
        )


def test_unfaithful_program_refuses_theory_derived_strength_shortcuts() -> None:
    context = _single_chain_context()
    program = program_embedding(context, Embedding(chains=((0, 1),)))
    tampered = replace(program, linear=(1.5, -0.5))
    evaluation = ExactEvaluator().evaluate(context, tampered, strength=2.0)

    assert not evaluation.audit.faithful_programming
    with pytest.raises(ValueError, match="faithful"):
        binding_log_solve_derivative(context, evaluation)
    with pytest.raises(ValueError, match="faithful"):
        select_uniform_strength(context, tampered)


def test_iteration_exhaustion_fails_closed_before_returning_an_uncertified_label() -> None:
    context = _interior_context()
    program = program_embedding(context, Embedding(chains=((0, 1), (2,))))

    with pytest.raises(RuntimeError, match="converge"):
        select_uniform_strength(
            context,
            program,
            strength_tolerance=1e-15,
            log_regret_tolerance=1e-15,
            max_iterations=1,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"strength_tolerance": 0.0}, "strength_tolerance"),
        ({"log_regret_tolerance": math.nan}, "log_regret_tolerance"),
        ({"max_iterations": 0}, "max_iterations"),
    ],
)
def test_invalid_optimizer_controls_are_rejected(kwargs, message: str) -> None:
    context = _single_chain_context()
    program = program_embedding(context, Embedding(chains=((0, 1),)))

    with pytest.raises(ValueError, match=message):
        select_uniform_strength(context, program, **kwargs)

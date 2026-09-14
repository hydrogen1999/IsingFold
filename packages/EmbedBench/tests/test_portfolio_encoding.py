from __future__ import annotations

import numpy as np
import pytest
from embedbench.apps import Instance, portfolio

PRODUCTION_ASSET_COUNTS = (6, 8, 10, 12, 14, 16)


def _state_table(instance: Instance) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    encoding = instance.derived["encoding"]
    n_assets = instance.kwargs["n_assets"]
    masks = np.arange(1 << n_assets, dtype=np.uint32)[:, None]
    bits = ((masks >> np.arange(n_assets, dtype=np.uint32)) & 1).astype(np.int64)

    qubo = np.full(len(bits), encoding["binary_offset_units"], dtype=np.int64)
    for node, units in encoding["qubo_linear_units"]:
        qubo += units * bits[:, node]
    for left, right, units in encoding["qubo_quadratic_units"]:
        qubo += units * bits[:, left] * bits[:, right]

    denominator = encoding["ising_coefficient_denominator"]
    spins = 2 * bits - 1
    ising_with_offset = np.full(len(bits), encoding["ising_offset_units"], dtype=np.int64)
    for node, coefficient in instance.problem.h.items():
        units = int(round(coefficient * denominator))
        assert coefficient == units / denominator
        ising_with_offset += units * spins[:, node]
    for (left, right), coefficient in instance.problem.j.items():
        units = int(round(coefficient * denominator))
        assert coefficient == units / denominator
        ising_with_offset += units * spins[:, left] * spins[:, right]

    covariance = np.asarray(instance.derived["covariance"]["sigma_units"], dtype=np.int64)
    expected_return = np.asarray(
        instance.derived["objective"]["expected_return_units"], dtype=np.int64
    )
    risk_units = instance.derived["objective"]["risk_units"]
    direct_objective = risk_units * np.einsum(
        "bi,ij,bj->b", bits, covariance, bits, optimize=True
    ) - bits @ expected_return
    valid = bits.sum(axis=1) == instance.derived["cardinality"]
    return qubo, ising_with_offset, direct_objective, valid


@pytest.mark.parametrize("n_assets", PRODUCTION_ASSET_COUNTS)
@pytest.mark.parametrize("seed", (0, 7, 31))
def test_portfolio_exact_qubo_ising_and_constrained_argmin(
    n_assets: int,
    seed: int,
) -> None:
    instance = portfolio(
        n_assets=n_assets,
        n_factors=min(3, n_assets),
        seed=seed,
    )
    qubo, ising_with_offset, direct_objective, valid = _state_table(instance)
    encoding = instance.derived["encoding"]

    loadings = np.asarray(instance.derived["covariance"]["factor_loadings"], dtype=np.int64)
    diagonal = np.asarray(
        instance.derived["covariance"]["idiosyncratic_diagonal"], dtype=np.int64
    )
    sigma = np.asarray(instance.derived["covariance"]["sigma_units"], dtype=np.int64)
    assert np.array_equal(sigma, loadings @ loadings.T + np.diag(diagonal))
    assert np.linalg.eigvalsh(sigma.astype(float)).min() > 0.0

    assert np.array_equal(4 * qubo, ising_with_offset)
    assert valid.any()
    assert np.array_equal(qubo[valid], direct_objective[valid])
    assert qubo.min() == direct_objective[valid].min()
    assert np.all(valid[qubo == qubo.min()])
    assert np.all(qubo[~valid] > encoding["feasible_witness_objective_units"])
    assert encoding["penalty_units"] == (
        encoding["feasible_witness_objective_units"]
        - encoding["global_objective_lower_bound_units"]
        + 1
    )


def test_portfolio_semantic_seed_grid_has_distinct_full_psd_problems() -> None:
    signatures = {
        (
            tuple(sorted(instance.problem.h.items())),
            tuple(sorted(instance.problem.j.items())),
        )
        for seed in range(64)
        for instance in (portfolio(n_assets=6, n_factors=3, seed=seed),)
    }

    assert len(signatures) == 64

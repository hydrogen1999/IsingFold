from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from embedbench.apps import graphcut_mrf


@pytest.mark.parametrize("connectivity", (4, 8))
def test_graphcut_ising_equals_twice_squared_unary_plus_potts(
    connectivity: int,
) -> None:
    rows, cols, seed, noise = 2, 3, 11, 0.6
    instance = graphcut_mrf(
        rows=rows,
        cols=cols,
        seed=seed,
        noise=noise,
        connectivity=connectivity,
    )
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    cy, cx = (rows - 1) / 2, (cols - 1) / 2
    radius = 0.30 * min(rows, cols)
    latent = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    observed = latent.astype(float) + rng.normal(0.0, noise, size=latent.shape)
    flattened = observed.reshape(-1)
    weights = {edge: -coefficient for edge, coefficient in instance.problem.j.items()}
    offset = float(np.sum(1.0 - 2.0 * flattened + 2.0 * flattened**2)) + sum(
        weights.values()
    )

    for raw_spins in product((-1, 1), repeat=rows * cols):
        spins = np.asarray(raw_spins, dtype=np.int64)
        labels = (spins + 1) // 2
        ising_energy = sum(
            coefficient * spins[node]
            for node, coefficient in instance.problem.h.items()
        ) + sum(
            coefficient * spins[left] * spins[right]
            for (left, right), coefficient in instance.problem.j.items()
        )
        application_energy = 2.0 * float(np.sum((labels - flattened) ** 2))
        application_energy += 2.0 * sum(
            weight * (labels[left] != labels[right])
            for (left, right), weight in weights.items()
        )
        assert application_energy == pytest.approx(ising_energy + offset, abs=1e-12)


def test_graphcut_latent_disc_is_not_claimed_as_a_planted_optimum() -> None:
    instance = graphcut_mrf(rows=3, cols=3, seed=0, connectivity=4)
    latent_spins = (-1, -1, -1, -1, 1, -1, -1, -1, -1)

    def energy(spins: tuple[int, ...]) -> float:
        return sum(
            coefficient * spins[node]
            for node, coefficient in instance.problem.h.items()
        ) + sum(
            coefficient * spins[left] * spins[right]
            for (left, right), coefficient in instance.problem.j.items()
        )

    optimum = min(energy(spins) for spins in product((-1, 1), repeat=9))
    assert energy(latent_spins) > optimum

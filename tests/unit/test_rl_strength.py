from __future__ import annotations

import numpy as np
import pytest

from isingfold.rl.strength import (
    FEATURE_ORDER,
    StrengthRecord,
    feature_vector,
    fit_selector,
)


def _features(strength: float) -> dict[str, float]:
    return {
        "strength": strength,
        "scale": 1.0,
        "qubits": 8.0,
        "max_chain": 3.0,
        "mean_chain": 2.0,
        "single_qubit_fraction": 0.25,
        "max_field": 0.5,
        "max_coupling": 1.5,
        "mean_contacts": 1.25,
        "single_contact_fraction": 0.75,
        "strength_over_jmax": strength / 1.5,
        "chain_edges": 5.0,
    }


def _record(hits: tuple[int, int, int, int]) -> StrengthRecord:
    return StrengthRecord(
        features=tuple(_features(value) for value in (0.5, 1.0, 2.0, 4.0)),
        hits=hits,
        reads=(16, 16, 16, 16),
        lineage="lineage-0",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hits": (1, 2, 3), "reads": (16, 16, 16)},
        {"hits": (1, 2, 3, 17), "reads": (16, 16, 16, 16)},
        {"hits": (1, 2, 3, 4), "reads": (16, 16, 0, 16)},
    ],
)
def test_strength_record_requires_four_complete_valid_integer_count_pairs(kwargs) -> None:
    with pytest.raises(ValueError):
        StrengthRecord(
            features=tuple(_features(value) for value in (0.5, 1.0, 2.0, 4.0)),
            lineage="bad",
            **kwargs,
        )


def test_selector_fit_is_finite_for_zero_and_all_hit_blocks() -> None:
    model = fit_selector([_record((0, 0, 0, 0)), _record((16, 16, 16, 16))], epochs=8)

    predictions = model.predict(_record((0, 0, 0, 0)).features)

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()
    assert ((0.0 <= predictions) & (predictions <= 1.0)).all()
    assert model.frozen


def test_feature_vector_uses_an_exact_deployable_allowlist() -> None:
    features = _features(1.0)
    baseline = feature_vector(features, 1)
    poisoned = dict(features)
    poisoned.update(
        {
            "ground_energy": -123.0,
            "evaluator_hits": 16.0,
            "witness_membership": 1.0,
            "random_key": 777.0,
        }
    )

    assert np.array_equal(feature_vector(poisoned, 1), baseline)
    assert baseline.shape == (len(FEATURE_ORDER) + 4,)


def test_missing_deployable_feature_is_rejected_instead_of_imputed() -> None:
    features = _features(1.0)
    del features[FEATURE_ORDER[0]]

    with pytest.raises(ValueError, match="missing"):
        feature_vector(features, 0)


def test_exact_tie_breaks_to_smallest_strength_index() -> None:
    model = fit_selector([_record((8, 8, 8, 8))], epochs=0)

    assert model.select((), _record((8, 8, 8, 8)).features) == 0

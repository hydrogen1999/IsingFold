from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from embedbench.hamiltonian_context import (
    HAMILTONIAN_CONTEXT_DIMENSION,
    HAMILTONIAN_CONTEXT_NAMES,
    HAMILTONIAN_CONTEXT_SCHEMA,
    HAMILTONIAN_CONTEXT_SCHEMA_VERSION,
    encode_hamiltonian_context,
    hamiltonian_context_contract,
)


def _record() -> dict:
    return {
        "focus": 0,
        "p_solve": [0.1, 0.9],
        "best_index": 1,
        "resource_index": 0,
        "problem": {
            "h": {"0": 0.25, "1": -0.5, "2": 1.25, "3": 0.0},
            "J": [[0, 1, -1.0], [1, 2, 0.75], [2, 3, -0.4], [1, 3, 0.2]],
            "e0": -3.5,
        },
    }


def test_context_has_a_fixed_self_describing_finite_float32_contract() -> None:
    context = encode_hamiltonian_context(_record())

    assert context.schema == HAMILTONIAN_CONTEXT_SCHEMA
    assert context.schema_version == HAMILTONIAN_CONTEXT_SCHEMA_VERSION == 1
    assert context.dimension == HAMILTONIAN_CONTEXT_DIMENSION
    assert context.names == HAMILTONIAN_CONTEXT_NAMES
    assert len(context.names) == context.dimension
    assert len(set(context.names)) == context.dimension
    assert context.values.shape == (context.dimension,)
    assert context.values.dtype == np.float32
    assert np.isfinite(context.values).all()
    assert context.values.flags.writeable is False
    assert hamiltonian_context_contract() == {
        "schema": HAMILTONIAN_CONTEXT_SCHEMA,
        "schema_version": HAMILTONIAN_CONTEXT_SCHEMA_VERSION,
        "dimension": HAMILTONIAN_CONTEXT_DIMENSION,
        "feature_names": list(HAMILTONIAN_CONTEXT_NAMES),
        "source": "problem.h_and_problem.J",
        "uses_ground_state_energy": False,
    }


def test_context_is_invariant_to_mapping_edge_order_and_edge_orientation() -> None:
    first = _record()
    second = deepcopy(first)
    second["problem"]["h"] = {"3": 0.0, "1": -0.5, "0": 0.25, "2": 1.25}
    second["problem"]["J"] = [
        [3, 1, 0.2],
        [3, 2, -0.4],
        [2, 1, 0.75],
        [1, 0, -1.0],
    ]

    np.testing.assert_array_equal(
        encode_hamiltonian_context(first).values,
        encode_hamiltonian_context(second).values,
    )


def test_context_uses_non_focus_fields_and_couplers_from_the_full_problem() -> None:
    base = _record()
    changed_non_focus_h = deepcopy(base)
    changed_non_focus_h["problem"]["h"]["2"] = 4.5
    changed_non_focus_j = deepcopy(base)
    changed_non_focus_j["problem"]["J"][2][2] = -2.75

    encoded = encode_hamiltonian_context(base).values
    assert not np.array_equal(
        encoded,
        encode_hamiltonian_context(changed_non_focus_h).values,
    )
    assert not np.array_equal(
        encoded,
        encode_hamiltonian_context(changed_non_focus_j).values,
    )


def test_context_ignores_ground_truth_labels_and_baseline_metadata() -> None:
    first = _record()
    second = deepcopy(first)
    second.update(
        focus=3,
        p_solve=[0.99, 0.01],
        best_index=0,
        resource_index=1,
        original_index=1,
        stage=[2, 1],
    )
    second["problem"]["e0"] = 999.0

    np.testing.assert_array_equal(
        encode_hamiltonian_context(first).values,
        encode_hamiltonian_context(second).values,
    )


def test_signed_cycle_summaries_distinguish_balanced_and_frustrated_triangles() -> None:
    balanced = {
        "problem": {
            "h": {"0": 0.0, "1": 0.0, "2": 0.0},
            "J": [[0, 1, -1.0], [1, 2, -1.0], [2, 0, -1.0]],
        }
    }
    frustrated = deepcopy(balanced)
    frustrated["problem"]["J"] = [
        [0, 1, 1.0],
        [1, 2, 1.0],
        [2, 0, 1.0],
    ]
    by_name = {name: index for index, name in enumerate(HAMILTONIAN_CONTEXT_NAMES)}

    balanced_values = encode_hamiltonian_context(balanced).values
    frustrated_values = encode_hamiltonian_context(frustrated).values

    assert balanced_values[by_name["frustrated_triangle_fraction"]] == 0.0
    assert frustrated_values[by_name["frustrated_triangle_fraction"]] == 1.0
    assert balanced_values[by_name["nodes_in_sign_frustrated_components_fraction"]] == 0.0
    assert frustrated_values[by_name["nodes_in_sign_frustrated_components_fraction"]] == 1.0


def test_extreme_finite_coefficients_still_produce_finite_float32_features() -> None:
    context = encode_hamiltonian_context(
        {
            "problem": {
                "h": {"0": 1e308, "1": -1e308},
                "J": [[0, 1, -1e308]],
            }
        }
    )

    assert np.isfinite(context.values).all()


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (None, "record must be a mapping"),
        ({}, "record requires problem"),
        ({"problem": []}, "problem must be a mapping"),
        ({"problem": {"J": []}}, "problem requires h"),
        ({"problem": {"h": {"0": 0.0}}}, "problem requires J"),
        ({"problem": {"h": {}, "J": []}}, "at least one logical variable"),
        (
            {"problem": {"h": {"01": 0.0}, "J": []}},
            "canonical integer string",
        ),
        (
            {"problem": {"h": {"0": float("nan")}, "J": []}},
            "finite real number",
        ),
        (
            {"problem": {"h": {"0": 0.0}, "J": [[0, 0, -1.0]]}},
            "self-coupling",
        ),
        (
            {"problem": {"h": {"0": 0.0}, "J": [[0, 1, -1.0]]}},
            "absent from h",
        ),
        (
            {
                "problem": {
                    "h": {"0": 0.0, "1": 0.0},
                    "J": [[0, 1, -1.0], [1, 0, -1.0]],
                }
            },
            "duplicate undirected coupling",
        ),
        (
            {
                "problem": {
                    "h": {"0": 0.0, "1": 0.0},
                    "J": [[0, 1, float("inf")]],
                }
            },
            "finite real number",
        ),
        (
            {
                "problem": {
                    "h": {"0": 0.0, "1": 0.0},
                    "J": [[0, 1]],
                }
            },
            "three-item sequence",
        ),
    ],
)
def test_malformed_problem_inputs_fail_closed(record, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        encode_hamiltonian_context(record)

from __future__ import annotations

import copy
import inspect

import networkx as nx
import pytest


def _case():
    logical = nx.path_graph(3)
    host = nx.cycle_graph(8)
    incumbent = {0: [0], 1: [1], 2: [2]}
    return logical, host, incumbent


def test_repair_once_is_deterministic_plain_and_does_not_mutate_inputs() -> None:
    from isingfold.repair import repair_once

    logical, host, incumbent = _case()
    frozen = copy.deepcopy(incumbent)

    first = repair_once(
        logical,
        host,
        incumbent,
        neighborhood=(1, 2),
        random_seed=71,
        max_candidates=8,
        max_transitions=40,
    )
    second = repair_once(
        logical,
        host,
        incumbent,
        neighborhood=(1, 2),
        random_seed=71,
        max_candidates=8,
        max_transitions=40,
    )

    assert first == second
    assert set(first) == {"chains", "transitions", "reason"}
    assert isinstance(first["transitions"], int)
    assert isinstance(first["reason"], str)
    assert incumbent == frozen
    if first["chains"] is not None:
        from lac_minorminer.validation import validate_embedding

        assert validate_embedding(logical, host, first["chains"]).valid


def test_repair_once_api_has_no_benchmark_or_label_arguments() -> None:
    from isingfold.repair import repair_once

    forbidden = {
        "audit",
        "decision",
        "evaluator",
        "group_id",
        "instance",
        "labeler",
        "manifest",
        "objective",
        "split",
    }

    assert forbidden.isdisjoint(inspect.signature(repair_once).parameters)


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"neighborhood": ()}, "neighborhood"),
        ({"neighborhood": (9,)}, "neighborhood|logical"),
        ({"random_seed": -1}, "seed"),
        ({"max_candidates": 0}, "max_candidates"),
        ({"max_transitions": -1}, "max_transitions"),
    ),
)
def test_repair_once_rejects_invalid_control_inputs(override, match) -> None:
    from isingfold.repair import repair_once

    logical, host, incumbent = _case()
    options = {
        "neighborhood": (0, 1),
        "random_seed": 3,
        "max_candidates": 8,
        "max_transitions": 20,
    }
    options.update(override)

    with pytest.raises((TypeError, ValueError), match=match):
        repair_once(logical, host, incumbent, **options)


def test_repair_once_rejects_an_invalid_incumbent_before_search() -> None:
    from isingfold.repair import repair_once

    logical, host, incumbent = _case()
    incumbent[2] = [7]

    with pytest.raises(ValueError, match="incumbent|embedding|realize"):
        repair_once(
            logical,
            host,
            incumbent,
            neighborhood=(0, 1),
            random_seed=3,
        )


def test_repair_once_preserves_non_orderable_hashable_target_labels() -> None:
    from isingfold.repair import repair_once

    logical = nx.complete_graph(3)
    host = nx.Graph([(0, "a"), ("a", 1), (1, "b"), ("b", 0)])
    incumbent = {0: [0], 1: ["a"], 2: [1, "b"]}

    result = repair_once(
        logical,
        host,
        incumbent,
        neighborhood=(0,),
        random_seed=7,
        max_transitions=40,
    )

    assert result["chains"] is not None
    from lac_minorminer.validation import validate_embedding

    assert validate_embedding(logical, host, result["chains"]).valid

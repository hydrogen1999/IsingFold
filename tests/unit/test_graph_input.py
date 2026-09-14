from __future__ import annotations

import pytest

from lac_minorminer._graph_input import normalize_graph


class GraphLike:
    def __init__(self, nodes, edges, *, directed: bool = False):
        self.nodes = nodes
        self.edges = edges
        self._directed = directed

    def is_directed(self) -> bool:
        return self._directed


def test_graph_like_input_preserves_labels_and_isolates() -> None:
    graph = normalize_graph(GraphLike(["z", "a", "isolated"], [("a", "z")]))

    assert graph.labels == ("z", "a", "isolated")
    assert graph.edges == ((0, 1),)
    assert graph.index("a") == 1


def test_edge_iterable_uses_first_appearance_and_deduplicates() -> None:
    graph = normalize_graph([("b", "a"), ("a", "b"), ("a", "c")])

    assert graph.labels == ("b", "a", "c")
    assert graph.edges == ((0, 1), (1, 2))


@pytest.mark.parametrize(
    "value, message",
    [
        ([("a", "a")], "self-loop"),
        (GraphLike([0, 1], [(0, 2)]), "not listed"),
        (GraphLike([0, 1], [(0, 1)], directed=True), "directed"),
    ],
)
def test_invalid_graph_input_is_rejected(value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_graph(value)

"""Deterministic graph normalization shared by the public Python boundary."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizedGraph:
    labels: tuple[Hashable, ...]
    edges: tuple[tuple[int, int], ...]
    _indices: Mapping[Hashable, int]

    def index(self, label: Hashable) -> int:
        try:
            return self._indices[label]
        except (KeyError, TypeError) as error:
            raise ValueError(f"node {label!r} is not listed in the graph") from error


def _items(attribute: Any) -> list[Any]:
    return list(attribute() if callable(attribute) else attribute)


def _pair(edge: Any) -> tuple[Hashable, Hashable]:
    try:
        first, second = edge
    except (TypeError, ValueError) as error:
        raise ValueError(f"each edge must contain exactly two endpoints, got {edge!r}") from error
    return first, second


def normalize_graph(value: Any) -> NormalizedGraph:
    """Map arbitrary hashable labels to compact IDs in deterministic input order."""

    if isinstance(value, NormalizedGraph):
        return value
    graph_like = hasattr(value, "nodes") and hasattr(value, "edges")
    if graph_like:
        is_directed = getattr(value, "is_directed", None)
        if callable(is_directed) and is_directed():
            raise ValueError("directed graphs are not supported")
        labels = _items(value.nodes)
        raw_edges = _items(value.edges)
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise TypeError("a graph must be graph-like or an iterable of endpoint pairs")
        raw_edges = list(value)
        labels = []
        seen: dict[Hashable, None] = {}
        for raw_edge in raw_edges:
            for label in _pair(raw_edge):
                try:
                    if label not in seen:
                        seen[label] = None
                        labels.append(label)
                except TypeError as error:
                    raise ValueError(f"graph labels must be hashable, got {label!r}") from error

    indices: dict[Hashable, int] = {}
    for index, label in enumerate(labels):
        try:
            if label in indices:
                raise ValueError(f"duplicate node label {label!r}")
            indices[label] = index
        except TypeError as error:
            raise ValueError(f"graph labels must be hashable, got {label!r}") from error

    edges: set[tuple[int, int]] = set()
    for raw_edge in raw_edges:
        first_label, second_label = _pair(raw_edge)
        try:
            first = indices[first_label]
            second = indices[second_label]
        except (KeyError, TypeError) as error:
            raise ValueError(f"edge endpoint is not listed in the graph: {raw_edge!r}") from error
        if first == second:
            raise ValueError(f"self-loop at {first_label!r} is not supported")
        edges.add((min(first, second), max(first, second)))

    return NormalizedGraph(tuple(labels), tuple(sorted(edges)), MappingProxyType(indices))

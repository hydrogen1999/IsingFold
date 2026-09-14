"""Independent, pure-Python verification for returned minor embeddings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ._graph_input import normalize_graph


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]


def _connected(chain: set[int], adjacency: list[set[int]]) -> bool:
    if not chain:
        return False
    reached = {next(iter(chain))}
    frontier = list(reached)
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] & chain:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return reached == chain


def validate_embedding(
    source: Any, target: Any, embedding: Mapping[Any, Iterable[Any]]
) -> ValidationReport:
    """Check chain coverage, connectivity, disjointness, and source-edge realization."""

    if not isinstance(embedding, Mapping):
        raise TypeError("embedding must be a mapping from source labels to target-node iterables")

    source_graph = normalize_graph(source)
    target_graph = normalize_graph(target)
    adjacency = [set() for _ in target_graph.labels]
    for first, second in target_graph.edges:
        adjacency[first].add(second)
        adjacency[second].add(first)

    errors: list[str] = []
    chains: list[set[int]] = []
    ownership: list[list[Any]] = [[] for _ in target_graph.labels]
    source_labels = set(source_graph.labels)
    for extra in embedding:
        if extra not in source_labels:
            errors.append(f"unknown source variable {extra!r} has a chain")

    for logical in source_graph.labels:
        if logical not in embedding:
            errors.append(f"source variable {logical!r} is missing a chain")
            chains.append(set())
            continue
        raw_chain = embedding[logical]
        if isinstance(raw_chain, (str, bytes)) or not isinstance(raw_chain, Iterable):
            errors.append(f"chain for {logical!r} must be an iterable of target nodes")
            chains.append(set())
            continue

        indices: list[int] = []
        for target_label in raw_chain:
            try:
                indices.append(target_graph.index(target_label))
            except ValueError:
                errors.append(
                    f"chain for {logical!r} contains unknown target node {target_label!r}"
                )
        chain = set(indices)
        chains.append(chain)
        if not indices:
            errors.append(f"chain for {logical!r} is empty")
        if len(chain) != len(indices):
            errors.append(f"chain for {logical!r} repeats a target node")
        if chain and not _connected(chain, adjacency):
            errors.append(f"chain for {logical!r} is not connected")
        for node in chain:
            ownership[node].append(logical)

    for node, owners in enumerate(ownership):
        if len(owners) > 1:
            errors.append(
                f"target node {target_graph.labels[node]!r} has chain overlap: {owners!r}"
            )

    for first, second in source_graph.edges:
        realized = any(
            neighbor in chains[second] for node in chains[first] for neighbor in adjacency[node]
        )
        if not realized:
            errors.append(
                f"source edge ({source_graph.labels[first]!r}, {source_graph.labels[second]!r}) "
                "is not realized"
            )

    return ValidationReport(not errors, tuple(errors))

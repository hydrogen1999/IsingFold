"""Deterministic, label-free global context for an Ising Hamiltonian.

The encoder reads only record["problem"]["h"] and record["problem"]["J"].
It intentionally ignores ground-state energy, embedding labels, candidate choices, and
baseline metadata, so the returned vector is available unchanged at deployment time.

Coefficient moments are computed after a signed log1p(abs(x)) transform. This keeps
the representation finite even for very large finite coefficients while retaining sign
and scale information. Graph summaries use non-zero logical couplers; explicit zero-valued
couplers contribute no edge because they do not change the Hamiltonian.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

HAMILTONIAN_CONTEXT_SCHEMA = "embedbench.hamiltonian-context"
HAMILTONIAN_CONTEXT_SCHEMA_VERSION = 1
HAMILTONIAN_CONTEXT_NAMES = (
    "log1p_num_variables",
    "log1p_num_nonzero_couplers",
    "nonzero_coupler_density",
    "mean_degree_fraction",
    "degree_std_fraction",
    "max_degree_fraction",
    "isolated_variable_fraction",
    "connected_component_fraction",
    "largest_component_fraction",
    "cycle_rank_per_nonzero_coupler",
    "triangle_density",
    "frustrated_triangle_fraction",
    "sign_frustrated_nontrivial_component_fraction",
    "nodes_in_sign_frustrated_components_fraction",
    "couplers_in_sign_frustrated_components_fraction",
    "h_signed_log_mean",
    "h_signed_log_std",
    "h_signed_log_min",
    "h_signed_log_max",
    "h_abs_log_mean",
    "h_abs_log_std",
    "h_abs_log_max",
    "h_positive_fraction",
    "h_negative_fraction",
    "h_zero_fraction",
    "j_signed_log_mean",
    "j_signed_log_std",
    "j_signed_log_min",
    "j_signed_log_max",
    "j_abs_log_mean",
    "j_abs_log_std",
    "j_abs_log_max",
    "j_positive_fraction",
    "j_negative_fraction",
    "all_coefficient_abs_log_mean",
    "h_minus_j_abs_log_mean",
)
HAMILTONIAN_CONTEXT_DIMENSION = len(HAMILTONIAN_CONTEXT_NAMES)


def hamiltonian_context_contract() -> dict[str, object]:
    """Return the portable feature contract stored in training artifacts."""

    return {
        "schema": HAMILTONIAN_CONTEXT_SCHEMA,
        "schema_version": HAMILTONIAN_CONTEXT_SCHEMA_VERSION,
        "dimension": HAMILTONIAN_CONTEXT_DIMENSION,
        "feature_names": list(HAMILTONIAN_CONTEXT_NAMES),
        "source": "problem.h_and_problem.J",
        "uses_ground_state_energy": False,
    }


@dataclass(frozen=True)
class HamiltonianContext:
    """A self-describing immutable global Hamiltonian feature vector."""

    values: np.ndarray
    schema: str = field(default=HAMILTONIAN_CONTEXT_SCHEMA, init=False)
    schema_version: int = field(default=HAMILTONIAN_CONTEXT_SCHEMA_VERSION, init=False)
    dimension: int = field(default=HAMILTONIAN_CONTEXT_DIMENSION, init=False)
    names: tuple[str, ...] = field(default=HAMILTONIAN_CONTEXT_NAMES, init=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"Hamiltonian context must have shape ({self.dimension},), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Hamiltonian context contains a non-finite feature")
        immutable = values.copy()
        immutable.flags.writeable = False
        object.__setattr__(self, "values", immutable)


@dataclass(frozen=True)
class _ParsedHamiltonian:
    nodes: tuple[int, ...]
    h: tuple[float, ...]
    edges: tuple[tuple[int, int, float], ...]


def _node_id(value: object, location: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{location} must be an integer node ID, not bool")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        try:
            converted = int(value)
        except ValueError as error:
            raise ValueError(f"{location} must be a canonical integer string") from error
        if str(converted) != value:
            raise ValueError(f"{location} must be a canonical integer string")
        return converted
    raise ValueError(f"{location} must be an integer node ID")


def _finite_real(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{location} must be a finite real number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{location} must be a finite real number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be a finite real number")
    return converted


def _parse_problem(record: Mapping[str, Any]) -> _ParsedHamiltonian:
    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")
    if "problem" not in record:
        raise ValueError("record requires problem")
    problem = record["problem"]
    if not isinstance(problem, Mapping):
        raise ValueError("problem must be a mapping")
    if "h" not in problem:
        raise ValueError("problem requires h")
    if "J" not in problem:
        raise ValueError("problem requires J")

    raw_h = problem["h"]
    if not isinstance(raw_h, Mapping):
        raise ValueError("problem h must be a mapping")
    h_by_node: dict[int, float] = {}
    for raw_node, raw_coefficient in raw_h.items():
        node = _node_id(raw_node, "h node")
        if node in h_by_node:
            raise ValueError(f"problem h repeats logical variable {node}")
        h_by_node[node] = _finite_real(raw_coefficient, f"h[{node}]")
    if not h_by_node:
        raise ValueError("problem requires at least one logical variable")

    raw_j = problem["J"]
    if not isinstance(raw_j, Sequence) or isinstance(raw_j, (str, bytes)):
        raise ValueError("problem J must be a sequence of coupling triples")
    edges_by_pair: dict[tuple[int, int], float] = {}
    for edge_index, raw_edge in enumerate(raw_j):
        if (
            not isinstance(raw_edge, Sequence)
            or isinstance(raw_edge, (str, bytes))
            or len(raw_edge) != 3
        ):
            raise ValueError(f"J[{edge_index}] must be a three-item sequence")
        left = _node_id(raw_edge[0], f"J[{edge_index}] left endpoint")
        right = _node_id(raw_edge[1], f"J[{edge_index}] right endpoint")
        if left == right:
            raise ValueError(f"J[{edge_index}] is a self-coupling")
        if left not in h_by_node or right not in h_by_node:
            raise ValueError(f"J[{edge_index}] endpoint is absent from h")
        pair = (left, right) if left < right else (right, left)
        if pair in edges_by_pair:
            raise ValueError(f"J contains duplicate undirected coupling {pair}")
        edges_by_pair[pair] = _finite_real(raw_edge[2], f"J[{edge_index}] coefficient")

    nodes = tuple(sorted(h_by_node))
    edges = tuple(
        (left, right, coefficient)
        for (left, right), coefficient in sorted(edges_by_pair.items())
        if coefficient != 0.0
    )
    return _ParsedHamiltonian(
        nodes=nodes,
        h=tuple(h_by_node[node] for node in nodes),
        edges=edges,
    )


def _components(
    nodes: tuple[int, ...], adjacency: Mapping[int, set[int]]
) -> tuple[frozenset[int], ...]:
    remaining = set(nodes)
    components: list[frozenset[int]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            reached = remaining.intersection(adjacency[current])
            for neighbour in sorted(reached, reverse=True):
                remaining.remove(neighbour)
                component.add(neighbour)
                frontier.append(neighbour)
        components.append(frozenset(component))
    return tuple(components)


def _component_is_sign_frustrated(
    component: frozenset[int],
    adjacency: Mapping[int, set[int]],
    couplings: Mapping[tuple[int, int], float],
) -> bool:
    """Whether preferred pair relations are inconsistent on any cycle."""

    root = min(component)
    gauge = {root: 1}
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for neighbour in sorted(adjacency[current]):
            pair = (current, neighbour) if current < neighbour else (neighbour, current)
            preferred_relation = -1 if couplings[pair] > 0.0 else 1
            expected = gauge[current] * preferred_relation
            if neighbour in gauge:
                if gauge[neighbour] != expected:
                    return True
            else:
                gauge[neighbour] = expected
                frontier.append(neighbour)
    return False


def _graph_features(problem: _ParsedHamiltonian) -> list[float]:
    nodes = problem.nodes
    node_count = len(nodes)
    edge_count = len(problem.edges)
    adjacency = {node: set() for node in nodes}
    couplings: dict[tuple[int, int], float] = {}
    for left, right, coefficient in problem.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
        couplings[(left, right)] = coefficient

    degrees = np.asarray([len(adjacency[node]) for node in nodes], dtype=np.float64)
    degree_scale = float(max(1, node_count - 1))
    components = _components(nodes, adjacency)
    cycle_rank = edge_count - node_count + len(components)

    triangle_count = 0
    frustrated_triangle_count = 0
    for left in nodes:
        for middle in sorted(node for node in adjacency[left] if node > left):
            for right in sorted(adjacency[left].intersection(adjacency[middle])):
                if right <= middle:
                    continue
                triangle_count += 1
                signs = (
                    couplings[(left, middle)],
                    couplings[(left, right)],
                    couplings[(middle, right)],
                )
                preferred_product = math.prod(-1 if value > 0.0 else 1 for value in signs)
                frustrated_triangle_count += preferred_product < 0

    nontrivial_components = [
        component for component in components if any(adjacency[node] for node in component)
    ]
    frustrated_components = [
        component
        for component in nontrivial_components
        if _component_is_sign_frustrated(component, adjacency, couplings)
    ]
    frustrated_nodes = set().union(*frustrated_components) if frustrated_components else set()
    frustrated_edges = sum(
        left in frustrated_nodes and right in frustrated_nodes for left, right, _ in problem.edges
    )

    possible_edges = node_count * (node_count - 1) / 2
    possible_triangles = node_count * (node_count - 1) * (node_count - 2) / 6
    return [
        math.log1p(node_count),
        math.log1p(edge_count),
        edge_count / possible_edges if possible_edges else 0.0,
        float(degrees.mean()) / degree_scale,
        float(degrees.std()) / degree_scale,
        float(degrees.max()) / degree_scale,
        float(np.count_nonzero(degrees == 0.0)) / node_count,
        len(components) / node_count,
        max(len(component) for component in components) / node_count,
        cycle_rank / max(1, edge_count),
        triangle_count / possible_triangles if possible_triangles else 0.0,
        (frustrated_triangle_count / triangle_count if triangle_count else 0.0),
        (len(frustrated_components) / len(nontrivial_components) if nontrivial_components else 0.0),
        len(frustrated_nodes) / node_count,
        frustrated_edges / edge_count if edge_count else 0.0,
    ]


def _coefficient_features(
    coefficients: Sequence[float], *, include_zero_fraction: bool
) -> list[float]:
    if coefficients:
        values = np.asarray(coefficients, dtype=np.float64)
        magnitude_log = np.log1p(np.abs(values))
        signed_log = np.sign(values) * magnitude_log
        features = [
            float(signed_log.mean()),
            float(signed_log.std()),
            float(signed_log.min()),
            float(signed_log.max()),
            float(magnitude_log.mean()),
            float(magnitude_log.std()),
            float(magnitude_log.max()),
            float(np.count_nonzero(values > 0.0)) / len(values),
            float(np.count_nonzero(values < 0.0)) / len(values),
        ]
        if include_zero_fraction:
            features.append(float(np.count_nonzero(values == 0.0)) / len(values))
        return features
    return [0.0] * (10 if include_zero_fraction else 9)


def encode_hamiltonian_context(record: Mapping[str, Any]) -> HamiltonianContext:
    """Encode the complete logical h/J problem without consuming any labels."""

    problem = _parse_problem(record)
    j_coefficients = tuple(coefficient for _, _, coefficient in problem.edges)
    h_features = _coefficient_features(problem.h, include_zero_fraction=True)
    j_features = _coefficient_features(j_coefficients, include_zero_fraction=False)
    all_coefficients = problem.h + j_coefficients
    all_abs_log_mean = float(
        np.log1p(np.abs(np.asarray(all_coefficients, dtype=np.float64))).mean()
    )
    features = [
        *_graph_features(problem),
        *h_features,
        *j_features,
        all_abs_log_mean,
        h_features[4] - j_features[4],
    ]
    values = np.asarray(features, dtype=np.float32)
    if values.shape != (HAMILTONIAN_CONTEXT_DIMENSION,):
        raise RuntimeError("Hamiltonian context implementation disagrees with its schema")
    if not np.isfinite(values).all():
        raise ValueError("Hamiltonian context features must be finite")
    return HamiltonianContext(values)


__all__ = [
    "HAMILTONIAN_CONTEXT_DIMENSION",
    "HAMILTONIAN_CONTEXT_NAMES",
    "HAMILTONIAN_CONTEXT_SCHEMA",
    "HAMILTONIAN_CONTEXT_SCHEMA_VERSION",
    "HamiltonianContext",
    "encode_hamiltonian_context",
    "hamiltonian_context_contract",
]

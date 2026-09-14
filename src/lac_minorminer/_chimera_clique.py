"""Bounded defect-aware clique support for Chimera targets.

The legacy search remains the first initializer.  This module is invoked only after that
search fails and only when the source is complete and the target can be authenticated as a
faulted Chimera graph.  It never reads Ising coefficients, solution samples, or evaluator
outcomes.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from numbers import Integral

from ._graph_input import NormalizedGraph
from .diagnostics import SearchWorkCounters


_MIN_NODE_COVERAGE = 0.75
_MIN_EDGE_COVERAGE = 0.75
_MAX_TILE = 16
_MAX_STRUCTURAL_CANDIDATES = 4_096
_MAX_CLIQUE_SEARCH_STATES = 100_000
_MAX_CLIQUE_ORDER = 256

_Coordinate = tuple[int, int, int, int]


class _BudgetExhausted(RuntimeError):
    def __init__(self, coordinate: str) -> None:
        super().__init__(coordinate)
        self.coordinate = coordinate


class _DeadlineExceeded(RuntimeError):
    pass


class _SearchStateLimit(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _Layout:
    rows: int
    columns: int
    tile: int
    coordinates: tuple[_Coordinate, ...]


@dataclass(frozen=True, slots=True)
class ChimeraCliqueResult:
    applicable: bool
    chains: tuple[tuple[int, ...], ...] | None
    work: SearchWorkCounters
    detail: str
    budget_exhausted_coordinate: str | None = None
    timed_out: bool = False


class _Ledger:
    def __init__(
        self,
        initial: SearchWorkCounters,
        cap: SearchWorkCounters | None,
        deadline: float | None,
    ) -> None:
        self.work = initial
        self.cap = cap
        self.deadline = deadline

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _DeadlineExceeded

    def charge(self, coordinate: str, amount: int = 1) -> None:
        self.check_deadline()
        if amount < 0:
            raise ValueError("structural work increments must be nonnegative")
        if amount == 0:
            return
        if self.cap is not None:
            current = getattr(self.work, coordinate)
            limit = getattr(self.cap, coordinate)
            if current > limit or amount > limit - current:
                raise _BudgetExhausted(coordinate)
        self.work = self.work.plus(**{coordinate: amount})


def _is_complete(graph: NormalizedGraph) -> bool:
    size = len(graph.labels)
    return size >= 3 and len(graph.edges) == size * (size - 1) // 2


def _edge_is_chimera(first: _Coordinate, second: _Coordinate) -> bool:
    i1, j1, u1, k1 = first
    i2, j2, u2, k2 = second
    if i1 == i2 and j1 == j2 and u1 != u2:
        return True
    if u1 != u2 or k1 != k2:
        return False
    if u1 == 0:
        return j1 == j2 and abs(i1 - i2) == 1
    return i1 == i2 and abs(j1 - j2) == 1


def _nominal_neighbors(
    coordinate: _Coordinate,
    *,
    rows: int,
    columns: int,
    tile: int,
) -> tuple[_Coordinate, ...]:
    i, j, u, k = coordinate
    result = [(i, j, 1 - u, other) for other in range(tile)]
    if u == 0:
        if i > 0:
            result.append((i - 1, j, u, k))
        if i + 1 < rows:
            result.append((i + 1, j, u, k))
    else:
        if j > 0:
            result.append((i, j - 1, u, k))
        if j + 1 < columns:
            result.append((i, j + 1, u, k))
    return tuple(result)


def _layout_score(
    layout: _Layout,
    target: NormalizedGraph,
    ledger: _Ledger,
) -> tuple[float, float, int, int] | None:
    nominal_nodes = 2 * layout.rows * layout.columns * layout.tile
    node_coverage = len(layout.coordinates) / nominal_nodes
    if node_coverage < _MIN_NODE_COVERAGE:
        return None

    coordinate_to_node: dict[_Coordinate, int] = {}
    for node, coordinate in enumerate(layout.coordinates):
        ledger.charge("feature_work")
        i, j, u, k = coordinate
        if (
            not 0 <= i < layout.rows
            or not 0 <= j < layout.columns
            or u not in (0, 1)
            or not 0 <= k < layout.tile
            or coordinate in coordinate_to_node
        ):
            return None
        coordinate_to_node[coordinate] = node

    observed_edges = set(target.edges)
    for first, second in target.edges:
        ledger.charge("feature_work")
        if not _edge_is_chimera(
            layout.coordinates[first], layout.coordinates[second]
        ):
            return None

    expected_edges: set[tuple[int, int]] = set()
    for node, coordinate in enumerate(layout.coordinates):
        for neighbor_coordinate in _nominal_neighbors(
            coordinate,
            rows=layout.rows,
            columns=layout.columns,
            tile=layout.tile,
        ):
            ledger.charge("feature_work")
            neighbor = coordinate_to_node.get(neighbor_coordinate)
            if neighbor is not None and neighbor != node:
                expected_edges.add((min(node, neighbor), max(node, neighbor)))
    if not expected_edges:
        return None
    edge_coverage = len(observed_edges) / len(expected_edges)
    if not observed_edges <= expected_edges or edge_coverage < _MIN_EDGE_COVERAGE:
        return None
    return (
        edge_coverage,
        node_coverage,
        min(layout.rows, layout.columns),
        layout.tile,
    )


def _coordinate_layout(
    target: NormalizedGraph,
    ledger: _Ledger,
) -> _Layout | None:
    coordinates: list[_Coordinate] = []
    for label in target.labels:
        ledger.charge("feature_work")
        if not isinstance(label, tuple) or len(label) != 4:
            return None
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in label):
            return None
        coordinate = tuple(int(value) for value in label)
        if any(value < 0 for value in coordinate) or coordinate[2] not in (0, 1):
            return None
        coordinates.append(coordinate)  # type: ignore[arg-type]
    if not coordinates:
        return None
    layout = _Layout(
        rows=max(item[0] for item in coordinates) + 1,
        columns=max(item[1] for item in coordinates) + 1,
        tile=max(item[3] for item in coordinates) + 1,
        coordinates=tuple(coordinates),
    )
    if (
        layout.rows < 2
        or layout.columns < 2
        or layout.tile < 1
        or layout.tile > _MAX_TILE
    ):
        return None
    return layout if _layout_score(layout, target, ledger) is not None else None


def _integer_layout(
    target: NormalizedGraph,
    ledger: _Ledger,
) -> _Layout | None:
    labels: list[int] = []
    for label in target.labels:
        ledger.charge("feature_work")
        if isinstance(label, bool) or not isinstance(label, Integral) or int(label) < 0:
            return None
        labels.append(int(label))
    if not labels:
        return None

    maximum = max(labels)
    if maximum + 1 > math.floor(len(labels) / _MIN_NODE_COVERAGE):
        return None
    best: tuple[tuple[float, float, int, int], _Layout] | None = None
    for tile in range(1, min(_MAX_TILE, (maximum + 1) // 8) + 1):
        max_columns = math.floor(
            len(labels) / (4 * tile * _MIN_NODE_COVERAGE)
        )
        if max_columns < 2:
            continue
        for columns in range(2, max_columns + 1):
            ledger.check_deadline()
            ledger.charge("feature_work")
            rows = math.ceil((maximum + 1) / (2 * columns * tile))
            if rows < 2:
                continue
            nominal_nodes = 2 * rows * columns * tile
            if maximum >= nominal_nodes or len(labels) / nominal_nodes < _MIN_NODE_COVERAGE:
                continue
            coordinates = []
            for label in labels:
                ledger.charge("feature_work")
                k = label % tile
                quotient = label // tile
                u = quotient % 2
                cell = quotient // 2
                coordinates.append((cell // columns, cell % columns, u, k))
            layout = _Layout(rows, columns, tile, tuple(coordinates))
            score = _layout_score(layout, target, ledger)
            if score is not None and (best is None or score > best[0]):
                best = (score, layout)
    return None if best is None else best[1]


def _infer_layout(target: NormalizedGraph, ledger: _Ledger) -> _Layout | None:
    if not target.labels or not target.edges:
        return None
    first = target.labels[0]
    if isinstance(first, tuple):
        return _coordinate_layout(target, ledger)
    return _integer_layout(target, ledger)


def _lane_components(
    layout: _Layout,
    target: NormalizedGraph,
    ledger: _Ledger,
) -> tuple[tuple[int, ...], ...]:
    parent = list(range(len(target.labels)))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second in target.edges:
        ledger.check_deadline()
        ledger.charge("feature_work")
        first_coordinate = layout.coordinates[first]
        second_coordinate = layout.coordinates[second]
        if first_coordinate[2] == second_coordinate[2]:
            union(first, second)

    groups: dict[int, list[int]] = {}
    for node in range(len(target.labels)):
        ledger.charge("route_expansions")
        groups.setdefault(root(node), []).append(node)
    components = tuple(tuple(sorted(groups[root(node)])) for node in range(len(target.labels)))
    return components


def _candidate_chains(
    layout: _Layout,
    target: NormalizedGraph,
    ledger: _Ledger,
) -> tuple[tuple[int, ...], ...]:
    components = _lane_components(layout, target, ledger)
    candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for first, second in target.edges:
        ledger.check_deadline()
        ledger.charge("feature_work")
        first_coordinate = layout.coordinates[first]
        second_coordinate = layout.coordinates[second]
        if not (
            first_coordinate[:2] == second_coordinate[:2]
            and first_coordinate[2] != second_coordinate[2]
        ):
            continue
        if len(candidates) >= _MAX_STRUCTURAL_CANDIDATES:
            raise _SearchStateLimit
        vertical = first if first_coordinate[2] == 0 else second
        horizontal = second if first_coordinate[2] == 0 else first
        ledger.charge("materializations")
        chain = tuple(sorted(set(components[vertical]) | set(components[horizontal])))
        if chain in seen:
            continue
        seen.add(chain)
        candidates.append(chain)
    return tuple(candidates)


def _compatibility_masks(
    candidates: tuple[tuple[int, ...], ...],
    target: NormalizedGraph,
    ledger: _Ledger,
) -> tuple[int, ...]:
    target_adjacency = [set() for _ in target.labels]
    for first, second in target.edges:
        target_adjacency[first].add(second)
        target_adjacency[second].add(first)

    chain_masks: list[int] = []
    neighbor_masks: list[int] = []
    for chain in candidates:
        chain_mask = 0
        neighbor_mask = 0
        for node in chain:
            chain_mask |= 1 << node
            for neighbor in target_adjacency[node]:
                ledger.charge("feature_work")
                neighbor_mask |= 1 << neighbor
        chain_masks.append(chain_mask)
        neighbor_masks.append(neighbor_mask)

    compatibility = [0] * len(candidates)
    for first in range(len(candidates)):
        ledger.check_deadline()
        for second in range(first + 1, len(candidates)):
            ledger.charge("feature_work")
            if chain_masks[first] & chain_masks[second]:
                continue
            ledger.charge("feature_work")
            if neighbor_masks[first] & chain_masks[second]:
                compatibility[first] |= 1 << second
                compatibility[second] |= 1 << first
    return tuple(compatibility)


def _find_fixed_clique(
    compatibility: tuple[int, ...],
    size: int,
    random_seed: int,
    ledger: _Ledger,
) -> tuple[int, ...] | None:
    if size > _MAX_CLIQUE_ORDER:
        raise _SearchStateLimit
    if len(compatibility) < size:
        return None
    tie_order = list(range(len(compatibility)))
    random.Random(random_seed).shuffle(tie_order)
    tie_rank = {node: rank for rank, node in enumerate(tie_order)}
    visited_states = 0

    def search(chosen: tuple[int, ...], possible: int) -> tuple[int, ...] | None:
        nonlocal visited_states
        ledger.check_deadline()
        if visited_states >= _MAX_CLIQUE_SEARCH_STATES:
            raise _SearchStateLimit
        ledger.charge("feature_work")
        visited_states += 1
        needed = size - len(chosen)
        if needed == 0:
            return chosen
        if possible.bit_count() < needed:
            return None

        vertices: list[tuple[int, int, int]] = []
        remaining = possible
        while remaining:
            bit = remaining & -remaining
            node = bit.bit_length() - 1
            ledger.charge("feature_work")
            degree = (compatibility[node] & possible).bit_count()
            vertices.append((-degree, tie_rank[node], node))
            remaining ^= bit
        vertices.sort()

        available = possible
        for _, _, node in vertices:
            bit = 1 << node
            if not available & bit:
                continue
            if available.bit_count() < needed:
                return None
            ledger.charge("feature_work")
            result = search(chosen + (node,), (available & ~bit) & compatibility[node])
            if result is not None:
                return result
            available &= ~bit
        return None

    return search((), (1 << len(compatibility)) - 1)


def find_chimera_clique_fallback(
    source: NormalizedGraph,
    target: NormalizedGraph,
    *,
    random_seed: int,
    initial_work: SearchWorkCounters,
    work_cap: SearchWorkCounters | None,
    deadline: float | None,
) -> ChimeraCliqueResult:
    """Find a complete-source embedding using a bounded structural construction."""

    ledger = _Ledger(initial_work, work_cap, deadline)
    if not _is_complete(source):
        return ChimeraCliqueResult(False, None, ledger.work, "source_not_complete")
    if len(source.labels) > _MAX_CLIQUE_ORDER:
        return ChimeraCliqueResult(False, None, ledger.work, "source_order_limit")
    try:
        ledger.check_deadline()
        layout = _infer_layout(target, ledger)
        if layout is None:
            return ChimeraCliqueResult(False, None, ledger.work, "target_not_chimera")
        candidates = _candidate_chains(layout, target, ledger)
        compatibility = _compatibility_masks(candidates, target, ledger)
        selected = _find_fixed_clique(
            compatibility,
            len(source.labels),
            random_seed,
            ledger,
        )
        chains = None if selected is None else tuple(candidates[index] for index in selected)
        return ChimeraCliqueResult(
            True,
            chains,
            ledger.work,
            "clique_not_found" if chains is None else "success",
        )
    except _BudgetExhausted as error:
        return ChimeraCliqueResult(
            True,
            None,
            ledger.work,
            "work_budget_exhausted",
            budget_exhausted_coordinate=error.coordinate,
        )
    except _DeadlineExceeded:
        return ChimeraCliqueResult(
            True,
            None,
            ledger.work,
            "timeout",
            timed_out=True,
        )
    except _SearchStateLimit:
        return ChimeraCliqueResult(True, None, ledger.work, "search_state_limit")


__all__ = ["ChimeraCliqueResult", "find_chimera_clique_fallback"]

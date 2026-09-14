"""Quality-first candidate value model and exact constrained selector.

The neural forward boundary accepts label-free local physical tensors plus a deterministic
summary of the complete logical Hamiltonian. Labels remain in the training/evaluation
layer. Exact resource and connectivity descriptors are reconstructed from immutable v1.1
quality records and are used for hard selection masks and auxiliary supervision.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from embedbench.hamiltonian_context import HAMILTONIAN_CONTEXT_DIMENSION
from embedbench.models_chain import CAND_F, build_chain_arch

ROBUSTNESS_NAMES = (
    "minimum_logical_contact_count",
    "mean_logical_contact_count",
    "total_logical_contact_count",
    "candidate_chain_cycle_rank",
)
"""Registered, exactly derived mechanism targets, in output-column order."""

ARTIFACT_SCHEMA = "embedbench.quality-value-v2"
ARTIFACT_SCHEMA_VERSION = 2
SELECTION_CONTRACT = "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
RESIDUAL_CONNECTIVITY_PROXY = "largest_remaining_free_component_size_over_original_window_size"
"""Registered auxiliary target; it is not a future-feasibility probability."""


@dataclass(frozen=True)
class ExactCandidateMetrics:
    """Exact complete-candidate descriptors aligned with candidate order.

    ``residual_connectivity_proxy`` is the largest connected component left after removing
    the candidate from the free window, divided by the original window size.  It is an
    auxiliary fragmentation descriptor, not a calibrated future-feasibility, routability,
    or completion-probability target.
    """

    total_qubits: np.ndarray
    max_chain: np.ndarray
    contact_counts: np.ndarray
    minimum_contacts: np.ndarray
    mean_contacts: np.ndarray
    total_contacts: np.ndarray
    cycle_rank: np.ndarray
    residual_connectivity_proxy: np.ndarray
    feasible: np.ndarray
    frozen_total_qubits: int
    release_focus_chain_q: np.ndarray | None

    @property
    def residual_capacity(self) -> np.ndarray:
        """Compatibility alias for the registered residual-connectivity proxy."""

        return self.residual_connectivity_proxy

    @property
    def residual_connectivity(self) -> np.ndarray:
        """Short alias for :attr:`residual_connectivity_proxy`."""

        return self.residual_connectivity_proxy

    @property
    def robustness(self) -> np.ndarray:
        return np.column_stack(
            (
                self.minimum_contacts,
                self.mean_contacts,
                self.total_contacts,
                self.cycle_rank,
            )
        ).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.total_qubits.shape[0])


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return converted


def _normalise_chain(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of qubit IDs")
    chain = tuple(_integer(qubit, name) for qubit in value)
    if len(chain) != len(set(chain)):
        raise ValueError(f"{name} contains duplicate qubits")
    return chain


def _component_count(nodes: set[int], adjacency: Mapping[int, set[int]]) -> int:
    remaining = set(nodes)
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            current = frontier.pop()
            reached = remaining & adjacency.get(current, set())
            remaining.difference_update(reached)
            frontier.extend(reached)
    return count


def _largest_component_size(nodes: set[int], adjacency: Mapping[int, set[int]]) -> int:
    remaining = set(nodes)
    largest = 0
    while remaining:
        frontier = [remaining.pop()]
        size = 0
        while frontier:
            current = frontier.pop()
            size += 1
            reached = remaining & adjacency.get(current, set())
            remaining.difference_update(reached)
            frontier.extend(reached)
        largest = max(largest, size)
    return largest


@dataclass(frozen=True)
class _HostView:
    nodes: frozenset[int]
    adjacency: Mapping[int, frozenset[int]]

    def has_edge(self, left: int, right: int) -> bool:
        return right in self.adjacency.get(left, frozenset())


def _verified_host_view(host: nx.Graph) -> _HostView:
    if not isinstance(host, nx.Graph) or host.is_directed() or host.is_multigraph():
        raise ValueError("host must be an undirected simple networkx.Graph")
    nodes = [_integer(node, "host node") for node in host.nodes]
    if len(nodes) != len(set(nodes)):
        raise ValueError("host node IDs are not unique integers")
    node_set = frozenset(nodes)
    adjacency: dict[int, set[int]] = {node: set() for node in node_set}
    for raw_left, raw_right in host.edges:
        left = _integer(raw_left, "host edge endpoint")
        right = _integer(raw_right, "host edge endpoint")
        if left == right:
            raise ValueError("host must not contain self-loops")
        if left not in node_set or right not in node_set:
            raise ValueError("host edge endpoint is absent from host nodes")
        adjacency[left].add(right)
        adjacency[right].add(left)
    return _HostView(
        nodes=node_set,
        adjacency={node: frozenset(neighbours) for node, neighbours in adjacency.items()},
    )


@cache
def _pristine_host_view(topology: str, size: int) -> _HostView:
    from embedbench.structural import host_graph

    return _verified_host_view(host_graph(topology, size))


def _resolve_host_view(record: Mapping[str, Any], host: nx.Graph | None) -> _HostView:
    if host is not None:
        return _verified_host_view(host)

    for field in ("defect_qubits", "defect_couplers"):
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a finite number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{field} must be a finite non-negative number")
        if float(value) != 0.0:
            raise ValueError(
                "defective-host records require an explicit host; pristine topology "
                f"reconstruction cannot represent {field}={value!r}"
            )

    topology = record.get("topology")
    if not isinstance(topology, str) or not topology:
        raise ValueError("pristine host reconstruction requires an explicit topology")
    size = _integer(record.get("size"), "size")
    if size <= 0:
        raise ValueError("size must be positive")
    try:
        return _pristine_host_view(topology, size)
    except (ImportError, ValueError) as error:
        raise ValueError(f"cannot reconstruct pristine {topology!r} host of size {size}") from error


def _problem_logical_edges(
    record: Mapping[str, Any],
    focus: int,
    all_chains: Mapping[int, tuple[int, ...]],
) -> tuple[tuple[int, int], ...]:
    problem = record.get("problem")
    if not isinstance(problem, Mapping):
        raise ValueError("exact feasibility requires problem.h and problem.J")
    raw_h = problem.get("h")
    if not isinstance(raw_h, Mapping):
        raise ValueError("exact feasibility requires problem.h")
    variables = {_integer(variable, "problem.h key") for variable in raw_h}
    expected_variables = set(all_chains) | {focus}
    if variables != expected_variables:
        raise ValueError("problem.h variables disagree with focus and all_chains")
    raw_j = problem.get("J")
    if not isinstance(raw_j, Sequence) or isinstance(raw_j, (str, bytes)):
        raise ValueError("exact feasibility requires candidate-independent problem.J")
    logical_edges: set[tuple[int, int]] = set()
    for index, raw_edge in enumerate(raw_j):
        if (
            not isinstance(raw_edge, Sequence)
            or isinstance(raw_edge, (str, bytes))
            or len(raw_edge) != 3
        ):
            raise ValueError(f"problem.J[{index}] must contain u, v, and coupling")
        left = _integer(raw_edge[0], f"problem.J[{index}][0]")
        right = _integer(raw_edge[1], f"problem.J[{index}][1]")
        if left == right or left not in variables or right not in variables:
            raise ValueError(f"problem.J[{index}] is not a valid logical edge")
        coupling = raw_edge[2]
        if isinstance(coupling, bool) or not isinstance(coupling, (int, float)):
            raise ValueError(f"problem.J[{index}][2] must be a finite number")
        if not math.isfinite(float(coupling)):
            raise ValueError(f"problem.J[{index}][2] must be a finite number")
        edge = (min(left, right), max(left, right))
        if edge in logical_edges:
            raise ValueError(f"problem.J contains duplicate logical edge {edge!r}")
        logical_edges.add(edge)
    return tuple(sorted(logical_edges))


def derive_exact_candidate_metrics(
    record: Mapping[str, Any], *, host: nx.Graph | None = None
) -> ExactCandidateMetrics:
    """Derive exact complete-embedding descriptors without changing ``record``.

    ``all_chains`` and ``problem.J`` are required because the legacy local fields cannot
    establish full-embedding validity.  When ``host`` is omitted, a pristine host is
    reconstructed from explicit ``topology`` and ``size``.  A caller handling a defective
    host must pass that exact graph; explicit non-zero defect fields fail closed otherwise.
    """

    raw_all_chains = record.get("all_chains")
    if not isinstance(raw_all_chains, Mapping):
        raise ValueError("exact complete-candidate metrics require a full record with all_chains")
    focus = _integer(record.get("focus"), "focus")
    all_chains: dict[int, tuple[int, ...]] = {}
    for raw_variable, raw_chain in raw_all_chains.items():
        variable = _integer(raw_variable, "all_chains key")
        if variable == focus:
            raise ValueError("all_chains must not contain the focus chain")
        if variable in all_chains:
            raise ValueError(f"all_chains contains duplicate logical variable {variable}")
        all_chains[variable] = _normalise_chain(raw_chain, f"all_chains[{raw_variable!r}]")
    if any(not chain for chain in all_chains.values()):
        raise ValueError("all non-focus chains must be non-empty")
    flattened_frozen = [qubit for chain in all_chains.values() for qubit in chain]
    if len(flattened_frozen) != len(set(flattened_frozen)):
        raise ValueError("all_chains must be pairwise disjoint")
    if record.get("n_vars") is not None:
        n_vars = _integer(record["n_vars"], "n_vars")
        if len(all_chains) != n_vars - 1:
            raise ValueError("all_chains must contain every non-focus logical variable")

    host_view = _resolve_host_view(record, host)
    for variable, chain in all_chains.items():
        chain_set = set(chain)
        if not chain_set <= host_view.nodes:
            raise ValueError(f"non-focus chain {variable} contains qubits outside the host")
        if _component_count(chain_set, host_view.adjacency) != 1:
            raise ValueError(f"non-focus chain {variable} is disconnected on the host")

    logical_edges = _problem_logical_edges(record, focus, all_chains)
    focus_neighbours = tuple(
        sorted(
            right if left == focus else left
            for left, right in logical_edges
            if focus in (left, right)
        )
    )
    raw_neighbours = record.get("neighbours")
    neighbours = _normalise_chain(raw_neighbours, "neighbours")
    if not set(focus_neighbours) <= set(neighbours):
        raise ValueError("neighbours omits a focus edge present in problem.J")
    unknown_neighbours = set(neighbours) - set(all_chains)
    if unknown_neighbours:
        raise ValueError(
            f"focus neighbours are absent from all_chains: {sorted(unknown_neighbours)!r}"
        )
    for left, right in logical_edges:
        if focus in (left, right):
            continue
        if not any(
            host_view.has_edge(left_qubit, right_qubit)
            for left_qubit in all_chains[left]
            for right_qubit in all_chains[right]
        ):
            raise ValueError(f"non-focus logical edge {(left, right)!r} is unrealised")

    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise ValueError("candidates must be a sequence")
    candidates = [
        _normalise_chain(candidate, f"candidates[{index}]")
        for index, candidate in enumerate(raw_candidates)
    ]
    if not candidates:
        raise ValueError("a quality record must contain at least one candidate")

    window = set(_normalise_chain(record.get("window_nodes"), "window_nodes"))
    raw_edges = record.get("window_edges")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
        raise ValueError("window_edges must be a sequence")
    if not window <= host_view.nodes:
        raise ValueError("window_nodes contains qubits outside the host")
    for index, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, Sequence) or len(raw_edge) != 2:
            raise ValueError(f"window_edges[{index}] must contain two qubit IDs")
        left = _integer(raw_edge[0], f"window_edges[{index}][0]")
        right = _integer(raw_edge[1], f"window_edges[{index}][1]")
        if left not in window or right not in window or left == right:
            raise ValueError(f"window_edges[{index}] is not a valid window edge")
        if not host_view.has_edge(left, right):
            raise ValueError(f"window_edges[{index}] is absent from the verified host")

    adjacency = {node: set(host_view.adjacency.get(node, frozenset())) & window for node in window}
    edges = {
        (min(left, right), max(left, right))
        for left in window
        for right in adjacency[left]
        if left < right
    }
    neighbour_chains = [set(all_chains[variable]) for variable in neighbours]
    contact_multiplicity = {
        qubit: [
            sum(1 for adjacent in host_view.adjacency.get(qubit, frozenset()) if adjacent in chain)
            for chain in neighbour_chains
        ]
        for qubit in window
    }

    base_total = sum(len(chain) for chain in all_chains.values())
    base_max = max((len(chain) for chain in all_chains.values()), default=0)
    blocked = set(flattened_frozen)
    l_cap_value = record.get("l_cap")
    l_cap = _integer(l_cap_value, "l_cap") if l_cap_value is not None else None

    total_qubits: list[int] = []
    max_chain: list[int] = []
    contact_rows: list[list[int]] = []
    cycle_rank: list[int] = []
    residual_connectivity_proxy: list[float] = []
    feasible: list[bool] = []
    for candidate in candidates:
        candidate_set = set(candidate)
        contacts = [
            sum(
                contact_multiplicity[qubit][index]
                for qubit in candidate
                if qubit in contact_multiplicity
            )
            for index in range(len(neighbour_chains))
        ]
        internal_edges = sum(
            1 for left, right in edges if left in candidate_set and right in candidate_set
        )
        components = _component_count(candidate_set, adjacency) if candidate_set else 0
        connected = bool(candidate_set) and components == 1
        valid = (
            connected
            and candidate_set <= window
            and candidate_set <= host_view.nodes
            and candidate_set.isdisjoint(blocked)
            and all(contact > 0 for contact in contacts)
            and (l_cap is None or len(candidate) <= l_cap)
        )

        total_qubits.append(base_total + len(candidate))
        max_chain.append(max(base_max, len(candidate)))
        contact_rows.append(contacts)
        cycle_rank.append(max(0, internal_edges - len(candidate) + components))
        remaining_free = window - blocked - candidate_set
        residual_connectivity_proxy.append(
            _largest_component_size(remaining_free, adjacency) / max(1, len(window))
        )
        feasible.append(valid)

    contacts_array = np.asarray(contact_rows, dtype=np.int64)
    if contacts_array.ndim == 1:
        contacts_array = contacts_array.reshape(len(candidates), 0)
    minimum_contacts = (
        contacts_array.min(axis=1)
        if contacts_array.shape[1]
        else np.zeros(len(candidates), dtype=np.int64)
    )
    mean_contacts = (
        contacts_array.mean(axis=1)
        if contacts_array.shape[1]
        else np.zeros(len(candidates), dtype=np.float64)
    )
    total_contacts = contacts_array.sum(axis=1)
    release_focus_chain_q = None
    if record.get("Q") is not None:
        raw_release_q = record["Q"]
        if not isinstance(raw_release_q, Sequence) or isinstance(raw_release_q, (str, bytes)):
            raise ValueError("release_focus_chain_Q must be a candidate-aligned sequence")
        release_focus_chain_q = np.asarray(
            [
                _integer(value, f"release_focus_chain_Q[{index}]")
                for index, value in enumerate(raw_release_q)
            ],
            dtype=np.int64,
        )
        expected_focus_q = np.asarray(total_qubits, dtype=np.int64) - base_total
        if release_focus_chain_q.shape != expected_focus_q.shape or not np.array_equal(
            release_focus_chain_q, expected_focus_q
        ):
            raise ValueError("release_focus_chain_Q disagrees with exact candidate-chain lengths")
    return ExactCandidateMetrics(
        total_qubits=np.asarray(total_qubits, dtype=np.int64),
        max_chain=np.asarray(max_chain, dtype=np.int64),
        contact_counts=contacts_array,
        minimum_contacts=np.asarray(minimum_contacts, dtype=np.int64),
        mean_contacts=np.asarray(mean_contacts, dtype=np.float32),
        total_contacts=np.asarray(total_contacts, dtype=np.int64),
        cycle_rank=np.asarray(cycle_rank, dtype=np.int64),
        residual_connectivity_proxy=np.asarray(residual_connectivity_proxy, dtype=np.float32),
        feasible=np.asarray(feasible, dtype=np.bool_),
        frozen_total_qubits=base_total,
        release_focus_chain_q=release_focus_chain_q,
    )


@dataclass
class QualityV2Output:
    """Candidate-aligned neural outputs before policy constraints are applied.

    ``future_capacity`` is retained as a checkpoint/API spelling, but its supervised target
    is only :data:`RESIDUAL_CONNECTIVITY_PROXY`.  The head is auxiliary and cannot override
    the quality score or exact feasibility mask.
    """

    quality_logit: torch.Tensor
    quality_log_concentration: torch.Tensor
    future_capacity: torch.Tensor
    robustness: torch.Tensor
    terminal_qubits: torch.Tensor | None

    @property
    def residual_connectivity_proxy(self) -> torch.Tensor:
        """Connectivity-aware auxiliary prediction under its uncalibrated legacy name."""

        return self.future_capacity

    @property
    def quality_mean(self) -> torch.Tensor:
        return torch.sigmoid(self.quality_logit)

    @property
    def quality_concentration(self) -> torch.Tensor:
        # A registered finite range avoids a degenerate infinite-confidence optimum.
        return 1.0 + 99.0 * torch.sigmoid(self.quality_log_concentration)


@dataclass(frozen=True)
class QualitySelection:
    index: int | None
    score: float | None
    statistic: Literal["mean", "lcb"]
    eligible_indices: tuple[int, ...]
    reason: str | None = None


def select_quality_candidate(
    output: QualityV2Output,
    metrics: ExactCandidateMetrics,
    *,
    budget: int,
    statistic: Literal["mean", "lcb"] = "mean",
    lcb_z: float = 1.0,
) -> QualitySelection:
    """Select quality-first after exact feasibility and physical-qubit masks.

    Ties are deterministic in candidate order.  An empty mask is represented explicitly;
    callers can then invoke the native classical fallback rather than silently violating the
    registered budget.
    """

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if statistic not in {"mean", "lcb"}:
        raise ValueError("statistic must be 'mean' or 'lcb'")
    if not math.isfinite(lcb_z) or lcb_z < 0.0:
        raise ValueError("lcb_z must be a finite non-negative number")
    candidate_count = output.quality_logit.numel()
    if output.quality_logit.ndim != 1 or len(metrics) != candidate_count:
        raise ValueError("model outputs and exact metrics must have one aligned candidate axis")
    if output.quality_log_concentration.shape != output.quality_logit.shape:
        raise ValueError("quality concentration must align with quality mean")
    if not bool(torch.isfinite(output.quality_logit).all()) or not bool(
        torch.isfinite(output.quality_log_concentration).all()
    ):
        raise ValueError("quality predictions must be finite")

    eligible = tuple(
        index
        for index in range(candidate_count)
        if bool(metrics.feasible[index]) and int(metrics.total_qubits[index]) <= budget
    )
    if not eligible:
        return QualitySelection(
            index=None,
            score=None,
            statistic=statistic,
            eligible_indices=(),
            reason="no_feasible_candidate_within_budget",
        )

    mean = output.quality_mean.detach().cpu()
    if statistic == "mean":
        scores = mean
    else:
        concentration = output.quality_concentration.detach().cpu()
        variance = mean * (1.0 - mean) / (concentration + 1.0)
        scores = mean - lcb_z * torch.sqrt(variance.clamp_min(0.0))
    selected = max(eligible, key=lambda index: (float(scores[index]), -index))
    return QualitySelection(
        index=selected,
        score=float(scores[selected]),
        statistic=statistic,
        eligible_indices=eligible,
    )


DEFAULT_BUDGET_RATIOS: tuple[float | None, ...] = (1.0, 1.1, 1.25, 1.5, None)


def _budget_key(ratio: float | None) -> str:
    return "uncapped" if ratio is None else f"{ratio:.2f}x"


def _probabilities(example: Any, candidate_count: int) -> list[float]:
    raw = getattr(example, "p", None)
    if not isinstance(raw, Sequence) or len(raw) != candidate_count:
        raise ValueError("evaluation p_solve labels must align with every candidate")
    labels = []
    for value in raw:
        if isinstance(value, bool):
            raise ValueError("evaluation p_solve labels must be finite probabilities")
        try:
            probability = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("evaluation p_solve labels must be finite probabilities") from error
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("evaluation p_solve labels must lie in [0, 1]")
        labels.append(probability)
    return labels


def _summary_for_selections(
    selections: list[int | None],
    regrets: list[float],
    top_count: int,
    feasible_count: int,
) -> dict[str, Any]:
    selected_count = sum(index is not None for index in selections)
    return {
        "regret": float(np.mean(regrets)) if regrets else None,
        "top": top_count / selected_count if selected_count else None,
        "feasibility_rate": feasible_count / selected_count if selected_count else None,
        "n_selected": selected_count,
        "selection_indices": selections,
    }


@dataclass(frozen=True)
class _BudgetEvaluationRow:
    input_index: int
    output: QualityV2Output
    labels: tuple[float, ...]
    metrics: ExactCandidateMetrics
    resource_index: int
    original_index: int | None


def _evaluate_budget_rows(
    rows: Sequence[_BudgetEvaluationRow],
    ratios: Sequence[float | None],
    keys: Sequence[str],
    *,
    reference_kind: Literal["stock_minorminer_original", "registered_resource"],
    reference_policy: str,
    reference_source: str,
    lcb_z: float,
    top_tolerance: float,
    random_seed: int,
    seed_namespace: str,
) -> dict[str, Any]:
    """Evaluate one cohort whose rows all share the same budget denominator."""

    rows = list(rows)
    budget_results: dict[str, Any] = {}
    for ratio, key in zip(ratios, keys, strict=True):
        policy_names = ("mean", "lcb", "resource", "random", "original")
        selections: dict[str, list[int | None]] = {name: [] for name in policy_names}
        regrets: dict[str, list[float]] = {name: [] for name in policy_names}
        top_counts = {name: 0 for name in policy_names}
        feasible_counts = {name: 0 for name in policy_names}
        reference_totals: list[int] = []
        reference_indices: list[int] = []
        budgets: list[int | None] = []
        eligible_counts: list[int] = []
        no_survivor = 0
        rng = random.Random(f"{seed_namespace}:{random_seed}:{key}")

        for row in rows:
            if reference_kind == "stock_minorminer_original":
                reference = row.original_index
                if reference is None or not bool(row.metrics.feasible[reference]):
                    raise RuntimeError("primary budget cohort contains an invalid Q_MM reference")
            else:
                reference = row.resource_index
            reference_total = int(row.metrics.total_qubits[reference])
            budget = (
                int(row.metrics.total_qubits.max())
                if ratio is None
                else int(math.floor(float(ratio) * reference_total + 1e-9))
            )
            reference_totals.append(reference_total)
            reference_indices.append(reference)
            budgets.append(None if ratio is None else budget)

            mean_selection = select_quality_candidate(
                row.output, row.metrics, budget=budget, statistic="mean", lcb_z=lcb_z
            )
            lcb_selection = select_quality_candidate(
                row.output, row.metrics, budget=budget, statistic="lcb", lcb_z=lcb_z
            )
            eligible = list(mean_selection.eligible_indices)
            if eligible != list(lcb_selection.eligible_indices):
                raise RuntimeError("registered selectors produced inconsistent exact support")
            eligible_counts.append(len(eligible))
            if not eligible:
                no_survivor += 1
                for name in policy_names:
                    selections[name].append(None)
                continue

            chosen = {
                "mean": mean_selection.index,
                "lcb": lcb_selection.index,
                "resource": row.resource_index if row.resource_index in eligible else None,
                "random": rng.choice(eligible),
                "original": row.original_index if row.original_index in eligible else None,
            }
            best = max(row.labels[index] for index in eligible)
            for name, index in chosen.items():
                selections[name].append(index)
                if index is None:
                    continue
                regrets[name].append(best - row.labels[index])
                top_counts[name] += row.labels[index] >= best - top_tolerance
                feasible_counts[name] += (
                    bool(row.metrics.feasible[index])
                    and int(row.metrics.total_qubits[index]) <= budget
                )

        selector_results = {
            name: _summary_for_selections(
                selections[name],
                regrets[name],
                top_counts[name],
                feasible_counts[name],
            )
            for name in policy_names
        }
        budget_results[key] = {
            "ratio": None if ratio is None else float(ratio),
            "reference_policy": reference_policy,
            "reference_sources": {reference_source: len(rows)},
            "input_record_indices": [row.input_index for row in rows],
            "reference_indices": reference_indices,
            "reference_total_qubits": reference_totals,
            "budgets": budgets,
            "n_records": len(rows),
            "mean_eligible_candidates": (
                float(np.mean(eligible_counts)) if eligible_counts else None
            ),
            "minimum_eligible_candidates": min(eligible_counts) if eligible_counts else None,
            "maximum_eligible_candidates": max(eligible_counts) if eligible_counts else None,
            "no_survivor": no_survivor,
            "no_survivor_rate": no_survivor / len(rows) if rows else None,
            "selectors": selector_results,
        }
    return budget_results


def evaluate_quality_v2_budgets(
    outputs: Sequence[QualityV2Output],
    examples: Sequence[Any],
    exact_metrics: Sequence[ExactCandidateMetrics],
    *,
    budget_ratios: Sequence[float | None] = DEFAULT_BUDGET_RATIOS,
    lcb_z: float = 1.0,
    top_tolerance: float = 0.02,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate mean and LCB selectors under the registered ``B/Q_MM`` contract.

    The presented candidate support is always complete.  For each registered ratio, a
    label-independent feasibility/budget mask is applied identically to learned and control
    policies. The primary sweep contains only rows with an exactly valid original stock
    minorminer candidate, whose complete-embedding qubit count is ``Q_MM``. Rows without that
    paired reference are excluded and reported, never re-based onto another denominator.
    A separately named diagnostic sweep uses the registered release resource candidate for
    every input row. Labels are consulted only after selection to calculate regret and
    top-choice metrics.
    """

    outputs = list(outputs)
    examples = list(examples)
    exact_metrics = list(exact_metrics)
    if not (len(outputs) == len(examples) == len(exact_metrics)):
        raise ValueError("outputs, examples, and exact metrics must align one-to-one")
    if not outputs:
        raise ValueError("budget evaluation requires at least one example")
    if not math.isfinite(lcb_z) or lcb_z < 0.0:
        raise ValueError("lcb_z must be a finite non-negative number")
    if not math.isfinite(top_tolerance) or top_tolerance < 0.0:
        raise ValueError("top_tolerance must be a finite non-negative number")
    ratios = list(budget_ratios)
    if not ratios:
        raise ValueError("budget_ratios must be non-empty")
    keys = []
    for ratio in ratios:
        if ratio is not None and (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or float(ratio) <= 0.0
        ):
            raise ValueError("budget ratios must be finite positive numbers or None")
        keys.append(_budget_key(None if ratio is None else float(ratio)))
    if len(keys) != len(set(keys)):
        raise ValueError("budget ratios must have unique registered names")

    prepared: list[_BudgetEvaluationRow] = []
    primary_rows: list[_BudgetEvaluationRow] = []
    included_input_indices: list[int] = []
    excluded_input_indices: list[int] = []
    exclusion_reasons = {
        "non_minorminer_source": 0,
        "missing_original_reference": 0,
        "infeasible_original_reference": 0,
    }
    fidelity_stages: set[int] = set()
    for input_index, (output, example, metrics) in enumerate(
        zip(outputs, examples, exact_metrics, strict=True)
    ):
        output = QualityV2Output(
            quality_logit=output.quality_logit.detach().cpu(),
            quality_log_concentration=output.quality_log_concentration.detach().cpu(),
            future_capacity=output.future_capacity.detach().cpu(),
            robustness=output.robustness.detach().cpu(),
            terminal_qubits=(
                output.terminal_qubits.detach().cpu()
                if output.terminal_qubits is not None
                else None
            ),
        )
        candidate_count = len(metrics)
        labels = _probabilities(example, candidate_count)
        # Calling the selector once at the maximum exact cost validates the output vectors
        # and their finiteness before evaluation begins.
        select_quality_candidate(
            output,
            metrics,
            budget=int(metrics.total_qubits.max()),
            statistic="mean",
        )
        valid_indices = [index for index in range(candidate_count) if bool(metrics.feasible[index])]
        raw_resource = getattr(example, "resource_index", None)
        if isinstance(raw_resource, bool) or not isinstance(raw_resource, (int, np.integer)):
            raise ValueError("resource_index must be an integer")
        resource_index = int(raw_resource)
        if not 0 <= resource_index < candidate_count:
            raise ValueError("resource_index lies outside the full candidate support")
        if not bool(metrics.feasible[resource_index]):
            raise ValueError("registered resource candidate is exactly infeasible")
        expected_resource = min(
            valid_indices,
            key=lambda index: (
                int(metrics.total_qubits[index]) - metrics.frozen_total_qubits,
                int(metrics.cycle_rank[index])
                + int(metrics.total_qubits[index])
                - metrics.frozen_total_qubits
                - 1,
                index,
            ),
        )
        if resource_index != expected_resource:
            raise ValueError(
                "resource_index disagrees with the registered focus-Q/internal-coupler "
                "resource policy"
            )
        raw_original = getattr(example, "original_index", -1)
        if isinstance(raw_original, bool) or not isinstance(raw_original, (int, np.integer)):
            raise ValueError("original_index must be an integer")
        original_index = int(raw_original)
        if original_index == -1:
            original_index = None
        elif not 0 <= original_index < candidate_count:
            raise ValueError("original_index lies outside the full candidate support")
        row = _BudgetEvaluationRow(
            input_index=input_index,
            output=output,
            labels=tuple(labels),
            metrics=metrics,
            resource_index=resource_index,
            original_index=original_index,
        )
        prepared.append(row)
        if getattr(example, "source", None) != "minorminer":
            exclusion_reason = "non_minorminer_source"
        elif original_index is None:
            exclusion_reason = "missing_original_reference"
        elif not bool(metrics.feasible[original_index]):
            exclusion_reason = "infeasible_original_reference"
        else:
            exclusion_reason = None
        if exclusion_reason is None:
            primary_rows.append(row)
            included_input_indices.append(input_index)
            stage = getattr(example, "stage", None)
            if isinstance(stage, Sequence) and len(stage) == candidate_count:
                fidelity_stages.update(int(value) for value in stage)
        else:
            exclusion_reasons[exclusion_reason] += 1
            excluded_input_indices.append(input_index)

    budget_results = _evaluate_budget_rows(
        primary_rows,
        ratios,
        keys,
        reference_kind="stock_minorminer_original",
        reference_policy="stock_minorminer_original_total_qubits",
        reference_source="stock_minorminer_original",
        lcb_z=lcb_z,
        top_tolerance=top_tolerance,
        random_seed=random_seed,
        seed_namespace="quality-v2",
    )
    diagnostic_budget_results = _evaluate_budget_rows(
        prepared,
        ratios,
        keys,
        reference_kind="registered_resource",
        reference_policy="registered_resource_candidate_total_qubits",
        reference_source="registered_resource_candidate",
        lcb_z=lcb_z,
        top_tolerance=top_tolerance,
        random_seed=random_seed,
        seed_namespace="quality-v2-resource-diagnostic",
    )

    finite_mean_regrets = [
        budget_results[key]["selectors"]["mean"]["regret"]
        for ratio, key in zip(ratios, keys, strict=True)
        if ratio is not None and budget_results[key]["selectors"]["mean"]["regret"] is not None
    ]
    primary_metric = float(np.mean(finite_mean_regrets)) if finite_mean_regrets else None
    if fidelity_stages == {2}:
        label_fidelity = "stage2-release"
    elif fidelity_stages:
        label_fidelity = "mixed-stage-release"
    else:
        label_fidelity = "release-unspecified"
    return {
        "candidate_support": "full",
        "support_uses_labels": False,
        "evaluation_mode": "development",
        "label_fidelity": label_fidelity,
        "provisional": True,
        "exact_mask": "minor_embedding_feasible_and_full_embedding_total_qubits",
        "release_Q_semantics": "release_focus_chain_Q",
        "budget_contract": "B/Q_MM",
        "budget_reference": "stock_minorminer_original_total_qubits",
        "primary_record_policy": "valid_stock_minorminer_original_reference_only",
        "primary_record_coverage": {
            "input_records": len(prepared),
            "included_records": len(primary_rows),
            "excluded_records": len(prepared) - len(primary_rows),
            "coverage_rate": len(primary_rows) / len(prepared),
            "included_input_indices": included_input_indices,
            "excluded_input_indices": excluded_input_indices,
            "exclusion_reasons": exclusion_reasons,
        },
        "budget_ratios": [None if ratio is None else float(ratio) for ratio in ratios],
        "selectors": ["mean", "lcb"],
        "lcb_z": float(lcb_z),
        "primary_selector": "mean",
        "primary_metric_name": "mean_finite_budget_regret",
        "primary_metric": primary_metric,
        "lcb_is_secondary_until_calibrated": True,
        "budgets": budget_results,
        "diagnostic_resource_relative_sweep": {
            "diagnostic_only": True,
            "included_in_primary_metric": False,
            "budget_contract": "B/Q_resource",
            "budget_reference": "registered_resource_candidate_total_qubits",
            "record_policy": "all_validated_input_records",
            "budget_ratios": [None if ratio is None else float(ratio) for ratio in ratios],
            "budgets": diagnostic_budget_results,
        },
    }


@dataclass(frozen=True)
class QualityV2Config:
    arch: str = "mpnn"
    hidden: int = 64
    layers: int = 3
    heads: int = 4
    neighbour_feats: bool = False
    predict_terminal_qubits: bool = False
    robustness_dim: int = len(ROBUSTNESS_NAMES)


class _CandidateRepresentation(nn.Module):
    """Reuse each V1 graph backbone while replacing only its scalar readout."""

    def __init__(self, config: QualityV2Config) -> None:
        super().__init__()
        self.output_dim = 3 * config.hidden + CAND_F + (2 if config.neighbour_feats else 0)
        self.encoder = build_chain_arch(
            config.arch,
            config.hidden,
            config.layers,
            config.heads,
            neighbour_feats=config.neighbour_feats,
        )
        if config.arch == "mpnn":
            self.encoder.read = nn.Identity()
        else:
            self.encoder.out.read = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(x, adjacency, candidate_masks, candidate_features)


class QualityValueV2(nn.Module):
    """Candidate-conditioned graph value model with registered auxiliary heads."""

    def __init__(self, config: QualityV2Config) -> None:
        super().__init__()
        if config.hidden <= 0 or config.layers <= 0 or config.heads <= 0:
            raise ValueError("hidden, layers, and heads must be positive")
        if config.robustness_dim != len(ROBUSTNESS_NAMES):
            raise ValueError("robustness_dim must match the registered robustness target schema")
        if config.arch not in {"mpnn", "gin", "gatv2", "gps"}:
            raise ValueError(f"unsupported quality V2 architecture {config.arch!r}")
        if config.arch in {"gatv2", "gps"} and config.hidden % config.heads:
            raise ValueError("hidden must be divisible by heads for attention backbones")
        self.config = config
        self.graph_encoder = _CandidateRepresentation(config)
        representation_dim = self.graph_encoder.output_dim
        self.task_adapter = nn.Sequential(
            nn.Linear(representation_dim, config.hidden),
            nn.ReLU(),
            nn.LayerNorm(config.hidden),
        )
        self.problem_adapter = nn.Sequential(
            nn.Linear(HAMILTONIAN_CONTEXT_DIMENSION, config.hidden),
            nn.ReLU(),
            nn.LayerNorm(config.hidden),
        )
        self.candidate_context = nn.Sequential(
            nn.Linear(3 * config.hidden, config.hidden),
            nn.ReLU(),
            nn.LayerNorm(config.hidden),
        )
        self.quality_head = nn.Linear(config.hidden, 2)
        self.capacity_head = nn.Linear(config.hidden, 1)
        self.robustness_head = nn.Linear(config.hidden, config.robustness_dim)
        self.terminal_qubits_head = (
            nn.Linear(config.hidden, 1) if config.predict_terminal_qubits else None
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_features: torch.Tensor,
        hamiltonian_context: torch.Tensor,
    ) -> QualityV2Output:
        if hamiltonian_context.shape != (HAMILTONIAN_CONTEXT_DIMENSION,) or not bool(
            torch.isfinite(hamiltonian_context).all()
        ):
            raise ValueError("hamiltonian_context must be one finite registered feature vector")
        representation = self.graph_encoder(x, adjacency, candidate_masks, candidate_features)
        local = self.task_adapter(representation)
        state = local.mean(dim=0, keepdim=True).expand_as(local)
        problem = self.problem_adapter(hamiltonian_context).unsqueeze(0).expand_as(local)
        conditioned = self.candidate_context(torch.cat((local, state, problem), dim=-1))
        quality = self.quality_head(conditioned)
        terminal = (
            F.softplus(self.terminal_qubits_head(conditioned).squeeze(-1))
            if self.terminal_qubits_head is not None
            else None
        )
        return QualityV2Output(
            quality_logit=quality[:, 0],
            quality_log_concentration=quality[:, 1],
            future_capacity=torch.sigmoid(self.capacity_head(conditioned).squeeze(-1)),
            robustness=F.softplus(self.robustness_head(conditioned)),
            terminal_qubits=terminal,
        )


def build_quality_v2_model(
    arch: str = "mpnn",
    hidden: int = 64,
    layers: int = 3,
    heads: int = 4,
    *,
    neighbour_feats: bool = False,
    predict_terminal_qubits: bool = False,
    robustness_dim: int = len(ROBUSTNESS_NAMES),
) -> nn.Module:
    config = QualityV2Config(
        arch=arch,
        hidden=hidden,
        layers=layers,
        heads=heads,
        neighbour_feats=neighbour_feats,
        predict_terminal_qubits=predict_terminal_qubits,
        robustness_dim=robustness_dim,
    )
    if arch == "hetero":
        if neighbour_feats:
            raise ValueError("neighbour_feats is not applicable to the heterogeneous backbone")
        from embedbench.models_quality_v2_hetero import build_quality_v2_hetero_model

        return build_quality_v2_hetero_model(
            hidden=hidden,
            layers=layers,
            heads=heads,
            predict_terminal_qubits=predict_terminal_qubits,
            robustness_dim=robustness_dim,
        )
    return QualityValueV2(config)


def tensors_from_chain(
    encoded: Any, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, ...]:
    """Extract only deployable, label-free arrays from a ``ChainEncoded`` value."""

    selected_device = torch.device(device)
    return (
        torch.as_tensor(encoded.x, dtype=torch.float32, device=selected_device),
        torch.as_tensor(encoded.adj, dtype=torch.float32, device=selected_device),
        torch.as_tensor(encoded.cand_masks, dtype=torch.float32, device=selected_device),
        torch.as_tensor(encoded.cand_feats, dtype=torch.float32, device=selected_device),
        torch.tensor(
            encoded.hamiltonian_context,
            dtype=torch.float32,
            device=selected_device,
        ),
    )


def _validated_mask(
    mask: torch.Tensor | None,
    target: torch.Tensor,
    expected_shape: torch.Size,
    name: str,
) -> torch.Tensor:
    if target.shape != expected_shape:
        raise ValueError(f"{name} target must have shape {tuple(expected_shape)}")
    if mask is None:
        return torch.isfinite(target)
    converted = mask.to(device=target.device, dtype=torch.bool)
    if converted.shape == expected_shape:
        return converted & torch.isfinite(target)
    if len(expected_shape) == 2 and converted.shape == expected_shape[:1]:
        return converted[:, None].expand(expected_shape) & torch.isfinite(target)
    raise ValueError(f"{name} mask must have shape {tuple(expected_shape)}")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return zero


def quality_v2_loss(
    output: QualityV2Output,
    *,
    p_solve: torch.Tensor,
    p_solve_mask: torch.Tensor | None = None,
    stage: torch.Tensor | None = None,
    future_capacity: torch.Tensor | None = None,
    future_capacity_mask: torch.Tensor | None = None,
    robustness: torch.Tensor | None = None,
    robustness_mask: torch.Tensor | None = None,
    terminal_qubits: torch.Tensor | None = None,
    terminal_qubits_mask: torch.Tensor | None = None,
    rank_margin: float = 0.05,
    stage1_rank_margin: float = 0.10,
    rank_weight: float = 0.5,
    lambda_capacity: float = 0.1,
    lambda_robustness: float = 0.1,
    lambda_terminal_q: float = 0.1,
    stage1_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Masked multi-head objective with within-state quality ranking.

    Stage-1 estimates remain usable for development training but receive lower probability
    and ranking weight.  Missing auxiliary targets contribute an exact differentiable zero.
    The legacy ``future_capacity`` argument is specifically the registered residual-
    connectivity proxy, not a continuation-value or feasibility-probability label.
    """

    if output.quality_logit.ndim != 1:
        raise ValueError("quality_logit must have one value per candidate")
    candidate_count = output.quality_logit.shape[0]
    expected = torch.Size((candidate_count,))
    target = p_solve.to(device=output.quality_logit.device, dtype=output.quality_logit.dtype)
    quality_mask = _validated_mask(p_solve_mask, target, expected, "p_solve")
    if bool(((target[quality_mask] < 0.0) | (target[quality_mask] > 1.0)).any()):
        raise ValueError("p_solve targets must lie in [0, 1]")
    if not bool(quality_mask.any()):
        raise ValueError("at least one p_solve target is required")
    if not math.isfinite(rank_margin) or rank_margin < 0.0:
        raise ValueError("rank_margin must be a finite non-negative number")
    if not math.isfinite(stage1_rank_margin) or stage1_rank_margin < rank_margin:
        raise ValueError("stage1_rank_margin must be finite and at least the stage-2 rank_margin")
    for name, value in (
        ("rank_weight", rank_weight),
        ("lambda_capacity", lambda_capacity),
        ("lambda_robustness", lambda_robustness),
        ("lambda_terminal_q", lambda_terminal_q),
        ("stage1_weight", stage1_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")

    if stage is None:
        stage_values = torch.full(expected, 2, device=target.device, dtype=torch.long)
    else:
        stage_values = stage.to(device=target.device, dtype=torch.long)
        if stage_values.shape != expected or bool(
            ((stage_values != 1) & (stage_values != 2)).any()
        ):
            raise ValueError("stage must contain one value in {1, 2} per candidate")
    fidelity_weight = torch.where(
        stage_values == 2,
        torch.ones_like(target),
        torch.full_like(target, stage1_weight),
    )

    mean = output.quality_mean
    concentration = output.quality_concentration
    # Replace masked non-finite placeholders before tensor arithmetic. Merely slicing them
    # out after BCE would leave NaNs in its backward pass (NaN multiplied by zero is NaN).
    safe_target = torch.where(quality_mask, target, torch.zeros_like(target))
    bce = F.binary_cross_entropy_with_logits(output.quality_logit, safe_target, reduction="none")
    # Positive heteroscedastic calibration term.  Its optimum trades residual error against
    # confidence, while the registered [1, 100] range prevents infinite concentration.
    calibration = (mean - safe_target).square() * concentration + torch.log(100.0 / concentration)
    probability_terms = fidelity_weight * (bce + 0.01 * calibration)
    quality_probability = probability_terms[quality_mask].sum() / fidelity_weight[
        quality_mask
    ].sum().clamp_min(torch.finfo(target.dtype).eps)

    zero = output.quality_logit.sum() * 0.0
    target_difference = safe_target[:, None] - safe_target[None, :]
    reliable_pair = (stage_values[:, None] == 2) & (stage_values[None, :] == 2)
    pair_margin = torch.where(
        reliable_pair,
        torch.full_like(target_difference, rank_margin),
        torch.full_like(target_difference, stage1_rank_margin),
    )
    pair_mask = (
        quality_mask[:, None]
        & quality_mask[None, :]
        & (target_difference > 0.0)
        & (target_difference >= pair_margin)
    )
    if bool(pair_mask.any()):
        score_difference = output.quality_logit[:, None] - output.quality_logit[None, :]
        rank_terms = F.softplus(-score_difference)
        pair_weight = torch.minimum(fidelity_weight[:, None], fidelity_weight[None, :])
        weighted_pair_mask = pair_weight * pair_mask
        within_state_rank = (
            rank_terms * weighted_pair_mask
        ).sum() / weighted_pair_mask.sum().clamp_min(torch.finfo(target.dtype).eps)
    else:
        within_state_rank = zero

    if future_capacity is None:
        capacity_loss = zero
    else:
        capacity_target = future_capacity.to(
            device=target.device, dtype=output.future_capacity.dtype
        )
        capacity_mask = _validated_mask(
            future_capacity_mask, capacity_target, expected, "future_capacity"
        )
        safe_capacity_target = torch.where(
            capacity_mask, capacity_target, torch.zeros_like(capacity_target)
        )
        capacity_loss = _masked_mean(
            (output.future_capacity - safe_capacity_target).square(), capacity_mask, zero
        )

    if robustness is None:
        robustness_loss = zero
    else:
        robustness_target = robustness.to(device=target.device, dtype=output.robustness.dtype)
        robustness_expected = output.robustness.shape
        robustness_mask_value = _validated_mask(
            robustness_mask, robustness_target, robustness_expected, "robustness"
        )
        safe_robustness_target = torch.where(
            robustness_mask_value, robustness_target, torch.zeros_like(robustness_target)
        )
        robustness_loss = _masked_mean(
            F.smooth_l1_loss(output.robustness, safe_robustness_target, reduction="none"),
            robustness_mask_value,
            zero,
        )

    if terminal_qubits is None:
        terminal_loss = zero
    else:
        if output.terminal_qubits is None:
            raise ValueError("terminal-qubit targets require predict_terminal_qubits=True")
        terminal_target = terminal_qubits.to(
            device=target.device, dtype=output.terminal_qubits.dtype
        )
        terminal_mask = _validated_mask(
            terminal_qubits_mask, terminal_target, expected, "terminal_qubits"
        )
        safe_terminal_target = torch.where(
            terminal_mask, terminal_target, torch.zeros_like(terminal_target)
        )
        terminal_loss = _masked_mean(
            F.smooth_l1_loss(output.terminal_qubits, safe_terminal_target, reduction="none"),
            terminal_mask,
            zero,
        )

    total = (
        quality_probability
        + rank_weight * within_state_rank
        + lambda_capacity * capacity_loss
        + lambda_robustness * robustness_loss
        + lambda_terminal_q * terminal_loss
    )
    return {
        "loss": total,
        "quality_probability": quality_probability,
        "within_state_rank": within_state_rank,
        "future_capacity": capacity_loss,
        "robustness": robustness_loss,
        "terminal_qubits": terminal_loss,
    }


def save_quality_v2_model(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a CPU-portable checkpoint with a complete reconstruction contract."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    supplied = dict(metadata or {})
    reserved = {"artifact_schema", "artifact_schema_version", "model_config"}
    overlap = reserved & supplied.keys()
    if overlap:
        raise ValueError(f"checkpoint metadata cannot override {sorted(overlap)}")
    config = getattr(model, "config", None)
    if not isinstance(config, QualityV2Config):
        raise TypeError("quality V2 checkpoint model must expose a QualityV2Config")
    if config.arch == "hetero":
        forward_inputs = [
            "qubit_features",
            "variable_features",
            "coupler_features",
            "logical_adjacency",
            "membership",
            "candidate_masks",
            "candidate_contacts",
            "focus_index",
            "focus_h",
            "hamiltonian_context",
        ]
    else:
        forward_inputs = [
            "x",
            "adjacency",
            "candidate_masks",
            "candidate_features",
            "hamiltonian_context",
        ]
    complete_metadata = {
        **supplied,
        "artifact_schema": ARTIFACT_SCHEMA,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_config": asdict(config),
        "robustness_names": list(ROBUSTNESS_NAMES),
        "selection_contract": SELECTION_CONTRACT,
        "auxiliary_heads_used_for_selection": False,
        "forward_inputs": forward_inputs,
    }
    state = {
        name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()
    }
    torch.save({"state": state, "meta": complete_metadata}, destination)


def load_quality_v2_model(
    path: str | Path,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a V2 model on CPU and return its validated portable metadata."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with the minimum torch extra
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("quality V2 checkpoint must be an object")
    metadata = checkpoint.get("meta")
    if not isinstance(metadata, Mapping):
        raise ValueError("quality V2 checkpoint has no metadata object")
    if (
        metadata.get("artifact_schema") != ARTIFACT_SCHEMA
        or metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported quality V2 checkpoint schema")
    raw_config = metadata.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("quality V2 checkpoint has no model_config")
    config = QualityV2Config(**dict(raw_config))
    model = build_quality_v2_model(
        arch=config.arch,
        hidden=config.hidden,
        layers=config.layers,
        heads=config.heads,
        neighbour_feats=config.neighbour_feats,
        predict_terminal_qubits=config.predict_terminal_qubits,
        robustness_dim=config.robustness_dim,
    )
    state = checkpoint.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("quality V2 checkpoint has no state dictionary")
    model.load_state_dict(state)
    model.eval()
    return model, dict(metadata)

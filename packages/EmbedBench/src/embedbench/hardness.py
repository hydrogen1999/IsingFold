"""Authenticated calibrated-hardness records for partial structural decisions.

This module separates complete starting-embedding provenance from a partial current context.
All externally supplied JSON artifacts are copied once, verified against independently supplied
digests, and converted to immutable tuples before any scientific measurement is made.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

import embedbench.candidate_protocol as candidate_protocol_module
import embedbench.exact as exact_module
import embedbench.hard_ood_schema as hard_ood_schema_module
import embedbench.objective as objective_module
import embedbench.realized_host as realized_host_module
from embedbench.candidate_protocol import (
    CandidateFacts,
    CandidateProtocolResult,
    FrozenChain,
    candidate_protocol_result_sha256,
    generate_candidate_bank,
)
from embedbench.exact import SearchAborted
from embedbench.ground_certificate import IsingProblem
from embedbench.hard_ood_schema import canonical_sha256
from embedbench.realized_host import (
    validate_minor_embedding,
    validate_realized_host_artifact,
    validate_window,
)

UINT64_MAX = 2**64 - 1

STARTING_EMBEDDING_SCHEMA = "embedbench.starting-embedding"
STARTING_EMBEDDING_SCHEMA_VERSION = 1
PARTIAL_CONTEXT_SCHEMA = "embedbench.partial-context"
PARTIAL_CONTEXT_SCHEMA_VERSION = 1

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_STARTING_SOURCES = frozenset({"witness", "minorminer", "cpp_baseline", "witness_perturbation"})
_VERIFICATION_SEAL = object()

_STARTING_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "problem_sha256",
        "host_sha256",
        "source",
        "source_artifact_sha256",
        "chains",
        "starting_embedding_sha256",
    }
)
_PARTIAL_CONTEXT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "problem_sha256",
        "host_sha256",
        "starting_embedding_sha256",
        "witness_embedding_sha256",
        "removal_rule_sha256",
        "focus",
        "frozen_chains",
        "window_nodes",
        "window_edges",
        "l_cap",
        "q_cap",
        "q_cap_slack",
        "witness_total_qubits",
        "terminal_q_cap",
        "partial_context_sha256",
    }
)


def _require_exact_keys(value: object, expected: Collection[str], name: str) -> dict[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise TypeError(f"{name} must be an exact JSON object with string keys")
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        raise ValueError(f"{name} schema fields differ: missing={missing}, unknown={unknown}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate")
    return value


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    integer = _require_int(value, name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _require_positive_int(value: object, name: str) -> int:
    integer = _require_int(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _require_uint64(value: object, name: str) -> int:
    integer = _require_int(value, name)
    if not 0 <= integer <= UINT64_MAX:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return integer


def _snapshot_json(value: object, name: str) -> dict[str, object]:
    """Take the sole caller-owned read before validating an external JSON artifact."""

    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    snapshot = copy.deepcopy(value)
    if type(snapshot) is not dict:
        raise TypeError(f"{name} snapshot must be an exact JSON object")
    return snapshot


def _canonical_nodes(value: object, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if type(value) not in {list, tuple}:
        raise TypeError(f"{name} must be an array or immutable tuple")
    nodes = tuple(_require_uint64(node, f"{name}[{index}]") for index, node in enumerate(value))
    if nonempty and not nodes:
        raise ValueError(f"{name} must not be empty")
    if nodes != tuple(sorted(nodes)) or len(nodes) != len(set(nodes)):
        raise ValueError(f"{name} must be sorted and duplicate-free")
    return nodes


def _nodes_from_json(value: object, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact JSON array")
    return _canonical_nodes(value, name, nonempty=nonempty)


def _canonical_edges(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if type(value) not in {list, tuple}:
        raise TypeError(f"{name} must be an array or immutable tuple")
    edges: list[tuple[int, int]] = []
    for index, raw in enumerate(value):
        if type(raw) not in {list, tuple} or len(raw) != 2:
            raise TypeError(f"{name}[{index}] must be a two-item edge")
        left = _require_uint64(raw[0], f"{name}[{index}][0]")
        right = _require_uint64(raw[1], f"{name}[{index}][1]")
        if left >= right:
            raise ValueError(f"{name} edges must be sorted pairs of distinct nodes")
        edges.append((left, right))
    result = tuple(edges)
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be sorted and duplicate-free")
    return result


def _edges_from_json(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if type(value) is not list or any(type(edge) is not list for edge in value):
        raise TypeError(f"{name} must be an exact JSON array of edge arrays")
    return _canonical_edges(value, name)


def _chains_from_json(value: object, name: str) -> tuple[FrozenChain, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact JSON array")
    chains: list[FrozenChain] = []
    for index, raw in enumerate(value):
        if type(raw) is not list or len(raw) != 2:
            raise TypeError(f"{name}[{index}] must be [logical_variable, chain_nodes]")
        logical_variable = _require_int(raw[0], f"{name}[{index}][0]")
        nodes = _nodes_from_json(raw[1], f"{name}[{index}][1]", nonempty=True)
        chains.append(FrozenChain(logical_variable, nodes))
    variables = tuple(chain.logical_variable for chain in chains)
    if variables != tuple(sorted(variables)) or len(variables) != len(set(variables)):
        raise ValueError(f"{name} must be sorted by duplicate-free logical variable")
    return tuple(chains)


def _chains_to_json(chains: tuple[FrozenChain, ...]) -> list[list[object]]:
    return [[chain.logical_variable, list(chain.nodes)] for chain in chains]


def _checked_frozen_chains(value: object, name: str) -> tuple[FrozenChain, ...]:
    if type(value) is not tuple or not all(type(chain) is FrozenChain for chain in value):
        raise TypeError(f"{name} must be an immutable tuple of exact FrozenChain values")
    chains = tuple(value)
    variables = tuple(chain.logical_variable for chain in chains)
    if variables != tuple(sorted(variables)) or len(variables) != len(set(variables)):
        raise ValueError(f"{name} must be sorted by duplicate-free logical variable")
    return chains


def _snapshot_graph(graph: nx.Graph, name: str = "graph") -> nx.Graph:
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TypeError(f"{name} must be an undirected simple NetworkX graph")
    nodes = tuple(sorted(_require_uint64(node, f"{name} node") for node in graph.nodes))
    edges = tuple(
        sorted(
            (
                min(
                    _require_uint64(left, "edge endpoint"), _require_uint64(right, "edge endpoint")
                ),
                max(
                    _require_uint64(left, "edge endpoint"), _require_uint64(right, "edge endpoint")
                ),
            )
            for left, right in graph.edges
        )
    )
    if any(left == right for left, right in edges):
        raise ValueError(f"{name} must not contain self-loops")
    detached = nx.Graph()
    detached.add_nodes_from(nodes)
    detached.add_edges_from(edges)
    return detached


@dataclass(frozen=True, slots=True)
class VerifiedProblem:
    """Immutable logical-graph view authenticated by an external problem digest."""

    problem_sha256: str
    variables: tuple[int, ...]
    logical_edges: tuple[tuple[int, int], ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VERIFICATION_SEAL:
            raise TypeError("VerifiedProblem values must be created by verify_problem")

    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.variables)
        graph.add_edges_from(self.logical_edges)
        return graph


@dataclass(frozen=True, slots=True)
class VerifiedHost:
    """Immutable realized-host snapshot authenticated against graph and artifact digests."""

    topology: str
    size: int
    requested_qubit_fraction: float
    requested_coupler_fraction: float
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    removed_nodes: tuple[int, ...]
    removed_edges: tuple[tuple[int, int], ...]
    host_sha256: str
    host_artifact_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VERIFICATION_SEAL:
            raise TypeError("VerifiedHost values must be created by verify_host")

    @property
    def pristine_node_count(self) -> int:
        return len(self.nodes) + len(self.removed_nodes)

    @property
    def post_qubit_edge_count(self) -> int:
        return len(self.edges) + len(self.removed_edges)

    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        return graph


@dataclass(frozen=True, slots=True)
class VerifiedStartingEmbedding:
    """Complete authenticated starting embedding used only for provenance and quotas."""

    problem_sha256: str
    host_sha256: str
    source: str
    source_artifact_sha256: str
    chains: tuple[FrozenChain, ...]
    starting_embedding_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VERIFICATION_SEAL:
            raise TypeError(
                "VerifiedStartingEmbedding values must be created by verify_starting_embedding"
            )

    @property
    def total_qubits(self) -> int:
        return sum(len(chain.nodes) for chain in self.chains)

    def chain_for(self, logical_variable: int) -> tuple[int, ...]:
        for chain in self.chains:
            if chain.logical_variable == logical_variable:
                return chain.nodes
        raise KeyError(logical_variable)

    def chain_map(self) -> dict[int, list[int]]:
        return {chain.logical_variable: list(chain.nodes) for chain in self.chains}


@dataclass(frozen=True, slots=True)
class PartialContext:
    """Authenticated placed context for one focus-chain decision."""

    problem_sha256: str
    host_sha256: str
    starting_embedding_sha256: str
    witness_embedding_sha256: str
    removal_rule_sha256: str
    focus: int
    frozen_chains: tuple[FrozenChain, ...]
    unplaced_variables: tuple[int, ...]
    required_focus_neighbors: tuple[int, ...]
    deferred_logical_edges: tuple[tuple[int, int], ...]
    window_nodes: tuple[int, ...]
    window_edges: tuple[tuple[int, int], ...]
    l_cap: int
    q_cap: None
    q_cap_slack: int
    witness_total_qubits: int
    terminal_q_cap: int
    partial_context_sha256: str
    original_focus_chain: tuple[int, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VERIFICATION_SEAL:
            raise TypeError("PartialContext values must be created by verify_partial_context")

    @property
    def placed_variables(self) -> tuple[int, ...]:
        return tuple(chain.logical_variable for chain in self.frozen_chains)

    @property
    def frozen_qubits(self) -> tuple[int, ...]:
        return tuple(sorted(node for chain in self.frozen_chains for node in chain.nodes))


def verify_problem(
    problem: IsingProblem,
    *,
    expected_problem_sha256: str,
) -> VerifiedProblem:
    """Bind one immutable Ising problem to an independently supplied identity."""

    if type(problem) is not IsingProblem:
        raise TypeError("problem must be an exact immutable IsingProblem")
    expected = _require_sha256(expected_problem_sha256, "expected_problem_sha256")
    if problem.problem_sha256 != expected:
        raise ValueError("problem digest disagrees with expected_problem_sha256")
    if len(problem.variables) < 2:
        raise ValueError("hardness schema requires at least two logical variables")
    edges = tuple((left, right) for left, right, _ in problem.quadratic)
    return VerifiedProblem(expected, tuple(problem.variables), edges, _VERIFICATION_SEAL)


def verify_host(
    artifact: object,
    *,
    seed_key: bytes,
    expected_host_artifact_sha256: str,
    expected_host_sha256: str,
) -> VerifiedHost:
    """Authenticate one detached realized-host artifact without rereading caller data."""

    snapshot = _snapshot_json(artifact, "realized host artifact")
    expected_artifact = _require_sha256(
        expected_host_artifact_sha256, "expected_host_artifact_sha256"
    )
    expected_graph = _require_sha256(expected_host_sha256, "expected_host_sha256")
    graph = validate_realized_host_artifact(
        snapshot,
        seed_key=seed_key,
        expected_host_artifact_sha256=expected_artifact,
        expected_host_sha256=expected_graph,
    )
    detached = _snapshot_graph(graph, "verified realized host")
    requested = snapshot["requested_defects"]
    return VerifiedHost(
        topology=_require_text(snapshot["topology"], "topology"),
        size=_require_positive_int(snapshot["size"], "size"),
        requested_qubit_fraction=float(requested["qubit_fraction"]),
        requested_coupler_fraction=float(requested["coupler_fraction"]),
        nodes=tuple(detached.nodes),
        edges=tuple(sorted((min(left, right), max(left, right)) for left, right in detached.edges)),
        removed_nodes=_canonical_nodes(snapshot["removed_nodes"], "removed_nodes"),
        removed_edges=_canonical_edges(snapshot["removed_edges"], "removed_edges"),
        host_sha256=expected_graph,
        host_artifact_sha256=expected_artifact,
        _seal=_VERIFICATION_SEAL,
    )


def build_starting_embedding_artifact(
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    source: str,
    source_artifact_sha256: str,
    chains: tuple[FrozenChain, ...],
) -> dict[str, object]:
    """Create a content-addressed complete starting-embedding artifact for generation."""

    if type(problem) is not VerifiedProblem or type(host) is not VerifiedHost:
        raise TypeError("problem and host must be verified boundary values")
    checked_source = _require_text(source, "source")
    if checked_source not in _STARTING_SOURCES:
        raise ValueError(f"source must be one of {sorted(_STARTING_SOURCES)}")
    checked_chains = _checked_frozen_chains(chains, "chains")
    variables = tuple(chain.logical_variable for chain in checked_chains)
    if variables != problem.variables:
        raise ValueError("starting embedding must contain every logical variable exactly once")
    validate_minor_embedding(
        host.graph(),
        logical_edges=[list(edge) for edge in problem.logical_edges],
        chains={chain.logical_variable: list(chain.nodes) for chain in checked_chains},
    )
    payload: dict[str, object] = {
        "schema": STARTING_EMBEDDING_SCHEMA,
        "schema_version": STARTING_EMBEDDING_SCHEMA_VERSION,
        "problem_sha256": problem.problem_sha256,
        "host_sha256": host.host_sha256,
        "source": checked_source,
        "source_artifact_sha256": _require_sha256(source_artifact_sha256, "source_artifact_sha256"),
        "chains": _chains_to_json(checked_chains),
    }
    return {**payload, "starting_embedding_sha256": canonical_sha256(payload)}


def verify_starting_embedding(
    artifact: object,
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    expected_starting_embedding_sha256: str,
) -> VerifiedStartingEmbedding:
    """Verify and detach a complete starting embedding against its external digest."""

    if type(problem) is not VerifiedProblem or type(host) is not VerifiedHost:
        raise TypeError("problem and host must be verified boundary values")
    snapshot = _snapshot_json(artifact, "starting embedding artifact")
    document = _require_exact_keys(snapshot, _STARTING_FIELDS, "starting embedding artifact")
    if document["schema"] != STARTING_EMBEDDING_SCHEMA:
        raise ValueError(f"starting embedding schema must equal {STARTING_EMBEDDING_SCHEMA!r}")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != STARTING_EMBEDDING_SCHEMA_VERSION
    ):
        raise ValueError(
            f"starting embedding schema_version must equal {STARTING_EMBEDDING_SCHEMA_VERSION}"
        )
    expected = _require_sha256(
        expected_starting_embedding_sha256, "expected_starting_embedding_sha256"
    )
    stored = _require_sha256(document["starting_embedding_sha256"], "starting_embedding_sha256")
    payload = {key: value for key, value in document.items() if key != "starting_embedding_sha256"}
    computed = canonical_sha256(payload)
    if stored != computed or stored != expected:
        raise ValueError(
            "starting embedding digest does not match its payload and external binding"
        )
    if document["problem_sha256"] != problem.problem_sha256:
        raise ValueError("starting embedding problem_sha256 does not match the verified problem")
    if document["host_sha256"] != host.host_sha256:
        raise ValueError("starting embedding host_sha256 does not match the verified host")
    source = _require_text(document["source"], "source")
    if source not in _STARTING_SOURCES:
        raise ValueError(f"source must be one of {sorted(_STARTING_SOURCES)}")
    chains = _chains_from_json(document["chains"], "chains")
    variables = tuple(chain.logical_variable for chain in chains)
    if variables != problem.variables:
        raise ValueError("starting embedding must contain every logical variable exactly once")
    validate_minor_embedding(
        host.graph(),
        logical_edges=[list(edge) for edge in problem.logical_edges],
        chains={chain.logical_variable: list(chain.nodes) for chain in chains},
    )
    return VerifiedStartingEmbedding(
        problem_sha256=problem.problem_sha256,
        host_sha256=host.host_sha256,
        source=source,
        source_artifact_sha256=_require_sha256(
            document["source_artifact_sha256"], "source_artifact_sha256"
        ),
        chains=chains,
        starting_embedding_sha256=stored,
        _seal=_VERIFICATION_SEAL,
    )


def _checked_frozen_variables(
    value: object, problem: VerifiedProblem, focus: int
) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError("frozen_variables must be an immutable tuple")
    variables = tuple(_require_int(item, "frozen variable") for item in value)
    if variables != tuple(sorted(variables)) or len(variables) != len(set(variables)):
        raise ValueError("frozen_variables must be sorted and duplicate-free")
    if focus in variables or not set(variables) < set(problem.variables):
        raise ValueError("frozen_variables must be a proper subset excluding focus")
    return variables


def build_partial_context_artifact(
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    starting_embedding: VerifiedStartingEmbedding,
    witness_embedding: VerifiedStartingEmbedding,
    focus: int,
    frozen_variables: tuple[int, ...],
    window_nodes: tuple[int, ...],
    window_edges: tuple[tuple[int, int], ...],
    l_cap: int,
    q_cap: None,
    q_cap_slack: int,
    removal_rule_sha256: str,
) -> dict[str, object]:
    """Create a content-addressed quota-directed partial-context artifact."""

    if (
        type(problem) is not VerifiedProblem
        or type(host) is not VerifiedHost
        or type(starting_embedding) is not VerifiedStartingEmbedding
        or type(witness_embedding) is not VerifiedStartingEmbedding
    ):
        raise TypeError(
            "problem, host, starting_embedding, and witness_embedding must be verified values"
        )
    if witness_embedding.source != "witness":
        raise ValueError("terminal resource cap requires a registered witness embedding")
    if (
        witness_embedding.problem_sha256 != problem.problem_sha256
        or witness_embedding.host_sha256 != host.host_sha256
    ):
        raise ValueError("witness embedding is not bound to the supplied problem and host")
    checked_focus = _require_int(focus, "focus")
    if checked_focus not in problem.variables:
        raise ValueError("focus must be a logical variable")
    checked_frozen = _checked_frozen_variables(frozen_variables, problem, checked_focus)
    chains = tuple(
        FrozenChain(variable, starting_embedding.chain_for(variable)) for variable in checked_frozen
    )
    checked_window = _canonical_nodes(window_nodes, "window_nodes", nonempty=True)
    checked_edges = _canonical_edges(window_edges, "window_edges")
    checked_l_cap = _require_positive_int(l_cap, "l_cap")
    if checked_l_cap > 6:
        raise ValueError("l_cap must not exceed the registered value 6")
    if q_cap is not None:
        raise ValueError("quota-directed partial-structural q_cap must be null")
    checked_slack = _require_nonnegative_int(q_cap_slack, "q_cap_slack")
    if checked_slack not in {0, 1}:
        raise ValueError("q_cap_slack must be exactly 0 or 1")
    witness_total = witness_embedding.total_qubits
    payload: dict[str, object] = {
        "schema": PARTIAL_CONTEXT_SCHEMA,
        "schema_version": PARTIAL_CONTEXT_SCHEMA_VERSION,
        "problem_sha256": problem.problem_sha256,
        "host_sha256": host.host_sha256,
        "starting_embedding_sha256": starting_embedding.starting_embedding_sha256,
        "witness_embedding_sha256": witness_embedding.starting_embedding_sha256,
        "removal_rule_sha256": _require_sha256(removal_rule_sha256, "removal_rule_sha256"),
        "focus": checked_focus,
        "frozen_chains": _chains_to_json(chains),
        "window_nodes": list(checked_window),
        "window_edges": [list(edge) for edge in checked_edges],
        "l_cap": checked_l_cap,
        "q_cap": None,
        "q_cap_slack": checked_slack,
        "witness_total_qubits": witness_total,
        "terminal_q_cap": witness_total + checked_slack,
    }
    document = {**payload, "partial_context_sha256": canonical_sha256(payload)}
    verify_partial_context(
        document,
        problem=problem,
        host=host,
        starting_embedding=starting_embedding,
        witness_embedding=witness_embedding,
        expected_partial_context_sha256=document["partial_context_sha256"],
    )
    return document


def verify_partial_context(
    artifact: object,
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    starting_embedding: VerifiedStartingEmbedding,
    witness_embedding: VerifiedStartingEmbedding,
    expected_partial_context_sha256: str,
) -> PartialContext:
    """Verify a partial context and derive all placed/deferred logical relations."""

    if (
        type(problem) is not VerifiedProblem
        or type(host) is not VerifiedHost
        or type(starting_embedding) is not VerifiedStartingEmbedding
        or type(witness_embedding) is not VerifiedStartingEmbedding
    ):
        raise TypeError(
            "problem, host, starting_embedding, and witness_embedding must be verified values"
        )
    if (
        starting_embedding.problem_sha256 != problem.problem_sha256
        or starting_embedding.host_sha256 != host.host_sha256
        or witness_embedding.problem_sha256 != problem.problem_sha256
        or witness_embedding.host_sha256 != host.host_sha256
    ):
        raise ValueError("starting or witness embedding is not bound to the problem and host")
    if witness_embedding.source != "witness":
        raise ValueError("terminal resource cap requires a registered witness embedding")
    snapshot = _snapshot_json(artifact, "partial context artifact")
    document = _require_exact_keys(snapshot, _PARTIAL_CONTEXT_FIELDS, "partial context artifact")
    if document["schema"] != PARTIAL_CONTEXT_SCHEMA:
        raise ValueError(f"partial context schema must equal {PARTIAL_CONTEXT_SCHEMA!r}")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != PARTIAL_CONTEXT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"partial context schema_version must equal {PARTIAL_CONTEXT_SCHEMA_VERSION}"
        )
    expected = _require_sha256(expected_partial_context_sha256, "expected_partial_context_sha256")
    stored = _require_sha256(document["partial_context_sha256"], "partial_context_sha256")
    payload = {key: value for key, value in document.items() if key != "partial_context_sha256"}
    if canonical_sha256(payload) != stored or stored != expected:
        raise ValueError("partial context digest does not match its payload and external binding")
    expected_bindings = {
        "problem_sha256": problem.problem_sha256,
        "host_sha256": host.host_sha256,
        "starting_embedding_sha256": starting_embedding.starting_embedding_sha256,
        "witness_embedding_sha256": witness_embedding.starting_embedding_sha256,
    }
    for name, expected_value in expected_bindings.items():
        if document[name] != expected_value:
            raise ValueError(f"partial context {name} does not match its verified source")

    focus = _require_int(document["focus"], "focus")
    if focus not in problem.variables:
        raise ValueError("focus must be a logical variable")
    frozen = _chains_from_json(document["frozen_chains"], "frozen_chains")
    placed = tuple(chain.logical_variable for chain in frozen)
    if focus in placed:
        raise ValueError("focus must not occur in frozen_chains")
    if not set(placed) < set(problem.variables):
        raise ValueError("frozen chains must be a proper subset of logical variables")
    for chain in frozen:
        if chain.nodes != starting_embedding.chain_for(chain.logical_variable):
            raise ValueError("frozen chains must equal their authenticated starting chains")
    unplaced = tuple(
        variable
        for variable in problem.variables
        if variable != focus and variable not in set(placed)
    )
    if not unplaced:
        raise ValueError("partial_structural context must contain at least one unplaced variable")

    window_nodes = _nodes_from_json(document["window_nodes"], "window_nodes", nonempty=True)
    window_edges = _edges_from_json(document["window_edges"], "window_edges")
    validate_window(
        host.graph(),
        list(window_nodes),
        [list(edge) for edge in window_edges],
    )
    active_variables = (focus, *unplaced)
    active_witness_nodes = {
        node for variable in active_variables for node in witness_embedding.chain_for(variable)
    }
    if not active_witness_nodes <= set(window_nodes):
        raise ValueError("window must contain the registered witness chains for active variables")
    free_window_count = len(
        set(window_nodes) - set(node for chain in frozen for node in chain.nodes)
    )
    if free_window_count > 28:
        raise ValueError("partial context exceeds 28 free window nodes")

    l_cap = _require_positive_int(document["l_cap"], "l_cap")
    if l_cap > 6:
        raise ValueError("l_cap must not exceed the registered value 6")
    if document["q_cap"] is not None:
        raise ValueError("quota-directed partial-structural q_cap must be null")
    slack = _require_nonnegative_int(document["q_cap_slack"], "q_cap_slack")
    if slack not in {0, 1}:
        raise ValueError("q_cap_slack must be exactly 0 or 1")
    witness_total = _require_positive_int(document["witness_total_qubits"], "witness_total_qubits")
    if witness_total != witness_embedding.total_qubits:
        raise ValueError("witness_total_qubits disagrees with the authenticated witness embedding")
    terminal_q_cap = _require_positive_int(document["terminal_q_cap"], "terminal_q_cap")
    if terminal_q_cap != witness_total + slack:
        raise ValueError("terminal_q_cap must equal witness_total_qubits plus q_cap_slack")

    graph = problem.graph()
    required_focus_neighbors = tuple(
        sorted(neighbor for neighbor in graph.neighbors(focus) if neighbor in set(placed))
    )
    deferred = tuple(
        edge
        for edge in problem.logical_edges
        if focus in edge and (edge[1] if edge[0] == focus else edge[0]) in set(unplaced)
    )
    return PartialContext(
        problem_sha256=problem.problem_sha256,
        host_sha256=host.host_sha256,
        starting_embedding_sha256=starting_embedding.starting_embedding_sha256,
        witness_embedding_sha256=witness_embedding.starting_embedding_sha256,
        removal_rule_sha256=_require_sha256(document["removal_rule_sha256"], "removal_rule_sha256"),
        focus=focus,
        frozen_chains=frozen,
        unplaced_variables=unplaced,
        required_focus_neighbors=required_focus_neighbors,
        deferred_logical_edges=deferred,
        window_nodes=window_nodes,
        window_edges=window_edges,
        l_cap=l_cap,
        q_cap=None,
        q_cap_slack=slack,
        witness_total_qubits=witness_total,
        terminal_q_cap=terminal_q_cap,
        partial_context_sha256=stored,
        original_focus_chain=starting_embedding.chain_for(focus),
        _seal=_VERIFICATION_SEAL,
    )


def residual_hardness(realized_host: nx.Graph, occupied_nodes: object) -> dict[str, object]:
    """Return exact residual-graph measurements using the registered LCC tie-break."""

    graph = _snapshot_graph(realized_host, "realized_host")
    occupied = _canonical_nodes(occupied_nodes, "occupied_nodes")
    host_nodes = set(graph.nodes)
    missing = tuple(sorted(set(occupied) - host_nodes))
    if missing:
        raise ValueError(f"occupied_nodes contains nodes absent from realized_host: {missing}")
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("residual hardness requires a nonempty host with at least one edge")

    free_nodes = tuple(sorted(host_nodes - set(occupied)))
    free_graph = graph.subgraph(free_nodes).copy()
    components = sorted(
        (tuple(sorted(component)) for component in nx.connected_components(free_graph)),
        key=lambda component: (-len(component), component),
    )
    largest_graph = free_graph.subgraph(components[0]).copy() if components else nx.Graph()
    host_node_count = graph.number_of_nodes()
    host_edge_count = graph.number_of_edges()
    largest_node_count = largest_graph.number_of_nodes()
    edge_connectivity = 0 if largest_node_count < 2 else int(nx.edge_connectivity(largest_graph))
    free_edge_count = free_graph.number_of_edges()
    largest_edge_count = largest_graph.number_of_edges()
    return {
        "free_node_numerator": len(free_nodes),
        "free_node_denominator": host_node_count,
        "free_node_fraction": len(free_nodes) / host_node_count,
        "free_edge_numerator": free_edge_count,
        "free_edge_denominator": host_edge_count,
        "free_edge_fraction": free_edge_count / host_edge_count,
        "largest_free_component_node_numerator": largest_node_count,
        "largest_free_component_node_denominator": host_node_count,
        "largest_free_component_node_fraction": largest_node_count / host_node_count,
        "largest_free_component_edge_numerator": largest_edge_count,
        "largest_free_component_edge_denominator": host_edge_count,
        "largest_free_component_edge_fraction": largest_edge_count / host_edge_count,
        "largest_free_component_edge_connectivity": edge_connectivity,
        "articulation_count": sum(1 for _ in nx.articulation_points(free_graph)),
        "largest_free_component_articulation_count": sum(
            1 for _ in nx.articulation_points(largest_graph)
        ),
    }


HARDNESS_SCHEMA = "embedbench.hardness"
HARDNESS_SCHEMA_VERSION = 1
CONTINUATION_CERTIFICATE_SCHEMA = "embedbench.exact-continuation-certificate"
CONTINUATION_CERTIFICATE_SCHEMA_VERSION = 1
CONTINUATION_CHECKER_ACCEPTANCE_SCHEMA = "embedbench.exact-continuation-checker-acceptance"
CONTINUATION_CHECKER_ACCEPTANCE_SCHEMA_VERSION = 1
EXACT_COMPLETION_NODE_BUDGET = 10_000_000

_CONTINUATION_CERTIFICATE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "problem_sha256",
        "host_sha256",
        "partial_context_sha256",
        "candidate_protocol_result_sha256",
        "canonical_index",
        "candidate",
        "node_budget",
        "expanded_nodes",
        "exact_completion_status",
        "completion_feasible",
        "terminal_total_qubits",
        "terminal_maximum_chain_length",
        "largest_free_component_node_numerator",
        "largest_free_component_node_denominator",
        "largest_free_component_edge_connectivity",
        "largest_free_component_articulation_count",
        "minimum_logical_contact_multiplicity",
        "assignment_sha256",
        "certificate_sha256",
    }
)
_CHECKER_ACCEPTANCE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "certificate_sha256",
        "checker_source_sha256",
        "checker_environment_sha256",
        "assignment_sha256",
        "checker_exit_code",
        "checker_acceptance_sha256",
    }
)
_HARDNESS_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "profile_id",
        "sources",
        "host",
        "logical",
        "starting_embedding",
        "partial_context",
        "reference_candidate_index",
        "candidates",
        "decision",
        "solver_profile_sha256",
    }
)
_HARDNESS_SOURCE_FIELDS = frozenset(
    {
        "problem_sha256",
        "host_artifact_sha256",
        "host_sha256",
        "starting_embedding_sha256",
        "witness_embedding_sha256",
        "partial_context_sha256",
        "candidate_protocol_result_sha256",
    }
)
_HARDNESS_HOST_FIELDS = frozenset(
    {
        "nodes",
        "edges",
        "component_count",
        "defect_qubit_numerator",
        "defect_qubit_denominator",
        "defect_coupler_numerator",
        "defect_coupler_denominator",
    }
)
_HARDNESS_LOGICAL_FIELDS = frozenset(
    {
        "variables",
        "edges",
        "average_degree_numerator",
        "average_degree_denominator",
        "density_numerator",
        "density_denominator",
        "maximum_degree",
        "component_count",
    }
)
_HARDNESS_CHAIN_FIELDS = frozenset(
    {
        "chain_count",
        "total_qubits",
        "host_fill_numerator",
        "host_fill_denominator",
        "mean_chain_length_numerator",
        "mean_chain_length_denominator",
        "maximum_chain_length",
        "chain_length_stddev",
    }
)
_HARDNESS_CONTEXT_FIELDS = frozenset(
    {
        "placed_variables",
        "unplaced_variables",
        "focus",
        "occupied_qubits",
        "q_cap",
        "terminal_q_cap",
    }
)
_HARDNESS_CANDIDATE_FIELDS = frozenset(
    {
        "canonical_index",
        "candidate",
        "immediate",
        "post_replacement_embedding",
        "residual",
        "deferred_logical_edges",
        "continuation",
    }
)
_HARDNESS_IMMEDIATE_FIELDS = frozenset(
    {
        "connected",
        "chain_disjoint",
        "realizes_every_required_coupler",
        "within_l_cap",
        "minor_valid",
        "current_total_qubits",
        "current_maximum_chain_length",
        "within_q_cap",
        "feasible_now",
    }
)
_HARDNESS_RESIDUAL_FIELDS = frozenset(
    {
        "free_node_numerator",
        "free_node_denominator",
        "free_edge_numerator",
        "free_edge_denominator",
        "largest_free_component_node_numerator",
        "largest_free_component_node_denominator",
        "largest_free_component_edge_numerator",
        "largest_free_component_edge_denominator",
        "largest_free_component_edge_connectivity",
        "articulation_count",
        "largest_free_component_articulation_count",
    }
)
_HARDNESS_CONTINUATION_FIELDS = frozenset(
    {
        "exact_completion_status",
        "certificate_sha256",
        "checker_acceptance_sha256",
        "completion_feasible",
        "terminal_total_qubits",
        "terminal_maximum_chain_length",
        "largest_free_component_node_numerator",
        "largest_free_component_node_denominator",
        "largest_free_component_edge_connectivity",
        "largest_free_component_articulation_count",
        "minimum_logical_contact_multiplicity",
    }
)
_HARDNESS_DECISION_FIELDS = frozenset(
    {
        "focus_degree",
        "window_nodes",
        "window_edges",
        "full_candidate_count",
        "offered_candidate_count",
        "exact_completion_numerator",
        "exact_completion_denominator",
        "exact_completion_fraction",
    }
)
_SUBJECTIVE_PROFILE_TOKENS = frozenset({"base", "easy", "hard", "medium"})


def _require_registered_profile_name(value: object) -> str:
    profile_id = _require_text(value, "profile_id")
    tokens = frozenset(token for token in re.split(r"[^a-z0-9]+", profile_id.lower()) if token)
    if tokens & _SUBJECTIVE_PROFILE_TOKENS:
        raise ValueError("profile_id must not contain a subjective hardness label")
    return profile_id


def _validate_hardness_document_structure(value: object) -> dict[str, object]:
    document = _require_exact_keys(value, _HARDNESS_FIELDS, "hardness")
    if type(document["schema"]) is not str or document["schema"] != HARDNESS_SCHEMA:
        raise ValueError(f"hardness schema must equal {HARDNESS_SCHEMA!r}")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != HARDNESS_SCHEMA_VERSION
    ):
        raise ValueError(f"hardness schema_version must equal {HARDNESS_SCHEMA_VERSION}")
    _require_registered_profile_name(document["profile_id"])
    sources = _require_exact_keys(document["sources"], _HARDNESS_SOURCE_FIELDS, "hardness sources")
    for field_name, digest in sources.items():
        _require_sha256(digest, f"hardness sources.{field_name}")
    _require_exact_keys(document["host"], _HARDNESS_HOST_FIELDS, "hardness host")
    _require_exact_keys(document["logical"], _HARDNESS_LOGICAL_FIELDS, "hardness logical")
    _require_exact_keys(
        document["starting_embedding"],
        _HARDNESS_CHAIN_FIELDS,
        "hardness starting_embedding",
    )
    _require_exact_keys(
        document["partial_context"],
        _HARDNESS_CONTEXT_FIELDS,
        "hardness partial_context",
    )
    _require_nonnegative_int(document["reference_candidate_index"], "reference_candidate_index")
    candidates = document["candidates"]
    if type(candidates) is not list or not candidates:
        raise TypeError("hardness candidates must be a nonempty exact JSON array")
    for index, candidate_value in enumerate(candidates):
        candidate = _require_exact_keys(
            candidate_value,
            _HARDNESS_CANDIDATE_FIELDS,
            f"hardness candidates[{index}]",
        )
        _nodes_from_json(
            candidate["candidate"],
            f"hardness candidates[{index}].candidate",
            nonempty=True,
        )
        _require_exact_keys(
            candidate["immediate"],
            _HARDNESS_IMMEDIATE_FIELDS,
            f"hardness candidates[{index}].immediate",
        )
        _require_exact_keys(
            candidate["post_replacement_embedding"],
            _HARDNESS_CHAIN_FIELDS,
            f"hardness candidates[{index}].post_replacement_embedding",
        )
        _require_exact_keys(
            candidate["residual"],
            _HARDNESS_RESIDUAL_FIELDS,
            f"hardness candidates[{index}].residual",
        )
        _edges_from_json(
            candidate["deferred_logical_edges"],
            f"hardness candidates[{index}].deferred_logical_edges",
        )
        _require_exact_keys(
            candidate["continuation"],
            _HARDNESS_CONTINUATION_FIELDS,
            f"hardness candidates[{index}].continuation",
        )
    _require_exact_keys(document["decision"], _HARDNESS_DECISION_FIELDS, "hardness decision")
    _require_sha256(document["solver_profile_sha256"], "solver_profile_sha256")
    return document


@dataclass(frozen=True, slots=True)
class VerifiedContinuationEvidence:
    """One checker-replayed, content-bound exact continuation result."""

    canonical_index: int
    candidate: tuple[int, ...]
    exact_completion_status: str
    completion_feasible: bool | None
    terminal_total_qubits: int | None
    terminal_maximum_chain_length: int | None
    largest_free_component_node_numerator: int | None
    largest_free_component_node_denominator: int | None
    largest_free_component_edge_connectivity: int | None
    largest_free_component_articulation_count: int | None
    minimum_logical_contact_multiplicity: int | None
    assignment_sha256: str | None
    certificate_sha256: str
    checker_acceptance_sha256: str
    expanded_nodes: int
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VERIFICATION_SEAL:
            raise TypeError("VerifiedContinuationEvidence values must be created by checker replay")

    def summary(self) -> dict[str, object]:
        return {
            "exact_completion_status": self.exact_completion_status,
            "certificate_sha256": self.certificate_sha256,
            "checker_acceptance_sha256": self.checker_acceptance_sha256,
            "completion_feasible": self.completion_feasible,
            "terminal_total_qubits": self.terminal_total_qubits,
            "terminal_maximum_chain_length": self.terminal_maximum_chain_length,
            "largest_free_component_node_numerator": (self.largest_free_component_node_numerator),
            "largest_free_component_node_denominator": (
                self.largest_free_component_node_denominator
            ),
            "largest_free_component_edge_connectivity": (
                self.largest_free_component_edge_connectivity
            ),
            "largest_free_component_articulation_count": (
                self.largest_free_component_articulation_count
            ),
            "minimum_logical_contact_multiplicity": (self.minimum_logical_contact_multiplicity),
        }


@dataclass(frozen=True, slots=True, init=False)
class VerifiedHardness:
    """Immutable content-addressed hardness object produced only by recomputation."""

    _serialized: str
    hardness_sha256: str
    _seal: object

    def to_dict(self) -> dict[str, object]:
        document = json.loads(self._serialized)
        if type(document) is not dict:  # pragma: no cover - constructor invariant
            raise RuntimeError("verified hardness payload is not an object")
        return document


def continuation_checker_source_sha256() -> str:
    """Hash every local source file that participates in continuation replay."""

    modules = (
        ("embedbench/candidate_protocol.py", candidate_protocol_module),
        ("embedbench/exact.py", exact_module),
        ("embedbench/hard_ood_schema.py", hard_ood_schema_module),
        ("embedbench/hardness.py", None),
        ("embedbench/objective.py", objective_module),
        ("embedbench/realized_host.py", realized_host_module),
    )
    entries: list[list[str]] = []
    for relative_name, module in modules:
        source_path = Path(__file__) if module is None else Path(module.__file__)
        entries.append([relative_name, hashlib.sha256(source_path.read_bytes()).hexdigest()])
    return canonical_sha256(
        {
            "schema": "embedbench.exact-continuation-checker-source-set",
            "schema_version": 1,
            "files": entries,
        }
    )


def _contact_count(
    host: nx.Graph,
    left_chain: Collection[int],
    right_chain: Collection[int],
) -> int:
    return sum(1 for left in left_chain for right in right_chain if host.has_edge(left, right))


def _assignment_sha256(
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    context: PartialContext,
    canonical_index: int,
    chains: dict[int, tuple[int, ...]],
) -> str:
    return canonical_sha256(
        {
            "schema": "embedbench.exact-continuation-assignment",
            "schema_version": 1,
            "problem_sha256": problem.problem_sha256,
            "host_sha256": host.host_sha256,
            "partial_context_sha256": context.partial_context_sha256,
            "canonical_index": canonical_index,
            "chains": [[variable, list(chains[variable])] for variable in sorted(chains)],
        }
    )


def _connected_subsets(
    graph: nx.Graph,
    free_nodes: tuple[int, ...],
    maximum_size: int,
):
    for length in range(1, min(maximum_size, len(free_nodes)) + 1):
        for candidate in itertools.combinations(free_nodes, length):
            if length == 1 or nx.is_connected(graph.subgraph(candidate)):
                yield candidate


def _replay_continuation(
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    context: PartialContext,
    canonical_index: int,
    candidate: tuple[int, ...],
    node_budget: int,
) -> dict[str, object]:
    """Exhaustively maximize Section 7.3 T(c), charging one node per DFS state."""

    checked_node_budget = _require_positive_int(node_budget, "node_budget")

    host_graph = host.graph()
    logical_graph = problem.graph()
    assigned: dict[int, tuple[int, ...]] = {
        chain.logical_variable: chain.nodes for chain in context.frozen_chains
    }
    assigned[context.focus] = candidate
    used = {node for chain in assigned.values() for node in chain}
    if len(used) != sum(len(chain) for chain in assigned.values()):
        raise ValueError("candidate overlaps the authenticated frozen context")
    available_window = tuple(sorted(set(context.window_nodes) - used))
    window_graph = host_graph.subgraph(context.window_nodes).copy()
    order = tuple(
        sorted(
            context.unplaced_variables,
            key=lambda variable: (-logical_graph.degree(variable), variable),
        )
    )
    expanded_nodes = 0
    best_score: tuple[int, int, int, int, int, int, int] | None = None
    best_assignment_key: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    best_measurements: dict[str, object] | None = None

    def expand() -> None:
        nonlocal expanded_nodes
        if expanded_nodes >= checked_node_budget:
            raise SearchAborted(f"exact continuation exceeded {checked_node_budget} nodes")
        expanded_nodes += 1

    def visit(position: int, free: tuple[int, ...], current_qubits: int) -> None:
        nonlocal best_score, best_assignment_key, best_measurements
        expand()
        remaining = len(order) - position
        if current_qubits + remaining > context.terminal_q_cap:
            return
        if position == len(order):
            complete = {variable: tuple(nodes) for variable, nodes in assigned.items()}
            validate_minor_embedding(
                host_graph,
                logical_edges=[list(edge) for edge in problem.logical_edges],
                chains={variable: list(nodes) for variable, nodes in complete.items()},
            )
            occupied = tuple(sorted(node for chain in complete.values() for node in chain))
            residual = residual_hardness(host_graph, occupied)
            contact_counts = tuple(
                _contact_count(host_graph, complete[left], complete[right])
                for left, right in problem.logical_edges
            )
            terminal_maximum = max(len(chain) for chain in complete.values())
            measurements: dict[str, object] = {
                "terminal_total_qubits": current_qubits,
                "terminal_maximum_chain_length": terminal_maximum,
                "largest_free_component_node_numerator": residual[
                    "largest_free_component_node_numerator"
                ],
                "largest_free_component_node_denominator": residual[
                    "largest_free_component_node_denominator"
                ],
                "largest_free_component_edge_connectivity": residual[
                    "largest_free_component_edge_connectivity"
                ],
                "largest_free_component_articulation_count": residual[
                    "largest_free_component_articulation_count"
                ],
                "minimum_logical_contact_multiplicity": min(contact_counts, default=0),
            }
            score = (
                1,
                int(measurements["largest_free_component_node_numerator"]),
                int(measurements["largest_free_component_edge_connectivity"]),
                -int(measurements["largest_free_component_articulation_count"]),
                int(measurements["minimum_logical_contact_multiplicity"]),
                -current_qubits,
                -terminal_maximum,
            )
            assignment_key = tuple((variable, complete[variable]) for variable in sorted(complete))
            if (
                best_score is None
                or score > best_score
                or (score == best_score and assignment_key < best_assignment_key)
            ):
                best_score = score
                best_assignment_key = assignment_key
                measurements["assignment_sha256"] = _assignment_sha256(
                    problem=problem,
                    host=host,
                    context=context,
                    canonical_index=canonical_index,
                    chains=complete,
                )
                best_measurements = measurements
            return

        variable = order[position]
        required_placed_neighbors = tuple(
            neighbor for neighbor in logical_graph.neighbors(variable) if neighbor in assigned
        )
        for chain in _connected_subsets(window_graph, free, context.l_cap):
            if not all(
                _contact_count(host_graph, chain, assigned[neighbor]) > 0
                for neighbor in required_placed_neighbors
            ):
                continue
            assigned[variable] = chain
            chain_nodes = set(chain)
            visit(
                position + 1,
                tuple(node for node in free if node not in chain_nodes),
                current_qubits + len(chain),
            )
            del assigned[variable]

    initial_qubits = sum(len(chain) for chain in assigned.values())
    try:
        visit(0, available_window, initial_qubits)
    except SearchAborted:
        return {
            "expanded_nodes": expanded_nodes,
            "exact_completion_status": "node_budget_exhausted",
            "completion_feasible": None,
            "terminal_total_qubits": None,
            "terminal_maximum_chain_length": None,
            "largest_free_component_node_numerator": None,
            "largest_free_component_node_denominator": None,
            "largest_free_component_edge_connectivity": None,
            "largest_free_component_articulation_count": None,
            "minimum_logical_contact_multiplicity": None,
            "assignment_sha256": None,
        }
    if best_measurements is None:
        return {
            "expanded_nodes": expanded_nodes,
            "exact_completion_status": "complete",
            "completion_feasible": False,
            "terminal_total_qubits": None,
            "terminal_maximum_chain_length": None,
            "largest_free_component_node_numerator": None,
            "largest_free_component_node_denominator": None,
            "largest_free_component_edge_connectivity": None,
            "largest_free_component_articulation_count": None,
            "minimum_logical_contact_multiplicity": None,
            "assignment_sha256": None,
        }
    return {
        "expanded_nodes": expanded_nodes,
        "exact_completion_status": "complete",
        "completion_feasible": True,
        **best_measurements,
    }


def _build_continuation_artifacts_from_environment_snapshot(
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    partial_context: PartialContext,
    candidate_protocol_result: CandidateProtocolResult,
    canonical_index: int,
    checker_environment_snapshot: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if type(candidate_protocol_result) is not CandidateProtocolResult:
        raise TypeError("candidate_protocol_result must be an exact immutable result")
    if candidate_protocol_result.full_candidates is None:
        raise ValueError("continuation requires a complete full candidate bank")
    if candidate_protocol_result.offered_to_full_indices is None:
        raise ValueError("continuation requires an offered candidate bank")
    checked_index = _require_nonnegative_int(canonical_index, "canonical_index")
    if checked_index not in candidate_protocol_result.offered_to_full_indices:
        raise ValueError("continuation canonical_index must name an offered candidate")
    candidate = candidate_protocol_result.full_candidates[checked_index]
    result_digest = candidate_protocol_result_sha256(candidate_protocol_result)
    replay = _replay_continuation(
        problem=problem,
        host=host,
        context=partial_context,
        canonical_index=checked_index,
        candidate=candidate,
        node_budget=EXACT_COMPLETION_NODE_BUDGET,
    )
    certificate_payload: dict[str, object] = {
        "schema": CONTINUATION_CERTIFICATE_SCHEMA,
        "schema_version": CONTINUATION_CERTIFICATE_SCHEMA_VERSION,
        "problem_sha256": problem.problem_sha256,
        "host_sha256": host.host_sha256,
        "partial_context_sha256": partial_context.partial_context_sha256,
        "candidate_protocol_result_sha256": result_digest,
        "canonical_index": checked_index,
        "candidate": list(candidate),
        "node_budget": EXACT_COMPLETION_NODE_BUDGET,
        **replay,
    }
    certificate = {
        **certificate_payload,
        "certificate_sha256": canonical_sha256(certificate_payload),
    }
    acceptance_payload: dict[str, object] = {
        "schema": CONTINUATION_CHECKER_ACCEPTANCE_SCHEMA,
        "schema_version": CONTINUATION_CHECKER_ACCEPTANCE_SCHEMA_VERSION,
        "certificate_sha256": certificate["certificate_sha256"],
        "checker_source_sha256": continuation_checker_source_sha256(),
        "checker_environment_sha256": canonical_sha256(checker_environment_snapshot),
        "assignment_sha256": certificate["assignment_sha256"],
        "checker_exit_code": 0,
    }
    acceptance = {
        **acceptance_payload,
        "checker_acceptance_sha256": canonical_sha256(acceptance_payload),
    }
    return certificate, acceptance


def _replay_candidate_protocol_for_continuation(
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    partial_context: PartialContext,
    candidate_protocol_result: CandidateProtocolResult,
    candidate_sample_seed_key: bytes,
    expected_candidate_protocol_result_sha256: str,
) -> CandidateProtocolResult:
    if (
        type(problem) is not VerifiedProblem
        or type(host) is not VerifiedHost
        or type(partial_context) is not PartialContext
    ):
        raise TypeError("continuation sources must be exact verified boundary values")
    if (
        partial_context.problem_sha256 != problem.problem_sha256
        or partial_context.host_sha256 != host.host_sha256
    ):
        raise ValueError("continuation sources do not share the same problem and host identity")
    if type(candidate_protocol_result) is not CandidateProtocolResult:
        raise TypeError("candidate_protocol_result must be an exact immutable result")
    expected = _require_sha256(
        expected_candidate_protocol_result_sha256,
        "expected_candidate_protocol_result_sha256",
    )
    replayed = generate_candidate_bank(
        host.graph(),
        window_nodes=partial_context.window_nodes,
        frozen_chains=partial_context.frozen_chains,
        required_logical_neighbors=partial_context.required_focus_neighbors,
        original_focus_chain=partial_context.original_focus_chain,
        l_cap=partial_context.l_cap,
        q_cap=partial_context.q_cap,
        candidate_sample_seed_key=candidate_sample_seed_key,
    )
    replayed_digest = candidate_protocol_result_sha256(replayed)
    supplied_digest = candidate_protocol_result_sha256(candidate_protocol_result)
    if (
        supplied_digest != expected
        or replayed_digest != expected
        or candidate_protocol_result != replayed
    ):
        raise ValueError("candidate protocol result disagrees with its binding and exact replay")
    return replayed


def build_continuation_artifacts(
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    partial_context: PartialContext,
    candidate_protocol_result: CandidateProtocolResult,
    candidate_sample_seed_key: bytes,
    expected_candidate_protocol_result_sha256: str,
    canonical_index: int,
    checker_environment: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the exact checker and build its content-addressed certificate and acceptance."""

    replayed_result = _replay_candidate_protocol_for_continuation(
        problem=problem,
        host=host,
        partial_context=partial_context,
        candidate_protocol_result=candidate_protocol_result,
        candidate_sample_seed_key=candidate_sample_seed_key,
        expected_candidate_protocol_result_sha256=expected_candidate_protocol_result_sha256,
    )
    environment_snapshot = _snapshot_json(checker_environment, "checker_environment")
    canonical_sha256(environment_snapshot)
    return _build_continuation_artifacts_from_environment_snapshot(
        problem=problem,
        host=host,
        partial_context=partial_context,
        candidate_protocol_result=replayed_result,
        canonical_index=canonical_index,
        checker_environment_snapshot=environment_snapshot,
    )


def verify_continuation_evidence(
    certificate_document: object,
    checker_acceptance_document: object,
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    partial_context: PartialContext,
    candidate_protocol_result: CandidateProtocolResult,
    candidate_sample_seed_key: bytes,
    expected_candidate_protocol_result_sha256: str,
    canonical_index: int,
    checker_environment: object,
    expected_certificate_sha256: str,
    expected_checker_acceptance_sha256: str,
    expected_checker_source_sha256: str,
    expected_checker_environment_sha256: str,
    expected_assignment_sha256: str | None,
) -> VerifiedContinuationEvidence:
    """Replay exact completion and authenticate every certificate dependency."""

    replayed_result = _replay_candidate_protocol_for_continuation(
        problem=problem,
        host=host,
        partial_context=partial_context,
        candidate_protocol_result=candidate_protocol_result,
        candidate_sample_seed_key=candidate_sample_seed_key,
        expected_candidate_protocol_result_sha256=expected_candidate_protocol_result_sha256,
    )

    certificate = _require_exact_keys(
        _snapshot_json(certificate_document, "continuation certificate"),
        _CONTINUATION_CERTIFICATE_FIELDS,
        "continuation certificate",
    )
    acceptance = _require_exact_keys(
        _snapshot_json(checker_acceptance_document, "checker acceptance"),
        _CHECKER_ACCEPTANCE_FIELDS,
        "checker acceptance",
    )
    environment = _snapshot_json(checker_environment, "checker_environment")
    expected_certificate_digest = _require_sha256(
        expected_certificate_sha256, "expected_certificate_sha256"
    )
    expected_acceptance_digest = _require_sha256(
        expected_checker_acceptance_sha256, "expected_checker_acceptance_sha256"
    )
    expected_source_digest = _require_sha256(
        expected_checker_source_sha256, "expected_checker_source_sha256"
    )
    expected_environment_digest = _require_sha256(
        expected_checker_environment_sha256, "expected_checker_environment_sha256"
    )
    if expected_assignment_sha256 is not None:
        _require_sha256(expected_assignment_sha256, "expected_assignment_sha256")

    certificate_payload = {
        key: value for key, value in certificate.items() if key != "certificate_sha256"
    }
    acceptance_payload = {
        key: value for key, value in acceptance.items() if key != "checker_acceptance_sha256"
    }
    if (
        canonical_sha256(certificate_payload) != certificate["certificate_sha256"]
        or certificate["certificate_sha256"] != expected_certificate_digest
    ):
        raise ValueError("continuation certificate digest is not externally authenticated")
    if (
        canonical_sha256(acceptance_payload) != acceptance["checker_acceptance_sha256"]
        or acceptance["checker_acceptance_sha256"] != expected_acceptance_digest
    ):
        raise ValueError("checker acceptance digest is not externally authenticated")
    actual_source_digest = continuation_checker_source_sha256()
    if (
        acceptance["checker_source_sha256"] != expected_source_digest
        or actual_source_digest != expected_source_digest
    ):
        raise ValueError("checker source digest does not authenticate the executing checker")
    actual_environment_digest = canonical_sha256(environment)
    if (
        acceptance["checker_environment_sha256"] != expected_environment_digest
        or actual_environment_digest != expected_environment_digest
    ):
        raise ValueError("checker environment digest is not externally authenticated")
    if certificate["assignment_sha256"] != expected_assignment_sha256:
        raise ValueError("continuation assignment digest is not externally authenticated")

    replayed_certificate, replayed_acceptance = (
        _build_continuation_artifacts_from_environment_snapshot(
            problem=problem,
            host=host,
            partial_context=partial_context,
            candidate_protocol_result=replayed_result,
            canonical_index=canonical_index,
            checker_environment_snapshot=environment,
        )
    )
    if certificate != replayed_certificate or acceptance != replayed_acceptance:
        raise ValueError("continuation artifacts disagree with exact checker replay")
    return VerifiedContinuationEvidence(
        canonical_index=canonical_index,
        candidate=tuple(certificate["candidate"]),
        exact_completion_status=str(certificate["exact_completion_status"]),
        completion_feasible=certificate["completion_feasible"],
        terminal_total_qubits=certificate["terminal_total_qubits"],
        terminal_maximum_chain_length=certificate["terminal_maximum_chain_length"],
        largest_free_component_node_numerator=certificate["largest_free_component_node_numerator"],
        largest_free_component_node_denominator=certificate[
            "largest_free_component_node_denominator"
        ],
        largest_free_component_edge_connectivity=certificate[
            "largest_free_component_edge_connectivity"
        ],
        largest_free_component_articulation_count=certificate[
            "largest_free_component_articulation_count"
        ],
        minimum_logical_contact_multiplicity=certificate["minimum_logical_contact_multiplicity"],
        assignment_sha256=certificate["assignment_sha256"],
        certificate_sha256=expected_certificate_digest,
        checker_acceptance_sha256=expected_acceptance_digest,
        expanded_nodes=int(certificate["expanded_nodes"]),
        _seal=_VERIFICATION_SEAL,
    )


def _logical_hardness(problem: VerifiedProblem) -> dict[str, object]:
    graph = problem.graph()
    variable_count = len(problem.variables)
    edge_count = len(problem.logical_edges)
    degree_numerator = 2 * edge_count
    return {
        "variables": variable_count,
        "edges": edge_count,
        "average_degree_numerator": degree_numerator,
        "average_degree_denominator": variable_count,
        "density_numerator": degree_numerator,
        "density_denominator": variable_count * (variable_count - 1),
        "maximum_degree": max(dict(graph.degree()).values()),
        "component_count": nx.number_connected_components(graph),
    }


def _chain_hardness(chains: tuple[FrozenChain, ...], host_node_count: int) -> dict[str, object]:
    if not chains:
        raise ValueError("chain hardness requires at least one placed chain")
    lengths = tuple(len(chain.nodes) for chain in chains)
    total = sum(lengths)
    mean = total / len(lengths)
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    return {
        "chain_count": len(lengths),
        "total_qubits": total,
        "host_fill_numerator": total,
        "host_fill_denominator": host_node_count,
        "mean_chain_length_numerator": total,
        "mean_chain_length_denominator": len(lengths),
        "maximum_chain_length": max(lengths),
        "chain_length_stddev": math.sqrt(variance),
    }


def _exact_residual_view(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key
        in {
            "free_node_numerator",
            "free_node_denominator",
            "free_edge_numerator",
            "free_edge_denominator",
            "largest_free_component_node_numerator",
            "largest_free_component_node_denominator",
            "largest_free_component_edge_numerator",
            "largest_free_component_edge_denominator",
            "largest_free_component_edge_connectivity",
            "articulation_count",
            "largest_free_component_articulation_count",
        }
    }


def _immediate_view(facts: CandidateFacts) -> dict[str, object]:
    return {
        "connected": facts.connected,
        "chain_disjoint": facts.chain_disjoint,
        "realizes_every_required_coupler": facts.realizes_every_required_coupler,
        "within_l_cap": facts.within_l_cap,
        "minor_valid": facts.minor_valid,
        "current_total_qubits": facts.current_total_qubits,
        "current_maximum_chain_length": facts.current_maximum_chain_length,
        "within_q_cap": facts.within_q_cap,
        "feasible_now": facts.feasible_now,
    }


def _verify_source_bindings(
    *,
    problem: VerifiedProblem,
    host: VerifiedHost,
    starting_embedding: VerifiedStartingEmbedding,
    witness_embedding: VerifiedStartingEmbedding,
    partial_context: PartialContext,
    expected_problem_sha256: str,
    expected_host_artifact_sha256: str,
    expected_host_sha256: str,
    expected_starting_embedding_sha256: str,
    expected_witness_embedding_sha256: str,
    expected_partial_context_sha256: str,
) -> None:
    bindings = (
        (problem.problem_sha256, expected_problem_sha256, "problem"),
        (host.host_artifact_sha256, expected_host_artifact_sha256, "host artifact"),
        (host.host_sha256, expected_host_sha256, "host graph"),
        (
            starting_embedding.starting_embedding_sha256,
            expected_starting_embedding_sha256,
            "starting embedding",
        ),
        (
            witness_embedding.starting_embedding_sha256,
            expected_witness_embedding_sha256,
            "witness embedding",
        ),
        (
            partial_context.partial_context_sha256,
            expected_partial_context_sha256,
            "partial context",
        ),
    )
    for actual, raw_expected, name in bindings:
        expected = _require_sha256(raw_expected, f"expected {name} digest")
        if actual != expected:
            raise ValueError(f"{name} digest disagrees with the state source binding")
    if (
        starting_embedding.problem_sha256 != problem.problem_sha256
        or starting_embedding.host_sha256 != host.host_sha256
        or witness_embedding.source != "witness"
        or witness_embedding.problem_sha256 != problem.problem_sha256
        or witness_embedding.host_sha256 != host.host_sha256
        or partial_context.problem_sha256 != problem.problem_sha256
        or partial_context.host_sha256 != host.host_sha256
        or partial_context.starting_embedding_sha256 != starting_embedding.starting_embedding_sha256
        or partial_context.witness_embedding_sha256 != witness_embedding.starting_embedding_sha256
    ):
        raise ValueError("verified source objects do not share the same identity chain")


def recompute_hardness(
    *,
    profile_id: str,
    solver_profile_sha256: str,
    problem: VerifiedProblem,
    host: VerifiedHost,
    starting_embedding: VerifiedStartingEmbedding,
    witness_embedding: VerifiedStartingEmbedding,
    partial_context: PartialContext,
    expected_problem_sha256: str,
    expected_host_artifact_sha256: str,
    expected_host_sha256: str,
    expected_starting_embedding_sha256: str,
    expected_witness_embedding_sha256: str,
    expected_partial_context_sha256: str,
    candidate_sample_seed_key: bytes,
    expected_candidate_protocol_result_sha256: str,
    continuation_certificates: tuple[object, ...],
    checker_acceptances: tuple[object, ...],
    expected_continuation_certificate_sha256s: tuple[str, ...],
    expected_checker_acceptance_sha256s: tuple[str, ...],
    expected_assignment_sha256s: tuple[str | None, ...],
    checker_environment: object,
    expected_checker_source_sha256: str,
    expected_checker_environment_sha256: str,
) -> VerifiedHardness:
    """Recompute one complete candidate-indexed partial-structural hardness object."""

    if (
        type(problem) is not VerifiedProblem
        or type(host) is not VerifiedHost
        or type(starting_embedding) is not VerifiedStartingEmbedding
        or type(witness_embedding) is not VerifiedStartingEmbedding
        or type(partial_context) is not PartialContext
    ):
        raise TypeError("hardness sources must be exact verified boundary values")
    _verify_source_bindings(
        problem=problem,
        host=host,
        starting_embedding=starting_embedding,
        witness_embedding=witness_embedding,
        partial_context=partial_context,
        expected_problem_sha256=expected_problem_sha256,
        expected_host_artifact_sha256=expected_host_artifact_sha256,
        expected_host_sha256=expected_host_sha256,
        expected_starting_embedding_sha256=expected_starting_embedding_sha256,
        expected_witness_embedding_sha256=expected_witness_embedding_sha256,
        expected_partial_context_sha256=expected_partial_context_sha256,
    )
    environment_snapshot = _snapshot_json(checker_environment, "checker_environment")
    expected_source_digest = _require_sha256(
        expected_checker_source_sha256, "expected_checker_source_sha256"
    )
    if continuation_checker_source_sha256() != expected_source_digest:
        raise ValueError("checker source digest does not match the executing source")
    expected_environment_digest = _require_sha256(
        expected_checker_environment_sha256, "expected_checker_environment_sha256"
    )
    if canonical_sha256(environment_snapshot) != expected_environment_digest:
        raise ValueError("checker environment digest does not match the supplied environment")

    candidate_result = generate_candidate_bank(
        host.graph(),
        window_nodes=partial_context.window_nodes,
        frozen_chains=partial_context.frozen_chains,
        required_logical_neighbors=partial_context.required_focus_neighbors,
        original_focus_chain=partial_context.original_focus_chain,
        l_cap=partial_context.l_cap,
        q_cap=partial_context.q_cap,
        candidate_sample_seed_key=candidate_sample_seed_key,
    )
    if candidate_result.attempt_status not in {"single_candidate", "candidate_bank_ready"}:
        raise ValueError(
            f"hardness requires a complete candidate bank, got {candidate_result.attempt_status!r}"
        )
    result_digest = candidate_protocol_result_sha256(candidate_result)
    expected_result_digest = _require_sha256(
        expected_candidate_protocol_result_sha256,
        "expected_candidate_protocol_result_sha256",
    )
    if result_digest != expected_result_digest:
        raise ValueError("candidate protocol result digest disagrees with exact replay")
    if (
        candidate_result.full_candidates is None
        or candidate_result.full_candidate_facts is None
        or candidate_result.offered_to_full_indices is None
        or candidate_result.offered_candidate_count is None
        or candidate_result.full_candidate_count is None
    ):
        raise RuntimeError("successful candidate protocol result has incomplete fields")

    aligned_values = (
        continuation_certificates,
        checker_acceptances,
        expected_continuation_certificate_sha256s,
        expected_checker_acceptance_sha256s,
        expected_assignment_sha256s,
    )
    if any(type(values) is not tuple for values in aligned_values):
        raise TypeError("continuation documents and expected digests must be immutable tuples")
    offered_count = candidate_result.offered_candidate_count
    if any(len(values) != offered_count for values in aligned_values):
        raise ValueError("continuation evidence must exactly match every offered candidate")

    verified_continuations: list[VerifiedContinuationEvidence] = []
    candidate_documents: list[dict[str, object]] = []
    for position, canonical_index in enumerate(candidate_result.offered_to_full_indices):
        facts = candidate_result.full_candidate_facts[canonical_index]
        evidence = verify_continuation_evidence(
            continuation_certificates[position],
            checker_acceptances[position],
            problem=problem,
            host=host,
            partial_context=partial_context,
            candidate_protocol_result=candidate_result,
            candidate_sample_seed_key=candidate_sample_seed_key,
            expected_candidate_protocol_result_sha256=expected_result_digest,
            canonical_index=canonical_index,
            checker_environment=environment_snapshot,
            expected_certificate_sha256=expected_continuation_certificate_sha256s[position],
            expected_checker_acceptance_sha256=expected_checker_acceptance_sha256s[position],
            expected_checker_source_sha256=expected_source_digest,
            expected_checker_environment_sha256=expected_environment_digest,
            expected_assignment_sha256=expected_assignment_sha256s[position],
        )
        verified_continuations.append(evidence)
        current_chains = (
            *partial_context.frozen_chains,
            FrozenChain(partial_context.focus, facts.candidate),
        )
        current_chains = tuple(sorted(current_chains, key=lambda chain: chain.logical_variable))
        post_embedding = _chain_hardness(current_chains, len(host.nodes))
        if (
            post_embedding["total_qubits"] != facts.current_total_qubits
            or post_embedding["maximum_chain_length"] != facts.current_maximum_chain_length
        ):
            raise RuntimeError("candidate protocol resource facts disagree with recomputation")
        occupied = tuple(sorted(node for chain in current_chains for node in chain.nodes))
        candidate_documents.append(
            {
                "canonical_index": canonical_index,
                "candidate": list(facts.candidate),
                "immediate": _immediate_view(facts),
                "post_replacement_embedding": post_embedding,
                "residual": _exact_residual_view(residual_hardness(host.graph(), occupied)),
                "deferred_logical_edges": [
                    list(edge) for edge in partial_context.deferred_logical_edges
                ],
                "continuation": evidence.summary(),
            }
        )

    completion_exhausted = any(
        evidence.exact_completion_status == "node_budget_exhausted"
        for evidence in verified_continuations
    )
    completion_count = sum(
        evidence.completion_feasible is True for evidence in verified_continuations
    )
    logical_graph = problem.graph()
    host_graph = host.graph()
    document: dict[str, object] = {
        "schema": HARDNESS_SCHEMA,
        "schema_version": HARDNESS_SCHEMA_VERSION,
        "profile_id": _require_registered_profile_name(profile_id),
        "sources": {
            "problem_sha256": problem.problem_sha256,
            "host_artifact_sha256": host.host_artifact_sha256,
            "host_sha256": host.host_sha256,
            "starting_embedding_sha256": starting_embedding.starting_embedding_sha256,
            "witness_embedding_sha256": witness_embedding.starting_embedding_sha256,
            "partial_context_sha256": partial_context.partial_context_sha256,
            "candidate_protocol_result_sha256": result_digest,
        },
        "host": {
            "nodes": len(host.nodes),
            "edges": len(host.edges),
            "component_count": nx.number_connected_components(host_graph),
            "defect_qubit_numerator": len(host.removed_nodes),
            "defect_qubit_denominator": host.pristine_node_count,
            "defect_coupler_numerator": len(host.removed_edges),
            "defect_coupler_denominator": host.post_qubit_edge_count,
        },
        "logical": _logical_hardness(problem),
        "starting_embedding": _chain_hardness(starting_embedding.chains, len(host.nodes)),
        "partial_context": {
            "placed_variables": len(partial_context.placed_variables),
            "unplaced_variables": len(partial_context.unplaced_variables),
            "focus": partial_context.focus,
            "occupied_qubits": len(partial_context.frozen_qubits),
            "q_cap": partial_context.q_cap,
            "terminal_q_cap": partial_context.terminal_q_cap,
        },
        "reference_candidate_index": candidate_result.full_candidates.index(
            partial_context.original_focus_chain
        ),
        "candidates": candidate_documents,
        "decision": {
            "focus_degree": logical_graph.degree[partial_context.focus],
            "window_nodes": len(partial_context.window_nodes),
            "window_edges": len(partial_context.window_edges),
            "full_candidate_count": candidate_result.full_candidate_count,
            "offered_candidate_count": offered_count,
            "exact_completion_numerator": None if completion_exhausted else completion_count,
            "exact_completion_denominator": offered_count,
            "exact_completion_fraction": (
                None if completion_exhausted else completion_count / offered_count
            ),
        },
        "solver_profile_sha256": _require_sha256(solver_profile_sha256, "solver_profile_sha256"),
    }
    _validate_hardness_document_structure(document)
    snapshot = copy.deepcopy(document)
    serialized = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    verified = object.__new__(VerifiedHardness)
    object.__setattr__(verified, "_serialized", serialized)
    object.__setattr__(verified, "hardness_sha256", canonical_sha256(snapshot))
    object.__setattr__(verified, "_seal", _VERIFICATION_SEAL)
    return verified


def validate_hardness(value: object) -> VerifiedHardness:
    """Validate the structure and integrity of an already recomputed hardness value."""

    if type(value) is not VerifiedHardness or value._seal is not _VERIFICATION_SEAL:
        raise TypeError("hardness must be an exact VerifiedHardness value")
    document = value.to_dict()
    _validate_hardness_document_structure(document)
    if canonical_sha256(document) != value.hardness_sha256:
        raise ValueError("verified hardness content digest is invalid")
    return value


def verify_hardness(
    stored_document: object,
    *,
    expected_hardness_sha256: str,
    recompute_arguments: object,
) -> VerifiedHardness:
    """Bind stored hardness bytes to a recomputation performed inside this call."""

    snapshot = _snapshot_json(stored_document, "stored hardness")
    expected = _require_sha256(expected_hardness_sha256, "expected_hardness_sha256")
    if canonical_sha256(snapshot) != expected:
        raise ValueError("stored hardness digest does not match its external binding")
    if type(recompute_arguments) is not dict or not all(
        type(name) is str for name in recompute_arguments
    ):
        raise TypeError("recompute_arguments must be an exact keyword dictionary")
    checked = validate_hardness(recompute_hardness(**dict(recompute_arguments)))
    if snapshot != checked.to_dict() or expected != checked.hardness_sha256:
        raise ValueError("stored hardness does not equal authenticated recomputed hardness")
    return checked


def hardness_model_inputs(value: object) -> dict[str, object]:
    """Return the positive-whitelist deployment view without future or identity fields."""

    document = validate_hardness(value).to_dict()
    return {
        "host": document["host"],
        "logical": document["logical"],
        "partial_context": document["partial_context"],
        "candidates": [
            {
                "canonical_index": candidate["canonical_index"],
                "candidate": candidate["candidate"],
                "immediate": candidate["immediate"],
                "post_replacement_embedding": candidate["post_replacement_embedding"],
                "residual": candidate["residual"],
                "deferred_logical_edges": candidate["deferred_logical_edges"],
            }
            for candidate in document["candidates"]
        ],
        "decision": {
            key: field_value
            for key, field_value in document["decision"].items()
            if not key.startswith("exact_completion")
        },
    }


__all__ = [
    "CONTINUATION_CERTIFICATE_SCHEMA",
    "CONTINUATION_CERTIFICATE_SCHEMA_VERSION",
    "CONTINUATION_CHECKER_ACCEPTANCE_SCHEMA",
    "CONTINUATION_CHECKER_ACCEPTANCE_SCHEMA_VERSION",
    "EXACT_COMPLETION_NODE_BUDGET",
    "HARDNESS_SCHEMA",
    "HARDNESS_SCHEMA_VERSION",
    "PARTIAL_CONTEXT_SCHEMA",
    "PARTIAL_CONTEXT_SCHEMA_VERSION",
    "STARTING_EMBEDDING_SCHEMA",
    "STARTING_EMBEDDING_SCHEMA_VERSION",
    "PartialContext",
    "VerifiedContinuationEvidence",
    "VerifiedHardness",
    "VerifiedHost",
    "VerifiedProblem",
    "VerifiedStartingEmbedding",
    "build_partial_context_artifact",
    "build_starting_embedding_artifact",
    "build_continuation_artifacts",
    "continuation_checker_source_sha256",
    "hardness_model_inputs",
    "recompute_hardness",
    "residual_hardness",
    "verify_host",
    "verify_continuation_evidence",
    "verify_hardness",
    "verify_partial_context",
    "verify_problem",
    "verify_starting_embedding",
    "validate_hardness",
]

"""Immutable realized-host artifacts and defect-aware embedding validation.

The hard/OOD protocol treats a defective host as data, not as a topology name that a
consumer may reconstruct.  This module materializes the exact fault mask from a registered
32-byte seed key, gives the pristine graph, realized graph, and provenance separate content
identities, and verifies every content-level claim before returning a graph to a caller.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Collection, Mapping, Sequence

import networkx as nx

from embedbench.hard_ood_schema import canonical_sha256
from embedbench.structural import host_graph

UINT64_MAX = 2**64 - 1
UINT256_MAX = 2**256 - 1

REALIZED_HOST_SCHEMA = "embedbench.realized-host"
REALIZED_HOST_SCHEMA_VERSION = 1
HOST_GRAPH_SCHEMA = "embedbench.host-graph"
PRISTINE_HOST_SCHEMA = "embedbench.pristine-host"
GRAPH_SCHEMA_VERSION = 1

_TOPOLOGIES = frozenset({"chimera", "pegasus", "zephyr"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_REALIZED_HOST_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "topology",
        "size",
        "pristine_host_sha256",
        "requested_defects",
        "removed_nodes",
        "removed_edges",
        "nodes",
        "edges",
        "realized_statistics",
        "host_sha256",
        "host_artifact_sha256",
    }
)
_REQUESTED_DEFECT_FIELDS = frozenset({"qubit_fraction", "coupler_fraction", "seed"})
_REALIZED_STATISTIC_FIELDS = frozenset(
    {
        "qubit_fraction",
        "coupler_fraction",
        "component_count",
        "largest_component_fraction",
    }
)
_SELF_DIGEST_FIELDS = frozenset({"host_sha256", "host_artifact_sha256"})


def _require_exact_keys(
    value: object,
    expected: Collection[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    expected_keys = set(expected)
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        raise ValueError(f"{name} schema fields differ: missing={missing}, unknown={unknown}")
    return value


def _require_seed_key(seed_key: object) -> bytes:
    if type(seed_key) is not bytes:
        raise TypeError("seed_key must be raw bytes")
    if len(seed_key) != 32:
        raise ValueError("seed_key must contain exactly 32 bytes")
    return seed_key


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_topology(value: object) -> str:
    if type(value) is not str or value not in _TOPOLOGIES:
        raise ValueError(f"topology must be one of {sorted(_TOPOLOGIES)}")
    return value


def _require_size(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("size must be a positive integer")
    return value


def _require_fraction(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite binary64 value in [0, 1]")
    return 0.0 if value == 0.0 else value


def _require_uint64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _require_logical_node(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _canonical_nodes(value: object, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    nodes = tuple(_require_uint64(node, f"{name}[{index}]") for index, node in enumerate(value))
    if nonempty and not nodes:
        raise ValueError(f"{name} must not be empty")
    if list(nodes) != sorted(nodes) or len(nodes) != len(set(nodes)):
        raise ValueError(f"{name} must be duplicate-free and sorted")
    return nodes


def _canonical_edges(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    edges: list[tuple[int, int]] = []
    for index, raw_edge in enumerate(value):
        if type(raw_edge) is not list or len(raw_edge) != 2:
            raise ValueError(f"{name}[{index}] must be a two-node JSON array")
        left = _require_uint64(raw_edge[0], f"{name}[{index}][0]")
        right = _require_uint64(raw_edge[1], f"{name}[{index}][1]")
        if left >= right:
            raise ValueError(f"{name}[{index}] must be sorted and contain two distinct nodes")
        edges.append((left, right))
    if edges != sorted(edges) or len(edges) != len(set(edges)):
        raise ValueError(f"{name} must be duplicate-free and lexicographically sorted")
    return tuple(edges)


def _canonical_graph_arrays(graph: nx.Graph) -> tuple[list[int], list[list[int]]]:
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TypeError("host must be an undirected simple NetworkX graph")
    nodes = sorted(_require_uint64(node, "host node") for node in graph.nodes)
    if len(nodes) != graph.number_of_nodes():
        raise ValueError("host contains duplicate normalized node identifiers")
    edges: list[list[int]] = []
    for raw_left, raw_right in graph.edges:
        left = _require_uint64(raw_left, "host edge endpoint")
        right = _require_uint64(raw_right, "host edge endpoint")
        if left == right:
            raise ValueError("host must not contain self-loops")
        edges.append([min(left, right), max(left, right)])
    edges.sort()
    if len(edges) != len({tuple(edge) for edge in edges}):
        raise ValueError("host must not contain parallel edges")
    return nodes, edges


def _graph_identity(graph: nx.Graph, *, schema: str) -> dict[str, object]:
    nodes, edges = _canonical_graph_arrays(graph)
    return {
        "schema": schema,
        "schema_version": GRAPH_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
    }


def host_graph_sha256(graph: nx.Graph) -> str:
    """Return the identity digest of only the realized physical graph."""

    return canonical_sha256(_graph_identity(graph, schema=HOST_GRAPH_SCHEMA))


def pristine_host_sha256(topology: str, size: int) -> str:
    """Regenerate and hash the pinned D-Wave NetworkX pristine topology."""

    checked_topology = _require_topology(topology)
    checked_size = _require_size(size)
    pristine = host_graph(checked_topology, checked_size)
    return canonical_sha256(_graph_identity(pristine, schema=PRISTINE_HOST_SCHEMA))


def _ranked_removed_nodes(
    nodes: Sequence[int],
    *,
    count: int,
    seed_key: bytes,
) -> list[int]:
    ranked = sorted(
        nodes,
        key=lambda node: (
            hashlib.sha256(seed_key + b":node:" + node.to_bytes(8, "big")).digest(),
            node,
        ),
    )
    return sorted(ranked[:count])


def _ranked_removed_edges(
    edges: Sequence[tuple[int, int]],
    *,
    count: int,
    seed_key: bytes,
) -> list[tuple[int, int]]:
    ranked = sorted(
        edges,
        key=lambda edge: (
            hashlib.sha256(
                seed_key + b":edge:" + edge[0].to_bytes(8, "big") + edge[1].to_bytes(8, "big")
            ).digest(),
            edge,
        ),
    )
    return sorted(ranked[:count])


def _component_statistics(graph: nx.Graph) -> tuple[int, float]:
    if graph.number_of_nodes() == 0:
        return 0, 0.0
    components = list(nx.connected_components(graph))
    return len(components), max(map(len, components)) / graph.number_of_nodes()


def _materialize_payload(
    *,
    topology: str,
    size: int,
    qubit_fraction: float,
    coupler_fraction: float,
    seed_key: bytes,
) -> tuple[dict[str, object], nx.Graph]:
    pristine = host_graph(topology, size)
    pristine_nodes, pristine_edges_json = _canonical_graph_arrays(pristine)
    if not pristine_nodes:
        raise ValueError("pristine host must contain at least one node")
    pristine_edges = [tuple(edge) for edge in pristine_edges_json]

    removed_node_count = math.floor(qubit_fraction * len(pristine_nodes))
    if qubit_fraction > 0.0 and removed_node_count == 0:
        raise ValueError("nonzero qubit defect request rounds to zero removed qubits")
    removed_nodes = _ranked_removed_nodes(
        pristine_nodes,
        count=removed_node_count,
        seed_key=seed_key,
    )
    removed_node_set = set(removed_nodes)
    surviving_nodes = [node for node in pristine_nodes if node not in removed_node_set]
    surviving_edges = [
        edge
        for edge in pristine_edges
        if edge[0] not in removed_node_set and edge[1] not in removed_node_set
    ]

    removed_edge_count = math.floor(coupler_fraction * len(surviving_edges))
    if coupler_fraction > 0.0 and removed_edge_count == 0:
        raise ValueError("nonzero coupler defect request rounds to zero removed couplers")
    removed_edges = _ranked_removed_edges(
        surviving_edges,
        count=removed_edge_count,
        seed_key=seed_key,
    )
    removed_edge_set = set(removed_edges)
    realized_edges = [edge for edge in surviving_edges if edge not in removed_edge_set]

    realized = nx.Graph()
    realized.add_nodes_from(surviving_nodes)
    realized.add_edges_from(realized_edges)
    component_count, largest_component_fraction = _component_statistics(realized)
    payload: dict[str, object] = {
        "schema": REALIZED_HOST_SCHEMA,
        "schema_version": REALIZED_HOST_SCHEMA_VERSION,
        "topology": topology,
        "size": size,
        "pristine_host_sha256": canonical_sha256(
            _graph_identity(pristine, schema=PRISTINE_HOST_SCHEMA)
        ),
        "requested_defects": {
            "qubit_fraction": qubit_fraction,
            "coupler_fraction": coupler_fraction,
            # This is the reversible 256-bit registry seed, not seed32 or an ordinal.
            # Fixed-width recovery preserves the raw seed_key, including leading zeroes.
            "seed": int.from_bytes(seed_key, "big"),
        },
        "removed_nodes": removed_nodes,
        "removed_edges": [list(edge) for edge in removed_edges],
        "nodes": surviving_nodes,
        "edges": [list(edge) for edge in realized_edges],
        "realized_statistics": {
            "qubit_fraction": removed_node_count / len(pristine_nodes),
            "coupler_fraction": (
                removed_edge_count / len(surviving_edges) if surviving_edges else 0.0
            ),
            "component_count": component_count,
            "largest_component_fraction": largest_component_fraction,
        },
    }
    return payload, realized


def build_realized_host_artifact(
    *,
    topology: str,
    size: int,
    qubit_fraction: float,
    coupler_fraction: float,
    seed_key: bytes,
) -> dict[str, object]:
    """Build the exact Section 6 realized-host v1 artifact.

    ``seed_key`` is the raw registered 32-byte host seed.  The JSON ``seed`` field stores its
    unsigned fixed-width registry-seed representation, not a 32-bit adapter seed or plan
    ordinal.  It can be recovered exactly with ``seed.to_bytes(32, "big")``.
    """

    checked_topology = _require_topology(topology)
    checked_size = _require_size(size)
    checked_qubit_fraction = _require_fraction(qubit_fraction, "qubit_fraction")
    checked_coupler_fraction = _require_fraction(coupler_fraction, "coupler_fraction")
    checked_seed_key = _require_seed_key(seed_key)

    payload, realized = _materialize_payload(
        topology=checked_topology,
        size=checked_size,
        qubit_fraction=checked_qubit_fraction,
        coupler_fraction=checked_coupler_fraction,
        seed_key=checked_seed_key,
    )
    artifact = dict(payload)
    artifact["host_sha256"] = host_graph_sha256(realized)
    artifact["host_artifact_sha256"] = canonical_sha256(payload)
    return artifact


def _validate_requested_defects(value: object, *, seed_key: bytes) -> tuple[float, float]:
    document = _require_exact_keys(value, _REQUESTED_DEFECT_FIELDS, "requested_defects")
    qubit_fraction = _require_fraction(document["qubit_fraction"], "qubit_fraction")
    coupler_fraction = _require_fraction(document["coupler_fraction"], "coupler_fraction")
    seed = document["seed"]
    if type(seed) is not int or not 0 <= seed <= UINT256_MAX:
        raise ValueError("requested_defects.seed must be an unsigned 256-bit integer")
    if seed.to_bytes(32, "big") != seed_key:
        raise ValueError("requested_defects.seed does not match the registered seed_key")
    return qubit_fraction, coupler_fraction


def _validate_realized_statistics(value: object) -> None:
    document = _require_exact_keys(value, _REALIZED_STATISTIC_FIELDS, "realized_statistics")
    _require_fraction(document["qubit_fraction"], "realized qubit_fraction")
    _require_fraction(document["coupler_fraction"], "realized coupler_fraction")
    component_count = document["component_count"]
    if type(component_count) is not int or component_count < 0:
        raise ValueError("component_count must be a non-negative integer")
    _require_fraction(document["largest_component_fraction"], "largest_component_fraction")


def _artifact_payload(document: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in document.items() if key not in _SELF_DIGEST_FIELDS}


def validate_realized_host_artifact(
    artifact: object,
    *,
    seed_key: bytes,
    expected_host_artifact_sha256: str | None,
    expected_host_sha256: str | None,
) -> nx.Graph:
    """Fully verify an artifact against external content identities and regenerate its mask.

    Expected digests are deliberately mandatory.  Reading the expected value out of an
    untrusted artifact would only prove internal consistency, not record-to-artifact binding.
    """

    checked_seed_key = _require_seed_key(seed_key)
    if expected_host_artifact_sha256 is None:
        raise ValueError("expected_host_artifact_sha256 is required from the referencing record")
    if expected_host_sha256 is None:
        raise ValueError("expected_host_sha256 is required from the referencing record")
    expected_artifact_digest = _require_sha256(
        expected_host_artifact_sha256,
        "expected_host_artifact_sha256",
    )
    expected_graph_digest = _require_sha256(expected_host_sha256, "expected_host_sha256")

    document = _require_exact_keys(artifact, _REALIZED_HOST_FIELDS, "realized host artifact")
    if type(document["schema"]) is not str or document["schema"] != REALIZED_HOST_SCHEMA:
        raise ValueError(f"realized host artifact requires schema {REALIZED_HOST_SCHEMA!r}")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != REALIZED_HOST_SCHEMA_VERSION
    ):
        raise ValueError(
            f"realized host artifact requires schema_version {REALIZED_HOST_SCHEMA_VERSION}"
        )
    topology = _require_topology(document["topology"])
    size = _require_size(document["size"])
    pristine_digest = _require_sha256(document["pristine_host_sha256"], "pristine_host_sha256")
    qubit_fraction, coupler_fraction = _validate_requested_defects(
        document["requested_defects"],
        seed_key=checked_seed_key,
    )
    removed_nodes = _canonical_nodes(document["removed_nodes"], "removed_nodes")
    removed_edges = _canonical_edges(document["removed_edges"], "removed_edges")
    realized_nodes = _canonical_nodes(document["nodes"], "nodes")
    realized_edges = _canonical_edges(document["edges"], "edges")
    _validate_realized_statistics(document["realized_statistics"])
    stored_graph_digest = _require_sha256(document["host_sha256"], "host_sha256")
    stored_artifact_digest = _require_sha256(
        document["host_artifact_sha256"],
        "host_artifact_sha256",
    )

    computed_artifact_digest = canonical_sha256(_artifact_payload(document))
    if stored_artifact_digest != computed_artifact_digest:
        raise ValueError("host_artifact_sha256 does not match the artifact payload")
    if stored_artifact_digest != expected_artifact_digest:
        raise ValueError("external host artifact digest does not match host_artifact_sha256")

    supplied_graph = nx.Graph()
    supplied_graph.add_nodes_from(realized_nodes)
    supplied_graph.add_edges_from(realized_edges)
    computed_graph_digest = host_graph_sha256(supplied_graph)
    if stored_graph_digest != computed_graph_digest:
        raise ValueError("host_sha256 does not match the realized graph")
    if stored_graph_digest != expected_graph_digest:
        raise ValueError("external host graph digest does not match host_sha256")

    expected_artifact = build_realized_host_artifact(
        topology=topology,
        size=size,
        qubit_fraction=qubit_fraction,
        coupler_fraction=coupler_fraction,
        seed_key=checked_seed_key,
    )
    if pristine_digest != expected_artifact["pristine_host_sha256"]:
        raise ValueError("pristine_host_sha256 does not match the regenerated pristine host")
    if list(removed_nodes) != expected_artifact["removed_nodes"]:
        raise ValueError("removed_nodes do not match the deterministic removed_nodes mask")
    if [list(edge) for edge in removed_edges] != expected_artifact["removed_edges"]:
        raise ValueError("removed_edges do not match the deterministic removed_edges mask")
    if list(realized_nodes) != expected_artifact["nodes"]:
        raise ValueError("nodes do not match the deterministic realized graph")
    if [list(edge) for edge in realized_edges] != expected_artifact["edges"]:
        raise ValueError("edges do not match the deterministic realized graph")
    if document["realized_statistics"] != expected_artifact["realized_statistics"]:
        raise ValueError("realized_statistics do not match the regenerated fault mask")
    if stored_graph_digest != expected_artifact["host_sha256"]:
        raise ValueError("host_sha256 does not match the regenerated fault mask")
    if stored_artifact_digest != expected_artifact["host_artifact_sha256"]:
        raise ValueError("host_artifact_sha256 does not match the regenerated artifact")
    return supplied_graph


def reconstruct_pristine_host_for_generation(*, topology: str, size: int) -> nx.Graph:
    """Construct a pristine host before a content-addressed artifact exists.

    This helper is only for generation.  Record consumers must use
    :func:`load_realized_host`, which requires the published artifact and both externally
    supplied content identities even when the requested defect fractions are zero.
    """

    checked_topology = _require_topology(topology)
    checked_size = _require_size(size)
    pristine = host_graph(checked_topology, checked_size)
    nodes, _ = _canonical_graph_arrays(pristine)
    if not nodes:
        raise ValueError("pristine host must contain at least one node")
    return pristine


def load_realized_host(
    *,
    topology: str,
    size: int,
    qubit_fraction: float,
    coupler_fraction: float,
    seed_key: bytes,
    artifact: object,
    expected_host_artifact_sha256: str,
    expected_host_sha256: str,
) -> nx.Graph:
    """Load one record-bound host after verifying its artifact and external identities."""

    checked_topology = _require_topology(topology)
    checked_size = _require_size(size)
    checked_qubit_fraction = _require_fraction(qubit_fraction, "qubit_fraction")
    checked_coupler_fraction = _require_fraction(coupler_fraction, "coupler_fraction")
    checked_seed_key = _require_seed_key(seed_key)
    if artifact is None:
        raise ValueError("record-facing loader requires the exact realized-host artifact")

    graph = validate_realized_host_artifact(
        artifact,
        seed_key=checked_seed_key,
        expected_host_artifact_sha256=expected_host_artifact_sha256,
        expected_host_sha256=expected_host_sha256,
    )
    document = artifact
    if document["topology"] != checked_topology or document["size"] != checked_size:
        raise ValueError("realized-host artifact topology or size differs from the request")
    defects = document["requested_defects"]
    if (
        defects["qubit_fraction"] != checked_qubit_fraction
        or defects["coupler_fraction"] != checked_coupler_fraction
    ):
        raise ValueError("realized-host artifact defect fractions differ from the request")
    return graph


def validate_host_nodes(graph: nx.Graph, nodes: object, *, name: str) -> tuple[int, ...]:
    """Validate one canonical node array against a verified realized graph."""

    checked = _canonical_nodes(nodes, name)
    missing = sorted(set(checked) - set(graph.nodes))
    if missing:
        raise ValueError(f"{name} contains nodes absent from the realized host: {missing}")
    return checked


def host_fill_fraction(graph: nx.Graph, occupied_nodes: object) -> float:
    """Compute occupied qubits divided by realized-host nodes after validating occupancy."""

    checked = validate_host_nodes(graph, occupied_nodes, name="occupied_nodes")
    return len(checked) / graph.number_of_nodes() if graph.number_of_nodes() else 0.0


def validate_host_edges(
    graph: nx.Graph,
    edges: object,
    *,
    name: str,
    allowed_nodes: object | None = None,
) -> tuple[tuple[int, int], ...]:
    """Validate one canonical edge array against a verified realized graph."""

    checked = _canonical_edges(edges, name)
    allowed = (
        None if allowed_nodes is None else set(_canonical_nodes(allowed_nodes, "allowed_nodes"))
    )
    for edge in checked:
        if not graph.has_edge(*edge):
            raise ValueError(f"{name} contains edge {edge} absent from the realized host")
        if allowed is not None and not set(edge) <= allowed:
            raise ValueError(f"{name} contains edge {edge} outside the allowed node set")
    return checked


def validate_window(graph: nx.Graph, nodes: object, edges: object) -> None:
    """Validate a complete realized-host induced window."""

    checked_nodes = validate_host_nodes(graph, nodes, name="window_nodes")
    checked_edges = validate_host_edges(
        graph,
        edges,
        name="window_edges",
        allowed_nodes=list(checked_nodes),
    )
    expected_edges = {
        tuple(sorted((_require_uint64(left, "window edge"), _require_uint64(right, "window edge"))))
        for left, right in graph.subgraph(checked_nodes).edges
    }
    if set(checked_edges) != expected_edges:
        raise ValueError("window_edges must equal all induced realized-host edges")


def validate_chain(
    graph: nx.Graph,
    nodes: object,
    edges: object | None = None,
    *,
    name: str,
) -> tuple[int, ...]:
    """Validate a nonempty connected chain and its optional programmed edge array."""

    checked_nodes = _canonical_nodes(nodes, name, nonempty=True)
    validate_host_nodes(graph, list(checked_nodes), name=name)
    if edges is None:
        chain_graph = graph.subgraph(checked_nodes)
    else:
        checked_edges = validate_host_edges(
            graph,
            edges,
            name=f"{name}_edges",
            allowed_nodes=list(checked_nodes),
        )
        chain_graph = nx.Graph()
        chain_graph.add_nodes_from(checked_nodes)
        chain_graph.add_edges_from(checked_edges)
    if not nx.is_connected(chain_graph):
        raise ValueError(f"{name} must be connected in the realized host")
    return checked_nodes


def validate_candidate_bank(graph: nx.Graph, candidates: object) -> tuple[tuple[int, ...], ...]:
    """Validate canonical, duplicate-free connected candidate chains."""

    if type(candidates) is not list:
        raise TypeError("candidates must be a JSON array")
    checked = tuple(
        validate_chain(graph, candidate, name=f"candidates[{index}]")
        for index, candidate in enumerate(candidates)
    )
    if len(checked) != len(set(checked)):
        raise ValueError("candidate bank must be duplicate-free")
    if list(checked) != sorted(checked, key=lambda candidate: (len(candidate), candidate)):
        raise ValueError("candidate bank must be sorted by (chain length, node array)")
    return checked


def validate_contact_edges(
    graph: nx.Graph,
    edges: object,
    *,
    left_chain: object,
    right_chain: object,
    name: str,
) -> tuple[tuple[int, int], ...]:
    """Validate physical contact couplers joining two declared chains."""

    left_nodes = set(_canonical_nodes(left_chain, "left_chain", nonempty=True))
    right_nodes = set(_canonical_nodes(right_chain, "right_chain", nonempty=True))
    checked = validate_host_edges(graph, edges, name=name)
    for left, right in checked:
        joins = (left in left_nodes and right in right_nodes) or (
            right in left_nodes and left in right_nodes
        )
        if not joins:
            raise ValueError(f"{name} edge {(left, right)} does not join the declared chains")
    return checked


def _logical_edges(value: object) -> tuple[tuple[int, int], ...]:
    if type(value) is not list:
        raise TypeError("logical_edges must be a JSON array")
    edges: list[tuple[int, int]] = []
    for index, raw_edge in enumerate(value):
        if type(raw_edge) is not list or len(raw_edge) != 2:
            raise ValueError(f"logical_edges[{index}] must be a two-node JSON array")
        left = _require_logical_node(raw_edge[0], f"logical_edges[{index}][0]")
        right = _require_logical_node(raw_edge[1], f"logical_edges[{index}][1]")
        if left >= right:
            raise ValueError("logical edges must be sorted pairs of distinct variables")
        edges.append((left, right))
    if edges != sorted(edges) or len(edges) != len(set(edges)):
        raise ValueError("logical_edges must be duplicate-free and lexicographically sorted")
    return tuple(edges)


def _chain_mapping(chains: object) -> dict[int, list[int]]:
    if not isinstance(chains, Mapping):
        raise TypeError("chains must be a mapping from logical nodes to canonical node arrays")
    checked: dict[int, list[int]] = {}
    for raw_logical_node, raw_chain in chains.items():
        logical_node = _require_logical_node(raw_logical_node, "chain logical node")
        if logical_node in checked:
            raise ValueError("chains contain a duplicate logical node")
        checked[logical_node] = list(
            _canonical_nodes(raw_chain, f"chain[{logical_node}]", nonempty=True)
        )
    return checked


def _edge_mapping(value: object, *, name: str) -> dict[int, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    output: dict[int, object] = {}
    for raw_key, raw_edges in value.items():
        key = _require_logical_node(raw_key, f"{name} key")
        if key in output:
            raise ValueError(f"{name} contains a duplicate key")
        output[key] = raw_edges
    return output


def _contact_mapping(value: object) -> dict[tuple[int, int], object]:
    if not isinstance(value, Mapping):
        raise TypeError("contact_edges must be a mapping")
    output: dict[tuple[int, int], object] = {}
    for raw_key, raw_edges in value.items():
        if type(raw_key) is not tuple or len(raw_key) != 2:
            raise ValueError("contact_edges keys must be sorted logical-edge tuples")
        left = _require_logical_node(raw_key[0], "contact edge logical endpoint")
        right = _require_logical_node(raw_key[1], "contact edge logical endpoint")
        if left >= right:
            raise ValueError("contact_edges keys must be sorted logical-edge tuples")
        key = (left, right)
        if key in output:
            raise ValueError("contact_edges contains a duplicate logical edge")
        output[key] = raw_edges
    return output


def validate_minor_embedding(
    graph: nx.Graph,
    *,
    logical_edges: object,
    chains: object,
    chain_edges: object | None = None,
    contact_edges: object | None = None,
) -> None:
    """Prove full minor-embedding feasibility on the verified realized host.

    Every chain is nonempty and connected, chains are pairwise disjoint, and every declared
    logical edge has at least one physical contact.  Optional recorded chain and contact edges
    are validated in addition to the independently recomputed feasibility predicate.
    """

    checked_logical_edges = _logical_edges(logical_edges)
    checked_chains = _chain_mapping(chains)
    programmed_edges = (
        None if chain_edges is None else _edge_mapping(chain_edges, name="chain_edges")
    )
    if programmed_edges is not None and set(programmed_edges) != set(checked_chains):
        raise ValueError("chain_edges keys must exactly match chains keys")

    owner: dict[int, int] = {}
    for logical_node in sorted(checked_chains):
        chain = checked_chains[logical_node]
        edges = None if programmed_edges is None else programmed_edges[logical_node]
        validate_chain(graph, chain, edges, name=f"chain[{logical_node}]")
        for node in chain:
            if node in owner:
                raise ValueError(
                    "chains must be pairwise disjoint; "
                    f"node {node} belongs to {owner[node]} and {logical_node}"
                )
            owner[node] = logical_node

    recorded_contacts = None if contact_edges is None else _contact_mapping(contact_edges)
    if recorded_contacts is not None and set(recorded_contacts) != set(checked_logical_edges):
        raise ValueError("contact_edges keys must exactly match logical_edges")

    for logical_left, logical_right in checked_logical_edges:
        if logical_left not in checked_chains or logical_right not in checked_chains:
            raise ValueError(
                f"logical edge {(logical_left, logical_right)} has an endpoint without a chain"
            )
        left_chain = checked_chains[logical_left]
        right_chain = checked_chains[logical_right]
        realized_contacts = {
            tuple(sorted((left, right)))
            for left in left_chain
            for right in right_chain
            if graph.has_edge(left, right)
        }
        if not realized_contacts:
            raise ValueError(
                f"logical edge {(logical_left, logical_right)} has no realized contact coupler"
            )
        if recorded_contacts is not None:
            declared = validate_contact_edges(
                graph,
                recorded_contacts[(logical_left, logical_right)],
                left_chain=left_chain,
                right_chain=right_chain,
                name=f"contact_edges[{logical_left},{logical_right}]",
            )
            if not declared:
                raise ValueError(
                    f"logical edge {(logical_left, logical_right)} has no recorded contact coupler"
                )


__all__ = [
    "GRAPH_SCHEMA_VERSION",
    "HOST_GRAPH_SCHEMA",
    "PRISTINE_HOST_SCHEMA",
    "REALIZED_HOST_SCHEMA",
    "REALIZED_HOST_SCHEMA_VERSION",
    "build_realized_host_artifact",
    "host_fill_fraction",
    "host_graph_sha256",
    "load_realized_host",
    "pristine_host_sha256",
    "reconstruct_pristine_host_for_generation",
    "validate_candidate_bank",
    "validate_chain",
    "validate_contact_edges",
    "validate_host_edges",
    "validate_host_nodes",
    "validate_minor_embedding",
    "validate_realized_host_artifact",
    "validate_window",
]

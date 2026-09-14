from __future__ import annotations

import copy
import hashlib
import math

import networkx as nx
import pytest
from embedbench.hard_ood_schema import canonical_sha256
from embedbench.realized_host import (
    build_realized_host_artifact,
    host_fill_fraction,
    host_graph_sha256,
    load_realized_host,
    pristine_host_sha256,
    reconstruct_pristine_host_for_generation,
    validate_candidate_bank,
    validate_chain,
    validate_contact_edges,
    validate_host_edges,
    validate_host_nodes,
    validate_minor_embedding,
    validate_realized_host_artifact,
    validate_window,
)
from embedbench.structural import host_graph

_SEED_KEY = bytes(range(32))

_FAULT_MASK_GOLDENS = [
    pytest.param(
        "chimera",
        5,
        0.02,
        0.05,
        [23, 78, 127, 176],
        26,
        "bc4c80bbaeb7df2b28d1644e3245bc61bb263d90c7c6c569d1db440ce6688c77",
        "a61d3e26eed73c9793db5fd8705d08a628043e9b22a13a99d779e4a94b9c56f8",
        "a7867d354bc376196ca7ecd965b35dca121aa8570b8f47ee6166b552b38c9000",
        "ed0e296001f12b9812055a38bfc5d14bf1ff16946d82c3f7aadf8bc8ac46462c",
        id="chimera-5-light",
    ),
    pytest.param(
        "chimera",
        5,
        0.05,
        0.10,
        [10, 23, 49, 78, 82, 127, 134, 170, 176, 179],
        50,
        "994e392a19fc65ef4d4576b367809b192523d29ec8d6c8bc1279d6ce142f71be",
        "a61d3e26eed73c9793db5fd8705d08a628043e9b22a13a99d779e4a94b9c56f8",
        "addf69d1a9b02c08d0adbd9ec95c471e5f3cb86fc02d918db59c3c7a4f2d54a0",
        "bf13152f268b7b1a28b5d725de3b125e62c42f5070aba51ca8f2559bc7e5caf4",
        id="chimera-5-heavy",
    ),
    pytest.param(
        "pegasus",
        3,
        0.02,
        0.05,
        [23, 78],
        34,
        "03f5ea104f2b186011e90a72c00050a9dac77b0614e4adbc7586d081b1523a9a",
        "c3d8884a94f2ed690e9ac23f04c2a0b5e8a34ea9897d852188505405d1f90c7f",
        "9e07ad1dec78a75ac35e04bb0c4f124f031f091090d3b5f6d84fda561a019ee0",
        "e3ba17cb3f75ad68e7fd0ffccac92f1227677e0d9a4d9931e885f783ee0b9018",
        id="pegasus-3-light",
    ),
    pytest.param(
        "pegasus",
        3,
        0.05,
        0.10,
        [10, 23, 78, 82, 127, 134],
        65,
        "a3f5e7e172ff203009059f77315b2eeea6d0f5c72919a4e91c0ad0cbdee1adf2",
        "c3d8884a94f2ed690e9ac23f04c2a0b5e8a34ea9897d852188505405d1f90c7f",
        "94cc2b451a8ede643b189b663470825eb2ea939fe64b0c6bbe74dd59b1b78bf2",
        "d922657dd122fa4e05061bf82304c4abbc6ef700d32d56f7cbf7e5cca5910eaa",
        id="pegasus-3-heavy",
    ),
    pytest.param(
        "zephyr",
        2,
        0.02,
        0.05,
        [23, 78, 127],
        58,
        "d619cb1099180b588ec3247790a90ec11c7dbdc53c47d3b48ac49ac20085f9f7",
        "421882451fee852a38c7f3493baee88342109bad22f89dda1158c43a1d01a481",
        "5cdef7ce5a0308ec164392515c90e237f376ae193c505706bc4a8fe5e46fbed7",
        "c832f7a477c989dce62f6c2418119cd9e0f026ad8572be67cb387c75a25e72b5",
        id="zephyr-2-light",
    ),
    pytest.param(
        "zephyr",
        2,
        0.05,
        0.10,
        [10, 23, 49, 74, 78, 82, 127, 134],
        111,
        "d1349efbd01fdcd85662356b76f0169a99d9e317bc98933a6abd2b1d97954786",
        "421882451fee852a38c7f3493baee88342109bad22f89dda1158c43a1d01a481",
        "615e1e649836464174dbda6f04bb5d1b50cd54c21b16f73fb0ab6ea6f749e541",
        "185b01ee85e80cf561057c1fc842890744fab83131b650bdb293dc7aac4ec270",
        id="zephyr-2-heavy",
    ),
]


def _verify(artifact: dict[str, object], *, seed_key: bytes = _SEED_KEY) -> nx.Graph:
    return validate_realized_host_artifact(
        artifact,
        seed_key=seed_key,
        expected_host_artifact_sha256=artifact["host_artifact_sha256"],
        expected_host_sha256=artifact["host_sha256"],
    )


def _resign(artifact: dict[str, object]) -> None:
    artifact["host_sha256"] = canonical_sha256(
        {
            "schema": "embedbench.host-graph",
            "schema_version": 1,
            "nodes": artifact["nodes"],
            "edges": artifact["edges"],
        }
    )
    artifact["host_artifact_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"host_sha256", "host_artifact_sha256"}
        }
    )


@pytest.mark.parametrize(
    ("topology", "size"),
    [("chimera", 2), ("pegasus", 3), ("zephyr", 2)],
)
def test_pristine_artifact_round_trips_for_all_registered_topologies(
    topology: str,
    size: int,
) -> None:
    artifact = build_realized_host_artifact(
        topology=topology,
        size=size,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=_SEED_KEY,
    )

    pristine = host_graph(topology, size)
    verified = _verify(artifact)
    assert set(verified.nodes) == set(pristine.nodes)
    assert {tuple(sorted(edge)) for edge in verified.edges} == {
        tuple(sorted(edge)) for edge in pristine.edges
    }
    assert artifact["removed_nodes"] == []
    assert artifact["removed_edges"] == []
    assert artifact["requested_defects"] == {
        "qubit_fraction": 0.0,
        "coupler_fraction": 0.0,
        "seed": int.from_bytes(_SEED_KEY, "big"),
    }
    assert artifact["pristine_host_sha256"] == pristine_host_sha256(topology, size)
    assert artifact["host_sha256"] == host_graph_sha256(pristine)
    assert artifact["realized_statistics"] == {
        "qubit_fraction": 0.0,
        "coupler_fraction": 0.0,
        "component_count": 1,
        "largest_component_fraction": 1.0,
    }


@pytest.mark.parametrize(
    (
        "topology",
        "size",
        "qubit_fraction",
        "coupler_fraction",
        "removed_nodes",
        "removed_edge_count",
        "mask_sha256",
        "expected_pristine_sha256",
        "expected_host_sha256",
        "expected_artifact_sha256",
    ),
    _FAULT_MASK_GOLDENS,
)
def test_fault_mask_and_artifact_golden_vectors(
    topology: str,
    size: int,
    qubit_fraction: float,
    coupler_fraction: float,
    removed_nodes: list[int],
    removed_edge_count: int,
    mask_sha256: str,
    expected_pristine_sha256: str,
    expected_host_sha256: str,
    expected_artifact_sha256: str,
) -> None:
    artifact = build_realized_host_artifact(
        topology=topology,
        size=size,
        qubit_fraction=qubit_fraction,
        coupler_fraction=coupler_fraction,
        seed_key=_SEED_KEY,
    )

    assert artifact["removed_nodes"] == removed_nodes
    assert len(artifact["removed_edges"]) == removed_edge_count
    assert (
        canonical_sha256(
            {
                "removed_nodes": artifact["removed_nodes"],
                "removed_edges": artifact["removed_edges"],
            }
        )
        == mask_sha256
    )
    assert artifact["pristine_host_sha256"] == expected_pristine_sha256
    assert artifact["host_sha256"] == expected_host_sha256
    assert artifact["host_artifact_sha256"] == expected_artifact_sha256
    _verify(artifact)


def test_light_fault_mask_uses_raw_seed_key_sha_order_and_floor_counts() -> None:
    pristine = host_graph("chimera", 4)
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=4,
        qubit_fraction=0.02,
        coupler_fraction=0.05,
        seed_key=_SEED_KEY,
    )

    expected_node_count = math.floor(0.02 * pristine.number_of_nodes())
    expected_nodes = sorted(
        pristine.nodes,
        key=lambda node: (
            hashlib.sha256(_SEED_KEY + b":node:" + int(node).to_bytes(8, "big")).digest(),
            int(node),
        ),
    )[:expected_node_count]
    surviving_edges = sorted(
        tuple(sorted((int(left), int(right))))
        for left, right in pristine.edges
        if left not in expected_nodes and right not in expected_nodes
    )
    expected_edge_count = math.floor(0.05 * len(surviving_edges))
    expected_edges = sorted(
        surviving_edges,
        key=lambda edge: (
            hashlib.sha256(
                _SEED_KEY + b":edge:" + edge[0].to_bytes(8, "big") + edge[1].to_bytes(8, "big")
            ).digest(),
            edge,
        ),
    )[:expected_edge_count]

    assert artifact["removed_nodes"] == sorted(expected_nodes)
    assert artifact["removed_edges"] == sorted([list(edge) for edge in expected_edges])
    assert len(artifact["removed_nodes"]) == expected_node_count
    assert len(artifact["removed_edges"]) == expected_edge_count
    graph = _verify(artifact)
    assert not set(expected_nodes) & set(graph.nodes)
    assert not set(expected_edges) & {tuple(sorted(edge)) for edge in graph.edges}
    assert artifact["realized_statistics"]["qubit_fraction"] == (
        expected_node_count / pristine.number_of_nodes()
    )
    assert artifact["realized_statistics"]["coupler_fraction"] == (
        expected_edge_count / len(surviving_edges)
    )


def test_heavy_fault_artifact_records_exact_component_statistics() -> None:
    artifact = build_realized_host_artifact(
        topology="zephyr",
        size=2,
        qubit_fraction=0.35,
        coupler_fraction=0.45,
        seed_key=bytes(reversed(range(32))),
    )

    graph = _verify(artifact, seed_key=bytes(reversed(range(32))))
    components = list(nx.connected_components(graph))
    statistics = artifact["realized_statistics"]
    assert len(components) > 1
    assert statistics["component_count"] == len(components)
    assert statistics["largest_component_fraction"] == (
        max(map(len, components), default=0) / graph.number_of_nodes()
        if graph.number_of_nodes()
        else 0.0
    )


def test_graph_and_provenance_digests_have_separate_identities() -> None:
    first = build_realized_host_artifact(
        topology="chimera",
        size=2,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=b"a" * 32,
    )
    second = build_realized_host_artifact(
        topology="chimera",
        size=2,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=b"b" * 32,
    )

    assert first["host_sha256"] == second["host_sha256"]
    assert first["pristine_host_sha256"] == second["pristine_host_sha256"]
    assert first["host_artifact_sha256"] != second["host_artifact_sha256"]
    assert first["host_artifact_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in first.items()
            if key not in {"host_sha256", "host_artifact_sha256"}
        }
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact["nodes"].append(2**64),
        lambda artifact: artifact["edges"].append([0, 2**64]),
        lambda artifact: artifact["requested_defects"].update(qubit_fraction=0.3),
        lambda artifact: artifact["realized_statistics"].update(component_count=999),
        lambda artifact: artifact.update(topology="pegasus"),
        lambda artifact: artifact.update(pristine_host_sha256="0" * 64),
    ],
)
def test_artifact_tampering_is_rejected_even_when_self_digests_are_recomputed(
    mutation,
) -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=3,
        qubit_fraction=0.1,
        coupler_fraction=0.1,
        seed_key=_SEED_KEY,
    )
    mutation(artifact)
    _resign(artifact)

    with pytest.raises((TypeError, ValueError)):
        _verify(artifact)


def test_regenerated_fault_mask_rejects_a_different_resigned_mask() -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=3,
        qubit_fraction=0.1,
        coupler_fraction=0.1,
        seed_key=_SEED_KEY,
    )
    removed = artifact["removed_nodes"]
    realized = artifact["nodes"]
    restored, newly_removed = removed[0], realized[0]
    artifact["removed_nodes"] = sorted([*removed[1:], newly_removed])
    artifact["nodes"] = sorted([*realized[1:], restored])
    all_nodes = set(artifact["nodes"])
    artifact["edges"] = sorted(edge for edge in artifact["edges"] if set(edge) <= all_nodes)
    _resign(artifact)

    with pytest.raises(ValueError, match="deterministic removed_nodes"):
        _verify(artifact)


@pytest.mark.parametrize(
    "field",
    ["nodes", "removed_nodes", "edges", "removed_edges"],
)
def test_artifact_requires_canonical_sorted_arrays(field: str) -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=3,
        qubit_fraction=0.1,
        coupler_fraction=0.1,
        seed_key=_SEED_KEY,
    )
    artifact[field] = list(reversed(artifact[field]))
    _resign(artifact)

    with pytest.raises(ValueError, match="sorted"):
        _verify(artifact)


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("schema_version",), True),
        (("size",), True),
        (("requested_defects", "qubit_fraction"), 0),
        (("requested_defects", "coupler_fraction"), False),
        (("requested_defects", "qubit_fraction"), float("nan")),
        (("requested_defects", "coupler_fraction"), float("inf")),
        (("requested_defects", "seed"), True),
        (("requested_defects", "seed"), -1),
        (("requested_defects", "seed"), 2**256),
        (("realized_statistics", "qubit_fraction"), 0),
        (("realized_statistics", "coupler_fraction"), False),
        (("realized_statistics", "component_count"), True),
        (("realized_statistics", "component_count"), -1),
        (("realized_statistics", "largest_component_fraction"), float("nan")),
    ],
    ids=[
        "bool-schema-version",
        "bool-size",
        "integer-requested-fraction",
        "bool-requested-fraction",
        "nan-requested-fraction",
        "infinite-requested-fraction",
        "bool-seed",
        "negative-seed",
        "overflow-seed",
        "integer-realized-fraction",
        "bool-realized-fraction",
        "bool-component-count",
        "negative-component-count",
        "nan-largest-component",
    ],
)
def test_artifact_rejects_noncanonical_scalar_types_and_ranges(
    path: tuple[str, ...],
    invalid: object,
) -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=3,
        qubit_fraction=0.1,
        coupler_fraction=0.1,
        seed_key=_SEED_KEY,
    )
    target = artifact
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = invalid

    with pytest.raises((TypeError, ValueError)):
        _verify(artifact)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("nodes", lambda values: tuple(values)),
        ("nodes", lambda values: [True, *values[1:]]),
        ("nodes", lambda values: [-1, *values[1:]]),
        ("nodes", lambda values: [*values[:-1], 2**64]),
        ("nodes", lambda values: [values[0], values[0], *values[2:]]),
        ("edges", lambda values: [tuple(values[0]), *values[1:]]),
        ("edges", lambda values: [list(reversed(values[0])), *values[1:]]),
        ("edges", lambda values: [[values[0][0], values[0][0]], *values[1:]]),
        ("edges", lambda values: [values[0], values[0], *values[2:]]),
        ("edges", lambda values: [[False, values[0][1]], *values[1:]]),
        ("removed_nodes", lambda values: [*values, values[-1]]),
        ("removed_edges", lambda values: [*values, values[-1]]),
    ],
    ids=[
        "tuple-node-array",
        "bool-node",
        "negative-node",
        "overflow-node",
        "duplicate-node",
        "tuple-edge",
        "reversed-edge",
        "self-loop",
        "duplicate-edge",
        "bool-edge-endpoint",
        "duplicate-removed-node",
        "duplicate-removed-edge",
    ],
)
def test_artifact_rejects_malformed_or_noncanonical_graph_arrays(
    field: str,
    replacement,
) -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=3,
        qubit_fraction=0.1,
        coupler_fraction=0.1,
        seed_key=_SEED_KEY,
    )
    artifact[field] = replacement(artifact[field])

    with pytest.raises((TypeError, ValueError)):
        _verify(artifact)


@pytest.mark.parametrize(
    "invalid_graph",
    [
        nx.DiGraph([(0, 1)]),
        nx.MultiGraph([(0, 1), (0, 1)]),
        nx.Graph([(True, 2)]),
        nx.Graph([(-1, 2)]),
        nx.Graph([(0, 2**64)]),
        nx.Graph([(0, 0)]),
    ],
    ids=["directed", "multigraph", "bool-node", "negative-node", "overflow-node", "self-loop"],
)
def test_graph_identity_rejects_non_simple_or_non_uint64_graphs(
    invalid_graph: nx.Graph,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        host_graph_sha256(invalid_graph)


def test_nonzero_fraction_that_rounds_to_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="qubit.*rounds to zero"):
        build_realized_host_artifact(
            topology="chimera",
            size=1,
            qubit_fraction=0.01,
            coupler_fraction=0.0,
            seed_key=_SEED_KEY,
        )

    with pytest.raises(ValueError, match="coupler.*rounds to zero"):
        build_realized_host_artifact(
            topology="chimera",
            size=1,
            qubit_fraction=0.0,
            coupler_fraction=0.001,
            seed_key=_SEED_KEY,
        )


@pytest.mark.parametrize(
    "invalid_fraction",
    [0, False, -0.1, 1.1, float("nan"), float("inf"), float("-inf")],
    ids=["integer", "bool", "negative", "above-one", "nan", "infinity", "negative-infinity"],
)
def test_builder_rejects_non_binary64_or_out_of_range_fractions(
    invalid_fraction: object,
) -> None:
    with pytest.raises(ValueError):
        build_realized_host_artifact(
            topology="chimera",
            size=2,
            qubit_fraction=invalid_fraction,
            coupler_fraction=0.0,
            seed_key=_SEED_KEY,
        )


@pytest.mark.parametrize(
    ("qubit_fraction", "coupler_fraction", "node_count", "edge_count", "component_count"),
    [(0.0, 1.0, 8, 0, 8), (1.0, 0.0, 0, 0, 0)],
)
def test_disconnected_and_empty_realized_hosts_have_exact_statistics(
    qubit_fraction: float,
    coupler_fraction: float,
    node_count: int,
    edge_count: int,
    component_count: int,
) -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=1,
        qubit_fraction=qubit_fraction,
        coupler_fraction=coupler_fraction,
        seed_key=_SEED_KEY,
    )
    graph = _verify(artifact)

    assert graph.number_of_nodes() == node_count
    assert graph.number_of_edges() == edge_count
    assert artifact["realized_statistics"]["component_count"] == component_count
    expected_largest_fraction = 0.0 if node_count == 0 else 1.0 / node_count
    assert artifact["realized_statistics"]["largest_component_fraction"] == (
        expected_largest_fraction
    )


def test_empty_pristine_topology_and_negative_zero_are_handled_explicitly() -> None:
    with pytest.raises(ValueError, match="pristine host must contain at least one node"):
        build_realized_host_artifact(
            topology="pegasus",
            size=1,
            qubit_fraction=0.0,
            coupler_fraction=0.0,
            seed_key=_SEED_KEY,
        )
    with pytest.raises(ValueError, match="pristine host must contain at least one node"):
        reconstruct_pristine_host_for_generation(topology="pegasus", size=1)

    artifact = build_realized_host_artifact(
        topology="chimera",
        size=1,
        qubit_fraction=-0.0,
        coupler_fraction=-0.0,
        seed_key=_SEED_KEY,
    )
    requested = artifact["requested_defects"]
    assert math.copysign(1.0, requested["qubit_fraction"]) == 1.0
    assert math.copysign(1.0, requested["coupler_fraction"]) == 1.0


def test_seed_key_must_be_exactly_32_raw_bytes() -> None:
    for invalid in (b"short", bytearray(32), bytes(33)):
        with pytest.raises((TypeError, ValueError), match="seed_key"):
            build_realized_host_artifact(
                topology="chimera",
                size=2,
                qubit_fraction=0.0,
                coupler_fraction=0.0,
                seed_key=invalid,
            )


@pytest.mark.parametrize(
    "seed_key",
    [bytes(32), bytes(31) + b"\x01", bytes([0x80, *range(1, 32)]), b"\xff" * 32],
    ids=["zero", "leading-zeroes", "high-bit", "uint256-max"],
)
def test_requested_seed_round_trips_all_unsigned_256_bit_boundaries(seed_key: bytes) -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=2,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=seed_key,
    )
    requested = artifact["requested_defects"]
    assert requested["seed"].to_bytes(32, "big") == seed_key
    _verify(artifact, seed_key=seed_key)


def test_requested_seed_rejects_bool_impersonation() -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=2,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=_SEED_KEY,
    )
    requested = artifact["requested_defects"]

    requested["seed"] = True
    _resign(artifact)
    with pytest.raises(ValueError, match="unsigned 256-bit integer"):
        _verify(artifact)


def test_external_content_digests_are_mandatory_and_verified() -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=2,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=_SEED_KEY,
    )

    with pytest.raises(ValueError, match="expected_host_artifact_sha256"):
        validate_realized_host_artifact(
            artifact,
            seed_key=_SEED_KEY,
            expected_host_artifact_sha256=None,
            expected_host_sha256=artifact["host_sha256"],
        )
    with pytest.raises(ValueError, match="external host artifact digest"):
        validate_realized_host_artifact(
            artifact,
            seed_key=_SEED_KEY,
            expected_host_artifact_sha256="0" * 64,
            expected_host_sha256=artifact["host_sha256"],
        )
    with pytest.raises(ValueError, match="expected_host_sha256"):
        validate_realized_host_artifact(
            artifact,
            seed_key=_SEED_KEY,
            expected_host_artifact_sha256=artifact["host_artifact_sha256"],
            expected_host_sha256=None,
        )
    with pytest.raises(ValueError, match="external host graph digest"):
        validate_realized_host_artifact(
            artifact,
            seed_key=_SEED_KEY,
            expected_host_artifact_sha256=artifact["host_artifact_sha256"],
            expected_host_sha256="0" * 64,
        )


def test_external_artifact_identity_is_not_copied_from_resigned_input() -> None:
    original = build_realized_host_artifact(
        topology="chimera",
        size=2,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=b"a" * 32,
    )
    resigned = build_realized_host_artifact(
        topology="chimera",
        size=2,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=b"b" * 32,
    )
    assert original["host_sha256"] == resigned["host_sha256"]

    with pytest.raises(ValueError, match="external host artifact digest"):
        validate_realized_host_artifact(
            resigned,
            seed_key=b"b" * 32,
            expected_host_artifact_sha256=original["host_artifact_sha256"],
            expected_host_sha256=original["host_sha256"],
        )


@pytest.mark.parametrize(
    ("qubit_fraction", "coupler_fraction"),
    [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1)],
)
def test_record_loader_refuses_missing_artifact(
    qubit_fraction: float,
    coupler_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="record-facing loader.*artifact"):
        load_realized_host(
            topology="chimera",
            size=3,
            qubit_fraction=qubit_fraction,
            coupler_fraction=coupler_fraction,
            seed_key=_SEED_KEY,
            artifact=None,
            expected_host_artifact_sha256="0" * 64,
            expected_host_sha256="0" * 64,
        )


def test_generation_only_helper_reconstructs_pristine_host() -> None:
    loaded = reconstruct_pristine_host_for_generation(topology="pegasus", size=3)
    pristine = host_graph("pegasus", 3)
    assert host_graph_sha256(loaded) == host_graph_sha256(pristine)


def test_record_loader_accepts_only_a_matching_bound_artifact() -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=3,
        qubit_fraction=0.1,
        coupler_fraction=0.1,
        seed_key=_SEED_KEY,
    )
    loaded = load_realized_host(
        topology="chimera",
        size=3,
        qubit_fraction=0.1,
        coupler_fraction=0.1,
        seed_key=_SEED_KEY,
        artifact=artifact,
        expected_host_artifact_sha256=artifact["host_artifact_sha256"],
        expected_host_sha256=artifact["host_sha256"],
    )
    assert host_graph_sha256(loaded) == artifact["host_sha256"]

    with pytest.raises(ValueError, match="defect fractions differ"):
        load_realized_host(
            topology="chimera",
            size=3,
            qubit_fraction=0.2,
            coupler_fraction=0.1,
            seed_key=_SEED_KEY,
            artifact=artifact,
            expected_host_artifact_sha256=artifact["host_artifact_sha256"],
            expected_host_sha256=artifact["host_sha256"],
        )


def test_window_node_and_edge_validation_is_exact() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3)])
    assert validate_host_nodes(graph, [0, 1, 2], name="window") == (0, 1, 2)
    assert validate_host_edges(
        graph,
        [[0, 1], [0, 2], [1, 2]],
        name="window_edges",
        allowed_nodes=[0, 1, 2],
    ) == ((0, 1), (0, 2), (1, 2))
    validate_window(graph, [0, 1, 2], [[0, 1], [0, 2], [1, 2]])

    with pytest.raises(ValueError, match="induced realized-host edges"):
        validate_window(graph, [0, 1, 2], [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="absent from the realized host"):
        validate_host_edges(graph, [[0, 3]], name="bad_edges")

    for invalid_nodes in ([True], [-1], [2**64], [0, 0], [1, 0]):
        with pytest.raises(ValueError):
            validate_host_nodes(graph, invalid_nodes, name="bad_nodes")
    for invalid_edges in ([[False, 1]], [[-1, 0]], [[0, 2**64]], [[1, 0]], [[0, 0]]):
        with pytest.raises(ValueError):
            validate_host_edges(graph, invalid_edges, name="bad_edges")


def test_host_fill_fraction_uses_realized_nodes_and_validates_occupancy() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    assert host_fill_fraction(graph, [0, 2, 3]) == 0.75
    assert host_fill_fraction(nx.Graph(), []) == 0.0

    with pytest.raises(ValueError, match="absent from the realized host"):
        host_fill_fraction(graph, [0, 99])


def test_chain_and_candidate_validation_catches_disconnected_or_missing_nodes() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (3, 4)])
    validate_chain(graph, [0, 1, 2], [[0, 1], [1, 2]], name="chain")
    validate_candidate_bank(graph, [[0], [0, 1], [3, 4]])

    with pytest.raises(ValueError, match="connected"):
        validate_chain(graph, [0, 2], [], name="chain")
    with pytest.raises(ValueError, match="absent from the realized host"):
        validate_candidate_bank(graph, [[0, 99]])
    with pytest.raises(ValueError, match="duplicate-free"):
        validate_candidate_bank(graph, [[0], [0]])
    with pytest.raises(ValueError, match="sorted by"):
        validate_candidate_bank(graph, [[0, 1], [0]])


def test_contact_edges_must_exist_and_join_the_declared_chains() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    validate_contact_edges(
        graph,
        [[1, 2]],
        left_chain=[0, 1],
        right_chain=[2, 3],
        name="rho_0_1",
    )

    with pytest.raises(ValueError, match="does not join"):
        validate_contact_edges(
            graph,
            [[0, 1]],
            left_chain=[0, 1],
            right_chain=[2, 3],
            name="rho_0_1",
        )


def test_minor_embedding_validation_accepts_full_feasible_embedding() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)])
    validate_minor_embedding(
        graph,
        logical_edges=[[0, 1], [1, 2]],
        chains={0: [0, 1], 1: [4], 2: [2, 3]},
        chain_edges={0: [[0, 1]], 1: [], 2: [[2, 3]]},
        contact_edges={(0, 1): [[1, 4]], (1, 2): [[3, 4]]},
    )


def test_minor_embedding_rejects_overlap_disconnection_and_missing_contact() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4)])

    with pytest.raises(ValueError, match="pairwise disjoint"):
        validate_minor_embedding(
            graph,
            logical_edges=[[0, 1]],
            chains={0: [0, 1], 1: [1, 2]},
        )
    with pytest.raises(ValueError, match="connected"):
        validate_minor_embedding(
            graph,
            logical_edges=[[0, 1]],
            chains={0: [0, 2], 1: [3, 4]},
        )
    with pytest.raises(ValueError, match="logical edge.*realized contact"):
        validate_minor_embedding(
            graph,
            logical_edges=[[0, 1]],
            chains={0: [0, 1], 1: [3, 4]},
        )


def test_minor_embedding_checks_every_logical_edge_and_optional_mapping_key() -> None:
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (4, 5)])

    with pytest.raises(ValueError, match=r"logical edge \(1, 2\).*realized contact"):
        validate_minor_embedding(
            graph,
            logical_edges=[[0, 1], [1, 2]],
            chains={0: [0], 1: [1], 2: [4, 5]},
        )
    with pytest.raises(ValueError, match="chain_edges keys must exactly match"):
        validate_minor_embedding(
            graph,
            logical_edges=[[0, 1]],
            chains={0: [0], 1: [1]},
            chain_edges={0: []},
        )
    with pytest.raises(ValueError, match="contact_edges keys must exactly match"):
        validate_minor_embedding(
            graph,
            logical_edges=[[0, 1]],
            chains={0: [0], 1: [1]},
            contact_edges={(0, 2): [[0, 1]]},
        )
    with pytest.raises(ValueError, match="no recorded contact"):
        validate_minor_embedding(
            graph,
            logical_edges=[[0, 1]],
            chains={0: [0], 1: [1]},
            contact_edges={(0, 1): []},
        )


def test_wrong_seed_and_unknown_schema_fields_are_rejected() -> None:
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=3,
        qubit_fraction=0.1,
        coupler_fraction=0.1,
        seed_key=_SEED_KEY,
    )
    with pytest.raises(ValueError, match="seed_key"):
        _verify(artifact, seed_key=b"x" * 32)

    unknown = copy.deepcopy(artifact)
    unknown["surprise"] = True
    _resign(unknown)
    with pytest.raises(ValueError, match="schema fields differ"):
        _verify(unknown)

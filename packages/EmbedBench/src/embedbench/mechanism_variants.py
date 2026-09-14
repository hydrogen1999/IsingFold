"""Certified, deterministic variants of EmbedBench's seven hand-test motifs.

This module is the fail-closed single-mechanism foundation required by Sections 9.2 and 12
of the hard/OOD corpus specification.  It deliberately does not implement generic hardware
placement or composed traps: those operations need additional registered topology and
interaction-certificate primitives.  Callers receive explicit unsupported statuses instead
of a weaker substitute.

Only label-free, fully precommitted inputs enter the transformation boundary.  Every retained
state is independently replayed through :mod:`embedbench.exact` via
``handtests.certify_motif``.  Quality labels, learned scores, and baseline decisions have no
field in any accepted schema.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Literal, NoReturn, cast

import networkx as nx

from embedbench.exact import SearchAborted, all_completions
from embedbench.handtests import MOTIFS, Motif, build_motif, certify_motif
from embedbench.hard_ood_schema import (
    SeedRequest,
    VerifiedSeedResolver,
    canonical_bytes,
    require_exact_keys,
)
from embedbench.objective import INFEASIBLE, Outcome, compare

UINT64_MAX = 2**64 - 1
MECHANISM_FAMILIES: tuple[str, ...] = (
    "dead_end",
    "articulation",
    "cut_capacity",
    "multi_neighbour",
    "high_degree_trap",
    "short_chain_trap",
    "ordering",
)
if tuple(MOTIFS) != MECHANISM_FAMILIES:
    raise RuntimeError("registered mechanism families drifted from embedbench.handtests.MOTIFS")
MECHANISM_PARTITIONS: frozenset[str] = frozenset({"mechanism_dev", "mechanism_locked"})
PLACEMENT_MODES: frozenset[str] = frozenset({"template_subgraph", "registered_hardware_subgraph"})
ORIENTATIONS: frozenset[str] = frozenset({"forward", "reverse"})

MECHANISM_PARAMETER_SCHEMA = "embedbench.mechanism-parameter-row"
MECHANISM_PARAMETER_SCHEMA_VERSION = 1
MECHANISM_VARIANT_SCHEMA = "embedbench.mechanism-variant"
MECHANISM_VARIANT_SCHEMA_VERSION = 1
MECHANISM_CERTIFICATE_SCHEMA = "embedbench.mechanism-exact-certificate"
MECHANISM_CERTIFICATE_SCHEMA_VERSION = 1
MECHANISM_RESULT_SCHEMA = "embedbench.mechanism-generation-result"
MECHANISM_RESULT_SCHEMA_VERSION = 1
MECHANISM_REGISTRY_SCHEMA = "embedbench.mechanism-identity-registry"
MECHANISM_REGISTRY_SCHEMA_VERSION = 1
MECHANISM_PARENT_MAP_SCHEMA = "embedbench.mechanism-parent-partition-map"
MECHANISM_PARENT_MAP_SCHEMA_VERSION = 1
MECHANISM_PLAN_SCHEMA = "embedbench.mechanism-generation-plan"
MECHANISM_PLAN_SCHEMA_VERSION = 1
MECHANISM_ATTEMPT_LEDGER_SCHEMA = "embedbench.mechanism-attempt-ledger"
MECHANISM_ATTEMPT_LEDGER_SCHEMA_VERSION = 1
AUTHORITATIVE_ISOMORPHISM_MANIFEST_SCHEMA = (
    "embedbench.authoritative-isomorphism-engine-manifest"
)
AUTHORITATIVE_ISOMORPHISM_MANIFEST_SCHEMA_VERSION = 1

GenerationStatus = Literal[
    "accepted",
    "unsupported_generic_hardware_placement",
    "unsupported_composed_mechanism",
    "exact_certification_failed",
    "hypothesized_winner_not_certified",
    "non_positive_margin",
    "byte_reconstruction_failed",
    "mechanism_semantics_not_certified",
    "unsupported_authoritative_isomorphism",
    "contextual_replay_failed",
    "identity_collision",
    "parent_partition_collision",
]
GENERATION_STATUSES: frozenset[str] = frozenset(
    {
        "accepted",
        "unsupported_generic_hardware_placement",
        "unsupported_composed_mechanism",
        "exact_certification_failed",
        "hypothesized_winner_not_certified",
        "non_positive_margin",
        "byte_reconstruction_failed",
        "mechanism_semantics_not_certified",
        "unsupported_authoritative_isomorphism",
        "contextual_replay_failed",
        "identity_collision",
        "parent_partition_collision",
    }
)


def _require_generation_status(value: object) -> GenerationStatus:
    if type(value) is not str or value not in GENERATION_STATUSES:
        raise ValueError("status is not a registered mechanism-generation status")
    return cast(GenerationStatus, value)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PARENT_ID_PATTERN = re.compile(r"mechanism-parent-sha256:([0-9a-f]{64})\Z")
_MAX_DECORATION_NODES = 8

Edge = tuple[int, int]
Relabeling = tuple[tuple[int, int], ...]
Chain = tuple[int, tuple[int, ...]]
Action = tuple[int, int]


def _require_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    canonical_bytes(value)
    return unicodedata.normalize("NFC", value)


def _require_uint64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    value = _require_int(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_positive_int(value: object, name: str) -> int:
    value = _require_int(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character hexadecimal value")
    return value


def _require_schema(document: dict[str, Any], schema: str, version: int, name: str) -> None:
    if document["schema"] != schema:
        raise ValueError(f"{name} schema must equal {schema!r}")
    if type(document["schema_version"]) is not int or document["schema_version"] != version:
        raise ValueError(f"{name} schema_version must equal {version}")


def _json_tuple(value: object, name: str) -> tuple[object, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    return tuple(cast(list[object], value))


def _canonical_nodes(value: object, name: str, *, allow_empty: bool = True) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    nodes = tuple(_require_uint64(node, f"{name}[{index}]") for index, node in enumerate(value))
    if not allow_empty and not nodes:
        raise ValueError(f"{name} must not be empty")
    if nodes != tuple(sorted(set(nodes))):
        raise ValueError(f"{name} must be sorted and duplicate-free")
    return nodes


def _canonical_edges(value: object, name: str) -> tuple[Edge, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    edges: list[Edge] = []
    for index, raw_edge in enumerate(value):
        if type(raw_edge) is not tuple or len(raw_edge) != 2:
            raise TypeError(f"{name}[{index}] must be an immutable endpoint pair")
        left = _require_uint64(raw_edge[0], f"{name}[{index}][0]")
        right = _require_uint64(raw_edge[1], f"{name}[{index}][1]")
        if left >= right:
            raise ValueError(f"{name}[{index}] endpoints must satisfy left < right")
        edges.append((left, right))
    result = tuple(edges)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be lexicographically sorted and duplicate-free")
    return result


def _canonical_logical_nodes(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    nodes = tuple(_require_int(node, f"{name}[{index}]") for index, node in enumerate(value))
    if nodes != tuple(sorted(set(nodes))):
        raise ValueError(f"{name} must be sorted and duplicate-free")
    return nodes


def _canonical_logical_edges(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    edges: list[tuple[int, int]] = []
    for index, raw_edge in enumerate(value):
        if type(raw_edge) is not tuple or len(raw_edge) != 2:
            raise TypeError(f"{name}[{index}] must be an immutable endpoint pair")
        left = _require_int(raw_edge[0], f"{name}[{index}][0]")
        right = _require_int(raw_edge[1], f"{name}[{index}][1]")
        if left >= right:
            raise ValueError(f"{name}[{index}] endpoints must satisfy left < right")
        edges.append((left, right))
    result = tuple(edges)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be lexicographically sorted and duplicate-free")
    return result


def _canonical_relabeling(
    value: object,
    name: str,
    *,
    domain: tuple[int, ...],
    target_uint64: bool,
) -> Relabeling:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    pairs: list[tuple[int, int]] = []
    for index, raw_pair in enumerate(value):
        if type(raw_pair) is not tuple or len(raw_pair) != 2:
            raise TypeError(f"{name}[{index}] must be an immutable pair")
        source = _require_int(raw_pair[0], f"{name}[{index}][0]")
        target = (
            _require_uint64(raw_pair[1], f"{name}[{index}][1]")
            if target_uint64
            else _require_int(raw_pair[1], f"{name}[{index}][1]")
        )
        pairs.append((source, target))
    result = tuple(pairs)
    if tuple(source for source, _ in result) != domain:
        raise ValueError(f"{name} source domain must exactly equal the canonical template domain")
    targets = tuple(target for _, target in result)
    if len(targets) != len(set(targets)):
        raise ValueError(f"{name} targets must be duplicate-free")
    return result


def _canonical_chains(
    value: object,
    name: str,
    *,
    logical_nodes: Collection[int] | None = None,
    host_nodes: Collection[int] | None = None,
) -> tuple[Chain, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    chains: list[Chain] = []
    for index, raw_chain in enumerate(value):
        if type(raw_chain) is not tuple or len(raw_chain) != 2:
            raise TypeError(f"{name}[{index}] must be an immutable (logical, nodes) pair")
        logical = _require_int(raw_chain[0], f"{name}[{index}][0]")
        nodes = _canonical_nodes(raw_chain[1], f"{name}[{index}][1]", allow_empty=False)
        chains.append((logical, nodes))
    result = tuple(chains)
    if tuple(logical for logical, _ in result) != tuple(
        sorted(set(logical for logical, _ in result))
    ):
        raise ValueError(f"{name} must be sorted by duplicate-free logical variable")
    if logical_nodes is not None and not set(logical for logical, _ in result) <= set(
        logical_nodes
    ):
        raise ValueError(f"{name} contains an unknown logical variable")
    occupied = tuple(node for _, nodes in result for node in nodes)
    if len(occupied) != len(set(occupied)):
        raise ValueError(f"{name} chains must be pairwise disjoint")
    if host_nodes is not None and not set(occupied) <= set(host_nodes):
        raise ValueError(f"{name} contains a node outside the realized host")
    return result


def _canonical_action(value: object, name: str) -> Action:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(f"{name} must be an immutable (logical, host-node) pair")
    return (
        _require_int(value[0], f"{name}[0]"),
        _require_uint64(value[1], f"{name}[1]"),
    )


def _canonical_actions(value: object, name: str) -> tuple[Action, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    actions = tuple(
        _canonical_action(action, f"{name}[{index}]") for index, action in enumerate(value)
    )
    if len(actions) < 2:
        raise ValueError(f"{name} must contain at least two candidates")
    if len(actions) != len(set(actions)):
        raise ValueError(f"{name} must be duplicate-free")
    return actions


def _graph_arrays(graph: nx.Graph) -> tuple[tuple[int, ...], tuple[Edge, ...]]:
    nodes = tuple(sorted(_require_uint64(node, "host node") for node in graph.nodes))
    edges = tuple(
        sorted(
            (
                min(
                    _require_uint64(left, "host endpoint"), _require_uint64(right, "host endpoint")
                ),
                max(
                    _require_uint64(left, "host endpoint"), _require_uint64(right, "host endpoint")
                ),
            )
            for left, right in graph.edges
        )
    )
    return nodes, edges


def _logical_graph_arrays(graph: nx.Graph) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    nodes = tuple(sorted(_require_int(node, "logical node") for node in graph.nodes))
    edges = tuple(
        sorted(
            (
                min(
                    _require_int(left, "logical endpoint"), _require_int(right, "logical endpoint")
                ),
                max(
                    _require_int(left, "logical endpoint"), _require_int(right, "logical endpoint")
                ),
            )
            for left, right in graph.edges
        )
    )
    return nodes, edges


def _automorphisms(graph: nx.Graph) -> tuple[dict[int, int], ...]:
    nodes = tuple(sorted(graph.nodes))
    identity = {node: node for node in nodes}
    mappings: dict[tuple[int, ...], dict[int, int]] = {}
    for raw_mapping in nx.algorithms.isomorphism.GraphMatcher(graph, graph).isomorphisms_iter():
        key = tuple(raw_mapping[node] for node in nodes)
        mappings[key] = {node: raw_mapping[node] for node in nodes}
    identity_key = tuple(nodes)
    ordered_keys = (identity_key,) + tuple(sorted(key for key in mappings if key != identity_key))
    return tuple(mappings.get(key, identity) for key in ordered_keys)


def _template_payload(motif: Motif) -> dict[str, object]:
    host_nodes, host_edges = _graph_arrays(motif.host)
    logical_nodes, logical_edges = _logical_graph_arrays(motif.logical)
    cores = tuple((logical, tuple(sorted(nodes))) for logical, nodes in sorted(motif.cores.items()))
    return {
        "schema": "embedbench.mechanism-motif-template",
        "schema_version": 1,
        "family": motif.name,
        "mechanism": motif.mechanism,
        "host_nodes": list(host_nodes),
        "host_edges": [list(edge) for edge in host_edges],
        "logical_nodes": list(logical_nodes),
        "logical_edges": [list(edge) for edge in logical_edges],
        "cores": [[logical, list(nodes)] for logical, nodes in cores],
        "actions": [list(action) for action in motif.actions],
        "hypothesis_winner": list(motif.hypothesis_winner),
        "l_cap": motif.l_cap,
        "q_cap": motif.q_cap,
        "original_candidate_index": 0,
    }


def motif_template_sha256(family: str) -> str:
    """Return the immutable source-template identity for one registered family."""

    if type(family) is not str or family not in MECHANISM_FAMILIES:
        raise ValueError(f"family must be one of {list(MECHANISM_FAMILIES)}")
    return hashlib.sha256(canonical_bytes(_template_payload(build_motif(family)))).hexdigest()


def host_automorphism_count(family: str) -> int:
    """Return the size of the deterministic registered host-symmetry table."""

    if type(family) is not str or family not in MECHANISM_FAMILIES:
        raise ValueError(f"family must be one of {list(MECHANISM_FAMILIES)}")
    return len(_automorphisms(build_motif(family).host))


def logical_symmetry_count(family: str) -> int:
    """Return the size of the deterministic registered logical-symmetry table."""

    if type(family) is not str or family not in MECHANISM_FAMILIES:
        raise ValueError(f"family must be one of {list(MECHANISM_FAMILIES)}")
    return len(_automorphisms(build_motif(family).logical))


def parent_seed_namespace(release_id: str, partition: str, family: str) -> str:
    return f"{_require_string(release_id, 'release_id')}/mechanism-parent/{partition}/{family}"


def transformation_seed_namespace(release_id: str, partition: str, family: str) -> str:
    return f"{_require_string(release_id, 'release_id')}/mechanism-transform/{partition}/{family}"


def _parent_scientific_payload(family: str) -> dict[str, object]:
    """Canonical pre-transform scientific state, with no split or provenance fields.

    This is intentionally derived before relabeling, orientation, decoration, and every
    stochastic transform.  ``family`` selects the registered source but is not serialized:
    two aliases for identical scientific content must identify the same parent.
    """

    if type(family) is not str or family not in MECHANISM_FAMILIES:
        raise ValueError(f"family must be one of {list(MECHANISM_FAMILIES)}")
    motif = build_motif(family)
    host_nodes, host_edges = _graph_arrays(motif.host)
    logical_nodes, logical_edges = _logical_graph_arrays(motif.logical)
    cores = tuple((logical, tuple(sorted(nodes))) for logical, nodes in sorted(motif.cores.items()))
    return {
        "schema": "embedbench.mechanism-parent-scientific-content",
        "schema_version": 1,
        "host_nodes": list(host_nodes),
        "host_edges": [list(edge) for edge in host_edges],
        "logical_nodes": list(logical_nodes),
        "logical_edges": [list(edge) for edge in logical_edges],
        "cores": [[logical, list(nodes)] for logical, nodes in cores],
        # The registered hand-test state uses the complete realized host as its window.
        "window_nodes": list(host_nodes),
        "actions": [list(action) for action in sorted(motif.actions)],
        "l_cap": motif.l_cap,
        "q_cap": motif.q_cap,
    }


def parent_scientific_sha256(family: str) -> str:
    """Hash only canonical pre-transform scientific content."""

    return hashlib.sha256(canonical_bytes(_parent_scientific_payload(family))).hexdigest()


def mechanism_parent_id(*, parent_scientific_sha256_value: str) -> str:
    """Return the parent identity without split, seed, release, or provenance leakage."""

    digest = _require_sha256(
        parent_scientific_sha256_value,
        "parent_scientific_sha256_value",
    )
    return f"mechanism-parent-sha256:{digest}"


_PARAMETER_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "partition",
        "family",
        "parent_ordinal",
        "transformation_ordinal",
        "parent_seed_namespace",
        "parent_seed_key",
        "transformation_seed_namespace",
        "transformation_seed_key",
        "mechanism_parent_id",
        "parent_scientific_sha256",
        "motif_template_sha256",
        "placement_mode",
        "host_automorphism_index",
        "logical_symmetry_index",
        "orientation",
        "host_node_relabeling",
        "logical_role_relabeling",
        "decoration_nodes",
        "decoration_edges",
    }
)


@dataclass(frozen=True, slots=True)
class MechanismParameterRow:
    """One complete precommitted, label-free single-mechanism transformation."""

    SCHEMA: ClassVar[str] = MECHANISM_PARAMETER_SCHEMA
    SCHEMA_VERSION: ClassVar[int] = MECHANISM_PARAMETER_SCHEMA_VERSION

    release_id: str
    partition: str
    family: str
    parent_ordinal: int
    transformation_ordinal: int
    parent_seed_namespace: str
    parent_seed_key: str
    transformation_seed_namespace: str
    transformation_seed_key: str
    mechanism_parent_id: str
    parent_scientific_sha256: str
    motif_template_sha256: str
    placement_mode: str
    host_automorphism_index: int
    logical_symmetry_index: int
    orientation: str
    host_node_relabeling: Relabeling
    logical_role_relabeling: Relabeling
    decoration_nodes: tuple[int, ...]
    decoration_edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        release_id = _require_string(self.release_id, "release_id")
        if type(self.partition) is not str or self.partition not in MECHANISM_PARTITIONS:
            raise ValueError(f"partition must be one of {sorted(MECHANISM_PARTITIONS)}")
        if type(self.family) is not str or self.family not in MECHANISM_FAMILIES:
            raise ValueError(f"family must be one of {list(MECHANISM_FAMILIES)}")
        ordinal = _require_nonnegative_int(self.parent_ordinal, "parent_ordinal")
        transformation_ordinal = _require_nonnegative_int(
            self.transformation_ordinal,
            "transformation_ordinal",
        )
        parent_key = _require_sha256(self.parent_seed_key, "parent_seed_key")
        transform_key = _require_sha256(self.transformation_seed_key, "transformation_seed_key")
        if parent_key == transform_key:
            raise ValueError("parent and transformation seed keys must be independent")
        expected_parent_namespace = parent_seed_namespace(release_id, self.partition, self.family)
        expected_transform_namespace = transformation_seed_namespace(
            release_id, self.partition, self.family
        )
        if self.parent_seed_namespace != expected_parent_namespace:
            raise ValueError("parent_seed_namespace does not match release/partition/family")
        if self.transformation_seed_namespace != expected_transform_namespace:
            raise ValueError(
                "transformation_seed_namespace does not match release/partition/family"
            )
        if self.parent_seed_namespace == self.transformation_seed_namespace:
            raise ValueError("parent and transformation seed namespaces must be independent")

        template_digest = _require_sha256(self.motif_template_sha256, "motif_template_sha256")
        expected_template_digest = motif_template_sha256(self.family)
        if template_digest != expected_template_digest:
            raise ValueError("motif_template_sha256 does not match the registered template")
        if (
            type(self.mechanism_parent_id) is not str
            or _PARENT_ID_PATTERN.fullmatch(self.mechanism_parent_id) is None
        ):
            raise ValueError("mechanism_parent_id must use the canonical prefixed SHA-256 form")
        scientific_digest = _require_sha256(
            self.parent_scientific_sha256,
            "parent_scientific_sha256",
        )
        if scientific_digest != parent_scientific_sha256(self.family):
            raise ValueError(
                "parent_scientific_sha256 does not match canonical pre-transform content"
            )
        expected_parent_id = mechanism_parent_id(
            parent_scientific_sha256_value=scientific_digest,
        )
        if self.mechanism_parent_id != expected_parent_id:
            raise ValueError("mechanism_parent_id does not match its parent identity fields")

        if type(self.placement_mode) is not str or self.placement_mode not in PLACEMENT_MODES:
            raise ValueError(f"placement_mode must be one of {sorted(PLACEMENT_MODES)}")
        host_auto_index = _require_nonnegative_int(
            self.host_automorphism_index, "host_automorphism_index"
        )
        logical_auto_index = _require_nonnegative_int(
            self.logical_symmetry_index, "logical_symmetry_index"
        )
        if type(self.orientation) is not str or self.orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {sorted(ORIENTATIONS)}")

        motif = build_motif(self.family)
        host_nodes, _ = _graph_arrays(motif.host)
        logical_nodes, _ = _logical_graph_arrays(motif.logical)
        if host_auto_index >= len(_automorphisms(motif.host)):
            raise ValueError("host_automorphism_index is outside the registered symmetry table")
        if logical_auto_index >= len(_automorphisms(motif.logical)):
            raise ValueError("logical_symmetry_index is outside the registered symmetry table")
        host_relabeling = _canonical_relabeling(
            self.host_node_relabeling,
            "host_node_relabeling",
            domain=host_nodes,
            target_uint64=True,
        )
        logical_relabeling = _canonical_relabeling(
            self.logical_role_relabeling,
            "logical_role_relabeling",
            domain=logical_nodes,
            target_uint64=False,
        )
        decoration_nodes = _canonical_nodes(self.decoration_nodes, "decoration_nodes")
        if len(decoration_nodes) > _MAX_DECORATION_NODES:
            raise ValueError(
                f"schema version 1 permits at most {_MAX_DECORATION_NODES} decoration nodes"
            )
        output_host_nodes = {target for _, target in host_relabeling}
        if output_host_nodes & set(decoration_nodes):
            raise ValueError("decoration nodes must be new in the transformed host namespace")
        decoration_edges = _canonical_edges(self.decoration_edges, "decoration_edges")
        available = output_host_nodes | set(decoration_nodes)
        if any(left not in available or right not in available for left, right in decoration_edges):
            raise ValueError("every decoration edge endpoint must exist in the output host")
        if any(
            left not in decoration_nodes and right not in decoration_nodes
            for left, right in decoration_edges
        ):
            raise ValueError("every decoration edge must touch at least one decoration node")
        output_labels = dict(host_relabeling)
        transformed_template_edges = {
            tuple(sorted((output_labels[left], output_labels[right])))
            for left, right in motif.host.edges
        }
        if transformed_template_edges & set(decoration_edges):
            raise ValueError("decoration_edges must not duplicate a transformed template edge")

        object.__setattr__(self, "release_id", release_id)
        object.__setattr__(self, "parent_ordinal", ordinal)
        object.__setattr__(self, "transformation_ordinal", transformation_ordinal)
        object.__setattr__(self, "parent_seed_key", parent_key)
        object.__setattr__(self, "transformation_seed_key", transform_key)
        object.__setattr__(self, "parent_scientific_sha256", scientific_digest)
        object.__setattr__(self, "motif_template_sha256", template_digest)
        object.__setattr__(self, "host_automorphism_index", host_auto_index)
        object.__setattr__(self, "logical_symmetry_index", logical_auto_index)
        object.__setattr__(self, "host_node_relabeling", host_relabeling)
        object.__setattr__(self, "logical_role_relabeling", logical_relabeling)
        object.__setattr__(self, "decoration_nodes", decoration_nodes)
        object.__setattr__(self, "decoration_edges", decoration_edges)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "release_id": self.release_id,
            "partition": self.partition,
            "family": self.family,
            "parent_ordinal": self.parent_ordinal,
            "transformation_ordinal": self.transformation_ordinal,
            "parent_seed_namespace": self.parent_seed_namespace,
            "parent_seed_key": self.parent_seed_key,
            "transformation_seed_namespace": self.transformation_seed_namespace,
            "transformation_seed_key": self.transformation_seed_key,
            "mechanism_parent_id": self.mechanism_parent_id,
            "parent_scientific_sha256": self.parent_scientific_sha256,
            "motif_template_sha256": self.motif_template_sha256,
            "placement_mode": self.placement_mode,
            "host_automorphism_index": self.host_automorphism_index,
            "logical_symmetry_index": self.logical_symmetry_index,
            "orientation": self.orientation,
            "host_node_relabeling": [list(pair) for pair in self.host_node_relabeling],
            "logical_role_relabeling": [list(pair) for pair in self.logical_role_relabeling],
            "decoration_nodes": list(self.decoration_nodes),
            "decoration_edges": [list(edge) for edge in self.decoration_edges],
        }

    @classmethod
    def from_dict(cls, value: object) -> MechanismParameterRow:
        document = require_exact_keys(value, _PARAMETER_FIELDS, "mechanism parameter row")
        _require_schema(document, cls.SCHEMA, cls.SCHEMA_VERSION, "mechanism parameter row")

        def pairs(field: str) -> tuple[tuple[object, object], ...]:
            rows = _json_tuple(document[field], field)
            result: list[tuple[object, object]] = []
            for index, raw in enumerate(rows):
                pair = _json_tuple(raw, f"{field}[{index}]")
                if len(pair) != 2:
                    raise ValueError(f"{field}[{index}] must contain two values")
                result.append((pair[0], pair[1]))
            return tuple(result)

        return cls(
            release_id=document["release_id"],
            partition=document["partition"],
            family=document["family"],
            parent_ordinal=document["parent_ordinal"],
            transformation_ordinal=document["transformation_ordinal"],
            parent_seed_namespace=document["parent_seed_namespace"],
            parent_seed_key=document["parent_seed_key"],
            transformation_seed_namespace=document["transformation_seed_namespace"],
            transformation_seed_key=document["transformation_seed_key"],
            mechanism_parent_id=document["mechanism_parent_id"],
            parent_scientific_sha256=document["parent_scientific_sha256"],
            motif_template_sha256=document["motif_template_sha256"],
            placement_mode=document["placement_mode"],
            host_automorphism_index=document["host_automorphism_index"],
            logical_symmetry_index=document["logical_symmetry_index"],
            orientation=document["orientation"],
            host_node_relabeling=cast(Relabeling, pairs("host_node_relabeling")),
            logical_role_relabeling=cast(Relabeling, pairs("logical_role_relabeling")),
            decoration_nodes=cast(
                tuple[int, ...],
                _json_tuple(document["decoration_nodes"], "decoration_nodes"),
            ),
            decoration_edges=cast(tuple[Edge, ...], pairs("decoration_edges")),
        )

    def to_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


def make_parameter_row(
    *,
    release_id: str,
    partition: str,
    family: str,
    parent_ordinal: int,
    transformation_ordinal: int = 0,
    parent_seed_key: str,
    transformation_seed_key: str,
    placement_mode: str = "template_subgraph",
    host_automorphism_index: int = 0,
    logical_symmetry_index: int = 0,
    orientation: str = "forward",
    host_node_relabeling: Relabeling | None = None,
    logical_role_relabeling: Relabeling | None = None,
    decoration_nodes: tuple[int, ...] = (),
    decoration_edges: tuple[Edge, ...] = (),
) -> MechanismParameterRow:
    """Build a row while deriving, rather than trusting, all parent self-identities."""

    release_id = _require_string(release_id, "release_id")
    if type(family) is not str or family not in MECHANISM_FAMILIES:
        raise ValueError(f"family must be one of {list(MECHANISM_FAMILIES)}")
    motif = build_motif(family)
    host_nodes, _ = _graph_arrays(motif.host)
    logical_nodes, _ = _logical_graph_arrays(motif.logical)
    template_digest = motif_template_sha256(family)
    parent_namespace = parent_seed_namespace(release_id, partition, family)
    transform_namespace = transformation_seed_namespace(release_id, partition, family)
    scientific_digest = parent_scientific_sha256(family)
    parent_id = mechanism_parent_id(
        parent_scientific_sha256_value=scientific_digest,
    )
    return MechanismParameterRow(
        release_id=release_id,
        partition=partition,
        family=family,
        parent_ordinal=parent_ordinal,
        transformation_ordinal=transformation_ordinal,
        parent_seed_namespace=parent_namespace,
        parent_seed_key=parent_seed_key,
        transformation_seed_namespace=transform_namespace,
        transformation_seed_key=transformation_seed_key,
        mechanism_parent_id=parent_id,
        parent_scientific_sha256=scientific_digest,
        motif_template_sha256=template_digest,
        placement_mode=placement_mode,
        host_automorphism_index=host_automorphism_index,
        logical_symmetry_index=logical_symmetry_index,
        orientation=orientation,
        host_node_relabeling=(
            tuple((node, node) for node in host_nodes)
            if host_node_relabeling is None
            else host_node_relabeling
        ),
        logical_role_relabeling=(
            tuple((node, node) for node in logical_nodes)
            if logical_role_relabeling is None
            else logical_role_relabeling
        ),
        decoration_nodes=decoration_nodes,
        decoration_edges=decoration_edges,
    )


def mechanism_seed_requests(
    *,
    release_id: str,
    partition: str,
    family: str,
    parent_ordinal: int,
    transformation_ordinal: int = 0,
) -> tuple[SeedRequest, SeedRequest]:
    """Derive the only two seed requests valid for a mechanism parameter row."""

    release_id = _require_string(release_id, "release_id")
    if type(partition) is not str or partition not in MECHANISM_PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(MECHANISM_PARTITIONS)}")
    if type(family) is not str or family not in MECHANISM_FAMILIES:
        raise ValueError(f"family must be one of {list(MECHANISM_FAMILIES)}")
    parent_ordinal = _require_nonnegative_int(parent_ordinal, "parent_ordinal")
    transformation_ordinal = _require_nonnegative_int(
        transformation_ordinal,
        "transformation_ordinal",
    )
    scientific_digest = parent_scientific_sha256(family)
    parent = SeedRequest(
        release_id=release_id,
        partition=partition,
        panel="mechanism",
        cell=family,
        task_type=None,
        candidate_index=None,
        strength_index=None,
        label_stage=None,
        purpose="problem",
        problem_sha256=None,
        state_sha256=None,
        replicate=parent_ordinal,
    )
    transformation = SeedRequest(
        release_id=release_id,
        partition=partition,
        panel="mechanism",
        cell=family,
        task_type=None,
        candidate_index=None,
        strength_index=None,
        label_stage=None,
        purpose="mechanism",
        problem_sha256=scientific_digest,
        state_sha256=scientific_digest,
        replicate=transformation_ordinal,
    )
    return parent, transformation


_PLAN_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "parameter_row",
        "parameter_row_sha256",
        "parent_seed_request",
        "transformation_seed_request",
        "seed_registry_terminal_root_sha256",
        "authoritative_isomorphism_manifest_sha256",
    }
)


def make_mechanism_plan_document(
    *,
    row: MechanismParameterRow,
    parent_seed_request: SeedRequest,
    transformation_seed_request: SeedRequest,
    seed_registry_terminal_root_sha256: str,
    authoritative_isomorphism_manifest_sha256: str | None,
) -> dict[str, object]:
    """Serialize a complete pre-outcome plan suitable for independent commitment."""

    if type(row) is not MechanismParameterRow:
        raise TypeError("row must be an exact MechanismParameterRow")
    if type(parent_seed_request) is not SeedRequest or type(
        transformation_seed_request
    ) is not SeedRequest:
        raise TypeError("plan seed requests must be exact SeedRequest values")
    root = _require_sha256(
        seed_registry_terminal_root_sha256,
        "seed_registry_terminal_root_sha256",
    )
    manifest = (
        None
        if authoritative_isomorphism_manifest_sha256 is None
        else _require_sha256(
            authoritative_isomorphism_manifest_sha256,
            "authoritative_isomorphism_manifest_sha256",
        )
    )
    return {
        "schema": MECHANISM_PLAN_SCHEMA,
        "schema_version": MECHANISM_PLAN_SCHEMA_VERSION,
        "parameter_row": row.to_dict(),
        "parameter_row_sha256": row.sha256,
        "parent_seed_request": parent_seed_request.to_dict(),
        "transformation_seed_request": transformation_seed_request.to_dict(),
        "seed_registry_terminal_root_sha256": root,
        "authoritative_isomorphism_manifest_sha256": manifest,
    }


_PLAN_VERIFICATION_SEAL = object()


@dataclass(frozen=True, slots=True)
class VerifiedMechanismPlan:
    """A plan bound to external plan/root commitments and a verified seed registry."""

    row: MechanismParameterRow
    parent_seed_request: SeedRequest
    transformation_seed_request: SeedRequest
    seed_registry_terminal_root_sha256: str
    authoritative_isomorphism_manifest_sha256: str | None
    plan_sha256: str
    canonical_plan_bytes: bytes
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _PLAN_VERIFICATION_SEAL:
            raise TypeError("VerifiedMechanismPlan must come from verify_mechanism_plan")
        if type(self.row) is not MechanismParameterRow:
            raise TypeError("verified plan row must be an exact MechanismParameterRow")
        if type(self.parent_seed_request) is not SeedRequest or type(
            self.transformation_seed_request
        ) is not SeedRequest:
            raise TypeError("verified plan requests must be exact SeedRequest values")
        root = _require_sha256(
            self.seed_registry_terminal_root_sha256,
            "seed_registry_terminal_root_sha256",
        )
        plan_digest = _require_sha256(self.plan_sha256, "plan_sha256")
        if type(self.canonical_plan_bytes) is not bytes:
            raise TypeError("canonical_plan_bytes must be exact bytes")
        if hashlib.sha256(self.canonical_plan_bytes).hexdigest() != plan_digest:
            raise ValueError("verified plan bytes do not match plan_sha256")
        manifest = self.authoritative_isomorphism_manifest_sha256
        if manifest is not None:
            manifest = _require_sha256(manifest, "authoritative isomorphism manifest SHA-256")
        object.__setattr__(self, "seed_registry_terminal_root_sha256", root)
        object.__setattr__(self, "authoritative_isomorphism_manifest_sha256", manifest)
        object.__setattr__(self, "plan_sha256", plan_digest)


def verify_mechanism_plan(
    value: object,
    *,
    expected_plan_sha256: str,
    resolver: VerifiedSeedResolver,
    expected_seed_registry_terminal_root_sha256: str,
) -> VerifiedMechanismPlan:
    """Verify plan bytes, exact schema, contextual seed requests, and sealed resolution."""

    expected_plan = _require_sha256(expected_plan_sha256, "expected plan SHA-256")
    expected_root = _require_sha256(
        expected_seed_registry_terminal_root_sha256,
        "expected seed-registry terminal root SHA-256",
    )
    if type(resolver) is not VerifiedSeedResolver:
        raise TypeError("resolver must be a VerifiedSeedResolver")
    document = require_exact_keys(value, _PLAN_FIELDS, "mechanism generation plan")
    _require_schema(
        document,
        MECHANISM_PLAN_SCHEMA,
        MECHANISM_PLAN_SCHEMA_VERSION,
        "mechanism generation plan",
    )
    plan_bytes = canonical_bytes(document)
    if hashlib.sha256(plan_bytes).hexdigest() != expected_plan:
        raise ValueError("mechanism plan SHA-256 does not match independent commitment")
    row = MechanismParameterRow.from_dict(document["parameter_row"])
    row_digest = _require_sha256(document["parameter_row_sha256"], "parameter_row_sha256")
    if row_digest != row.sha256:
        raise ValueError("mechanism plan parameter-row digest mismatch")
    parent_request = SeedRequest.from_dict(document["parent_seed_request"])
    transformation_request = SeedRequest.from_dict(document["transformation_seed_request"])
    expected_parent, expected_transformation = mechanism_seed_requests(
        release_id=row.release_id,
        partition=row.partition,
        family=row.family,
        parent_ordinal=row.parent_ordinal,
        transformation_ordinal=row.transformation_ordinal,
    )
    if parent_request != expected_parent:
        raise ValueError("parent seed request is not the canonical request for parameter row")
    if transformation_request != expected_transformation:
        raise ValueError(
            "transformation seed request is not the canonical request for parameter row"
        )
    if row.parent_seed_key != parent_request.seed_key_hex:
        raise ValueError("parameter row is not bound to its registered parent seed request")
    if row.transformation_seed_key != transformation_request.seed_key_hex:
        raise ValueError(
            "parameter row is not bound to its registered transformation seed request"
        )
    root = _require_sha256(
        document["seed_registry_terminal_root_sha256"],
        "seed_registry_terminal_root_sha256",
    )
    if root != expected_root:
        raise ValueError("mechanism plan seed root does not match independent commitment")
    resolved = resolver.resolve_many((parent_request, transformation_request))
    if tuple(entry.request for entry in resolved) != (parent_request, transformation_request):
        raise ValueError("verified seed resolver returned a contextual request mismatch")
    if tuple(entry.seed_key_sha256 for entry in resolved) != (
        row.parent_seed_key,
        row.transformation_seed_key,
    ):
        raise ValueError("verified seed resolver returned a key mismatch")
    manifest_value = document["authoritative_isomorphism_manifest_sha256"]
    manifest = (
        None
        if manifest_value is None
        else _require_sha256(
            manifest_value,
            "authoritative_isomorphism_manifest_sha256",
        )
    )
    return VerifiedMechanismPlan(
        row=row,
        parent_seed_request=parent_request,
        transformation_seed_request=transformation_request,
        seed_registry_terminal_root_sha256=root,
        authoritative_isomorphism_manifest_sha256=manifest,
        plan_sha256=expected_plan,
        canonical_plan_bytes=plan_bytes,
        _seal=_PLAN_VERIFICATION_SEAL,
    )


@dataclass(frozen=True, slots=True)
class _MaterializedState:
    host_nodes: tuple[int, ...]
    host_edges: tuple[Edge, ...]
    logical_nodes: tuple[int, ...]
    logical_edges: tuple[tuple[int, int], ...]
    cores: tuple[Chain, ...]
    actions: tuple[Action, ...]
    original_candidate_index: int
    hypothesized_winner_index: int
    l_cap: int
    q_cap: int | None

    def host_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.host_nodes)
        graph.add_edges_from(self.host_edges)
        return graph

    def logical_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.logical_nodes)
        graph.add_edges_from(self.logical_edges)
        return graph


def _materialize(row: MechanismParameterRow) -> _MaterializedState:
    motif = build_motif(row.family)
    host_symmetry = _automorphisms(motif.host)[row.host_automorphism_index]
    logical_symmetry = _automorphisms(motif.logical)[row.logical_symmetry_index]
    host_labels = dict(row.host_node_relabeling)
    logical_labels = dict(row.logical_role_relabeling)

    host_mapping = {
        source: host_labels[host_symmetry[source]] for source in sorted(motif.host.nodes)
    }
    logical_mapping = {
        source: logical_labels[logical_symmetry[source]] for source in sorted(motif.logical.nodes)
    }

    host_nodes = tuple(sorted((*host_mapping.values(), *row.decoration_nodes)))
    base_host_edges = {
        tuple(sorted((host_mapping[left], host_mapping[right]))) for left, right in motif.host.edges
    }
    if base_host_edges & set(row.decoration_edges):
        raise ValueError("decoration_edges must not duplicate a transformed template edge")
    host_edges = tuple(sorted(base_host_edges | set(row.decoration_edges)))
    logical_nodes = tuple(sorted(logical_mapping.values()))
    logical_edges = tuple(
        sorted(
            tuple(sorted((logical_mapping[left], logical_mapping[right])))
            for left, right in motif.logical.edges
        )
    )
    cores = tuple(
        sorted(
            (
                logical_mapping[logical],
                tuple(sorted(host_mapping[node] for node in nodes)),
            )
            for logical, nodes in motif.cores.items()
        )
    )

    transformed_actions = tuple(
        (logical_mapping[logical], host_mapping[node]) for logical, node in motif.actions
    )
    transformed_original = transformed_actions[0]
    transformed_hypothesis = (
        logical_mapping[motif.hypothesis_winner[0]],
        host_mapping[motif.hypothesis_winner[1]],
    )
    actions = (
        transformed_actions
        if row.orientation == "forward"
        else tuple(reversed(transformed_actions))
    )
    return _MaterializedState(
        host_nodes=host_nodes,
        host_edges=cast(tuple[Edge, ...], host_edges),
        logical_nodes=logical_nodes,
        logical_edges=cast(tuple[tuple[int, int], ...], logical_edges),
        cores=cores,
        actions=actions,
        original_candidate_index=actions.index(transformed_original),
        hypothesized_winner_index=actions.index(transformed_hypothesis),
        l_cap=motif.l_cap,
        q_cap=motif.q_cap,
    )


def _state_identity_payload(state: _MaterializedState) -> dict[str, object]:
    candidate_roles = sorted(
        (
            {
                "action": list(action),
                "original_candidate": index == state.original_candidate_index,
            }
            for index, action in enumerate(state.actions)
        ),
        key=lambda entry: (
            cast(list[int], entry["action"])[0],
            cast(list[int], entry["action"])[1],
            not cast(bool, entry["original_candidate"]),
        ),
    )
    return {
        "schema": "embedbench.mechanism-state-identity",
        "schema_version": 1,
        "host_nodes": list(state.host_nodes),
        "host_edges": [list(edge) for edge in state.host_edges],
        "logical_nodes": list(state.logical_nodes),
        "logical_edges": [list(edge) for edge in state.logical_edges],
        "cores": [[logical, list(nodes)] for logical, nodes in state.cores],
        "window_nodes": list(state.host_nodes),
        "candidate_roles": candidate_roles,
        "l_cap": state.l_cap,
        "q_cap": state.q_cap,
    }


def _state_sha256(state: _MaterializedState) -> str:
    return hashlib.sha256(canonical_bytes(_state_identity_payload(state))).hexdigest()


def _canonical_colored_graph_bytes(
    colors: Sequence[bytes], adjacency: Sequence[Collection[int]]
) -> bytes:
    """Exact individualization/refinement canonical form for these small colored motifs.

    This is intentionally an exact canonicalizer, not a Weisfeiler-Lehman digest.  The
    schema-v1 decoration cap keeps the only potentially symmetric color class small enough
    for exhaustive individualization.  A future large-host implementation must replace this
    with the release-pinned Traces binary and therefore requires a schema bump.
    """

    vertex_count = len(colors)
    if len(adjacency) != vertex_count:
        raise ValueError("colored graph adjacency length mismatch")
    adjacency_sets = tuple(frozenset(neighbors) for neighbors in adjacency)
    if any(
        neighbor < 0 or neighbor >= vertex_count
        for neighbors in adjacency_sets
        for neighbor in neighbors
    ):
        raise ValueError("colored graph has an invalid neighbor index")
    if any(vertex in adjacency_sets[vertex] for vertex in range(vertex_count)):
        raise ValueError("colored graph must not contain self-loops")
    if any(
        vertex not in adjacency_sets[neighbor]
        for vertex, neighbors in enumerate(adjacency_sets)
        for neighbor in neighbors
    ):
        raise ValueError("colored graph adjacency must be symmetric")

    palette = {value: index for index, value in enumerate(sorted(set(colors)))}
    initial = tuple(palette[value] for value in colors)

    def refine(current: tuple[int, ...]) -> tuple[int, ...]:
        while True:
            signatures = tuple(
                (
                    current[vertex],
                    tuple(sorted(current[neighbor] for neighbor in adjacency_sets[vertex])),
                )
                for vertex in range(vertex_count)
            )
            signature_palette = {
                signature: index for index, signature in enumerate(sorted(set(signatures)))
            }
            updated = tuple(signature_palette[signature] for signature in signatures)
            if updated == current:
                return current
            current = updated

    memo: dict[tuple[int, ...], bytes] = {}

    def search(current: tuple[int, ...]) -> bytes:
        current = refine(current)
        cached = memo.get(current)
        if cached is not None:
            return cached
        cells: dict[int, list[int]] = defaultdict(list)
        for vertex, color in enumerate(current):
            cells[color].append(vertex)
        non_singletons = [
            (len(vertices), color, tuple(vertices))
            for color, vertices in cells.items()
            if len(vertices) > 1
        ]
        if not non_singletons:
            order = tuple(
                vertex for vertex, _ in sorted(enumerate(current), key=lambda item: item[1])
            )
            inverse = {vertex: index for index, vertex in enumerate(order)}
            payload = {
                "colors": [colors[vertex].hex() for vertex in order],
                "edges": [
                    list(edge)
                    for edge in sorted(
                        (inverse[left], inverse[right])
                        for left in order
                        for right in adjacency_sets[left]
                        if inverse[left] < inverse[right]
                    )
                ],
            }
            result = canonical_bytes(payload)
            memo[current] = result
            return result

        _, _, target_cell = min(non_singletons)
        individualized_color = max(current, default=-1) + 1
        best: bytes | None = None
        for vertex in target_cell:
            child = list(current)
            child[vertex] = individualized_color
            candidate = search(tuple(child))
            if best is None or candidate < best:
                best = candidate
        if best is None:  # pragma: no cover - target_cell is known non-empty
            raise RuntimeError("canonicalization did not examine a branch")
        memo[current] = best
        return best

    return search(initial)


def _mechanism_incidence(
    state: _MaterializedState,
) -> tuple[tuple[bytes, ...], tuple[frozenset[int], ...]]:
    colors: list[bytes] = []
    adjacency: list[set[int]] = []

    def vertex(color: str, attributes: object | None = None) -> int:
        encoded = color.encode("ascii")
        if attributes is not None:
            encoded += b":" + canonical_bytes(attributes)
        index = len(colors)
        colors.append(encoded)
        adjacency.append(set())
        return index

    def connect(left: int, right: int) -> None:
        adjacency[left].add(right)
        adjacency[right].add(left)

    global_vertex = vertex("global", {"l_cap": state.l_cap, "q_cap": state.q_cap})
    host_vertices = {node: vertex("host") for node in state.host_nodes}
    logical_vertices = {node: vertex("logical") for node in state.logical_nodes}
    # The global vertex makes graph-wide numeric caps part of every connected component.
    for host_vertex in host_vertices.values():
        membership = vertex("window_membership")
        connect(global_vertex, membership)
        connect(membership, host_vertex)
    for left, right in state.host_edges:
        relation = vertex("host_edge")
        connect(relation, host_vertices[left])
        connect(relation, host_vertices[right])
    for left, right in state.logical_edges:
        relation = vertex("logical_edge")
        connect(relation, logical_vertices[left])
        connect(relation, logical_vertices[right])
    for logical, nodes in state.cores:
        for node in nodes:
            relation = vertex("frozen_chain_member")
            connect(relation, logical_vertices[logical])
            connect(relation, host_vertices[node])
    for index, (logical, node) in enumerate(state.actions):
        relation = vertex(
            "candidate",
            {"original_candidate": index == state.original_candidate_index},
        )
        connect(relation, logical_vertices[logical])
        connect(relation, host_vertices[node])
    return tuple(colors), tuple(frozenset(neighbors) for neighbors in adjacency)


@lru_cache(maxsize=4_096)
def _diagnostic_iso_sha256(state: _MaterializedState) -> str:
    """Return a test-only canonical digest with no release-validity claim."""

    colors, adjacency = _mechanism_incidence(state)
    canonical = _canonical_colored_graph_bytes(colors, adjacency)
    preimage = (
        len(b"embedbench.diagnostic-mechanism-isomorphism-v1").to_bytes(8, "big")
        + b"embedbench.diagnostic-mechanism-isomorphism-v1"
        + len(canonical).to_bytes(8, "big")
        + canonical
    )
    return hashlib.sha256(preimage).hexdigest()


_EXACT_CANDIDATE_FIELDS = frozenset({"action", "outcome", "completion"})


@dataclass(frozen=True, slots=True)
class ExactCandidateRecord:
    """Exact terminal outcome and one attaining completion for a candidate action."""

    action: Action
    outcome: Outcome
    completion: tuple[Chain, ...] | None

    def __post_init__(self) -> None:
        action = _canonical_action(self.action, "candidate action")
        if type(self.outcome) is not tuple or len(self.outcome) != 3:
            raise TypeError("candidate outcome must be an immutable three-integer tuple")
        outcome = cast(
            Outcome,
            tuple(
                _require_int(item, f"candidate outcome[{index}]")
                for index, item in enumerate(self.outcome)
            ),
        )
        if outcome[0] not in {0, 1}:
            raise ValueError("candidate outcome feasibility must be zero or one")
        if outcome[0] == 0 and outcome != INFEASIBLE:
            raise ValueError(
                "an infeasible exact outcome must equal the canonical INFEASIBLE tuple"
            )
        if outcome[0] == 1 and (outcome[1] >= 0 or outcome[2] >= 0):
            raise ValueError("a feasible exact outcome must contain negative Q and L_max")
        if self.completion is None:
            if outcome != INFEASIBLE:
                raise ValueError("a feasible outcome requires an attaining completion")
            completion = None
        else:
            completion = _canonical_chains(self.completion, "candidate completion")
            if outcome == INFEASIBLE:
                raise ValueError("an infeasible outcome cannot carry a completion")
            qubits = sum(len(nodes) for _, nodes in completion)
            maximum = max((len(nodes) for _, nodes in completion), default=0)
            if outcome != (1, -qubits, -maximum):
                raise ValueError("candidate outcome disagrees with completion resources")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "completion", completion)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": list(self.action),
            "outcome": list(self.outcome),
            "completion": (
                None
                if self.completion is None
                else [[logical, list(nodes)] for logical, nodes in self.completion]
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> ExactCandidateRecord:
        document = require_exact_keys(value, _EXACT_CANDIDATE_FIELDS, "exact candidate record")
        action_values = _json_tuple(document["action"], "candidate action")
        outcome_values = _json_tuple(document["outcome"], "candidate outcome")
        completion_value = document["completion"]
        completion: tuple[Chain, ...] | None
        if completion_value is None:
            completion = None
        else:
            raw_chains = _json_tuple(completion_value, "candidate completion")
            parsed_chains: list[Chain] = []
            for index, raw_chain in enumerate(raw_chains):
                pair = _json_tuple(raw_chain, f"candidate completion[{index}]")
                if len(pair) != 2:
                    raise ValueError("candidate completion entry must have two values")
                parsed_chains.append(
                    (
                        cast(int, pair[0]),
                        cast(tuple[int, ...], _json_tuple(pair[1], "completion chain nodes")),
                    )
                )
            completion = tuple(parsed_chains)
        return cls(
            action=cast(Action, action_values),
            outcome=cast(Outcome, outcome_values),
            completion=completion,
        )


def _deterministic_ranking(candidates: Sequence[ExactCandidateRecord]) -> tuple[int, ...]:
    """Rank by exact value, then by canonical action rather than input enumeration order."""

    return tuple(
        sorted(
            range(len(candidates)),
            key=lambda index: (
                -candidates[index].outcome[0],
                -candidates[index].outcome[1],
                -candidates[index].outcome[2],
                candidates[index].action,
            ),
        )
    )


_CERTIFICATE_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "family",
        "state_sha256",
        "search_method",
        "search_node_budget",
        "hypothesized_winner_index",
        "certified_winner_index",
        "margin_kind",
        "margin_value",
        "candidates",
        "certificate_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class MechanismExactCertificate:
    """Replayable exact counterfactual certificate for all offered actions."""

    SCHEMA: ClassVar[str] = MECHANISM_CERTIFICATE_SCHEMA
    SCHEMA_VERSION: ClassVar[int] = MECHANISM_CERTIFICATE_SCHEMA_VERSION
    SEARCH_METHOD: ClassVar[str] = "embedbench.handtests.certify_motif/exhaustive"

    family: str
    state_sha256: str
    hypothesized_winner_index: int
    certified_winner_index: int
    margin_kind: str
    margin_value: int
    candidates: tuple[ExactCandidateRecord, ...]

    def __post_init__(self) -> None:
        if type(self.family) is not str or self.family not in MECHANISM_FAMILIES:
            raise ValueError(f"certificate family must be one of {list(MECHANISM_FAMILIES)}")
        state_digest = _require_sha256(self.state_sha256, "certificate state_sha256")
        if type(self.candidates) is not tuple or not all(
            type(candidate) is ExactCandidateRecord for candidate in self.candidates
        ):
            raise TypeError(
                "certificate candidates must be an immutable ExactCandidateRecord tuple"
            )
        if len(self.candidates) < 2:
            raise ValueError("certificate must cover at least two candidate actions")
        actions = tuple(candidate.action for candidate in self.candidates)
        if len(actions) != len(set(actions)):
            raise ValueError("certificate candidate actions must be duplicate-free")
        hypothesis_index = _require_nonnegative_int(
            self.hypothesized_winner_index, "hypothesized_winner_index"
        )
        certified_index = _require_nonnegative_int(
            self.certified_winner_index, "certified_winner_index"
        )
        if hypothesis_index >= len(self.candidates) or certified_index >= len(self.candidates):
            raise ValueError("certificate winner index is outside the candidate bank")
        ranking = _deterministic_ranking(self.candidates)
        if certified_index != ranking[0]:
            raise ValueError("certified_winner_index disagrees with exact candidate outcomes")
        expected_kind, expected_value = compare(
            self.candidates[ranking[0]].outcome,
            self.candidates[ranking[1]].outcome,
        )
        if type(self.margin_kind) is not str or self.margin_kind != expected_kind:
            raise ValueError("certificate margin_kind disagrees with exact candidate outcomes")
        margin_value = _require_nonnegative_int(self.margin_value, "margin_value")
        if margin_value != expected_value:
            raise ValueError("certificate margin_value disagrees with exact candidate outcomes")
        object.__setattr__(self, "state_sha256", state_digest)
        object.__setattr__(self, "hypothesized_winner_index", hypothesis_index)
        object.__setattr__(self, "certified_winner_index", certified_index)
        object.__setattr__(self, "margin_value", margin_value)

    def _digest_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "family": self.family,
            "state_sha256": self.state_sha256,
            "search_method": self.SEARCH_METHOD,
            "search_node_budget": None,
            "hypothesized_winner_index": self.hypothesized_winner_index,
            "certified_winner_index": self.certified_winner_index,
            "margin_kind": self.margin_kind,
            "margin_value": self.margin_value,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @property
    def certificate_sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self._digest_payload())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self._digest_payload(), "certificate_sha256": self.certificate_sha256}

    @classmethod
    def from_dict(cls, value: object) -> MechanismExactCertificate:
        document = require_exact_keys(value, _CERTIFICATE_FIELDS, "mechanism exact certificate")
        _require_schema(document, cls.SCHEMA, cls.SCHEMA_VERSION, "mechanism exact certificate")
        if document["search_method"] != cls.SEARCH_METHOD:
            raise ValueError("mechanism exact certificate has an unregistered search_method")
        if document["search_node_budget"] is not None:
            raise ValueError("schema-v1 motif certification requires untruncated exhaustive search")
        candidates = tuple(
            ExactCandidateRecord.from_dict(candidate)
            for candidate in _json_tuple(document["candidates"], "certificate candidates")
        )
        certificate = cls(
            family=document["family"],
            state_sha256=document["state_sha256"],
            hypothesized_winner_index=document["hypothesized_winner_index"],
            certified_winner_index=document["certified_winner_index"],
            margin_kind=document["margin_kind"],
            margin_value=document["margin_value"],
            candidates=candidates,
        )
        claimed = _require_sha256(document["certificate_sha256"], "certificate_sha256")
        if claimed != certificate.certificate_sha256:
            raise ValueError("mechanism exact certificate digest mismatch")
        return certificate


_VARIANT_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "parameter_row",
        "parameter_row_sha256",
        "host_nodes",
        "host_edges",
        "logical_nodes",
        "logical_edges",
        "cores",
        "actions",
        "original_candidate_index",
        "hypothesized_winner_index",
        "l_cap",
        "q_cap",
        "state_sha256",
        "diagnostic_iso_sha256",
        "certificate",
    }
)


@dataclass(frozen=True, slots=True)
class MechanismVariant:
    """Immutable accepted state plus its independently replayable exact certificate."""

    SCHEMA: ClassVar[str] = MECHANISM_VARIANT_SCHEMA
    SCHEMA_VERSION: ClassVar[int] = MECHANISM_VARIANT_SCHEMA_VERSION

    parameter_row: MechanismParameterRow
    host_nodes: tuple[int, ...]
    host_edges: tuple[Edge, ...]
    logical_nodes: tuple[int, ...]
    logical_edges: tuple[tuple[int, int], ...]
    cores: tuple[Chain, ...]
    actions: tuple[Action, ...]
    original_candidate_index: int
    hypothesized_winner_index: int
    l_cap: int
    q_cap: int | None
    certificate: MechanismExactCertificate

    def __post_init__(self) -> None:
        if type(self.parameter_row) is not MechanismParameterRow:
            raise TypeError("parameter_row must be an exact MechanismParameterRow")
        if self.parameter_row.placement_mode != "template_subgraph":
            raise ValueError("an accepted variant requires template_subgraph placement")
        host_nodes = _canonical_nodes(self.host_nodes, "host_nodes", allow_empty=False)
        host_edges = _canonical_edges(self.host_edges, "host_edges")
        if any(left not in host_nodes or right not in host_nodes for left, right in host_edges):
            raise ValueError("host edge endpoint is absent from host_nodes")
        logical_nodes = _canonical_logical_nodes(self.logical_nodes, "logical_nodes")
        if not logical_nodes:
            raise ValueError("logical_nodes must not be empty")
        logical_edges = _canonical_logical_edges(self.logical_edges, "logical_edges")
        if any(
            left not in logical_nodes or right not in logical_nodes for left, right in logical_edges
        ):
            raise ValueError("logical edge endpoint is absent from logical_nodes")
        cores = _canonical_chains(
            self.cores,
            "cores",
            logical_nodes=logical_nodes,
            host_nodes=host_nodes,
        )
        actions = _canonical_actions(self.actions, "actions")
        if any(logical not in logical_nodes or node not in host_nodes for logical, node in actions):
            raise ValueError("candidate action leaves the logical or realized-host graph")
        occupied = {node for _, nodes in cores for node in nodes}
        if any(node in occupied for _, node in actions):
            raise ValueError("candidate action uses a node occupied by the frozen context")
        original_index = _require_nonnegative_int(
            self.original_candidate_index, "original_candidate_index"
        )
        hypothesis_index = _require_nonnegative_int(
            self.hypothesized_winner_index, "hypothesized_winner_index"
        )
        if original_index >= len(actions) or hypothesis_index >= len(actions):
            raise ValueError("candidate role index is outside the action bank")
        l_cap = _require_positive_int(self.l_cap, "l_cap")
        q_cap = None if self.q_cap is None else _require_nonnegative_int(self.q_cap, "q_cap")
        if type(self.certificate) is not MechanismExactCertificate:
            raise TypeError("certificate must be an exact MechanismExactCertificate")

        object.__setattr__(self, "host_nodes", host_nodes)
        object.__setattr__(self, "host_edges", host_edges)
        object.__setattr__(self, "logical_nodes", logical_nodes)
        object.__setattr__(self, "logical_edges", logical_edges)
        object.__setattr__(self, "cores", cores)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "original_candidate_index", original_index)
        object.__setattr__(self, "hypothesized_winner_index", hypothesis_index)
        object.__setattr__(self, "l_cap", l_cap)
        object.__setattr__(self, "q_cap", q_cap)

        if self.certificate.family != self.parameter_row.family:
            raise ValueError("certificate family disagrees with parameter row")
        if tuple(candidate.action for candidate in self.certificate.candidates) != actions:
            raise ValueError("certificate candidate actions disagree with variant actions")
        if self.certificate.hypothesized_winner_index != hypothesis_index:
            raise ValueError("certificate hypothesized winner disagrees with variant")
        if self.certificate.certified_winner_index != hypothesis_index:
            raise ValueError("accepted variant's hypothesized winner is not the exact winner")
        if self.certificate.margin_value <= 0 or self.certificate.margin_kind == "none":
            raise ValueError("accepted variant requires a positive exact decision margin")
        if self.certificate.state_sha256 != self.state_sha256:
            raise ValueError("certificate state_sha256 disagrees with variant state")

    def _state(self) -> _MaterializedState:
        return _MaterializedState(
            host_nodes=self.host_nodes,
            host_edges=self.host_edges,
            logical_nodes=self.logical_nodes,
            logical_edges=self.logical_edges,
            cores=self.cores,
            actions=self.actions,
            original_candidate_index=self.original_candidate_index,
            hypothesized_winner_index=self.hypothesized_winner_index,
            l_cap=self.l_cap,
            q_cap=self.q_cap,
        )

    @property
    def state_sha256(self) -> str:
        return _state_sha256(self._state())

    @property
    def diagnostic_iso_sha256(self) -> str:
        """Bespoke small-graph digest for diagnostics, never release admission."""

        return _diagnostic_iso_sha256(self._state())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "parameter_row": self.parameter_row.to_dict(),
            "parameter_row_sha256": self.parameter_row.sha256,
            "host_nodes": list(self.host_nodes),
            "host_edges": [list(edge) for edge in self.host_edges],
            "logical_nodes": list(self.logical_nodes),
            "logical_edges": [list(edge) for edge in self.logical_edges],
            "cores": [[logical, list(nodes)] for logical, nodes in self.cores],
            "actions": [list(action) for action in self.actions],
            "original_candidate_index": self.original_candidate_index,
            "hypothesized_winner_index": self.hypothesized_winner_index,
            "l_cap": self.l_cap,
            "q_cap": self.q_cap,
            "state_sha256": self.state_sha256,
            "diagnostic_iso_sha256": self.diagnostic_iso_sha256,
            "certificate": self.certificate.to_dict(),
        }

    def to_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> MechanismVariant:
        document = require_exact_keys(value, _VARIANT_FIELDS, "mechanism variant")
        _require_schema(document, cls.SCHEMA, cls.SCHEMA_VERSION, "mechanism variant")
        row = MechanismParameterRow.from_dict(document["parameter_row"])
        claimed_parameter_digest = _require_sha256(
            document["parameter_row_sha256"], "parameter_row_sha256"
        )
        if claimed_parameter_digest != row.sha256:
            raise ValueError("mechanism variant parameter-row digest mismatch")

        def pairs(field: str) -> tuple[tuple[object, object], ...]:
            rows = _json_tuple(document[field], field)
            result: list[tuple[object, object]] = []
            for index, raw in enumerate(rows):
                pair = _json_tuple(raw, f"{field}[{index}]")
                if len(pair) != 2:
                    raise ValueError(f"{field}[{index}] must contain two values")
                result.append((pair[0], pair[1]))
            return tuple(result)

        parsed_cores: list[Chain] = []
        for index, raw in enumerate(_json_tuple(document["cores"], "cores")):
            pair = _json_tuple(raw, f"cores[{index}]")
            if len(pair) != 2:
                raise ValueError(f"cores[{index}] must contain two values")
            parsed_cores.append(
                (
                    cast(int, pair[0]),
                    cast(tuple[int, ...], _json_tuple(pair[1], f"cores[{index}][1]")),
                )
            )
        variant = cls(
            parameter_row=row,
            host_nodes=cast(tuple[int, ...], _json_tuple(document["host_nodes"], "host_nodes")),
            host_edges=cast(tuple[Edge, ...], pairs("host_edges")),
            logical_nodes=cast(
                tuple[int, ...], _json_tuple(document["logical_nodes"], "logical_nodes")
            ),
            logical_edges=cast(tuple[tuple[int, int], ...], pairs("logical_edges")),
            cores=tuple(parsed_cores),
            actions=cast(tuple[Action, ...], pairs("actions")),
            original_candidate_index=document["original_candidate_index"],
            hypothesized_winner_index=document["hypothesized_winner_index"],
            l_cap=document["l_cap"],
            q_cap=document["q_cap"],
            certificate=MechanismExactCertificate.from_dict(document["certificate"]),
        )
        claimed_state = _require_sha256(document["state_sha256"], "state_sha256")
        claimed_iso = _require_sha256(
            document["diagnostic_iso_sha256"],
            "diagnostic_iso_sha256",
        )
        if claimed_state != variant.state_sha256:
            raise ValueError("mechanism variant state_sha256 mismatch")
        if claimed_iso != variant.diagnostic_iso_sha256:
            raise ValueError("mechanism variant diagnostic_iso_sha256 mismatch")
        replay = generate_single_mechanism_variant(row)
        if replay.status != "accepted" or replay.variant is None:
            raise ValueError(
                "mechanism variant parameter row does not replay to an accepted exact state"
            )
        if replay.variant.to_bytes() != variant.to_bytes():
            raise ValueError("transformation metadata does not reconstruct variant byte-for-byte")
        return variant


_RESULT_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "parameter_row_sha256",
        "constituent_families",
        "variant",
    }
)


@dataclass(frozen=True, slots=True)
class MechanismGenerationResult:
    """Closed outcome for an accepted, rejected, or intentionally unsupported attempt."""

    SCHEMA: ClassVar[str] = MECHANISM_RESULT_SCHEMA
    SCHEMA_VERSION: ClassVar[int] = MECHANISM_RESULT_SCHEMA_VERSION

    status: GenerationStatus
    parameter_row_sha256: str
    constituent_families: tuple[str, ...]
    variant: MechanismVariant | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _require_generation_status(self.status))
        parameter_digest = _require_sha256(self.parameter_row_sha256, "parameter_row_sha256")
        if type(self.constituent_families) is not tuple:
            raise TypeError("constituent_families must be an immutable tuple")
        families = tuple(self.constituent_families)
        if not families or any(
            type(family) is not str or family not in MECHANISM_FAMILIES for family in families
        ):
            raise ValueError("constituent_families must contain registered mechanism families")
        if self.status == "unsupported_composed_mechanism":
            if len(families) not in {2, 3}:
                raise ValueError(
                    "unsupported composed status requires exactly two or three families"
                )
        elif len(families) != 1:
            raise ValueError("single-mechanism result must name exactly one constituent family")
        if self.status == "accepted":
            if type(self.variant) is not MechanismVariant:
                raise ValueError("accepted result requires an exact MechanismVariant")
            if self.variant.parameter_row.sha256 != parameter_digest:
                raise ValueError("accepted result does not match parameter_row_sha256")
            if families != (self.variant.parameter_row.family,):
                raise ValueError("accepted result family does not match its variant")
        elif self.variant is not None:
            raise ValueError("non-accepted result must not carry a variant")
        object.__setattr__(self, "parameter_row_sha256", parameter_digest)
        object.__setattr__(self, "constituent_families", families)

    def require_variant(self) -> MechanismVariant:
        if self.status != "accepted" or self.variant is None:
            raise ValueError(f"mechanism generation did not produce a variant: {self.status}")
        return self.variant

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "status": self.status,
            "parameter_row_sha256": self.parameter_row_sha256,
            "constituent_families": list(self.constituent_families),
            "variant": None if self.variant is None else self.variant.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> MechanismGenerationResult:
        document = require_exact_keys(value, _RESULT_FIELDS, "mechanism generation result")
        _require_schema(document, cls.SCHEMA, cls.SCHEMA_VERSION, "mechanism generation result")
        variant_value = document["variant"]
        return cls(
            status=document["status"],
            parameter_row_sha256=document["parameter_row_sha256"],
            constituent_families=cast(
                tuple[str, ...],
                _json_tuple(document["constituent_families"], "constituent_families"),
            ),
            variant=None if variant_value is None else validate_variant(variant_value),
        )


def _result(
    row: MechanismParameterRow,
    status: GenerationStatus,
    variant: MechanismVariant | None = None,
) -> MechanismGenerationResult:
    return MechanismGenerationResult(
        status=status,
        parameter_row_sha256=row.sha256,
        constituent_families=(row.family,),
        variant=variant,
    )


def _completion_record(
    completion: Mapping[int, Collection[int]] | None,
) -> tuple[Chain, ...] | None:
    if completion is None:
        return None
    return tuple((logical, tuple(sorted(nodes))) for logical, nodes in sorted(completion.items()))


def _deterministic_optimal_completion(
    state: _MaterializedState,
    action: Action,
    outcome: Outcome,
) -> tuple[Chain, ...] | None:
    """Select the lexicographically least attaining witness from exhaustive completions.

    :mod:`embedbench.exact` intentionally exposes one optimum found by its branch-and-bound
    traversal.  Set iteration is not a release-level tie-break, so mechanism certificates
    independently enumerate the tiny hand-test state and choose canonical minimum bytes.
    """

    if outcome == INFEASIBLE:
        return None
    cores = {logical: frozenset(nodes) for logical, nodes in state.cores}
    logical, node = action
    existing = cores.get(logical, frozenset())
    cores[logical] = existing | {node}
    completions = all_completions(
        state.host_graph(),
        state.logical_graph(),
        cores,
        l_cap=state.l_cap,
        q_cap=state.q_cap,
    )
    attaining: list[tuple[Chain, ...]] = []
    for completion in completions:
        record = _completion_record(completion)
        if record is None:  # pragma: no cover - all_completions never returns None
            continue
        qubits = sum(len(nodes) for _, nodes in record)
        maximum = max((len(nodes) for _, nodes in record), default=0)
        if outcome == (1, -qubits, -maximum):
            attaining.append(record)
    if not attaining:
        raise ValueError("exact outcome has no independently enumerated attaining completion")
    return min(attaining, key=lambda record: canonical_bytes(
        [[logical, list(nodes)] for logical, nodes in record]
    ))


def _multi_neighbour_causality_holds(
    state: _MaterializedState,
    certificate: MechanismExactCertificate,
) -> bool:
    """Require each single-neighbour ablation to strictly reverse the winner."""

    action_variable = state.actions[certificate.certified_winner_index][0]
    frozen = {logical: frozenset(nodes) for logical, nodes in state.cores}
    neighbours = tuple(
        sorted(node for node in state.logical_graph().neighbors(action_variable) if node in frozen)
    )
    if len(neighbours) != 2:
        return False
    winner_action = state.actions[certificate.certified_winner_index]
    for removed in neighbours:
        logical = state.logical_graph()
        logical.remove_node(removed)
        ablated_cores = {node: nodes for node, nodes in frozen.items() if node != removed}
        try:
            ablated = certify_motif(
                Motif(
                    name="multi_neighbour",
                    mechanism="registered neighbour-ablation check",
                    host=state.host_graph(),
                    logical=logical,
                    cores=ablated_cores,
                    actions=state.actions,
                    hypothesis_winner=winner_action,
                    l_cap=state.l_cap,
                    q_cap=state.q_cap,
                )
            )
        except (SearchAborted, ValueError):
            return False
        values = ablated.sample.values
        winning_value = values[state.actions.index(winner_action)]
        alternatives = tuple(
            value for index, value in enumerate(values) if state.actions[index] != winner_action
        )
        if not alternatives or max(alternatives) <= winning_value:
            return False
    return True


def _mechanism_semantics_certified(
    state: _MaterializedState,
    certificate: MechanismExactCertificate,
) -> bool:
    if certificate.family == "short_chain_trap":
        return certificate.margin_kind == "max_chain"
    if certificate.family == "multi_neighbour":
        return (
            certificate.margin_kind == "qubits"
            and _multi_neighbour_causality_holds(state, certificate)
        )
    return True


def generate_single_mechanism_variant(
    row: MechanismParameterRow,
) -> MechanismGenerationResult:
    """Materialize and independently certify one explicit single-mechanism row.

    Schema errors raise because the precommit itself is malformed.  Scientific attrition
    returns a typed result so it remains countable and cannot be mistaken for an accepted
    example.
    """

    if type(row) is not MechanismParameterRow:
        raise TypeError("row must be an exact MechanismParameterRow")
    if row.placement_mode != "template_subgraph":
        return _result(row, "unsupported_generic_hardware_placement")

    state = _materialize(row)
    host = state.host_graph()
    logical = state.logical_graph()
    cores = {logical_node: frozenset(nodes) for logical_node, nodes in state.cores}
    hypothesis = state.actions[state.hypothesized_winner_index]
    transformed_motif = Motif(
        name=row.family,
        mechanism=build_motif(row.family).mechanism,
        host=host,
        logical=logical,
        cores=cores,
        actions=state.actions,
        hypothesis_winner=hypothesis,
        l_cap=state.l_cap,
        q_cap=state.q_cap,
    )
    try:
        exact = certify_motif(transformed_motif)
    except SearchAborted:
        return _result(row, "exact_certification_failed")

    state_digest = _state_sha256(state)
    try:
        candidates = tuple(
            ExactCandidateRecord(
                action=action,
                outcome=outcome,
                completion=_deterministic_optimal_completion(state, action, outcome),
            )
            for action, outcome in zip(
                exact.sample.actions,
                exact.sample.values,
                strict=True,
            )
        )
    except SearchAborted:
        return _result(row, "exact_certification_failed")
    ranking = _deterministic_ranking(candidates)
    certificate = MechanismExactCertificate(
        family=row.family,
        state_sha256=state_digest,
        hypothesized_winner_index=state.hypothesized_winner_index,
        certified_winner_index=ranking[0],
        margin_kind=exact.margin[0],
        margin_value=exact.margin[1],
        candidates=candidates,
    )
    if certificate.margin_value <= 0:
        return _result(row, "non_positive_margin")
    if certificate.certified_winner_index != certificate.hypothesized_winner_index:
        return _result(row, "hypothesized_winner_not_certified")
    if not _mechanism_semantics_certified(state, certificate):
        return _result(row, "mechanism_semantics_not_certified")

    variant = MechanismVariant(
        parameter_row=row,
        host_nodes=state.host_nodes,
        host_edges=state.host_edges,
        logical_nodes=state.logical_nodes,
        logical_edges=state.logical_edges,
        cores=state.cores,
        actions=state.actions,
        original_candidate_index=state.original_candidate_index,
        hypothesized_winner_index=state.hypothesized_winner_index,
        l_cap=state.l_cap,
        q_cap=state.q_cap,
        certificate=certificate,
    )
    if variant.state_sha256 != state_digest:
        return _result(row, "byte_reconstruction_failed")
    return _result(row, "accepted", variant)


def unsupported_composed_mechanism(
    *, parameter_row_sha256: str, constituent_families: tuple[str, ...]
) -> MechanismGenerationResult:
    """Fail closed until interaction and ablation certificates have a registered primitive."""

    return MechanismGenerationResult(
        status="unsupported_composed_mechanism",
        parameter_row_sha256=parameter_row_sha256,
        constituent_families=constituent_families,
        variant=None,
    )


def verify_byte_reconstruction(variant: MechanismVariant) -> bool:
    """Rebuild from only the committed row and require identical canonical variant bytes."""

    if type(variant) is not MechanismVariant:
        raise TypeError("variant must be an exact MechanismVariant")
    replay = generate_single_mechanism_variant(variant.parameter_row)
    if replay.status != "accepted" or replay.variant is None:
        raise ValueError(f"parameter-row replay did not remain accepted: {replay.status}")
    if replay.variant.to_bytes() != variant.to_bytes():
        raise ValueError("transformation metadata does not reconstruct variant byte-for-byte")
    return True


def validate_variant(value: object) -> MechanismVariant:
    """Strictly parse and independently exact-recertify an untrusted variant record."""

    if type(value) is MechanismVariant:
        variant = MechanismVariant.from_dict(value.to_dict())
    else:
        variant = MechanismVariant.from_dict(value)
    return variant


class MechanismCollisionError(ValueError):
    """A state, isomorphism class, parent, or seed stream was already admitted."""


_PARENT_MAP_ENTRY_FIELDS = frozenset({"mechanism_parent_id", "partition"})


@dataclass(frozen=True, slots=True)
class MechanismParentPartitionEntry:
    mechanism_parent_id: str
    partition: str

    def __post_init__(self) -> None:
        if (
            type(self.mechanism_parent_id) is not str
            or _PARENT_ID_PATTERN.fullmatch(self.mechanism_parent_id) is None
        ):
            raise ValueError("parent-map entry has an invalid mechanism_parent_id")
        if type(self.partition) is not str or self.partition not in MECHANISM_PARTITIONS:
            raise ValueError("parent-map entry has an invalid partition")

    def to_dict(self) -> dict[str, object]:
        return {
            "mechanism_parent_id": self.mechanism_parent_id,
            "partition": self.partition,
        }

    @classmethod
    def from_dict(cls, value: object) -> MechanismParentPartitionEntry:
        document = require_exact_keys(value, _PARENT_MAP_ENTRY_FIELDS, "parent-map entry")
        return cls(
            mechanism_parent_id=document["mechanism_parent_id"],
            partition=document["partition"],
        )


_PARENT_MAP_FIELDS = frozenset(
    {"schema", "schema_version", "release_id", "entries", "parent_map_sha256"}
)


@dataclass(frozen=True, slots=True)
class MechanismParentPartitionMap:
    """Canonical parent-to-partition map; repeated siblings collapse to one assignment."""

    SCHEMA: ClassVar[str] = MECHANISM_PARENT_MAP_SCHEMA
    SCHEMA_VERSION: ClassVar[int] = MECHANISM_PARENT_MAP_SCHEMA_VERSION

    release_id: str
    entries: tuple[MechanismParentPartitionEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_id", _require_string(self.release_id, "release_id"))
        if type(self.entries) is not tuple or not all(
            type(entry) is MechanismParentPartitionEntry for entry in self.entries
        ):
            raise TypeError("parent-map entries must be an exact immutable entry tuple")
        expected = tuple(
            sorted(self.entries, key=lambda entry: (entry.mechanism_parent_id, entry.partition))
        )
        if self.entries != expected:
            raise ValueError("parent-map entries must use canonical order")
        parents: dict[str, str] = {}
        for entry in self.entries:
            previous = parents.setdefault(entry.mechanism_parent_id, entry.partition)
            if previous != entry.partition:
                raise MechanismCollisionError(
                    "one scientific parent crosses mechanism partitions"
                )
            if previous == entry.partition and sum(
                candidate.mechanism_parent_id == entry.mechanism_parent_id
                for candidate in self.entries
            ) > 1:
                raise ValueError("parent-map entries must be duplicate-free")

    def _digest_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "release_id": self.release_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def parent_map_sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self._digest_payload())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self._digest_payload(), "parent_map_sha256": self.parent_map_sha256}

    @classmethod
    def from_dict(cls, value: object) -> MechanismParentPartitionMap:
        document = require_exact_keys(value, _PARENT_MAP_FIELDS, "parent-partition map")
        _require_schema(document, cls.SCHEMA, cls.SCHEMA_VERSION, "parent-partition map")
        entries = tuple(
            MechanismParentPartitionEntry.from_dict(entry)
            for entry in _json_tuple(document["entries"], "parent-map entries")
        )
        result = cls(release_id=document["release_id"], entries=entries)
        claimed = _require_sha256(document["parent_map_sha256"], "parent_map_sha256")
        if claimed != result.parent_map_sha256:
            raise ValueError("parent-partition map digest mismatch")
        return result


def make_parent_partition_map(
    *,
    release_id: str,
    assignments: Sequence[tuple[str, str]],
) -> MechanismParentPartitionMap:
    """Build a map while allowing any number of same-partition siblings."""

    if type(assignments) not in {tuple, list}:
        raise TypeError("assignments must be a sequence of exact pairs")
    by_parent: dict[str, str] = {}
    for index, assignment in enumerate(assignments):
        if type(assignment) is not tuple or len(assignment) != 2:
            raise TypeError(f"assignments[{index}] must be an exact pair")
        entry = MechanismParentPartitionEntry(
            mechanism_parent_id=assignment[0],
            partition=assignment[1],
        )
        prior = by_parent.setdefault(entry.mechanism_parent_id, entry.partition)
        if prior != entry.partition:
            raise MechanismCollisionError("one scientific parent crosses mechanism partitions")
    return MechanismParentPartitionMap(
        release_id=release_id,
        entries=tuple(
            MechanismParentPartitionEntry(parent, partition)
            for parent, partition in sorted(by_parent.items())
        ),
    )


_PARENT_MAP_VERIFICATION_SEAL = object()


@dataclass(frozen=True, slots=True)
class VerifiedMechanismParentPartitionMap:
    value: MechanismParentPartitionMap
    committed_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _PARENT_MAP_VERIFICATION_SEAL:
            raise TypeError("VerifiedMechanismParentPartitionMap must come from verifier")
        if type(self.value) is not MechanismParentPartitionMap:
            raise TypeError("verified parent map must wrap an exact map")
        committed = _require_sha256(self.committed_sha256, "committed parent-map SHA-256")
        if committed != self.value.parent_map_sha256:
            raise ValueError("verified parent map does not match its commitment")
        object.__setattr__(self, "committed_sha256", committed)


def verify_parent_partition_map(
    value: object,
    *,
    expected_parent_map_sha256: str,
) -> VerifiedMechanismParentPartitionMap:
    """Bind a parent map to an independently supplied commitment."""

    expected = _require_sha256(expected_parent_map_sha256, "expected parent-map SHA-256")
    parsed = (
        MechanismParentPartitionMap.from_dict(value.to_dict())
        if type(value) is MechanismParentPartitionMap
        else MechanismParentPartitionMap.from_dict(value)
    )
    if parsed.parent_map_sha256 != expected:
        raise ValueError("parent-partition map does not match independent commitment")
    return VerifiedMechanismParentPartitionMap(parsed, expected, _PARENT_MAP_VERIFICATION_SEAL)


def compare_committed_parent_maps(
    public: VerifiedMechanismParentPartitionMap,
    custodian_locked: VerifiedMechanismParentPartitionMap,
) -> bool:
    """Reject any scientific parent appearing in both independently committed maps."""

    if type(public) is not VerifiedMechanismParentPartitionMap or type(
        custodian_locked
    ) is not VerifiedMechanismParentPartitionMap:
        raise TypeError("both parent maps must be independently verified commitments")
    if public.value.release_id != custodian_locked.value.release_id:
        raise ValueError("committed parent maps use different release_id values")
    public_ids = {entry.mechanism_parent_id for entry in public.value.entries}
    locked_ids = {entry.mechanism_parent_id for entry in custodian_locked.value.entries}
    if public_ids & locked_ids:
        raise MechanismCollisionError("scientific parent occurs in both committed parent maps")
    return True


_REGISTRY_ENTRY_FIELDS: frozenset[str] = frozenset(
    {
        "partition",
        "mechanism_parent_id",
        "parent_seed_key",
        "transformation_seed_key",
        "plan_sha256",
        "state_sha256",
        "release_mechanism_iso_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class MechanismIdentityEntry:
    partition: str
    mechanism_parent_id: str
    parent_seed_key: str
    transformation_seed_key: str
    plan_sha256: str
    state_sha256: str
    release_mechanism_iso_sha256: str

    def __post_init__(self) -> None:
        if type(self.partition) is not str or self.partition not in MECHANISM_PARTITIONS:
            raise ValueError("registry entry has an invalid mechanism partition")
        if (
            type(self.mechanism_parent_id) is not str
            or _PARENT_ID_PATTERN.fullmatch(self.mechanism_parent_id) is None
        ):
            raise ValueError("registry entry has an invalid mechanism_parent_id")
        for name in (
            "parent_seed_key",
            "transformation_seed_key",
            "plan_sha256",
            "state_sha256",
            "release_mechanism_iso_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))

    @classmethod
    def _from_verified_attempt(
        cls,
        *,
        plan: VerifiedMechanismPlan,
        variant: MechanismVariant,
        release_mechanism_iso_sha256: str,
    ) -> MechanismIdentityEntry:
        if type(plan) is not VerifiedMechanismPlan:
            raise TypeError("identity entry requires a VerifiedMechanismPlan")
        if type(variant) is not MechanismVariant:
            raise TypeError("identity entry requires an exact reconstructed variant")
        row = variant.parameter_row
        if row != plan.row:
            raise ValueError("identity entry variant disagrees with verified plan")
        return cls(
            partition=row.partition,
            mechanism_parent_id=row.mechanism_parent_id,
            parent_seed_key=row.parent_seed_key,
            transformation_seed_key=row.transformation_seed_key,
            plan_sha256=plan.plan_sha256,
            state_sha256=variant.state_sha256,
            release_mechanism_iso_sha256=release_mechanism_iso_sha256,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "partition": self.partition,
            "mechanism_parent_id": self.mechanism_parent_id,
            "parent_seed_key": self.parent_seed_key,
            "transformation_seed_key": self.transformation_seed_key,
            "plan_sha256": self.plan_sha256,
            "state_sha256": self.state_sha256,
            "release_mechanism_iso_sha256": self.release_mechanism_iso_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> MechanismIdentityEntry:
        document = require_exact_keys(value, _REGISTRY_ENTRY_FIELDS, "mechanism registry entry")
        return cls(
            partition=document["partition"],
            mechanism_parent_id=document["mechanism_parent_id"],
            parent_seed_key=document["parent_seed_key"],
            transformation_seed_key=document["transformation_seed_key"],
            plan_sha256=document["plan_sha256"],
            state_sha256=document["state_sha256"],
            release_mechanism_iso_sha256=document["release_mechanism_iso_sha256"],
        )


_REGISTRY_FIELDS: frozenset[str] = frozenset(
    {"schema", "schema_version", "entries", "registry_sha256"}
)


@dataclass(frozen=True, slots=True)
class MechanismIdentityRegistry:
    """Immutable release-wide collision set for accepted mechanism states."""

    SCHEMA: ClassVar[str] = MECHANISM_REGISTRY_SCHEMA
    SCHEMA_VERSION: ClassVar[int] = MECHANISM_REGISTRY_SCHEMA_VERSION

    entries: tuple[MechanismIdentityEntry, ...]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or not all(
            type(entry) is MechanismIdentityEntry for entry in self.entries
        ):
            raise TypeError("registry entries must be an immutable MechanismIdentityEntry tuple")
        expected_order = tuple(
            sorted(
                self.entries,
                key=lambda entry: (
                    entry.state_sha256,
                    entry.release_mechanism_iso_sha256,
                    entry.mechanism_parent_id,
                    entry.plan_sha256,
                ),
            )
        )
        if self.entries != expected_order:
            raise ValueError("registry entries must use canonical digest order")
        self._assert_unique()

    def _assert_unique(self) -> None:
        fields = (
            "state_sha256",
            "release_mechanism_iso_sha256",
            "transformation_seed_key",
            "plan_sha256",
        )
        for field in fields:
            values = [getattr(entry, field) for entry in self.entries]
            if len(values) != len(set(values)):
                raise MechanismCollisionError(f"duplicate {field} in mechanism identity registry")
        parent_partitions: dict[str, str] = {}
        for entry in self.entries:
            previous = parent_partitions.setdefault(
                entry.mechanism_parent_id,
                entry.partition,
            )
            if previous != entry.partition:
                raise MechanismCollisionError(
                    "one scientific parent crosses mechanism partitions"
                )

    @classmethod
    def empty(cls) -> MechanismIdentityRegistry:
        return cls(entries=())

    def _admit_verified(self, incoming: MechanismIdentityEntry) -> MechanismIdentityRegistry:
        """Internal atomic-boundary insertion; public variants are never admitted directly."""

        if type(incoming) is not MechanismIdentityEntry:
            raise TypeError("incoming must be an exact MechanismIdentityEntry")
        checks = (
            "state_sha256",
            "release_mechanism_iso_sha256",
            "transformation_seed_key",
            "plan_sha256",
        )
        for field in checks:
            if any(
                getattr(existing, field) == getattr(incoming, field) for existing in self.entries
            ):
                raise MechanismCollisionError(f"duplicate {field} across mechanism variants")
        for existing in self.entries:
            if (
                existing.mechanism_parent_id == incoming.mechanism_parent_id
                and existing.partition != incoming.partition
            ):
                raise MechanismCollisionError(
                    "one scientific parent crosses mechanism partitions"
                )
        return MechanismIdentityRegistry(
            entries=tuple(
                sorted(
                    (*self.entries, incoming),
                    key=lambda entry: (
                        entry.state_sha256,
                        entry.release_mechanism_iso_sha256,
                        entry.mechanism_parent_id,
                        entry.plan_sha256,
                    ),
                )
            )
        )

    def _digest_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def registry_sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self._digest_payload())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self._digest_payload(), "registry_sha256": self.registry_sha256}

    @classmethod
    def from_dict(cls, value: object) -> MechanismIdentityRegistry:
        document = require_exact_keys(value, _REGISTRY_FIELDS, "mechanism identity registry")
        _require_schema(document, cls.SCHEMA, cls.SCHEMA_VERSION, "mechanism identity registry")
        entries = tuple(
            MechanismIdentityEntry.from_dict(entry)
            for entry in _json_tuple(document["entries"], "registry entries")
        )
        registry = cls(entries=entries)
        claimed = _require_sha256(document["registry_sha256"], "registry_sha256")
        if claimed != registry.registry_sha256:
            raise ValueError("mechanism identity registry digest mismatch")
        return registry


class AuthoritativeIsomorphismExecutionUnavailableError(RuntimeError):
    """Raised until an execution-attested external canonicalizer is implemented."""


def verify_authoritative_isomorphism_engine(
    provider: object,
    manifest: object,
    *,
    expected_manifest_sha256: str,
) -> NoReturn:
    """Reject callback-only providers until an external execution verifier exists.

    Provider attributes cannot authenticate executable bytes or prove that the callback invokes
    the committed implementation.  A future producer must execute a securely captured binary,
    bind its runtime and fixtures, and return immutable execution evidence.  Until then no
    release isomorphism digest can enter the registry.
    """

    del provider, manifest, expected_manifest_sha256
    raise AuthoritativeIsomorphismExecutionUnavailableError(
        "callback-only canonicalizers do not provide executable execution evidence"
    )


_ATTEMPT_ENTRY_FIELDS = frozenset(
    {
        "plan_sha256",
        "parameter_row_sha256",
        "mechanism_parent_id",
        "partition",
        "family",
        "status",
        "state_sha256",
        "diagnostic_iso_sha256",
        "release_mechanism_iso_sha256",
        "certificate_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class MechanismAttemptLedgerEntry:
    """Typed audit record for one precommitted attempt, accepted or rejected."""

    plan_sha256: str
    parameter_row_sha256: str
    mechanism_parent_id: str
    partition: str
    family: str
    status: GenerationStatus
    state_sha256: str | None
    diagnostic_iso_sha256: str | None
    release_mechanism_iso_sha256: str | None
    certificate_sha256: str | None

    def __post_init__(self) -> None:
        for name in ("plan_sha256", "parameter_row_sha256"):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        if (
            type(self.mechanism_parent_id) is not str
            or _PARENT_ID_PATTERN.fullmatch(self.mechanism_parent_id) is None
        ):
            raise ValueError("attempt entry has an invalid mechanism_parent_id")
        if type(self.partition) is not str or self.partition not in MECHANISM_PARTITIONS:
            raise ValueError("attempt entry has an invalid partition")
        if type(self.family) is not str or self.family not in MECHANISM_FAMILIES:
            raise ValueError("attempt entry has an invalid family")
        object.__setattr__(self, "status", _require_generation_status(self.status))
        for name in (
            "state_sha256",
            "diagnostic_iso_sha256",
            "release_mechanism_iso_sha256",
            "certificate_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _require_sha256(value, name))
        if self.status == "accepted":
            if self.release_mechanism_iso_sha256 is None:
                raise ValueError("accepted attempt requires an authoritative release digest")
            if self.state_sha256 is None or self.certificate_sha256 is None:
                raise ValueError("accepted attempt requires state and exact-certificate digests")
        elif self.release_mechanism_iso_sha256 is not None:
            raise ValueError("rejected attempt must withhold release mechanism digest")
        if (
            self.status == "unsupported_authoritative_isomorphism"
            and self.diagnostic_iso_sha256 is None
        ):
            raise ValueError("unsupported-isomorphism attempt requires a diagnostic digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_sha256": self.plan_sha256,
            "parameter_row_sha256": self.parameter_row_sha256,
            "mechanism_parent_id": self.mechanism_parent_id,
            "partition": self.partition,
            "family": self.family,
            "status": self.status,
            "state_sha256": self.state_sha256,
            "diagnostic_iso_sha256": self.diagnostic_iso_sha256,
            "release_mechanism_iso_sha256": self.release_mechanism_iso_sha256,
            "certificate_sha256": self.certificate_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> MechanismAttemptLedgerEntry:
        document = require_exact_keys(value, _ATTEMPT_ENTRY_FIELDS, "mechanism attempt entry")
        return cls(
            plan_sha256=document["plan_sha256"],
            parameter_row_sha256=document["parameter_row_sha256"],
            mechanism_parent_id=document["mechanism_parent_id"],
            partition=document["partition"],
            family=document["family"],
            status=document["status"],
            state_sha256=document["state_sha256"],
            diagnostic_iso_sha256=document["diagnostic_iso_sha256"],
            release_mechanism_iso_sha256=document["release_mechanism_iso_sha256"],
            certificate_sha256=document["certificate_sha256"],
        )


_ATTEMPT_LEDGER_FIELDS = frozenset(
    {"schema", "schema_version", "entries", "ledger_sha256"}
)


@dataclass(frozen=True, slots=True)
class MechanismAttemptLedger:
    SCHEMA: ClassVar[str] = MECHANISM_ATTEMPT_LEDGER_SCHEMA
    SCHEMA_VERSION: ClassVar[int] = MECHANISM_ATTEMPT_LEDGER_SCHEMA_VERSION

    entries: tuple[MechanismAttemptLedgerEntry, ...]

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or not all(
            type(entry) is MechanismAttemptLedgerEntry for entry in self.entries
        ):
            raise TypeError("attempt-ledger entries must be an exact immutable tuple")
        expected = tuple(sorted(self.entries, key=lambda entry: entry.plan_sha256))
        if self.entries != expected:
            raise ValueError("attempt-ledger entries must use canonical plan-digest order")
        plans = tuple(entry.plan_sha256 for entry in self.entries)
        if len(plans) != len(set(plans)):
            raise MechanismCollisionError("duplicate precommitted plan in mechanism ledger")

    @classmethod
    def empty(cls) -> MechanismAttemptLedger:
        return cls(entries=())

    def append(self, entry: MechanismAttemptLedgerEntry) -> MechanismAttemptLedger:
        if type(entry) is not MechanismAttemptLedgerEntry:
            raise TypeError("entry must be an exact MechanismAttemptLedgerEntry")
        return MechanismAttemptLedger(
            entries=tuple(sorted((*self.entries, entry), key=lambda item: item.plan_sha256))
        )

    def _digest_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @property
    def ledger_sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(self._digest_payload())).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {**self._digest_payload(), "ledger_sha256": self.ledger_sha256}

    @classmethod
    def from_dict(cls, value: object) -> MechanismAttemptLedger:
        document = require_exact_keys(value, _ATTEMPT_LEDGER_FIELDS, "mechanism attempt ledger")
        _require_schema(document, cls.SCHEMA, cls.SCHEMA_VERSION, "mechanism attempt ledger")
        result = cls(
            entries=tuple(
                MechanismAttemptLedgerEntry.from_dict(entry)
                for entry in _json_tuple(document["entries"], "attempt-ledger entries")
            )
        )
        claimed = _require_sha256(document["ledger_sha256"], "ledger_sha256")
        if claimed != result.ledger_sha256:
            raise ValueError("mechanism attempt ledger digest mismatch")
        return result


@dataclass(frozen=True, slots=True)
class MechanismAdmissionOutcome:
    """Atomic result: registry and ledger either advance together or only rejection is logged."""

    status: GenerationStatus
    plan_sha256: str
    registry: MechanismIdentityRegistry
    ledger: MechanismAttemptLedger
    variant: MechanismVariant | None
    release_mechanism_iso_sha256: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _require_generation_status(self.status))
        object.__setattr__(self, "plan_sha256", _require_sha256(self.plan_sha256, "plan_sha256"))
        if type(self.registry) is not MechanismIdentityRegistry:
            raise TypeError("admission outcome registry must be an exact registry")
        if type(self.ledger) is not MechanismAttemptLedger:
            raise TypeError("admission outcome ledger must be an exact ledger")
        release_digest = self.release_mechanism_iso_sha256
        if release_digest is not None:
            release_digest = _require_sha256(
                release_digest,
                "release_mechanism_iso_sha256",
            )
            object.__setattr__(self, "release_mechanism_iso_sha256", release_digest)
        if self.status == "accepted":
            if type(self.variant) is not MechanismVariant or release_digest is None:
                raise ValueError("accepted admission requires reconstructed variant and digest")
        elif self.variant is not None or release_digest is not None:
            raise ValueError("rejected admission must not expose variant or release digest")
        matching = tuple(
            entry for entry in self.ledger.entries if entry.plan_sha256 == self.plan_sha256
        )
        if len(matching) != 1 or matching[0].status != self.status:
            raise ValueError("admission outcome is not represented exactly once in its ledger")


def _attempt_entry(
    plan: VerifiedMechanismPlan,
    status: GenerationStatus,
    variant: MechanismVariant | None = None,
    *,
    release_mechanism_iso_sha256: str | None = None,
) -> MechanismAttemptLedgerEntry:
    return MechanismAttemptLedgerEntry(
        plan_sha256=plan.plan_sha256,
        parameter_row_sha256=plan.row.sha256,
        mechanism_parent_id=plan.row.mechanism_parent_id,
        partition=plan.row.partition,
        family=plan.row.family,
        status=status,
        state_sha256=None if variant is None else variant.state_sha256,
        diagnostic_iso_sha256=None if variant is None else variant.diagnostic_iso_sha256,
        release_mechanism_iso_sha256=release_mechanism_iso_sha256,
        certificate_sha256=(
            None if variant is None else variant.certificate.certificate_sha256
        ),
    )


def _closed_admission(
    *,
    plan: VerifiedMechanismPlan,
    status: GenerationStatus,
    registry: MechanismIdentityRegistry,
    ledger: MechanismAttemptLedger,
    diagnostic_variant: MechanismVariant | None = None,
) -> MechanismAdmissionOutcome:
    entry = _attempt_entry(plan, status, diagnostic_variant)
    return MechanismAdmissionOutcome(
        status=status,
        plan_sha256=plan.plan_sha256,
        registry=registry,
        ledger=ledger.append(entry),
        variant=None,
        release_mechanism_iso_sha256=None,
    )


def execute_mechanism_attempt(
    *,
    plan: VerifiedMechanismPlan,
    registry: MechanismIdentityRegistry,
    ledger: MechanismAttemptLedger,
    authoritative_engine: object | None,
) -> MechanismAdmissionOutcome:
    """Generate, contextually reconstruct, and admit one plan as one immutable operation."""

    if type(plan) is not VerifiedMechanismPlan:
        raise TypeError("plan must be a VerifiedMechanismPlan from verify_mechanism_plan")
    if type(registry) is not MechanismIdentityRegistry:
        raise TypeError("registry must be an exact MechanismIdentityRegistry")
    if type(ledger) is not MechanismAttemptLedger:
        raise TypeError("ledger must be an exact MechanismAttemptLedger")
    if any(entry.plan_sha256 == plan.plan_sha256 for entry in ledger.entries):
        raise MechanismCollisionError("precommitted plan was already attempted")

    generated = generate_single_mechanism_variant(plan.row)
    if generated.status != "accepted" or generated.variant is None:
        return _closed_admission(
            plan=plan,
            status=generated.status,
            registry=registry,
            ledger=ledger,
        )

    candidate = generated.variant
    try:
        candidate_bytes = candidate.to_bytes()
        reconstructed = MechanismVariant.from_dict(json.loads(candidate_bytes))
        if reconstructed.parameter_row != plan.row:
            raise ValueError("reconstructed row differs from verified plan")
        if reconstructed.to_bytes() != candidate_bytes:
            raise ValueError("contextual reconstruction changed canonical variant bytes")
    except (TypeError, ValueError, json.JSONDecodeError):
        return _closed_admission(
            plan=plan,
            status="contextual_replay_failed",
            registry=registry,
            ledger=ledger,
        )

    # A self-reported Python callback cannot prove which executable produced its answer.  The
    # authoritative_engine argument is reserved for the future external execution-evidence
    # adapter; current releases always stop at the diagnostic identity boundary.
    del authoritative_engine
    return _closed_admission(
        plan=plan,
        status="unsupported_authoritative_isomorphism",
        registry=registry,
        ledger=ledger,
        diagnostic_variant=reconstructed,
    )


__all__ = [
    "AUTHORITATIVE_ISOMORPHISM_MANIFEST_SCHEMA",
    "AUTHORITATIVE_ISOMORPHISM_MANIFEST_SCHEMA_VERSION",
    "AuthoritativeIsomorphismExecutionUnavailableError",
    "GENERATION_STATUSES",
    "MECHANISM_ATTEMPT_LEDGER_SCHEMA",
    "MECHANISM_ATTEMPT_LEDGER_SCHEMA_VERSION",
    "MECHANISM_CERTIFICATE_SCHEMA",
    "MECHANISM_CERTIFICATE_SCHEMA_VERSION",
    "MECHANISM_FAMILIES",
    "MECHANISM_PARAMETER_SCHEMA",
    "MECHANISM_PARAMETER_SCHEMA_VERSION",
    "MECHANISM_PARENT_MAP_SCHEMA",
    "MECHANISM_PARENT_MAP_SCHEMA_VERSION",
    "MECHANISM_PARTITIONS",
    "MECHANISM_PLAN_SCHEMA",
    "MECHANISM_PLAN_SCHEMA_VERSION",
    "MECHANISM_REGISTRY_SCHEMA",
    "MECHANISM_REGISTRY_SCHEMA_VERSION",
    "MECHANISM_RESULT_SCHEMA",
    "MECHANISM_RESULT_SCHEMA_VERSION",
    "MECHANISM_VARIANT_SCHEMA",
    "MECHANISM_VARIANT_SCHEMA_VERSION",
    "ExactCandidateRecord",
    "MechanismAdmissionOutcome",
    "MechanismAttemptLedger",
    "MechanismAttemptLedgerEntry",
    "MechanismCollisionError",
    "MechanismExactCertificate",
    "MechanismGenerationResult",
    "MechanismIdentityEntry",
    "MechanismIdentityRegistry",
    "MechanismParameterRow",
    "MechanismParentPartitionEntry",
    "MechanismParentPartitionMap",
    "MechanismVariant",
    "VerifiedMechanismParentPartitionMap",
    "VerifiedMechanismPlan",
    "compare_committed_parent_maps",
    "execute_mechanism_attempt",
    "generate_single_mechanism_variant",
    "host_automorphism_count",
    "logical_symmetry_count",
    "make_mechanism_plan_document",
    "make_parameter_row",
    "make_parent_partition_map",
    "mechanism_parent_id",
    "mechanism_seed_requests",
    "motif_template_sha256",
    "parent_scientific_sha256",
    "parent_seed_namespace",
    "transformation_seed_namespace",
    "unsupported_composed_mechanism",
    "validate_variant",
    "verify_byte_reconstruction",
    "verify_authoritative_isomorphism_engine",
    "verify_mechanism_plan",
    "verify_parent_partition_map",
]

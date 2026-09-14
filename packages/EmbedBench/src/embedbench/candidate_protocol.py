"""Deterministic hard/OOD candidate enumeration and label-free truncation.

This module implements the candidate-bank boundary in Sections 7.3 and 10 of the
hard/OOD corpus specification.  It consumes an already verified realized host and
canonical immutable context.  It does not accept quality labels, learned scores, or
baseline decisions.

The result is a deterministic computation artifact, not standalone provenance.  The
enclosing immutable state record and verified manifests bind the host identity, window,
frozen context, original chain, caps, and registered candidate-sample seed request.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, Literal

import networkx as nx

from embedbench.hard_ood_schema import canonical_bytes

UINT64_MAX = 2**64 - 1
ENUMERATION_CAP = 4_000
OFFERED_BANK_CAP = 64
CANDIDATE_PROTOCOL_SCHEMA = "embedbench.candidate-protocol-result"
CANDIDATE_PROTOCOL_SCHEMA_VERSION = 1

AttemptStatus = Literal[
    "starting_embedding_unavailable",
    "invalid_generation",
    "enumeration_aborted",
    "no_candidate",
    "single_candidate",
    "candidate_bank_ready",
]
EnumerationStatus = Literal["complete", "aborted_cap"]
Candidate = tuple[int, ...]

ATTEMPT_STATUSES: frozenset[str] = frozenset(
    {
        "starting_embedding_unavailable",
        "invalid_generation",
        "enumeration_aborted",
        "no_candidate",
        "single_candidate",
        "candidate_bank_ready",
    }
)
ENUMERATION_STATUSES: frozenset[str] = frozenset({"complete", "aborted_cap"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _require_uint64(value: object, name: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _require_logical_variable(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_nodes(
    value: object,
    name: str,
    *,
    nonempty: bool,
) -> Candidate:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    nodes = tuple(_require_uint64(node, f"{name}[{index}]") for index, node in enumerate(value))
    if nonempty and not nodes:
        raise ValueError(f"{name} must not be empty")
    if len(nodes) != len(set(nodes)):
        raise ValueError(f"{name} must be duplicate-free")
    if nodes != tuple(sorted(nodes)):
        raise ValueError(f"{name} must be sorted")
    return nodes


def _canonical_logical_variables(value: object, name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    variables = tuple(
        _require_logical_variable(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(variables) != len(set(variables)):
        raise ValueError(f"{name} must be duplicate-free")
    if variables != tuple(sorted(variables)):
        raise ValueError(f"{name} must be sorted")
    return variables


def _canonical_bank(value: object, name: str, *, allow_empty: bool) -> tuple[Candidate, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    candidates = tuple(
        _canonical_nodes(candidate, f"{name}[{index}]", nonempty=True)
        for index, candidate in enumerate(value)
    )
    if not allow_empty and not candidates:
        raise ValueError(f"{name} must not be empty")
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"{name} must be duplicate-free")
    if candidates != tuple(sorted(candidates, key=lambda candidate: (len(candidate), candidate))):
        raise ValueError(f"{name} must be sorted by (chain length, node tuple)")
    return candidates


def _require_seed_key(value: object) -> bytes:
    if type(value) is not bytes:
        raise TypeError("candidate_sample_seed_key must be raw bytes")
    if len(value) != 32:
        raise ValueError("candidate_sample_seed_key must contain exactly 32 bytes")
    return value


def _verified_graph_snapshot(host: nx.Graph) -> nx.Graph:
    """Validate and detach the content of an already verified realized host."""

    if not isinstance(host, nx.Graph) or host.is_directed() or host.is_multigraph():
        raise TypeError("host must be an undirected simple NetworkX graph")
    nodes = tuple(sorted(_require_uint64(node, "host node") for node in host.nodes))
    if len(nodes) != host.number_of_nodes():
        raise ValueError("host contains duplicate normalized node identifiers")
    edges: list[tuple[int, int]] = []
    for raw_left, raw_right in host.edges:
        left = _require_uint64(raw_left, "host edge endpoint")
        right = _require_uint64(raw_right, "host edge endpoint")
        if left == right:
            raise ValueError("host must not contain self-loops")
        edges.append((min(left, right), max(left, right)))
    if len(edges) != len(set(edges)):
        raise ValueError("host must not contain parallel edges")
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(sorted(edges))
    return graph


@dataclass(frozen=True, slots=True)
class FrozenChain:
    """One canonical non-focus chain keyed by its logical variable."""

    logical_variable: int
    nodes: Candidate

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_variable",
            _require_logical_variable(self.logical_variable, "logical_variable"),
        )
        object.__setattr__(
            self,
            "nodes",
            _canonical_nodes(self.nodes, "frozen chain nodes", nonempty=True),
        )

    def to_dict(self) -> dict[str, object]:
        return {"logical_variable": self.logical_variable, "nodes": list(self.nodes)}


@dataclass(frozen=True, slots=True)
class CandidateFacts:
    """Exact immediate occupancy, contact, and residual facts for one candidate."""

    candidate: Candidate
    connected: bool
    chain_disjoint: bool
    realizes_every_required_coupler: bool
    within_l_cap: bool
    minor_valid: bool
    current_total_qubits: int
    current_maximum_chain_length: int
    within_q_cap: bool
    feasible_now: bool
    internal_chain_edge_count: int
    logical_contact_counts: tuple[int, ...]
    minimum_logical_contact_count: int
    total_logical_contact_count: int
    sorted_logical_contact_counts: tuple[int, ...]
    largest_free_component_nodes: Candidate
    largest_free_component_node_numerator: int
    largest_free_component_node_denominator: int
    largest_free_component_edge_connectivity: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate",
            _canonical_nodes(self.candidate, "candidate", nonempty=True),
        )
        for name in (
            "connected",
            "chain_disjoint",
            "realizes_every_required_coupler",
            "within_l_cap",
            "minor_valid",
            "within_q_cap",
            "feasible_now",
        ):
            _require_bool(getattr(self, name), name)
        expected_minor_valid = (
            self.connected
            and self.chain_disjoint
            and self.realizes_every_required_coupler
            and self.within_l_cap
        )
        if self.minor_valid != expected_minor_valid:
            raise ValueError("minor_valid must equal the conjunction of immediate-validity facts")
        if self.feasible_now != (self.minor_valid and self.within_q_cap):
            raise ValueError("feasible_now must equal minor_valid and within_q_cap")

        for name in (
            "current_total_qubits",
            "current_maximum_chain_length",
            "internal_chain_edge_count",
            "minimum_logical_contact_count",
            "total_logical_contact_count",
            "largest_free_component_node_numerator",
            "largest_free_component_edge_connectivity",
        ):
            _require_nonnegative_int(getattr(self, name), name)
        _require_positive_int(
            self.largest_free_component_node_denominator,
            "largest_free_component_node_denominator",
        )
        if self.current_total_qubits == 0 or self.current_maximum_chain_length == 0:
            raise ValueError("a candidate replacement must occupy at least one qubit")
        if self.current_total_qubits < len(self.candidate):
            raise ValueError("current_total_qubits cannot be smaller than the candidate chain")
        if not (
            len(self.candidate) <= self.current_maximum_chain_length <= self.current_total_qubits
        ):
            raise ValueError("current_maximum_chain_length is inconsistent with occupied chains")
        if self.internal_chain_edge_count > len(self.candidate) * (len(self.candidate) - 1) // 2:
            raise ValueError("internal_chain_edge_count exceeds a simple induced graph")

        if type(self.logical_contact_counts) is not tuple:
            raise TypeError("logical_contact_counts must be an immutable tuple")
        contacts = tuple(
            _require_nonnegative_int(value, f"logical_contact_counts[{index}]")
            for index, value in enumerate(self.logical_contact_counts)
        )
        if type(self.sorted_logical_contact_counts) is not tuple:
            raise TypeError("sorted_logical_contact_counts must be an immutable tuple")
        sorted_contacts = tuple(
            _require_nonnegative_int(value, f"sorted_logical_contact_counts[{index}]")
            for index, value in enumerate(self.sorted_logical_contact_counts)
        )
        if sorted_contacts != tuple(sorted(contacts)):
            raise ValueError("sorted_logical_contact_counts must sort logical_contact_counts")
        expected_minimum = min(contacts, default=0)
        if self.minimum_logical_contact_count != expected_minimum:
            raise ValueError("minimum_logical_contact_count disagrees with contact counts")
        if self.total_logical_contact_count != sum(contacts):
            raise ValueError("total_logical_contact_count disagrees with contact counts")
        object.__setattr__(self, "logical_contact_counts", contacts)
        object.__setattr__(self, "sorted_logical_contact_counts", sorted_contacts)

        largest_nodes = _canonical_nodes(
            self.largest_free_component_nodes,
            "largest_free_component_nodes",
            nonempty=False,
        )
        if len(largest_nodes) != self.largest_free_component_node_numerator:
            raise ValueError("largest free-component numerator must equal its node count")
        if (
            self.largest_free_component_node_numerator
            > self.largest_free_component_node_denominator
        ):
            raise ValueError("largest free-component numerator exceeds its denominator")
        maximum_connectivity = max(0, len(largest_nodes) - 1)
        if self.largest_free_component_edge_connectivity > maximum_connectivity:
            raise ValueError("largest free-component edge connectivity is impossible")
        object.__setattr__(self, "largest_free_component_nodes", largest_nodes)

    @property
    def largest_free_component_node_fraction(self) -> float:
        return (
            self.largest_free_component_node_numerator
            / self.largest_free_component_node_denominator
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": list(self.candidate),
            "connected": self.connected,
            "chain_disjoint": self.chain_disjoint,
            "realizes_every_required_coupler": self.realizes_every_required_coupler,
            "within_l_cap": self.within_l_cap,
            "minor_valid": self.minor_valid,
            "current_total_qubits": self.current_total_qubits,
            "current_maximum_chain_length": self.current_maximum_chain_length,
            "within_q_cap": self.within_q_cap,
            "feasible_now": self.feasible_now,
            "internal_chain_edge_count": self.internal_chain_edge_count,
            "logical_contact_counts": list(self.logical_contact_counts),
            "minimum_logical_contact_count": self.minimum_logical_contact_count,
            "total_logical_contact_count": self.total_logical_contact_count,
            "sorted_logical_contact_counts": list(self.sorted_logical_contact_counts),
            "largest_free_component_nodes": list(self.largest_free_component_nodes),
            "largest_free_component_node_numerator": (self.largest_free_component_node_numerator),
            "largest_free_component_node_denominator": (
                self.largest_free_component_node_denominator
            ),
            "largest_free_component_node_fraction": (self.largest_free_component_node_fraction),
            "largest_free_component_edge_connectivity": (
                self.largest_free_component_edge_connectivity
            ),
        }


@dataclass(frozen=True, slots=True)
class CandidateProtocolResult:
    """Closed immutable outcome for one candidate-generation attempt.

    This object intentionally omits input provenance.  Its enclosing state record and
    manifests must bind the host/context, caps, and registered sample-seed request.
    """

    SCHEMA: ClassVar[str] = CANDIDATE_PROTOCOL_SCHEMA
    SCHEMA_VERSION: ClassVar[int] = CANDIDATE_PROTOCOL_SCHEMA_VERSION

    attempt_status: AttemptStatus
    enumeration_status: EnumerationStatus | None
    enumeration_cap: int
    enumerated_prefix_count: int | None
    enumerated_prefix_sha256: str | None
    full_candidates: tuple[Candidate, ...] | None
    full_candidate_facts: tuple[CandidateFacts, ...] | None
    full_candidate_count: int | None
    full_candidate_bank_sha256: str | None
    offered_candidates: tuple[Candidate, ...] | None
    offered_candidate_count: int | None
    offered_candidate_bank_sha256: str | None
    offered_to_full_indices: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if type(self.attempt_status) is not str or self.attempt_status not in ATTEMPT_STATUSES:
            raise ValueError("attempt_status is not a registered candidate-generation status")
        if self.enumeration_status is not None and (
            type(self.enumeration_status) is not str
            or self.enumeration_status not in ENUMERATION_STATUSES
        ):
            raise ValueError("enumeration_status is not registered")
        _require_positive_int(self.enumeration_cap, "enumeration_cap")
        if self.enumeration_cap != ENUMERATION_CAP:
            raise ValueError(f"enumeration_cap must equal {ENUMERATION_CAP}")

        early_null_fields = (
            self.enumeration_status,
            self.enumerated_prefix_count,
            self.enumerated_prefix_sha256,
            self.full_candidates,
            self.full_candidate_facts,
            self.full_candidate_count,
            self.full_candidate_bank_sha256,
            self.offered_candidates,
            self.offered_candidate_count,
            self.offered_candidate_bank_sha256,
            self.offered_to_full_indices,
        )
        if self.attempt_status in {"starting_embedding_unavailable", "invalid_generation"}:
            if any(value is not None for value in early_null_fields):
                raise ValueError(
                    "pre-enumeration failures require null enumeration and bank fields"
                )
            return

        if self.enumerated_prefix_count is None:
            raise ValueError("an enumerated attempt requires enumerated_prefix_count")
        _require_nonnegative_int(self.enumerated_prefix_count, "enumerated_prefix_count")
        if self.enumerated_prefix_sha256 is None:
            raise ValueError("an enumerated attempt requires enumerated_prefix_sha256")
        _require_sha256(self.enumerated_prefix_sha256, "enumerated_prefix_sha256")

        bank_fields = (
            self.full_candidates,
            self.full_candidate_facts,
            self.full_candidate_count,
            self.full_candidate_bank_sha256,
            self.offered_candidates,
            self.offered_candidate_count,
            self.offered_candidate_bank_sha256,
            self.offered_to_full_indices,
        )
        if self.attempt_status == "enumeration_aborted":
            if self.enumeration_status != "aborted_cap":
                raise ValueError("enumeration_aborted requires enumeration_status='aborted_cap'")
            if self.enumerated_prefix_count != ENUMERATION_CAP + 1:
                raise ValueError("aborted enumeration must contain the cap-plus-one prefix")
            if any(value is not None for value in bank_fields):
                raise ValueError("aborted enumeration requires null full and offered bank fields")
            return

        if self.enumeration_status != "complete":
            raise ValueError("a non-aborted enumerated attempt requires complete enumeration")
        if self.attempt_status == "no_candidate":
            if self.enumerated_prefix_count != 0:
                raise ValueError("no_candidate requires an empty enumerated prefix")
            if self.enumerated_prefix_sha256 != candidate_bank_sha256(()) or any(
                value is not None for value in bank_fields
            ):
                raise ValueError("no_candidate requires the canonical empty prefix and null banks")
            return

        if self.attempt_status not in {"single_candidate", "candidate_bank_ready"}:
            raise ValueError("unsupported attempt-status field combination")
        if any(value is None for value in bank_fields):
            raise ValueError("decision-producing statuses require complete bank fields")

        full = _canonical_bank(self.full_candidates, "full_candidates", allow_empty=False)
        offered = _canonical_bank(self.offered_candidates, "offered_candidates", allow_empty=False)
        if type(self.full_candidate_facts) is not tuple or not all(
            type(facts) is CandidateFacts for facts in self.full_candidate_facts
        ):
            raise TypeError("full_candidate_facts must be an immutable CandidateFacts tuple")
        if len(self.full_candidate_facts) != len(full) or any(
            facts.candidate != candidate
            for facts, candidate in zip(self.full_candidate_facts, full, strict=True)
        ):
            raise ValueError("full_candidate_facts must align exactly with full_candidates")
        if any(not facts.minor_valid for facts in self.full_candidate_facts):
            raise ValueError("a full candidate bank may contain only minor-valid candidates")
        if (
            len(
                {
                    facts.largest_free_component_node_denominator
                    for facts in self.full_candidate_facts
                }
            )
            != 1
        ):
            raise ValueError("candidate facts must share one realized-host denominator")

        if self.full_candidate_count != len(full):
            raise ValueError("full_candidate_count disagrees with full_candidates")
        if self.offered_candidate_count != len(offered):
            raise ValueError("offered_candidate_count disagrees with offered_candidates")
        _require_positive_int(self.full_candidate_count, "full_candidate_count")
        _require_positive_int(self.offered_candidate_count, "offered_candidate_count")
        if len(full) > ENUMERATION_CAP:
            raise ValueError("a complete candidate bank cannot exceed enumeration_cap")
        if not 1 <= len(offered) <= OFFERED_BANK_CAP:
            raise ValueError("offered candidate bank must contain between 1 and 64 candidates")
        if self.enumerated_prefix_count != len(full):
            raise ValueError("complete prefix count must equal the full candidate count")
        expected_full_digest = candidate_bank_sha256(full)
        expected_offered_digest = candidate_bank_sha256(offered)
        if self.full_candidate_bank_sha256 != expected_full_digest:
            raise ValueError("full candidate-bank digest mismatch")
        if self.enumerated_prefix_sha256 != expected_full_digest:
            raise ValueError("complete enumerated-prefix digest must equal the full-bank digest")
        if self.offered_candidate_bank_sha256 != expected_offered_digest:
            raise ValueError("offered candidate-bank digest mismatch")

        if type(self.offered_to_full_indices) is not tuple:
            raise TypeError("offered_to_full_indices must be an immutable tuple")
        indices = tuple(
            _require_nonnegative_int(index, f"offered_to_full_indices[{offset}]")
            for offset, index in enumerate(self.offered_to_full_indices)
        )
        if indices != tuple(sorted(set(indices))):
            raise ValueError("offered_to_full_indices must be sorted and duplicate-free")
        if len(indices) != len(offered) or any(index >= len(full) for index in indices):
            raise ValueError("offered_to_full_indices is not a full-bank mapping")
        if offered != tuple(full[index] for index in indices):
            raise ValueError("offered candidates disagree with their full-bank indices")
        expected_offered_count = min(len(full), OFFERED_BANK_CAP)
        if len(offered) != expected_offered_count:
            raise ValueError("offered bank does not have the required protocol cardinality")
        if self.attempt_status == "single_candidate" and len(full) != 1:
            raise ValueError("single_candidate requires exactly one full candidate")
        if self.attempt_status == "candidate_bank_ready" and len(full) < 2:
            raise ValueError("candidate_bank_ready requires at least two full candidates")

    @classmethod
    def invalid_generation(cls) -> CandidateProtocolResult:
        """Return the scientific failure used when the original chain is not valid."""

        return cls(
            attempt_status="invalid_generation",
            enumeration_status=None,
            enumeration_cap=ENUMERATION_CAP,
            enumerated_prefix_count=None,
            enumerated_prefix_sha256=None,
            full_candidates=None,
            full_candidate_facts=None,
            full_candidate_count=None,
            full_candidate_bank_sha256=None,
            offered_candidates=None,
            offered_candidate_count=None,
            offered_candidate_bank_sha256=None,
            offered_to_full_indices=None,
        )

    @classmethod
    def enumeration_aborted(cls, *, prefix_sha256: str) -> CandidateProtocolResult:
        """Return the cap-plus-one scientific enumeration failure."""

        return cls(
            attempt_status="enumeration_aborted",
            enumeration_status="aborted_cap",
            enumeration_cap=ENUMERATION_CAP,
            enumerated_prefix_count=ENUMERATION_CAP + 1,
            enumerated_prefix_sha256=prefix_sha256,
            full_candidates=None,
            full_candidate_facts=None,
            full_candidate_count=None,
            full_candidate_bank_sha256=None,
            offered_candidates=None,
            offered_candidate_count=None,
            offered_candidate_bank_sha256=None,
            offered_to_full_indices=None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.SCHEMA_VERSION,
            "attempt_status": self.attempt_status,
            "enumeration_status": self.enumeration_status,
            "enumeration_cap": self.enumeration_cap,
            "enumerated_prefix_count": self.enumerated_prefix_count,
            "enumerated_prefix_sha256": self.enumerated_prefix_sha256,
            "full_candidates": (
                None
                if self.full_candidates is None
                else [list(candidate) for candidate in self.full_candidates]
            ),
            "full_candidate_facts": (
                None
                if self.full_candidate_facts is None
                else [facts.to_dict() for facts in self.full_candidate_facts]
            ),
            "full_candidate_count": self.full_candidate_count,
            "full_candidate_bank_sha256": self.full_candidate_bank_sha256,
            "offered_candidates": (
                None
                if self.offered_candidates is None
                else [list(candidate) for candidate in self.offered_candidates]
            ),
            "offered_candidate_count": self.offered_candidate_count,
            "offered_candidate_bank_sha256": self.offered_candidate_bank_sha256,
            "offered_to_full_indices": (
                None if self.offered_to_full_indices is None else list(self.offered_to_full_indices)
            ),
        }


def canonical_candidate_bytes(candidate: Candidate) -> bytes:
    """Encode one sorted candidate with uint64 big-endian length and node words."""

    checked = _canonical_nodes(candidate, "candidate", nonempty=True)
    return len(checked).to_bytes(8, "big") + b"".join(node.to_bytes(8, "big") for node in checked)


def canonical_candidate_bank_bytes(candidates: tuple[Candidate, ...]) -> bytes:
    """Encode a canonical bank count followed by self-delimiting candidates."""

    checked = _canonical_bank(candidates, "candidate bank", allow_empty=True)
    return len(checked).to_bytes(8, "big") + b"".join(
        canonical_candidate_bytes(candidate) for candidate in checked
    )


def candidate_bank_sha256(candidates: tuple[Candidate, ...]) -> str:
    """Hash the uncompressed canonical candidate-bank array."""

    return hashlib.sha256(canonical_candidate_bank_bytes(candidates)).hexdigest()


def candidate_protocol_result_sha256(result: CandidateProtocolResult) -> str:
    """Hash every versioned field and recomputed fact in a protocol result."""

    if type(result) is not CandidateProtocolResult:
        raise TypeError("result must be an exact CandidateProtocolResult")
    return hashlib.sha256(canonical_bytes(result.to_dict())).hexdigest()


def _validate_context(
    host: nx.Graph,
    *,
    window_nodes: object,
    frozen_chains: object,
    required_logical_neighbors: object,
    original_focus_chain: object,
) -> tuple[
    Candidate,
    tuple[FrozenChain, ...],
    tuple[int, ...],
    Candidate,
]:
    window = _canonical_nodes(window_nodes, "window_nodes", nonempty=False)
    host_nodes = set(host.nodes)
    missing_window = tuple(sorted(set(window) - host_nodes))
    if missing_window:
        raise ValueError(
            f"window_nodes contains nodes absent from the realized host: {missing_window}"
        )

    if type(frozen_chains) is not tuple:
        raise TypeError("frozen_chains must be an immutable tuple")
    if not all(type(chain) is FrozenChain for chain in frozen_chains):
        raise TypeError("frozen_chains must contain only FrozenChain records")
    frozen = tuple(frozen_chains)
    logical_variables = tuple(chain.logical_variable for chain in frozen)
    if len(logical_variables) != len(set(logical_variables)):
        raise ValueError("frozen_chains must have duplicate-free logical variables")
    if logical_variables != tuple(sorted(logical_variables)):
        raise ValueError("frozen_chains must be sorted by logical_variable")

    owner: dict[int, int] = {}
    for chain in frozen:
        missing = tuple(sorted(set(chain.nodes) - host_nodes))
        if missing:
            raise ValueError(
                f"frozen chain {chain.logical_variable} contains nodes absent from "
                f"the realized host: {missing}"
            )
        if not nx.is_connected(host.subgraph(chain.nodes)):
            raise ValueError(f"frozen chain {chain.logical_variable} must be connected")
        for node in chain.nodes:
            if node in owner:
                raise ValueError(
                    "frozen chains must be pairwise disjoint; "
                    f"node {node} belongs to {owner[node]} and {chain.logical_variable}"
                )
            owner[node] = chain.logical_variable

    required = _canonical_logical_variables(
        required_logical_neighbors,
        "required_logical_neighbors",
    )
    unknown = tuple(sorted(set(required) - set(logical_variables)))
    if unknown:
        raise ValueError(
            f"every required logical neighbor must name a frozen chain; missing={unknown}"
        )
    original = _canonical_nodes(
        original_focus_chain,
        "original_focus_chain",
        nonempty=True,
    )
    return window, frozen, required, original


def _is_connected(host: nx.Graph, candidate: Candidate) -> bool:
    if not candidate or any(node not in host for node in candidate):
        return False
    if len(candidate) == 1:
        return True
    allowed = set(candidate)
    seen = {candidate[0]}
    frontier = [candidate[0]]
    while frontier:
        node = frontier.pop()
        for neighbor in host.neighbors(node):
            if neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == len(candidate)


def _contact_counts(
    host: nx.Graph,
    candidate: Candidate,
    required_chains: tuple[Candidate, ...],
) -> tuple[int, ...]:
    return tuple(
        sum(
            1
            for candidate_node in candidate
            for frozen_node in frozen_chain
            if host.has_edge(candidate_node, frozen_node)
        )
        for frozen_chain in required_chains
    )


def _candidate_predicates(
    host: nx.Graph,
    *,
    candidate: Candidate,
    window: frozenset[int],
    blocked: frozenset[int],
    required_chains: tuple[Candidate, ...],
    l_cap: int,
) -> tuple[bool, bool, bool, bool, bool]:
    candidate_nodes = set(candidate)
    inside_host_and_window = candidate_nodes <= set(host.nodes) and candidate_nodes <= window
    connected = _is_connected(host, candidate)
    disjoint = candidate_nodes.isdisjoint(blocked)
    realizes_couplers = all(
        count > 0 for count in _contact_counts(host, candidate, required_chains)
    )
    within_l_cap = len(candidate) <= l_cap
    enumerable = (
        inside_host_and_window and connected and disjoint and realizes_couplers and within_l_cap
    )
    return connected, disjoint, realizes_couplers, within_l_cap, enumerable


def _largest_free_component(host: nx.Graph, occupied: frozenset[int]) -> tuple[Candidate, int]:
    free_nodes = set(host.nodes) - occupied
    if not free_nodes:
        return (), 0
    free_graph = host.subgraph(free_nodes)
    components = [tuple(sorted(component)) for component in nx.connected_components(free_graph)]
    largest = min(components, key=lambda component: (-len(component), component))
    if len(largest) < 2:
        return largest, 0
    connectivity = nx.edge_connectivity(free_graph.subgraph(largest))
    return largest, int(connectivity)


def _candidate_facts(
    host: nx.Graph,
    *,
    candidate: Candidate,
    frozen: tuple[FrozenChain, ...],
    required_chains: tuple[Candidate, ...],
    l_cap: int,
    q_cap: int | None,
) -> CandidateFacts:
    blocked = frozenset(node for chain in frozen for node in chain.nodes)
    connected = _is_connected(host, candidate)
    disjoint = set(candidate).isdisjoint(blocked)
    contacts = _contact_counts(host, candidate, required_chains)
    realizes_couplers = all(count > 0 for count in contacts)
    within_l_cap = len(candidate) <= l_cap
    minor_valid = connected and disjoint and realizes_couplers and within_l_cap
    current_total_qubits = sum(len(chain.nodes) for chain in frozen) + len(candidate)
    current_maximum_chain_length = max(
        (len(candidate), *(len(chain.nodes) for chain in frozen)),
    )
    within_q_cap = q_cap is None or current_total_qubits <= q_cap
    occupied = blocked | frozenset(candidate)
    largest_nodes, edge_connectivity = _largest_free_component(host, occupied)
    internal_edges = host.subgraph(candidate).number_of_edges()
    return CandidateFacts(
        candidate=candidate,
        connected=connected,
        chain_disjoint=disjoint,
        realizes_every_required_coupler=realizes_couplers,
        within_l_cap=within_l_cap,
        minor_valid=minor_valid,
        current_total_qubits=current_total_qubits,
        current_maximum_chain_length=current_maximum_chain_length,
        within_q_cap=within_q_cap,
        feasible_now=minor_valid and within_q_cap,
        internal_chain_edge_count=internal_edges,
        logical_contact_counts=contacts,
        minimum_logical_contact_count=min(contacts, default=0),
        total_logical_contact_count=sum(contacts),
        sorted_logical_contact_counts=tuple(sorted(contacts)),
        largest_free_component_nodes=largest_nodes,
        largest_free_component_node_numerator=len(largest_nodes),
        largest_free_component_node_denominator=host.number_of_nodes(),
        largest_free_component_edge_connectivity=edge_connectivity,
    )


def _offered_indices(
    candidates: tuple[Candidate, ...],
    facts: tuple[CandidateFacts, ...],
    *,
    original_focus_chain: Candidate,
    candidate_sample_seed_key: bytes,
) -> tuple[int, ...]:
    if len(candidates) <= OFFERED_BANK_CAP:
        return tuple(range(len(candidates)))

    original_index = candidates.index(original_focus_chain)
    selected = {original_index}

    selected.add(
        min(
            range(len(candidates)),
            key=lambda index: (
                len(candidates[index]),
                -facts[index].internal_chain_edge_count,
                index,
            ),
        )
    )
    selected.add(
        max(
            range(len(candidates)),
            key=lambda index: (
                Fraction(
                    facts[index].largest_free_component_node_numerator,
                    facts[index].largest_free_component_node_denominator,
                ),
                facts[index].largest_free_component_edge_connectivity,
                -index,
            ),
        )
    )
    selected.add(
        max(
            range(len(candidates)),
            key=lambda index: (
                facts[index].minimum_logical_contact_count,
                facts[index].total_logical_contact_count,
                facts[index].sorted_logical_contact_counts,
                -index,
            ),
        )
    )

    strata: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (candidate, candidate_facts) in enumerate(zip(candidates, facts, strict=True)):
        if index in selected:
            continue
        key = (
            len(candidate),
            candidate_facts.minimum_logical_contact_count,
            candidate_facts.total_logical_contact_count,
        )
        strata[key].append(index)

    seed_hex = candidate_sample_seed_key.hex()
    for indices in strata.values():
        indices.sort(
            key=lambda index: (
                hashlib.sha256(canonical_bytes([seed_hex, list(candidates[index])])).digest(),
                index,
            )
        )

    stratum_keys = tuple(sorted(strata))
    offsets = {key: 0 for key in stratum_keys}
    while len(selected) < OFFERED_BANK_CAP:
        progress = False
        for key in stratum_keys:
            offset = offsets[key]
            if offset >= len(strata[key]):
                continue
            selected.add(strata[key][offset])
            offsets[key] = offset + 1
            progress = True
            if len(selected) == OFFERED_BANK_CAP:
                break
        if not progress:  # Defensive: a bank larger than 64 makes this unreachable.
            raise RuntimeError("candidate strata exhausted before filling the offered bank")
    return tuple(sorted(selected))


def generate_candidate_bank(
    host: nx.Graph,
    *,
    window_nodes: tuple[int, ...],
    frozen_chains: tuple[FrozenChain, ...],
    required_logical_neighbors: tuple[int, ...],
    original_focus_chain: Candidate,
    l_cap: int,
    candidate_sample_seed_key: bytes,
    q_cap: int | None = None,
) -> CandidateProtocolResult:
    """Enumerate and deterministically truncate one hard/OOD candidate bank.

    ``host`` must be the graph returned by realized-host artifact verification.  This
    function revalidates its graph-level shape and all supplied context, then detaches a
    canonical snapshot so caller mutation cannot affect an in-flight computation.
    ``q_cap`` constrains current frozen-plus-candidate occupancy only; an exact continuation
    protocol must enforce any terminal completion cap separately.
    """

    graph = _verified_graph_snapshot(host)
    checked_l_cap = _require_positive_int(l_cap, "l_cap")
    checked_q_cap = None if q_cap is None else _require_nonnegative_int(q_cap, "q_cap")
    seed_key = _require_seed_key(candidate_sample_seed_key)
    window, frozen, required, original = _validate_context(
        graph,
        window_nodes=window_nodes,
        frozen_chains=frozen_chains,
        required_logical_neighbors=required_logical_neighbors,
        original_focus_chain=original_focus_chain,
    )
    frozen_by_logical = {chain.logical_variable: chain.nodes for chain in frozen}
    required_chains = tuple(frozen_by_logical[logical] for logical in required)
    blocked = frozenset(node for chain in frozen for node in chain.nodes)
    window_set = frozenset(window)

    *_, original_is_enumerable = _candidate_predicates(
        graph,
        candidate=original,
        window=window_set,
        blocked=blocked,
        required_chains=required_chains,
        l_cap=checked_l_cap,
    )
    if not original_is_enumerable:
        return CandidateProtocolResult.invalid_generation()

    eligible_nodes = tuple(node for node in window if node not in blocked)
    candidates: list[Candidate] = []
    for length in range(1, min(checked_l_cap, len(eligible_nodes)) + 1):
        for candidate in itertools.combinations(eligible_nodes, length):
            if not _is_connected(graph, candidate):
                continue
            if not all(count > 0 for count in _contact_counts(graph, candidate, required_chains)):
                continue
            candidates.append(candidate)
            if len(candidates) == ENUMERATION_CAP + 1:
                prefix = tuple(candidates)
                return CandidateProtocolResult.enumeration_aborted(
                    prefix_sha256=candidate_bank_sha256(prefix)
                )

    full = tuple(candidates)
    if original not in full:  # The precheck and traversal must agree byte-for-byte.
        return CandidateProtocolResult.invalid_generation()
    if not full:  # Unreachable with a valid required original; retained as a protocol guard.
        return CandidateProtocolResult(
            attempt_status="no_candidate",
            enumeration_status="complete",
            enumeration_cap=ENUMERATION_CAP,
            enumerated_prefix_count=0,
            enumerated_prefix_sha256=candidate_bank_sha256(()),
            full_candidates=None,
            full_candidate_facts=None,
            full_candidate_count=None,
            full_candidate_bank_sha256=None,
            offered_candidates=None,
            offered_candidate_count=None,
            offered_candidate_bank_sha256=None,
            offered_to_full_indices=None,
        )

    facts = tuple(
        _candidate_facts(
            graph,
            candidate=candidate,
            frozen=frozen,
            required_chains=required_chains,
            l_cap=checked_l_cap,
            q_cap=checked_q_cap,
        )
        for candidate in full
    )
    offered_indices = _offered_indices(
        full,
        facts,
        original_focus_chain=original,
        candidate_sample_seed_key=seed_key,
    )
    offered = tuple(full[index] for index in offered_indices)
    full_digest = candidate_bank_sha256(full)
    return CandidateProtocolResult(
        attempt_status="single_candidate" if len(full) == 1 else "candidate_bank_ready",
        enumeration_status="complete",
        enumeration_cap=ENUMERATION_CAP,
        enumerated_prefix_count=len(full),
        enumerated_prefix_sha256=full_digest,
        full_candidates=full,
        full_candidate_facts=facts,
        full_candidate_count=len(full),
        full_candidate_bank_sha256=full_digest,
        offered_candidates=offered,
        offered_candidate_count=len(offered),
        offered_candidate_bank_sha256=candidate_bank_sha256(offered),
        offered_to_full_indices=offered_indices,
    )


__all__ = [
    "ATTEMPT_STATUSES",
    "CANDIDATE_PROTOCOL_SCHEMA",
    "CANDIDATE_PROTOCOL_SCHEMA_VERSION",
    "ENUMERATION_CAP",
    "ENUMERATION_STATUSES",
    "OFFERED_BANK_CAP",
    "AttemptStatus",
    "Candidate",
    "CandidateFacts",
    "CandidateProtocolResult",
    "EnumerationStatus",
    "FrozenChain",
    "candidate_bank_sha256",
    "candidate_protocol_result_sha256",
    "canonical_candidate_bank_bytes",
    "canonical_candidate_bytes",
    "generate_candidate_bank",
]

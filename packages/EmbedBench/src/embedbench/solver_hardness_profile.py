"""Fail-closed Section 8.3 solver-hardness artifacts.

The scientific profile is an exact canonical-byte artifact.  Wall-clock observations
live in a separate, deliberately noncanonical sidecar and therefore cannot change the
scientific identity.  A heuristic failure is only a failed completed attempt; it is
never represented as a proof of infeasibility.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import networkx as nx

from embedbench.ground_certificate import IsingProblem
from embedbench.hard_ood_protocols import (
    SOLVER_ATTEMPT_COUNT,
    SOLVER_SEED_PROJECTION,
    SOLVER_SEED_SCHEDULE_RULE,
    VerifiedSolverProtocol,
    solver_attempt_seed_schedule,
    solver_attempt_seed_schedule_sha256,
)
from embedbench.hard_ood_provenance import parse_canonical_json_bytes
from embedbench.hard_ood_schema import (
    SeedRequest,
    VerifiedSeedResolver,
    canonical_bytes,
    canonical_sha256,
    require_exact_keys,
    validate_seed_resolver,
)
from embedbench.realized_host import host_graph_sha256

SOLVER_HARDNESS_PROFILE_SCHEMA = "embedbench.solver-hardness-profile"
SOLVER_HARDNESS_PROFILE_SCHEMA_VERSION = 1
SOLVER_RUNTIME_SIDECAR_SCHEMA = "embedbench.solver-runtime-sidecar"
SOLVER_RUNTIME_SIDECAR_SCHEMA_VERSION = 1
SOLVER_EMBEDDING_OUTPUT_SCHEMA = "embedbench.solver-embedding-output"
SOLVER_EMBEDDING_OUTPUT_SCHEMA_VERSION = 1
EXACT_FEASIBILITY_NODE_LIMIT = 50_000_000
EXACT_FEASIBILITY_METHOD = "deterministic-exact-minor-embedding-search-v1"
MAX_SOLVER_OUTPUT_BYTES = 1_048_576

SOLVER_ORDER = ("minorminer", "cpp_baseline")
FAILURE_CATEGORIES = (
    "no_embedding_returned",
    "invalid_embedding",
    "watchdog",
    "process_launch_failure",
    "process_nonzero_exit",
    "process_signal",
    "malformed_output",
    "executor_attestation_failure",
)
HEURISTIC_FAILURE_CATEGORIES = frozenset(
    {
        "no_embedding_returned",
        "invalid_embedding",
    }
)
ENVIRONMENT_FAILURE_CATEGORIES = frozenset(
    {
        "watchdog",
        "process_launch_failure",
        "process_nonzero_exit",
        "process_signal",
        "malformed_output",
        "executor_attestation_failure",
    }
)

_PROFILE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "partition",
        "problem_sha256",
        "host_graph_sha256",
        "n_vars",
        "realized_host_node_count",
        "seed_registry_terminal_root_sha256",
        "seed_schedule",
        "solver_runs",
        "exact_feasibility",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "rule",
        "projection",
        "attempt_count",
        "schedule_sha256",
        "requests",
        "request_key_sha256s",
        "registry_entry_sha256s",
        "seed32s",
    }
)
_RUN_FIELDS = frozenset(
    {
        "solver_id",
        "solver_protocol_sha256",
        "solver_protocol",
        "seed_schedule_sha256",
        "request_key_sha256s",
        "registry_entry_sha256s",
        "attempts",
        "accounting",
        "success_resource_distributions",
    }
)
_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_index",
        "request_key_sha256",
        "registry_entry_sha256",
        "seed32",
        "process_evidence",
        "output_sha256",
        "output_bytes_hex",
    }
)
_ACCOUNTING_FIELDS = frozenset(
    {
        "planned_attempt_count",
        "completed_attempt_count",
        "success_count",
        "heuristic_failure_count",
        "environment_failure_count",
        "success_rate",
        "failure_category_counts",
    }
)
_SUCCESS_RATE_FIELDS = frozenset({"numerator", "denominator"})
_FAILURE_COUNT_FIELDS = frozenset({"category", "count"})
_RESOURCE_DISTRIBUTION_FIELDS = frozenset(
    {
        "total_qubits",
        "maximum_chain_length",
        "sum_squared_chain_lengths",
    }
)
_EXACT_FEASIBILITY_FIELDS = frozenset(
    {
        "requirement",
        "method",
        "node_limit",
        "status",
        "nodes_visited",
        "evidence_sha256",
    }
)
_PROCESS_EVIDENCE_FIELDS = frozenset(
    {
        "executor_attested",
        "termination_mode",
        "exit_code",
        "signal_number",
        "stdout_sha256",
        "stderr_sha256",
    }
)
_OUTPUT_FIELDS = frozenset({"schema", "schema_version", "result", "embedding"})
_CHAIN_FIELDS = frozenset({"logical_variable", "chain"})
_SIDECAR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "profile_sha256",
        "solver_observations",
    }
)
_SOLVER_OBSERVATION_FIELDS = frozenset({"solver_id", "attempts"})
_RUNTIME_OBSERVATION_FIELDS = frozenset(
    {
        "attempt_index",
        "request_key_sha256",
        "runtime_seconds",
        "termination_mode",
    }
)
_SHA256_ALPHABET = frozenset("0123456789abcdef")
_PROFILE_SEAL = object()
_HOST_SNAPSHOT_SEAL = object()
_EXACT_EVIDENCE_SEAL = object()
_SOLVER_EXECUTION_EVIDENCE_SEAL = object()
_RUNTIME_SIDECAR_SEAL = object()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _require_uint32(value: object, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result >= 2**32:
        raise ValueError(f"{name} must be an unsigned 32-bit integer")
    return result


def _require_uint64(value: object, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result >= 2**64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return result


def _require_exact_value(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} does not match the registered value")


def _require_digest_list(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not list or len(value) != SOLVER_ATTEMPT_COUNT:
        raise ValueError(f"{name} must contain exactly {SOLVER_ATTEMPT_COUNT} digests")
    return tuple(_require_sha256(item, f"{name} item") for item in value)


def _require_int_list(value: object, name: str, *, unsigned32: bool = False) -> tuple[int, ...]:
    if type(value) is not list or len(value) != SOLVER_ATTEMPT_COUNT:
        raise ValueError(f"{name} must contain exactly {SOLVER_ATTEMPT_COUNT} integers")
    validator = _require_uint32 if unsigned32 else _require_nonnegative_int
    return tuple(validator(item, f"{name} item") for item in value)


class ExactFeasibilityEvidenceUnavailableError(RuntimeError):
    """The mandatory exact profile cannot be trusted without an upstream verifier."""


class SolverExecutionEvidenceUnavailableError(RuntimeError):
    """Solver outcomes cannot be trusted until a real execution verifier exists."""


@dataclass(frozen=True, slots=True, init=False)
class VerifiedRealizedHostGraphSnapshot:
    """Immutable realized-host graph authenticated by an external graph digest."""

    host_graph_sha256: str
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    _seal: object

    def __new__(cls) -> VerifiedRealizedHostGraphSnapshot:
        raise TypeError("realized-host snapshots must be produced by the verifier")

    def graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        return graph


@dataclass(frozen=True, slots=True, init=False)
class VerifiedExactFeasibilityEvidence:
    """Placeholder for a future upstream replayed exact-feasibility receipt.

    There is intentionally no producer in this module.  Accepting a constructor or a
    status-to-capsule helper would merely rename the unsupported naked claim.
    """

    evidence_sha256: str
    problem_sha256: str
    host_graph_sha256: str
    method: str
    node_limit: int
    status: str
    nodes_visited: int
    _seal: object

    def __new__(cls) -> VerifiedExactFeasibilityEvidence:
        raise TypeError(
            "exact-feasibility evidence must be produced by an upstream exact-feasibility verifier"
        )


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSolverExecutionEvidence:
    """Placeholder for evidence produced by a future solver-execution verifier.

    Structural parsing of a profile proves that its outputs are valid embeddings and
    that its aggregates are self-consistent.  It does not prove that the registered
    binaries produced those outputs.  Consequently this module deliberately has no
    constructor or producer for this capsule.  A real runner must authenticate the
    exact run receipts before a scientific profile can be emitted.
    """

    evidence_sha256: str
    release_id: str
    partition: str
    problem_sha256: str
    host_graph_sha256: str
    seed_registry_terminal_root_sha256: str
    seed_schedule_sha256: str
    solver_protocol_sha256s: tuple[str, str]
    solver_runs_sha256: str
    _seal: object

    def __new__(cls) -> VerifiedSolverExecutionEvidence:
        raise TypeError("solver-execution evidence must be produced by a real execution verifier")


def verify_realized_host_graph_snapshot(
    graph: nx.Graph,
    *,
    expected_host_graph_sha256: str,
) -> VerifiedRealizedHostGraphSnapshot:
    """Detach one simple host graph and bind the immutable copy to an external digest."""

    if type(graph) is not nx.Graph:
        raise TypeError("realized host must be an exact undirected simple NetworkX Graph")
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("realized host must be an exact undirected simple NetworkX Graph")
    nodes = tuple(sorted(_require_uint64(node, "host node") for node in graph.nodes))
    if len(nodes) != graph.number_of_nodes():
        raise ValueError("realized host contains duplicate normalized nodes")
    edges = tuple(
        sorted(
            (min(left, right), max(left, right))
            for raw_left, raw_right in graph.edges
            for left, right in (
                (
                    _require_uint64(raw_left, "host edge endpoint"),
                    _require_uint64(raw_right, "host edge endpoint"),
                ),
            )
        )
    )
    if any(left == right for left, right in edges):
        raise ValueError("realized host must not contain self-loops")
    if len(edges) != len(set(edges)):
        raise ValueError("realized host must not contain parallel edges")
    detached = nx.Graph()
    detached.add_nodes_from(nodes)
    detached.add_edges_from(edges)
    digest = host_graph_sha256(detached)
    expected = _require_sha256(expected_host_graph_sha256, "expected host graph SHA-256")
    if digest != expected:
        raise ValueError("realized host graph disagrees with its external host graph digest")
    verified = object.__new__(VerifiedRealizedHostGraphSnapshot)
    object.__setattr__(verified, "host_graph_sha256", digest)
    object.__setattr__(verified, "nodes", nodes)
    object.__setattr__(verified, "edges", edges)
    object.__setattr__(verified, "_seal", _HOST_SNAPSHOT_SEAL)
    return verified


def validate_realized_host_graph_snapshot(
    value: object,
    *,
    expected_host_graph_sha256: str,
) -> VerifiedRealizedHostGraphSnapshot:
    """Replay an immutable host snapshot against an independent graph commitment."""

    if type(value) is not VerifiedRealizedHostGraphSnapshot:
        raise TypeError("realized_host must be an exact verified host snapshot")
    snapshot = value
    if getattr(snapshot, "_seal", None) is not _HOST_SNAPSHOT_SEAL:
        raise TypeError("realized host snapshot must be produced by its verifier")
    replayed = verify_realized_host_graph_snapshot(
        snapshot.graph(),
        expected_host_graph_sha256=expected_host_graph_sha256,
    )
    if (
        snapshot.host_graph_sha256 != replayed.host_graph_sha256
        or snapshot.nodes != replayed.nodes
        or snapshot.edges != replayed.edges
    ):
        raise ValueError("verified host snapshot fields disagree with its immutable graph")
    return snapshot


def _validate_problem(problem: object, *, expected_problem_sha256: str) -> IsingProblem:
    if type(problem) is not IsingProblem:
        raise TypeError("problem must be an exact immutable IsingProblem")
    try:
        snapshot = IsingProblem(
            variables=tuple(problem.variables),
            linear=tuple(problem.linear),
            quadratic=tuple(problem.quadratic),
        )
    except AttributeError as error:
        raise TypeError("problem must be a complete immutable IsingProblem") from error
    expected = _require_sha256(expected_problem_sha256, "expected problem SHA-256")
    if snapshot.problem_sha256 != expected:
        raise ValueError("problem digest disagrees with its external problem digest")
    if snapshot != problem:
        raise ValueError("IsingProblem fields disagree with their validated snapshot")
    return snapshot


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSolverHardnessProfile:
    """A profile capsule authenticated by independently supplied commitments."""

    release_id: str
    partition: str
    problem_sha256: str
    host_graph_sha256: str
    n_vars: int
    realized_host_node_count: int
    profile_sha256: str
    seed_registry_terminal_root_sha256: str
    seed_schedule_sha256: str
    solver_execution_evidence_sha256: str
    _payload: bytes
    _problem: IsingProblem
    _realized_host: VerifiedRealizedHostGraphSnapshot
    _seed_resolver: VerifiedSeedResolver
    _solver_protocols: tuple[VerifiedSolverProtocol, VerifiedSolverProtocol]
    _solver_protocol_sha256s: tuple[str, str]
    _attempt_outcomes: tuple[tuple[str, ...], tuple[str, ...]]
    _attempt_termination_modes: tuple[tuple[str, ...], tuple[str, ...]]
    _solver_execution_evidence: VerifiedSolverExecutionEvidence
    _exact_feasibility_evidence: VerifiedExactFeasibilityEvidence | None
    _exact_feasibility_evidence_sha256: str | None
    _seal: object

    def __new__(cls) -> VerifiedSolverHardnessProfile:
        raise TypeError("solver-hardness profiles must be produced by the verifier")

    def to_bytes(
        self,
        *,
        expected_profile_sha256: str,
        expected_release_id: str,
        expected_partition: str,
        expected_problem_sha256: str,
        expected_host_graph_sha256: str,
        expected_solver_execution_evidence_sha256: str,
    ) -> bytes:
        validate_solver_hardness_profile(
            self,
            expected_profile_sha256=expected_profile_sha256,
            expected_release_id=expected_release_id,
            expected_partition=expected_partition,
            expected_problem_sha256=expected_problem_sha256,
            expected_host_graph_sha256=expected_host_graph_sha256,
            expected_solver_execution_evidence_sha256=(expected_solver_execution_evidence_sha256),
        )
        return bytes(self._payload)

    def to_dict(
        self,
        *,
        expected_profile_sha256: str,
        expected_release_id: str,
        expected_partition: str,
        expected_problem_sha256: str,
        expected_host_graph_sha256: str,
        expected_solver_execution_evidence_sha256: str,
    ) -> dict[str, object]:
        return parse_canonical_json_bytes(
            self.to_bytes(
                expected_profile_sha256=expected_profile_sha256,
                expected_release_id=expected_release_id,
                expected_partition=expected_partition,
                expected_problem_sha256=expected_problem_sha256,
                expected_host_graph_sha256=expected_host_graph_sha256,
                expected_solver_execution_evidence_sha256=(
                    expected_solver_execution_evidence_sha256
                ),
            ),
            name="verified solver-hardness profile",
        )


@dataclass(frozen=True, slots=True, init=False)
class ValidatedSolverRuntimeSidecar:
    """Validated runtime provenance that is intentionally outside profile identity."""

    profile_sha256: str
    success_runtime_distributions: Mapping[str, tuple[float, ...]]
    _seal: object

    def __new__(cls) -> ValidatedSolverRuntimeSidecar:
        raise TypeError("runtime sidecars must be produced by the validator")


def _validate_protocol_pair(
    solver_protocols: object,
    expected_solver_protocol_sha256s: object,
    *,
    expected_release_id: str,
) -> tuple[
    tuple[VerifiedSolverProtocol, VerifiedSolverProtocol],
    tuple[str, str],
    tuple[dict[str, object], dict[str, object]],
]:
    if type(solver_protocols) is not tuple or len(solver_protocols) != 2:
        raise TypeError("solver_protocols must be an exact two-item tuple")
    if any(type(protocol) is not VerifiedSolverProtocol for protocol in solver_protocols):
        raise TypeError("solver protocols must be exact VerifiedSolverProtocol objects")
    protocols = cast(tuple[VerifiedSolverProtocol, VerifiedSolverProtocol], solver_protocols)
    try:
        solver_ids = tuple(protocol.solver_id for protocol in protocols)
    except AttributeError as error:
        raise TypeError("solver protocol must be a complete verified solver protocol") from error
    if solver_ids != SOLVER_ORDER:
        raise ValueError(
            "solver protocols must be exactly the minorminer and cpp_baseline pair in order"
        )
    if (
        type(expected_solver_protocol_sha256s) is not tuple
        or len(expected_solver_protocol_sha256s) != 2
    ):
        raise TypeError("expected solver protocol roots must be an exact two-item tuple")
    roots = cast(tuple[object, object], expected_solver_protocol_sha256s)
    protocol_roots = (
        _require_sha256(roots[0], "expected minorminer protocol SHA-256"),
        _require_sha256(roots[1], "expected cpp_baseline protocol SHA-256"),
    )
    try:
        documents = tuple(
            protocol.to_dict(
                expected_protocol_sha256=protocol_sha256,
                expected_release_id=expected_release_id,
            )
            for protocol, protocol_sha256 in zip(protocols, protocol_roots, strict=True)
        )
    except AttributeError as error:
        raise TypeError("solver protocol must be a complete verified solver protocol") from error
    return protocols, protocol_roots, cast(tuple[dict[str, object], dict[str, object]], documents)


def _validate_schedule(
    value: object,
    *,
    release_id: str,
    partition: str,
    problem_sha256: str,
    resolver: VerifiedSeedResolver,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    schedule = require_exact_keys(value, _SCHEDULE_FIELDS, "seed_schedule")
    _require_exact_value(schedule["rule"], SOLVER_SEED_SCHEDULE_RULE, "seed schedule rule")
    _require_exact_value(schedule["projection"], SOLVER_SEED_PROJECTION, "seed projection")
    _require_exact_value(schedule["attempt_count"], SOLVER_ATTEMPT_COUNT, "attempt_count")

    raw_requests = schedule["requests"]
    if type(raw_requests) is not list or len(raw_requests) != SOLVER_ATTEMPT_COUNT:
        raise ValueError("seed schedule must contain exactly 32 complete requests")
    requests = tuple(SeedRequest.from_dict(item) for item in raw_requests)
    expected_requests = solver_attempt_seed_schedule(
        release_id=release_id,
        partition=partition,
        problem_sha256=problem_sha256,
    )
    if tuple(request.canonical_preimage() for request in requests) != tuple(
        request.canonical_preimage() for request in expected_requests
    ):
        raise ValueError("seed schedule is not the registered solver-neutral schedule")
    schedule_sha256 = solver_attempt_seed_schedule_sha256(requests)
    if _require_sha256(schedule["schedule_sha256"], "schedule digest") != schedule_sha256:
        raise ValueError("seed schedule digest disagrees with the complete ordered requests")

    entries = resolver.resolve_many(requests)
    for entry in entries:
        if entry.seed32_required is not True or entry.seed32 is None:
            raise ValueError("every solver request must have seed32_required=true")
        if entry.collision_counter is None:
            raise ValueError("every solver request needs collision-managed seed32 provenance")
    expected_request_keys = tuple(request.seed_key_hex for request in requests)
    expected_entry_digests = tuple(canonical_sha256(entry.to_dict()) for entry in entries)
    expected_seed32s = tuple(cast(int, entry.seed32) for entry in entries)
    request_keys = _require_digest_list(schedule["request_key_sha256s"], "request keys")
    entry_digests = _require_digest_list(
        schedule["registry_entry_sha256s"], "registry-entry digests"
    )
    seed32s = _require_int_list(schedule["seed32s"], "seed32 list", unsigned32=True)
    if request_keys != expected_request_keys:
        raise ValueError("ordered request keys disagree with the registered schedule")
    if entry_digests != expected_entry_digests:
        raise ValueError("ordered registry-entry digests disagree with the verified registry")
    if seed32s != expected_seed32s:
        raise ValueError("ordered seed32 list disagrees with collision-managed registry entries")
    return schedule_sha256, request_keys, entry_digests, seed32s


@dataclass(frozen=True, slots=True)
class _AttemptResult:
    outcome: str
    failure_category: str | None
    resources: tuple[int, int, int] | None
    runtime_termination_mode: str


def _validate_process_evidence(value: object) -> tuple[str | None, str]:
    evidence = require_exact_keys(value, _PROCESS_EVIDENCE_FIELDS, "process evidence")
    attested = evidence["executor_attested"]
    if type(attested) is not bool:
        raise ValueError("process evidence executor_attested must be a boolean")
    mode = _require_text(evidence["termination_mode"], "process evidence termination_mode")
    exit_code = evidence["exit_code"]
    signal_number = evidence["signal_number"]
    stdout_sha256 = evidence["stdout_sha256"]
    stderr_sha256 = evidence["stderr_sha256"]
    if mode == "executor_attestation_failure":
        if attested or any(
            item is not None for item in (exit_code, signal_number, stdout_sha256, stderr_sha256)
        ):
            raise ValueError("executor-attestation process evidence is incoherent")
        return "executor_attestation_failure", "launch_failure"
    if not attested:
        raise ValueError("unattested process evidence must be an executor_attestation_failure")
    if mode == "launch_failure":
        if any(
            item is not None for item in (exit_code, signal_number, stdout_sha256, stderr_sha256)
        ):
            raise ValueError("launch-failure process evidence is incoherent")
        return "process_launch_failure", "launch_failure"
    stdout = _require_sha256(stdout_sha256, "process stdout digest")
    stderr = _require_sha256(stderr_sha256, "process stderr digest")
    if not stdout or not stderr:  # pragma: no cover - digest validator is definitive
        raise AssertionError("unreachable empty digest")
    if mode == "watchdog":
        if exit_code is not None or signal_number is not None:
            raise ValueError("watchdog process evidence requires null exit_code and signal_number")
        return "watchdog", "watchdog"
    if mode == "signal":
        if exit_code is not None:
            raise ValueError("signal process evidence requires a null exit_code")
        signal = _require_positive_int(signal_number, "process signal_number")
        if signal > 255:
            raise ValueError("process signal_number must be at most 255")
        return "process_signal", "process_signal"
    if mode != "exited":
        raise ValueError("process evidence termination_mode is not registered")
    if signal_number is not None:
        raise ValueError("exited process evidence requires a null signal_number")
    code = _require_nonnegative_int(exit_code, "process exit_code")
    if code > 255:
        raise ValueError("process exit_code must be at most 255")
    return (None if code == 0 else "process_nonzero_exit"), "completed"


def _validated_output_bytes(attempt: Mapping[str, object]) -> bytes | None:
    raw_digest = attempt["output_sha256"]
    raw_hex = attempt["output_bytes_hex"]
    if raw_digest is None or raw_hex is None:
        if raw_digest is not None or raw_hex is not None:
            raise ValueError("output digest and output bytes must both be present or both be null")
        return None
    digest = _require_sha256(raw_digest, "output digest")
    if type(raw_hex) is not str or not raw_hex or raw_hex != raw_hex.lower():
        raise ValueError("output_bytes_hex must be nonempty lowercase hexadecimal")
    try:
        payload = bytes.fromhex(raw_hex)
    except ValueError as error:
        raise ValueError("output_bytes_hex must be nonempty lowercase hexadecimal") from error
    if payload.hex() != raw_hex:
        raise ValueError("output_bytes_hex must use exact lowercase hexadecimal")
    if len(payload) > MAX_SOLVER_OUTPUT_BYTES:
        raise ValueError("solver output exceeds the registered byte limit")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("output digest disagrees with the retained output bytes")
    return payload


def _validate_embedding(
    value: object,
    *,
    problem: IsingProblem,
    host: VerifiedRealizedHostGraphSnapshot,
) -> tuple[int, int, int]:
    if type(value) is not list:
        raise ValueError("embedding must be a JSON array")
    chains: dict[int, tuple[int, ...]] = {}
    for index, raw_chain in enumerate(value):
        item = require_exact_keys(raw_chain, _CHAIN_FIELDS, f"embedding chain {index}")
        variable = item["logical_variable"]
        if type(variable) is not int:
            raise ValueError("embedding logical_variable must be an integer")
        raw_nodes = item["chain"]
        if type(raw_nodes) is not list or not raw_nodes:
            raise ValueError("embedding chain must be a nonempty JSON array")
        nodes = tuple(_require_uint64(node, "embedding chain node") for node in raw_nodes)
        if nodes != tuple(sorted(set(nodes))):
            raise ValueError("embedding chain nodes must be sorted and duplicate-free")
        if variable in chains:
            raise ValueError("embedding contains a duplicate logical variable")
        chains[variable] = nodes
    if tuple(chains) != problem.variables or len(chains) != len(problem.variables):
        raise ValueError("embedding is incomplete or not in canonical logical-variable order")
    host_nodes = set(host.nodes)
    host_edges = set(host.edges)
    owner: dict[int, int] = {}
    for variable, chain in chains.items():
        missing = set(chain) - host_nodes
        if missing:
            raise ValueError("embedding uses a qubit outside the realized host")
        for node in chain:
            if node in owner:
                raise ValueError("embedding chains overlap")
            owner[node] = variable
        reached = {chain[0]}
        frontier = [chain[0]]
        chain_set = set(chain)
        while frontier:
            current = frontier.pop()
            for left, right in host_edges:
                neighbor: int | None = None
                if left == current and right in chain_set:
                    neighbor = right
                elif right == current and left in chain_set:
                    neighbor = left
                if neighbor is not None and neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        if reached != chain_set:
            raise ValueError("embedding contains a disconnected chain")
    for left, right, _coefficient in problem.quadratic:
        if not any(
            (min(left_node, right_node), max(left_node, right_node)) in host_edges
            for left_node in chains[left]
            for right_node in chains[right]
        ):
            raise ValueError("embedding is missing a required logical-edge contact")
    lengths = tuple(len(chains[variable]) for variable in problem.variables)
    return sum(lengths), max(lengths), sum(length * length for length in lengths)


def _classify_completed_output(
    payload: bytes | None,
    *,
    problem: IsingProblem,
    host: VerifiedRealizedHostGraphSnapshot,
) -> _AttemptResult:
    if payload is None:
        return _AttemptResult("environment_failure", "malformed_output", None, "completed")
    try:
        output = parse_canonical_json_bytes(payload, name="solver embedding output")
        document = require_exact_keys(output, _OUTPUT_FIELDS, "solver embedding output")
        _require_exact_value(
            document["schema"], SOLVER_EMBEDDING_OUTPUT_SCHEMA, "solver output schema"
        )
        _require_exact_value(
            document["schema_version"],
            SOLVER_EMBEDDING_OUTPUT_SCHEMA_VERSION,
            "solver output schema_version",
        )
        result = _require_text(document["result"], "solver output result")
    except (TypeError, ValueError):
        return _AttemptResult("environment_failure", "malformed_output", None, "completed")
    if result == "no_embedding":
        if document["embedding"] is not None:
            return _AttemptResult("environment_failure", "malformed_output", None, "completed")
        return _AttemptResult("heuristic_failure", "no_embedding_returned", None, "completed")
    if result != "embedding":
        return _AttemptResult("environment_failure", "malformed_output", None, "completed")
    try:
        resources = _validate_embedding(document["embedding"], problem=problem, host=host)
    except (TypeError, ValueError):
        return _AttemptResult("heuristic_failure", "invalid_embedding", None, "completed")
    return _AttemptResult("success", None, resources, "completed")


def _validate_attempt(
    value: object,
    *,
    index: int,
    problem: IsingProblem,
    host: VerifiedRealizedHostGraphSnapshot,
    request_key: str,
    entry_digest: str,
    seed32: int,
) -> _AttemptResult:
    attempt = require_exact_keys(value, _ATTEMPT_FIELDS, f"attempt {index}")
    _require_exact_value(attempt["attempt_index"], index, f"attempt {index} attempt_index")
    _require_exact_value(attempt["request_key_sha256"], request_key, f"attempt {index} request key")
    _require_exact_value(
        attempt["registry_entry_sha256"], entry_digest, f"attempt {index} registry entry"
    )
    _require_exact_value(attempt["seed32"], seed32, f"attempt {index} seed32")
    process_failure, runtime_mode = _validate_process_evidence(attempt["process_evidence"])
    output_bytes = _validated_output_bytes(attempt)
    if process_failure is None:
        return _classify_completed_output(output_bytes, problem=problem, host=host)
    if process_failure in {"executor_attestation_failure", "process_launch_failure"} and (
        output_bytes is not None
    ):
        raise ValueError("a process that was not launched cannot carry solver output bytes")
    return _AttemptResult("environment_failure", process_failure, None, runtime_mode)


def _validate_failure_counts(
    value: object,
    *,
    observed: Mapping[str, int],
) -> None:
    if type(value) is not list or len(value) != len(FAILURE_CATEGORIES):
        raise ValueError("failure_category_counts must cover every registered category")
    for expected_category, raw_item in zip(FAILURE_CATEGORIES, value, strict=True):
        item = require_exact_keys(raw_item, _FAILURE_COUNT_FIELDS, "failure category count")
        _require_exact_value(item["category"], expected_category, "failure category order")
        count = _require_nonnegative_int(item["count"], "failure category count")
        if count != observed[expected_category]:
            raise ValueError("failure category count disagrees with attempts")


def _validate_distribution(value: object, expected: list[int], name: str) -> None:
    if type(value) is not list:
        raise ValueError(f"{name} resource distribution must be a JSON array")
    observed = [_require_positive_int(item, name) for item in value]
    if observed != sorted(observed):
        raise ValueError(f"{name} resource distribution must be sorted")
    if observed != sorted(expected):
        raise ValueError(f"{name} resource distribution disagrees with successful attempts")


def _validate_accounting_and_resources(
    accounting_value: object,
    resources_value: object,
    *,
    outcomes: tuple[str, ...],
    failure_categories: tuple[str | None, ...],
    resources: tuple[tuple[int, int, int] | None, ...],
) -> None:
    success_count = outcomes.count("success")
    heuristic_failure_count = outcomes.count("heuristic_failure")
    environment_failure_count = outcomes.count("environment_failure")
    completed_attempt_count = success_count + heuristic_failure_count
    accounting = require_exact_keys(accounting_value, _ACCOUNTING_FIELDS, "solver accounting")
    expected_counts = {
        "planned_attempt_count": SOLVER_ATTEMPT_COUNT,
        "completed_attempt_count": completed_attempt_count,
        "success_count": success_count,
        "heuristic_failure_count": heuristic_failure_count,
        "environment_failure_count": environment_failure_count,
    }
    for field, expected in expected_counts.items():
        observed = _require_nonnegative_int(accounting[field], field)
        if observed != expected:
            raise ValueError(f"{field} disagrees with attempt outcomes")
    if completed_attempt_count == 0:
        if accounting["success_rate"] is not None:
            raise ValueError("success_rate must be null when no scientific attempt completed")
    else:
        rate = require_exact_keys(accounting["success_rate"], _SUCCESS_RATE_FIELDS, "success_rate")
        numerator = _require_nonnegative_int(rate["numerator"], "success-rate numerator")
        denominator = _require_positive_int(rate["denominator"], "success-rate denominator")
        if numerator != success_count:
            raise ValueError("success-rate numerator disagrees with success_count")
        if denominator != completed_attempt_count:
            raise ValueError(
                "success-rate denominator must exclude environment failures from infeasibility"
            )

    observed_failures = {category: 0 for category in FAILURE_CATEGORIES}
    for failure in failure_categories:
        if failure is not None:
            observed_failures[failure] += 1
    _validate_failure_counts(accounting["failure_category_counts"], observed=observed_failures)

    resource_distributions = require_exact_keys(
        resources_value,
        _RESOURCE_DISTRIBUTION_FIELDS,
        "success_resource_distributions",
    )
    successful_resources = [item for item in resources if item is not None]
    _validate_distribution(
        resource_distributions["total_qubits"],
        [item[0] for item in successful_resources],
        "total_qubits",
    )
    _validate_distribution(
        resource_distributions["maximum_chain_length"],
        [item[1] for item in successful_resources],
        "maximum_chain_length",
    )
    _validate_distribution(
        resource_distributions["sum_squared_chain_lengths"],
        [item[2] for item in successful_resources],
        "sum_squared_chain_lengths",
    )


def _validate_solver_run(
    value: object,
    *,
    expected_solver_id: str,
    protocol_sha256: str,
    protocol_document: dict[str, object],
    schedule_sha256: str,
    request_keys: tuple[str, ...],
    entry_digests: tuple[str, ...],
    seed32s: tuple[int, ...],
    problem: IsingProblem,
    host: VerifiedRealizedHostGraphSnapshot,
) -> tuple[_AttemptResult, ...]:
    run = require_exact_keys(value, _RUN_FIELDS, f"{expected_solver_id} run receipt")
    _require_exact_value(run["solver_id"], expected_solver_id, "solver_id")
    if _require_sha256(run["solver_protocol_sha256"], "solver protocol digest") != (
        protocol_sha256
    ):
        raise ValueError("solver protocol digest disagrees with its external commitment")
    if canonical_bytes(run["solver_protocol"]) != canonical_bytes(protocol_document):
        raise ValueError("solver protocol snapshot disagrees with its verified protocol")
    if _require_sha256(run["seed_schedule_sha256"], "run schedule digest") != schedule_sha256:
        raise ValueError("run receipt schedule digest disagrees with shared schedule digest")
    if _require_digest_list(run["request_key_sha256s"], "run ordered request keys") != request_keys:
        raise ValueError("run receipt ordered request keys disagree with shared schedule")
    if (
        _require_digest_list(run["registry_entry_sha256s"], "run registry-entry digests")
        != entry_digests
    ):
        raise ValueError("run receipt registry-entry digests disagree with shared schedule")
    attempts_value = run["attempts"]
    if type(attempts_value) is not list or len(attempts_value) != SOLVER_ATTEMPT_COUNT:
        raise ValueError("solver run must contain exactly 32 attempt receipts")
    checked_attempts = tuple(
        _validate_attempt(
            raw_attempt,
            index=index,
            problem=problem,
            host=host,
            request_key=request_keys[index],
            entry_digest=entry_digests[index],
            seed32=seed32s[index],
        )
        for index, raw_attempt in enumerate(attempts_value)
    )
    _validate_accounting_and_resources(
        run["accounting"],
        run["success_resource_distributions"],
        outcomes=tuple(item.outcome for item in checked_attempts),
        failure_categories=tuple(item.failure_category for item in checked_attempts),
        resources=tuple(item.resources for item in checked_attempts),
    )
    return checked_attempts


def _validate_solver_execution_evidence(
    evidence: VerifiedSolverExecutionEvidence | None,
    *,
    expected_evidence_sha256: str | None,
    release_id: str,
    partition: str,
    problem_sha256: str,
    host_graph_sha256: str,
    seed_registry_terminal_root_sha256: str,
    seed_schedule_sha256: str,
    solver_protocol_sha256s: tuple[str, str],
    solver_runs_sha256: str,
) -> tuple[VerifiedSolverExecutionEvidence, str]:
    """Require an execution-verifier capsule before trusting structural receipts."""

    if evidence is not None and type(evidence) is not VerifiedSolverExecutionEvidence:
        raise TypeError("solver-execution evidence must come from a real execution verifier")
    if evidence is not None and (
        getattr(evidence, "_seal", None) is not _SOLVER_EXECUTION_EVIDENCE_SEAL
    ):
        raise TypeError("solver-execution evidence must come from a real execution verifier")
    if evidence is None or expected_evidence_sha256 is None:
        raise SolverExecutionEvidenceUnavailableError(
            "real solver-execution evidence is required but its runner/verifier is not implemented"
        )
    external_evidence_sha256 = _require_sha256(
        expected_evidence_sha256,
        "expected solver-execution evidence SHA-256",
    )
    try:
        observed_protocol_roots = evidence.solver_protocol_sha256s
        if type(observed_protocol_roots) is not tuple or len(observed_protocol_roots) != 2:
            raise ValueError("solver-execution evidence has invalid protocol roots")
        checked_protocol_roots = (
            _require_sha256(
                observed_protocol_roots[0],
                "execution-evidence minorminer protocol SHA-256",
            ),
            _require_sha256(
                observed_protocol_roots[1],
                "execution-evidence cpp protocol SHA-256",
            ),
        )
        agrees = (
            evidence.evidence_sha256 == external_evidence_sha256
            and evidence.release_id == release_id
            and evidence.partition == partition
            and evidence.problem_sha256 == problem_sha256
            and evidence.host_graph_sha256 == host_graph_sha256
            and evidence.seed_registry_terminal_root_sha256 == seed_registry_terminal_root_sha256
            and evidence.seed_schedule_sha256 == seed_schedule_sha256
            and checked_protocol_roots == solver_protocol_sha256s
            and evidence.solver_runs_sha256 == solver_runs_sha256
        )
    except AttributeError as error:
        raise TypeError(
            "solver-execution evidence must be complete and come from a real execution verifier"
        ) from error
    if not agrees:
        raise ValueError("solver profile disagrees with verified solver-execution evidence")
    return evidence, external_evidence_sha256


def _validate_exact_feasibility(
    value: object,
    *,
    problem: IsingProblem,
    host: VerifiedRealizedHostGraphSnapshot,
    evidence: VerifiedExactFeasibilityEvidence | None,
    expected_evidence_sha256: str | None,
) -> None:
    exact = require_exact_keys(value, _EXACT_FEASIBILITY_FIELDS, "exact feasibility profile")
    is_required = len(problem.variables) <= 16 and len(host.nodes) <= 128
    if not is_required:
        expected = {
            "requirement": "not_required",
            "method": None,
            "node_limit": None,
            "status": "not_run",
            "nodes_visited": None,
            "evidence_sha256": None,
        }
        if canonical_bytes(exact) != canonical_bytes(expected):
            raise ValueError("exact feasibility must be not_run outside the registered envelope")
        if evidence is not None or expected_evidence_sha256 is not None:
            raise ValueError("exact-feasibility evidence is forbidden outside its envelope")
        return
    _require_exact_value(exact["requirement"], "required", "exact feasibility requirement")
    _require_exact_value(exact["method"], EXACT_FEASIBILITY_METHOD, "exact feasibility method")
    _require_exact_value(
        exact["node_limit"], EXACT_FEASIBILITY_NODE_LIMIT, "exact feasibility node limit"
    )
    status = _require_text(exact["status"], "exact feasibility status")
    if status not in {"feasible", "infeasible", "node_budget_exhausted"}:
        raise ValueError("exact feasibility status must not describe a timeout")
    nodes_visited = _require_nonnegative_int(
        exact["nodes_visited"], "exact feasibility nodes_visited"
    )
    if nodes_visited > EXACT_FEASIBILITY_NODE_LIMIT:
        raise ValueError("exact feasibility nodes_visited exceeds its frozen node limit")
    if status == "node_budget_exhausted" and nodes_visited != EXACT_FEASIBILITY_NODE_LIMIT:
        raise ValueError(
            "exact feasibility node_budget_exhausted requires consuming the exact node limit"
        )
    claimed_evidence_sha256 = _require_sha256(
        exact["evidence_sha256"], "exact-feasibility evidence digest"
    )
    if evidence is not None and type(evidence) is not VerifiedExactFeasibilityEvidence:
        raise TypeError(
            "exact-feasibility evidence must come from an upstream exact-feasibility verifier"
        )
    if evidence is not None and getattr(evidence, "_seal", None) is not _EXACT_EVIDENCE_SEAL:
        raise TypeError(
            "exact-feasibility evidence must come from an upstream exact-feasibility verifier"
        )
    if evidence is None or expected_evidence_sha256 is None:
        raise ExactFeasibilityEvidenceUnavailableError(
            "upstream replayed exact-feasibility evidence is required but not implemented"
        )
    external_evidence_sha256 = _require_sha256(
        expected_evidence_sha256, "expected exact-feasibility evidence SHA-256"
    )
    if claimed_evidence_sha256 != external_evidence_sha256:
        raise ValueError("exact-feasibility evidence disagrees with its external commitment")
    if (
        evidence.evidence_sha256 != external_evidence_sha256
        or evidence.problem_sha256 != problem.problem_sha256
        or evidence.host_graph_sha256 != host.host_graph_sha256
        or evidence.method != EXACT_FEASIBILITY_METHOD
        or evidence.node_limit != EXACT_FEASIBILITY_NODE_LIMIT
        or evidence.status != status
        or evidence.nodes_visited != nodes_visited
    ):
        raise ValueError("exact-feasibility profile disagrees with verified upstream evidence")


def verify_solver_hardness_profile(
    profile_bytes: bytes,
    *,
    expected_profile_sha256: str,
    expected_release_id: str,
    expected_partition: str,
    problem: IsingProblem,
    expected_problem_sha256: str,
    realized_host: VerifiedRealizedHostGraphSnapshot,
    expected_host_graph_sha256: str,
    seed_resolver: VerifiedSeedResolver,
    expected_seed_registry_terminal_root_sha256: str,
    solver_protocols: tuple[VerifiedSolverProtocol, VerifiedSolverProtocol],
    expected_solver_protocol_sha256s: tuple[str, str],
    solver_execution_evidence: VerifiedSolverExecutionEvidence | None,
    expected_solver_execution_evidence_sha256: str | None,
    exact_feasibility_evidence: VerifiedExactFeasibilityEvidence | None,
    expected_exact_feasibility_evidence_sha256: str | None,
) -> VerifiedSolverHardnessProfile:
    """Verify one complete profile against all independent roots."""

    expected_profile = _require_sha256(expected_profile_sha256, "expected profile SHA-256")
    if type(profile_bytes) is not bytes:
        raise TypeError("solver-hardness profile must be exact bytes")
    actual_profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    if actual_profile_sha256 != expected_profile:
        raise ValueError("solver-hardness profile disagrees with its external commitment")
    document = parse_canonical_json_bytes(profile_bytes, name="solver-hardness profile")
    profile = require_exact_keys(document, _PROFILE_FIELDS, "solver-hardness profile")
    _require_exact_value(profile["schema"], SOLVER_HARDNESS_PROFILE_SCHEMA, "profile schema")
    _require_exact_value(
        profile["schema_version"],
        SOLVER_HARDNESS_PROFILE_SCHEMA_VERSION,
        "schema_version",
    )
    release_id = _require_text(expected_release_id, "expected_release_id")
    _require_exact_value(profile["release_id"], release_id, "release_id")
    scientific_problem = _validate_problem(problem, expected_problem_sha256=expected_problem_sha256)
    problem_sha256 = scientific_problem.problem_sha256
    _require_exact_value(profile["problem_sha256"], problem_sha256, "problem_sha256")
    host = validate_realized_host_graph_snapshot(
        realized_host,
        expected_host_graph_sha256=expected_host_graph_sha256,
    )
    _require_exact_value(profile["host_graph_sha256"], host.host_graph_sha256, "host_graph_sha256")
    partition = _require_text(expected_partition, "expected_partition")
    _require_exact_value(profile["partition"], partition, "partition")
    n_vars = len(scientific_problem.variables)
    _require_exact_value(profile["n_vars"], n_vars, "n_vars")
    host_node_count = len(host.nodes)
    _require_exact_value(
        profile["realized_host_node_count"],
        host_node_count,
        "realized_host_node_count",
    )

    expected_seed_root = _require_sha256(
        expected_seed_registry_terminal_root_sha256,
        "expected seed-registry terminal-root SHA-256",
    )
    try:
        resolver = validate_seed_resolver(
            seed_resolver,
            expected_terminal_root_sha256=expected_seed_root,
        )
    except AttributeError as error:
        raise TypeError("seed_resolver must be a complete verified resolver") from error
    if (
        _require_sha256(
            profile["seed_registry_terminal_root_sha256"],
            "profile seed-registry terminal-root SHA-256",
        )
        != expected_seed_root
    ):
        raise ValueError("profile seed registry disagrees with its external terminal root")
    protocols, protocol_roots, protocol_documents = _validate_protocol_pair(
        solver_protocols,
        expected_solver_protocol_sha256s,
        expected_release_id=release_id,
    )
    schedule_sha256, request_keys, entry_digests, seed32s = _validate_schedule(
        profile["seed_schedule"],
        release_id=release_id,
        partition=partition,
        problem_sha256=problem_sha256,
        resolver=resolver,
    )
    raw_runs = profile["solver_runs"]
    if type(raw_runs) is not list or len(raw_runs) != 2:
        raise ValueError("solver_runs must contain exactly minorminer and cpp_baseline")
    checked_runs: list[tuple[_AttemptResult, ...]] = []
    for index, solver_id in enumerate(SOLVER_ORDER):
        checked_runs.append(
            _validate_solver_run(
                raw_runs[index],
                expected_solver_id=solver_id,
                protocol_sha256=protocol_roots[index],
                protocol_document=protocol_documents[index],
                schedule_sha256=schedule_sha256,
                request_keys=request_keys,
                entry_digests=entry_digests,
                seed32s=seed32s,
                problem=scientific_problem,
                host=host,
            )
        )
    _validate_exact_feasibility(
        profile["exact_feasibility"],
        problem=scientific_problem,
        host=host,
        evidence=exact_feasibility_evidence,
        expected_evidence_sha256=expected_exact_feasibility_evidence_sha256,
    )
    execution_evidence, execution_evidence_sha256 = _validate_solver_execution_evidence(
        solver_execution_evidence,
        expected_evidence_sha256=expected_solver_execution_evidence_sha256,
        release_id=release_id,
        partition=partition,
        problem_sha256=problem_sha256,
        host_graph_sha256=host.host_graph_sha256,
        seed_registry_terminal_root_sha256=expected_seed_root,
        seed_schedule_sha256=schedule_sha256,
        solver_protocol_sha256s=protocol_roots,
        solver_runs_sha256=canonical_sha256(raw_runs),
    )

    verified = object.__new__(VerifiedSolverHardnessProfile)
    object.__setattr__(verified, "release_id", release_id)
    object.__setattr__(verified, "partition", partition)
    object.__setattr__(verified, "problem_sha256", problem_sha256)
    object.__setattr__(verified, "host_graph_sha256", host.host_graph_sha256)
    object.__setattr__(verified, "n_vars", n_vars)
    object.__setattr__(verified, "realized_host_node_count", host_node_count)
    object.__setattr__(verified, "profile_sha256", actual_profile_sha256)
    object.__setattr__(verified, "seed_registry_terminal_root_sha256", expected_seed_root)
    object.__setattr__(verified, "seed_schedule_sha256", schedule_sha256)
    object.__setattr__(
        verified,
        "solver_execution_evidence_sha256",
        execution_evidence_sha256,
    )
    object.__setattr__(verified, "_payload", bytes(profile_bytes))
    object.__setattr__(verified, "_problem", scientific_problem)
    object.__setattr__(verified, "_realized_host", host)
    object.__setattr__(verified, "_seed_resolver", resolver)
    object.__setattr__(verified, "_solver_protocols", protocols)
    object.__setattr__(verified, "_solver_protocol_sha256s", protocol_roots)
    object.__setattr__(
        verified,
        "_attempt_outcomes",
        tuple(tuple(item.outcome for item in run) for run in checked_runs),
    )
    object.__setattr__(
        verified,
        "_attempt_termination_modes",
        tuple(tuple(item.runtime_termination_mode for item in run) for run in checked_runs),
    )
    object.__setattr__(verified, "_solver_execution_evidence", execution_evidence)
    object.__setattr__(verified, "_exact_feasibility_evidence", exact_feasibility_evidence)
    object.__setattr__(
        verified,
        "_exact_feasibility_evidence_sha256",
        expected_exact_feasibility_evidence_sha256,
    )
    object.__setattr__(verified, "_seal", _PROFILE_SEAL)
    return verified


def validate_solver_hardness_profile(
    value: object,
    *,
    expected_profile_sha256: str,
    expected_release_id: str,
    expected_partition: str,
    expected_problem_sha256: str,
    expected_host_graph_sha256: str,
    expected_solver_execution_evidence_sha256: str,
) -> VerifiedSolverHardnessProfile:
    """Replay a verified profile capsule before exposing its retained bytes."""

    if type(value) is not VerifiedSolverHardnessProfile:
        raise TypeError("profile must be an exact VerifiedSolverHardnessProfile")
    profile = value
    if (
        getattr(profile, "_seal", None) is not _PROFILE_SEAL
        or type(getattr(profile, "_payload", None)) is not bytes
    ):
        raise TypeError("profile must be produced by verify_solver_hardness_profile")
    replayed = verify_solver_hardness_profile(
        profile._payload,
        expected_profile_sha256=expected_profile_sha256,
        expected_release_id=expected_release_id,
        expected_partition=expected_partition,
        problem=profile._problem,
        expected_problem_sha256=expected_problem_sha256,
        realized_host=profile._realized_host,
        expected_host_graph_sha256=expected_host_graph_sha256,
        seed_resolver=profile._seed_resolver,
        expected_seed_registry_terminal_root_sha256=profile.seed_registry_terminal_root_sha256,
        solver_protocols=profile._solver_protocols,
        expected_solver_protocol_sha256s=profile._solver_protocol_sha256s,
        solver_execution_evidence=profile._solver_execution_evidence,
        expected_solver_execution_evidence_sha256=(expected_solver_execution_evidence_sha256),
        exact_feasibility_evidence=profile._exact_feasibility_evidence,
        expected_exact_feasibility_evidence_sha256=(profile._exact_feasibility_evidence_sha256),
    )
    public_fields = (
        "release_id",
        "partition",
        "problem_sha256",
        "host_graph_sha256",
        "n_vars",
        "realized_host_node_count",
        "profile_sha256",
        "seed_registry_terminal_root_sha256",
        "seed_schedule_sha256",
        "solver_execution_evidence_sha256",
    )
    if any(getattr(profile, field) != getattr(replayed, field) for field in public_fields):
        raise ValueError("verified profile fields disagree with retained canonical bytes")
    if profile._seed_resolver is not replayed._seed_resolver or any(
        retained is not checked
        for retained, checked in zip(
            profile._solver_protocols, replayed._solver_protocols, strict=True
        )
    ):
        raise ValueError("verified profile dependencies were replaced")
    if (
        profile._problem != replayed._problem
        or profile._realized_host is not replayed._realized_host
        or profile._attempt_outcomes != replayed._attempt_outcomes
        or profile._attempt_termination_modes != replayed._attempt_termination_modes
        or profile._solver_execution_evidence is not replayed._solver_execution_evidence
        or profile._exact_feasibility_evidence is not replayed._exact_feasibility_evidence
        or profile._exact_feasibility_evidence_sha256 != replayed._exact_feasibility_evidence_sha256
    ):
        raise ValueError("verified profile scientific evidence was replaced")
    return profile


def validate_solver_runtime_sidecar(
    value: object,
    *,
    profile: VerifiedSolverHardnessProfile,
    expected_profile_sha256: str,
    expected_release_id: str,
    expected_partition: str,
    expected_problem_sha256: str,
    expected_host_graph_sha256: str,
    expected_solver_execution_evidence_sha256: str,
) -> ValidatedSolverRuntimeSidecar:
    """Validate runtime provenance without creating a scientific content digest."""

    checked_profile = validate_solver_hardness_profile(
        profile,
        expected_profile_sha256=expected_profile_sha256,
        expected_release_id=expected_release_id,
        expected_partition=expected_partition,
        expected_problem_sha256=expected_problem_sha256,
        expected_host_graph_sha256=expected_host_graph_sha256,
        expected_solver_execution_evidence_sha256=(expected_solver_execution_evidence_sha256),
    )
    document = require_exact_keys(value, _SIDECAR_FIELDS, "solver runtime sidecar")
    _require_exact_value(document["schema"], SOLVER_RUNTIME_SIDECAR_SCHEMA, "sidecar schema")
    _require_exact_value(
        document["schema_version"], SOLVER_RUNTIME_SIDECAR_SCHEMA_VERSION, "schema_version"
    )
    if _require_sha256(document["profile_sha256"], "sidecar profile SHA-256") != (
        checked_profile.profile_sha256
    ):
        raise ValueError("runtime sidecar does not bind the verified scientific profile")
    profile_document = parse_canonical_json_bytes(
        checked_profile._payload, name="verified solver-hardness profile"
    )
    profile_runs = cast(list[dict[str, object]], profile_document["solver_runs"])
    raw_observations = document["solver_observations"]
    if type(raw_observations) is not list or len(raw_observations) != 2:
        raise ValueError("runtime sidecar must cover exactly both registered solver runs")
    success_distributions: dict[str, tuple[float, ...]] = {}
    for run_index, solver_id in enumerate(SOLVER_ORDER):
        solver_observations = require_exact_keys(
            raw_observations[run_index],
            _SOLVER_OBSERVATION_FIELDS,
            f"{solver_id} runtime observations",
        )
        _require_exact_value(solver_observations["solver_id"], solver_id, "sidecar solver_id")
        raw_attempts = solver_observations["attempts"]
        if type(raw_attempts) is not list or len(raw_attempts) != SOLVER_ATTEMPT_COUNT:
            raise ValueError("runtime sidecar must contain all 32 attempt observations")
        scientific_attempts = cast(list[dict[str, object]], profile_runs[run_index]["attempts"])
        success_runtimes: list[float] = []
        for attempt_index, raw_observation in enumerate(raw_attempts):
            observation = require_exact_keys(
                raw_observation,
                _RUNTIME_OBSERVATION_FIELDS,
                f"runtime observation {attempt_index}",
            )
            scientific_attempt = scientific_attempts[attempt_index]
            _require_exact_value(
                observation["attempt_index"], attempt_index, "runtime attempt_index"
            )
            _require_exact_value(
                observation["request_key_sha256"],
                scientific_attempt["request_key_sha256"],
                "runtime request key",
            )
            runtime = observation["runtime_seconds"]
            if type(runtime) not in {int, float}:
                raise ValueError("runtime_seconds must be a finite nonnegative number")
            try:
                runtime_float = float(runtime)
            except OverflowError as error:
                raise ValueError("runtime_seconds must be a finite nonnegative number") from error
            if not math.isfinite(runtime_float) or runtime_float < 0:
                raise ValueError("runtime_seconds must be a finite nonnegative number")
            termination_mode = _require_text(observation["termination_mode"], "termination_mode")
            expected_mode = checked_profile._attempt_termination_modes[run_index][attempt_index]
            if termination_mode != expected_mode:
                raise ValueError("termination_mode disagrees with verified process evidence")
            if checked_profile._attempt_outcomes[run_index][attempt_index] == "success":
                success_runtimes.append(runtime_float)
        success_distributions[solver_id] = tuple(sorted(success_runtimes))
    validated = object.__new__(ValidatedSolverRuntimeSidecar)
    object.__setattr__(validated, "profile_sha256", checked_profile.profile_sha256)
    object.__setattr__(
        validated,
        "success_runtime_distributions",
        MappingProxyType(success_distributions),
    )
    object.__setattr__(validated, "_seal", _RUNTIME_SIDECAR_SEAL)
    return validated


__all__ = [
    "ENVIRONMENT_FAILURE_CATEGORIES",
    "EXACT_FEASIBILITY_METHOD",
    "EXACT_FEASIBILITY_NODE_LIMIT",
    "ExactFeasibilityEvidenceUnavailableError",
    "FAILURE_CATEGORIES",
    "HEURISTIC_FAILURE_CATEGORIES",
    "SOLVER_HARDNESS_PROFILE_SCHEMA",
    "SOLVER_HARDNESS_PROFILE_SCHEMA_VERSION",
    "SOLVER_EMBEDDING_OUTPUT_SCHEMA",
    "SOLVER_EMBEDDING_OUTPUT_SCHEMA_VERSION",
    "SOLVER_RUNTIME_SIDECAR_SCHEMA",
    "SOLVER_RUNTIME_SIDECAR_SCHEMA_VERSION",
    "SolverExecutionEvidenceUnavailableError",
    "ValidatedSolverRuntimeSidecar",
    "VerifiedExactFeasibilityEvidence",
    "VerifiedRealizedHostGraphSnapshot",
    "VerifiedSolverExecutionEvidence",
    "VerifiedSolverHardnessProfile",
    "validate_realized_host_graph_snapshot",
    "validate_solver_hardness_profile",
    "validate_solver_runtime_sidecar",
    "verify_realized_host_graph_snapshot",
    "verify_solver_hardness_profile",
]

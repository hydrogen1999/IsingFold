from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import networkx as nx
import pytest
from embedbench.ground_certificate import IsingProblem
from embedbench.hard_ood_protocols import (
    SOLVER_ENTRYPOINTS,
    SOLVER_PROTOCOL_SCHEMA,
    SOLVER_RESOLVED_PARAMETERS,
    SOLVER_SEED_PROJECTION,
    SOLVER_SEED_SCHEDULE_RULE,
    SOLVER_SELECTION_ORDER,
    SOLVER_WORK_LIMITS,
    VerifiedSolverProtocol,
    solver_attempt_seed_schedule,
    solver_attempt_seed_schedule_sha256,
    verify_solver_protocol,
)
from embedbench.hard_ood_provenance import (
    SOURCE_BUNDLE_SCHEMA,
    VerifiedSourceBundle,
    verify_source_bundle,
)
from embedbench.hard_ood_schema import (
    SeedRegistration,
    VerifiedSeedResolver,
    canonical_bytes,
    canonical_sha256,
    verify_seed_registry_shards,
    write_seed_registry_shards,
)
from embedbench.realized_host import host_graph_sha256
from embedbench.solver_hardness_profile import (
    EXACT_FEASIBILITY_METHOD,
    EXACT_FEASIBILITY_NODE_LIMIT,
    SOLVER_EMBEDDING_OUTPUT_SCHEMA,
    SOLVER_HARDNESS_PROFILE_SCHEMA,
    SOLVER_HARDNESS_PROFILE_SCHEMA_VERSION,
    SOLVER_RUNTIME_SIDECAR_SCHEMA,
    ExactFeasibilityEvidenceUnavailableError,
    SolverExecutionEvidenceUnavailableError,
    ValidatedSolverRuntimeSidecar,
    VerifiedExactFeasibilityEvidence,
    VerifiedRealizedHostGraphSnapshot,
    VerifiedSolverExecutionEvidence,
    VerifiedSolverHardnessProfile,
    validate_realized_host_graph_snapshot,
    validate_solver_hardness_profile,
    validate_solver_runtime_sidecar,
    verify_realized_host_graph_snapshot,
    verify_solver_hardness_profile,
)

_RELEASE = "embedbench-hard-ood-v1.0.0"
_PARTITION = "hard_dev"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_FAILURE_CATEGORIES = (
    "no_embedding_returned",
    "invalid_embedding",
    "watchdog",
    "process_launch_failure",
    "process_nonzero_exit",
    "process_signal",
    "malformed_output",
    "executor_attestation_failure",
)


def _problem(variable_count: int = 17) -> IsingProblem:
    variables = tuple(range(variable_count))
    return IsingProblem(
        variables=variables,
        linear=tuple((variable, 0.0) for variable in variables),
        quadratic=tuple((variable, variable + 1, 1.0) for variable in variables[:-1]),
    )


def _host(variable_count: int = 17) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(40))
    graph.add_edges_from((variable, variable + 1) for variable in range(variable_count - 1))
    graph.add_edges_from(
        (20 + variable, 20 + variable + 1) for variable in range(variable_count - 1)
    )
    return graph


def _portable_native_executable() -> bytes:
    payload = Path(sys.executable).read_bytes()
    if payload[:4] == b"\x7fELF" or payload[:4] in {
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
        b"\xca\xfe\xba\xbf",
        b"\xbf\xba\xfe\xca",
    }:
        return payload
    fixture = bytearray(121)
    fixture[:8] = b"\x7fELF\x02\x01\x01\x00"
    struct.pack_into("<HHI", fixture, 16, 2, 62, 1)
    struct.pack_into("<QQ", fixture, 24, 0x400078, 64)
    struct.pack_into("<H", fixture, 52, 64)
    struct.pack_into("<HH", fixture, 54, 56, 1)
    struct.pack_into("<II", fixture, 64, 1, 5)
    struct.pack_into("<QQQQQ", fixture, 72, 120, 0x400078, 0, 1, 1)
    fixture[120] = 0xC3
    return bytes(fixture)


def _source_bundle(tmp_path: Path, solver_id: str) -> VerifiedSourceBundle:
    entrypoint = SOLVER_ENTRYPOINTS[solver_id]
    payload = _portable_native_executable()
    root = tmp_path / f"bundle-{solver_id}"
    target = root / entrypoint
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    target.chmod(0o755)
    manifest = canonical_bytes(
        {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "schema_version": 1,
            "release_id": _RELEASE,
            "role": solver_id,
            "files": [
                {
                    "relative_path": entrypoint,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                    "executable": True,
                }
            ],
        }
    )
    return verify_source_bundle(
        root,
        manifest,
        expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        expected_role=solver_id,
    )


def _solver_protocol(tmp_path: Path, solver_id: str) -> tuple[VerifiedSolverProtocol, str]:
    bundle = _source_bundle(tmp_path, solver_id)
    work_limit_unit, work_limit = SOLVER_WORK_LIMITS[solver_id]
    document = {
        "schema": SOLVER_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "release_id": _RELEASE,
        "solver_id": solver_id,
        "source_bundle_sha256": bundle.manifest_sha256,
        "entrypoint": SOLVER_ENTRYPOINTS[solver_id],
        "binary_sha256": bundle.file_entry(
            SOLVER_ENTRYPOINTS[solver_id],
            expected_manifest_sha256=bundle.manifest_sha256,
            expected_role=solver_id,
            expected_release_id=_RELEASE,
        ).sha256,
        "attempt_count": 32,
        "seed_schedule_rule": SOLVER_SEED_SCHEDULE_RULE,
        "seed_projection": SOLVER_SEED_PROJECTION,
        "selection_order": list(SOLVER_SELECTION_ORDER),
        "deterministic_work_limit": work_limit,
        "work_limit_unit": work_limit_unit,
        "safety_watchdog_seconds": 600,
        "resolved_parameters": dict(SOLVER_RESOLVED_PARAMETERS[solver_id]),
        "reference_executor": {
            "site": "apollo",
            "cpu_model": "fixture-cpu",
            "logical_cores": 1,
            "operating_system": "fixture-os",
        },
        "software_versions": {
            "solver": f"{solver_id}-fixture-1",
            "compiler_id": "fixture-compiler",
            "compiler_version": "fixture-compiler-1",
            "dependency_versions": {"fixture-dependency": "1"},
        },
    }
    payload = canonical_bytes(document)
    digest = hashlib.sha256(payload).hexdigest()
    return (
        verify_solver_protocol(
            payload,
            expected_protocol_sha256=digest,
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        ),
        digest,
    )


def _seed_resolver(
    tmp_path: Path,
    problem_sha256: str,
    *,
    seed32_required: bool = True,
) -> tuple[VerifiedSeedResolver, str]:
    schedule = solver_attempt_seed_schedule(
        release_id=_RELEASE,
        partition=_PARTITION,
        problem_sha256=problem_sha256,
    )
    suffix = "seed32" if seed32_required else "pcg"
    directory = tmp_path / f"seed-registry-{problem_sha256[:12]}-{suffix}"
    root = write_seed_registry_shards(
        (
            SeedRegistration(request=request, seed32_required=seed32_required)
            for request in schedule
        ),
        directory=directory,
        release_id=_RELEASE,
        stage_id="solver-hardness",
        shard_entry_limit=7,
        sort_chunk_limit=5,
    )
    verified = verify_seed_registry_shards(directory, expected_root_sha256=root.sha256)
    return VerifiedSeedResolver(verified, expected_terminal_root_sha256=root.sha256), root.sha256


def _embedding_output_bytes(chains: dict[int, list[int]]) -> bytes:
    return canonical_bytes(
        {
            "schema": SOLVER_EMBEDDING_OUTPUT_SCHEMA,
            "schema_version": 1,
            "result": "embedding",
            "embedding": [
                {"logical_variable": variable, "chain": chain}
                for variable, chain in sorted(chains.items())
            ],
        }
    )


def _no_embedding_output_bytes() -> bytes:
    return canonical_bytes(
        {
            "schema": SOLVER_EMBEDDING_OUTPUT_SCHEMA,
            "schema_version": 1,
            "result": "no_embedding",
            "embedding": None,
        }
    )


def _output_fields(payload: bytes | None) -> dict[str, object]:
    if payload is None:
        return {"output_sha256": None, "output_bytes_hex": None}
    return {
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "output_bytes_hex": payload.hex(),
    }


def _process_evidence(
    termination_mode: str,
    *,
    exit_code: int | None = None,
    signal_number: int | None = None,
) -> dict[str, object]:
    prelaunch = termination_mode in {"executor_attestation_failure", "launch_failure"}
    return {
        "executor_attested": termination_mode != "executor_attestation_failure",
        "termination_mode": termination_mode,
        "exit_code": exit_code,
        "signal_number": signal_number,
        "stdout_sha256": None if prelaunch else _EMPTY_SHA256,
        "stderr_sha256": None if prelaunch else _EMPTY_SHA256,
    }


def _failure_counts(**nonzero: int) -> list[dict[str, object]]:
    return [
        {"category": category, "count": nonzero.get(category, 0)}
        for category in _FAILURE_CATEGORIES
    ]


def _run_document(
    protocol: VerifiedSolverProtocol,
    protocol_sha256: str,
    schedule_digest: str,
    request_keys: list[str],
    entry_digests: list[str],
    seed32s: list[int],
    chains: dict[int, list[int]],
) -> dict[str, object]:
    embedding_output = _embedding_output_bytes(chains)
    no_embedding_output = _no_embedding_output_bytes()
    attempts: list[dict[str, object]] = []
    for index, (request_key, entry_digest, seed32) in enumerate(
        zip(request_keys, entry_digests, seed32s, strict=True)
    ):
        if index < 30:
            process = _process_evidence("exited", exit_code=0)
            output = embedding_output
        elif index == 30:
            process = _process_evidence("exited", exit_code=0)
            output = no_embedding_output
        else:
            process = _process_evidence("watchdog")
            output = None
        attempts.append(
            {
                "attempt_index": index,
                "request_key_sha256": request_key,
                "registry_entry_sha256": entry_digest,
                "seed32": seed32,
                "process_evidence": process,
                **_output_fields(output),
            }
        )
    total_qubits = sum(len(chain) for chain in chains.values())
    maximum_chain_length = max(len(chain) for chain in chains.values())
    sum_squared = sum(len(chain) ** 2 for chain in chains.values())
    return {
        "solver_id": protocol.solver_id,
        "solver_protocol_sha256": protocol_sha256,
        "solver_protocol": protocol.to_dict(
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        ),
        "seed_schedule_sha256": schedule_digest,
        "request_key_sha256s": request_keys,
        "registry_entry_sha256s": entry_digests,
        "attempts": attempts,
        "accounting": {
            "planned_attempt_count": 32,
            "completed_attempt_count": 31,
            "success_count": 30,
            "heuristic_failure_count": 1,
            "environment_failure_count": 1,
            "success_rate": {"numerator": 30, "denominator": 31},
            "failure_category_counts": _failure_counts(
                no_embedding_returned=1,
                watchdog=1,
            ),
        },
        "success_resource_distributions": {
            "total_qubits": [total_qubits] * 30,
            "maximum_chain_length": [maximum_chain_length] * 30,
            "sum_squared_chain_lengths": [sum_squared] * 30,
        },
    }


def _make_capsule(
    tmp_path: Path,
    *,
    problem: IsingProblem | None = None,
    host_graph: nx.Graph | None = None,
) -> dict[str, Any]:
    scientific_problem = _problem() if problem is None else problem
    mutable_host = _host(len(scientific_problem.variables)) if host_graph is None else host_graph
    host_sha256 = host_graph_sha256(mutable_host)
    host_snapshot = verify_realized_host_graph_snapshot(
        mutable_host,
        expected_host_graph_sha256=host_sha256,
    )
    resolver, seed_root = _seed_resolver(tmp_path, scientific_problem.problem_sha256)
    minorminer, minorminer_sha256 = _solver_protocol(tmp_path, "minorminer")
    cpp, cpp_sha256 = _solver_protocol(tmp_path, "cpp_baseline")
    protocols = (minorminer, cpp)
    protocol_sha256s = (minorminer_sha256, cpp_sha256)
    schedule = solver_attempt_seed_schedule(
        release_id=_RELEASE,
        partition=_PARTITION,
        problem_sha256=scientific_problem.problem_sha256,
    )
    entries = resolver.resolve_many(schedule)
    request_keys = [request.seed_key_hex for request in schedule]
    entry_digests = [canonical_sha256(entry.to_dict()) for entry in entries]
    seed32s = [entry.seed32 for entry in entries]
    assert all(type(seed32) is int for seed32 in seed32s)
    checked_seed32s = [cast(int, seed32) for seed32 in seed32s]
    schedule_digest = solver_attempt_seed_schedule_sha256(schedule)
    first_chains = {variable: [variable] for variable in scientific_problem.variables}
    second_chains = {variable: [20 + variable] for variable in scientific_problem.variables}
    mandatory_exact = len(scientific_problem.variables) <= 16 and len(host_snapshot.nodes) <= 128
    exact_feasibility = (
        {
            "requirement": "required",
            "method": EXACT_FEASIBILITY_METHOD,
            "node_limit": EXACT_FEASIBILITY_NODE_LIMIT,
            "status": "feasible",
            "nodes_visited": 123,
            "evidence_sha256": "e" * 64,
        }
        if mandatory_exact
        else {
            "requirement": "not_required",
            "method": None,
            "node_limit": None,
            "status": "not_run",
            "nodes_visited": None,
            "evidence_sha256": None,
        }
    )
    document = {
        "schema": SOLVER_HARDNESS_PROFILE_SCHEMA,
        "schema_version": SOLVER_HARDNESS_PROFILE_SCHEMA_VERSION,
        "release_id": _RELEASE,
        "partition": _PARTITION,
        "problem_sha256": scientific_problem.problem_sha256,
        "host_graph_sha256": host_sha256,
        "n_vars": len(scientific_problem.variables),
        "realized_host_node_count": len(host_snapshot.nodes),
        "seed_registry_terminal_root_sha256": seed_root,
        "seed_schedule": {
            "rule": SOLVER_SEED_SCHEDULE_RULE,
            "projection": SOLVER_SEED_PROJECTION,
            "attempt_count": 32,
            "schedule_sha256": schedule_digest,
            "requests": [request.to_dict() for request in schedule],
            "request_key_sha256s": request_keys,
            "registry_entry_sha256s": entry_digests,
            "seed32s": checked_seed32s,
        },
        "solver_runs": [
            _run_document(
                minorminer,
                minorminer_sha256,
                schedule_digest,
                request_keys,
                entry_digests,
                checked_seed32s,
                first_chains,
            ),
            _run_document(
                cpp,
                cpp_sha256,
                schedule_digest,
                request_keys,
                entry_digests,
                checked_seed32s,
                second_chains,
            ),
        ],
        "exact_feasibility": exact_feasibility,
    }
    return {
        "document": document,
        "problem": scientific_problem,
        "host_graph": mutable_host,
        "host_snapshot": host_snapshot,
        "host_sha256": host_sha256,
        "resolver": resolver,
        "seed_root": seed_root,
        "protocols": protocols,
        "protocol_sha256s": protocol_sha256s,
    }


@pytest.fixture
def capsule(tmp_path: Path) -> dict[str, Any]:
    return _make_capsule(tmp_path)


def _verify(
    capsule: dict[str, Any],
    document: dict[str, Any] | None = None,
    *,
    expected_profile_sha256: str | None = None,
    exact_evidence: VerifiedExactFeasibilityEvidence | None = None,
    exact_evidence_sha256: str | None = None,
    execution_evidence: VerifiedSolverExecutionEvidence | None = None,
    execution_evidence_sha256: str | None = None,
) -> VerifiedSolverHardnessProfile:
    checked_document = capsule["document"] if document is None else document
    payload = canonical_bytes(checked_document)
    return verify_solver_hardness_profile(
        payload,
        expected_profile_sha256=(
            hashlib.sha256(payload).hexdigest()
            if expected_profile_sha256 is None
            else expected_profile_sha256
        ),
        expected_release_id=_RELEASE,
        expected_partition=_PARTITION,
        problem=capsule["problem"],
        expected_problem_sha256=capsule["problem"].problem_sha256,
        realized_host=capsule["host_snapshot"],
        expected_host_graph_sha256=capsule["host_sha256"],
        seed_resolver=capsule["resolver"],
        expected_seed_registry_terminal_root_sha256=capsule["seed_root"],
        solver_protocols=capsule["protocols"],
        expected_solver_protocol_sha256s=capsule["protocol_sha256s"],
        solver_execution_evidence=execution_evidence,
        expected_solver_execution_evidence_sha256=execution_evidence_sha256,
        exact_feasibility_evidence=exact_evidence,
        expected_exact_feasibility_evidence_sha256=exact_evidence_sha256,
    )


def _replace_attempt_output(attempt: dict[str, Any], payload: bytes | None) -> None:
    attempt.update(_output_fields(payload))


def test_verified_evidence_capsules_cannot_be_constructed_or_forged() -> None:
    with pytest.raises(TypeError, match="verifier"):
        VerifiedSolverHardnessProfile()
    with pytest.raises(TypeError, match="verifier"):
        VerifiedRealizedHostGraphSnapshot()
    with pytest.raises(TypeError, match="upstream exact-feasibility verifier"):
        VerifiedExactFeasibilityEvidence()
    with pytest.raises(TypeError, match="real execution verifier"):
        VerifiedSolverExecutionEvidence()
    with pytest.raises(TypeError, match="validator"):
        ValidatedSolverRuntimeSidecar()

    forged = object.__new__(VerifiedSolverHardnessProfile)
    with pytest.raises(TypeError, match="produced"):
        validate_solver_hardness_profile(
            forged,
            expected_profile_sha256="a" * 64,
            expected_release_id=_RELEASE,
            expected_partition=_PARTITION,
            expected_problem_sha256="b" * 64,
            expected_host_graph_sha256="c" * 64,
            expected_solver_execution_evidence_sha256="d" * 64,
        )


def test_realized_host_snapshot_is_immutable_and_externally_bound() -> None:
    graph = _host()
    digest = host_graph_sha256(graph)
    snapshot = verify_realized_host_graph_snapshot(
        graph,
        expected_host_graph_sha256=digest,
    )
    graph.remove_node(0)

    assert 0 in snapshot.nodes
    assert (
        validate_realized_host_graph_snapshot(
            snapshot,
            expected_host_graph_sha256=digest,
        )
        is snapshot
    )
    with pytest.raises(ValueError, match="external host graph"):
        verify_realized_host_graph_snapshot(
            _host(),
            expected_host_graph_sha256="a" * 64,
        )


def test_structurally_valid_profile_fails_closed_without_execution_evidence(
    capsule: dict[str, Any],
) -> None:
    with pytest.raises(
        SolverExecutionEvidenceUnavailableError,
        match="runner/verifier is not implemented",
    ):
        _verify(capsule)


def test_fabricated_self_consistent_32_of_32_run_cannot_become_verified(
    capsule: dict[str, Any],
) -> None:
    document = copy.deepcopy(capsule["document"])
    valid_output = _embedding_output_bytes(
        {variable: [variable] for variable in capsule["problem"].variables}
    )
    run = document["solver_runs"][0]
    for attempt in run["attempts"]:
        attempt["process_evidence"] = _process_evidence("exited", exit_code=0)
        _replace_attempt_output(attempt, valid_output)
    run["accounting"] = {
        "planned_attempt_count": 32,
        "completed_attempt_count": 32,
        "success_count": 32,
        "heuristic_failure_count": 0,
        "environment_failure_count": 0,
        "success_rate": {"numerator": 32, "denominator": 32},
        "failure_category_counts": _failure_counts(),
    }
    run["success_resource_distributions"] = {
        "total_qubits": [17] * 32,
        "maximum_chain_length": [1] * 32,
        "sum_squared_chain_lengths": [17] * 32,
    }

    with pytest.raises(
        SolverExecutionEvidenceUnavailableError,
        match="real solver-execution evidence is required",
    ):
        _verify(capsule, document)

    with pytest.raises(TypeError, match="real execution verifier"):
        _verify(
            capsule,
            document,
            execution_evidence=cast(Any, {"evidence_sha256": "d" * 64}),
            execution_evidence_sha256="d" * 64,
        )

    forged = object.__new__(VerifiedSolverExecutionEvidence)
    with pytest.raises(TypeError, match="real execution verifier"):
        _verify(
            capsule,
            document,
            execution_evidence=forged,
            execution_evidence_sha256="d" * 64,
        )


def test_profile_requires_exact_problem_and_host_context(capsule: dict[str, Any]) -> None:
    payload = canonical_bytes(capsule["document"])
    digest = hashlib.sha256(payload).hexdigest()
    kwargs = {
        "expected_profile_sha256": digest,
        "expected_release_id": _RELEASE,
        "expected_partition": _PARTITION,
        "problem": capsule["problem"],
        "expected_problem_sha256": capsule["problem"].problem_sha256,
        "realized_host": capsule["host_snapshot"],
        "expected_host_graph_sha256": capsule["host_sha256"],
        "seed_resolver": capsule["resolver"],
        "expected_seed_registry_terminal_root_sha256": capsule["seed_root"],
        "solver_protocols": capsule["protocols"],
        "expected_solver_protocol_sha256s": capsule["protocol_sha256s"],
        "solver_execution_evidence": None,
        "expected_solver_execution_evidence_sha256": None,
        "exact_feasibility_evidence": None,
        "expected_exact_feasibility_evidence_sha256": None,
    }
    kwargs["problem"] = object()
    with pytest.raises(TypeError, match="exact immutable IsingProblem"):
        verify_solver_hardness_profile(payload, **kwargs)

    kwargs["problem"] = capsule["problem"]
    kwargs["expected_problem_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="problem digest"):
        verify_solver_hardness_profile(payload, **kwargs)

    kwargs["expected_problem_sha256"] = capsule["problem"].problem_sha256
    kwargs["realized_host"] = object()
    with pytest.raises(TypeError, match="host snapshot"):
        verify_solver_hardness_profile(payload, **kwargs)

    kwargs["realized_host"] = capsule["host_snapshot"]
    kwargs["expected_host_graph_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="external host graph"):
        verify_solver_hardness_profile(payload, **kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda chains: chains.update({1: [0]}),
            "success_count|resource distribution",
        ),
        (
            lambda chains: chains.update({0: [0, 17]}),
            "success_count|resource distribution",
        ),
        (
            lambda chains: chains.pop(16),
            "success_count|resource distribution",
        ),
        (
            lambda chains: chains.update({1: [17]}),
            "success_count|resource distribution",
        ),
    ],
    ids=["overlap", "disconnected", "incomplete", "missing-contact"],
)
def test_success_output_must_be_a_complete_valid_minor_embedding(
    capsule: dict[str, Any],
    mutation: Callable[[dict[int, list[int]]], object],
    message: str,
) -> None:
    document = copy.deepcopy(capsule["document"])
    chains = {variable: [variable] for variable in capsule["problem"].variables}
    mutation(chains)
    _replace_attempt_output(
        document["solver_runs"][0]["attempts"][0],
        _embedding_output_bytes(chains),
    )

    with pytest.raises(ValueError, match=message):
        _verify(capsule, document)


def test_success_output_rejects_duplicate_logical_variable_rows(
    capsule: dict[str, Any],
) -> None:
    document = copy.deepcopy(capsule["document"])
    output = json.loads(
        _embedding_output_bytes({variable: [variable] for variable in capsule["problem"].variables})
    )
    output["embedding"].insert(
        0,
        {
            "logical_variable": capsule["problem"].variables[0],
            "chain": [max(capsule["host_graph"].nodes) + 1],
        },
    )
    _replace_attempt_output(
        document["solver_runs"][0]["attempts"][0],
        canonical_bytes(output),
    )

    with pytest.raises(ValueError, match="success_count"):
        _verify(capsule, document)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda attempt: attempt.update({"output_sha256": "a" * 64}),
            "output digest",
        ),
        (
            lambda attempt: attempt.update({"output_sha256": None, "output_bytes_hex": None}),
            "completed_attempt_count|success_count|failure category",
        ),
        (
            lambda attempt: attempt.update({"outcome": "success"}),
            "unknown",
        ),
        (
            lambda attempt: attempt.update({"total_qubits": 17}),
            "unknown",
        ),
    ],
)
def test_attempts_reject_missing_mismatched_output_and_naked_claims(
    capsule: dict[str, Any],
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    document = copy.deepcopy(capsule["document"])
    mutate(document["solver_runs"][0]["attempts"][0])

    with pytest.raises((TypeError, ValueError), match=message):
        _verify(capsule, document)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda run: run["accounting"].update({"success_count": 31}),
            "success_count",
        ),
        (
            lambda run: run["accounting"]["success_rate"].update({"denominator": 32}),
            "environment failures",
        ),
        (
            lambda run: run["success_resource_distributions"].update({"total_qubits": [999] * 30}),
            "resource distribution",
        ),
        (
            lambda run: run["attempts"][0]["process_evidence"].update(
                {"termination_mode": "watchdog", "exit_code": 0}
            ),
            "watchdog.*exit_code|process evidence",
        ),
        (
            lambda run: run["attempts"][30]["process_evidence"].update({"exit_code": 5}),
            "completed_attempt_count|heuristic_failure_count|failure category",
        ),
    ],
)
def test_process_evidence_determines_classification_and_aggregates(
    capsule: dict[str, Any],
    mutate: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    document = copy.deepcopy(capsule["document"])
    mutate(document["solver_runs"][0])

    with pytest.raises((TypeError, ValueError), match=message):
        _verify(capsule, document)


def test_missing_output_can_only_be_environment_failure_when_aggregates_agree(
    capsule: dict[str, Any],
) -> None:
    document = copy.deepcopy(capsule["document"])
    run = document["solver_runs"][0]
    _replace_attempt_output(run["attempts"][0], None)
    run["accounting"].update(
        {
            "success_count": 29,
            "heuristic_failure_count": 1,
            "environment_failure_count": 2,
            "completed_attempt_count": 30,
            "success_rate": {"numerator": 29, "denominator": 30},
            "failure_category_counts": _failure_counts(
                no_embedding_returned=1,
                watchdog=1,
                malformed_output=1,
            ),
        }
    )
    for distribution in run["success_resource_distributions"].values():
        distribution.pop()

    with pytest.raises(SolverExecutionEvidenceUnavailableError):
        _verify(capsule, document)


def test_noncanonical_duplicate_and_solver_specific_schedule_are_rejected(
    capsule: dict[str, Any],
) -> None:
    payload = canonical_bytes(capsule["document"])
    pretty = json.dumps(capsule["document"], indent=2, sort_keys=True).encode()
    kwargs = {
        "expected_release_id": _RELEASE,
        "expected_partition": _PARTITION,
        "problem": capsule["problem"],
        "expected_problem_sha256": capsule["problem"].problem_sha256,
        "realized_host": capsule["host_snapshot"],
        "expected_host_graph_sha256": capsule["host_sha256"],
        "seed_resolver": capsule["resolver"],
        "expected_seed_registry_terminal_root_sha256": capsule["seed_root"],
        "solver_protocols": capsule["protocols"],
        "expected_solver_protocol_sha256s": capsule["protocol_sha256s"],
        "solver_execution_evidence": None,
        "expected_solver_execution_evidence_sha256": None,
        "exact_feasibility_evidence": None,
        "expected_exact_feasibility_evidence_sha256": None,
    }
    with pytest.raises(ValueError, match="canonical"):
        verify_solver_hardness_profile(
            pretty,
            expected_profile_sha256=hashlib.sha256(pretty).hexdigest(),
            **kwargs,
        )

    duplicate = payload.replace(
        b'{"exact_feasibility":',
        b'{"schema":"duplicate","exact_feasibility":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate"):
        verify_solver_hardness_profile(
            duplicate,
            expected_profile_sha256=hashlib.sha256(duplicate).hexdigest(),
            **kwargs,
        )

    document = copy.deepcopy(capsule["document"])
    document["seed_schedule"]["requests"][0]["cell"] = "minorminer-only"
    with pytest.raises(ValueError, match="solver-neutral"):
        _verify(capsule, document)


def test_seed_and_protocol_roots_and_pair_remain_external(capsule: dict[str, Any]) -> None:
    payload = canonical_bytes(capsule["document"])
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="terminal-root|terminal root"):
        verify_solver_hardness_profile(
            payload,
            expected_profile_sha256=digest,
            expected_release_id=_RELEASE,
            expected_partition=_PARTITION,
            problem=capsule["problem"],
            expected_problem_sha256=capsule["problem"].problem_sha256,
            realized_host=capsule["host_snapshot"],
            expected_host_graph_sha256=capsule["host_sha256"],
            seed_resolver=capsule["resolver"],
            expected_seed_registry_terminal_root_sha256="a" * 64,
            solver_protocols=capsule["protocols"],
            expected_solver_protocol_sha256s=capsule["protocol_sha256s"],
            solver_execution_evidence=None,
            expected_solver_execution_evidence_sha256=None,
            exact_feasibility_evidence=None,
            expected_exact_feasibility_evidence_sha256=None,
        )

    with pytest.raises(ValueError, match="external commitment"):
        verify_solver_hardness_profile(
            payload,
            expected_profile_sha256=digest,
            expected_release_id=_RELEASE,
            expected_partition=_PARTITION,
            problem=capsule["problem"],
            expected_problem_sha256=capsule["problem"].problem_sha256,
            realized_host=capsule["host_snapshot"],
            expected_host_graph_sha256=capsule["host_sha256"],
            seed_resolver=capsule["resolver"],
            expected_seed_registry_terminal_root_sha256=capsule["seed_root"],
            solver_protocols=capsule["protocols"],
            expected_solver_protocol_sha256s=("b" * 64, "c" * 64),
            solver_execution_evidence=None,
            expected_solver_execution_evidence_sha256=None,
            exact_feasibility_evidence=None,
            expected_exact_feasibility_evidence_sha256=None,
        )


def test_profile_digest_prevents_consistent_output_and_aggregate_whole_rebase(
    capsule: dict[str, Any],
) -> None:
    old_payload = canonical_bytes(capsule["document"])
    old_digest = hashlib.sha256(old_payload).hexdigest()
    document = copy.deepcopy(capsule["document"])
    chains = {variable: [20 + variable] for variable in capsule["problem"].variables}
    _replace_attempt_output(
        document["solver_runs"][0]["attempts"][0],
        _embedding_output_bytes(chains),
    )
    with pytest.raises(ValueError, match="external commitment"):
        _verify(capsule, document, expected_profile_sha256=old_digest)


def test_mandatory_exact_feasibility_fails_closed_without_upstream_evidence(
    tmp_path: Path,
) -> None:
    small_problem = _problem(4)
    capsule = _make_capsule(
        tmp_path,
        problem=small_problem,
        host_graph=_host(4),
    )

    with pytest.raises(
        ExactFeasibilityEvidenceUnavailableError,
        match="upstream.*exact-feasibility",
    ):
        _verify(capsule)

    forged = object.__new__(VerifiedExactFeasibilityEvidence)
    with pytest.raises(TypeError, match="upstream exact-feasibility verifier"):
        _verify(
            capsule,
            exact_evidence=forged,
            exact_evidence_sha256="e" * 64,
        )


def test_runtime_sidecar_cannot_bypass_missing_execution_evidence(
    capsule: dict[str, Any],
) -> None:
    with pytest.raises(SolverExecutionEvidenceUnavailableError):
        _verify(capsule)

    forged_profile = object.__new__(VerifiedSolverHardnessProfile)
    with pytest.raises(TypeError, match="produced"):
        validate_solver_runtime_sidecar(
            {
                "schema": SOLVER_RUNTIME_SIDECAR_SCHEMA,
                "schema_version": 1,
                "profile_sha256": "a" * 64,
                "solver_observations": [],
            },
            profile=forged_profile,
            expected_profile_sha256="a" * 64,
            expected_release_id=_RELEASE,
            expected_partition=_PARTITION,
            expected_problem_sha256=capsule["problem"].problem_sha256,
            expected_host_graph_sha256=capsule["host_sha256"],
            expected_solver_execution_evidence_sha256="d" * 64,
        )

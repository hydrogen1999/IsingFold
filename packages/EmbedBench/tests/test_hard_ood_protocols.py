from __future__ import annotations

import hashlib
import io
import struct
import sys
import zipfile
from pathlib import Path

import embedbench.hard_ood_protocols as protocols
import pytest
from embedbench.hard_ood_protocols import (
    ALLOWED_Q_CAP_SLACK,
    CANDIDATE_CHAIN_ORDER_RULE,
    CONTINUATION_COMMAND_TEMPLATE,
    CONTINUATION_INPUT_BYTE_LIMIT,
    CONTINUATION_INPUT_SCHEMA,
    CONTINUATION_NODE_BUDGET,
    CONTINUATION_NODE_COUNT_RULE,
    CONTINUATION_OBJECTIVE_ORDER,
    CONTINUATION_OUTPUT_BYTE_LIMIT,
    CONTINUATION_OUTPUT_SCHEMA,
    CONTINUATION_PATH_DEPTH_LIMIT,
    CONTINUATION_PROTOCOL_SCHEMA,
    CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT,
    CONTINUATION_STDERR_BYTE_LIMIT,
    CONTINUATION_STDOUT_BYTE_LIMIT,
    CONTINUATION_WATCHDOG_SECONDS,
    CONTINUATION_ZIP_MEMBER_LIMIT,
    CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT,
    FOCUS_RULE,
    GENERATION_PROTOCOL_SCHEMA,
    REMOVAL_RULE,
    SOLVER_ATTEMPT_COUNT,
    SOLVER_ENTRYPOINTS,
    SOLVER_PROTOCOL_SCHEMA,
    SOLVER_RESOLVED_PARAMETERS,
    SOLVER_SEED_PROJECTION,
    SOLVER_SEED_SCHEDULE_RULE,
    SOLVER_WORK_LIMITS,
    STARTING_SOURCE_ROTATION,
    VARIABLE_ORDER_RULE,
    ContinuationProtocolSnapshot,
    snapshot_continuation_protocol,
    solver_attempt_seed_request,
    solver_attempt_seed_schedule,
    solver_attempt_seed_schedule_sha256,
    verify_continuation_protocol,
    verify_generation_protocol,
    verify_solver_protocol,
)
from embedbench.hard_ood_provenance import SOURCE_BUNDLE_SCHEMA, verify_source_bundle
from embedbench.hard_ood_schema import canonical_bytes

_RELEASE = "embedbench-hard-ood-v1.0.0"


def _source_bundle(
    tmp_path: Path,
    role: str,
    files: dict[str, bytes],
    *,
    executable_paths: frozenset[str] = frozenset(),
):
    root = tmp_path / role
    for relative_path, payload in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if relative_path in executable_paths:
            target.chmod(0o755)
    manifest = canonical_bytes(
        {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "schema_version": 1,
            "release_id": _RELEASE,
            "role": role,
            "files": [
                {
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                    "executable": relative_path in executable_paths,
                }
                for relative_path, payload in sorted(files.items())
            ],
        }
    )
    return verify_source_bundle(
        root,
        manifest,
        expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        expected_role=role,
    )


def _bundle_entry(bundle, relative_path: str):
    return bundle.file_entry(
        relative_path,
        expected_manifest_sha256=bundle.manifest_sha256,
        expected_role=bundle.role,
        expected_release_id=_RELEASE,
    )


def _checker_zipapp() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("__main__.py", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, b"raise SystemExit(0)\n")
    return output.getvalue()


def _checker_zipapp_with_members(
    members: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return output.getvalue()


def _portable_runtime() -> bytes:
    return Path(sys.executable).read_bytes()


def _non_executable_elf_header() -> bytes:
    payload = bytearray(64)
    payload[:8] = b"\x7fELF\x02\x01\x01\x00"
    struct.pack_into("<HHI", payload, 16, 3, 62, 1)
    struct.pack_into("<H", payload, 52, 64)
    return bytes(payload)


def _non_executable_macho_header() -> bytes:
    payload = bytearray(40)
    payload[:4] = b"\xcf\xfa\xed\xfe"
    struct.pack_into("<IIIIII", payload, 4, 0x01000007, 3, 2, 1, 8, 0)
    struct.pack_into("<II", payload, 32, 0, 8)
    return bytes(payload)


def _continuation_bundles(tmp_path: Path):
    checker = _source_bundle(
        tmp_path,
        "continuation_checker",
        {"checker.pyz": _checker_zipapp()},
    )
    runtime = _source_bundle(
        tmp_path,
        "portable_runtime",
        {"bin/python3": _portable_runtime()},
        executable_paths=frozenset({"bin/python3"}),
    )
    return checker, runtime


def _verify_generation(document: dict[str, object], source_bundle):
    payload = canonical_bytes(document)
    return verify_generation_protocol(
        payload,
        expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
        expected_release_id=_RELEASE,
        source_bundle=source_bundle,
    )


def _generation_document(bundle_sha256: str) -> dict[str, object]:
    return {
        "schema": GENERATION_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "release_id": _RELEASE,
        "protocol_id": "hard-ood-generation-v1",
        "source_bundle_sha256": bundle_sha256,
        "witness_method": "target-guided-inkdrop-v1",
        "focus_rule": FOCUS_RULE,
        "removal_rule": REMOVAL_RULE,
        "window_rule": "witness-active-induced-v1",
        "starting_source_rotation": list(STARTING_SOURCE_ROTATION),
        "max_window_free": 28,
        "maximum_chain_length": 6,
        "allowed_q_cap_slack": list(ALLOWED_Q_CAP_SLACK),
    }


def _solver_document(
    bundle_sha256: str,
    solver_id: str,
    binary_sha256: str,
) -> dict[str, object]:
    work_limit_unit, work_limit = SOLVER_WORK_LIMITS[solver_id]
    return {
        "schema": SOLVER_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "release_id": _RELEASE,
        "solver_id": solver_id,
        "source_bundle_sha256": bundle_sha256,
        "entrypoint": SOLVER_ENTRYPOINTS[solver_id],
        "binary_sha256": binary_sha256,
        "attempt_count": 32,
        "seed_schedule_rule": SOLVER_SEED_SCHEDULE_RULE,
        "seed_projection": SOLVER_SEED_PROJECTION,
        "selection_order": [
            "total_qubits",
            "maximum_chain_length",
            "sum_squared_chain_lengths",
            "canonical_embedding_bytes",
        ],
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
            "solver": "fixture-solver-1",
            "compiler_id": "fixture-compiler",
            "compiler_version": "fixture-compiler-1",
            "dependency_versions": {"fixture-dependency": "1"},
        },
    }


def _continuation_document(checker_sha256: str, runtime_sha256: str) -> dict[str, object]:
    return {
        "schema": CONTINUATION_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "release_id": _RELEASE,
        "protocol_id": "exact-continuation-v1",
        "checker_source_bundle_sha256": checker_sha256,
        "runtime_bundle_sha256": runtime_sha256,
        "runtime_python": "bin/python3",
        "checker_zipapp": "checker.pyz",
        "command_template": list(CONTINUATION_COMMAND_TEMPLATE),
        "clean_environment": {},
        "input_schema": CONTINUATION_INPUT_SCHEMA,
        "output_schema": CONTINUATION_OUTPUT_SCHEMA,
        "node_budget": CONTINUATION_NODE_BUDGET,
        "node_count_rule": CONTINUATION_NODE_COUNT_RULE,
        "max_window_free": 28,
        "maximum_chain_length": 6,
        "variable_order_rule": VARIABLE_ORDER_RULE,
        "candidate_chain_order_rule": CANDIDATE_CHAIN_ORDER_RULE,
        "objective_order": list(CONTINUATION_OBJECTIVE_ORDER),
        "safety_watchdog_seconds": CONTINUATION_WATCHDOG_SECONDS,
        "input_byte_limit": CONTINUATION_INPUT_BYTE_LIMIT,
        "output_byte_limit": CONTINUATION_OUTPUT_BYTE_LIMIT,
        "stdout_byte_limit": CONTINUATION_STDOUT_BYTE_LIMIT,
        "stderr_byte_limit": CONTINUATION_STDERR_BYTE_LIMIT,
        "retained_bundle_byte_limit": CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT,
        "zip_member_limit": CONTINUATION_ZIP_MEMBER_LIMIT,
        "zip_uncompressed_byte_limit": CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT,
        "path_depth_limit": CONTINUATION_PATH_DEPTH_LIMIT,
    }


def test_generation_protocol_binds_registered_source_bundle(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path, "generation", {"generator.py": b"pass\n"})
    document = _generation_document(bundle.manifest_sha256)
    payload = canonical_bytes(document)
    verified = _verify_generation(document, bundle)

    assert verified.release_id == _RELEASE
    assert verified.source_bundle_sha256 == bundle.manifest_sha256
    assert (
        verified.to_dict(
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
        )["max_window_free"]
        == 28
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("max_window_free", 29),
        ("maximum_chain_length", 5),
        ("allowed_q_cap_slack", [0, 2]),
        ("starting_source_rotation", list(reversed(STARTING_SOURCE_ROTATION))),
        ("protocol_id", "easy"),
    ],
)
def test_generation_protocol_rejects_posthoc_contract_changes(
    tmp_path: Path,
    field_name: str,
    replacement: object,
) -> None:
    bundle = _source_bundle(tmp_path, "generation", {"generator.py": b"pass\n"})
    document = _generation_document(bundle.manifest_sha256)
    document[field_name] = replacement

    with pytest.raises(ValueError, match="registered|does not match"):
        _verify_generation(document, bundle)


@pytest.mark.parametrize("solver_id", ["minorminer", "cpp_baseline"])
def test_solver_protocol_binds_role_attempts_selection_and_limits(
    tmp_path: Path,
    solver_id: str,
) -> None:
    entrypoint = SOLVER_ENTRYPOINTS[solver_id]
    bundle = _source_bundle(
        tmp_path,
        solver_id,
        {entrypoint: _portable_runtime()},
        executable_paths=frozenset({entrypoint}),
    )
    document = _solver_document(
        bundle.manifest_sha256,
        solver_id,
        _bundle_entry(bundle, entrypoint).sha256,
    )
    payload = canonical_bytes(document)
    verified = verify_solver_protocol(
        payload,
        expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
        expected_release_id=_RELEASE,
        source_bundle=bundle,
    )

    assert verified.solver_id == solver_id
    assert verified.deterministic_work_limit == SOLVER_WORK_LIMITS[solver_id][1]


def test_solver_protocol_rejects_wrong_role_and_attempt_count(tmp_path: Path) -> None:
    entrypoint = SOLVER_ENTRYPOINTS["minorminer"]
    bundle = _source_bundle(
        tmp_path,
        "minorminer",
        {entrypoint: _portable_runtime()},
        executable_paths=frozenset({entrypoint}),
    )
    binary_sha256 = _bundle_entry(bundle, entrypoint).sha256
    wrong_role = _solver_document(bundle.manifest_sha256, "cpp_baseline", binary_sha256)
    wrong_payload = canonical_bytes(wrong_role)
    with pytest.raises(ValueError, match="wrong role"):
        verify_solver_protocol(
            wrong_payload,
            expected_protocol_sha256=hashlib.sha256(wrong_payload).hexdigest(),
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )

    wrong_count = _solver_document(bundle.manifest_sha256, "minorminer", binary_sha256)
    wrong_count["attempt_count"] = 31
    count_payload = canonical_bytes(wrong_count)
    with pytest.raises(ValueError, match="32 attempts"):
        verify_solver_protocol(
            count_payload,
            expected_protocol_sha256=hashlib.sha256(count_payload).hexdigest(),
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )


@pytest.mark.parametrize("solver_id", ["minorminer", "cpp_baseline"])
def test_solver_protocol_rejects_under_specified_or_unregistered_budget(
    tmp_path: Path,
    solver_id: str,
) -> None:
    entrypoint = SOLVER_ENTRYPOINTS[solver_id]
    bundle = _source_bundle(
        tmp_path,
        solver_id,
        {entrypoint: _portable_runtime()},
        executable_paths=frozenset({entrypoint}),
    )
    document = _solver_document(
        bundle.manifest_sha256,
        solver_id,
        _bundle_entry(bundle, entrypoint).sha256,
    )
    document["deterministic_work_limit"] = 1
    payload = canonical_bytes(document)
    with pytest.raises(ValueError, match="work limit"):
        verify_solver_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )

    document = _solver_document(
        bundle.manifest_sha256,
        solver_id,
        _bundle_entry(bundle, entrypoint).sha256,
    )
    document["resolved_parameters"] = {}
    payload = canonical_bytes(document)
    with pytest.raises(ValueError, match="resolved_parameters"):
        verify_solver_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )


@pytest.mark.parametrize(
    "binary",
    [
        b"arbitrary executable bytes",
        b"\x7fELF",
        _non_executable_elf_header(),
        _non_executable_macho_header(),
    ],
)
def test_solver_protocol_rejects_arbitrary_or_truncated_executable_bytes(
    tmp_path: Path,
    binary: bytes,
) -> None:
    entrypoint = SOLVER_ENTRYPOINTS["cpp_baseline"]
    bundle = _source_bundle(
        tmp_path,
        "cpp_baseline",
        {entrypoint: binary},
        executable_paths=frozenset({entrypoint}),
    )
    document = _solver_document(
        bundle.manifest_sha256,
        "cpp_baseline",
        _bundle_entry(bundle, entrypoint).sha256,
    )
    payload = canonical_bytes(document)
    with pytest.raises(ValueError, match="native executable"):
        verify_solver_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )


def test_structure_solver_adapters_share_the_exact_registered_seed32_request() -> None:
    request_fields = {
        "release_id": _RELEASE,
        "partition": "hard_dev",
        "problem_sha256": "1" * 64,
    }
    shared_schedule = solver_attempt_seed_schedule(**request_fields)

    minorminer_request = shared_schedule[7]
    cpp_request = solver_attempt_seed_request(**request_fields, attempt_index=7)

    assert minorminer_request == cpp_request
    assert minorminer_request.seed_key == cpp_request.seed_key
    assert minorminer_request.purpose == "solver_attempt"
    assert "solver_id" not in minorminer_request.to_dict()
    assert SOLVER_SEED_PROJECTION == "collision-managed-uint32-v1"
    assert {
        parameters["random_seed_source"] for parameters in SOLVER_RESOLVED_PARAMETERS.values()
    } == {"registered_attempt_seed32"}
    assert SOLVER_RESOLVED_PARAMETERS["minorminer"]["timeout_seconds"] is None
    assert len(shared_schedule) == SOLVER_ATTEMPT_COUNT
    assert (
        solver_attempt_seed_schedule_sha256(shared_schedule)
        == hashlib.sha256(
            canonical_bytes([request.to_dict() for request in shared_schedule])
        ).hexdigest()
    )


@pytest.mark.parametrize("attempt_index", [-1, 32, True])
def test_solver_seed_schedule_rejects_out_of_range_attempts(attempt_index: int) -> None:
    with pytest.raises(ValueError, match="attempt_index"):
        solver_attempt_seed_request(
            release_id=_RELEASE,
            partition="hard_dev",
            problem_sha256="1" * 64,
            attempt_index=attempt_index,
        )


def test_solver_seed_schedule_digest_rejects_adapter_specific_divergence() -> None:
    schedule = solver_attempt_seed_schedule(
        release_id=_RELEASE,
        partition="hard_dev",
        problem_sha256="1" * 64,
    )
    divergent = list(schedule)
    divergent[7] = solver_attempt_seed_request(
        release_id=_RELEASE,
        partition="hard_dev",
        problem_sha256="2" * 64,
        attempt_index=7,
    )

    with pytest.raises(ValueError, match="registered solver-neutral"):
        solver_attempt_seed_schedule_sha256(tuple(divergent))


def test_solver_seed_schedule_rejects_equality_proxy_objects() -> None:
    class EqualityProxy:
        release_id = _RELEASE
        partition = "hard_dev"
        problem_sha256 = "1" * 64

        def __eq__(self, _other: object) -> bool:
            return True

        def to_dict(self) -> dict[str, object]:
            return {"attacker": True}

    hostile = tuple(EqualityProxy() for _ in range(SOLVER_ATTEMPT_COUNT))

    with pytest.raises(TypeError, match="exact SeedRequest"):
        solver_attempt_seed_schedule_sha256(hostile)  # type: ignore[arg-type]


def test_solver_protocol_rejects_a_different_seed_projection(tmp_path: Path) -> None:
    entrypoint = SOLVER_ENTRYPOINTS["cpp_baseline"]
    bundle = _source_bundle(
        tmp_path,
        "cpp_baseline",
        {entrypoint: _portable_runtime()},
        executable_paths=frozenset({entrypoint}),
    )
    document = _solver_document(
        bundle.manifest_sha256,
        "cpp_baseline",
        _bundle_entry(bundle, entrypoint).sha256,
    )
    document["seed_projection"] = "truncate-seed-key-to-uint64"
    payload = canonical_bytes(document)

    with pytest.raises(ValueError, match="seed_projection"):
        verify_solver_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )


def test_continuation_protocol_binds_portable_runtime_and_checker_bytes(tmp_path: Path) -> None:
    checker, runtime = _continuation_bundles(tmp_path)
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    payload = canonical_bytes(document)
    verified = verify_continuation_protocol(
        payload,
        expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
        expected_release_id=_RELEASE,
        checker_source_bundle=checker,
        runtime_bundle=runtime,
    )

    assert verified.node_budget == 10_000_000
    assert verified.runtime_python == "bin/python3"
    assert verified.checker_zipapp == "checker.pyz"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("node_budget", 9_999_999),
        ("node_count_rule", "proposal_count"),
        ("maximum_chain_length", 5),
        ("command_template", ["python3", "checker.pyz"]),
        ("clean_environment", {"PYTHONPATH": "attacker"}),
        ("objective_order", list(reversed(CONTINUATION_OBJECTIVE_ORDER))),
    ],
)
def test_continuation_protocol_rejects_runtime_contract_changes(
    tmp_path: Path,
    field_name: str,
    replacement: object,
) -> None:
    checker, runtime = _continuation_bundles(tmp_path)
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    document[field_name] = replacement
    payload = canonical_bytes(document)

    with pytest.raises(ValueError, match="registered|clean environment|does not match"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )


def test_continuation_protocol_rejects_arbitrary_runtime_and_invalid_zipapp(
    tmp_path: Path,
) -> None:
    bad_checker = _source_bundle(
        tmp_path,
        "continuation_checker",
        {"checker.pyz": b"not a zipapp"},
    )
    runtime = _source_bundle(
        tmp_path,
        "portable_runtime",
        {"bin/python3": _portable_runtime()},
        executable_paths=frozenset({"bin/python3"}),
    )
    document = _continuation_document(bad_checker.manifest_sha256, runtime.manifest_sha256)
    payload = canonical_bytes(document)
    with pytest.raises(ValueError, match="zipapp"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=bad_checker,
            runtime_bundle=runtime,
        )

    other_root = tmp_path / "other"
    bad_runtime = _source_bundle(
        other_root,
        "portable_runtime",
        {"bin/python3": b"arbitrary executable bytes"},
        executable_paths=frozenset({"bin/python3"}),
    )
    checker = _source_bundle(
        other_root,
        "continuation_checker",
        {"checker.pyz": _checker_zipapp()},
    )
    document = _continuation_document(checker.manifest_sha256, bad_runtime.manifest_sha256)
    payload = canonical_bytes(document)
    with pytest.raises(ValueError, match="native executable"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=bad_runtime,
        )


def test_continuation_protocol_freezes_watchdog_and_retains_snapshot_bytes(
    tmp_path: Path,
) -> None:
    checker, runtime = _continuation_bundles(tmp_path)
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    document["safety_watchdog_seconds"] = CONTINUATION_WATCHDOG_SECONDS + 1
    payload = canonical_bytes(document)
    with pytest.raises(ValueError, match="watchdog"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )

    document["safety_watchdog_seconds"] = CONTINUATION_WATCHDOG_SECONDS
    payload = canonical_bytes(document)
    verified = verify_continuation_protocol(
        payload,
        expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
        expected_release_id=_RELEASE,
        checker_source_bundle=checker,
        runtime_bundle=runtime,
    )
    protocol_sha256 = hashlib.sha256(payload).hexdigest()
    assert (
        verified.runtime_file_bytes(
            "bin/python3",
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )
        == _portable_runtime()
    )
    assert (
        verified.checker_file_bytes(
            "checker.pyz",
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )
        == _checker_zipapp()
    )

    object.__setattr__(verified, "runtime_python", "attacker")
    with pytest.raises(ValueError, match="verified protocol fields"):
        verified.runtime_file_bytes(
            "bin/python3",
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )


@pytest.mark.parametrize(
    ("field_name", "registered_value"),
    [
        ("input_byte_limit", CONTINUATION_INPUT_BYTE_LIMIT),
        ("output_byte_limit", CONTINUATION_OUTPUT_BYTE_LIMIT),
        ("stdout_byte_limit", CONTINUATION_STDOUT_BYTE_LIMIT),
        ("stderr_byte_limit", CONTINUATION_STDERR_BYTE_LIMIT),
        ("retained_bundle_byte_limit", CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT),
        ("zip_member_limit", CONTINUATION_ZIP_MEMBER_LIMIT),
        ("zip_uncompressed_byte_limit", CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT),
        ("path_depth_limit", CONTINUATION_PATH_DEPTH_LIMIT),
    ],
)
@pytest.mark.parametrize("replacement_kind", ["different", "boolean"])
def test_continuation_resource_caps_are_exact_protocol_commitments(
    tmp_path: Path,
    field_name: str,
    registered_value: int,
    replacement_kind: str,
) -> None:
    checker, runtime = _continuation_bundles(tmp_path)
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    original_sha256 = hashlib.sha256(canonical_bytes(document)).hexdigest()
    document[field_name] = registered_value + 1 if replacement_kind == "different" else True
    payload = canonical_bytes(document)

    assert hashlib.sha256(payload).hexdigest() != original_sha256
    with pytest.raises(ValueError, match="registered protocol"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )


def test_continuation_zip_member_limit_is_checked_before_zipfile_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker_payload = _checker_zipapp_with_members(
        {
            "__main__.py": b"raise SystemExit(0)\n",
            "one.py": b"",
            "two.py": b"",
        }
    )
    checker = _source_bundle(
        tmp_path,
        "continuation_checker",
        {"checker.pyz": checker_payload},
    )
    runtime = _source_bundle(
        tmp_path,
        "portable_runtime",
        {"bin/python3": _portable_runtime()},
        executable_paths=frozenset({"bin/python3"}),
    )
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    document["zip_member_limit"] = 2
    payload = canonical_bytes(document)
    monkeypatch.setattr(protocols, "CONTINUATION_ZIP_MEMBER_LIMIT", 2)

    def forbidden_zipfile(*args, **kwargs):
        raise AssertionError("over-limit member inventory reached ZipFile")

    monkeypatch.setattr(protocols.zipfile, "ZipFile", forbidden_zipfile)
    with pytest.raises(ValueError, match="member count"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )


def test_continuation_zip_member_limit_rejects_a_forged_eocd_count_before_zipfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker_payload = bytearray(
        _checker_zipapp_with_members(
            {
                "__main__.py": b"raise SystemExit(0)\n",
                "one.py": b"",
                "two.py": b"",
            }
        )
    )
    eocd_offset = checker_payload.rfind(b"PK\x05\x06")
    assert eocd_offset >= 0
    checker_payload[eocd_offset + 8 : eocd_offset + 12] = (1).to_bytes(2, "little") * 2
    checker = _source_bundle(
        tmp_path,
        "continuation_checker",
        {"checker.pyz": bytes(checker_payload)},
    )
    runtime = _source_bundle(
        tmp_path,
        "portable_runtime",
        {"bin/python3": _portable_runtime()},
        executable_paths=frozenset({"bin/python3"}),
    )
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    document["zip_member_limit"] = 2
    payload = canonical_bytes(document)
    monkeypatch.setattr(protocols, "CONTINUATION_ZIP_MEMBER_LIMIT", 2)

    def forbidden_zipfile(*args, **kwargs):
        raise AssertionError("forged EOCD member count reached ZipFile")

    monkeypatch.setattr(protocols.zipfile, "ZipFile", forbidden_zipfile)
    with pytest.raises(ValueError, match="member count"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )


@pytest.mark.parametrize("member_name", ["", "/"])
def test_continuation_zipapp_rejects_empty_or_root_member_paths(
    tmp_path: Path,
    member_name: str,
) -> None:
    checker_payload = _checker_zipapp_with_members(
        {
            "__main__.py": b"raise SystemExit(0)\n",
            member_name: b"",
        }
    )
    checker = _source_bundle(
        tmp_path,
        "continuation_checker",
        {"checker.pyz": checker_payload},
    )
    runtime = _source_bundle(
        tmp_path,
        "portable_runtime",
        {"bin/python3": _portable_runtime()},
        executable_paths=frozenset({"bin/python3"}),
    )
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    payload = canonical_bytes(document)

    with pytest.raises(ValueError, match="relative_path"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )


def test_continuation_zip_bomb_declared_payload_is_rejected_before_decompression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker_payload = _checker_zipapp_with_members(
        {"__main__.py": b"x" * 4096},
        compression=zipfile.ZIP_DEFLATED,
    )
    checker = _source_bundle(
        tmp_path,
        "continuation_checker",
        {"checker.pyz": checker_payload},
    )
    runtime = _source_bundle(
        tmp_path,
        "portable_runtime",
        {"bin/python3": _portable_runtime()},
        executable_paths=frozenset({"bin/python3"}),
    )
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    document["zip_uncompressed_byte_limit"] = 1024
    payload = canonical_bytes(document)
    monkeypatch.setattr(protocols, "CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT", 1024)

    with pytest.raises(ValueError, match="uncompressed payload"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )


def test_continuation_zip_member_path_depth_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker_payload = _checker_zipapp_with_members(
        {
            "__main__.py": b"raise SystemExit(0)\n",
            "one/two/three.py": b"",
        }
    )
    checker = _source_bundle(
        tmp_path,
        "continuation_checker",
        {"checker.pyz": checker_payload},
    )
    runtime = _source_bundle(
        tmp_path,
        "portable_runtime",
        {"bin/python3": _portable_runtime()},
        executable_paths=frozenset({"bin/python3"}),
    )
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    document["path_depth_limit"] = 2
    payload = canonical_bytes(document)
    monkeypatch.setattr(protocols, "CONTINUATION_PATH_DEPTH_LIMIT", 2)

    with pytest.raises(ValueError, match="path depth"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )


def test_continuation_combined_retained_bundle_payload_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker, runtime = _continuation_bundles(tmp_path)
    total_payload_bytes = sum(len(entry.payload) for entry in checker.files + runtime.files)
    aggregate_limit = total_payload_bytes - 1
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    document["retained_bundle_byte_limit"] = aggregate_limit
    payload = canonical_bytes(document)
    monkeypatch.setattr(
        protocols,
        "CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT",
        aggregate_limit,
    )

    with pytest.raises(ValueError, match="retained bundles.*aggregate"):
        verify_continuation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id=_RELEASE,
            checker_source_bundle=checker,
            runtime_bundle=runtime,
        )


def test_continuation_snapshot_is_detached_and_constructor_sealed(tmp_path: Path) -> None:
    checker, runtime = _continuation_bundles(tmp_path)
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    payload = canonical_bytes(document)
    protocol_sha256 = hashlib.sha256(payload).hexdigest()
    verified = verify_continuation_protocol(
        payload,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
        checker_source_bundle=checker,
        runtime_bundle=runtime,
    )

    snapshot = snapshot_continuation_protocol(
        verified,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )
    assert snapshot.protocol_bytes == payload
    assert snapshot.checker_files is not checker.files
    assert snapshot.runtime_files is not runtime.files
    assert snapshot.input_byte_limit == CONTINUATION_INPUT_BYTE_LIMIT
    object.__setattr__(checker.files[0], "payload", b"attacker")
    assert snapshot.checker_files[0].payload == _checker_zipapp()
    with pytest.raises(TypeError):
        ContinuationProtocolSnapshot()


def test_continuation_snapshot_rejects_check_use_bundle_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker, runtime = _continuation_bundles(tmp_path)
    document = _continuation_document(checker.manifest_sha256, runtime.manifest_sha256)
    payload = canonical_bytes(document)
    protocol_sha256 = hashlib.sha256(payload).hexdigest()
    verified = verify_continuation_protocol(
        payload,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
        checker_source_bundle=checker,
        runtime_bundle=runtime,
    )
    real_snapshot = protocols.snapshot_source_bundle
    snapshot_calls = 0

    def mutate_after_runtime_snapshot(*args, **kwargs):
        nonlocal snapshot_calls
        detached = real_snapshot(*args, **kwargs)
        snapshot_calls += 1
        if snapshot_calls == 2:
            object.__setattr__(checker.files[0], "payload", b"attacker")
        return detached

    monkeypatch.setattr(protocols, "snapshot_source_bundle", mutate_after_runtime_snapshot)
    with pytest.raises(ValueError, match="changed while"):
        snapshot_continuation_protocol(
            verified,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )


def test_continuation_snapshot_rejects_whole_capsule_self_rehash(
    tmp_path: Path,
) -> None:
    honest_checker, honest_runtime = _continuation_bundles(tmp_path / "honest")
    attacker_checker = _source_bundle(
        tmp_path / "attacker",
        "continuation_checker",
        {"checker.pyz": _checker_zipapp_with_members({"__main__.py": b"raise SystemExit(7)\n"})},
    )
    attacker_runtime = _source_bundle(
        tmp_path / "attacker",
        "portable_runtime",
        {"bin/python3": _portable_runtime()},
        executable_paths=frozenset({"bin/python3"}),
    )
    honest_document = _continuation_document(
        honest_checker.manifest_sha256,
        honest_runtime.manifest_sha256,
    )
    attacker_document = _continuation_document(
        attacker_checker.manifest_sha256,
        attacker_runtime.manifest_sha256,
    )
    honest_payload = canonical_bytes(honest_document)
    attacker_payload = canonical_bytes(attacker_document)
    honest_sha256 = hashlib.sha256(honest_payload).hexdigest()
    honest = verify_continuation_protocol(
        honest_payload,
        expected_protocol_sha256=honest_sha256,
        expected_release_id=_RELEASE,
        checker_source_bundle=honest_checker,
        runtime_bundle=honest_runtime,
    )
    attacker = verify_continuation_protocol(
        attacker_payload,
        expected_protocol_sha256=hashlib.sha256(attacker_payload).hexdigest(),
        expected_release_id=_RELEASE,
        checker_source_bundle=attacker_checker,
        runtime_bundle=attacker_runtime,
    )
    for field_name in (
        "protocol_sha256",
        "checker_source_bundle_sha256",
        "runtime_bundle_sha256",
        "_payload",
        "_checker_source_bundle",
        "_runtime_bundle",
    ):
        object.__setattr__(honest, field_name, getattr(attacker, field_name))

    with pytest.raises(ValueError, match="external commitment"):
        snapshot_continuation_protocol(
            honest,
            expected_protocol_sha256=honest_sha256,
            expected_release_id=_RELEASE,
        )


def test_protocol_rejects_wrong_digest_release_unknown_field_and_bool_version(
    tmp_path: Path,
) -> None:
    bundle = _source_bundle(tmp_path, "generation", {"generator.py": b"pass\n"})
    document = _generation_document(bundle.manifest_sha256)
    payload = canonical_bytes(document)
    with pytest.raises(ValueError, match="external commitment"):
        verify_generation_protocol(
            payload,
            expected_protocol_sha256="0" * 64,
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )
    with pytest.raises(ValueError, match="release identities"):
        verify_generation_protocol(
            payload,
            expected_protocol_sha256=hashlib.sha256(payload).hexdigest(),
            expected_release_id="different-release",
            source_bundle=bundle,
        )

    document["unknown"] = None
    unknown = canonical_bytes(document)
    with pytest.raises(ValueError, match="schema fields differ"):
        verify_generation_protocol(
            unknown,
            expected_protocol_sha256=hashlib.sha256(unknown).hexdigest(),
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )

    del document["unknown"]
    document["schema_version"] = True
    boolean_version = canonical_bytes(document)
    with pytest.raises(ValueError, match="schema_version"):
        verify_generation_protocol(
            boolean_version,
            expected_protocol_sha256=hashlib.sha256(boolean_version).hexdigest(),
            expected_release_id=_RELEASE,
            source_bundle=bundle,
        )


def test_protocol_whole_capsule_rebase_cannot_replace_external_commitment(
    tmp_path: Path,
) -> None:
    honest_bundle = _source_bundle(
        tmp_path / "honest",
        "generation",
        {"generator.py": b"honest\n"},
    )
    attacker_bundle = _source_bundle(
        tmp_path / "attacker",
        "generation",
        {"generator.py": b"attacker\n"},
    )
    honest_document = _generation_document(honest_bundle.manifest_sha256)
    attacker_document = _generation_document(attacker_bundle.manifest_sha256)
    honest_payload = canonical_bytes(honest_document)
    attacker_payload = canonical_bytes(attacker_document)
    honest_digest = hashlib.sha256(honest_payload).hexdigest()
    honest = verify_generation_protocol(
        honest_payload,
        expected_protocol_sha256=honest_digest,
        expected_release_id=_RELEASE,
        source_bundle=honest_bundle,
    )
    attacker = verify_generation_protocol(
        attacker_payload,
        expected_protocol_sha256=hashlib.sha256(attacker_payload).hexdigest(),
        expected_release_id=_RELEASE,
        source_bundle=attacker_bundle,
    )

    for field in (
        "release_id",
        "protocol_sha256",
        "source_bundle_sha256",
        "_payload",
        "_source_bundle",
    ):
        object.__setattr__(honest, field, getattr(attacker, field))

    with pytest.raises(ValueError, match="external commitment"):
        honest.to_dict(
            expected_protocol_sha256=honest_digest,
            expected_release_id=_RELEASE,
        )

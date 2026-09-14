"""Verified protocol documents for hard/OOD generation and continuation checking."""

from __future__ import annotations

import hashlib
import io
import stat
import struct
import zipfile
import zlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Self, cast

from embedbench.hard_ood_provenance import (
    SourceBundleSnapshot,
    VerifiedSourceBundle,
    VerifiedSourceFile,
    parse_canonical_json_bytes,
    snapshot_source_bundle,
    validate_relative_path,
    validate_source_bundle,
)
from embedbench.hard_ood_schema import SeedRequest, canonical_bytes, require_exact_keys

GENERATION_PROTOCOL_SCHEMA = "embedbench.generation-protocol"
SOLVER_PROTOCOL_SCHEMA = "embedbench.solver-protocol"
CONTINUATION_PROTOCOL_SCHEMA = "embedbench.continuation-protocol"
PROTOCOL_SCHEMA_VERSION = 1

GENERATION_PROTOCOL_ID = "hard-ood-generation-v1"
WITNESS_METHOD = "target-guided-inkdrop-v1"
FOCUS_RULE = "degree-quartile-seeded-v1"
REMOVAL_RULE = "four-state-slots-v1"
WINDOW_RULE = "witness-active-induced-v1"
STARTING_SOURCE_ROTATION = (
    "witness",
    "minorminer",
    "cpp_baseline",
    "witness_perturbation",
)
ALLOWED_Q_CAP_SLACK = (0, 1)

SOLVER_IDS = frozenset({"minorminer", "cpp_baseline"})
SOLVER_ATTEMPT_COUNT = 32
SOLVER_SEED_SCHEDULE_RULE = "shared-registered-32-per-problem-v1"
SOLVER_SEED_PROJECTION = "collision-managed-uint32-v1"
SOLVER_SEED_PANEL = "structure"
SOLVER_SEED_CELL = "paired-baseline-comparison"
SOLVER_SELECTION_ORDER = (
    "total_qubits",
    "maximum_chain_length",
    "sum_squared_chain_lengths",
    "canonical_embedding_bytes",
)
SOLVER_ENTRYPOINTS = MappingProxyType(
    {
        "minorminer": "bin/minorminer-runner",
        "cpp_baseline": "bin/lac-embedder",
    }
)
SOLVER_WORK_LIMITS = MappingProxyType(
    {
        "minorminer": ("inner_rounds_per_try", 100),
        "cpp_baseline": ("state_transitions_per_try", 10_000),
    }
)
SOLVER_RESOLVED_PARAMETERS = MappingProxyType(
    {
        "minorminer": MappingProxyType(
            {
                "chainlength_patience": 10,
                "fixed_chains": None,
                "initial_chains": None,
                "inner_rounds": 100,
                "interactive": False,
                "max_beta": None,
                "max_fill": None,
                "max_no_improvement": 10,
                "random_seed_source": "registered_attempt_seed32",
                "restrict_chains": None,
                "return_overlap": False,
                "skip_initialization": False,
                "suspend_chains": None,
                "threads": 1,
                "timeout_seconds": None,
                "tries": 10,
                "verbose": 0,
            }
        ),
        "cpp_baseline": MappingProxyType(
            {
                "anytime": False,
                "max_candidates": 8,
                "max_transitions": 10_000,
                "random_seed_source": "registered_attempt_seed32",
                "return_diagnostics": True,
                "search_profile": "lns_v1",
                "threads": 1,
                "timeout_seconds": None,
                "tries": 10,
            }
        ),
    }
)

CONTINUATION_PROTOCOL_ID = "exact-continuation-v1"
CONTINUATION_NODE_BUDGET = 10_000_000
CONTINUATION_WATCHDOG_SECONDS = 900
CONTINUATION_NODE_COUNT_RULE = "dfs_state_entry_v1"
CONTINUATION_COMMAND_TEMPLATE = (
    "{runtime_root}/{runtime_python}",
    "-I",
    "{checker_root}/{checker_zipapp}",
    "{input_path}",
    "{output_path}",
)
CONTINUATION_OBJECTIVE_ORDER = (
    "completion_feasible",
    "largest_free_component_node_fraction",
    "largest_free_component_edge_connectivity",
    "negative_largest_free_component_articulation_count",
    "minimum_logical_contact_multiplicity",
    "negative_terminal_total_qubits",
    "negative_terminal_maximum_chain_length",
)
VARIABLE_ORDER_RULE = "negative_logical_degree_then_variable_v1"
CANDIDATE_CHAIN_ORDER_RULE = "length_then_unsigned_nodes_v1"
CONTINUATION_INPUT_SCHEMA = "embedbench.continuation-batch-input-v1"
CONTINUATION_OUTPUT_SCHEMA = "embedbench.continuation-batch-certificate-v1"
RUNTIME_PYTHON_PATH = "bin/python3"
CHECKER_ZIPAPP_PATH = "checker.pyz"
CONTINUATION_INPUT_BYTE_LIMIT = 8 * 1024 * 1024
CONTINUATION_OUTPUT_BYTE_LIMIT = 32 * 1024 * 1024
CONTINUATION_STDOUT_BYTE_LIMIT = 1024 * 1024
CONTINUATION_STDERR_BYTE_LIMIT = 1024 * 1024
CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT = 512 * 1024 * 1024
CONTINUATION_ZIP_MEMBER_LIMIT = 1024
CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT = 64 * 1024 * 1024
CONTINUATION_PATH_DEPTH_LIMIT = 16

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_PROTOCOL_SEAL = object()
_GENERATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "protocol_id",
        "source_bundle_sha256",
        "witness_method",
        "focus_rule",
        "removal_rule",
        "window_rule",
        "starting_source_rotation",
        "max_window_free",
        "maximum_chain_length",
        "allowed_q_cap_slack",
    }
)
_SOLVER_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "solver_id",
        "source_bundle_sha256",
        "entrypoint",
        "binary_sha256",
        "attempt_count",
        "seed_schedule_rule",
        "seed_projection",
        "selection_order",
        "deterministic_work_limit",
        "work_limit_unit",
        "safety_watchdog_seconds",
        "resolved_parameters",
        "reference_executor",
        "software_versions",
    }
)
_CONTINUATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "release_id",
        "protocol_id",
        "checker_source_bundle_sha256",
        "runtime_bundle_sha256",
        "runtime_python",
        "checker_zipapp",
        "command_template",
        "clean_environment",
        "input_schema",
        "output_schema",
        "node_budget",
        "node_count_rule",
        "max_window_free",
        "maximum_chain_length",
        "variable_order_rule",
        "candidate_chain_order_rule",
        "objective_order",
        "safety_watchdog_seconds",
        "input_byte_limit",
        "output_byte_limit",
        "stdout_byte_limit",
        "stderr_byte_limit",
        "retained_bundle_byte_limit",
        "zip_member_limit",
        "zip_uncompressed_byte_limit",
        "path_depth_limit",
    }
)
_REFERENCE_EXECUTOR_FIELDS = frozenset({"site", "cpu_model", "logical_cores", "operating_system"})
_SOFTWARE_VERSION_FIELDS = frozenset(
    {"solver", "compiler_id", "compiler_version", "dependency_versions"}
)


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


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_registered_value(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} does not match the registered protocol")


def _validate_reference_executor(value: object) -> None:
    executor = require_exact_keys(value, _REFERENCE_EXECUTOR_FIELDS, "reference_executor")
    _require_registered_value(executor["site"], "apollo", "reference_executor.site")
    _require_registered_value(
        executor["logical_cores"],
        1,
        "reference_executor.logical_cores",
    )
    _require_text(executor["cpu_model"], "reference_executor.cpu_model")
    _require_text(executor["operating_system"], "reference_executor.operating_system")


def _validate_software_versions(value: object) -> None:
    versions = require_exact_keys(value, _SOFTWARE_VERSION_FIELDS, "software_versions")
    _require_text(versions["solver"], "software_versions.solver")
    _require_text(versions["compiler_id"], "software_versions.compiler_id")
    _require_text(versions["compiler_version"], "software_versions.compiler_version")
    dependencies = versions["dependency_versions"]
    if type(dependencies) is not dict or not dependencies:
        raise ValueError("software_versions.dependency_versions must be a nonempty object")
    for dependency, version in dependencies.items():
        _require_text(dependency, "software dependency name")
        _require_text(version, f"software dependency version for {dependency!r}")


def solver_attempt_seed_request(
    *,
    release_id: str,
    partition: str,
    problem_sha256: str,
    attempt_index: int,
) -> SeedRequest:
    """Build the one solver-neutral request consumed by both structure baselines."""

    if type(attempt_index) is not int or not 0 <= attempt_index < SOLVER_ATTEMPT_COUNT:
        raise ValueError(f"attempt_index must be in [0, {SOLVER_ATTEMPT_COUNT})")
    return SeedRequest(
        release_id=release_id,
        purpose="solver_attempt",
        partition=partition,
        panel=SOLVER_SEED_PANEL,
        cell=SOLVER_SEED_CELL,
        task_type=None,
        problem_sha256=problem_sha256,
        state_sha256=None,
        candidate_index=None,
        strength_index=None,
        label_stage=None,
        replicate=attempt_index,
    )


def solver_attempt_seed_schedule(
    *,
    release_id: str,
    partition: str,
    problem_sha256: str,
) -> tuple[SeedRequest, ...]:
    """Construct the single registered 32-request schedule shared by both adapters."""

    return tuple(
        solver_attempt_seed_request(
            release_id=release_id,
            partition=partition,
            problem_sha256=problem_sha256,
            attempt_index=attempt_index,
        )
        for attempt_index in range(SOLVER_ATTEMPT_COUNT)
    )


def solver_attempt_seed_schedule_sha256(requests: tuple[SeedRequest, ...]) -> str:
    """Authenticate an exact, complete solver-neutral attempt schedule."""

    if type(requests) is not tuple or len(requests) != SOLVER_ATTEMPT_COUNT:
        raise ValueError(f"solver seed schedule must contain {SOLVER_ATTEMPT_COUNT} requests")
    if any(type(request) is not SeedRequest for request in requests):
        raise TypeError("solver seed schedule entries must be exact SeedRequest values")
    first = requests[0]
    expected = solver_attempt_seed_schedule(
        release_id=first.release_id,
        partition=first.partition,
        problem_sha256=_require_sha256(first.problem_sha256, "problem_sha256"),
    )
    preimages = tuple(request.canonical_preimage() for request in requests)
    expected_preimages = tuple(request.canonical_preimage() for request in expected)
    if preimages != expected_preimages:
        raise ValueError("solver seed schedule is not the registered solver-neutral schedule")
    return hashlib.sha256(canonical_bytes([request.to_dict() for request in expected])).hexdigest()


def _is_thin_macho_executable(payload: bytes) -> bool:
    magic = payload[:4]
    formats = {
        b"\xfe\xed\xfa\xce": (">", 28),
        b"\xce\xfa\xed\xfe": ("<", 28),
        b"\xfe\xed\xfa\xcf": (">", 32),
        b"\xcf\xfa\xed\xfe": ("<", 32),
    }
    format_spec = formats.get(magic)
    if format_spec is None:
        return False
    endian, header_size = format_spec
    if len(payload) < header_size:
        return False
    cpu_type, _cpu_subtype, file_type, command_count, command_bytes, _flags = struct.unpack_from(
        f"{endian}IIIIII", payload, 4
    )
    accepted_cpu_types = {0x01000007, 0x0100000C} if header_size == 32 else {7, 12}
    command_end = header_size + command_bytes
    if (
        cpu_type not in accepted_cpu_types
        or file_type != 2
        or command_count == 0
        or command_count > 4096
        or command_bytes == 0
        or command_end > len(payload)
    ):
        return False

    executable_ranges: list[tuple[int, int]] = []
    main_entry_offsets: list[int] = []
    has_unix_thread_entry = False
    command_offset = header_size
    for _index in range(command_count):
        if command_offset + 8 > command_end:
            return False
        command, command_size = struct.unpack_from(f"{endian}II", payload, command_offset)
        command_alignment = 8 if header_size == 32 else 4
        if (
            command_size < 8
            or command_size % command_alignment != 0
            or command_offset + command_size > command_end
        ):
            return False
        if command == 0x19 and header_size == 32:
            if command_size < 72:
                return False
            file_offset, file_size = struct.unpack_from(f"{endian}QQ", payload, command_offset + 40)
            initial_protection = struct.unpack_from(f"{endian}I", payload, command_offset + 60)[0]
            if file_offset + file_size > len(payload):
                return False
            if file_size > 0 and initial_protection & 0x4:
                executable_ranges.append((file_offset, file_offset + file_size))
        elif command == 0x1 and header_size == 28:
            if command_size < 56:
                return False
            file_offset, file_size = struct.unpack_from(f"{endian}II", payload, command_offset + 32)
            initial_protection = struct.unpack_from(f"{endian}I", payload, command_offset + 44)[0]
            if file_offset + file_size > len(payload):
                return False
            if file_size > 0 and initial_protection & 0x4:
                executable_ranges.append((file_offset, file_offset + file_size))
        elif command == 0x80000028:
            if command_size < 24:
                return False
            main_entry_offsets.append(
                struct.unpack_from(f"{endian}Q", payload, command_offset + 8)[0]
            )
        elif command == 0x5:
            has_unix_thread_entry = True
        command_offset += command_size

    if command_offset != command_end or not executable_ranges:
        return False
    if main_entry_offsets:
        return len(main_entry_offsets) == 1 and any(
            start <= main_entry_offsets[0] < end for start, end in executable_ranges
        )
    return has_unix_thread_entry


def _is_elf_executable(payload: bytes) -> bool:
    if len(payload) < 16 or payload[:4] != b"\x7fELF":
        return False
    elf_class = payload[4]
    data_encoding = payload[5]
    if elf_class not in {1, 2} or data_encoding not in {1, 2} or payload[6] != 1:
        return False
    if payload[7] not in {0, 3}:
        return False
    endian = "<" if data_encoding == 1 else ">"
    header_size = 52 if elf_class == 1 else 64
    header_size_offset = 40 if elf_class == 1 else 52
    if len(payload) < header_size:
        return False
    file_type, machine, version = struct.unpack_from(f"{endian}HHI", payload, 16)
    recorded_header_size = struct.unpack_from(f"{endian}H", payload, header_size_offset)[0]
    accepted_formats = {
        (1, 1, 3),
        (1, 1, 40),
        (2, 1, 62),
        (2, 1, 183),
    }
    if (
        file_type not in {2, 3}
        or (elf_class, data_encoding, machine) not in accepted_formats
        or version != 1
        or recorded_header_size != header_size
    ):
        return False
    if elf_class == 1:
        entry_point, program_header_offset = struct.unpack_from(f"{endian}II", payload, 24)
        program_header_entry_size, program_header_count = struct.unpack_from(
            f"{endian}HH", payload, 42
        )
        expected_entry_size = 32
    else:
        entry_point, program_header_offset = struct.unpack_from(f"{endian}QQ", payload, 24)
        program_header_entry_size, program_header_count = struct.unpack_from(
            f"{endian}HH", payload, 54
        )
        expected_entry_size = 56
    table_end = program_header_offset + program_header_entry_size * program_header_count
    if (
        program_header_offset < header_size
        or program_header_entry_size != expected_entry_size
        or program_header_count == 0
        or program_header_count > 4096
        or table_end > len(payload)
    ):
        return False
    for index in range(program_header_count):
        offset = program_header_offset + index * program_header_entry_size
        if elf_class == 1:
            (
                segment_type,
                file_offset,
                virtual_address,
                _physical_address,
                file_size,
                memory_size,
            ) = struct.unpack_from(f"{endian}IIIIII", payload, offset)
            flags = struct.unpack_from(f"{endian}I", payload, offset + 24)[0]
        else:
            segment_type, flags = struct.unpack_from(f"{endian}II", payload, offset)
            file_offset, virtual_address, _physical_address, file_size, memory_size = (
                struct.unpack_from(f"{endian}QQQQQ", payload, offset + 8)
            )
        if (
            segment_type == 1
            and flags & 0x1
            and file_size > 0
            and memory_size >= file_size
            and file_offset + file_size <= len(payload)
            and virtual_address <= entry_point < virtual_address + memory_size
        ):
            return True
    return False


def _is_fat_macho_executable(payload: bytes) -> bool:
    magic = payload[:4]
    formats = {
        b"\xca\xfe\xba\xbe": (">", False),
        b"\xbe\xba\xfe\xca": ("<", False),
        b"\xca\xfe\xba\xbf": (">", True),
        b"\xbf\xba\xfe\xca": ("<", True),
    }
    format_spec = formats.get(magic)
    if format_spec is None or len(payload) < 8:
        return False
    endian, is_64_bit = format_spec
    architecture_count = struct.unpack_from(f"{endian}I", payload, 4)[0]
    entry_size = 32 if is_64_bit else 20
    table_end = 8 + architecture_count * entry_size
    if architecture_count == 0 or architecture_count > 64 or table_end > len(payload):
        return False
    ranges: list[tuple[int, int]] = []
    cpu_types: set[int] = set()
    for index in range(architecture_count):
        offset = 8 + index * entry_size
        if is_64_bit:
            cpu_type, _subtype, slice_offset, slice_size, alignment, reserved = struct.unpack_from(
                f"{endian}IIQQII", payload, offset
            )
            if reserved != 0:
                return False
        else:
            cpu_type, _subtype, slice_offset, slice_size, alignment = struct.unpack_from(
                f"{endian}IIIII", payload, offset
            )
        slice_end = slice_offset + slice_size
        if (
            cpu_type == 0
            or cpu_type in cpu_types
            or alignment > 31
            or slice_size == 0
            or slice_offset < table_end
            or slice_offset % (1 << alignment) != 0
            or slice_end > len(payload)
        ):
            return False
        nested_magic = payload[slice_offset : slice_offset + 4]
        nested_endian = ">" if nested_magic in {b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf"} else "<"
        if slice_size < 8:
            return False
        nested_cpu_type = struct.unpack_from(f"{nested_endian}I", payload, slice_offset + 4)[0]
        if nested_cpu_type != cpu_type:
            return False
        cpu_types.add(cpu_type)
        ranges.append((slice_offset, slice_end))
        if not _is_thin_macho_executable(payload[slice_offset:slice_end]):
            return False
    ordered_ranges = sorted(ranges)
    return all(
        left_end <= right_start
        for (_, left_end), (right_start, _) in zip(ordered_ranges, ordered_ranges[1:], strict=False)
    )


def _has_valid_native_executable_header(payload: bytes) -> bool:
    """Perform a static format preflight, not an execution attestation."""

    return (
        _is_elf_executable(payload)
        or _is_thin_macho_executable(payload)
        or _is_fat_macho_executable(payload)
    )


def _snapshot_entry(
    bundle: SourceBundleSnapshot,
    relative_path: str,
) -> VerifiedSourceFile:
    if type(bundle) is not SourceBundleSnapshot:
        raise TypeError("continuation bundle must be an immutable source snapshot")
    return bundle.file_entry(relative_path)


def _require_native_executable_snapshot(
    bundle: SourceBundleSnapshot,
    relative_path: str,
) -> None:
    entry = _snapshot_entry(bundle, relative_path)
    if not entry.executable:
        raise ValueError("portable runtime entrypoint must carry executable intent")
    if not _has_valid_native_executable_header(entry.payload):
        raise ValueError("entrypoint must pass the native executable static preflight")


def _require_native_executable(
    bundle: VerifiedSourceBundle,
    relative_path: str,
    *,
    expected_manifest_sha256: str,
    expected_role: str,
    expected_release_id: str,
) -> None:
    entry = bundle.file_entry(
        relative_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_role=expected_role,
        expected_release_id=expected_release_id,
    )
    if not entry.executable:
        raise ValueError("portable runtime entrypoint must carry executable intent")
    if not _has_valid_native_executable_header(entry.payload):
        raise ValueError("entrypoint must pass the native executable static preflight")


def _zip_eocd_member_count(payload: bytes, *, member_limit: int) -> int:
    """Count the classic central directory before ZipFile materializes metadata."""

    minimum_eocd_size = 22
    maximum_comment_size = (1 << 16) - 1
    search_start = max(0, len(payload) - minimum_eocd_size - maximum_comment_size)
    eocd_offset = payload.rfind(b"PK\x05\x06", search_start)
    if eocd_offset < 0 or eocd_offset + minimum_eocd_size > len(payload):
        raise ValueError("continuation checker entrypoint must be a valid zipapp")
    (
        signature,
        disk_number,
        central_directory_disk,
        disk_member_count,
        member_count,
        central_directory_size,
        central_directory_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", payload, eocd_offset)
    if signature != b"PK\x05\x06" or eocd_offset + minimum_eocd_size + comment_size != len(payload):
        raise ValueError("continuation checker entrypoint has an invalid zip directory")
    if (
        disk_number != 0
        or central_directory_disk != 0
        or disk_member_count != member_count
        or member_count == 0xFFFF
        or central_directory_size == 0xFFFFFFFF
        or central_directory_offset == 0xFFFFFFFF
    ):
        raise ValueError("continuation checker zipapp must be one non-ZIP64 archive")
    if member_count > member_limit:
        raise ValueError("continuation checker zipapp exceeds the member count limit")
    central_directory_start = eocd_offset - central_directory_size
    if central_directory_start < 0 or central_directory_offset > central_directory_start:
        raise ValueError("continuation checker entrypoint has an invalid zip directory")

    position = central_directory_start
    actual_member_count = 0
    central_header_size = 46
    while position < eocd_offset:
        if (
            position + central_header_size > eocd_offset
            or payload[position : position + 4] != b"PK\x01\x02"
        ):
            raise ValueError("continuation checker zipapp member inventory is inconsistent")
        filename_size = struct.unpack_from("<H", payload, position + 28)[0]
        extra_size = struct.unpack_from("<H", payload, position + 30)[0]
        member_comment_size = struct.unpack_from("<H", payload, position + 32)[0]
        disk_start = struct.unpack_from("<H", payload, position + 34)[0]
        compressed_size = struct.unpack_from("<L", payload, position + 20)[0]
        uncompressed_size = struct.unpack_from("<L", payload, position + 24)[0]
        local_header_offset = struct.unpack_from("<L", payload, position + 42)[0]
        if (
            disk_start != 0
            or compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_header_offset == 0xFFFFFFFF
        ):
            raise ValueError("continuation checker zipapp must be one non-ZIP64 archive")
        position += central_header_size + filename_size + extra_size + member_comment_size
        if position > eocd_offset:
            raise ValueError("continuation checker zipapp member inventory is inconsistent")
        actual_member_count += 1
        if actual_member_count > member_limit:
            raise ValueError("continuation checker zipapp exceeds the member count limit")
    if actual_member_count != member_count:
        raise ValueError("continuation checker zipapp member inventory is inconsistent")
    return actual_member_count


def _require_valid_zipapp(
    bundle: SourceBundleSnapshot,
    relative_path: str,
    *,
    member_limit: int,
    uncompressed_byte_limit: int,
    path_depth_limit: int,
) -> None:
    entry = _snapshot_entry(bundle, relative_path)
    expected_member_count = _zip_eocd_member_count(
        entry.payload,
        member_limit=member_limit,
    )
    try:
        archive = zipfile.ZipFile(io.BytesIO(entry.payload), "r")
    except zipfile.BadZipFile as error:
        raise ValueError("continuation checker entrypoint must be a valid zipapp") from error
    with archive:
        infos = archive.infolist()
        if len(infos) != expected_member_count or len(infos) > member_limit:
            raise ValueError("continuation checker zipapp member inventory is inconsistent")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or "__main__.py" not in names:
            raise ValueError("continuation checker zipapp requires one unique __main__.py")
        declared_uncompressed_bytes = 0
        for info in infos:
            path = info.filename[:-1] if info.is_dir() else info.filename
            validate_relative_path(path)
            if len(path.split("/")) > path_depth_limit:
                raise ValueError("continuation checker zipapp exceeds the path depth limit")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError("continuation checker zipapp must not contain symlinks")
            if info.flag_bits & 0x1:
                raise ValueError("continuation checker zipapp must not contain encrypted members")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError(
                    "continuation checker zipapp uses an unsupported compression method"
                )
            if type(info.file_size) is not int or info.file_size < 0:
                raise ValueError("continuation checker zipapp has an invalid member size")
            declared_uncompressed_bytes += info.file_size
            if declared_uncompressed_bytes > uncompressed_byte_limit:
                raise ValueError(
                    "continuation checker zipapp exceeds the uncompressed payload limit"
                )

        actual_uncompressed_bytes = 0
        try:
            for info in infos:
                member_byte_count = 0
                with archive.open(info, "r") as member:
                    while True:
                        remaining = uncompressed_byte_limit - actual_uncompressed_bytes
                        chunk = member.read(min(1024 * 1024, remaining + 1))
                        if not chunk:
                            break
                        member_byte_count += len(chunk)
                        actual_uncompressed_bytes += len(chunk)
                        if actual_uncompressed_bytes > uncompressed_byte_limit:
                            raise ValueError(
                                "continuation checker zipapp exceeds the uncompressed payload limit"
                            )
                if member_byte_count != info.file_size:
                    raise ValueError("continuation checker zipapp member size is inconsistent")
        except (RuntimeError, zipfile.BadZipFile, zlib.error) as error:
            raise ValueError("continuation checker zipapp failed bounded validation") from error
        if actual_uncompressed_bytes != declared_uncompressed_bytes:
            raise ValueError("continuation checker zipapp payload size is inconsistent")


def _exact_json_array(value: object, expected: tuple[object, ...], name: str) -> None:
    if type(value) is not list or tuple(value) != expected:
        raise ValueError(f"{name} does not match the registered protocol")
    for item in value:
        if type(item) not in {str, int}:
            raise TypeError(f"{name} contains a noncanonical scalar")


def _parse_protocol(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_schema: str,
    fields: frozenset[str],
) -> tuple[dict[str, object], str]:
    expected = _require_sha256(expected_sha256, "expected_protocol_sha256")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("protocol digest disagrees with its external commitment")
    document = require_exact_keys(
        parse_canonical_json_bytes(payload, name="protocol document"),
        fields,
        "protocol document",
    )
    if type(document["schema"]) is not str or document["schema"] != expected_schema:
        raise ValueError(f"protocol requires schema {expected_schema!r}")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != PROTOCOL_SCHEMA_VERSION
    ):
        raise ValueError(f"protocol requires schema_version {PROTOCOL_SCHEMA_VERSION}")
    return document, expected


def _require_release_bundle(
    document: dict[str, object],
    *,
    expected_release_id: str,
    bundle: VerifiedSourceBundle,
    expected_role: str,
    digest_field: str,
) -> VerifiedSourceBundle:
    release_id = _require_text(expected_release_id, "expected_release_id")
    if type(document["release_id"]) is not str or document["release_id"] != release_id:
        raise ValueError("protocol and source bundle release identities do not match")
    manifest_sha256 = _require_sha256(document[digest_field], digest_field)
    checked_bundle = validate_source_bundle(
        bundle,
        expected_manifest_sha256=manifest_sha256,
        expected_role=expected_role,
        expected_release_id=release_id,
    )
    return checked_bundle


@dataclass(frozen=True, slots=True, init=False)
class VerifiedGenerationProtocol:
    release_id: str
    protocol_sha256: str
    source_bundle_sha256: str
    _payload: bytes
    _source_bundle: VerifiedSourceBundle
    _seal: object

    def to_dict(
        self,
        *,
        expected_protocol_sha256: str,
        expected_release_id: str,
    ) -> dict[str, object]:
        validate_protocol(
            self,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_release_id=expected_release_id,
        )
        return parse_canonical_json_bytes(self._payload, name="verified generation protocol")


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSolverProtocol:
    release_id: str
    solver_id: str
    protocol_sha256: str
    source_bundle_sha256: str
    entrypoint: str
    binary_sha256: str
    deterministic_work_limit: int
    work_limit_unit: str
    _payload: bytes
    _source_bundle: VerifiedSourceBundle
    _seal: object

    def to_dict(
        self,
        *,
        expected_protocol_sha256: str,
        expected_release_id: str,
    ) -> dict[str, object]:
        validate_protocol(
            self,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_release_id=expected_release_id,
        )
        return parse_canonical_json_bytes(self._payload, name="verified solver protocol")


@dataclass(frozen=True, slots=True, init=False)
class VerifiedContinuationProtocol:
    release_id: str
    protocol_sha256: str
    checker_source_bundle_sha256: str
    runtime_bundle_sha256: str
    runtime_python: str
    checker_zipapp: str
    node_budget: int
    safety_watchdog_seconds: int
    input_byte_limit: int
    output_byte_limit: int
    stdout_byte_limit: int
    stderr_byte_limit: int
    retained_bundle_byte_limit: int
    zip_member_limit: int
    zip_uncompressed_byte_limit: int
    path_depth_limit: int
    _payload: bytes
    _checker_source_bundle: VerifiedSourceBundle
    _runtime_bundle: VerifiedSourceBundle
    _seal: object

    def to_dict(
        self,
        *,
        expected_protocol_sha256: str,
        expected_release_id: str,
    ) -> dict[str, object]:
        snapshot = snapshot_continuation_protocol(
            self,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_release_id=expected_release_id,
        )
        return parse_canonical_json_bytes(
            snapshot.protocol_bytes,
            name="verified continuation protocol",
        )

    def runtime_file_bytes(
        self,
        relative_path: str,
        *,
        expected_protocol_sha256: str,
        expected_release_id: str,
    ) -> bytes:
        snapshot = snapshot_continuation_protocol(
            self,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_release_id=expected_release_id,
        )
        normalized = validate_relative_path(relative_path)
        for entry in snapshot.runtime_files:
            if entry.relative_path == normalized:
                return entry.payload
        raise KeyError(relative_path)

    def checker_file_bytes(
        self,
        relative_path: str,
        *,
        expected_protocol_sha256: str,
        expected_release_id: str,
    ) -> bytes:
        snapshot = snapshot_continuation_protocol(
            self,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_release_id=expected_release_id,
        )
        normalized = validate_relative_path(relative_path)
        for entry in snapshot.checker_files:
            if entry.relative_path == normalized:
                return entry.payload
        raise KeyError(relative_path)


_CONTINUATION_SNAPSHOT_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class ContinuationProtocolSnapshot:
    """One detached, closed protocol plus its exact retained execution payloads."""

    release_id: str
    protocol_sha256: str
    protocol_bytes: bytes
    checker_source_bundle_sha256: str
    runtime_bundle_sha256: str
    runtime_python: str
    checker_zipapp: str
    command_template: tuple[str, ...]
    input_schema: str
    output_schema: str
    watchdog_seconds: int
    input_byte_limit: int
    output_byte_limit: int
    stdout_byte_limit: int
    stderr_byte_limit: int
    retained_bundle_byte_limit: int
    zip_member_limit: int
    zip_uncompressed_byte_limit: int
    path_depth_limit: int
    checker_files: tuple[VerifiedSourceFile, ...]
    runtime_files: tuple[VerifiedSourceFile, ...]
    checker_zipapp_sha256: str
    runtime_python_sha256: str
    _seal: object

    def __new__(cls) -> Self:
        raise TypeError("continuation snapshots can only be constructed by the verifier")


def verify_generation_protocol(
    protocol_bytes: bytes,
    *,
    expected_protocol_sha256: str,
    expected_release_id: str,
    source_bundle: VerifiedSourceBundle,
) -> VerifiedGenerationProtocol:
    document, digest = _parse_protocol(
        protocol_bytes,
        expected_sha256=expected_protocol_sha256,
        expected_schema=GENERATION_PROTOCOL_SCHEMA,
        fields=_GENERATION_FIELDS,
    )
    bundle = _require_release_bundle(
        document,
        expected_release_id=expected_release_id,
        bundle=source_bundle,
        expected_role="generation",
        digest_field="source_bundle_sha256",
    )
    required_scalars = {
        "protocol_id": GENERATION_PROTOCOL_ID,
        "witness_method": WITNESS_METHOD,
        "focus_rule": FOCUS_RULE,
        "removal_rule": REMOVAL_RULE,
        "window_rule": WINDOW_RULE,
        "max_window_free": 28,
        "maximum_chain_length": 6,
    }
    for name, expected in required_scalars.items():
        _require_registered_value(document[name], expected, name)
    _exact_json_array(
        document["starting_source_rotation"],
        STARTING_SOURCE_ROTATION,
        "starting_source_rotation",
    )
    _exact_json_array(
        document["allowed_q_cap_slack"],
        ALLOWED_Q_CAP_SLACK,
        "allowed_q_cap_slack",
    )

    verified = object.__new__(VerifiedGenerationProtocol)
    object.__setattr__(verified, "release_id", bundle.release_id)
    object.__setattr__(verified, "protocol_sha256", digest)
    object.__setattr__(verified, "source_bundle_sha256", bundle.manifest_sha256)
    object.__setattr__(verified, "_payload", bytes(protocol_bytes))
    object.__setattr__(verified, "_source_bundle", bundle)
    object.__setattr__(verified, "_seal", _PROTOCOL_SEAL)
    return verified


def verify_solver_protocol(
    protocol_bytes: bytes,
    *,
    expected_protocol_sha256: str,
    expected_release_id: str,
    source_bundle: VerifiedSourceBundle,
) -> VerifiedSolverProtocol:
    document, digest = _parse_protocol(
        protocol_bytes,
        expected_sha256=expected_protocol_sha256,
        expected_schema=SOLVER_PROTOCOL_SCHEMA,
        fields=_SOLVER_FIELDS,
    )
    solver_id = _require_text(document["solver_id"], "solver_id")
    if solver_id not in SOLVER_IDS:
        raise ValueError("solver_id is not registered")
    bundle = _require_release_bundle(
        document,
        expected_release_id=expected_release_id,
        bundle=source_bundle,
        expected_role=solver_id,
        digest_field="source_bundle_sha256",
    )
    entrypoint = _require_text(document["entrypoint"], "entrypoint")
    _require_registered_value(entrypoint, SOLVER_ENTRYPOINTS[solver_id], "solver entrypoint")
    validate_relative_path(entrypoint)
    source_manifest_sha256 = _require_sha256(
        document["source_bundle_sha256"],
        "source_bundle_sha256",
    )
    _require_native_executable(
        bundle,
        entrypoint,
        expected_manifest_sha256=source_manifest_sha256,
        expected_role=solver_id,
        expected_release_id=expected_release_id,
    )
    binary_sha256 = _require_sha256(document["binary_sha256"], "binary_sha256")
    if (
        binary_sha256
        != bundle.file_entry(
            entrypoint,
            expected_manifest_sha256=source_manifest_sha256,
            expected_role=solver_id,
            expected_release_id=expected_release_id,
        ).sha256
    ):
        raise ValueError("solver binary digest does not match its verified bundle entrypoint")
    if (
        type(document["attempt_count"]) is not int
        or document["attempt_count"] != SOLVER_ATTEMPT_COUNT
    ):
        raise ValueError(f"solver protocol requires exactly {SOLVER_ATTEMPT_COUNT} attempts")
    _require_registered_value(
        document["seed_schedule_rule"],
        SOLVER_SEED_SCHEDULE_RULE,
        "solver seed_schedule_rule",
    )
    _require_registered_value(
        document["seed_projection"],
        SOLVER_SEED_PROJECTION,
        "solver seed_projection",
    )
    _exact_json_array(document["selection_order"], SOLVER_SELECTION_ORDER, "selection_order")
    work_limit = _require_positive_int(
        document["deterministic_work_limit"],
        "deterministic_work_limit",
    )
    expected_unit, expected_limit = SOLVER_WORK_LIMITS[solver_id]
    _require_registered_value(work_limit, expected_limit, "solver deterministic work limit")
    work_limit_unit = _require_text(document["work_limit_unit"], "work_limit_unit")
    _require_registered_value(work_limit_unit, expected_unit, "solver work_limit_unit")
    _require_registered_value(
        document["safety_watchdog_seconds"],
        600,
        "solver safety watchdog",
    )
    if type(document["resolved_parameters"]) is not dict:
        raise TypeError("resolved_parameters must be an exact JSON object")
    if canonical_bytes(document["resolved_parameters"]) != canonical_bytes(
        dict(SOLVER_RESOLVED_PARAMETERS[solver_id])
    ):
        raise ValueError("resolved_parameters do not match the registered solver profile")
    _validate_reference_executor(document["reference_executor"])
    _validate_software_versions(document["software_versions"])

    verified = object.__new__(VerifiedSolverProtocol)
    object.__setattr__(verified, "release_id", bundle.release_id)
    object.__setattr__(verified, "solver_id", solver_id)
    object.__setattr__(verified, "protocol_sha256", digest)
    object.__setattr__(verified, "source_bundle_sha256", bundle.manifest_sha256)
    object.__setattr__(verified, "entrypoint", entrypoint)
    object.__setattr__(verified, "binary_sha256", binary_sha256)
    object.__setattr__(verified, "deterministic_work_limit", work_limit)
    object.__setattr__(verified, "work_limit_unit", work_limit_unit)
    object.__setattr__(verified, "_payload", bytes(protocol_bytes))
    object.__setattr__(verified, "_source_bundle", bundle)
    object.__setattr__(verified, "_seal", _PROTOCOL_SEAL)
    return verified


def _validate_continuation_payload(
    protocol_bytes: bytes,
    *,
    expected_protocol_sha256: str,
    expected_release_id: str,
    checker_source_bundle: VerifiedSourceBundle,
    runtime_bundle: VerifiedSourceBundle,
) -> tuple[
    dict[str, object],
    str,
    SourceBundleSnapshot,
    SourceBundleSnapshot,
]:
    document, digest = _parse_protocol(
        protocol_bytes,
        expected_sha256=expected_protocol_sha256,
        expected_schema=CONTINUATION_PROTOCOL_SCHEMA,
        fields=_CONTINUATION_FIELDS,
    )
    release_id = _require_text(expected_release_id, "expected_release_id")
    if type(document["release_id"]) is not str or document["release_id"] != release_id:
        raise ValueError("protocol and source bundle release identities do not match")
    registered_values: dict[str, object] = {
        "protocol_id": CONTINUATION_PROTOCOL_ID,
        "node_budget": CONTINUATION_NODE_BUDGET,
        "node_count_rule": CONTINUATION_NODE_COUNT_RULE,
        "max_window_free": 28,
        "maximum_chain_length": 6,
        "variable_order_rule": VARIABLE_ORDER_RULE,
        "candidate_chain_order_rule": CANDIDATE_CHAIN_ORDER_RULE,
        "input_schema": CONTINUATION_INPUT_SCHEMA,
        "output_schema": CONTINUATION_OUTPUT_SCHEMA,
        "input_byte_limit": CONTINUATION_INPUT_BYTE_LIMIT,
        "output_byte_limit": CONTINUATION_OUTPUT_BYTE_LIMIT,
        "stdout_byte_limit": CONTINUATION_STDOUT_BYTE_LIMIT,
        "stderr_byte_limit": CONTINUATION_STDERR_BYTE_LIMIT,
        "retained_bundle_byte_limit": CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT,
        "zip_member_limit": CONTINUATION_ZIP_MEMBER_LIMIT,
        "zip_uncompressed_byte_limit": CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT,
        "path_depth_limit": CONTINUATION_PATH_DEPTH_LIMIT,
    }
    for name, expected in registered_values.items():
        _require_registered_value(document[name], expected, name)
    _exact_json_array(
        document["command_template"],
        CONTINUATION_COMMAND_TEMPLATE,
        "command_template",
    )
    _exact_json_array(
        document["objective_order"],
        CONTINUATION_OBJECTIVE_ORDER,
        "objective_order",
    )
    if type(document["clean_environment"]) is not dict or document["clean_environment"]:
        raise ValueError("continuation checker requires an empty clean environment")
    runtime_python = _require_text(document["runtime_python"], "runtime_python")
    checker_zipapp = _require_text(document["checker_zipapp"], "checker_zipapp")
    _require_registered_value(runtime_python, RUNTIME_PYTHON_PATH, "runtime_python")
    _require_registered_value(checker_zipapp, CHECKER_ZIPAPP_PATH, "checker_zipapp")
    validate_relative_path(runtime_python)
    validate_relative_path(checker_zipapp)
    runtime_manifest_sha256 = _require_sha256(
        document["runtime_bundle_sha256"],
        "runtime_bundle_sha256",
    )
    checker_manifest_sha256 = _require_sha256(
        document["checker_source_bundle_sha256"],
        "checker_source_bundle_sha256",
    )
    retained_bundle_byte_limit = _require_positive_int(
        document["retained_bundle_byte_limit"],
        "retained_bundle_byte_limit",
    )
    path_depth_limit = _require_positive_int(
        document["path_depth_limit"],
        "path_depth_limit",
    )
    checker = snapshot_source_bundle(
        checker_source_bundle,
        expected_manifest_sha256=checker_manifest_sha256,
        expected_role="continuation_checker",
        expected_release_id=release_id,
        maximum_total_payload_bytes=retained_bundle_byte_limit,
        maximum_path_depth=path_depth_limit,
    )
    runtime = snapshot_source_bundle(
        runtime_bundle,
        expected_manifest_sha256=runtime_manifest_sha256,
        expected_role="portable_runtime",
        expected_release_id=release_id,
        maximum_total_payload_bytes=retained_bundle_byte_limit,
        maximum_path_depth=path_depth_limit,
    )
    if checker.total_payload_bytes + runtime.total_payload_bytes > retained_bundle_byte_limit:
        raise ValueError("continuation retained bundles exceed the aggregate payload byte limit")
    _require_native_executable_snapshot(
        runtime,
        runtime_python,
    )
    _require_valid_zipapp(
        checker,
        checker_zipapp,
        member_limit=_require_positive_int(document["zip_member_limit"], "zip_member_limit"),
        uncompressed_byte_limit=_require_positive_int(
            document["zip_uncompressed_byte_limit"],
            "zip_uncompressed_byte_limit",
        ),
        path_depth_limit=path_depth_limit,
    )
    watchdog = document["safety_watchdog_seconds"]
    _require_registered_value(
        watchdog,
        CONTINUATION_WATCHDOG_SECONDS,
        "continuation safety watchdog",
    )
    return document, digest, checker, runtime


def verify_continuation_protocol(
    protocol_bytes: bytes,
    *,
    expected_protocol_sha256: str,
    expected_release_id: str,
    checker_source_bundle: VerifiedSourceBundle,
    runtime_bundle: VerifiedSourceBundle,
) -> VerifiedContinuationProtocol:
    document, digest, checker, runtime = _validate_continuation_payload(
        protocol_bytes,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_release_id=expected_release_id,
        checker_source_bundle=checker_source_bundle,
        runtime_bundle=runtime_bundle,
    )
    runtime_python = cast(str, document["runtime_python"])
    checker_zipapp = cast(str, document["checker_zipapp"])
    watchdog = cast(int, document["safety_watchdog_seconds"])

    verified = object.__new__(VerifiedContinuationProtocol)
    object.__setattr__(verified, "release_id", checker.release_id)
    object.__setattr__(verified, "protocol_sha256", digest)
    object.__setattr__(verified, "checker_source_bundle_sha256", checker.manifest_sha256)
    object.__setattr__(verified, "runtime_bundle_sha256", runtime.manifest_sha256)
    object.__setattr__(verified, "runtime_python", runtime_python)
    object.__setattr__(verified, "checker_zipapp", checker_zipapp)
    object.__setattr__(verified, "node_budget", CONTINUATION_NODE_BUDGET)
    object.__setattr__(verified, "safety_watchdog_seconds", watchdog)
    for field_name in (
        "input_byte_limit",
        "output_byte_limit",
        "stdout_byte_limit",
        "stderr_byte_limit",
        "retained_bundle_byte_limit",
        "zip_member_limit",
        "zip_uncompressed_byte_limit",
        "path_depth_limit",
    ):
        object.__setattr__(verified, field_name, cast(int, document[field_name]))
    object.__setattr__(verified, "_payload", bytes(protocol_bytes))
    object.__setattr__(verified, "_checker_source_bundle", checker_source_bundle)
    object.__setattr__(verified, "_runtime_bundle", runtime_bundle)
    object.__setattr__(verified, "_seal", _PROTOCOL_SEAL)
    return verified


def _retained_bundle_identity(bundle: VerifiedSourceBundle) -> tuple[object, ...]:
    files = bundle.files
    identity: list[object] = [
        bundle._seal,
        bundle.release_id,
        bundle.role,
        bundle.manifest_sha256,
        id(bundle._manifest_payload),
        bundle._manifest_payload,
        id(files),
        len(files) if type(files) is tuple else -1,
    ]
    if type(files) is tuple:
        for entry in files:
            if type(entry) is not VerifiedSourceFile:
                identity.extend((id(entry), None))
                continue
            identity.extend(
                (
                    id(entry),
                    entry.relative_path,
                    entry.sha256,
                    id(entry.payload),
                    entry.payload,
                    entry.executable,
                )
            )
    return tuple(identity)


def _continuation_capsule_identity(value: VerifiedContinuationProtocol) -> tuple[object, ...]:
    return (
        value._seal,
        value.release_id,
        value.protocol_sha256,
        value.checker_source_bundle_sha256,
        value.runtime_bundle_sha256,
        value.runtime_python,
        value.checker_zipapp,
        value.node_budget,
        value.safety_watchdog_seconds,
        value.input_byte_limit,
        value.output_byte_limit,
        value.stdout_byte_limit,
        value.stderr_byte_limit,
        value.retained_bundle_byte_limit,
        value.zip_member_limit,
        value.zip_uncompressed_byte_limit,
        value.path_depth_limit,
        id(value._payload),
        value._payload,
        id(value._checker_source_bundle),
        _retained_bundle_identity(value._checker_source_bundle),
        id(value._runtime_bundle),
        _retained_bundle_identity(value._runtime_bundle),
    )


def snapshot_continuation_protocol(
    value: object,
    *,
    expected_protocol_sha256: str,
    expected_release_id: str,
) -> ContinuationProtocolSnapshot:
    """Atomically detach a continuation protocol and both closed retained bundles."""

    if type(value) is not VerifiedContinuationProtocol or value._seal is not _PROTOCOL_SEAL:
        raise TypeError("protocol must be produced by the continuation verifier")
    before_identity = _continuation_capsule_identity(value)
    protocol_bytes = value._payload
    checker_source_bundle = value._checker_source_bundle
    runtime_bundle = value._runtime_bundle
    document, digest, checker, runtime = _validate_continuation_payload(
        protocol_bytes,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_release_id=expected_release_id,
        checker_source_bundle=checker_source_bundle,
        runtime_bundle=runtime_bundle,
    )
    if _continuation_capsule_identity(value) != before_identity:
        raise ValueError("verified continuation protocol changed while it was being snapshotted")

    expected_fields: dict[str, object] = {
        "release_id": document["release_id"],
        "protocol_sha256": digest,
        "checker_source_bundle_sha256": checker.manifest_sha256,
        "runtime_bundle_sha256": runtime.manifest_sha256,
        "runtime_python": document["runtime_python"],
        "checker_zipapp": document["checker_zipapp"],
        "node_budget": document["node_budget"],
        "safety_watchdog_seconds": document["safety_watchdog_seconds"],
        "input_byte_limit": document["input_byte_limit"],
        "output_byte_limit": document["output_byte_limit"],
        "stdout_byte_limit": document["stdout_byte_limit"],
        "stderr_byte_limit": document["stderr_byte_limit"],
        "retained_bundle_byte_limit": document["retained_bundle_byte_limit"],
        "zip_member_limit": document["zip_member_limit"],
        "zip_uncompressed_byte_limit": document["zip_uncompressed_byte_limit"],
        "path_depth_limit": document["path_depth_limit"],
    }
    if any(getattr(value, name) != expected for name, expected in expected_fields.items()):
        raise ValueError("verified protocol fields disagree with retained protocol bytes")

    runtime_python = cast(str, document["runtime_python"])
    checker_zipapp = cast(str, document["checker_zipapp"])
    snapshot = object.__new__(ContinuationProtocolSnapshot)
    object.__setattr__(snapshot, "release_id", cast(str, document["release_id"]))
    object.__setattr__(snapshot, "protocol_sha256", digest)
    object.__setattr__(snapshot, "protocol_bytes", bytes(protocol_bytes))
    object.__setattr__(snapshot, "checker_source_bundle_sha256", checker.manifest_sha256)
    object.__setattr__(snapshot, "runtime_bundle_sha256", runtime.manifest_sha256)
    object.__setattr__(snapshot, "runtime_python", runtime_python)
    object.__setattr__(snapshot, "checker_zipapp", checker_zipapp)
    object.__setattr__(
        snapshot,
        "command_template",
        tuple(cast(list[str], document["command_template"])),
    )
    object.__setattr__(snapshot, "input_schema", cast(str, document["input_schema"]))
    object.__setattr__(snapshot, "output_schema", cast(str, document["output_schema"]))
    object.__setattr__(
        snapshot,
        "watchdog_seconds",
        cast(int, document["safety_watchdog_seconds"]),
    )
    for field_name in (
        "input_byte_limit",
        "output_byte_limit",
        "stdout_byte_limit",
        "stderr_byte_limit",
        "retained_bundle_byte_limit",
        "zip_member_limit",
        "zip_uncompressed_byte_limit",
        "path_depth_limit",
    ):
        object.__setattr__(snapshot, field_name, cast(int, document[field_name]))
    object.__setattr__(snapshot, "checker_files", checker.files)
    object.__setattr__(snapshot, "runtime_files", runtime.files)
    object.__setattr__(
        snapshot,
        "checker_zipapp_sha256",
        checker.file_entry(checker_zipapp).sha256,
    )
    object.__setattr__(
        snapshot,
        "runtime_python_sha256",
        runtime.file_entry(runtime_python).sha256,
    )
    object.__setattr__(snapshot, "_seal", _CONTINUATION_SNAPSHOT_SEAL)
    return snapshot


def validate_protocol(
    value: object,
    *,
    expected_protocol_sha256: str,
    expected_release_id: str,
) -> VerifiedGenerationProtocol | VerifiedSolverProtocol | VerifiedContinuationProtocol:
    """Replay a capsule against independently supplied roots before field consumption."""

    if type(value) not in {
        VerifiedGenerationProtocol,
        VerifiedSolverProtocol,
        VerifiedContinuationProtocol,
    }:
        raise TypeError("protocol must be produced by its verifier")
    checked = cast(
        VerifiedGenerationProtocol | VerifiedSolverProtocol | VerifiedContinuationProtocol,
        value,
    )
    if checked._seal is not _PROTOCOL_SEAL:
        raise TypeError("protocol must be produced by its verifier")
    if type(checked._payload) is not bytes:
        raise ValueError("verified protocol fields disagree with retained protocol bytes")
    actual_sha256 = hashlib.sha256(checked._payload).hexdigest()
    if actual_sha256 != checked.protocol_sha256:
        raise ValueError("verified protocol fields disagree with retained protocol bytes")
    external_sha256 = _require_sha256(expected_protocol_sha256, "expected_protocol_sha256")
    if actual_sha256 != external_sha256:
        raise ValueError("verified protocol disagrees with its external commitment")
    release_id = _require_text(expected_release_id, "expected_release_id")
    replayed: VerifiedGenerationProtocol | VerifiedSolverProtocol | VerifiedContinuationProtocol
    fields: tuple[str, ...]
    if isinstance(checked, VerifiedGenerationProtocol):
        generation = checked
        replayed = verify_generation_protocol(
            generation._payload,
            expected_protocol_sha256=external_sha256,
            expected_release_id=release_id,
            source_bundle=generation._source_bundle,
        )
        fields = ("release_id", "protocol_sha256", "source_bundle_sha256")
    elif isinstance(checked, VerifiedSolverProtocol):
        solver = checked
        replayed = verify_solver_protocol(
            solver._payload,
            expected_protocol_sha256=external_sha256,
            expected_release_id=release_id,
            source_bundle=solver._source_bundle,
        )
        fields = (
            "release_id",
            "solver_id",
            "protocol_sha256",
            "source_bundle_sha256",
            "entrypoint",
            "binary_sha256",
            "deterministic_work_limit",
            "work_limit_unit",
        )
    else:
        continuation = checked
        replayed = verify_continuation_protocol(
            continuation._payload,
            expected_protocol_sha256=external_sha256,
            expected_release_id=release_id,
            checker_source_bundle=continuation._checker_source_bundle,
            runtime_bundle=continuation._runtime_bundle,
        )
        fields = (
            "release_id",
            "protocol_sha256",
            "checker_source_bundle_sha256",
            "runtime_bundle_sha256",
            "runtime_python",
            "checker_zipapp",
            "node_budget",
            "safety_watchdog_seconds",
            "input_byte_limit",
            "output_byte_limit",
            "stdout_byte_limit",
            "stderr_byte_limit",
            "retained_bundle_byte_limit",
            "zip_member_limit",
            "zip_uncompressed_byte_limit",
            "path_depth_limit",
        )
    if any(getattr(checked, field) != getattr(replayed, field) for field in fields):
        raise ValueError("verified protocol fields disagree with retained protocol bytes")
    return checked


__all__ = [
    "ALLOWED_Q_CAP_SLACK",
    "CANDIDATE_CHAIN_ORDER_RULE",
    "CONTINUATION_COMMAND_TEMPLATE",
    "CONTINUATION_INPUT_BYTE_LIMIT",
    "CONTINUATION_INPUT_SCHEMA",
    "CONTINUATION_NODE_BUDGET",
    "CONTINUATION_NODE_COUNT_RULE",
    "CONTINUATION_OBJECTIVE_ORDER",
    "CONTINUATION_OUTPUT_BYTE_LIMIT",
    "CONTINUATION_OUTPUT_SCHEMA",
    "CONTINUATION_PROTOCOL_SCHEMA",
    "CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT",
    "CONTINUATION_STDERR_BYTE_LIMIT",
    "CONTINUATION_STDOUT_BYTE_LIMIT",
    "CONTINUATION_WATCHDOG_SECONDS",
    "CONTINUATION_PATH_DEPTH_LIMIT",
    "CONTINUATION_ZIP_MEMBER_LIMIT",
    "CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT",
    "FOCUS_RULE",
    "GENERATION_PROTOCOL_SCHEMA",
    "PROTOCOL_SCHEMA_VERSION",
    "REMOVAL_RULE",
    "RUNTIME_PYTHON_PATH",
    "CHECKER_ZIPAPP_PATH",
    "SOLVER_ENTRYPOINTS",
    "SOLVER_ATTEMPT_COUNT",
    "SOLVER_PROTOCOL_SCHEMA",
    "SOLVER_RESOLVED_PARAMETERS",
    "SOLVER_SEED_PROJECTION",
    "SOLVER_SEED_PANEL",
    "SOLVER_SEED_CELL",
    "SOLVER_SEED_SCHEDULE_RULE",
    "SOLVER_WORK_LIMITS",
    "STARTING_SOURCE_ROTATION",
    "VARIABLE_ORDER_RULE",
    "ContinuationProtocolSnapshot",
    "VerifiedContinuationProtocol",
    "VerifiedGenerationProtocol",
    "VerifiedSolverProtocol",
    "verify_continuation_protocol",
    "verify_generation_protocol",
    "verify_solver_protocol",
    "solver_attempt_seed_request",
    "solver_attempt_seed_schedule",
    "solver_attempt_seed_schedule_sha256",
    "snapshot_continuation_protocol",
]

"""Authenticated adapter from reference-export bundles to IsingFold source rows.

The adapter changes serialization only.  It does not recompute a ground state or
promote a best-known scalar to certified authority.  Authority comes from the four
out-of-band SHA-256 pins supplied by the caller and from the exact-reference exporter
that produced the pinned inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedbench.candidate_bank import canonical_json_bytes, content_digest

EXPORT_MANIFEST_NAME = "problem_reference_export_manifest.json"
PROTOCOL_NAME = "problem_reference_protocol.json"
REFERENCES_NAME = "problem_references.jsonl"
CERTIFICATES_NAME = "problem_reference_certificates.jsonl"
BEST_KNOWN_NAME = "problem_references.best_known.jsonl"
OUTPUT_REFERENCES_NAME = "isingfold_references.jsonl"

EXPORT_SCHEMA = "embedbench.problem-reference-export"
PROTOCOL_SCHEMA = "embedbench.problem-reference-evaluator-protocol"
SOURCE_REFERENCE_SCHEMA = "embedbench.problem-reference"
SOURCE_CERTIFICATE_SCHEMA = "embedbench.problem-reference-certificate"
OUTPUT_REFERENCE_SCHEMA = "embedbench.isingfold-reference"
SCHEMA_VERSION = 1

_HEX = frozenset("0123456789abcdef")
_MAX_CONTROL_BYTES = 256 * 1024 * 1024
_OUTPUT_NAMES = {
    BEST_KNOWN_NAME,
    CERTIFICATES_NAME,
    PROTOCOL_NAME,
    REFERENCES_NAME,
}
_MANIFEST_FIELDS = {
    "certified_problems",
    "excluded_problems",
    "importer_complete",
    "max_exact_variables",
    "outputs",
    "protocol_sha256",
    "schema",
    "schema_version",
    "source_checksums",
    "source_checksums_sha256",
    "source_corpora",
    "source_rows",
    "unique_problems",
}
_PROTOCOL_FIELDS = {
    "coefficient_semantics",
    "enumeration_order",
    "ground_certificate_source_sha256",
    "maximum_variables",
    "problem_identity",
    "purpose",
    "schema",
    "schema_version",
}
_SOURCE_REFERENCE_FIELDS = {
    "certificate_digest",
    "evaluator_protocol_digest",
    "problem_digest",
    "reference_energy",
    "reference_status",
    "schema",
    "schema_version",
}
_CERTIFICATE_FIELDS = {
    "assignment",
    "checked_state_count",
    "declared_release_energy",
    "enumeration_order",
    "evaluator_protocol_digest",
    "exact_energy",
    "final_gray_word",
    "minimum_gray_step",
    "problem_digest",
    "problem_sha256",
    "reference_energy",
    "reference_status",
    "schema",
    "schema_version",
    "spin_assignment_sha256",
    "variable_order",
}
_CERTIFIED_STATUSES = frozenset({"planted_proof", "exact_enumeration", "certified_optimal"})


@dataclass(frozen=True, slots=True)
class ReferenceAdapterReceipt:
    """Stable identities of one immutable adapter publication."""

    output_directory: Path
    reference_index_path: Path
    reference_index_sha256: str
    reference_count: int
    certificate_directory: Path
    certificate_count: int
    source_manifest_sha256: str
    source_protocol_sha256: str
    source_references_sha256: str
    source_certificates_sha256: str
    evaluator_protocol_digest: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate")
    return value


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _version(value: object, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} requires schema_version {expected}")
    return value


def _filename(value: object, name: str) -> str:
    text = _text(value, name)
    if text in {".", ".."} or "/" in text or "\\" in text or "\x00" in text:
        raise ValueError(f"{name} must be a safe single-component filename")
    return text


def _exact(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an exact JSON object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} schema fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return value


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        value[key] = item
    return value


def _constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _object(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
        canonical_json_bytes(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{name} is invalid UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _read_regular(path: Path, name: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing or unreadable") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a regular file, not a symlink")
    if before.st_size > _MAX_CONTROL_BYTES:
        raise ValueError(f"{name} exceeds its byte limit")
    payload = path.read_bytes()
    after = path.lstat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise ValueError(f"{name} changed while it was being read")
    return payload


def _pinned(path: Path, expected: object, name: str) -> bytes:
    payload = _read_regular(path, name)
    if _sha256_bytes(payload) != _sha256(expected, f"expected {name} SHA-256"):
        raise ValueError(f"{name} differs from its external SHA-256 commitment")
    return payload


def _canonical_object(payload: bytes, name: str) -> dict[str, Any]:
    value = _object(payload, name)
    if payload != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} must be canonical JSON followed by one line feed")
    return value


def _canonical_jsonl(
    payload: bytes,
    name: str,
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    if not payload:
        if allow_empty:
            return []
        raise ValueError(f"{name} must be nonempty")
    if not payload.endswith(b"\n"):
        raise ValueError(f"{name} must end with one line feed")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ValueError(f"{name} line {line_number} is blank")
        row = _object(line, f"{name} line {line_number}")
        if line != canonical_json_bytes(row):
            raise ValueError(f"{name} line {line_number} is not canonical JSON")
        rows.append(row)
    return rows


def _bundle_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path)
    try:
        metadata = root.lstat()
    except OSError as error:
        raise ValueError("reference-export root is missing or unreadable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("reference-export root must be a directory, not a symlink")
    return root


def _output_descriptor(value: object, name: str) -> tuple[int, str]:
    descriptor = _exact(value, {"bytes", "sha256"}, f"{name} output descriptor")
    return (
        _integer(descriptor["bytes"], f"{name} output byte count"),
        _sha256(descriptor["sha256"], f"{name} output SHA-256"),
    )


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    payloads: Mapping[str, bytes],
) -> tuple[int, list[dict[str, Any]]]:
    _exact(manifest, _MANIFEST_FIELDS, "reference-export manifest")
    if manifest["schema"] != EXPORT_SCHEMA:
        raise ValueError("unsupported reference-export manifest schema")
    _version(manifest["schema_version"], SCHEMA_VERSION, "reference-export manifest")
    certified = _integer(manifest["certified_problems"], "certified problem count", positive=True)
    excluded = _integer(manifest["excluded_problems"], "excluded problem count")
    unique = _integer(manifest["unique_problems"], "unique problem count", positive=True)
    if manifest["importer_complete"] is not True or excluded != 0:
        raise ValueError("reference export contains best-known or uncertified problems")
    if unique != certified + excluded:
        raise ValueError("reference-export problem counts are inconsistent")
    _integer(manifest["max_exact_variables"], "maximum exact variables", positive=True)
    _sha256(manifest["protocol_sha256"], "reference evaluator protocol digest")
    _filename(manifest["source_checksums"], "source checksum filename")
    _sha256(manifest["source_checksums_sha256"], "source checksum SHA-256")

    outputs = _exact(manifest["outputs"], _OUTPUT_NAMES, "reference-export outputs")
    for filename in sorted(_OUTPUT_NAMES):
        byte_count, digest = _output_descriptor(outputs[filename], filename)
        payload = payloads[filename]
        if byte_count != len(payload) or digest != _sha256_bytes(payload):
            raise ValueError(f"{filename} differs from its manifest descriptor")

    raw_corpora = manifest["source_corpora"]
    if type(raw_corpora) is not list or not raw_corpora:
        raise ValueError("reference-export source_corpora must be a nonempty list")
    corpora: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_corpora, start=1):
        row = _exact(
            raw,
            {"file", "manifest", "manifest_sha256", "rows", "sha256"},
            f"source corpus {index}",
        )
        corpora.append(
            {
                "file": _filename(row["file"], f"source corpus {index} filename"),
                "manifest": _filename(row["manifest"], f"source corpus {index} manifest filename"),
                "manifest_sha256": _sha256(
                    row["manifest_sha256"], f"source corpus {index} manifest SHA-256"
                ),
                "rows": _integer(row["rows"], f"source corpus {index} rows", positive=True),
                "sha256": _sha256(row["sha256"], f"source corpus {index} SHA-256"),
            }
        )
    if corpora != sorted(corpora, key=lambda row: row["file"]):
        raise ValueError("reference-export source corpora are not canonically sorted")
    if len({row["file"] for row in corpora}) != len(corpora):
        raise ValueError("reference-export source corpora repeat a filename")
    source_rows = _integer(manifest["source_rows"], "source row count", positive=True)
    if sum(row["rows"] for row in corpora) != source_rows:
        raise ValueError("reference-export source row count is inconsistent")
    return certified, corpora


def _validate_protocol(protocol: dict[str, Any], manifest: Mapping[str, Any]) -> str:
    _exact(protocol, _PROTOCOL_FIELDS, "reference evaluator protocol")
    if protocol["schema"] != PROTOCOL_SCHEMA:
        raise ValueError("unsupported reference evaluator protocol schema")
    _version(protocol["schema_version"], SCHEMA_VERSION, "reference evaluator protocol")
    registered = {
        "coefficient_semantics": "exact_ieee754_binary64_dyadic",
        "enumeration_order": "binary_reflected_gray_code_lsb_first_v1",
        "problem_identity": "sha256_canonical_h_J_e0_v1",
        "purpose": "independent_ground_reference_validation_before_downstream_evaluation",
    }
    if any(protocol[field] != value for field, value in registered.items()):
        raise ValueError("reference evaluator protocol differs from the registered protocol")
    _sha256(
        protocol["ground_certificate_source_sha256"],
        "ground-certificate source SHA-256",
    )
    maximum = _integer(protocol["maximum_variables"], "protocol maximum variables", positive=True)
    if maximum != manifest["max_exact_variables"]:
        raise ValueError("protocol variable limit differs from the export manifest")
    digest = content_digest(protocol)
    if digest != manifest["protocol_sha256"]:
        raise ValueError("protocol content digest differs from the export manifest")
    return digest


def _validate_source_references(
    rows: Sequence[dict[str, Any]],
    *,
    protocol_digest: str,
    expected_count: int,
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    certificate_digests: set[str] = set()
    order: list[str] = []
    for line_number, raw in enumerate(rows, start=1):
        row = _exact(
            raw,
            _SOURCE_REFERENCE_FIELDS,
            f"problem-reference line {line_number}",
        )
        if row["schema"] != SOURCE_REFERENCE_SCHEMA:
            raise ValueError("unsupported problem-reference schema")
        _version(row["schema_version"], SCHEMA_VERSION, f"problem-reference line {line_number}")
        problem_digest = _sha256(
            row["problem_digest"], f"problem-reference line {line_number} problem digest"
        )
        certificate_digest = _sha256(
            row["certificate_digest"],
            f"problem-reference line {line_number} certificate digest",
        )
        if problem_digest in references:
            raise ValueError("problem-reference sidecar repeats a problem")
        if certificate_digest in certificate_digests:
            raise ValueError("problem-reference sidecar repeats a certificate")
        if row["evaluator_protocol_digest"] != protocol_digest:
            raise ValueError("problem-reference row uses another evaluator protocol")
        status = row["reference_status"]
        if status not in _CERTIFIED_STATUSES:
            raise ValueError("problem-reference row is best-known or uncertified")
        references[problem_digest] = {
            "certificate_digest": certificate_digest,
            "evaluator_protocol_digest": protocol_digest,
            "problem_digest": problem_digest,
            "reference_energy": _finite(
                row["reference_energy"],
                f"problem-reference line {line_number} reference energy",
            ),
            "reference_status": status,
        }
        certificate_digests.add(certificate_digest)
        order.append(problem_digest)
    if len(references) != expected_count:
        raise ValueError("problem-reference count differs from the export manifest")
    if order != sorted(order):
        raise ValueError("problem-reference rows must be sorted by problem_digest")
    return references


def _validate_assignment(certificate: Mapping[str, Any], name: str) -> None:
    variables = certificate["variable_order"]
    if (
        type(variables) is not list
        or not variables
        or any(type(variable) is not int for variable in variables)
        or variables != sorted(set(variables))
    ):
        raise ValueError(f"{name} variable_order must be sorted and duplicate-free")
    assignment = certificate["assignment"]
    if type(assignment) is not list or len(assignment) != len(variables):
        raise ValueError(f"{name} assignment must cover every variable")
    checked: list[int] = []
    for pair in assignment:
        if (
            type(pair) is not list
            or len(pair) != 2
            or type(pair[0]) is not int
            or type(pair[1]) is not int
            or pair[1] not in {-1, 1}
        ):
            raise ValueError(f"{name} assignment rows must be [integer, -1|1]")
        checked.append(pair[0])
    if checked != variables:
        raise ValueError(f"{name} assignment order differs from variable_order")


def _validate_certificate(
    raw: object,
    *,
    reference: Mapping[str, Any],
    protocol_digest: str,
    name: str,
) -> tuple[str, bytes]:
    certificate = _exact(raw, _CERTIFICATE_FIELDS, name)
    if certificate["schema"] != SOURCE_CERTIFICATE_SCHEMA:
        raise ValueError("unsupported problem-reference certificate schema")
    _version(certificate["schema_version"], SCHEMA_VERSION, name)
    problem_digest = _sha256(certificate["problem_digest"], f"{name} problem digest")
    problem_sha256 = _sha256(certificate["problem_sha256"], f"{name} problem SHA-256")
    if problem_digest != reference["problem_digest"]:
        raise ValueError("certificate references the wrong source problem")
    if certificate["evaluator_protocol_digest"] != protocol_digest:
        raise ValueError("certificate uses another evaluator protocol")
    status = certificate["reference_status"]
    if status != reference["reference_status"] or status not in _CERTIFIED_STATUSES:
        raise ValueError("certificate reference status differs from its reference row")
    energy = _finite(certificate["reference_energy"], f"{name} reference energy")
    if energy != reference["reference_energy"]:
        raise ValueError("certificate energy differs from its reference row")
    _finite(certificate["declared_release_energy"], f"{name} declared release energy")
    _integer(certificate["checked_state_count"], f"{name} checked state count", positive=True)
    _integer(certificate["final_gray_word"], f"{name} final Gray word")
    _integer(certificate["minimum_gray_step"], f"{name} minimum Gray step")
    if certificate["enumeration_order"] != "binary_reflected_gray_code_lsb_first_v1":
        raise ValueError("certificate uses another enumeration order")
    exact_energy = _exact(
        certificate["exact_energy"], {"integer", "power_of_two"}, f"{name} exact energy"
    )
    if type(exact_energy["integer"]) is not int or type(exact_energy["power_of_two"]) is not int:
        raise ValueError(f"{name} exact energy must use integer dyadic fields")
    _sha256(certificate["spin_assignment_sha256"], f"{name} assignment SHA-256")
    _validate_assignment(certificate, name)
    payload = canonical_json_bytes(certificate)
    if _sha256_bytes(payload) != reference["certificate_digest"]:
        raise ValueError("certificate bytes differ from the reference certificate digest")
    return problem_sha256, payload


def _validate_source_record(
    raw: object,
    *,
    corpora: Mapping[str, Mapping[str, Any]],
    name: str,
) -> tuple[str, int, str]:
    row = _exact(raw, {"corpus", "instance_id", "line", "record_sha256"}, name)
    corpus = _filename(row["corpus"], f"{name} corpus")
    if corpus not in corpora:
        raise ValueError("certificate evidence references an unknown source corpus")
    line = _integer(row["line"], f"{name} line", positive=True)
    if line > corpora[corpus]["rows"]:
        raise ValueError("certificate evidence references a source line outside its corpus")
    _text(row["instance_id"], f"{name} instance ID")
    digest = _sha256(row["record_sha256"], f"{name} record SHA-256")
    return corpus, line, digest


def _validate_evidence(
    rows: Sequence[dict[str, Any]],
    *,
    references: Mapping[str, Mapping[str, Any]],
    corpora: Sequence[Mapping[str, Any]],
    protocol_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    corpora_by_name = {str(row["file"]): row for row in corpora}
    evidence_problems: set[str] = set()
    problem_sha256s: set[str] = set()
    source_coverage: set[tuple[str, int]] = set()
    source_order: list[str] = []
    output_rows: list[dict[str, Any]] = []
    certificates: dict[str, bytes] = {}
    for line_number, raw in enumerate(rows, start=1):
        evidence = _exact(
            raw,
            {"certificate", "certificate_digest", "source_records"},
            f"certificate evidence line {line_number}",
        )
        certificate_digest = _sha256(
            evidence["certificate_digest"],
            f"certificate evidence line {line_number} digest",
        )
        certificate_raw = evidence["certificate"]
        if type(certificate_raw) is not dict:
            raise ValueError("certificate evidence must contain one certificate object")
        problem_digest = _sha256(
            certificate_raw.get("problem_digest"),
            f"certificate evidence line {line_number} problem digest",
        )
        reference = references.get(problem_digest)
        if reference is None:
            raise ValueError("certificate evidence references an unknown source problem")
        if problem_digest in evidence_problems:
            raise ValueError("certificate evidence repeats a source problem")
        if certificate_digest != reference["certificate_digest"]:
            raise ValueError("certificate evidence references the wrong certificate digest")
        problem_sha256, certificate_bytes = _validate_certificate(
            certificate_raw,
            reference=reference,
            protocol_digest=protocol_digest,
            name=f"certificate evidence line {line_number} certificate",
        )
        if problem_sha256 in problem_sha256s:
            raise ValueError("certificate evidence repeats an Ising problem identity")

        source_records = evidence["source_records"]
        if type(source_records) is not list or not source_records:
            raise ValueError("certificate evidence source_records must be nonempty")
        record_order: list[tuple[str, int, str]] = []
        for position, source in enumerate(source_records, start=1):
            identity = _validate_source_record(
                source,
                corpora=corpora_by_name,
                name=f"certificate evidence line {line_number} source record {position}",
            )
            if identity[:2] in source_coverage:
                raise ValueError("certificate evidence repeats one source corpus row")
            source_coverage.add(identity[:2])
            record_order.append(identity)
        if record_order != sorted(record_order):
            raise ValueError("certificate source records must be canonically sorted")

        payload = {
            "certificate": {
                "path": f"certificates/{certificate_digest}",
                "sha256": certificate_digest,
            },
            "evaluator_protocol_digest": protocol_digest,
            "problem_sha256": problem_sha256,
            "reference_energy": reference["reference_energy"],
            "reference_status": reference["reference_status"],
            "schema": OUTPUT_REFERENCE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
        }
        output_rows.append({**payload, "record_digest": content_digest(payload)})
        certificates[certificate_digest] = certificate_bytes
        evidence_problems.add(problem_digest)
        problem_sha256s.add(problem_sha256)
        source_order.append(problem_digest)

    if evidence_problems != set(references):
        raise ValueError("certificate evidence coverage differs from problem references")
    covered_by_corpus = {str(corpus["file"]): 0 for corpus in corpora}
    for corpus, _ in source_coverage:
        covered_by_corpus[corpus] += 1
    if any(covered_by_corpus[str(corpus["file"])] != corpus["rows"] for corpus in corpora):
        raise ValueError("certificate evidence does not cover every source corpus row exactly once")
    if source_order != sorted(source_order):
        raise ValueError("certificate evidence rows must be sorted by problem_digest")
    output_rows.sort(key=lambda row: row["problem_sha256"])
    return output_rows, certificates


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _publish_stage(stage: Path, destination: Path) -> None:
    if _path_exists(destination):
        raise FileExistsError(f"reference-adapter output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError:
        raise FileExistsError(f"reference-adapter output already exists: {destination}") from None
    created = destination.lstat()
    try:
        target_certificates = destination / "certificates"
        target_certificates.mkdir()
        for source in sorted((stage / "certificates").iterdir(), key=lambda path: path.name):
            os.link(source, target_certificates / source.name)
        _fsync_directory(target_certificates)
        _fsync_directory(destination)
        # This index is the protocol commit marker and becomes visible only after
        # every certificate directory entry is durable.
        os.link(
            stage / OUTPUT_REFERENCES_NAME,
            destination / OUTPUT_REFERENCES_NAME,
        )
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            current = destination.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (
            created.st_dev,
            created.st_ino,
        ):
            shutil.rmtree(destination)
        raise


def adapt_reference_export_bundle(
    reference_export_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str,
    expected_protocol_sha256: str,
    expected_references_sha256: str,
    expected_certificates_sha256: str,
) -> ReferenceAdapterReceipt:
    """Publish the exact-reference export in the source publisher's wire format.

    All four authority-bearing inputs require independent SHA-256 commitments.  The
    best-known sidecar is transitively authenticated by the pinned export manifest and
    must be empty.  The adapter preserves source status, energy, and protocol identity;
    it never turns an uncertified result into a certified one.
    """

    root = _bundle_root(reference_export_dir)
    destination = Path(output_dir)
    if _path_exists(destination):
        raise FileExistsError(f"reference-adapter output already exists: {destination}")
    manifest_raw = _pinned(
        root / EXPORT_MANIFEST_NAME,
        expected_manifest_sha256,
        "reference-export manifest",
    )
    protocol_raw = _pinned(
        root / PROTOCOL_NAME,
        expected_protocol_sha256,
        "reference evaluator protocol",
    )
    references_raw = _pinned(
        root / REFERENCES_NAME,
        expected_references_sha256,
        "problem-reference sidecar",
    )
    certificates_raw = _pinned(
        root / CERTIFICATES_NAME,
        expected_certificates_sha256,
        "problem-reference certificate evidence",
    )
    best_known_raw = _read_regular(root / BEST_KNOWN_NAME, "best-known sidecar")

    manifest = _canonical_object(manifest_raw, "reference-export manifest")
    certified_count, corpora = _validate_manifest(
        manifest,
        payloads={
            BEST_KNOWN_NAME: best_known_raw,
            CERTIFICATES_NAME: certificates_raw,
            PROTOCOL_NAME: protocol_raw,
            REFERENCES_NAME: references_raw,
        },
    )
    if _canonical_jsonl(best_known_raw, "best-known sidecar", allow_empty=True):
        raise ValueError("reference export contains best-known or uncertified entries")
    protocol = _canonical_object(protocol_raw, "reference evaluator protocol")
    protocol_digest = _validate_protocol(protocol, manifest)
    source_references = _validate_source_references(
        _canonical_jsonl(references_raw, "problem-reference sidecar"),
        protocol_digest=protocol_digest,
        expected_count=certified_count,
    )
    output_rows, certificates = _validate_evidence(
        _canonical_jsonl(
            certificates_raw,
            "problem-reference certificate evidence",
        ),
        references=source_references,
        corpora=corpora,
        protocol_digest=protocol_digest,
    )
    if len(certificates) != certified_count:
        raise ValueError("certificate count differs from the export manifest")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.adapt-", dir=destination.parent))
    output_raw = _jsonl_bytes(output_rows)
    try:
        for digest, payload in sorted(certificates.items()):
            _write_exclusive(stage / "certificates" / digest, payload)
        _write_exclusive(stage / OUTPUT_REFERENCES_NAME, output_raw)
        _publish_stage(stage, destination)
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    return ReferenceAdapterReceipt(
        output_directory=destination,
        reference_index_path=destination / OUTPUT_REFERENCES_NAME,
        reference_index_sha256=_sha256_bytes(output_raw),
        reference_count=len(output_rows),
        certificate_directory=destination / "certificates",
        certificate_count=len(certificates),
        source_manifest_sha256=_sha256_bytes(manifest_raw),
        source_protocol_sha256=_sha256_bytes(protocol_raw),
        source_references_sha256=_sha256_bytes(references_raw),
        source_certificates_sha256=_sha256_bytes(certificates_raw),
        evaluator_protocol_digest=protocol_digest,
    )


__all__ = ["ReferenceAdapterReceipt", "adapt_reference_export_bundle"]

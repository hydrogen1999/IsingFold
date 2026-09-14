"""Authenticated exact-reference export for legacy quality release v1.1.

The v1.1 JSONL rows contain a complete Ising problem and a scalar ``e0``, but the
release did not type that scalar as exact, planted, or best-known.  This module never
upgrades the scalar by assertion.  It authenticates the corpus bytes, independently
enumerates every eligible problem with exact dyadic arithmetic, and emits the closed
``embedbench.problem-reference`` sidecar consumed by IsingFold.

The legacy graph-cut rows are deliberately quarantined.  Their stored scalar came from
tabu search, and the release carries no registered proof for it.  In incomplete audit
mode they are written to a separate best-known file and never enter the certified
sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedbench import ground_certificate
from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.ground_certificate import (
    GRAY_CODE_ENUMERATION,
    MAX_EXHAUSTIVE_VARIABLES,
    IsingProblem,
    spin_assignment_sha256,
)

REFERENCE_SCHEMA = "embedbench.problem-reference"
REFERENCE_SCHEMA_VERSION = 1
CERTIFICATE_SCHEMA = "embedbench.problem-reference-certificate"
CERTIFICATE_SCHEMA_VERSION = 1
PROTOCOL_SCHEMA = "embedbench.problem-reference-evaluator-protocol"
PROTOCOL_SCHEMA_VERSION = 1
EXPORT_SCHEMA = "embedbench.problem-reference-export"
EXPORT_SCHEMA_VERSION = 1
DEFAULT_MAX_EXACT_VARIABLES = 22

_REFERENCE_FIELDS = {
    "certificate_digest",
    "evaluator_protocol_digest",
    "problem_digest",
    "reference_energy",
    "reference_status",
    "schema",
    "schema_version",
}
_HEX = frozenset("0123456789abcdef")


class IncompleteReferenceCoverageError(ValueError):
    """At least one selected problem cannot enter the certified sidecar."""


@dataclass(slots=True)
class _ProblemSources:
    problem: IsingProblem
    declared_energy: float
    modes: set[str]
    source_records: list[dict[str, object]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_json(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    # Canonicalisation recursively rejects infinities created by overflowing exponents.
    canonical_json_bytes(value)
    return value


def _jsonl(path: Path, name: str, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{name} line {line_number} is blank")
        rows.append(_strict_json(line, f"{name} line {line_number}"))
    if not rows and not allow_empty:
        raise ValueError(f"{name} is empty")
    return rows


def _checksum_table(path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None:
            raise ValueError(f"SHA256SUMS line {line_number} is not canonical")
        digest, filename = match.groups()
        if filename in table:
            raise ValueError(f"SHA256SUMS repeats {filename!r}")
        table[filename] = digest
    if not table:
        raise ValueError("SHA256SUMS is empty")
    return table


def _verify_checksum(path: Path, table: Mapping[str, str]) -> str:
    expected = table.get(path.name)
    if expected is None:
        raise ValueError(f"{path.name} is not authenticated by SHA256SUMS")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")
    return actual


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _problem(raw: object) -> tuple[IsingProblem, float, str]:
    if not isinstance(raw, Mapping) or set(raw) != {"J", "e0", "h"}:
        raise ValueError("problem must contain exactly h, J, and e0")
    raw_h = raw["h"]
    if not isinstance(raw_h, Mapping) or not raw_h:
        raise ValueError("problem.h must be a non-empty object")
    linear: list[tuple[int, float]] = []
    for key, value in raw_h.items():
        if not isinstance(key, str) or re.fullmatch(r"0|[1-9][0-9]*", key) is None:
            raise ValueError("problem.h keys must be canonical non-negative integers")
        linear.append((int(key), _finite(value, f"problem.h[{key}]")))
    linear.sort()
    variables = tuple(node for node, _ in linear)
    variable_set = set(variables)
    raw_j = raw["J"]
    if not isinstance(raw_j, list):
        raise ValueError("problem.J must be a list")
    quadratic: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for index, edge in enumerate(raw_j):
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError(f"problem.J[{index}] must be [u,v,bias]")
        if any(
            isinstance(endpoint, bool) or not isinstance(endpoint, int)
            for endpoint in edge[:2]
        ):
            raise ValueError(f"problem.J[{index}] endpoints must be integers")
        left, right = sorted((edge[0], edge[1]))
        if left == right or left not in variable_set or right not in variable_set:
            raise ValueError(f"problem.J[{index}] has invalid endpoints")
        pair = (left, right)
        if pair in seen:
            raise ValueError("problem.J contains a duplicate edge")
        seen.add(pair)
        coefficient = _finite(edge[2], f"problem.J[{index}] coefficient")
        if coefficient == 0.0:
            raise ValueError("problem.J must omit zero coefficients")
        quadratic.append((left, right, coefficient))
    quadratic.sort()
    reference = _finite(raw["e0"], "problem.e0")
    canonical = {
        "J": quadratic,
        "e0": reference,
        "h": linear,
    }
    digest = hashlib.sha256(
        json.dumps(
            canonical,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return (
        IsingProblem(
            variables=variables,
            linear=tuple(linear),
            quadratic=tuple(quadratic),
        ),
        reference,
        digest,
    )


def _protocol(max_exact_variables: int) -> dict[str, object]:
    source = Path(inspect.getsourcefile(ground_certificate) or ground_certificate.__file__)
    return {
        "coefficient_semantics": "exact_ieee754_binary64_dyadic",
        "enumeration_order": GRAY_CODE_ENUMERATION,
        "ground_certificate_source_sha256": _sha256(source),
        "maximum_variables": max_exact_variables,
        "problem_identity": "sha256_canonical_h_J_e0_v1",
        "purpose": "independent_ground_reference_validation_before_downstream_evaluation",
        "schema": PROTOCOL_SCHEMA,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
    }


def _certificate(
    problem: IsingProblem,
    problem_digest: str,
    declared_energy: float,
    protocol_sha256: str,
) -> dict[str, object]:
    # This is package-internal reuse of the exact implementation also exercised by
    # verify_ground_state_certificate.  It lifts binary64 coefficients to dyadics and
    # evaluates every state using unbounded integer arithmetic.
    result = ground_certificate._enumerate_problem(problem)
    exact_energy = result.energy.to_float()
    if not math.isclose(exact_energy, declared_energy, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"problem {problem_digest} release e0={declared_energy!r} disagrees with exact "
            f"enumeration energy={exact_energy!r}"
        )
    return {
        "assignment": [list(item) for item in result.assignment],
        "checked_state_count": result.checked_state_count,
        "declared_release_energy": declared_energy,
        "enumeration_order": GRAY_CODE_ENUMERATION,
        "evaluator_protocol_digest": protocol_sha256,
        "exact_energy": result.energy.to_dict(),
        "final_gray_word": result.final_gray_word,
        "minimum_gray_step": result.minimum_gray_step,
        "problem_digest": problem_digest,
        "problem_sha256": problem.problem_sha256,
        "reference_energy": exact_energy,
        "reference_status": "exact_enumeration",
        "schema": CERTIFICATE_SCHEMA,
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "spin_assignment_sha256": spin_assignment_sha256(problem, result.assignment),
        "variable_order": list(problem.variables),
    }


def _reference(certificate: Mapping[str, object], certificate_digest: str) -> dict[str, object]:
    return {
        "certificate_digest": certificate_digest,
        "evaluator_protocol_digest": certificate["evaluator_protocol_digest"],
        "problem_digest": certificate["problem_digest"],
        "reference_energy": certificate["reference_energy"],
        "reference_status": "exact_enumeration",
        "schema": REFERENCE_SCHEMA,
        "schema_version": REFERENCE_SCHEMA_VERSION,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _source_problems(
    corpus_paths: Sequence[Path],
    checksums: Mapping[str, str],
) -> tuple[dict[str, _ProblemSources], list[dict[str, object]], int]:
    problems: dict[str, _ProblemSources] = {}
    source_files: list[dict[str, object]] = []
    row_count = 0
    for corpus in sorted(corpus_paths, key=lambda item: item.name):
        corpus_sha = _verify_checksum(corpus, checksums)
        manifest_path = Path(f"{corpus}.manifest.json")
        manifest_sha = _verify_checksum(manifest_path, checksums)
        manifest = _strict_json(manifest_path.read_bytes(), f"manifest for {corpus.name}")
        if Path(str(manifest.get("file", ""))).name != corpus.name:
            raise ValueError(f"manifest for {corpus.name} points to another corpus")
        if _require_sha256(manifest.get("sha256"), "manifest corpus SHA-256") != corpus_sha:
            raise ValueError(f"manifest for {corpus.name} disagrees with authenticated content")
        lines = corpus.read_bytes().splitlines()
        if not lines:
            raise ValueError(f"{corpus.name} is empty")
        source_files.append(
            {
                "file": corpus.name,
                "manifest": manifest_path.name,
                "manifest_sha256": manifest_sha,
                "rows": len(lines),
                "sha256": corpus_sha,
            }
        )
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise ValueError(f"{corpus.name} line {line_number} is blank")
            raw = _strict_json(line, f"{corpus.name} line {line_number}")
            mode = raw.get("mode")
            instance_id = raw.get("instance_id")
            if not isinstance(mode, str) or not mode:
                raise ValueError("quality row mode must be a non-empty string")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError("quality row instance_id must be a non-empty string")
            problem, reference, digest = _problem(raw.get("problem"))
            source_record = {
                "corpus": corpus.name,
                "instance_id": instance_id,
                "line": line_number,
                "record_sha256": hashlib.sha256(line).hexdigest(),
            }
            entry = problems.setdefault(
                digest,
                _ProblemSources(
                    declared_energy=reference,
                    modes=set(),
                    problem=problem,
                    source_records=[],
                ),
            )
            if entry.problem != problem or entry.declared_energy != reference:
                raise RuntimeError("a semantic problem digest collision was detected")
            entry.modes.add(mode)
            entry.source_records.append(source_record)
            row_count += 1
    return problems, source_files, row_count


def _validate_inputs(corpus_paths: Sequence[str | os.PathLike[str]]) -> list[Path]:
    paths = [Path(path) for path in corpus_paths]
    if not paths:
        raise ValueError("at least one release-v1.1 quality corpus is required")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("quality corpus basenames must be unique")
    if any(not path.name.startswith("quality_") or path.suffix != ".jsonl" for path in paths):
        raise ValueError("only quality_*.jsonl corpora are accepted")
    return paths


def _exclude_reason(modes: set[str], variable_count: int, max_exact_variables: int) -> str | None:
    if any(mode.startswith("graphcut") for mode in modes):
        return "legacy_graphcut_reference_was_generated_by_tabu"
    if variable_count > max_exact_variables:
        return "problem_exceeds_registered_exact_enumeration_limit"
    return None


def _best_known_record(
    problem_digest: str,
    item: _ProblemSources,
    reason: str,
) -> dict[str, object]:
    modes = sorted(item.modes)
    source_records = sorted(
        item.source_records,
        key=lambda source: (source["corpus"], source["line"], source["record_sha256"]),
    )
    payload = {
        "mode": modes[0] if len(modes) == 1 else modes,
        "n_variables": len(item.problem.variables),
        "problem_digest": problem_digest,
        "reason": reason,
        "reference_energy": item.declared_energy,
        "schema": "embedbench.problem-reference-best-known",
        "schema_version": 1,
        "source_records": source_records,
        "status": "uncertified_best_known",
    }
    return {**payload, "record_digest": content_digest(payload)}


def export_problem_references(
    corpus_paths: Sequence[str | os.PathLike[str]],
    *,
    checksums_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    max_exact_variables: int = DEFAULT_MAX_EXACT_VARIABLES,
    allow_incomplete: bool = False,
) -> dict[str, object]:
    """Export exact references atomically from authenticated legacy corpus bytes.

    By default every selected semantic problem must be certifiable.  Set
    ``allow_incomplete`` only to produce an audit classification; that output is marked
    ``importer_complete=false`` and is not a complete sidecar for the selected corpora.
    """

    paths = _validate_inputs(corpus_paths)
    if (
        isinstance(max_exact_variables, bool)
        or not isinstance(max_exact_variables, int)
        or not 1 <= max_exact_variables <= MAX_EXHAUSTIVE_VARIABLES
    ):
        raise ValueError(
            f"max_exact_variables must lie in [1, {MAX_EXHAUSTIVE_VARIABLES}]"
        )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"reference export already exists: {destination}")
    source_checksums_path = Path(checksums_path)
    checksums = _checksum_table(source_checksums_path)
    problems, source_files, source_rows = _source_problems(paths, checksums)
    excluded_reasons = {
        digest: reason
        for digest, item in problems.items()
        if (
            reason := _exclude_reason(
                item.modes,
                len(item.problem.variables),
                max_exact_variables,
            )
        )
    }
    if excluded_reasons and not allow_incomplete:
        graphcuts = sum(
            reason == "legacy_graphcut_reference_was_generated_by_tabu"
            for reason in excluded_reasons.values()
        )
        raise IncompleteReferenceCoverageError(
            f"selected corpora contain {len(excluded_reasons)} uncertified problems "
            f"({graphcuts} graph-cut tabu references); select exact-only corpora or use "
            "allow_incomplete for an audit artifact"
        )

    protocol = _protocol(max_exact_variables)
    protocol_sha = content_digest(protocol)
    references: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    best_known: list[dict[str, object]] = []
    for digest, item in sorted(problems.items()):
        source_records = sorted(
            item.source_records,
            key=lambda source: (source["corpus"], source["line"], source["record_sha256"]),
        )
        reason = excluded_reasons.get(digest)
        if reason is not None:
            best_known.append(_best_known_record(digest, item, reason))
            continue
        certificate = _certificate(
            item.problem,
            digest,
            item.declared_energy,
            protocol_sha,
        )
        certificate_digest = content_digest(certificate)
        references.append(_reference(certificate, certificate_digest))
        evidence.append(
            {
                "certificate": certificate,
                "certificate_digest": certificate_digest,
                "source_records": source_records,
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        payloads: dict[str, bytes] = {
            "problem_reference_certificates.jsonl": _jsonl_bytes(evidence),
            "problem_references.best_known.jsonl": _jsonl_bytes(best_known),
            "problem_references.jsonl": _jsonl_bytes(references),
            "problem_reference_protocol.json": canonical_json_bytes(protocol) + b"\n",
        }
        for filename, file_payload in payloads.items():
            _atomic_write(temporary / filename, file_payload)
        outputs = {
            filename: {
                "bytes": len(file_payload),
                "sha256": hashlib.sha256(file_payload).hexdigest(),
            }
            for filename, file_payload in sorted(payloads.items())
        }
        manifest = {
            "certified_problems": len(references),
            "excluded_problems": len(best_known),
            "importer_complete": not best_known,
            "max_exact_variables": max_exact_variables,
            "outputs": outputs,
            "protocol_sha256": protocol_sha,
            "schema": EXPORT_SCHEMA,
            "schema_version": EXPORT_SCHEMA_VERSION,
            "source_checksums": source_checksums_path.name,
            "source_checksums_sha256": _sha256(source_checksums_path),
            "source_corpora": source_files,
            "source_rows": source_rows,
            "unique_problems": len(problems),
        }
        manifest_payload = canonical_json_bytes(manifest) + b"\n"
        _atomic_write(temporary / "problem_reference_export_manifest.json", manifest_payload)
        generated = {
            **{name: item["sha256"] for name, item in outputs.items()},
            "problem_reference_export_manifest.json": hashlib.sha256(manifest_payload).hexdigest(),
        }
        overlap = set(checksums).intersection(generated)
        if overlap:
            raise ValueError(
                f"source SHA256SUMS already contains generated names: {sorted(overlap)}"
            )
        checksum_payload = "".join(
            f"{digest}  {filename}\n"
            for filename, digest in (*checksums.items(), *sorted(generated.items()))
        ).encode("utf-8")
        _atomic_write(temporary / "SHA256SUMS", checksum_payload)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def verify_problem_reference_export(
    corpus_paths: Sequence[str | os.PathLike[str]],
    *,
    checksums_path: str | os.PathLike[str],
    export_dir: str | os.PathLike[str],
    verify_output_checksums: bool = True,
) -> dict[str, int]:
    """Re-authenticate source bytes and replay every emitted exact certificate."""

    paths = _validate_inputs(corpus_paths)
    source_checksums_path = Path(checksums_path)
    source_checksums = _checksum_table(source_checksums_path)
    problems, source_files, source_rows = _source_problems(paths, source_checksums)
    root = Path(export_dir)
    manifest_path = root / "problem_reference_export_manifest.json"
    manifest = _strict_json(manifest_path.read_bytes(), "reference export manifest")
    protocol_path = root / "problem_reference_protocol.json"
    reference_path = root / "problem_references.jsonl"
    evidence_path = root / "problem_reference_certificates.jsonl"
    excluded_path = root / "problem_references.best_known.jsonl"
    output_paths = (manifest_path, protocol_path, reference_path, evidence_path, excluded_path)
    if verify_output_checksums:
        augmented = _checksum_table(root / "SHA256SUMS")
        for path in output_paths:
            _verify_checksum(path, augmented)
    if manifest.get("schema") != EXPORT_SCHEMA or manifest.get("schema_version") != 1:
        raise ValueError("unsupported reference export manifest")
    if manifest.get("source_checksums_sha256") != _sha256(source_checksums_path):
        raise ValueError("reference export binds a different source SHA256SUMS")
    if manifest.get("source_corpora") != source_files or manifest.get("source_rows") != source_rows:
        raise ValueError("reference export source coverage differs from authenticated corpora")
    max_exact_variables = manifest.get("max_exact_variables")
    if isinstance(max_exact_variables, bool) or not isinstance(max_exact_variables, int):
        raise ValueError("reference export max_exact_variables is invalid")
    protocol = _strict_json(protocol_path.read_bytes(), "reference evaluator protocol")
    protocol_sha = content_digest(protocol)
    if (
        protocol != _protocol(max_exact_variables)
        or manifest.get("protocol_sha256") != protocol_sha
    ):
        raise ValueError("reference evaluator protocol differs from the registered implementation")

    references = _jsonl(reference_path, "problem-reference sidecar", allow_empty=True)
    evidence_rows = _jsonl(evidence_path, "problem-reference certificates", allow_empty=True)
    exclusions = _jsonl(excluded_path, "best-known exclusions", allow_empty=True)
    evidence_by_digest: dict[str, dict[str, Any]] = {}
    for row in evidence_rows:
        if set(row) != {"certificate", "certificate_digest", "source_records"}:
            raise ValueError("certificate evidence schema differs")
        certificate = row["certificate"]
        if not isinstance(certificate, Mapping):
            raise ValueError("certificate must be an object")
        digest = _require_sha256(row["certificate_digest"], "certificate digest")
        if content_digest(certificate) != digest:
            raise ValueError("certificate digest mismatch")
        problem_digest = _require_sha256(certificate.get("problem_digest"), "problem digest")
        if problem_digest in evidence_by_digest:
            raise ValueError("certificate evidence repeats a problem")
        item = problems.get(problem_digest)
        if item is None:
            raise ValueError("certificate references an unknown source problem")
        expected = _certificate(
            item.problem,
            problem_digest,
            item.declared_energy,
            protocol_sha,
        )
        if certificate != expected:
            raise ValueError("certificate transcript differs from exact replay")
        expected_sources = sorted(
            item.source_records,
            key=lambda source: (source["corpus"], source["line"], source["record_sha256"]),
        )
        if row["source_records"] != expected_sources:
            raise ValueError("certificate source-record coverage differs")
        evidence_by_digest[problem_digest] = row

    reference_by_digest: dict[str, dict[str, Any]] = {}
    for row in references:
        if set(row) != _REFERENCE_FIELDS:
            raise ValueError("problem-reference row schema differs")
        if row["schema"] != REFERENCE_SCHEMA or row["schema_version"] != 1:
            raise ValueError("unsupported problem-reference schema")
        problem_digest = _require_sha256(row["problem_digest"], "problem digest")
        evidence = evidence_by_digest.get(problem_digest)
        if evidence is None or row != _reference(
            evidence["certificate"], evidence["certificate_digest"]
        ):
            raise ValueError("problem-reference row differs from its exact certificate")
        if problem_digest in reference_by_digest:
            raise ValueError("problem-reference sidecar repeats a problem")
        reference_by_digest[problem_digest] = row

    excluded_digests = set()
    for row in exclusions:
        digest = _require_sha256(row.get("problem_digest"), "excluded problem digest")
        if digest in excluded_digests:
            raise ValueError("best-known exclusions repeat a problem")
        item = problems.get(digest)
        if item is None:
            raise ValueError("best-known exclusion references an unknown source problem")
        reason = _exclude_reason(item.modes, len(item.problem.variables), max_exact_variables)
        if reason is None or row != _best_known_record(digest, item, reason):
            raise ValueError("best-known exclusion differs from its authenticated source facts")
        excluded_digests.add(digest)
    expected_excluded = {
        digest
        for digest, item in problems.items()
        if _exclude_reason(item.modes, len(item.problem.variables), max_exact_variables)
        is not None
    }
    if excluded_digests != expected_excluded:
        raise ValueError("best-known exclusion coverage differs from the registered policy")
    if set(reference_by_digest) != set(problems) - expected_excluded:
        raise ValueError("certified reference coverage differs from the registered policy")
    if set(reference_by_digest) != set(evidence_by_digest):
        raise ValueError("reference and certificate coverage differ")
    if set(reference_by_digest) | excluded_digests != set(problems):
        raise ValueError("reference export does not exactly classify every source problem")
    if bool(manifest.get("importer_complete")) != (not excluded_digests):
        raise ValueError("reference export importer_complete flag is inconsistent")
    if manifest.get("certified_problems") != len(reference_by_digest):
        raise ValueError("reference export certified count differs")
    if manifest.get("excluded_problems") != len(excluded_digests):
        raise ValueError("reference export excluded count differs")
    if manifest.get("unique_problems") != len(problems):
        raise ValueError("reference export unique-problem count differs")
    return {"certified_problems": len(reference_by_digest), "source_rows": source_rows}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "verify"))
    parser.add_argument("corpora", nargs="+")
    parser.add_argument("--checksums", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-exact-variables", type=int, default=DEFAULT_MAX_EXACT_VARIABLES)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result: Mapping[str, object]
    if args.mode == "export":
        result = export_problem_references(
            args.corpora,
            checksums_path=args.checksums,
            output_dir=args.output_dir,
            max_exact_variables=args.max_exact_variables,
            allow_incomplete=args.allow_incomplete,
        )
    else:
        result = verify_problem_reference_export(
            args.corpora,
            checksums_path=args.checksums,
            export_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the public functions
    raise SystemExit(main())

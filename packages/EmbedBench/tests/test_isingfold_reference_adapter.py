from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_reference_adapter import adapt_reference_export_bundle
from embedbench.reference_export import export_problem_references


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_row(
    instance_id: str = "fixture-random",
    *,
    first_field: float = -1.0,
    second_field: float = 0.0,
    coupling: float = -0.5,
    energy: float = -1.5,
    mode: str = "random",
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "mode": mode,
        "problem": {
            "J": [[0, 1, coupling]],
            "e0": energy,
            "h": {"0": first_field, "1": second_field},
        },
    }


def _release(root: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    corpus = root / "quality_fixture_random.jsonl"
    corpus.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    corpus_sha256 = _sha256(corpus)
    manifest = root / f"{corpus.name}.manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "file": f"runs/release_v1_1/{corpus.name}",
                "sha256": corpus_sha256,
            }
        )
        + b"\n"
    )
    checksums = root / "SHA256SUMS"
    checksums.write_text(
        f"{corpus_sha256}  {corpus.name}\n{_sha256(manifest)}  {manifest.name}\n",
        encoding="utf-8",
    )
    return corpus, checksums


def _reference_export(
    root: Path,
    rows: list[dict[str, object]] | None = None,
    *,
    allow_incomplete: bool = False,
) -> Path:
    corpus, checksums = _release(root / "release", rows or [_quality_row()])
    output = root / "reference-export"
    export_problem_references(
        [corpus],
        checksums_path=checksums,
        output_dir=output,
        allow_incomplete=allow_incomplete,
    )
    return output


def _adapt(source: Path, output: Path):
    return adapt_reference_export_bundle(
        source,
        output,
        expected_manifest_sha256=_sha256(source / "problem_reference_export_manifest.json"),
        expected_protocol_sha256=_sha256(source / "problem_reference_protocol.json"),
        expected_references_sha256=_sha256(source / "problem_references.jsonl"),
        expected_certificates_sha256=_sha256(source / "problem_reference_certificates.jsonl"),
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _rebind_output(source: Path, filename: str) -> None:
    manifest_path = source / "problem_reference_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = (source / filename).read_bytes()
    manifest["outputs"][filename] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")


def test_adapts_real_reference_export_to_publisher_contract(tmp_path: Path) -> None:
    source = _reference_export(tmp_path)
    output = tmp_path / "isingfold-references"

    receipt = _adapt(source, output)

    index = output / "isingfold_references.jsonl"
    rows = _jsonl(index)
    assert len(rows) == 1
    assert set(rows[0]) == {
        "certificate",
        "evaluator_protocol_digest",
        "problem_sha256",
        "record_digest",
        "reference_energy",
        "reference_status",
        "schema",
        "schema_version",
    }
    assert rows[0]["schema"] == "embedbench.isingfold-reference"
    assert rows[0]["schema_version"] == 1
    assert rows[0]["reference_status"] == "exact_enumeration"
    payload = {key: value for key, value in rows[0].items() if key != "record_digest"}
    assert rows[0]["record_digest"] == content_digest(payload)
    assert index.read_bytes() == canonical_json_bytes(rows[0]) + b"\n"

    descriptor = rows[0]["certificate"]
    certificate = output / descriptor["path"]
    assert descriptor == {
        "path": f"certificates/{descriptor['sha256']}",
        "sha256": _sha256(certificate),
    }
    assert json.loads(certificate.read_bytes())["problem_sha256"] == rows[0]["problem_sha256"]
    assert receipt.output_directory == output
    assert receipt.reference_index_path == index
    assert receipt.reference_index_sha256 == _sha256(index)
    assert receipt.reference_count == receipt.certificate_count == 1
    assert receipt.evaluator_protocol_digest == rows[0]["evaluator_protocol_digest"]


def test_external_reference_pin_rejects_byte_tamper(tmp_path: Path) -> None:
    source = _reference_export(tmp_path)
    references = source / "problem_references.jsonl"
    original_digest = _sha256(references)
    references.write_bytes(references.read_bytes() + b" ")

    with pytest.raises(ValueError, match="external SHA-256 commitment"):
        adapt_reference_export_bundle(
            source,
            tmp_path / "output",
            expected_manifest_sha256=_sha256(source / "problem_reference_export_manifest.json"),
            expected_protocol_sha256=_sha256(source / "problem_reference_protocol.json"),
            expected_references_sha256=original_digest,
            expected_certificates_sha256=_sha256(source / "problem_reference_certificates.jsonl"),
        )
    assert not (tmp_path / "output").exists()


def test_incomplete_export_cannot_promote_best_known_rows(tmp_path: Path) -> None:
    source = _reference_export(
        tmp_path,
        [
            _quality_row(),
            _quality_row(
                "fixture-graphcut",
                first_field=-0.75,
                energy=-1.25,
                mode="graphcut5x5",
            ),
        ],
        allow_incomplete=True,
    )

    with pytest.raises(ValueError, match="best-known or uncertified"):
        _adapt(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_certificate_evidence_coverage_is_exact(tmp_path: Path, mutation: str) -> None:
    source = _reference_export(
        tmp_path,
        [
            _quality_row(),
            _quality_row(
                "fixture-second",
                first_field=-0.5,
                second_field=-0.25,
                coupling=-0.25,
                energy=-1.0,
            ),
        ],
    )
    evidence_path = source / "problem_reference_certificates.jsonl"
    evidence = _jsonl(evidence_path)
    evidence = evidence[:-1] if mutation == "missing" else [*evidence, dict(evidence[-1])]
    _write_jsonl(evidence_path, evidence)
    _rebind_output(source, evidence_path.name)

    with pytest.raises(ValueError, match="coverage|repeats"):
        _adapt(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_problem_reference_coverage_is_exact(tmp_path: Path, mutation: str) -> None:
    source = _reference_export(
        tmp_path,
        [
            _quality_row(),
            _quality_row(
                "fixture-second",
                first_field=-0.5,
                second_field=-0.25,
                coupling=-0.25,
                energy=-1.0,
            ),
        ],
    )
    references_path = source / "problem_references.jsonl"
    references = _jsonl(references_path)
    references = references[:-1] if mutation == "missing" else [*references, dict(references[-1])]
    _write_jsonl(references_path, references)
    _rebind_output(source, references_path.name)

    with pytest.raises(ValueError, match="count|repeats"):
        _adapt(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_uncertified_reference_status_is_never_upgraded(tmp_path: Path) -> None:
    source = _reference_export(tmp_path)
    references_path = source / "problem_references.jsonl"
    references = _jsonl(references_path)
    references[0]["reference_status"] = "uncertified_reference"
    _write_jsonl(references_path, references)
    _rebind_output(source, references_path.name)

    with pytest.raises(ValueError, match="best-known or uncertified"):
        _adapt(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_rebound_certificate_tamper_still_fails_content_digest(tmp_path: Path) -> None:
    source = _reference_export(tmp_path)
    evidence_path = source / "problem_reference_certificates.jsonl"
    evidence = _jsonl(evidence_path)
    evidence[0]["certificate"]["checked_state_count"] += 1
    _write_jsonl(evidence_path, evidence)
    _rebind_output(source, evidence_path.name)

    with pytest.raises(ValueError, match="certificate bytes"):
        _adapt(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_certificate_evidence_covers_every_source_row(tmp_path: Path) -> None:
    source = _reference_export(
        tmp_path,
        [_quality_row("duplicate-a"), _quality_row("duplicate-b")],
    )
    evidence_path = source / "problem_reference_certificates.jsonl"
    evidence = _jsonl(evidence_path)
    evidence[0]["source_records"].pop()
    _write_jsonl(evidence_path, evidence)
    _rebind_output(source, evidence_path.name)

    with pytest.raises(ValueError, match="every source corpus row"):
        _adapt(source, tmp_path / "output")


def test_forged_huge_source_count_fails_without_materializing_the_range(
    tmp_path: Path,
) -> None:
    source = _reference_export(tmp_path)
    manifest_path = source / "problem_reference_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_corpora"][0]["rows"] = 10**12
    manifest["source_rows"] = 10**12
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ValueError, match="every source corpus row"):
        _adapt(source, tmp_path / "output")


def test_input_symlink_and_manifest_path_traversal_are_rejected(tmp_path: Path) -> None:
    source = _reference_export(tmp_path / "symlink-case")
    protocol = source / "problem_reference_protocol.json"
    outside = tmp_path / "protocol-copy.json"
    outside.write_bytes(protocol.read_bytes())
    protocol.unlink()
    protocol.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file, not a symlink"):
        _adapt(source, tmp_path / "symlink-output")

    traversal_source = _reference_export(tmp_path / "traversal-case")
    manifest_path = traversal_source / "problem_reference_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_checksums"] = "../SHA256SUMS"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="safe single-component"):
        _adapt(traversal_source, tmp_path / "traversal-output")


def test_existing_output_is_preserved(tmp_path: Path) -> None:
    source = _reference_export(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "owned-by-caller"
    sentinel.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        _adapt(source, output)

    assert sentinel.read_bytes() == b"keep"
    assert {path.name for path in output.iterdir()} == {sentinel.name}

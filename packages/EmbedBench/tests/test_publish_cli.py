from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from embedbench import publish_cli


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_source_command_forwards_all_external_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_publish(**kwargs):
        observed.update(kwargs)
        return {
            "schema": "embedbench.isingfold-source-publication",
            "record_digest": "a" * 64,
        }

    import embedbench.isingfold_publication as publication

    monkeypatch.setattr(publication, "publish_isingfold_source", fake_publish)
    result = publish_cli.main(
        [
            "source",
            "--bank",
            str(tmp_path / "bank.jsonl"),
            "--bank-manifest",
            str(tmp_path / "bank.manifest.json"),
            "--publication-index",
            str(tmp_path / "publication-index.json"),
            "--reference-index",
            str(tmp_path / "references.jsonl"),
            "--expected-bank-manifest-sha256",
            "1" * 64,
            "--expected-publication-index-sha256",
            "2" * 64,
            "--expected-reference-index-sha256",
            "3" * 64,
            "--out",
            str(tmp_path / "published"),
        ]
    )

    assert result == 0
    assert observed == {
        "bank_path": tmp_path / "bank.jsonl",
        "bank_manifest_path": tmp_path / "bank.manifest.json",
        "publication_index_path": tmp_path / "publication-index.json",
        "reference_index_path": tmp_path / "references.jsonl",
        "output_dir": tmp_path / "published",
        "expected_bank_manifest_sha256": "1" * 64,
        "expected_publication_index_sha256": "2" * 64,
        "expected_reference_index_sha256": "3" * 64,
    }
    assert json.loads(capsys.readouterr().out)["record_digest"] == "a" * 64


def test_attest_command_builds_exact_digest_to_path_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    certificate_root = tmp_path / "certificates"
    certificate_root.mkdir()
    first = b"first certificate"
    second = b"second certificate"
    first_digest = _digest(first)
    second_digest = _digest(second)
    (certificate_root / first_digest).write_bytes(first)
    (certificate_root / second_digest).write_bytes(second)
    observed: dict[str, object] = {}

    def fake_build(*args, **kwargs):
        from embedbench.isingfold_attestation import PublisherAttestationReceipt

        observed["args"] = args
        observed["kwargs"] = kwargs
        return PublisherAttestationReceipt(
            output_directory=tmp_path / "attestation",
            attestation_path=tmp_path / "attestation" / "publisher-attestation.json",
            attestation_record_digest="4" * 64,
            attestation_sha256="5" * 64,
            publisher_id="publisher",
            publication_id="publication",
            prepared_manifest_sha256="6" * 64,
            target_count=7,
            certificate_count=2,
            verifier=kwargs["verifier"],
        )

    import embedbench.isingfold_attestation as attestation

    monkeypatch.setattr(attestation, "build_publisher_attestation_v2", fake_build)
    result = publish_cli.main(
        [
            "attest",
            "--prepared",
            str(tmp_path / "prepared"),
            "--certificate-root",
            str(certificate_root),
            "--publisher-id",
            "publisher",
            "--publication-id",
            "publication",
            "--verifier-name",
            "independent-ground-verifier",
            "--verifier-version",
            "1.0.0",
            "--verifier-implementation-sha256",
            "7" * 64,
            "--out",
            str(tmp_path / "attestation"),
        ]
    )

    assert result == 0
    assert observed["args"] == (
        tmp_path / "prepared",
        {
            first_digest: certificate_root / first_digest,
            second_digest: certificate_root / second_digest,
        },
        tmp_path / "attestation",
    )
    verifier = observed["kwargs"]["verifier"]
    assert verifier.as_dict() == {
        "implementation_sha256": "7" * 64,
        "name": "independent-ground-verifier",
        "version": "1.0.0",
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed["attestation_record_digest"] == "4" * 64
    assert printed["certificate_count"] == 2
    assert printed["attestation_path"].endswith("publisher-attestation.json")


def test_adapt_references_command_forwards_all_external_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def fake_adapt(*args, **kwargs):
        from embedbench.isingfold_reference_adapter import ReferenceAdapterReceipt

        observed["args"] = args
        observed["kwargs"] = kwargs
        return ReferenceAdapterReceipt(
            output_directory=tmp_path / "adapted",
            reference_index_path=tmp_path / "adapted" / "isingfold_references.jsonl",
            reference_index_sha256="5" * 64,
            reference_count=3,
            certificate_directory=tmp_path / "adapted" / "certificates",
            certificate_count=3,
            source_manifest_sha256="1" * 64,
            source_protocol_sha256="2" * 64,
            source_references_sha256="3" * 64,
            source_certificates_sha256="4" * 64,
            evaluator_protocol_digest="6" * 64,
        )

    import embedbench.isingfold_reference_adapter as adapter

    monkeypatch.setattr(adapter, "adapt_reference_export_bundle", fake_adapt)
    result = publish_cli.main(
        [
            "adapt-references",
            "--reference-export",
            str(tmp_path / "reference-export"),
            "--expected-manifest-sha256",
            "1" * 64,
            "--expected-protocol-sha256",
            "2" * 64,
            "--expected-references-sha256",
            "3" * 64,
            "--expected-certificates-sha256",
            "4" * 64,
            "--out",
            str(tmp_path / "adapted"),
        ]
    )

    assert result == 0
    assert observed == {
        "args": (tmp_path / "reference-export", tmp_path / "adapted"),
        "kwargs": {
            "expected_manifest_sha256": "1" * 64,
            "expected_protocol_sha256": "2" * 64,
            "expected_references_sha256": "3" * 64,
            "expected_certificates_sha256": "4" * 64,
        },
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed["reference_count"] == 3
    assert printed["reference_index_sha256"] == "5" * 64


def test_certificate_registry_rejects_unexpected_names_and_symlinks(tmp_path: Path) -> None:
    certificate_root = tmp_path / "certificates"
    certificate_root.mkdir()
    (certificate_root / "not-a-digest").write_bytes(b"certificate")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        publish_cli.certificate_registry(certificate_root)

    (certificate_root / "not-a-digest").unlink()
    payload = b"certificate"
    digest = _digest(payload)
    outside = tmp_path / "outside"
    outside.write_bytes(payload)
    (certificate_root / digest).symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        publish_cli.certificate_registry(certificate_root)


def test_console_entry_point_is_registered() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'embedbench-publish = "embedbench.publish_cli:main"' in pyproject.read_text()

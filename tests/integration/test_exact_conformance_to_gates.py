from __future__ import annotations

import hashlib
import json
from pathlib import Path

from isingfold.rl.cli import _load_exact_conformance_tasks, main
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from tests.unit.test_rl_exact_conformance_protocol import _population


def test_cli_producer_verifier_and_gate_loader_share_target_free_selector(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    population = _population()
    corpus = tmp_path / "prepared-v4"
    corpus.mkdir()
    manifest_payload = {
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": 4,
    }
    manifest = {**manifest_payload, "record_digest": content_digest(manifest_payload)}
    manifest_path = corpus / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    for filename in (
        "initializers.jsonl",
        "policy_instances.jsonl",
        "provenance.jsonl",
        "splits.json",
    ):
        (corpus / filename).write_bytes(b"")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    evaluator_flags: list[bool] = []

    def public_loader(directory: Path, partition: str, *, include_evaluator: bool):
        assert Path(directory) == corpus
        assert partition == "val"
        evaluator_flags.append(include_evaluator)
        return type("Loaded", (), {"tasks": tuple(population), "target_access": None})()

    monkeypatch.setattr(
        "isingfold.rl.data.exact_conformance.load_prepared_partition", public_loader
    )
    registry = tmp_path / "exact-conformance.json"
    main(
        [
            "make-exact-conformance-corpus",
            "--corpus",
            str(corpus),
            "--expected-corpus-manifest-sha256",
            manifest_sha,
            "--out",
            str(registry),
        ]
    )
    make_output = json.loads(capsys.readouterr().out)
    registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()
    assert make_output["file_sha256"] == registry_sha

    receipt = tmp_path / "verification.json"
    main(
        [
            "verify-exact-conformance-corpus",
            "--corpus",
            str(corpus),
            "--expected-corpus-manifest-sha256",
            manifest_sha,
            "--registry",
            str(registry),
            "--expected-registry-sha256",
            registry_sha,
            "--receipt-out",
            str(receipt),
        ]
    )
    verify_output = json.loads(capsys.readouterr().out)
    verification = json.loads(receipt.read_text())
    assert verification["exact_reproduction"] is True
    assert verification["selector_implementation"]["version"]
    assert receipt.read_bytes() == canonical_json_bytes(verification) + b"\n"
    assert verify_output["receipt_file_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()

    main(
        [
            "verify-exact-conformance-corpus",
            "--corpus",
            str(corpus),
            "--expected-corpus-manifest-sha256",
            manifest_sha,
            "--registry",
            str(registry),
            "--expected-registry-sha256",
            registry_sha,
        ]
    )
    local_verification = json.loads(capsys.readouterr().out)
    assert local_verification["exact_reproduction"] is True
    assert local_verification["record_digest"] == content_digest(
        {key: value for key, value in local_verification.items() if key != "record_digest"}
    )

    gate_tasks, gate_identity = _load_exact_conformance_tasks(
        registry,
        expected_sha256=registry_sha,
        corpus=corpus,
        expected_corpus_manifest_sha256=manifest_sha,
    )
    assert [task.task_id for task in gate_tasks] == verification["selected_task_ids"]
    assert gate_identity["exact_reproduction"] is True
    assert evaluator_flags == [False, False, False, False]

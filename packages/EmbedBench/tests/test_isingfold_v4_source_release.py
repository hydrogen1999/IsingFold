from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from embedbench.candidate_bank import canonical_json_bytes, content_digest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data" / "provenance" / "isingfold_corpus_v4" / "plan_v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_source_release_is_canonical_and_binds_current_source() -> None:
    source_release_path = RELEASE / "SOURCE_RELEASE.json"
    raw = source_release_path.read_bytes()
    document = json.loads(raw)
    assert raw == canonical_json_bytes(document) + b"\n"
    payload = {key: value for key, value in document.items() if key != "record_digest"}
    assert document["record_digest"] == content_digest(payload)

    helper = ROOT / "scripts" / "verify_isingfold_corpus_runtime.sh"
    observed_source = subprocess.run(
        [
            "bash",
            str(helper),
            "source-sha256",
            str(ROOT),
            sys.executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert observed_source == document["source_inventory"]["sha256"]

    plan = RELEASE / "corpus_plan.json"
    assert _sha256(plan) == document["plan"]["sha256"]
    plan_document = json.loads(plan.read_bytes())
    assert plan_document["record_digest"] == document["plan"]["record_digest"]
    assert len(plan_document["lineages"]) == document["plan"]["lineage_count"] == 3082


def test_v4_source_release_binds_runtime_and_operational_files() -> None:
    document = json.loads((RELEASE / "SOURCE_RELEASE.json").read_bytes())
    operational_paths = {
        "apollo_launcher_sha256": ROOT / "scripts" / "apollo_isingfold_corpus.sh",
        "goose_builder_sha256": (
            ROOT / "scripts" / "goose_build_isingfold_corpus_runtime.sbatch"
        ),
        "goose_launcher_sha256": ROOT / "scripts" / "goose_isingfold_corpus.sbatch",
        "runtime_helper_sha256": (
            ROOT / "scripts" / "verify_isingfold_corpus_runtime.sh"
        ),
        "runtime_validator_sha256": (
            ROOT / "scripts" / "validate_isingfold_corpus_image.py"
        ),
    }
    runtime_paths = {
        "definition_sha256": ROOT / "runtime" / "isingfold-corpus" / "isingfold-corpus.def",
        "requirements_lock_sha256": (
            ROOT
            / "runtime"
            / "isingfold-corpus"
            / "requirements-linux-x86_64-py312.lock"
        ),
        "runtime_lock_sha256": (
            ROOT / "runtime" / "isingfold-corpus" / "runtime-lock.json"
        ),
    }
    for field, path in operational_paths.items():
        assert _sha256(path) == document["operational_files"][field]
    for field, path in runtime_paths.items():
        assert _sha256(path) == document["runtime"][field]
    assert document["resource_policy"] == {
        "corpus_generation": "cpu-only",
        "gpu_requested": False,
        "threads_per_worker": 1,
    }


def test_latest_local_audit_has_3082_globally_unique_identities() -> None:
    audit = RELEASE / "audit_only" / "local_macos_source-5f8f636d"
    preflight_path = audit / "prospective_preflight_v2.json"
    receipt = json.loads((audit / "diagnostic_receipt.json").read_bytes())
    preflight = json.loads(preflight_path.read_bytes())
    entries = preflight["prospective_identity_map"]["entries"]

    assert _sha256(preflight_path) == receipt["preflight_sha256"]
    assert preflight["record_digest"] == receipt["preflight_record_digest"]
    assert preflight["prospective_identity_map"]["record_digest"] == receipt[
        "prospective_identity_map_digest"
    ]
    assert len(entries) == receipt["prospective_identity_count"] == 3082
    assert len({entry["lineage_id"] for entry in entries}) == 3082
    assert len({entry["split_unit_id"] for entry in entries}) == 3082
    assert len({entry["problem_sha256"] for entry in entries}) == 3082

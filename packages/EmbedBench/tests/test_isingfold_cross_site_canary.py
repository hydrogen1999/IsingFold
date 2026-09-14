from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_corpus_io import write_corpus_shard
from embedbench.isingfold_corpus_plan import (
    build_production_corpus_plan,
    write_corpus_plan,
)
from embedbench.isingfold_corpus_shard import (
    ShardConfig,
    generate_corpus_shard,
    lineage_shard,
)
from embedbench.isingfold_cross_site_canary import (
    publish_parity_attestation,
    verify_parity_attestation,
)

THREAD_CONTROLS = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


@pytest.fixture(scope="module")
def generated_canary(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("cross-site-canary").resolve()
    previous = {name: os.environ.get(name) for name in THREAD_CONTROLS}
    os.environ.update(THREAD_CONTROLS)
    try:
        plan = build_production_corpus_plan(root_seed=260912, shard_count=64)
        counts = [
            sum(
                lineage_shard(item.request.lineage_id, plan.shard_count) == index
                for item in plan.lineages
            )
            for index in range(plan.shard_count)
        ]
        shard_index = counts.index(min(counts))
        plan_path = root / "plan.json"
        plan_receipt = write_corpus_plan(plan, plan_path)
        screening = plan.screening
        config = ShardConfig(
            root_seed=plan.root_seed,
            shard_index=shard_index,
            shard_count=plan.shard_count,
            attempt_slots=screening.attempt_slots,
            incumbent_search_slots=screening.incumbent_search_slots,
            max_split_search=screening.max_split_search,
            strengths=screening.strengths,
            reads=screening.reads,
            sweeps=screening.sweeps,
        )
        shard = generate_corpus_shard(
            config,
            tuple(item.request for item in plan.lineages),
        )
        shard_root = root / "shard"
        write_corpus_shard(shard, shard_root)
        commitments = {
            "generation_provenance_sha256": shard.provenance_digest,
            "installation_sha256": "b" * 64,
            "plan_sha256": plan_receipt.sha256,
            "source_sha256": "d" * 64,
        }
        yield {
            "commitments": commitments,
            "plan": plan_path,
            "shard": shard_root,
        }
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _inputs(tmp_path: Path, generated_canary) -> tuple[Path, Path, Path, dict[str, str]]:
    plan = (tmp_path / "plan.json").resolve()
    shutil.copyfile(generated_canary["plan"], plan)
    apollo = shutil.copytree(generated_canary["shard"], tmp_path / "apollo").resolve()
    goose = shutil.copytree(generated_canary["shard"], tmp_path / "goose").resolve()
    return plan, apollo, goose, generated_canary["commitments"]


def _publish(tmp_path: Path, generated_canary) -> tuple[Path, dict[str, object]]:
    plan, apollo, goose, commitments = _inputs(tmp_path, generated_canary)
    output = (tmp_path / "parity.json").resolve()
    document = publish_parity_attestation(
        apollo_shard=apollo,
        goose_shard=goose,
        plan=plan,
        output=output,
        **commitments,
    )
    return output, document


def test_publishes_and_verifies_exact_cross_site_parity_once(
    tmp_path: Path,
    generated_canary,
) -> None:
    output, document = _publish(tmp_path, generated_canary)
    raw = output.read_bytes()

    assert raw == canonical_json_bytes(document) + b"\n"
    assert set(document["files"]) == {
        "SHA256SUMS",
        "corpus_shard.json",
        "shard_manifest.json",
    }
    verified = verify_parity_attestation(
        attestation=output,
        expected_attestation_sha256=hashlib.sha256(raw).hexdigest(),
        plan=(tmp_path / "plan.json").resolve(),
        **generated_canary["commitments"],
    )
    assert verified == document


def test_rejects_cross_site_byte_drift_and_unknown_files(
    tmp_path: Path,
    generated_canary,
) -> None:
    plan, apollo, goose, commitments = _inputs(tmp_path, generated_canary)
    (goose / "SHA256SUMS").write_bytes((goose / "SHA256SUMS").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256SUMS"):
        publish_parity_attestation(
            apollo_shard=apollo,
            goose_shard=goose,
            plan=plan,
            output=(tmp_path / "parity.json").resolve(),
            **commitments,
        )

    (goose / "SHA256SUMS").write_bytes((apollo / "SHA256SUMS").read_bytes())
    (goose / "unexpected").write_bytes(b"unregistered")
    with pytest.raises(ValueError, match="inventory"):
        publish_parity_attestation(
            apollo_shard=apollo,
            goose_shard=goose,
            plan=plan,
            output=(tmp_path / "parity.json").resolve(),
            **commitments,
        )


def test_rejects_identical_shards_detached_from_authenticated_plan(
    tmp_path: Path,
    generated_canary,
) -> None:
    _, apollo, goose, commitments = _inputs(tmp_path, generated_canary)
    other_plan = build_production_corpus_plan(root_seed=260913, shard_count=64)
    other_path = (tmp_path / "unrelated-plan.json").resolve()
    other_receipt = write_corpus_plan(other_plan, other_path)
    changed = {**commitments, "plan_sha256": other_receipt.sha256}

    with pytest.raises(ValueError, match="config|plan-assigned"):
        publish_parity_attestation(
            apollo_shard=apollo,
            goose_shard=goose,
            plan=other_path,
            output=(tmp_path / "parity.json").resolve(),
            **changed,
        )


def test_rejects_identical_rehashed_but_internally_malformed_shards(
    tmp_path: Path,
    generated_canary,
) -> None:
    plan, apollo, goose, commitments = _inputs(tmp_path, generated_canary)
    shard = json.loads((apollo / "corpus_shard.json").read_bytes())
    del shard["lineages"][0]["group"]
    shard_payload = {key: value for key, value in shard.items() if key != "record_digest"}
    shard["record_digest"] = content_digest(shard_payload)
    shard_raw = canonical_json_bytes(shard) + b"\n"
    manifest = json.loads((apollo / "shard_manifest.json").read_bytes())
    manifest["artifact"] = {
        "byte_count": len(shard_raw),
        "path": "corpus_shard.json",
        "sha256": hashlib.sha256(shard_raw).hexdigest(),
    }
    manifest["shard_record_digest"] = shard["record_digest"]
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "record_digest"
    }
    manifest["record_digest"] = content_digest(manifest_payload)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    sums = (
        f"{hashlib.sha256(shard_raw).hexdigest()}  corpus_shard.json\n"
        f"{hashlib.sha256(manifest_raw).hexdigest()}  shard_manifest.json\n"
    ).encode("ascii")
    for directory in (apollo, goose):
        (directory / "corpus_shard.json").write_bytes(shard_raw)
        (directory / "shard_manifest.json").write_bytes(manifest_raw)
        (directory / "SHA256SUMS").write_bytes(sums)

    with pytest.raises(ValueError, match="generated lineage.*fields differ"):
        publish_parity_attestation(
            apollo_shard=apollo,
            goose_shard=goose,
            plan=plan,
            output=(tmp_path / "parity.json").resolve(),
            **commitments,
        )


def test_parity_attestation_publication_never_overwrites(
    tmp_path: Path,
    generated_canary,
) -> None:
    output, _ = _publish(tmp_path, generated_canary)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_parity_attestation(
            apollo_shard=(tmp_path / "apollo").resolve(),
            goose_shard=(tmp_path / "goose").resolve(),
            plan=(tmp_path / "plan.json").resolve(),
            output=output,
            **generated_canary["commitments"],
        )
    assert output.read_bytes() == original


def test_production_verifier_rejects_missing_tampered_and_legacy_attestation(
    tmp_path: Path,
    generated_canary,
) -> None:
    output, document = _publish(tmp_path, generated_canary)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    plan = (tmp_path / "plan.json").resolve()

    with pytest.raises(ValueError, match="regular non-symlink"):
        verify_parity_attestation(
            attestation=(tmp_path / "missing.json").resolve(),
            expected_attestation_sha256=digest,
            plan=plan,
            **generated_canary["commitments"],
        )
    with pytest.raises(ValueError, match="external commitment"):
        verify_parity_attestation(
            attestation=output,
            expected_attestation_sha256="e" * 64,
            plan=plan,
            **generated_canary["commitments"],
        )

    legacy_payload = {
        key: value for key, value in document.items() if key != "record_digest"
    }
    legacy_payload["schema_version"] = 0
    legacy = {**legacy_payload, "record_digest": content_digest(legacy_payload)}
    legacy_path = (tmp_path / "legacy.json").resolve()
    legacy_path.write_bytes(canonical_json_bytes(legacy) + b"\n")
    with pytest.raises(ValueError, match="schema"):
        verify_parity_attestation(
            attestation=legacy_path,
            expected_attestation_sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
            plan=plan,
            **generated_canary["commitments"],
        )


def test_production_verifier_rejects_commitment_drift(
    tmp_path: Path,
    generated_canary,
) -> None:
    output, _ = _publish(tmp_path, generated_canary)
    raw = output.read_bytes()
    changed = {**generated_canary["commitments"], "source_sha256": "f" * 64}

    with pytest.raises(ValueError, match="commitments differ"):
        verify_parity_attestation(
            attestation=output,
            expected_attestation_sha256=hashlib.sha256(raw).hexdigest(),
            plan=(tmp_path / "plan.json").resolve(),
            **changed,
        )

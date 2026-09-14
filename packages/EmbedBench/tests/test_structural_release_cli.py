from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from embedbench import structural_release_cli
from embedbench.candidate_bank import canonical_json_bytes
from embedbench.structural_release import (
    DECISIONS_FILENAME,
    StructuralReleasePlan,
    make_structural_plan_row,
)


def _tiny_plan() -> StructuralReleasePlan:
    row = make_structural_plan_row(
        root_seed=1,
        topology="chimera",
        size=2,
        mode="compact",
        split="train",
        ood=False,
        difficulty="easy",
        replicate=0,
        faulted=False,
        n_vars=6,
        chain_size=3,
        k_in_play=3,
        radius=1,
        l_cap=4,
        max_window_free=18,
        max_nodes=200_000,
        samples_per_instance=4,
        max_actions=8,
    )
    return StructuralReleasePlan.build(root_seed=1, rows=[row])


def _write_plan(path: Path, plan: StructuralReleasePlan) -> str:
    raw = canonical_json_bytes(plan.to_dict()) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_plan_command_atomically_writes_canonical_pinned_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "nested" / "production-plan.json"
    assert (
        structural_release_cli.main(
            [
                "plan",
                "--root-seed",
                "260912",
                "--replicates-per-cell",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    raw = output.read_bytes()
    document = json.loads(raw)
    assert raw == canonical_json_bytes(document) + b"\n"
    plan = StructuralReleasePlan.from_dict(document)
    assert len(plan.rows) == 144
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "lineage_count": 144,
        "plan_digest": plan.plan_digest,
        "plan_path": str(output.resolve()),
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
    }

    before = raw
    assert (
        structural_release_cli.main(
            [
                "plan",
                "--root-seed",
                "260912",
                "--replicates-per-cell",
                "1",
                "--out",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert output.read_bytes() == before
    assert not list(output.parent.glob(".*.staging-*"))


def test_load_pinned_plan_rejects_wrong_hash_noncanonical_and_symlink(
    tmp_path: Path,
) -> None:
    plan = _tiny_plan()
    path = tmp_path / "plan.json"
    digest = _write_plan(path, plan)
    assert structural_release_cli.load_pinned_plan(path, digest) == plan
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        structural_release_cli.load_pinned_plan(path, "0" * 64)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(plan.to_dict(), indent=2))
    noncanonical_digest = hashlib.sha256(noncanonical.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="canonical"):
        structural_release_cli.load_pinned_plan(noncanonical, noncanonical_digest)

    link = tmp_path / "plan-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular file"):
        structural_release_cli.load_pinned_plan(link, digest)


def test_load_pinned_plan_rejects_duplicate_keys_and_bad_semantic_digest(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":1,"schema":2}\n')
    duplicate_sha256 = hashlib.sha256(duplicate.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate key"):
        structural_release_cli.load_pinned_plan(duplicate, duplicate_sha256)

    plan = _tiny_plan()
    altered = plan.to_dict()
    altered["root_seed"] += 1
    bad_digest_path = tmp_path / "bad-digest.json"
    bad_raw = canonical_json_bytes(altered) + b"\n"
    bad_digest_path.write_bytes(bad_raw)
    with pytest.raises(ValueError, match="plan_digest mismatch"):
        structural_release_cli.load_pinned_plan(
            bad_digest_path, hashlib.sha256(bad_raw).hexdigest()
        )


def test_generate_shard_and_merge_commands_use_mandatory_plan_pin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _tiny_plan()
    plan_path = tmp_path / "tiny-plan.json"
    plan_sha256 = _write_plan(plan_path, plan)
    shard = tmp_path / "shard-0"
    assert (
        structural_release_cli.main(
            [
                "generate-shard",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha256,
                "--shard-index",
                "0",
                "--shard-count",
                "1",
                "--out",
                str(shard),
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out)
    assert generated["plan_digest"] == plan.plan_digest
    assert generated["shard_index"] == 0
    assert generated["shard_count"] == 1
    assert generated["record_count"] > 0

    release = tmp_path / "release"
    assert (
        structural_release_cli.main(
            [
                "merge",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha256,
                "--shard-dir",
                str(shard),
                "--out",
                str(release),
            ]
        )
        == 0
    )
    merged = json.loads(capsys.readouterr().out)
    assert merged["plan_digest"] == plan.plan_digest
    assert merged["lineage_count"] == 1
    assert (release / DECISIONS_FILENAME).read_bytes() == (
        shard / DECISIONS_FILENAME
    ).read_bytes()


def test_bad_plan_pin_fails_before_creating_shard_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _tiny_plan())
    output = tmp_path / "must-not-exist"
    result = structural_release_cli.main(
        [
            "generate-shard",
            "--plan",
            str(plan_path),
            "--expected-plan-sha256",
            "f" * 64,
            "--shard-index",
            "0",
            "--shard-count",
            "1",
            "--out",
            str(output),
        ]
    )
    assert result == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert not output.exists()


def test_generate_shard_requires_an_explicit_plan_hash_pin(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        structural_release_cli.main(
            [
                "generate-shard",
                "--plan",
                str(tmp_path / "plan.json"),
                "--shard-index",
                "0",
                "--shard-count",
                "1",
                "--out",
                str(tmp_path / "shard"),
            ]
        )
    assert raised.value.code == 2


def test_console_entry_point_is_registered() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert (
        'embedbench-structural-release = "embedbench.structural_release_cli:main"'
        in pyproject.read_text()
    )

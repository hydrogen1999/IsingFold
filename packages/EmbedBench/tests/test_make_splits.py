from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "make_splits.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_splits", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _legacy_split(instance_id: str) -> str:
    value = int(hashlib.sha256(instance_id.encode()).hexdigest()[:8], 16) / 2**32
    return "train" if value < 0.7 else ("val" if value < 0.8 else "test")


def _run(files: list[Path], output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *(str(path) for path in files),
            "--out",
            str(output),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_problem_digest_groups_quality_and_preserves_structural_assignments(
    tmp_path: Path,
) -> None:
    structural = tmp_path / "structural_chimera5.jsonl"
    quality_a = tmp_path / "quality_chimera5_app.jsonl"
    quality_b = tmp_path / "quality_pegasus3_app.jsonl"
    output = tmp_path / "splits_problem_v2.json"
    problem = {
        "h": {"0": 1.0, "1": -1.0},
        "J": [[0, 1, -0.5], [1, 2, 0.25]],
        "e0": -2.0,
    }
    equivalent_problem = {
        "e0": -2,
        "J": [[2, 1, 0.25], [1, 0, -0.5]],
        "h": {"1": -1, "0": 1},
    }
    _write_records(structural, [{"instance_id": "structural-case-7"}])
    _write_records(
        quality_a,
        [{"instance_id": "chimera5-app-case-7-w", "problem": problem}],
    )
    _write_records(
        quality_b,
        [{"instance_id": "pegasus3-app-case-7-m", "problem": equivalent_problem}],
    )

    result = _run(
        [structural, quality_a, quality_b],
        output,
        "--quality-group",
        "problem-digest",
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["schema"] == "embedbench.split-manifest"
    assert manifest["schema_version"] == 2
    assert manifest["provenance"]["quality_group"] == "problem-digest"
    assert manifest["splits"][structural.name]["structural-case-7"] == _legacy_split(
        "structural-case-7"
    )
    quality_group_a = manifest["groups"][quality_a.name]["chimera5-app-case-7-w"]
    quality_group_b = manifest["groups"][quality_b.name]["pegasus3-app-case-7-m"]
    assert quality_group_a == quality_group_b
    assert quality_group_a.startswith("problem-digest:")
    assert (
        manifest["splits"][quality_a.name]["chimera5-app-case-7-w"]
        == manifest["splits"][quality_b.name]["pegasus3-app-case-7-m"]
    )


def test_manifest_bytes_are_deterministic_across_input_argument_order(tmp_path: Path) -> None:
    quality_a = tmp_path / "quality_a.jsonl"
    quality_b = tmp_path / "quality_b.jsonl"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    _write_records(
        quality_a,
        [
            {
                "instance_id": "quality-a",
                "problem": {"h": {"0": 0.0}, "J": [], "e0": 0.0},
            }
        ],
    )
    _write_records(
        quality_b,
        [
            {
                "instance_id": "quality-b",
                "problem": {"h": {"0": 1.0}, "J": [], "e0": -1.0},
            }
        ],
    )

    first = _run(
        [quality_a, quality_b],
        first_output,
        "--quality-group",
        "problem-digest",
    )
    second = _run(
        [quality_b, quality_a],
        second_output,
        "--quality-group",
        "problem-digest",
    )

    assert first.returncode == second.returncode == 0
    assert first_output.read_bytes() == second_output.read_bytes()


def test_default_mode_retains_legacy_instance_id_assignment(tmp_path: Path) -> None:
    quality = tmp_path / "quality_chimera5_app.jsonl"
    output = tmp_path / "legacy-compatible.json"
    records = [
        {
            "instance_id": "case-a",
            "problem": {"h": {"0": 0.0}, "J": [], "e0": 0.0},
        },
        {
            "instance_id": "case-b",
            "problem": {"h": {"0": 0.0}, "J": [], "e0": 0.0},
        },
    ]
    _write_records(quality, records)

    result = _run([quality], output)

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["provenance"]["quality_group"] == "instance-id"
    assert manifest["splits"][quality.name] == {
        record["instance_id"]: _legacy_split(record["instance_id"]) for record in records
    }


def test_problem_digest_mode_rejects_quality_record_without_problem(tmp_path: Path) -> None:
    quality = tmp_path / "quality_chimera5_app.jsonl"
    output = tmp_path / "unsafe.json"
    _write_records(quality, [{"instance_id": "missing-payload"}])

    result = _run(
        [quality],
        output,
        "--quality-group",
        "problem-digest",
    )

    assert result.returncode != 0
    assert "full problem payload" in result.stderr
    assert not output.exists()


def test_group_disjoint_validation_rejects_tampered_manifest() -> None:
    make_splits = _load_module()
    manifest = {
        "splits": {
            "quality_a.jsonl": {"a": "train"},
            "quality_b.jsonl": {"b": "test"},
        },
        "groups": {
            "quality_a.jsonl": {"a": "problem-digest:same"},
            "quality_b.jsonl": {"b": "problem-digest:same"},
        },
    }

    with pytest.raises(ValueError, match="split leakage.*problem-digest:same"):
        make_splits.validate_group_disjoint(manifest)

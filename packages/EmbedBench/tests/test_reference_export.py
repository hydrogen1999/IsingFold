from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from embedbench.candidate_bank import canonical_json_bytes
from embedbench.reference_export import (
    IncompleteReferenceCoverageError,
    export_problem_references,
    verify_problem_reference_export,
)


def _row(*, mode: str = "random", e0: float = -1.5) -> dict[str, object]:
    first_field = -0.75 if mode.startswith("graphcut") else -1.0
    if mode.startswith("graphcut") and e0 == -1.5:
        e0 = -1.25
    return {
        "instance_id": f"fixture-{mode}",
        "mode": mode,
        "problem": {
            "h": {"0": first_field, "1": 0.0},
            "J": [[0, 1, -0.5]],
            "e0": e0,
        },
    }


def _release(
    root: Path,
    rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    root.mkdir()
    corpus = root / "quality_fixture_random.jsonl"
    corpus.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    manifest = root / f"{corpus.name}.manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "file": f"runs/release_v1_1/{corpus.name}",
                "sha256": corpus_sha,
            }
        )
        + b"\n"
    )
    checksums = root / "SHA256SUMS"
    checksums.write_text(
        f"{corpus_sha}  {corpus.name}\n"
        f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  {manifest.name}\n",
        encoding="utf-8",
    )
    return corpus, checksums


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exact_export_matches_the_strict_release_v1_sidecar_schema(tmp_path: Path) -> None:
    corpus, checksums = _release(tmp_path / "release", [_row(), _row()])
    output = tmp_path / "references"

    manifest = export_problem_references(
        [corpus],
        checksums_path=checksums,
        output_dir=output,
    )

    references = _jsonl(output / "problem_references.jsonl")
    evidence = _jsonl(output / "problem_reference_certificates.jsonl")
    assert len(references) == len(evidence) == 1
    assert set(references[0]) == {
        "certificate_digest",
        "evaluator_protocol_digest",
        "problem_digest",
        "reference_energy",
        "reference_status",
        "schema",
        "schema_version",
    }
    assert references[0]["schema"] == "embedbench.problem-reference"
    assert references[0]["schema_version"] == 1
    assert references[0]["reference_status"] == "exact_enumeration"
    assert references[0]["reference_energy"] == -1.5
    assert references[0]["certificate_digest"] == evidence[0]["certificate_digest"]
    assert references[0]["evaluator_protocol_digest"] == manifest["protocol_sha256"]
    assert manifest["source_rows"] == 2
    assert manifest["unique_problems"] == 1
    assert manifest["certified_problems"] == 1
    assert manifest["excluded_problems"] == 0
    assert manifest["importer_complete"] is True
    assert verify_problem_reference_export(
        [corpus],
        checksums_path=checksums,
        export_dir=output,
    ) == {"certified_problems": 1, "source_rows": 2}

    augmented = (output / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"  {corpus.name}\n" in augmented
    assert "  problem_references.jsonl\n" in augmented


def test_graphcut_tabu_rows_are_never_promoted_to_certified_references(tmp_path: Path) -> None:
    corpus, checksums = _release(
        tmp_path / "release",
        [_row(), _row(mode="graphcut5x5")],
    )

    with pytest.raises(IncompleteReferenceCoverageError, match="graph-cut"):
        export_problem_references(
            [corpus],
            checksums_path=checksums,
            output_dir=tmp_path / "strict",
        )
    assert not (tmp_path / "strict").exists()

    output = tmp_path / "audit"
    manifest = export_problem_references(
        [corpus],
        checksums_path=checksums,
        output_dir=output,
        allow_incomplete=True,
    )
    references = _jsonl(output / "problem_references.jsonl")
    excluded = _jsonl(output / "problem_references.best_known.jsonl")
    assert len(references) == 1
    assert len(excluded) == 1
    assert excluded[0]["status"] == "uncertified_best_known"
    assert excluded[0]["reason"] == "legacy_graphcut_reference_was_generated_by_tabu"
    assert excluded[0]["mode"] == "graphcut5x5"
    assert manifest["importer_complete"] is False


def test_declared_energy_must_match_independent_exact_dyadic_enumeration(tmp_path: Path) -> None:
    corpus, checksums = _release(tmp_path / "release", [_row(e0=-999.0)])

    with pytest.raises(ValueError, match="disagrees with exact enumeration"):
        export_problem_references(
            [corpus],
            checksums_path=checksums,
            output_dir=tmp_path / "references",
        )
    assert not (tmp_path / "references").exists()


def test_source_corpus_and_manifest_are_both_authenticated(tmp_path: Path) -> None:
    corpus, checksums = _release(tmp_path / "release", [_row()])
    corpus.write_bytes(corpus.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_problem_references(
            [corpus],
            checksums_path=checksums,
            output_dir=tmp_path / "references",
        )


def test_verifier_recomputes_certificate_content_instead_of_trusting_its_digest(
    tmp_path: Path,
) -> None:
    corpus, checksums = _release(tmp_path / "release", [_row()])
    output = tmp_path / "references"
    export_problem_references([corpus], checksums_path=checksums, output_dir=output)
    evidence_path = output / "problem_reference_certificates.jsonl"
    evidence = _jsonl(evidence_path)
    evidence[0]["certificate"]["checked_state_count"] = 3
    evidence_path.write_bytes(canonical_json_bytes(evidence[0]) + b"\n")

    with pytest.raises(ValueError, match="certificate digest mismatch"):
        verify_problem_reference_export(
            [corpus],
            checksums_path=checksums,
            export_dir=output,
            verify_output_checksums=False,
        )

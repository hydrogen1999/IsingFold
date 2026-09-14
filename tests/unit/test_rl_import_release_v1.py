"""Contract tests for the strict EmbedBench release-v1.1 adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.import_release_v1 import (
    ReleaseV1CompatibilityError,
    _host_graph,
    _problem_split,
    prepare_release_v1,
    quality_problem_digest,
)
from isingfold.rl.data.prepared import load_prepared_tasks
from tests.unit.quality_attestation_support import (
    FIXTURE_CERTIFICATE_DIGEST,
    attest_prepared,
)


def _config() -> dict[str, Any]:
    return {
        "alpha": 0.6,
        "chain_size": 3,
        "defect_couplers": 0.0,
        "defect_qubits": 0.0,
        "degree": 3.0,
        "difficulty": "base",
        "graph": "random",
        "l_cap": 4,
        "loop_max": 8,
        "loop_min": 3,
        "max_candidates": 60,
        "max_enum": 4000,
        "max_window_free": 14,
        "min_spread": 0.05,
        "modes": ["compact", "elongated", "cut_congested", "near_capacity"],
        "n_strengths": 4,
        "n_vars": 2,
        "num_reads": 100,
        "num_sweeps": 200,
        "refine_reads": 400,
        "refine_top": 3,
        "samples_per_instance": 4,
        "size": 1,
        "source": "both",
        "topology": "chimera",
    }


def _problem() -> dict[str, Any]:
    return {"J": [[0, 1, -1.0]], "e0": -1.0, "h": {"0": 0.0, "1": 0.0}}


def _record(*, focus: int = 0) -> dict[str, Any]:
    if focus == 0:
        all_chains = {"1": [4]}
        candidates = [[0], [1]]
        frozen = {"1": [4]}
        window_nodes = [0, 1, 2, 3]
        neighbours = [1]
    else:
        all_chains = {"0": [1]}
        candidates = [[4], [5]]
        frozen = {"0": [1]}
        window_nodes = [4, 5, 6, 7]
        neighbours = [0]
    source_seed = int.from_bytes(hashlib.sha256(b"200:random:0").digest()[:4], "big")
    return {
        "Q": [1, 1],
        "all_chains": all_chains,
        "best_F": [1.0, 1.0],
        "best_index": 0,
        "candidates": candidates,
        "difficulty": "base",
        "edge_J": [-1.0],
        "focus": focus,
        "focus_h": 0.0,
        "frozen": frozen,
        "frozen_adjacency": {},
        "ground_energy": -1.0,
        "instance_id": f"chimera1-random-random-0-s{source_seed}-m",
        "j_scale": 1.0,
        "l_cap": 4,
        "margin": 0.3,
        "mode": "random",
        "n_enumerated": 2,
        "n_vars": 2,
        "neighbour_chain_size": [1],
        "neighbour_degree": [1],
        "neighbours": neighbours,
        "original_agrees": focus == 0,
        # These choices make both focus records reconstruct {0:[1], 1:[4]}.
        "original_index": 1 if focus == 0 else 0,
        "p_solve": [0.8, 0.5],
        "problem": _problem(),
        "resource_agrees": True,
        "resource_index": 0,
        "sa_seed": 17 + focus,
        "size": 1,
        "source": "minorminer",
        "spread": 0.3,
        "stage": [2, 2],
        "topology": "chimera",
        "window_edges": [],
        "window_nodes": window_nodes,
    }


def _write_release(
    root: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    include_references: bool = True,
    split_grouping: str = "problem-digest",
) -> tuple[list[Path], Path, Path, Path, Path]:
    root.mkdir()
    records = rows or [_record(), _record(focus=1)]
    corpus = root / "quality_chimera1_random.jsonl"
    corpus.write_bytes(
        b"".join(
            json.dumps(row, allow_nan=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
            + b"\n"
            for row in records
        )
    )
    manifest = {
        "config": _config(),
        "file": "runs/release_v1_1/quality_chimera1_random.jsonl",
        "jobs": 1,
        "n_instances": 1,
        "objective": "chain seam: max_F p_solve (T1) per candidate chain",
        "seed": 200,
        "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "stats": {"samples_kept": len(records)},
    }
    manifest_path = Path(str(corpus) + ".manifest.json")
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    problem_digest = quality_problem_digest(_problem())
    partition = _problem_split(problem_digest)
    source_id = records[0]["instance_id"]
    group_id = (
        f"problem-digest:{problem_digest}"
        if split_grouping == "problem-digest"
        else f"instance-id:{source_id}"
    )
    counts = {"train": 0, "val": 0, "test": 0}
    counts[partition] = len(records)
    split = {
        "group_counts": {"train": 0, "val": 0, "test": 0, partition: 1},
        "groups": {corpus.name: {source_id: group_id}},
        "provenance": {
            "assignment": {
                "hash": "sha256",
                "prefix_hex_chars": 8,
                "denominator": 2**32,
                "thresholds": {"train": 0.7, "val": 0.8, "test": 1.0},
            },
            "generator": "EmbedBench/scripts/make_splits.py",
            "inputs": [{"file": corpus.name, "sha256": manifest["sha256"]}],
            "quality_group": split_grouping,
        },
        "record_counts": {corpus.name: counts},
        "rule": "fixture",
        "schema": "embedbench.split-manifest",
        "schema_version": 2,
        "splits": {corpus.name: {source_id: partition}},
    }
    split_path = root / "splits_quality_problem_v2.json"
    split_path.write_bytes(canonical_json_bytes(split) + b"\n")

    references = root / "problem_references.jsonl"
    if include_references:
        reference = {
            "certificate_digest": FIXTURE_CERTIFICATE_DIGEST,
            "evaluator_protocol_digest": "b" * 64,
            "problem_digest": problem_digest,
            "reference_energy": -1.0,
            "reference_status": "exact_enumeration",
            "schema": "embedbench.problem-reference",
            "schema_version": 1,
        }
        references.write_bytes(canonical_json_bytes(reference) + b"\n")

    authenticated = [corpus, manifest_path]
    if include_references:
        authenticated.append(references)
    checksums = root / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in authenticated
        ),
        encoding="utf-8",
    )
    return [corpus], split_path, checksums, references, root / "prepared"


def _prepare(root: Path, **kwargs: Any) -> Path:
    corpora, split, checksums, references, output = _write_release(root, **kwargs)
    prepare_release_v1(corpora, split, checksums, references, output, qubit_cap=8)
    return output


def test_topology_reconstruction_matches_full_yield_release_hosts() -> None:
    expected = {
        ("chimera", 5): (
            200,
            560,
            "7c099bad1d9fab24766a92d49d13ba272efba517313d95b9cb84de5a3c2b5164",
        ),
        ("pegasus", 3): (
            128,
            704,
            "25c046e91298091e46f095032353a33a53a9b1b8fecc0e352a77c14bf8c7a238",
        ),
        ("zephyr", 2): (
            160,
            1224,
            "c273d082109f58ce016dac8cf94584c90417783e57a5e24dcb809f402fb3f24f",
        ),
    }
    for (topology, size), (node_count, edge_count, digest) in expected.items():
        nodes, edges = _host_graph(topology, size)
        assert (len(nodes), len(edges)) == (node_count, edge_count)
        assert nodes == sorted(set(nodes))
        assert [tuple(edge) for edge in edges] == sorted({tuple(edge) for edge in edges})
        assert content_digest({"nodes": nodes, "edges": edges}) == digest


def test_problem_digest_matches_embedbench_schema_v2_contract() -> None:
    assert (
        quality_problem_digest(_problem())
        == "4249c2ec9a9f8a2c26b6f23b258df2e757de74f2ab1cb1b49e586f01daee283d"
    )


def test_conversion_is_byte_compatible_with_prepared_loader_and_deduplicates(
    tmp_path: Path,
) -> None:
    output = _prepare(tmp_path / "release")

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {
        "evaluator_targets": 1,
        "initializers": 1,
        "policy_instances": 1,
    }
    public_bytes = (output / "policy_instances.jsonl").read_bytes() + (
        output / "initializers.jsonl"
    ).read_bytes()
    for forbidden in (b"p_solve", b"best_index", b"resource_index", b"ground_energy"):
        assert forbidden not in public_bytes

    with pytest.raises(ValueError, match="provenance-complete prepared schema v2 or v3"):
        load_prepared_tasks(output, include_evaluator=False)
    deployment = load_prepared_tasks(
        output,
        include_evaluator=False,
        require_provenance=False,
        require_corpus_design=False,
    )
    pin = attest_prepared(output, tmp_path / "trust")
    trusted = load_prepared_tasks(
        output,
        include_evaluator=True,
        require_provenance=False,
        require_corpus_design=False,
        quality_attestation_pin=pin,
    )
    assert len(deployment) == len(trusted) == 1
    assert deployment[0].task.ground_energy is None
    assert trusted[0].task.ground_energy == -1.0
    assert trusted[0].task.initial_embedding == {0: frozenset({1}), 1: frozenset({4})}


def test_missing_authenticated_reference_sidecar_fails_closed(tmp_path: Path) -> None:
    corpora, split, checksums, references, output = _write_release(
        tmp_path / "release", include_references=False
    )

    with pytest.raises(ReleaseV1CompatibilityError, match="untyped ground_energy"):
        prepare_release_v1(corpora, split, checksums, references, output, qubit_cap=8)
    assert not output.exists()


def test_original_instance_id_split_is_rejected_as_leaky(tmp_path: Path) -> None:
    corpora, split, checksums, references, output = _write_release(
        tmp_path / "release", split_grouping="instance-id"
    )

    with pytest.raises(ReleaseV1CompatibilityError, match="problem-digest grouping"):
        prepare_release_v1(corpora, split, checksums, references, output, qubit_cap=8)


def test_corpus_and_per_file_manifest_must_both_match_sha256sums(tmp_path: Path) -> None:
    corpora, split, checksums, references, output = _write_release(tmp_path / "release")
    corpus = corpora[0]
    corpus.write_bytes(corpus.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_release_v1(corpora, split, checksums, references, output, qubit_cap=8)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.__setitem__("surprise", 1), "schema differs"),
        (lambda row: row["p_solve"].__setitem__(0, float("nan")), "non-finite"),
        (
            lambda row: (
                row["all_chains"].__setitem__("1", [1]),
                row["frozen"].__setitem__("1", [1]),
            ),
            "overlap",
        ),
        (lambda row: row.__setitem__("ground_energy", -2.0), "differs from problem.e0"),
    ],
)
def test_legacy_record_schema_and_semantics_are_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    row = _record()
    mutation(row)
    corpora, split, checksums, references, output = _write_release(tmp_path / "release", rows=[row])

    with pytest.raises(ValueError, match=message):
        prepare_release_v1(corpora, split, checksums, references, output, qubit_cap=8)


def test_preparation_is_immutable(tmp_path: Path) -> None:
    root = tmp_path / "release"
    output = _prepare(root)
    corpora = [root / "quality_chimera1_random.jsonl"]

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_release_v1(
            corpora,
            root / "splits_quality_problem_v2.json",
            root / "SHA256SUMS",
            root / "problem_references.jsonl",
            output,
            qubit_cap=8,
        )

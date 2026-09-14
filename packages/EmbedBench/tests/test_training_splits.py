from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "training_splits.py"
    spec = importlib.util.spec_from_file_location("training_splits", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, instance_ids: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"instance_id": instance_id}) + "\n" for instance_id in instance_ids),
        encoding="utf-8",
    )


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_splits(path: Path, filename: str, assignments: dict[str, str]) -> None:
    path.write_text(
        json.dumps({"rule": "test rule", "splits": {filename: assignments}}),
        encoding="utf-8",
    )


def _write_v2_splits(path: Path, corpus: Path, assignments: dict[str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
                "provenance": {
                    "inputs": [
                        {
                            "file": corpus.name,
                            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                        }
                    ]
                },
                "splits": {corpus.name: assignments},
            }
        ),
        encoding="utf-8",
    )


def test_load_split_records_uses_release_assignments(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "structural.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["i-test", "i-train", "i-val", "i-train"])
    _write_splits(
        split_path,
        corpus.name,
        {"i-train": "train", "i-val": "val", "i-test": "test"},
    )

    partitions = training_splits.load_split_records([corpus], split_path)

    assert [record["instance_id"] for record in partitions.train] == ["i-train", "i-train"]
    assert [record["instance_id"] for record in partitions.val] == ["i-val"]
    assert [record["instance_id"] for record in partitions.test] == ["i-test"]


def test_file_sha256_records_exact_split_manifest(tmp_path: Path) -> None:
    training_splits = _load_module()
    split_path = tmp_path / "splits.json"
    split_path.write_bytes(b"abc")

    assert training_splits.file_sha256(split_path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_schema_v2_rejects_corpus_changed_after_manifest_creation(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "structural.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["i-train"])
    _write_v2_splits(split_path, corpus, {"i-train": "train"})
    _write_jsonl(corpus, ["i-train", "new-record"])

    with pytest.raises(ValueError, match="SHA-256 mismatch.*structural.jsonl"):
        training_splits.load_split_records([corpus], split_path)


def test_schema_v2_accepts_corpus_matching_manifest_provenance(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "structural.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["i-train", "i-val", "i-test"])
    _write_v2_splits(
        split_path,
        corpus,
        {"i-train": "train", "i-val": "val", "i-test": "test"},
    )

    partitions = training_splits.load_split_records([corpus], split_path)

    assert [record["instance_id"] for record in partitions.train] == ["i-train"]
    assert [record["instance_id"] for record in partitions.val] == ["i-val"]
    assert [record["instance_id"] for record in partitions.test] == ["i-test"]


def test_schema_v2_rejects_incomplete_input_provenance(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "structural.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["i-train"])
    _write_v2_splits(split_path, corpus, {"i-train": "train"})
    document = json.loads(split_path.read_text(encoding="utf-8"))
    document["provenance"]["inputs"] = []
    split_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance inputs and split tables"):
        training_splits.load_split_records([corpus], split_path)


def test_schema_v2_rejects_omitted_declared_corpus(tmp_path: Path) -> None:
    training_splits = _load_module()
    first = tmp_path / "quality_first.jsonl"
    second = tmp_path / "quality_second.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(first, ["first-train"])
    _write_jsonl(second, ["second-val"])
    split_path.write_text(
        json.dumps(
            {
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
                "provenance": {
                    "inputs": [
                        {
                            "file": path.name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                        for path in (first, second)
                    ]
                },
                "splits": {
                    first.name: {"first-train": "train"},
                    second.name: {"second-val": "val"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires the exact corpus input set"):
        training_splits.load_split_records([first], split_path)
    with pytest.raises(ValueError, match="requires the exact corpus input set"):
        training_splits.data_provenance([first], split_path)


def test_schema_v2_rejects_duplicate_input_provenance(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "structural.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["i-train"])
    _write_v2_splits(split_path, corpus, {"i-train": "train"})
    document = json.loads(split_path.read_text(encoding="utf-8"))
    document["provenance"]["inputs"].append(document["provenance"]["inputs"][0])
    split_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate provenance input.*structural.jsonl"):
        training_splits.load_split_records([corpus], split_path)


def test_load_split_records_limit_preserves_input_prefix(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "structural.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["i-test", "i-train", "i-val"])
    _write_splits(
        split_path,
        corpus.name,
        {"i-train": "train", "i-val": "val", "i-test": "test"},
    )

    partitions = training_splits.load_split_records([corpus], split_path, limit=2)

    assert [record["instance_id"] for record in partitions.train] == ["i-train"]
    assert partitions.val == []
    assert [record["instance_id"] for record in partitions.test] == ["i-test"]


def test_load_split_records_rejects_record_missing_from_release_map(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "quality.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["known", "unmapped"])
    _write_splits(split_path, corpus.name, {"known": "train"})

    with pytest.raises(ValueError, match="unmapped.*quality.jsonl"):
        training_splits.load_split_records([corpus], split_path)


def test_quality_source_pair_cannot_cross_release_splits(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "quality.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["problem-7-w", "problem-7-m"])
    _write_splits(
        split_path,
        corpus.name,
        {"problem-7-w": "train", "problem-7-m": "test"},
    )

    with pytest.raises(ValueError, match="split leakage.*problem-7"):
        training_splits.load_split_records(
            [corpus],
            split_path,
            group_key=training_splits.quality_problem_id,
        )


def test_quality_source_pair_stays_together_when_assignments_agree(tmp_path: Path) -> None:
    training_splits = _load_module()
    corpus = tmp_path / "quality.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["problem-7-w", "problem-7-m", "other-m"])
    _write_splits(
        split_path,
        corpus.name,
        {"problem-7-w": "val", "problem-7-m": "val", "other-m": "train"},
    )

    partitions = training_splits.load_split_records(
        [corpus],
        split_path,
        group_key=training_splits.quality_problem_id,
    )

    assert [record["instance_id"] for record in partitions.val] == [
        "problem-7-w",
        "problem-7-m",
    ]
    assert [record["instance_id"] for record in partitions.train] == ["other-m"]


def test_same_application_problem_cannot_cross_topology_splits(tmp_path: Path) -> None:
    training_splits = _load_module()
    chimera = tmp_path / "quality_chimera5_app.jsonl"
    pegasus = tmp_path / "quality_pegasus3_app.jsonl"
    split_path = tmp_path / "splits.json"
    chimera_id = "chimera5-app-portfolio16-0-s123-m"
    pegasus_id = "pegasus3-app-portfolio16-0-s123-m"
    _write_jsonl(chimera, [chimera_id])
    _write_jsonl(pegasus, [pegasus_id])
    split_path.write_text(
        json.dumps(
            {
                "splits": {
                    chimera.name: {chimera_id: "train"},
                    pegasus.name: {pegasus_id: "test"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split leakage.*app-portfolio16-0-s123"):
        training_splits.load_split_records(
            [chimera, pegasus],
            split_path,
            group_key=training_splits.quality_problem_id,
        )


def test_problem_digest_is_invariant_to_mapping_edge_order_and_orientation() -> None:
    training_splits = _load_module()
    first = {
        "problem": {
            "h": {"0": 1, "1": -1.0},
            "J": [[0, 1, -0.5], [1, 2, 0.25]],
            "e0": -2,
        }
    }
    reordered = {
        "problem": {
            "e0": -2.0,
            "J": [[2, 1, 0.25], [1, 0, -0.5]],
            "h": {"1": -1, "0": 1.0},
        }
    }

    assert training_splits.quality_problem_digest(first) == (
        training_splits.quality_problem_digest(reordered)
    )


@pytest.mark.parametrize(
    "changed_problem",
    [
        {"h": {"0": 0.5}, "J": [[0, 1, -0.5]], "e0": -1.5},
        {"h": {"0": 1.0}, "J": [[0, 1, 0.5]], "e0": -1.5},
        {"h": {"0": 1.0}, "J": [[0, 1, -0.5]], "e0": -2.0},
    ],
)
def test_problem_digest_commits_to_every_ising_field(changed_problem: dict) -> None:
    training_splits = _load_module()
    original = {"problem": {"h": {"0": 1.0}, "J": [[0, 1, -0.5]], "e0": -1.5}}

    assert training_splits.quality_problem_digest(original) != (
        training_splits.quality_problem_digest({"problem": changed_problem})
    )


def test_problem_digest_rejects_cross_topology_leakage_even_with_unrelated_ids(
    tmp_path: Path,
) -> None:
    training_splits = _load_module()
    chimera = tmp_path / "quality_chimera5_app.jsonl"
    pegasus = tmp_path / "quality_pegasus3_app.jsonl"
    split_path = tmp_path / "splits.json"
    problem = {"h": {"0": 1.0}, "J": [[0, 1, -0.5]], "e0": -1.5}
    _write_records(chimera, [{"instance_id": "unrelated-a", "problem": problem}])
    _write_records(pegasus, [{"instance_id": "unrelated-b", "problem": problem}])
    split_path.write_text(
        json.dumps(
            {
                "splits": {
                    chimera.name: {"unrelated-a": "train"},
                    pegasus.name: {"unrelated-b": "test"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split leakage.*problem payload"):
        training_splits.load_split_records(
            [chimera, pegasus],
            split_path,
            record_group_key=training_splits.quality_problem_digest,
            limit=1,
        )


@pytest.mark.parametrize("graph", ["app", "random"])
def test_quality_problem_id_groups_shared_problems_across_topologies(graph: str) -> None:
    training_splits = _load_module()

    chimera = training_splits.quality_problem_id(f"chimera5-{graph}-case-0-s123-m")
    pegasus = training_splits.quality_problem_id(f"pegasus3-{graph}-case-0-s123-m")

    assert chimera == pegasus


@pytest.mark.parametrize(
    "instance_id",
    ["plain", "problem-worker", "problem-mm", "chimera5-inkdrop-case-0-s123-m"],
)
def test_quality_problem_id_only_strips_known_source_suffix(instance_id: str) -> None:
    training_splits = _load_module()

    expected = instance_id[:-2] if instance_id.endswith("-m") else instance_id
    assert training_splits.quality_problem_id(instance_id) == expected


@pytest.mark.parametrize("script", ["train_structural.py", "train_chain.py"])
def test_training_cli_exposes_fixed_release_splits(script: str) -> None:
    root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--splits" in result.stdout
    assert "--evaluate-test" in result.stdout


def test_chain_training_cli_rejects_source_pair_leakage(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    corpus = tmp_path / "quality.jsonl"
    split_path = tmp_path / "splits.json"
    _write_jsonl(corpus, ["problem-7-w", "problem-7-m"])
    _write_splits(
        split_path,
        corpus.name,
        {"problem-7-w": "train", "problem-7-m": "test"},
    )

    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "train_chain.py"),
            str(corpus),
            "--splits",
            str(split_path),
            "--out",
            str(tmp_path / "result.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "split leakage" in result.stderr


def test_fixed_split_grid_keeps_structural_test_metrics_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    trainer = _load_script("train_structural.py")
    output = tmp_path / "result.json"
    train_records = [{"instance_id": "train"}]
    val_records = [{"instance_id": "val"}]
    test_records = [{"instance_id": "test"}]

    class FakeModel:
        def parameters(self):
            return []

    evaluated = []
    monkeypatch.setattr(
        trainer,
        "load_split_records",
        lambda *args, **kwargs: (train_records, val_records, test_records),
    )
    monkeypatch.setattr(
        trainer,
        "data_provenance",
        lambda *args, **kwargs: {
            "split_manifest": {"sha256": "digest"},
            "corpus_inputs": [],
        },
    )
    monkeypatch.setattr(trainer, "encode", lambda record: record)
    monkeypatch.setattr(trainer, "build_arch", lambda *args: FakeModel())
    monkeypatch.setattr(trainer, "train", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda model, records, **kwargs: evaluated.append(records) or {"n": len(records)},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_structural.py",
            "corpus.jsonl",
            "--splits",
            "splits.json",
            "--out",
            str(output),
        ],
    )

    trainer.main()

    result = json.loads(output.read_text(encoding="utf-8"))["results"][0]
    assert result["validation"] == {"n": 1}
    assert result["test_evaluated"] is False
    assert result["test"] is None
    assert evaluated == [val_records]

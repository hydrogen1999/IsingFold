from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path

import pytest

from isingfold.rl import cli
from isingfold.rl.data.import_embedbench import canonical_json_bytes
from tests.unit.test_rl_quality_integrity import (
    _make_quality_artifact,
    _make_quality_shards,
    _merge_args,
    _pin,
)


def _spawn_round_trip_selection(selection: cli._QualityJsonlSelection):
    return selection


def _forbid_records_read_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_open = Path.open

    def guarded(path: Path) -> bytes:
        if path.name == "records.jsonl":
            raise AssertionError("quality JSONL must be streamed, not materialized with read_bytes")
        return original_read_bytes(path)

    def guarded_text(path: Path, *args, **kwargs) -> str:
        if path.name == "records.jsonl":
            raise AssertionError("quality JSONL must not be materialized with read_text")
        return original_read_text(path, *args, **kwargs)

    class StreamingReadGuard:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.handle)

        def read(self, size: int = -1):
            if size < 0:
                raise AssertionError("quality JSONL reads must have an explicit bounded size")
            return self.handle.read(size)

        def readlines(self, *_args, **_kwargs):
            raise AssertionError("quality JSONL must not be materialized with readlines")

        def __getattr__(self, name: str):
            return getattr(self.handle, name)

    def guarded_open(path: Path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if path.name == "records.jsonl" and "r" in mode:
            return StreamingReadGuard(handle)
        return handle

    monkeypatch.setattr(Path, "read_bytes", guarded)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "open", guarded_open)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("rb") as handle:
        return [json.loads(line) for line in handle]


def _read_streamed(path: Path) -> bytes:
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            chunks.append(chunk)
    return b"".join(chunks)


def test_quality_loader_never_materializes_records_jsonl_with_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, labels, bundle, context = _make_quality_artifact(tmp_path, monkeypatch)
    _forbid_records_read_bytes(monkeypatch)

    records, loaded = cli._load_quality_labels(
        labels,
        corpus=corpus,
        selector=bundle,
        context=context,
        quality_attestation_pin=_pin(tmp_path),
        allow_diagnostic_legacy=True,
    )

    assert len(records) == loaded["record_count"] == 1


def test_quality_merge_streams_input_and_preserves_canonical_output_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, shards, _bundle, _context = _make_quality_shards(
        tmp_path,
        monkeypatch,
        lineage_count=6,
        shard_count=3,
    )
    source_rows = [
        row
        for shard in shards
        for row in _read_jsonl(shard / "records.jsonl")
    ]
    source_rows.sort(
        key=lambda row: (
            str(row["lineage"]),
            str(row["task_id"]),
            tuple(row["prefix"]),
            str(row["state_fingerprint"]),
        )
    )
    expected = b"".join(canonical_json_bytes(row) + b"\n" for row in source_rows)
    _forbid_records_read_bytes(monkeypatch)

    merged = tmp_path / "quality-merged-streaming"
    args = _merge_args(corpus, list(reversed(shards)), merged)
    args.func(args)

    actual = _read_streamed(merged / "records.jsonl")
    manifest = json.loads((merged / "manifest.json").read_text())
    assert actual == expected
    assert manifest["record_count"] == len(source_rows)
    assert manifest["records_sha256"] == hashlib.sha256(expected).hexdigest()


def test_preflight_worker_selection_reads_only_its_assigned_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, shards, bundle, context = _make_quality_shards(
        tmp_path,
        monkeypatch,
        lineage_count=4,
        shard_count=2,
    )
    merged = tmp_path / "quality-merged-for-selection"
    merge_args = _merge_args(corpus, shards, merged)
    merge_args.func(merge_args)
    manifest = json.loads((merged / "manifest.json").read_text())
    _forbid_records_read_bytes(monkeypatch)
    record_index = cli._build_quality_jsonl_index(merged, manifest)
    selected_lineage = sorted(record_index.lineages)[0]
    selected = record_index.select(frozenset({selected_lineage}))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        assert executor.submit(_spawn_round_trip_selection, selected).result() == selected

    from isingfold.rl.data import quality

    original_replay = quality.run_continuation
    replayed_lineages: list[str] = []
    implementation_contract = manifest["label_protocol"]["implementation_contract"]

    def traced_replay(task, *args, **kwargs):
        replayed_lineages.append(task.lineage)
        return original_replay(task, *args, **kwargs)

    monkeypatch.setattr(quality, "run_continuation", traced_replay)
    monkeypatch.setattr(
        cli,
        "_quality_implementation_contract",
        lambda _context: implementation_contract,
    )
    records, loaded = cli._load_quality_labels(
        merged,
        corpus=corpus,
        selector=bundle,
        context=context,
        enforce_resolution=False,
        quality_attestation_pin=_pin(tmp_path),
        _replay_lineages=frozenset({selected_lineage}),
        _record_selection=selected,
        _collect_decoded=False,
        allow_diagnostic_legacy=True,
    )

    assert records == []
    assert loaded["loader_summary"]["label_lineages"] == 1
    assert set(replayed_lineages) == {selected_lineage}


def test_authenticated_range_selection_fails_closed_after_file_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _corpus, labels, _bundle, _context = _make_quality_artifact(tmp_path, monkeypatch)
    manifest = json.loads((labels / "manifest.json").read_text())
    record_index = cli._build_quality_jsonl_index(labels, manifest)
    records_path = labels / "records.jsonl"
    with records_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"[")

    with pytest.raises(ValueError, match="changed after authentication"):
        list(cli._iter_quality_jsonl_selection(labels, record_index.select()))

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import embedbench.hard_ood_provenance as provenance
import pytest
from embedbench.hard_ood_provenance import (
    SOURCE_BUNDLE_SCHEMA,
    SourceBundleSnapshot,
    parse_canonical_json_bytes,
    snapshot_source_bundle,
    validate_source_bundle,
    verify_source_bundle,
)
from embedbench.hard_ood_schema import canonical_bytes


def _manifest(
    files: dict[str, bytes],
    *,
    role: str = "generation",
    executable_paths: frozenset[str] = frozenset(),
) -> bytes:
    return canonical_bytes(
        {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "schema_version": 1,
            "release_id": "embedbench-hard-ood-v1.0.0",
            "role": role,
            "files": [
                {
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                    "executable": relative_path in executable_paths,
                }
                for relative_path, payload in sorted(files.items())
            ],
        }
    )


def _write_bundle(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, payload in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def test_source_bundle_authenticates_and_detaches_file_bytes(tmp_path: Path) -> None:
    files = {"checker/main.py": b"print('ok')\n", "protocol.json": b"{}"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)
    verified = verify_source_bundle(
        tmp_path,
        manifest,
        expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        expected_role="generation",
    )

    (tmp_path / "checker/main.py").write_bytes(b"changed after verification\n")

    assert (
        validate_source_bundle(
            verified,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
            expected_release_id="embedbench-hard-ood-v1.0.0",
        )
        is verified
    )
    assert (
        verified.file_bytes(
            "checker/main.py",
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
            expected_release_id="embedbench-hard-ood-v1.0.0",
        )
        == b"print('ok')\n"
    )
    assert verified.release_id == "embedbench-hard-ood-v1.0.0"
    with pytest.raises(KeyError):
        verified.file_bytes(
            "missing.py",
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
            expected_release_id="embedbench-hard-ood-v1.0.0",
        )


def test_source_bundle_rejects_file_mutation_and_role_substitution(tmp_path: Path) -> None:
    files = {"main.py": b"original\n"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)
    (tmp_path / "main.py").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="byte count mismatch|digest mismatch"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )

    (tmp_path / "main.py").write_bytes(files["main.py"])
    with pytest.raises(ValueError, match="role"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="continuation_checker",
        )


def test_source_bundle_rejects_symlinks_and_path_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "link.py").symlink_to(outside)
    files = {"link.py": outside.read_bytes()}
    manifest = _manifest(files)

    with pytest.raises(OSError):
        verify_source_bundle(
            bundle_root,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )

    escaped = json.loads(manifest)
    escaped["files"][0]["relative_path"] = "../outside.py"
    escaped_manifest = canonical_bytes(escaped)
    with pytest.raises(ValueError, match="normalized relative"):
        verify_source_bundle(
            bundle_root,
            escaped_manifest,
            expected_manifest_sha256=hashlib.sha256(escaped_manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_rejects_symlink_in_root_ancestry(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "main.py").write_bytes(b"pass\n")
    link = tmp_path / "linked"
    link.symlink_to(actual, target_is_directory=True)
    manifest = _manifest({"main.py": b"pass\n"})

    with pytest.raises(OSError):
        verify_source_bundle(
            link,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_rejects_undeclared_files_and_special_entries(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_bytes(b"pass\n")
    (tmp_path / "undeclared.py").write_bytes(b"raise SystemExit\n")
    manifest = _manifest({"main.py": b"pass\n"})

    with pytest.raises(ValueError, match="undeclared file"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_rejects_undeclared_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_bytes(b"pass\n")
    (tmp_path / "undeclared").mkdir()
    manifest = _manifest({"main.py": b"pass\n"})

    with pytest.raises(ValueError, match="undeclared directory"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_rejects_unexpected_file_before_reading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "main.py").write_bytes(b"pass\n")
    (tmp_path / "unexpected.bin").write_bytes(b"attacker-controlled")
    manifest = _manifest({"main.py": b"pass\n"})
    original_reader = provenance._read_open_regular_file

    def guarded_reader(file_fd, metadata, *, expected_byte_count):
        if metadata.st_size == len(b"attacker-controlled"):
            raise AssertionError("unexpected file must be rejected before reading")
        return original_reader(
            file_fd,
            metadata,
            expected_byte_count=expected_byte_count,
        )

    monkeypatch.setattr(provenance, "_read_open_regular_file", guarded_reader)
    with pytest.raises(ValueError, match="undeclared file"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_rejects_declared_size_mismatch_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "main.py").write_bytes(b"attacker-controlled replacement")
    manifest = _manifest({"main.py": b"pass\n"})

    def forbidden_read(_file_descriptor: int, _byte_count: int) -> bytes:
        raise AssertionError("size-mismatched declared file must not be read")

    monkeypatch.setattr(provenance.os, "read", forbidden_read)
    with pytest.raises(ValueError, match="byte count mismatch before file read"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_growing_file_is_read_only_to_declared_count_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"pass\n"
    (tmp_path / "main.py").write_bytes(payload)
    manifest = _manifest({"main.py": payload})
    read_requests: list[int] = []

    def endlessly_growing_read(_file_descriptor: int, byte_count: int) -> bytes:
        read_requests.append(byte_count)
        if sum(read_requests) > len(payload) + 1:
            raise AssertionError("source capture attempted to read an unbounded growing file")
        return b"x" * byte_count

    monkeypatch.setattr(provenance.os, "read", endlessly_growing_read)
    with pytest.raises(ValueError, match="declared byte count"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )

    assert read_requests == [len(payload) + 1]


def test_source_bundle_fails_closed_without_secure_posix_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest({"main.py": b"pass\n"})
    monkeypatch.setattr(provenance, "_SECURE_IO_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="O_NOFOLLOW"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_uses_streaming_scandir_not_listdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"main.py": b"pass\n"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)

    def forbidden_listdir(_path):
        raise AssertionError("verification must not materialize an untrusted directory listing")

    monkeypatch.setattr(provenance.os, "listdir", forbidden_listdir)
    verified = verify_source_bundle(
        tmp_path,
        manifest,
        expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        expected_role="generation",
    )

    assert verified.manifest_sha256 == hashlib.sha256(manifest).hexdigest()


def test_source_bundle_rejects_manifest_file_count_above_absolute_limit(tmp_path: Path) -> None:
    entry = {
        "relative_path": "main.py",
        "sha256": hashlib.sha256(b"pass\n").hexdigest(),
        "byte_count": 5,
        "executable": False,
    }
    manifest = canonical_bytes(
        {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "schema_version": 1,
            "release_id": "embedbench-hard-ood-v1.0.0",
            "role": "generation",
            "files": [entry] * (provenance.MAX_SOURCE_BUNDLE_FILES + 1),
        }
    )

    with pytest.raises(ValueError, match="file count"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_consumer_replays_manifest_file_count_limit(tmp_path: Path) -> None:
    files = {"main.py": b"pass\n"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)
    verified = verify_source_bundle(
        tmp_path,
        manifest,
        expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        expected_role="generation",
    )
    oversized_manifest = canonical_bytes(
        {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "schema_version": 1,
            "release_id": "embedbench-hard-ood-v1.0.0",
            "role": "generation",
            "files": [
                {
                    "relative_path": f"f{index:05d}.py",
                    "sha256": hashlib.sha256(b"").hexdigest(),
                    "byte_count": 0,
                    "executable": False,
                }
                for index in range(provenance.MAX_SOURCE_BUNDLE_FILES + 1)
            ],
        }
    )
    object.__setattr__(verified, "_manifest_payload", oversized_manifest)
    object.__setattr__(
        verified,
        "manifest_sha256",
        hashlib.sha256(oversized_manifest).hexdigest(),
    )

    with pytest.raises(ValueError, match="file count"):
        validate_source_bundle(
            verified,
            expected_manifest_sha256=hashlib.sha256(oversized_manifest).hexdigest(),
            expected_role="generation",
            expected_release_id="embedbench-hard-ood-v1.0.0",
        )


@pytest.mark.parametrize("unsafe_path", ["C:payload.py", "line\nbreak.py", "safe\u202eevil.py"])
def test_source_bundle_rejects_nonportable_or_invisible_paths(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    document = json.loads(_manifest({"main.py": b"pass\n"}))
    document["files"][0]["relative_path"] = unsafe_path
    payload = canonical_bytes(document)

    with pytest.raises(ValueError, match="portable ASCII"):
        verify_source_bundle(
            tmp_path,
            payload,
            expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_binds_executable_intent(tmp_path: Path) -> None:
    executable = tmp_path / "tool"
    executable.write_bytes(b"binary")
    executable.chmod(0o755)
    manifest = _manifest(
        {"tool": b"binary"},
        role="portable_runtime",
        executable_paths=frozenset({"tool"}),
    )
    verified = verify_source_bundle(
        tmp_path,
        manifest,
        expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        expected_role="portable_runtime",
    )

    assert (
        verified.file_entry(
            "tool",
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="portable_runtime",
            expected_release_id="embedbench-hard-ood-v1.0.0",
        ).executable
        is True
    )

    executable.chmod(0o644)
    with pytest.raises(ValueError, match="executable intent"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="portable_runtime",
        )


def test_source_bundle_validation_rejects_mutated_verified_fields(tmp_path: Path) -> None:
    files = {"main.py": b"pass\n"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)
    verified = verify_source_bundle(
        tmp_path,
        manifest,
        expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        expected_role="generation",
    )

    object.__setattr__(verified, "role", "continuation_checker")
    with pytest.raises(ValueError, match="verified source bundle fields"):
        validate_source_bundle(
            verified,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
            expected_release_id="embedbench-hard-ood-v1.0.0",
        )


def test_source_bundle_whole_capsule_rebase_cannot_replace_external_commitment(
    tmp_path: Path,
) -> None:
    honest_root = tmp_path / "honest"
    attacker_root = tmp_path / "attacker"
    honest_files = {"main.py": b"honest\n"}
    attacker_files = {"main.py": b"attacker\n"}
    _write_bundle(honest_root, honest_files)
    _write_bundle(attacker_root, attacker_files)
    honest_manifest = _manifest(honest_files)
    attacker_manifest = _manifest(attacker_files)
    honest_digest = hashlib.sha256(honest_manifest).hexdigest()
    honest = verify_source_bundle(
        honest_root,
        honest_manifest,
        expected_manifest_sha256=honest_digest,
        expected_role="generation",
    )
    attacker = verify_source_bundle(
        attacker_root,
        attacker_manifest,
        expected_manifest_sha256=hashlib.sha256(attacker_manifest).hexdigest(),
        expected_role="generation",
    )

    for field in ("release_id", "role", "manifest_sha256", "files", "_manifest_payload"):
        object.__setattr__(honest, field, getattr(attacker, field))

    with pytest.raises(ValueError, match="external commitment"):
        honest.file_bytes(
            "main.py",
            expected_manifest_sha256=honest_digest,
            expected_role="generation",
            expected_release_id="embedbench-hard-ood-v1.0.0",
        )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":1}',
        b'{"value":NaN}',
        b'{ "a":1}',
        b'{"a":1}\n',
        b"[]",
    ],
)
def test_canonical_json_bytes_reject_noncanonical_or_ambiguous_inputs(payload: bytes) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_canonical_json_bytes(payload, name="fixture")


def test_canonical_json_parser_inverts_registered_binary64_encoding() -> None:
    payload = canonical_bytes({"nested": {"value": 0.5}, "negative_zero": -0.0})

    assert parse_canonical_json_bytes(payload, name="fixture") == {
        "nested": {"value": 0.5},
        "negative_zero": 0.0,
    }


@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_source_bundle_rejects_non_integer_schema_version(
    tmp_path: Path,
    invalid_version: object,
) -> None:
    files = {"main.py": b"pass\n"}
    _write_bundle(tmp_path, files)
    document = json.loads(_manifest(files))
    document["schema_version"] = invalid_version
    manifest = canonical_bytes(document)

    with pytest.raises(ValueError):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_requires_independent_manifest_digest(tmp_path: Path) -> None:
    files = {"main.py": b"pass\n"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)

    with pytest.raises(ValueError, match="external commitment"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256="0" * 64,
            expected_role="generation",
        )


def test_source_bundle_rejects_unknown_fields_and_unsorted_entries(tmp_path: Path) -> None:
    files = {"a.py": b"a", "b.py": b"b"}
    _write_bundle(tmp_path, files)
    document = json.loads(_manifest(files))
    document["unknown"] = None
    unknown = canonical_bytes(document)
    with pytest.raises(ValueError, match="schema fields differ"):
        verify_source_bundle(
            tmp_path,
            unknown,
            expected_manifest_sha256=hashlib.sha256(unknown).hexdigest(),
            expected_role="generation",
        )

    del document["unknown"]
    document["files"].reverse()
    unsorted = canonical_bytes(document)
    with pytest.raises(ValueError, match="sorted"):
        verify_source_bundle(
            tmp_path,
            unsorted,
            expected_manifest_sha256=hashlib.sha256(unsorted).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_rejects_aggregate_payload_before_file_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"main.py": b"12345"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)
    monkeypatch.setattr(provenance, "MAX_SOURCE_BUNDLE_TOTAL_BYTES", 4)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("over-limit aggregate reached bundle filesystem IO")

    monkeypatch.setattr(provenance, "_open_root_without_symlinks", forbidden_open)
    with pytest.raises(ValueError, match="aggregate payload"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_rejects_manifest_path_depth_before_file_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"a/b/main.py": b"pass\n"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)
    monkeypatch.setattr(provenance, "MAX_SOURCE_BUNDLE_PATH_DEPTH", 2)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("over-depth path reached bundle filesystem IO")

    monkeypatch.setattr(provenance, "_open_root_without_symlinks", forbidden_open)
    with pytest.raises(ValueError, match="path depth"):
        verify_source_bundle(
            tmp_path,
            manifest,
            expected_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            expected_role="generation",
        )


def test_source_bundle_snapshot_recomputes_manifest_and_detaches_records(
    tmp_path: Path,
) -> None:
    files = {"main.py": b"honest\n"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    verified = verify_source_bundle(
        tmp_path,
        manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_role="generation",
    )

    snapshot = snapshot_source_bundle(
        verified,
        expected_manifest_sha256=manifest_sha256,
        expected_role="generation",
        expected_release_id="embedbench-hard-ood-v1.0.0",
        maximum_total_payload_bytes=len(files["main.py"]),
        maximum_path_depth=1,
    )
    assert snapshot.manifest_bytes == manifest
    assert snapshot.total_payload_bytes == len(files["main.py"])
    assert snapshot.files is not verified.files
    assert snapshot.files[0] is not verified.files[0]

    object.__setattr__(verified.files[0], "payload", b"attacker\n")
    assert snapshot.file_entry("main.py").payload == b"honest\n"
    with pytest.raises(TypeError):
        SourceBundleSnapshot()


def test_source_bundle_snapshot_rejects_check_use_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {"main.py": b"honest\n"}
    _write_bundle(tmp_path, files)
    manifest = _manifest(files)
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    verified = verify_source_bundle(
        tmp_path,
        manifest,
        expected_manifest_sha256=manifest_sha256,
        expected_role="generation",
    )
    real_identity = provenance._source_bundle_identity
    identity_calls = 0

    def mutate_on_final_identity(value):
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 2:
            object.__setattr__(value.files[0], "payload", b"attacker\n")
        return real_identity(value)

    monkeypatch.setattr(provenance, "_source_bundle_identity", mutate_on_final_identity)
    with pytest.raises(ValueError, match="changed while"):
        snapshot_source_bundle(
            verified,
            expected_manifest_sha256=manifest_sha256,
            expected_role="generation",
            expected_release_id="embedbench-hard-ood-v1.0.0",
        )

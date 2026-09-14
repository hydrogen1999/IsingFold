"""Create and verify the Apollo/Goose same-shard byte-parity attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_corpus_io import _load_corpus_shard
from embedbench.isingfold_corpus_plan import CorpusGenerationPlan, read_corpus_plan
from embedbench.isingfold_corpus_shard import ShardConfig, lineage_shard

_FILES = ("SHA256SUMS", "corpus_shard.json", "shard_manifest.json")
_AUTHENTICATED_FILES = ("corpus_shard.json", "shard_manifest.json")
_HEX = frozenset("0123456789abcdef")
_THREAD_CONTROLS = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_SCHEMA = "embedbench.isingfold-cross-site-canary-attestation"
_VERSION = 1
_FIELDS = {
    "commitments",
    "files",
    "record_digest",
    "schema",
    "schema_version",
    "shard_count",
    "shard_index",
    "shard_manifest_record_digest",
    "shard_record_digest",
}
_COMMITMENT_FIELDS = {
    "generation_provenance_sha256",
    "installation_sha256",
    "plan_sha256",
    "plan_record_digest",
    "source_sha256",
    "thread_controls",
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _exact_object(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with text keys")
    if set(value) != fields:
        raise ValueError(f"{name} fields differ from the registered schema")
    return value


def _strict_json(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} must be canonical JSON with one terminal line feed")
    return value


def _read_regular(file_name: Path, name: str) -> bytes:
    try:
        before = file_name.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} must be a regular file, not a link")
        raw = file_name.read_bytes()
        after = file_name.lstat()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {error}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(raw) != before.st_size:
        raise ValueError(f"{name} changed while it was being read")
    return raw


def _canonical_file(file_name: Path, name: str) -> Path:
    if not file_name.is_absolute() or file_name.is_symlink() or not file_name.is_file():
        raise ValueError(f"{name} must be an absolute regular non-symlink file")
    resolved = file_name.resolve(strict=True)
    if resolved != file_name:
        raise ValueError(f"{name} must use a canonical physical path")
    return resolved


def _canonical_directory(directory: Path, name: str) -> Path:
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{name} must be an absolute real directory")
    resolved = directory.resolve(strict=True)
    if resolved != directory:
        raise ValueError(f"{name} must use a canonical physical path")
    return resolved


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_record(document: dict[str, Any], name: str) -> str:
    recorded = _require_sha256(document.get("record_digest"), f"{name} record digest")
    payload = {key: value for key, value in document.items() if key != "record_digest"}
    if content_digest(payload) != recorded:
        raise ValueError(f"{name} record digest mismatch")
    return recorded


def _expected_config(
    plan: CorpusGenerationPlan,
    shard_index: int,
) -> ShardConfig:
    screening = plan.screening
    return ShardConfig(
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


def _load_shard(
    directory: Path,
    expected_provenance_sha256: str,
    plan: CorpusGenerationPlan,
) -> dict[str, Any]:
    directory = _canonical_directory(directory, "canary shard")
    entries = {entry.name: entry for entry in directory.iterdir()}
    if set(entries) != set(_FILES):
        raise ValueError("canary shard directory inventory differs from the registered files")
    raw = {name: _read_regular(entries[name], f"canary {name}") for name in _FILES}
    expected_sums = "".join(
        f"{_sha256(raw[name])}  {name}\n" for name in sorted(_AUTHENTICATED_FILES)
    ).encode("utf-8")
    if raw["SHA256SUMS"] != expected_sums:
        raise ValueError("canary SHA256SUMS is noncanonical or does not authenticate the shard")

    loaded = _load_corpus_shard(directory)
    shard = loaded.shard
    provenance = dict(shard.provenance)
    provenance_digest = shard.provenance_digest
    if provenance_digest != expected_provenance_sha256:
        raise ValueError("canary generation provenance differs from its external commitment")
    if provenance.get("protocol") != "embedbench.isingfold-corpus-generator-v4":
        raise ValueError("canary generator protocol is not supported")
    if provenance.get("thread_controls") != _THREAD_CONTROLS:
        raise ValueError("canary thread controls differ from the runtime contract")

    shard_index = shard.shard_index
    shard_count = shard.shard_count
    if shard_count != plan.shard_count or shard.config != _expected_config(plan, shard_index):
        raise ValueError("canary shard generation config differs from the authenticated plan")
    expected_requests = tuple(
        asdict(item.request)
        for item in plan.lineages
        if lineage_shard(item.request.lineage_id, plan.shard_count) == shard_index
    )
    observed_requests = tuple(asdict(item.request) for item in shard.lineages)
    if observed_requests != expected_requests:
        raise ValueError("canary shard requests differ from its exact plan-assigned subset")
    return {
        "files": {
            name: {"byte_count": len(raw[name]), "sha256": _sha256(raw[name])}
            for name in _FILES
        },
        "lineage_count": len(shard.lineages),
        "manifest_record_digest": loaded.manifest_record_digest,
        "raw": raw,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "shard_record_digest": shard.record_digest,
    }


def _commitments(
    *,
    plan_sha256: str,
    source_sha256: str,
    generation_provenance_sha256: str,
    installation_sha256: str,
    plan_record_digest: str,
) -> dict[str, object]:
    return {
        "generation_provenance_sha256": _require_sha256(
            generation_provenance_sha256, "generation provenance SHA-256"
        ),
        "installation_sha256": _require_sha256(
            installation_sha256, "installation SHA-256"
        ),
        "plan_sha256": _require_sha256(plan_sha256, "plan SHA-256"),
        "plan_record_digest": _require_sha256(
            plan_record_digest, "plan record digest"
        ),
        "source_sha256": _require_sha256(source_sha256, "source SHA-256"),
        "thread_controls": dict(_THREAD_CONTROLS),
    }


def publish_parity_attestation(
    *,
    apollo_shard: Path,
    goose_shard: Path,
    plan: Path,
    output: Path,
    plan_sha256: str,
    source_sha256: str,
    generation_provenance_sha256: str,
    installation_sha256: str,
) -> dict[str, Any]:
    """Compare two canary shards and publish one immutable parity attestation."""

    expected_provenance = _require_sha256(
        generation_provenance_sha256, "generation provenance SHA-256"
    )
    plan = _canonical_file(plan, "canary plan")
    plan_document = read_corpus_plan(plan, expected_sha256=plan_sha256)
    apollo = _load_shard(apollo_shard, expected_provenance, plan_document)
    goose = _load_shard(goose_shard, expected_provenance, plan_document)
    if (apollo["shard_index"], apollo["shard_count"]) != (
        goose["shard_index"],
        goose["shard_count"],
    ):
        raise ValueError("Apollo and Goose canaries are not the same scientific shard")
    for name in _FILES:
        if apollo["raw"][name] != goose["raw"][name]:
            raise ValueError(f"Apollo and Goose canary bytes differ for {name}")

    payload = {
        "commitments": _commitments(
            plan_sha256=plan_sha256,
            source_sha256=source_sha256,
            generation_provenance_sha256=generation_provenance_sha256,
            installation_sha256=installation_sha256,
            plan_record_digest=plan_document.record_digest,
        ),
        "files": apollo["files"],
        "schema": _SCHEMA,
        "schema_version": _VERSION,
        "shard_count": apollo["shard_count"],
        "shard_index": apollo["shard_index"],
        "shard_manifest_record_digest": apollo["manifest_record_digest"],
        "shard_record_digest": apollo["shard_record_digest"],
    }
    document = {**payload, "record_digest": content_digest(payload)}
    raw = canonical_json_bytes(document) + b"\n"
    output_parent = _canonical_directory(output.parent, "attestation output parent")
    if not output.is_absolute() or output != output_parent / output.name:
        raise ValueError("attestation output must use a canonical absolute path")
    claim = output.parent / f".{output.name}.claim"
    if output.exists() or output.is_symlink() or claim.exists() or claim.is_symlink():
        raise FileExistsError("refusing to overwrite a canary parity attestation or claim")
    claim.mkdir()
    _fsync_directory(output_parent)
    partial = claim / "attestation.partial"
    with partial.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, output)
    _fsync_directory(output_parent)
    os.unlink(partial)
    with (claim / "SUCCESS").open("xb") as stream:
        stream.write((_sha256(raw) + "\n").encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(claim)
    _fsync_directory(output_parent)
    return document


def verify_parity_attestation(
    *,
    attestation: Path,
    expected_attestation_sha256: str,
    plan_sha256: str,
    source_sha256: str,
    generation_provenance_sha256: str,
    installation_sha256: str,
    plan: Path,
) -> dict[str, Any]:
    """Verify an out-of-band-pinned parity attestation for production launch."""

    attestation = _canonical_file(attestation, "canary parity attestation")
    plan = _canonical_file(plan, "canary plan")
    plan_document = read_corpus_plan(plan, expected_sha256=plan_sha256)
    raw = _read_regular(attestation, "canary parity attestation")
    if _sha256(raw) != _require_sha256(
        expected_attestation_sha256, "canary parity attestation SHA-256"
    ):
        raise ValueError("canary parity attestation differs from its external commitment")
    document = _exact_object(_strict_json(raw, "canary parity attestation"), _FIELDS, "attestation")
    if document["schema"] != _SCHEMA or document["schema_version"] != _VERSION:
        raise ValueError("canary parity attestation schema is not supported")
    _verify_record(document, "canary parity attestation")
    commitments = _exact_object(
        document["commitments"], _COMMITMENT_FIELDS, "attestation commitments"
    )
    expected = _commitments(
        plan_sha256=plan_sha256,
        source_sha256=source_sha256,
        generation_provenance_sha256=generation_provenance_sha256,
        installation_sha256=installation_sha256,
        plan_record_digest=plan_document.record_digest,
    )
    if commitments != expected:
        raise ValueError("canary parity attestation commitments differ from this launch")
    if (
        type(document["shard_count"]) is not int
        or document["shard_count"] <= 0
        or type(document["shard_index"]) is not int
        or not 0 <= document["shard_index"] < document["shard_count"]
    ):
        raise ValueError("canary parity attestation coordinates are invalid")
    if document["shard_count"] != plan_document.shard_count:
        raise ValueError("canary parity attestation shard count differs from the plan")
    files = _exact_object(document["files"], set(_FILES), "attestation files")
    for name, descriptor in files.items():
        row = _exact_object(descriptor, {"byte_count", "sha256"}, f"{name} descriptor")
        if type(row["byte_count"]) is not int or row["byte_count"] <= 0:
            raise ValueError(f"{name} byte count must be positive")
        _require_sha256(row["sha256"], f"{name} SHA-256")
    _require_sha256(document["shard_record_digest"], "shard record digest")
    _require_sha256(
        document["shard_manifest_record_digest"], "shard manifest record digest"
    )
    return document


def _add_commitments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--generation-provenance-sha256", required=True)
    parser.add_argument("--installation-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--apollo-shard", required=True, type=Path)
    publish.add_argument("--goose-shard", required=True, type=Path)
    publish.add_argument("--plan", required=True, type=Path)
    publish.add_argument("--out", required=True, type=Path)
    _add_commitments(publish)
    verify = commands.add_parser("verify")
    verify.add_argument("--attestation", required=True, type=Path)
    verify.add_argument("--expected-attestation-sha256", required=True)
    verify.add_argument("--plan", required=True, type=Path)
    _add_commitments(verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kwargs = {
        "plan_sha256": args.plan_sha256,
        "source_sha256": args.source_sha256,
        "generation_provenance_sha256": args.generation_provenance_sha256,
        "installation_sha256": args.installation_sha256,
    }
    if args.command == "publish":
        result = publish_parity_attestation(
            apollo_shard=args.apollo_shard,
            goose_shard=args.goose_shard,
            plan=args.plan,
            output=args.out,
            **kwargs,
        )
    else:
        result = verify_parity_attestation(
            attestation=args.attestation,
            expected_attestation_sha256=args.expected_attestation_sha256,
            plan=args.plan,
            **kwargs,
        )
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

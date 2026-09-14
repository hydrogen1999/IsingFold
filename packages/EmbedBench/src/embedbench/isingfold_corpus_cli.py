"""Operational CLI for the prospective IsingFold corpus release pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_corpus_io import merge_corpus_shards, write_corpus_shard
from embedbench.isingfold_corpus_plan import (
    build_production_corpus_plan,
    read_corpus_plan,
    write_corpus_plan,
)
from embedbench.isingfold_corpus_preflight import (
    DEFAULT_PREFLIGHT_WORKERS,
    publish_prospective_preflight,
    read_prospective_preflight,
    require_worker_count,
    verify_prospective_preflight,
)
from embedbench.isingfold_corpus_shard import ShardConfig, generate_corpus_shard
from embedbench.isingfold_design import (
    LineageFact,
    TaskFact,
    build_corpus_design_v2,
)

_HEX = frozenset("0123456789abcdef")


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _json_value(item) for key, item in vars(value).items()}
    return value


def _emit(value: object) -> None:
    print(json.dumps(_json_value(value), allow_nan=False, sort_keys=True))


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        output[key] = value
    return output


def _read_regular(path: Path, name: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing or unreadable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return path.read_bytes()


def _canonical_json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_closed_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid UTF-8 JSON") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} must be one canonical JSON object with a terminal newline")
    return value


def _verify_record(value: Mapping[str, Any], name: str) -> None:
    digest = _sha256(value.get("record_digest"), f"{name} record digest")
    payload = {key: item for key, item in value.items() if key != "record_digest"}
    if digest != content_digest(payload):
        raise ValueError(f"{name} record digest mismatch")


def _preflight_workers(raw: str) -> int:
    try:
        value = int(raw)
        return require_worker_count(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _fact_rows(
    path: Path,
    *,
    expected_sha256: str,
    schema: str,
) -> list[dict[str, Any]]:
    raw = _read_regular(path, path.name)
    if hashlib.sha256(raw).hexdigest() != _sha256(expected_sha256, f"{path.name} SHA-256"):
        raise ValueError(f"{path.name} differs from the release manifest")
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{path.name} must be a nonempty canonical JSONL file")
    facts: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        row = _canonical_json(line + b"\n", f"{path.name} line {line_number}")
        if set(row) != {"fact", "record_digest", "schema", "schema_version"}:
            raise ValueError(f"{path.name} line {line_number} has an unsupported schema")
        if row["schema"] != schema or row["schema_version"] != 1:
            raise ValueError(f"{path.name} line {line_number} has an unsupported schema")
        _verify_record(row, f"{path.name} line {line_number}")
        fact = row["fact"]
        if type(fact) is not dict:
            raise ValueError(f"{path.name} line {line_number} fact must be an object")
        facts.append(fact)
    return facts


def _lineage_fact(value: Mapping[str, Any]) -> LineageFact:
    expected = {
        "application_family",
        "base_lineage_key",
        "generator_id",
        "generator_implementation_sha256",
        "generator_kind",
        "measurements",
        "source_instance_record_digests",
    }
    if set(value) != expected:
        raise ValueError("lineage fact fields differ from the registered schema")
    measurements = value["measurements"]
    digests = value["source_instance_record_digests"]
    if type(measurements) is not list or type(digests) is not list:
        raise ValueError("lineage fact arrays are malformed")
    return LineageFact(
        base_lineage_key=value["base_lineage_key"],
        application_family=value["application_family"],
        generator_id=value["generator_id"],
        generator_implementation_sha256=value["generator_implementation_sha256"],
        generator_kind=value["generator_kind"],
        source_instance_record_digests=tuple(digests),
        measurements=tuple(tuple(item) for item in measurements),
    )


def _task_fact(value: Mapping[str, Any]) -> TaskFact:
    expected = {
        "active_topology_identity",
        "base_parent_lineage",
        "calibration_identity",
        "descendant_transform_identity",
        "distribution",
        "fault_identity",
        "group_id",
        "nominal_topology_identity",
    }
    if set(value) != expected:
        raise ValueError("task fact fields differ from the registered schema")
    return TaskFact(**value)


def _release_facts(
    release_root: Path,
    *,
    expected_release_manifest_sha256: str,
    expected_plan_record_digest: str,
) -> tuple[list[LineageFact], list[TaskFact], str]:
    manifest_path = release_root / "release_manifest.json"
    manifest_raw = _read_regular(manifest_path, "release manifest")
    release_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if release_sha256 != _sha256(
        expected_release_manifest_sha256,
        "expected release manifest SHA-256",
    ):
        raise ValueError("release manifest differs from its external commitment")
    manifest = _canonical_json(manifest_raw, "release manifest")
    _verify_record(manifest, "release manifest")
    if manifest.get("source_plan") != {
        "lineage_request_count": sum(
            int(manifest.get("counts", {}).get(name, 0)) for name in ("lineages",)
        ),
        "record_digest": expected_plan_record_digest,
    }:
        raise ValueError("release is not bound to the pinned prospective plan")
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict:
        raise ValueError("release manifest has no artifact inventory")
    lineage_descriptor = artifacts.get("lineage_facts.jsonl")
    task_descriptor = artifacts.get("task_facts.jsonl")
    if type(lineage_descriptor) is not dict or type(task_descriptor) is not dict:
        raise ValueError("release manifest omits corpus-design facts")
    lineages = [
        _lineage_fact(value)
        for value in _fact_rows(
            release_root / "lineage_facts.jsonl",
            expected_sha256=lineage_descriptor.get("sha256"),
            schema="embedbench.isingfold-lineage-fact",
        )
    ]
    tasks = [
        _task_fact(value)
        for value in _fact_rows(
            release_root / "task_facts.jsonl",
            expected_sha256=task_descriptor.get("sha256"),
            schema="embedbench.isingfold-task-fact",
        )
    ]
    return lineages, tasks, release_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedbench-isingfold-corpus")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="freeze the prospective 3,082-lineage plan")
    plan.add_argument("--root-seed", type=int, default=260912)
    plan.add_argument("--shard-count", type=int, default=64)
    plan.add_argument("--out", type=Path, required=True)

    preflight = commands.add_parser(
        "preflight",
        help="prospect the complete pinned plan without proposals or quality evaluation",
    )
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--expected-plan-sha256", required=True)
    preflight.add_argument("--expected-source-sha256", required=True)
    preflight.add_argument(
        "--workers",
        type=_preflight_workers,
        default=DEFAULT_PREFLIGHT_WORKERS,
    )
    preflight.add_argument("--out", type=Path, required=True)

    verify_preflight = commands.add_parser(
        "verify-preflight",
        help="authenticate and replay every identity in a pinned prospective preflight",
    )
    verify_preflight.add_argument("--plan", type=Path, required=True)
    verify_preflight.add_argument("--expected-plan-sha256", required=True)
    verify_preflight.add_argument("--expected-source-sha256", required=True)
    verify_preflight.add_argument("--preflight", type=Path, required=True)
    verify_preflight.add_argument("--expected-preflight-sha256", required=True)
    verify_preflight.add_argument("--expected-preflight-record-digest", required=True)
    verify_preflight.add_argument("--expected-identity-map-digest", required=True)
    verify_preflight.add_argument("--expected-generation-provenance-digest", required=True)
    verify_preflight.add_argument(
        "--workers",
        type=_preflight_workers,
        default=DEFAULT_PREFLIGHT_WORKERS,
    )

    shard = commands.add_parser("generate-shard", help="generate one deterministic plan shard")
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--expected-plan-sha256", required=True)
    shard.add_argument("--expected-source-sha256", required=True)
    shard.add_argument("--preflight", type=Path, required=True)
    shard.add_argument("--expected-preflight-sha256", required=True)
    shard.add_argument("--expected-preflight-record-digest", required=True)
    shard.add_argument("--expected-identity-map-digest", required=True)
    shard.add_argument("--expected-generation-provenance-digest", required=True)
    shard.add_argument("--index", type=int, required=True)
    shard.add_argument("--out", type=Path, required=True)

    merge = commands.add_parser("merge", help="merge the exact complete shard census")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--expected-plan-sha256", required=True)
    merge.add_argument("--shard-root", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)

    design = commands.add_parser("design", help="publish the production corpus design")
    design.add_argument("--plan", type=Path, required=True)
    design.add_argument("--expected-plan-sha256", required=True)
    design.add_argument("--release", type=Path, required=True)
    design.add_argument("--expected-release-manifest-sha256", required=True)
    design.add_argument("--publisher-id", required=True)
    design.add_argument("--source-release-id", required=True)
    design.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        plan = build_production_corpus_plan(
            root_seed=args.root_seed,
            shard_count=args.shard_count,
        )
        receipt = write_corpus_plan(plan, args.out)
    else:
        plan = read_corpus_plan(
            args.plan,
            expected_sha256=args.expected_plan_sha256,
        )
        if args.command == "preflight":
            receipt = publish_prospective_preflight(
                plan,
                expected_plan_sha256=args.expected_plan_sha256,
                expected_source_sha256=args.expected_source_sha256,
                workers=args.workers,
                output_path=args.out,
            )
        elif args.command == "verify-preflight":
            receipt = verify_prospective_preflight(
                plan,
                expected_plan_sha256=args.expected_plan_sha256,
                expected_source_sha256=args.expected_source_sha256,
                preflight_path=args.preflight,
                expected_preflight_sha256=args.expected_preflight_sha256,
                expected_preflight_record_digest=args.expected_preflight_record_digest,
                expected_identity_map_digest=args.expected_identity_map_digest,
                expected_generation_provenance_digest=(
                    args.expected_generation_provenance_digest
                ),
                workers=args.workers,
            )
        elif args.command == "generate-shard":
            if not 0 <= args.index < plan.shard_count:
                raise ValueError("shard index lies outside the pinned plan")
            read_prospective_preflight(
                plan,
                expected_plan_sha256=args.expected_plan_sha256,
                expected_source_sha256=args.expected_source_sha256,
                preflight_path=args.preflight,
                expected_preflight_sha256=args.expected_preflight_sha256,
                expected_preflight_record_digest=args.expected_preflight_record_digest,
                expected_identity_map_digest=args.expected_identity_map_digest,
                expected_generation_provenance_digest=(
                    args.expected_generation_provenance_digest
                ),
            )
            screening = plan.screening
            config = ShardConfig(
                root_seed=plan.root_seed,
                shard_index=args.index,
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
            receipt = write_corpus_shard(shard, args.out)
        elif args.command == "merge":
            shard_dirs = tuple(
                args.shard_root / f"shard-{index:04d}-of-{plan.shard_count:04d}"
                for index in range(plan.shard_count)
            )
            receipt = merge_corpus_shards(
                shard_dirs,
                args.out,
                expected_shard_count=plan.shard_count,
                expected_plan_record_digest=plan.record_digest,
                expected_lineage_requests=tuple(item.request for item in plan.lineages),
            )
        elif args.command == "design":
            lineages, tasks, release_sha256 = _release_facts(
                args.release,
                expected_release_manifest_sha256=args.expected_release_manifest_sha256,
                expected_plan_record_digest=plan.record_digest,
            )
            receipt = build_corpus_design_v2(
                lineages=lineages,
                tasks=tasks,
                output_directory=args.out,
                publisher_id=args.publisher_id,
                source_release_id=args.source_release_id,
                source_release_manifest_sha256=release_sha256,
                split_manifest_sha256=args.expected_plan_sha256,
                measurement_budget=plan.measurement_budget,
                difficulty_rules=plan.difficulty_rules,
            )
        else:  # pragma: no cover - argparse owns the closed command set.
            raise RuntimeError(f"unsupported command {args.command!r}")
    _emit(receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

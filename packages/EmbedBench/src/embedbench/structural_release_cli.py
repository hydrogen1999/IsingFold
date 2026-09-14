"""Production CLI for the certified structural release boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from embedbench.candidate_bank import canonical_json_bytes
from embedbench.structural_release import (
    StructuralReleasePlan,
    build_production_plan,
    generate_structural_shard,
    merge_structural_shards,
)

_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class StructuralPlanFileReceipt:
    """Stable identities needed to pin later generate and merge commands."""

    plan_path: Path
    plan_sha256: str
    plan_digest: str
    lineage_count: int


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_regular(path: Path, name: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ValueError(f"{name} is missing: {path}") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file, not a symlink or directory")
    return path.read_bytes()


def _strict_plan_document(raw: bytes) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"plan contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"plan contains non-finite number {token}")

    try:
        document = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("plan is invalid JSON") from error
    if not isinstance(document, dict):
        raise ValueError("plan must contain one JSON object")
    try:
        canonical = canonical_json_bytes(document) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("plan does not contain a canonical finite JSON value") from error
    if raw != canonical:
        raise ValueError("plan must be canonical JSON with exactly one terminal newline")
    return document


def load_pinned_plan(
    plan_path: str | os.PathLike[str], expected_plan_sha256: str
) -> StructuralReleasePlan:
    """Load a canonical regular plan file only when its exact byte hash is pinned."""

    expected = _require_sha256(expected_plan_sha256, "expected_plan_sha256")
    path = Path(plan_path)
    raw = _read_regular(path, "plan")
    actual = _sha256(raw)
    if actual != expected:
        raise ValueError(
            f"plan SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return StructuralReleasePlan.from_dict(_strict_plan_document(raw))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_new_file(destination: Path, raw: bytes) -> Path:
    """Publish bytes atomically without ever replacing an existing path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_metadata = destination.parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode) or destination.parent.is_symlink():
        raise ValueError("plan output parent must be a real directory")
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"plan output already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(f"plan output already exists: {destination}") from None
        _fsync_directory(destination.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return destination.resolve(strict=True)


def write_production_plan(
    *,
    root_seed: int,
    replicates_per_cell: int,
    output_path: str | os.PathLike[str],
) -> StructuralPlanFileReceipt:
    """Build and atomically publish the canonical production plan."""

    plan = build_production_plan(
        root_seed=root_seed, replicates_per_cell=replicates_per_cell
    )
    raw = canonical_json_bytes(plan.to_dict()) + b"\n"
    path = _atomic_new_file(Path(output_path), raw)
    return StructuralPlanFileReceipt(
        plan_path=path,
        plan_sha256=_sha256(raw),
        plan_digest=plan.plan_digest,
        lineage_count=len(plan.rows),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return value


def _emit(value: Any) -> None:
    print(json.dumps(_json_value(value), allow_nan=False, sort_keys=True))


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _sha256_argument(value: str) -> str:
    try:
        return _require_sha256(value, "plan hash pin")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_plan(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "plan", help="atomically publish the deterministic structural production plan"
    )
    parser.add_argument("--root-seed", type=_nonnegative_int, required=True)
    parser.add_argument("--replicates-per-cell", type=_positive_int, required=True)
    parser.add_argument("--out", type=Path, required=True)


def _add_plan_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--expected-plan-sha256", type=_sha256_argument, required=True
    )


def _add_generate_shard(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "generate-shard", help="generate one deterministic exact structural shard"
    )
    _add_plan_input(parser)
    parser.add_argument("--shard-index", type=_nonnegative_int, required=True)
    parser.add_argument("--shard-count", type=_positive_int, required=True)
    parser.add_argument("--out", type=Path, required=True)


def _add_merge(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "merge", help="verify exact shard coverage and publish the structural release"
    )
    _add_plan_input(parser)
    parser.add_argument("--shard-dir", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedbench-structural-release")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_plan(subparsers)
    _add_generate_shard(subparsers)
    _add_merge(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            receipt = write_production_plan(
                root_seed=args.root_seed,
                replicates_per_cell=args.replicates_per_cell,
                output_path=args.out,
            )
        elif args.command == "generate-shard":
            plan = load_pinned_plan(args.plan, args.expected_plan_sha256)
            receipt = generate_structural_shard(
                plan, args.shard_index, args.shard_count, args.out
            )
        elif args.command == "merge":
            plan = load_pinned_plan(args.plan, args.expected_plan_sha256)
            receipt = merge_structural_shards(plan, args.shard_dir, args.out)
        else:  # pragma: no cover - argparse owns the closed command set.
            raise RuntimeError(f"unknown command {args.command!r}")
    except (OSError, TypeError, ValueError) as error:
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        return 2
    _emit(receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "StructuralPlanFileReceipt",
    "load_pinned_plan",
    "main",
    "write_production_plan",
]

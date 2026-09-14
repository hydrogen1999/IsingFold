"""Command-line boundary for publishing IsingFold source and evidence bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

_HEX = frozenset("0123456789abcdef")


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def certificate_registry(root: str | os.PathLike[str]) -> dict[str, Path]:
    """Authenticate a flat content-addressed certificate directory.

    The downstream publisher independently checks this registry again.  Checking it here gives
    operators an early, precise failure for accidental files, renamed artifacts, and symlinks.
    """

    directory = Path(root)
    if directory.is_symlink():
        raise ValueError("certificate root must not be a symbolic link")
    if not directory.is_dir():
        raise ValueError("certificate root must be an existing directory")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise ValueError("certificate artifact must not be a symbolic link")
        if not path.is_file():
            raise ValueError("certificate root must contain only regular artifact files")
        digest = _sha256(path.name, "certificate artifact filename")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"certificate artifact bytes do not match filename {digest}")
        result[digest] = path
    if not result:
        raise ValueError("certificate root must contain at least one artifact")
    return result


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


def _add_source(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "source",
        help="publish the authenticated CandidateBank inputs consumed by IsingFold prepare",
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--publication-index", type=Path, required=True)
    parser.add_argument("--reference-index", type=Path, required=True)
    parser.add_argument("--expected-bank-manifest-sha256", required=True)
    parser.add_argument("--expected-publication-index-sha256", required=True)
    parser.add_argument("--expected-reference-index-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)


def _add_attest(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "attest",
        help="publish partitioned certificate evidence after IsingFold prepare-v4",
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--certificate-root", type=Path, required=True)
    parser.add_argument("--publisher-id", required=True)
    parser.add_argument("--publication-id", required=True)
    parser.add_argument("--verifier-name", required=True)
    parser.add_argument("--verifier-version", required=True)
    parser.add_argument("--verifier-implementation-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)


def _add_reference_adapter(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "adapt-references",
        help="adapt an authenticated exact-reference export for IsingFold publication",
    )
    parser.add_argument("--reference-export", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-references-sha256", required=True)
    parser.add_argument("--expected-certificates-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)


def _add_verify(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "verify-attestation",
        help="independently verify a published partitioned certificate bundle",
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--expected-attestation-record-digest", required=True)
    parser.add_argument("--expected-publisher-id", required=True)
    parser.add_argument("--verifier-name", required=True)
    parser.add_argument("--verifier-version", required=True)
    parser.add_argument("--verifier-implementation-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedbench-publish")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_source(subparsers)
    _add_reference_adapter(subparsers)
    _add_attest(subparsers)
    _add_verify(subparsers)
    return parser


def _verifier(args: argparse.Namespace):
    from embedbench.isingfold_attestation import VerifierIdentity

    return VerifierIdentity(
        name=args.verifier_name,
        version=args.verifier_version,
        implementation_sha256=args.verifier_implementation_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source":
        from embedbench.isingfold_publication import publish_isingfold_source

        receipt = publish_isingfold_source(
            bank_path=args.bank,
            bank_manifest_path=args.bank_manifest,
            publication_index_path=args.publication_index,
            reference_index_path=args.reference_index,
            output_dir=args.out,
            expected_bank_manifest_sha256=args.expected_bank_manifest_sha256,
            expected_publication_index_sha256=args.expected_publication_index_sha256,
            expected_reference_index_sha256=args.expected_reference_index_sha256,
        )
    elif args.command == "adapt-references":
        from embedbench.isingfold_reference_adapter import adapt_reference_export_bundle

        receipt = adapt_reference_export_bundle(
            args.reference_export,
            args.out,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_protocol_sha256=args.expected_protocol_sha256,
            expected_references_sha256=args.expected_references_sha256,
            expected_certificates_sha256=args.expected_certificates_sha256,
        )
    elif args.command == "attest":
        from embedbench.isingfold_attestation import build_publisher_attestation_v2

        receipt = build_publisher_attestation_v2(
            args.prepared,
            certificate_registry(args.certificate_root),
            args.out,
            publisher_id=args.publisher_id,
            publication_id=args.publication_id,
            verifier=_verifier(args),
        )
    elif args.command == "verify-attestation":
        from embedbench.isingfold_attestation import verify_publisher_attestation_v2

        receipt = verify_publisher_attestation_v2(
            args.prepared,
            args.publication,
            expected_attestation_record_digest=args.expected_attestation_record_digest,
            expected_publisher_id=args.expected_publisher_id,
            expected_verifier=_verifier(args),
        )
    else:  # pragma: no cover - argparse owns the closed command set.
        raise RuntimeError(f"unknown command {args.command!r}")
    _emit(receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

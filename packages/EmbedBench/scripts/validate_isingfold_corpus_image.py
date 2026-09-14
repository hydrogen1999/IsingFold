#!/usr/bin/env python3
"""Fail-closed validation of an IsingFold corpus-generation image.

The image supplies Python and installed distributions.  The staged EmbedBench
tree supplies the generator source through PYTHONPATH.  This script binds those
two identities and emits a small JSON receipt only after ``pip check`` and the
generator's own provenance calculation succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOCK_FIELDS = {
    "base_image",
    "build_tool_versions",
    "installed_inventory",
    "platform",
    "provenance",
    "python",
    "runtime_support_versions",
    "schema",
    "thread_controls",
}
_INVENTORY_FIELDS = {
    "algorithm",
    "distribution_count",
    "file_count",
    "installation_sha256",
    "native_libraries_sha256",
    "native_library_count",
    "payload_sha256",
}
_INVENTORY_DIGEST_FIELDS = _INVENTORY_FIELDS - {"installation_sha256"}
_THREAD_VARIABLES = {
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
}
_IGNORED_PAYLOAD_NAMES = {"INSTALLER", "RECORD", "REQUESTED", "direct_url.json"}


def _object(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with text keys")
    if set(value) != fields:
        raise ValueError(f"{name} fields differ from the runtime contract")
    return value


def _version_map(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object")
    if any(
        type(key) is not str or type(version) is not str or not version
        for key, version in value.items()
    ):
        raise ValueError(f"{name} must map package names to non-empty versions")
    return dict(value)


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _validate_distribution_set(expected_names: set[str]) -> None:
    """Reject shadow metadata and every distribution outside the lock."""

    observed: set[str] = set()
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("installed distribution has no valid metadata Name")
        name = _normalized_distribution_name(raw_name)
        if name in observed:
            raise RuntimeError(f"duplicate installed distribution metadata for {name!r}")
        observed.add(name)
    normalized_expected = {_normalized_distribution_name(name) for name in expected_names}
    if observed != normalized_expected:
        raise RuntimeError(
            "installed distribution set differs from runtime lock: "
            f"missing={sorted(normalized_expected - observed)!r}, "
            f"unexpected={sorted(observed - normalized_expected)!r}"
        )


def _canonical_sha256(document: object) -> str:
    raw = json.dumps(
        document,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _installation_sha256(inventory: dict[str, object]) -> str:
    """Bind the complete installed payload and native-library inventories."""

    if set(inventory) != _INVENTORY_DIGEST_FIELDS:
        raise ValueError("installation inventory fields differ from the runtime contract")
    return _canonical_sha256(
        {
            "inventory": inventory,
            "schema": "isingfold-installed-runtime-inventory-v1",
        }
    )


def _ignore_payload_path(path: PurePosixPath) -> bool:
    return (
        ".." in path.parts
        or "__pycache__" in path.parts
        or path.suffix in {".pyc", ".pyo"}
        or path.name in _IGNORED_PAYLOAD_NAMES
    )


def _is_native_library(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((".a", ".dll", ".dylib", ".pyd", ".so")) or ".so." in lowered


def _installed_inventory(expected_versions: dict[str, str]) -> dict[str, Any]:
    """Hash immutable wheel payloads, excluding install-path-specific metadata."""

    distributions: list[dict[str, Any]] = []
    for name in sorted(expected_versions):
        distribution = importlib.metadata.distribution(name)
        files: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in distribution.files or ():
            logical = PurePosixPath(str(entry))
            if _ignore_payload_path(logical):
                continue
            logical_text = logical.as_posix()
            if logical_text in seen:
                raise RuntimeError(f"distribution {name!r} repeats payload {logical_text!r}")
            seen.add(logical_text)
            target = Path(distribution.locate_file(entry))
            if target.is_symlink() or not target.is_file():
                raise RuntimeError(
                    f"distribution {name!r} payload is missing, non-regular, or linked: "
                    f"{logical_text}"
                )
            payload = target.read_bytes()
            files.append(
                {
                    "path": logical_text,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
        files.sort(key=lambda row: str(row["path"]))
        distributions.append(
            {"files": files, "name": name, "version": expected_versions[name]}
        )

    payload_document = {
        "distributions": distributions,
        "schema": "isingfold-installed-distribution-inventory-v1",
    }
    native_libraries = [
        {"distribution": distribution["name"], **file_record}
        for distribution in distributions
        for file_record in distribution["files"]
        if _is_native_library(str(file_record["path"]))
    ]
    native_document = {
        "libraries": native_libraries,
        "schema": "isingfold-native-library-inventory-v1",
    }
    return {
        "algorithm": "isingfold-installed-wheel-payload-v1",
        "distribution_count": len(distributions),
        "distribution_summaries": [
            {
                "file_count": len(distribution["files"]),
                "name": distribution["name"],
                "payload_sha256": _canonical_sha256(
                    {
                        "files": distribution["files"],
                        "name": distribution["name"],
                        "version": distribution["version"],
                    }
                ),
                "version": distribution["version"],
            }
            for distribution in distributions
        ],
        "file_count": sum(
            len(distribution["files"]) for distribution in distributions
        ),
        "native_libraries": native_libraries,
        "native_libraries_sha256": _canonical_sha256(native_document),
        "native_library_count": len(native_libraries),
        "payload_sha256": _canonical_sha256(payload_document),
    }


def _read_lock(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("--runtime-lock must be an absolute regular non-symlink file")
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(
            stream,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"runtime lock contains non-finite JSON: {token}")
            ),
        )
    lock = _object(raw, _LOCK_FIELDS, "runtime lock")
    if lock["schema"] != "isingfold-corpus-runtime-lock-v2":
        raise ValueError("runtime lock schema is not supported")
    _object(lock["base_image"], {"reference"}, "base_image")
    _object(lock["platform"], {"machine", "system"}, "platform")
    _object(lock["python"], {"executable", "version"}, "python")
    provenance = _object(lock["provenance"], {"dependency_versions", "protocol"}, "provenance")
    _version_map(provenance["dependency_versions"], "provenance dependency_versions")
    _version_map(lock["runtime_support_versions"], "runtime_support_versions")
    _version_map(lock["build_tool_versions"], "build_tool_versions")
    inventory = _object(lock["installed_inventory"], _INVENTORY_FIELDS, "installed_inventory")
    if inventory["algorithm"] != "isingfold-installed-wheel-payload-v1":
        raise ValueError("installed inventory algorithm is not supported")
    for field in ("distribution_count", "file_count", "native_library_count"):
        if type(inventory[field]) is not int or inventory[field] <= 0:
            raise ValueError(f"installed_inventory.{field} must be a positive integer")
    for field in ("payload_sha256", "native_libraries_sha256"):
        if type(inventory[field]) is not str or not _SHA256.fullmatch(inventory[field]):
            raise ValueError(f"installed_inventory.{field} must be a lowercase SHA-256")
    installation_sha256 = inventory["installation_sha256"]
    if type(installation_sha256) is not str or not _SHA256.fullmatch(installation_sha256):
        raise ValueError(
            "installed_inventory.installation_sha256 must be a lowercase SHA-256"
        )
    inventory_digest_contract = {
        key: inventory[key] for key in _INVENTORY_DIGEST_FIELDS
    }
    if _installation_sha256(inventory_digest_contract) != installation_sha256:
        raise ValueError("installed_inventory.installation_sha256 is not canonical")
    thread_controls = _version_map(lock["thread_controls"], "thread_controls")
    if set(thread_controls) != _THREAD_VARIABLES or set(thread_controls.values()) != {"1"}:
        raise ValueError("thread_controls must pin all four registered variables to one")
    return lock


def validate(
    *,
    runtime_lock: Path,
    source_root: Path,
    expected_installation_sha256: str,
    expected_provenance_sha256: str,
    expected_python_executable: str | None = None,
) -> dict[str, object]:
    """Validate the bound runtime and return its canonical receipt payload."""

    lock = _read_lock(runtime_lock)
    if not _SHA256.fullmatch(expected_installation_sha256):
        raise ValueError("--expected-installation-sha256 must be one lowercase SHA-256")
    if not _SHA256.fullmatch(expected_provenance_sha256):
        raise ValueError("--expected-provenance-sha256 must be one lowercase SHA-256")
    if not source_root.is_absolute() or not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("--source-root must be an absolute directory, not a symlink")
    source_root = source_root.resolve(strict=True)

    python_lock = lock["python"]
    assert isinstance(python_lock, dict)
    expected_executable = expected_python_executable or str(python_lock["executable"])
    if not os.path.isabs(expected_executable):
        raise ValueError("expected Python executable must be absolute")
    if os.path.abspath(sys.executable) != expected_executable:
        raise RuntimeError(
            f"validator runs under {sys.executable!r}, expected {expected_executable!r}"
        )
    if platform.python_version() != python_lock["version"]:
        raise RuntimeError("Python version differs from runtime lock")

    platform_lock = lock["platform"]
    assert isinstance(platform_lock, dict)
    observed_platform = {"machine": platform.machine(), "system": platform.system()}
    if observed_platform != platform_lock:
        raise RuntimeError(
            f"platform differs from runtime lock: observed={observed_platform!r}"
        )

    provenance_lock = lock["provenance"]
    assert isinstance(provenance_lock, dict)
    dependency_versions = _version_map(
        provenance_lock["dependency_versions"], "provenance dependency_versions"
    )
    support_versions = _version_map(lock["runtime_support_versions"], "runtime support")
    tool_versions = _version_map(lock["build_tool_versions"], "build tools")
    expected_versions = {**dependency_versions, **support_versions, **tool_versions}
    _validate_distribution_set(set(expected_versions))
    observed_versions = {
        name: importlib.metadata.version(name) for name in sorted(expected_versions)
    }
    if observed_versions != dict(sorted(expected_versions.items())):
        raise RuntimeError(
            "installed distribution versions differ from runtime lock: "
            f"observed={observed_versions!r}"
        )

    thread_controls = _version_map(lock["thread_controls"], "thread_controls")
    observed_thread_controls = {name: os.environ.get(name) for name in sorted(thread_controls)}
    if observed_thread_controls != dict(sorted(thread_controls.items())):
        raise RuntimeError(
            "thread controls differ from runtime lock: "
            f"observed={observed_thread_controls!r}"
        )

    inventory_versions = dict(expected_versions)
    inventory_versions.pop("embedbench")
    observed_inventory = _installed_inventory(inventory_versions)
    inventory_lock = lock["installed_inventory"]
    assert isinstance(inventory_lock, dict)
    inventory_contract = {
        key: observed_inventory[key] for key in _INVENTORY_DIGEST_FIELDS
    }
    locked_inventory_contract = {
        key: inventory_lock[key] for key in _INVENTORY_DIGEST_FIELDS
    }
    if inventory_contract != locked_inventory_contract:
        raise RuntimeError(
            "installed distribution/native-library inventory differs from runtime lock: "
            f"observed={inventory_contract!r}"
        )
    observed_installation_sha256 = _installation_sha256(inventory_contract)
    if observed_installation_sha256 != inventory_lock["installation_sha256"]:
        raise RuntimeError("installed runtime inventory differs from its lock commitment")
    if observed_installation_sha256 != expected_installation_sha256:
        raise RuntimeError(
            "installed runtime inventory differs from the cross-site commitment: "
            f"observed={observed_installation_sha256}"
        )

    for name in sorted(expected_versions):
        direct_url_text = importlib.metadata.distribution(name).read_text("direct_url.json")
        if direct_url_text is None:
            continue
        direct_url = json.loads(direct_url_text)
        editable = bool(direct_url.get("dir_info", {}).get("editable", False))
        if name != "embedbench" or editable:
            raise RuntimeError(
                f"distribution {name!r} is editable, VCS-installed, or direct-URL-installed; "
                "the production runtime requires the hashed wheel lock"
            )

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        text=True,
    )
    if pip_check.returncode != 0:
        raise RuntimeError(
            f"pip check failed ({pip_check.returncode}): "
            f"{pip_check.stdout}{pip_check.stderr}"
        )

    import embedbench.isingfold_corpus_shard as shard_module
    from embedbench.candidate_bank import content_digest
    from embedbench.isingfold_corpus_shard import _generation_provenance

    expected_module = (source_root / "src" / "embedbench" / "isingfold_corpus_shard.py").resolve()
    actual_module = Path(shard_module.__file__).resolve()
    if actual_module != expected_module:
        raise RuntimeError(
            f"generator imported from {actual_module}, expected staged source {expected_module}"
        )
    provenance = _generation_provenance()
    if provenance.get("protocol") != provenance_lock["protocol"]:
        raise RuntimeError("generator protocol differs from runtime lock")
    if provenance.get("python_version") != python_lock["version"]:
        raise RuntimeError("generator provenance reports the wrong Python version")
    if provenance.get("dependency_versions") != dependency_versions:
        raise RuntimeError("generator provenance dependencies differ from runtime lock")
    if provenance.get("thread_controls") != thread_controls:
        raise RuntimeError("generator provenance thread controls differ from runtime lock")
    observed_provenance_sha256 = content_digest(provenance)
    if observed_provenance_sha256 != expected_provenance_sha256:
        raise RuntimeError(
            "generation provenance differs from the cross-site commitment: "
            f"observed={observed_provenance_sha256}"
        )

    return {
        "dependency_versions": observed_versions,
        "generation_provenance_sha256": observed_provenance_sha256,
        "installation_sha256": observed_installation_sha256,
        "installed_inventory": observed_inventory,
        "pip_check": "ok",
        "platform": observed_platform,
        "python_executable": os.path.abspath(sys.executable),
        "python_version": platform.python_version(),
        "runtime_schema": lock["schema"],
        "source_root": str(source_root),
        "thread_controls": observed_thread_controls,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-lock", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-installation-sha256", required=True)
    parser.add_argument("--expected-provenance-sha256", required=True)
    parser.add_argument(
        "--expected-python-executable",
        help="override the image path when validating the clean Apollo virtualenv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = validate(
        runtime_lock=args.runtime_lock,
        source_root=args.source_root,
        expected_installation_sha256=args.expected_installation_sha256,
        expected_provenance_sha256=args.expected_provenance_sha256,
        expected_python_executable=args.expected_python_executable,
    )
    print(json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

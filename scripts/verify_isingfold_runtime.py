#!/usr/bin/env python3
"""Construct and verify IsingFold's immutable scientific runtime receipts.

The verifier intentionally uses only the Python standard library before it has
authenticated the installed environment.  It binds the exact project wheel,
the native ``lac_minorminer`` extension, every installed distribution payload,
the checkpoint source registry, the staged operational source, and (on Goose)
the Apptainer image that encloses the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IGNORED_DISTRIBUTION_FILES = {"INSTALLER", "RECORD", "REQUESTED", "direct_url.json"}
_THREAD_NAMES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
_SOURCE_TOP_LEVEL_FILES = ("CMakeLists.txt", "README.md", "pyproject.toml")
_SOURCE_TREES: tuple[tuple[str, frozenset[str]], ...] = (
    ("src", frozenset({".py"})),
    ("cpp", frozenset({".cpp", ".h", ".hpp", ".txt"})),
    ("configs", frozenset({".json"})),
    ("scripts", frozenset({".py", ".sh", ".sbatch"})),
    ("runtime/isingfold-training", frozenset({".def", ".json", ".lock", ".md"})),
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an absolute regular non-symbolic-link file")
    physical = path.parent.resolve(strict=True) / path.name
    if path != physical:
        raise ValueError(f"{name} must use a canonical physical path")
    return path


def _require_directory(path: Path, name: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be an absolute existing non-symbolic-link directory")
    physical = path.resolve(strict=True)
    if path != physical:
        raise ValueError(f"{name} must use a canonical physical path")
    return path


def _require_executable(path: Path, name: str) -> Path:
    if not path.is_absolute() or not path.exists() or not os.access(path, os.X_OK):
        raise ValueError(f"{name} must be an absolute executable")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must resolve to a regular file")
    return path


def _file_record(path: Path, relative: PurePosixPath) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"inventory member is not a regular file: {path}")
    payload = path.read_bytes()
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def _walk_source_tree(
    root: Path,
    relative_root: str,
    suffixes: frozenset[str],
) -> Iterable[dict[str, object]]:
    tree = root / relative_root
    if not tree.exists():
        return
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError(f"source tree is missing or a symbolic link: {relative_root}")
    for current, directory_names, filenames in os.walk(tree, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in sorted(directory_names):
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"source inventory contains a symbolic link: {child}")
            if name not in {"__pycache__", ".pytest_cache", ".ruff_cache"}:
                kept.append(name)
        directory_names[:] = kept
        for name in sorted(filenames):
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"source inventory contains a symbolic link: {path}")
            if path.suffix not in suffixes:
                continue
            relative = PurePosixPath(path.relative_to(root).as_posix())
            yield _file_record(path, relative)


def source_inventory(source_root: Path) -> dict[str, object]:
    """Hash the complete scientific and operational source closure."""

    root = _require_directory(source_root, "source root")
    records: list[dict[str, object]] = []
    for relative_name in _SOURCE_TOP_LEVEL_FILES:
        path = root / relative_name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required source file is absent or linked: {relative_name}")
        records.append(_file_record(path, PurePosixPath(relative_name)))
    for relative_root, suffixes in _SOURCE_TREES:
        records.extend(_walk_source_tree(root, relative_root, suffixes))
    records.sort(key=lambda row: str(row["path"]))
    paths = [str(row["path"]) for row in records]
    if len(paths) != len(set(paths)):
        raise ValueError("source inventory repeats a logical path")
    payload = {
        "files": records,
        "schema": "isingfold.runtime-source-inventory",
        "schema_version": 1,
    }
    return {
        **payload,
        "file_count": len(records),
        "source_sha256": canonical_digest(payload),
    }


def inspect_project_wheel(path: Path) -> dict[str, object]:
    """Inspect a project wheel without importing or extracting its contents."""

    wheel = _require_regular_file(path, "project wheel")
    if wheel.suffix != ".whl":
        raise ValueError("project wheel must end in .whl")
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        names = [PurePosixPath(info.filename) for info in infos if not info.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("project wheel repeats a logical member")
        unsafe = [name for name in names if name.is_absolute() or ".." in name.parts]
        if unsafe:
            raise ValueError("project wheel contains an unsafe member path")
        native_names = [
            name
            for name in names
            if name.parts and name.parts[0] == "lac_minorminer"
            and name.name.startswith("_core") and name.suffix == ".so"
        ]
        if len(native_names) != 1:
            raise ValueError("project wheel must contain exactly one lac_minorminer/_core*.so")
        native_name = native_names[0]
        native_payload = archive.read(native_name.as_posix())
        source_records: list[dict[str, object]] = []
        for name in names:
            if name.suffix != ".py" or not name.parts:
                continue
            if name.parts[0] not in {"isingfold", "lac_minorminer", "isingfold_lac_b"}:
                continue
            payload = archive.read(name.as_posix())
            source_records.append(
                {
                    "path": name.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            )
    source_records.sort(key=lambda row: str(row["path"]))
    source_payload = {
        "files": source_records,
        "schema": "isingfold.project-wheel-python-source",
        "schema_version": 1,
    }
    return {
        "native_extension_path": native_name.as_posix(),
        "native_extension_sha256": hashlib.sha256(native_payload).hexdigest(),
        "python_source_count": len(source_records),
        "python_source_sha256": canonical_digest(source_payload),
        "wheel_filename": wheel.name,
        "wheel_sha256": file_sha256(wheel),
        "wheel_size": wheel.stat().st_size,
    }


def wheelhouse_manifest(wheelhouse: Path) -> dict[str, object]:
    root = _require_directory(wheelhouse, "wheelhouse")
    wheels = sorted(root.glob("*.whl"), key=lambda item: item.name)
    if not wheels or any(path.is_symlink() or not path.is_file() for path in wheels):
        raise ValueError("wheelhouse must contain regular non-symbolic-link wheels")
    files = [
        {"filename": path.name, "sha256": file_sha256(path), "size": path.stat().st_size}
        for path in wheels
    ]
    project = [row for row in files if str(row["filename"]).startswith("isingfold-0.1.0-")]
    if len(project) != 1:
        raise ValueError("wheelhouse must contain exactly one IsingFold 0.1.0 wheel")
    payload = {
        "files": files,
        "project_wheel": project[0]["filename"],
        "schema": "isingfold.runtime-wheelhouse-manifest",
        "schema_version": 1,
    }
    return {**payload, "manifest_sha256": canonical_digest(payload)}


def verify_wheelhouse_manifest(wheelhouse: Path, manifest: Mapping[str, object]) -> None:
    observed = wheelhouse_manifest(wheelhouse)
    expected_files = manifest.get("files")
    observed_files = observed["files"]
    if not isinstance(expected_files, list):
        raise ValueError("wheelhouse manifest files must be a list")
    expected_names = [row.get("filename") for row in expected_files if isinstance(row, dict)]
    observed_names = [row["filename"] for row in observed_files]
    if expected_names != observed_names:
        raise ValueError("wheel set differs from the authenticated manifest")
    if dict(manifest) != observed:
        raise ValueError("wheelhouse bytes differ from the authenticated manifest")


def publish_receipt(path: Path, payload: Mapping[str, object]) -> None:
    """Publish a canonical self-digested JSON file without replacement."""

    if not path.is_absolute():
        raise ValueError("receipt path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    if "receipt_sha256" in document:
        raise ValueError("receipt payload cannot predeclare receipt_sha256")
    document["receipt_sha256"] = canonical_digest(document)
    raw = canonical_json(document) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _strict_object(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with text keys")
    if set(value) != fields:
        raise ValueError(f"{name} fields differ from the runtime contract")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, name: str) -> dict[str, Any]:
    regular = _require_regular_file(path, name)
    try:
        with regular.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"{name} contains non-finite JSON: {token}")
                ),
                object_pairs_hook=_unique_json_object,
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def load_receipt(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    expected = _require_sha256(expected_file_sha256, "expected receipt file digest")
    receipt_path = _require_regular_file(path, "receipt")
    observed = file_sha256(receipt_path)
    if observed != expected:
        raise ValueError(
            f"receipt file digest differs from its external commitment: {observed}"
        )
    document = _read_json(receipt_path, "receipt")
    embedded = _require_sha256(document.get("receipt_sha256"), "receipt_sha256")
    payload = dict(document)
    payload.pop("receipt_sha256")
    if canonical_digest(payload) != embedded:
        raise ValueError("receipt self-digest is invalid")
    return document


def _version_map(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object")
    if any(
        type(key) is not str or not key or type(version) is not str or not version
        for key, version in value.items()
    ):
        raise ValueError(f"{name} must map non-empty names to non-empty versions")
    normalized = {_normalized_distribution_name(key): version for key, version in value.items()}
    if len(normalized) != len(value):
        raise ValueError(f"{name} repeats a normalized distribution name")
    return normalized


def read_runtime_lock(path: Path) -> dict[str, Any]:
    lock = _read_json(path, "runtime lock")
    _strict_object(
        lock,
        {
            "accelerator",
            "base_image",
            "distribution_versions",
            "platform",
            "project",
            "python",
            "schema",
            "schema_version",
            "thread_controls",
            "wheelhouse",
        },
        "runtime lock",
    )
    if lock["schema"] != "isingfold.training-runtime-lock" or lock["schema_version"] != 1:
        raise ValueError("runtime lock schema is not supported")
    base = _strict_object(lock["base_image"], {"reference"}, "base_image")
    reference = base["reference"]
    if reference != (
        "docker.io/library/python@sha256:"
        "afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63"
    ):
        raise ValueError("base_image.reference must equal the registered digest-only image")
    platform_lock = _strict_object(lock["platform"], {"machine", "system"}, "platform")
    if platform_lock != {"machine": "x86_64", "system": "Linux"}:
        raise ValueError("runtime platform must be Linux x86_64")
    python_lock = _strict_object(
        lock["python"], {"container_executable", "implementation", "version"}, "python"
    )
    if python_lock["implementation"] != "CPython":
        raise ValueError("runtime Python implementation must be CPython")
    if python_lock["version"] != "3.12.3":
        raise ValueError("runtime Python patch must be 3.12.3")
    if python_lock["container_executable"] != "/opt/isingfold/venv/bin/python":
        raise ValueError("container Python path differs from the registered path")
    accelerator = _strict_object(
        lock["accelerator"],
        {"cublas_workspace_config", "cuda_runtime", "require_cuda", "torch"},
        "accelerator",
    )
    if accelerator != {
        "cublas_workspace_config": ":4096:8",
        "cuda_runtime": "12.8",
        "require_cuda": True,
        "torch": "2.11.0+cu128",
    }:
        raise ValueError("accelerator contract differs from CUDA 12.8 Torch 2.11")
    project = _strict_object(
        lock["project"],
        {"distribution", "native_extension_glob", "version"},
        "project",
    )
    if project != {
        "distribution": "isingfold",
        "native_extension_glob": "lac_minorminer/_core*.so",
        "version": "0.1.0",
    }:
        raise ValueError("project wheel contract differs from IsingFold 0.1.0")
    versions = _version_map(lock["distribution_versions"], "distribution_versions")
    if versions.get("isingfold") != project["version"]:
        raise ValueError("project version differs from distribution_versions")
    if versions.get("torch") != accelerator["torch"]:
        raise ValueError("Torch version differs from accelerator contract")
    thread_controls = lock["thread_controls"]
    if (
        not isinstance(thread_controls, dict)
        or set(thread_controls) != set(_THREAD_NAMES)
        or set(thread_controls.values()) != {"1"}
    ):
        raise ValueError("thread controls must pin all four registered variables to one")
    wheelhouse = _strict_object(
        lock["wheelhouse"], {"manifest_schema", "manifest_version"}, "wheelhouse"
    )
    if wheelhouse != {
        "manifest_schema": "isingfold.runtime-wheelhouse-manifest",
        "manifest_version": 1,
    }:
        raise ValueError("wheelhouse manifest contract is unsupported")
    return lock


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _ignore_distribution_file(path: PurePosixPath) -> bool:
    return (
        ".." in path.parts
        or "__pycache__" in path.parts
        or path.suffix in {".pyc", ".pyo"}
        or path.name in _IGNORED_DISTRIBUTION_FILES
    )


def _is_native_library(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith((".a", ".dll", ".dylib", ".pyd", ".so")) or ".so." in lowered


def installed_inventory(expected_versions: Mapping[str, str]) -> dict[str, object]:
    """Hash every stable installed payload byte for the exact distribution set."""

    expected = _version_map(dict(expected_versions), "expected distribution versions")
    discovered: dict[str, Any] = {}
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("installed distribution metadata has no valid Name")
        name = _normalized_distribution_name(raw_name)
        if name in discovered:
            raise ValueError(f"installed distribution metadata repeats {name!r}")
        discovered[name] = distribution
    if set(discovered) != set(expected):
        raise ValueError(
            "installed distribution set differs from runtime lock: "
            f"missing={sorted(set(expected) - set(discovered))!r}, "
            f"unexpected={sorted(set(discovered) - set(expected))!r}"
        )

    distributions: list[dict[str, object]] = []
    for name in sorted(expected):
        distribution = discovered[name]
        if distribution.version != expected[name]:
            raise ValueError(
                f"installed distribution versions differ for {name}: "
                f"expected {expected[name]}, observed {distribution.version}"
            )
        files: list[dict[str, object]] = []
        seen: set[str] = set()
        for entry in distribution.files or ():
            logical = PurePosixPath(str(entry))
            if _ignore_distribution_file(logical):
                continue
            logical_text = logical.as_posix()
            if logical_text in seen:
                raise ValueError(f"distribution {name!r} repeats payload {logical_text!r}")
            seen.add(logical_text)
            target = Path(distribution.locate_file(entry))
            if target.is_symlink() or not target.is_file():
                raise ValueError(
                    f"distribution {name!r} payload is missing or linked: {logical_text}"
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
        distributions.append({"files": files, "name": name, "version": expected[name]})

    payload_document = {
        "distributions": distributions,
        "schema": "isingfold.installed-distribution-inventory",
        "schema_version": 1,
    }
    native_libraries = [
        {"distribution": distribution["name"], **record}
        for distribution in distributions
        for record in distribution["files"]  # type: ignore[union-attr]
        if _is_native_library(str(record["path"]))
    ]
    native_document = {
        "libraries": native_libraries,
        "schema": "isingfold.installed-native-library-inventory",
        "schema_version": 1,
    }
    summaries = [
        {
            "file_count": len(distribution["files"]),  # type: ignore[arg-type]
            "name": distribution["name"],
            "payload_sha256": canonical_digest(distribution),
            "version": distribution["version"],
        }
        for distribution in distributions
    ]
    contract = {
        "algorithm": "isingfold.installed-wheel-payload-v1",
        "distribution_count": len(distributions),
        "distribution_summaries": summaries,
        "file_count": sum(len(row["files"]) for row in distributions),  # type: ignore[arg-type]
        "native_libraries": native_libraries,
        "native_libraries_sha256": canonical_digest(native_document),
        "native_library_count": len(native_libraries),
        "payload_sha256": canonical_digest(payload_document),
    }
    return {
        **contract,
        "installation_sha256": canonical_digest(
            {"inventory": contract, "schema": "isingfold.installed-runtime-v1"}
        ),
    }


def verify_project_direct_url(*, project_wheel: Path, expected_wheel_sha256: str) -> None:
    wheel = _require_regular_file(project_wheel, "project wheel")
    expected = _require_sha256(expected_wheel_sha256, "expected project wheel digest")
    if file_sha256(wheel) != expected:
        raise ValueError("project wheel digest differs from its external commitment")
    distribution = importlib_metadata.distribution("isingfold")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise ValueError("IsingFold installation has no project-wheel provenance")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise ValueError("IsingFold direct_url.json is malformed") from exc
    if direct_url.get("vcs_info") is not None:
        raise ValueError("IsingFold cannot be installed from VCS")
    if bool(direct_url.get("dir_info", {}).get("editable", False)):
        raise ValueError("IsingFold cannot be installed editable")
    archive = direct_url.get("archive_info")
    if not isinstance(archive, dict):
        raise ValueError("IsingFold must be installed from an archive wheel")
    hashes = archive.get("hashes")
    stated_hash = archive.get("hash")
    observed = hashes.get("sha256") if isinstance(hashes, dict) else None
    if observed is None and isinstance(stated_hash, str) and stated_hash.startswith("sha256="):
        observed = stated_hash.removeprefix("sha256=")
    if observed != expected:
        raise ValueError("installed IsingFold wheel digest differs from the attested wheel digest")
    url = direct_url.get("url")
    if not isinstance(url, str) or PurePosixPath(url.removeprefix("file://")).name != wheel.name:
        raise ValueError("installed IsingFold wheel filename differs from the attested wheel")


_CUDA_SMOKE_PROGRAM = r'''from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os

import torch


if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is not available")
if torch.version.cuda != "12.8":
    raise SystemExit(f"Torch reports CUDA {torch.version.cuda!r}, expected '12.8'")
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise SystemExit("CUBLAS_WORKSPACE_CONFIG is not pinned to :4096:8")

torch.use_deterministic_algorithms(True)
torch.manual_seed(271828)
torch.cuda.manual_seed_all(271828)
device = torch.device("cuda:0")
model = torch.nn.Sequential(
    torch.nn.Linear(5, 7),
    torch.nn.GELU(),
    torch.nn.Dropout(p=0.25),
    torch.nn.Linear(7, 3),
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
initial_model = copy.deepcopy(model.state_dict())
initial_optimizer = copy.deepcopy(optimizer.state_dict())
initial_cpu_rng = torch.get_rng_state().clone()
initial_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]


def one_step():
    model.train()
    optimizer.zero_grad(set_to_none=True)
    features = torch.tensor(
        [[0.25, -0.5, 0.75, 1.0, -1.25], [-0.1, 0.2, -0.3, 0.4, -0.5]],
        dtype=torch.float32,
        device=device,
    )
    target = torch.tensor(
        [[0.2, -0.4, 0.6], [-0.3, 0.5, -0.7]],
        dtype=torch.float32,
        device=device,
    )
    output = model(features)
    loss = torch.square(output - target).mean()
    loss.backward()
    optimizer.step()
    return output.detach().clone(), copy.deepcopy(model.state_dict()), copy.deepcopy(
        optimizer.state_dict()
    )


def tensor_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            tensor_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tensor_tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


first = one_step()
model.load_state_dict(initial_model)
optimizer.load_state_dict(initial_optimizer)
torch.set_rng_state(initial_cpu_rng)
torch.cuda.set_rng_state_all(initial_cuda_rng)
second = one_step()
if not all(tensor_tree_equal(left, right) for left, right in zip(first, second)):
    raise SystemExit("CUDA forward/backward resume replay was not bitwise identical")

digest = hashlib.sha256()
for tensor in (first[0], *first[1].values()):
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
properties = torch.cuda.get_device_properties(0)
cuda_driver = ctypes.CDLL("libcuda.so.1")
driver_version = ctypes.c_int()
if cuda_driver.cuDriverGetVersion(ctypes.byref(driver_version)) != 0:
    raise SystemExit("CUDA driver version query failed")
print(json.dumps({
    "capability": [properties.major, properties.minor],
    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    "cuda_driver_api_version": driver_version.value,
    "cuda_runtime": torch.version.cuda,
    "device_count": torch.cuda.device_count(),
    "device_name": properties.name,
    "replay_sha256": digest.hexdigest(),
}, allow_nan=False, separators=(",", ":"), sort_keys=True))
'''


def cuda_smoke_check(
    python_executable: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run a forward/backward CUDA step twice across an exact state restore."""

    python_path = _require_executable(python_executable, "runtime Python")
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8") as program:
        program.write(_CUDA_SMOKE_PROGRAM)
        program.flush()
        completed = subprocess.run(
            [str(python_path), "-I", program.name],
            check=False,
            capture_output=True,
            env=None if environment is None else dict(environment),
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"CUDA smoke check failed ({completed.returncode}): "
            f"{completed.stdout}{completed.stderr}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CUDA smoke check did not emit one JSON document") from exc
    required = {
        "capability",
        "cublas_workspace_config",
        "cuda_driver_api_version",
        "cuda_runtime",
        "device_count",
        "device_name",
        "replay_sha256",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise RuntimeError("CUDA smoke check report fields differ from the contract")
    if (
        report["cublas_workspace_config"] != ":4096:8"
        or type(report["cuda_driver_api_version"]) is not int
        or report["cuda_driver_api_version"] <= 0
        or report["cuda_runtime"] != "12.8"
        or type(report["device_count"]) is not int
        or report["device_count"] < 1
    ):
        raise RuntimeError("CUDA smoke check reports the wrong runtime or no device")
    _require_sha256(report["replay_sha256"], "CUDA replay digest")
    return report


def _write_json_exclusive(path: Path, document: Mapping[str, object]) -> None:
    if not path.is_absolute():
        raise ValueError("output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(dict(document)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_wheelhouse_manifest(path: Path, expected_file_sha256: str | None) -> dict[str, Any]:
    manifest_path = _require_regular_file(path, "wheelhouse manifest")
    if expected_file_sha256 is not None:
        expected = _require_sha256(
            expected_file_sha256, "expected wheelhouse-manifest file digest"
        )
        if file_sha256(manifest_path) != expected:
            raise ValueError("wheelhouse manifest file digest differs from its commitment")
    manifest = _read_json(manifest_path, "wheelhouse manifest")
    _strict_object(
        manifest,
        {"files", "manifest_sha256", "project_wheel", "schema", "schema_version"},
        "wheelhouse manifest",
    )
    if (
        manifest["schema"] != "isingfold.runtime-wheelhouse-manifest"
        or manifest["schema_version"] != 1
    ):
        raise ValueError("wheelhouse manifest schema is unsupported")
    payload = dict(manifest)
    embedded = _require_sha256(payload.pop("manifest_sha256"), "wheelhouse manifest digest")
    if canonical_digest(payload) != embedded:
        raise ValueError("wheelhouse manifest self-digest is invalid")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("wheelhouse manifest files must be a non-empty list")
    names: list[str] = []
    for row in files:
        parsed = _strict_object(row, {"filename", "sha256", "size"}, "wheel record")
        filename = parsed["filename"]
        if (
            type(filename) is not str
            or PurePosixPath(filename).name != filename
            or not filename.endswith(".whl")
        ):
            raise ValueError("wheel record filename must be one basename ending in .whl")
        _require_sha256(parsed["sha256"], "wheel digest")
        if type(parsed["size"]) is not int or parsed["size"] <= 0:
            raise ValueError("wheel size must be positive")
        names.append(filename)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("wheelhouse manifest wheel names must be sorted and unique")
    if manifest["project_wheel"] not in names:
        raise ValueError("wheelhouse project_wheel does not name a manifest member")
    return manifest


def _project_wheel_path(wheelhouse: Path, manifest: Mapping[str, object]) -> Path:
    root = _require_directory(wheelhouse, "wheelhouse")
    filename = manifest.get("project_wheel")
    if type(filename) is not str or PurePosixPath(filename).name != filename:
        raise ValueError("wheelhouse manifest has an invalid project wheel filename")
    return _require_regular_file(root / filename, "project wheel")


def _verify_external_install_origins(expected_versions: Mapping[str, str]) -> None:
    for name in sorted(expected_versions):
        distribution = importlib_metadata.distribution(name)
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None or name == "isingfold":
            continue
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"distribution {name!r} has malformed direct_url.json") from exc
        editable = bool(direct_url.get("dir_info", {}).get("editable", False))
        if editable or direct_url.get("vcs_info") is not None or direct_url.get("url") is not None:
            raise ValueError(
                f"external distribution {name!r} is editable, VCS, or direct-URL installed"
            )


def _pip_check(python_executable: Path) -> None:
    completed = subprocess.run(
        [str(python_executable), "-I", "-m", "pip", "check"],
        check=False,
        capture_output=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"pip check failed ({completed.returncode}): {completed.stdout}{completed.stderr}"
        )


def _native_extension() -> dict[str, object]:
    specification = importlib.util.find_spec("lac_minorminer._core")
    origin = None if specification is None else specification.origin
    if origin is None:
        raise ValueError("cannot resolve lac_minorminer._core")
    path = Path(origin)
    if path.is_symlink() or not path.is_file():
        raise ValueError("lac_minorminer._core is not one regular installed file")
    return {
        "artifact_kind": "extension-module",
        "filename": path.name,
        "sha256": file_sha256(path),
        "size": path.stat().st_size,
    }


def _verify_project_import_origins() -> None:
    """Reject source-tree, editable, or injected imports at execution time."""

    distribution = importlib_metadata.distribution("isingfold")
    install_root = Path(distribution.locate_file("")).resolve(strict=True)
    registered_files = {
        Path(distribution.locate_file(entry)).resolve(strict=True)
        for entry in distribution.files or ()
        if not _ignore_distribution_file(PurePosixPath(str(entry)))
    }
    for module_name in (
        "isingfold",
        "isingfold.rl.cli",
        "lac_minorminer",
        "lac_minorminer._core",
    ):
        specification = importlib.util.find_spec(module_name)
        origin = None if specification is None else specification.origin
        if origin is None:
            raise ValueError(f"cannot resolve required project module {module_name!r}")
        path = Path(origin)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"project module {module_name!r} is missing or linked")
        physical = path.resolve(strict=True)
        try:
            physical.relative_to(install_root)
        except ValueError as exc:
            raise ValueError(
                f"project module {module_name!r} was imported outside the attested wheel"
            ) from exc
        if physical not in registered_files:
            raise ValueError(
                f"project module {module_name!r} is not a registered attested-wheel payload"
            )


def _checkpoint_registry() -> dict[str, object]:
    from isingfold.rl.checkpoint import runtime_implementation_registry

    registry = runtime_implementation_registry()
    if not isinstance(registry, dict):
        raise ValueError("checkpoint runtime registry must be one JSON object")
    if registry.get("schema") != "isingfold.rl-runtime-implementation":
        raise ValueError("checkpoint runtime registry schema is unsupported")
    modules = registry.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("checkpoint runtime registry has no module source closure")
    return json.loads(canonical_json(registry))


def _installed_project_source() -> dict[str, object]:
    distribution = importlib_metadata.distribution("isingfold")
    records: list[dict[str, object]] = []
    for entry in distribution.files or ():
        logical = PurePosixPath(str(entry))
        if (
            logical.suffix != ".py"
            or not logical.parts
            or logical.parts[0] not in {"isingfold", "lac_minorminer", "isingfold_lac_b"}
        ):
            continue
        target = Path(distribution.locate_file(entry))
        records.append(_file_record(target, logical))
    records.sort(key=lambda row: str(row["path"]))
    payload = {
        "files": records,
        "schema": "isingfold.project-wheel-python-source",
        "schema_version": 1,
    }
    return {
        "python_source_count": len(records),
        "python_source_sha256": canonical_digest(payload),
    }


def _command_identity(name: str) -> dict[str, object]:
    executable = shutil.which(name)
    if executable is None:
        raise ValueError(f"required build program is absent: {name}")
    path = Path(executable).resolve(strict=True)
    completed = subprocess.run(
        [str(path), "--version"], check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot identify build program {name}")
    first_line = (completed.stdout + completed.stderr).splitlines()
    if not first_line:
        raise ValueError(f"build program {name} emitted no version")
    return {
        "executable_sha256": file_sha256(path),
        "version_line": first_line[0],
    }


def _critical_source_files(inventory: Mapping[str, object]) -> dict[str, str]:
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("source inventory has no files")
    critical: dict[str, str] = {}
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("source inventory contains a malformed file record")
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("source inventory contains a malformed file record")
        if path.startswith("scripts/") or path.startswith("runtime/isingfold-training/"):
            critical[path] = digest
    required = {
        "scripts/publication_runtime.sh",
        "scripts/verify_runtime_source.sh",
        "scripts/verify_isingfold_runtime.py",
    }
    if not required.issubset(critical):
        raise ValueError("source inventory omits a required runtime helper")
    return dict(sorted(critical.items()))


def create_build_receipt(
    *,
    source_root: Path,
    runtime_lock_path: Path,
    runtime_requirements: Path,
    build_requirements: Path,
    definition: Path,
    wheelhouse: Path,
    wheelhouse_manifest_path: Path,
    second_project_wheel: Path,
) -> dict[str, object]:
    """Create the content-addressed canonical build evidence payload."""

    lock = read_runtime_lock(runtime_lock_path)
    observed_build_platform = {"machine": platform.machine(), "system": platform.system()}
    if observed_build_platform != lock["platform"]:
        raise ValueError("wheel-build platform differs from runtime lock")
    if (
        platform.python_implementation() != lock["python"]["implementation"]
        or platform.python_version() != lock["python"]["version"]
    ):
        raise ValueError("wheel-build Python differs from runtime lock")
    observed_threads = {name: os.environ.get(name) for name in _THREAD_NAMES}
    if observed_threads != lock["thread_controls"]:
        raise ValueError("wheel-build thread environment differs from runtime lock")
    source = source_inventory(source_root)
    manifest = _load_wheelhouse_manifest(wheelhouse_manifest_path, None)
    verify_wheelhouse_manifest(wheelhouse, manifest)
    project_wheel = _project_wheel_path(wheelhouse, manifest)
    first = inspect_project_wheel(project_wheel)
    second = inspect_project_wheel(second_project_wheel)
    if first["wheel_sha256"] != second["wheel_sha256"]:
        raise ValueError("the two independent project-wheel builds are not byte-identical")
    if first != {**second, "wheel_filename": first["wheel_filename"]}:
        raise ValueError("the two project-wheel inspections differ")

    versions = _version_map(lock["distribution_versions"], "distribution_versions")
    inventory = installed_inventory(versions)
    verify_project_direct_url(
        project_wheel=project_wheel,
        expected_wheel_sha256=str(first["wheel_sha256"]),
    )
    _verify_external_install_origins(versions)
    _pip_check(Path(sys.executable))
    installed_source = _installed_project_source()
    if installed_source["python_source_sha256"] != first["python_source_sha256"]:
        raise ValueError("installed project Python sources differ from the project wheel")
    _verify_project_import_origins()
    native = _native_extension()
    if native["sha256"] != first["native_extension_sha256"]:
        raise ValueError("installed lac_minorminer extension differs from the project wheel")
    registry = _checkpoint_registry()
    registry_native = registry.get("native_artifacts", {})
    if not isinstance(registry_native, dict):
        raise ValueError("checkpoint registry native_artifacts is malformed")
    core = registry_native.get("lac_minorminer._core")
    if not isinstance(core, dict) or core.get("sha256") != native["sha256"]:
        raise ValueError("checkpoint registry does not bind the loaded native extension")

    contract_files = {
        "runtime_lock": file_sha256(_require_regular_file(runtime_lock_path, "runtime lock")),
        "runtime_requirements": file_sha256(
            _require_regular_file(runtime_requirements, "runtime requirements")
        ),
        "build_requirements": file_sha256(
            _require_regular_file(build_requirements, "build requirements")
        ),
        "apptainer_definition": file_sha256(
            _require_regular_file(definition, "Apptainer definition")
        ),
        "wheelhouse_manifest_file": file_sha256(wheelhouse_manifest_path),
    }
    return {
        "build_environment": {
            "build_programs": {
                name: _command_identity(name) for name in ("c++", "cmake", "ninja")
            },
            "machine": observed_build_platform["machine"],
            "python_implementation": platform.python_implementation(),
            "python_executable_sha256": file_sha256(Path(sys.executable).resolve(strict=True)),
            "python_version": platform.python_version(),
            "system": observed_build_platform["system"],
            "thread_controls": observed_threads,
        },
        "checkpoint_runtime_registry": registry,
        "checkpoint_runtime_registry_sha256": canonical_digest(registry),
        "contract_files": contract_files,
        "critical_launcher_helper_sha256": _critical_source_files(source),
        "installed_inventory": inventory,
        "project_wheel": first,
        "reproducible_project_wheel_build": {
            "build_count": 2,
            "byte_identical": True,
            "first_sha256": first["wheel_sha256"],
            "second_sha256": second["wheel_sha256"],
        },
        "runtime_lock": lock,
        "schema": "isingfold.runtime-build-receipt",
        "schema_version": 1,
        "source_inventory": source,
        "wheelhouse_manifest": manifest,
    }


def _validate_build_receipt(document: Mapping[str, object]) -> dict[str, Any]:
    receipt = _strict_object(
        dict(document),
        {
            "build_environment",
            "checkpoint_runtime_registry",
            "checkpoint_runtime_registry_sha256",
            "contract_files",
            "critical_launcher_helper_sha256",
            "installed_inventory",
            "project_wheel",
            "receipt_sha256",
            "reproducible_project_wheel_build",
            "runtime_lock",
            "schema",
            "schema_version",
            "source_inventory",
            "wheelhouse_manifest",
        },
        "build receipt",
    )
    if (
        receipt["schema"] != "isingfold.runtime-build-receipt"
        or receipt["schema_version"] != 1
    ):
        raise ValueError("build receipt schema is unsupported")
    build_environment = _strict_object(
        receipt["build_environment"],
        {
            "build_programs",
            "machine",
            "python_executable_sha256",
            "python_implementation",
            "python_version",
            "system",
            "thread_controls",
        },
        "build environment",
    )
    _require_sha256(
        build_environment["python_executable_sha256"],
        "build Python executable digest",
    )
    programs = _strict_object(
        build_environment["build_programs"], {"c++", "cmake", "ninja"}, "build programs"
    )
    for name, value in programs.items():
        program = _strict_object(
            value, {"executable_sha256", "version_line"}, f"build program {name}"
        )
        _require_sha256(program["executable_sha256"], f"build program {name} digest")
        if type(program["version_line"]) is not str or not program["version_line"]:
            raise ValueError(f"build program {name} has no version identity")
    _require_sha256(
        receipt["checkpoint_runtime_registry_sha256"],
        "checkpoint runtime registry digest",
    )
    registry = receipt["checkpoint_runtime_registry"]
    if canonical_digest(registry) != receipt["checkpoint_runtime_registry_sha256"]:
        raise ValueError("build receipt checkpoint registry digest is invalid")
    reproducible = _strict_object(
        receipt["reproducible_project_wheel_build"],
        {"build_count", "byte_identical", "first_sha256", "second_sha256"},
        "reproducible project wheel build",
    )
    if reproducible["build_count"] != 2 or reproducible["byte_identical"] is not True:
        raise ValueError("build receipt does not prove two byte-identical project wheels")
    first = _require_sha256(reproducible["first_sha256"], "first project wheel digest")
    second = _require_sha256(reproducible["second_sha256"], "second project wheel digest")
    project = receipt["project_wheel"]
    if not isinstance(project, dict):
        raise ValueError("build receipt project_wheel is malformed")
    if first != second or project.get("wheel_sha256") != first:
        raise ValueError("build receipt project wheel reproducibility evidence is inconsistent")
    inventory = receipt["installed_inventory"]
    if not isinstance(inventory, dict):
        raise ValueError("build receipt installed inventory is malformed")
    _require_sha256(inventory.get("installation_sha256"), "installation digest")
    source = receipt["source_inventory"]
    if not isinstance(source, dict):
        raise ValueError("build receipt source inventory is malformed")
    _require_sha256(source.get("source_sha256"), "source inventory digest")
    manifest = receipt["wheelhouse_manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("build receipt wheelhouse manifest is malformed")
    _require_sha256(manifest.get("manifest_sha256"), "wheelhouse manifest digest")
    return receipt


def verify_runtime(
    *,
    runtime_lock_path: Path,
    build_receipt_path: Path,
    expected_build_receipt_sha256: str,
    source_root: Path,
    expected_source_sha256: str,
    wheelhouse_manifest_path: Path,
    expected_wheelhouse_manifest_sha256: str,
    project_wheel: Path,
    expected_project_wheel_sha256: str,
    expected_native_extension_sha256: str,
    expected_installation_sha256: str,
    expected_python_executable_sha256: str,
    execution_mode: str,
    accelerator: str,
    image: Path | None = None,
    expected_image_sha256: str | None = None,
    wheelhouse: Path | None = None,
) -> dict[str, object]:
    """Validate a live Apollo venv or Goose image against one build receipt."""

    lock = read_runtime_lock(runtime_lock_path)
    receipt = _validate_build_receipt(
        load_receipt(build_receipt_path, expected_build_receipt_sha256)
    )
    if execution_mode not in {"pinned-venv", "apptainer"}:
        raise ValueError("execution mode must be pinned-venv or apptainer")
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError("accelerator must be cpu or gpu")
    if (
        accelerator == "gpu"
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != lock["accelerator"]["cublas_workspace_config"]
    ):
        raise ValueError("GPU runtime CUBLAS_WORKSPACE_CONFIG differs from runtime lock")

    expected_source = _require_sha256(expected_source_sha256, "expected source digest")
    expected_wheel = _require_sha256(
        expected_project_wheel_sha256, "expected project wheel digest"
    )
    expected_native = _require_sha256(
        expected_native_extension_sha256, "expected native extension digest"
    )
    expected_installation = _require_sha256(
        expected_installation_sha256, "expected installation digest"
    )
    expected_runtime_python = _require_sha256(
        expected_python_executable_sha256, "expected runtime Python executable digest"
    )
    current_source = source_inventory(source_root)
    if current_source != receipt["source_inventory"]:
        raise ValueError("staged scientific or launcher/helper source differs from build receipt")
    if current_source["source_sha256"] != expected_source:
        raise ValueError("staged source differs from its external commitment")
    if _critical_source_files(current_source) != receipt["critical_launcher_helper_sha256"]:
        raise ValueError("critical launcher/helper source differs from build receipt")

    if lock != receipt["runtime_lock"]:
        raise ValueError("runtime lock differs from build receipt")
    contract_files = receipt["contract_files"]
    if not isinstance(contract_files, dict):
        raise ValueError("build receipt contract_files is malformed")
    source_runtime = source_root / "runtime" / "isingfold-training"
    observed_contract_files = {
        "runtime_lock": file_sha256(runtime_lock_path),
        "runtime_requirements": file_sha256(
            source_runtime / "requirements-linux-x86_64-py312-cu128.lock"
        ),
        "build_requirements": file_sha256(
            source_runtime / "requirements-build-linux-x86_64-py312.lock"
        ),
        "apptainer_definition": file_sha256(
            source_runtime / "isingfold-training.def"
        ),
        "wheelhouse_manifest_file": file_sha256(wheelhouse_manifest_path),
    }
    if observed_contract_files != contract_files:
        raise ValueError("runtime contract file bytes differ from build receipt")

    manifest = _load_wheelhouse_manifest(
        wheelhouse_manifest_path, expected_wheelhouse_manifest_sha256
    )
    if manifest != receipt["wheelhouse_manifest"]:
        raise ValueError("wheelhouse manifest differs from build receipt")
    if wheelhouse is not None:
        verify_wheelhouse_manifest(wheelhouse, manifest)
    project_report = inspect_project_wheel(project_wheel)
    if project_report != receipt["project_wheel"]:
        raise ValueError("project wheel differs from build receipt")
    if project_report["wheel_sha256"] != expected_wheel:
        raise ValueError("project wheel differs from its external commitment")
    if project_report["native_extension_sha256"] != expected_native:
        raise ValueError("project wheel native extension differs from its external commitment")

    python_path = Path(sys.executable)
    _require_executable(python_path, "runtime Python")
    python_lock = lock["python"]
    platform_lock = lock["platform"]
    if platform.python_implementation() != python_lock["implementation"]:
        raise ValueError("runtime Python implementation differs from runtime lock")
    if platform.python_version() != python_lock["version"]:
        raise ValueError("runtime Python patch differs from runtime lock")
    observed_platform = {"machine": platform.machine(), "system": platform.system()}
    if observed_platform != platform_lock:
        raise ValueError("runtime platform differs from runtime lock")
    thread_lock = lock["thread_controls"]
    observed_threads = {name: os.environ.get(name) for name in _THREAD_NAMES}
    if observed_threads != thread_lock:
        raise ValueError(
            f"runtime thread environment differs from one-thread lock: {observed_threads!r}"
        )

    build_environment = receipt["build_environment"]
    _require_sha256(
        build_environment.get("python_executable_sha256"),
        "build Python executable digest",
    )
    observed_python_binary = file_sha256(python_path.resolve(strict=True))
    if observed_python_binary != expected_runtime_python:
        raise ValueError("runtime Python executable differs from its external commitment")
    expected_build_identity = {
        "machine": platform_lock["machine"],
        "python_implementation": python_lock["implementation"],
        "python_version": python_lock["version"],
        "system": platform_lock["system"],
        "thread_controls": thread_lock,
    }
    observed_build_identity = {
        name: build_environment.get(name) for name in expected_build_identity
    }
    if observed_build_identity != expected_build_identity:
        raise ValueError("wheel-build interpreter/platform contract differs from runtime lock")

    versions = _version_map(lock["distribution_versions"], "distribution_versions")
    observed_inventory = installed_inventory(versions)
    if observed_inventory != receipt["installed_inventory"]:
        raise ValueError("installed distribution files/native libraries differ from build receipt")
    if observed_inventory["installation_sha256"] != expected_installation:
        raise ValueError("installed runtime differs from its external commitment")
    verify_project_direct_url(
        project_wheel=project_wheel,
        expected_wheel_sha256=expected_wheel,
    )
    _verify_external_install_origins(versions)
    _pip_check(python_path)

    installed_source = _installed_project_source()
    if installed_source["python_source_sha256"] != project_report["python_source_sha256"]:
        raise ValueError("installed project source differs from the attested project wheel")
    _verify_project_import_origins()
    native = _native_extension()
    if native["sha256"] != expected_native:
        raise ValueError("loaded lac_minorminer extension differs from the external commitment")
    registry = _checkpoint_registry()
    if registry != receipt["checkpoint_runtime_registry"]:
        raise ValueError("registered checkpoint source closure differs from build receipt")
    if canonical_digest(registry) != receipt["checkpoint_runtime_registry_sha256"]:
        raise ValueError("registered checkpoint source-closure digest differs from build receipt")

    image_record: dict[str, object] | None = None
    if execution_mode == "apptainer":
        if image is None or expected_image_sha256 is None:
            raise ValueError("apptainer execution requires image path and external digest")
        image_path = _require_regular_file(image, "Apptainer image")
        expected_image = _require_sha256(expected_image_sha256, "expected image digest")
        observed_image = file_sha256(image_path)
        if observed_image != expected_image:
            raise ValueError("Apptainer image differs from its external commitment")
        image_record = {
            "path": str(image_path),
            "sha256": observed_image,
            "size": image_path.stat().st_size,
        }
    elif image is not None or expected_image_sha256 is not None:
        raise ValueError("non-Apptainer execution cannot claim a container image")

    cuda = (
        cuda_smoke_check(python_path, environment=os.environ)
        if accelerator == "gpu"
        else {"status": "not-required-for-cpu-stage"}
    )
    return {
        "accelerator": accelerator,
        "build_receipt_file_sha256": expected_build_receipt_sha256,
        "checkpoint_runtime_registry_sha256": canonical_digest(registry),
        "cuda": cuda,
        "execution_mode": execution_mode,
        "execution_site": {
            "hostname": platform.node(),
            "kernel_release": platform.release(),
            "libc": list(platform.libc_ver()),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "image": image_record,
        "installation_sha256": observed_inventory["installation_sha256"],
        "native_extension_sha256": native["sha256"],
        "pip_check": "ok",
        "platform": observed_platform,
        "project_wheel_sha256": project_report["wheel_sha256"],
        "python_executable_sha256": observed_python_binary,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "schema": "isingfold.runtime-validation-receipt",
        "schema_version": 1,
        "source_sha256": current_source["source_sha256"],
        "thread_controls": observed_threads,
        "wheelhouse_manifest_file_sha256": expected_wheelhouse_manifest_sha256,
    }


def _add_path_argument(parser: argparse.ArgumentParser, flag: str, **kwargs: object) -> None:
    parser.add_argument(flag, required=True, type=Path, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser("source-inventory", allow_abbrev=False)
    _add_path_argument(source, "--source-root")

    make_manifest = subparsers.add_parser("make-wheelhouse-manifest", allow_abbrev=False)
    _add_path_argument(make_manifest, "--wheelhouse")
    _add_path_argument(make_manifest, "--out")

    verify_manifest = subparsers.add_parser("verify-wheelhouse", allow_abbrev=False)
    _add_path_argument(verify_manifest, "--wheelhouse")
    _add_path_argument(verify_manifest, "--manifest")
    verify_manifest.add_argument("--expected-manifest-file-sha256")

    project_path = subparsers.add_parser("project-wheel-path", allow_abbrev=False)
    _add_path_argument(project_path, "--wheelhouse")
    _add_path_argument(project_path, "--manifest")
    project_path.add_argument("--expected-manifest-file-sha256")

    receipt = subparsers.add_parser("verify-receipt", allow_abbrev=False)
    _add_path_argument(receipt, "--path")
    receipt.add_argument("--expected-file-sha256", required=True)
    receipt.add_argument("--schema", required=True)
    receipt.add_argument("--schema-version", required=True, type=int)

    build = subparsers.add_parser("create-build-receipt", allow_abbrev=False)
    _add_path_argument(build, "--source-root")
    _add_path_argument(build, "--runtime-lock")
    _add_path_argument(build, "--runtime-requirements")
    _add_path_argument(build, "--build-requirements")
    _add_path_argument(build, "--definition")
    _add_path_argument(build, "--wheelhouse")
    _add_path_argument(build, "--wheelhouse-manifest")
    _add_path_argument(build, "--second-project-wheel")
    _add_path_argument(build, "--out")

    validate = subparsers.add_parser("verify-runtime", allow_abbrev=False)
    _add_path_argument(validate, "--runtime-lock")
    _add_path_argument(validate, "--build-receipt")
    validate.add_argument("--expected-build-receipt-sha256", required=True)
    _add_path_argument(validate, "--source-root")
    validate.add_argument("--expected-source-sha256", required=True)
    _add_path_argument(validate, "--wheelhouse-manifest")
    validate.add_argument("--expected-wheelhouse-manifest-sha256", required=True)
    _add_path_argument(validate, "--project-wheel")
    validate.add_argument("--expected-project-wheel-sha256", required=True)
    validate.add_argument("--expected-native-extension-sha256", required=True)
    validate.add_argument("--expected-installation-sha256", required=True)
    validate.add_argument("--expected-python-executable-sha256", required=True)
    validate.add_argument(
        "--execution-mode", choices=("pinned-venv", "apptainer"), required=True
    )
    validate.add_argument("--accelerator", choices=("cpu", "gpu"), required=True)
    validate.add_argument("--image", type=Path)
    validate.add_argument("--expected-image-sha256")
    validate.add_argument("--wheelhouse", type=Path)
    validate.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source-inventory":
        print(canonical_json(source_inventory(args.source_root)).decode("utf-8"))
        return 0
    if args.command == "make-wheelhouse-manifest":
        _write_json_exclusive(args.out, wheelhouse_manifest(args.wheelhouse))
        return 0
    if args.command == "verify-wheelhouse":
        manifest = _load_wheelhouse_manifest(
            args.manifest, args.expected_manifest_file_sha256
        )
        verify_wheelhouse_manifest(args.wheelhouse, manifest)
        return 0
    if args.command == "project-wheel-path":
        manifest = _load_wheelhouse_manifest(
            args.manifest, args.expected_manifest_file_sha256
        )
        verify_wheelhouse_manifest(args.wheelhouse, manifest)
        print(_project_wheel_path(args.wheelhouse, manifest))
        return 0
    if args.command == "verify-receipt":
        document = load_receipt(args.path, args.expected_file_sha256)
        if document.get("schema") != args.schema or document.get("schema_version") != args.schema_version:
            raise ValueError("receipt schema differs from the requested contract")
        return 0
    if args.command == "create-build-receipt":
        payload = create_build_receipt(
            source_root=args.source_root,
            runtime_lock_path=args.runtime_lock,
            runtime_requirements=args.runtime_requirements,
            build_requirements=args.build_requirements,
            definition=args.definition,
            wheelhouse=args.wheelhouse,
            wheelhouse_manifest_path=args.wheelhouse_manifest,
            second_project_wheel=args.second_project_wheel,
        )
        publish_receipt(args.out, payload)
        return 0
    if args.command == "verify-runtime":
        payload = verify_runtime(
            runtime_lock_path=args.runtime_lock,
            build_receipt_path=args.build_receipt,
            expected_build_receipt_sha256=args.expected_build_receipt_sha256,
            source_root=args.source_root,
            expected_source_sha256=args.expected_source_sha256,
            wheelhouse_manifest_path=args.wheelhouse_manifest,
            expected_wheelhouse_manifest_sha256=(
                args.expected_wheelhouse_manifest_sha256
            ),
            project_wheel=args.project_wheel,
            expected_project_wheel_sha256=args.expected_project_wheel_sha256,
            expected_native_extension_sha256=args.expected_native_extension_sha256,
            expected_installation_sha256=args.expected_installation_sha256,
            expected_python_executable_sha256=(
                args.expected_python_executable_sha256
            ),
            execution_mode=args.execution_mode,
            accelerator=args.accelerator,
            image=args.image,
            expected_image_sha256=args.expected_image_sha256,
            wheelhouse=args.wheelhouse,
        )
        if args.out is None:
            print(canonical_json(payload).decode("utf-8"))
        else:
            publish_receipt(args.out, payload)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

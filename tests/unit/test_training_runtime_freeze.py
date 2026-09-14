"""Fail-closed contracts for the paper-eligible IsingFold training runtime."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "runtime" / "isingfold-training"
RUNTIME_LOCK = RUNTIME_ROOT / "runtime-lock.json"
RUNTIME_REQUIREMENTS = RUNTIME_ROOT / "requirements-linux-x86_64-py312-cu128.lock"
BUILD_REQUIREMENTS = RUNTIME_ROOT / "requirements-build-linux-x86_64-py312.lock"
DEFINITION = RUNTIME_ROOT / "isingfold-training.def"
VERIFIER = ROOT / "scripts" / "verify_isingfold_runtime.py"
SHELL_GUARD = ROOT / "scripts" / "verify_runtime_source.sh"
APOLLO_BUILDER = ROOT / "scripts" / "apollo_build_isingfold_runtime.sh"
GOOSE_BUILDER = ROOT / "scripts" / "goose_build_isingfold_runtime.sbatch"
GOOSE_VERIFIER = ROOT / "scripts" / "goose_verify_isingfold_runtime.sbatch"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_isingfold_runtime", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_project_wheel(path: Path, *, native: bytes = b"native-v1") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("isingfold/__init__.py", '"""fixture"""\n')
        archive.writestr("isingfold/rl/checkpoint.py", "VALUE = 1\n")
        archive.writestr("lac_minorminer/__init__.py", "from . import _core\n")
        archive.writestr(
            "lac_minorminer/_core.cpython-312-x86_64-linux-gnu.so",
            native,
        )
        archive.writestr(
            "isingfold-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: isingfold\nVersion: 0.1.0\n",
        )
        archive.writestr("isingfold-0.1.0.dist-info/RECORD", "")


def test_runtime_assets_pin_exact_linux_cuda_environment() -> None:
    lock = json.loads(RUNTIME_LOCK.read_text(encoding="utf-8"))
    assert lock["schema"] == "isingfold.training-runtime-lock"
    assert lock["schema_version"] == 1
    assert lock["platform"] == {"machine": "x86_64", "system": "Linux"}
    assert lock["python"] == {
        "container_executable": "/opt/isingfold/venv/bin/python",
        "implementation": "CPython",
        "version": "3.12.3",
    }
    assert lock["accelerator"]["torch"] == "2.11.0+cu128"
    assert lock["accelerator"]["cuda_runtime"] == "12.8"
    assert lock["accelerator"]["cublas_workspace_config"] == ":4096:8"
    assert lock["thread_controls"] == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert lock["project"] == {
        "distribution": "isingfold",
        "native_extension_glob": "lac_minorminer/_core*.so",
        "version": "0.1.0",
    }
    versions = lock["distribution_versions"]
    assert versions["isingfold"] == "0.1.0"
    assert versions["torch"] == "2.11.0+cu128"
    assert versions["minorminer"] == "0.2.22"
    assert versions["numpy"] == "2.4.4"
    assert versions["scipy"] == "1.18.0"
    assert all(type(value) is str and value for value in versions.values())

    runtime_text = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    assert "--only-binary=:all:" in runtime_text
    assert "--require-hashes" in runtime_text
    assert "https://download.pytorch.org/whl/cu128" in runtime_text
    for name, version in versions.items():
        if name == "isingfold":
            continue
        requirement_name = name.replace("_", "-")
        assert f"{requirement_name}=={version}" in runtime_text
    requirement_lines = [
        line
        for line in runtime_text.splitlines()
        if line and not line.startswith(("#", "--", " "))
    ]
    assert requirement_lines
    assert all("==" in line for line in requirement_lines)
    assert runtime_text.count("--hash=sha256:") == len(requirement_lines)

    build_text = BUILD_REQUIREMENTS.read_text(encoding="utf-8")
    assert "--only-binary=:all:" in build_text
    assert "--require-hashes" in build_text
    for requirement in (
        "cmake==4.4.3",
        "ninja==1.13.0",
        "pybind11==3.1.0",
        "scikit-build-core==1.0.3",
    ):
        assert requirement in build_text


def test_definition_is_offline_and_installs_the_prebuilt_project_wheel() -> None:
    text = DEFINITION.read_text(encoding="utf-8")
    assert (
        "docker.io/library/python@sha256:"
        "afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63"
    ) in text
    assert ":3.12.3-slim-bookworm@sha256:" not in text
    assert "--no-index" in text
    assert "PIP_NO_INDEX=1" in text
    assert "--require-hashes" in text
    assert "/mnt/isingfold-wheelhouse" in text
    assert "pip install -e" not in text
    assert "git clone" not in text
    assert "curl " not in text
    assert "wget " not in text
    assert "MKL_NUM_THREADS=1" in text
    assert "OPENBLAS_NUM_THREADS=1" in text
    assert "/opt/isingfold/venv/bin/python" in text


def test_source_inventory_is_complete_deterministic_and_symlink_closed(
    tmp_path: Path,
) -> None:
    runtime = _load_verifier()
    source = tmp_path / "source"
    (source / "src" / "isingfold").mkdir(parents=True)
    (source / "src" / "lac_minorminer").mkdir(parents=True)
    (source / "cpp" / "src").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='isingfold'\n")
    (source / "CMakeLists.txt").write_text("project(isingfold)\n")
    (source / "README.md").write_text("# Fixture\n")
    (source / "src" / "isingfold" / "__init__.py").write_text("VALUE = 1\n")
    (source / "src" / "lac_minorminer" / "__init__.py").write_text("VALUE = 2\n")
    (source / "cpp" / "src" / "core.cpp").write_text("int value = 1;\n")
    (source / "scripts" / "apollo_cell.sh").write_text("#!/bin/sh\n")
    cache = source / "src" / "isingfold" / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")

    first = runtime.source_inventory(source)
    second = runtime.source_inventory(source)

    assert first == second
    paths = {row["path"] for row in first["files"]}
    assert "src/isingfold/__init__.py" in paths
    assert "src/lac_minorminer/__init__.py" in paths
    assert "cpp/src/core.cpp" in paths
    assert "scripts/apollo_cell.sh" in paths
    assert all("__pycache__" not in path for path in paths)

    (source / "cpp" / "src" / "core.cpp").write_text("int value = 2;\n")
    assert runtime.source_inventory(source)["source_sha256"] != first["source_sha256"]

    (source / "src" / "isingfold" / "alias.py").symlink_to("__init__.py")
    with pytest.raises(ValueError, match="symbolic link"):
        runtime.source_inventory(source)


def test_project_wheel_inspection_binds_native_extension_and_source_bytes(
    tmp_path: Path,
) -> None:
    runtime = _load_verifier()
    wheel = tmp_path / "isingfold-0.1.0-cp312-cp312-linux_x86_64.whl"
    _write_project_wheel(wheel)

    report = runtime.inspect_project_wheel(wheel)

    assert report["wheel_sha256"] == _sha256(wheel)
    assert report["native_extension_path"].startswith("lac_minorminer/_core")
    assert report["native_extension_sha256"] == hashlib.sha256(b"native-v1").hexdigest()
    assert report["python_source_count"] == 3
    assert len(report["python_source_sha256"]) == 64

    second = tmp_path / "isingfold-0.1.0-second.whl"
    _write_project_wheel(second, native=b"native-v2")
    assert (
        runtime.inspect_project_wheel(second)["native_extension_sha256"]
        != report["native_extension_sha256"]
    )


def test_wheelhouse_manifest_rejects_missing_extra_and_mutated_files(
    tmp_path: Path,
) -> None:
    runtime = _load_verifier()
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    project = wheelhouse / "isingfold-0.1.0-cp312-cp312-linux_x86_64.whl"
    dependency = wheelhouse / "networkx-3.6.1-py3-none-any.whl"
    _write_project_wheel(project)
    dependency.write_bytes(b"networkx wheel")
    manifest = runtime.wheelhouse_manifest(wheelhouse)

    runtime.verify_wheelhouse_manifest(wheelhouse, manifest)

    dependency.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="wheelhouse bytes"):
        runtime.verify_wheelhouse_manifest(wheelhouse, manifest)
    dependency.write_bytes(b"networkx wheel")
    (wheelhouse / "extra.whl").write_bytes(b"extra")
    with pytest.raises(ValueError, match="wheel set"):
        runtime.verify_wheelhouse_manifest(wheelhouse, manifest)


def test_runtime_receipt_is_self_digested_and_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    runtime = _load_verifier()
    output = tmp_path / "receipt.json"
    payload = {"schema": "fixture", "schema_version": 1, "value": 7}

    runtime.publish_receipt(output, payload)
    document = json.loads(output.read_text(encoding="utf-8"))

    digest = document.pop("receipt_sha256")
    assert digest == runtime.canonical_digest(document)
    with pytest.raises(FileExistsError):
        runtime.publish_receipt(output, payload)


class _FakeDistribution:
    def __init__(
        self,
        root: Path,
        name: str,
        version: str,
        files: tuple[str, ...],
        *,
        direct_url: str | None = None,
    ) -> None:
        self.root = root
        self.metadata = {"Name": name}
        self.version = version
        self.files = tuple(PurePosixPath(path) for path in files)
        self._direct_url = direct_url

    def locate_file(self, path: PurePosixPath) -> Path:
        return self.root / path

    def read_text(self, filename: str) -> str | None:
        return self._direct_url if filename == "direct_url.json" else None


def test_installed_inventory_binds_every_distribution_file_and_native_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _load_verifier()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "module.py").write_bytes(b"VALUE = 1\n")
    (tmp_path / "alpha" / "native.so").write_bytes(b"native-alpha")
    (tmp_path / "alpha-1.dist-info").mkdir()
    (tmp_path / "alpha-1.dist-info" / "RECORD").write_text("variable install record")
    alpha = _FakeDistribution(
        tmp_path,
        "Alpha",
        "1.0",
        (
            "alpha/module.py",
            "alpha/native.so",
            "alpha-1.dist-info/RECORD",
        ),
    )
    monkeypatch.setattr(runtime.importlib_metadata, "distributions", lambda: (alpha,))
    monkeypatch.setattr(runtime.importlib_metadata, "distribution", lambda _: alpha)

    inventory = runtime.installed_inventory({"alpha": "1.0"})

    assert inventory["distribution_count"] == 1
    assert inventory["file_count"] == 2
    assert inventory["native_library_count"] == 1
    assert inventory["distribution_summaries"][0]["name"] == "alpha"
    assert inventory["native_libraries"][0]["path"] == "alpha/native.so"
    assert len(inventory["installation_sha256"]) == 64

    with pytest.raises(ValueError, match="versions"):
        runtime.installed_inventory({"alpha": "2.0"})
    with pytest.raises(ValueError, match="distribution set"):
        runtime.installed_inventory({"alpha": "1.0", "missing": "1.0"})


def test_project_install_must_come_from_the_exact_attested_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _load_verifier()
    wheel = tmp_path / "isingfold-0.1.0-cp312-cp312-linux_x86_64.whl"
    _write_project_wheel(wheel)
    wheel_digest = _sha256(wheel)
    project = _FakeDistribution(
        tmp_path,
        "isingfold",
        "0.1.0",
        (),
        direct_url=json.dumps(
            {
                "archive_info": {
                    "hash": f"sha256={wheel_digest}",
                    "hashes": {"sha256": wheel_digest},
                },
                "url": wheel.as_uri(),
            }
        ),
    )
    monkeypatch.setattr(runtime.importlib_metadata, "distribution", lambda _: project)

    runtime.verify_project_direct_url(
        project_wheel=wheel,
        expected_wheel_sha256=wheel_digest,
    )

    with pytest.raises(ValueError, match="wheel digest"):
        runtime.verify_project_direct_url(
            project_wheel=wheel,
            expected_wheel_sha256="0" * 64,
        )
    project._direct_url = json.dumps(
        {"dir_info": {"editable": True}, "url": tmp_path.as_uri()}
    )
    with pytest.raises(ValueError, match="editable"):
        runtime.verify_project_direct_url(
            project_wheel=wheel,
            expected_wheel_sha256=wheel_digest,
        )


def test_runtime_rejects_project_imports_outside_the_attested_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _load_verifier()
    install = tmp_path / "site-packages"
    install.mkdir()
    origins: dict[str, Path] = {}
    for index, name in enumerate(
        ("isingfold", "isingfold.rl.cli", "lac_minorminer", "lac_minorminer._core")
    ):
        path = install / f"module-{index}.py"
        path.write_text(f"# {name}\n")
        origins[name] = path
    distribution = _FakeDistribution(
        install,
        "isingfold",
        "0.1.0",
        tuple(path.name for path in origins.values()),
    )
    monkeypatch.setattr(runtime.importlib_metadata, "distribution", lambda _: distribution)
    monkeypatch.setattr(
        runtime.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(origins[name])),
    )

    runtime._verify_project_import_origins()

    injected = tmp_path / "injected.py"
    injected.write_text("# injected\n")
    origins["isingfold.rl.cli"] = injected
    with pytest.raises(ValueError, match="outside the attested wheel"):
        runtime._verify_project_import_origins()

    origins["isingfold.rl.cli"] = install / "unregistered.py"
    origins["isingfold.rl.cli"].write_text("# unregistered\n")
    with pytest.raises(ValueError, match="not a registered"):
        runtime._verify_project_import_origins()


def test_self_digested_receipt_loader_rejects_any_mutation(tmp_path: Path) -> None:
    runtime = _load_verifier()
    receipt = tmp_path / "receipt.json"
    runtime.publish_receipt(receipt, {"schema": "fixture", "schema_version": 1})
    expected = _sha256(receipt)
    assert runtime.load_receipt(receipt, expected)["schema"] == "fixture"

    document = json.loads(receipt.read_text(encoding="utf-8"))
    document["schema_version"] = 2
    receipt.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="file digest"):
        runtime.load_receipt(receipt, expected)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"first","schema":"second"}\n')
    with pytest.raises(ValueError, match="repeats key"):
        runtime._read_json(duplicate.resolve(), "duplicate fixture")


def test_runtime_lock_parser_rejects_unknown_fields_and_unpinned_threads(
    tmp_path: Path,
) -> None:
    runtime = _load_verifier()
    accepted = runtime.read_runtime_lock(RUNTIME_LOCK)
    assert accepted["python"]["version"] == "3.12.3"

    mutated = dict(accepted)
    mutated["unexpected"] = True
    invalid = tmp_path / "runtime-lock.json"
    invalid.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        runtime.read_runtime_lock(invalid.resolve())

    mutated = dict(accepted)
    mutated["thread_controls"] = {**accepted["thread_controls"], "OMP_NUM_THREADS": "2"}
    invalid.unlink()
    invalid.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ValueError, match="thread"):
        runtime.read_runtime_lock(invalid.resolve())


def test_cuda_smoke_check_requires_forward_backward_and_exact_resume(
    tmp_path: Path,
) -> None:
    runtime = _load_verifier()
    fake_python = tmp_path / "python"
    captured = tmp_path / "program.py"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cp \"$2\" \"$CAPTURED_PROGRAM\"\n"
        "printf '%s\\n' '{\"capability\":[12,0],"
        "\"cublas_workspace_config\":\":4096:8\","
        "\"cuda_driver_api_version\":12080,\"cuda_runtime\":\"12.8\","
        "\"device_count\":1,\"device_name\":\"fixture\","
        "\"replay_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}'\n"
    )
    fake_python.chmod(0o755)
    monkey_env = {
        **os.environ,
        "CAPTURED_PROGRAM": str(captured),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    report = runtime.cuda_smoke_check(fake_python, environment=monkey_env)

    assert report["cuda_runtime"] == "12.8"
    program = captured.read_text(encoding="utf-8")
    assert ".backward()" in program
    assert "optimizer.state_dict()" in program
    assert "torch.cuda.get_rng_state_all()" in program
    assert "torch.set_rng_state" in program
    assert "torch.equal" in program
    assert "cuDriverGetVersion" in program

    fake_python.write_text("#!/usr/bin/env bash\nexit 7\n")
    fake_python.chmod(0o755)
    with pytest.raises(RuntimeError, match="CUDA smoke"):
        runtime.cuda_smoke_check(fake_python, environment=monkey_env)


def test_runtime_shell_entrypoints_fail_closed_on_wrong_host_context() -> None:
    apollo = subprocess.run(
        ["bash", str(APOLLO_BUILDER)],
        cwd=ROOT,
        env={**os.environ, "SLURM_JOB_ID": "123"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert apollo.returncode == 69
    assert "Apollo" in apollo.stderr and "Slurm" in apollo.stderr

    for launcher in (GOOSE_BUILDER, GOOSE_VERIFIER):
        environment = os.environ.copy()
        for name in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURMD_NODENAME"):
            environment.pop(name, None)
        goose = subprocess.run(
            ["bash", str(launcher)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert goose.returncode == 69
        assert "Slurm" in goose.stderr


def test_goose_runtime_launchers_use_real_site_binaries_and_generic_gpu() -> None:
    for launcher in (GOOSE_BUILDER, GOOSE_VERIFIER):
        text = launcher.read_text(encoding="utf-8")
        assert "/opt/slurm/bin/srun" in text
        assert "/usr/bin/apptainer" in text
        assert "/opt/apptainer/bin/apptainer" not in text
        assert "#SBATCH --partition=gpu" in text
        assert "#SBATCH --gres=gpu:1" in text
        assert "SLURM_JOB_ID" in text
        assert "SLURM_JOB_PARTITION" in text
        assert "SLURMD_NODENAME" in text
        assert "ISINGFOLD_RUNTIME_EXPECTED_HOST_PYTHON_SHA256" in text
        assert "--nv" in text


def test_goose_builds_the_native_wheel_twice_then_builds_image_offline() -> None:
    builder = GOOSE_BUILDER.read_text(encoding="utf-8")
    assert "isingfold-wheel-builder.def" in builder
    assert "isingfold-training.def" in builder
    assert builder.count('"$srun_bin"') >= 4
    assert "ISINGFOLD_RUNTIME_EXPECTED_SRUN_SHA256" in builder
    assert "ISINGFOLD_RUNTIME_EXPECTED_APPTAINER_SHA256" in builder
    assert ': "${ISINGFOLD_RUNTIME_SOURCE_ROOT:?set ISINGFOLD_RUNTIME_SOURCE_ROOT}"' in builder
    assert 'source_root=$(cd -- "$ISINGFOLD_RUNTIME_SOURCE_ROOT" && pwd -P)' in builder
    assert 'dirname -- "${BASH_SOURCE[0]}"' not in builder
    assert "--fakeroot" in builder
    assert "--bind" in builder
    assert ":ro" in builder
    assert "verify-runtime" in builder
    assert "--accelerator gpu" in builder
    assert "runtime-validation.json" in builder

    definition = (RUNTIME_ROOT / "isingfold-wheel-builder.def").read_text(
        encoding="utf-8"
    )
    assert (
        "docker.io/library/python@sha256:"
        "afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63"
        in definition
    )
    assert ":3.12.3-slim-bookworm@sha256:" not in definition
    assert definition.count("pip wheel") == 2
    assert "SOURCE_DATE_EPOCH=0" in definition
    assert "CMAKE_BUILD_PARALLEL_LEVEL=1" in definition
    assert "cmp " in definition
    assert "source-before.json" in definition
    assert "source-after.json" in definition
    assert (
        "https://snapshot.debian.org/archive/debian/20240513T000000Z "
        "bookworm main"
    ) in definition
    assert (
        "https://snapshot.debian.org/archive/debian-security/20240513T000000Z "
        "bookworm-security main"
    ) in definition
    assert "--require-hashes" in definition
    assert "-DBUILD_TESTING=ON" in definition
    assert "-DLAC_MINORMINER_BUILD_PYTHON=OFF" in definition
    assert "ctest" in definition and "--output-on-failure" in definition
    receipt_index = definition.index("create-build-receipt")
    cleanup_index = definition.index(
        "rm -rf /opt/isingfold/build-venv /opt/isingfold/venv"
    )
    test_section = definition.index("%test")
    assert receipt_index < cleanup_index < test_section
    assert "/usr/local/bin/python3.12 -I" in definition[test_section:]


def test_apollo_consumes_the_goose_wheelhouse_without_compiling_or_network() -> None:
    text = APOLLO_BUILDER.read_text(encoding="utf-8")
    assert "--no-index" in text
    assert "--find-links" in text
    assert "pip wheel" not in text
    assert "pip install -e" not in text
    assert "EXPECTED_BASE_PYTHON_SHA256" in text
    assert "verify-runtime" in text
    assert "--execution-mode pinned-venv" in text
    assert "--accelerator gpu" in text


def test_full_runtime_guard_supports_explicit_cpu_and_gpu_modes() -> None:
    text = SHELL_GUARD.read_text(encoding="utf-8")
    assert "usage: verify_runtime_source.sh PYTHON ACCELERATOR" in text
    assert '"$accelerator" != "cpu"' in text
    assert '"$accelerator" != "gpu"' in text

    cpu_names = {
        "apollo_bootstrap_bank.sh",
        "apollo_final_strength_audit.sh",
        "apollo_initializer_bank.sh",
        "apollo_seal_initializer_bank.sh",
        "apollo_selector_labels.sh",
        "goose_bootstrap_bank.sbatch",
        "goose_bootstrap_bank_packed.sbatch",
        "goose_final_strength_audit.sbatch",
        "goose_initializer_bank.sbatch",
        "goose_initializer_bank_packed.sbatch",
        "goose_seal_bootstrap_bank.sbatch",
        "goose_seal_initializer_bank.sbatch",
        "goose_selector_labels.sbatch",
    }
    dynamic_apollo_quality_modes = {
        "apollo_quality_control.sh": "$runtime_accelerator",
        "apollo_quality_preflight_shards.sh": "$ISINGFOLD_QUALITY_ACCELERATOR",
        "apollo_quality_resolution_shards.sh": "$ISINGFOLD_QUALITY_ACCELERATOR",
        "apollo_quality_shards.sh": "$quality_accelerator",
    }
    for launcher in sorted((ROOT / "scripts").glob("apollo_*.sh")):
        if launcher == APOLLO_BUILDER:
            continue
        if launcher.name in dynamic_apollo_quality_modes:
            text = launcher.read_text()
            assert "require_quality_accelerator_control" in text
            mode = dynamic_apollo_quality_modes[launcher.name]
            assert (
                f'verify_runtime_source.sh" "$python_bin" "{mode}"' in text
            )
            continue
        mode = "cpu" if launcher.name in cpu_names else "gpu"
        assert f'verify_runtime_source.sh" "$python_bin" {mode}' in launcher.read_text()
    for launcher in sorted((ROOT / "scripts").glob("goose_*.sbatch")):
        if launcher in (GOOSE_BUILDER, GOOSE_VERIFIER):
            continue
        if launcher.name == "goose_quality_shards.sbatch":
            text = launcher.read_text()
            assert "require_quality_accelerator_control" in text
            assert (
                'verify_runtime_source.sh" "$container_python" "$quality_accelerator"'
                in text
            )
            continue
        mode = "cpu" if launcher.name in cpu_names else "gpu"
        assert f'verify_runtime_source.sh" "$container_python" {mode}' in launcher.read_text()


def test_gpu_runtime_guard_rejects_a_cpu_device_override_before_work() -> None:
    completed = subprocess.run(
        ["/bin/bash", str(SHELL_GUARD), sys.executable, "gpu"],
        cwd=ROOT,
        env={**os.environ, "ISINGFOLD_DEVICE": "cpu"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 78
    assert "reject ISINGFOLD_DEVICE" in completed.stderr


def test_paper_launchers_call_the_full_runtime_verifier() -> None:
    assert SHELL_GUARD.is_file()
    guard_text = SHELL_GUARD.read_text(encoding="utf-8")
    assert "verify_isingfold_runtime.py" in guard_text
    assert "ISINGFOLD_RUNTIME_BUILD_RECEIPT" in guard_text
    assert "ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256" in guard_text
    assert "ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256" in guard_text
    assert "ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256" in guard_text
    assert "ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256" in guard_text
    assert "ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256" in guard_text
    assert "ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256" in guard_text
    assert "canonical_environment_lock" in guard_text

    for launcher in sorted((ROOT / "scripts").glob("apollo_*.sh")):
        if launcher == APOLLO_BUILDER:
            continue
        assert "verify_runtime_source.sh" in launcher.read_text(encoding="utf-8"), launcher
    for launcher in sorted((ROOT / "scripts").glob("goose_*.sbatch")):
        if launcher in (GOOSE_BUILDER, GOOSE_VERIFIER):
            continue
        assert "verify_runtime_source.sh" in launcher.read_text(encoding="utf-8"), launcher


def test_runtime_scripts_are_executable() -> None:
    for path in (VERIFIER, SHELL_GUARD, APOLLO_BUILDER, GOOSE_BUILDER, GOOSE_VERIFIER):
        assert path.stat().st_mode & stat.S_IXUSR, path

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from embedbench.isingfold_corpus_shard import _DEPENDENCIES

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime" / "isingfold-corpus"

EXPECTED_PROVENANCE_VERSIONS = {
    "dimod": "0.12.22",
    "dwave-networkx": "0.8.19",
    "dwave-samplers": "1.8.0",
    "embedbench": "0.1.0",
    "minorminer": "0.2.22",
    "networkx": "3.6.1",
    "numpy": "2.4.4",
    "scipy": "1.18.0",
}
EXPECTED_SUPPORT_VERSIONS = {
    "dwave-graphs": "1.0.0",
    "fasteners": "0.20",
    "homebase": "1.0.1",
}
EXPECTED_BUILD_VERSIONS = {
    "packaging": "26.3",
    "pip": "26.2.1",
    "setuptools": "78.1.0",
    "wheel": "0.47.0",
}


def _requirements_versions(text: str) -> dict[str, str]:
    return {
        name: version
        for name, version in re.findall(r"^([a-z][a-z0-9-]*)==([^ \\\n]+)", text, re.MULTILINE)
    }


def test_runtime_lock_and_hashed_requirements_are_in_exact_agreement() -> None:
    runtime_lock = json.loads((RUNTIME / "runtime-lock.json").read_text(encoding="utf-8"))
    requirements = (RUNTIME / "requirements-linux-x86_64-py312.lock").read_text(
        encoding="utf-8"
    )

    assert runtime_lock["schema"] == "isingfold-corpus-runtime-lock-v2"
    assert runtime_lock["python"] == {
        "executable": "/opt/isingfold/venv/bin/python",
        "version": "3.12.3",
    }
    assert runtime_lock["platform"] == {"machine": "x86_64", "system": "Linux"}
    assert runtime_lock["provenance"] == {
        "dependency_versions": EXPECTED_PROVENANCE_VERSIONS,
        "protocol": "embedbench.isingfold-corpus-generator-v4",
    }
    assert set(runtime_lock["provenance"]["dependency_versions"]) == set(_DEPENDENCIES)
    assert runtime_lock["runtime_support_versions"] == EXPECTED_SUPPORT_VERSIONS
    assert runtime_lock["build_tool_versions"] == EXPECTED_BUILD_VERSIONS
    assert runtime_lock["thread_controls"] == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert runtime_lock["installed_inventory"] == {
        "algorithm": "isingfold-installed-wheel-payload-v1",
        "distribution_count": 14,
        "file_count": 4328,
        "installation_sha256": (
            "52521e9af36a79c15160b648035bc3a3eaaf28ee339da19686dfaf27c2b7a99f"
        ),
        "native_library_count": 160,
        "native_libraries_sha256": (
            "adb5b7d18a9879a2fa3099ef46e9456813a16760d6bdbd4ac7e50530f2720b89"
        ),
        "payload_sha256": "d2ee41a478e50a45381b8804fa151e9e9fe06980fc38fd7ed56e69215fd5d94b",
    }

    expected_external = {
        **{k: v for k, v in EXPECTED_PROVENANCE_VERSIONS.items() if k != "embedbench"},
        **EXPECTED_SUPPORT_VERSIONS,
        **EXPECTED_BUILD_VERSIONS,
    }
    assert _requirements_versions(requirements) == expected_external
    assert requirements.count("--hash=sha256:") == len(expected_external)
    assert "--require-hashes" in requirements
    assert "--only-binary=:all:" in requirements
    assert " -e " not in requirements and "git+" not in requirements


def test_runtime_readme_separates_authoritative_preflight_and_cpu_policy() -> None:
    readme = (RUNTIME / "README.md").read_text(encoding="utf-8")

    assert "must be created by the frozen Linux runtime" in readme
    assert "verify-preflight" in readme
    assert "3,082 identities" in readme
    assert "developer-machine preflight" in readme
    assert "Corpus prospecting and shard generation are CPU-bound" in readme
    assert "GPUs belong to" in readme
    assert "neural-model training stage" in readme


def test_definition_pins_base_and_exposes_the_registered_python() -> None:
    definition = (RUNTIME / "isingfold-corpus.def").read_text(encoding="utf-8")
    lock = json.loads((RUNTIME / "runtime-lock.json").read_text(encoding="utf-8"))

    reference = lock["base_image"]["reference"]
    assert re.fullmatch(r"docker\.io/library/[a-z0-9._-]+@sha256:[0-9a-f]{64}", reference)
    base_image = reference.removeprefix("docker.io/library/")
    assert f"From: {base_image}" in definition
    assert "runtime/isingfold-corpus/requirements-linux-x86_64-py312.lock" in definition
    assert definition.count("/opt/isingfold/venv/bin/python") >= 5
    assert "python -m pip check" in definition
    assert "--no-build-isolation" in definition
    for variable in lock["thread_controls"]:
        assert f"export {variable}=1" in definition


def test_goose_builder_requires_slurm_and_validates_before_publish() -> None:
    launcher = (ROOT / "scripts" / "goose_build_isingfold_corpus_runtime.sbatch").read_text(
        encoding="utf-8"
    )

    assert "${SLURM_JOB_ID:-}" in launcher
    assert "${SLURMD_NODENAME:-}" in launcher
    assert "/opt/slurm/bin/srun" in launcher
    assert '"$ISINGFOLD_CORPUS_APPTAINER" build --fakeroot' in launcher
    validation_at = launcher.index("validate_isingfold_corpus_image.py")
    publish_at = launcher.index('ln -- "$partial_image" "$ISINGFOLD_CORPUS_CONTAINER_IMAGE"')
    assert validation_at < publish_at
    assert "refusing to overwrite existing runtime artifact" in launcher
    assert "ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" in launcher
    assert "ISINGFOLD_CORPUS_EXPECTED_INSTALLATION_SHA256" in launcher
    assert "--expected-installation-sha256" in launcher


def test_goose_corpus_requests_one_cpu_and_no_gpu_for_single_thread_generation() -> None:
    launcher = (ROOT / "scripts" / "goose_isingfold_corpus.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --cpus-per-task=1" in launcher
    assert "#SBATCH --mem=8G" in launcher
    assert "#SBATCH --gres" not in launcher
    assert 'if (( threads != 1 )); then' in launcher


def test_goose_builder_claims_once_and_never_replaces_published_runtime() -> None:
    launcher = (ROOT / "scripts" / "goose_build_isingfold_corpus_runtime.sbatch").read_text(
        encoding="utf-8"
    )

    claim_at = launcher.index('if ! mkdir -- "$build_claim"')
    build_at = launcher.index('"$ISINGFOLD_CORPUS_APPTAINER" build --fakeroot')
    image_publish_at = launcher.index(
        'ln -- "$partial_image" "$ISINGFOLD_CORPUS_CONTAINER_IMAGE"'
    )
    validation_publish_at = launcher.index(
        'ln -- "$partial_validation" "$validation_receipt"'
    )
    commit_publish_at = launcher.index(
        'ln -- "$partial_digest_receipt" "$image_digest_receipt"'
    )

    assert claim_at < build_at < image_publish_at < validation_publish_at < commit_publish_at
    assert 'build_claim="$ISINGFOLD_CORPUS_IMAGE_OUTPUT_ROOT/.' in launcher
    assert 'build_claim=' in launcher and "$SLURM_JOB_ID.build-claim" not in launcher
    assert 'mv -- "$partial_image" "$ISINGFOLD_CORPUS_CONTAINER_IMAGE"' not in launcher
    assert 'mv -- "$partial_validation" "$validation_receipt"' not in launcher


def test_runtime_claim_has_exactly_one_concurrent_winner(tmp_path: Path) -> None:
    claim = tmp_path / ".runtime.sif.build-claim"

    def compete() -> bool:
        try:
            claim.mkdir()
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = tuple(pool.map(lambda _: compete(), range(2)))
    assert sorted(winners) == [False, True]


def test_runtime_hard_links_never_replace_and_digest_is_commit_marker(
    tmp_path: Path,
) -> None:
    partial_image = tmp_path / ".runtime.sif.partial"
    partial_validation = tmp_path / ".runtime.validation.partial"
    partial_digest = tmp_path / ".runtime.sha256.partial"
    image = tmp_path / "runtime.sif"
    validation = tmp_path / "runtime.sif.validation.json"
    digest = tmp_path / "runtime.sif.sha256"
    partial_image.write_bytes(b"new-image")
    partial_validation.write_bytes(b"new-validation")
    partial_digest.write_bytes(b"new-digest")

    os.link(partial_image, image)
    os.link(partial_validation, validation)
    assert image.exists() and validation.exists() and not digest.exists()
    os.link(partial_digest, digest)
    assert digest.read_bytes() == b"new-digest"

    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    for target in (image, validation, digest):
        original = target.read_bytes()
        with pytest.raises(FileExistsError):
            os.link(replacement, target)
        assert target.read_bytes() == original


def test_goose_builder_rejects_direct_non_slurm_execution() -> None:
    launcher = ROOT / "scripts" / "goose_build_isingfold_corpus_runtime.sbatch"
    completed = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        env={"PATH": os.environ["PATH"]},
        text=True,
    )

    assert completed.returncode == 69
    assert "requires a Goose Slurm allocation" in completed.stderr


@pytest.mark.parametrize(
    "launcher_name",
    ["apollo_isingfold_corpus.sh", "goose_isingfold_corpus.sbatch"],
)
def test_generation_launcher_authenticates_helper_before_sourcing(
    launcher_name: str,
) -> None:
    launcher = (ROOT / "scripts" / launcher_name).read_text(encoding="utf-8")

    digest_check_at = launcher.index(
        'if [[ "$observed_runtime_helper_sha256" '
        '!= "$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" ]]'
    )
    source_at = launcher.index('source "$runtime_helper"')
    assert "ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" in launcher
    assert digest_check_at < source_at
    assert "runtime_helper_sha256=$ISINGFOLD_CORPUS_EXPECTED_RUNTIME_HELPER_SHA256" in launcher


@pytest.mark.parametrize(
    "launcher_name",
    ["apollo_isingfold_corpus.sh", "goose_isingfold_corpus.sbatch"],
)
def test_generation_launcher_enforces_single_thread_runtime_contract(
    launcher_name: str,
) -> None:
    launcher = (ROOT / "scripts" / launcher_name).read_text(encoding="utf-8")

    assert "the pinned corpus runtime requires THREADS=1" in launcher
    for variable in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        assert re.search(rf'{variable}="?\$threads"?', launcher)


def test_shared_helper_rejects_noncontract_thread_count() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "verify_isingfold_corpus_runtime.sh"),
            "provenance-sha256",
            str(ROOT),
            sys.executable,
            "2",
        ],
        check=False,
        capture_output=True,
        env={"PATH": os.environ["PATH"]},
        text=True,
    )

    assert completed.returncode == 64
    assert "requires exactly one worker thread" in completed.stderr


def test_source_inventory_ignores_generated_distribution_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path.resolve()
    scripts = source_root / "scripts"
    package = source_root / "src" / "embedbench"
    metadata = source_root / "src" / "embedbench.egg-info"
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    metadata.mkdir()
    shutil.copyfile(ROOT / "pyproject.toml", source_root / "pyproject.toml")
    for name in (
        "apollo_isingfold_corpus.sh",
        "goose_isingfold_corpus.sbatch",
        "verify_isingfold_corpus_runtime.sh",
    ):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    (package / "__init__.py").write_text("VERSION = 1\n", encoding="utf-8")
    (metadata / "PKG-INFO").write_text("Version: 0.1.0\n", encoding="utf-8")

    helper = scripts / "verify_isingfold_corpus_runtime.sh"

    def digest() -> str:
        return subprocess.run(
            ["bash", str(helper), "source-sha256", str(source_root), sys.executable],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    baseline = digest()
    (metadata / "PKG-INFO").write_text("Version: 999\n", encoding="utf-8")
    assert digest() == baseline

    (package / "__init__.py").write_text("VERSION = 2\n", encoding="utf-8")
    assert digest() != baseline


def test_shared_mode_guard_makes_canary_only_single_shard() -> None:
    helper = str(ROOT / "scripts" / "verify_isingfold_corpus_runtime.sh")
    valid = subprocess.run(
        [
            "bash",
            helper,
            "validate-mode",
            "canary-only",
            "10",
            "10",
            "1",
            "1",
            "/diagnostic/canary/shards",
            "/diagnostic/canary",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0

    for first, last, step, tasks in (
        ("10", "11", "1", "1"),
        ("10", "10", "2", "1"),
        ("10", "10", "1", "2"),
    ):
        invalid = subprocess.run(
            [
                "bash",
                helper,
                "validate-mode",
                "canary-only",
                first,
                last,
                step,
                tasks,
                "/diagnostic/canary/shards",
                "/diagnostic/canary",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert invalid.returncode == 64
        assert "exactly one shard and one task" in invalid.stderr


@pytest.mark.parametrize(
    "launcher_name",
    ["apollo_isingfold_corpus.sh", "goose_isingfold_corpus.sbatch"],
)
def test_generation_launchers_gate_production_on_pinned_parity_attestation(
    launcher_name: str,
) -> None:
    launcher = (ROOT / "scripts" / launcher_name).read_text(encoding="utf-8")

    assert "ISINGFOLD_CORPUS_MODE" in launcher
    assert "canary-only" in launcher and "production" in launcher
    assert "ISINGFOLD_CORPUS_EXPECTED_PARITY_ATTESTATION_SHA256" in launcher
    assert "embedbench.isingfold_cross_site_canary verify" in launcher
    assert "canary-only SHARD_ROOT must be CANARY_OUTPUT_ROOT/shards" in launcher


@pytest.mark.parametrize(
    "launcher_name",
    ["apollo_isingfold_corpus.sh", "goose_isingfold_corpus.sbatch"],
)
def test_generation_launchers_validate_installed_bytes_before_generation(
    launcher_name: str,
) -> None:
    launcher = (ROOT / "scripts" / launcher_name).read_text(encoding="utf-8")

    validation_token = (
        "corpus_validate_apollo_runtime"
        if launcher_name == "apollo_isingfold_corpus.sh"
        else "--expected-installation-sha256"
    )
    validation_at = launcher.index(validation_token)
    generation_at = launcher.index("generate-shard")
    assert validation_at < generation_at
    assert "ISINGFOLD_CORPUS_EXPECTED_RUNTIME_LOCK_SHA256" in launcher
    assert "ISINGFOLD_CORPUS_EXPECTED_VALIDATOR_SHA256" in launcher


@pytest.mark.parametrize(
    "launcher_name",
    ["apollo_isingfold_corpus.sh", "goose_isingfold_corpus.sbatch"],
)
def test_generation_launchers_require_full_preflight_replay_and_shard_binding(
    launcher_name: str,
) -> None:
    launcher = (ROOT / "scripts" / launcher_name).read_text(encoding="utf-8")

    for control in (
        "ISINGFOLD_CORPUS_PREFLIGHT",
        "ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_SHA256",
        "ISINGFOLD_CORPUS_EXPECTED_PREFLIGHT_RECORD_DIGEST",
        "ISINGFOLD_CORPUS_EXPECTED_IDENTITY_MAP_DIGEST",
        "ISINGFOLD_CORPUS_EXPECTED_PROVENANCE_SHA256",
    ):
        assert control in launcher
    replay_token = (
        "corpus_validate_preflight"
        if launcher_name == "apollo_isingfold_corpus.sh"
        else "validate-preflight"
    )
    replay_at = launcher.index(replay_token)
    generation_at = launcher.index("generate-shard")
    assert replay_at < generation_at
    generation_command = launcher[generation_at:]
    for argument in (
        "--expected-source-sha256",
        "--preflight",
        "--expected-preflight-sha256",
        "--expected-preflight-record-digest",
        "--expected-identity-map-digest",
        "--expected-generation-provenance-digest",
    ):
        assert argument in generation_command

    helper = (ROOT / "scripts" / "verify_isingfold_corpus_runtime.sh").read_text(
        encoding="utf-8"
    )
    assert "corpus_validate_preflight()" in helper
    assert "embedbench.isingfold_corpus_cli verify-preflight" in helper


def test_shared_helper_rejects_noneditable_external_direct_url_install(
    tmp_path: Path,
) -> None:
    source_root = tmp_path.resolve()
    package = source_root / "src" / "embedbench"
    package.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "src" / "embedbench" / "isingfold_corpus_shard.py",
        package / "isingfold_corpus_shard.py",
    )
    fake_distribution = source_root / "src" / "minorminer-0.2.22.dist-info"
    fake_distribution.mkdir()
    (fake_distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: minorminer\nVersion: 0.2.22\n",
        encoding="utf-8",
    )
    (fake_distribution / "direct_url.json").write_text(
        json.dumps({"dir_info": {"editable": False}, "url": "file:///untrusted/tree"}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "verify_isingfold_corpus_runtime.sh"),
            "validate-externals",
            str(source_root),
            sys.executable,
        ],
        check=False,
        capture_output=True,
        env={"PATH": os.environ["PATH"]},
        text=True,
    )

    assert completed.returncode != 0
    assert "minorminer: direct-URL install" in completed.stderr

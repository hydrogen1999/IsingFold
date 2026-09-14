from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _validator_module() -> ModuleType:
    path = ROOT / "scripts" / "validate_isingfold_corpus_image.py"
    spec = importlib.util.spec_from_file_location("isingfold_runtime_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installation_digest_is_canonical_and_self_excluding() -> None:
    validator = _validator_module()
    contract = {
        "algorithm": "isingfold-installed-wheel-payload-v1",
        "distribution_count": 14,
        "file_count": 4328,
        "native_library_count": 160,
        "native_libraries_sha256": (
            "adb5b7d18a9879a2fa3099ef46e9456813a16760d6bdbd4ac7e50530f2720b89"
        ),
        "payload_sha256": (
            "d2ee41a478e50a45381b8804fa151e9e9fe06980fc38fd7ed56e69215fd5d94b"
        ),
    }

    assert validator._installation_sha256(contract) == (
        "52521e9af36a79c15160b648035bc3a3eaaf28ee339da19686dfaf27c2b7a99f"
    )


def test_validator_requires_external_installation_commitment() -> None:
    validator = _validator_module()
    parser = validator._parser()
    action_names = {action.dest for action in parser._actions}

    assert "expected_installation_sha256" in action_names


def test_validator_rejects_unexpected_or_duplicate_installed_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _validator_module()

    class FakeDistribution:
        def __init__(self, name: str) -> None:
            self.metadata = {"Name": name}

    monkeypatch.setattr(
        validator.importlib.metadata,
        "distributions",
        lambda: iter(
            (
                FakeDistribution("EmbedBench"),
                FakeDistribution("minorminer"),
                FakeDistribution("unexpected"),
            )
        ),
    )
    with pytest.raises(RuntimeError, match="distribution set differs"):
        validator._validate_distribution_set({"embedbench", "minorminer"})

    monkeypatch.setattr(
        validator.importlib.metadata,
        "distributions",
        lambda: iter(
            (
                FakeDistribution("EmbedBench"),
                FakeDistribution("embedbench"),
                FakeDistribution("minorminer"),
            )
        ),
    )
    with pytest.raises(RuntimeError, match="duplicate installed distribution"):
        validator._validate_distribution_set({"embedbench", "minorminer"})

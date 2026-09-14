from __future__ import annotations


def test_native_backend_is_importable() -> None:
    import lac_minorminer

    info = lac_minorminer.backend_info()

    assert info["cpp_standard"] == 17
    assert info["backend"] == "lac_minorminer_cpp"
    assert info["package_version"] == lac_minorminer.__version__
    assert info["compiler_family"] in {"clang", "gcc", "msvc", "unknown"}
    assert isinstance(info["assertions_enabled"], bool)
    assert info["stock_minorminer_linked"] is False


def test_quality_v2_runtime_is_available_from_the_public_package() -> None:
    from lac_minorminer import (
        QUALITY_V2_FEATURE_NAMES,
        LinearQualityV2Checkpoint,
        QualityV2AcceptancePolicy,
        QualityV2BatchInput,
        QualityV2Decision,
        QualityV2InferenceError,
        QualityV2Predictions,
        QualityV2Predictor,
        QualityV2Problem,
        QualityV2Scorer,
        WideCandidateProvider,
        load_quality_v2_checkpoint,
    )

    assert WideCandidateProvider is not None
    assert "exact_total_qubits" in QUALITY_V2_FEATURE_NAMES
    assert QualityV2Scorer is not None
    assert QualityV2AcceptancePolicy is not None
    assert QualityV2BatchInput is not None
    assert QualityV2Decision is not None
    assert QualityV2InferenceError is not None
    assert QualityV2Predictions is not None
    assert QualityV2Predictor is not None
    assert QualityV2Problem is not None
    assert LinearQualityV2Checkpoint is not None
    assert load_quality_v2_checkpoint is not None

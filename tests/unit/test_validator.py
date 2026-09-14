from __future__ import annotations

from lac_minorminer.validation import validate_embedding


def test_validator_accepts_a_minor_embedding_with_a_multinode_chain() -> None:
    source = [("a", "b"), ("a", "c"), ("b", "c")]
    target = [(0, 1), (1, 2), (2, 3), (3, 0)]
    embedding = {"a": [0, 1], "b": [2], "c": [3]}

    report = validate_embedding(source, target, embedding)

    assert report.valid
    assert report.errors == ()


def test_validator_reports_structural_failures_without_native_code() -> None:
    source = [("a", "b")]
    target = [(0, 1), (1, 2)]

    disconnected = validate_embedding(source, target, {"a": [0, 2], "b": [1]})
    overlap = validate_embedding(source, target, {"a": [0, 1], "b": [1, 2]})
    missing_edge = validate_embedding(source, target, {"a": [0], "b": [2]})
    missing_chain = validate_embedding(source, target, {"a": [0]})

    assert not disconnected.valid and "not connected" in disconnected.errors[0]
    assert not overlap.valid and any("overlap" in error for error in overlap.errors)
    assert not missing_edge.valid and any("not realized" in error for error in missing_edge.errors)
    assert not missing_chain.valid and any(
        "missing a chain" in error for error in missing_chain.errors
    )

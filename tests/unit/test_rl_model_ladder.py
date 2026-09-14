"""The registered three-family representation screen is executable, not config-only."""

from __future__ import annotations

import pytest

from isingfold.rl.model import IFCore, IFDual, IFMLP, build_model


@pytest.mark.parametrize(
    ("name", "kind", "local", "fusion"),
    (
        ("if-mlp", IFMLP, 0, 0),
        ("if-dual", IFDual, 3, 0),
        ("if-core", IFCore, 3, 2),
    ),
)
def test_model_factory_materializes_the_registered_ladder(
    name: str,
    kind: type[IFCore],
    local: int,
    fusion: int,
) -> None:
    model = build_model(name)

    assert isinstance(model, kind)
    assert len(model.local_l) == local
    assert len(model.local_h) == local
    assert len(model.fusion) == fusion
    assert model.parameter_count() > 0


def test_model_factory_rejects_unregistered_architectures() -> None:
    with pytest.raises(ValueError, match="model family"):
        build_model("Transformer-because-bigger")

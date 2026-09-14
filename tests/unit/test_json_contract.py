from __future__ import annotations

import pytest

from isingfold_lac_b.json_contract import strict_json_loads, strict_json_object


def test_strict_json_rejects_nested_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON object key: 'x'"):
        strict_json_object('{"outer":{"x":1,"x":2}}', name="fixture")


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonfinite_numbers(token: str) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_loads(f'{{"value":{token}}}', name="fixture")


def test_strict_json_object_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="one JSON object"):
        strict_json_object("[]", name="fixture")

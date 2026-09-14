"""Minimal canonical-JSON primitives used by the isolated paper certificate checker."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Collection
from typing import Any, cast


def _normalize_unicode_scalar(value: str, name: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate code point")
    return unicodedata.normalize("NFC", value)


def canonical_value(value: object) -> object:
    """Return the registered cross-platform canonical JSON view."""

    value_type = type(value)
    if value_type is float:
        number = cast(float, value)
        if not math.isfinite(number):
            raise ValueError("non-finite float")
        normalized = 0.0 if number == 0.0 else number
        return {"__float64_hex__": normalized.hex().lower()}
    if value is None or value_type is bool or value_type is int:
        return value
    if value_type is str:
        return _normalize_unicode_scalar(cast(str, value), "canonical string")
    if value_type is list or value_type is tuple:
        return [canonical_value(item) for item in cast(list[Any] | tuple[Any, ...], value)]
    if value_type is dict:
        mapping = cast(dict[object, object], value)
        if not all(type(key) is str for key in mapping):
            raise TypeError("canonical object keys must be strings")
        keys = [
            _normalize_unicode_scalar(cast(str, key), "canonical object key") for key in mapping
        ]
        if len(keys) != len(set(keys)) or "__float64_hex__" in keys:
            raise ValueError("duplicate normalized or reserved key")
        return {
            _normalize_unicode_scalar(key, "canonical object key"): canonical_value(item)
            for key, item in cast(dict[str, object], mapping).items()
        }
    raise TypeError(f"unsupported canonical type: {type(value)!r}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_exact_keys(
    value: object,
    expected: Collection[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a JSON object")
    if not all(type(key) is str for key in value):
        raise TypeError(f"{name} keys must be strings")
    expected_keys = set(expected)
    if not all(isinstance(key, str) for key in expected_keys):
        raise TypeError("expected schema keys must be strings")
    actual = set(value)
    if actual != expected_keys:
        missing = sorted(expected_keys - actual)
        unknown = sorted(actual - expected_keys)
        raise ValueError(f"{name} schema fields differ: missing={missing}, unknown={unknown}")
    return cast(dict[str, Any], value)


def validate_versioned_object(
    value: object,
    *,
    schema: str,
    schema_version: int,
    fields: Collection[str],
    name: str,
) -> dict[str, Any]:
    if type(schema) is not str or not schema:
        raise ValueError("expected schema must be a non-empty string")
    if type(schema_version) is not int or schema_version <= 0:
        raise ValueError("expected schema_version must be a positive integer")
    document = require_exact_keys(value, {"schema", "schema_version", *fields}, name)
    actual_schema = document["schema"]
    if type(actual_schema) is not str or actual_schema != schema:
        raise ValueError(f"{name} requires schema {schema!r}")
    actual_version = document["schema_version"]
    if type(actual_version) is not int or actual_version != schema_version:
        raise ValueError(f"{name} requires schema_version {schema_version}")
    return document

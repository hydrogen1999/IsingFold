"""Shared validation for public facade and direct orchestration options."""

from __future__ import annotations

import math
import secrets
from numbers import Integral, Real
from typing import Any


def integer_option(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def seed_option(value: Any, *, allow_none: bool) -> int:
    if value is None:
        if allow_none:
            return secrets.randbits(64)
        raise ValueError("random_seed must be an unsigned 64-bit integer")
    seed = integer_option("random_seed", value, minimum=0)
    if seed >= 2**64:
        raise ValueError("random_seed must fit in an unsigned 64-bit integer")
    return seed


def timeout_option(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("timeout must be a finite non-negative number or None")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0.0:
        raise ValueError("timeout must be a finite non-negative number or None")
    return timeout

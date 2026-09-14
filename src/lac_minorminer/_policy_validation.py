"""Validation gates for all values returned by replaceable Python policies."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral, Real
from typing import Any


def validated_selection(value: Any, eligible: list[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in eligible:
        raise ValueError("vertex selector must return one of the eligible logical IDs")
    return int(value)


def validated_repair_neighborhood(value: Any, logical_count: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("repair selector must return unique in-range logical IDs")
    raw_ids = list(value)
    if not raw_ids or any(
        isinstance(logical, bool)
        or not isinstance(logical, Integral)
        or not 0 <= int(logical) < logical_count
        for logical in raw_ids
    ):
        raise ValueError("repair selector must return unique in-range logical IDs")
    logical_ids = tuple(int(logical) for logical in raw_ids)
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("repair selector must return unique in-range logical IDs")
    return logical_ids


def validated_costs(value: Any, target_size: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("route-cost provider must return one finite non-negative cost per target")
    costs = list(value)
    if len(costs) != target_size or any(
        isinstance(cost, bool)
        or not isinstance(cost, Real)
        or not math.isfinite(float(cost))
        or float(cost) < 0.0
        for cost in costs
    ):
        raise ValueError("route-cost provider must return one finite non-negative cost per target")
    return [float(cost) for cost in costs]


def validated_scores(value: Any, candidate_count: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("scorer must return one finite score per candidate")
    scores = list(value)
    if len(scores) != candidate_count or any(
        isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score))
        for score in scores
    ):
        raise ValueError("scorer must return one finite score per candidate")
    return [float(score) for score in scores]


def validated_choice(value: Any, candidate_count: int) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) < candidate_count
    ):
        raise ValueError("acceptance policy must return a candidate index or None")
    return int(value)

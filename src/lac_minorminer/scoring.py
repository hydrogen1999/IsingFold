"""Candidate scoring adapters."""

from __future__ import annotations

from typing import Any


class ResourceScorer:
    """Encode the exact native lexicographic resource order as dense scalar ranks."""

    @staticmethod
    def _key(candidate: Any) -> tuple[int, int, int, float]:
        rank = candidate.rank
        return (
            rank.max_occupancy,
            rank.total_excess_occupancy,
            rank.used_target_nodes,
            rank.route_cost,
        )

    def score(self, snapshot: Any, candidates: Any) -> list[float]:
        keys = [self._key(candidate) for candidate in candidates.candidates]
        dense_rank = {key: float(index) for index, key in enumerate(sorted(set(keys)))}
        return [dense_rank[key] for key in keys]

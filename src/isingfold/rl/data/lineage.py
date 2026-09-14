"""Leakage-resistant lineages and splits.

Spec: Rev2 section "Leakage-resistant lineages and data contracts". Base problem lineages are
split *before* trajectories, gauges, embedding candidates or topology variants are generated,
and every descendant of one base problem stays in one partition.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


def digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Lineage:
    """An immutable base-problem identity and its generator provenance."""

    lineage_id: str
    family: str
    host: str
    size: int
    generator_version: str
    parent: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def root(self) -> str:
        return self.parent or self.lineage_id


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    ood: tuple[str, ...] = ()

    def partitions_for(self, lineage_root: str) -> tuple[str, ...]:
        """Return every membership so malformed overlapping registries cannot fail open."""

        return tuple(
            name for name in ("train", "validation", "test", "ood")
            if lineage_root in getattr(self, name)
        )

    def partition_of(self, lineage_root: str) -> str:
        memberships = self.partitions_for(lineage_root)
        if not memberships:
            return "unassigned"
        if len(memberships) > 1:
            return "ambiguous"
        return memberships[0]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "ood": list(self.ood),
        }


def split_by_lineage(
    lineages: Iterable[Lineage],
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ood_predicate=None,
    seed: int = 0,
) -> Split:
    """Group by root lineage, hold out an explicit OOD stratum, then split the remainder.

    An ordinary random split does not establish topology transfer, so ``ood_predicate``
    withholds a whole family, size range or host regime by construction.
    """

    roots: dict[str, Lineage] = {}
    for lineage in lineages:
        roots.setdefault(lineage.root, lineage)
    ood = sorted(r for r, item in roots.items() if ood_predicate is not None and ood_predicate(item))
    rest = sorted(r for r in roots if r not in set(ood))
    random.Random(seed).shuffle(rest)
    n = len(rest)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return Split(
        train=tuple(sorted(rest[:n_train])),
        validation=tuple(sorted(rest[n_train : n_train + n_val])),
        test=tuple(sorted(rest[n_train + n_val :])),
        ood=tuple(ood),
    )


def check_no_leakage(split: Split, records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Every record must land in exactly one partition through its root lineage."""

    seen: dict[str, set[str]] = {}
    unassigned = 0
    registry_conflicts: dict[str, list[str]] = {}
    for record in records:
        root = str(record.get("lineage_root") or record.get("lineage") or "")
        memberships = split.partitions_for(root)
        if not memberships:
            unassigned += 1
            seen.setdefault(root, set()).add("unassigned")
            continue
        if len(memberships) > 1:
            registry_conflicts[root] = sorted(memberships)
        seen.setdefault(root, set()).update(memberships)
    conflicts = {
        **{root: sorted(parts) for root, parts in seen.items() if len(parts) > 1},
        **registry_conflicts,
    }
    return {"roots": len(seen), "unassigned": unassigned, "conflicts": conflicts, "ok": not conflicts and not unassigned}

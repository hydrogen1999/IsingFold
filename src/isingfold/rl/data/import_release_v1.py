"""Strict offline adapter for the legacy EmbedBench release-v1.1 quality corpus.

Release v1.1 predates CandidateBank.  Its quality records carry a full Ising problem and
enough chain data to rebuild an initializer, but they mix public state with outcome labels
and carry only an untyped ``ground_energy`` scalar.  This module authenticates the legacy
files, reconstructs their full-yield hardware graph without importing EmbedBench, and emits
the same five separated artifacts as :mod:`isingfold.rl.data.import_embedbench`.

The scalar reference in a legacy row is not promoted to a certified target by assertion.
Conversion therefore requires an authenticated problem-reference sidecar.  The published
v1.1 archive does not contain that sidecar, so it intentionally fails closed until the
missing certificates are released.  In particular, graph-cut application records were
generated with a tabu reference and cannot be called exact from the JSONL alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from isingfold.rl.data.import_embedbench import (
    CERTIFIED_REFERENCE_STATUSES,
    canonical_json_bytes,
    content_digest,
)

_QUALITY_FIELDS = {
    "Q",
    "all_chains",
    "best_F",
    "best_index",
    "candidates",
    "difficulty",
    "edge_J",
    "focus",
    "focus_h",
    "frozen",
    "frozen_adjacency",
    "ground_energy",
    "instance_id",
    "j_scale",
    "l_cap",
    "margin",
    "mode",
    "n_enumerated",
    "n_vars",
    "neighbour_chain_size",
    "neighbour_degree",
    "neighbours",
    "original_agrees",
    "original_index",
    "p_solve",
    "problem",
    "resource_agrees",
    "resource_index",
    "sa_seed",
    "size",
    "source",
    "spread",
    "stage",
    "topology",
    "window_edges",
    "window_nodes",
}
_CONFIG_FIELDS = {
    "alpha",
    "chain_size",
    "defect_couplers",
    "defect_qubits",
    "degree",
    "difficulty",
    "graph",
    "l_cap",
    "loop_max",
    "loop_min",
    "max_candidates",
    "max_enum",
    "max_window_free",
    "min_spread",
    "modes",
    "n_strengths",
    "n_vars",
    "num_reads",
    "num_sweeps",
    "refine_reads",
    "refine_top",
    "samples_per_instance",
    "size",
    "source",
    "topology",
}
_MANIFEST_FIELDS = {"config", "file", "jobs", "n_instances", "objective", "seed", "sha256", "stats"}
_SPLIT_FIELDS = {
    "group_counts",
    "groups",
    "provenance",
    "record_counts",
    "rule",
    "schema",
    "schema_version",
    "splits",
}
_REFERENCE_FIELDS = {
    "certificate_digest",
    "evaluator_protocol_digest",
    "problem_digest",
    "reference_energy",
    "reference_status",
    "schema",
    "schema_version",
}
_HEX = frozenset("0123456789abcdef")
_SOURCE_ID = re.compile(
    r"^(?P<host>chimera|pegasus|zephyr)(?P<size>[1-9][0-9]*)-"
    r"(?P<graph>inkdrop|random|app)-(?P<mode>[A-Za-z0-9_]+)-"
    r"(?P<index>[0-9]+)-s(?P<seed>[0-9]+)-(?P<source>[wm])$"
)


class ReleaseV1CompatibilityError(ValueError):
    """The authenticated legacy release lacks a fact required by the current contract."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} schema differs: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _check_finite(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite(item, name)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item, name)


def _strict_object(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    _check_finite(value, name)
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    return _strict_object(path.read_bytes(), name)


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{name} line {line_number} is blank")
        rows.append(_strict_object(line, f"{name} line {line_number}"))
    if not rows:
        raise ValueError(f"{name} is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _checksum_table(path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        if match is None:
            raise ValueError(f"SHA256SUMS line {line_number} is not canonical")
        digest, filename = match.groups()
        if filename in table:
            raise ValueError(f"SHA256SUMS repeats {filename!r}")
        table[filename] = digest
    if not table:
        raise ValueError("SHA256SUMS is empty")
    return table


def _verify_checksum(path: Path, table: Mapping[str, str]) -> str:
    expected = table.get(path.name)
    if expected is None:
        raise ReleaseV1CompatibilityError(
            f"{path.name} is not authenticated by the supplied SHA256SUMS"
        )
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")
    return actual


def _canonical_problem(
    raw: object,
) -> tuple[list[int], list[list[float | int]], list[list[float | int]], float]:
    if not isinstance(raw, Mapping):
        raise ValueError("problem must be an object")
    _exact_keys(raw, {"J", "e0", "h"}, "quality problem")
    raw_h = raw["h"]
    if not isinstance(raw_h, Mapping) or not raw_h:
        raise ValueError("problem h must be a non-empty object")
    h: list[list[float | int]] = []
    for key, bias in raw_h.items():
        if not isinstance(key, str) or not re.fullmatch(r"0|[1-9][0-9]*", key):
            raise ValueError("problem h keys must be canonical non-negative integer strings")
        h.append([int(key), _number(bias, f"h[{key}]")])
    h.sort(key=lambda item: int(item[0]))
    nodes = [int(item[0]) for item in h]
    allowed = set(nodes)
    raw_j = raw["J"]
    if not isinstance(raw_j, list):
        raise ValueError("problem J must be a list")
    couplers: list[list[float | int]] = []
    seen: set[tuple[int, int]] = set()
    for index, edge in enumerate(raw_j):
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError(f"problem J[{index}] must be [u,v,bias]")
        u = _integer(edge[0], f"J[{index}] u", minimum=0)
        v = _integer(edge[1], f"J[{index}] v", minimum=0)
        if u == v or u not in allowed or v not in allowed:
            raise ValueError(f"problem J[{index}] has invalid endpoints")
        pair = (min(u, v), max(u, v))
        if pair in seen:
            raise ValueError("problem J contains a duplicate edge")
        seen.add(pair)
        weight = _number(edge[2], f"J[{index}] bias")
        if weight == 0.0:
            raise ValueError("problem J must not store zero couplers")
        couplers.append([pair[0], pair[1], weight])
    couplers.sort(key=lambda edge: (int(edge[0]), int(edge[1])))
    return nodes, h, couplers, _number(raw["e0"], "problem e0")


def quality_problem_digest(problem: object) -> str:
    """Return the exact schema-v2 EmbedBench semantic problem digest."""

    _, h, couplers, e0 = _canonical_problem(problem)
    payload = json.dumps(
        {"J": [tuple(edge) for edge in couplers], "e0": e0, "h": [tuple(item) for item in h]},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _host_graph(topology: str, size: int) -> tuple[list[int], list[list[int]]]:
    """Reproduce full-yield D-Wave NetworkX integer-labelled topology graphs."""

    if size <= 0:
        raise ValueError("host size must be positive")
    edges: set[tuple[int, int]] = set()
    if topology == "chimera":
        tile = 4

        def label(i: int, j: int, u: int, k: int) -> int:
            return i * size * 2 * tile + j * 2 * tile + u * tile + k

        nodes = list(range(size * size * 2 * tile))
        for i in range(size):
            for j in range(size):
                for a in range(tile):
                    for b in range(tile):
                        edges.add((label(i, j, 0, a), label(i, j, 1, b)))
                if i + 1 < size:
                    for k in range(tile):
                        edges.add(tuple(sorted((label(i, j, 0, k), label(i + 1, j, 0, k)))))
                if j + 1 < size:
                    for k in range(tile):
                        edges.add(tuple(sorted((label(i, j, 1, k), label(i, j + 1, 1, k)))))
    elif topology == "pegasus":
        if size < 2:
            return [], []
        m1 = size - 1
        off0 = (2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6)
        off1 = (6, 6, 6, 6, 2, 2, 2, 2, 10, 10, 10, 10)
        start = (min(off1), min(off0))
        end = (12 - max(off1), 12 - max(off0))

        def label(u: int, w: int, k: int, z: int) -> int:
            return u * 12 * size * m1 + w * 12 * m1 + k * m1 + z

        def keep(u: int, w: int, k: int, _z: int) -> bool:
            if w == 0:
                return k >= start[u]
            if w == m1:
                return k < 12 - end[u]
            return True

        for u in (0, 1):
            for w in range(size):
                lower = start[u] if w == 0 else 0
                upper = 12 - (end[u] if w == m1 else 0)
                for k in range(lower, upper):
                    for z in range(m1 - 1):
                        edges.add(tuple(sorted((label(u, w, k, z), label(u, w, k, z + 1)))))
                for k in range(lower, upper, 2):
                    for z in range(m1):
                        edges.add(tuple(sorted((label(u, w, k, z), label(u, w, k + 1, z)))))
        for w in range(size):
            for kk in range(12):
                lower = 0 if w else off1[kk]
                upper = 12 if w < m1 else off1[kk]
                for k in range(lower, upper):
                    for z in range(m1):
                        left = (0, w, k, z)
                        right = (1, z + int(kk < off0[k]), kk, w - int(k < off1[kk]))
                        if keep(*left) and keep(*right):
                            edges.add(tuple(sorted((label(*left), label(*right)))))
        nodes = sorted({node for edge in edges for node in edge})
    elif topology == "zephyr":
        tile = 4
        width = 2 * size + 1

        def label(u: int, w: int, k: int, j: int, z: int) -> int:
            return (((u * width + w) * tile + k) * 2 + j) * size + z

        nodes = list(range(4 * tile * size * width))
        for u in (0, 1):
            for w in range(width):
                for k in range(tile):
                    for j in (0, 1):
                        for z in range(size - 1):
                            edges.add(
                                tuple(sorted((label(u, w, k, j, z), label(u, w, k, j, z + 1))))
                            )
                    for a in (0, 1):
                        for z in range(a, size):
                            edges.add(
                                tuple(sorted((label(u, w, k, 0, z), label(u, w, k, 1, z - a))))
                            )
        for w in range(size):
            for z in range(size):
                for h in range(tile):
                    for k in range(tile):
                        for i in (0, 1):
                            for j in (0, 1):
                                for a in (0, 1):
                                    for b in (0, 1):
                                        left = label(0, 2 * w + 1 + a * (2 * i - 1), k, j, z)
                                        right = label(1, 2 * z + 1 + b * (2 * j - 1), h, i, w)
                                        edges.add(tuple(sorted((left, right))))
    else:
        raise ValueError(f"unsupported topology {topology!r}")
    return nodes, [list(edge) for edge in sorted(edges)]


def _chains(raw: object, name: str, *, allow_empty: bool = False) -> dict[int, list[int]]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be an object")
    result: dict[int, list[int]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not re.fullmatch(r"0|[1-9][0-9]*", key):
            raise ValueError(f"{name} keys must be canonical non-negative integer strings")
        node = int(key)
        if not isinstance(value, list) or (not value and not allow_empty):
            raise ValueError(f"{name}[{key}] must be a non-empty qubit list")
        chain = [_integer(q, f"{name}[{key}] qubit", minimum=0) for q in value]
        if len(chain) != len(set(chain)) or chain != sorted(chain):
            raise ValueError(f"{name}[{key}] must be sorted and unique")
        result[node] = chain
    return result


def _list_ints(raw: object, name: str, *, nonempty: bool = False) -> list[int]:
    if not isinstance(raw, list) or (nonempty and not raw):
        raise ValueError(f"{name} must be {'a non-empty' if nonempty else 'a'} list")
    result = [_integer(value, f"{name} value", minimum=0) for value in raw]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


def _connected(chain: Sequence[int], adjacency: Mapping[int, set[int]]) -> bool:
    seen = {chain[0]}
    pending = [chain[0]]
    allowed = set(chain)
    while pending:
        current = pending.pop()
        new = (adjacency[current] & allowed) - seen
        seen.update(new)
        pending.extend(new)
    return seen == allowed


def _validate_embedding(
    chains: Mapping[int, Sequence[int]],
    nodes: Sequence[int],
    logical_edges: Sequence[Sequence[int]],
    host_nodes: Sequence[int],
    host_edges: Sequence[Sequence[int]],
    qubit_cap: int,
) -> dict[str, Any]:
    if set(chains) != set(nodes):
        raise ValueError("initializer chains do not exactly cover the logical variables")
    hardware = set(host_nodes)
    couplers = {tuple(edge) for edge in host_edges}
    adjacency = {node: set() for node in hardware}
    for u, v in couplers:
        adjacency[u].add(v)
        adjacency[v].add(u)
    occupied: set[int] = set()
    for node in nodes:
        chain = list(chains[node])
        if not chain or not set(chain).issubset(hardware):
            raise ValueError(f"logical variable {node} has an empty or off-host chain")
        if occupied.intersection(chain):
            raise ValueError("initializer chains overlap")
        if not _connected(chain, adjacency):
            raise ValueError(f"logical variable {node} has a disconnected chain")
        occupied.update(chain)
    if len(occupied) > qubit_cap:
        raise ValueError("initializer exceeds the registered qubit cap")
    for u, v in logical_edges:
        if not any(tuple(sorted((a, b))) in couplers for a in chains[u] for b in chains[v]):
            raise ValueError(f"initializer does not realize logical edge {(u, v)}")
    return {
        "connected": True,
        "disjoint": True,
        "logical_edge_contacts": len(logical_edges),
        "on_host": True,
        "realizes_logical_edges": True,
        "total_qubits": len(occupied),
        "within_qubit_cap": True,
    }


def _load_manifest(path: Path, corpus: Path, checksums: Mapping[str, str]) -> dict[str, Any]:
    _verify_checksum(path, checksums)
    manifest = _read_json(path, f"manifest for {corpus.name}")
    _exact_keys(manifest, _MANIFEST_FIELDS, f"manifest for {corpus.name}")
    config = manifest["config"]
    if not isinstance(config, Mapping):
        raise ValueError("quality manifest config must be an object")
    _exact_keys(config, _CONFIG_FIELDS, "quality manifest config")
    if Path(str(manifest["file"])).name != corpus.name:
        raise ValueError("quality manifest points to a different corpus basename")
    digest = _verify_checksum(corpus, checksums)
    if _require_sha(manifest["sha256"], "manifest corpus SHA-256") != digest:
        raise ValueError("quality manifest and SHA256SUMS disagree on corpus content")
    if manifest["objective"] != "chain seam: max_F p_solve (T1) per candidate chain":
        raise ValueError("unsupported legacy quality objective")
    for key in ("n_instances", "seed", "jobs"):
        _integer(manifest[key], f"manifest {key}", minimum=1)
    if config["topology"] not in {"chimera", "pegasus", "zephyr"}:
        raise ValueError("unsupported quality topology")
    if config["graph"] not in {"app", "inkdrop", "random"}:
        raise ValueError("unsupported quality graph family")
    if (
        _number(config["defect_qubits"], "defect_qubits") != 0.0
        or _number(config["defect_couplers"], "defect_couplers") != 0.0
    ):
        raise ReleaseV1CompatibilityError(
            "release-v1.1 rows do not store the defect mask, so an active defective host "
            "cannot be reconstructed exactly"
        )
    return manifest


def _load_split(
    path: Path, corpus_hashes: Mapping[str, str]
) -> tuple[dict[str, Mapping[str, str]], dict[str, Mapping[str, str]], str]:
    split = _read_json(path, "quality split manifest")
    _exact_keys(split, _SPLIT_FIELDS, "quality split manifest")
    if split["schema"] != "embedbench.split-manifest" or split["schema_version"] != 2:
        raise ReleaseV1CompatibilityError("quality preparation requires split schema v2")
    provenance = split["provenance"]
    if not isinstance(provenance, Mapping) or provenance.get("quality_group") != "problem-digest":
        raise ReleaseV1CompatibilityError(
            "quality preparation requires problem-digest grouping; the original v1.1 "
            "instance-id split leaks repeated problems"
        )
    inputs = provenance.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("split provenance inputs must be a list")
    declared: dict[str, str] = {}
    for entry in inputs:
        if not isinstance(entry, Mapping) or set(entry) != {"file", "sha256"}:
            raise ValueError("split provenance input schema differs")
        filename = entry["file"]
        if not isinstance(filename, str) or not filename or filename in declared:
            raise ValueError("split provenance contains an invalid or duplicate filename")
        declared[filename] = _require_sha(entry["sha256"], "split input SHA-256")
    for filename, digest in corpus_hashes.items():
        if declared.get(filename) != digest:
            raise ValueError(f"split manifest does not authenticate {filename}")
    splits = split["splits"]
    groups = split["groups"]
    if not isinstance(splits, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("split/group tables must be objects")
    selected_splits: dict[str, Mapping[str, str]] = {}
    selected_groups: dict[str, Mapping[str, str]] = {}
    assignments: dict[str, str] = {}
    for filename in corpus_hashes:
        table = splits.get(filename)
        group_table = groups.get(filename)
        if not isinstance(table, Mapping) or not isinstance(group_table, Mapping):
            raise ValueError(f"split manifest has no tables for {filename}")
        if set(table) != set(group_table):
            raise ValueError(f"split and group tables differ for {filename}")
        for source_id, partition in table.items():
            group_id = group_table[source_id]
            if partition not in {"train", "val", "test"} or not isinstance(group_id, str):
                raise ValueError("invalid split assignment")
            prior = assignments.setdefault(group_id, partition)
            if prior != partition:
                raise ValueError(f"split leakage for canonical group {group_id}")
        selected_splits[filename] = table
        selected_groups[filename] = group_table
    return selected_splits, selected_groups, _sha256(path)


def _load_references(path: Path, checksums: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ReleaseV1CompatibilityError(
            "release v1.1 stores only an untyped ground_energy scalar. An authenticated "
            "problem-reference sidecar with a certified/planted status and certificate "
            "digest is required; none was supplied"
        )
    _verify_checksum(path, checksums)
    references: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path, "problem-reference sidecar"):
        _exact_keys(row, _REFERENCE_FIELDS, "problem-reference row")
        if row["schema"] != "embedbench.problem-reference" or row["schema_version"] != 1:
            raise ValueError("unsupported problem-reference schema")
        digest = _require_sha(row["problem_digest"], "problem digest")
        status = row["reference_status"]
        if status not in CERTIFIED_REFERENCE_STATUSES:
            raise ReleaseV1CompatibilityError("problem reference is not certified or planted")
        normalized = {
            "certificate_digest": _require_sha(row["certificate_digest"], "certificate digest"),
            "evaluator_protocol_digest": _require_sha(
                row["evaluator_protocol_digest"], "evaluator protocol digest"
            ),
            "problem_digest": digest,
            "reference_energy": _number(row["reference_energy"], "reference energy"),
            "reference_status": status,
            "schema": "embedbench.problem-reference",
            "schema_version": 1,
        }
        if digest in references:
            raise ValueError("problem-reference sidecar repeats a problem digest")
        references[digest] = normalized
    return references


def _source_seed(root_seed: int, mode: str, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{root_seed}:{mode}:{index}".encode()).digest()[:4], "big"
    )


def _problem_split(problem_digest: str) -> str:
    value = int(hashlib.sha256(problem_digest.encode()).hexdigest()[:8], 16) / 2**32
    if value < 0.7:
        return "train"
    if value < 0.8:
        return "val"
    return "test"


def _validate_record(
    raw: Mapping[str, Any], manifest: Mapping[str, Any], qubit_cap: int
) -> dict[str, Any]:
    _exact_keys(raw, _QUALITY_FIELDS, "quality record")
    config = manifest["config"]
    topology = raw["topology"]
    size = _integer(raw["size"], "record size", minimum=1)
    if topology != config["topology"] or size != config["size"]:
        raise ValueError("quality record topology differs from its manifest")
    graph_family = config["graph"]
    match = _SOURCE_ID.fullmatch(str(raw["instance_id"]))
    if match is None:
        raise ValueError("quality record instance_id does not match the v1.1 grammar")
    source = raw["source"]
    source_code = {"witness": "w", "minorminer": "m"}.get(source)
    if (
        match["host"] != topology
        or int(match["size"]) != size
        or match["graph"] != graph_family
        or match["mode"] != raw["mode"]
        or match["source"] != source_code
    ):
        raise ValueError("quality record identity fields disagree")
    expected_seed = _source_seed(int(manifest["seed"]), str(raw["mode"]), int(match["index"]))
    if int(match["seed"]) != expected_seed:
        raise ValueError("quality record instance seed disagrees with the generator manifest")

    logical_nodes, h, j, e0 = _canonical_problem(raw["problem"])
    if _integer(raw["n_vars"], "n_vars", minimum=1) != len(logical_nodes):
        raise ValueError("n_vars differs from the full problem")
    if not math.isclose(_number(raw["ground_energy"], "ground_energy"), e0, abs_tol=1e-12):
        raise ValueError("ground_energy differs from problem.e0")
    focus = _integer(raw["focus"], "focus", minimum=0)
    if focus not in logical_nodes:
        raise ValueError("focus is outside the logical problem")
    all_chains = _chains(raw["all_chains"], "all_chains")
    if set(all_chains) != set(logical_nodes) - {focus}:
        raise ValueError("all_chains must contain every non-focus logical variable exactly once")
    frozen = _chains(raw["frozen"], "frozen")
    if any(node not in all_chains or chain != all_chains[node] for node, chain in frozen.items()):
        raise ValueError("frozen is not an exact subset of all_chains")

    host_nodes, host_edges = _host_graph(str(topology), size)
    host_edge_set = {tuple(edge) for edge in host_edges}
    window = _list_ints(raw["window_nodes"], "window_nodes", nonempty=True)
    if window != sorted(window) or not set(window).issubset(host_nodes):
        raise ValueError("window_nodes must be sorted and on the active host")
    raw_window_edges = raw["window_edges"]
    if not isinstance(raw_window_edges, list):
        raise ValueError("window_edges must be a list")
    window_edges: list[list[int]] = []
    for edge in raw_window_edges:
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError("window edge must have two endpoints")
        u = _integer(edge[0], "window edge endpoint", minimum=0)
        v = _integer(edge[1], "window edge endpoint", minimum=0)
        pair = (min(u, v), max(u, v))
        if u == v or set(pair) - set(window) or pair not in host_edge_set:
            raise ValueError("window edge is invalid")
        window_edges.append(list(pair))
    induced = [edge for edge in host_edges if edge[0] in set(window) and edge[1] in set(window)]
    if window_edges != sorted(window_edges) or window_edges != induced:
        raise ValueError("window_edges must be the complete sorted induced host subgraph")

    raw_candidates = raw["candidates"]
    if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
        raise ValueError("quality record needs at least two candidates")
    candidates: list[list[int]] = []
    for index, candidate in enumerate(raw_candidates):
        values = _list_ints(candidate, f"candidate {index}", nonempty=True)
        if values != sorted(values) or not set(values).issubset(window):
            raise ValueError("candidate chains must be sorted subsets of the window")
        candidates.append(values)
    if len({tuple(candidate) for candidate in candidates}) != len(candidates):
        raise ValueError("quality record repeats a candidate chain")
    count = len(candidates)
    q_values = raw["Q"]
    if not isinstance(q_values, list) or len(q_values) != count:
        raise ValueError("Q must align with candidates")
    if [_integer(value, "Q value", minimum=1) for value in q_values] != [
        len(candidate) for candidate in candidates
    ]:
        raise ValueError("Q values differ from candidate sizes")
    for field in ("p_solve", "best_F", "stage"):
        if not isinstance(raw[field], list) or len(raw[field]) != count:
            raise ValueError(f"{field} must align with candidates")
    probabilities = [_number(value, "p_solve") for value in raw["p_solve"]]
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("p_solve values must lie in [0,1]")
    if any(_number(value, "best_F") <= 0 for value in raw["best_F"]):
        raise ValueError("best_F values must be positive")
    stages = [_integer(value, "stage", minimum=1) for value in raw["stage"]]
    if any(value not in {1, 2} for value in stages):
        raise ValueError("stage values must be 1 or 2")
    indices = {
        name: _integer(raw[name], name, minimum=0)
        for name in ("best_index", "original_index", "resource_index")
    }
    if any(index >= count for index in indices.values()):
        raise ValueError("quality record index is outside the candidate support")
    if stages[indices["best_index"]] != 2:
        raise ValueError("best_index must identify a fresh-seed stage-2 candidate")

    logical_edges = [[int(edge[0]), int(edge[1])] for edge in j]
    selected = dict(all_chains)
    selected[focus] = candidates[indices["original_index"]]
    validation = _validate_embedding(
        selected, logical_nodes, logical_edges, host_nodes, host_edges, qubit_cap
    )
    # Every stored action was claimed valid by release v1.1. Verify each one rather than
    # letting a corrupt non-selected alternative survive authentication.
    for candidate in candidates:
        alternative = dict(all_chains)
        alternative[focus] = candidate
        _validate_embedding(
            alternative, logical_nodes, logical_edges, host_nodes, host_edges, qubit_cap
        )

    neighbours = _list_ints(raw["neighbours"], "neighbours")
    if any(node not in all_chains for node in neighbours):
        raise ValueError("neighbours includes an unknown or focus variable")
    if not math.isclose(
        _number(raw["focus_h"], "focus_h"),
        dict((int(node), float(value)) for node, value in h)[focus],
        abs_tol=1e-12,
    ):
        raise ValueError("focus_h differs from the full problem")
    edge_j = raw["edge_J"]
    if not isinstance(edge_j, list) or len(edge_j) != len(neighbours):
        raise ValueError("edge_J must align with neighbours")
    coupling = {(int(u), int(v)): float(value) for u, v, value in j}
    for neighbour, value in zip(neighbours, edge_j, strict=True):
        pair = (min(focus, neighbour), max(focus, neighbour))
        if not math.isclose(_number(value, "edge_J"), coupling.get(pair, 0.0), abs_tol=1e-12):
            raise ValueError("edge_J differs from the full problem")

    problem_digest = quality_problem_digest(raw["problem"])
    return {
        "chains": [selected[node] for node in logical_nodes],
        "family": f"release_v1_1/{graph_family}/{raw['mode']}",
        "h": h,
        "host_edges": host_edges,
        "host_nodes": host_nodes,
        "logical_edges": logical_edges,
        "logical_nodes": logical_nodes,
        "j": j,
        "problem_digest": problem_digest,
        "source_id": raw["instance_id"],
        "source_record_digest": content_digest(raw),
        "topology": topology,
        "validation": validation,
        "e0": e0,
    }


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _jsonl(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _write(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_release_v1(
    corpus_paths: Sequence[str | os.PathLike[str]],
    split_manifest_path: str | os.PathLike[str],
    checksums_path: str | os.PathLike[str],
    problem_references_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    qubit_cap: int,
) -> dict[str, Any]:
    """Authenticate and convert selected release-v1.1 quality corpora atomically.

    Inputs can be a deliberate subset of the nine quality files, but each basename must be
    present in the leakage-safe problem-digest split manifest.  A corpus file's sibling
    ``.manifest.json`` is mandatory.  ``output_dir`` must not already exist.
    """

    cap = _integer(qubit_cap, "qubit_cap", minimum=1)
    paths = [Path(path) for path in corpus_paths]
    if not paths:
        raise ValueError("at least one quality corpus is required")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("quality corpus basenames must be unique")
    if any(not path.name.startswith("quality_") or path.suffix != ".jsonl" for path in paths):
        raise ValueError("only release-v1.1 quality JSONL corpora are accepted")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"prepared output already exists: {destination}")

    checksums_file = Path(checksums_path)
    checksums = _checksum_table(checksums_file)
    manifests: dict[str, dict[str, Any]] = {}
    corpus_hashes: dict[str, str] = {}
    for corpus in sorted(paths, key=lambda path: path.name):
        manifest_path = Path(str(corpus) + ".manifest.json")
        manifests[corpus.name] = _load_manifest(manifest_path, corpus, checksums)
        corpus_hashes[corpus.name] = _sha256(corpus)
    split_tables, group_tables, split_sha = _load_split(Path(split_manifest_path), corpus_hashes)
    references = _load_references(Path(problem_references_path), checksums)

    normalized: list[dict[str, Any]] = []
    seen_source_rows: set[tuple[str, str]] = set()
    for corpus in sorted(paths, key=lambda path: path.name):
        rows = _read_jsonl(corpus, corpus.name)
        manifest = manifests[corpus.name]
        stats = manifest["stats"]
        if not isinstance(stats, Mapping) or stats.get("samples_kept") != len(rows):
            raise ValueError(f"{corpus.name} row count differs from manifest stats")
        source_splits = split_tables[corpus.name]
        source_groups = group_tables[corpus.name]
        source_ids: set[str] = set()
        for raw in rows:
            item = _validate_record(raw, manifest, cap)
            source_id = item["source_id"]
            source_ids.add(source_id)
            if source_id not in source_splits:
                raise ValueError(f"split manifest omits source instance {source_id!r}")
            declared_group = source_groups[source_id]
            if declared_group != f"problem-digest:{item['problem_digest']}":
                raise ValueError("split problem group differs from record content")
            if source_splits[source_id] != _problem_split(item["problem_digest"]):
                raise ValueError("split partition differs from the registered digest rule")
            key = (source_id, item["source_record_digest"])
            if key in seen_source_rows:
                raise ValueError("release corpus repeats an identical source row")
            seen_source_rows.add(key)
            item["partition"] = source_splits[source_id]
            item["source_corpus"] = corpus.name
            normalized.append(item)
        if source_ids != set(source_splits):
            raise ValueError(f"split table coverage differs from {corpus.name} source IDs")

    problem_digests = {item["problem_digest"] for item in normalized}
    if set(references) != problem_digests:
        raise ReleaseV1CompatibilityError(
            "problem-reference sidecar must exactly cover selected semantic problems; "
            f"missing={sorted(problem_digests - set(references))}, "
            f"unknown={sorted(set(references) - problem_digests)}"
        )
    for item in normalized:
        reference = references[item["problem_digest"]]
        if not math.isclose(reference["reference_energy"], item["e0"], abs_tol=1e-12):
            raise ReleaseV1CompatibilityError(
                "certified/planted reference energy differs from the legacy problem payload"
            )

    policies: dict[str, dict[str, Any]] = {}
    policy_reference: dict[str, dict[str, Any]] = {}
    tasks: dict[str, dict[str, Any]] = {}
    task_sources: dict[str, set[str]] = {}
    split_tasks: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    lineage_to_split: dict[str, str] = {}
    for item in normalized:
        lineage = "logical-" + content_digest(
            {
                "domain": "isingfold-logical-problem-v1",
                "h": item["h"],
                "j": item["j"],
                "logical_edges": item["logical_edges"],
                "logical_nodes": item["logical_nodes"],
            }
        )
        instance_id = "instance-" + content_digest(
            {
                "domain": "isingfold-instance-v1",
                "host_edges": item["host_edges"],
                "host_nodes": item["host_nodes"],
                "split_unit_id": lineage,
            }
        )
        policy_payload = {
            "family": item["family"],
            "h": item["h"],
            "host_edges": item["host_edges"],
            "host_nodes": item["host_nodes"],
            "instance_id": instance_id,
            "j": item["j"],
            "logical_edges": item["logical_edges"],
            "logical_nodes": item["logical_nodes"],
            "schema": "isingfold.policy-instance",
            "schema_version": 1,
            "topology": item["topology"],
        }
        policy = _with_digest(policy_payload)
        previous_policy = policies.setdefault(instance_id, policy)
        if previous_policy != policy:
            raise ValueError("one semantic policy instance has conflicting public content")
        policy_reference[instance_id] = references[item["problem_digest"]]
        partition = item["partition"]
        prior = lineage_to_split.setdefault(lineage, partition)
        if prior != partition:
            raise ValueError("one base logical lineage appears in multiple partitions")
        task_id = "task-" + content_digest(
            {
                "chains": item["chains"],
                "domain": "isingfold-release-v1-profile-i-task-v1",
                "instance_id": instance_id,
            }
        )
        task_sources.setdefault(task_id, set()).add(item["source_record_digest"])
        candidate_id = "candidate-" + content_digest(
            {
                "chains": item["chains"],
                "domain": "isingfold-release-v1-initializer-v1",
                "instance_id": instance_id,
            }
        )
        group_id = "group-" + content_digest(
            {"domain": "isingfold-release-v1-group-v1", "task_id": task_id}
        )
        task = {
            "candidate_id": candidate_id,
            "chains": item["chains"],
            "group_id": group_id,
            "instance_id": instance_id,
            "instance_record_digest": policy["record_digest"],
            "qubit_cap": cap,
            "schema": "isingfold.profile-i-initializer",
            "schema_version": 1,
            "task_id": task_id,
            "validation": item["validation"],
        }
        previous_task = tasks.setdefault(task_id, task)
        if previous_task != task:
            raise ValueError("one deduplicated task has conflicting initializer content")
        split_tasks[partition].add(task_id)

    initializers = []
    for task_id, task in sorted(tasks.items()):
        group_record_digest = content_digest(
            {
                "domain": "isingfold-release-v1-source-group-v1",
                "source_record_digests": sorted(task_sources[task_id]),
                "task_id": task_id,
            }
        )
        initializers.append(_with_digest({**task, "group_record_digest": group_record_digest}))
    policy_records = sorted(policies.values(), key=lambda row: row["instance_id"])
    targets = []
    for instance_id, policy in sorted(policies.items()):
        reference = policy_reference[instance_id]
        targets.append(
            _with_digest(
                {
                    "certificate_digest": reference["certificate_digest"],
                    "evaluator_protocol_digest": reference["evaluator_protocol_digest"],
                    "instance_id": instance_id,
                    "instance_record_digest": policy["record_digest"],
                    "reference_energy": reference["reference_energy"],
                    "reference_status": reference["reference_status"],
                    "schema": "isingfold.evaluator-target",
                    "schema_version": 1,
                }
            )
        )
    split_record = _with_digest(
        {
            "lineage_to_split": dict(sorted(lineage_to_split.items())),
            "schema": "isingfold.lineage-splits",
            "schema_version": 1,
            "test": sorted(split_tasks["test"]),
            "train": sorted(split_tasks["train"]),
            "val": sorted(split_tasks["val"]),
        }
    )
    contents = {
        "policy_instances.jsonl": _jsonl(policy_records),
        "initializers.jsonl": _jsonl(initializers),
        "evaluator_targets.jsonl": _jsonl(targets),
        "splits.json": canonical_json_bytes(split_record) + b"\n",
    }
    outputs = {
        filename: {
            "records": (
                len(policy_records)
                if filename == "policy_instances.jsonl"
                else len(initializers)
                if filename == "initializers.jsonl"
                else len(targets)
                if filename == "evaluator_targets.jsonl"
                else 1
            ),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for filename, raw in contents.items()
    }
    source_commitment = {
        "corpora": dict(sorted(corpus_hashes.items())),
        "manifests": {
            name: _sha256(Path(str(path) + ".manifest.json"))
            for name, path in sorted((path.name, path) for path in paths)
        },
        "problem_references": _sha256(Path(problem_references_path)),
        "sha256sums": _sha256(checksums_file),
        "split_manifest": split_sha,
    }
    manifest = _with_digest(
        {
            "counts": {
                "evaluator_targets": len(targets),
                "initializers": len(initializers),
                "policy_instances": len(policy_records),
            },
            "outputs": outputs,
            "policy_model_feature_allowlist": [
                "family",
                "h",
                "host_edges",
                "host_nodes",
                "j",
                "logical_edges",
                "logical_nodes",
                "topology",
            ],
            "qubit_cap": cap,
            "schema": "isingfold.prepared-candidate-bank",
            "schema_version": 1,
            "source_bank_manifest_record_digest": content_digest(source_commitment),
            "source_sha256": source_commitment,
        }
    )
    contents["manifest.json"] = canonical_json_bytes(manifest) + b"\n"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.prepare-", dir=destination.parent)
    )
    try:
        for filename, raw in contents.items():
            _write(temporary / filename, raw)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


__all__ = [
    "ReleaseV1CompatibilityError",
    "prepare_release_v1",
    "quality_problem_digest",
]

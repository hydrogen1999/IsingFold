#!/usr/bin/env python3
"""Rescore quality candidates with a traceable, high-read surrogate audit.

The emitted schema-v2 rows bind every score vector to the exact corpus bytes and
ordered candidate support.  They also record the registered strength and seed
schedules needed for paper-facing reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

import networkx as nx
import numpy as np
from embedbench.embedding import Embedding, LogicalProblem
from embedbench.models_chain import (
    AUDIT_AGGREGATION,
    AUDIT_GENERATOR,
    AUDIT_SCHEMA,
    AUDIT_SCHEMA_VERSION,
    AUDIT_SEED_SCHEDULE,
    AUDIT_STRENGTH_SCHEDULE,
    MIN_HIGH_READS_PER_STRENGTH,
    REGISTERED_STRENGTH_COUNT,
    quality_candidate_signature,
)
from embedbench.structural import host_graph
from embedbench.surrogate import (
    EXACT_GROUND_STATE_MAX_N,
    default_strength_grid,
    solve_probability_at,
)
from quality_v2_paper_contract import (
    load_paper_preregistration,
    require_exact_json,
    strict_json_loads,
    verify_preregistered_policy_freezes,
)
from training_artifacts import runtime_provenance
from training_splits import (
    data_provenance,
    load_split_records,
    quality_problem_digest,
    quality_problem_id,
)

NUM_SWEEPS = 200
EXACT_ENUMERATION_BATCH_STATES = 1 << 17
AUDIT_SHARD_MANIFEST_SCHEMA = "embedbench.quality-v2-audit-shard"
AUDIT_SHARD_MANIFEST_SCHEMA_VERSION = 2
AUDIT_RELEASE_MANIFEST_SCHEMA = "embedbench.quality-v2-audit-release"
AUDIT_RELEASE_MANIFEST_SCHEMA_VERSION = 2


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_audit_identity(filename: object, record: Mapping[str, object]) -> bytes:
    """Return the contract-registered, cross-process audit identity bytes."""

    instance_id = record.get("instance_id")
    focus = record.get("focus")
    if (
        type(filename) is not str
        or not filename
        or Path(filename).name != filename
        or type(instance_id) is not str
        or not instance_id
        or type(focus) is not int
    ):
        raise ValueError("audit identity requires basename file, string instance_id, and int focus")
    return json.dumps(
        [filename, instance_id, focus],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def audit_shard_index(identity: bytes, shard_count: int) -> int:
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise ValueError("audit shard_count must be a positive integer")
    return int.from_bytes(hashlib.sha256(identity).digest(), "big") % shard_count


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_audit_release_commitment(
    audit_paths: Sequence[str | Path],
    *,
    audit_contract_binding: Mapping[str, object],
    preregistration_binding: Mapping[str, object],
    selection_binding: Mapping[str, object],
    audit_source_binding: Mapping[str, object],
    expected_shard_count: int,
) -> tuple[dict[str, object], bytes]:
    """Build the byte commitment a custodian retains outside the audit release.

    Per-shard manifests are useful integrity indexes but cannot authenticate themselves. This
    document commits to both every JSONL and every shard-manifest byte stream. Its returned
    SHA-256 must be registered independently before a paper evaluator is allowed to consume
    any score payload.
    """

    if (
        isinstance(expected_shard_count, bool)
        or not isinstance(expected_shard_count, int)
        or expected_shard_count <= 0
        or len(audit_paths) != expected_shard_count
    ):
        raise ValueError("audit release requires exactly the registered shard count")
    entries: list[dict[str, object]] = []
    seen_files: set[str] = set()
    seen_indices: set[int] = set()
    for raw_path in audit_paths:
        path = Path(raw_path)
        if path.name in seen_files or Path(path.name).name != path.name:
            raise ValueError("audit release shard basenames must be unique")
        manifest_path = path.with_name(f"{path.name}.manifest.json")
        try:
            shard_payload = path.read_bytes()
            manifest_payload = manifest_path.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid audit release shard {path}") from error
        manifest = strict_json_loads(
            manifest_payload,
            location=f"audit shard manifest {manifest_path}",
        )
        jsonl = manifest.get("jsonl") if isinstance(manifest, Mapping) else None
        shard_index = manifest.get("shard_index") if isinstance(manifest, Mapping) else None
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("schema") != AUDIT_SHARD_MANIFEST_SCHEMA
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != AUDIT_SHARD_MANIFEST_SCHEMA_VERSION
            or manifest.get("shard_count") != expected_shard_count
            or isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or not 0 <= shard_index < expected_shard_count
            or shard_index in seen_indices
            or manifest.get("audit_contract") != audit_contract_binding
            or manifest.get("preregistration") != preregistration_binding
            or manifest.get("selection") != selection_binding
            or manifest.get("audit_source") != audit_source_binding
            or not isinstance(jsonl, Mapping)
            or set(jsonl) != {"file", "sha256", "bytes"}
            or jsonl.get("file") != path.name
            or jsonl.get("sha256") != hashlib.sha256(shard_payload).hexdigest()
            or type(jsonl.get("bytes")) is not int
            or jsonl.get("bytes") != len(shard_payload)
        ):
            raise ValueError(f"audit shard {path} cannot enter the external release commitment")
        seen_files.add(path.name)
        seen_indices.add(shard_index)
        entries.append(
            {
                "shard_index": shard_index,
                "file": path.name,
                "sha256": hashlib.sha256(shard_payload).hexdigest(),
                "bytes": len(shard_payload),
                "manifest_file": manifest_path.name,
                "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "manifest_bytes": len(manifest_payload),
            }
        )
    if seen_indices != set(range(expected_shard_count)):
        raise ValueError("audit release does not cover every registered shard exactly once")
    entries.sort(key=lambda entry: int(entry["shard_index"]))
    document: dict[str, object] = {
        "schema": AUDIT_RELEASE_MANIFEST_SCHEMA,
        "schema_version": AUDIT_RELEASE_MANIFEST_SCHEMA_VERSION,
        "commitment_scope": "exact_audit_jsonl_and_shard_manifest_bytes",
        "audit_contract": dict(audit_contract_binding),
        "preregistration": dict(preregistration_binding),
        "selection": dict(selection_binding),
        "audit_source": dict(audit_source_binding),
        "shard_count": expected_shard_count,
        "shards": entries,
    }
    payload = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    return document, payload


def finalize_paper_audit_shard(
    output: str | Path,
    paper_payloads: Sequence[tuple[bytes, Path, bytes]],
    *,
    audit_contract_binding: Mapping[str, object],
    preregistration_binding: Mapping[str, object],
    selection_binding: Mapping[str, object],
    audit_source_binding: Mapping[str, object],
    shard_count: int,
    shard_index: int,
    complete_fixed_test_record_count: int,
) -> dict[str, object]:
    """Atomically materialize the canonical JSONL and its byte-level shard manifest."""

    destination = Path(output)
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
        or isinstance(complete_fixed_test_record_count, bool)
        or not isinstance(complete_fixed_test_record_count, int)
        or complete_fixed_test_record_count < 0
    ):
        raise ValueError("paper audit shard has invalid registered dimensions")
    identities = [identity for identity, _, _ in paper_payloads]
    if len(identities) != len(set(identities)):
        raise ValueError("paper audit shard contains duplicate canonical identities")
    if any(audit_shard_index(identity, shard_count) != shard_index for identity in identities):
        raise ValueError("paper audit shard contains an identity assigned to another shard")
    canonical_payloads = sorted(paper_payloads, key=lambda item: item[0])
    for identity, record_path, payload in canonical_payloads:
        expected_record_path = destination.with_name(f"{destination.name}.records") / (
            f"{hashlib.sha256(identity).hexdigest()}.json"
        )
        if (
            record_path.resolve() != expected_record_path.resolve()
            or record_path.read_bytes() != payload
        ):
            raise ValueError("paper audit record is not at its canonical byte-identical path")
    merged_payload = b"".join(payload for _, _, payload in canonical_payloads)
    _atomic_write(destination, merged_payload)
    manifest: dict[str, object] = {
        "schema": AUDIT_SHARD_MANIFEST_SCHEMA,
        "schema_version": AUDIT_SHARD_MANIFEST_SCHEMA_VERSION,
        "audit_contract": dict(audit_contract_binding),
        "preregistration": dict(preregistration_binding),
        "selection": dict(selection_binding),
        "audit_source": dict(audit_source_binding),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "assignment": "sha256(canonical_audit_identity) mod shard_count",
        "canonical_audit_identity": "json_compact_utf8_array[file,instance_id,focus]",
        "merge_order": "canonical_audit_identity_lexicographic",
        "complete_fixed_test_record_count": complete_fixed_test_record_count,
        "record_count": len(canonical_payloads),
        "records": [
            {
                "identity": identity.decode("utf-8"),
                "relative_path": record_path.relative_to(destination.parent).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for identity, record_path, payload in canonical_payloads
        ],
        "jsonl": {
            "file": destination.name,
            "sha256": hashlib.sha256(merged_payload).hexdigest(),
            "bytes": len(merged_payload),
        },
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(destination.with_name(f"{destination.name}.manifest.json"), manifest_payload)
    return manifest


def corpus_margin(record: Mapping[str, object]) -> float:
    probabilities = sorted(record["p_solve"], reverse=True)
    return float(probabilities[0] - probabilities[1])


def _records_for_audit(
    records: Sequence[dict],
    *,
    per_file: int,
    low_margin: float,
    rng: random.Random,
    paper_mode: bool,
) -> list[dict]:
    if paper_mode:
        if len(records) > per_file:
            raise ValueError("paper audit --per-file must cover complete fixed-test record support")
        # The locked audit must not inspect release p_solve, best_index, or stage labels,
        # even for prioritisation. Full support makes label-based sampling unnecessary.
        return sorted(
            records,
            key=lambda record: (
                str(record.get("instance_id")),
                int(record.get("focus", -1)),
                quality_problem_digest(record),
            ),
        )

    low_margin_records = [record for record in records if corpus_margin(record) < low_margin]
    remaining = [record for record in records if corpus_margin(record) >= low_margin]
    rng.shuffle(low_margin_records)
    rng.shuffle(remaining)
    return (low_margin_records + remaining)[:per_file]


def build_audit_record(
    corpus_path: str | Path,
    record: Mapping[str, object],
    *,
    scores: Sequence[float],
    reads: int,
    base_seed: int,
    corpus_digest: str | None = None,
    paper_binding: Mapping[str, object] | None = None,
    realized_strengths: Sequence[float] | None = None,
    realized_seeds: Sequence[Sequence[int]] | None = None,
    strength_p_solve: Sequence[Sequence[float]] | None = None,
    strength_success_counts: Sequence[Sequence[int]] | None = None,
    ground_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one portable audit row without rerunning the surrogate."""
    path = Path(corpus_path)
    normalized_scores = [float(score) for score in scores]
    high_read_best = int(np.argmax(normalized_scores))
    row = {
        "audit_schema": AUDIT_SCHEMA,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "file": path.name,
        "corpus_sha256": corpus_digest or file_sha256(path),
        "instance_id": record["instance_id"],
        "focus": record["focus"],
        "candidate_signature": quality_candidate_signature(record),
        "resource_index": record["resource_index"],
        "original_index": record.get("original_index", -1),
        "high_read_best": high_read_best,
        "high_read_scores": normalized_scores,
        "reads": reads,
        "provenance": {
            "generator": AUDIT_GENERATOR,
            "reads_per_strength": reads,
            "num_sweeps": NUM_SWEEPS,
            "base_seed": base_seed,
            "registered_strength_count": REGISTERED_STRENGTH_COUNT,
            "seed_schedule": AUDIT_SEED_SCHEDULE,
            "strength_schedule": AUDIT_STRENGTH_SCHEDULE,
            "aggregation": AUDIT_AGGREGATION,
        },
    }
    if paper_binding is None:
        row.update(
            {
                "corpus_best": record["best_index"],
                "corpus_margin": corpus_margin(record),
                "corpus_scores": record["p_solve"],
            }
        )
    paper_values = (realized_strengths, realized_seeds, ground_reference)
    if paper_binding is not None:
        if any(value is None for value in paper_values):
            raise ValueError("paper audit rows require realized schedules and ground evidence")
        row["paper_binding"] = {
            **dict(paper_binding),
            "ground_reference": dict(ground_reference or {}),
            "realized_strengths": [float(value) for value in realized_strengths or ()],
            "realized_seeds": [
                [int(seed) for seed in candidate_seeds] for candidate_seeds in realized_seeds or ()
            ],
        }
        row["provenance"].update(
            {
                "score_evidence": "integer_success_counts_per_candidate_strength",
                "probability_derivation": ("success_count_divided_by_reads_per_strength_binary64"),
            }
        )
        if strength_p_solve is None or strength_success_counts is None:
            raise ValueError(
                "paper audit rows require per-strength probabilities and integer success counts"
            )
        strength_count = len(realized_strengths or ())
        if (
            isinstance(reads, bool)
            or not isinstance(reads, int)
            or reads <= 0
            or strength_count <= 0
            or len(strength_p_solve) != len(normalized_scores)
            or len(strength_success_counts) != len(normalized_scores)
            or any(len(values) != strength_count for values in strength_p_solve)
            or any(len(values) != strength_count for values in strength_success_counts)
            or any(
                isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= reads
                for candidate_counts in strength_success_counts
                for count in candidate_counts
            )
            or any(
                count / reads != float(score)
                for candidate_scores, candidate_counts in zip(
                    strength_p_solve,
                    strength_success_counts,
                    strict=True,
                )
                for score, count in zip(candidate_scores, candidate_counts, strict=True)
            )
            or any(
                max(float(score) for score in candidate_scores) != aggregate
                for candidate_scores, aggregate in zip(
                    strength_p_solve,
                    normalized_scores,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "paper audit probabilities require complete exact integer success-count evidence"
            )
        row["strength_p_solve"] = [
            [float(score) for score in candidate_scores] for candidate_scores in strength_p_solve
        ]
        row["strength_success_counts"] = [
            [int(count) for count in candidate_counts]
            for candidate_counts in strength_success_counts
        ]
    elif any(value is not None for value in paper_values):
        raise ValueError("realized paper schedules require a paper binding")
    return row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--per-file", type=int, default=20)
    parser.add_argument("--reads", type=int, default=4000)
    parser.add_argument(
        "--low-margin",
        type=float,
        default=0.06,
        help="prefer records whose corpus best-vs-second margin is below this",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--splits",
        required=True,
        help=(
            "required schema-v2 splits.json; rescore only the explicitly selected split "
            "(per-file cap still applies)"
        ),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument(
        "--audit-contract",
        help="required with --splits --split test; frozen paper-audit contract",
    )
    parser.add_argument(
        "--audit-contract-sha256",
        help="pre-registered SHA-256 of --audit-contract",
    )
    parser.add_argument(
        "--selection",
        help="completed validation-only selection artifact; required for the test split",
    )
    parser.add_argument(
        "--selection-sha256",
        help="pre-registered SHA-256 of --selection; required for the test split",
    )
    parser.add_argument(
        "--selection-root",
        help="staged root containing the grid, receipts, and winner checkpoints",
    )
    parser.add_argument(
        "--preregistration-manifest",
        help="single preregistration manifest authenticated before test-label generation",
    )
    parser.add_argument(
        "--preregistration-manifest-sha256",
        help="manifest SHA-256 retained independently before test-label generation",
    )
    parser.add_argument(
        "--policy-freezes",
        nargs="+",
        help="all registered winner-seed policy freezes bound by the preregistration",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="deterministic audit shard count (paper contract registers 64)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="zero-based deterministic audit shard index",
    )
    return parser


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid quality corpus {path}") from error
    for line_number, line in enumerate(payload.splitlines(), start=1):
        record = strict_json_loads(line, location=f"{path}:{line_number} quality row")
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: quality row must be an object")
        records.append(record)
    return records


def _rescore_record(
    record,
    host,
    reads: int,
    base_seed: int,
) -> tuple[list[float], list[list[float]], list[list[int]], list[float], list[list[int]]]:
    problem, strengths, realized_seeds = _audit_schedule(record, base_seed)
    raw_problem = record["problem"]
    frozen = {
        int(variable): chain
        for variable, chain in (record.get("all_chains") or record["frozen"]).items()
    }
    scores = []
    strength_p_solve: list[list[float]] = []
    strength_success_counts: list[list[int]] = []
    for candidate_index, candidate in enumerate(record["candidates"]):
        chains = {variable: frozenset(chain) for variable, chain in frozen.items()}
        chains[int(record["focus"])] = frozenset(candidate)
        embedding = Embedding.from_chains(chains, host, problem)
        candidate_seeds = realized_seeds[candidate_index]
        candidate_scores: list[float] = []
        candidate_counts: list[int] = []
        for strength_index, strength in enumerate(strengths):
            result = solve_probability_at(
                embedding,
                problem,
                raw_problem["e0"],
                strength,
                num_reads=reads,
                num_sweeps=NUM_SWEEPS,
                seed=candidate_seeds[strength_index],
            )
            if result.num_reads != reads:
                raise RuntimeError("surrogate returned an unregistered read count")
            success_count = round(float(result.p_solve) * reads)
            derived_probability = success_count / reads
            if not math.isclose(
                float(result.p_solve),
                derived_probability,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError("surrogate probability is not backed by an integer hit count")
            candidate_scores.append(derived_probability)
            candidate_counts.append(success_count)
        strength_p_solve.append(candidate_scores)
        strength_success_counts.append(candidate_counts)
        scores.append(max(candidate_scores))
    return scores, strength_p_solve, strength_success_counts, strengths, realized_seeds


def _audit_schedule(
    record: Mapping[str, object],
    base_seed: int,
) -> tuple[LogicalProblem, list[float], list[list[int]]]:
    raw_problem = record["problem"]
    problem = _strict_legacy_problem(raw_problem)
    strengths = default_strength_grid(problem, REGISTERED_STRENGTH_COUNT)
    realized_seeds = [
        [
            (base_seed + 7919 * candidate_index + 31 * strength_index) % (2**31)
            for strength_index in range(len(strengths))
        ]
        for candidate_index in range(len(record["candidates"]))
    ]
    return problem, strengths, realized_seeds


def _strict_legacy_problem(raw_problem: object) -> LogicalProblem:
    """Parse legacy JSON Ising payloads without allowing identifier aliases."""

    if not isinstance(raw_problem, Mapping):
        raise ValueError("paper audit record has no problem payload")
    raw_h = raw_problem.get("h")
    raw_j = raw_problem.get("J")
    if not isinstance(raw_h, Mapping) or not raw_h or not isinstance(raw_j, list):
        raise ValueError("paper audit problem payload must contain nonempty h and J")
    h: dict[int, float] = {}
    for raw_node, raw_bias in raw_h.items():
        if not isinstance(raw_node, str):
            raise ValueError("paper audit problem.h keys must be canonical integer strings")
        try:
            node = int(raw_node)
        except ValueError as error:
            raise ValueError(
                "paper audit problem.h keys must be canonical integer strings"
            ) from error
        if str(node) != raw_node or node in h:
            raise ValueError("paper audit problem.h keys must be canonical integer strings")
        h[node] = float(_finite_fraction(raw_bias, f"problem.h[{raw_node!r}]"))
    j: dict[tuple[int, int], float] = {}
    for index, edge in enumerate(raw_j):
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError(f"problem.J[{index}] must be a canonical edge triple")
        left, right, raw_weight = edge
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or left >= right
            or left not in h
            or right not in h
            or (left, right) in j
        ):
            raise ValueError(f"problem.J[{index}] is not a canonical logical edge")
        j[(left, right)] = float(_finite_fraction(raw_weight, f"problem.J[{index}][2]"))
    return LogicalProblem.from_dicts(h, j)


def _paper_row_payload(
    path: Path,
    record: Mapping[str, object],
    *,
    scores: Sequence[float],
    reads: int,
    base_seed: int,
    corpus_digest: str,
    paper_binding: Mapping[str, object],
    strengths: Sequence[float],
    realized_seeds: Sequence[Sequence[int]],
    ground_reference: Mapping[str, object],
    strength_p_solve: Sequence[Sequence[float]],
    strength_success_counts: Sequence[Sequence[int]],
) -> tuple[dict[str, object], bytes]:
    if len(scores) != len(record["candidates"]) or any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
        for score in scores
    ):
        raise ValueError("paper audit scores must be a complete finite probability vector")
    if (
        len(strength_p_solve) != len(scores)
        or any(len(candidate_scores) != len(strengths) for candidate_scores in strength_p_solve)
        or any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            for candidate_scores in strength_p_solve
            for score in candidate_scores
        )
        or any(
            not math.isclose(
                max(float(value) for value in candidate_scores),
                float(aggregate),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for candidate_scores, aggregate in zip(strength_p_solve, scores, strict=True)
        )
    ):
        raise ValueError("paper audit per-strength scores are incomplete or inconsistent")
    if (
        len(strength_success_counts) != len(scores)
        or any(
            len(candidate_counts) != len(strengths) for candidate_counts in strength_success_counts
        )
        or any(
            isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= reads
            for candidate_counts in strength_success_counts
            for count in candidate_counts
        )
        or any(
            count / reads != float(score)
            for candidate_scores, candidate_counts in zip(
                strength_p_solve,
                strength_success_counts,
                strict=True,
            )
            for score, count in zip(candidate_scores, candidate_counts, strict=True)
        )
    ):
        raise ValueError("paper audit probabilities require exact integer success-count evidence")
    row = build_audit_record(
        path,
        record,
        scores=scores,
        reads=reads,
        base_seed=base_seed,
        corpus_digest=corpus_digest,
        paper_binding=paper_binding,
        realized_strengths=strengths,
        realized_seeds=realized_seeds,
        strength_p_solve=strength_p_solve,
        strength_success_counts=strength_success_counts,
        ground_reference=ground_reference,
    )
    payload = (
        json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return row, payload


def _load_resumable_paper_row(
    record_path: Path,
    path: Path,
    record: Mapping[str, object],
    **expected: object,
) -> tuple[dict[str, object], bytes] | None:
    """Never trust a partial paper-audit cache as score evidence.

    A cached row cannot authenticate its own ``high_read_scores``. Paper shards are small,
    so interrupted shards are recomputed in full unless a future protocol preregisters a
    complete shard digest independently. The parameters remain for API compatibility with
    older tooling and tests.
    """

    del record_path, path, record, expected
    return None


def _finite_fraction(value: object, location: str) -> Fraction:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite and numeric")
    return Fraction(str(value))


def _ferromagnetic_min_cut_certificate(
    problem: LogicalProblem,
) -> tuple[float, dict[str, object]] | None:
    """Certify a ferromagnetic Ising optimum by an integer-capacity s-t min-cut.

    With ``s_i = 2 x_i - 1`` and every ``J_ij <= 0``, the pair term contributes
    ``-2 J_ij`` when the two binary labels differ.  Signed fields become terminal
    capacities after a constant shift.  Decimal JSON coefficients are lifted to exact
    rationals and then to one common integer scale, so the flow computation itself has no
    floating-point comparison ambiguity.
    """

    h = {
        int(node): _finite_fraction(value, f"problem.h[{node!r}]")
        for node, value in problem.h.items()
    }
    j = {
        (int(u), int(v)): _finite_fraction(value, f"problem.J[{u!r},{v!r}]")
        for (u, v), value in problem.j.items()
    }
    if any(weight > 0 for weight in j.values()):
        return None

    scale = 1
    for value in (*h.values(), *j.values()):
        scale = math.lcm(scale, value.denominator)

    source = ("embedbench", "source")
    sink = ("embedbench", "sink")
    graph = nx.DiGraph()
    graph.add_nodes_from((source, sink, *h))
    for node, field in h.items():
        source_capacity = max(2 * field, Fraction(0))
        sink_capacity = max(-2 * field, Fraction(0))
        if source_capacity:
            graph.add_edge(source, node, capacity=int(source_capacity * scale))
        if sink_capacity:
            graph.add_edge(node, sink, capacity=int(sink_capacity * scale))
    for (u, v), weight in j.items():
        capacity = int((-2 * weight) * scale)
        if capacity:
            graph.add_edge(u, v, capacity=capacity)
            graph.add_edge(v, u, capacity=capacity)

    cut_value, partition = nx.minimum_cut(
        graph,
        source,
        sink,
        capacity="capacity",
        flow_func=nx.algorithms.flow.preflow_push,
    )
    if isinstance(cut_value, bool) or int(cut_value) != cut_value:
        raise RuntimeError("integer min-cut unexpectedly returned a non-integer capacity")
    cut_value_scaled = int(cut_value)
    source_side, sink_side = partition
    source_variables = sorted(node for node in h if node in source_side)
    sink_variables = sorted(node for node in h if node in sink_side)
    if set(source_variables).isdisjoint(sink_variables) is False or set(
        source_variables + sink_variables
    ) != set(h):
        raise RuntimeError("min-cut partition does not cover every logical variable")

    spins = {node: (-1 if node in source_side else 1) for node in h}
    witness_energy = sum(h[node] * spins[node] for node in h) + sum(
        weight * spins[u] * spins[v] for (u, v), weight in j.items()
    )
    exact_energy = (
        Fraction(cut_value_scaled, scale)
        - sum(abs(field) for field in h.values())
        + sum(j.values())
    )
    if witness_energy != exact_energy:
        raise RuntimeError("min-cut witness energy disagrees with the cut reduction")
    witness_payload = json.dumps(
        [[node, spins[node]] for node in sorted(spins)],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    certificate = {
        "algorithm": "networkx_preflow_push_integer_capacities_v1",
        "capacity_scale": scale,
        "cut_value_scaled": cut_value_scaled,
        "energy_numerator": exact_energy.numerator,
        "energy_denominator": exact_energy.denominator,
        "source_variables": source_variables,
        "sink_variables": sink_variables,
        "witness_sha256": hashlib.sha256(witness_payload).hexdigest(),
    }
    return float(exact_energy), certificate


def _exhaustive_ground_energy(problem: LogicalProblem) -> float:
    """Enumerate every logical spin state in bounded NumPy batches."""

    nodes = sorted(problem.h)
    node_index = {node: index for index, node in enumerate(nodes)}
    fields = np.asarray([float(problem.h[node]) for node in nodes], dtype=np.float64)
    couplers = [
        (node_index[u], node_index[v], float(weight)) for (u, v), weight in problem.j.items()
    ]
    if not np.isfinite(fields).all() or any(not math.isfinite(weight) for _, _, weight in couplers):
        raise ValueError("exact ground-state enumeration requires finite coefficients")

    bit_positions = np.arange(len(nodes), dtype=np.uint64)
    state_count = 1 << len(nodes)
    best = math.inf
    for first in range(0, state_count, EXACT_ENUMERATION_BATCH_STATES):
        stop = min(first + EXACT_ENUMERATION_BATCH_STATES, state_count)
        states = np.arange(first, stop, dtype=np.uint64)
        spins = (1 - 2 * ((states[:, None] >> bit_positions) & 1)).astype(
            np.int8,
            copy=False,
        )
        energies = spins @ fields
        for u, v, weight in couplers:
            energies += weight * spins[:, u] * spins[:, v]
        batch_best = float(energies.min())
        if batch_best < best:
            best = batch_best
    return best


def _certified_ground_reference(
    record: Mapping[str, object],
    cache: dict[str, dict[str, object]],
    *,
    exhaustive_max_n: int = EXACT_GROUND_STATE_MAX_N,
) -> dict[str, object]:
    """Certify ``e0`` exactly by exhaustive search or a ferromagnetic min-cut."""

    digest = quality_problem_digest(record)
    if digest in cache:
        return cache[digest]
    raw_problem = record.get("problem")
    if not isinstance(raw_problem, Mapping):
        raise ValueError("paper audit record has no problem payload")
    problem = _strict_legacy_problem(raw_problem)
    reference = float(raw_problem["e0"])
    if len(problem.h) <= exhaustive_max_n:
        exact_energy = _exhaustive_ground_energy(problem)
        method = "exhaustive_enumeration"
        certificate = None
    else:
        min_cut = _ferromagnetic_min_cut_certificate(problem)
        if min_cut is None:
            exact_energy = None
            method = "not_exactly_verifiable_by_registered_auditor"
            certificate = None
        else:
            exact_energy, certificate = min_cut
            method = "ferromagnetic_s_t_min_cut"

    if exact_energy is None:
        result = {
            "status": "uncertified_best_known",
            "method": method,
            "energy": reference,
            "n_variables": len(problem.h),
        }
    elif not math.isclose(exact_energy, reference, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(
            "paper audit problem.e0 disagrees with its registered exact ground certificate"
        )
    else:
        result = {
            "status": "certified_exact",
            "method": method,
            "energy": reference,
            "n_variables": len(problem.h),
        }
        if certificate is not None:
            result["certificate"] = certificate
    cache[digest] = result
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.per_file <= 0:
        parser.error("--per-file must be positive")
    if args.reads < MIN_HIGH_READS_PER_STRENGTH:
        parser.error(
            f"--reads must be at least {MIN_HIGH_READS_PER_STRENGTH} for a high-read audit"
        )
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    contract_requested = args.audit_contract is not None or args.audit_contract_sha256 is not None
    if contract_requested and (args.audit_contract is None or args.audit_contract_sha256 is None):
        parser.error("--audit-contract and --audit-contract-sha256 must be supplied together")
    if args.split == "test" and not contract_requested:
        parser.error("locked test rescoring requires --audit-contract and --audit-contract-sha256")
    paper_contract = None
    if contract_requested:
        from quality_v2_paper_contract import load_paper_audit_contract

        try:
            paper_contract = load_paper_audit_contract(
                args.audit_contract,
                args.audit_contract_sha256,
            )
            if not args.splits or args.split != paper_contract.document["partition"]:
                raise ValueError("paper audit contract is valid only for its fixed split partition")
            if (
                paper_contract.audit_labels["schema"] != AUDIT_SCHEMA
                or paper_contract.audit_labels["schema_version"] != AUDIT_SCHEMA_VERSION
            ):
                raise ValueError("rescore_quality audit schema disagrees with the paper contract")
            paper_contract.validate_audit_provenance(
                {
                    "generator": AUDIT_GENERATOR,
                    "reads_per_strength": args.reads,
                    "num_sweeps": NUM_SWEEPS,
                    "base_seed": args.seed,
                    "registered_strength_count": REGISTERED_STRENGTH_COUNT,
                    "seed_schedule": AUDIT_SEED_SCHEDULE,
                    "strength_schedule": AUDIT_STRENGTH_SCHEDULE,
                    "aggregation": AUDIT_AGGREGATION,
                    "score_evidence": "integer_success_counts_per_candidate_strength",
                    "probability_derivation": (
                        "success_count_divided_by_reads_per_strength_binary64"
                    ),
                },
                location="rescore_quality paper protocol",
            )
        except ValueError as error:
            parser.error(str(error))

    selection_values = (
        args.selection,
        args.selection_sha256,
        args.selection_root,
    )
    if args.split == "test" and any(value is None for value in selection_values):
        parser.error(
            "locked test rescoring requires --selection, --selection-sha256, and --selection-root"
        )
    if args.split != "test" and any(value is not None for value in selection_values):
        parser.error("selection binding is valid only for locked test rescoring")

    selection_document = None
    if args.split == "test":
        from select_training_grid import revalidate_selection_artifact

        try:
            assert paper_contract is not None
            selection_document = revalidate_selection_artifact(
                args.selection,
                args.selection_sha256,
                root=args.selection_root,
                audit_contract=paper_contract,
            )
        except ValueError as error:
            parser.error(str(error))

    preregistration = None
    preregistration_values = (
        args.preregistration_manifest,
        args.preregistration_manifest_sha256,
        args.policy_freezes,
    )
    if args.split == "test" and any(value is None for value in preregistration_values):
        parser.error(
            "locked test rescoring requires --preregistration-manifest, "
            "--preregistration-manifest-sha256, and --policy-freezes"
        )
    if args.split != "test" and any(value is not None for value in preregistration_values):
        parser.error("paper preregistration is valid only for locked test rescoring")
    if args.split == "test":
        try:
            assert paper_contract is not None and selection_document is not None
            preregistration = load_paper_preregistration(
                args.preregistration_manifest,
                args.preregistration_manifest_sha256,
                audit_contract=paper_contract,
                selection_document=selection_document,
                selection_file=Path(args.selection).name,
                selection_sha256=args.selection_sha256,
                audit_source_root=Path(__file__).parents[1],
            )
            verify_preregistered_policy_freezes(preregistration, args.policy_freezes)
        except ValueError as error:
            parser.error(str(error))

    if paper_contract is not None:
        execution = paper_contract.document.get("audit_execution")
        expected_execution = {
            "shard_count": args.shard_count,
            "assignment": "sha256(canonical_audit_identity) mod shard_count",
            "canonical_audit_identity": "json_compact_utf8_array[file,instance_id,focus]",
            "record_output": "atomic_recomputed_one_record_per_audit_identity_no_partial_reuse",
            "merge_coverage": "exactly_once_complete_fixed_test_partition",
            "merge_order": "canonical_audit_identity_lexicographic",
            "release_commitment": "exact_jsonl_and_shard_manifest_bytes",
            "release_trust_anchor": "sha256_registered_out_of_band_before_phase_2",
        }
        if not isinstance(execution, Mapping):
            parser.error("rescore_quality sharding disagrees with the frozen paper contract")
        try:
            require_exact_json(
                execution,
                expected_execution,
                location="rescore_quality audit execution contract",
            )
        except ValueError as error:
            parser.error(str(error))

    paths = tuple(Path(raw_path) for raw_path in args.files)
    try:
        current_provenance = data_provenance(paths, args.splits)
        partitions = load_split_records(
            paths,
            args.splits,
            group_key=quality_problem_id,
            record_group_key=quality_problem_digest,
        )
        split_path = Path(args.splits)
        split_document = strict_json_loads(
            split_path.read_bytes(),
            location=f"split manifest {split_path}",
        )
        split_tables = split_document["splits"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"invalid fixed split/data provenance: {error}")
    if selection_document is not None:
        try:
            require_exact_json(
                selection_document.get("data_provenance"),
                current_provenance,
                location="selection data provenance for audit inputs",
            )
        except ValueError as error:
            parser.error(str(error))

    records_by_path: dict[Path, list[dict]] = {}
    selected_count = 0
    try:
        for path in paths:
            table = split_tables[path.name]
            selected = []
            for record in _load_records(path):
                if table.get(str(record.get("instance_id"))) != args.split:
                    continue
                if len(record.get("candidates", ())) < 2 or "problem" not in record:
                    raise ValueError(
                        f"{path}: selected record lacks complete candidate/problem support"
                    )
                selected.append(record)
            records_by_path[path] = selected
            selected_count += len(selected)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"invalid fixed split record support: {error}")
    expected_partition = getattr(partitions, args.split)
    if selected_count != len(expected_partition):
        parser.error("fixed split record materialization changed validated record support")
    if paper_contract is not None:
        fixed_identities = [
            canonical_audit_identity(path.name, record)
            for path, records in records_by_path.items()
            for record in records
        ]
        if len(fixed_identities) != len(set(fixed_identities)):
            parser.error("fixed test partition repeats a canonical audit identity")
        for path, records in records_by_path.items():
            if len(records) > args.per_file:
                parser.error("paper audit --per-file must cover complete fixed-test record support")
            records_by_path[path] = [
                record
                for record in records
                if audit_shard_index(
                    canonical_audit_identity(path.name, record),
                    args.shard_count,
                )
                == args.shard_index
            ]

    paper_binding = None
    ground_references: dict[str, dict[str, object]] = {}
    if selection_document is not None:
        from quality_v2_paper_contract import (
            AUDIT_SOURCE_HASH_ALGORITHM,
            audit_source_sha256,
            selection_audit_binding,
        )

        assert paper_contract is not None
        actual_audit_source = audit_source_sha256(Path(__file__).parents[1])
        if selection_document.get("audit_source_sha256") != actual_audit_source:
            parser.error("selection audit source does not match the executing auditor")
        paper_binding = {
            "audit_contract": paper_contract.public_binding(),
            "preregistration": preregistration.public_binding(),
            "selection": selection_audit_binding(
                selection_document,
                selection_file=args.selection,
                selection_sha256=args.selection_sha256,
            ),
            "audit_source": {
                "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
                "sha256": actual_audit_source,
            },
            "runtime_provenance": runtime_provenance("cpu"),
        }
        ground_cache: dict[str, dict[str, object]] = {}
        for records in records_by_path.values():
            for record in records:
                ground = _certified_ground_reference(record, ground_cache)
                if ground["status"] != "certified_exact":
                    parser.error(
                        "locked exact-p_solve audit rejects an uncertified ground reference: "
                        f"{record.get('instance_id')}"
                    )
                ground_references[quality_problem_digest(record)] = ground

    rng = random.Random(args.seed)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    paper_payloads: list[tuple[bytes, Path, bytes]] = []
    record_dir = output.with_name(f"{output.name}.records")
    audit_stream = output.open("w", encoding="utf-8") if paper_contract is None else None
    try:
        for path in paths:
            records = records_by_path[path]
            sample = _records_for_audit(
                records,
                per_file=args.per_file,
                low_margin=args.low_margin,
                rng=rng,
                paper_mode=paper_contract is not None,
            )
            host = host_graph(sample[0]["topology"], sample[0]["size"]) if sample else None
            digest = file_sha256(path)
            flips = agreement = 0
            corpus_regrets = []
            resource_regrets = []
            for record in sample:
                if paper_binding is not None:
                    _, strengths, realized_seeds = _audit_schedule(record, args.seed)
                    ground_reference = ground_references[quality_problem_digest(record)]
                    identity = canonical_audit_identity(path.name, record)
                    identity_sha256 = hashlib.sha256(identity).hexdigest()
                    record_path = record_dir / f"{identity_sha256}.json"
                    expected = {
                        "reads": args.reads,
                        "base_seed": args.seed,
                        "corpus_digest": digest,
                        "paper_binding": paper_binding,
                        "strengths": strengths,
                        "realized_seeds": realized_seeds,
                        "ground_reference": ground_reference,
                    }
                    resumed = _load_resumable_paper_row(
                        record_path,
                        path,
                        record,
                        **expected,
                    )
                    if resumed is None:
                        (
                            scores,
                            strength_p_solve,
                            strength_success_counts,
                            observed_strengths,
                            observed_seeds,
                        ) = _rescore_record(
                            record,
                            host,
                            args.reads,
                            args.seed,
                        )
                        if observed_strengths != strengths or observed_seeds != realized_seeds:
                            raise RuntimeError(
                                "paper audit realized schedule changed during scoring"
                            )
                        row, payload = _paper_row_payload(
                            path,
                            record,
                            scores=scores,
                            strength_p_solve=strength_p_solve,
                            strength_success_counts=strength_success_counts,
                            **expected,
                        )
                        _atomic_write(record_path, payload)
                    else:
                        row, payload = resumed
                        scores = [float(value) for value in row["high_read_scores"]]
                    paper_payloads.append((identity, record_path, payload))
                else:
                    (
                        scores,
                        _strength_p_solve,
                        _strength_success_counts,
                        strengths,
                        realized_seeds,
                    ) = _rescore_record(record, host, args.reads, args.seed)
                    row = build_audit_record(
                        path,
                        record,
                        scores=scores,
                        reads=args.reads,
                        base_seed=args.seed,
                        corpus_digest=digest,
                    )
                high_read_best = int(np.argmax(scores))
                if paper_contract is None:
                    flips += high_read_best != record["best_index"]
                    agreement += high_read_best == record["best_index"]
                    corpus_regrets.append(scores[high_read_best] - scores[record["best_index"]])
                resource_regrets.append(scores[high_read_best] - scores[record["resource_index"]])
                if audit_stream is not None:
                    audit_stream.write(json.dumps(row) + "\n")
                    audit_stream.flush()
            count = len(sample)
            summary[str(path)] = {
                "n": count,
                "release_labels_consumed": paper_contract is None,
                "flip_rate": flips / max(1, count) if paper_contract is None else None,
                "top1_agreement": (agreement / max(1, count) if paper_contract is None else None),
                "regret_corpus_best": (float(np.mean(corpus_regrets)) if corpus_regrets else None),
                "regret_resource": (float(np.mean(resource_regrets)) if resource_regrets else None),
            }
            print(path, summary[str(path)], flush=True)
    finally:
        if audit_stream is not None:
            audit_stream.close()

    if paper_contract is not None:
        expected_record_paths = {record_path.resolve() for _, record_path, _ in paper_payloads}
        actual_record_paths = (
            {path.resolve() for path in record_dir.glob("*.json")} if record_dir.exists() else set()
        )
        if actual_record_paths != expected_record_paths:
            unexpected = sorted(str(path) for path in actual_record_paths - expected_record_paths)
            missing = sorted(str(path) for path in expected_record_paths - actual_record_paths)
            raise ValueError(
                "paper audit record cache does not exactly match this shard; "
                f"missing={missing}, unexpected={unexpected}"
            )
        assert paper_binding is not None
        finalize_paper_audit_shard(
            output,
            paper_payloads,
            audit_contract_binding=paper_contract.public_binding(),
            preregistration_binding=paper_binding["preregistration"],
            selection_binding=paper_binding["selection"],
            audit_source_binding=paper_binding["audit_source"],
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            complete_fixed_test_record_count=selected_count,
        )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

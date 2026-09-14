"""Filesystem authentication helpers for the post-freeze strength audit.

The statistical protocol lives in :mod:`isingfold.rl.final_strength_audit`.  This module
owns the narrower trust boundary between six complete-system result directories and that
protocol.  In particular, a self-digested report is never enough: every report file must
also match a caller-supplied SHA-256 value.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from isingfold.rl.complete_system import (
    EmbeddingTask,
    read_complete_system_evidence,
    read_complete_system_outcomes,
    read_complete_system_receipts,
)
from isingfold.rl.complete_system_aggregate import (
    COMPLETE_SYSTEM_EVALUATION_SCHEMA,
    COMPLETE_SYSTEM_EVALUATION_VERSION,
    AuthenticatedCompleteSystemSeedRun,
)
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.experiment_selection import FrozenRLValueSelection
from isingfold.rl.external_pairing import (
    AuthenticatedExternalCompleteRun,
    context_snapshot,
    load_authenticated_external_complete_run,
)
from isingfold.rl.final_strength_audit import ARMS, REGISTERED_TRAINING_SEEDS


class FinalStrengthWorkflowError(RuntimeError):
    """A source path, external pin, or complete-system artifact is invalid."""


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_object(content: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise FinalStrengthWorkflowError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise FinalStrengthWorkflowError(f"{label} contains non-finite number {token}")

    try:
        value = json.loads(content, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalStrengthWorkflowError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise FinalStrengthWorkflowError(f"{label} must contain one JSON object")

    def finite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise FinalStrengthWorkflowError(f"{label} contains a non-finite number")
        if isinstance(item, Mapping):
            for child in item.values():
                finite(child)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                finite(child)

    finite(value)
    return value


def frozen_learned_selection_payload(
    selection: FrozenRLValueSelection,
) -> dict[str, object]:
    """Project the validation freeze into the exact final-audit selection identity."""

    if not isinstance(selection, FrozenRLValueSelection):
        raise TypeError("final-strength learned selection has the wrong type")
    return {
        "schema": "isingfold.final-strength-frozen-learned-selection",
        "schema_version": 1,
        "selection_receipt_sha256": selection.receipt_sha256,
        "selection_record_digest": selection.record_digest,
        "grid_manifest_sha256": selection.grid_manifest_sha256,
        "selected_model_family": selection.model_family,
        "selected_grid_model_family": selection.grid_model_family,
        "selected_method": selection.method,
        "training_seeds": list(selection.training_seeds),
        "source_cell_ids": list(selection.cell_ids),
        "source_checkpoint_payload_digests": list(
            selection.checkpoint_payload_digests
        ),
        "representation_selection_sha256": (
            selection.representation_selection_sha256
        ),
        "representation_selection_record_digest": (
            selection.representation_selection_record_digest
        ),
        "runtime_implementation_registry": dict(
            selection.runtime_implementation_registry
        ),
        "runtime_implementation_digest": selection.runtime_implementation_digest,
        "quality_preflight_receipt_sha256": (
            selection.quality_preflight_receipt_sha256
        ),
        "quality_preflight_record_digest": (
            selection.quality_preflight_record_digest
        ),
        "seed_selection_forbidden": True,
    }


def load_authenticated_learned_complete_run(
    directory: str | Path,
    *,
    expected_report_sha256: str,
    tasks: Sequence[EmbeddingTask],
    context: Context,
) -> AuthenticatedCompleteSystemSeedRun:
    """Load one learned test run through its report and all three raw sidecars."""

    if not _is_digest(expected_report_sha256):
        raise FinalStrengthWorkflowError(
            "learned complete-system report requires a lowercase SHA-256 pin"
        )
    root = Path(directory)
    report_path = root / "report.json"
    content = report_path.read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(observed, expected_report_sha256):
        raise FinalStrengthWorkflowError(
            "learned complete-system report differs from its out-of-band SHA-256 pin"
        )
    report = _strict_json_object(content, label="learned complete-system report")
    if canonical_json_bytes(report) + b"\n" != content:
        raise FinalStrengthWorkflowError(
            "learned complete-system report is not canonical JSON"
        )
    payload = {name: value for name, value in report.items() if name != "record_digest"}
    if report.get("record_digest") != content_digest(payload):
        raise FinalStrengthWorkflowError(
            "learned complete-system report record digest differs"
        )
    if (
        report.get("schema") != COMPLETE_SYSTEM_EVALUATION_SCHEMA
        or report.get("schema_version") != COMPLETE_SYSTEM_EVALUATION_VERSION
        or report.get("partition") != "test"
        or report.get("sealed_test_opened") is not True
    ):
        raise FinalStrengthWorkflowError(
            "learned complete-system report is not the registered sealed test endpoint"
        )
    if (
        report.get("context") != context_snapshot(context)
        or report.get("context_digest") != stable_digest(context_snapshot(context))
    ):
        raise FinalStrengthWorkflowError(
            "learned complete-system report uses another execution context"
        )
    artifacts = report.get("artifacts")
    expected_paths = {
        "complete_receipts": "complete_receipts.jsonl",
        "terminal_evidence": "terminal_evidence.jsonl",
        "outcomes": "outcomes.jsonl",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_paths):
        raise FinalStrengthWorkflowError(
            "learned complete-system report has an incomplete artifact registry"
        )
    entries: dict[str, Mapping[str, object]] = {}
    for name, filename in expected_paths.items():
        entry = artifacts.get(name)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256", "count"}
            or entry.get("path") != filename
            or not _is_digest(entry.get("sha256"))
            or type(entry.get("count")) is not int
            or int(entry["count"]) <= 0
        ):
            raise FinalStrengthWorkflowError(
                f"learned complete-system {name} artifact entry is invalid"
            )
        entries[name] = entry
    receipts = tuple(
        read_complete_system_receipts(
            root / expected_paths["complete_receipts"],
            expected_sha256=str(entries["complete_receipts"]["sha256"]),
        )
    )
    outcomes = tuple(
        read_complete_system_outcomes(
            root / expected_paths["outcomes"],
            receipts=receipts,
            expected_sha256=str(entries["outcomes"]["sha256"]),
        )
    )
    evidence = tuple(
        read_complete_system_evidence(
            root / expected_paths["terminal_evidence"],
            receipts=receipts,
            tasks=tasks,
            context=context,
            expected_sha256=str(entries["terminal_evidence"]["sha256"]),
        )
    )
    counts = {
        "complete_receipts": len(receipts),
        "terminal_evidence": len(evidence),
        "outcomes": len(outcomes),
    }
    if any(int(entries[name]["count"]) != count for name, count in counts.items()):
        raise FinalStrengthWorkflowError(
            "learned complete-system artifact count differs from its report"
        )
    run = AuthenticatedCompleteSystemSeedRun(report, receipts, evidence)
    if not hmac.compare_digest(run.report_file_sha256, expected_report_sha256):
        raise FinalStrengthWorkflowError(
            "learned complete-system typed report identity is noncanonical"
        )
    return run


@dataclass(frozen=True)
class AuthenticatedFinalStrengthSources:
    """Exactly three learned and three tuned-stock runs plus their external pins."""

    learned: tuple[AuthenticatedCompleteSystemSeedRun, ...]
    stock: tuple[AuthenticatedExternalCompleteRun, ...]
    report_sha256_pins: Mapping[str, str]

    def __post_init__(self) -> None:
        learned = tuple(sorted(self.learned, key=lambda run: int(run.report["training_seed"])))
        stock = tuple(sorted(self.stock, key=lambda run: int(run.report["training_seed"])))
        learned_seeds = tuple(int(run.report["training_seed"]) for run in learned)
        stock_seeds = tuple(int(run.report["training_seed"]) for run in stock)
        if learned_seeds != REGISTERED_TRAINING_SEEDS or stock_seeds != REGISTERED_TRAINING_SEEDS:
            raise FinalStrengthWorkflowError(
                "final-strength sources must cover each registered training seed exactly once"
            )
        pins = dict(self.report_sha256_pins)
        expected = {
            f"{arm}:{seed}" for arm in ARMS for seed in REGISTERED_TRAINING_SEEDS
        }
        if set(pins) != expected or any(not _is_digest(value) for value in pins.values()):
            raise FinalStrengthWorkflowError(
                "final-strength sources require exactly six report SHA-256 pins"
            )
        object.__setattr__(self, "learned", learned)
        object.__setattr__(self, "stock", stock)
        object.__setattr__(self, "report_sha256_pins", pins)


def authenticate_final_strength_sources(
    *,
    learned_directories: Sequence[str | Path],
    learned_report_sha256s: Sequence[str],
    stock_directories: Sequence[str | Path],
    stock_report_sha256s: Sequence[str],
    tasks: Sequence[EmbeddingTask],
    context: Context,
) -> AuthenticatedFinalStrengthSources:
    """Authenticate the complete six-run census used by every execution shard."""

    arguments = (
        tuple(learned_directories),
        tuple(learned_report_sha256s),
        tuple(stock_directories),
        tuple(stock_report_sha256s),
    )
    if any(len(values) != 3 for values in arguments):
        raise FinalStrengthWorkflowError(
            "final-strength execution requires three learned roots and pins and three "
            "tuned-stock roots and pins"
        )
    learned = tuple(
        load_authenticated_learned_complete_run(
            root,
            expected_report_sha256=pin,
            tasks=tasks,
            context=context,
        )
        for root, pin in zip(arguments[0], arguments[1], strict=True)
    )
    stock = tuple(
        load_authenticated_external_complete_run(
            root,
            expected_report_sha256=pin,
            tasks=tasks,
            context=context,
        )
        for root, pin in zip(arguments[2], arguments[3], strict=True)
    )
    pins = {
        **{
            f"learned:{int(run.report['training_seed'])}": pin
            for run, pin in zip(learned, arguments[1], strict=True)
        },
        **{
            f"tuned-stock:{int(run.report['training_seed'])}": pin
            for run, pin in zip(stock, arguments[3], strict=True)
        },
    }
    return AuthenticatedFinalStrengthSources(learned, stock, pins)


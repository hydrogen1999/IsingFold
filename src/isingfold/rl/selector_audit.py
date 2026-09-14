"""Fail-closed post-freeze audit for the independent IF-Q3-S0 selector.

The audit never fits or calibrates a model.  It evaluates one complete authenticated audit
partition and publishes every empirical count, decision and diagnostic needed to recompute
the aggregate.  The four-count maximum is an optimistic empirical oracle diagnostic, not a
deployable policy and not an estimate corrected for winner's noise.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.strength import N_STRENGTHS, StrengthRecord

AUDIT_SCHEMA = "isingfold.strength-selector-audit"
AUDIT_ROW_SCHEMA = "isingfold.strength-selector-audit-row"
AUDIT_SCHEMA_VERSION = 2
FIXED_F2_INDEX = 1
AUDIT_PARTITIONS = frozenset({"audit_val", "audit_test"})


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _finite_probabilities(values: object) -> tuple[float, ...]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.shape != (N_STRENGTHS,) or not np.isfinite(array).all():
        raise ValueError("selector must return four finite probabilities")
    converted = tuple(float(value) for value in array)
    if any(value < 0.0 or value > 1.0 for value in converted):
        raise ValueError("selector probabilities must be in [0, 1]")
    return converted


def audit_selector_records(
    model: object,
    records: Sequence[StrengthRecord],
    *,
    partition: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Evaluate every row in one audit partition without any parameter update."""

    if partition not in AUDIT_PARTITIONS:
        raise ValueError("selector audit partition must be audit_val or audit_test")
    if not bool(getattr(model, "frozen", False)) or not bool(
        getattr(model, "deployment_ready", False)
    ):
        raise ValueError("selector audit requires a frozen deployment-ready graph selector")
    if not records:
        raise ValueError("selector audit requires a non-empty authenticated partition")

    output: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    seen_source_records: set[str] = set()
    lineages: set[str] = set()
    with torch.no_grad():
        for record in records:
            if record.selector_partition != partition:
                raise ValueError("selector audit record belongs to a different partition")
            if record.graph_inputs is None:
                raise ValueError("selector audit requires graph-complete records")
            if not record.task_id or not record.source_record_id or not record.lineage:
                raise ValueError("selector audit records require authenticated source identities")
            if record.task_id in seen_tasks:
                raise ValueError(f"duplicate task in selector audit: {record.task_id}")
            if record.source_record_id in seen_source_records:
                raise ValueError(
                    f"duplicate source record in selector audit: {record.source_record_id}"
                )
            seen_tasks.add(record.task_id)
            seen_source_records.add(record.source_record_id)
            lineages.add(record.lineage)

            probabilities = _finite_probabilities(model(record.graph_inputs))  # type: ignore[operator]
            # NumPy argmax implements the registered lowest-index tie rule.  Evaluate the
            # network only once so this audit has a fixed and directly countable inference cost.
            selected = int(np.argmax(np.asarray(probabilities)))

            rates = tuple(
                hit / read for hit, read in zip(record.hits, record.reads, strict=True)
            )
            oracle_rate = max(rates)
            oracle_indices = tuple(
                index for index, rate in enumerate(rates) if rate == oracle_rate
            )
            selected_rate = rates[selected]
            fixed_rate = rates[FIXED_F2_INDEX]
            random_expected = float(np.mean(rates))
            row = _with_digest(
                {
                    "schema": AUDIT_ROW_SCHEMA,
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "partition": partition,
                    "task_id": record.task_id,
                    "lineage": record.lineage,
                    "source_selector_record_id": record.source_record_id,
                    "hits": list(record.hits),
                    "reads": list(record.reads),
                    "empirical_rates": list(rates),
                    "selector_probabilities": list(probabilities),
                    "selected_strength_index": selected,
                    "selected_strength": float(record.graph_inputs[selected].strength),
                    "selected_rate": selected_rate,
                    "fixed_f2_strength_index": FIXED_F2_INDEX,
                    "fixed_f2_strength": float(
                        record.graph_inputs[FIXED_F2_INDEX].strength
                    ),
                    "fixed_f2_rate": fixed_rate,
                    "fixed_f2_top1": FIXED_F2_INDEX in oracle_indices,
                    "uniform_random_expected_rate": random_expected,
                    "uniform_random_expected_top1_probability": (
                        len(oracle_indices) / N_STRENGTHS
                    ),
                    "empirical_oracle_indices": list(oracle_indices),
                    "empirical_oracle_rate": oracle_rate,
                    "selected_regret": oracle_rate - selected_rate,
                    "fixed_f2_regret": oracle_rate - fixed_rate,
                    "uniform_random_expected_regret": oracle_rate - random_expected,
                    "top1": selected in oracle_indices,
                }
            )
            output.append(row)

    # Canonical ordering makes the raw artifact reproducible across loader implementations.
    output.sort(key=lambda row: (str(row["task_id"]), str(row["source_selector_record_id"])))

    def summarize(grouped_rows: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, float]:
        selected_rates = np.asarray(
            [np.mean([float(row["selected_rate"]) for row in group]) for group in grouped_rows]
        )
        fixed_rates = np.asarray(
            [np.mean([float(row["fixed_f2_rate"]) for row in group]) for group in grouped_rows]
        )
        random_rates = np.asarray(
            [
                np.mean([float(row["uniform_random_expected_rate"]) for row in group])
                for group in grouped_rows
            ]
        )
        oracle_rates = np.asarray(
            [
                np.mean([float(row["empirical_oracle_rate"]) for row in group])
                for group in grouped_rows
            ]
        )
        selected_top1_rates = np.asarray(
            [np.mean([bool(row["top1"]) for row in group]) for group in grouped_rows]
        )
        fixed_top1_rates = np.asarray(
            [
                np.mean([bool(row["fixed_f2_top1"]) for row in group])
                for group in grouped_rows
            ]
        )
        random_top1_rates = np.asarray(
            [
                np.mean(
                    [
                        float(row["uniform_random_expected_top1_probability"])
                        for row in group
                    ]
                )
                for group in grouped_rows
            ]
        )
        return {
            "selected_mean": float(selected_rates.mean()),
            "fixed_f2_mean": float(fixed_rates.mean()),
            "uniform_random_expected_mean": float(random_rates.mean()),
            "empirical_oracle_mean": float(oracle_rates.mean()),
            "selected_minus_fixed_f2_mean": float((selected_rates - fixed_rates).mean()),
            "selected_minus_uniform_random_expected_mean": float(
                (selected_rates - random_rates).mean()
            ),
            "selected_regret_mean": float((oracle_rates - selected_rates).mean()),
            "fixed_f2_regret_mean": float((oracle_rates - fixed_rates).mean()),
            "uniform_random_expected_regret_mean": float(
                (oracle_rates - random_rates).mean()
            ),
            "top1_rate": float(selected_top1_rates.mean()),
            "selected_top1_rate": float(selected_top1_rates.mean()),
            "fixed_f2_top1_rate": float(fixed_top1_rates.mean()),
            "uniform_random_expected_top1_rate": float(random_top1_rates.mean()),
        }

    record_summary = summarize(tuple((row,) for row in output))
    by_lineage: dict[str, list[Mapping[str, Any]]] = {
        lineage: [] for lineage in sorted(lineages)
    }
    for row in output:
        by_lineage[str(row["lineage"])].append(row)
    lineage_summary = summarize(tuple(tuple(group) for group in by_lineage.values()))
    aggregate = {
        "records": len(output),
        "lineages": len(lineages),
        "record_weighted": record_summary,
        "equal_lineage_weighted": lineage_summary,
        # Flat fields retain a compact record-level summary while the explicit independent-unit
        # aggregate above is the appropriate basis for inferential analysis.
        **record_summary,
        "top1_count": sum(bool(row["top1"]) for row in output),
        "fixed_f2_top1_count": sum(bool(row["fixed_f2_top1"]) for row in output),
    }
    summaries = (*record_summary.values(), *lineage_summary.values())
    if not all(math.isfinite(value) for value in summaries):
        raise RuntimeError("selector audit produced a non-finite aggregate")
    return tuple(output), aggregate


def _write_sync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_selector_audit(
    destination: str | os.PathLike[str],
    *,
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    provenance: Mapping[str, Any],
    access_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish raw rows and their authenticated aggregate receipt."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"selector-audit output already exists: {target}")
    if not rows:
        raise ValueError("selector audit cannot publish an empty raw artifact")
    partitions = {row.get("partition") for row in rows}
    if len(partitions) != 1 or next(iter(partitions)) not in AUDIT_PARTITIONS:
        raise ValueError("raw selector-audit rows must share one registered audit partition")
    partition = next(iter(partitions))
    task_ids: set[object] = set()
    source_ids: set[object] = set()
    for row in rows:
        if (
            row.get("schema") != AUDIT_ROW_SCHEMA
            or row.get("schema_version") != AUDIT_SCHEMA_VERSION
        ):
            raise ValueError("raw selector-audit row has an unsupported schema")
        digest = row.get("record_digest")
        payload = {key: value for key, value in row.items() if key != "record_digest"}
        if digest != content_digest(payload):
            raise ValueError("raw selector-audit row digest mismatch")
        task_id = row.get("task_id")
        source_id = row.get("source_selector_record_id")
        lineage = row.get("lineage")
        if any(
            not isinstance(value, str) or not value
            for value in (task_id, source_id, lineage)
        ):
            raise ValueError("raw selector-audit row has invalid source identities")
        if task_id in task_ids or source_id in source_ids:
            raise ValueError("raw selector-audit rows contain duplicate source identities")
        task_ids.add(task_id)
        source_ids.add(source_id)
    if aggregate.get("records") != len(rows) or aggregate.get("lineages") != len(
        {row.get("lineage") for row in rows}
    ):
        raise ValueError("selector-audit aggregate/raw census differs")

    required = access_control.get("rl_value_selection_required")
    if required is not (partition == "audit_test"):
        raise ValueError("selector-audit access control disagrees with its partition")
    protected_fields = (
        "grid_manifest_sha256",
        "rl_value_selection_receipt_sha256",
        "rl_value_selection_record_digest",
        "representation_selection_receipt_sha256",
        "representation_selection_record_digest",
    )
    if partition == "audit_test":
        values = tuple(access_control.get(name) for name in protected_fields)
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ) or access_control.get("selected_model_family") not in {
            "if-mlp",
            "if-dual",
            "if-core",
        } or access_control.get("selected_method") not in {
            "supervised-only",
            "ppo-warm-start",
            "ppo-from-scratch",
        }:
            raise ValueError("audit_test has no authenticated final RL-value freeze")
    elif any(
        access_control.get(name) is not None
        for name in (*protected_fields, "selected_model_family", "selected_method")
    ):
        raise ValueError("audit_val must not claim a final-selection unsealing receipt")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.selector-audit-", dir=target.parent)
    )
    try:
        raw = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
        _write_sync(temporary / "rows.jsonl", raw)
        receipt = _with_digest(
            {
                "schema": AUDIT_SCHEMA,
                "schema_version": AUDIT_SCHEMA_VERSION,
                "scope": "post-freeze-diagnostic-only-no-training-or-calibration",
                "partition": partition,
                "random_control": "exact-uniform-expectation-over-four-strengths",
                "oracle_control": (
                    "maximum-empirical-rate-from-the-same-four-audit-count-blocks;"
                    "diagnostic-only"
                ),
                "fixed_control": "second-registered-strength-index-1",
                "aggregation_contract": {
                    "record_weighted": "descriptive-over-every-program-record",
                    "equal_lineage_weighted": (
                        "average-records-within-lineage-then-average-independent-lineages"
                    ),
                },
                "raw_rows": {
                    "path": "rows.jsonl",
                    "schema": AUDIT_ROW_SCHEMA,
                    "count": len(rows),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "aggregate": dict(aggregate),
                "provenance": dict(provenance),
                "access_control": dict(access_control),
            }
        )
        _write_sync(
            temporary / "receipt.json", canonical_json_bytes(receipt) + b"\n"
        )
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return receipt


__all__ = [
    "AUDIT_PARTITIONS",
    "AUDIT_ROW_SCHEMA",
    "AUDIT_SCHEMA",
    "AUDIT_SCHEMA_VERSION",
    "audit_selector_records",
    "publish_selector_audit",
]

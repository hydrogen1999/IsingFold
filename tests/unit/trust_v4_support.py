from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from tests.unit.quality_attestation_support import FIXTURE_CERTIFICATE_DIGEST
from tests.unit.test_rl_import_embedbench import _write_design_inputs


def _record(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _write_record(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    record = _record(payload)
    raw = canonical_json_bytes(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return record["record_digest"], hashlib.sha256(raw).hexdigest()


def write_v4_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path, str]:
    """Upgrade the compact production-design fixture to evidence-backed design v2."""

    bank, bank_manifest, targets, provenance, design, _ = _write_design_inputs(root)
    target_rows = [json.loads(line) for line in targets.read_text().splitlines()]
    for row in target_rows:
        row["certificate_digest"] = FIXTURE_CERTIFICATE_DIGEST
    targets.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in target_rows)
    )

    bank_rows = [json.loads(line) for line in bank.read_text().splitlines()]
    instances = {
        row["record"]["instance_id"]: row["record"]
        for row in bank_rows
        if row["kind"] == "instance"
    }
    groups = {
        row["record"]["group_id"]: row["record"]
        for row in bank_rows
        if row["kind"] == "group"
    }
    provenance_rows = [json.loads(line) for line in provenance.read_text().splitlines()]
    design_record = json.loads(design.read_text())
    lineages = design_record["lineage_registry"]

    strata = root / "strata"
    protocol_payload = {
        "outcome_blind": True,
        "rules": [
            {
                "field": field,
                "hard_if": "greater-than-or-equal",
                "metric": metric,
                "threshold": 0.5,
            }
            for field, metric in (
                ("decision_difficulty", "decision_score"),
                ("embedding_difficulty", "embedding_score"),
                ("sampling_difficulty", "sampling_score"),
            )
        ],
        "schema": "embedbench.difficulty-protocol",
        "schema_version": 1,
    }
    protocol_digest, protocol_sha = _write_record(strata / "protocol.json", protocol_payload)
    budget_payload = {
        "limits": {
            "decision_evaluations": 8,
            "embedding_attempts": 8,
            "sampler_reads": 8,
        },
        "measurement_budget_id": "fixture-budget-v1",
        "schema": "embedbench.difficulty-budget",
        "schema_version": 1,
    }
    budget_digest, budget_sha = _write_record(strata / "budget.json", budget_payload)
    base_lineages = sorted({row["base_lineage_key"] for row in lineages})
    panel_payload = {
        "base_lineages": base_lineages,
        "sampling_frame": "all-registered-lineages",
        "selection_outcome_blind": True,
        "schema": "embedbench.difficulty-panel",
        "schema_version": 1,
    }
    panel_digest, panel_sha = _write_record(strata / "panel.json", panel_payload)

    labels_by_lineage = {row["base_lineage_key"]: row for row in lineages}
    measurement_rows = []
    for base in base_lineages:
        labels = labels_by_lineage[base]
        row_payload = {
            "base_lineage_key": base,
            "metrics": [
                ["decision_score", 1.0 if labels["decision_difficulty"] == "hard" else 0.0],
                ["embedding_score", 1.0 if labels["embedding_difficulty"] == "hard" else 0.0],
                ["sampling_score", 1.0 if labels["sampling_difficulty"] == "hard" else 0.0],
            ],
        }
        measurement_rows.append(_record(row_payload))
    evidence_payload = {
        "budget_record_digest": budget_digest,
        "measurements": measurement_rows,
        "panel_record_digest": panel_digest,
        "protocol_record_digest": protocol_digest,
        "schema": "embedbench.difficulty-evidence",
        "schema_version": 1,
    }
    evidence_digest, evidence_sha = _write_record(strata / "evidence.json", evidence_payload)

    origin_rows: dict[str, dict[str, Any]] = {}
    for provenance_row in provenance_rows:
        group = groups[provenance_row["group_id"]]
        instance = instances[group["instance_id"]]
        base = provenance_row["base_parent_lineage"]
        label = labels_by_lineage[base]
        entry = origin_rows.setdefault(
            base,
            {
                "application_family": instance["family"],
                "base_lineage_key": base,
                "generator": {
                    "generator_id": f"fixture/{instance['family']}",
                    "implementation_sha256": "9" * 64,
                    "kind": (
                        "application"
                        if label["problem_origin"] == "application-derived"
                        else "synthetic"
                    ),
                },
                "source_instance_record_digests": [],
            },
        )
        entry["source_instance_record_digests"].append(instance["record_digest"])
    origin_records = []
    for base in sorted(origin_rows):
        row = origin_rows[base]
        row["source_instance_record_digests"] = sorted(
            set(row["source_instance_record_digests"])
        )
        origin_records.append(_record(row))
    origin_payload = {
        "publisher_id": "embedbench-fixture-publisher",
        "records": origin_records,
        "schema": "embedbench.origin-provenance",
        "schema_version": 1,
        "source_release_id": "embedbench-fixture-v2",
        "source_release_manifest_sha256": "7" * 64,
    }
    origin_digest, origin_sha = _write_record(strata / "origin.json", origin_payload)

    authority_payload = {
        "artifacts": {
            "budget_record_digest": budget_digest,
            "budget_sha256": budget_sha,
            "evidence_record_digest": evidence_digest,
            "evidence_sha256": evidence_sha,
            "origin_record_digest": origin_digest,
            "origin_sha256": origin_sha,
            "panel_record_digest": panel_digest,
            "panel_sha256": panel_sha,
            "protocol_record_digest": protocol_digest,
            "protocol_sha256": protocol_sha,
        },
        "publisher_id": "embedbench-fixture-publisher",
        "schema": "embedbench.stratum-publisher-authority",
        "schema_version": 1,
        "source_release_id": "embedbench-fixture-v2",
        "source_release_manifest_sha256": "7" * 64,
        "statement": "publisher-attests-origin-and-outcome-blind-difficulty-evidence",
    }
    _, authority_sha = _write_record(strata / "authority.json", authority_payload)

    def descriptor(name: str, digest: str) -> dict[str, str]:
        return {"path": f"strata/{name}.json", "sha256": digest}
    design_payload = {
        key: value
        for key, value in design_record.items()
        if key not in {"record_digest", "difficulty_calibration"}
    }
    design_payload["corpus_design_version"] = "if-core-v2"
    design_payload["schema_version"] = 2
    design_payload["difficulty_calibration"] = {
        "authority": descriptor("authority", authority_sha),
        "budget": descriptor("budget", budget_sha),
        "evidence": descriptor("evidence", evidence_sha),
        "origin": descriptor("origin", origin_sha),
        "outcome_blind": True,
        "panel": descriptor("panel", panel_sha),
        "protocol": descriptor("protocol", protocol_sha),
        "publisher_id": "embedbench-fixture-publisher",
    }
    design_record = _record(design_payload)
    design.write_bytes(canonical_json_bytes(design_record) + b"\n")
    return (
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        hashlib.sha256(design.read_bytes()).hexdigest(),
    )


__all__ = ["write_v4_inputs"]

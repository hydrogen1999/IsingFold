from __future__ import annotations

from collections.abc import Sequence

from isingfold.rl.contracts import stable_digest
from isingfold.rl.evaluation_strata import (
    ConfirmatoryEvaluationDesign,
    EvaluationStratum,
)


def evaluation_contract(
    identities: Sequence[tuple[str, str]], *, partition: str = "test"
) -> tuple[tuple[EvaluationStratum, ...], ConfirmatoryEvaluationDesign]:
    strata = tuple(
        EvaluationStratum(
            lineage=lineage,
            instance=instance,
            learning_partition=partition,
            application_family="frustrated-loop",
            problem_origin="synthetic",
            host_family="path-a",
            fault_status="none",
            distribution_regime="ood",
            calibration_status="not_applicable",
            calibration_sha256=None,
            embedding_difficulty="easy",
            sampling_difficulty="hard",
            decision_difficulty="hard",
            nominal_size=index + 2,
            source_registry_row_digest=stable_digest(
                {"registry": [lineage, instance]}
            ),
            source_provenance_record_digests=(
                stable_digest({"provenance": [lineage, instance]}),
            ),
        )
        for index, (lineage, instance) in enumerate(sorted(identities))
    )
    target = {
        "target_id": f"{partition}-valid-return-noninferiority",
        "endpoint": "valid-return-noninferiority",
        "alternative": "one-sided-noninferiority",
        "alpha": 0.05,
        "target_power": 0.51,
        "assumed_discordance": 0.01,
        "assumed_true_difference": 0.47,
        "noninferiority_margin": 0.02,
        "power_separation": 0.49,
        "filter": {
            "application_family": ["frustrated-loop"],
            "problem_origin": ["synthetic"],
            "host_family": ["path-a"],
            "fault_status": ["none"],
            "distribution_regime": ["ood"],
            "calibration_status": ["not_applicable"],
            "embedding_difficulty": ["easy"],
            "sampling_difficulty": ["hard"],
            "decision_difficulty": ["hard"],
        },
        "learning_partition": partition,
        "method": "paired-binary-normal-approximation",
        "minimum_base_lineages": 1,
    }
    design = ConfirmatoryEvaluationDesign.from_prepared_receipt(
        {
            "manifest_sha256": stable_digest({"design": "file"}),
            "manifest_record_digest": stable_digest({"design": "record"}),
            "power_targets": [target],
            "precision_targets": [
                {
                    "target_id": f"{partition}-paired-utility-precision",
                    "endpoint": "learned-minus-stock-unconditional-if-q3-s0",
                    "learning_partition": partition,
                    "filter": target["filter"],
                    "method": "bounded-paired-difference-worst-case-normal",
                    "confidence_level": 0.51,
                    "half_width": 0.99,
                    "minimum_base_lineages": 1,
                    "outcome_bounds": [-1.0, 1.0],
                    "variance_bound": 1.0,
                }
            ],
        },
        partition=partition,
        noninferiority_margin=0.02,
    )
    return strata, design


__all__ = ["evaluation_contract"]

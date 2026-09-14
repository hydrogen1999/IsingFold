from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from isingfold.rl.contracts import stable_digest
from isingfold.rl.evaluation_strata import (
    SIZE_BIN_BOUNDARY_CONVENTION,
    SIZE_BIN_PROTOCOL,
    ConfirmatoryEvaluationDesign,
    EvaluationStratum,
    PowerTarget,
    PrecisionTarget,
    evaluation_contract_from_prepared,
    nominal_size_bin,
)


def _filter() -> dict[str, list[str]]:
    return {
        "application_family": ["portfolio"],
        "problem_origin": ["application-derived"],
        "host_family": ["pegasus"],
        "fault_status": ["faulted"],
        "distribution_regime": ["ood"],
        "calibration_status": ["recorded"],
        "embedding_difficulty": ["hard"],
        "sampling_difficulty": ["medium"],
        "decision_difficulty": ["hard"],
    }


def _target(*, target_id: str = "test-feasibility", alpha: float = 0.05) -> dict[str, object]:
    return {
        "target_id": target_id,
        "endpoint": "valid-return-noninferiority",
        "alternative": "one-sided-noninferiority",
        "alpha": alpha,
        "target_power": 0.51,
        "assumed_discordance": 0.01,
        "assumed_true_difference": 0.47,
        "noninferiority_margin": 0.02,
        "power_separation": 0.49,
        "filter": _filter(),
        "learning_partition": "test",
        "method": "paired-binary-normal-approximation",
        "minimum_base_lineages": 1,
    }


def _design_receipt(targets: list[dict[str, object]]) -> dict[str, object]:
    return {
        "manifest_sha256": stable_digest({"source": "design-file"}),
        "manifest_record_digest": stable_digest({"source": "design-record"}),
        "power_targets": targets,
        "precision_targets": [_precision_target()],
    }


def _precision_target() -> dict[str, object]:
    return {
        "target_id": "test-learned-minus-stock-if-q3-s0-precision",
        "endpoint": "learned-minus-stock-unconditional-if-q3-s0",
        "learning_partition": "test",
        "filter": _filter(),
        "method": "bounded-paired-difference-worst-case-normal",
        "confidence_level": 0.51,
        "half_width": 0.99,
        "minimum_base_lineages": 1,
        "outcome_bounds": [-1.0, 1.0],
        "variance_bound": 1.0,
    }


def _stratum() -> EvaluationStratum:
    return EvaluationStratum(
        lineage="base-lineage-001",
        instance="policy-instance-007",
        learning_partition="test",
        application_family="portfolio",
        problem_origin="application-derived",
        host_family="pegasus",
        fault_status="faulted",
        distribution_regime="ood",
        calibration_status="recorded",
        calibration_sha256="a" * 64,
        embedding_difficulty="hard",
        sampling_difficulty="medium",
        decision_difficulty="hard",
        nominal_size=17,
        source_registry_row_digest="b" * 64,
        source_provenance_record_digests=("c" * 64,),
    )


def test_evaluation_stratum_round_trip_binds_derived_size_bin_and_digest() -> None:
    stratum = _stratum()

    assert nominal_size_bin(17) == "000016-000031"
    assert stratum.size_bin == "000016-000031"
    assert stratum.size_bin_protocol == SIZE_BIN_PROTOCOL
    assert stratum.size_bin_boundary_convention == SIZE_BIN_BOUNDARY_CONVENTION
    assert EvaluationStratum.from_mapping(stratum.as_dict()) == stratum

    tampered = copy.deepcopy(stratum.as_dict())
    tampered["size_bin"] = "000032-000063"
    with pytest.raises(ValueError, match="size bin"):
        EvaluationStratum.from_mapping(tampered)

    tampered = copy.deepcopy(stratum.as_dict())
    tampered["application_family"] = "graph-cut"
    with pytest.raises(ValueError, match="record digest"):
        EvaluationStratum.from_mapping(tampered)


def test_power_target_round_trip_preserves_exact_registered_row() -> None:
    source = _target()
    target = PowerTarget.from_mapping(source)

    assert target.as_dict() == source
    assert target.record_digest == stable_digest(source)
    assert target.design_filter.matches(_stratum())

    precision_source = _precision_target()
    precision = PrecisionTarget.from_mapping(precision_source)
    assert precision.as_dict() == precision_source
    assert precision.record_digest == stable_digest(precision_source)


def test_confirmatory_design_rejects_underpowered_or_alpha_inflated_registry() -> None:
    underpowered = _target()
    underpowered["assumed_true_difference"] = 0.0
    underpowered["power_separation"] = 0.02
    with pytest.raises(ValueError, match="registered power calculation"):
        ConfirmatoryEvaluationDesign.from_prepared_receipt(
            _design_receipt([underpowered]),
            partition="test",
            noninferiority_margin=0.02,
        )

    first = _target(target_id="a", alpha=0.03)
    second = _target(target_id="b", alpha=0.03)
    with pytest.raises(ValueError, match="alpha spending"):
        ConfirmatoryEvaluationDesign.from_prepared_receipt(
            _design_receipt([first, second]),
            partition="test",
            noninferiority_margin=0.02,
        )

    underprecise = _design_receipt([_target()])
    underprecise["precision_targets"][0]["half_width"] = 0.05
    with pytest.raises(ValueError, match="registered precision calculation"):
        ConfirmatoryEvaluationDesign.from_prepared_receipt(
            underprecise,
            partition="test",
            noninferiority_margin=0.02,
        )


def test_confirmatory_design_round_trip_binds_source_and_exact_target() -> None:
    receipt = _design_receipt([_target()])
    design = ConfirmatoryEvaluationDesign.from_prepared_receipt(
        receipt, partition="test", noninferiority_margin=0.02
    )

    assert design.power_targets[0].power_separation == 0.49
    assert design.noninferiority_margin == 0.02
    assert design.power_targets[0].alpha == 0.05
    assert design.power_targets[0].endpoint == "valid-return-noninferiority"
    assert design.source_power_targets_digest == stable_digest(receipt["power_targets"])
    assert design.source_precision_targets_digest == stable_digest(
        receipt["precision_targets"]
    )
    assert design.precision_targets[0].endpoint == (
        "learned-minus-stock-unconditional-if-q3-s0"
    )
    assert ConfirmatoryEvaluationDesign.from_mapping(design.as_dict()) == design

    tampered = copy.deepcopy(design.as_dict())
    tampered["power_targets"][0]["target_id"] = "tampered-feasibility"
    with pytest.raises(ValueError, match="record digest"):
        ConfirmatoryEvaluationDesign.from_mapping(tampered)


def test_prepared_projection_collapses_candidate_rows_without_losing_provenance() -> None:
    condition = SimpleNamespace(
        base_lineage_key="base-lineage-001",
        learning_partition="test",
        application_family="portfolio",
        problem_origin="application-derived",
        host_family="pegasus",
        fault_status="faulted",
        distribution_regime="ood",
        calibration_status="recorded",
        calibration_sha256="a" * 64,
        embedding_difficulty="hard",
        sampling_difficulty="medium",
        decision_difficulty="hard",
        registry_row_digest="b" * 64,
    )
    rows = [
        SimpleNamespace(
            partition="test",
            task=SimpleNamespace(lineage="base-lineage-001"),
            instance_id="policy-instance-007",
            design_condition=condition,
            provenance=SimpleNamespace(
                active_topology="pegasus",
                fault_status="faulted",
                distribution_regime="ood",
                calibration_status="recorded",
                calibration_sha256="a" * 64,
                nominal_size=17,
                record_digest=digest,
            ),
        )
        for digest in ("c" * 64, "d" * 64)
    ]

    strata, design = evaluation_contract_from_prepared(
        rows,
        corpus_design_receipt=_design_receipt([_target()]),
        partition="test",
        noninferiority_margin=0.02,
    )

    assert len(strata) == 1
    assert strata[0].source_provenance_record_digests == ("c" * 64, "d" * 64)
    assert strata[0].problem_origin == "application-derived"
    assert design.power_targets[0].noninferiority_margin == 0.02

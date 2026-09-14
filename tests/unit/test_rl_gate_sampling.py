from __future__ import annotations

from types import SimpleNamespace

from isingfold.rl.cli import _stratified_gate_sample


def _prepared(
    lineage: str,
    task_id: str,
    host_family: str,
    *,
    fault_status: str = "none",
    problem_origin: str = "synthetic",
    difficulty: str = "hard",
) -> SimpleNamespace:
    task = SimpleNamespace(lineage=lineage)
    condition = {
        "application_family": (
            "portfolio" if problem_origin == "application-derived" else "frustrated-loop"
        ),
        "problem_origin": problem_origin,
        "calibration_status": "not_applicable",
        "decision_difficulty": difficulty,
        "distribution_regime": "iid",
        "embedding_difficulty": difficulty,
        "fault_status": fault_status,
        "host_family": host_family,
        "learning_partition": "val",
        "sampling_difficulty": difficulty,
        # These are condition identities, not categorical stratum coordinates.
        "base_lineage_key": lineage,
        "calibration_sha256": None,
        "registry_row_digest": ("a" if lineage == "L1" else "b") * 64,
    }
    return SimpleNamespace(task=task, task_id=task_id, design_condition=condition)


def test_scalable_gate_sampler_balances_real_strata_before_reusing_a_lineage() -> None:
    # Under the old lineage-first implementation, the stable within-lineage hashes select
    # both ``host-a`` rows for seed 7.  A genuine stratum-first round robin must instead use
    # both hosts while retaining one independent row per base lineage.
    rows = [
        _prepared("L1", "L1-a-1", "host-a"),
        _prepared("L1", "L1-b-1", "host-b"),
        _prepared("L2", "L2-a-0", "host-a"),
        _prepared("L2", "L2-b-0", "host-b"),
    ]

    selected, receipt = _stratified_gate_sample(rows, count=2, seed=7)
    reversed_selected, reversed_receipt = _stratified_gate_sample(
        list(reversed(rows)), count=2, seed=7
    )

    assert [item.task_id for item in selected] == [item.task_id for item in reversed_selected]
    assert receipt == reversed_receipt
    assert len({item.task.lineage for item in selected}) == 2
    assert {item.design_condition["host_family"] for item in selected} == {
        "host-a",
        "host-b",
    }


def test_scalable_gate_sampler_balances_marginal_axes_not_only_joint_strata() -> None:
    # A seeded ordering of distinct joint strata can still take two rows from one host or one
    # difficulty level.  The gate sample instead maximises authenticated marginal coverage.
    rows = [
        _prepared("L1", "t1", "host-a", difficulty="easy"),
        _prepared(
            "L2",
            "t2",
            "host-b",
            fault_status="faulted",
            problem_origin="application-derived",
            difficulty="hard",
        ),
        _prepared("L3", "t3", "host-a", difficulty="hard"),
        _prepared(
            "L4",
            "t4",
            "host-b",
            fault_status="faulted",
            problem_origin="application-derived",
            difficulty="easy",
        ),
    ]

    selected, receipt = _stratified_gate_sample(rows, count=2, seed=0)

    for field in (
        "host_family",
        "fault_status",
        "problem_origin",
        "embedding_difficulty",
        "sampling_difficulty",
        "decision_difficulty",
    ):
        assert len({item.design_condition[field] for item in selected}) == 2
        assert set(receipt["selected_axis_census"][field].values()) == {1}

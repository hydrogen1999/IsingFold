"""Fail-closed diagnostics for cross-embedding q_psi ordering."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from isingfold.rl.checkpoint import runtime_implementation_registry  # noqa: E402
from isingfold.rl.cli import build_parser  # noqa: E402
from isingfold.rl.contracts import Context, WorkVector, stable_digest  # noqa: E402
from isingfold.rl.data.import_embedbench import content_digest  # noqa: E402
from isingfold.rl.data.quality import (  # noqa: E402
    CONTINUATION_POLICY_ID,
    CONTINUATION_RECEIPT_SCHEMA,
    CONTINUATION_RECEIPT_VERSION,
    _canonical_program,
    _encoded_chains,
    legacy_quality_initializer_binding,
)
from isingfold.rl.qpsi_audit import (  # noqa: E402
    audit_qpsi_ordering,
    publish_qpsi_ordering_audit,
)
from isingfold.rl.validate import p_return  # noqa: E402
from tests.unit.test_rl_selector_labels import _prepared_task  # noqa: E402


class _FrozenQpsi:
    frozen = True
    deployment_ready = True
    normalizer_digest = "selector-normalizer-unit-v1"
    coefficient_transform_scale = torch.tensor(1.0, dtype=torch.float64)

    def __call__(self, inputs):
        maximum_claimed_qubit = int(inputs[0].index_claims[1].max())
        score = 0.9 if maximum_claimed_qubit > 1 else 0.3
        return torch.tensor((score, 0.2, 0.1, 0.05), dtype=torch.float32)


def _receipt(
    task,
    chains,
    *,
    hits: int,
    reads: int,
    seed: int,
    selected_index: int = 0,
) -> dict[str, object]:
    context = Context(qubit_cap=4)
    validation, programs = p_return(
        chains,
        task.logical,
        task.host,
        task.problem,
        context,
    )
    assert validation.valid
    encoded_embedding = _encoded_chains(chains)
    encoded_programs = [_canonical_program(program) for program in programs]
    evidence = {
        "embedding": encoded_embedding,
        "programs": encoded_programs,
        "selected_index": selected_index,
        "selected_strength": float(programs[selected_index].strength),
        "selected_program_digest": stable_digest(
            {
                "program": encoded_programs[selected_index],
                "embedding": encoded_embedding,
            }
        ),
        "evaluator_count_block": {
            "seed": seed,
            "hits": hits,
            "reads": reads,
            "num_sweeps": context.num_sweeps,
        },
    }
    payload = {
        "schema": CONTINUATION_RECEIPT_SCHEMA,
        "schema_version": CONTINUATION_RECEIPT_VERSION,
        "continuation_seed": seed,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "continuation_steps": 1,
        "action_trace": [],
        "returned_valid": True,
        "terminal_reason": "COMMIT",
        "reward": hits / reads,
        "requested_reward_reads": reads,
        "validation_receipt": validation.as_dict(),
        "cumulative_work": WorkVector().as_dict(),
        "terminal_evidence": evidence,
        "initializer_binding": legacy_quality_initializer_binding(),
    }
    return {**payload, "record_digest": stable_digest(payload)}


def _quality_row(item, receipts: list[dict[str, object]]) -> dict[str, object]:
    payload = {
        "schema": "isingfold.quality-counterfactual",
        "schema_version": 7,
        "task_id": item.task_id,
        "instance_id": item.instance_id,
        "lineage": item.task.lineage,
        "partition": "train",
        "state_fingerprint": f"state-{item.task_id}",
        "evaluated": [{"continuation_receipts": receipts}],
    }
    return {**payload, "record_digest": content_digest(payload)}


def _two_lineage_inputs():
    first = _prepared_task(
        "qpsi-a",
        "lineage-a",
        "train",
        partition_target_count=2,
    )
    second = _prepared_task(
        "qpsi-b",
        "lineage-b",
        "train",
        partition_target_count=2,
    )
    low = {0: frozenset((0,)), 1: frozenset((1,))}
    high = {0: frozenset((2,)), 1: frozenset((3,))}
    rows = (
        _quality_row(
            first,
            [
                _receipt(first.task, high, hits=8, reads=10, seed=101),
                _receipt(first.task, low, hits=2, reads=10, seed=102),
            ],
        ),
        _quality_row(
            second,
            [
                _receipt(second.task, high, hits=7, reads=10, seed=201),
                _receipt(second.task, low, hits=2, reads=10, seed=202),
            ],
        ),
    )
    return (first, second), rows


def _signed(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _provenance() -> dict[str, object]:
    from isingfold.rl import evaluator, program, qpsi_audit, strength, strength_tensorize

    runtime = runtime_implementation_registry()
    quality_contract = {"label_version": "fixture-v1"}
    evaluator_protocols = [
        {"task_id": "qpsi-a", "evaluator_protocol_digest": "1" * 64},
        {"task_id": "qpsi-b", "evaluator_protocol_digest": "2" * 64},
    ]
    return {
        "partition": "train",
        "sealed_validation_or_test_opened": False,
        "source_corpus_manifest_sha256": "3" * 64,
        "source_quality_manifest_sha256": "4" * 64,
        "source_quality_manifest_record_digest": "5" * 64,
        "source_quality_records_sha256": "6" * 64,
        "quality_preflight_receipt_sha256": "7" * 64,
        "quality_preflight_record_digest": "8" * 64,
        "selector_digest": "9" * 64,
        "selector_file_sha256": "a" * 64,
        "selector_fit_receipt_sha256": "b" * 64,
        "selector_fit_record_digest": "c" * 64,
        "normalizer_digest": "d" * 64,
        "quality_authority": _signed({"authority": "train"}),
        "target_access": _signed({"partition": "train"}),
        "ground_partition_receipt": _signed({"partition": "train"}),
        "quality_implementation_contract": quality_contract,
        "quality_implementation_contract_digest": content_digest(quality_contract),
        "task_evaluator_protocols": evaluator_protocols,
        "task_evaluator_protocols_digest": content_digest(evaluator_protocols),
        "runtime_implementation_registry": runtime,
        "runtime_implementation_digest": content_digest(runtime),
        "implementation_sources": {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in {
                "evaluator": evaluator,
                "program": program,
                "qpsi_audit": qpsi_audit,
                "selector": strength,
                "selector_tensorizer": strength_tensorize,
            }.items()
        },
    }


def test_qpsi_audit_reports_equal_lineage_instance_ranking_and_last_baseline() -> None:
    tasks, records = _two_lineage_inputs()

    rows, aggregate = audit_qpsi_ordering(
        _FrozenQpsi(),
        records,
        tasks,
        Context(qubit_cap=4),
        bootstrap_replicates=1_000,
        bootstrap_seed=71,
    )

    assert len(rows) == 2
    assert rows[0]["distinct_terminal_embeddings"] == 2
    assert rows[0]["spearman_rho"] == pytest.approx(1.0)
    assert rows[0]["pairwise_sign_accuracy"] == pytest.approx(1.0)
    assert rows[0]["qpsi_top1_regret"] == pytest.approx(0.0)
    assert rows[0]["qpsi_top_minus_last"] == pytest.approx(0.6)
    primary = aggregate["equal_lineage_then_equal_instance"]
    assert primary["spearman_rho"] == pytest.approx(1.0)
    assert primary["pairwise_sign_accuracy"] == pytest.approx(1.0)
    assert primary["qpsi_top1_regret"] == pytest.approx(0.0)
    assert primary["qpsi_top_minus_last"] == pytest.approx(0.55)
    assert primary["qpsi_top_beats_last_rate"] == pytest.approx(1.0)
    assert aggregate["pairwise_counts"] == {
        "correct": 2,
        "incorrect": 0,
        "score_ties": 0,
        "target_ties": 0,
        "comparable": 2,
    }
    assert aggregate["exact_work_census"] == {
        "selector_forward_calls": 4,
        "selector_program_inputs": 16,
        "program_compiler_calls": 16,
    }
    delta_ci = aggregate["lineage_cluster_bootstrap"]["qpsi_top_minus_last"]
    assert delta_ci["lower"] == pytest.approx(0.5)
    assert delta_ci["upper"] == pytest.approx(0.6)


def test_qpsi_audit_rejects_rehashed_program_or_selector_mismatch() -> None:
    tasks, records = _two_lineage_inputs()
    tampered = deepcopy(records)
    receipt = tampered[0]["evaluated"][0]["continuation_receipts"][0]
    evidence = receipt["terminal_evidence"]
    evidence["programs"][0]["scale"] += 0.25
    evidence["selected_program_digest"] = stable_digest(
        {"program": evidence["programs"][0], "embedding": evidence["embedding"]}
    )
    receipt["record_digest"] = stable_digest(
        {key: value for key, value in receipt.items() if key != "record_digest"}
    )
    tampered[0]["record_digest"] = content_digest(
        {key: value for key, value in tampered[0].items() if key != "record_digest"}
    )

    with pytest.raises(ValueError, match="compiled programs"):
        audit_qpsi_ordering(
            _FrozenQpsi(),
            tampered,
            tasks,
            Context(qubit_cap=4),
            bootstrap_replicates=1_000,
            bootstrap_seed=71,
        )

    invalidated = deepcopy(records)
    invalid_receipt = invalidated[0]["evaluated"][0]["continuation_receipts"][0]
    invalid_receipt["validation_receipt"]["valid"] = False
    invalid_receipt["record_digest"] = stable_digest(
        {
            key: value
            for key, value in invalid_receipt.items()
            if key != "record_digest"
        }
    )
    invalidated[0]["record_digest"] = content_digest(
        {key: value for key, value in invalidated[0].items() if key != "record_digest"}
    )
    with pytest.raises(ValueError, match="validation receipt"):
        audit_qpsi_ordering(
            _FrozenQpsi(),
            invalidated,
            tasks,
            Context(qubit_cap=4),
            bootstrap_replicates=1_000,
            bootstrap_seed=71,
        )

    mismatched = deepcopy(records)
    for row in mismatched:
        for action in row["evaluated"]:
            for continuation in action["continuation_receipts"]:
                continuation["terminal_evidence"]["selected_index"] = 1
                continuation["terminal_evidence"]["selected_strength"] = continuation[
                    "terminal_evidence"
                ]["programs"][1]["strength"]
                continuation["terminal_evidence"]["selected_program_digest"] = stable_digest(
                    {
                        "program": continuation["terminal_evidence"]["programs"][1],
                        "embedding": continuation["terminal_evidence"]["embedding"],
                    }
                )
                continuation["record_digest"] = stable_digest(
                    {
                        key: value
                        for key, value in continuation.items()
                        if key != "record_digest"
                    }
                )
        row["record_digest"] = content_digest(
            {key: value for key, value in row.items() if key != "record_digest"}
        )
    with pytest.raises(ValueError, match="frozen selector choice"):
        audit_qpsi_ordering(
            _FrozenQpsi(),
            mismatched,
            tasks,
            Context(qubit_cap=4),
            bootstrap_replicates=1_000,
            bootstrap_seed=71,
        )


def test_qpsi_audit_fails_closed_without_cross_embedding_or_independent_lineages() -> None:
    tasks, records = _two_lineage_inputs()
    single_embedding = (deepcopy(records[0]),)
    single_embedding[0]["evaluated"][0]["continuation_receipts"] = single_embedding[0][
        "evaluated"
    ][0]["continuation_receipts"][:1]
    single_embedding[0]["record_digest"] = content_digest(
        {
            key: value
            for key, value in single_embedding[0].items()
            if key != "record_digest"
        }
    )
    with pytest.raises(ValueError, match="minimum cross-embedding contract"):
        audit_qpsi_ordering(
            _FrozenQpsi(),
            single_embedding,
            tasks[:1],
            Context(qubit_cap=4),
            bootstrap_replicates=1_000,
            bootstrap_seed=71,
        )

    with pytest.raises(ValueError, match="two independent train lineages"):
        audit_qpsi_ordering(
            _FrozenQpsi(),
            records[:1],
            tasks[:1],
            Context(qubit_cap=4),
            bootstrap_replicates=1_000,
            bootstrap_seed=71,
        )


def test_qpsi_publication_is_diagnostic_only_and_rejects_false_provenance(
    tmp_path: Path,
) -> None:
    tasks, records = _two_lineage_inputs()
    rows, aggregate = audit_qpsi_ordering(
        _FrozenQpsi(),
        records,
        tasks,
        Context(qubit_cap=4),
        bootstrap_replicates=1_000,
        bootstrap_seed=71,
    )
    destination = tmp_path / "qpsi-audit"
    receipt = publish_qpsi_ordering_audit(
        destination,
        rows=rows,
        aggregate=aggregate,
        provenance=_provenance(),
    )

    assert receipt["scope"] == "DIAGNOSTIC_ONLY"
    assert receipt["archive_policy_authorized"] is False
    assert receipt["fifo_archive_comparator"]["available"] is False
    assert set(path.name for path in destination.iterdir()) == {
        "opportunities.jsonl",
        "receipt.json",
    }
    assert json.loads((destination / "receipt.json").read_text()) == receipt

    forged = _provenance()
    forged["sealed_validation_or_test_opened"] = True
    with pytest.raises(ValueError, match="train-only"):
        publish_qpsi_ordering_audit(
            tmp_path / "forged",
            rows=rows,
            aggregate=aggregate,
            provenance=forged,
        )

    forged_source = _provenance()
    forged_source["implementation_sources"]["program"] = "e" * 64
    with pytest.raises(ValueError, match="implementation source"):
        publish_qpsi_ordering_audit(
            tmp_path / "forged-source",
            rows=rows,
            aggregate=aggregate,
            provenance=forged_source,
        )


def test_cli_registers_train_only_qpsi_ordering_diagnostic() -> None:
    args = build_parser().parse_args(
        [
            "audit-qpsi-ordering",
            "--corpus",
            "prepared",
            "--selector",
            "selector",
            "--quality-attestation",
            "attestation.json",
            "--expected-quality-attestation-digest",
            "a" * 64,
            "--expected-quality-publisher-id",
            "publisher",
            "--ground-certificate-root",
            "ground-root.json",
            "--expected-ground-certificate-root-sha256",
            "b" * 64,
            "--quality-labels",
            "quality",
            "--initializer-bank",
            "initializer-bank",
            "--expected-initializer-bank-manifest-sha256",
            "d" * 64,
            "--complete-config",
            "complete-system.json",
            "--quality-preflight-receipt",
            "preflight.json",
            "--expected-quality-preflight-sha256",
            "c" * 64,
            "--out",
            "qpsi-audit",
        ]
    )

    assert args.command == "audit-qpsi-ordering"
    assert args.bootstrap_replicates == 20_000
    assert not hasattr(args, "partition")

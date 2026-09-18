"""Evidence must come from evaluation rows, not a header's split declaration."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from constructor_evidence_audit import audit_log


ROOT = Path(__file__).resolve().parents[2]


def header(**changes):
    return {"protocol": "constructor-curriculum-gate2-v1", "train": ["train-0", "train-1"],
            "heldout": ["test-0"], "heldout_role": "test", "manifest_split": True,
            "eval_sets": "both", **changes}


def evaluation(**changes):
    return {"evaluation": "final", "set": "heldout", "instances": 1,
            "per_instance": {"test-0": 0.5}, **changes}


def write_log(tmp_path, *rows):
    path = tmp_path / "run.log"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def codes(report):
    return {item["code"] for item in report["errors"]}


def test_valid_final_test_evidence_is_counted(tmp_path):
    path = write_log(tmp_path, header(), evaluation())
    report = audit_log(path, require_heldout_test=True)
    assert report["ok"] and report["heldout_test_evidence"]
    assert report["header"]["declared_instances"] == {"train": 2, "heldout": 1}
    assert report["evaluations"][0]["observed_instances"] == 1


def test_completed_real_curriculum_log_accepts_normal_footer():
    path = ROOT / "results" / "curriculum" / "b_local_s0.log"
    assert path.read_text().splitlines()[-1] == "CURRICULUM GATE 2 DONE"
    report = audit_log(path)
    assert report["ok"], report["errors"]
    assert any(row["evaluation"] == "final" and row["set"] == "heldout"
               and row["valid"] for row in report["evaluations"])


def test_completion_footer_cannot_replace_final_evidence(tmp_path):
    path = write_log(tmp_path, header(), evaluation(evaluation="init"),
                     {"summary": "gate2", "heldout": {"final": 1.0}})
    with path.open("a", encoding="utf-8") as stream:
        stream.write("CURRICULUM GATE 2 DONE\n")
    report = audit_log(path, require_heldout_test=True)
    assert not report["ok"]
    assert {"missing_final_evaluation", "missing_heldout_test_evidence"} <= codes(report)
    assert "malformed_json" not in codes(report)


@pytest.mark.parametrize("extra", ["CURRICULUM GATE 2 DONE with errors", '{"evaluation":'])
def test_normal_footer_does_not_hide_unrecognized_or_malformed_lines(tmp_path, extra):
    path = write_log(tmp_path, header(), evaluation())
    with path.open("a", encoding="utf-8") as stream:
        stream.write("CURRICULUM GATE 2 DONE\n" + extra + "\n")
    report = audit_log(path)
    assert not report["ok"] and "malformed_json" in codes(report)


def test_header_and_checkpoint_summary_cannot_supply_test_evidence(tmp_path):
    path = write_log(tmp_path, header(init_summary={"heldout": {"final": 1.0}}),
                     evaluation(evaluation="init", set="train", instances=2,
                                per_instance={"train-0": 0.5, "train-1": 0.5}))
    report = audit_log(path, require_heldout_test=True)
    assert not report["ok"] and not report["heldout_test_evidence"]
    assert {"missing_final_evaluation", "missing_heldout_test_evidence"} <= codes(report)
    assert report["evaluations"][0]["set"] == "train"


@pytest.mark.parametrize("path", sorted((ROOT / "results" / "transfer").glob("m_*.log")),
                         ids=lambda path: path.name)
def test_archived_transfer_logs_do_not_prove_heldout_test_results(path):
    report = audit_log(path, require_heldout_test=True)
    assert not report["ok"] and not report["heldout_test_evidence"]
    assert report["header"]["heldout_role"] == "test"
    assert report["header"]["declared_instances"] == {"train": 16, "heldout": 15}
    assert {"missing_final_evaluation", "missing_heldout_test_evidence"} <= codes(report)
    assert [(row["evaluation"], row["set"], row["observed_instances"])
            for row in report["evaluations"]] == [("init", "train", 16)]


def test_partial_inspection_is_explicit_and_never_proves_final_test_evidence(tmp_path):
    path = write_log(tmp_path, header(), evaluation(evaluation="init"))
    assert "missing_final_evaluation" in codes(audit_log(path))
    assert audit_log(path, require_final=False)["ok"]
    assert not audit_log(path, require_final=False, require_heldout_test=True)["ok"]


def test_validation_run_requires_each_configured_final_split(tmp_path):
    path = write_log(tmp_path, header(heldout_role="validation"), evaluation())
    report = audit_log(path)
    assert not report["ok"]
    assert any(error["code"] == "missing_final_evaluation" and error["set"] == "train"
               for error in report["errors"])
    path = write_log(tmp_path, header(heldout_role="validation", eval_sets="heldout"), evaluation())
    assert audit_log(path)["ok"]
    assert not audit_log(path, require_heldout_test=True)["ok"]


def test_deployment_evaluation_needs_only_heldout_final_rows(tmp_path):
    path = write_log(tmp_path, header(heldout_role="validation", objective="feasibility",
                                      evaluation_objective="deployment"), evaluation())
    assert audit_log(path)["ok"]


@pytest.mark.parametrize("objective", ["quality", "deployment"])
def test_quality_and_deployment_rows_supply_instance_evidence(tmp_path, objective):
    row = evaluation(objective=objective)
    del row["per_instance"]
    row["rows"] = [{"instance": "test-0", "policy": {"valid": True}}]
    path = write_log(tmp_path, header(protocol="constructor-quality-v2"), row)
    assert audit_log(path, require_heldout_test=True)["ok"]
    row["rows"].append(row["rows"][0])
    row["instances"] = 2
    path = write_log(tmp_path, header(), row)
    assert "duplicate_identity" in codes(audit_log(path))


@pytest.mark.parametrize(("changes", "code"), [
    ({"per_instance": {"train-0": 0.5}}, "split_mismatch"),
    ({"per_instance": {}}, "split_mismatch"),
    ({"per_instance": {"test-0": 0.5, "unknown": 1.0}}, "count_mismatch"),
    ({"instances": 2}, "count_mismatch"),
    ({"instances": True}, "invalid_count"),
    ({"instances": 1.0}, "invalid_count"),
    ({"per_instance": []}, "malformed_per_instance"),
    ({"per_instance": {"": 0.5}}, "malformed_identities"),
    ({"set": "test"}, "invalid_set"),
    ({"set": []}, "invalid_set"),
    ({"evaluation": None}, "invalid_evaluation"),
])
def test_rejects_malformed_or_mismatched_evaluation(tmp_path, changes, code):
    report = audit_log(write_log(tmp_path, header(), evaluation(**changes)))
    assert not report["ok"] and code in codes(report)


@pytest.mark.parametrize(("changes", "code"), [
    ({"train": ["train-0", "train-0"]}, "duplicate_identity"),
    ({"heldout": ["train-0"]}, "split_overlap"),
    ({"heldout": None}, "malformed_identities"),
    ({"heldout": [None]}, "malformed_identities"),
    ({"eval_sets": "train"}, "invalid_eval_sets"),
    ({"heldout_role": "train"}, "invalid_heldout_role"),
    ({"protocol": "unrelated"}, "unsupported_protocol"),
    ({"protocol": []}, "unsupported_protocol"),
    ({"protocol": {}}, "unsupported_protocol"),
])
def test_rejects_invalid_split_declarations(tmp_path, changes, code):
    report = audit_log(write_log(tmp_path, header(**changes), evaluation()))
    assert not report["ok"] and code in codes(report)


def test_rejects_missing_and_duplicate_records(tmp_path):
    assert "missing_header" in codes(audit_log(write_log(tmp_path, evaluation())))
    assert "missing_evaluation" in codes(audit_log(write_log(tmp_path, header())))
    assert "duplicate_header" in codes(audit_log(write_log(tmp_path, header(), header(), evaluation())))
    assert "duplicate_evaluation" in codes(audit_log(
        write_log(tmp_path, header(), evaluation(), evaluation())))
    assert "evaluation_before_header" in codes(audit_log(
        write_log(tmp_path, evaluation(), header())))
    row = evaluation()
    del row["per_instance"]
    assert "missing_instance_evidence" in codes(audit_log(write_log(tmp_path, header(), row)))
    assert "unreadable_log" in codes(audit_log(tmp_path / "missing.log"))


@pytest.mark.parametrize(("raw", "code"), [
    ('{"evaluation": "final",', "malformed_json"),
    ('[]', "malformed_record"),
    ('{"evaluation":"final","set":"heldout","instances":1,'
     '"per_instance":{"test-0":0.5,"test-0":1.0}}', "duplicate_json_key"),
])
def test_invalid_json_and_duplicate_json_keys_are_not_silently_accepted(tmp_path, raw, code):
    path = write_log(tmp_path, header(), evaluation())
    with path.open("a", encoding="utf-8") as stream:
        stream.write(raw + "\n")
    report = audit_log(path)
    assert not report["ok"] and code in codes(report)


def test_cli_emits_json_and_nonzero_status_on_failed_audit(tmp_path):
    path = write_log(tmp_path, header(), evaluation(evaluation="init"))
    command = [sys.executable, str(ROOT / "probes" / "constructor_evidence_audit.py"), str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 1 and not result.stderr
    assert not json.loads(result.stdout)["ok"]
    result = subprocess.run([*command, "--allow-incomplete"], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0 and json.loads(result.stdout)["ok"]
    result = subprocess.run([*command, "--allow-incomplete", "--require-heldout-test"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1 and not json.loads(result.stdout)["ok"]


@pytest.mark.parametrize("field", ["evaluation", "set", "instances"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_malformed_nonfinite_metadata_still_yields_strict_machine_readable_json(
    tmp_path, field, value
):
    path = write_log(tmp_path, header(), evaluation(**{field: value}))
    result = subprocess.run(
        [sys.executable, str(ROOT / "probes" / "constructor_evidence_audit.py"), str(path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1 and not result.stderr

    def reject_nonfinite(token):
        pytest.fail(f"audit emitted a non-JSON value: {token}")

    assert not json.loads(result.stdout, parse_constant=reject_nonfinite)["ok"]


def test_test_evidence_requires_explicit_manifest_test_role(tmp_path):
    for changes in ({"manifest_split": False}, {"manifest_split": "true"},
                    {"heldout_role": "validation", "eval_sets": "heldout"}):
        report = audit_log(write_log(tmp_path, header(**changes), evaluation()),
                           require_heldout_test=True)
        assert not report["ok"] and "missing_heldout_test_evidence" in codes(report)

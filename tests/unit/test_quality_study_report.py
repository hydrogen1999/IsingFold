"""Attribution must preserve failures, experimental controls and independent units."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "probes" / "quality_study_report.py"
_spec = importlib.util.spec_from_file_location("quality_study_report", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def arm(valid=True, residual=.2, p_solve=.4, utility=.8):
    return {"valid": valid, "residual": residual if valid else None,
            "p_solve": p_solve if valid else None,
            "quality_utility": utility if valid else 0., "assessment_ok": valid,
            "assessment_failures": 0, "selection_failures": 0,
            "budget": {"deadline_seconds": 30, "assessment_reads_per_call": 512}}


def evaluation(tag="final", seed=0, arms=None):
    arms = arms or [arm(), arm()]
    return {"evaluation": tag, "set": "heldout", "objective": "quality", "stage": "Z",
            "seed": seed, "instances": len(arms), "rows": [
                {"instance": "instance%d" % i, "lineage": "lineage%d" % i, "policy": a}
                for i, a in enumerate(arms)]}


def header(objective="quality", seed=0):
    options = {k: 1 for k in report.REQUIRED_OPTIONS}
    options.update(objective=objective, stage="Z", seed=seed, actor="linear", features="tiny",
                   support="registered", init="feas.pt", evaluation_objective="quality",
                   heldout_role="validation")
    return {"protocol": report.VERSION, "contract": {
        "version": report.VERSION, "source_commit": "abc", "source_dirty": False,
        "initial_checkpoint_sha256": "checkpoint-bytes-sha256",
        "train_fingerprints": ["train-hash"], "heldout_fingerprints": ["h1", "h2"],
        "options": options}}


def write_log(tmp_path, name, objective="quality", final=None, first=None):
    path = tmp_path / name
    records = [header(objective)]
    if first is not None:
        records.append(first)
    records.append(final or evaluation())
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def test_matched_control_and_within_run_deltas_have_correct_signs(tmp_path):
    improved = evaluation(arms=[arm(residual=.1, p_solve=.6, utility=.9)] * 2)
    q = write_log(tmp_path, "quality.log", final=improved, first=evaluation("init"))
    c = write_log(tmp_path, "control.log", "feasibility")
    result = report.build_report(q, c, draws=500)
    paired = result["quality_minus_continued_feasibility"]
    assert paired["blocking_reasons"] == []
    for name, expected in (("deployment_p_solve", .2), ("quality_utility", .1),
                           ("conditional_residual", .1)):
        metric = paired["metrics"][name]
        assert metric["mean_gain"] == pytest.approx(expected)
        assert metric["inference_eligible"]
    assert result["within_run_final_minus_init"]["metrics"]["quality_utility"]["mean_gain"] == pytest.approx(.1)


def test_failure_is_zero_deployment_utility_not_excluded():
    left = evaluation(arms=[arm(residual=.1, utility=.9), arm(valid=False)])
    right = evaluation()
    paired = report.paired_report(left, right, draws=500)
    assert paired["coverage"] == {"left": .5, "right": 1.}
    assert paired["metrics"]["quality_utility"]["mean_gain"] == pytest.approx(-.35)
    residual = paired["metrics"]["conditional_residual"]
    assert residual["mean_gain"] == pytest.approx(.1)
    assert residual["instances"] == 1 and residual["ci95"] is None


def test_assessment_failure_blocks_inference_instead_of_becoming_zero():
    left = evaluation()
    left["rows"][0]["policy"].update(assessment_failures=1, assessment_ok=False,
                                        residual=None, quality_utility=0.)
    paired = report.paired_report(left, evaluation(), draws=500)
    assert paired["diagnostic_only"] and paired["measurement_errors"]
    for metric in paired["metrics"].values():
        assert metric["mean_gain"] is None and metric["ci95"] is None
        assert not metric["inference_eligible"]


def test_missing_probability_on_valid_output_does_not_become_zero():
    left = evaluation()
    left["rows"][0]["policy"]["p_solve"] = None
    paired = report.paired_report(left, evaluation(), draws=500)
    metric = paired["metrics"]["deployment_p_solve"]
    assert metric["mean_gain"] is None and metric["ci95"] is None
    assert metric["missing_instances"] == ["instance0"]


def test_cluster_bootstrap_uses_lineages_not_rows_or_reads():
    a = report.paired_bootstrap([("a", 1.), ("a", 1.), ("b", 0.)], draws=500)
    b = report.paired_bootstrap([("a", 1.), ("b", 0.)], draws=500)
    assert a["lineages"] == 2 and a["instances"] == 3
    assert a["mean_gain"] == b["mean_gain"] == .5
    assert a["ci95"] == b["ci95"]


@pytest.mark.parametrize("changed", ["deadline", "assessment_reads", "features", "support", "init", "seed"])
def test_changed_budget_representation_initialization_or_seed_blocks_attribution(changed):
    q, c = {"header": header()}, {"header": header("feasibility")}
    c["header"]["contract"]["options"][changed] = "different"
    reasons = report.attribution_reasons(q, c, evaluation(), evaluation())
    assert "different option " + changed in reasons


def test_different_fingerprints_block_named_instance_overlap():
    q, c = {"header": header()}, {"header": header("feasibility")}
    c["header"]["contract"]["heldout_fingerprints"] = ["changed", "h2"]
    assert "different heldout_fingerprints" in report.attribution_reasons(q, c, evaluation(), evaluation())


def test_same_checkpoint_path_with_changed_bytes_blocks_attribution():
    q, c = {"header": header()}, {"header": header("feasibility")}
    c["header"]["contract"]["initial_checkpoint_sha256"] = "different-bytes-sha256"
    reasons = report.attribution_reasons(q, c, evaluation(), evaluation())
    assert "different initial_checkpoint_sha256" in reasons


def test_warm_start_requires_hash_but_cold_starts_accept_none():
    q, c = {"header": header()}, {"header": header("feasibility")}
    for log in (q, c):
        log["header"]["contract"]["initial_checkpoint_sha256"] = None
    assert "missing initial checkpoint content hash" in report.attribution_reasons(
        q, c, evaluation(), evaluation())
    for log in (q, c):
        log["header"]["contract"]["options"]["init"] = ""
    assert report.attribution_reasons(q, c, evaluation(), evaluation()) == []


def test_legacy_log_is_only_descriptive(tmp_path):
    old = evaluation()
    for r in old["rows"]:
        r.pop("lineage")
    path = tmp_path / "old.log"
    first = copy.deepcopy(old)
    first["evaluation"] = "init"
    path.write_text(json.dumps(first) + "\n" + json.dumps(old))
    within = report.build_report(path, draws=500)["within_run_final_minus_init"]
    assert within["diagnostic_only"]
    assert within["metrics"]["conditional_residual"]["mean_gain"] == 0.
    assert within["metrics"]["conditional_residual"]["ci95"] is None


def test_requires_exact_instance_list_and_never_selects_best_checkpoint(tmp_path):
    q = write_log(tmp_path, "q.log", final=evaluation("iter 49"))
    with pytest.raises(ValueError, match="expected one"):
        report.build_report(q, draws=500)
    assert report.build_report(q, tag="iter 49", draws=500)["selected_tag"] == "iter 49"
    left, right = evaluation(), evaluation()
    right["rows"].reverse()
    with pytest.raises(ValueError, match="ordered instance"):
        report.paired_report(left, right, draws=500)


def test_multiple_run_headers_reject_seed_pooling(tmp_path):
    path = tmp_path / "pooled.log"
    path.write_text(json.dumps(header()) + "\n" + json.dumps(header(seed=1)))
    with pytest.raises(ValueError, match="pooled seeds"):
        report.load_log(path)

import importlib.util
import json
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "probes" / "ablation_table.py"
_spec = importlib.util.spec_from_file_location("ablation_table", _path)
table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(table)


def run(features="tiny", seed=0):
    header = {key: None for key in table.CONTROLS}
    header.update(actor="linear", support="registered", stop_bias=0, max_steps=240,
                  train=["train-a"], heldout=["valid-a"],
                  train_sizes=[[12, 64]], heldout_sizes=[[12, 64]])
    return dict(header=header, stage="F", features=features, seed=seed, arm=features,
                init=.1, final=.833, gain=.733)


def test_local_features_do_not_hide_support_and_stop_bias_confound():
    tiny, local = run(), run("local")
    local["header"].update(support="wide", stop_bias=-6)
    reasons = table.comparison_reasons(tiny, local)
    assert reasons == ["different support", "different stop_bias"]
    assert table.feature_comparisons([tiny, local])[0][3] == 0


def test_same_names_need_same_budgets_initialization_and_sets():
    tiny, local = run(), run("local")
    local["header"].update(init="warm.pt", max_steps=1500, heldout=["valid-b"])
    reasons = table.comparison_reasons(tiny, local)
    assert set(reasons) == {"different init", "different max_steps", "different heldout"}
    assert table.control_signature(tiny) != table.control_signature(local)


def test_missing_legacy_controls_are_unknown_not_current_defaults():
    tiny, local = run(), run("local")
    del tiny["header"]["support"]
    del tiny["header"]["init"]
    reasons = table.comparison_reasons(tiny, local)
    assert "unknown support" in reasons and "unknown init" in reasons
    assert table.control_signature(tiny) != table.control_signature(local)


def test_identical_controls_allow_feature_comparison_only_with_paired_seeds():
    tiny, local = run(), run("local")
    assert table.comparison_reasons(tiny, local) == []
    assert table.feature_comparisons([tiny, local])[0][3:] == (1, [])
    local["seed"] = 1
    report = table.feature_comparisons([tiny, local])[0]
    assert report[3] == 0 and "no shared seed" in report[4]


def test_two_equal_rates_and_duplicate_seeds_do_not_produce_zero_width_ci():
    assert table.descriptive_note([run(seed=0), run(seed=1)]) == "2 seed(s); descriptive only; no CI"
    assert "duplicate" in table.descriptive_note([run(), run()])
    note = table.descriptive_note([run(seed=i) for i in range(3)])
    assert "SD" in note and "no CI" in note


def test_old_log_still_prints_descriptive_rate(tmp_path, capsys):
    summary = {"summary": "gate2", "stage": "F", "seed": 0, "features": "tiny",
               "heldout": {"init": .1, "final": .8, "gain": .7}}
    (tmp_path / "F_linear_s0.log").write_text(json.dumps(summary) + "\n")
    assert table.main(["--logs", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "0.80" in output and "no CI" in output and "ABLATION TABLE DONE" in output


def test_different_horizons_are_not_pooled_as_seeds(tmp_path, capsys):
    for seed, horizon in enumerate((240, 1500)):
        header = run(seed=seed)["header"]
        header.update(protocol="v1", stage="F", max_steps=horizon)
        summary = {"summary": "gate2", "stage": "F", "seed": seed,
                   "heldout": {"init": .1, "final": .8, "gain": .7}}
        (tmp_path / ("F_linear_s%d.log" % seed)).write_text(
            json.dumps(header) + "\n" + json.dumps(summary) + "\n")
    table.main(["--logs", str(tmp_path)])
    output = capsys.readouterr().out
    assert output.count("1 seed(s); descriptive only; no CI") == 2

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes"))
from run_quality_study import build_plan


def test_quality_and_feasibility_controls_have_identical_inputs_and_budgets():
    config = json.loads((ROOT / "configs/constructor_quality_study_v2.json").read_text())
    jobs = build_plan(config)
    assert len(jobs) == 12
    for host in config["hosts"]:
        for feature in config["features"]:
            arms = {j["arm"]: j for j in jobs if j["host"] == host and j["features"] == feature}
            def options(job):
                # Flags are deliberately preserved in comparison as well.
                result = list(job["command"])
                for flag in ("--objective", "--out"):
                    i = result.index(flag)
                    del result[i:i+2]
                return result
            assert options(arms["quality"]) == options(arms["continued_feasibility"])
            assert arms["quality"]["required"] == arms["frozen"]["required"]
            assert "--manifest-split" in arms["quality"]["command"]
            assert "--prefix-fraction" not in arms["quality"]["command"]
            assert ("--expand-features" in arms["quality"]["command"]) == (feature == "physics")
            assert "--heldout-role" not in arms["quality"]["command"]

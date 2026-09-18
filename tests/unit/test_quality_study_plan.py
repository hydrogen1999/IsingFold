import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probes"))
import run_quality_study as study
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


@pytest.fixture
def small_study(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    source = tmp_path / "mutable.best.pt"
    source.write_bytes(b"initial weights")
    (tmp_path / "corpus").mkdir()
    config = {"protocol": "test-study", "data_seed": 7, "seeds": [0],
              "hosts": {"host": {"corpus": "corpus", "init": source.name}},
              "features": ["local", "physics"], "common": {},
              "pilot": {"iterations": 2, "eval_every": 1}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        checkpoint = Path(command[command.index("--init") + 1])
        assert checkpoint.read_bytes() == b"initial weights"
        assert checkpoint != source
        kwargs["stdout"].write("run completed\n")

    monkeypatch.setattr(study.subprocess, "run", launch)
    return tmp_path, config, config_path, source, commands


def test_plan_does_not_create_outputs_or_read_checkpoint_bytes(small_study, capsys):
    directory, _, config_path, source, commands = small_study
    source.unlink()  # Plans must work even before inputs have been staged.
    before = sorted(directory.rglob("*"))
    assert study.main(["--config", str(config_path), "--out-dir", "study"]) == 0
    assert sorted(directory.rglob("*")) == before
    assert not commands
    assert len(json.loads(capsys.readouterr().out)["jobs"]) == 6


def test_separate_only_jobs_share_snapshot_and_manifest(small_study):
    directory, config, config_path, source, commands = small_study
    args = ["--config", str(config_path), "--out-dir", "study", "--execute"]
    study.main(args + ["--only", "host_local_frozen_s0"])
    manifest_path = directory / "study" / study.MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest_stat = manifest_path.stat()
    for tag in ("host_local_continued_feasibility_s0", "host_local_quality_s0",
                "host_physics_quality_s0"):
        study.main(args + ["--only", tag])
    paths = {command[command.index("--init") + 1] for command in commands}
    assert len(paths) == 1
    snapshot = Path(paths.pop())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert snapshot.name == digest + ".pt"
    assert snapshot.stat().st_mode & 0o222 == 0
    assert snapshot.stat().st_ino != source.stat().st_ino  # A copy, not a hard link to the source.
    assert manifest_path.read_bytes() == manifest_bytes
    assert manifest_path.stat().st_ino == manifest_stat.st_ino
    assert json.loads(manifest_bytes)["config"] == config


def test_changed_source_blocks_next_only_job_without_replacing_lock(small_study, capsys):
    directory, _, config_path, source, commands = small_study
    args = ["--config", str(config_path), "--out-dir", "study", "--execute"]
    study.main(args + ["--only", "host_local_frozen_s0"])
    manifest = directory / "study" / study.MANIFEST_NAME
    original_manifest = manifest.read_bytes()
    original_snapshot = Path(commands[0][commands[0].index("--init") + 1])
    source.write_bytes(b"updated best weights")
    with pytest.raises(SystemExit) as error:
        study.main(args + ["--only", "host_local_quality_s0"])
    assert error.value.code == 2
    assert "use a new --out-dir" in capsys.readouterr().err
    assert len(commands) == 1
    assert manifest.read_bytes() == original_manifest
    assert original_snapshot.read_bytes() == b"initial weights"
    assert not (directory / "study/host_local_quality_s0.log").exists()
    fresh = study.freeze_study_inputs(json.loads(config_path.read_text()), phase="pilot",
                                     out_dir="new-study")
    assert fresh["initial_checkpoints"]["host"]["sha256"] != original_snapshot.stem


def test_changed_source_during_execution_blocks_remaining_arms(small_study, monkeypatch, capsys):
    directory, _, config_path, source, commands = small_study
    launch = study.subprocess.run

    def update_best_after_arm(command, **kwargs):
        launch(command, **kwargs)
        source.write_bytes(b"new best checkpoint")

    monkeypatch.setattr(study.subprocess, "run", update_best_after_arm)
    with pytest.raises(SystemExit):
        study.main(["--config", str(config_path), "--out-dir", "study", "--execute"])
    assert len(commands) == 1
    assert "source changed" in capsys.readouterr().err
    assert not (directory / "study/host_local_continued_feasibility_s0.log").exists()


def test_full_configuration_is_locked_before_only_filter(small_study):
    directory, config, config_path, _, _ = small_study
    second = directory / "second.best.pt"
    second.write_bytes(b"other host")
    config["hosts"]["second"] = {"corpus": "corpus", "init": second.name}
    config_path.write_text(json.dumps(config))
    study.main(["--config", str(config_path), "--out-dir", "study", "--execute",
                "--only", "host_local_frozen_s0", "--hosts", "host"])
    manifest = json.loads((directory / "study" / study.MANIFEST_NAME).read_text())
    assert set(manifest["initial_checkpoints"]) == {"host", "second"}
    assert Path(manifest["initial_checkpoints"]["second"]["snapshot"]).read_bytes() == b"other host"


def test_corrupt_snapshot_and_changed_config_are_rejected(small_study):
    directory, config, _, _, _ = small_study
    manifest = study.freeze_study_inputs(config, phase="pilot", out_dir="study")
    snapshot = Path(manifest["initial_checkpoints"]["host"]["snapshot"])
    snapshot.chmod(0o644)
    snapshot.write_bytes(b"damaged snapshot")
    with pytest.raises(ValueError, match="snapshot changed"):
        study.freeze_study_inputs(config, phase="pilot", out_dir="study")
    assert snapshot.read_bytes() == b"damaged snapshot"  # Never silently repair by overwrite.
    config["pilot"]["iterations"] += 1
    with pytest.raises(ValueError, match="configuration changed"):
        study.freeze_study_inputs(config, phase="pilot", out_dir="study")
    assert not list((directory / "study").rglob(".pending-*"))


def test_unlocked_existing_arm_prevents_adopting_new_inputs(small_study):
    directory, config, _, _, _ = small_study
    output = directory / "study"
    output.mkdir()
    (output / "host_physics_quality_s0.log").write_text("previous run")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        study.freeze_study_inputs(config, phase="pilot", out_dir="study")
    assert not (output / study.MANIFEST_NAME).exists()
    assert not (output / "initial_checkpoints").exists()


def test_atomic_publish_reuses_equal_bytes_and_never_replaces_conflict(tmp_path):
    target = tmp_path / "inputs/checkpoint.pt"
    study._publish_once(target, b"original")
    inode = target.stat().st_ino
    study._publish_once(target, b"original")
    with pytest.raises(ValueError, match="refusing to replace"):
        study._publish_once(target, b"different")
    assert target.read_bytes() == b"original"
    assert target.stat().st_ino == inode
    assert not list(target.parent.glob(".pending-*"))

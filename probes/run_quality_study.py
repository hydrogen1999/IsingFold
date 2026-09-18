"""Plan or execute matched quality/continued-feasibility/frozen-checkpoint controls.

No SSH, background jobs or implicit final-test access. Plans are the default;
--execute runs the selected jobs sequentially and refuses missing inputs or outputs
that would be overwritten. Classical solvers occur only in comparison arms.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "study_manifest.json"


def build_plan(config, *, phase="pilot", hosts=None, features=None, out_dir="runs/quality_study_v2"):
    settings = config[phase]
    jobs = []
    for host in hosts or config["hosts"]:
        dataset = config["hosts"][host]
        for feature in features or config["features"]:
            for seed in settings.get("seeds", config["seeds"]):
                for arm in ("frozen", "continued_feasibility", "quality"):
                    tag = f"{host}_{feature}_{arm}_s{seed}"
                    output = str(Path(out_dir) / tag)
                    options = dict(config["common"], stage="corpus", corpus=dataset["corpus"],
                                   init=dataset["init"], features=feature, seed=seed,
                                   data_seed=config["data_seed"],
                                   objective="feasibility" if arm == "continued_feasibility" else "quality",
                                   iterations=0 if arm == "frozen" else settings["iterations"],
                                   eval_every=settings["eval_every"], out=output + ".pt")
                    command = [sys.executable, "-u", str(ROOT / "probes/constructor_curriculum.py")]
                    command.append("--manifest-split")
                    if feature == "physics":
                        command.append("--expand-features")
                    for key, value in options.items():
                        command.extend(["--" + key.replace("_", "-"), str(value)])
                    jobs.append({"tag": tag, "phase": phase, "host": host, "arm": arm,
                                 "features": feature, "seed": seed, "command": command,
                                 "log": output + ".log", "checkpoint": output + ".pt",
                                 "required": [dataset["corpus"], dataset["init"]]})
    return jobs


def _publish_once(path, data):
    """Publish complete, read-only bytes atomically, without replacing a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError(f"refusing to replace conflicting study input: {path}")
    finally:
        os.unlink(temporary)


def validate_study_inputs(manifest):
    """A changed source requires a new study, even when its snapshot survives."""
    for entry in manifest["initial_checkpoints"].values():
        for key in ("source", "snapshot"):
            path = Path(entry[key])
            if key == "snapshot" and path.is_symlink():
                raise ValueError(f"study snapshot must not be a symlink: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"study checkpoint {key} changed: {path}; use a new --out-dir")


def freeze_study_inputs(config, *, phase, out_dir):
    """Lock every configured host before filtering arms, features, or seeds.

    Atomic publication lets separate --only invocations safely share the same
    manifest. A source can never silently update the initialization of one arm.
    """
    directory = (ROOT / out_dir).resolve()
    path = directory / MANIFEST_NAME
    contents, entries = {}, {}
    for host, dataset in config["hosts"].items():
        source = (ROOT / dataset["init"]).resolve()
        if source not in contents:
            contents[source] = source.read_bytes()
        digest = hashlib.sha256(contents[source]).hexdigest()
        entries[host] = {"source": str(source), "sha256": digest,
                         "snapshot": str(directory / "initial_checkpoints" / (digest + ".pt"))}
    manifest = {"schema": "constructor-quality-study-inputs-v1", "phase": phase,
                "config": config, "initial_checkpoints": entries}
    if path.exists() or path.is_symlink():
        if path.is_symlink() or json.loads(path.read_text()) != manifest:
            raise ValueError("study source checkpoint or configuration changed; use a new --out-dir")
    else:
        # Do not retroactively attach a new lock to runs launched without one.
        check_outputs_absent(build_plan(config, phase=phase, out_dir=out_dir))
        for entry in entries.values():
            _publish_once(Path(entry["snapshot"]), contents[Path(entry["source"])])
        validate_study_inputs(manifest)
        _publish_once(path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    validate_study_inputs(manifest)
    return manifest


def apply_study_inputs(jobs, manifest):
    """Bind executable commands to the frozen bytes and expose their provenance."""
    result = []
    for job in jobs:
        entry = manifest["initial_checkpoints"][job["host"]]
        bound = dict(job, command=list(job["command"]), required=list(job["required"]),
                     initial_checkpoint_sha256=entry["sha256"], initial_source=entry["source"])
        bound["command"][bound["command"].index("--init") + 1] = entry["snapshot"]
        bound["required"][1] = entry["snapshot"]
        result.append(bound)
    return result


def check_outputs_absent(jobs):
    for job in jobs:
        for name in (job["log"], job["checkpoint"],
                     str(Path(job["checkpoint"]).with_suffix(".best.pt"))):
            path = ROOT / name
            if path.exists() or path.is_symlink():
                raise ValueError("refusing to overwrite existing artifact: " + str(path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/constructor_quality_study_v2.json"))
    p.add_argument("--phase", choices=("pilot", "replication"), default="pilot")
    p.add_argument("--hosts", nargs="+")
    p.add_argument("--features", nargs="+", choices=("local", "physics"))
    p.add_argument("--only", help="exact job tag to execute or display")
    p.add_argument("--out-dir", default="runs/quality_study_v2")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    config = json.loads(Path(a.config).read_text())
    if a.hosts and not set(a.hosts) <= set(config["hosts"]):
        p.error("unknown host")
    if a.features and not set(a.features) <= set(config["features"]):
        p.error("feature is not registered in this study")
    jobs = build_plan(config, phase=a.phase, hosts=a.hosts, features=a.features, out_dir=a.out_dir)
    if a.only:
        jobs = [job for job in jobs if job["tag"] == a.only]
        if not jobs:
            p.error("--only does not identify a selected job")
    if a.execute:
        # Preflight the selected outputs before creating any study inputs.
        try:
            check_outputs_absent(jobs)
            for job in jobs:
                for path in job["required"]:
                    if not (ROOT / path).exists():
                        raise ValueError("required input is absent: " + path)
            manifest = freeze_study_inputs(config, phase=a.phase, out_dir=a.out_dir)
            jobs = apply_study_inputs(jobs, manifest)
        except (OSError, ValueError) as exc:
            p.error(str(exc))
    print(json.dumps({"protocol": config["protocol"], "phase": a.phase,
                      "jobs": jobs, "execution": a.execute,
                      "study_manifest": str(Path(a.out_dir) / MANIFEST_NAME),
                      "input_policy": "execution freezes all configured initial checkpoints"},
                     indent=2), flush=True)
    if not a.execute:
        return 0
    for job in jobs:
        try:
            validate_study_inputs(manifest)
            check_outputs_absent([job])
            log = ROOT / job["log"]
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("x")
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        with stream:
            subprocess.run(job["command"], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plan or execute matched quality/continued-feasibility/frozen-checkpoint controls.

No SSH, background jobs or implicit final-test access. Plans are the default;
--execute runs the selected jobs sequentially and refuses missing inputs or outputs
that would be overwritten. Classical solvers occur only in comparison arms.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


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
    jobs = build_plan(config, phase=a.phase, hosts=a.hosts, features=a.features, out_dir=a.out_dir)
    if a.only:
        jobs = [job for job in jobs if job["tag"] == a.only]
        if not jobs:
            p.error("--only does not identify a selected job")
    print(json.dumps({"protocol": config["protocol"], "phase": a.phase,
                      "jobs": jobs, "execution": a.execute}, indent=2), flush=True)
    if not a.execute:
        return 0
    # Check the entire selected plan before starting a potentially long run.
    for job in jobs:
        for path in job["required"]:
            if not (ROOT / path).exists():
                p.error("required input is absent: " + path)
        for path in (job["log"], job["checkpoint"],
                     str(Path(job["checkpoint"]).with_suffix(".best.pt"))):
            if (ROOT / path).exists():
                p.error("refusing to overwrite existing artifact: " + path)
    for job in jobs:
        log = ROOT / job["log"]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as stream:
            subprocess.run(job["command"], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

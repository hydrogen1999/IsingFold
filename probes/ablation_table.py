"""Descriptive ladder results, with an explicit audit of feature comparisons.

Changing support, termination bias, initialization or budget changes the experiment.
Final rates alone cannot isolate a feature effect, even on identically named tasks.
Old logs remain readable; absent metadata is unknown, never inferred from today's
settings. No confidence interval is manufactured from two seeds.
"""
import argparse
import itertools
import json
import pathlib
import re
import statistics

RUNG = {"a": "a, 2 to 4 vars", "b": "b, 4 to 8 vars",
        "p": "Pegasus 2 fragments, 12 to 24 qubits", "z": "Zephyr 1 fragments, 12 to 24",
        "P": "Pegasus 3 fragments, 32 to 64", "Z": "Zephyr 2 fragments, 32 to 64",
        "F": "Pegasus 3 fragments, 64 to 128", "G": "Zephyr 2 fragments, 64 to 128",
        "corpus": "corpus"}

# Feature width/parameter count deliberately differ in a feature ablation. Every
# other logged training/evaluation control must match within a seed.
CONTROLS = ("actor", "width", "baseline", "learning_rate", "entropy_coef",
            "iterations", "instances_per_iteration", "episodes_per_instance",
            "eval_episodes", "max_steps", "episode_seconds", "train_episode_seconds",
            "support", "stop_bias", "init", "prefix_curriculum", "prefix_unit",
            "prefix_schedule", "prefix_shared", "prefix_empty_mix", "manifest_split",
            "heldout_role", "corpus", "cells")
DATA = ("train", "heldout", "train_sizes", "heldout_sizes")


def arm(name):
    stem = re.sub(r"_s\d+$", "", name)
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else "linear"


def read_run(path, objective="feasibility"):
    header, summary = {}, None
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "protocol" in row and "stage" in row:
            header = row
        if "summary" in row:
            summary = row
    if not summary or summary.get("objective", "feasibility") != objective:
        return None
    held = summary.get("heldout") or {}
    if "init" not in held or "final" not in held:
        return None
    return {"init": held["init"], "final": held["final"], "gain": held.get("gain"),
            "seed": summary.get("seed"), "stage": summary.get("stage", "?"),
            "features": summary.get("features", header.get("features")),
            "arm": arm(path.stem), "log": path.name, "header": header}


def control_signature(row):
    """Keep runs with different controls out of the same descriptive aggregate."""
    h = row["header"]
    # A documented null init is a cold start; a missing init is unknown.
    return json.dumps([(key, key in h, h.get(key)) for key in CONTROLS], sort_keys=True)


def comparison_reasons(left, right):
    """Reasons two runs cannot isolate a feature effect (empty means eligible).

    Matching names/sizes is necessary, not a certificate of identical graph
    content or source revision. A final comparison must pin those too.
    """
    reasons = []
    if left["stage"] != right["stage"]:
        reasons.append("different rung")
    if left["seed"] is None or left["seed"] != right["seed"]:
        reasons.append("different or missing seed")
    for key in CONTROLS + DATA:
        a, b = left["header"], right["header"]
        if key not in a or key not in b:
            reasons.append("unknown " + key)
        elif a[key] != b[key]:
            reasons.append("different " + key)
    return reasons


def descriptive_note(rows):
    seeds = [r["seed"] for r in rows]
    if None in seeds or len(set(seeds)) != len(seeds):
        return "duplicate/missing seeds; descriptive runs only; no CI"
    if len(rows) < 3:
        return "%d seed(s); descriptive only; no CI" % len(rows)
    # SD describes variation across runs, not population uncertainty.
    return "seed SD %.3f; descriptive only; no CI" % statistics.stdev(r["final"] for r in rows)


def feature_comparisons(runs):
    groups = {}
    for row in runs:
        groups.setdefault((row["stage"], row["arm"]), []).append(row)
    reports = []
    for (ka, aa), (kb, bb) in itertools.combinations(sorted(groups.items()), 2):
        if ka[0] != kb[0] or {r["features"] for r in aa} == {r["features"] for r in bb}:
            continue
        pairs = [(a, b) for a in aa for b in bb if a["seed"] == b["seed"]]
        reasons, eligible = set(), 0
        for a, b in pairs:
            problems = comparison_reasons(a, b)
            reasons.update(problems)
            eligible += not problems
        if not pairs:
            reasons.add("no shared seed")
        if any(sum(r["seed"] == seed for r in rows) > 1
               for rows in (aa, bb) for seed in {r["seed"] for r in rows}):
            reasons.add("duplicate seed; select one registered run per arm")
        if {r["seed"] for r in aa} != {r["seed"] for r in bb}:
            reasons.add("unequal seed sets; aggregate difference is unpaired")
        reports.append((ka[0], ka[1], kb[1], eligible, sorted(reasons)))
    return reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results/curriculum")
    ap.add_argument("--objective", default="feasibility")
    args = ap.parse_args(argv)
    runs = [r for p in sorted(pathlib.Path(args.logs).glob("*.log"))
            if (r := read_run(p, args.objective)) is not None]
    groups = {}
    for row in runs:
        key = (row["stage"], row["arm"], control_signature(row))
        groups.setdefault(key, []).append(row)
    print("Descriptive final rates are not evidence of a causal feature effect.")
    print("Configurations differing in logged controls are separate rows; missing controls are unknown.")
    print("  %-22s %-18s %5s %7s %7s %8s %s"
          % ("rung", "arm", "seeds", "init", "FINAL", "gain", "note"))
    for (stage, name, _), rows in sorted(groups.items()):
        avg = lambda key: statistics.mean(r[key] for r in rows)
        print("  %-22s %-18s %5d %7.2f %7.2f %8.3f %s"
              % (RUNG.get(stage, stage), name, len({r["seed"] for r in rows}),
                 avg("init"), avg("final"),
                 avg("gain") if all(r["gain"] is not None for r in rows) else float("nan"),
                 descriptive_note(rows)))
    print("Feature comparison audit (eligibility is not significance or superiority):")
    for stage, a, b, n, reasons in feature_comparisons(runs):
        note = "; ".join(reasons) if reasons else "logged controls match; pin code/data and use paired inference"
        print("  %s: %s vs %s: %d eligible seed pair(s); %s" % (stage, a, b, n, note))
    print("ABLATION TABLE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

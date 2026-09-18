"""One table from the ladder logs: what each design choice is worth, on the same rung.

The record grew a design choice at a time and the evidence for each sits in its own run, so a
reviewer cannot see the comparison the paper claims. This reads the summary record every
curriculum run emits and groups by rung, so actor, feature set, baseline and learning rate are
read off one page against the same held-out instances.

It reports only what the logs contain. A cell with fewer seeds than another is marked, because
a single seed is not a comparison and should not be printed as if it were one.
"""
import argparse, json, math, pathlib, re, statistics

# Read from constructor_curriculum.HARDWARE, which maps every uppercase stage to a FRAGMENT of
# a small host, not to a full one. An earlier version of this table called P and Z the full
# Pegasus 16 and Zephyr 15 and that was wrong: they are 32 to 64 qubit fragments of Pegasus 3 and
# Zephyr 2. The only full-host results in the record come from the ink-drop corpora.
RUNG = {"a": "a, 2 to 4 vars", "b": "b, 4 to 8 vars",
        "p": "Pegasus 2 fragments, 12 to 24 qubits", "z": "Zephyr 1 fragments, 12 to 24",
        "P": "Pegasus 3 fragments, 32 to 64", "Z": "Zephyr 2 fragments, 32 to 64",
        "F": "Pegasus 3 fragments, 64 to 128", "G": "Zephyr 2 fragments, 64 to 128",
        "corpus": "corpus"}


def arm(name):
    """The design choice a run's file name encodes, stripped of rung and seed."""
    stem = re.sub(r"_s\d+$", "", name)
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else "linear"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results/curriculum")
    ap.add_argument("--objective", default="feasibility")
    a = ap.parse_args()

    groups: dict[tuple[str, str], list[dict]] = {}
    for p in sorted(pathlib.Path(a.logs).glob("*.log")):
        summary = None
        for line in p.read_text().splitlines():
            if line.startswith("{") and '"summary"' in line:
                summary = json.loads(line)
        if not summary or summary.get("objective", "feasibility") != a.objective:
            continue
        held = summary.get("heldout") or {}
        if "init" not in held or "final" not in held:
            continue
        key = (summary.get("stage", "?"), arm(p.stem))
        groups.setdefault(key, []).append(
            {"init": held["init"], "final": held["final"], "gain": held.get("gain"),
             "seed": summary.get("seed"), "log": p.name})

    # Final rate is the comparison, not gain. Arms within a rung start from different
    # checkpoints, so a larger gain can simply mean a worse starting point; the local-feature
    # arms begin at 0.42 and 0.54 where the others begin near 0.10.
    inits: dict[str, list[float]] = {}
    for (stage, name), rows in groups.items():
        inits.setdefault(stage, []).extend(r["init"] for r in rows)

    print("  %-22s %-18s %5s %7s %7s %8s %s"
          % ("rung", "arm", "seeds", "init", "FINAL", "gain", "note"), flush=True)
    for (stage, name) in sorted(groups, key=lambda k: (k[0], -sum(r["final"] for r in k and groups[k]) / len(groups[k]))):
        rows = groups[(stage, name)]
        n = len(rows)
        mean = lambda k: sum(r[k] for r in rows) / n
        finals = [r["final"] for r in rows]
        if n > 1:
            h = 1.96 * statistics.stdev(finals) / math.sqrt(n)
            note = "final [%+.3f, %+.3f]" % (mean("final") - h, mean("final") + h)
        else:
            note = "one seed, not a comparison"
        spread = max(inits[stage]) - min(inits[stage])
        if spread > 0.15:
            note += "; inits differ by %.2f in this rung, compare FINAL not gain" % spread
        print("  %-22s %-18s %5d %7.2f %7.2f %8.3f %s"
              % (RUNG.get(stage, stage), name, n, mean("init"), mean("final"),
                 mean("gain") if all(r["gain"] is not None for r in rows) else float("nan"),
                 note), flush=True)
    print("ABLATION TABLE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

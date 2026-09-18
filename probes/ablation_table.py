"""One table from the ladder logs: what each design choice is worth, on the same rung.

The record grew a design choice at a time and the evidence for each sits in its own run, so a
reviewer cannot see the comparison the paper claims. This reads the summary record every
curriculum run emits and groups by rung, so actor, feature set, baseline and learning rate are
read off one page against the same held-out instances.

It reports only what the logs contain. A cell with fewer seeds than another is marked, because
a single seed is not a comparison and should not be printed as if it were one.
"""
import argparse, json, math, pathlib, re, statistics

RUNG = {"a": "a, 2 to 4 vars", "b": "b, 4 to 8 vars", "p": "Pegasus 2 fragments",
        "z": "Zephyr 1 fragments", "P": "Pegasus 16 full", "Z": "Zephyr 15 full",
        "F": "F, 20 vars", "G": "G, 24 vars", "corpus": "corpus"}


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

    print("  %-22s %-18s %5s %7s %7s %8s %s"
          % ("rung", "arm", "seeds", "init", "final", "gain", "note"), flush=True)
    for (stage, name) in sorted(groups, key=lambda k: (k[0], k[1])):
        rows = groups[(stage, name)]
        n = len(rows)
        mean = lambda k: sum(r[k] for r in rows) / n
        gains = [r["gain"] for r in rows if r["gain"] is not None]
        if len(gains) > 1:
            sd = statistics.stdev(gains)
            h = 1.96 * sd / math.sqrt(len(gains))
            note = "[%+.3f, %+.3f]" % (sum(gains) / len(gains) - h, sum(gains) / len(gains) + h)
        else:
            note = "one seed, not a comparison"
        print("  %-22s %-18s %5d %7.2f %7.2f %8.3f %s"
              % (RUNG.get(stage, stage), name, n, mean("init"), mean("final"),
                 mean("gain") if gains else float("nan"), note), flush=True)
    print("ABLATION TABLE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

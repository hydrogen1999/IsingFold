"""Tidy per-point data for every table and figure, extracted from the logs themselves.

The results document carries means and intervals, which is what a reader needs and what a plot
cannot be drawn from. A figure needs the points: one row per instance, per chain, per branch
point, per evaluation. This reads the committed logs and writes one tidy file per figure, in long
format with one observation a row, so a plot is a group-by rather than a reshape.

Nothing is recomputed from memory. Every value here is read out of a log under results/, and the
manifest records which log each file came from, so a figure can be traced back to the run that
produced it.
"""
import argparse, csv, json, math, pathlib, re, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_json_lines(path, needle=None):
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("{") and (needle is None or needle in line):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def write(name, rows, fields, sources, manifest):
    if not rows:
        manifest.append({"file": name, "rows": 0, "skipped": "no data found", "sources": sources})
        return
    path = ROOT / "data" / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    manifest.append({"file": name, "rows": len(rows), "columns": fields, "sources": sources})
    print("  %-38s %5d rows" % (name, len(rows)), flush=True)


def spearman(a, b):
    n = len(a)
    if n < 3:
        return None
    def rank(xs):
        o = sorted(range(n), key=lambda i: xs[i]); r = [0.] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[o[j + 1]] == xs[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2. + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b); ma, mb = sum(ra) / n, sum(rb) / n
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return None if den == 0 else sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den


def selection_signals(manifest):
    """One row per instance and rule: how well that rule ordered the pool it was given."""
    rows, sources = [], []
    for host, path in (("pegasus6", "results/rules/pegasus6.log"),
                       ("zephyr4", "results/rules/zephyr4.log")):
        p = ROOT / path
        if not p.exists():
            continue
        sources.append(path)
        for r in read_json_lines(p, '"assess"'):
            rules = {"fewest qubits": [-q for q in r["qubits"]],
                     "shortest chain": [-c for c in r["chain"]],
                     "256-read measurement": r["select"]}
            for rule, x in rules.items():
                v = spearman(x, r["assess"])
                if v is None:
                    continue
                rows.append({"host": host, "task": r["task"], "rule": rule, "rho": v,
                             "draws": r["draws"], "qubit_spread": max(r["qubits"]) - min(r["qubits"]),
                             "assessed_mean": sum(r["assess"]) / len(r["assess"])})
    write("fig_selection_signal_rho.csv", rows,
          ["host", "task", "rule", "rho", "draws", "qubit_spread", "assessed_mean"],
          sources, manifest)


def solvability(manifest):
    """One row per instance, cell and anneal depth across the frontier sweep."""
    rows, sources = [], []
    for p in sorted((ROOT / "results" / "frontier").glob("*.log")):
        for r in read_json_lines(p, '"p_best"'):
            if "vars" not in r:
                continue
            sources.append("results/frontier/" + p.name)
            rows.append({"host": r["host"], "variables": r["vars"], "qubits": r["qubits"],
                         "named_fill": round(r["realised_fill"], 3),
                         "mean_chain": round(r["mean_chain"], 3), "field_rate": r["field_rate"],
                         "sweeps": r["sweeps"], "reads": r["reads"],
                         "p_registered": r["p_registered"], "p_best": r["p_best"],
                         "minorminer_valid": int(bool(r["mm_valid"])),
                         "p_minorminer": r.get("p_minorminer"),
                         "source": p.name})
    write("fig_solvability_frontier.csv", rows,
          ["host", "variables", "qubits", "named_fill", "mean_chain", "field_rate", "sweeps",
           "reads", "p_registered", "p_best", "minorminer_valid", "p_minorminer", "source"],
          sorted(set(sources)), manifest)


def chain_breaking(manifest):
    """One row per chain: its structure and the break rate measured for it."""
    rows, sources = [], []
    for name in ("breakpred2_p3.log",):
        p = ROOT / "results" / "quality" / name
        if not p.exists():
            continue
        sources.append("results/quality/" + name)
        for r in read_json_lines(p, '"chain"'):
            c = r["chain"]
            rows.append({k: c.get(k) for k in
                         ("task_index", "source", "break_rate", "length", "internal_edges",
                          "redundancy", "cut_vertices", "logical_degree", "coupling_mass",
                          "contacts", "mass_per_contact", "max_contact_load",
                          "load_concentration", "field")})
    write("fig_chain_breaking.csv", rows,
          ["task_index", "source", "break_rate", "length", "internal_edges", "redundancy",
           "cut_vertices", "logical_degree", "coupling_mass", "contacts", "mass_per_contact",
           "max_contact_load", "load_concentration", "field"], sources, manifest)


def embedding_quality(manifest):
    """One row per embedding: its shape, its breaking and its measured quality."""
    rows, sources = [], []
    for name in ("broken_pre_p3.log", "broken_clone_local.log", "broken_clone_physics.log",
                 "excess_p3_f30.log", "excess_clone_p3.log", "excess20_pre_p3.log",
                 "excess20_clone_p3.log"):
        p = ROOT / "results" / "quality" / name
        if not p.exists():
            continue
        sources.append("results/quality/" + name)
        for r in read_json_lines(p, '"removable_fraction"'):
            for arm in ("policy", "pruned", "minorminer"):
                a = r.get(arm)
                if not a:
                    continue
                rows.append({"source": name, "task": r["task"], "variables": r["variables"],
                             "arm": arm, "qubits": a.get("qubits"), "longest": a.get("longest"),
                             "residual": a.get("residual"), "p_solve": a.get("p_solve"),
                             "broken": a.get("broken")})
    write("fig_embedding_quality.csv", rows,
          ["source", "task", "variables", "arm", "qubits", "longest", "residual", "p_solve",
           "broken"], sources, manifest)


def branch_points(manifest):
    """One row per branch point and action: the residual that action led to."""
    rows, sources = [], []
    for p in sorted((ROOT / "results" / "quality").glob("branch*.jsonl")):
        sources.append("results/quality/" + p.name)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            measured = {int(k): v for k, v in r["residual_by_action"].items()}
            best = min(measured, key=measured.get)
            for action, value in sorted(measured.items()):
                rows.append({"source": p.name, "task": r["task"], "depth": r["depth"],
                             "action": action, "residual": value,
                             "is_policy_pick": int(action == r["policy_pick"]),
                             "is_measured_best": int(action == best),
                             "spread": max(measured.values()) - min(measured.values())})
    write("fig_branch_points.csv", rows,
          ["source", "task", "depth", "action", "residual", "is_policy_pick", "is_measured_best",
           "spread"], sources, manifest)


def learning_curves(manifest):
    """One row per evaluation of every training run: the held-out rate at that point."""
    rows, sources = [], []
    for sub, tag in (("quality", "quality-cell ladder"), ("inkdrop", "hardware scale"),
                     ("curriculum", "curriculum"), ("reverse", "reverse start")):
        d = ROOT / "results" / sub
        if not d.exists():
            continue
        for p in sorted(d.glob("*.log")):
            evs = [r for r in read_json_lines(p, '"evaluation"') if r.get("set") == "heldout"]
            if not evs:
                continue
            sources.append("results/%s/%s" % (sub, p.name))
            for order, r in enumerate(evs):
                rows.append({"group": tag, "run": p.stem, "order": order,
                             "evaluation": r.get("evaluation"), "rate": r.get("rate"),
                             "instances": r.get("instances")})
    write("fig_learning_curves.csv", rows,
          ["group", "run", "order", "evaluation", "rate", "instances"],
          sorted(set(sources))[:40], manifest)


def paired_study(manifest):
    """One row per arm and instance from the three-arm quality study."""
    rows, sources = [], []
    d = ROOT / "results" / "quality_study"
    for p in sorted(d.glob("*_s0.log")) if d.exists() else []:
        arm = re.sub(r"^.*?_(frozen|continued_feasibility|quality)_s0$", r"\1", p.stem)
        evs = read_json_lines(p, '"evaluation"')
        if not evs:
            continue
        sources.append("results/quality_study/" + p.name)
        d_last = evs[-1]
        for r in d_last.get("rows", []):
            pol = r.get("policy") or {}
            rows.append({"study": p.stem.split("_")[0], "schema": p.stem.split("_")[1],
                         "arm": arm, "task": r.get("task"),
                         "policy_residual": pol.get("residual", r.get("residual")),
                         "policy_p_solve": pol.get("p_solve"),
                         "minorminer_residual": (r.get("minorminer") or {}).get("residual"),
                         "evaluation": d_last.get("evaluation")})
    write("fig_paired_quality_study.csv", rows,
          ["study", "schema", "arm", "task", "policy_residual", "policy_p_solve",
           "minorminer_residual", "evaluation"], sources, manifest)


def congestion(manifest):
    """One row per cell: how often an anytime router succeeds at each deadline."""
    rows, sources = [], []
    for host, name in (("pegasus3", "anytime_pegasus3_300.log"),
                       ("zephyr2", "anytime_zephyr2_300.log")):
        p = ROOT / "results" / "control" / name
        if not p.exists():
            continue
        sources.append("results/control/" + name)
        for line in p.read_text().splitlines():
            m = re.match(r"\s+fill(\d+)-a([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                rows.append({"host": host, "fill": int(m.group(1)) / 100.0,
                             "chain_shape": "short" if m.group(2) == "3.0" else "long",
                             "instances": int(m.group(3)), "valid_at_60s": float(m.group(4)),
                             "valid_at_300s": float(m.group(5)), "mean_attempts": float(m.group(6))})
    write("fig_congestion_wallclock.csv", rows,
          ["host", "fill", "chain_shape", "instances", "valid_at_60s", "valid_at_300s",
           "mean_attempts"], sources, manifest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/figures")
    a = ap.parse_args()
    manifest = []
    print("writing tidy per-point data under %s\n" % a.out, flush=True)
    for fn in (selection_signals, solvability, chain_breaking, embedding_quality, branch_points,
               learning_curves, paired_study, congestion):
        fn(manifest)
    path = ROOT / a.out / "MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"files": manifest}, indent=1) + "\n")
    total = sum(m.get("rows", 0) for m in manifest)
    print("\n  %d files, %d rows, manifest at %s" % (len(manifest), total, path.relative_to(ROOT)),
          flush=True)
    print("EXPORT DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

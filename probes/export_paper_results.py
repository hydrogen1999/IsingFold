"""Emit the paper's experimental numbers as JSON, one file per table or figure.

Numbers only. Every record carries the log it came from so a table in the paper can be
regenerated without reading prose. Values that were computed by an earlier probe are
transcribed with their source; values that can be recomputed from the raw logs are
recomputed here and the recomputation is what ships.

Run from the project root:  python3 probes/export_paper_results.py
"""

import csv
import json
import os
import random
import statistics as st

OUT = "docs/paper/results"
FIGDATA = "data/figures"


# ---------------------------------------------------------------- helpers

def jsonl(path, want=None):
    """Every JSON object on its own line in a log, optionally filtered by a required key."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if want is None or all(k in d for k in want):
                out.append(d)
    return out


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rank(a), rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def boot_ci(values, reps=10000, seed=0):
    """Percentile bootstrap of the mean over the independent experimental unit."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(st.mean(rng.choice(values) for _ in range(n)) for _ in range(reps))
    return [round(means[int(0.025 * reps)], 6), round(means[int(0.975 * reps) - 1], 6)]


def front(points, resource, quality, higher_is_better):
    """Indices of the non-dominated points: least resource, best quality."""
    keep = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            cheaper = q[resource] <= p[resource]
            better = (q[quality] >= p[quality]) if higher_is_better else (q[quality] <= p[quality])
            strict = q[resource] < p[resource] or (
                q[quality] > p[quality] if higher_is_better else q[quality] < p[quality])
            if cheaper and better and strict:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def write(name, payload):
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path, os.path.getsize(path)


# ------------------------------------------------- F3: resource-quality Pareto

def pareto():
    """Three populations whose resource axis spans 2, 8-18 and 30 qubits.

    The figure asks whether quality follows resource. Each population is a different
    way of moving qubits: the router's own sampling noise, a controlled absorption of
    free qubits, and a change of method. Quality is solve probability, assessed on a
    read block disjoint from any block used to select.
    """
    populations = []

    # P1: the deployed router's own eight draws. results/rules/*.log
    pts, rhos = [], []
    for host, path in (("pegasus6", "results/rules/pegasus6.log"),
                       ("zephyr4", "results/rules/zephyr4.log")):
        for d in jsonl(path, want=("task", "qubits", "assess")):
            q, a = d["qubits"], d["assess"]
            if len(q) != len(a) or len(q) < 3:
                continue
            idx = front([{"q": x, "v": y} for x, y in zip(q, a)], "q", "v", True)
            for k, (x, y) in enumerate(zip(q, a)):
                pts.append({"host": host, "task": d["task"], "draw": k, "qubits": x,
                            "p_solve": round(y, 6),
                            "p_solve_selection_block": round(d["select"][k], 6),
                            "on_front": k in idx})
            r = spearman(q, a)
            rhos.append({"host": host, "task": d["task"],
                         "rho": None if r is None else round(r, 6),
                         "qubit_spread": max(q) - min(q), "front_size": len(idx)})
    populations.append({
        "id": "router_draws",
        "how_resource_moves": "sampling noise of twenty-try minorminer, eight draws an instance",
        "instances": len(rhos), "points": len(pts),
        "qubit_spread": {"min": min(r["qubit_spread"] for r in rhos),
                         "max": max(r["qubit_spread"] for r in rhos),
                         "mean": round(st.mean(r["qubit_spread"] for r in rhos), 3)},
        "rho_qubits_vs_p_solve": {
            "defined_on": sum(1 for r in rhos if r["rho"] is not None),
            "undefined_constant_resource": sum(1 for r in rhos if r["rho"] is None),
            "mean": round(st.mean(r["rho"] for r in rhos if r["rho"] is not None), 6),
            "ci95": boot_ci([r["rho"] for r in rhos if r["rho"] is not None], seed=1),
            "negative_fraction": round(
                sum(1 for r in rhos if r["rho"] is not None and r["rho"] < 0)
                / max(1, sum(1 for r in rhos if r["rho"] is not None)), 3)},
        "front_size": {"mean": round(st.mean(r["front_size"] for r in rhos), 3),
                       "of_draws": 8},
        "per_instance": rhos, "measurements": pts,
        "source": ["results/rules/pegasus6.log", "results/rules/zephyr4.log"],
        "sign_convention": "p_solve higher is better, qubits lower is better"})

    # P2: free qubits absorbed into random chains. results/quality/resource_rho_*.log
    for channel, higher in (("solve probability", True), ("residual", False)):
        pts, rhos = [], []
        for host, path in (("pegasus3", "results/quality/resource_rho_pegasus3.log"),
                           ("zephyr2", "results/quality/resource_rho_zephyr2.log")):
            for d in jsonl(path, want=("task", "qubits", "independent_blocks")):
                q = d["qubits"]
                blocks = d["independent_blocks"][channel]
                v = [st.mean(b[i] for b in blocks) for i in range(len(q))]
                se = [st.stdev(b[i] for b in blocks) / len(blocks) ** 0.5 for i in range(len(q))]
                idx = front([{"q": x, "v": y} for x, y in zip(q, v)], "q", "v", higher)
                for k in range(len(q)):
                    pts.append({"host": host, "task": d["task"], "embedding": k,
                                "qubits": q[k], "value": round(v[k], 6),
                                "se": round(se[k], 6), "blocks": len(blocks),
                                "on_front": k in idx})
                r = spearman(q, v)
                rhos.append({"host": host, "task": d["task"],
                             "rho": None if r is None else round(r, 6),
                             "qubit_spread": max(q) - min(q), "front_size": len(idx)})
        # Is anything in this pool resolvable at all? Compare the spread between
        # embeddings against the standard error of each embedding's own mean.
        bytask = {}
        for m in pts:
            bytask.setdefault(m["task"], []).append(m)
        spread = st.mean(max(x["value"] for x in v) - min(x["value"] for x in v)
                         for v in bytask.values())
        se = st.mean(st.mean(x["se"] for x in v) for v in bytask.values())
        populations.append({
            "id": "absorbed_growth_" + channel.replace(" ", "_"),
            "resolution": {
                "within_instance_spread": round(spread, 6),
                "standard_error_of_an_embedding_mean": round(se, 6),
                "spread_over_se": round(spread / se, 3),
                "verdict": ("resolvable" if spread / se >= 3 else
                            "not resolvable at this read budget"),
                "reading": ("a null here bounds the effect below the detection floor; "
                            "it does not establish that the added qubits change nothing")},
            "how_resource_moves": "free qubits absorbed into random chains, every contact preserved",
            "channel": channel, "instances": len(rhos), "points": len(pts),
            "qubit_spread": {"min": min(r["qubit_spread"] for r in rhos),
                             "max": max(r["qubit_spread"] for r in rhos),
                             "mean": round(st.mean(r["qubit_spread"] for r in rhos), 3)},
            "rho_qubits_vs_quality": {
                "defined_on": sum(1 for r in rhos if r["rho"] is not None),
                "undefined_constant_resource": sum(1 for r in rhos if r["rho"] is None),
                "mean": round(st.mean(r["rho"] for r in rhos if r["rho"] is not None), 6),
                "ci95": boot_ci([r["rho"] for r in rhos if r["rho"] is not None], seed=2),
                "per_instance_range": [
                    round(min(r["rho"] for r in rhos if r["rho"] is not None), 6),
                    round(max(r["rho"] for r in rhos if r["rho"] is not None), 6)]},
            "front_size": {"mean": round(st.mean(r["front_size"] for r in rhos), 3),
                           "of_embeddings": 8},
            "per_instance": rhos, "measurements": pts,
            "source": ["results/quality/resource_rho_pegasus3.log",
                       "results/quality/resource_rho_zephyr2.log"],
            "sign_convention": ("higher is better" if higher else "lower is better")})

    # P3: a change of method. data/figures/fig_embedding_quality.csv
    rows = list(csv.DictReader(open(os.path.join(FIGDATA, "fig_embedding_quality.csv"))))
    by_arm, pts = {}, []
    for r in rows:
        rec = {"source": r["source"], "task": r["task"], "variables": int(r["variables"]),
               "arm": r["arm"], "qubits": float(r["qubits"]),
               "longest": float(r["longest"]) if r["longest"] not in ("", "None") else None,
               "residual": float(r["residual"]), "p_solve": float(r["p_solve"]),
               "broken": float(r["broken"]) if r["broken"] not in ("", "None") else None}
        pts.append(rec)
        by_arm.setdefault(r["arm"], []).append(rec)
    arms = []
    for a, v in sorted(by_arm.items()):
        arms.append({"arm": a, "n": len(v),
                     "qubits": round(st.mean(x["qubits"] for x in v), 3),
                     "qubits_ci95": boot_ci([x["qubits"] for x in v], seed=3),
                     "p_solve": round(st.mean(x["p_solve"] for x in v), 6),
                     "p_solve_ci95": boot_ci([x["p_solve"] for x in v], seed=4),
                     "residual": round(st.mean(x["residual"] for x in v), 6),
                     "residual_ci95": boot_ci([x["residual"] for x in v], seed=5)})
    # paired, per task, so the arms are compared on the same instances
    paired = {}
    for r in pts:
        paired.setdefault((r["source"], r["task"]), {})[r["arm"]] = r
    def delta(a, b, key):
        d = [p[b][key] - p[a][key] for p in paired.values() if a in p and b in p]
        return {"n": len(d), "mean": round(st.mean(d), 6), "ci95": boot_ci(d, seed=6)}
    populations.append({
        "id": "across_methods",
        "how_resource_moves": "a different construction: the router, the learned policy, the policy pruned",
        "instances": len(paired), "points": len(pts),
        "arms": arms,
        "paired_differences": {
            "policy_minus_minorminer": {k: delta("minorminer", "policy", k)
                                        for k in ("qubits", "p_solve", "residual")},
            "pruned_minus_policy": {k: delta("policy", "pruned", k)
                                    for k in ("qubits", "p_solve", "residual")}},
        "measurements": pts,
        "source": [os.path.join(FIGDATA, "fig_embedding_quality.csv")],
        "sign_convention": "p_solve higher is better, residual lower is better, qubits lower is better"})

    return {
        "figure": "F2",
        "title": "resource-quality Pareto",
        "question": "does spending qubits buy quality",
        "resource_axis": "physical qubits used by the embedding",
        "quality_axis": ["solve probability", "normalized energy residual"],
        "protocol": {"sampler": "classical simulated annealing",
                     "beta_range": [0.1, 2.0], "sweeps": 200,
                     "selection_and_assessment": "disjoint read blocks"},
        "populations": populations}


def fig(name):
    return list(csv.DictReader(open(os.path.join(FIGDATA, name))))


def num(x):
    return None if x in ("", "None", None) else float(x)


# --------------------------------------------------- T1: selection signal

def t1_selection_signal():
    rows = fig("fig_selection_signal_rho.csv")
    out = {}
    for r in rows:
        key = (r["host"], r["rule"])
        out.setdefault(key, []).append(r)
    table = []
    for (host, rule), v in sorted(out.items()):
        rhos = [num(x["rho"]) for x in v if num(x["rho"]) is not None]
        if not rhos:
            continue
        table.append({"host": host, "rule": rule, "instances": len(v),
                      "defined_on": len(rhos),
                      "rho": round(st.mean(rhos), 6), "ci95": boot_ci(rhos, seed=11),
                      "qubit_spread": round(st.mean(num(x["qubit_spread"]) for x in v
                                                    if num(x["qubit_spread"]) is not None), 4)})
    return {"table": "T1", "claim": "a short measurement orders candidates better than qubit count",
            "unit": "within-instance Spearman against a disjoint assessment block",
            "recomputed_from": os.path.join(FIGDATA, "fig_selection_signal_rho.csv"),
            "source": ["results/rules/pegasus6.log", "results/rules/zephyr4.log"],
            "rows": table,
            "utility_gain_over_first_draw": {
                "derivation": "transcribed", "source": "results/signal/ranking.log",
                "unit": "solve probability gained over the first draw, eight draws an instance",
                "pegasus6": {"measured": 0.147, "fewest_qubits": 0.042, "oracle": 0.163},
                "zephyr4": {"measured": 0.153, "fewest_qubits": 0.041, "oracle": 0.163},
                "instances_per_host": 30}}


# ------------------------------------------- T2: what decides quality

def t2_quality_mechanism():
    rows = fig("fig_chain_breaking.csv")
    groups = {}
    for r in rows:
        groups.setdefault(r["source"], []).append(r)
    fields = ["length", "internal_edges", "redundancy", "cut_vertices", "logical_degree",
              "coupling_mass", "contacts", "mass_per_contact", "max_contact_load",
              "load_concentration"]
    # within instance, multi-qubit chains only: a singleton cannot break
    by_task = {}
    for r in rows:
        if num(r["length"]) and num(r["length"]) > 1:
            by_task.setdefault(r["task_index"], []).append(r)
    within = {}
    for f in fields:
        rs = []
        for v in by_task.values():
            a = [num(x[f]) for x in v]
            b = [num(x["break_rate"]) for x in v]
            if any(z is None for z in a + b):
                continue
            rho = spearman(a, b)
            if rho is not None:
                rs.append(rho)
        if rs:
            within[f] = {"instances": len(rs), "rho": round(st.mean(rs), 6),
                         "ci95": boot_ci(rs, seed=12)}
    # holding length fixed, inside each instance-and-length group of at least six chains
    at_length = {}
    for f in fields:
        rs = []
        for v in by_task.values():
            bylen = {}
            for x in v:
                bylen.setdefault(num(x["length"]), []).append(x)
            for g in bylen.values():
                if len(g) < 6:
                    continue
                a = [num(x[f]) for x in g]
                b = [num(x["break_rate"]) for x in g]
                if any(z is None for z in a + b):
                    continue
                rho = spearman(a, b)
                if rho is not None:
                    rs.append(rho)
        if rs:
            at_length[f] = {"groups": len(rs), "rho": round(st.mean(rs), 6),
                            "ci95": boot_ci(rs, seed=13)}
    return {"table": "T2",
            "claim": "chain breaking decides quality, and at a fixed length redundancy decides breaking",
            "chains_total": len(rows),
            "chains_multi_qubit": sum(len(v) for v in by_task.values()),
            "singleton_fraction": round(1 - sum(len(v) for v in by_task.values()) / len(rows), 4),
            "singletons_excluded_because": "a one-qubit chain cannot break",
            "recomputed_from": os.path.join(FIGDATA, "fig_chain_breaking.csv"),
            "source": ["results/quality/breakpred2_p3.log"],
            "within_instance_vs_break_rate": within,
            "at_fixed_length_vs_break_rate": at_length,
            "residual_predictors": {
                "derivation": "transcribed", "source": "results/quality/broken_*.log",
                "unit": "within-instance Spearman with residual", "embeddings_per_instance": 9,
                "instances": 20,
                "broken_chain_fraction": {"rho": 0.728, "ci95": [0.568, 0.888]},
                "qubit_count": {"rho": 0.439, "ci95": [0.303, 0.576]},
                "broken_with_qubits": {"rho": 0.686, "ci95": [0.602, 0.769]}}}


# ------------------------------------------- T6: the quality arms

def t6_quality_arms():
    rows = fig("fig_paired_quality_study.csv")
    arms = {}
    for r in rows:
        arms.setdefault((r["study"], r["schema"], r["arm"]), []).append(r)
    table = []
    for (study, schema, arm), v in sorted(arms.items()):
        res = [num(x["policy_residual"]) for x in v if num(x["policy_residual"]) is not None]
        ps = [num(x["policy_p_solve"]) for x in v if num(x["policy_p_solve"]) is not None]
        table.append({"study": study, "schema": schema, "arm": arm, "n": len(v),
                      "residual": round(st.mean(res), 6) if res else None,
                      "residual_ci95": boot_ci(res, seed=14),
                      "p_solve": round(st.mean(ps), 6) if ps else None})
    return {"table": "T6",
            "claim": "training improves what the constructor builds; rewarding quality does not improve it further",
            "recomputed_from": os.path.join(FIGDATA, "fig_paired_quality_study.csv"),
            "source": ["results/quality_study/pegasus3_physics_*_s0.log"],
            "arms": table,
            "headline": {
                "derivation": "transcribed", "instances": 30, "updates": 40, "seed": 1,
                "deployment_seconds": 300,
                "per_arm": [
                    {"arm": "frozen", "residual": 0.0647, "p_solve": 0.354, "qubits": 67.4,
                     "longest_chain": 6.5, "valid": 0.967},
                    {"arm": "feasibility_reward", "residual": 0.0557, "p_solve": 0.402,
                     "qubits": 68.8, "longest_chain": 6.6, "valid": 1.000},
                    {"arm": "quality_reward", "residual": 0.0590, "p_solve": 0.373,
                     "qubits": 68.9, "longest_chain": 6.1, "valid": 1.000},
                    {"arm": "minorminer", "residual": 0.0289, "p_solve": 0.537, "qubits": 29.9,
                     "longest_chain": 1.7, "valid": 1.000}],
                "paired_differences": [
                    {"name": "feasibility_minus_quality", "value": -0.0033,
                     "ci95": [-0.0084, 0.0018], "registered_kill_threshold_upper": 0.002,
                     "verdict": "configuration dead",
                     "quality_better_on_instances": 10, "of": 30},
                    {"name": "quality_minus_frozen", "value": 0.0060, "ci95": [-0.0025, 0.0145]},
                    {"name": "feasibility_minus_frozen", "value": 0.0094,
                     "ci95": [0.0025, 0.0163], "verdict": "clear of zero"}]}}


# ------------------------------------------- T7: the branch decomposition

def t7_branch():
    rows = fig("fig_branch_points.csv")
    pts = {}
    for r in rows:
        pts.setdefault((r["source"], r["task"], r["depth"]), []).append(r)
    spreads, hits = [], []
    for v in pts.values():
        res = [num(x["residual"]) for x in v if num(x["residual"]) is not None]
        if len(res) < 2:
            continue
        spreads.append(max(res) - min(res))
        pick = [x for x in v if x["is_policy_pick"] in ("True", "true", "1")]
        best = [x for x in v if x["is_measured_best"] in ("True", "true", "1")]
        if pick and best:
            hits.append(1.0 if pick[0] is best[0] or
                        num(pick[0]["residual"]) == num(best[0]["residual"]) else 0.0)
    return {"table": "T7",
            "claim": "one decision carries about twice the whole performance gap, and the policy chooses at chance",
            "recomputed_from": os.path.join(FIGDATA, "fig_branch_points.csv"),
            "source": ["results/quality/branch_all.jsonl", "results/quality/branch_p3.log"],
            "branch_points": len(spreads),
            "residual_spread_across_four_actions": {
                "mean": round(st.mean(spreads), 6), "ci95": boot_ci(spreads, seed=15)},
            "policy_pick_was_best": {"n": len(hits),
                                     "rate": round(st.mean(hits), 6) if hits else None,
                                     "ci95": boot_ci(hits, seed=16), "chance": 0.25},
            "policy_minus_minorminer_gap": {"derivation": "transcribed", "value": 0.0268},
            "supervised_on_measured_orderings": {
                "derivation": "transcribed", "source": "probes/branch_learnable.py",
                "folds": 5, "grouped_by": "instance", "chance": 0.250,
                "linear": {"held_out": 0.257, "fitted": 0.321},
                "small_network": {"held_out": 0.290, "fitted": 0.692,
                                  "standard_errors_from_chance": 2.1}}}


# ------------------------------------------- T9: congestion under a wall clock

def t9_congestion():
    rows = fig("fig_congestion_wallclock.csv")
    return {"table": "T9",
            "claim": "the shape of the target embedding defeats the router, not its occupancy",
            "protocol": {"router": "minorminer restarting until a shared deadline",
                         "deadline_seconds": 300, "instances_per_cell": 12},
            "recomputed_from": os.path.join(FIGDATA, "fig_congestion_wallclock.csv"),
            "source": ["results/control/anytime_pegasus3_300.log",
                       "results/control/anytime_zephyr2_300.log"],
            "cells": [{"host": r["host"], "fill": num(r["fill"]),
                       "chain_shape": r["chain_shape"], "instances": int(r["instances"]),
                       "valid_at_60s": num(r["valid_at_60s"]),
                       "valid_at_300s": num(r["valid_at_300s"]),
                       "mean_attempts": num(r["mean_attempts"])} for r in rows]}


# ------------------------------------------- T15: where solve probability is measurable

def t15_solvability():
    rows = fig("fig_solvability_frontier.csv")
    return {"table": "T15",
            "claim": "solve probability decays close to exponentially in the variable count and depth does not change the slope",
            "recomputed_from": os.path.join(FIGDATA, "fig_solvability_frontier.csv"),
            "source": ["results/frontier/*.log"],
            "decay": {"derivation": "transcribed", "unit": "slope of log solve probability per variable",
                      "fitted_on": "cells that produced a resolvable rate",
                      "by_sweeps": [{"sweeps": 200, "slope": -0.0499, "variables_at_0.05": 66},
                                    {"sweeps": 2000, "slope": -0.0383, "variables_at_0.05": 83},
                                    {"sweeps": 20000, "slope": -0.0457, "variables_at_0.05": 80}]},
            "depth_stops_paying": {"variables": 94, "p_solve_at_20000_sweeps": 0.084,
                                   "p_solve_at_200000_sweeps": 0.082},
            "local_field_knob": {"nodes_fraction_max": 0.5, "p_solve": 0.0,
                                 "from_variables": 195},
            "residual_still_separates": {"named_fill": 0.50, "perturbation_qubits": 64,
                                         "variables": 274, "effect": 0.0047,
                                         "ci95": [0.0022, 0.0073],
                                         "p_solve_there": 0.0},
            "cells": [{"host": r["host"], "variables": int(r["variables"]),
                       "qubits": num(r["qubits"]), "named_fill": num(r["named_fill"]),
                       "mean_chain": num(r["mean_chain"]), "field_rate": num(r["field_rate"]),
                       "sweeps": num(r["sweeps"]), "reads": num(r["reads"]),
                       "p_registered": num(r["p_registered"]),
                       "minorminer_valid": num(r["minorminer_valid"]),
                       "p_minorminer": num(r["p_minorminer"])} for r in rows]}


# ------------------------------------------- T13: ablations and learning curves

def t13_ablations():
    rows = fig("fig_learning_curves.csv")
    runs = {}
    for r in rows:
        runs.setdefault((r["group"], r["run"]), []).append(r)
    curves = []
    for (group, run), v in sorted(runs.items()):
        v.sort(key=lambda x: int(x["order"]))
        rates = [num(x["rate"]) for x in v]
        known = [x for x in rates if x is not None]
        if not known:
            continue
        curves.append({"group": group, "run": run, "evaluations": len(v),
                       "initial": known[0], "final": known[-1],
                       "best": max(known), "series": rates})
    return {"table": "T13",
            "claim": "four local-capacity channels buy what two hundred and fourteen more channels buy",
            "reported_as": "final held-out rate, not gain: arms within a rung start from different checkpoints",
            "recomputed_from": os.path.join(FIGDATA, "fig_learning_curves.csv"),
            "source": ["results/ablation/feasibility.log"],
            "curves": curves,
            "summary": {"derivation": "transcribed",
                        "rows": [
                            {"rung": "zephyr2 fragments 64-128 qubits", "observation": "20 channels",
                             "seeds": 2, "final": 0.96, "ci95": [0.931, 0.986]},
                            {"rung": "zephyr2 fragments 64-128 qubits", "observation": "230 channels",
                             "seeds": 1, "final": 0.94},
                            {"rung": "zephyr2 fragments 64-128 qubits", "observation": "16 channels",
                             "seeds": 1, "final": 0.88},
                            {"rung": "pegasus3 fragments 64-128 qubits", "observation": "20 channels",
                             "seeds": 2, "final": 0.83},
                            {"rung": "pegasus3 fragments 64-128 qubits", "observation": "16 channels",
                             "seeds": 1, "final": 0.80}],
                        "rung_b_gains_three_seeds_common_start_0.13": [
                            {"arm": "230 channels", "gain": 0.760, "ci95": [0.696, 0.823]},
                            {"arm": "16 channels", "gain": 0.607, "ci95": [0.566, 0.648]},
                            {"arm": "contextual actor, leave-one-out baseline", "gain": 0.388,
                             "ci95": [-0.093, 0.868]},
                            {"arm": "contextual actor, value baseline", "gain": 0.257,
                             "ci95": [-0.112, 0.626]}],
                        "step_cost_seconds": {"20 channels": 0.087, "230 channels": 0.74}}}


# ------------------------------------------- transcribed tables

def transcribed():
    return {
        "t3_resource_interventions.json": {
            "table": "T3",
            "claim": "two independent interventions on the resource axis, both effective on resources, both inert on quality",
            "source": ["results/quality/excess_*.log", "results/quality/excess20_*_p3.log"],
            "interventions": [
                {"name": "greedy pruning, every contact preserved", "qubits_change": -0.34,
                 "broken_change": -0.12, "residual_recovered": 0.0022, "of_gap": 0.080},
                {"name": "cloning the witness construction, 250 epochs", "qubits_change": -0.42,
                 "broken_change": -0.18, "residual_change": -0.0200,
                 "ci95": [-0.0428, 0.0028], "instances": 20,
                 "note": "reversed sign between an eight-instance sample and a twenty-instance replication"}]},

        "t4_feasibility_ladder.json": {
            "table": "T4",
            "claim": "held-out feasibility from an empty host, minorminer forbidden after generation",
            "protocol": {"evaluation_start": "empty", "instances": "held out, unseen"},
            "source": ["results/inkdrop/ink24_*.log", "results/quality/r30_*.log",
                       "results/quality/r39_*.log", "results/curriculum/w[23]_f9*.log"],
            "rungs": [
                {"cell": "ink-drop, full Pegasus 16", "host_qubits": 5640, "variables": 24,
                 "fill": 0.02, "initial": 0.10, "final": 1.00, "seeds": 1},
                {"cell": "ink-drop, full Zephyr 15", "host_qubits": 7440, "variables": 24,
                 "fill": 0.02, "initial": 0.35, "final": 0.95, "seeds": 1},
                {"cell": "Pegasus 3 and Zephyr 2", "variables": 28, "fill": 0.30,
                 "initial_range": [0.75, 1.00], "final": 1.00, "seeds": 4},
                {"cell": "Pegasus 3", "variables": 39, "fill": 0.50,
                 "initial_range": [0.05, 0.11], "final_per_seed": [1.00, 0.94], "seeds": 2},
                {"cell": "Zephyr 2", "variables": 39, "fill": 0.50,
                 "initial_range": [0.00, 0.05], "final_per_seed": [0.61, 0.44], "seeds": 2},
                {"cell": "Pegasus 3 and Zephyr 2", "variables_range": [93, 116],
                 "fill_values": [0.90, 0.95], "initial": 0.00, "final": 0.00,
                 "minorminer_valid_at_300s": 0.00}]},

        "t5_transfer.json": {
            "table": "T5",
            "claim": "training at hardware scale transfers across sizes and across topologies",
            "protocol": {"decisions": 400, "seconds": 300, "episodes_per_instance": 5,
                         "checkpoints": "frozen", "trained_on_variables": 24},
            "caveat": "both lists were opened once at a looser budget before this rerun, so they are development diagnostics",
            "source": ["results/transfer/m_*.log"],
            "tests": [
                {"test_host": "pegasus16", "checkpoint": "trained on Pegasus 16",
                 "overall": 0.60, "by_variables": {"24": 0.90, "48": 0.65, "100": 0.43}},
                {"test_host": "pegasus16", "checkpoint": "trained on Zephyr 15, cross-topology",
                 "overall": 0.55, "by_variables": {"24": 0.90, "48": 0.62, "100": 0.33}},
                {"test_host": "pegasus16", "checkpoint": "fragment, never trained at scale",
                 "overall": 0.38, "by_variables": {"24": 0.70, "48": 0.33, "100": 0.33}},
                {"test_host": "zephyr15", "checkpoint": "trained on Zephyr 15",
                 "overall": 0.81, "by_variables": {"24": 0.93, "48": 1.00, "100": 0.52}},
                {"test_host": "zephyr15", "checkpoint": "trained on Pegasus 16, cross-topology",
                 "overall": 0.84, "by_variables": {"24": 0.95, "48": 0.93, "100": 0.60}},
                {"test_host": "zephyr15", "checkpoint": "fragment, never trained at scale",
                 "overall": 0.73, "by_variables": {"24": 0.93, "48": 0.80, "100": 0.36}}],
            "effects": {"scale_training_over_fragment": [0.22, 0.08],
                        "cross_topology_cost": [-0.05, 0.03]}},

        "t8_expressiveness.json": {
            "table": "T8",
            "claim": "the observation carries real signal but capacity and the physics channels are not the binding constraint",
            "protocol": {"records": "candidate rows cached at each witness decision",
                         "split": "over training instances, because a held-out instance carries no witness",
                         "loss": "one ranking loss for both models"},
            "caveat": "the witness is one valid solution among many, so a move that is not witness-consistent may still be good",
            "source": ["results/quality/express4_p3.log", "results/quality/express_big*.log"],
            "rows": [
                {"paths": 8, "decisions": 272, "schema": "physics32", "model": "linear",
                 "fitted": 0.276, "held_out": 0.278, "chance": 0.031},
                {"paths": 8, "decisions": 272, "schema": "physics32", "model": "small network",
                 "fitted": 0.522, "held_out": 0.187, "chance": 0.031},
                {"paths": 39, "decisions": 1293, "schema": "physics32", "model": "linear",
                 "fitted": 0.363, "held_out": 0.234, "chance": 0.026},
                {"paths": 39, "decisions": 1293, "schema": "physics32", "model": "small network",
                 "fitted": 0.512, "held_out": 0.267, "chance": 0.026},
                {"paths": 39, "decisions": 1293, "schema": "local20", "model": "linear",
                 "fitted": 0.354, "held_out": 0.249, "chance": 0.026},
                {"paths": 39, "decisions": 1293, "schema": "local20", "model": "small network",
                 "fitted": 0.396, "held_out": 0.261, "chance": 0.026}]},

        "t10_fill_certification.json": {
            "table": "T10",
            "claim": "named fill overstates the occupancy an embedding actually needs",
            "note": "both the witness and its pruning are sufficient occupancies, neither is a minimum",
            "source": ["results/frontier/prune_*.log"],
            "rows": [
                {"host": "pegasus2", "named_fill": 0.90, "variables": 30, "pruned_fill": 0.84,
                 "lower_bound": 0.75, "minorminer_valid_at_200_tries": 0.58},
                {"host": "pegasus3", "named_fill": 0.90, "variables": 93, "pruned_fill": 0.86,
                 "lower_bound": 0.72, "minorminer_valid_at_200_tries": 0.00},
                {"host": "pegasus6", "named_fill": 0.90, "variables": 488, "pruned_fill": 0.87,
                 "lower_bound": 0.72, "minorminer_valid_at_200_tries": 0.00},
                {"host": "zephyr1", "named_fill": 0.95, "variables": 38, "pruned_fill": 0.92,
                 "lower_bound": 0.79, "minorminer_valid_at_200_tries": 0.17},
                {"host": "zephyr2", "named_fill": 0.90, "variables": 116, "pruned_fill": 0.87,
                 "lower_bound": 0.72, "minorminer_valid_at_200_tries": 0.00}]},

        "t11_reward_channel.json": {
            "table": "T11",
            "claim": "residual trains, solve probability assesses",
            "protocol": {"embeddings_per_instance": 8, "blocks_per_embedding": 8,
                         "reads_per_block": 256, "instances_per_host": 12,
                         "method": "measurement variance removed from between-candidate variance"},
            "source": ["results/quality/reward_*.log"],
            "rows": [
                {"host": "pegasus3", "channel": "solve probability", "snr_at_256_reads": 2.75,
                 "assessed_gain_from_ranking": 0.038},
                {"host": "pegasus3", "channel": "residual", "snr_at_256_reads": 19.1,
                 "assessed_gain_from_ranking": 0.027},
                {"host": "zephyr2", "channel": "solve probability", "snr_at_256_reads": 0.73,
                 "assessed_gain_from_ranking": 0.007},
                {"host": "zephyr2", "channel": "residual", "snr_at_256_reads": 39.8,
                 "assessed_gain_from_ranking": 0.032}]},

        "t12_sample_size.json": {
            "table": "T12",
            "claim": "0.017 residual is the measurable bar and it is thirty-seven percent of the gap to the router",
            "evaluation_noise": {"paired_per_instance_spread": [0.0231, 0.0454],
                                 "measured_by": "the same arm evaluated twice on the same instances"},
            "design": {"confidence": 0.95, "power": 0.80, "test": "paired"},
            "rows": [
                {"effect": 0.005, "instances_at_sd_0.023": 166, "instances_at_sd_0.045": 636,
                 "core_hours_per_evaluation_at_24_instances": [69, 265]},
                {"effect": 0.017, "instances_at_sd_0.023": 15, "instances_at_sd_0.045": 55,
                 "core_hours_per_evaluation_at_24_instances": [10, 20]}],
            "full_study": {"arms": 3, "evaluations": 3, "seeds": 3,
                           "core_hours_at_effect_0.005": [1900, 7000]}},

        "t14_support_cost.json": {
            "table": "T14",
            "claim": "the wide candidate registration is required, and a hinted replay cannot establish that",
            "source": ["results/quality/replay_*.log", "results/control/replay_p3_f90_*.log"],
            "rows": [
                {"replay": "unhinted", "support": "registered 64 candidates", "fill": 0.50,
                 "valid": 0, "of": 6,
                 "failure": "states offering no legal placement while variables remain"},
                {"replay": "unhinted", "support": "wide 512 candidates", "fill": 0.50,
                 "valid": 6, "of": 6, "decisions": 58, "seconds": 13.9},
                {"replay": "unhinted", "support": "registered 64 candidates", "fill": 0.90,
                 "valid": 1, "of": 4},
                {"replay": "unhinted", "support": "wide 512 candidates", "fill": 0.90,
                 "valid": 4, "of": 4, "decisions": 107, "seconds": 80.7}],
            "step_cost": {"support": "wide", "qubits": 115, "seconds": 0.53}},

        "retractions.json": {
            "note": "every entry was reported and then withdrawn by a later measurement",
            "entries": [
                {"claim": "solve probability is exactly zero at scale",
                 "withdrawn_by": "zero hits in 512 reads bounds the rate below 0.006"},
                {"claim": "annealing deeper erases the embedding signal",
                 "withdrawn_by": "the table compared each embedding's own best chain strength on the block it was scored on",
                 "corrected_value": -0.0005, "ci95": [-0.0036, 0.0026]},
                {"claim": "residual separates embeddings at the congested cell",
                 "withdrawn_by": "a fresh twenty-four instances",
                 "first": {"value": 0.0042, "ci95": [0.0008, 0.0076], "instances": 12},
                 "replication": {"value": 0.0004, "ci95": [-0.0031, 0.0038], "instances": 24}},
                {"claim": "the cold-start pair is a curriculum ablation",
                 "withdrawn_by": "both logs hold only the initial evaluations and not one training iteration"},
                {"claim": "the congested cell runs inside the registered support",
                 "withdrawn_by": "unhinted, the registered support reaches a COMMIT on 0 of 6"},
                {"claim": "cloning closes 46 percent of the quality gap",
                 "withdrawn_by": "a twenty-instance replication",
                 "first": {"value": 0.0366, "ci95": [-0.0029, 0.0761], "instances": 8},
                 "replication": {"value": -0.0200, "ci95": [-0.0428, 0.0028], "instances": 20}},
                {"claim": "twenty updates of the feasibility reward take 53 percent of the gap",
                 "withdrawn_by": "a mid-training evaluation on four instances; the arm ended worse than it started"},
                {"claim": "named fill is the certified congestion",
                 "withdrawn_by": "pruning shows a fill-0.90 witness needs 0.84 to 0.87"},
                {"claim": "stage P and Z are the full Pegasus 16 and Zephyr 15",
                 "withdrawn_by": "they are 32 to 64 qubit fragments of Pegasus 3 and Zephyr 2"},
                {"claim": "qubit count is not a quality signal",
                 "withdrawn_by": "it is, at rank correlation 0.15 to 0.27; measurement is three to four times stronger"}]},

        "open.json": {
            "entries": [
                {"item": "the quality objective failed its pre-registered test",
                 "why": "the quality at a decision is 0.0491 [0.0467, 0.0515] and neither the policy nor a model fitted to the measured orderings can find it from the rows the environment offers",
                 "next": "the action representation, not the reward"},
                {"item": "the congested cells have a benchmark result and no method result",
                 "variables_range": [93, 116], "fill_values": [0.90, 0.95],
                 "minorminer_valid": 0, "of": 12, "constructor_valid": 0.00,
                 "technique_under_test": "reverse-start curriculum",
                 "status": "prefix frozen at 0.969 and 0.959, training validity 0.274 to 0.364 against a 0.8 advance threshold"},
                {"item": "no QPU result exists",
                 "scope": "every quality statement is classical simulated annealing under the registered schedule"}]}}


def main():
    written = []
    for name, payload in [("f2_pareto_resource_quality.json", pareto()),
                          ("t1_selection_signal.json", t1_selection_signal()),
                          ("t2_quality_mechanism.json", t2_quality_mechanism()),
                          ("t6_quality_arms.json", t6_quality_arms()),
                          ("t7_branch_decomposition.json", t7_branch()),
                          ("t9_congestion_wallclock.json", t9_congestion()),
                          ("t13_ablations.json", t13_ablations()),
                          ("t15_solvability_decay.json", t15_solvability())]:
        written.append(write(name, payload))
    for name, payload in sorted(transcribed().items()):
        written.append(write(name, payload))

    manifest = {
        "protocol": {"sampler": "classical simulated annealing", "beta_range": [0.1, 2.0],
                     "sweeps": 200,
                     "selection_and_assessment": "disjoint read blocks",
                     "qpu": "no QPU result exists and none is claimed"},
        "conventions": {"ci95": "95 percent interval",
                        "residual": "normalized energy residual, lower is better",
                        "p_solve": "decoded solve probability, higher is better",
                        "derivation": "recomputed from the exported figure data unless a field says transcribed"},
        "files": [{"file": os.path.basename(p), "bytes": n} for p, n in written]}
    written.append(write("MANIFEST.json", manifest))
    for p, n in written:
        print("%-38s %7d bytes" % (p, n))


if __name__ == "__main__":
    main()

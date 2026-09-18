"""Paired, single-seed attribution report for empty-start quality training.

Select a registered evaluation tag, never the best observed checkpoint. The
continued-feasibility control must share v2 data/code/settings, differing only in
training objective. Bootstrap units are lineages (instance fallback is diagnostic).
A sampler/assessment failure is missing evidence, never a zero-quality label.

Example: python probes/quality_study_report.py --quality-log quality.log \
    --control-log continued_feasibility.log --tag final
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

VERSION = "constructor-quality-v2"
REQUIRED_OPTIONS = ("stage", "seed", "actor", "features", "support", "init", "iterations",
                    "learning_rate", "baseline", "max_steps", "episode_seconds", "deadline",
                    "select_cap", "selection_reads", "assessment_reads", "reward_reads",
                    "evaluation_objective", "heldout_role")
METRICS = {"deployment_p_solve": ("p_solve", 1), "quality_utility": ("quality_utility", 1),
           "conditional_residual": ("residual", -1)}


def load_log(path):
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.startswith("{"):
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON record") from exc
    headers = [r for r in records if "protocol" in r]
    if len(headers) > 1:
        raise ValueError("multiple run headers: provide one run per log, never pooled seeds")
    return {"path": str(path), "header": headers[0] if headers else {}, "records": records}


def evaluation(log, tag, split="heldout"):
    found = [r for r in log["records"] if r.get("evaluation") == tag and r.get("set") == split
             and r.get("objective") == "quality"]
    if len(found) != 1:
        raise ValueError(f"expected one quality evaluation tag={tag!r}, set={split!r}; got {len(found)}")
    row = found[0]
    ids = [r.get("instance") for r in row.get("rows", [])]
    if not ids or None in ids or len(set(ids)) != len(ids):
        raise ValueError("evaluation needs unique nonempty instance identities")
    if row.get("instances") != len(ids):
        raise ValueError("declared instance count differs from evaluation rows")
    return row


def contract_reasons(log, record):
    header, reasons = log["header"], []
    contract = header.get("contract", {})
    if header.get("protocol") != VERSION or contract.get("version") != VERSION:
        reasons.append("legacy or missing v2 contract: diagnostic only")
    for key in ("source_commit", "train_fingerprints", "heldout_fingerprints"):
        if not contract.get(key):
            reasons.append("missing contract " + key)
    if contract.get("source_dirty") is not False:
        reasons.append("source is dirty or its status is unknown")
    opts = contract.get("options", {})
    checkpoint_hash = contract.get("initial_checkpoint_sha256")
    if opts.get("init") and not checkpoint_hash:
        reasons.append("missing initial checkpoint content hash")
    if not opts.get("init") and checkpoint_hash is not None:
        reasons.append("checkpoint hash is present for a run without initialization")
    for key in REQUIRED_OPTIONS:
        if key not in opts:
            reasons.append("missing option " + key)
    if opts.get("evaluation_objective") != "quality":
        reasons.append("evaluation objective is not explicitly quality")
    if record.get("seed") != opts.get("seed") or record.get("stage") != opts.get("stage"):
        reasons.append("record and contract seed/stage differ")
    split_key = "train_fingerprints" if record.get("set") == "train" else "heldout_fingerprints"
    if len(contract.get(split_key, [])) != len(record["rows"]):
        reasons.append("split fingerprints do not cover evaluation rows")
    if any(not r.get("lineage") for r in record["rows"]):
        reasons.append("lineage identities missing: instance grouping is diagnostic only")
    return reasons


def attribution_reasons(left_log, right_log, left, right):
    reasons = contract_reasons(left_log, left) + contract_reasons(right_log, right)
    a, b = left_log["header"].get("contract", {}), right_log["header"].get("contract", {})
    for key in ("source_commit", "train_fingerprints", "heldout_fingerprints",
                "initial_checkpoint_sha256"):
        if a.get(key) != b.get(key):
            reasons.append("different " + key)
    x, y = a.get("options", {}), b.get("options", {})
    if x.get("objective") != "quality" or y.get("objective") != "feasibility":
        reasons.append("need quality training versus continued feasibility training")
    # Paths for output do not affect the experiment; every other setting,
    # including seed and initial checkpoint, must match except the treatment.
    for key in sorted((set(x) | set(y)) - {"objective", "out"}):
        if key not in x or key not in y or x[key] != y[key]:
            reasons.append("different option " + key)
    return sorted(set(reasons))


def aligned_rows(left, right):
    left_ids = [r["instance"] for r in left["rows"]]
    right_ids = [r["instance"] for r in right["rows"]]
    if left_ids != right_ids:
        raise ValueError("paired evaluations require the same exact ordered instance list")
    for a, b in zip(left["rows"], right["rows"]):
        if a.get("lineage", a["instance"]) != b.get("lineage", b["instance"]):
            raise ValueError("instance lineage assignment differs")
        if "policy" not in a or "policy" not in b:
            raise ValueError("both evaluation rows require the policy arm")
    return list(zip(left["rows"], right["rows"]))


def assessment_problem(arm):
    if arm.get("assessment_failures", 0) or arm.get("selection_failures", 0):
        return "measurement failure"
    if arm.get("valid") and (arm.get("assessment_ok") is False or arm.get("residual") is None):
        return "valid construction has no successful assessment"
    return None


def metric_value(arm, field):
    if not isinstance(arm.get("valid"), bool):
        raise ValueError("validity must be a Boolean")
    if not arm["valid"]:
        return None if field == "residual" else 0.0
    value = arm.get(field)
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if value < 0 or (field != "residual" and value > 1):
        raise ValueError("quality metric lies outside its declared range")
    return float(value)


def paired_bootstrap(values, draws=10000, seed=0, allow_ci=True):
    """Equal-weight lineage means; repetitions/reads never become bootstrap units."""
    grouped = defaultdict(list)
    for lineage, delta in values:
        grouped[lineage].append(delta)
    means = [statistics.mean(v) for _, v in sorted(grouped.items())]
    point = statistics.mean(means) if means else None
    ci = None
    if allow_ci and len(means) >= 2:
        rng = random.Random(seed)
        samples = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(draws))
        ci = [samples[int(.025 * (draws - 1))], samples[int(.975 * (draws - 1))]]
    return {"mean_gain": point, "ci95": ci, "lineages": len(means), "instances": len(values),
            "estimand": "equal-weight mean of lineage-level paired gains"}


def paired_report(left, right, reasons=(), draws=10000, seed=0):
    pairs = aligned_rows(left, right)
    reasons = list(reasons)
    if left.get("seed") != right.get("seed"):
        reasons.append("different training seeds; single-seed attribution is required")
    if left.get("stage") != right.get("stage") or left.get("set") != right.get("set"):
        reasons.append("different evaluation stage or split")
    measurement_errors = []
    for a, b in pairs:
        for side, row in (("left", a), ("right", b)):
            problem = assessment_problem(row["policy"])
            if problem:
                measurement_errors.append({"instance": row["instance"], "side": side, "reason": problem})
        ba, bb = a["policy"].get("budget"), b["policy"].get("budget")
        if ba != bb or ba is None:
            reasons.append("missing or unequal per-instance evaluation budgets")
    if measurement_errors:
        reasons.append("assessment/selection failure: do not replace missing measurements with zero")
    coverage = {side: sum(r["policy"]["valid"] for r in rec["rows"]) / len(pairs)
                for side, rec in (("left", left), ("right", right))}
    results = {}
    for name, (field, sign) in METRICS.items():
        values, missing = [], []
        for a, b in pairs:
            if name == "conditional_residual" and not (a["policy"]["valid"] and b["policy"]["valid"]):
                continue
            x, y = metric_value(a["policy"], field), metric_value(b["policy"], field)
            if x is None or y is None:
                missing.append(a["instance"])
                continue
            values.append((a.get("lineage", a["instance"]), sign * (x - y)))
        blocked = bool(reasons or missing)
        result = paired_bootstrap(values, draws=draws, seed=seed, allow_ci=not blocked)
        if measurement_errors or missing:
            # A subset mean may be mistaken for an unconditional deployment metric.
            result["mean_gain"] = None
        result.update(direction="positive favours left", raw_delta_sign=sign,
                      missing_instances=missing, inference_eligible=not blocked and result["lineages"] >= 2)
        if name == "conditional_residual":
            result["condition"] = "both policies constructed and independently assessed; failures excluded"
        results[name] = result
    return {"left_tag": left["evaluation"], "right_tag": right["evaluation"],
            "seed": left.get("seed"), "coverage": coverage, "instances": len(pairs),
            "diagnostic_only": bool(reasons), "blocking_reasons": sorted(set(reasons)),
            "measurement_errors": measurement_errors, "metrics": results}


def build_report(quality_log, control_log=None, tag="final", control_tag=None,
                 split="heldout", draws=10000, seed=0):
    if draws < 100:
        raise ValueError("bootstrap requires at least 100 draws")
    log = load_log(quality_log)
    final = evaluation(log, tag, split)
    out = {"report": "quality-training-attribution-v1", "quality_log": str(quality_log),
           "selected_tag": tag, "split": split,
           "note": "one paired training seed; no cross-seed pooling or automatic best-checkpoint selection"}
    initial = [r for r in log["records"] if r.get("evaluation") == "init" and r.get("set") == split
               and r.get("objective") == "quality"]
    if initial and tag != "init":
        init = evaluation(log, "init", split)
        out["within_run_final_minus_init"] = paired_report(
            final, init, contract_reasons(log, final), draws, seed)
        out["within_run_final_minus_init"]["attribution"] = "training progress; does not isolate quality reward"
    else:
        out["within_run_final_minus_init"] = None
    if control_log:
        control = load_log(control_log)
        cf = evaluation(control, control_tag or tag, split)
        reasons = attribution_reasons(log, control, final, cf)
        out["quality_minus_continued_feasibility"] = paired_report(final, cf, reasons, draws, seed)
    else:
        out["quality_minus_continued_feasibility"] = None
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-log", required=True)
    parser.add_argument("--control-log")
    parser.add_argument("--tag", default="final")
    parser.add_argument("--control-tag")
    parser.add_argument("--split", default="heldout", choices=("heldout", "train"))
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.quality_log, args.control_log, args.tag, args.control_tag,
                              args.split, args.bootstrap_draws, args.bootstrap_seed)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

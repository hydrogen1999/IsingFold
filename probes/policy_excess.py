"""Audit removable qubits and measured quality on an explicit corpus split.

This frozen-checkpoint diagnostic compares the first valid policy construction,
its greedy pruning, and a separate minorminer draw. It does not isolate a causal
placement effect or impose a shared solver wall-time budget. Every requested task
contributes to coverage and all-instance utility, including construction failures.
"""
# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
sys.path[:0] = [os.environ.get("ISINGFOLD_SRC", str(ROOT / "src")), str(ROOT / "probes")]

import numpy as np
import torch
from isingfold.rl.evaluator import sample_program
from isingfold.rl.program import compile_program, strength_registry

import constructor_curriculum as cc
from constructor_checkpoint import run_contract
from constructor_objective import quality_utility
from constructor_provenance import training_provenance
from constructor_rollout import episode
from _context import REGISTERED_BETA_RANGE, host_context
from _initializers import minorminer_initializer
from witness_prune import prune


def measure(task, chains, ctx, index, reads, seed):
    """Residual, solve probability and broken-chain fraction from the same block."""
    strengths = strength_registry(task.problem, ctx.strength_ratios, ctx.epsilon_strength)
    program = compile_program(chains, task.host, task.problem, strengths[index], index)
    block = sample_program(program, chains, task.problem, task.ground_energy, num_reads=reads,
                           seed=int(seed), num_sweeps=200, beta_range=REGISTERED_BETA_RANGE)
    residual, p_solve = float(block.mean_residual), float(block.rate)
    if block.reads != reads or block.strength_index != index:
        raise ValueError("quality assessment violated its read/strength receipt")
    if not math.isfinite(residual) or residual < 0 or not math.isfinite(p_solve) or not 0 <= p_solve <= 1:
        raise ValueError("quality assessment returned invalid metrics")
    broken = float(block.broken_fraction)
    if not math.isfinite(broken) or not 0 <= broken <= 1:
        raise ValueError("broken-chain fraction must be finite and in [0, 1]")
    return residual, p_solve, broken


def shape(chains):
    sizes = [len(c) for c in chains.values()]
    return sum(sizes), max(sizes), sum(sizes) / len(sizes)


def audit_tasks(args):
    """Use only the evaluation pool; no task receives a training witness."""
    if args.exploratory_repartition and args.role == "test":
        raise ValueError("test evaluation requires the manifest split")
    cells = [c for c in args.cells.split(",") if c]
    _, tasks = cc.build_corpus_sets(args.corpus, cells, 0, args.n_tasks, args.seed,
                                   use_manifest_split=not args.exploratory_repartition,
                                   heldout_role=args.role)
    return tasks


def source_digest():
    """Hash the Python source imported from this checkout/source root."""
    source = Path(os.environ.get("ISINGFOLD_SRC", str(ROOT / "src"))) / "isingfold"
    digest = hashlib.sha256()
    for label, directory in (("probes", ROOT / "probes"), ("isingfold", source)):
        for path in sorted(directory.rglob("*.py")):
            digest.update((label + "/" + path.relative_to(directory).as_posix()).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def assessment(task, chains, ctx, args, seed, failure):
    if chains is None:
        return {"valid": False, "failure": failure, "qubits": None, "longest": None,
                "residual": None, "p_solve": 0.0, "quality_utility": 0.0,
                "broken": None,
                "assessment_seed": None, "reads": 0, "strength_index": args.strength_index}
    residual, p_solve, broken = measure(task, chains, ctx, args.strength_index, args.reads, seed)
    qubits, longest, _ = shape(chains)
    return {"valid": True, "failure": None, "qubits": qubits, "longest": longest,
            "residual": residual, "p_solve": p_solve, "broken": broken,
            "quality_utility": quality_utility(task, residual), "assessment_seed": seed,
            "reads": args.reads, "strength_index": args.strength_index}


def summarize(rows):
    """Separate conditional quality from metrics whose denominator is all tasks."""
    def mean(values):
        return sum(values) / len(values) if values else None
    result = {"summary": "policy_excess", "tasks": len(rows), "arms": {}}
    for arm in ("policy", "pruned", "minorminer"):
        valid = [row[arm] for row in rows if row[arm]["valid"]]
        result["arms"][arm] = {
            "valid_tasks": len(valid), "failed_tasks": len(rows) - len(valid),
            "coverage": len(valid) / len(rows),
            "mean_residual_conditional": mean([r["residual"] for r in valid]),
            "mean_p_solve_conditional": mean([r["p_solve"] for r in valid]),
            "mean_broken_conditional": mean([r["broken"] for r in valid]),
            "mean_p_solve_all_instances": mean([r[arm]["p_solve"] for r in rows]),
            "mean_quality_utility_all_instances": mean([r[arm]["quality_utility"] for r in rows]),
            "mean_qubits_conditional": mean([r["qubits"] for r in valid]),
            "assessment_reads": sum(r[arm]["reads"] for r in rows),
        }
    pruned = [r for r in rows if r["policy"]["valid"] and r["pruned"]["valid"]]
    paired = [r for r in rows if r["policy"]["valid"] and r["minorminer"]["valid"]]
    result.update({
        "pruning_pairs": len(pruned),
        "mean_removable_fraction_conditional": mean([r["removable_fraction"] for r in pruned]),
        "recovered_by_pruning_conditional": mean([
            r["policy"]["residual"] - r["pruned"]["residual"] for r in pruned]),
        "policy_minorminer_pairs": len(paired),
        "mean_policy_minus_minorminer_residual_paired": mean([
            r["policy"]["residual"] - r["minorminer"]["residual"] for r in paired]),
        "reading": "Descriptive pruning/checkpoint diagnostic; no causal placement mechanism identified.",
    })
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--init", required=True)
    ap.add_argument("--features", choices=tuple(cc.FEATURE_WIDTHS), default="local")
    ap.add_argument("--actor", choices=("linear", "mlp", "contextual"), default="linear")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--support", choices=("registered", "wide"), default="wide")
    ap.add_argument("--role", choices=("validation", "test"), default="validation")
    ap.add_argument("--exploratory-repartition", action="store_true",
                    help="explicit development-only random repartition; never a held-out test")
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--episode-seconds", type=float, default=90.)
    ap.add_argument("--reads", type=int, default=4096)
    ap.add_argument("--strength-index", type=int, default=1)
    ap.add_argument("--mm-tries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args(argv)
    if min(a.n_tasks, a.episodes, a.max_steps, a.reads, a.mm_tries, a.width) < 1:
        ap.error("task/episode/step/read/baseline counts and width must be positive")
    if not math.isfinite(a.episode_seconds) or a.episode_seconds <= 0 or a.strength_index < 0:
        ap.error("episode seconds must be finite and positive; strength index must be nonnegative")
    if a.exploratory_repartition and a.role == "test":
        ap.error("--role test cannot be combined with --exploratory-repartition")

    tasks = audit_tasks(a)
    provenance = training_provenance(a.init, current_train=[], heldout=tasks, training=False)
    contract = run_contract(a, [], tasks)
    contract.update(version="policy-excess-v2", code_sha256=source_digest())
    actor = cc.make_actor(a.actor, a.width, cc.FEATURE_WIDTHS[a.features])
    cc.load_init(a.init, actor, a.actor, a.features)
    actor.eval()
    mm = minorminer_initializer(a.mm_tries)
    print(json.dumps({"probe": "policy_excess", "protocol": "policy-excess-v2",
                      "contract": contract, "config": vars(a), "training_provenance": provenance,
                      "evaluation_role": "exploratory" if a.exploratory_repartition else a.role,
                      "heldout_ancestry_verified": bool(provenance["complete"] and not a.exploratory_repartition),
                      "scope": "frozen-checkpoint diagnostic; solver construction budgets are not matched",
                      "tasks": [{"name": t.name, "lineage": t.lineage or t.name} for t in tasks],
                      "assessment": {"num_sweeps": 200, "beta_range": REGISTERED_BETA_RANGE,
                                     "selection": "first valid policy episode; separate minorminer draw",
                                     "metrics": "residual, p_solve and broken-chain fraction from the same read block"}}), flush=True)

    rows = []
    for k, t in enumerate(tasks):
        ctx = host_context(t.host.number_of_nodes())
        if a.strength_index >= len(ctx.strength_ratios):
            raise ValueError("strength index exceeds the registered strength grid")
        fc = cc.make_features(a.features, t)
        found, attempts = None, []
        for e in range(a.episodes):
            episode_seed = a.seed + 100 * k + e
            started = time.monotonic()
            with torch.no_grad(), cc.no_completion_solver():
                rec = episode(t, actor, fc, 1., a.max_steps,
                              np.random.default_rng(episode_seed), a.episode_seconds,
                              train=True, objective="quality", evaluate_reward=False,
                              wide=a.support == "wide")
            attempts.append({"seed": episode_seed, "valid": bool(rec["valid"]),
                             "reason": rec["reason"], "steps": rec["steps"],
                             "seconds": time.monotonic() - started})
            if rec["valid"]:
                found = {v: frozenset(c) for v, c in rec["terminal"].embedding.items()}
                break
        tight = prune(t.host, found, t.logical) if found is not None else None
        mm_seed = a.seed + k
        started = time.monotonic()
        draw = mm(t.logical, t.host, mm_seed)
        mm_seconds = time.monotonic() - started
        mm_chains = {v: frozenset(draw[v]) for v in t.logical.nodes()} if draw else None
        seed = a.seed + 10000 * k
        policy = assessment(t, found, ctx, a, seed, "no valid COMMIT within episode budget")
        pruned = assessment(t, tight, ctx, a, seed + 1, "policy construction failed")
        baseline = assessment(t, mm_chains, ctx, a, seed + 2, "minorminer returned no embedding")
        removable = policy["qubits"] - pruned["qubits"] if found is not None else None
        row = {"task": t.name, "lineage": t.lineage or t.name,
               "variables": t.logical.number_of_nodes(), "policy": policy,
               "pruned": pruned, "minorminer": baseline,
               "policy_attempts": attempts, "minorminer_seed": mm_seed,
               "minorminer_seconds": mm_seconds, "removable_qubits": removable,
               "removable_fraction": removable / policy["qubits"] if found is not None else None}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps(summarize(rows)), flush=True)
    print("POLICY EXCESS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

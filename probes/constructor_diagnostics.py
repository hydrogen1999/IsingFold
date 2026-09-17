"""Two diagnostics behind the constructor ladder's open bottlenecks.

``support``: what the policy sees and does at scale. For a few train tasks of a corpus (or a
generated stage) and a checkpoint (or zero weights), roll episodes with the exact sampling of
``constructor_rollout.episode`` and record per step the support size, the candidate count and
the policy's probability mass per action family, and the chosen family; report episode
length, termination reason, progress at the end, mean mass per family and P(STOP) per step.

``snr``: what the quality reward's policy gradient sees. For a few tasks and a checkpoint, take
several valid embeddings the policy produces, measure each residual several times on the
reward read count, and compare the between-embedding variance of the true value with the
within-embedding measurement variance, in residual and in reward-utility units, for the
leave-one-out advantage the trainer uses.
"""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys
import time
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [finder for finder in sys.meta_path
                    if not ("editable" in str(type(finder)).lower()
                            and "isingfold" in str(type(finder)).lower())]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "probes")]

import numpy as np
import torch

import constructor_curriculum as cc
from constructor_rollout import _context, _progress, episode
from constructor_tiny_gate import no_completion_solver
from isingfold.rl.contracts import DecisionState, TerminalRecord
from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector

FAMILIES = ("PLACE", "ROUTE", "GROW", "SHRINK", "REWRITE", "REPAIR", "RESTART", "COMMIT", "STOP")


def family(opcode):
    for f in FAMILIES:
        if opcode.startswith(f):
            return f
    return "OTHER"


def support_episode(task, actor, fc, max_steps, seconds, rng, initializer=None):
    """Mirror of constructor_rollout.episode that records the support and the policy mass."""
    torch.set_num_threads(1)
    started = time.monotonic()
    ctx = _context(task, len(task.host), max_steps, 8, None, 2)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=initializer,
                       selector=fixed_strength_selector(), reward_reads=8, build_observation=False)
    env.generator.allow_satisfied_growth = True
    current = env.reset(int(rng.integers(0, 2 ** 31)))
    steps, reason = [], None
    while isinstance(current, DecisionState):
        if time.monotonic() - started >= seconds:
            reason = "DEADLINE"; break
        if len(steps) >= max_steps:
            reason = "HORIZON"; break
        legal = [i for i, ok in enumerate(current.legal_mask) if ok]
        if not legal:
            reason = "NO_LEGAL_ACTION"; break
        chains = env.state.chains
        rows = [fc.observe(current.candidates[i], chains, state=env.state, ctx=ctx,
                           steps_left=max_steps - len(steps), max_steps=max_steps) for i in legal]
        feats = torch.as_tensor(np.stack(rows), dtype=torch.float32)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=actor(feats))
        probs = dist.probs.numpy().astype(np.float64); probs /= probs.sum()
        fams = [family(current.candidates[i].opcode.value) for i in legal]
        mass, count = defaultdict(float), Counter(fams)
        for f, p in zip(fams, probs):
            mass[f] += float(p)
        action = int(rng.choice(len(legal), p=probs))
        steps.append({"support": len(legal), "mass": dict(mass), "count": dict(count),
                      "chosen": fams[action], "progress": _progress(task, chains)})
        current = env.step(current, legal[action], evaluate_training_reward=False).next_decision_or_terminal
    terminal = current if isinstance(current, TerminalRecord) else None
    if reason is None:
        reason = terminal.terminal_reason.value if terminal is not None else "NO_TERMINAL"
    valid = bool(terminal is not None and terminal.returned_valid and terminal.embedding is not None
                 and steps and steps[-1]["chosen"] == "COMMIT")
    return steps, reason, valid, env.state.chains, time.monotonic() - started


def summarise(steps):
    if not steps:
        return {}
    fams = sorted({f for s in steps for f in s["mass"]})
    return {"steps": len(steps),
            "mean_support": float(np.mean([s["support"] for s in steps])),
            "mean_mass": {f: float(np.mean([s["mass"].get(f, 0.0) for s in steps])) for f in fams},
            "mean_count": {f: float(np.mean([s["count"].get(f, 0) for s in steps])) for f in fams},
            "chosen": dict(Counter(s["chosen"] for s in steps)),
            "first_progress": steps[0]["progress"], "last_progress": steps[-1]["progress"]}


def run_support(args, tasks):
    actor = cc.make_actor(args.actor, args.width, cc.FEATURE_WIDTHS[args.features])
    if args.init:
        cc.load_init(args.init, actor, args.actor, args.features)
    rows = []
    for k, t in enumerate(tasks[:args.n_tasks]):
        fc = cc.make_features(args.features, t)
        for e in range(args.episodes):
            seed = args.seed * 1000 + k * 100 + e
            init = cc.prefix_initializer(t, args.prefix_fraction, seed) if args.prefix_fraction else None
            steps, reason, valid, chains, secs = support_episode(
                t, actor, fc, args.max_steps, args.episode_seconds, np.random.default_rng(seed), init)
            row = {"task": t.name, "variables": t.logical.number_of_nodes(), "host": len(t.host),
                   "episode": e, "prefix_fraction": args.prefix_fraction or 0.0, "init": args.init or None,
                   "reason": reason, "valid": valid, "seconds": secs,
                   "seconds_per_step": secs / max(1, len(steps)),
                   "final_progress": _progress(t, chains), **summarise(steps)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    fams = sorted({f for r in rows for f in r.get("mean_mass", {})})
    print(json.dumps({"summary": "support", "episodes": len(rows),
                      "mean_steps": float(np.mean([r["steps"] for r in rows if "steps" in r] or [0])),
                      "reasons": dict(Counter(r["reason"] for r in rows)),
                      "valid": sum(r["valid"] for r in rows),
                      "mean_final_progress": float(np.mean([r["final_progress"] for r in rows])),
                      "mean_seconds_per_step": float(np.mean([r["seconds_per_step"] for r in rows])),
                      "mean_mass": {f: float(np.mean([r["mean_mass"].get(f, 0.0) for r in rows if "mean_mass" in r])) for f in fams},
                      "mean_support": float(np.mean([r["mean_support"] for r in rows if "mean_support" in r] or [0]))}), flush=True)


def run_snr(args, tasks):
    from constructor_objective import measure_terminal, quality_utility
    actor = cc.make_actor(args.actor, args.width, cc.FEATURE_WIDTHS[args.features])
    if args.init:
        cc.load_init(args.init, actor, args.actor, args.features)
    out = []
    for k, t in enumerate(tasks[:args.n_tasks]):
        fc = cc.make_features(args.features, t)
        terminals, attempts = [], 0
        while len(terminals) < args.embeddings and attempts < 6 * args.embeddings:
            attempts += 1
            with torch.no_grad():
                rec = episode(t, actor, fc, 1., args.max_steps, np.random.default_rng(args.seed * 1000 + k * 100 + attempts),
                              args.episode_seconds, train=True, objective="quality", evaluate_reward=False)
            if rec["valid"]:
                terminals.append(rec["terminal"])
        if len(terminals) < 2:
            print(json.dumps({"task": t.name, "skipped": "fewer than two valid embeddings", "attempts": attempts}), flush=True)
            continue
        residuals = np.array([[measure_terminal(t, term, args.seed * 100000 + i * 1000 + r, args.reads)
                               for r in range(args.repeats)] for i, term in enumerate(terminals)])
        utilities = np.vectorize(lambda x: quality_utility(t, x))(residuals)
        within = residuals.var(axis=1, ddof=1).mean()
        between_obs = residuals.mean(axis=1).var(ddof=1)
        between_true = max(0.0, between_obs - within / args.repeats)
        within_u = utilities.var(axis=1, ddof=1).mean()
        between_u = max(0.0, utilities.mean(axis=1).var(ddof=1) - within_u / args.repeats)
        K = args.episodes_per_instance
        # single-measurement leave-one-out advantage: signal var = between_true * (1 + 1/(K-1)),
        # noise var = within * (1 + 1/(K-1)); the ratio is independent of K
        snr = between_true / within if within > 0 else float("inf")
        row = {"task": t.name, "embeddings": len(terminals), "attempts": attempts, "reads": args.reads,
               "mean_residual": float(residuals.mean()), "within_std_residual": float(np.sqrt(within)),
               "between_std_residual_true": float(np.sqrt(between_true)),
               "within_std_utility": float(np.sqrt(within_u)), "between_std_utility_true": float(np.sqrt(between_u)),
               "snr_single_measurement": float(snr),
               "reads_for_snr_1": (int(np.ceil(args.reads / snr)) if snr > 0 else None),
               "qubits": [sum(len(c) for c in term.embedding.values()) for term in terminals]}
        out.append(row)
        print(json.dumps(row), flush=True)
    if out:
        print(json.dumps({"summary": "snr", "tasks": len(out),
                          "median_snr": float(np.median([r["snr_single_measurement"] for r in out])),
                          "median_within_std_residual": float(np.median([r["within_std_residual"] for r in out])),
                          "median_between_std_residual_true": float(np.median([r["between_std_residual_true"] for r in out])),
                          "median_reads_for_snr_1": float(np.median([r["reads_for_snr_1"] for r in out if r["reads_for_snr_1"]] or [float("nan")]))}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("support", "snr"), required=True)
    ap.add_argument("--stage", choices=cc.STAGES + ("corpus",), default="corpus")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--cells", default="")
    ap.add_argument("--n-tasks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actor", choices=("linear", "mlp", "contextual"), default="linear")
    ap.add_argument("--features", choices=("tiny", "construction"), default="tiny")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--init", default="")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--prefix-fraction", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--episode-seconds", type=float, default=300.)
    ap.add_argument("--embeddings", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--episodes-per-instance", type=int, default=8)
    args = ap.parse_args(argv)
    if args.stage == "corpus":
        if not args.corpus:
            ap.error("--corpus is required for --stage corpus")
        cells = [c for c in args.cells.split(",") if c]
        train, _ = cc.build_corpus_sets(args.corpus, cells, args.n_tasks, 1, args.seed)
    else:
        train, _ = cc.build_sets(args.stage, args.n_tasks, 1, args.seed)
    print(json.dumps({"diagnostic": args.mode, "tasks": [t.name for t in train], "config": vars(args)}), flush=True)
    with no_completion_solver():
        (run_support if args.mode == "support" else run_snr)(args, train)
    print("DIAGNOSTIC DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

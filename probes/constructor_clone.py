"""Behaviour cloning of train witnesses with manifest validation from empty.

The default monotone teacher labels offered legal actions that strictly extend chains
inside the witness, and labels only COMMIT when available. The legacy ``witness_set``
ablation also labels witness-subset shrink/relocation actions and refinements at COMMIT;
those labels can teach workspace cycles even when the selected teacher walk succeeds.
Neither teacher guarantees a path through truncated support or teaches Ising quality.

    L = -log sum_{a in S(s)} pi(a | s),  S(s) = eligible offered legal candidates

Only train-lineage witnesses are used, and only to build the teacher trajectory. Evaluation
is the curriculum's own: episodes from empty on unseen instances, no witness reachable, no
completion solver. A cloned actor is a starting point for RL, not a result by itself.
"""
# ruff: noqa: E402
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [finder for finder in sys.meta_path
                    if not ("editable" in str(type(finder)).lower()
                            and "isingfold" in str(type(finder)).lower())]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "probes")]

import numpy as np
import torch

import constructor_curriculum as cc
from constructor_checkpoint import run_contract
from constructor_provenance import training_provenance
from constructor_rollout import _context, _progress
from constructor_tiny_gate import no_completion_solver
from isingfold.rl.contracts import DecisionState, Opcode, TerminalRecord
from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector
from witness_replay import consistent


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clone_corpus_sets(args, cells):
    """Resolve manifest IDs before selecting cells; expose witnesses on train only."""
    from isingfold.rl.data.generate import load_instances

    root = Path(args.corpus)
    tasks = load_instances(root)
    manifest_path, splits_path = root / "manifest.json", root / "splits.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    split = json.loads(splits_path.read_text()) if splits_path.exists() else manifest.get("split", {})
    if not args.exploratory_repartition:
        if not isinstance(split, dict) or any(not isinstance(split.get(role), list)
                                              for role in ("train", "validation")):
            raise ValueError("cloning requires manifest train/validation lists; "
                             "--exploratory-repartition explicitly enables diagnostic repartitioning")
        if any(not isinstance(split.get(role, []), list) for role in ("train", "validation", "test")):
            raise ValueError("manifest train/validation/test entries must be lists")
        train, heldout = cc.manifest_split_sets(
            tasks, split, cells, args.train, args.heldout, args.seed, "validation")
    else:
        train, heldout = cc.corpus_sets_from_tasks(
            tasks, cells, args.train, args.heldout, args.seed)

    # The protection includes reserved lineages outside the requested cell/subset.
    # Manifests may mix instance names and lineage roots in the same list.
    manifest_lineages = {}
    for role in ("train", "validation", "test"):
        names = set(split.get(role, [])) if isinstance(split, dict) and isinstance(split.get(role), list) else set()
        manifest_lineages[role] = sorted({t.lineage or t.name for t in tasks
                                        if t.name in names or t.lineage in names})
    if any(t.prefix_source is not None for t in heldout):
        raise ValueError("held-out tasks must not expose teacher witnesses")
    files = {p.name: _sha256(p) for p in
             (manifest_path, splits_path, root / "instances.jsonl", root / "evaluator_only.jsonl")
             if p.exists()}
    metadata = {
        "manifest_split": not args.exploratory_repartition,
        "heldout_role": "diagnostic" if args.exploratory_repartition else "validation",
        "corpus_files_sha256": files,
        "split_source": "splits.json" if splits_path.exists() else "manifest.json:split",
        "split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
        "manifest_lineages": manifest_lineages,
        "train": [t.name for t in train], "heldout": [t.name for t in heldout],
        "train_lineages": sorted({t.lineage or t.name for t in train}),
        "heldout_lineages": sorted({t.lineage or t.name for t in heldout}),
    }
    return train, heldout, metadata


def teacher_candidates(current, witness, chains, mode="monotone"):
    """Choose labels from existing support only; never inject a witness action."""
    if mode not in ("monotone", "witness_set"):
        raise ValueError("teacher mode must be monotone or witness_set")
    legal = [i for i, ok in enumerate(current.legal_mask) if ok]
    good = [i for i in legal if consistent(current.candidates[i], witness, chains)]
    if mode == "witness_set":
        return good
    if any(not chain <= witness.get(v, frozenset()) for v, chain in chains.items()):
        return []
    commit = [i for i in good if current.candidates[i].opcode is Opcode.COMMIT]
    if commit:
        return commit
    # Every changed chain must retain its old qubits; at least one gains qubits.
    # Immediate edge/contact progress is not required: intermediate growth can be
    # necessary before a future PLACE or ROUTE realizes an edge.
    return [i for i in good
            if all(chains.get(v, frozenset()) <= chain
                   for v, chain in current.candidates[i].new_chains.items())
            and any(chains.get(v, frozenset()) < chain
                    for v, chain in current.candidates[i].new_chains.items())]


def teacher_steps(task, witness, features, max_steps, seconds, wide=True, rng=None,
                  teacher_mode="monotone"):
    """Walk from empty using the first eligible offered action, COMMIT when offered.

    A missing eligible action is a blocked teacher, with no fallback or completion
    solver. Successful walks alone are retained by default.
    """
    started = time.monotonic()
    ctx = _context(task, len(task.host), max_steps, 8, None, 2, wide=wide)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8, build_observation=False)
    env.generator.allow_satisfied_growth = True
    current = env.reset(0 if rng is None else int(rng.integers(0, 2 ** 31)))
    out, reason = [], None
    while isinstance(current, DecisionState):
        if time.monotonic() - started > seconds:
            reason = "DEADLINE"; break
        if len(out) >= max_steps:
            reason = "HORIZON"; break
        chains = env.state.chains
        legal = [i for i, ok in enumerate(current.legal_mask) if ok]
        if not legal:
            reason = "NO_LEGAL_ACTION"; break
        good = teacher_candidates(current, witness, chains, teacher_mode)
        if not good:
            reason = "stuck"; break
        rows = np.stack([features.observe(current.candidates[i], chains, state=env.state, ctx=ctx,
                                          steps_left=max_steps - len(out), max_steps=max_steps)
                         for i in legal])
        out.append((rows, [legal.index(i) for i in good]))
        commit = [i for i in good if current.candidates[i].opcode is Opcode.COMMIT]
        pick = commit[0] if commit else good[0]
        current = env.step(current, pick, evaluate_training_reward=False).next_decision_or_terminal
    terminal = current if isinstance(current, TerminalRecord) else None
    valid = bool(terminal is not None and terminal.returned_valid)
    if reason is None:
        reason = "valid" if valid else "terminal"
    return out, reason, valid, _progress(task, env.state.chains), time.monotonic() - started


def clone_loss(actor, rows, good, temperature=1.0):
    """-log of the probability mass the policy puts on the witness-consistent candidates."""
    logits = actor(torch.as_tensor(rows, dtype=torch.float32)) / temperature
    total = torch.logsumexp(logits, dim=0)
    picked = torch.logsumexp(logits[torch.as_tensor(good, dtype=torch.long)], dim=0)
    return total - picked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--exploratory-repartition", action="store_true",
                    help="diagnostic only: ignore manifest partitions and randomly repartition "
                         "all corpus lineages, including reserved validation/test lineages")
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--heldout", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actor", choices=("linear", "mlp", "contextual"), default="linear")
    # Taken from the curriculum so a new schema does not have to be listed twice.
    ap.add_argument("--features", choices=tuple(cc.FEATURE_WIDTHS), default="tiny")
    ap.add_argument("--expand-features", action="store_true",
                    help="zero-pad a narrower checkpoint into this schema, so the added "
                         "channels start with no influence on any action logit")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--init", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--teacher-seconds", type=float, default=3600.)
    ap.add_argument("--teacher-mode", choices=("monotone", "witness_set"), default="monotone",
                    help="monotone: strict witness-subset chain extensions, COMMIT-only labels "
                         "when offered; witness_set: legacy subset-label ablation")
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--episode-seconds", type=float, default=600.)
    ap.add_argument("--support", choices=("registered", "wide"), default="wide")
    ap.add_argument("--stop-bias", type=float, default=0.0)
    ap.add_argument("--keep-incomplete", action="store_true",
                    help="teach stalled teacher walks too; off, because a walk that does "
                         "not finish teaches the prefix of a construction that fails")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if min(a.train, a.heldout, a.epochs, a.eval_episodes, a.eval_every, a.max_steps) < 1:
        ap.error("set sizes, epochs, evaluation and horizon must be positive")
    if a.learning_rate <= 0 or a.teacher_seconds <= 0 or a.episode_seconds <= 0:
        ap.error("learning rate and deadlines must be positive")

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    cells = [c for c in a.cells.split(",") if c]
    train, heldout, corpus = clone_corpus_sets(a, cells)
    forbidden = (corpus["manifest_lineages"]["validation"] + corpus["manifest_lineages"]["test"]
                 if corpus["manifest_split"] else [])
    provenance = training_provenance(a.init, train, heldout, forbidden_lineages=forbidden)
    diagnostics = []
    if a.exploratory_repartition:
        diagnostics.append("exploratory corpus repartition; manifest holdouts are not preserved")
    if not provenance["complete"]:
        diagnostics.append("initial checkpoint training-lineage provenance is incomplete or unknown")
    contract = dict(run_contract(a, train, heldout), **corpus)
    contract["version"] = "constructor-behaviour-cloning-v2"
    contract["source_files_sha256"] = {
        name: _sha256(ROOT / "probes" / name) for name in
        ("constructor_clone.py", "constructor_curriculum.py", "constructor_checkpoint.py",
         "constructor_rollout.py", "constructor_provenance.py", "witness_replay.py")}
    wide = a.support == "wide"
    actor = cc.make_actor(a.actor, a.width, cc.FEATURE_WIDTHS[a.features])
    if a.init:
        cc.load_init(a.init, actor, a.actor, a.features, expand=a.expand_features)
    cc.apply_terminal_bias(actor, a.features, a.stop_bias)
    optimizer = torch.optim.Adam(actor.parameters(), lr=a.learning_rate)
    features = {t.name: cc.make_features(a.features, t) for t in train + heldout}
    print(json.dumps({"protocol": "constructor-behaviour-cloning-v2", "corpus": a.corpus,
                      "contract": contract, "training_provenance": provenance,
                      "diagnostic": bool(diagnostics), "diagnostic_reasons": diagnostics,
                      "heldout_role": corpus["heldout_role"],
                      "scope": "feasibility development only; no quality or final-test claim",
                      "cells": cells or None, "train": [t.name for t in train],
                      "heldout": [t.name for t in heldout], "support": a.support,
                      "actor": a.actor, "features": a.features, "epochs": a.epochs,
                      "teacher_mode": a.teacher_mode,
                      "teacher": "set-valued over witness-consistent candidates; selected train lineages only",
                      "evaluation": "from empty, no witness, no completion solver",
                      "learning_rate": a.learning_rate, "stop_bias": a.stop_bias,
                      "max_steps": a.max_steps, "config": vars(a)}), flush=True)

    def evaluate(tag):
        rates = {}
        with torch.no_grad():
            for k, t in enumerate(heldout):
                from constructor_rollout import episode
                valid = 0
                for e in range(a.eval_episodes):
                    rec = episode(t, actor, features[t.name], 1., a.max_steps,
                                  np.random.default_rng(100000 + 1000 * k + e), a.episode_seconds,
                                  train=True, objective="feasibility", evaluate_reward=False, wide=wide)
                    valid += bool(rec["valid"])
                rates[t.name] = valid / a.eval_episodes
        row = {"evaluation": tag, "set": "heldout", "rate": float(np.mean(list(rates.values()))),
               "instances": len(heldout), "per_instance": rates,
               "heldout_role": corpus["heldout_role"], "diagnostic": bool(diagnostics),
               "diagnostic_reasons": diagnostics}
        print(json.dumps(row), flush=True)
        return row

    # one teacher trajectory per train instance, reused every epoch
    trajectories = []
    with no_completion_solver():
        for t in train:
            witness = {v: frozenset(c) for v, c in (t.prefix_source or {}).items()}
            if not witness:
                print(json.dumps({"skipped": t.name, "why": "no witness on this task"}), flush=True)
                continue
            steps, reason, valid, progress, secs = teacher_steps(
                t, witness, features[t.name], a.max_steps, a.teacher_seconds, wide,
                teacher_mode=a.teacher_mode)
            print(json.dumps({"teacher": t.name, "lineage": t.lineage,
                              "steps": len(steps), "reason": reason,
                              "teacher_mode": a.teacher_mode,
                              "blocked_support": reason in ("stuck", "NO_LEGAL_ACTION"),
                              "valid": valid, "progress": progress, "seconds": secs}), flush=True)
            # Only trajectories that reached a valid COMMIT are taught. A walk that stalls
            # part way is still a sequence of legal actions, and keeping it teaches the actor to
            # reproduce the prefix of a construction that does not finish, which is the habit the
            # policy already has. Requiring validity here costs teacher coverage and is worth it.
            if steps and (valid or a.keep_incomplete):
                trajectories.append(steps)
            elif steps:
                print(json.dumps({"dropped": t.name, "why": "teacher did not reach a valid COMMIT",
                                  "reason": reason, "progress": progress}), flush=True)
        if not trajectories:
            raise SystemExit("no teacher trajectory reached a valid COMMIT")
        evaluate("init")
        order = np.random.default_rng(a.seed + 1)
        for epoch in range(a.epochs):
            losses = []
            for index in order.permutation(len(trajectories)):
                steps = trajectories[int(index)]
                optimizer.zero_grad()
                loss = torch.stack([clone_loss(actor, rows, good) for rows, good in steps]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)),
                              "trajectories": len(trajectories)}), flush=True)
            if (epoch + 1) % a.eval_every == 0 and epoch + 1 < a.epochs:
                evaluate("epoch %d" % epoch)
        final = evaluate("final")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": actor.state_dict(), "actor": a.actor, "width": a.width,
                    "features": a.features, "support": a.support, "seed": a.seed,
                    "contract": contract, "training_provenance": provenance,
                    "diagnostic": bool(diagnostics), "diagnostic_reasons": diagnostics,
                    "summary": {"summary": "clone", "heldout": final}}, a.out)
    print("CLONE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

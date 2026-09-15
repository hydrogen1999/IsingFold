#!/usr/bin/env python3
"""Train the specified policy by imitating its own best rollouts.

The headroom probe measures, for each development lineage, the spread of terminal utility over
many random rollouts: the best of forty-eight sits +0.23 to +0.42 above their mean. So the action
space contains good trajectories and the problem is finding them. Policy gradients see that
spread as variance and average it away, which is what the two PPO runs did: no arm's training
return moved outside noise.

This is the AlphaGo Zero loop instead. Each round samples K complete episodes per lineage under
the current policy, keeps the single best by terminal reward, and trains the policy by cross
entropy on the actions that trajectory took. Search generates the target; the network learns to
reach it without the search. The loss is low variance because it is supervised, and the target
improves every round because the policy that generates the rollouts improves.

    python3 probes/train_bestof.py --corpus runs/corpus_c4x10 --family if-core \
        --rounds 40 --lineages 24 --k 8 --out runs/bestof_if-core_s0
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

_pin = os.environ.get("ISINGFOLD_SRC")
if _pin:
    sys.path.insert(0, _pin)
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                                and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]

import torch
import torch.nn.functional as F

from isingfold.rl.contracts import Context, InitFailureRecord, Mode
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import (LEGACY_ONLINE_INITIALIZER_RESTARTS_V1, EmbeddingEnv,
                              fixed_strength_selector)
from isingfold.rl.evaluate import (first_commit_controller, random_masked_controller,
                                   run_controller, secondary_metrics, torch_controller)
from isingfold.rl.model import build_model
from isingfold.rl.ppo import PPOConfig, collect
from isingfold.rl.proposal import router_initializer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _initializers import pick_initializer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--require-gain", action="store_true", default=True,
                    help="teach only on rollouts that beat returning their own starting "
                         "embedding, measured by a control block that shares no reads with them")
    ap.add_argument("--no-require-gain", dest="require_gain", action="store_false")
    ap.add_argument("--gain-margin", type=float, default=0.0,
                    help="how far a rollout must beat its control before it is taught on")
    ap.add_argument("--initializer", default="router", choices=["router", "minorminer"],
                    help="what protects the episode floor. 'minorminer' starts every "
                         "episode from the standard tool's embedding, so the policy is "
                         "learning to improve on it rather than to replace it")
    ap.add_argument("--mm-tries", type=int, default=10,
                    help="minorminer's own internal tries, when it is the initializer")
    ap.add_argument("--family", default="if-core")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--lineages", type=int, default=24, help="training lineages sampled per round")
    ap.add_argument("--k", type=int, default=8, help="rollouts per lineage per round")
    ap.add_argument("--epochs", type=int, default=2, help="passes over the kept trajectories")
    ap.add_argument("--window", type=int, default=10,
                    help="rounds of best trajectories kept for training; one round of 24 lineages "
                         "yields about 130 transitions, which is too few for a gradient step on "
                         "its own and is forgotten by the next round")
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-reads", type=int, default=128)
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-instances", type=int, default=40)
    ap.add_argument("--init-attempts", type=int, default=32)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    train_roots, dev_roots = set(split["train"]), set(split["validation"]) | set(split["test"])
    ctx = Context(qubit_cap=a.qubit_cap)
    initializer = pick_initializer(a.initializer, a.mm_tries)
    selector = fixed_strength_selector()

    def make_env(task, seed: int, chains=None):
        """Build one episode's environment.

        `chains` pins the starting embedding. The K rollouts of one lineage must share it: with
        a fresh initializer draw per episode, the best of K is partly the best initializer draw,
        and copying the winner's actions teaches the policy to take credit for a good start it
        did not produce. An external audit made this point with a worked counterexample, and it
        is right. The episode seed still varies, so the policy and the evaluator keep their own
        randomness; only the starting state is held fixed.
        """
        init = initializer if chains is None else (lambda l, h, sd, c=chains: c)
        return EmbeddingEnv(task, ctx, mode=Mode.IMPROVEMENT, initializer=init,
                            selector=selector, reward_reads=a.reward_reads, seed=seed,
                            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1)

    def shared_initial(task, seed: int):
        """Draw one starting embedding and measure what returning it unchanged is worth.

        The measurement is the control every gain is stated against. It uses its own seed range
        so that no read can serve both as the control and as a rollout's terminal block.
        """
        env = make_env(task, seed)
        rec = env.reset(seed)
        if isinstance(rec, InitFailureRecord):
            return None, None
        chains = {n: frozenset(c) for n, c in env.state.chains.items()}
        out = run_controller([task], ctx, first_commit_controller,
                             initializer=(lambda l, h, sd, c=chains: c), selector=selector,
                             reward_reads=a.reward_reads, repetitions=1,
                             seed=60_000_000 + seed)
        o = out[0]
        base = float(o.utility) if o.returned_valid and o.utility is not None else None
        return chains, base

    train = [t for t in tasks if t.lineage in train_roots]
    train = [t for t in train
             if all(not isinstance(make_env(t, 1000 + k).reset(), InitFailureRecord) for k in range(3))]
    dev = [t for t in tasks if t.lineage in dev_roots][: a.eval_instances]
    print(json.dumps({"train": len(train), "dev": len(dev), "device": str(device)}), flush=True)

    torch.manual_seed(a.seed)
    model = build_model(a.family, improvement_mode=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    common = dict(initializer=initializer, selector=selector, reward_reads=a.reward_reads,
                  repetitions=1)
    rng = np.random.default_rng(a.seed)
    replay: list[list] = []
    history = []
    t0 = time.time()

    for rnd in range(a.rounds):
        picked = [train[int(i)] for i in rng.integers(0, len(train), a.lineages)]
        kept, stats = [], []
        gains, reassessed, verified, attempted = [], [], 0, 0
        for j, task in enumerate(picked):
            init_seed = 30_000_000 + 7919 * rnd + j
            chains, base = shared_initial(task, init_seed)
            if chains is None or base is None:
                continue
            attempted += 1
            cfg = PPOConfig(episodes_per_batch=a.k, seed=a.seed + 7919 * rnd + j,
                            init_attempt_cap=a.init_attempts)
            model.eval()
            buffer, _ = collect(lambda s, idx=0, t=task, c=chains: make_env(t, s, c), model, cfg,
                                update_index=rnd, device=device)
            if not buffer.n_episodes:
                continue
            rewards = [e.terminal_reward for e in buffer.episodes]
            best = int(np.argmax(rewards))
            stats.append((float(np.max(rewards)), float(np.mean(rewards))))
            gains.append(float(np.max(rewards)) - base)
            # The line above is the selection maximum minus an independent control, so it still
            # carries the inflation a maximum over K noisy blocks produces: about +0.06 at K=8
            # and 128 reads even when every candidate is identical. Re-measuring the winner on
            # reads that took no part in choosing it gives the gain without that term.
            won = buffer.episodes[best]
            chains_out = getattr(won, "returned_embedding", None) or getattr(won, "embedding", None)
            if chains_out is not None:
                fresh = run_controller(
                    [task], ctx, first_commit_controller,
                    initializer=(lambda l, h, sd, c=chains_out: {n: frozenset(v) for n, v in c.items()}),
                    selector=selector, reward_reads=a.reward_reads, repetitions=1,
                    seed=70_000_000 + init_seed)[0]
                if fresh.returned_valid and fresh.utility is not None:
                    reassessed.append(float(fresh.utility) - base)
            # A winning rollout that did not beat returning its own starting embedding is not a
            # teacher. Without this the target is whichever episode drew the luckiest read
            # block, and a synthetic control shows that alone manufactures about +0.06 of
            # apparent headroom at K=8 and 128 reads when every candidate is identical.
            if a.require_gain and np.max(rewards) <= base + a.gain_margin:
                continue
            verified += 1
            # Episode.transitions holds indices into the buffer, not the rows themselves.
            kept.extend(buffer.transitions[int(i)] for i in buffer.episodes[best].transitions)
        if not kept:
            continue
        replay.append(kept)
        if len(replay) > a.window:
            replay.pop(0)
        pool = [row for block in replay for row in block]
        model.train()
        losses = []
        for _ in range(a.epochs):
            order = rng.permutation(len(pool))
            for start in range(0, len(pool), a.minibatch):
                rows = [pool[int(i)] for i in order[start : start + a.minibatch]]
                outputs = model([r.observation for r in rows], device=device)
                chosen = torch.as_tensor([r.chosen_index for r in rows], dtype=torch.long, device=device)
                logp = outputs.masked_log_probs[torch.arange(len(rows), device=device), chosen]
                loss = -logp.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
        best_mean = float(np.mean([s[0] for s in stats])); roll_mean = float(np.mean([s[1] for s in stats]))
        rec = dict(round=rnd, kept_transitions=len(kept), pool=len(pool), loss=float(np.mean(losses)),
                   rollout_best=best_mean, rollout_mean=roll_mean,
                   gain_over_initial=float(np.mean(gains)) if gains else None,
                   reassessed_gain=float(np.mean(reassessed)) if reassessed else None,
                   verified_lineages=verified, attempted_lineages=attempted,
                   seconds=round(time.time() - t0))
        if rnd % a.eval_every == 0 or rnd == a.rounds - 1:
            model.eval()
            arm = run_controller(dev, ctx, torch_controller(model, device), seed=17, **common)
            base = run_controller(dev, ctx, first_commit_controller, seed=17, **common)
            rec["dev_policy"] = secondary_metrics(arm)["utility_mean"]
            rec["dev_initial"] = secondary_metrics(base)["utility_mean"]
            torch.save(model.state_dict(), out / "policy.pt")
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=1, default=float))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

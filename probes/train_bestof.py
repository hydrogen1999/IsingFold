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

from isingfold.rl.contracts import Context, InitFailureRecord, Mode, Opcode
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



def hindsight_prefix(env_factory, rows, target, max_steps=64):
    """Replay a winning episode and stop the moment its output could have been returned.

    The reason this exists: the winner's trajectory is cloned in full, and a third external audit
    measured that 87 of 154 non-terminal actions in such trajectories happen after the returned
    embedding already existed, while 29 of 32 commits take an archived embedding rather than the
    final workspace. More than half of what the teacher teaches did not produce the output.

    The audit is also explicit that truncating the transition list and reusing the old commit
    index is wrong: at an earlier decision the masks, the archive indices and the remaining
    budget are all different, so the old index may point at another action or at none. So this
    replays the episode step by step in a fresh environment and, at each decision, asks whether
    some legal candidate would return exactly the target embedding. The first decision where one
    does is where the episode should have stopped.

    Returns (prefix_length, commit_index, decision) or None when no such decision exists, which
    happens when the target only becomes reachable through the actions that were actually taken.
    """
    want = {n: frozenset(c) for n, c in target.items()}
    env = env_factory()
    dec = env.reset()
    actions = [r.chosen_index for r in rows]
    for step in range(min(len(actions), max_steps) + 1):
        # Replaying with the wrong seed produces a different candidate list, and the recorded
        # action index then points at another action or outside the support entirely. Rather
        # than trust the reconstruction, check it: the environment stamps each decision with a
        # support fingerprint, and the recorded transition carries the one that held when the
        # action was chosen. A mismatch means this is not the same episode.
        if step < len(rows) and hasattr(dec, "support_fingerprint"):
            recorded = getattr(rows[step], "support_fingerprint", "")
            if recorded and dec.support_fingerprint != recorded:
                return None
        if not hasattr(dec, "candidates"):
            return None
        # Only a COMMIT returns anything, and what it returns is the archive entry it names,
        # not the workspace. The first version of this matched any legal candidate whose
        # workspace-plus-new_chains equalled the target and handed that index back as the
        # commit: on a fixture whose winner was REWRITE_ONE then COMMIT it produced a teacher of
        # one REWRITE_ONE labelled as the commit, and on an archive with two entries it picked
        # the wrong one. A fourth external audit reproduced both.
        for i, (cand, ok) in enumerate(zip(dec.candidates, dec.legal_mask)):
            if not ok or cand.opcode is not Opcode.COMMIT:
                continue
            ref = cand.archive_ref
            if ref is None or ref >= len(env.state.archive):
                continue
            entry = {n: frozenset(c) for n, c in env.state.archive[ref].chains.items()}
            if entry == want:
                return step, i, dec
        if step >= len(actions):
            return None
        nxt = env.step(dec, int(actions[step]),
                       evaluate_training_reward=False).next_decision_or_terminal
        if not hasattr(nxt, "candidates"):
            return None
        dec = nxt
    return None


class _TeachRow:
    """The two fields the cross-entropy update reads, for a decision that was never taken."""

    __slots__ = ("observation", "chosen_index")

    def __init__(self, observation, chosen_index):
        self.observation = observation
        self.chosen_index = int(chosen_index)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dev-split", default="validation",
                    choices=["validation", "test", "both"],
                    help="which lineages the in-training curve is measured on. Keep it at "
                         "validation: the curve is looked at repeatedly and steers decisions, "
                         "so whatever it reads stops being a held-out set")
    ap.add_argument("--strict-prefix", action="store_true",
                    help="when no commit for the returned embedding is found along the replay, "
                         "drop the lineage instead of falling back to the whole trajectory, so "
                         "a prefix arm measures the prefix teacher and nothing else")
    ap.add_argument("--teacher", default="prefix", choices=["prefix", "whole"],
                    help="'whole' clones every action of the winning rollout, including the "
                         "ones taken after its output already existed. 'prefix' replays the "
                         "episode and teaches the shortest sequence that could have returned "
                         "that output, plus the commit that returns it")
    ap.add_argument("--require-gain", action="store_true", default=True,
                    help="teach only on rollouts whose re-measured gain over their own starting "
                         "embedding is positive. The re-measurement shares no reads with the "
                         "block that made the rollout the winner, which is the point")
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
    train_roots = set(split["train"])
    # Validation only, by default. Pooling test in here and watching the curve every few rounds
    # is how a held-out set stops being held out: no gradient touches it, but every decision
    # about rounds, learning rate and which checkpoint to keep is made against it. The external
    # audit called this correctly. Anything measured on this set is a development diagnostic.
    dev_roots = set(split[a.dev_split]) if a.dev_split != "both" else (
        set(split["validation"]) | set(split["test"]))
    ctx = Context(qubit_cap=a.qubit_cap)
    initializer = pick_initializer(a.initializer, a.mm_tries)
    selector = fixed_strength_selector()

    def make_env(task, seed: int, chains=None):
        """Build one episode's environment.

        `chains` pins the starting embedding. The K rollouts of one lineage must share it: with
        a fresh initializer draw per episode, the best of K is partly the best initializer draw,
        and copying the winner's actions teaches the policy to take credit for a good start it
        did not produce. The episode seed still varies, so the policy and the evaluator keep
        their own randomness; only the starting state is held fixed.

        The environment calls one initializer for two different jobs: it produces the protected
        starting embedding, and RESTART draws from it too. Pinning both to one embedding made
        RESTART a move that returns where the episode began, while at deployment the same action
        draws a fresh minorminer embedding. The policy then trains against an action it will not
        meet. So the two are separated here: the first call returns the shared start, and every
        later call goes to the real initializer, which is what RESTART is supposed to reach.
        An external audit found this and reproduced the difference in candidate support.
        """
        if chains is None:
            init = initializer
        else:
            state = {"served": False}

            def init(logical, host, sd, c=chains, st=state):
                if not st["served"]:
                    st["served"] = True
                    return c
                return initializer(logical, host, sd)

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
    print(json.dumps({"train": len(train), "dev": len(dev), "dev_split": a.dev_split,
                      "device": str(device)}), flush=True)

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
        selected, remeasured, accepted, attempted = [], [], 0, 0
        skipped_no_embedding = skipped_no_fresh = prefix_not_found = 0
        taught_steps, full_steps = [], []
        for j, task in enumerate(picked):
            init_seed = 30_000_000 + 7919 * rnd + j
            chains, base = shared_initial(task, init_seed)
            if chains is None or base is None:
                continue
            attempted += 1
            cfg = PPOConfig(episodes_per_batch=a.k, seed=a.seed + 7919 * rnd + j,
                            init_attempt_cap=a.init_attempts)
            model.eval()
            # The collector derives a seed per episode and hands it to the factory; the prefix
            # replay needs that exact seed, so it is recorded as the factory is called rather
            # than reconstructed from the collector\'s internals.
            episode_seeds: dict[int, int] = {}

            def factory(s, idx=0, t=task, c=chains, store=episode_seeds):
                store[int(idx)] = int(s)
                return make_env(t, s, c)

            buffer, _ = collect(factory, model, cfg, update_index=rnd, device=device)
            if not buffer.n_episodes:
                continue
            rewards = [e.terminal_reward for e in buffer.episodes]
            best = int(np.argmax(rewards))
            stats.append((float(np.max(rewards)), float(np.mean(rewards))))
            selected.append(float(np.max(rewards)) - base)
            # The selection maximum minus the control still carries what a maximum over K noisy
            # blocks invents: about +0.06 at K=8 and 128 reads even when every candidate is
            # identical. It is logged, and it decides nothing. The gate below uses the winner
            # re-measured on reads that took no part in choosing it, because a gate driven by
            # the selection block admits exactly the rollouts that got lucky in it. The first
            # version of this code logged the honest number and gated on the noisy one; an
            # external audit reproduced a case where the winner's fresh measurement was 0.10
            # below its control and the trajectory was taught on anyway.
            won = buffer.episodes[best]
            chains_out = getattr(won, "returned_embedding", None)
            if chains_out is None:
                skipped_no_embedding += 1
                continue
            fresh = run_controller(
                [task], ctx, first_commit_controller,
                initializer=(lambda l, h, sd, c=chains_out: {n: frozenset(v) for n, v in c.items()}),
                selector=selector, reward_reads=a.reward_reads, repetitions=1,
                seed=70_000_000 + init_seed)[0]
            if not fresh.returned_valid or fresh.utility is None:
                skipped_no_fresh += 1
                continue
            fresh_gain = float(fresh.utility) - base
            remeasured.append(fresh_gain)
            # A winning rollout that did not beat returning its own starting embedding is not a
            # teacher. Without this the target is whichever episode drew the luckiest read
            # block, and a synthetic control shows that alone manufactures about +0.06 of
            # apparent headroom at K=8 and 128 reads when every candidate is identical.
            if a.require_gain and fresh_gain <= a.gain_margin:
                continue
            accepted += 1
            rows_idx = [int(i) for i in buffer.episodes[best].transitions]
            episode_rows = [buffer.transitions[i] for i in rows_idx]
            if a.teacher == "whole":
                kept.extend(episode_rows)
                taught_steps.append(len(episode_rows))
                full_steps.append(len(episode_rows))
                continue
            # Teach the shortest sequence that could have produced this output, and the commit
            # that returns it, rather than everything the rollout happened to do afterwards.
            # The index the collector hands the factory is its own schedule counter, not the
            # buffer index of the episode, and the two stop agreeing once episodes finish out of
            # order. Rather than guess the mapping, try the seeds it used and let the support
            # fingerprints say which one replays this episode. There are only K of them, the
            # replay is cheap, and a wrong seed is rejected rather than silently accepted.
            candidate_seeds = [episode_seeds.get(best)] + list(episode_seeds.values())
            found = None
            for sd_try in [x for x in candidate_seeds if x is not None]:
                found = hindsight_prefix(
                    lambda t=task, c=chains, sd=sd_try: make_env(t, sd, c),
                    episode_rows, chains_out)
                if found is not None:
                    break
            if found is None:
                # A prefix ablation that silently falls back to the whole trajectory is not an
                # ablation. The fallback is counted and, when --strict-prefix is set, the
                # lineage is dropped instead of being taught the thing under test.
                prefix_not_found += 1
                if a.strict_prefix:
                    continue
                kept.extend(episode_rows)
                taught_steps.append(len(episode_rows))
                full_steps.append(len(episode_rows))
                continue
            cut, commit_index, dec = found
            kept.extend(episode_rows[:cut])
            kept.append(_TeachRow(dec.observation, commit_index))
            taught_steps.append(cut + 1)
            full_steps.append(len(episode_rows))
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
                   selected_gain=float(np.mean(selected)) if selected else None,
                   remeasured_gain=float(np.mean(remeasured)) if remeasured else None,
                   accepted_teacher=accepted, attempted_lineages=attempted,
                   skipped_no_embedding=skipped_no_embedding,
                   skipped_no_fresh=skipped_no_fresh,
                   teacher=a.teacher, prefix_not_found=prefix_not_found,
                   taught_steps=float(np.mean(taught_steps)) if taught_steps else None,
                   episode_steps=float(np.mean(full_steps)) if full_steps else None,
                   seconds=round(time.time() - t0))
        if rnd % a.eval_every == 0 or rnd == a.rounds - 1:
            model.eval()
            arm = run_controller(dev, ctx, torch_controller(model, device), seed=17, **common)
            base = run_controller(dev, ctx, first_commit_controller, seed=17, **common)
            rec["dev_split"] = a.dev_split
            rec["dev_policy"] = secondary_metrics(arm)["utility_mean"]
            rec["dev_initial"] = secondary_metrics(base)["utility_mean"]
            torch.save(model.state_dict(), out / "policy.pt")
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=1, default=float))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

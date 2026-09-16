"""Reinforcement learning of the hybrid: the policy places roots, minorminer completes, the
reward is whether the completion succeeds within a deadline.

Half of the witness's roots make minorminer succeed where it fails from scratch, and roots
agreeing with one particular witness are not the point: minorminer needs a globally
consistent layout, which a locally trained scorer does not give. So the layout is learned
against the only signal that matters. An episode: PLACE-only construction from an empty
embedding, one root per variable sampled from the prioritiser's softmax over the offered
PLACE candidates; then minorminer with those roots as initial chains, restarted until a
short deadline; reward 1 if a valid embedding came back, else 0. REINFORCE with the mean
over K episodes of the instance as baseline. Evaluation is the deployment protocol on
held-out instances: greedy roots, then minorminer until the full deadline.
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
from isingfold.rl.contracts import DecisionState, Opcode
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector

from _context import construction_context, qubit_budget
from candidate_features import FeatureContext
from seeded_minorminer import attempt
from train_constructor_rl import candidate_tuple
from train_prioritiser import Prioritiser

PLACE_ONLY = {"place": 64, "route": 0, "grow": 0, "shrink": 0}


def place_roots(task, model, fc, temperature, rng, train):
    ctx = construction_context(task.host.number_of_nodes(), task.logical.number_of_nodes(),
                               task.logical.number_of_edges(), quotas=PLACE_ONLY)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8)
    dec = env.reset(int(rng.integers(0, 2 ** 31)))
    logps, steps = [], 0
    n = task.logical.number_of_nodes()
    while isinstance(dec, DecisionState) and steps < 4 * n + 8:
        chains = env.state.chains
        idx = [i for i, ok in enumerate(dec.legal_mask) if ok and dec.candidates[i].opcode is Opcode.PLACE]
        if not idx:
            break
        feats = np.stack([fc.candidate(candidate_tuple(dec.candidates[i]), chains) for i in idx])
        scores = model(torch.as_tensor(feats)) / temperature
        if train:
            dist = torch.distributions.Categorical(logits=scores)
            j = int(dist.sample())
            logps.append(dist.log_prob(torch.tensor(j)))
        else:
            j = int(torch.argmax(scores))
        dec = env.step(dec, idx[j], evaluate_training_reward=False).next_decision_or_terminal
        steps += 1
    roots = {v: frozenset(c) for v, c in env.state.chains.items() if c}
    return roots, logps


def complete(task, roots, deadline, tries, seed):
    t0, ok, n = time.time(), None, 0
    while ok is None and time.time() - t0 < deadline:
        n += 1
        ok = attempt(task, roots, seed + n, tries)
    return ok is not None, time.time() - t0, n


def partial_score(task, roots, seed, tries=2):
    """A graded score for a failed completion: the router with overlaps allowed returns an
    embedding in which contested qubits are shared; the fraction of variables whose chain
    shares no qubit is how close the layout came. Without it every episode of a hard
    instance scores zero and REINFORCE has no gradient at all, which is what the first run
    showed: no update in nine iterations."""
    import minorminer
    edges = list(task.logical.edges())
    in_edges = {u for e in edges for u in e}
    kw = {"tries": tries, "random_seed": seed % (2 ** 31), "return_overlap": True}
    if roots:
        kw["initial_chains"] = {v: list(c) for v, c in roots.items() if v in in_edges}
    emb, _ = minorminer.find_embedding(edges, list(task.host.edges()), **kw)
    if not emb:
        return 0.0
    owners = {}
    for v, c in emb.items():
        for q in c:
            owners.setdefault(q, set()).add(v)
    clean = sum(1 for v, c in emb.items() if all(len(owners[q]) == 1 for q in c))
    return clean / max(1, len(in_edges))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--instances-per-iteration", type=int, default=4)
    ap.add_argument("--episodes-per-instance", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--train-deadline", type=float, default=10.0)
    ap.add_argument("--eval-deadline", type=float, default=60.0)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--holdout-fraction", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-only", action="store_true",
                    help="train on cells where unseeded minorminer is not already at one")
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    names = sorted(t.name for t in tasks)
    held = set(rng.choice(names, size=max(1, int(len(names) * a.holdout_fraction)), replace=False))
    train_tasks = [t for t in tasks if t.name not in held]
    eval_tasks = [t for t in tasks if t.name in held]
    if a.hard_only:
        train_tasks = [t for t in train_tasks if not t.name.split("-")[1].startswith("fill70")]
    model = Prioritiser(a.width)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["state"])
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    fcs = {}

    def fc_for(task):
        if task.name not in fcs:
            fcs[task.name] = FeatureContext(task, qubit_budget({v: frozenset(c) for v, c in task.witness.items()}))
        return fcs[task.name]

    print(json.dumps({"corpus": a.corpus, "train": len(train_tasks), "held_out": len(eval_tasks),
                      "init": a.init or None, "train_deadline": a.train_deadline,
                      "eval_deadline": a.eval_deadline}), flush=True)

    def evaluate(tag):
        """Policy plus search: layouts are sampled from the policy until the deadline, each
        handed to the router with a short budget, and the instance counts as solved if any
        layout completes. The baseline is the router alone restarted until the same
        deadline. A single greedy layout would give the baseline a search the policy does
        not get."""
        model.eval()
        cells = defaultdict(list)
        for k, t in enumerate(eval_tasks):
            t0, ok, layouts = time.time(), False, 0
            while not ok and time.time() - t0 < a.eval_deadline:
                roots, _ = place_roots(t, model, fc_for(t), a.temperature, rng, train=True)
                layouts += 1
                left = a.eval_deadline - (time.time() - t0)
                ok, _, _ = complete(t, roots, min(a.train_deadline, max(0.5, left)), a.tries,
                                    70_000 + 1000 * k + 13 * layouts)
            ok0, _, _ = complete(t, None, a.eval_deadline, a.tries, 80_000 + 1000 * k)
            cells[t.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]].append((ok, ok0, layouts))
        model.train()
        allr = [r for rs in cells.values() for r in rs]
        print("  %s held-out within %.0fs: policy+search %.2f  router alone %.2f  layouts tried %.1f  over %d"
              % (tag, a.eval_deadline, np.mean([r[0] for r in allr]), np.mean([r[1] for r in allr]),
                 np.mean([r[2] for r in allr]), len(allr)), flush=True)
        print("    " + "  ".join("%s %.2f/%.2f" % (c, np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]))
                                 for c, rs in sorted(cells.items())), flush=True)
        return float(np.mean([r[0] for r in allr]))

    best = -1.0
    evaluate("init")
    for it in range(a.iterations):
        batch = rng.choice(len(train_tasks), size=min(a.instances_per_iteration, len(train_tasks)), replace=False)
        loss, n, wins, secs = 0.0, 0, [], []
        for idx in batch:
            t = train_tasks[idx]
            eps = []
            for e in range(a.episodes_per_instance):
                roots, logps = place_roots(t, model, fc_for(t), a.temperature, rng, train=True)
                ok, s, _ = complete(t, roots, a.train_deadline, a.tries, 90_000 + 7 * e + 100 * it)
                r = 1.0 if ok else 0.5 * partial_score(t, roots, 91_000 + 7 * e + 100 * it)
                eps.append((r, logps)); wins.append(ok); secs.append(s)
            base = np.mean([r for r, _ in eps])
            for r, logps in eps:
                if logps and abs(r - base) > 1e-9:
                    loss = loss - (r - base) * torch.stack(logps).sum() / len(logps)
                    n += 1
        if n:
            opt.zero_grad(); (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print("  iter %4d  train success %.2f  mm secs %.1f  updates %d"
              % (it, np.mean(wins), np.mean(secs), n), flush=True)
        if (it + 1) % a.eval_every == 0:
            v = evaluate("iter %d" % it)
            if v > best:
                best = v
                torch.save({"state": model.state_dict(), "width": a.width}, a.out)
    torch.save({"state": model.state_dict(), "width": a.width}, a.out + ".last")
    print("\nHYBRID RL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

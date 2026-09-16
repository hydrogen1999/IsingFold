"""Learn where to spend a qubit: contact growth chosen by a policy, rewarded by the objective.

Under the registered schedule, growing a chain toward the chains of its coupled variables
is the one way of spending qubits that does not hurt the measured objective, and it is
slightly positive at small spends; lengthening and redundancy hurt. Which contacts to add
is therefore the decision worth learning. Starting from the router's embedding, an episode
adds m qubits one at a time, each a free qubit adjacent to one chain and adjacent to the
chain of a coupled variable, chosen by a softmax over the prioritiser's scores of the
candidate (variable, qubit) pairs; the reward is the measured utility of the grown
embedding minus the start's, under the registered schedule, on an independent block.
REINFORCE with the instance's mean over K episodes as baseline.

Evaluation on held-out lineages, matched reads: policy best of K episodes selected by a
measurement block, random contact growth best of K the same way, and the start, all
assessed on a fresh block. Policy minus random is the learned contribution; both minus the
start is what the spend is worth at all.
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
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller

from _context import host_context
from _initializers import minorminer_initializer
from candidate_features import FRONTIER_WIDTH, WIDTH, FeatureContext, frontier_features
from train_prioritiser import Prioritiser

SELECT_BASE, ASSESS_BASE = 41_000_000, 42_000_000


def contact_moves(task, chains, occupied, cap_per_var=6):
    """Candidate additions: a free qubit adjacent to chain(v) that touches the chain of a
    logical neighbour of v. Returned as (v, r, touches)."""
    host, logical = task.host, task.logical
    out = []
    for v, chain in chains.items():
        seen = set()
        for q in chain:
            for r in host.neighbors(q):
                if r in occupied or r in seen:
                    continue
                seen.add(r)
                t = 0
                for u in logical.neighbors(v):
                    if any(host.has_edge(r, x) for x in chains[u]):
                        t += 1
                if t > 0:
                    out.append((v, r, t))
        # keep the strongest few per variable so the softmax is over a bounded set
    by_var = defaultdict(list)
    for v, r, t in out:
        by_var[v].append((v, r, t))
    kept = []
    for v, lst in by_var.items():
        kept += sorted(lst, key=lambda x: (-x[2], str(x[1])))[:cap_per_var]
    return kept


def episode(task, start, model, fc, m, temperature, rng, train=True):
    chains = {v: set(c) for v, c in start.items()}
    occupied = {q for c in chains.values() for q in c}
    logps = []
    for _ in range(m):
        moves = contact_moves(task, chains, occupied)
        if not moves:
            break
        frozen = {v: frozenset(c) for v, c in chains.items()}
        feats = np.stack([np.concatenate([fc.pair(v, [r], frozen, "ROUTE"),
                                          frontier_features(task.host, task.logical, frozen, v, r)])
                          for v, r, _ in moves]).astype(np.float32)
        scores = model(torch.as_tensor(feats)) / temperature
        if train:
            dist = torch.distributions.Categorical(logits=scores)
            j = int(dist.sample()); logps.append(dist.log_prob(torch.tensor(j)))
        else:
            j = int(rng.integers(0, len(moves))) if model is None else int(torch.argmax(scores))
        v, r, _ = moves[j]
        chains[v].add(r); occupied.add(r)
    return {v: frozenset(c) for v, c in chains.items()}, logps


def random_episode(task, start, m, rng):
    chains = {v: set(c) for v, c in start.items()}
    occupied = {q for c in chains.values() for q in c}
    for _ in range(m):
        moves = contact_moves(task, chains, occupied)
        if not moves:
            break
        v, r, _ = moves[int(rng.integers(0, len(moves)))]
        chains[v].add(r); occupied.add(r)
    return {v: frozenset(c) for v, c in chains.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--qubit-cap", type=int, default=248)
    ap.add_argument("--spend", type=float, default=0.10, help="qubits added as a fraction of the start")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--instances-per-iteration", type=int, default=4)
    ap.add_argument("--episodes-per-instance", type=int, default=4)
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--assess-reads", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--train-lineages", type=int, default=60)
    ap.add_argument("--eval-lineages", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-k", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="router", choices=("router", "witness"),
                    help="witness: begin from the planted valid embedding, which on the fill "
                         "corpora occupies 80 to 95 percent of the host; the budget is then "
                         "the space that is left")
    ap.add_argument("--objective", default="utility", choices=("utility", "residual"),
                    help="residual: the mean energy residual, lower is better, for scales "
                         "where solve probability reads zero")
    ap.add_argument("--fail-penalty", type=float, default=0.05,
                    help="an episode that exceeds the budget or cannot be measured counts as the "
                         "start's value minus this, not as missing")
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    lineages = sorted({t.lineage for t in tasks})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(lineages)
    train_l = set(lineages[: a.train_lineages]); eval_l = set(lineages[a.train_lineages: a.train_lineages + a.eval_lineages])
    train_tasks = [t for t in tasks if t.lineage in train_l][: a.train_lineages]
    eval_tasks = [t for t in tasks if t.lineage in eval_l][: a.eval_lineages]
    ctx = host_context(a.qubit_cap)
    mm = minorminer_initializer(20)
    torch.manual_seed(a.seed)
    model = Prioritiser(a.width, in_dim=WIDTH + FRONTIER_WIDTH)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["state"])
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    print(json.dumps({"corpus": a.corpus, "train": len(train_tasks), "held_out": len(eval_tasks),
                      "spend": a.spend, "start": a.start, "objective": a.objective,
                      "beta_range": list(ctx.beta_range)}), flush=True)

    def measure(task, chains, seed, reads):
        def fixed(l, h, s):
            return chains
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=reads,
                                 repetitions=1, seed=seed)
        except Exception:
            return None
        o = out[0]
        if not o.returned_valid:
            return None
        if a.objective == "residual":
            # sign flipped so that higher is better everywhere below
            return -float(o.mean_energy_residual) if o.mean_energy_residual is not None else None
        return float(o.utility) if o.utility is not None else None

    starts, fcs = {}, {}

    def start_of(task, k):
        if task.name not in starts:
            if a.start == "witness":
                ch = {v: frozenset(c) for v, c in task.witness.items()}
            else:
                ch = mm(task.logical, task.host, 5000 + k)
            starts[task.name] = ch
            fcs[task.name] = FeatureContext(task, a.qubit_cap)
        return starts[task.name], fcs[task.name]

    def spend_of(start):
        used = sum(len(c) for c in start.values())
        return max(1, min(int(round(a.spend * used)), a.qubit_cap - used))

    def evaluate(tag):
        model.eval()
        rows = []
        for k, t in enumerate(eval_tasks):
            start, fc = start_of(t, 9000 + k)
            if start is None:
                continue
            m = spend_of(start)
            arms = {}
            for name in ("policy", "random"):
                cands = []
                for e in range(a.eval_k):
                    if name == "policy":
                        ch, _ = episode(t, start, model, fc, m, a.temperature, rng, train=True)
                    else:
                        ch = random_episode(t, start, m, rng)
                    if sum(len(c) for c in ch.values()) > a.qubit_cap:
                        continue
                    u = measure(t, ch, SELECT_BASE + 1000 * k + 7 * e + (0 if name == "policy" else 500), a.reads)
                    if u is not None:
                        cands.append((u, ch))
                if not cands:
                    arms[name] = None; continue
                best = max(cands, key=lambda x: x[0])[1]
                arms[name] = measure(t, best, ASSESS_BASE + 1000 * k + (0 if name == "policy" else 500), a.assess_reads)
            arms["start"] = measure(t, start, ASSESS_BASE + 1000 * k + 900, a.assess_reads)
            if all(v is not None for v in arms.values()):
                rows.append(arms)
        model.train()
        if not rows:
            print("  %s: nothing measured" % tag, flush=True); return 0.0
        def boot(fn):
            d = np.array([fn(r) for r in rows]); rng2 = np.random.default_rng(0)
            bs = [d[rng2.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        pr = boot(lambda r: r["policy"] - r["random"]); ps = boot(lambda r: r["policy"] - r["start"]); rs = boot(lambda r: r["random"] - r["start"])
        print("  %s held-out over %d: policy-random %+.4f [%+.4f, %+.4f] | policy-start %+.4f [%+.4f, %+.4f] | random-start %+.4f [%+.4f, %+.4f]"
              % (tag, len(rows), *pr, *ps, *rs), flush=True)
        return float(pr[0])

    best = -1.0
    evaluate("init")
    for it in range(a.iterations):
        batch = rng.choice(len(train_tasks), size=min(a.instances_per_iteration, len(train_tasks)), replace=False)
        loss, n, gains, rho = 0.0, 0, [], []
        for idx in batch:
            t = train_tasks[idx]
            start, fc = start_of(t, int(idx))
            if start is None:
                continue
            m = spend_of(start)
            u0 = measure(t, start, SELECT_BASE + 100 * it + 3 * int(idx), a.reads)
            if u0 is None:
                continue
            eps = []
            for e in range(a.episodes_per_instance):
                ch, logps = episode(t, start, model, fc, m, a.temperature, rng, train=True)
                if not logps:
                    continue
                used = sum(len(c) for c in ch.values())
                if used > a.qubit_cap:
                    eps.append((-a.fail_penalty, logps)); gains.append(-a.fail_penalty); continue
                u = measure(t, ch, SELECT_BASE + 100 * it + 3 * int(idx) + 11 * (e + 1), a.reads)
                if u is None:
                    eps.append((-a.fail_penalty, logps)); gains.append(-a.fail_penalty); continue
                eps.append((u - u0, logps)); gains.append(u - u0)
                rho.append(used / t.host.number_of_nodes())
            if len(eps) < 2:
                continue
            base = np.mean([g for g, _ in eps])
            for g, logps in eps:
                loss = loss - (g - base) * torch.stack(logps).sum() / len(logps); n += 1
        if n:
            opt.zero_grad(); (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print("  iter %4d  mean gain %+.4f  updates %d  occupancy %.2f" % (it, np.mean(gains) if gains else 0.0, n, np.mean(rho) if rho else 0.0), flush=True)
        if (it + 1) % a.eval_every == 0:
            v = evaluate("iter %d" % it)
            if v > best:
                best = v; torch.save({"state": model.state_dict(), "width": a.width}, a.out)
    torch.save({"state": model.state_dict(), "width": a.width}, a.out + ".last")
    print("\nCONTACT POLICY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

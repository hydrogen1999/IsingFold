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

from _context import construction_context, host_context, qubit_budget
from candidate_features import FeatureContext
from seeded_minorminer import attempt
from fast_layout import sample_layout
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


def complete(task, roots, deadline, tries, seed, budget=None):
    t0, ok, n = time.time(), None, 0
    while ok is None and time.time() - t0 < deadline:
        n += 1
        ok = attempt(task, roots, seed + n, tries, budget=budget)
    complete.last = ok
    return ok is not None, time.time() - t0, n


_ctx_cache = {}


def measure_residual(task, chains, seed, reads=256):
    """Mean energy residual above the planted ground energy under the registered schedule,
    the objective at this scale (solve probability reads zero for every embedding here).
    Lower is better. None if the evaluator rejects the embedding."""
    from isingfold.rl.env import fixed_strength_selector
    from isingfold.rl.evaluate import first_commit_controller, run_controller
    cap = task.host.number_of_nodes()
    if cap not in _ctx_cache:
        _ctx_cache[cap] = host_context(cap)
    ctx = _ctx_cache[cap]
    fixed = {v: frozenset(c) for v, c in chains.items()}
    try:
        out = run_controller([task], ctx, first_commit_controller, initializer=lambda l, h, s: fixed,
                             selector=fixed_strength_selector(), reward_reads=reads,
                             repetitions=1, seed=seed)
    except Exception:
        return None
    o = out[0]
    if not o.returned_valid or o.mean_energy_residual is None:
        return None
    return float(o.mean_energy_residual)


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
    ap.add_argument("--fast", action="store_true",
                    help="sample layouts outside the environment (milliseconds a layout)")
    ap.add_argument("--layout-router-secs", type=float, default=2.0,
                    help="router budget per sampled layout in the search and in training; the "
                         "router finishes a right layout in under a second, so a long budget "
                         "only pays for wrong ones")
    ap.add_argument("--layout-tries", type=int, default=2)
    ap.add_argument("--objective", default="valid", choices=("valid", "quality"),
                    help="quality: a valid completion is rewarded by its measured energy "
                         "residual against the router-alone embedding of the same instance, "
                         "which is the objective; validity stays the gate")
    ap.add_argument("--quality-weight", type=float, default=10.0)
    ap.add_argument("--select-cap", type=int, default=6,
                    help="valid candidates measured per arm at evaluation, the read budget both arms get")
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

    budgets = {}
    base_eval = {}

    def budget_of(task):
        if task.name not in budgets:
            budgets[task.name] = qubit_budget({v: frozenset(c) for v, c in task.witness.items()})
        return budgets[task.name]

    ref_res = {}

    def reference_residual(task, k):
        """The planted witness embedding, measured once: a per-instance reference that does
        not depend on whether the router alone succeeds, so the quality reward keeps its
        meaning on the hardest instances."""
        if task.name not in ref_res:
            w = {v: frozenset(c) for v, c in task.witness.items()}
            ref_res[task.name] = measure_residual(task, w, 300 + k)
        return ref_res[task.name]

    def reward_of(task, roots, ok, k, e):
        """Invalid: at most 0.5, by how close the completion came. Valid: at least 0.6, so a
        valid embedding always outranks an invalid one, plus the quality term against the
        witness reference, clipped."""
        if not ok:
            return 0.5 * partial_score(task, roots, 91_000 + 7 * e + 100 * k)
        if a.objective == "valid":
            return 1.0
        res = measure_residual(task, complete.last, 400 + 7 * e + 100 * k)
        ref = reference_residual(task, k)
        if res is None or ref is None:
            return 1.0
        return float(np.clip(1.0 + a.quality_weight * (ref - res), 0.6, 3.0))

    layout = sample_layout if a.fast else place_roots
    print(json.dumps({"corpus": a.corpus, "train": len(train_tasks), "held_out": len(eval_tasks),
                      "init": a.init or None, "train_deadline": a.train_deadline,
                      "eval_deadline": a.eval_deadline, "fast": a.fast}), flush=True)

    def evaluate(tag):
        """ADR-004's comparison. Both arms get the same deadline, the same qubit budget and
        the same measured selection: candidates are collected until the deadline (layouts
        from the policy completed by the router in one arm, the router restarted alone in
        the other), each valid one is measured on a selection block up to a cap per arm,
        the best is assessed on a fresh block. Validity and the paired residual are reported."""
        model.eval()
        cells = defaultdict(list)
        pairs, reads_used = [], {"policy": 0, "router": 0}

        def collect(t, k, use_policy):
            t0, cands, tried = time.time(), [], 0
            while time.time() - t0 < a.eval_deadline:
                tried += 1
                left = a.eval_deadline - (time.time() - t0)
                if use_policy:
                    roots, _ = layout(t, model, fc_for(t), a.temperature, rng, train=True)
                    got, _, _ = complete(t, roots, min(a.layout_router_secs, max(0.5, left)),
                                         a.layout_tries, 70_000 + 1000 * k + 13 * tried, budget=budget_of(t))
                else:
                    got, _, _ = complete(t, None, min(a.layout_router_secs, max(0.5, left)),
                                         a.layout_tries, 80_000 + 1000 * k + 13 * tried, budget=budget_of(t))
                if got:
                    key = tuple(sorted((str(v), tuple(sorted(c, key=str))) for v, c in complete.last.items()))
                    if key not in {c[0] for c in cands}:
                        cands.append((key, complete.last))
                if a.objective == "valid" and cands:
                    break
                if len(cands) >= a.select_cap:
                    break
            return cands, tried

        for k, t in enumerate(eval_tasks):
            arms = {}
            for name, use_policy in (("policy", True), ("router", False)):
                if name == "router" and t.name in base_eval:
                    arms[name] = base_eval[t.name]; continue
                cands, tried = collect(t, k, use_policy)
                chosen, chosen_sel = None, None
                if cands and a.objective == "quality":
                    scored = []
                    for i, (_, ch) in enumerate(cands[: a.select_cap]):
                        sel = measure_residual(t, ch, 500 + 31 * i + 1000 * k)
                        reads_used[name] += 256
                        if sel is not None:
                            scored.append((sel, ch))
                    if scored:
                        chosen_sel, chosen = min(scored, key=lambda x: x[0])
                elif cands:
                    chosen = cands[0][1]
                fresh = measure_residual(t, chosen, 900 + 1000 * k) if (chosen is not None and a.objective == "quality") else None
                arms[name] = (bool(cands), fresh, tried, len(cands))
                if name == "router":
                    base_eval[t.name] = arms[name]
            (ok, q, tried, n_c), (ok0, q0, _, n_c0) = arms["policy"], arms["router"]
            if q is not None and q0 is not None:
                pairs.append(q - q0)
            cells[t.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]].append((ok, ok0, tried, n_c, n_c0))
        model.train()
        allr = [r for rs in cells.values() for r in rs]
        print("  %s held-out within %.0fs: policy+search %.2f  router+search %.2f  layouts %.1f  valid candidates %.1f/%.1f  over %d"
              % (tag, a.eval_deadline, np.mean([r[0] for r in allr]), np.mean([r[1] for r in allr]),
                 np.mean([r[2] for r in allr]), np.mean([r[3] for r in allr]), np.mean([r[4] for r in allr]), len(allr)), flush=True)
        print("    " + "  ".join("%s %.2f/%.2f" % (c, np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]))
                                 for c, rs in sorted(cells.items())), flush=True)
        if pairs:
            d = np.array(pairs)
            rng2 = np.random.default_rng(0)
            bs = [d[rng2.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print("    residual, policy minus router, both selected by measurement, fresh block, where both valid: %+.4f [%+.4f, %+.4f] over %d"
                  % (d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)), flush=True)
        score = float(np.mean([r[0] for r in allr]))
        if pairs:
            score -= float(np.mean(pairs))
        return score

    best = -1.0
    evaluate("init")
    for it in range(a.iterations):
        batch = rng.choice(len(train_tasks), size=min(a.instances_per_iteration, len(train_tasks)), replace=False)
        loss, n, wins, secs = 0.0, 0, [], []
        for idx in batch:
            t = train_tasks[idx]
            eps = []
            for e in range(a.episodes_per_instance):
                roots, logps = layout(t, model, fc_for(t), a.temperature, rng, train=True)
                ok, s, _ = complete(t, roots, a.layout_router_secs, a.layout_tries,
                                    90_000 + 7 * e + 100 * it, budget=budget_of(t))
                r = reward_of(t, roots, ok, int(idx), e)
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

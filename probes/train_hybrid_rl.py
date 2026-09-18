"""Hybrid RL: a sampled root layout is completed by minorminer within a qubit/time budget.

Validity training uses a completion reward and a bounded partial-overlap shaping signal.
Quality training uses bounded residual improvement over a training-only measured witness
reference. REINFORCE uses summed trajectory log probabilities and a leave-one-out baseline;
the opt-in contextual actor also supports a learned, detached state-value baseline.
Validation holds out whole lineages. Both learned and unhinted arms restart, measure and
select under one deployment deadline; fresh assessment reads are reporting-only. A witness
fill is a corpus property, not occupancy of these root-construction episodes.
"""
import argparse, hashlib, json, os, sys, time
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
from candidate_features import FeatureContext, WIDTH as LEGACY_WIDTH
from layout_features import LayoutFeatureContext, WIDTH as LAYOUT_WIDTH, FEATURE_VERSION
from layout_policy import LayoutActorCritic, sample_layout_v2
from layout_learning import trajectory_loss, witness_prefix_loss
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


def complete(task, roots, deadline, tries, seed, budget=None):
    """Cooperative router deadline; late completions never count as success."""
    t0, ok, n = time.monotonic(), None, 0
    complete.last = None
    while ok is None and time.monotonic() - t0 < deadline:
        n += 1
        left = deadline - (time.monotonic() - t0)
        ok = attempt(task, roots, seed + n, tries, budget=budget, timeout=left)
    elapsed = time.monotonic() - t0
    if elapsed > deadline:
        ok = None
    complete.last = ok
    return ok is not None, elapsed, n


def experiment_seed(seed, phase, *parts):
    """Stable, phase-separated sampler streams, without arithmetic seed collisions."""
    payload = json.dumps([int(seed), phase, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.blake2s(payload.encode(), digest_size=4).digest(), "little") % (2 ** 31)


def split_by_lineage(tasks, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must lie strictly between zero and one")
    groups = sorted({t.lineage or t.name for t in tasks})
    if len(groups) < 2:
        raise ValueError("at least two independent lineages are required")
    rng = np.random.default_rng(seed)
    size = min(len(groups) - 1, max(1, int(len(groups) * fraction)))
    held = set(rng.choice(groups, size=size, replace=False))
    return ([t for t in tasks if (t.lineage or t.name) not in held],
            [t for t in tasks if (t.lineage or t.name) in held])


def quality_reward(residual, reference, weight):
    """Bounded monotone quality reward; every valid completion exceeds partial failure.

    A missing measurement receives the valid floor, not an invented neutral quality label.
    This scalar surrogate is not a lexicographic feasibility guarantee in expectation.
    """
    if residual is None or reference is None:
        return 0.6
    return float(np.clip(1.0 + weight * (reference - residual), 0.6, 3.0))


def validation_checkpoint_key(records, objective):
    """Validity, measurement coverage, then lower raw residual on the policy's outputs.

    Explicit lexicographic checkpoint rule, not a learned resource objective. It never
    restricts quality to the baseline-success intersection or transforms the residual.
    """
    valid = [r for r in records if r["valid"]]
    coverage = len(valid) / len(records)
    if objective == "valid":
        return (coverage,)
    measured = [r["residual"] for r in valid if r["residual"] is not None]
    return (coverage, len(measured) / len(records),
            -float(np.mean(measured)) if measured else -float("inf"))


def evaluate_arm(task, propose, *, deadline, router_seconds, tries, budget, select_cap,
                 objective, seed, selection_reads=256, assessment_reads=256):
    """Matched deployment budget for proposal, routing and measured selection.

    Assessment is an independent reporting-only block, outside the deployment deadline.
    Blocking backend calls may overrun: their work is logged, their late result discarded.
    """
    started = time.monotonic()
    seen, chosen, best = set(), None, float("inf")
    attempts = selection_used = measurements = 0
    while time.monotonic() - started < deadline:
        attempts += 1
        roots = propose(attempts) if propose is not None else None
        left = deadline - (time.monotonic() - started)
        if left <= 0:
            break
        ok, _, _ = complete(task, roots, min(router_seconds, left), tries,
                            experiment_seed(seed, "router", task.name, attempts), budget=budget)
        if time.monotonic() - started > deadline:
            break
        if not ok:
            continue
        chains = complete.last
        key = frozenset((v, frozenset(c)) for v, c in chains.items())
        if key in seen:
            continue
        seen.add(key)
        if objective == "valid":
            chosen = chains
            break
        residual = measure_residual(task, chains,
                                    experiment_seed(seed, "selection", task.name, measurements),
                                    reads=selection_reads)
        selection_used += selection_reads
        measurements += 1
        if time.monotonic() - started > deadline:
            break
        if residual is not None and residual < best:
            chosen, best = chains, residual
        if measurements >= select_cap:
            break
    deployment_seconds = time.monotonic() - started
    fresh, assessment_used = None, 0
    if chosen is not None and objective == "quality":
        fresh = measure_residual(task, chosen, experiment_seed(seed, "assessment", task.name),
                                 reads=assessment_reads)
        assessment_used = assessment_reads
    return {"valid": chosen is not None, "residual": fresh, "attempts": attempts,
            "candidates": len(seen), "selection_reads": selection_used,
            "assessment_reads": assessment_used, "deployment_seconds": deployment_seconds,
            "total_seconds": time.monotonic() - started,
            "deadline_overrun_seconds": max(0.0, deployment_seconds - deadline)}


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
    residual = float(o.mean_energy_residual)
    return residual if np.isfinite(residual) else None


def partial_score(task, roots, seed, tries=2, timeout=2.0):
    """A graded score for a failed completion: the router with overlaps allowed returns an
    embedding in which contested qubits are shared; the fraction of variables whose chain
    shares no qubit is how close the layout came. Without it every episode of a hard
    instance scores zero and REINFORCE has no gradient at all, which is what the first run
    showed: no update in nine iterations."""
    import minorminer
    edges = list(task.logical.edges())
    in_edges = {u for e in edges for u in e}
    if not edges or timeout <= 0:
        return 0.0
    kw = {"tries": tries, "random_seed": seed % (2 ** 31), "return_overlap": True,
          "timeout": timeout}
    if roots:
        kw["initial_chains"] = {v: list(c) for v, c in roots.items() if v in in_edges}
    emb, _ = minorminer.find_embedding(edges, list(task.host.edges()), **kw)
    if not emb:
        return 0.0
    owners = {}
    for v, c in emb.items():
        for q in c:
            owners.setdefault(q, set()).add(v)
    clean = sum(1 for v, c in emb.items() if c and all(len(owners[q]) == 1 for q in c))
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
    ap.add_argument("--actor", choices=("local", "contextual"), default="local")
    ap.add_argument("--features", choices=("legacy", "capacity"), default="legacy")
    ap.add_argument("--root-support", choices=("legacy", "all_free"), default="legacy")
    ap.add_argument("--baseline", choices=("loo", "value"), default="loo")
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--warmstart-epochs", type=int, default=0,
                    help="train-only witness-set placement teacher; not a quality teacher")
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
                    help="quality: bounded residual improvement over a measured training "
                         "witness reference; no resource-minimization reward")
    ap.add_argument("--quality-weight", type=float, default=10.0)
    ap.add_argument("--select-cap", type=int, default=6,
                    help="valid candidates measured per arm at evaluation, the read budget both arms get")
    ap.add_argument("--qubit-cap", type=int, default=0,
                    help="deployment qubit budget for both arms; 0 allows the full host")
    ap.add_argument("--witness-budget", action="store_true",
                    help="legacy experiment: cap at 1.1 times witness occupancy; privileged "
                         "budget metadata, not the deployment default")
    a = ap.parse_args()
    if not all(np.isfinite(x) for x in (a.temperature, a.quality_weight, a.train_deadline,
                                        a.eval_deadline, a.layout_router_secs,
                                        a.entropy_coef, a.value_coef, a.learning_rate)):
        ap.error("training parameters must be finite")
    if (a.episodes_per_instance < 2 or a.select_cap < 1 or a.eval_every < 1
            or a.temperature <= 0 or a.quality_weight < 0 or a.qubit_cap < 0
            or min(a.train_deadline, a.eval_deadline, a.layout_router_secs) <= 0):
        ap.error("positive deadlines/temperature/caps and at least two episodes are required")
    if a.witness_budget and a.qubit_cap:
        ap.error("choose explicit --qubit-cap or legacy --witness-budget")
    if min(a.entropy_coef, a.value_coef, a.warmstart_epochs, a.iterations) < 0:
        ap.error("loss coefficients, epochs and iterations must be nonnegative")
    if (a.width < 1 or a.instances_per_iteration < 1 or a.learning_rate <= 0
            or a.tries < 1 or a.layout_tries < 1):
        ap.error("width, batch size, learning rate and router tries must be positive")
    if a.baseline == "value" and a.actor != "contextual":
        ap.error("--baseline value requires --actor contextual")
    if not a.fast and (a.actor != "local" or a.features != "legacy"
                       or a.root_support != "legacy" or a.baseline != "loo"
                       or a.entropy_coef or a.warmstart_epochs):
        ap.error("versioned layout features, support and actor require --fast")
    tasks = load_instances(a.corpus)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    train_tasks, eval_tasks = split_by_lineage(tasks, a.holdout_fraction, a.seed)
    if a.hard_only:
        train_tasks = [t for t in train_tasks if not t.name.split("-")[1].startswith("fill70")]
    if not train_tasks:
        ap.error("no training instances remain after filtering")
    in_dim = LAYOUT_WIDTH if a.features == "capacity" else LEGACY_WIDTH
    model_spec = {"actor": a.actor, "feature_version": FEATURE_VERSION if a.features == "capacity"
                  else "candidate-v1", "in_dim": in_dim, "width": a.width}
    model = (LayoutActorCritic(a.width, in_dim=in_dim) if a.actor == "contextual"
             else Prioritiser(a.width, in_dim=in_dim))
    if a.init:
        checkpoint = torch.load(a.init, map_location="cpu", weights_only=True)
        saved_spec = checkpoint.get("model_spec", {"actor": "local", "feature_version": "candidate-v1",
                                                   "in_dim": LEGACY_WIDTH, "width": checkpoint["width"]})
        if saved_spec != model_spec:
            ap.error("checkpoint model/feature schema differs; match --actor, --features and --width")
        model.load_state_dict(checkpoint["state"])
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    fcs = {}

    def budget_of(task):
        cap = task.host.number_of_nodes()
        if a.witness_budget:
            return min(cap, qubit_budget(task.witness))
        return min(cap, a.qubit_cap) if a.qubit_cap else cap

    def fc_for(task):
        if task.name not in fcs:
            cls = LayoutFeatureContext if a.features == "capacity" else FeatureContext
            fcs[task.name] = cls(task, budget_of(task))
        return fcs[task.name]

    base_eval = {}

    ref_res = {}

    def reference_residual(task, k):
        """The planted witness embedding, measured once: a per-instance reference that does
        not depend on whether the router alone succeeds, so the quality reward keeps its
        meaning on the hardest instances."""
        if task.name not in ref_res:
            w = {v: frozenset(c) for v, c in task.witness.items()}
            ref_res[task.name] = measure_residual(task, w, experiment_seed(a.seed, "reference", task.name))
        return ref_res[task.name]

    def reward_of(task, roots, ok, k, e, iteration):
        """Invalid: at most 0.5, by how close the completion came. Valid: at least 0.6, so a
        valid embedding always outranks an invalid one, plus the quality term against the
        witness reference, clipped."""
        if not ok:
            return 0.5 * partial_score(task, roots,
                                       experiment_seed(a.seed, "partial", task.name, iteration, e),
                                       timeout=min(a.train_deadline, a.layout_router_secs))
        if a.objective == "valid":
            return 1.0
        res = measure_residual(task, complete.last,
                               experiment_seed(a.seed, "reward", task.name, iteration, e))
        ref = reference_residual(task, k)
        return quality_reward(res, ref, a.quality_weight)

    def layout(task, policy, fc, temperature, rng, train):
        if a.fast:
            return sample_layout_v2(task, policy, fc, temperature, rng, train=train,
                                    support=a.root_support)
        return place_roots(task, policy, fc, temperature, rng, train)

    print(json.dumps({"corpus": a.corpus, "train": len(train_tasks), "held_out": len(eval_tasks),
                      "init": a.init or None, "train_deadline": a.train_deadline,
                      "eval_deadline": a.eval_deadline, "fast": a.fast,
                      "protocol": "hybrid-matched-deployment-v2",
                      "split_unit": "lineage", "evaluation_role": "validation",
                      "train_lineages": sorted({t.lineage or t.name for t in train_tasks}),
                      "validation_lineages": sorted({t.lineage or t.name for t in eval_tasks}),
                      "qubit_cap": a.qubit_cap, "legacy_witness_budget": a.witness_budget,
                      "model_spec": model_spec, "root_support": a.root_support,
                      "policy_sampler": "layout-v2-numpy" if a.fast else "environment-torch",
                      "baseline": a.baseline, "entropy_coef": a.entropy_coef,
                      "warmstart_epochs": a.warmstart_epochs,
                      "learning_algorithm": "terminal-MC-policy-gradient-single-update",
                      "config": vars(a),
                      "deadline_scope": "proposal+router+selection; reporting assessment separate"}), flush=True)

    def evaluate(tag):
        """ADR-004's comparison. Both arms get the same deadline, the same qubit budget and
        the same measured selection: candidates are collected until the deadline (layouts
        from the policy completed by the router in one arm, the router restarted alone in
        the other), each valid one is measured on a selection block up to a cap per arm,
        the best is assessed on a fresh block. Validity and the paired residual are reported."""
        model.eval()
        cells = defaultdict(list)
        pairs, records = [], []
        for k, t in enumerate(eval_tasks):
            arms = {}
            for name, use_policy in (("policy", True), ("router", False)):
                if name == "router" and t.name in base_eval:
                    arms[name] = base_eval[t.name]
                    records.append({"instance": t.name, "arm": name, "cached": True, **arms[name]})
                    continue
                # Validation must not advance training's numpy or torch random stream.
                proposal_rng = np.random.default_rng(experiment_seed(a.seed, "proposal", t.name))
                def propose(attempt_number):
                    with torch.random.fork_rng(devices=[]), torch.no_grad():
                        torch.manual_seed(experiment_seed(a.seed, "layout", t.name, attempt_number))
                        return layout(t, model, fc_for(t), a.temperature, proposal_rng, train=True)[0]
                arms[name] = evaluate_arm(
                    t, propose if use_policy else None, deadline=a.eval_deadline,
                    router_seconds=a.layout_router_secs, tries=a.layout_tries,
                    budget=budget_of(t), select_cap=a.select_cap, objective=a.objective,
                    seed=experiment_seed(a.seed, "validation", t.name))
                records.append({"instance": t.name, "arm": name, "cached": False, **arms[name]})
                if name == "router":
                    base_eval[t.name] = arms[name]
            learned, baseline = arms["policy"], arms["router"]
            ok, q, tried, n_c = (learned["valid"], learned["residual"],
                                  learned["attempts"], learned["candidates"])
            ok0, q0, n_c0 = baseline["valid"], baseline["residual"], baseline["candidates"]
            if q is not None and q0 is not None:
                pairs.append(q - q0)
            cells[t.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]].append((ok, ok0, tried, n_c, n_c0))
        print(json.dumps({"evaluation": tag, "role": "validation", "arms": records}), flush=True)
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
        score = validation_checkpoint_key([r for r in records if r["arm"] == "policy"], a.objective)
        print("    checkpoint key: %s (validity, measured coverage, negative mean policy residual)"
              % (score,), flush=True)
        return score

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(path, score=None):
        torch.save({"state": model.state_dict(), "width": a.width, "config": vars(a),
                    "model_spec": model_spec, "validation_score": score}, path)

    # Split first. Witnesses of validation tasks are never read by warm start.
    for epoch in range(a.warmstart_epochs):
        metrics, losses = [], []
        for idx in rng.permutation(len(train_tasks)):
            t = train_tasks[int(idx)]
            loss, metric = witness_prefix_loss(t, t.witness, model, fc_for(t), rng,
                                               support=a.root_support, temperature=a.temperature)
            metrics.append(metric)
            if loss is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
        print(json.dumps({"warmstart_epoch": epoch, "role": "training",
                          "teacher": "compatible-root-set-feasibility-only",
                          "teacher_coverage": sum(m["teacher_covered"] for m in metrics)
                          / max(1, sum(m["teacher_steps"] for m in metrics)),
                          "loss": float(np.mean(losses)) if losses else None}), flush=True)
    best = evaluate("post_warmstart" if a.warmstart_epochs else "init")
    save_checkpoint(a.out, best)
    for it in range(a.iterations):
        batch = rng.choice(len(train_tasks), size=min(a.instances_per_iteration, len(train_tasks)), replace=False)
        n, wins, secs, diagnostics = 0, [], [], []
        opt.zero_grad()
        for idx in batch:
            t = train_tasks[idx]
            eps = []
            for e in range(a.episodes_per_instance):
                roots, logps = layout(t, model, fc_for(t), a.temperature, rng, train=True)
                ok, s, _ = complete(t, roots, min(a.train_deadline, a.layout_router_secs), a.layout_tries,
                                    experiment_seed(a.seed, "train-router", t.name, it, e), budget=budget_of(t))
                r = reward_of(t, roots, ok, int(idx), e, it)
                eps.append((r, logps)); wins.append(ok); secs.append(s)
            if a.fast:
                loss, diag = trajectory_loss(eps, baseline=a.baseline,
                                             entropy_coef=a.entropy_coef, value_coef=a.value_coef)
                diagnostics.append(diag)
                n += diag["updated_episodes"]
            else:
                reward_sum = sum(r for r, _ in eps)
                terms = [-(r - (reward_sum - r) / (len(eps) - 1)) * torch.stack(logps).sum()
                         for r, logps in eps if logps]
                loss = torch.stack(terms).mean() if terms else None
                n += len(terms)
            if loss is not None:
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite policy loss")
                # Accumulate gradients with fixed parameters, freeing each task's
                # computation graph before the next all-free candidate rollout.
                (loss / len(batch)).backward()
        grad_norm = 0.0
        if n:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                            error_if_nonfinite=True))
            opt.step()
        if diagnostics:
            summary = {key: float(np.mean([d[key] for d in diagnostics if d[key] is not None]))
                       if any(d[key] is not None for d in diagnostics) else None
                       for key in diagnostics[0]}
            print(json.dumps({"training_iteration": it, "role": "training",
                              "gradient_norm_before_clip": grad_norm, **summary}), flush=True)
        print("  iter %4d  train success %.2f  mm secs %.1f  updates %d"
              % (it, np.mean(wins), np.mean(secs), n), flush=True)
        if (it + 1) % a.eval_every == 0:
            v = evaluate("iter %d" % it)
            if v > best:
                best = v
                save_checkpoint(a.out, best)
    save_checkpoint(a.out + ".last")
    print("\nHYBRID RL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

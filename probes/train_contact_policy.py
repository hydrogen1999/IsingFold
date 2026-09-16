"""Learn where to spend a qubit: contact growth chosen by a policy, rewarded by the objective.

This probe tests whether contact growth improves the registered downstream objective; it
does not assume that adding contacts helps on every instance. Starting from a router or
witness embedding, an episode adds at most m qubits, each free and adjacent to one chain and to the
chain of a coupled variable, chosen by a softmax over the prioritiser's scores of the
candidate (variable, qubit) pairs; the reward is the measured utility of the grown
embedding minus the start's, under the registered schedule, on an independent block.
REINFORCE uses a leave-one-out instance baseline and the sum of trajectory log probabilities.

Validation on disjoint lineages, matched read caps: policy best of K candidates selected by a
measurement block, random contact growth best of K the same way, and the start, all
assessed on a fresh block. Policy minus random is the learned contribution; both minus the
start is what the spend is worth at all. Both candidate pools contain K growth proposals;
the unchanged start is assessed separately as a reference and is never added as a fallback
candidate. Failed arm evaluations receive the declared failure penalty rather than being
dropped. A failed restart remains a failed proposal, never the supplied start, and consumes
one of the K attempts without receiving replacement draws or recycled measurement reads.
The protocol records actual read calls and timings; it does not match wall time.
This is a training/continuation diagnostic from a supplied valid state, not an
end-to-end embedder evaluation: deployed embedding construction must begin from empty.
Witness starts are allowed only for learning or controlled diagnostics. Checkpoint selection
uses validation, not an untouched final test. Witness fill describes actual starting occupancy
only when --start=witness; requested generator fill alone is not an occupancy measurement.
"""
import argparse, hashlib, json, os, sys
import time
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

def block_seed(seed, domain, *parts):
    """Stable domain separation: no arithmetic overlap between training/select/assess blocks."""
    payload = json.dumps([int(seed), domain, *parts], sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


def split_representatives(tasks, train_lineages, eval_lineages, seed):
    """One deterministic representative per lineage; never truncate across flattened tasks."""
    groups = defaultdict(list)
    for task in tasks:
        if not task.lineage:
            raise ValueError("contact-policy splitting requires a nonempty lineage")
        groups[task.lineage].append(task)
    lineages = sorted(groups)
    np.random.default_rng(seed).shuffle(lineages)
    if len(lineages) < train_lineages + eval_lineages:
        raise ValueError("corpus has fewer distinct lineages than requested train + validation")
    representatives = [sorted(groups[lineage], key=lambda t: t.name)[0] for lineage in lineages]
    return (representatives[:train_lineages],
            representatives[train_lineages:train_lineages + eval_lineages])


def spend_limit(start, fraction, qubit_cap, host_size):
    """Additional qubits allowed by both the physical host and the declared experiment cap."""
    used = sum(len(c) for c in start.values())
    slack = max(0, min(qubit_cap, host_size) - used)
    return min(max(0, int(round(fraction * used))), slack)


def reinforce_loss(episodes):
    """An action-independent leave-one-out baseline; no variable-length trajectory bias."""
    if len(episodes) < 2:
        return None
    total = sum(g for g, _ in episodes)
    losses = []
    for gain, logps in episodes:
        if logps:
            baseline = (total - gain) / (len(episodes) - 1)
            losses.append(-(gain - baseline) * torch.stack(logps).sum())
    return torch.stack(losses).mean() if losses else None


def select_and_assess(candidates, measure, select_seed, assess_seed, baseline, penalty, cap,
                      *, accounting=None):
    """Keep failed attempts and their seed slots; never measure or replace a missing proposal.

    Optional accounting counts measurement *calls*, including unsuccessful measurements.
    Skipped candidates consume an attempt but no read calls, with no budget reallocation.
    """
    counts = {"candidate_attempts": len(candidates), "unavailable_candidates": 0,
              "over_cap_candidates": 0, "selection_calls": 0,
              "selection_successes": 0, "assessment_calls": 0, "assessment_failures": 0}
    if accounting is not None:
        accounting.update(counts)
        counts = accounting
    measured = []
    for index, chains in enumerate(candidates):
        if not chains:
            counts["unavailable_candidates"] += 1
            continue
        if sum(len(c) for c in chains.values()) > cap:
            counts["over_cap_candidates"] += 1
            continue
        counts["selection_calls"] += 1
        score = measure(chains, select_seed(index), False)
        if score is not None and np.isfinite(score):
            counts["selection_successes"] += 1
            measured.append((float(score), chains))
    failures = len(candidates) - len(measured)
    if not measured:
        return baseline - penalty, failures, True
    chosen = max(measured, key=lambda item: item[0])[1]
    counts["assessment_calls"] += 1
    assessed = measure(chosen, assess_seed, True)
    if assessed is None or not np.isfinite(assessed):
        counts["assessment_failures"] += 1
        return baseline - penalty, failures, True
    return float(assessed), failures, False


def contact_moves(task, chains, occupied, cap_per_var=6):
    """Candidate additions: a free qubit adjacent to chain(v) that touches the chain of a
    logical neighbour of v. Returned as (v, r, touches)."""
    host, logical = task.host, task.logical
    out = []
    for v in sorted(chains, key=repr):
        chain = chains[v]
        seen = set()
        for q in sorted(chain, key=repr):
            for r in sorted(host.neighbors(q), key=repr):
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
        if model is None:
            j = int(rng.integers(0, len(moves)))
            v, r, _ = moves[j]
            chains[v].add(r); occupied.add(r)
            continue
        scores = model(torch.as_tensor(feats)) / temperature
        if train:
            dist = torch.distributions.Categorical(logits=scores)
            # Sampling uses the caller's RNG, so evaluating checkpoints cannot perturb
            # subsequent training trajectories through PyTorch's global generator.
            probabilities = dist.probs.detach().cpu().numpy().astype(np.float64)
            probabilities /= probabilities.sum()
            j = int(rng.choice(len(moves), p=probabilities))
            logps.append(dist.log_prob(torch.tensor(j, device=scores.device)))
        else:
            j = int(torch.argmax(scores))
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
    ap.add_argument("--eval-k", type=int, default=4,
                    help="growth proposals per arm; the unchanged start is a separate reference")
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
    for key in ("qubit_cap", "instances_per_iteration", "reads", "assess_reads",
                "train_lineages", "eval_lineages", "eval_every", "eval_k", "width"):
        if getattr(a, key) <= 0:
            ap.error("--" + key.replace("_", "-") + " must be positive")
    if a.episodes_per_instance < 2:
        ap.error("--episodes-per-instance must be at least 2 for the leave-one-out baseline")
    if a.iterations < 0 or not np.isfinite(a.spend) or a.spend < 0:
        ap.error("iterations and spend must be nonnegative and spend must be finite")
    if not np.isfinite(a.temperature) or a.temperature <= 0:
        ap.error("temperature must be finite and positive")
    if not np.isfinite(a.fail_penalty) or a.fail_penalty <= 0:
        ap.error("fail-penalty must be finite and positive")
    tasks = load_instances(a.corpus)
    rng = np.random.default_rng(a.seed)
    train_tasks, eval_tasks = split_representatives(tasks, a.train_lineages, a.eval_lineages, a.seed)
    ctx = host_context(a.qubit_cap)
    mm = minorminer_initializer(20)
    torch.manual_seed(a.seed)
    model = Prioritiser(a.width, in_dim=WIDTH + FRONTIER_WIDTH)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["state"])
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    print(json.dumps({"corpus": a.corpus, "train": len(train_tasks), "validation": len(eval_tasks),
                      "spend": a.spend, "start": a.start, "objective": a.objective,
                      "selection_candidates": a.eval_k,
                      "evaluation_scope": "continuation_diagnostic_not_end_to_end_embedding",
                      "selection_read_budget_per_arm": a.eval_k * a.reads,
                      "selection_read_budget_type": "cap_no_refill_on_failed_proposal",
                      "matched_wall_time": False,
                      "assessment_reads_per_arm": a.assess_reads,
                      "failure_penalty": a.fail_penalty,
                      "train_lineages": [t.lineage for t in train_tasks],
                      "validation_lineages": [t.lineage for t in eval_tasks],
                      "beta_range": list(ctx.beta_range)}), flush=True)

    def measure(task, chains, seed, reads):
        def fixed(l, h, s):
            return chains
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=reads,
                                 repetitions=1, seed=seed)
        except Exception as exc:
            print(json.dumps({"measurement_failure": type(exc).__name__, "task": task.name,
                              "seed": seed, "reads": reads}), flush=True)
            return None
        o = out[0]
        if not o.returned_valid:
            return None
        if a.objective == "residual":
            # sign flipped so that higher is better everywhere below
            return -float(o.mean_energy_residual) if o.mean_energy_residual is not None else None
        return float(o.utility) if o.utility is not None else None

    starts, fcs = {}, {}

    def start_of(task):
        key = (task.lineage, task.name)
        if key not in starts:
            if a.start == "witness":
                ch = {v: frozenset(c) for v, c in task.witness.items()}
            else:
                ch = mm(task.logical, task.host, block_seed(a.seed, "initializer", *key))
            starts[key] = ch
            fcs[key] = FeatureContext(task, min(a.qubit_cap, task.host.number_of_nodes()))
        return starts[key], fcs[key]

    def spend_of(task, start):
        return spend_limit(start, a.spend, a.qubit_cap, task.host.number_of_nodes())

    def evaluate(tag):
        model.eval()
        rows = []
        unavailable_starts = 0
        candidate_failures = {"policy": 0, "random": 0, "restart": 0}
        arm_failures = {"policy": 0, "random": 0, "restart": 0}
        arm_costs = {name: defaultdict(int, proposal_seconds=0.0,
                                      selection_seconds=0.0, assessment_seconds=0.0)
                     for name in candidate_failures}
        initial_occupancy = []
        for k, t in enumerate(eval_tasks):
            start, fc = start_of(t)
            if not start or sum(len(c) for c in start.values()) > a.qubit_cap:
                unavailable_starts += 1
                continue
            start_score = measure(t, start, block_seed(a.seed, "validation-start-assessment", t.name), a.assess_reads)
            if start_score is None or not np.isfinite(start_score):
                unavailable_starts += 1
                continue
            initial_occupancy.append(sum(len(c) for c in start.values()) / t.host.number_of_nodes())
            m = spend_of(t, start)
            arms = {}
            for name in ("policy", "random", "restart"):
                cands = []
                proposal_begin = time.perf_counter()
                for e in range(a.eval_k):
                    # Fixed validation seeds at every checkpoint; no mutation of training RNG.
                    eval_rng = np.random.default_rng(block_seed(a.seed, "validation-proposal", t.name, name, e))
                    if name == "policy":
                        with torch.no_grad():
                            ch, _ = episode(t, start, model, fc, m, a.temperature, eval_rng, train=True)
                    elif name == "random":
                        ch = random_episode(t, start, m, eval_rng)
                    else:
                        # The control that decides the claim: the resource-first router drawn
                        # afresh, as many times as the other arms propose, selected and
                        # assessed the same way. Without it the start is one draw against a
                        # search of K and the comparison is not the paper's.
                        ch = mm(t.logical, t.host, int(block_seed(a.seed, "validation-restart", t.name, e)) % (2 ** 31))
                    cands.append(ch)
                arm_costs[name]["proposal_seconds"] += time.perf_counter() - proposal_begin
                def arm_measure(chains, seed, assess):
                    begin = time.perf_counter()
                    result = measure(t, chains, seed, a.assess_reads if assess else a.reads)
                    field = "assessment_seconds" if assess else "selection_seconds"
                    arm_costs[name][field] += time.perf_counter() - begin
                    return result
                # Matching block seeds across arms permits common random numbers;
                # assessment remains domain-disjoint from all selection measurements.
                accounting = {}
                score, failed_candidates, failed_arm = select_and_assess(
                    cands, arm_measure,
                    lambda index: block_seed(a.seed, "validation-selection", t.name, index),
                    block_seed(a.seed, "validation-assessment", t.name),
                    start_score, a.fail_penalty, a.qubit_cap, accounting=accounting)
                for field, count in accounting.items():
                    arm_costs[name][field] += count
                arm_costs[name]["selection_reads_requested"] += accounting["selection_calls"] * a.reads
                arm_costs[name]["assessment_reads_requested"] += accounting["assessment_calls"] * a.assess_reads
                arms[name] = score
                candidate_failures[name] += failed_candidates
                arm_failures[name] += int(failed_arm)
            arms["start"] = start_score
            rows.append(arms)
        model.train()
        print(json.dumps({"validation_tag": tag, "requested_instances": len(eval_tasks),
                          "measurable_starts": len(rows), "unavailable_starts": unavailable_starts,
                          "failed_candidates": candidate_failures, "failed_arms": arm_failures,
                          "successful_arms": {name: len(rows) - count
                                              for name, count in arm_failures.items()},
                          "candidate_measurement_coverage": {
                              name: (cost["selection_successes"] / cost["candidate_attempts"]
                                     if cost["candidate_attempts"] else None)
                              for name, cost in arm_costs.items()},
                          "arm_costs": arm_costs,
                          "mean_actual_start_occupancy": float(np.mean(initial_occupancy)) if initial_occupancy else None,
                          "score": "downstream objective with declared failure penalty"}), flush=True)
        if not rows:
            print("  %s: nothing measured" % tag, flush=True); return -float("inf")
        def boot(fn):
            d = np.array([fn(r) for r in rows]); rng2 = np.random.default_rng(0)
            bs = [d[rng2.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        pr = boot(lambda r: r["policy"] - r["random"]); ps = boot(lambda r: r["policy"] - r["start"]); rs = boot(lambda r: r["random"] - r["start"])
        print("  %s validation over %d: policy-random %+.4f [%+.4f, %+.4f] | policy-start %+.4f [%+.4f, %+.4f] | random-start %+.4f [%+.4f, %+.4f]"
              % (tag, len(rows), *pr, *ps, *rs), flush=True)
        rt = boot(lambda r: r["restart"] - r["start"]); prt = boot(lambda r: r["policy"] - r["restart"]); rrt = boot(lambda r: r["random"] - r["restart"])
        print("  %s control: restart-start %+.4f [%+.4f, %+.4f] | policy-restart %+.4f [%+.4f, %+.4f] | random-restart %+.4f [%+.4f, %+.4f]"
              % (tag, *rt, *prt, *rrt), flush=True)
        return float(pr[0])

    def save_checkpoint(path):
        torch.save({"state": model.state_dict(), "width": a.width,
                    "in_dim": WIDTH + FRONTIER_WIDTH, "objective": a.objective,
                    "start": a.start, "spend": a.spend, "qubit_cap": a.qubit_cap,
                    "evaluation_scope": "continuation_diagnostic_not_end_to_end_embedding",
                    "checkpoint_selection_split": "validation",
                    "validation_lineages": [t.lineage for t in eval_tasks]}, path)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    best = evaluate("init")
    save_checkpoint(a.out)
    for it in range(a.iterations):
        batch = rng.choice(len(train_tasks), size=min(a.instances_per_iteration, len(train_tasks)), replace=False)
        loss, n, gains, rho = 0.0, 0, [], []
        for idx in batch:
            t = train_tasks[idx]
            start, fc = start_of(t)
            if not start or sum(len(c) for c in start.values()) > a.qubit_cap:
                continue
            m = spend_of(t, start)
            u0 = measure(t, start, block_seed(a.seed, "train-start", it, t.name), a.reads)
            if u0 is None or not np.isfinite(u0):
                continue
            eps = []
            for e in range(a.episodes_per_instance):
                ch, logps = episode(t, start, model, fc, m, a.temperature, rng, train=True)
                used = sum(len(c) for c in ch.values())
                if used > a.qubit_cap:
                    eps.append((-a.fail_penalty, logps)); gains.append(-a.fail_penalty); continue
                u = measure(t, ch, block_seed(a.seed, "train-growth", it, t.name, e), a.reads)
                if u is None or not np.isfinite(u):
                    eps.append((-a.fail_penalty, logps)); gains.append(-a.fail_penalty); continue
                eps.append((u - u0, logps)); gains.append(u - u0)
                rho.append(used / t.host.number_of_nodes())
            instance_loss = reinforce_loss(eps)
            if instance_loss is not None:
                loss = loss + instance_loss; n += 1
        if n:
            opt.zero_grad(); (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print("  iter %4d  mean gain %+.4f  updates %d  occupancy %.2f" % (it, np.mean(gains) if gains else 0.0, n, np.mean(rho) if rho else 0.0), flush=True)
        if (it + 1) % a.eval_every == 0:
            v = evaluate("iter %d" % it)
            if v > best:
                best = v; save_checkpoint(a.out)
    save_checkpoint(a.out + ".last")
    print("\nCONTACT POLICY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

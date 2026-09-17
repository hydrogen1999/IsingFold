"""Gate 2 of the constructor ladder: small instances with dead ends, train set and held-out.

The tiny gate showed the empty-start environment and a 16-weight linear REINFORCE actor can
learn one instance (K3 into C5). This gate asks the next question in order: does the same
actor, trained on a fixed set of small instances whose hosts carry dead-end branches and
placement ambiguity, raise its valid-COMMIT rate on those instances, and on instances it has
never seen? Two stages: ``a`` (2 to 4 variables on cycles with pendant dead ends and a chord)
and ``b`` (4 to 8 variables on small grids with holes and dead ends). Hosts and logical graphs
are generated from seeds; every instance is certified embeddable by minorminer at generation
time and that embedding is discarded. During training and evaluation minorminer is forbidden,
and the task raises on any access to a witness, an initial embedding or a ground energy.
Feasibility only: no annealing measurement, no quality claim, no qubit penalty.
"""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [finder for finder in sys.meta_path
                    if not ("editable" in str(type(finder)).lower()
                            and "isingfold" in str(type(finder)).lower())]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "probes")]

import minorminer
import networkx as nx
import numpy as np
import torch

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import OPCODES
from constructor_learning import constructor_loss
from constructor_rollout import episode
from constructor_tiny_gate import Actor, Features, no_completion_solver

STAGES = ("a", "b")
FEATURE_WIDTH = len(OPCODES) + 8


class Task:
    """A feasibility task; nothing evaluator-only is reachable from it."""

    def __init__(self, name, logical, host, lineage):
        self.logical, self.host = logical, host
        self.problem = LogicalProblem.from_dicts(
            {v: 0. for v in logical}, {edge: -1. for edge in logical.edges()})
        self.name, self.lineage = name, lineage

    @property
    def witness(self):
        raise AssertionError("witness accessed")

    @property
    def ground_energy(self):
        raise AssertionError("quality labels accessed in feasibility gate")

    @property
    def initial_embedding(self):
        raise AssertionError("initial embedding accessed")


def logical_graph(rng, n):
    """A connected logical graph on n nodes from a small family; nodes 0..n-1."""
    families = ["path"]
    if n >= 3:
        families += ["cycle", "star"]
    if n <= 4:
        families.append("complete")
    if n >= 4:
        families += ["random", "triangle_tail"]
    family = str(rng.choice(families))
    if family == "path":
        g = nx.path_graph(n)
    elif family == "cycle":
        g = nx.cycle_graph(n)
    elif family == "star":
        g = nx.star_graph(n - 1)
    elif family == "complete":
        g = nx.complete_graph(n)
    elif family == "triangle_tail":
        g = nx.complete_graph(3)
        g.add_edges_from((i - 1 if i > 3 else 2, i) for i in range(3, n))
    else:
        while True:
            g = nx.gnp_random_graph(n, .5, seed=int(rng.integers(2 ** 31)))
            if nx.is_connected(g):
                break
    return nx.convert_node_labels_to_integers(g), family


def _attach_dead_ends(g, rng, count, max_length):
    """Pendant paths: a root placed on one must either be long or block the way back."""
    anchors = rng.choice(sorted(g.nodes()), size=min(count, g.number_of_nodes()), replace=False)
    for anchor in anchors:
        previous = int(anchor)
        for _ in range(int(rng.integers(1, max_length + 1))):
            node = g.number_of_nodes()
            g.add_edge(previous, node)
            previous = node
    return g


def host_graph(rng, stage):
    """Stage a: cycle plus dead ends and a chord. Stage b: grid with holes plus dead ends."""
    if stage == "a":
        m = int(rng.integers(5, 10))
        g = nx.cycle_graph(m)
        u, v = rng.choice(m, size=2, replace=False)
        if not g.has_edge(int(u), int(v)):
            g.add_edge(int(u), int(v))
        g = _attach_dead_ends(g, rng, int(rng.integers(1, 4)), 2)
    elif stage == "b":
        rows, cols = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols))
        for _ in range(int(rng.integers(0, 3))):
            candidates = [n for n in g.nodes() if nx.is_connected(g.subgraph(set(g) - {n}))]
            if candidates:
                g.remove_node(int(rng.choice(candidates)))
        g = nx.convert_node_labels_to_integers(g)
        g = _attach_dead_ends(g, rng, int(rng.integers(1, 4)), 3)
    else:
        raise ValueError("stage must be one of %s" % (STAGES,))
    return nx.convert_node_labels_to_integers(g)


def certified_embeddable(logical, host, seed, tries=50):
    """Generation-time certificate only; the embedding it finds is thrown away."""
    found = minorminer.find_embedding(list(logical.edges()), list(host.edges()),
                                      tries=tries, random_seed=int(seed) % (2 ** 31))
    return bool(found)


def isomorphic_pair(a, b):
    return nx.is_isomorphic(a.logical, b.logical) and nx.is_isomorphic(a.host, b.host)


def generate(stage, count, seed, lineage, exclude=(), max_attempts=2000):
    """Certified-embeddable instances, none isomorphic (logical and host) to ``exclude``."""
    rng = np.random.default_rng(seed)
    lo, hi = (2, 4) if stage == "a" else (4, 8)
    tasks = []
    for attempt in range(max_attempts):
        if len(tasks) == count:
            break
        n = int(rng.integers(lo, hi + 1))
        logical, family = logical_graph(rng, n)
        host = host_graph(rng, stage)
        if logical.number_of_nodes() > host.number_of_nodes():
            continue
        if not certified_embeddable(logical, host, rng.integers(2 ** 31)):
            continue
        task = Task("%s-%s-n%d-m%d-%d" % (lineage, family, n, host.number_of_nodes(), len(tasks)),
                    logical, host, lineage)
        if any(isomorphic_pair(task, other) for other in list(exclude) + tasks):
            continue
        tasks.append(task)
    if len(tasks) < count:
        raise RuntimeError("could not generate %d certified instances for stage %s" % (count, stage))
    return tasks


def build_sets(stage, n_train, n_heldout, seed):
    train = generate(stage, n_train, seed, "train-%s-s%d" % (stage, seed))
    heldout = generate(stage, n_heldout, seed + 7919, "heldout-%s-s%d" % (stage, seed), exclude=train)
    return train, heldout


def make_actor(kind, width):
    if kind == "linear":
        linear = torch.nn.Linear(FEATURE_WIDTH, 1, bias=False)
        torch.nn.init.zeros_(linear.weight)
        return Actor(linear)
    if kind == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(FEATURE_WIDTH, width), torch.nn.SiLU(),
                                  torch.nn.Linear(width, 1, bias=False))
        torch.nn.init.zeros_(net[-1].weight)
        return Actor(net)
    raise ValueError("actor must be linear or mlp")


def paired_boot(values, seed=0, draws=2000):
    d = np.asarray(values, dtype=float)
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(draws)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run(args, train, heldout):
    """Train and evaluate on prebuilt sets; the caller forbids the completion solver."""
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    features = {t.name: Features(t) for t in train + heldout}
    actor = make_actor(args.actor, args.width)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)

    def draw(task, seed, grad):
        with torch.set_grad_enabled(grad):
            return episode(task, actor, features[task.name], 1., args.max_steps,
                           np.random.default_rng(seed), args.episode_seconds,
                           train=True, objective="feasibility", evaluate_reward=False)

    def evaluate(tag):
        out = {}
        for name, tasks in (("train", train), ("heldout", heldout)):
            rates = {}
            for k, t in enumerate(tasks):
                records = [draw(t, 100000 + 1000 * k + i, False) for i in range(args.eval_episodes)]
                rates[t.name] = float(np.mean([r["valid"] for r in records]))
            row = {"evaluation": tag, "set": name, "seed": args.seed, "stage": args.stage,
                   "rate": float(np.mean(list(rates.values()))), "instances": len(tasks),
                   "episodes_per_instance": args.eval_episodes, "per_instance": rates}
            print(json.dumps(row), flush=True)
            out[name] = rates
        return out

    print(json.dumps({
        "protocol": "constructor-curriculum-gate2-v1",
        "scope": "feasibility on a fixed small train set and unseen held-out instances; "
                 "no quality claim; no completion solver at train or test time",
        "stage": args.stage, "seed": args.seed, "actor": args.actor, "width": args.width,
        "parameters": sum(p.numel() for p in actor.parameters()), "features": FEATURE_WIDTH,
        "train": [t.name for t in train], "heldout": [t.name for t in heldout],
        "train_sizes": [(t.logical.number_of_nodes(), t.host.number_of_nodes()) for t in train],
        "heldout_sizes": [(t.logical.number_of_nodes(), t.host.number_of_nodes()) for t in heldout],
        "iterations": args.iterations, "instances_per_iteration": args.instances_per_iteration,
        "episodes_per_instance": args.episodes, "eval_episodes": args.eval_episodes,
        "learning_rate": args.learning_rate, "baseline": "loo within instance", "entropy_coef": 0.,
        "max_steps": args.max_steps, "episode_seconds": args.episode_seconds,
        "certificate": "minorminer at generation only; forbidden afterwards",
    }), flush=True)
    started = time.monotonic()
    history = {"init": evaluate("init")}
    order = np.random.default_rng(args.seed + 1)
    for iteration in range(args.iterations):
        picks = order.choice(len(train), size=min(args.instances_per_iteration, len(train)), replace=False)
        losses, valid, rewards, entropies = [], 0, [], []
        for slot, idx in enumerate(picks):
            t = train[int(idx)]
            records = [draw(t, args.seed * 1000000 + iteration * 1000 + slot * 100 + i, True)
                       for i in range(args.episodes)]
            loss, metrics = constructor_loss(records, baseline="loo", entropy_coef=0.)
            losses.append(loss); valid += sum(r["valid"] for r in records)
            rewards.append(metrics["reward_mean"]); entropies.append(metrics["normalized_entropy"])
        total = torch.stack(losses).mean()
        optimizer.zero_grad()
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        if iteration % 10 == 9:
            print(json.dumps({"iteration": iteration, "seed": args.seed, "train_valid": valid,
                              "train_episodes": len(picks) * args.episodes,
                              "reward": float(np.mean(rewards)), "entropy": float(np.mean(entropies)),
                              "grad_norm": float(norm), "seconds": time.monotonic() - started}), flush=True)
        if (iteration + 1) % args.eval_every == 0 and iteration + 1 < args.iterations:
            history["iter %d" % iteration] = evaluate("iter %d" % iteration)
    history["final"] = evaluate("final")
    summary = {"summary": "gate2", "stage": args.stage, "seed": args.seed, "actor": args.actor}
    for name in ("train", "heldout"):
        before, after = history["init"][name], history["final"][name]
        diff = paired_boot([after[k] - before[k] for k in before])
        summary[name] = {"init": float(np.mean(list(before.values()))),
                         "final": float(np.mean(list(after.values()))),
                         "gain": diff[0], "gain_ci": [diff[1], diff[2]], "instances": len(before)}
    summary["seconds"] = time.monotonic() - started
    print(json.dumps(summary), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": actor.state_dict(), "actor": args.actor, "width": args.width,
                    "features": FEATURE_WIDTH, "stage": args.stage, "seed": args.seed,
                    "summary": summary}, args.out)
    print("CURRICULUM GATE 2 DONE", flush=True)
    return summary


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="a")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--heldout", type=int, default=12)
    parser.add_argument("--actor", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--instances-per-iteration", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--episode-seconds", type=float, default=30.)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be a nonnegative 32-bit integer")
    if min(args.train, args.heldout, args.iterations, args.eval_episodes, args.eval_every,
           args.instances_per_iteration, args.max_steps, args.width) < 1:
        parser.error("set sizes, iterations, evaluation, width and horizon must be positive")
    if args.episodes < 2:
        parser.error("leave-one-out requires at least two episodes per instance")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.episode_seconds <= 0:
        parser.error("learning rate and episode seconds must be positive and finite")
    return args


def main(argv=None):
    args = parse(argv)
    # the certificate needs minorminer; everything after this line must not
    train, heldout = build_sets(args.stage, args.train, args.heldout, args.seed)
    with no_completion_solver():
        run(args, train, heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

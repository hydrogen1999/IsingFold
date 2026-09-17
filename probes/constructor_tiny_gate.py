"""Same-instance feasibility overfit gate on K3 embedded into a five-node cycle.

This runs the real empty-start construction environment and its existing macro-action
support. A zero-initialized 16-weight linear actor learns by REINFORCE with a leave-one-out
baseline. It is a learnability diagnostic, not generalization or annealing-quality evidence.
No witness, supplied initial embedding, quality labels or minorminer calls are allowed.
"""
# Pin imports to this checkout before loading the package.
# ruff: noqa: E402
import argparse
from collections import Counter
from contextlib import contextmanager
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


class Task:
    def __init__(self):
        self.logical = nx.complete_graph(3)
        self.host = nx.cycle_graph(5)
        self.problem = LogicalProblem.from_dicts(
            {v: 0. for v in self.logical}, {edge: -1. for edge in self.logical.edges()})
        self.name = "triangle-on-cycle5"
        self.lineage = "tiny-overfit-only"

    @property
    def witness(self):
        raise AssertionError("witness accessed")

    @property
    def ground_energy(self):
        raise AssertionError("quality labels accessed in feasibility diagnostic")

    @property
    def initial_embedding(self):
        raise AssertionError("initial embedding accessed")


class Features:
    """Opcode, five successor deltas and three terminal/restart indicators.

    The summary of a state is computed once per state and reused across the candidates of
    a decision; a candidate's successor summary is the state's summary adjusted by the
    variables the candidate touches, so an observation costs the affected chains rather
    than every chain (at four hundred placed chains the full recomputation per candidate
    was two seconds a step). ``summary_reference`` is the direct computation, kept for the
    equality test."""

    def __init__(self, task):
        self.task = task
        self.budget = len(task.host)
        self._neighbours = {q: tuple(task.host[q]) for q in task.host}
        self._logical_neighbours = {v: tuple(task.logical[v]) for v in task.logical}
        self._n = len(task.logical)
        self._m = max(1, task.logical.number_of_edges())
        self._state_key = None
        self._state_counts = None
        self._connected_cache = {}

    def _connected(self, chain):
        chain = frozenset(chain)
        hit = self._connected_cache.get(chain)
        if hit is None:
            if not chain:
                hit = True
            else:
                start = next(iter(chain)); seen = {start}; stack = [start]
                while stack:
                    q = stack.pop()
                    for r in self._neighbours.get(q, ()):
                        if r in chain and r not in seen:
                            seen.add(r); stack.append(r)
                hit = len(seen) == len(chain)
            if len(self._connected_cache) > 4096:
                self._connected_cache.clear()
            self._connected_cache[chain] = hit
        return hit

    def _touch(self, chains, v, u):
        a, b = chains.get(v, ()), chains.get(u, ())
        if not a or not b:
            return False
        if len(a) > len(b):
            a, b = b, a
        bset = b if isinstance(b, (set, frozenset)) else set(b)
        return any(r in bset for x in a for r in self._neighbours.get(x, ()))

    def _counts(self, chains):
        """(placed, realised edges, occupied qubits, memberships, disconnected chains)."""
        placed = sum(1 for v in self._logical_neighbours if chains.get(v))
        realized = sum(1 for v, u in self.task.logical.edges() if self._touch(chains, v, u))
        used = set().union(*chains.values()) if chains else set()
        memberships = sum(len(c) for c in chains.values())
        disconnected = sum(1 for c in chains.values() if c and not self._connected(c))
        return placed, realized, len(used), memberships, disconnected

    def _to_row(self, counts):
        placed, realized, used, memberships, disconnected = counts
        host = len(self.task.host)
        return np.array([placed / self._n, realized / self._m, used / host,
                         (memberships - used) / host, disconnected / self._n], dtype=np.float32)

    def summary_reference(self, chains):
        return self._to_row(self._counts(chains))

    def summary(self, chains):
        key = frozenset((v, frozenset(c)) for v, c in chains.items() if c)
        if key != self._state_key:
            self._state_key = key
            self._state_counts = self._counts(chains)
        return self._to_row(self._state_counts)

    def successor_summary(self, chains, after, affected):
        """The successor's summary from the state's counts and the affected variables."""
        self.summary(chains)
        placed, realized, used, memberships, disconnected = self._state_counts
        affected = set(affected)
        edges = {tuple(sorted((v, u), key=repr)) for v in affected for u in self._logical_neighbours[v]}
        for v in affected:
            before, now = chains.get(v, frozenset()), after.get(v, frozenset())
            placed += bool(now) - bool(before)
            memberships += len(now) - len(before)
            disconnected += (bool(now) and not self._connected(now)) - (bool(before) and not self._connected(before))
        for v, u in edges:
            realized += self._touch(after, v, u) - self._touch(chains, v, u)
        # occupied qubits: recount only if the affected chains changed the union
        old_q = set().union(*(chains.get(v, frozenset()) for v in affected)) if affected else set()
        new_q = set().union(*(after.get(v, frozenset()) for v in affected)) if affected else set()
        if old_q != new_q:
            others = set()
            for v, c in chains.items():
                if v not in affected and c:
                    others |= c
            used = len(others | new_q)
        return self._to_row((placed, realized, used, memberships, disconnected))

    def observe(self, candidate, chains, *, state, ctx, steps_left, max_steps):
        opcode = candidate.opcode.value
        if opcode == "COMMIT":
            after = dict(state.archive[candidate.archive_ref].chains)
            before, new = self.summary(chains), self.summary_reference(after)
        else:
            after = dict(chains)
            after.update(candidate.new_chains)
            affected = [v for v in candidate.new_chains if after.get(v, frozenset()) != chains.get(v, frozenset())]
            before = self.summary(chains)
            new = self.successor_summary(chains, after, affected) if affected else before
        onehot = [float(opcode == getattr(item, "value", item)) for item in OPCODES]
        return np.asarray(onehot + list(new - before)
                          + [float(opcode == "COMMIT") * new[0],
                             float(opcode == "STOP") * new[1],
                             float(opcode == "RESTART") * before[1]], dtype=np.float32)


class Actor(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.m = linear

    def forward(self, rows):
        return self.m(rows).squeeze(-1)


def forbidden(*args, **kwargs):
    raise AssertionError("minorminer called")


@contextmanager
def no_completion_solver():
    """Make accidental completion calls fail, restoring the module for test callers."""
    original = minorminer.find_embedding
    minorminer.find_embedding = forbidden
    try:
        yield
    finally:
        minorminer.find_embedding = original


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    task = Task()
    features = Features(task)
    linear = torch.nn.Linear(len(OPCODES) + 8, 1, bias=False)
    torch.nn.init.zeros_(linear.weight)
    actor = Actor(linear)
    optimizer = torch.optim.Adam(actor.parameters(), lr=.03)

    def draw(seed, grad):
        with torch.set_grad_enabled(grad):
            return episode(task, actor, features, 1., 32, np.random.default_rng(seed), 30.,
                           train=True, objective="feasibility", evaluate_reward=False)

    def evaluate(tag):
        records = [draw(100000 + i, False) for i in range(args.eval_episodes)]
        row = {
            "evaluation": tag, "seed": args.seed,
            "valid": sum(r["valid"] for r in records), "episodes": len(records),
            "rate": float(np.mean([r["valid"] for r in records])),
            "steps": float(np.mean([r["steps"] for r in records])),
            "seconds": sum(r["secs"] for r in records),
            "ops": dict(Counter(d.opcode for r in records for d in r["decisions"])),
            "reasons": dict(Counter(r["reason"] for r in records)),
        }
        for result in records:
            if result["valid"]:
                assert any(len(chain) > 1 for chain in result["embedding"].values()), (
                    "triangle cannot embed as singletons in cycle5")
        print(json.dumps(row), flush=True)
        return row

    print(json.dumps({
        "protocol": "constructor-tiny-feasibility-gate-v1",
        "scope": "same-instance feasibility overfit; no generalization or quality claim",
        "task": task.name, "seed": args.seed, "iterations": args.iterations,
        "episodes_per_iteration": args.episodes, "evaluation_episodes": args.eval_episodes,
        "parameters": sum(p.numel() for p in actor.parameters()), "features": len(OPCODES) + 8,
        "actor": "zero-initialized linear, no bias", "baseline": "loo",
        "learning_rate": .03, "entropy_coef": 0., "gradient_clip": 1.,
        "temperature": 1., "max_steps": 32, "episode_seconds": 30.,
        "qubit_cap": len(task.host), "completion_solver": None,
        "evaluation_seed_start": 100000,
        "training_seed_rule": "seed*1000000 + iteration*episodes + episode_index",
    }), flush=True)
    started = time.monotonic()
    evaluate("init")
    for iteration in range(args.iterations):
        records = [draw(args.seed * 1000000 + iteration * args.episodes + i, True)
                   for i in range(args.episodes)]
        loss, metrics = constructor_loss(records, baseline="loo", entropy_coef=0.)
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        if iteration % 10 == 9:
            print(json.dumps({"iteration": iteration, "seed": args.seed,
                              "valid": sum(r["valid"] for r in records),
                              "reward": metrics["reward_mean"],
                              "entropy": metrics["normalized_entropy"],
                              "grad_norm": float(norm)}), flush=True)
    evaluate("final")
    print(json.dumps({"total_seconds": time.monotonic() - started}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2 ** 32:
        parser.error("seed must be a nonnegative 32-bit integer")
    if args.iterations < 1 or args.eval_episodes < 1:
        parser.error("iterations and eval-episodes must be positive")
    if args.episodes < 2:
        parser.error("leave-one-out requires at least two episodes per iteration")
    with no_completion_solver():
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

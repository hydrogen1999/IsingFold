"""Seconds per constructor step against host size, split between the environment and the
observation, for the 16-channel tiny features and the 230-channel construction features.
A uniform policy (zero weights) on a fixed logical graph; this measures throughput only."""
# ruff: noqa: E402
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [finder for finder in sys.meta_path
                    if not ("editable" in str(type(finder)).lower()
                            and "isingfold" in str(type(finder)).lower())]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "probes")]

import networkx as nx
import numpy as np
import torch

from constructor_curriculum import FEATURE_WIDTHS, Task, hardware_fragment, logical_graph, make_actor, make_features
from constructor_rollout import episode
from constructor_tiny_gate import no_completion_solver

WRAP_SECONDS = {}


def timed(fc):
    """Count the seconds spent inside observe(), the constructor's own feature work."""
    original = fc.observe

    def observe(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            WRAP_SECONDS["observe"] = WRAP_SECONDS.get("observe", 0.) + time.perf_counter() - started
    fc.observe = observe
    return fc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=("pegasus", "zephyr"), default="pegasus")
    ap.add_argument("--generator-size", type=int, default=6)
    ap.add_argument("--sizes", default="32,64,128,256,512")
    ap.add_argument("--variables", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    rng = np.random.default_rng(a.seed)
    logical, family = logical_graph(rng, a.variables)
    for size in [int(x) for x in a.sizes.split(",")]:
        host = hardware_fragment(np.random.default_rng(a.seed), a.family, size, size, a.generator_size)
        task = Task("cost-%s-%d" % (a.family, size), logical, host, "cost")
        for kind in ("tiny", "construction"):
            WRAP_SECONDS.clear()
            fc = timed(make_features(kind, task))
            actor = make_actor("linear", 32, FEATURE_WIDTHS[kind])
            started = time.perf_counter()
            with no_completion_solver(), torch.no_grad():
                record = episode(task, actor, fc, 1., a.steps, np.random.default_rng(a.seed), 3600.,
                                 train=True, objective="feasibility", evaluate_reward=False)
            total = time.perf_counter() - started
            steps = max(1, record["steps"])
            print(json.dumps({"family": a.family, "host_qubits": host.number_of_nodes(),
                              "host_edges": host.number_of_edges(), "variables": a.variables,
                              "features": kind, "steps": record["steps"], "reason": record["reason"],
                              "seconds_per_step": total / steps,
                              "observe_seconds_per_step": WRAP_SECONDS.get("observe", 0.) / steps,
                              "environment_seconds_per_step": (total - WRAP_SECONDS.get("observe", 0.)) / steps}),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

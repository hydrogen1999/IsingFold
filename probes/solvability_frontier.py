"""Where solve probability is measurable at all on the congestion axis.

The congested corpora report residual rather than solve probability because at 434 variables the
witness scores zero hits in a 512-read block. Zero hits bounds the per-read rate below about
0.006 at 95 percent confidence; it does not establish that the rate is zero, and no wording here
should say it does. Fill sets how hard an instance is to embed; frustrated-loop density sets how
hard it is to solve once embedded, and the two were being read as one number.

Loop density is not available as a knob. Every support node survives as a field entry, but an
edge survives only where the summed coupling is nonzero, so a sparser problem is a sparser
embedding problem and the congestion claim moves with it.

Three knobs remain, and none of them is free. Shrinking the host at a fixed fill builds a
different logical graph, so it holds one ratio rather than the instance; anneal depth belongs to
the solver, and a second depth is a newly declared operating point, not the registered one; a
local field leaves the edge set alone but sets each field to the negative of a planted spin, so
it publishes part of the optimum and it also moves the root-mean-square scale the registered
chain strengths are derived from. This maps decoded solve probability, residual energy and broken
chains over all three, and asks minorminer at the same cells, so the operating regime can be
sited on measurement rather than on assumption.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import networkx as nx
import numpy as np
from isingfold.rl.data.generate import host_graph
from isingfold.rl.data.planting import PlantingError, frustrated_loops
from isingfold.rl.evaluator import sample_program
from isingfold.rl.program import compile_program, strength_registry

from _context import host_context
from _initializers import minorminer_initializer
from gen_fill_corpus import plant_partition, quotient, witness_occupancy


def grow(chains_map, host, extra, rng):
    """The paper's growth control: absorb free qubits into random chains, one contact at a time.

    A longer chain is a weaker effective ferromagnet at the same strength, so this is a valid
    embedding of the same logical graph that should decode worse. It is the cheapest evidence
    that a restored solve probability responds to the embedding and not only to the instance.
    """
    used = {q for c in chains_map.values() for q in c}
    out = {v: set(c) for v, c in chains_map.items()}
    order = list(out)
    added = 0
    for _ in range(40 * max(1, extra)):
        if added >= extra:
            break
        v = order[rng.integers(0, len(order))]
        frontier = [t for q in out[v] for t in host.neighbors(q) if t not in used]
        if not frontier:
            continue
        q = frontier[rng.integers(0, len(frontier))]
        out[v].add(q)
        used.add(q)
        added += 1
    return {v: frozenset(c) for v, c in out.items()}, added


def measure(chains_map, host, problem, ground, ctx, sweeps, reads, seed):
    """Every registered strength, reporting all three read-block channels, not only the rate.

    Solve probability, residual energy and broken-chain fraction answer different questions and
    are not interchangeable. A benchmark that reports only the first cannot tell an embedding
    that never breaks from one whose breaks are invisible because the instance is easy.
    """
    strengths = strength_registry(problem, ctx.strength_ratios, ctx.epsilon_strength)
    blocks = []
    for index, strength in enumerate(strengths):
        program = compile_program(chains_map, host, problem, strength, index)
        blocks.append(sample_program(program, chains_map, problem, ground, num_reads=reads,
                                     seed=seed + index, num_sweeps=sweeps,
                                     beta_range=ctx.beta_range))
    return blocks


def channels(blocks):
    """The registered strength, and the best strength by each channel's own sign."""
    best = min(range(len(blocks)), key=lambda i: blocks[i].mean_residual)
    return {"p_registered": blocks[1].rate,
            "p_best": max(b.rate for b in blocks),
            "p_by_strength": [b.rate for b in blocks],
            "residual_registered": blocks[1].mean_residual,
            "residual_best": blocks[best].mean_residual,
            "residual_by_strength": [b.mean_residual for b in blocks],
            "broken_registered": blocks[1].broken_fraction,
            "broken_best": blocks[best].broken_fraction}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="pegasus")
    ap.add_argument("--sizes", default="2,3,4,5,6")
    ap.add_argument("--fill", type=float, default=0.90)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--alpha-clause", type=float, default=0.9)
    ap.add_argument("--clause-length", type=int, default=5)
    ap.add_argument("--weights", default="0.4,1.0,2.5")
    ap.add_argument("--field-rates", default="0.0")
    ap.add_argument("--sweeps", default="200,2000,20000")
    ap.add_argument("--reads", type=int, default=512)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--lmin", type=int, default=1)
    ap.add_argument("--lmax", type=int, default=6)
    ap.add_argument("--mm-tries", type=int, default=20)
    ap.add_argument("--degrade", type=int, default=0,
                    help="free qubits to absorb into random chains, as a worse "
                         "but still valid embedding of the same logical graph")
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    sweeps_grid = [int(x) for x in a.sweeps.split(",")]
    field_rates = [float(x) for x in a.field_rates.split(",")]
    weights = tuple(float(x) for x in a.weights.split(","))
    mm = minorminer_initializer(a.mm_tries)

    print(json.dumps({"probe": "solvability_frontier", "host": a.host, "sizes": sizes,
                      "fill": a.fill, "sweeps": sweeps_grid, "reads": a.reads,
                      "field_rates": field_rates, "instances": a.instances}), flush=True)
    print("  %-9s %5s %5s %6s %7s %6s %8s %8s %8s %8s %8s"
          % ("host", "vars", "qubits", "fill", "chain", "field", "sweeps", "p_reg", "p_best",
             "p_mm", "p_grown"), flush=True)

    local = 0
    for size in sizes:
        host = host_graph(a.host, size)
        cap = host.number_of_nodes()
        ctx = host_context(cap)
        rng = np.random.default_rng(a.seed + size)
        for i in range(a.instances):
            local += 1
            chains = plant_partition(host, a.fill, a.alpha, a.lmin, a.lmax, rng)
            logical = quotient(host, chains)
            if not nx.is_connected(logical):
                comp = max(nx.connected_components(logical), key=len)
                chains = [chains[k] for k in sorted(comp)]
                logical = quotient(host, chains)
            for rate in field_rates:
                try:
                    planted = frustrated_loops(logical, alpha=a.alpha_clause,
                                               seed=a.seed + local, max_length=a.clause_length,
                                               weight_choices=weights, field_clause_rate=rate)
                except PlantingError:
                    continue
                graph = planted.problem.graph
                witness = {v: frozenset(chains[v]) for v in graph.nodes()}
                used = witness_occupancy(witness)
                mean_chain = used / float(graph.number_of_nodes())

                t0 = time.time()
                found = mm(graph, host, a.seed + local)
                mm_secs = time.time() - t0

                degraded, added = (None, 0)
                if a.degrade > 0:
                    degraded, added = grow(witness, host, a.degrade, rng)

                for sweeps in sweeps_grid:
                    def channels_of(cm):
                        return channels(measure(cm, host, planted.problem,
                                                planted.ground_energy, ctx, sweeps, a.reads,
                                                a.seed + 1000 * local))

                    wit = channels_of(witness)
                    rates = wit["p_by_strength"]
                    grown = channels_of(degraded) if degraded is not None else None
                    grown_rate = grown["p_best"] if grown else None
                    mm_ch = None
                    if found is not None:
                        mm_ch = channels_of({v: frozenset(found[v]) for v in graph.nodes()})
                    mm_rate = mm_ch["p_best"] if mm_ch else None
                    row = {"host": "%s%d" % (a.host, size), "qubits": cap,
                           "vars": graph.number_of_nodes(), "witness_qubits": used,
                           "realised_fill": used / float(cap), "mean_chain": mean_chain,
                           "field_rate": rate, "sweeps": sweeps, "reads": a.reads,
                           "witness": wit, "grown": grown, "minorminer": mm_ch,
                           "p_registered": rates[1], "p_by_strength": rates,
                           "p_best": max(rates), "mm_valid": found is not None,
                           "mm_qubits": (sum(len(c) for c in found.values())
                                         if found is not None else None),
                           "mm_secs": mm_secs, "p_minorminer": mm_rate,
                           "p_grown": grown_rate, "grown_extra": added,
                           "instance": local}
                    print(json.dumps(row), flush=True)
                    print("  %-9s %5d %6d %6.2f %7.2f %6.2f %8d %8.3f %8.3f %8s %8s"
                          % (row["host"], row["vars"], used, row["realised_fill"], mean_chain,
                             rate, sweeps, rates[1], max(rates),
                             ("%.3f" % mm_rate) if mm_rate is not None else "-",
                             ("%.3f" % grown_rate) if grown_rate is not None else "-"),
                          flush=True)
                    print("      residual  witness %.4f  grown %s  minorminer %s   broken "
                          "witness %.3f  grown %s"
                          % (wit["residual_best"],
                             ("%.4f" % grown["residual_best"]) if grown else "-",
                             ("%.4f" % mm_ch["residual_best"]) if mm_ch else "-",
                             wit["broken_best"],
                             ("%.3f" % grown["broken_best"]) if grown else "-"), flush=True)
    print("SOLVABILITY FRONTIER DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

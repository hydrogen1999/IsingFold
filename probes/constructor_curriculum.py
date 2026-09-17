"""Gate 2 of the constructor ladder: small instances with dead ends, train set and held-out.

The tiny gate showed the empty-start environment and a 16-weight linear REINFORCE actor can
learn one instance (K3 into C5). This gate asks the next question in order: does the same
actor, trained on a fixed set of small instances whose hosts carry dead-end branches and
placement ambiguity, raise its valid-COMMIT rate on those instances, and on instances it has
never seen? Stages: ``a`` (2 to 4 variables on cycles with pendant dead ends and a chord), ``b`` (4 to 8
variables on small grids with holes and dead ends), ``p`` and ``z`` (4 to 8 variables on
connected 12 to 24 qubit fragments of Pegasus 2 and Zephyr 1, the target topologies), ``P``
and ``Z`` (8 to 14 variables on 32 to 64 qubit fragments of Pegasus 3 and Zephyr 2), ``F``
and ``G`` (12 to 20 variables on 64 to 128 qubit fragments of the same hosts). Hosts and logical graphs
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
from constructor_features import ConstructorFeatureContext, WIDTH as CONSTRUCTION_WIDTH
from constructor_protocol import evaluate_search, experiment_seed
from constructor_tiny_gate import Actor, Features, no_completion_solver
from layout_policy import LayoutActorCritic

# captured before any guard replaces it; only the minorminer comparison arm may use it
_ORIGINAL_FIND_EMBEDDING = minorminer.find_embedding

STAGES = ("a", "b", "p", "z", "P", "Z", "F", "G")
VARIABLES = {"a": (2, 4), "b": (4, 8), "p": (4, 8), "z": (4, 8), "P": (8, 14), "Z": (8, 14),
             "F": (12, 20), "G": (12, 20)}
FRAGMENT = (12, 24)
# stage -> (hardware family, generator size, fragment size range)
HARDWARE = {"p": ("pegasus", 2, (12, 24)), "z": ("zephyr", 1, (12, 24)),
            "P": ("pegasus", 3, (32, 64)), "Z": ("zephyr", 2, (32, 64)),
            "F": ("pegasus", 3, (64, 128)), "G": ("zephyr", 2, (64, 128))}
FEATURE_WIDTH = len(OPCODES) + 8
FEATURE_WIDTHS = {"tiny": FEATURE_WIDTH, "construction": CONSTRUCTION_WIDTH}


class Task:
    """A curriculum task. The witness and any initial embedding are never reachable; the
    ground energy is reachable only when supplied, and only the quality reward and
    assessment backends read it."""

    def __init__(self, name, logical, host, lineage, problem=None, ground_energy=None,
                 prefix_source=None):
        self.logical, self.host = logical, host
        self.problem = problem if problem is not None else LogicalProblem.from_dicts(
            {v: 0. for v in logical}, {edge: -1. for edge in logical.edges()})
        self.name, self.lineage = name, lineage
        self._ground_energy = ground_energy
        # a valid embedding used only to start *training* episodes part-way (curriculum);
        # never set on held-out tasks, never read by features, evaluation starts from empty
        self.prefix_source = prefix_source

    @property
    def witness(self):
        raise AssertionError("witness accessed")

    @property
    def ground_energy(self):
        if self._ground_energy is None:
            raise AssertionError("quality labels accessed in feasibility gate")
        return self._ground_energy

    @property
    def initial_embedding(self):
        raise AssertionError("initial embedding accessed")


def exact_ground_energy(problem, limit=20):
    """Minimum Ising energy by enumeration in chunks; for the generated instances (at most
    20 variables in any stage)."""
    nodes = sorted(problem.h)
    n = len(nodes)
    if n > limit:
        raise ValueError("enumeration is limited to %d variables" % limit)
    index = {v: i for i, v in enumerate(nodes)}
    h = np.array([float(problem.h[v]) for v in nodes])
    couplings = [(index[u], index[v], float(j)) for (u, v), j in problem.j.items()]
    best = float("inf")
    chunk = 1 << min(n, 16)
    for start in range(0, 2 ** n, chunk):
        codes = np.arange(start, min(start + chunk, 2 ** n))
        states = ((codes[:, None] >> np.arange(n)) & 1) * 2 - 1
        energy = states @ h
        for a, b, j in couplings:
            energy = energy + j * states[:, a] * states[:, b]
        best = min(best, float(energy.min()))
    return best


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


def hardware_fragment(rng, family, lo=FRAGMENT[0], hi=FRAGMENT[1], size=None):
    """A connected induced subgraph of a Pegasus or Zephyr graph with the hardware's own
    degree structure and dead ends: a breadth-first ball, then random removals that keep
    it connected, down to a size drawn from [lo, hi]."""
    import dwave_networkx as dnx
    if size is None:
        size = 2 if family == "pegasus" else 1
    full = dnx.pegasus_graph(size) if family == "pegasus" else dnx.zephyr_graph(size)
    full = nx.convert_node_labels_to_integers(full, ordering="sorted")
    target = int(rng.integers(lo, hi + 1))
    start = int(rng.choice(sorted(full.nodes())))
    order, seen = [start], {start}
    for node in order:
        for neighbour in sorted(full.neighbors(node)):
            if neighbour not in seen:
                seen.add(neighbour); order.append(neighbour)
        if len(order) >= 2 * target:
            break
    if target >= 512:
        # a breadth-first prefix is connected by construction; the random-removal shaping
        # below is quadratic and only affordable for the small fragments
        sub = full.subgraph(order[:target]).copy()
        return nx.convert_node_labels_to_integers(sub, ordering="sorted")
    sub = full.subgraph(order[:2 * target]).copy()
    while sub.number_of_nodes() > target:
        removable = [n for n in sorted(sub.nodes())
                     if nx.is_connected(sub.subgraph(set(sub) - {n}))]
        sub.remove_node(int(rng.choice(removable)))
    return nx.convert_node_labels_to_integers(sub, ordering="sorted")


def host_graph(rng, stage):
    """a: cycle plus dead ends and a chord. b: grid with holes plus dead ends.
    p, z: 12 to 24 qubit fragments of Pegasus 2 and Zephyr 1. P, Z: 32 to 64 qubit
    fragments of Pegasus 3 and Zephyr 2 for 8 to 14 variables. F, G: 64 to 128 qubit
    fragments of the same hosts for 12 to 20 variables, the size of the small corpora."""
    if stage in HARDWARE:
        family, size, (lo, hi) = HARDWARE[stage]
        return hardware_fragment(rng, family, lo, hi, size)
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
    return bool(certificate_embedding(logical, host, seed, tries))


def certificate_embedding(logical, host, seed, tries=50):
    """The certificate itself, kept only as a training prefix source for train tasks."""
    found = minorminer.find_embedding(list(logical.edges()), list(host.edges()),
                                      tries=tries, random_seed=int(seed) % (2 ** 31))
    if not found:
        return None
    return {v: frozenset(c) for v, c in found.items()}


def isomorphic_pair(a, b):
    return nx.is_isomorphic(a.logical, b.logical) and nx.is_isomorphic(a.host, b.host)


def generate(stage, count, seed, lineage, exclude=(), max_attempts=2000):
    """Certified-embeddable instances, none isomorphic (logical and host) to ``exclude``."""
    rng = np.random.default_rng(seed)
    lo, hi = VARIABLES[stage]
    tasks = []
    for attempt in range(max_attempts):
        if len(tasks) == count:
            break
        n = int(rng.integers(lo, hi + 1))
        logical, family = logical_graph(rng, n)
        host = host_graph(rng, stage)
        if logical.number_of_nodes() > host.number_of_nodes():
            continue
        certificate = certificate_embedding(logical, host, rng.integers(2 ** 31))
        if not certificate:
            continue
        task = Task("%s-%s-n%d-m%d-%d" % (lineage, family, n, host.number_of_nodes(), len(tasks)),
                    logical, host, lineage)
        task._ground_energy = exact_ground_energy(task.problem)
        task._certificate = certificate
        if any(isomorphic_pair(task, other) for other in list(exclude) + tasks):
            continue
        tasks.append(task)
    if len(tasks) < count:
        raise RuntimeError("could not generate %d certified instances for stage %s" % (count, stage))
    return tasks


def corpus_sets_from_tasks(tasks, cells, n_train, n_heldout, seed):
    """One guarded task per lineage from a planted corpus, split by lineage; the witness
    that certifies embeddability stays behind in the corpus task and is never reachable."""
    keep = [t for t in tasks if not cells or any(cell in t.name for cell in cells)]
    by_lineage = {}
    for t in keep:
        by_lineage.setdefault(t.lineage or t.name, []).append(t)
    lineages = sorted(by_lineage)
    np.random.default_rng(seed).shuffle(lineages)
    if len(lineages) < n_train + n_heldout:
        raise ValueError("corpus has %d lineages in the chosen cells, %d requested"
                         % (len(lineages), n_train + n_heldout))
    picked = [sorted(by_lineage[l], key=lambda t: t.name)[0] for l in lineages[:n_train + n_heldout]]
    wrapped = [Task(t.name, t.logical, t.host, t.lineage or t.name, getattr(t, "problem", None),
                    getattr(t, "ground_energy", None)) for t in picked]
    for w, t in zip(wrapped[:n_train], picked[:n_train]):
        w.prefix_source = getattr(t, "witness", None)   # train tasks only
    return wrapped[:n_train], wrapped[n_train:]


def manifest_split_sets(tasks, split, cells, n_train, n_heldout, seed, heldout_role="validation"):
    """Honour a corpus manifest's split: train from its train list, held-out from its
    validation list, and never touch its test list. Lists may name instances or lineages.
    ``heldout_role="test"`` is for the final table only: an evaluation-only pass over the
    test list, once, with a checkpoint chosen on validation."""
    def member(t, names):
        return t.name in names or (t.lineage or "") in names
    if heldout_role not in ("validation", "test"):
        raise ValueError("heldout_role must be validation or test")
    train_names, val_names, test_names = (set(split.get(k, [])) for k in ("train", "validation", "test"))
    keep = [t for t in tasks if not cells or any(cell in t.name for cell in cells)]
    train_pool = [t for t in keep if member(t, train_names) and not member(t, test_names)]
    val_pool = ([t for t in keep if member(t, val_names) and not member(t, test_names)]
                if heldout_role == "validation" else [t for t in keep if member(t, test_names)])
    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool); rng.shuffle(val_pool)
    if len(train_pool) < n_train or len(val_pool) < n_heldout:
        raise ValueError("manifest split holds %d train and %d validation tasks in the chosen cells, "
                         "%d and %d requested" % (len(train_pool), len(val_pool), n_train, n_heldout))
    wrapped = [Task(t.name, t.logical, t.host, t.lineage or t.name, getattr(t, "problem", None),
                    getattr(t, "ground_energy", None)) for t in train_pool[:n_train] + val_pool[:n_heldout]]
    for w, t in zip(wrapped[:n_train], train_pool[:n_train]):
        w.prefix_source = getattr(t, "witness", None)
    return wrapped[:n_train], wrapped[n_train:]


def build_corpus_sets(path, cells, n_train, n_heldout, seed, use_manifest_split=False,
                      heldout_role="validation"):
    from isingfold.rl.data.generate import load_instances
    tasks = load_instances(path)
    if use_manifest_split:
        split = {}
        splits_file = Path(path) / "splits.json"
        if splits_file.exists():
            split = json.load(open(splits_file))
        else:
            split = (json.load(open(Path(path) / "manifest.json")).get("split") or {})
        if not any(isinstance(split.get(k), list) for k in ("train", "validation", "test")):
            raise ValueError("corpus has no train/validation/test lists to honour")
        return manifest_split_sets(tasks, split, cells, n_train, n_heldout, seed, heldout_role)
    if heldout_role == "test":
        raise ValueError("a test-list evaluation needs --manifest-split")
    return corpus_sets_from_tasks(tasks, cells, n_train, n_heldout, seed)


def load_init(path, actor, kind, features):
    """Warm start from a lower rung: same actor kind and feature schema, or refuse."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    saved = blob.get("features")
    if isinstance(saved, (int, np.integer)):
        # checkpoints written before the feature option carried the width, not the name
        saved = {width: name for name, width in FEATURE_WIDTHS.items()}.get(int(saved), saved)
    if blob.get("actor") != kind or saved != features:
        raise ValueError("checkpoint actor/features %s/%s do not match %s/%s"
                         % (blob.get("actor"), saved, kind, features))
    actor.load_state_dict(blob["state"])
    return blob.get("summary")


def build_sets(stage, n_train, n_heldout, seed):
    train = generate(stage, n_train, seed, "train-%s-s%d" % (stage, seed))
    heldout = generate(stage, n_heldout, seed + 7919, "heldout-%s-s%d" % (stage, seed), exclude=train)
    for t in train:
        t.prefix_source = t._certificate
    for t in heldout:
        t.prefix_source = None
    return train, heldout


def prefix_fraction(schedule, iteration, iterations):
    """Linear schedule 'start:end' over the iterations; None when no curriculum."""
    if not schedule:
        return None
    start, end = (float(x) for x in schedule.split(":"))
    if iterations <= 1:
        return end
    return start + (end - start) * iteration / (iterations - 1)


def prefix_initializer(task, fraction, seed):
    """A random subset of the task's prefix source covering ``fraction`` of the variables,
    as the environment's partial-start initializer; None when nothing to start from."""
    source = getattr(task, "prefix_source", None)
    if not source or fraction is None or fraction <= 0:
        return None
    rng = np.random.default_rng(seed)
    variables = sorted(source, key=repr)
    # at least one variable stays unplaced: a complete prefix would be a valid embedding,
    # which construction mode must build, not be handed
    k = min(int(round(fraction * len(variables))), len(variables) - 1)
    if k <= 0:
        return None
    chosen = [variables[i] for i in sorted(rng.choice(len(variables), size=k, replace=False))]
    partial = {v: frozenset(source[v]) for v in chosen}
    return lambda logical, host, seed_: partial


def make_actor(kind, width, in_dim=FEATURE_WIDTH):
    """linear and mlp score candidates independently; contextual is the PR's candidate-set
    actor-critic (Deep Sets pooling, state value head) on the same rows."""
    if kind == "linear":
        linear = torch.nn.Linear(in_dim, 1, bias=False)
        torch.nn.init.zeros_(linear.weight)
        return Actor(linear)
    if kind == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(in_dim, width), torch.nn.SiLU(),
                                  torch.nn.Linear(width, 1, bias=False))
        torch.nn.init.zeros_(net[-1].weight)
        return Actor(net)
    if kind == "contextual":
        return LayoutActorCritic(width, in_dim=in_dim)
    raise ValueError("actor must be linear, mlp or contextual")


def make_features(kind, task):
    if kind == "tiny":
        return Features(task)
    if kind == "construction":
        return ConstructorFeatureContext(task, len(task.host))
    raise ValueError("features must be tiny or construction")


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
    features = {t.name: make_features(args.features, t) for t in train + heldout}
    actor = make_actor(args.actor, args.width, FEATURE_WIDTHS[args.features])
    init_summary = load_init(args.init, actor, args.actor, args.features) if args.init else None
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)

    quality = args.objective == "quality"

    wide = args.support == "wide"

    def draw(task, seed, grad, reward=None, initializer=None):
        with torch.set_grad_enabled(grad):
            return episode(task, actor, features[task.name], 1., args.max_steps,
                           np.random.default_rng(seed), args.episode_seconds,
                           train=True, objective=args.objective, reward_reads=args.reward_reads,
                           evaluate_reward=quality if reward is None else reward,
                           initializer=initializer, wide=wide)

    def evaluate_quality(tag):
        """The paper's protocol: proposals until the deadline, selection by measurement,
        assessment on a fresh block; the minorminer arm under the same deadline."""
        out = {}
        for name, tasks in (("train", train), ("heldout", heldout)):
            rows = []
            for k, t in enumerate(tasks):
                def learned(seed, seconds_left, t=t):
                    with torch.no_grad():
                        result = episode(t, actor, features[t.name], 1., args.max_steps,
                                         np.random.default_rng(seed), min(seconds_left, args.episode_seconds),
                                         train=True, objective="quality", evaluate_reward=False, wide=wide)
                    return result["terminal"] if result["valid"] else None

                def baseline(seed, seconds_left, t=t):
                    from constructor_baseline import baseline_proposal
                    minorminer.find_embedding = _ORIGINAL_FIND_EMBEDDING
                    try:
                        return baseline_proposal(t, len(t.host), seed, seconds_left,
                                                 args.baseline_tries, args.baseline_router_seconds)
                    finally:
                        minorminer.find_embedding = forbidden_solver

                def grown(seed, seconds_left, t=t):
                    minorminer.find_embedding = _ORIGINAL_FIND_EMBEDDING
                    try:
                        return grown_baseline_proposal(t, len(t.host), seed, seconds_left, args.grow_extra,
                                                       args.baseline_tries, args.baseline_router_seconds)
                    finally:
                        minorminer.find_embedding = forbidden_solver
                arms = [("policy", learned)]
                if args.comparison in ("minorminer", "minorminer_grown"):
                    arms.append(("minorminer", baseline))
                if args.comparison == "minorminer_grown":
                    arms.append(("minorminer_grown", grown))
                if experiment_seed(args.seed, "arm-order", t.name) % 2:
                    arms.reverse()
                row = {"instance": t.name}
                for arm, proposer in arms:
                    shapes = []

                    def measure(task, terminal, seed, reads, shapes=shapes):
                        # the last call of a search is the assessment of the chosen candidate
                        from constructor_objective import measure_terminal
                        chains = terminal.embedding
                        shapes.append({"qubits": sum(len(c) for c in chains.values()),
                                       "longest_chain": max(len(c) for c in chains.values()),
                                       "reads": reads})
                        return measure_terminal(task, terminal, seed, reads)
                    row[arm] = evaluate_search(t, proposer, deadline=args.deadline, select_cap=args.select_cap,
                                               selection_reads=args.selection_reads,
                                               assessment_reads=args.assessment_reads, objective="quality",
                                               seed=experiment_seed(args.seed, "validation", t.name, k),
                                               measure=measure)
                    chosen = [x for x in shapes if x["reads"] == args.assessment_reads]
                    row[arm]["chosen_qubits"] = chosen[-1]["qubits"] if chosen else None
                    row[arm]["chosen_longest_chain"] = chosen[-1]["longest_chain"] if chosen else None
                    row[arm]["candidate_qubits_mean"] = float(np.mean([x["qubits"] for x in shapes])) if shapes else None
                rows.append(row)
            valid = {arm: float(np.mean([r[arm]["valid"] for r in rows])) for arm in rows[0] if arm != "instance"}
            residual = {arm: [r[arm]["residual"] for r in rows if r[arm]["valid"] and r[arm]["residual"] is not None]
                        for arm in valid}
            both = [r for r in rows if "minorminer" in r and r["policy"]["valid"] and r["minorminer"]["valid"]
                    and r["policy"]["residual"] is not None and r["minorminer"]["residual"] is not None]
            paired = paired_boot([r["policy"]["residual"] - r["minorminer"]["residual"] for r in both])
            both_grown = [r for r in rows if "minorminer_grown" in r and r["policy"]["valid"] and r["minorminer_grown"]["valid"]
                          and r["policy"]["residual"] is not None and r["minorminer_grown"]["residual"] is not None]
            paired_grown = paired_boot([r["policy"]["residual"] - r["minorminer_grown"]["residual"] for r in both_grown])
            shape = {arm: {"chosen_qubits": float(np.mean([r[arm]["chosen_qubits"] for r in rows if r[arm]["chosen_qubits"] is not None] or [np.nan])),
                           "chosen_longest_chain": float(np.mean([r[arm]["chosen_longest_chain"] for r in rows if r[arm]["chosen_longest_chain"] is not None] or [np.nan]))}
                     for arm in valid}
            summary = {"evaluation": tag, "set": name, "seed": args.seed, "stage": args.stage,
                       "objective": "quality", "instances": len(tasks), "valid": valid, "chosen_shape": shape,
                       "mean_residual": {arm: (float(np.mean(v)) if v else None) for arm, v in residual.items()},
                       "measured": {arm: len(v) for arm, v in residual.items()},
                       "paired_residual_policy_minus_minorminer": paired, "paired_over": len(both),
                       "paired_residual_policy_minus_minorminer_grown": paired_grown,
                       "paired_grown_over": len(both_grown),
                       "policy_valid_minorminer_not": sum(1 for r in rows if "minorminer" in r
                                                          and r["policy"]["valid"] and not r["minorminer"]["valid"]),
                       "rows": rows}
            print(json.dumps(summary), flush=True)
            out[name] = summary
        return out

    def evaluate(tag):
        if quality:
            return evaluate_quality(tag)
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
        "corpus": args.corpus or None, "cells": args.cells or None,
        "manifest_split": bool(args.manifest_split), "heldout_role": args.heldout_role,
        "init": args.init or None, "init_summary": init_summary,
        "parameters": sum(p.numel() for p in actor.parameters()),
        "features": args.features, "feature_width": FEATURE_WIDTHS[args.features],
        "train": [t.name for t in train], "heldout": [t.name for t in heldout],
        "train_sizes": [(t.logical.number_of_nodes(), t.host.number_of_nodes()) for t in train],
        "heldout_sizes": [(t.logical.number_of_nodes(), t.host.number_of_nodes()) for t in heldout],
        "iterations": args.iterations, "instances_per_iteration": args.instances_per_iteration,
        "episodes_per_instance": args.episodes, "eval_episodes": args.eval_episodes,
        "learning_rate": args.learning_rate, "baseline": args.baseline + " within instance",
        "entropy_coef": 0., "prefix_curriculum": args.prefix_fraction or None,
        "train_prefix_sources": sum(1 for t in train if getattr(t, "prefix_source", None)),
        "heldout_prefix_sources": sum(1 for t in heldout if getattr(t, "prefix_source", None)),
        "max_steps": args.max_steps, "episode_seconds": args.episode_seconds, "support": args.support,
        "certificate": "minorminer at generation only; forbidden afterwards",
    }), flush=True)
    started = time.monotonic()
    history = {"init": evaluate("init")}
    order = np.random.default_rng(args.seed + 1)
    for iteration in range(args.iterations):
        picks = order.choice(len(train), size=min(args.instances_per_iteration, len(train)), replace=False)
        losses, valid, rewards, entropies = [], 0, [], []
        fraction = prefix_fraction(args.prefix_fraction, iteration, args.iterations)
        for slot, idx in enumerate(picks):
            t = train[int(idx)]
            records = [draw(t, args.seed * 1000000 + iteration * 1000 + slot * 100 + i, True,
                            initializer=prefix_initializer(
                                t, fraction, args.seed * 1000000 + iteration * 1000 + slot * 100 + i))
                       for i in range(args.episodes)]
            loss, metrics = constructor_loss(records, baseline=args.baseline, entropy_coef=0.)
            losses.append(loss); valid += sum(r["valid"] for r in records)
            rewards.append(metrics["reward_mean"]); entropies.append(metrics["normalized_entropy"])
        total = torch.stack(losses).mean()
        optimizer.zero_grad()
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        if iteration % 10 == 9:
            print(json.dumps({"iteration": iteration, "seed": args.seed, "train_valid": valid,
                              "train_episodes": len(picks) * args.episodes, "prefix_fraction": fraction,
                              "reward": float(np.mean(rewards)), "entropy": float(np.mean(entropies)),
                              "grad_norm": float(norm), "seconds": time.monotonic() - started}), flush=True)
        if (iteration + 1) % args.eval_every == 0 and iteration + 1 < args.iterations:
            history["iter %d" % iteration] = evaluate("iter %d" % iteration)
    history["final"] = evaluate("final")
    summary = {"summary": "gate2", "stage": args.stage, "seed": args.seed, "actor": args.actor,
               "features": args.features, "baseline": args.baseline, "objective": args.objective}
    for name in ("train", "heldout"):
        before, after = history["init"][name], history["final"][name]
        if quality:
            summary[name] = {"init_valid": before["valid"], "final_valid": after["valid"],
                             "init_residual": before["mean_residual"], "final_residual": after["mean_residual"],
                             "final_shape": after["chosen_shape"],
                             "final_paired_policy_minus_minorminer": after["paired_residual_policy_minus_minorminer"],
                             "paired_over": after["paired_over"], "instances": before["instances"]}
            continue
        diff = paired_boot([after[k] - before[k] for k in before])
        summary[name] = {"init": float(np.mean(list(before.values()))),
                         "final": float(np.mean(list(after.values()))),
                         "gain": diff[0], "gain_ci": [diff[1], diff[2]], "instances": len(before)}
    summary["seconds"] = time.monotonic() - started
    print(json.dumps(summary), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": actor.state_dict(), "actor": args.actor, "width": args.width,
                    "features": args.features, "feature_width": FEATURE_WIDTHS[args.features],
                    "baseline": args.baseline, "stage": args.stage, "seed": args.seed,
                    "support": args.support, "summary": summary}, args.out)
    print("CURRICULUM GATE 2 DONE", flush=True)
    return summary


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES + ("corpus",), default="a")
    parser.add_argument("--corpus", default="", help="stage corpus: a planted corpus directory")
    parser.add_argument("--cells", default="", help="stage corpus: comma-separated name filters")
    parser.add_argument("--init", default="", help="warm start from a lower rung's checkpoint")
    parser.add_argument("--manifest-split", action="store_true",
                        help="stage corpus: train from the manifest's train list, held-out from its "
                             "validation list, never its test list")
    parser.add_argument("--heldout-role", choices=("validation", "test"), default="validation",
                        help="test: the final table only; evaluation-only (--iterations 0) with --init, "
                             "the held-out set is the locked test list, once")
    parser.add_argument("--objective", choices=("feasibility", "quality"), default="feasibility")
    parser.add_argument("--support", choices=("registered", "wide"), default="registered",
                        help="wide: the 512-candidate construction support, every frontier placement offered")
    parser.add_argument("--reward-reads", type=int, default=256)
    parser.add_argument("--selection-reads", type=int, default=256)
    parser.add_argument("--assessment-reads", type=int, default=512)
    parser.add_argument("--select-cap", type=int, default=6)
    parser.add_argument("--deadline", type=float, default=60., help="quality evaluation: seconds an arm may propose and select")
    parser.add_argument("--comparison", choices=("none", "minorminer", "minorminer_grown"),
                        default="minorminer_grown")
    parser.add_argument("--prefix-fraction", default="",
                        help="training curriculum 'start:end': fraction of a train task's prefix "
                             "source pre-placed at the first and last iteration; evaluation is from empty")
    parser.add_argument("--grow-extra", type=int, default=1,
                        help="minorminer_grown arm: random contact-growth qubits added to the router's draw")
    parser.add_argument("--baseline-tries", type=int, default=2)
    parser.add_argument("--baseline-router-seconds", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--heldout", type=int, default=12)
    parser.add_argument("--actor", choices=("linear", "mlp", "contextual"), default="linear")
    parser.add_argument("--features", choices=("tiny", "construction"), default="tiny")
    parser.add_argument("--baseline", choices=("loo", "value"), default="loo")
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
    if min(args.train, args.heldout, args.eval_episodes, args.eval_every,
           args.instances_per_iteration, args.max_steps, args.width) < 1 or args.iterations < 0:
        parser.error("set sizes, evaluation, width and horizon must be positive; iterations nonnegative")
    if args.episodes < 2:
        parser.error("leave-one-out requires at least two episodes per instance")
    if args.baseline == "value" and args.actor != "contextual":
        parser.error("--baseline value needs the contextual actor's value head")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.episode_seconds <= 0:
        parser.error("learning rate and episode seconds must be positive and finite")
    if (args.stage == "corpus") != bool(args.corpus):
        parser.error("--stage corpus and --corpus PATH go together")
    if args.heldout_role == "test" and (args.iterations != 0 or not args.init or not args.manifest_split):
        parser.error("--heldout-role test is evaluation-only: needs --iterations 0, --init and --manifest-split")
    if args.prefix_fraction:
        try:
            lo, hi = (float(x) for x in args.prefix_fraction.split(":"))
        except ValueError:
            parser.error("--prefix-fraction must be 'start:end'")
        if not (0 <= lo <= 1 and 0 <= hi <= 1):
            parser.error("--prefix-fraction values must lie in [0, 1]")
    if min(args.reward_reads, args.selection_reads, args.assessment_reads, args.select_cap,
           args.baseline_tries) < 1 or args.deadline <= 0 or args.baseline_router_seconds <= 0 or args.grow_extra < 0:
        parser.error("reads, caps, deadline and router seconds must be positive")
    return args


def grow_chains(chains, host, extra, rng):
    """Add ``extra`` free qubits, each adjacent to an existing chain, keeping chains
    connected and disjoint: the platform's random contact growth, no learning."""
    chains = {v: set(c) for v, c in chains.items()}
    used = set().union(*chains.values()) if chains else set()
    for _ in range(extra):
        options = [(v, n) for v, c in chains.items() for q in c for n in host[q] if n not in used]
        if not options:
            break
        v, n = options[int(rng.integers(len(options)))]
        chains[v].add(n); used.add(n)
    return {v: frozenset(c) for v, c in chains.items()}


def grown_baseline_proposal(task, budget, seed, seconds_left, extra, tries=2, router_seconds=2.0):
    """minorminer's own draw, then ``extra`` random contact-growth qubits, compiled and
    validated through the same final gate as every other arm."""
    import time as _time
    from seeded_minorminer import attempt
    from _context import host_context
    from isingfold.rl.contracts import DecisionState, Opcode
    from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector
    started = _time.monotonic()
    chains = attempt(task, None, seed, tries, budget=budget, timeout=min(seconds_left, router_seconds))
    if chains is None:
        return None
    chains = grow_chains(chains, task.host, extra, np.random.default_rng(seed))
    if sum(len(c) for c in chains.values()) > budget or _time.monotonic() - started >= seconds_left:
        return None
    env = EmbeddingEnv(task, host_context(budget, quotas={}), mode=Mode.IMPROVEMENT,
                       initializer=lambda *_: chains, selector=fixed_strength_selector(), reward_reads=8)
    dec = env.reset(seed)
    if not isinstance(dec, DecisionState):
        return None
    choices = [i for i, (c, ok) in enumerate(zip(dec.candidates, dec.legal_mask)) if ok and c.opcode is Opcode.COMMIT]
    if not choices:
        return None
    terminal = env.step(dec, choices[0], evaluate_training_reward=False).next_decision_or_terminal
    return terminal if _time.monotonic() - started <= seconds_left else None


def forbidden_solver(*args, **kwargs):
    raise AssertionError("minorminer called outside the comparison arm")


def main(argv=None):
    args = parse(argv)
    # the certificate needs minorminer; everything after this line must not
    if args.stage == "corpus":
        cells = [c for c in args.cells.split(",") if c]
        train, heldout = build_corpus_sets(args.corpus, cells, args.train, args.heldout, args.seed,
                                           use_manifest_split=args.manifest_split,
                                           heldout_role=args.heldout_role)
    else:
        train, heldout = build_sets(args.stage, args.train, args.heldout, args.seed)
    with no_completion_solver():
        run(args, train, heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare read allocation on the same frozen candidate pools.

Uniform uses K*R selection reads; halving, UCB and optional halving+prior each use exactly
floor(K*R/2). Prior scores may eliminate candidates without measurements, but their preprocessing
cost is reported separately. Random and prior-only picks use no selection reads. All chosen
candidates receive independent assessment reads, shared when multiple arms select the same
candidate. Selection and assessment seed namespaces are separate. This measures selection
quality, not the quality of newly generated embeddings or a minimum-qubit objective.
"""
import argparse, hashlib, json, os, pickle, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import _derived_seed, first_commit_controller, run_controller

from _context import host_context
from _provenance import CacheProvenanceError, label_provenance

class EvaluationSeeds:
    """Unique selection seeds below 2**30; independent assessment seeds above it.

    Counters are per run, not arithmetic offsets per state that can overlap when pools grow.
    Assessment uses a state/candidate pairing, so its seed does not depend on selection calls.
    """
    def __init__(self):
        self.counter = 0

    def selection(self):
        self.counter += 1
        if self.counter >= 2**30:
            raise ValueError("selection seed namespace exhausted")
        return self.counter

    @staticmethod
    def assessment(state, candidate):
        pair = (state + candidate) * (state + candidate + 1) // 2 + candidate
        if min(state, candidate) < 0 or pair >= 2**30:
            raise ValueError("assessment seed namespace exhausted")
        return 2**30 + pair


def validate_cache(blob, corpus, ctx, allow_code_change=False):
    """An explicit code-only override must never hide a changed corpus or sampler context."""
    recorded = blob.get("provenance")
    current = label_provenance(corpus, ctx)
    if recorded == current:
        return
    if (allow_code_change and isinstance(recorded, dict)
            and recorded.keys() == current.keys()
            and recorded.get("code_sha256")
            and all(recorded[key] == value for key, value in current.items()
                    if key != "code_sha256")):
        print("  WARNING: code-only cache mismatch accepted; recorded=%s current=%s"
              % (recorded["code_sha256"], current["code_sha256"]), flush=True)
        return
    raise CacheProvenanceError("cache provenance differs or is missing; --allow-stale-cache "
                               "can override only a code hash, never corpus/context provenance")


def mean_rate(pair):
    return pair[0] / pair[1] if pair[1] else float("-inf")


def successive_halving(candidates, budget, sample):
    """Spend exactly the requested budget, including failed calls, on one candidate pool.

    ``sample(candidate, requested_reads)`` returns (hits, actual_reads). Exceptions and
    invalid outcomes are fatal in the evaluator; missing outcomes supplied by a test/adapter
    consume their requested budget and are ranked below observed candidates.
    """
    alive = list(candidates)
    if len(alive) < 2 or len(set(alive)) != len(alive):
        raise ValueError("halving needs at least two distinct candidates")
    counts, n = [], len(alive)
    while n > 1:
        counts.append(n)
        n = (n + 1) // 2
    if budget < sum(counts):
        raise ValueError("read budget too small to observe each surviving candidate")
    acc = {j: [0, 0] for j in alive}
    remaining = budget
    for stage, n in enumerate(counts):
        # Reserve at least one read per candidate for every later round.
        reserve = sum(counts[stage + 1:])
        allocation = min(remaining - reserve, max(n, remaining // (len(counts) - stage)))
        each, extra = divmod(allocation, n)
        for pos, j in enumerate(alive):
            request = each + int(pos < extra)
            h = sample(j, request)
            if h is not None:
                acc[j][0] += h[0]
                acc[j][1] += h[1]
        remaining -= allocation
        alive.sort(key=lambda j: -mean_rate(acc[j]))
        alive = alive[:(len(alive) + 1) // 2]
    return alive[0], budget - remaining


def ucb_select(size, budget, block, sample):
    """UCB with a strict requested-read budget, also charged on failed measurements."""
    if size < 1 or block < 1 or budget < size:
        raise ValueError("UCB needs a positive block and at least one read per candidate")
    acc = {j: [0, 0] for j in range(size)}
    used = 0
    initial = min(block, budget // size)
    for j in range(size):
        h = sample(j, initial)
        used += initial
        if h is not None:
            acc[j][0] += h[0]
            acc[j][1] += h[1]
    while used < budget:
        bounds = [acc[j][0] / acc[j][1] + np.sqrt(2 * np.log(max(2, used)) / acc[j][1])
                  if acc[j][1] else float("inf") for j in range(size)]
        j = int(np.argmax(bounds))
        request = min(block, budget - used)
        h = sample(j, request)
        used += request
        if h is not None:
            acc[j][0] += h[0]
            acc[j][1] += h[1]
    return max(range(size), key=lambda j: mean_rate(acc[j])), used


def prior_config(checkpoint, width=None, metadata_path=None):
    """Use the feature contract saved by train_successor, never silently base features."""
    path = Path(checkpoint)
    options = [Path(metadata_path)] if metadata_path else [path.with_suffix(".json"), path.with_suffix("")]
    sidecar = next((candidate for candidate in options if candidate.is_file()), None)
    if sidecar is None:
        raise ValueError("prior needs its train_successor JSON metadata; pass --prior-metadata")
    meta = json.loads(sidecar.read_text())
    if (meta.get("checkpoint_sha256") is not None
            and hashlib.sha256(path.read_bytes()).hexdigest() != meta["checkpoint_sha256"]):
        raise ValueError("prior checkpoint does not match its metadata fingerprint")
    if meta.get("model") != "SuccessorScorer":
        raise ValueError("prior metadata must describe a SuccessorScorer")
    if any(flag not in meta for flag in ("coords", "physics", "space", "width")):
        raise ValueError("prior metadata lacks its feature contract")
    if width is not None and width != int(meta["width"]):
        raise ValueError("--width differs from the saved prior width")
    if meta.get("legacy_compile"):
        raise ValueError("legacy-compiled prior cannot score the registered physical program")
    if any(type(meta[flag]) is not bool for flag in ("coords", "physics", "space")):
        raise ValueError("prior feature flags must be booleans")
    return meta


def verify_prior_split(meta, states, exploratory=False):
    trained = meta.get("training_lineages")
    if trained is None:
        if not exploratory:
            raise ValueError("prior lacks training lineage provenance; retrain/save metadata or "
                             "use --allow-unverified-prior for exploratory results only")
        print("  WARNING: prior training split is unverified; EXPLORATORY ONLY", flush=True)
        return
    used = set(trained) | set(meta.get("selection_lineages", []))
    if used & {st["lineage"] for st in states}:
        raise ValueError("prior evaluation overlaps training/checkpoint-selection lineages")


def check_prior_rows(states, built):
    if len(states) != len(built):
        raise ValueError("prior compilation dropped states; all arms must use identical pools")
    for original, compiled in zip(states, built):
        if (original["state_id"] != compiled["state_id"]
                or len(original["rows"]) != len(compiled["rows"])
                or any(r["succ"] != b["succ"]
                       for r, b in zip(original["rows"], compiled["rows"]))):
            raise ValueError("prior compilation changed candidate membership or ordering")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--lineages", type=int, default=32)
    ap.add_argument("--reads", type=int, default=256, help="R, the uniform per-candidate reads")
    ap.add_argument("--block", type=int, default=32, help="UCB read increment; the final block may be smaller")
    ap.add_argument("--assess-reads", type=int, default=512)
    ap.add_argument("--qubit-cap", type=int, default=248)
    ap.add_argument("--split", default="eval", choices=("eval", "train"))
    ap.add_argument("--prior", default="",
                    help="a trained successor scorer (train_successor.py checkpoint); adds the "
                         "arms prior (its argmax, no reads) and halving+prior (its bottom half "
                         "dropped unread, halving on the rest at the same total reads)")
    ap.add_argument("--width", type=int, default=None, help="optional assertion of prior width")
    ap.add_argument("--prior-metadata", default="")
    ap.add_argument("--allow-unverified-prior", action="store_true",
                    help="legacy checkpoint without training lineage metadata; exploratory only")
    ap.add_argument("--allow-stale-cache", action="store_true",
                    help="accept a cache whose recorded code hash differs; only when the change "
                         "is known not to touch labelling, and it is printed")
    a = ap.parse_args()
    with open(a.cache, "rb") as fh:
        blob = pickle.load(fh)
    ctx = host_context(a.qubit_cap)
    if min(a.lineages, a.reads, a.block, a.assess_reads) < 1:
        ap.error("lineages, reads, block and assess-reads must be positive")
    validate_cache(blob, blob["key"][0], ctx, a.allow_stale_cache)
    states = blob[a.split]
    roots = sorted({st["lineage"] for st in states})[: a.lineages]
    states = [st for st in states if st["lineage"] in set(roots)]
    print(json.dumps({"cache": a.cache, "split": a.split, "lineages": len(roots),
                      "states": len(states), "reads": a.reads, "block": a.block,
                      "beta_range": list(ctx.beta_range), "prior": a.prior or None}), flush=True)
    prior_scores = {}
    prior_started = time.monotonic()
    if a.prior:
        import torch
        from successor_scorer import COORD_WIDTH, PHYS_WIDTH, SuccessorScorer
        from space_features import SPACE_WIDTH
        from train_successor import build
        device = torch.device("cpu")
        meta = prior_config(a.prior, a.width, a.prior_metadata or None)
        verify_prior_split(meta, states, a.allow_unverified_prior)
        print("  prior contract: " + json.dumps({key: meta.get(key) for key in (
            "width", "coords", "physics", "space", "legacy_compile", "label_provenance",
            "allow_stale_cache", "checkpoint_sha256")}, sort_keys=True), flush=True)
        node_dim = 4 + (COORD_WIDTH + 1 if meta["coords"] else 0)
        chain_dim = 4 + (PHYS_WIDTH if meta["physics"] else 0) + (SPACE_WIDTH if meta["space"] else 0)
        model = SuccessorScorer(width=int(meta["width"]), node_dim=node_dim, chain_dim=chain_dim)
        model.load_state_dict(torch.load(a.prior, map_location="cpu", weights_only=True))
        model.eval()
        built = build(states, ctx, device, "prior states", meta["coords"], meta["physics"], meta["space"])
        check_prior_rows(states, built)
        with torch.no_grad():
            for st in built:
                prior_scores[st["state_id"]] = [float(model(r["graph"])) for r in st["rows"]]

    if a.prior:
        print("  prior preprocessing/inference seconds %.6f" % (time.monotonic() - prior_started), flush=True)
    seeds = EvaluationSeeds()
    calls = {"n": 0}
    evaluator_seeds = set()

    def hits(task, chains, seed, reads):
        """Hits and reads from one evaluator block, a real sampler call."""
        calls["n"] += 1
        def fixed(l, h, s):
            return {v: frozenset(c) for v, c in chains.items()}
        # run_controller hashes its base seed into a 31-bit sampler seed. Avoid rare
        # birthday collisions *before* spending reads, rather than aborting a large run.
        while _derived_seed(seed, "final-evaluator", task, 0) in evaluator_seeds:
            seed = seeds.selection() if seed < 2**30 else seed + 1
        out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                             selector=fixed_strength_selector(), reward_reads=reads,
                             repetitions=1, seed=seed)
        o = out[0]
        if not o.returned_valid or o.utility is None:
            raise RuntimeError("candidate measurement failed; refusing to silently drop a state")
        if o.evaluator_hits is None or o.evaluator_reads != reads:
            raise RuntimeError("evaluator must report exact hit counts and requested reads")
        if o.evaluator_seed in evaluator_seeds or o.evaluator_seed is None:
            raise RuntimeError("evaluator seed reused/missing; measurements must be independent")
        evaluator_seeds.add(o.evaluator_seed)
        h = int(o.evaluator_hits)
        if not 0 <= h <= reads:
            raise RuntimeError("invalid evaluator hit count")
        return h, reads

    def assess(task, chains, k, j):
        h, n = hits(task, chains, seeds.assessment(k, j), a.assess_reads)
        return h / n

    per_state = []
    started = time.time()
    for k, st in enumerate(states):
        rows = st["rows"]
        K = len(rows)
        if K < 4:
            continue
        task = st["task"]
        spent = {}
        arm_seconds = {}
        def sample(j, reads):
            return hits(task, rows[j]["succ"], seeds.selection(), reads)
        arm_started = time.monotonic()
        est = [mean_rate(sample(j, a.reads)) for j in range(K)]
        pick_uniform = int(np.argmax(est)); spent["uniform"] = K * a.reads
        arm_seconds["uniform"] = time.monotonic() - arm_started
        budget = K * a.reads // 2
        arm_started = time.monotonic()
        pick_halving, spent["halving"] = successive_halving(range(K), budget, sample)
        arm_seconds["halving"] = time.monotonic() - arm_started
        arm_started = time.monotonic()
        pick_ucb, spent["ucb"] = ucb_select(K, budget, a.block, sample)
        arm_seconds["ucb"] = time.monotonic() - arm_started
        rng = np.random.default_rng(k)
        picks = {"uniform": pick_uniform, "halving": pick_halving, "ucb": pick_ucb,
                 "random": int(rng.integers(0, K))}
        if a.prior:
            pr = np.asarray(prior_scores[st["state_id"]])
            if pr.shape != (K,) or not np.isfinite(pr).all():
                raise ValueError("prior must provide one finite score per candidate")
            picks["prior"] = int(np.argmax(pr)); spent["prior"] = 0
            # plain ints: the assessment seed is derived through JSON, which numpy ints break
            alive = [int(i) for i in np.argsort(-pr, kind="stable")[:max(2, (K + 1) // 2)]]
            arm_started = time.monotonic()
            pick_hp, spent_hp = successive_halving(alive, budget, sample)
            picks["halving+prior"], spent["halving+prior"] = int(pick_hp), spent_hp
            arm_seconds["halving+prior"] = time.monotonic() - arm_started
        # one assessment block per distinct chosen candidate, shared across arms that agree
        assessed = {}
        for name, j in picks.items():
            if j not in assessed:
                assessed[j] = assess(task, rows[j]["succ"], k, j)
        per_state.append({"lineage": st["lineage"], "spent": spent,
                          "arm_seconds": arm_seconds,
                          "assessment_reads_shared": len(assessed) * a.assess_reads,
                          **{name: assessed[j] for name, j in picks.items()}})
        if (k + 1) % 10 == 0:
            print("  %d/%d states, %d evaluator calls, %.0fs"
                  % (k + 1, len(states), calls["n"], time.time() - started), flush=True)

    if not per_state:
        raise ValueError("no eligible states with at least four candidates")

    def boot(fn):
        by = defaultdict(list)
        for s in per_state:
            by[s["lineage"]].append(fn(s))
        means = np.array([np.mean(v) for v in by.values()])
        rng = np.random.default_rng(0)
        bs = [means[rng.integers(0, len(means), len(means))].mean() for _ in range(4000)]
        return means.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print("\n  %d states in %d lineages, assessed on %d independent reads"
          % (len(per_state), len({s['lineage'] for s in per_state}), a.assess_reads))
    print("  quality and differences are equally weighted over lineages")
    print("  %-14s %10s %10s  %s" % ("arm", "reads", "quality", "differences, 95% over lineages"))
    arms = ["uniform", "halving", "ucb", "random"] + (["prior", "halving+prior"] if a.prior else [])
    for name in arms:
        rows_ok = [s for s in per_state if name in s]
        if not rows_ok:
            continue
        reads = np.mean([s["spent"].get(name, 0) for s in rows_ok])
        q = boot(lambda s: s[name])[0]
        d, lo, hi = boot(lambda s: s[name] - s["uniform"]) if len(rows_ok) == len(per_state) else (float("nan"),) * 3
        dh, hlo, hhi = boot(lambda s: s[name] - s["halving"])
        print("  %-14s %10.0f %10.4f  vs uniform %+.4f [%+.4f, %+.4f]  vs halving %+.4f [%+.4f, %+.4f]"
              % (name, reads, q, d, lo, hi, dh, hlo, hhi))
    print("  selection wall seconds per state (prior preprocessing excluded): " + json.dumps({
        name: float(np.mean([s["arm_seconds"].get(name, 0.0) for s in per_state])) for name in arms}), flush=True)
    print("  assessment reads: %d per arm/state; %d actually used after sharing identical picks"
          % (a.assess_reads, sum(s["assessment_reads_shared"] for s in per_state)), flush=True)
    print("\nADAPTIVE ALLOCATION DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The objective-guided embedder: minorminer construction, then refinement inside the modular
minorminer search with a learned proposal order and the downstream surrogate in the loop.

    embedder = ObjectiveEmbedder(problem, budget=100, scorers=[...], policy="ens_ucb")
    chains = embedder(logical, host, seed)

`problem` carries the fields and couplings the objective is evaluated on (the embedder is
problem-aware by design: two embeddings of the same graph differ in solve probability only
through h and J). Policies: "ens_ucb" (ensemble mean + kappa*spread - lambda*overlap,
tabu on rejected chains), "learned" (single scorer order, evaluated), "random" (random
order, evaluated: the strong classical baseline), "none" (minorminer only).

Requires the Track B harness (`lac_minorminer`) for the proposal seam; it is imported lazily
so the rest of the benchmark stays importable without it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx
import numpy as np

from embedbench.embedding import Embedding, LogicalProblem
from embedbench.objective import outcome_of
from embedbench.surrogate import default_strength_grid, solve_probability_at


def embedding_features(logical, problem, chains, host):
    """Twelve per-embedding features for the selection shortlist (L-135): qubits, longest,
    mean and sd of chain length, single-qubit fraction, |J|-weighted length, mean contacts
    per coupling, single-contact fraction, |h|-weighted length, length of the strongest
    couplings, tree surplus per chain, free-neighbour capacity per qubit. Identical to
    the historical shortlist experiment that trained the packaged ranker."""
    lens = {v: len(c) for v, c in chains.items()}; L = list(lens.values())
    js = {e: abs(w) for e, w in problem.j.items()}; jmax = max(js.values()) if js else 1.0
    contacts = {e: sum(1 for a in chains[e[0]] for b in chains[e[1]] if host.has_edge(a, b)) for e in js}
    jw = sum(js[e] * (lens[e[0]] + lens[e[1]]) for e in js) / max(1e-9, sum(js.values()))
    single = sum(1 for e in js if contacts[e] == 1) / max(1, len(js))
    hl = sum(abs(problem.h.get(v, 0.0)) * lens[v] for v in lens) / max(1e-9, sum(abs(problem.h.get(v, 0.0)) for v in lens))
    surplus = sum(max(0, host.subgraph(c).number_of_edges() - (len(c) - 1)) for c in chains.values())
    strong = [lens[e[0]] + lens[e[1]] for e, w in js.items() if w >= 0.7 * jmax]
    used = {q for c in chains.values() for q in c}
    cap = len({n for q in used for n in host.neighbors(q) if n not in used})
    return [sum(L), max(L), float(np.mean(L)), float(np.std(L)), float(np.mean([l == 1 for l in L])), jw,
            float(np.mean(list(contacts.values()))) if contacts else 0.0, single, hl, float(np.mean(strong)) if strong else 0.0,
            surplus / max(1, len(L)), cap / max(1, sum(L))]


def move_features(logical, problem, cur, new, host, changed):
    """Features of a repaired state relative to the current one, computed before any
    evaluation: what a post-repair screen sees. changed = variables whose chain differs.
    Thirteen base features (screen v1) plus three mechanism features (L-132): change in
    chain cut connectivity (edge surplus over a spanning tree of each chain, summed), change
    in residual capacity (free qubits adjacent to the changed chains), and the number of
    changed chains that are single-qubit."""
    def stats(ch):
        lens = {v: len(c) for v, c in ch.items()}
        js = {e: abs(w) for e, w in problem.j.items()}
        contacts = {e: sum(1 for a in ch[e[0]] for b in ch[e[1]] if host.has_edge(a, b)) for e in js}
        jw = sum(js[e] * (lens[e[0]] + lens[e[1]]) for e in js) / max(1e-9, sum(js.values()))
        single = sum(1 for e in js if contacts[e] == 1) / max(1, len(js))
        hl = sum(abs(problem.h.get(v, 0.0)) * lens[v] for v in lens) / max(1e-9, sum(abs(problem.h.get(v, 0.0)) for v in lens))
        return sum(lens.values()), max(lens.values()), jw, sum(contacts.values()) / max(1, len(js)), single, hl
    q0, l0, jw0, c0, s0, h0 = stats(cur); q1, l1, jw1, c1, s1, h1 = stats(new)
    ch_lens = sorted(len(new[v]) for v in changed); old_lens = sorted(len(cur[v]) for v in changed)
    jsum = sum(abs(problem.j.get((min(v, u), max(v, u)), 0.0)) for v in changed for u in logical.neighbors(v))
    def surplus(ch, vs):  # chain edges beyond a spanning tree: cycles inside a chain (cut robustness)
        return sum(max(0, host.subgraph(ch[v]).number_of_edges() - (len(ch[v]) - 1)) for v in vs)
    def capacity(ch, vs):
        used = {q for c in ch.values() for q in c}
        return len({n for v in vs for q in ch[v] for n in host.neighbors(q) if n not in used})
    d_cut = surplus(new, changed) - surplus(cur, changed)
    d_cap = capacity(new, changed) - capacity(cur, changed)
    n_single = sum(1 for v in changed if len(new[v]) == 1)
    return [q1 - q0, l1 - l0, jw1 - jw0, c1 - c0, s1 - s0, h1 - h0, len(changed), sum(ch_lens) - sum(old_lens), max(ch_lens, default=0), jsum, q0, l0, c0,
            d_cut, d_cap, n_single]



def _pick_focus(rng, chains, pool, rule: str) -> int:
    """Which variable a destroy-and-repair move resets, chosen among `pool`.

    'random' is the released rule. 'longest' takes the variable whose chain occupies the most
    qubits, ties broken at random; 'weighted' samples proportionally to chain length.
    """

    if rule == "random" or not pool:
        return rng.choice(pool)
    sizes = [len(chains[i]) for i in pool]
    if rule == "longest":
        top = max(sizes)
        return rng.choice([i for i, sz in zip(pool, sizes) if sz == top])
    if rule == "weighted":
        pick = rng.random() * (float(sum(sizes)) or 1.0)
        run = 0.0
        for i, sz in zip(pool, sizes):
            run += sz
            if pick <= run:
                return i
        return pool[-1]
    raise ValueError(f"unknown destroy focus rule {rule}")


@dataclass
class ObjectiveEmbedder:
    problem: LogicalProblem
    e0: float
    budget: int = 400
    """Surrogate evaluations, ten of them the selection probe. With ``reads`` at 50 the
    refinement spends 20,000 reads, the same total as the previous 100 evaluations of 200
    reads, and gains +0.030 [+0.009, +0.052] on untouched seeds (L-151, confirmed in L-152)."""
    scorers: list | None = None
    policy: str = "ens_ucb"
    kappa: float = 1.0
    lam: float = 0.5
    reads: int = 50
    n_strengths: int = 4
    rounds_per_eval: int = 10
    max_candidates: int = 8
    verify_reads: int = 400
    objective: str = "auto"
    """What the refinement reads. 'psolve' is the released objective. 'residual' is the mean
    relative residual energy of the decoded reads, which still separates embeddings where
    p_solve is a flat zero. 'auto' probes the starting embedding once and takes psolve when it
    is above `residual_switch`, residual otherwise."""
    residual_switch: float = 0.02
    _auto_objective: str | None = None
    move: str = "perturb3"
    destroy_focus: str = "random"
    """Which variable a destroy-and-repair move resets: 'random' (released), 'longest' (the
    chain occupying the most qubits) or 'weighted' (proportional to chain length)."""
    select_k: int = 10
    q_cap: float | None = None
    """Cap on total qubits as a multiple of the stock embedding's (1.10 = at most ten percent
    more); repaired or replaced states over the cap are never evaluated. None = uncapped."""
    """Start from the best of `select_k` stock minorminer embeddings (different seeds) by a
    surrogate probe at `select_reads`; the probe evaluations count against `budget`. At
    100 to 200 variables selection is the whole gain (L-105); on small hosts refinement is."""
    select_reads: int = 100
    select_pool: int = 0
    """Generate this many stock embeddings (0 or <= select_k: exactly select_k, all probed) and
    let `selector` choose the select_k to probe (L-135: a learned shortlist recovers part of
    the gap to probing the whole pool at no evaluation and no qubit cost)."""
    selector: object | None = "default"
    """A fitted regressor on the 12 embedding features (train_shortlist.py) ranking the pool;
    "default" loads the packaged shortlist_v1; None takes the first select_k (the classical
    selection). Only used when select_pool > select_k."""
    screen_model: object | None = "default"
    """A fitted classifier with predict_proba on the 13 move features (train_screen.py); with
    screen_topk > 1, every destroy-and-repair step generates screen_topk repaired states and
    evaluates the one the screen ranks first (L-123, confirmed L-128). "default" loads the
    packaged screen_v1; None disables the learned decision (the classical embedder)."""
    screen_topk: int = 20
    """'chain': replace one chain per step from the search's proposals (ordered by `policy`).
    'perturbK': destroy-and-repair, reset a random variable and its K-1 most strongly coupled
    neighbours through the search's repair seam, repair to validity, evaluate (L-103: K = 3
    nearly doubles the gain on application graphs at 60 to 80 percent more qubits).
    'perturbmix': the neighbourhood size is drawn from {2, 3, 4} for every repaired state.
    'perturbauto': the size is 31 percent of the logical variables, between 3 and 8, so the
    move keeps its share of the problem as the problem grows."""
    """After the search, the start and the refined embedding are rescored once with fresh
    seeds at this read count and the better one is returned (ties go to the start, which
    uses fewer qubits). 0 disables the check. Costs one extra evaluation at the higher read
    count and removes refinements that were accepted on evaluation noise."""

    def _selector(self):
        if self.selector == "default":
            import joblib
            from importlib import resources
            with resources.as_file(resources.files("embedbench") / "models" / "shortlist_v1.joblib") as fp:
                self.selector = joblib.load(fp)
        return self.selector

    def _screen(self):
        if self.screen_model == "default":
            import joblib
            from importlib import resources
            with resources.as_file(resources.files("embedbench") / "models" / "screen_v1.joblib") as fp:
                self.screen_model = joblib.load(fp)
        return self.screen_model

    def surrogate(self, host, chains, seed):
        emb = Embedding.from_chains(chains, host, self.problem)
        results = [solve_probability_at(emb, self.problem, self.e0, f, num_reads=self.reads,
                                        num_sweeps=200, seed=(seed + 31 * k) % (2**31))
                   for k, f in enumerate(default_strength_grid(self.problem, self.n_strengths))]
        if self._objective() == "residual":
            # Where the host is roomy relative to the problem, or the problem is large, every
            # candidate embedding solves zero reads out of fifty and p_solve is a flat zero:
            # the search is then optimising a constant and its moves are random. The mean
            # relative residual energy of the decoded reads still separates them, so that is
            # what the search reads instead. Which one is in force is decided once, on the
            # starting embedding, and never changes inside a run.
            return max(-r.mean_residual for r in results)
        return max(r.p_solve for r in results)

    def _objective(self) -> str:
        if self.objective in ("psolve", "residual"):
            return self.objective
        return self._auto_objective or "psolve"

    def __call__(self, logical: nx.Graph, host: nx.Graph, seed: int) -> dict:
        import minorminer
        isolated = [v for v in logical.nodes() if logical.degree(v) == 0]
        emb = minorminer.find_embedding(list(logical.edges()), list(host.edges()), random_seed=seed % (2**31), tries=10)
        if not emb or set(emb) != set(logical.nodes()) - set(isolated):
            return {}
        cur = {v: frozenset(c) for v, c in emb.items()}
        used = {q for c in cur.values() for q in c}; fr = iter(sorted(set(host.nodes) - used))
        for v in isolated: cur[v] = frozenset([next(fr)])
        spent = 0
        self._auto_objective = None
        if self.objective == "auto":
            # One probe of the constructed embedding decides what the run reads. It is charged
            # to the budget like any other evaluation.
            self._auto_objective = "psolve"
            probe = self.surrogate(host, cur, seed + 991)
            self._auto_objective = "psolve" if probe > self.residual_switch else "residual"
            spent += 1
        if self.select_k > 1:
            pool = [cur]
            n_pool = max(self.select_k, self.select_pool)
            for k in range(1, n_pool):
                e_k = minorminer.find_embedding(list(logical.edges()), list(host.edges()), random_seed=(seed + 17 * k) % (2**31), tries=10)
                if e_k and set(e_k) == set(logical.nodes()) - set(isolated):
                    c_k = {v: frozenset(c) for v, c in e_k.items()}
                    used = {q for c in c_k.values() for q in c}; fr = iter(sorted(set(host.nodes) - used))
                    for v in isolated: c_k[v] = frozenset([next(fr)])
                    pool.append(c_k)
            if len(pool) > self.select_k:
                sel = self._selector() if self.select_pool > self.select_k else None
                if sel is not None:
                    sc = sel.predict(np.array([embedding_features(logical, self.problem, c, host) for c in pool]))
                    pool = [pool[j] for j in np.argsort(-sc)[: self.select_k]]
                else:
                    pool = pool[: self.select_k]
            r0 = self.reads; self.reads = self.select_reads
            try:
                scores = [self.surrogate(host, c, seed + 3 + j) for j, c in enumerate(pool)]
            finally:
                self.reads = r0
            cur = pool[int(np.argmax(scores))]; spent = len(pool)
        start = dict(cur)
        q_limit = (sum(len(c) for c in cur.values()) * self.q_cap) if self.q_cap is not None else None
        def over_cap(ch):
            return q_limit is not None and sum(len(c) for c in ch.values()) > q_limit
        if self.policy == "none" or self.budget <= spent:
            return cur
        from lac_minorminer.orchestrator import SearchOrchestrator
        from lac_minorminer.scoring import ResourceScorer
        from lac_minorminer.session import SearchSession
        probe = SearchSession(list(logical.edges()), list(host.edges()), random_seed=seed, max_candidates=self.max_candidates)
        src, tgt = probe.normalized_source, probe.normalized_target
        sess = SearchSession.from_chains(list(logical.edges()), list(host.edges()), [[tgt.index(q) for q in cur[l]] for l in src.labels],
                                         random_seed=seed, max_candidates=self.max_candidates)
        orch = SearchOrchestrator(scorer=ResourceScorer())
        rng = random.Random(seed); p_cur = self.surrogate(host, cur, seed + 7); evals = spent; tabu = set(); tried = {}
        seam = None
        if self.policy in ("ens_ucb", "learned") and self.scorers:
            from embedbench.seam import LearnedTieBreakScorer
            seam = [LearnedTieBreakScorer(self.problem, logical, host, sess, sc) for sc in self.scorers]
        mixed = self.move == "perturbmix"
        if self.move == "perturbauto":
            # Five variables out of sixteen is what the development sweep settled on, which is
            # just under a third of the problem. A fixed five touches a third of a 16-variable
            # embedding and a twentieth of a 100-variable one, so the move shrinks relative to
            # what it has to repair exactly as the problem gets harder. The fraction is held
            # instead, with a floor that keeps the move legal and a ceiling that keeps one
            # repair affordable.
            size = max(3, min(8, round(0.31 * logical.number_of_nodes())))
        elif mixed:
            size = 3
        elif self.move.startswith("perturb"):
            size = int(self.move[7:] or 2)
        else:
            size = 0
        def _screen_score(model, feats):
            n_in = getattr(model, "n_features_in_", len(feats)); arr = np.array([feats[:n_in]])
            return float(model.predict_proba(arr)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(arr)[0])
        for r in range(self.budget * self.rounds_per_eval):
            if evals >= self.budget: break
            branch = sess.fork(random_seed=seed + r)
            if size:
                k_states = max(1, self.screen_topk if self._screen() is not None else 1)
                best = None
                for kk in range(k_states):
                    b2 = branch if kk == 0 else sess.fork(random_seed=seed + 100003 * r + kk)
                    size_k = rng.choice((2, 3, 4)) if mixed else size
                    try:
                        before = b2.snapshot(); allv = list(range(len(before.chains)))
                        li = _pick_focus(rng, before.chains, allv, self.destroy_focus)
                        focus = src.labels[li]
                        nbrs = sorted(logical.neighbors(focus), key=lambda u: -abs(self.problem.j.get((min(focus, u), max(focus, u)), 0.0)))
                        b2.perturb([li] + [src.index(u) for u in nbrs[: size_k - 1]])
                        for _ in range(200):
                            if b2.snapshot().valid: break
                            orch.transition(b2, rng)
                    except ValueError:
                        continue
                    if not b2.snapshot().valid: continue
                    new2 = {k: frozenset(v) for k, v in b2.embedding().items()}
                    if new2 == cur or over_cap(new2) or outcome_of(new2, logical, host)[0] != 1: continue
                    if k_states == 1:
                        best = (0.0, b2, new2); break
                    changed = [v for v in new2 if new2[v] != cur[v]]
                    prob = _screen_score(self._screen(), move_features(logical, self.problem, cur, new2, host, changed))
                    if best is None or prob > best[0]: best = (prob, b2, new2)
                if best is None: continue
                _, branch, new = best
                p = self.surrogate(host, new, seed + 7); evals += 1
                if p >= p_cur:
                    sess, cur, p_cur = branch, new, p
                continue
            try:
                before = branch.snapshot(); eligible = branch.eligible_variables() or list(range(len(before.chains)))
                li = orch.selector.select(before, eligible, rng); batch = branch.propose(li)
            except ValueError:
                break
            focus = src.labels[li]
            cands = [(i, frozenset(tgt.labels[t] for t in c.chain)) for i, c in enumerate(batch.candidates) if c.rank.max_occupancy <= 1]
            cands = [(i, ch) for i, ch in cands if ch != cur[focus] and (focus, ch) not in tabu]
            if not cands:
                branch.discard(batch); continue
            if self.policy == "random" or seam is None:
                j = rng.randrange(len(cands))
            else:
                idxs = [i for i, _ in cands]
                per = np.array([[c - cs for c in s_.score_with_current(before, batch, idxs, cur[focus])[0]] for s_ in seam for cs in [s_.score_with_current(before, batch, idxs, cur[focus])[1]]])
                mean, spread = per.mean(0), per.std(0)
                if self.policy == "learned":
                    j = int(np.argmax(mean))
                else:
                    tried_chains = [ch for (v_, ch) in tabu if v_ == focus] + tried.get(focus, [])
                    overlap = np.array([max((len(ch & t) / max(1, len(ch | t)) for t in tried_chains), default=0.0) for _, ch in cands])
                    j = int(np.argmax(mean + self.kappa * spread - self.lam * overlap))
            i, new_chain = cands[j]
            cand = dict(cur); cand[focus] = new_chain
            if over_cap(cand) or outcome_of(cand, logical, host)[0] != 1:
                branch.discard(batch); continue
            p = self.surrogate(host, cand, seed + 7); evals += 1
            tried.setdefault(focus, []).append(new_chain)
            if p >= p_cur:
                branch.apply(batch, i); new = {k: frozenset(v) for k, v in branch.embedding().items()}
                if outcome_of(new, logical, host)[0] == 1:
                    sess, cur, p_cur = branch, new, p; continue
            tabu.add((focus, new_chain)); branch.discard(batch)
        if self.verify_reads and cur != start:
            r0, r1 = self.reads, self.verify_reads
            self.reads = r1
            try:
                p_start, p_end = self.surrogate(host, start, seed + 991), self.surrogate(host, cur, seed + 991)
            finally:
                self.reads = r0
            if p_end <= p_start:
                return start
        return cur

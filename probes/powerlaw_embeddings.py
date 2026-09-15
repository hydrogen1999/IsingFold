"""Embeddings whose chain lengths follow a power law, and the resource sweep that falls out.

Two things the project needs and does not have.

The advisor asked for chain lengths drawn from P(L = l) proportional to l^-alpha between L_min and
L_max, so that a corpus holds many short chains and still enough long ones for weak connectivity
and chain breaks to be learnable. Every generator here instead produces whatever lengths the
placement happened to give: about 2 on the ink-drop corpus and 4.36 on the clique corpus. Neither
is a controlled distribution, so nothing has ever been shown a deliberate mix.

The second thing is a consequence. Starting from one valid embedding and growing its chains to
drawn targets produces several embeddings of the same logical problem at different qubit counts,
all still valid, all realising the same couplings. That is the comparison the budget question
needs and could not have: every earlier attempt compared candidates five qubits apart, a local
perturbation, and called it a resource sweep.

Growth only ever takes free qubits and only ever attaches to the chain it is growing, so the
result stays an embedding: chains stay connected, stay disjoint, and every logical edge that had a
contact still has it. Extra qubits can only add contacts, never remove them.
"""
from __future__ import annotations

import numpy as np


def sample_lengths(n, alpha, lmin, lmax, rng):
    """Draw n chain lengths from a truncated power law over the integers [lmin, lmax].

    Truncated, because an untruncated law on a finite processor draws lengths the hardware cannot
    hold and the corpus silently becomes whatever the clipping did. The exponent, the bounds and
    the truncation are all data-design decisions; they are arguments here rather than constants so
    that an ablation over them is possible.
    """
    ls = np.arange(int(lmin), int(lmax) + 1, dtype=float)
    w = ls ** (-float(alpha))
    w = w / w.sum()
    return rng.choice(ls.astype(int), size=int(n), p=w)


def grow_chain(host, chains, node, target, rng, occupied=None, mode="length",
               neighbours=()):
    """Grow one chain into free qubits until it reaches `target`, or until it cannot.

    Returns the grown chain and how many qubits it fell short by, because a chain that could not
    reach its drawn length is a fact about the hardware at that point and should be counted, not
    silently accepted as if the draw had been smaller.
    """
    if occupied is None:
        occupied = {q for c in chains.values() for q in c}
    chain = set(chains[node])
    while len(chain) < target:
        border = []
        for q in chain:
            for r in host.neighbors(q):
                if r not in occupied and r not in chain:
                    border.append(r)
        if not border:
            break
        # Two ways to spend a qubit, and they are not the same spend.
        #
        # "length" prefers a neighbour with room around it, so the chain reaches further without
        # dead-ending. Every qubit it adds extends a path, so every internal edge stays a bridge
        # and the cheapest cut through the chain does not get any dearer: a longer chain of this
        # kind is a chain that breaks more easily.
        #
        # "redundant" prefers a neighbour already adjacent to several chain members, which closes
        # a cycle instead of extending a path. That lowers the fraction of internal edges that are
        # bridges and raises the margin of the cheapest cut, which is the quantity the chain
        # robustness feature computes. It is what "spending qubits strategically" has to mean if
        # it means anything measurable.
        #
        # "contact" grows toward the chains of logically coupled variables. The compiler divides
        # a logical coupling over the contacts that realise it, so a second contact halves what
        # each one carries. Since the autoscale shrinks every coefficient to fit the coupler
        # limit, and the chain couplers are what press against that limit, spreading a coupling
        # over more contacts relieves the pressure instead of adding to it. That is the only one
        # of these three spends whose arithmetic points the right way.
        scores = []
        for r in set(border):
            free_around = sum(1 for t in host.neighbors(r) if t not in occupied and t not in chain)
            attached = sum(1 for t in host.neighbors(r) if t in chain)
            if mode == "redundant":
                score = attached * 10 + free_around
            elif mode == "contact":
                touching = sum(1 for t in host.neighbors(r) for c in neighbours if t in c)
                score = touching * 10 + free_around
            else:
                score = free_around
            scores.append((score, r))
        best = max(s for s, _ in scores)
        pick = [r for s, r in scores if s == best]
        chosen = pick[int(rng.integers(0, len(pick)))]
        chain.add(chosen)
        occupied.add(chosen)
    return frozenset(chain), max(0, int(target) - len(chain))


def grown_embedding(host, chains, alpha, lmin, lmax, rng, mode="length", logical=None):
    """One embedding of the same logical problem, with chain lengths drawn from the law.

    Targets below a chain's current length are left alone: shrinking would have to decide which
    qubits to drop and could break a contact the problem needs, so this only ever grows. A corpus
    that needs short chains should draw them at placement time, not remove them afterwards.
    """
    order = sorted(chains, key=lambda v: len(chains[v]))
    targets = sample_lengths(len(order), alpha, lmin, lmax, rng)
    occupied = {q for c in chains.values() for q in c}
    out, shortfall = {}, 0
    for node, target in zip(order, targets):
        nb = ()
        if mode == "contact" and logical is not None and node in logical:
            current = {**chains, **out}
            nb = tuple(current[u] for u in logical.neighbors(node) if u in current)
        grown, missing = grow_chain(host, {**chains, **out}, node, target, rng, occupied,
                                    mode=mode, neighbours=nb)
        out[node] = grown
        shortfall += missing
    for node in chains:
        out.setdefault(node, frozenset(chains[node]))
    return out, shortfall


def realises(chains, logical, host):
    """Every logical edge still has a physical contact, and chains are disjoint and connected."""
    owner = {}
    for v, c in chains.items():
        if not c or not host.subgraph(c).number_of_nodes() or not _connected(host, c):
            return False
        for q in c:
            if q in owner:
                return False
            owner[q] = v
    for u, v in logical.edges():
        if not any(owner.get(r) == v for q in chains[u] for r in host.neighbors(q)):
            return False
    return True


def _connected(host, chain):
    seen = {next(iter(chain))}
    stack = list(seen)
    while stack:
        q = stack.pop()
        for r in host.neighbors(q):
            if r in chain and r not in seen:
                seen.add(r)
                stack.append(r)
    return len(seen) == len(chain)

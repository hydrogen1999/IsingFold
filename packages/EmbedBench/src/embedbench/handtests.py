"""Failure-mechanism motifs, section 6 of the 2026-09-08 meeting note.

Each motif is a small (host, logical graph, partial state, candidate actions) built to isolate
one global decision mechanism. The intended winner is recorded as a
*hypothesis*; `certify_motif` computes V* of every action by exhaustive completion and
returns the certified ranking and margin. A motif whose exact ranking disagrees with its
hypothesis is reported, not silently kept: that is the point of section 6's warning that the
generator's own path is not a label.

Motif families implemented (the meeting's list):
    dead_end            the nearest candidate enters a pocket with no way onward;
    articulation        one candidate consumes the only bridge another chain needs;
    cut_capacity        the immediate expansion takes too much of a narrow cut;
    multi_neighbour     either neighbour alone favours a choice that loses jointly;
    high_degree_trap    a hub looks attractive but is the scarce resource another chain needs;
    short_chain_trap    equal-Q choices differ only in final maximum chain length;
    ordering            which variable to place first: one order blocks, the other completes.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from embedbench.embedding import Node, Qubit
from embedbench.exact import DecisionSample, sample_at


@dataclass(frozen=True)
class Motif:
    name: str
    mechanism: str
    host: nx.Graph
    logical: nx.Graph
    cores: dict[Node, frozenset[Qubit]]
    actions: tuple[tuple[Node, Qubit], ...]
    hypothesis_winner: tuple[Node, Qubit]
    l_cap: int = 4
    q_cap: int | None = None


def _path(*qs: int) -> list[tuple[int, int]]:
    return list(zip(qs[:-1], qs[1:], strict=True))


def _dead_end() -> Motif:
    # Host: corridor 0-1-2-3-4 with a pocket leaf 5 hanging off 1. Logical path a-b-c.
    # a sits on 0. b's candidates: 5 (pocket, adjacent to 1? no: adjacent to a? we make 5
    # adjacent to 0) or 1 (corridor). If b takes the pocket, c cannot touch b.
    h = nx.Graph(_path(0, 1, 2, 3, 4) + [(0, 5)])
    g = nx.Graph([(0, 1), (1, 2)])
    cores = {0: frozenset([0])}
    return Motif("dead_end", "nearest candidate enters a pocket", h, g, cores,
                 ((1, 5), (1, 1)), hypothesis_winner=(1, 1), l_cap=2)


def _articulation() -> Motif:
    # Host: two lobes joined by a single bridge qubit 3: lobe {0,1,2} triangle, lobe {4,5,6}
    # triangle, bridge 2-3-4. Logical: a-b, a-c, where a is at 0, b must reach lobe 2 (its
    # core is at 6) and c is unplaced. Candidates for c: qubit 3 (the bridge; compact,
    # adjacent to a's lobe via 2? we place a at 2 so 3 is adjacent) or qubit 1.
    h = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 6)])
    g = nx.Graph([(0, 1), (0, 2)])
    cores = {0: frozenset([2]), 1: frozenset([6])}
    return Motif("articulation", "candidate consumes the only bridge", h, g, cores,
                 ((2, 3), (2, 1)), hypothesis_winner=(2, 1), l_cap=3)


def _cut_capacity() -> Motif:
    # Host: left block L={0,1,2} (path), right block R={6,7,8} (path), cut couplers 2-3-6 and
    # 1-4-7 via two corridor qubits 3 and 4 (each a 1-wide channel), plus 5 a spare on the
    # right. Logical: a-c, b-d, a-b. a at 0? Let a core {2}, b core {1}; c must be in R: core
    # {6}; d in R: core {7}. Candidates for a: 3 (crosses cut, fine) or grab both 3 and 4?
    # Single-qubit actions: a takes 4 (the corridor b needs) vs a takes 3.
    h = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 6), (1, 4), (4, 7), (6, 7), (7, 8), (8, 5), (2, 4)])
    g = nx.Graph([(0, 2), (1, 3), (0, 1)])
    cores = {0: frozenset([2]), 1: frozenset([1]), 2: frozenset([6]), 3: frozenset([7])}
    return Motif("cut_capacity", "expansion consumes a narrow cut", h, g, cores,
                 ((0, 4), (0, 3)), hypothesis_winner=(0, 3), l_cap=3)


def _multi_neighbour() -> Motif:
    # Logical star centre a must connect to leaves fixed at 100 and 101. Candidate 10 wins
    # against candidate 0 for either leaf in isolation (Q=5 versus Q=6). With both leaves,
    # however, candidate 0 amortizes the shared 0-1-2-3 trunk and wins by one qubit (Q=8
    # versus Q=9). Thus each single-neighbour ablation strictly reverses the exact ranking.
    h = nx.Graph(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 100),
            (3, 5),
            (5, 101),
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 100),
            (10, 14),
            (14, 15),
            (15, 16),
            (16, 101),
        ]
    )
    g = nx.Graph([(0, 1), (0, 2)])
    cores = {1: frozenset([100]), 2: frozenset([101])}
    return Motif(
        "multi_neighbour",
        "joint-neighbour routing reverses both single-neighbour preferences",
        h,
        g,
        cores,
        ((0, 10), (0, 0)),
        hypothesis_winner=(0, 0),
        l_cap=7,
    )


def _high_degree_trap() -> Motif:
    # Hub qubit 3 adjacent to 0,1,2,4,5. Logical: a-b, c-d, c-e. a at 0. b's candidates:
    # hub 3 (free neighbours galore) or qubit 1 (adjacent to 0, low degree). c, d, e are
    # unplaced and c must touch both d and e; without the hub, c's only option to reach two
    # others is... make d core {4}, e core {5}: c must touch 4 and 5, only the hub does.
    h = nx.Graph([(0, 1), (0, 3), (1, 3), (2, 3), (3, 4), (3, 5)])
    g = nx.Graph([(0, 1), (2, 3), (2, 4)])
    cores = {0: frozenset([0]), 3: frozenset([4]), 4: frozenset([5])}
    return Motif("high_degree_trap", "hub is scarce and needed elsewhere", h, g, cores,
                 ((1, 3), (1, 1)), hypothesis_winner=(1, 1), l_cap=2)


def _short_chain_trap() -> Motif:
    # Host and logical graph are paths with their corresponding endpoints fixed. Seeding
    # logical node 1 at host node 4 concentrates the remaining slack and forces a terminal
    # chain of length 3. Seeding it at node 2 has the same feasible Q=8 completion but keeps
    # every terminal chain at length at most 2, isolating the max-chain objective component.
    h = nx.Graph(_path(0, 1, 2, 3, 4, 5, 6, 7))
    g = nx.Graph(_path(0, 1, 2, 3, 4))
    cores = {0: frozenset([0]), 4: frozenset([7])}
    return Motif(
        "short_chain_trap",
        "equal-qubit completions differ in terminal maximum chain length",
        h,
        g,
        cores,
        ((1, 4), (1, 2)),
        hypothesis_winner=(1, 2),
        l_cap=4,
    )


def _ordering() -> Motif:
    # Two variables a, b (logical edge a-b) and a third c adjacent to a only. Host: 0-1-2 path
    # plus 3 adjacent to 0. Placing b first at 1 (the central qubit) leaves a with {0} and c
    # with {3}: feasible. Placing a first at 1 leaves b and c needing 0,2 and 3: c must touch
    # a: 3 is not adjacent to 1 -> c={0}? then b={2}: feasible too. Make c need two contacts:
    # logical c-a and c-b. Host: 0-1-2 path, 3 adjacent to 1 only. Placing a at 1 first: b at
    # 0 or 2, c must touch both a(1) and b: c={3}? 3 adj 1 but not adj 0/2 -> c={2} if b={0}:
    # 2 adj 1 yes, 2 adj 0 no. c={0,?}. Infeasible with l_cap 1... but with l_cap 2 c={2}
    # and b={0}: c-b needs 2~0: no. c={3}: 3~1 yes, 3~0 no. Infeasible. Placing b at 1 first
    # is symmetric (a and b symmetric except c needs both). Real ordering motif: a has degree
    # 2 (a-b, a-c) and b, c leaves. Placing a leaf first at the centre blocks. Host: star with
    # centre 1 and leaves 0,2,3. Actions at the empty state: (b, 1) [leaf takes centre] vs
    # (a, 1) [hub variable takes centre]. l_cap 1 makes it strict.
    h = nx.Graph([(0, 1), (1, 2), (1, 3)])
    g = nx.Graph([(0, 1), (0, 2)])
    cores: dict[Node, frozenset[Qubit]] = {}
    return Motif(
        "ordering",
        "a leaf placed first takes the hub the centre variable needs",
        h,
        g,
        cores,
        ((1, 1), (0, 1)),
        hypothesis_winner=(0, 1),
        l_cap=1,
    )


MOTIFS = {
    "dead_end": _dead_end,
    "articulation": _articulation,
    "cut_capacity": _cut_capacity,
    "multi_neighbour": _multi_neighbour,
    "high_degree_trap": _high_degree_trap,
    "short_chain_trap": _short_chain_trap,
    "ordering": _ordering,
}


def build_motif(name: str) -> Motif:
    return MOTIFS[name]()


@dataclass(frozen=True)
class Certificate:
    motif: str
    sample: DecisionSample
    certified_winner: tuple[Node, Qubit]
    hypothesis_winner: tuple[Node, Qubit]
    margin: tuple[str, int]

    @property
    def agrees(self) -> bool:
        return self.certified_winner == self.hypothesis_winner and self.margin[1] > 0


def certify_motif(m: Motif) -> Certificate:
    s = sample_at(m.host, m.logical, m.cores, m.actions, l_cap=m.l_cap, q_cap=m.q_cap)
    r = s.ranking()
    return Certificate(m.name, s, s.actions[r[0]], m.hypothesis_winner, s.margin())


def certify_all() -> list[Certificate]:
    return [certify_motif(build()) for build in MOTIFS.values()]

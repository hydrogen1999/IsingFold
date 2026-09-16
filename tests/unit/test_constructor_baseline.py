"""The optional comparison consumes only its own unhinted router output."""
from types import SimpleNamespace

import networkx as nx

from isingfold.embedding import LogicalProblem


def test_baseline_uses_unhinted_own_output_and_same_commit_gate(monkeypatch):
    import seeded_minorminer
    from constructor_baseline import baseline_proposal
    graph = nx.path_graph(2)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    task = SimpleNamespace(logical=graph, host=nx.path_graph(5), problem=problem,
                           name="baseline-toy", lineage="baseline", ground_energy=-1.0)
    seen = []

    def unhinted(t, roots, seed, tries, **kwargs):
        assert t is task and roots is None
        assert kwargs["budget"] == 5 and 0 < kwargs["timeout"] <= 2.0
        seen.append(seed)
        return {0: frozenset({0}), 1: frozenset({1})}

    monkeypatch.setattr(seeded_minorminer, "attempt", unhinted)
    result = baseline_proposal(task, 5, 123, 10.0)
    assert seen == [123]
    assert result.returned_valid and result.embedding == {0: frozenset({0}), 1: frozenset({1})}
    assert result.selected_program.strength_index == result.selected_index


def test_baseline_failure_has_no_supplied_embedding_fallback(monkeypatch):
    import seeded_minorminer
    from constructor_baseline import baseline_proposal
    monkeypatch.setattr(seeded_minorminer, "attempt", lambda *args, **kw: None)
    assert baseline_proposal(object(), 5, 0, 1.0) is None

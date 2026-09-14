from __future__ import annotations

import networkx as nx
import pytest
from embedbench.exact import SearchAborted, best_completion


@pytest.mark.parametrize(
    ("max_nodes", "aborted", "expected_nodes"),
    [(2, True, 2), (3, False, 3), (4, False, 3)],
)
def test_best_completion_budget_counts_dfs_state_entries(
    max_nodes: int,
    aborted: bool,
    expected_nodes: int,
) -> None:
    host = nx.path_graph(2)
    logical = nx.Graph()
    logical.add_node(0)
    stats: dict[str, int] = {}

    if aborted:
        with pytest.raises(SearchAborted):
            best_completion(host, logical, l_cap=1, max_nodes=max_nodes, stats=stats)
    else:
        outcome, chains = best_completion(
            host,
            logical,
            l_cap=1,
            max_nodes=max_nodes,
            stats=stats,
        )
        assert outcome == (1, -1, -1)
        assert chains is not None
    assert stats["nodes"] == expected_nodes


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.0])
def test_best_completion_rejects_invalid_node_budgets(invalid: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        best_completion(nx.path_graph(2), nx.empty_graph(1), max_nodes=invalid)

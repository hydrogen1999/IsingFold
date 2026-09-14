from __future__ import annotations

import networkx as nx

from lac_minorminer import (
    SearchWorkCounters,
    TerminationReason,
    find_embedding,
    validate_embedding,
)


def _chimera_graph(
    *,
    coordinates: bool = False,
    rows: int = 4,
    columns: int = 4,
    tile: int = 4,
) -> nx.Graph:
    graph = nx.Graph()

    def integer_label(i: int, j: int, u: int, k: int) -> int:
        return ((i * columns + j) * 2 + u) * tile + k

    def label(i: int, j: int, u: int, k: int):
        if coordinates:
            return (i, j, u, k)
        return integer_label(i, j, u, k)

    for i in range(rows):
        for j in range(columns):
            for u in range(2):
                for k in range(tile):
                    graph.add_node(label(i, j, u, k))
            for vertical in range(tile):
                for horizontal in range(tile):
                    graph.add_edge(
                        label(i, j, 0, vertical),
                        label(i, j, 1, horizontal),
                    )
    for i in range(rows - 1):
        for j in range(columns):
            for k in range(tile):
                graph.add_edge(label(i, j, 0, k), label(i + 1, j, 0, k))
    for i in range(rows):
        for j in range(columns - 1):
            for k in range(tile):
                graph.add_edge(label(i, j, 1, k), label(i, j + 1, 1, k))
    return graph


def _faulted_chimera(*, coordinates: bool = False) -> nx.Graph:
    graph = _chimera_graph(coordinates=coordinates)

    def label(i: int, j: int, u: int, k: int):
        if coordinates:
            return (i, j, u, k)
        return ((i * 4 + j) * 2 + u) * 4 + k

    graph.remove_nodes_from(
        (
            label(0, 0, 0, 0),
            label(3, 3, 1, 3),
        )
    )
    graph.remove_edges_from(
        (
            (label(0, 1, 0, 1), label(0, 1, 1, 1)),
            (label(1, 2, 0, 2), label(2, 2, 0, 2)),
            (label(2, 1, 1, 0), label(2, 2, 1, 0)),
        )
    )
    return graph


def test_hybrid_profile_embeds_k12_on_faulted_chimera() -> None:
    logical = nx.complete_graph([f"x{index}" for index in range(12)])
    host = _faulted_chimera()
    _, legacy = find_embedding(
        logical,
        host,
        random_seed=2207,
        tries=1,
        max_transitions=31,
        max_candidates=8,
        search_profile="v0",
        return_diagnostics=True,
    )

    embedding, diagnostics = find_embedding(
        logical,
        host,
        random_seed=2207,
        tries=1,
        max_transitions=31,
        max_candidates=8,
        search_profile="hybrid_chimera_clique_v1",
        return_diagnostics=True,
    )

    assert not legacy.success
    assert validate_embedding(logical, host, embedding).valid
    assert diagnostics.success
    assert diagnostics.termination_reason is TerminationReason.SUCCESS
    assert diagnostics.search_profile == "hybrid_chimera_clique_v1"
    assert diagnostics.profile_detail == "v0_failed_chimera_clique_success"
    assert diagnostics.transitions == diagnostics.work.decisions > 0
    assert diagnostics.trace == legacy.trace
    assert (
        diagnostics.transitions,
        diagnostics.proposals,
        diagnostics.applied,
        diagnostics.discarded,
    ) == (
        legacy.transitions,
        legacy.proposals,
        legacy.applied,
        legacy.discarded,
    )
    assert all(
        diagnostics.work.as_dict()[name] >= value
        for name, value in legacy.work.as_dict().items()
    )
    assert diagnostics.work.materializations > 0
    assert sum(map(len, embedding.values())) > logical.number_of_nodes()


def test_hybrid_profile_supports_coordinate_host_labels_and_is_deterministic() -> None:
    logical = nx.complete_graph([f"logical-{index}" for index in range(12)])
    host = _faulted_chimera(coordinates=True)

    first_embedding, first = find_embedding(
        logical,
        host,
        random_seed=3301,
        tries=1,
        max_transitions=31,
        max_candidates=8,
        search_profile="hybrid_chimera_clique_v1",
        return_diagnostics=True,
    )
    second_embedding, second = find_embedding(
        logical,
        host,
        random_seed=3301,
        tries=1,
        max_transitions=31,
        max_candidates=8,
        search_profile="hybrid_chimera_clique_v1",
        return_diagnostics=True,
    )

    assert first_embedding == second_embedding
    assert first.structural_dict() == second.structural_dict()
    assert validate_embedding(logical, host, first_embedding).valid


def test_hybrid_profile_falls_back_to_v0_without_changing_legacy_search() -> None:
    logical = nx.path_graph(3)
    host = nx.cycle_graph(5)
    common = {
        "random_seed": 41,
        "tries": 1,
        "max_transitions": 20,
        "max_candidates": 4,
        "return_diagnostics": True,
    }

    legacy_embedding, legacy = find_embedding(
        logical,
        host,
        search_profile="v0",
        **common,
    )
    hybrid_embedding, hybrid = find_embedding(
        logical,
        host,
        search_profile="hybrid_chimera_clique_v1",
        **common,
    )

    assert hybrid_embedding == legacy_embedding
    assert hybrid.work == legacy.work
    assert hybrid.trace == legacy.trace
    assert hybrid.incumbents == legacy.incumbents
    assert hybrid.profile_detail == "v0_success"


def test_hybrid_profile_stops_prospectively_at_materialization_cap() -> None:
    work_cap = SearchWorkCounters(
        decisions=0,
        route_expansions=10_000,
        materializations=0,
        validator_calls=10_000,
        feature_work=1_000_000,
    )

    embedding, diagnostics = find_embedding(
        nx.complete_graph(12),
        _faulted_chimera(),
        random_seed=2207,
        tries=1,
        max_transitions=0,
        max_candidates=8,
        search_profile="hybrid_chimera_clique_v1",
        work_cap=work_cap,
        return_diagnostics=True,
    )

    assert embedding == {}
    assert diagnostics.termination_reason is TerminationReason.WORK_BUDGET_EXHAUSTED
    assert diagnostics.work_budget_exhausted_coordinate == "materializations"
    assert diagnostics.work.materializations == 0
    assert diagnostics.work_cap == work_cap


def test_hybrid_profile_rejects_sparse_huge_integer_labels_with_bounded_work() -> None:
    host = nx.Graph([(0, 1), (1, 2), (2, 3)])
    host.add_node(10**12)

    embedding, diagnostics = find_embedding(
        nx.complete_graph(5),
        host,
        random_seed=7,
        tries=1,
        max_transitions=0,
        max_candidates=2,
        search_profile="hybrid_chimera_clique_v1",
        return_diagnostics=True,
    )

    assert embedding == {}
    assert diagnostics.profile_detail.endswith("target_not_chimera")
    assert diagnostics.work.feature_work < 1_000


def test_hybrid_profile_reports_final_validator_cap_without_success_detail() -> None:
    logical = nx.complete_graph(12)
    host = _faulted_chimera()
    _, uncapped = find_embedding(
        logical,
        host,
        random_seed=2207,
        tries=1,
        max_transitions=31,
        max_candidates=8,
        search_profile="hybrid_chimera_clique_v1",
        return_diagnostics=True,
    )
    cap_values = uncapped.work.as_dict()
    cap_values["validator_calls"] -= 1

    embedding, diagnostics = find_embedding(
        logical,
        host,
        random_seed=2207,
        tries=1,
        max_transitions=31,
        max_candidates=8,
        search_profile="hybrid_chimera_clique_v1",
        work_cap=SearchWorkCounters(**cap_values),
        return_diagnostics=True,
    )

    assert embedding == {}
    assert diagnostics.termination_reason is TerminationReason.WORK_BUDGET_EXHAUSTED
    assert diagnostics.work_budget_exhausted_coordinate == "validator_calls"
    assert diagnostics.profile_detail.endswith("validator_budget_exhausted")


def test_hybrid_profile_preserves_seeded_initializer_diversity() -> None:
    logical = nx.complete_graph(12)
    host = _faulted_chimera()
    embeddings = []
    for seed in range(10):
        embedding = find_embedding(
            logical,
            host,
            random_seed=seed,
            tries=1,
            max_transitions=31,
            max_candidates=8,
            search_profile="hybrid_chimera_clique_v1",
        )
        assert validate_embedding(logical, host, embedding).valid
        embeddings.append(
            tuple(tuple(chain) for chain in embedding.values())
        )

    assert len(set(embeddings)) == len(embeddings)


def test_coordinate_chimera_above_registered_tile_limit_is_inapplicable() -> None:
    embedding, diagnostics = find_embedding(
        nx.complete_graph(20),
        _chimera_graph(coordinates=True, rows=2, columns=2, tile=17),
        random_seed=5,
        tries=1,
        max_transitions=0,
        max_candidates=2,
        search_profile="hybrid_chimera_clique_v1",
        return_diagnostics=True,
    )

    assert embedding == {}
    assert diagnostics.structural_fallback_invoked
    assert diagnostics.profile_detail.endswith("target_not_chimera")

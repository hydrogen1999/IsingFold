from __future__ import annotations

import hashlib
import inspect
import itertools
from dataclasses import FrozenInstanceError, replace

import networkx as nx
import pytest
from embedbench.candidate_protocol import (
    ATTEMPT_STATUSES,
    CANDIDATE_PROTOCOL_SCHEMA,
    CANDIDATE_PROTOCOL_SCHEMA_VERSION,
    ENUMERATION_CAP,
    CandidateProtocolResult,
    FrozenChain,
    candidate_bank_sha256,
    candidate_protocol_result_sha256,
    canonical_candidate_bank_bytes,
    canonical_candidate_bytes,
    generate_candidate_bank,
)
from embedbench.hard_ood_schema import canonical_bytes

SEED_KEY = bytes(range(32))


def _brute_candidates(
    host: nx.Graph,
    *,
    window_nodes: tuple[int, ...],
    frozen_chains: tuple[FrozenChain, ...],
    required_logical_neighbors: tuple[int, ...],
    l_cap: int,
) -> tuple[tuple[int, ...], ...]:
    frozen_by_logical = {chain.logical_variable: set(chain.nodes) for chain in frozen_chains}
    blocked = set().union(*(set(chain.nodes) for chain in frozen_chains))
    candidates: list[tuple[int, ...]] = []
    for length in range(1, l_cap + 1):
        for candidate in itertools.combinations(window_nodes, length):
            nodes = set(candidate)
            if not nodes.isdisjoint(blocked) or not nx.is_connected(host.subgraph(nodes)):
                continue
            if all(
                any(host.has_edge(left, right) for left in nodes for right in frozen_by_logical[n])
                for n in required_logical_neighbors
            ):
                candidates.append(candidate)
    return tuple(candidates)


def _generate(
    host: nx.Graph,
    *,
    window_nodes: tuple[int, ...],
    frozen_chains: tuple[FrozenChain, ...] = (),
    required_logical_neighbors: tuple[int, ...] = (),
    original_focus_chain: tuple[int, ...] = (0,),
    l_cap: int = 3,
    q_cap: int | None = None,
) -> CandidateProtocolResult:
    return generate_candidate_bank(
        host,
        window_nodes=window_nodes,
        frozen_chains=frozen_chains,
        required_logical_neighbors=required_logical_neighbors,
        original_focus_chain=original_focus_chain,
        l_cap=l_cap,
        candidate_sample_seed_key=SEED_KEY,
        q_cap=q_cap,
    )


def test_full_enumeration_matches_independent_exhaustive_reference() -> None:
    host = nx.Graph(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (2, 5),
            (4, 6),
            (1, 7),
            (3, 8),
        ]
    )
    frozen = (FrozenChain(10, (7,)), FrozenChain(20, (8,)))
    expected = _brute_candidates(
        host,
        window_nodes=(0, 1, 2, 3, 4, 5, 6),
        frozen_chains=frozen,
        required_logical_neighbors=(10, 20),
        l_cap=4,
    )

    result = _generate(
        host,
        window_nodes=(0, 1, 2, 3, 4, 5, 6),
        frozen_chains=frozen,
        required_logical_neighbors=(10, 20),
        original_focus_chain=(1, 2, 3),
        l_cap=4,
    )

    assert result.attempt_status == (
        "single_candidate" if len(expected) == 1 else "candidate_bank_ready"
    )
    assert result.enumeration_status == "complete"
    assert result.full_candidates == expected
    assert result.offered_candidates == expected
    assert result.offered_to_full_indices == tuple(range(len(expected)))
    assert all(facts.minor_valid for facts in result.full_candidate_facts or ())


def test_complete_bank_is_canonical_duplicate_free_and_contains_original() -> None:
    host = nx.complete_graph(8)
    result = _generate(
        host,
        window_nodes=tuple(range(8)),
        original_focus_chain=(5, 7),
        l_cap=2,
    )

    assert result.full_candidates is not None
    assert result.full_candidates == tuple(
        sorted(set(result.full_candidates), key=lambda candidate: (len(candidate), candidate))
    )
    original_index = result.full_candidates.index((5, 7))
    assert result.offered_to_full_indices is not None
    assert original_index in result.offered_to_full_indices


def test_bank_over_64_uses_all_four_anchors_and_exact_round_robin() -> None:
    # Every window node contacts logical neighbour 99.  There are 12 + C(12,2) = 78
    # candidates, enough to exercise truncation while keeping the reference transparent.
    host = nx.complete_graph(13)
    frozen = (FrozenChain(99, (12,)),)
    result = _generate(
        host,
        window_nodes=tuple(range(12)),
        frozen_chains=frozen,
        required_logical_neighbors=(99,),
        original_focus_chain=(11,),
        l_cap=2,
    )

    full = result.full_candidates
    facts = result.full_candidate_facts
    assert full is not None and facts is not None
    assert len(full) == 78

    selected = {full.index((11,))}
    selected.add(
        min(
            range(len(full)),
            key=lambda i: (len(full[i]), -facts[i].internal_chain_edge_count, i),
        )
    )
    selected.add(
        max(
            range(len(full)),
            key=lambda i: (
                facts[i].largest_free_component_node_numerator,
                facts[i].largest_free_component_edge_connectivity,
                -i,
            ),
        )
    )
    selected.add(
        max(
            range(len(full)),
            key=lambda i: (
                facts[i].minimum_logical_contact_count,
                facts[i].total_logical_contact_count,
                facts[i].sorted_logical_contact_counts,
                -i,
            ),
        )
    )
    strata: dict[tuple[int, int, int], list[int]] = {}
    for index, (candidate, candidate_facts) in enumerate(zip(full, facts, strict=True)):
        if index in selected:
            continue
        stratum = (
            len(candidate),
            candidate_facts.minimum_logical_contact_count,
            candidate_facts.total_logical_contact_count,
        )
        strata.setdefault(stratum, []).append(index)
    for indices in strata.values():
        indices.sort(
            key=lambda i: (
                hashlib.sha256(canonical_bytes([SEED_KEY.hex(), list(full[i])])).digest(),
                full[i],
                i,
            )
        )
    ordered_strata = sorted(strata)
    cursor = {key: 0 for key in ordered_strata}
    while len(selected) < 64:
        for key in ordered_strata:
            offset = cursor[key]
            if offset < len(strata[key]):
                selected.add(strata[key][offset])
                cursor[key] += 1
                if len(selected) == 64:
                    break
    expected_indices = tuple(sorted(selected))

    assert result.offered_candidate_count == 64
    assert result.offered_to_full_indices == expected_indices
    assert result.offered_candidates == tuple(full[index] for index in expected_indices)
    assert full.index((11,)) in expected_indices


def test_anchor_ties_use_smallest_full_bank_canonical_index() -> None:
    host = nx.complete_graph(13)
    result = _generate(
        host,
        window_nodes=tuple(range(12)),
        frozen_chains=(FrozenChain(99, (12,)),),
        required_logical_neighbors=(99,),
        original_focus_chain=(11,),
        l_cap=2,
    )

    assert result.full_candidates is not None
    assert result.offered_to_full_indices is not None
    # Steps 2 and 3 both tie across all singleton candidates and must choose index zero.
    assert result.full_candidates[0] == (0,)
    assert 0 in result.offered_to_full_indices
    # Step 4 favours two contacts, then the first canonical pair.
    assert result.full_candidates[12] == (0, 1)
    assert 12 in result.offered_to_full_indices


def test_round_robin_handles_more_than_64_distinct_strata() -> None:
    host = nx.Graph()
    window = tuple(range(65))
    neighbour_chain = tuple(range(100, 165))
    host.add_nodes_from((*window, *neighbour_chain))
    host.add_edges_from(zip(neighbour_chain[:-1], neighbour_chain[1:], strict=True))
    for candidate_node in window:
        host.add_edges_from(
            (candidate_node, frozen_node) for frozen_node in neighbour_chain[: candidate_node + 1]
        )

    result = _generate(
        host,
        window_nodes=window,
        frozen_chains=(FrozenChain(9, neighbour_chain),),
        required_logical_neighbors=(9,),
        original_focus_chain=(64,),
        l_cap=1,
    )

    assert result.full_candidate_facts is not None
    assert (
        len(
            {
                (
                    len(facts.candidate),
                    facts.minimum_logical_contact_count,
                    facts.total_logical_contact_count,
                )
                for facts in result.full_candidate_facts
            }
        )
        == 65
    )
    # Resource/connectivity anchors choose index 0, contact/original choose index 64;
    # the first round-robin pass then visits ascending remaining strata until slot 64.
    assert result.offered_to_full_indices == (*range(63), 64)


def test_enumeration_aborts_only_after_candidate_4001() -> None:
    # K18 has 4,047 connected subsets of sizes one through four.
    host = nx.complete_graph(18)
    result = _generate(
        host,
        window_nodes=tuple(range(18)),
        original_focus_chain=(0,),
        l_cap=4,
    )
    prefix = tuple(
        candidate
        for length in range(1, 5)
        for candidate in itertools.combinations(range(18), length)
    )[: ENUMERATION_CAP + 1]

    assert result.attempt_status == "enumeration_aborted"
    assert result.enumeration_status == "aborted_cap"
    assert result.enumerated_prefix_count == 4_001
    assert result.enumerated_prefix_sha256 == candidate_bank_sha256(prefix)
    assert result.full_candidates is None
    assert result.full_candidate_count is None
    assert result.full_candidate_bank_sha256 is None
    assert result.offered_candidates is None
    assert result.offered_candidate_count is None
    assert result.offered_candidate_bank_sha256 is None
    assert result.offered_to_full_indices is None


def test_original_not_in_full_bank_is_invalid_generation_before_enumeration() -> None:
    host = nx.path_graph(4)
    result = _generate(
        host,
        window_nodes=(0, 1, 2),
        frozen_chains=(FrozenChain(7, (3,)),),
        required_logical_neighbors=(7,),
        original_focus_chain=(0,),  # no contact to chain (3,)
        l_cap=2,
    )

    assert result.attempt_status == "invalid_generation"
    assert result.enumeration_status is None
    assert result.enumerated_prefix_count is None
    assert result.enumerated_prefix_sha256 is None
    assert result.full_candidates is None
    assert result.offered_candidates is None


def test_single_original_candidate_has_single_candidate_status() -> None:
    host = nx.Graph()
    host.add_node(9)
    result = _generate(
        host,
        window_nodes=(9,),
        original_focus_chain=(9,),
        l_cap=1,
    )

    assert result.attempt_status == "single_candidate"
    assert result.full_candidates == ((9,),)
    assert result.offered_candidates == ((9,),)
    assert result.offered_to_full_indices == (0,)


def test_l_cap_larger_than_window_has_no_empty_subset_iterations() -> None:
    host = nx.Graph()
    host.add_node(9)
    result = _generate(
        host,
        window_nodes=(9,),
        original_focus_chain=(9,),
        l_cap=2**64,
    )

    assert result.full_candidates == ((9,),)


def test_exact_candidate_facts_use_whole_realized_host_and_replaced_embedding() -> None:
    host = nx.Graph(
        [
            (0, 1),
            (0, 4),
            (0, 5),
            (1, 2),
            (1, 4),
            (1, 5),
            (2, 3),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 6),
        ]
    )
    frozen = (FrozenChain(10, (4,)), FrozenChain(20, (2, 3)))
    result = _generate(
        host,
        window_nodes=(0, 1),
        frozen_chains=frozen,
        required_logical_neighbors=(10, 20),
        original_focus_chain=(1,),
        l_cap=2,
        q_cap=4,
    )
    assert result.full_candidates == ((1,), (0, 1))
    facts = result.full_candidate_facts
    assert facts is not None

    first, second = facts
    assert first.logical_contact_counts == (1, 1)
    assert first.sorted_logical_contact_counts == (1, 1)
    assert first.minimum_logical_contact_count == 1
    assert first.total_logical_contact_count == 2
    assert first.current_total_qubits == 4
    assert first.current_maximum_chain_length == 2
    assert first.within_q_cap is True
    assert first.feasible_now is True
    # Free nodes are {0,5,6}; they form a path in the whole host, not merely the window.
    assert first.largest_free_component_nodes == (0, 5, 6)
    assert first.largest_free_component_node_numerator == 3
    assert first.largest_free_component_node_denominator == 7
    assert first.largest_free_component_edge_connectivity == 1

    assert second.internal_chain_edge_count == 1
    assert second.current_total_qubits == 5
    assert second.within_q_cap is False
    assert second.feasible_now is False


def test_largest_free_component_tie_uses_smallest_sorted_uint64_tuple() -> None:
    host = nx.Graph()
    host.add_nodes_from(range(6))
    host.add_edges_from(((1, 4), (2, 3)))
    result = _generate(
        host,
        window_nodes=(0,),
        original_focus_chain=(0,),
        l_cap=1,
    )

    assert result.full_candidate_facts is not None
    facts = result.full_candidate_facts[0]
    assert facts.largest_free_component_nodes == (1, 4)
    assert facts.largest_free_component_node_numerator == 2
    assert facts.largest_free_component_edge_connectivity == 1


def test_none_q_cap_is_exactly_uncapped() -> None:
    host = nx.path_graph(3)
    result = _generate(
        host,
        window_nodes=(0, 1),
        frozen_chains=(FrozenChain(8, (2,)),),
        required_logical_neighbors=(8,),
        original_focus_chain=(1,),
        l_cap=2,
        q_cap=None,
    )

    assert result.full_candidate_facts is not None
    assert all(facts.within_q_cap is True for facts in result.full_candidate_facts)
    assert all(facts.feasible_now == facts.minor_valid for facts in result.full_candidate_facts)


def test_candidate_and_bank_encoding_have_frozen_golden_digests() -> None:
    candidate = (1, 2**64 - 1)
    assert canonical_candidate_bytes(candidate) == (
        (2).to_bytes(8, "big") + (1).to_bytes(8, "big") + (2**64 - 1).to_bytes(8, "big")
    )
    assert canonical_candidate_bank_bytes(()) == (0).to_bytes(8, "big")
    bank = ((1,), (2, 3))
    assert canonical_candidate_bank_bytes(bank) == (
        (2).to_bytes(8, "big")
        + (1).to_bytes(8, "big")
        + (1).to_bytes(8, "big")
        + (2).to_bytes(8, "big")
        + (2).to_bytes(8, "big")
        + (3).to_bytes(8, "big")
    )
    assert candidate_bank_sha256(()) == (
        "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
    )
    assert candidate_bank_sha256(bank) == (
        "94700a443c6ab618ad4f5508857f9a2d739cf8b715b59c7bd7dd22510cf6c89f"
    )


def test_full_and_offered_digests_bind_their_exact_canonical_arrays() -> None:
    host = nx.complete_graph(13)
    result = _generate(
        host,
        window_nodes=tuple(range(12)),
        frozen_chains=(FrozenChain(90, (12,)),),
        required_logical_neighbors=(90,),
        original_focus_chain=(11,),
        l_cap=2,
    )

    assert result.full_candidates is not None
    assert result.offered_candidates is not None
    assert result.full_candidate_bank_sha256 == candidate_bank_sha256(result.full_candidates)
    assert result.offered_candidate_bank_sha256 == candidate_bank_sha256(result.offered_candidates)
    assert result.enumerated_prefix_sha256 == result.full_candidate_bank_sha256
    assert result.full_candidate_bank_sha256 == (
        "759820c8bf974db0c476e95d7d87fcba24353a3063589a97e08d6a56c6d37a3a"
    )
    assert result.offered_candidate_bank_sha256 == (
        "12aea159c90a1f2a029df0a5d464041d94e91a652abcb92cb9e6468bf58b54b0"
    )


def test_output_is_closed_immutable_and_statuses_are_exact() -> None:
    result = _generate(nx.path_graph(2), window_nodes=(0, 1), l_cap=2)

    assert CANDIDATE_PROTOCOL_SCHEMA == "embedbench.candidate-protocol-result"
    assert CANDIDATE_PROTOCOL_SCHEMA_VERSION == 1
    assert CandidateProtocolResult.SCHEMA == CANDIDATE_PROTOCOL_SCHEMA
    assert CandidateProtocolResult.SCHEMA_VERSION == CANDIDATE_PROTOCOL_SCHEMA_VERSION
    assert result.to_dict()["schema"] == "embedbench.candidate-protocol-result"
    assert result.to_dict()["schema_version"] == 1

    assert (
        frozenset(
            {
                "starting_embedding_unavailable",
                "invalid_generation",
                "enumeration_aborted",
                "no_candidate",
                "single_candidate",
                "candidate_bank_ready",
            }
        )
        == ATTEMPT_STATUSES
    )
    with pytest.raises(FrozenInstanceError):
        result.attempt_status = "no_candidate"  # type: ignore[misc]
    with pytest.raises(ValueError, match="registered candidate-generation status"):
        replace(result, attempt_status="invented")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="full_candidate_count"):
        replace(result, full_candidate_count=True)
    for invalid_cap in (True, 4_000.0):
        with pytest.raises(ValueError, match="enumeration_cap must be a positive integer"):
            replace(result, enumeration_cap=invalid_cap)
    assert not hasattr(result, "__dict__")
    assert set(result.to_dict()) == {
        "schema",
        "schema_version",
        "attempt_status",
        "enumeration_status",
        "enumeration_cap",
        "enumerated_prefix_count",
        "enumerated_prefix_sha256",
        "full_candidates",
        "full_candidate_facts",
        "full_candidate_count",
        "full_candidate_bank_sha256",
        "offered_candidates",
        "offered_candidate_count",
        "offered_candidate_bank_sha256",
        "offered_to_full_indices",
    }
    assert {
        "host_sha256",
        "window_nodes",
        "frozen_chains",
        "original_focus_chain",
        "l_cap",
        "q_cap",
        "candidate_sample_seed_key",
    }.isdisjoint(result.to_dict())
    assert "enclosing state record" in (CandidateProtocolResult.__doc__ or "")


def test_protocol_result_digest_binds_schema_banks_indices_and_candidate_facts() -> None:
    result = _generate(nx.path_graph(3), window_nodes=(0, 1, 2), l_cap=2, q_cap=1)
    uncapped = _generate(nx.path_graph(3), window_nodes=(0, 1, 2), l_cap=2, q_cap=None)

    digest = candidate_protocol_result_sha256(result)
    assert digest == "c35b461abefa5ef116ba6eed199f0499a444fababb0da889e0ec189dd25aa8a8"
    assert digest == hashlib.sha256(canonical_bytes(result.to_dict())).hexdigest()
    assert result.full_candidate_bank_sha256 == uncapped.full_candidate_bank_sha256
    assert candidate_protocol_result_sha256(uncapped) != digest

    with pytest.raises(TypeError, match="exact CandidateProtocolResult"):
        candidate_protocol_result_sha256(result.to_dict())  # type: ignore[arg-type]


def test_public_generation_api_cannot_consume_labels_or_baseline_selections() -> None:
    parameters = inspect.signature(generate_candidate_bank).parameters
    forbidden = {
        "p_solve",
        "ground_energy",
        "ground_state_energy",
        "baseline_selected_index",
        "learned_score",
        "labels",
    }
    assert forbidden.isdisjoint(parameters)
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )

    with pytest.raises(TypeError, match="p_solve"):
        generate_candidate_bank(  # type: ignore[call-arg]
            nx.path_graph(2),
            window_nodes=(0, 1),
            frozen_chains=(),
            required_logical_neighbors=(),
            original_focus_chain=(0,),
            l_cap=1,
            candidate_sample_seed_key=SEED_KEY,
            p_solve=(0.9,),
        )


@pytest.mark.parametrize(
    ("host", "message"),
    [
        (nx.DiGraph([(0, 1)]), "undirected simple"),
        (nx.MultiGraph([(0, 1)]), "undirected simple"),
        (nx.Graph([(0, 0)]), "self-loops"),
        (nx.Graph([(False, 1)]), "unsigned 64-bit"),
    ],
)
def test_invalid_realized_graph_is_rejected(host: nx.Graph, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _generate(host, window_nodes=(0,), original_focus_chain=(0,), l_cap=1)


def test_malformed_frozen_context_and_required_contacts_are_rejected() -> None:
    host = nx.path_graph(6)
    with pytest.raises(ValueError, match="pairwise disjoint"):
        _generate(
            host,
            window_nodes=(0, 1),
            frozen_chains=(FrozenChain(7, (2, 3)), FrozenChain(8, (3, 4))),
            original_focus_chain=(0,),
            l_cap=1,
        )
    with pytest.raises(ValueError, match="connected"):
        _generate(
            host,
            window_nodes=(0, 1),
            frozen_chains=(FrozenChain(7, (2, 4)),),
            original_focus_chain=(0,),
            l_cap=1,
        )
    with pytest.raises(ValueError, match="required logical neighbor"):
        _generate(
            host,
            window_nodes=(0, 1),
            frozen_chains=(FrozenChain(7, (2,)),),
            required_logical_neighbors=(8,),
            original_focus_chain=(1,),
            l_cap=1,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"window_nodes": [0, 1]}, "tuple"),
        ({"window_nodes": (1, 0)}, "sorted"),
        ({"window_nodes": (0, 0)}, "duplicate-free"),
        ({"l_cap": True}, "positive integer"),
        ({"l_cap": 1.5}, "positive integer"),
        ({"q_cap": False}, "non-negative integer"),
        ({"q_cap": float("nan")}, "non-negative integer"),
        ({"candidate_sample_seed_key": b"short"}, "32 bytes"),
        ({"candidate_sample_seed_key": bytearray(32)}, "raw bytes"),
        ({"original_focus_chain": ()}, "must not be empty"),
    ],
)
def test_bool_nonfinite_and_malformed_generation_inputs_are_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "window_nodes": (0, 1),
        "frozen_chains": (),
        "required_logical_neighbors": (),
        "original_focus_chain": (0,),
        "l_cap": 2,
        "candidate_sample_seed_key": SEED_KEY,
        "q_cap": None,
    }
    arguments.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        generate_candidate_bank(nx.path_graph(2), **arguments)  # type: ignore[arg-type]


def test_frozen_chain_schema_rejects_bool_unsorted_duplicate_and_out_of_range_nodes() -> None:
    with pytest.raises(ValueError, match="logical_variable must be an integer"):
        FrozenChain(True, (0,))
    with pytest.raises(ValueError, match="sorted"):
        FrozenChain(1, (2, 1))
    with pytest.raises(ValueError, match="duplicate-free"):
        FrozenChain(1, (2, 2))
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        FrozenChain(1, (-1,))


def test_generation_is_independent_of_unrelated_mutable_label_objects() -> None:
    host = nx.complete_graph(6)
    before = _generate(host, window_nodes=tuple(range(6)), original_focus_chain=(4,), l_cap=2)
    unrelated_labels = {
        candidate: index / 10 for index, candidate in enumerate(before.full_candidates or ())
    }
    for candidate in unrelated_labels:
        unrelated_labels[candidate] = 1.0 - unrelated_labels[candidate]
    after = _generate(host, window_nodes=tuple(range(6)), original_focus_chain=(4,), l_cap=2)

    assert after == before


def test_generation_is_independent_of_host_insertion_order() -> None:
    edges = ((0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (2, 5))
    forward = nx.Graph()
    forward.add_nodes_from(range(6))
    forward.add_edges_from(edges)
    reverse = nx.Graph()
    reverse.add_nodes_from(reversed(range(6)))
    reverse.add_edges_from((right, left) for left, right in reversed(edges))
    arguments = {
        "window_nodes": (0, 1, 2, 3),
        "frozen_chains": (FrozenChain(7, (4,)), FrozenChain(8, (5,))),
        "required_logical_neighbors": (7, 8),
        "original_focus_chain": (1, 2),
        "l_cap": 3,
        "candidate_sample_seed_key": SEED_KEY,
    }

    assert generate_candidate_bank(forward, **arguments) == generate_candidate_bank(
        reverse, **arguments
    )

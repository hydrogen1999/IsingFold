#include "lac_minorminer/search_session.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Operation>
void require_logic_error(Operation operation, const std::string& message) {
    try {
        operation();
    } catch (const std::logic_error&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    using lac::minorminer::Graph;
    using lac::minorminer::SearchSession;

    try {
        SearchSession session(Graph(1, {}), Graph(3, {{0, 1}, {1, 2}}), 17, 2);
        const auto initial = session.snapshot();
        require(initial.chains.size() == 1 && initial.chains[0].size() == 1,
                "restart must initialize every logical chain");
        require(initial.valid, "a placed isolated variable should be a valid embedding");

        const auto batch = session.propose(0, {4.0, 1.0, 2.0});
        require(batch.candidates.size() == 2, "session did not honor its candidate bound");
        require(batch.candidates[0].chain == std::vector<Graph::Node>({1}),
                "session must forward dense custom costs");
        require_logic_error([&] { static_cast<void>(session.propose(0)); },
                            "only one proposal may be outstanding");
        session.discard(batch);
        require_logic_error([&] { session.discard(batch); }, "a proposal token must be single-use");

        const auto audited = session.propose_with_audit(0, 3, {4.0, 1.0, 2.0});
        require(audited.decision_batch.candidates.size() == 2,
                "audit mode must preserve the configured decision bound");
        require(audited.audit_batch.candidates.size() == 3,
                "audit mode must expose the wider pre-truncation batch");
        for (std::size_t index = 0; index < audited.decision_batch.candidates.size(); ++index) {
            require(audited.decision_batch.candidates[index].candidate_id ==
                            audited.audit_batch.candidates[index].candidate_id,
                    "decision and audit batches must share candidate IDs");
        }
        require_logic_error([&] { session.apply(audited.audit_batch, 0); },
                            "an audit-only batch must never mutate the session");
        session.discard(audited.decision_batch);

        SearchSession wide_session(Graph(1, {}), Graph(3, {{0, 1}, {1, 2}}), 17, 1);
        const auto narrow = wide_session.propose(0, {4.0, 1.0, 2.0});
        require(narrow.candidates.size() == 1 && narrow.candidates[0].chain ==
                        std::vector<Graph::Node>({1}),
                "the default proposal must retain its configured resource-first bound");
        wide_session.discard(narrow);

        const auto wide = wide_session.propose_applicable(0, 3, {4.0, 1.0, 2.0});
        require(wide.session_id == wide_session.session_id(),
                "candidate batches must expose their owning native session");
        require(wide.candidates.size() == 3,
                "the applicable scoring batch must honor the caller-selected wider bound");
        require(wide.candidates[2].candidate_id == 2 &&
                        wide.candidates[2].chain == std::vector<Graph::Node>({0}),
                "the wider applicable batch must preserve pre-truncation candidate IDs");
        wide_session.apply(wide, 2);
        require(wide_session.snapshot().chains[0] == std::vector<Graph::Node>({0}),
                "a candidate outside the default decision bound must be applicable");
        require_logic_error([&] { wide_session.apply(wide, 2); },
                            "an applicable scoring batch must be single-use");

        auto validation_session = SearchSession::from_chains(
                Graph(2, {{0, 1}}), Graph(3, {{0, 1}, {1, 2}}), {{0}, {2}},
                17, 2);
        const auto validation_batch = validation_session->materialize(0, {{1, 2}, {1}});
        std::size_t overlap_index = validation_batch.candidates.size();
        std::size_t valid_index = validation_batch.candidates.size();
        for (std::size_t index = 0; index < validation_batch.candidates.size(); ++index) {
            if (validation_batch.candidates[index].chain == std::vector<Graph::Node>({1, 2})) {
                overlap_index = index;
            }
            if (validation_batch.candidates[index].chain == std::vector<Graph::Node>({1})) {
                valid_index = index;
            }
        }
        require(overlap_index < validation_batch.candidates.size() &&
                        valid_index < validation_batch.candidates.size(),
                "validation fixture candidates were not materialized");
        require(!validation_session->candidate_is_valid(validation_batch, overlap_index),
                "native candidate validation must reject chain overlap");
        require(validation_session->candidate_is_valid(validation_batch, valid_index),
                "native candidate validation rejected an exact embedding");
        auto foreign_validation = SearchSession::from_chains(
                Graph(2, {{0, 1}}), Graph(3, {{0, 1}, {1, 2}}), {{0}, {2}},
                17, 2);
        require_logic_error(
                [&] { static_cast<void>(foreign_validation->candidate_is_valid(
                              validation_batch, valid_index)); },
                "candidate validation must reject a foreign session handle");
        validation_session->apply(validation_batch, valid_index);
        require(validation_session->snapshot().valid,
                "candidate validation must not consume the applicable handle");

        const auto stale_wide = wide_session.propose_applicable(0, 3);
        wide_session.restart();
        require_logic_error([&] { wide_session.apply(stale_wide, 0); },
                            "restart must invalidate an applicable scoring batch");

        SearchSession foreign_wide_session(Graph(1, {}), Graph(3, {{0, 1}, {1, 2}}), 17, 1);
        const auto foreign_wide = wide_session.propose_applicable(0, 3);
        const auto foreign_destination_batch = foreign_wide_session.propose_applicable(0, 3);
        require_logic_error([&] { foreign_wide_session.apply(foreign_wide, 0); },
                            "applicable scoring batches must remain session-local");
        foreign_wide_session.discard(foreign_destination_batch);
        wide_session.discard(foreign_wide);

        const auto wrong_index_wide = wide_session.propose_applicable(0, 3);
        try {
            wide_session.apply(wrong_index_wide, wrong_index_wide.candidates.size());
            throw std::runtime_error("wide candidate bounds must be checked before apply");
        } catch (const std::out_of_range&) {
        }
        wide_session.apply(wrong_index_wide, 0);

        try {
            static_cast<void>(wide_session.propose_applicable(0, 0));
            throw std::runtime_error("zero applicable scoring bounds must be rejected");
        } catch (const std::invalid_argument&) {
        }

        try {
            static_cast<void>(session.materialize(0, {{0, 2}}));
            throw std::runtime_error("disconnected supplied chains must be rejected");
        } catch (const std::invalid_argument&) {
        }
        const auto materialized = session.materialize(0, {{0, 1}, {2}});
        require(materialized.candidates.size() == 2, "valid supplied chains were not annotated");
        const auto before_apply = session.snapshot().generation;
        try {
            session.apply(materialized, 2);
            throw std::runtime_error("out-of-range candidate IDs must be rejected");
        } catch (const std::out_of_range&) {
        }
        session.apply(materialized, 0);
        const auto applied = session.snapshot();
        require(applied.generation == before_apply + 1, "apply must advance the generation");
        require(applied.chains[0] == std::vector<Graph::Node>({0, 1}),
                "apply must use the selected candidate");
        require_logic_error([&] { session.apply(materialized, 0); },
                            "an applied proposal must be single-use");

        const auto pending = session.propose(0);
        const auto before_restart = session.snapshot().generation;
        session.restart();
        require(session.snapshot().generation == before_restart + 1,
                "restart must advance the generation exactly once");
        require_logic_error([&] { session.discard(pending); },
                            "restart must invalidate outstanding proposals");

        const auto local = session.propose(0);
        SearchSession other(Graph(1, {}), Graph(3, {{0, 1}, {1, 2}}), 17, 2);
        require_logic_error([&] { other.apply(local, 0); },
                            "a proposal from another session must be rejected");
        session.discard(local);

        SearchSession fork_source(Graph(1, {}), Graph(4, {{0, 1}, {1, 2}, {2, 3}}),
                                  123, 4);
        const auto source_update = fork_source.materialize(0, {{0, 1}});
        fork_source.apply(source_update, 0);
        auto branch = fork_source.fork();
        require(branch->snapshot().chains == fork_source.snapshot().chains,
                "fork must preserve the complete embedding state");
        require(branch->snapshot().generation == fork_source.snapshot().generation,
                "fork must preserve the observable generation");

        const auto parent_update = fork_source.materialize(0, {{2, 3}});
        fork_source.apply(parent_update, 0);
        const auto branch_update = branch->materialize(0, {{1, 2}});
        branch->apply(branch_update, 0);
        require(fork_source.snapshot().chains[0] == std::vector<Graph::Node>({2, 3}),
                "parent update drifted after branch mutation");
        require(branch->snapshot().chains[0] == std::vector<Graph::Node>({1, 2}),
                "forked branch must mutate independently");
        require_logic_error([&] { branch->apply(parent_update, 0); },
                            "proposal handles must remain session-local after fork");

        SearchSession rng_source(Graph(3, {{0, 1}, {1, 2}}),
                                 Graph(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}}), 991,
                                 3);
        auto rng_branch = rng_source.fork();
        rng_source.restart();
        rng_branch->restart();
        require(rng_source.snapshot().chains == rng_branch->snapshot().chains,
                "fork must preserve the native RNG state for common-random-number branches");

        auto reseeded_a = rng_source.fork_with_seed(444);
        auto reseeded_b = rng_source.fork_with_seed(444);
        reseeded_a->restart();
        reseeded_b->restart();
        require(reseeded_a->snapshot().chains == reseeded_b->snapshot().chains,
                "same-seed counterfactual forks must use common native random numbers");

        auto restored_a = SearchSession::from_chains(
                Graph(2, {{0, 1}}), Graph(4, {{0, 1}, {1, 2}, {2, 3}}),
                {{0, 1}, {2}}, 27101, 3);
        auto restored_b = SearchSession::from_chains(
                Graph(2, {{0, 1}}), Graph(4, {{0, 1}, {1, 2}, {2, 3}}),
                {{0, 1}, {2}}, 27101, 3);
        require(restored_a->snapshot().chains == restored_b->snapshot().chains &&
                        restored_a->snapshot().valid,
                "from_chains must reconstruct the exact semantic embedding state");
        restored_a->restart();
        restored_b->restart();
        require(restored_a->snapshot().chains == restored_b->snapshot().chains,
                "restored sessions with the same seed must preserve common randomness");

        auto repair_a = SearchSession::from_chains(
                Graph(5, {}), Graph(8, {{0, 1}, {1, 2}, {2, 3}, {3, 4},
                                        {4, 5}, {5, 6}, {6, 7}}),
                {{0, 1}, {2}, {3}, {4}, {5}}, 808, 4);
        auto repair_b = SearchSession::from_chains(
                Graph(5, {}), Graph(8, {{0, 1}, {1, 2}, {2, 3}, {3, 4},
                                        {4, 5}, {5, 6}, {6, 7}}),
                {{0, 1}, {2}, {3}, {4}, {5}}, 808, 4);
        const auto repair_generation = repair_a->snapshot().generation;
        repair_a->perturb({0, 2, 4});
        repair_b->perturb({0, 2, 4});
        const auto repaired = repair_a->snapshot();
        require(repaired.generation == repair_generation + 1,
                "one local perturbation must advance the generation exactly once");
        require(repaired.chains == repair_b->snapshot().chains,
                "same-seed perturbations must preserve common random numbers");
        require(repaired.chains[1] == std::vector<Graph::Node>({2}) &&
                        repaired.chains[3] == std::vector<Graph::Node>({4}),
                "local perturbation must preserve chains outside its neighborhood");
        require(repaired.chains[0].size() == 1 && repaired.chains[2].size() == 1 &&
                        repaired.chains[4].size() == 1,
                "local perturbation must reset every selected chain to a singleton");
        require(repaired.max_occupancy == 1,
                "local perturbation must prefer distinct currently free target nodes");

        try {
            repair_a->perturb({});
            throw std::runtime_error("empty repair neighborhoods must be rejected");
        } catch (const std::invalid_argument&) {
        }
        try {
            repair_a->perturb({0, 0});
            throw std::runtime_error("duplicate repair variables must be rejected");
        } catch (const std::invalid_argument&) {
        }
        try {
            repair_a->perturb({5});
            throw std::runtime_error("out-of-range repair variables must be rejected");
        } catch (const std::out_of_range&) {
        }
        const auto pending_repair = repair_a->propose(0);
        repair_a->perturb({0});
        require_logic_error([&] { repair_a->discard(pending_repair); },
                            "local perturbation must invalidate outstanding proposals");

        const auto outstanding = rng_source.propose(0);
        require_logic_error([&] { static_cast<void>(rng_source.fork()); },
                            "fork must reject a session with an outstanding proposal");
        rng_source.discard(outstanding);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}

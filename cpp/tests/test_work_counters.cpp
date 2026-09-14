#include "lac_minorminer/search_session.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    using lac::minorminer::Graph;
    using lac::minorminer::NativeWorkCounters;
    using lac::minorminer::NativeWorkLimits;
    using lac::minorminer::SearchSession;
    using lac::minorminer::WorkBudgetExceeded;

    try {
        SearchSession session(Graph(2, {{0, 1}}), Graph(2, {{0, 1}}), 17, 4);
        const NativeWorkCounters initial = session.work_counters();
        require(initial.decisions == 0 && initial.route_expansions == 0 &&
                        initial.materializations == 0 && initial.validator_calls == 0 &&
                        initial.restart_work == 0 && initial.feature_work == 0,
                "session construction must not leak into native internal-work counters");

        static_cast<void>(session.snapshot());
        const auto after_snapshot = session.work_counters();
        // Snapshot features are two occupancy values, two chain memberships, two conflicts,
        // two missing-demand values, and five aggregate scalars.
        require(after_snapshot.validator_calls == 1 && after_snapshot.feature_work == 13,
                "snapshot validity and scalar-feature charges must be exact");

        const auto batch = session.propose(0);
        const auto after_proposal = session.work_counters();
        require(batch.candidates.size() == 1,
                "two-node route fixture must materialize one complete successor");
        require(after_proposal.route_expansions == 2,
                "router must count exactly its two settled target vertices");
        require(after_proposal.materializations == 1,
                "candidate successor must be materialized exactly once");
        // The proposal adds two route-cost scalars and one five-scalar ResourceRank.
        require(after_proposal.feature_work == 20,
                "route-cost and rank feature charges must be exact");
        session.discard(batch);
        const auto consumed = session.work_counters();
        require(consumed.decisions == 1,
                "discarding one proposal must consume exactly one decision");
        require(consumed.compiler_calls == 0 && consumed.cut_edge_visits == 0 &&
                        consumed.evaluator_reads == 0,
                "outcome-blind native search must not invent forbidden subsystem work");

        session.restart();
        require(session.work_counters().restart_work == 1,
                "one post-construction state restart must charge one restart-work unit");
        session.perturb({0});
        require(session.work_counters().restart_work == 2,
                "one stochastic partial reinitialization must charge one restart-work unit");

        auto restored = SearchSession::from_chains(
                Graph(1, {}), Graph(3, {{0, 1}, {1, 2}}), {{0}}, 19, 3);
        const auto supplied = restored->materialize(0, {{0}, {1}, {2}});
        require(supplied.candidates.size() == 3 &&
                        restored->work_counters().materializations == 3,
                "explicit candidate materialization must charge every complete successor");
        restored->discard(supplied);
        require(restored->work_counters().decisions == 1,
                "materialized proposal consumption must share the decision ledger");

        NativeWorkLimits route_limited;
        route_limited.route_expansions = 1;
        SearchSession bounded_route(Graph(2, {{0, 1}}), Graph(2, {{0, 1}}), 17, 4,
                                    route_limited);
        static_cast<void>(bounded_route.snapshot());
        try {
            static_cast<void>(bounded_route.propose(0));
            throw std::runtime_error("route work must stop before crossing its cap");
        } catch (const WorkBudgetExceeded& error) {
            require(error.coordinate() == "route_expansions",
                    "budget exception must identify the exhausted coordinate");
        }
        const auto bounded_route_work = bounded_route.work_counters();
        require(bounded_route_work.route_expansions == 1 &&
                        bounded_route_work.materializations == 0,
                "a rejected route boundary must neither exceed the cap nor materialize a successor");

        NativeWorkLimits decision_limited;
        decision_limited.decisions = 0;
        SearchSession bounded_decision(Graph(1, {}), Graph(3, {{0, 1}, {1, 2}}), 17, 2,
                                       decision_limited);
        try {
            static_cast<void>(bounded_decision.propose(0));
            throw std::runtime_error("proposal work must not start without a decision allowance");
        } catch (const WorkBudgetExceeded& error) {
            require(error.coordinate() == "decisions",
                    "decision exhaustion must be explicit");
        }
        require(bounded_decision.work_counters().route_expansions == 0 &&
                        bounded_decision.work_counters().materializations == 0,
                "decision exhaustion must stop before candidate generation");

        NativeWorkLimits snapshot_limited;
        snapshot_limited.validator_calls = 0;
        SearchSession bounded_snapshot(Graph(1, {}), Graph(1, {}), 17, 2,
                                       snapshot_limited);
        try {
            static_cast<void>(bounded_snapshot.snapshot());
            throw std::runtime_error("snapshot validation must respect its cap");
        } catch (const WorkBudgetExceeded& error) {
            require(error.coordinate() == "validator_calls",
                    "snapshot exhaustion must identify validator_calls");
        }
        require(bounded_snapshot.work_counters().validator_calls == 0 &&
                        bounded_snapshot.work_counters().feature_work == 0,
                "multi-coordinate snapshot charging must be atomic");

        NativeWorkLimits fork_limits;
        fork_limits.decisions = 1;
        SearchSession fork_source(Graph(1, {}), Graph(2, {{0, 1}}), 23, 2, fork_limits);
        const auto fork_batch = fork_source.propose(0);
        fork_source.discard(fork_batch);
        auto fork_branch = fork_source.fork();
        require(fork_branch->work_counters().decisions == 1,
                "fork must preserve the consumed work prefix");
        for (SearchSession* lineage : {&fork_source, fork_branch.get()}) {
            try {
                static_cast<void>(lineage->propose(0));
                throw std::runtime_error("forked lineages must preserve the prospective cap");
            } catch (const WorkBudgetExceeded& error) {
                require(error.coordinate() == "decisions",
                        "forked cap exhaustion must identify decisions");
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

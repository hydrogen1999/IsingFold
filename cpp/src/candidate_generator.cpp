#include "lac_minorminer/candidate.hpp"

#include "lac_minorminer/weighted_router.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace lac::minorminer {
namespace {

bool candidate_less(const Candidate& first, const Candidate& second) {
    if (first.rank < second.rank) {
        return true;
    }
    if (second.rank < first.rank) {
        return false;
    }
    return first.chain < second.chain;
}

}  // namespace

std::vector<Candidate> CandidateGenerator::propose(
        const EmbeddingState& state, Node logical, std::size_t max_candidates,
        const std::vector<double>& target_costs, NativeWorkCounters* work,
        const NativeWorkLimits* work_limits) const {
    static_cast<void>(state.chain(logical));
    if (max_candidates == 0) {
        throw std::invalid_argument("max_candidates must be positive");
    }
    const auto costs = routing_costs(state, logical, target_costs, work, work_limits);
    const auto& logical_neighbors = state.source().neighbors(logical);
    std::vector<Candidate> candidates;

    if (logical_neighbors.empty()) {
        for (std::size_t target = 0; target < state.target().num_nodes(); ++target) {
            const auto node = static_cast<Node>(target);
            // Singleton placement explicitly expands this root without running Dijkstra.
            if (work != nullptr) {
                NativeWorkCounters delta;
                delta.add_route_expansions();
                delta.add_materializations();
                delta.add_feature_work(5);
                if (work_limits == nullptr) {
                    work->add_route_expansions();
                    work->add_materializations();
                    work->add_feature_work(5);
                } else {
                    work->charge(delta, *work_limits);
                }
            }
            candidates.push_back(
                    {{node}, make_rank(state, logical, {node}, costs[target], nullptr,
                                       nullptr)});
        }
    } else {
        const WeightedRouter router(state.target());
        std::vector<RouteTree> route_trees;
        std::vector<bool> neighbor_chain_node(state.target().num_nodes(), false);
        route_trees.reserve(logical_neighbors.size());
        for (const Node neighbor : logical_neighbors) {
            const auto& sources = state.chain(neighbor);
            if (sources.empty()) {
                return {};
            }
            for (const Node source : sources) {
                neighbor_chain_node[static_cast<std::size_t>(source)] = true;
            }
            route_trees.push_back(router.run(sources, costs, work, work_limits));
        }

        std::vector<std::pair<double, Node>> roots;
        for (std::size_t root = 0; root < state.target().num_nodes(); ++root) {
            if (neighbor_chain_node[root]) {
                continue;
            }
            double total = 0.0;
            for (const auto& tree : route_trees) {
                total += tree.distance[root];
            }
            if (std::isfinite(total)) {
                roots.emplace_back(total, static_cast<Node>(root));
            }
        }
        std::sort(roots.begin(), roots.end());

        std::set<EmbeddingState::Chain> seen;
        for (const auto& [route_cost, root] : roots) {
            NativeWorkCounters materialization_delta;
            materialization_delta.add_materializations();
            if (work != nullptr && work_limits != nullptr) {
                work->require(materialization_delta, *work_limits);
            }
            EmbeddingState::Chain chain;
            bool complete = true;
            for (const auto& tree : route_trees) {
                auto path = tree.path_to_source(root);
                if (path.size() < 2) {
                    complete = false;
                    break;
                }
                path.pop_back();
                chain.insert(chain.end(), path.begin(), path.end());
            }
            if (!complete) {
                continue;
            }
            std::sort(chain.begin(), chain.end());
            chain.erase(std::unique(chain.begin(), chain.end()), chain.end());
            // Charge before deduplication: a complete successor was actually constructed.
            if (work != nullptr) {
                if (work_limits == nullptr) {
                    work->add_materializations();
                } else {
                    work->charge(materialization_delta, *work_limits);
                }
            }
            if (seen.insert(chain).second) {
                candidates.push_back(
                        {chain, make_rank(state, logical, chain, route_cost, work,
                                         work_limits)});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), candidate_less);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        candidates[index].candidate_id = static_cast<std::uint64_t>(index);
    }
    if (candidates.size() > max_candidates) {
        candidates.resize(max_candidates);
    }
    return candidates;
}

}  // namespace lac::minorminer

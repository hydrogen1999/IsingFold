#include "lac_minorminer/candidate.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace lac::minorminer {

CandidateGenerator::CandidateGenerator(RoutingConfig config) : config_(config) {
    if (!std::isfinite(config_.occupancy_base_offset) ||
        config_.occupancy_base_offset <= 0.0 || config_.occupancy_base_offset > 1'000'000.0 ||
        config_.occupancy_exponent_cap == 0 || config_.occupancy_exponent_cap > 32) {
        throw std::invalid_argument("invalid occupancy routing configuration");
    }
}

bool ResourceRank::operator<(const ResourceRank& other) const noexcept {
    return std::tie(max_occupancy, total_excess_occupancy, used_target_nodes, route_cost) <
           std::tie(other.max_occupancy, other.total_excess_occupancy,
                    other.used_target_nodes, other.route_cost);
}

std::vector<std::size_t> CandidateGenerator::released_occupancy(const EmbeddingState& state,
                                                                 Node logical) {
    auto occupancy = state.occupancy();
    for (const Node node : state.chain(logical)) {
        --occupancy[static_cast<std::size_t>(node)];
    }
    return occupancy;
}

std::vector<double> CandidateGenerator::routing_costs(
        const EmbeddingState& state, Node logical,
        const std::vector<double>& target_costs,
        NativeWorkCounters* work,
        const NativeWorkLimits* work_limits) const {
    if (!target_costs.empty()) {
        if (target_costs.size() != state.target().num_nodes()) {
            throw std::invalid_argument("target-cost vector has the wrong length");
        }
        for (const double cost : target_costs) {
            if (!std::isfinite(cost) || cost < 0.0) {
                throw std::invalid_argument("target costs must be finite and non-negative");
            }
        }
        if (work != nullptr) {
            work->add_feature_work(target_costs.size(), work_limits);
        }
        return target_costs;
    }

    if (work != nullptr) {
        work->add_feature_work(state.target().num_nodes(), work_limits);
    }
    const auto occupancy = released_occupancy(state, logical);
    const double base =
            static_cast<double>(state.target().num_nodes()) + config_.occupancy_base_offset;
    std::vector<double> costs;
    costs.reserve(occupancy.size());
    for (const auto count : occupancy) {
        costs.push_back(std::pow(
                base, static_cast<double>(std::min(count, config_.occupancy_exponent_cap))));
    }
    return costs;
}

ResourceRank CandidateGenerator::make_rank(const EmbeddingState& state, Node logical,
                                            const EmbeddingState::Chain& chain,
                                            double route_cost,
                                            NativeWorkCounters* work,
                                            const NativeWorkLimits* work_limits) {
    // ResourceRank exposes exactly five scalar features to the Python policy/scorer.
    if (work != nullptr) {
        work->add_feature_work(5, work_limits);
    }
    auto occupancy = released_occupancy(state, logical);
    for (const Node node : chain) {
        ++occupancy[static_cast<std::size_t>(node)];
    }
    ResourceRank rank;
    rank.chain_length = chain.size();
    rank.route_cost = route_cost;
    for (const auto count : occupancy) {
        rank.max_occupancy = std::max(rank.max_occupancy, count);
        rank.total_excess_occupancy += count > 1 ? count - 1 : 0;
        rank.used_target_nodes += count != 0 ? 1 : 0;
    }
    return rank;
}

}  // namespace lac::minorminer

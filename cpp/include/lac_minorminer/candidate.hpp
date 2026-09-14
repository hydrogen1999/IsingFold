#pragma once

#include "lac_minorminer/configuration.hpp"
#include "lac_minorminer/embedding_state.hpp"
#include "lac_minorminer/work_counters.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lac::minorminer {

struct ResourceRank {
    std::size_t max_occupancy = 0;
    std::size_t total_excess_occupancy = 0;
    std::size_t used_target_nodes = 0;
    std::size_t chain_length = 0;
    double route_cost = 0.0;

    [[nodiscard]] bool operator<(const ResourceRank& other) const noexcept;
};

struct Candidate {
    EmbeddingState::Chain chain;
    ResourceRank rank;
    std::uint64_t candidate_id = 0;
};

class CandidateGenerator {
public:
    using Node = Graph::Node;

    explicit CandidateGenerator(RoutingConfig config = {});

    [[nodiscard]] std::vector<Candidate> propose(
            const EmbeddingState& state, Node logical, std::size_t max_candidates,
            const std::vector<double>& target_costs = {},
            NativeWorkCounters* work = nullptr,
            const NativeWorkLimits* work_limits = nullptr) const;
    [[nodiscard]] std::vector<Candidate> materialize(
            const EmbeddingState& state, Node logical,
            const std::vector<EmbeddingState::Chain>& chains,
            NativeWorkCounters* work = nullptr,
            const NativeWorkLimits* work_limits = nullptr) const;

private:
    [[nodiscard]] static std::vector<std::size_t> released_occupancy(
            const EmbeddingState& state, Node logical);
    [[nodiscard]] std::vector<double> routing_costs(
            const EmbeddingState& state, Node logical,
            const std::vector<double>& target_costs,
            NativeWorkCounters* work,
            const NativeWorkLimits* work_limits) const;
    [[nodiscard]] static ResourceRank make_rank(const EmbeddingState& state, Node logical,
                                                const EmbeddingState::Chain& chain,
                                                double route_cost,
                                                NativeWorkCounters* work,
                                                const NativeWorkLimits* work_limits);

    RoutingConfig config_;
};

}  // namespace lac::minorminer

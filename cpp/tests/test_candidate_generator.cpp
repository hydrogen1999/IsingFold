#include "lac_minorminer/candidate.hpp"

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

}  // namespace

int main() {
    using lac::minorminer::CandidateGenerator;
    using lac::minorminer::EmbeddingState;
    using lac::minorminer::Graph;
    using lac::minorminer::ResourceRank;

    try {
        EmbeddingState triangle(Graph(3, {{0, 1}, {0, 2}, {1, 2}}),
                                Graph(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}}));
        triangle.replace_chain(0, {0});
        triangle.replace_chain(1, {1});
        triangle.replace_chain(2, {2});

        const CandidateGenerator generator;
        const ResourceRank short_chain{1, 0, 3, 1, 2.0};
        const ResourceRank long_chain{1, 0, 3, 4, 2.0};
        require(!(short_chain < long_chain) && !(long_chain < short_chain),
                "chain length must be a feature, not a hidden fifth resource criterion");
        const auto candidates = generator.propose(triangle, 0, 8);
        require(!candidates.empty(), "a routable variable must have a candidate");
        require(candidates.front().chain == std::vector<Graph::Node>({0, 3}),
                "candidate must unite root-to-neighbor paths and omit neighbor-chain endpoints");
        require(candidates.front().rank.max_occupancy == 1, "rank did not release the old chain");
        require(candidates.front().rank.total_excess_occupancy == 0,
                "conflict-free candidate must have no excess occupancy");
        require(candidates.front().rank.used_target_nodes == 4, "used-node rank is incorrect");
        require(candidates.front().rank.chain_length == 2, "chain length rank is incorrect");

        EmbeddingState isolated(Graph(1, {}), Graph(3, {{0, 1}, {1, 2}}));
        isolated.replace_chain(0, {2});
        const auto singleton = generator.propose(isolated, 0, 2, {4.0, 1.0, 2.0});
        require(singleton.size() == 2, "max_candidates must bound the batch");
        require(singleton[0].chain == std::vector<Graph::Node>({1}),
                "custom target costs must order isolated-variable candidates");
        require(singleton[1].chain == std::vector<Graph::Node>({2}),
                "custom target costs must be used deterministically");
        const auto audit = generator.propose(isolated, 0, 3, {4.0, 1.0, 2.0});
        require(audit.size() == 3, "a wider audit limit must expose pre-truncation candidates");
        for (std::size_t index = 0; index < singleton.size(); ++index) {
            require(singleton[index].candidate_id == audit[index].candidate_id,
                    "candidate identity must survive a change in truncation limit");
            require(singleton[index].chain == audit[index].chain,
                    "widening must preserve every decision-batch candidate");
        }
        require(audit[0].candidate_id != audit[1].candidate_id &&
                        audit[1].candidate_id != audit[2].candidate_id,
                "candidate IDs must be unique inside the pre-truncation batch");

        try {
            static_cast<void>(generator.propose(triangle, 0, 0));
            throw std::runtime_error("zero max_candidates must be rejected");
        } catch (const std::invalid_argument&) {
        }
        try {
            static_cast<void>(CandidateGenerator({1.0, 0}));
            throw std::runtime_error("invalid routing configuration must be rejected");
        } catch (const std::invalid_argument&) {
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}

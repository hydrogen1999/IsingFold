#include "lac_minorminer/embedding_state.hpp"

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
    using lac::minorminer::EmbeddingState;
    using lac::minorminer::Graph;

    try {
        EmbeddingState state(Graph(3, {{0, 1}}), Graph(4, {{0, 1}, {1, 2}, {2, 3}}));
        require(!state.is_valid_embedding(), "an empty state must not be a valid embedding");

        state.replace_chain(0, {1, 0});
        state.replace_chain(1, {1, 2});
        state.replace_chain(2, {3});
        require(state.chain(0) == std::vector<Graph::Node>({0, 1}), "chains must be canonical");
        require(state.max_occupancy() == 2, "maximum occupancy is stale");
        require(state.total_excess_occupancy() == 1, "excess occupancy is stale");
        require(state.used_target_nodes() == 4, "used-node count is stale");
        require(state.missing_source_edges() == 0, "realized source edge was reported missing");
        require(state.conflict_count(0) == 1, "logical conflict count is stale");
        require(!state.is_valid_embedding(), "overlapping chains must be invalid");

        const auto generation_before_rejection = state.generation();
        try {
            state.replace_chain(1, {0, 2});
            throw std::runtime_error("disconnected chains must be rejected");
        } catch (const std::invalid_argument&) {
        }
        require(state.generation() == generation_before_rejection, "a rejected edit mutated state");

        state.replace_chain(1, {2});
        require(state.max_occupancy() == 1, "overlap was not removed");
        require(state.total_excess_occupancy() == 0, "excess was not removed");
        require(state.is_valid_embedding(), "valid chains were not recognized");

        state.remove_chain(1);
        require(state.chain(1).empty(), "remove_chain did not clear the chain");
        require(state.missing_source_edges() == 1, "unembedded endpoint must leave an edge missing");
        require(!state.is_valid_embedding(), "an empty chain must make the state invalid");

        try {
            state.replace_chain(1, {1, 1});
            throw std::runtime_error("duplicate target nodes must be rejected");
        } catch (const std::invalid_argument&) {
        }
        try {
            state.replace_chain(1, {4});
            throw std::runtime_error("out-of-range target nodes must be rejected");
        } catch (const std::invalid_argument&) {
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}

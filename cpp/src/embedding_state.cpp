#include "lac_minorminer/embedding_state.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lac::minorminer {

EmbeddingState::EmbeddingState(Graph source, Graph target)
        : source_(std::move(source)),
          target_(std::move(target)),
          chains_(source_.num_nodes()),
          occupancy_(target_.num_nodes(), 0) {}

std::size_t EmbeddingState::checked_logical(Node logical) const {
    if (logical < 0 || static_cast<std::size_t>(logical) >= chains_.size()) {
        throw std::out_of_range("logical node is outside the source graph");
    }
    return static_cast<std::size_t>(logical);
}

EmbeddingState::Chain EmbeddingState::canonical_chain(Chain chain) const {
    if (chain.empty()) {
        throw std::invalid_argument("a replacement chain cannot be empty");
    }
    for (const Node node : chain) {
        if (node < 0 || static_cast<std::size_t>(node) >= target_.num_nodes()) {
            throw std::invalid_argument("chain node is outside the target graph");
        }
    }
    std::sort(chain.begin(), chain.end());
    if (std::adjacent_find(chain.begin(), chain.end()) != chain.end()) {
        throw std::invalid_argument("a chain cannot contain duplicate target nodes");
    }

    std::vector<bool> reached(target_.num_nodes(), false);
    std::vector<Node> frontier{chain.front()};
    reached[static_cast<std::size_t>(chain.front())] = true;
    std::size_t cursor = 0;
    while (cursor < frontier.size()) {
        for (const Node neighbor : target_.neighbors(frontier[cursor++])) {
            const auto index = static_cast<std::size_t>(neighbor);
            if (!reached[index] && std::binary_search(chain.begin(), chain.end(), neighbor)) {
                reached[index] = true;
                frontier.push_back(neighbor);
            }
        }
    }
    if (frontier.size() != chain.size()) {
        throw std::invalid_argument("a chain must induce a connected target subgraph");
    }
    return chain;
}

void EmbeddingState::replace_chain(Node logical, Chain chain) {
    const auto logical_index = checked_logical(logical);
    Chain replacement = canonical_chain(std::move(chain));
    if (replacement == chains_[logical_index]) {
        return;
    }
    for (const Node node : chains_[logical_index]) {
        --occupancy_[static_cast<std::size_t>(node)];
    }
    for (const Node node : replacement) {
        ++occupancy_[static_cast<std::size_t>(node)];
    }
    chains_[logical_index] = std::move(replacement);
    ++generation_;
}

void EmbeddingState::remove_chain(Node logical) {
    const auto logical_index = checked_logical(logical);
    if (chains_[logical_index].empty()) {
        return;
    }
    for (const Node node : chains_[logical_index]) {
        --occupancy_[static_cast<std::size_t>(node)];
    }
    chains_[logical_index].clear();
    ++generation_;
}

const Graph& EmbeddingState::source() const noexcept { return source_; }
const Graph& EmbeddingState::target() const noexcept { return target_; }

const EmbeddingState::Chain& EmbeddingState::chain(Node logical) const {
    return chains_[checked_logical(logical)];
}

const std::vector<std::size_t>& EmbeddingState::occupancy() const noexcept { return occupancy_; }
std::uint64_t EmbeddingState::generation() const noexcept { return generation_; }

}  // namespace lac::minorminer

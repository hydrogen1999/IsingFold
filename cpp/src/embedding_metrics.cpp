#include "lac_minorminer/embedding_state.hpp"

#include <algorithm>

namespace lac::minorminer {

std::size_t EmbeddingState::max_occupancy() const noexcept {
    if (occupancy_.empty()) {
        return 0;
    }
    return *std::max_element(occupancy_.begin(), occupancy_.end());
}

std::size_t EmbeddingState::total_excess_occupancy() const noexcept {
    std::size_t total = 0;
    for (const auto count : occupancy_) {
        total += count > 1 ? count - 1 : 0;
    }
    return total;
}

std::size_t EmbeddingState::used_target_nodes() const noexcept {
    return static_cast<std::size_t>(
            std::count_if(occupancy_.begin(), occupancy_.end(), [](const auto count) {
                return count != 0;
            }));
}

bool EmbeddingState::source_edge_realized(Node first, Node second) const {
    const auto& first_chain = chain(first);
    const auto& second_chain = chain(second);
    for (const Node first_target : first_chain) {
        for (const Node second_target : second_chain) {
            if (first_target != second_target && target_.adjacent(first_target, second_target)) {
                return true;
            }
        }
    }
    return false;
}

std::size_t EmbeddingState::missing_source_edges() const {
    std::size_t missing = 0;
    for (std::size_t first = 0; first < source_.num_nodes(); ++first) {
        for (const Node second : source_.neighbors(static_cast<Node>(first))) {
            if (first < static_cast<std::size_t>(second) &&
                !source_edge_realized(static_cast<Node>(first), second)) {
                ++missing;
            }
        }
    }
    return missing;
}

std::size_t EmbeddingState::conflict_count(Node logical) const {
    std::size_t conflicts = 0;
    for (const Node target : chain(logical)) {
        const auto count = occupancy_[static_cast<std::size_t>(target)];
        conflicts += count > 1 ? count - 1 : 0;
    }
    return conflicts;
}

std::size_t EmbeddingState::missing_incident_edges(Node logical) const {
    static_cast<void>(checked_logical(logical));
    return static_cast<std::size_t>(
            std::count_if(source_.neighbors(logical).begin(), source_.neighbors(logical).end(),
                          [this, logical](const Node neighbor) {
                              return !source_edge_realized(logical, neighbor);
                          }));
}

bool EmbeddingState::is_valid_embedding() const {
    if (std::any_of(chains_.begin(), chains_.end(), [](const auto& value) { return value.empty(); })) {
        return false;
    }
    return max_occupancy() <= 1 && missing_source_edges() == 0;
}

}  // namespace lac::minorminer

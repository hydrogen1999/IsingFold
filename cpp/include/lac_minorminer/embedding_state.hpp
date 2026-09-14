#pragma once

#include "lac_minorminer/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lac::minorminer {

class EmbeddingState {
public:
    using Node = Graph::Node;
    using Chain = std::vector<Node>;

    EmbeddingState(Graph source, Graph target);

    void replace_chain(Node logical, Chain chain);
    void remove_chain(Node logical);

    [[nodiscard]] const Graph& source() const noexcept;
    [[nodiscard]] const Graph& target() const noexcept;
    [[nodiscard]] const Chain& chain(Node logical) const;
    [[nodiscard]] const std::vector<std::size_t>& occupancy() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;

    [[nodiscard]] std::size_t max_occupancy() const noexcept;
    [[nodiscard]] std::size_t total_excess_occupancy() const noexcept;
    [[nodiscard]] std::size_t used_target_nodes() const noexcept;
    [[nodiscard]] std::size_t missing_source_edges() const;
    [[nodiscard]] std::size_t conflict_count(Node logical) const;
    [[nodiscard]] std::size_t missing_incident_edges(Node logical) const;
    [[nodiscard]] bool is_valid_embedding() const;

private:
    [[nodiscard]] std::size_t checked_logical(Node logical) const;
    [[nodiscard]] Chain canonical_chain(Chain chain) const;
    [[nodiscard]] bool source_edge_realized(Node first, Node second) const;

    Graph source_;
    Graph target_;
    std::vector<Chain> chains_;
    std::vector<std::size_t> occupancy_;
    std::uint64_t generation_ = 0;
};

}  // namespace lac::minorminer

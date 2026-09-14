#include "lac_minorminer/graph.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace lac::minorminer {
namespace {

std::size_t checked_node_count(std::size_t num_nodes) {
    if (num_nodes > static_cast<std::size_t>(std::numeric_limits<Graph::Node>::max())) {
        throw std::invalid_argument("graph has too many nodes for the compact node type");
    }
    return num_nodes;
}

}  // namespace

Graph::Graph(std::size_t num_nodes, std::vector<Edge> edges)
        : adjacency_(checked_node_count(num_nodes)) {
    for (auto& [first, second] : edges) {
        if (first < 0 || second < 0 || static_cast<std::size_t>(first) >= num_nodes ||
            static_cast<std::size_t>(second) >= num_nodes) {
            throw std::invalid_argument("edge endpoint is outside the graph");
        }
        if (first == second) {
            throw std::invalid_argument("self-loops are not supported");
        }
        if (second < first) {
            std::swap(first, second);
        }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    num_edges_ = edges.size();

    for (const auto& [first, second] : edges) {
        adjacency_[static_cast<std::size_t>(first)].push_back(second);
        adjacency_[static_cast<std::size_t>(second)].push_back(first);
    }
    for (auto& neighbors : adjacency_) {
        std::sort(neighbors.begin(), neighbors.end());
    }
}

std::size_t Graph::num_nodes() const noexcept { return adjacency_.size(); }

std::size_t Graph::num_edges() const noexcept { return num_edges_; }

std::size_t Graph::checked_index(Node node) const {
    if (node < 0 || static_cast<std::size_t>(node) >= adjacency_.size()) {
        throw std::out_of_range("node is outside the graph");
    }
    return static_cast<std::size_t>(node);
}

const std::vector<Graph::Node>& Graph::neighbors(Node node) const {
    return adjacency_[checked_index(node)];
}

bool Graph::adjacent(Node first, Node second) const {
    const auto& first_neighbors = neighbors(first);
    static_cast<void>(checked_index(second));
    return std::binary_search(first_neighbors.begin(), first_neighbors.end(), second);
}

}  // namespace lac::minorminer

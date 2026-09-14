#include "lac_minorminer/graph.hpp"

#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_invalid_argument(const std::function<void()>& operation, const std::string& message) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    using lac::minorminer::Graph;

    try {
        const Graph graph(4, {{1, 0}, {0, 1}, {3, 2}});
        require(graph.num_nodes() == 4, "node count must preserve isolates");
        require(graph.num_edges() == 2, "undirected duplicates must be canonicalized");
        require(graph.neighbors(0) == std::vector<Graph::Node>{1}, "adjacency must be sorted");
        require(graph.neighbors(2) == std::vector<Graph::Node>{3}, "edge must be symmetric");
        require(graph.neighbors(3) == std::vector<Graph::Node>{2}, "edge must be symmetric");
        require(graph.neighbors(1) == std::vector<Graph::Node>{0}, "edge must be symmetric");
        require(!graph.adjacent(0, 3), "non-edge reported as an edge");
        require(graph.adjacent(0, 1), "edge reported as a non-edge");

        require_invalid_argument([] { Graph(2, {{0, 0}}); }, "self-loops must be rejected");
        require_invalid_argument([] { Graph(2, {{0, 2}}); }, "out-of-range nodes must be rejected");
        require_invalid_argument([] { Graph(2, {{-1, 1}}); }, "negative nodes must be rejected");
        require_invalid_argument(
                [] {
                    Graph(static_cast<std::size_t>(std::numeric_limits<Graph::Node>::max()) + 1,
                          {});
                },
                "oversized compact graphs must be rejected before allocation");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}

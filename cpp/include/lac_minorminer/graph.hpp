#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace lac::minorminer {

class Graph {
public:
    using Node = std::int32_t;
    using Edge = std::pair<Node, Node>;

    Graph(std::size_t num_nodes, std::vector<Edge> edges);

    [[nodiscard]] std::size_t num_nodes() const noexcept;
    [[nodiscard]] std::size_t num_edges() const noexcept;
    [[nodiscard]] const std::vector<Node>& neighbors(Node node) const;
    [[nodiscard]] bool adjacent(Node first, Node second) const;

private:
    [[nodiscard]] std::size_t checked_index(Node node) const;

    std::vector<std::vector<Node>> adjacency_;
    std::size_t num_edges_ = 0;
};

}  // namespace lac::minorminer

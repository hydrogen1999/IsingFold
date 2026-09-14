#pragma once

#include "lac_minorminer/graph.hpp"
#include "lac_minorminer/work_counters.hpp"

#include <cstdint>
#include <vector>

namespace lac::minorminer {

struct RouteTree {
    using Node = Graph::Node;

    std::vector<double> distance;
    std::vector<Node> parent;
    std::vector<std::uint8_t> source;

    [[nodiscard]] std::vector<Node> path_to_source(Node root) const;
};

class WeightedRouter {
public:
    using Node = Graph::Node;

    explicit WeightedRouter(const Graph& graph) noexcept;
    [[nodiscard]] RouteTree run(std::vector<Node> sources,
                                const std::vector<double>& node_costs,
                                NativeWorkCounters* work = nullptr,
                                const NativeWorkLimits* work_limits = nullptr) const;

private:
    const Graph& graph_;
};

}  // namespace lac::minorminer

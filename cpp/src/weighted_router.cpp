#include "lac_minorminer/weighted_router.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace lac::minorminer {

std::vector<RouteTree::Node> RouteTree::path_to_source(Node root) const {
    if (root < 0 || static_cast<std::size_t>(root) >= distance.size()) {
        throw std::out_of_range("route root is outside the graph");
    }
    if (!std::isfinite(distance[static_cast<std::size_t>(root)])) {
        return {};
    }

    std::vector<Node> path;
    Node current = root;
    while (path.size() <= parent.size()) {
        path.push_back(current);
        const auto index = static_cast<std::size_t>(current);
        if (source[index]) {
            return path;
        }
        current = parent[index];
        if (current < 0) {
            return {};
        }
    }
    throw std::logic_error("route parent relation contains a cycle");
}

WeightedRouter::WeightedRouter(const Graph& graph) noexcept : graph_(graph) {}

RouteTree WeightedRouter::run(std::vector<Node> sources,
                              const std::vector<double>& node_costs,
                              NativeWorkCounters* work,
                              const NativeWorkLimits* work_limits) const {
    const auto num_nodes = graph_.num_nodes();
    if (sources.empty()) {
        throw std::invalid_argument("routing requires at least one source");
    }
    if (node_costs.size() != num_nodes) {
        throw std::invalid_argument("target-cost vector has the wrong length");
    }
    for (const double cost : node_costs) {
        if (!std::isfinite(cost) || cost < 0.0) {
            throw std::invalid_argument("target costs must be finite and non-negative");
        }
    }

    std::sort(sources.begin(), sources.end());
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
    RouteTree result{
            std::vector<double>(num_nodes, std::numeric_limits<double>::infinity()),
            std::vector<Node>(num_nodes, -1),
            std::vector<std::uint8_t>(num_nodes, 0),
    };
    using QueueEntry = std::pair<double, Node>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> queue;
    for (const Node source : sources) {
        static_cast<void>(graph_.neighbors(source));
        const auto index = static_cast<std::size_t>(source);
        result.distance[index] = 0.0;
        result.source[index] = 1;
        queue.emplace(0.0, source);
    }

    while (!queue.empty()) {
        const auto [distance, current] = queue.top();
        queue.pop();
        if (distance != result.distance[static_cast<std::size_t>(current)]) {
            continue;
        }
        // One expansion is one non-stale queue entry whose outgoing adjacency is examined.
        if (work != nullptr) {
            work->add_route_expansions(1, work_limits);
        }
        for (const Node neighbor : graph_.neighbors(current)) {
            const auto index = static_cast<std::size_t>(neighbor);
            if (result.source[index]) {
                continue;
            }
            const double candidate = distance + node_costs[index];
            if (candidate < result.distance[index]) {
                result.distance[index] = candidate;
                result.parent[index] = current;
                queue.emplace(candidate, neighbor);
            }
        }
    }
    return result;
}

}  // namespace lac::minorminer

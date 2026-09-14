#include "lac_minorminer/weighted_router.hpp"

#include <cmath>
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

}  // namespace

int main() {
    using lac::minorminer::Graph;
    using lac::minorminer::WeightedRouter;

    try {
        const Graph diamond(5, {{0, 1}, {0, 2}, {1, 3}, {2, 3}});
        const WeightedRouter router(diamond);

        const auto equal = router.run({3}, {1.0, 1.0, 1.0, 1.0, 1.0});
        require(equal.distance[0] == 2.0, "distance must charge entered non-source nodes");
        require(equal.path_to_source(0) == std::vector<Graph::Node>({0, 1, 3}),
                "equal-cost paths must use deterministic node-ID tie breaking");
        require(equal.path_to_source(3) == std::vector<Graph::Node>({3}),
                "a source must route to itself");
        require(std::isinf(equal.distance[4]), "unreachable nodes must keep infinite distance");
        require(equal.path_to_source(4).empty(), "an unreachable node must have no path");

        const auto weighted = router.run({3}, {1.0, 5.0, 1.0, 1.0, 1.0});
        require(weighted.path_to_source(0) == std::vector<Graph::Node>({0, 2, 3}),
                "routing must respect caller-supplied target costs");

        const auto multisource = router.run({3, 0}, {1.0, 1.0, 1.0, 1.0, 1.0});
        require(multisource.path_to_source(1) == std::vector<Graph::Node>({1, 0}),
                "multi-source ties must be deterministic");

        const Graph zero_path(4, {{3, 2}, {2, 1}});
        const auto zero_cost = WeightedRouter(zero_path).run({3}, {0.0, 0.0, 0.0, 0.0});
        require(zero_cost.path_to_source(1) == std::vector<Graph::Node>({1, 2, 3}),
                "zero-cost ties must not create a parent cycle");

        try {
            static_cast<void>(router.run({3}, {1.0, -1.0, 1.0, 1.0, 1.0}));
            throw std::runtime_error("negative costs must be rejected");
        } catch (const std::invalid_argument&) {
        }
        try {
            static_cast<void>(router.run({}, {1.0, 1.0, 1.0, 1.0, 1.0}));
            throw std::runtime_error("empty source sets must be rejected");
        } catch (const std::invalid_argument&) {
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}

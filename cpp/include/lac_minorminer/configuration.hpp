#pragma once

#include <cstddef>

namespace lac::minorminer {

struct RoutingConfig {
    double occupancy_base_offset = 1.0;
    std::size_t occupancy_exponent_cap = 16;
};

}  // namespace lac::minorminer

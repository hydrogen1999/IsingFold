#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace lac::minorminer {

inline constexpr const char* kWorkCounterSchema = "lac-minorminer.native-work";
inline constexpr std::uint32_t kWorkCounterVersion = 3;

struct NativeWorkLimits {
    std::uint64_t decisions = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t route_expansions = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t materializations = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t compiler_calls = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t validator_calls = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t cut_edge_visits = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t restart_work = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t evaluator_reads = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t feature_work = std::numeric_limits<std::uint64_t>::max();
};

class WorkBudgetExceeded : public std::runtime_error {
public:
    explicit WorkBudgetExceeded(std::string coordinate)
            : std::runtime_error("native work budget exhausted: " + coordinate),
              coordinate_(std::move(coordinate)) {}

    [[nodiscard]] const std::string& coordinate() const noexcept { return coordinate_; }

private:
    std::string coordinate_;
};

/** Exact semantic-operation ledger for one native search-session lineage.
 *
 * These are registered algorithmic events, not timing estimates or CPU-instruction
 * approximations.  A route expansion is one non-stale priority-queue vertex settlement
 * (or one explicitly enumerated singleton root).  A materialization is one complete
 * successor chain constructed before truncation, including a duplicate later removed.
 * A validator call is one top-level embedding predicate invocation.  Feature work counts
 * registered scalar predicates and ranks.  The v3 wrapper additionally counts lane-component
 * vertex settlements as route expansions, complete structural L-chains as materializations,
 * and topology, compatibility, and bounded clique-search predicates as feature work.  Restart
 * work counts stochastic state-reinitialization calls after construction.
 *
 * The native initializer contains no Ising compiler, cut generator, or outcome evaluator,
 * so compiler_calls, cut_edge_visits, and evaluator_reads remain exactly zero.  Their
 * presence here is intentional: all nine publication ledger coordinates are always
 * returned, and unsupported work can never be silently represented as a convenient zero.
 */
struct NativeWorkCounters {
    std::uint64_t decisions = 0;
    std::uint64_t route_expansions = 0;
    std::uint64_t materializations = 0;
    std::uint64_t compiler_calls = 0;
    std::uint64_t validator_calls = 0;
    std::uint64_t cut_edge_visits = 0;
    std::uint64_t restart_work = 0;
    std::uint64_t evaluator_reads = 0;
    std::uint64_t feature_work = 0;

    void add_decisions(std::uint64_t amount = 1,
                       const NativeWorkLimits* limits = nullptr) {
        charge_coordinate(decisions, amount, limits == nullptr ? nullptr : &limits->decisions,
                          "decisions");
    }
    void add_route_expansions(std::uint64_t amount = 1,
                              const NativeWorkLimits* limits = nullptr) {
        charge_coordinate(route_expansions, amount,
                          limits == nullptr ? nullptr : &limits->route_expansions,
                          "route_expansions");
    }
    void add_materializations(std::uint64_t amount = 1,
                              const NativeWorkLimits* limits = nullptr) {
        charge_coordinate(materializations, amount,
                          limits == nullptr ? nullptr : &limits->materializations,
                          "materializations");
    }
    void add_validator_calls(std::uint64_t amount = 1,
                             const NativeWorkLimits* limits = nullptr) {
        charge_coordinate(validator_calls, amount,
                          limits == nullptr ? nullptr : &limits->validator_calls,
                          "validator_calls");
    }
    void add_restart_work(std::uint64_t amount = 1,
                          const NativeWorkLimits* limits = nullptr) {
        charge_coordinate(restart_work, amount,
                          limits == nullptr ? nullptr : &limits->restart_work,
                          "restart_work");
    }
    void add_feature_work(std::uint64_t amount,
                          const NativeWorkLimits* limits = nullptr) {
        charge_coordinate(feature_work, amount,
                          limits == nullptr ? nullptr : &limits->feature_work,
                          "feature_work");
    }

    void require(const NativeWorkCounters& delta, const NativeWorkLimits& limits) const {
        require_coordinate(decisions, delta.decisions, limits.decisions, "decisions");
        require_coordinate(route_expansions, delta.route_expansions,
                           limits.route_expansions, "route_expansions");
        require_coordinate(materializations, delta.materializations,
                           limits.materializations, "materializations");
        require_coordinate(compiler_calls, delta.compiler_calls,
                           limits.compiler_calls, "compiler_calls");
        require_coordinate(validator_calls, delta.validator_calls,
                           limits.validator_calls, "validator_calls");
        require_coordinate(cut_edge_visits, delta.cut_edge_visits,
                           limits.cut_edge_visits, "cut_edge_visits");
        require_coordinate(restart_work, delta.restart_work,
                           limits.restart_work, "restart_work");
        require_coordinate(evaluator_reads, delta.evaluator_reads,
                           limits.evaluator_reads, "evaluator_reads");
        require_coordinate(feature_work, delta.feature_work,
                           limits.feature_work, "feature_work");
    }

    void charge(const NativeWorkCounters& delta, const NativeWorkLimits& limits) {
        require(delta, limits);
        checked_add(decisions, delta.decisions);
        checked_add(route_expansions, delta.route_expansions);
        checked_add(materializations, delta.materializations);
        checked_add(compiler_calls, delta.compiler_calls);
        checked_add(validator_calls, delta.validator_calls);
        checked_add(cut_edge_visits, delta.cut_edge_visits);
        checked_add(restart_work, delta.restart_work);
        checked_add(evaluator_reads, delta.evaluator_reads);
        checked_add(feature_work, delta.feature_work);
    }

private:
    static void charge_coordinate(std::uint64_t& coordinate, std::uint64_t amount,
                                  const std::uint64_t* limit, const char* name) {
        if (limit != nullptr) {
            require_coordinate(coordinate, amount, *limit, name);
        }
        checked_add(coordinate, amount);
    }

    static void require_coordinate(std::uint64_t current, std::uint64_t amount,
                                   std::uint64_t limit, const char* coordinate) {
        if (current > limit || amount > limit - current) {
            throw WorkBudgetExceeded(coordinate);
        }
    }

    static void checked_add(std::uint64_t& coordinate, std::uint64_t amount) {
        if (amount > std::numeric_limits<std::uint64_t>::max() - coordinate) {
            throw std::overflow_error("native work counter overflow");
        }
        coordinate += amount;
    }
};

}  // namespace lac::minorminer

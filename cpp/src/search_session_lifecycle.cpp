#include "lac_minorminer/search_session.hpp"

#include <algorithm>
#include <atomic>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace lac::minorminer {
namespace {

std::uint64_t allocate_session_id() {
    static std::atomic<std::uint64_t> next_id{1};
    return next_id.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

SearchSession::SearchSession(Graph source, Graph target, std::uint64_t random_seed,
                             std::size_t max_candidates,
                             NativeWorkLimits work_limits)
        : state_(std::move(source), std::move(target)),
          random_(random_seed),
          max_candidates_(max_candidates),
          session_id_(allocate_session_id()),
          work_limits_(work_limits) {
    if (max_candidates_ == 0) {
        throw std::invalid_argument("max_candidates must be positive");
    }
    restart();
}

std::unique_ptr<SearchSession> SearchSession::from_chains(
        Graph source, Graph target,
        const std::vector<EmbeddingState::Chain>& chains,
        std::uint64_t random_seed, std::size_t max_candidates,
        NativeWorkLimits work_limits) {
    auto session = std::make_unique<SearchSession>(std::move(source), std::move(target), 0,
                                                   max_candidates, work_limits);
    if (chains.size() != session->state_.source().num_nodes()) {
        throw std::invalid_argument("from_chains requires exactly one chain per logical node");
    }
    EmbeddingState restored(session->state_.source(), session->state_.target());
    for (std::size_t logical = 0; logical < chains.size(); ++logical) {
        restored.replace_chain(static_cast<Node>(logical), chains[logical]);
    }
    session->state_ = std::move(restored);
    session->random_.seed(random_seed);
    session->generation_ = 0;
    session->next_token_ = 1;
    session->outstanding_.reset();
    return session;
}

StateSnapshot SearchSession::snapshot() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    NativeWorkCounters delta;
    delta.add_validator_calls();
    delta.add_feature_work(static_cast<std::uint64_t>(state_.target().num_nodes()));
    delta.add_feature_work(
            2 * static_cast<std::uint64_t>(state_.source().num_nodes()));
    delta.add_feature_work(5);
    for (std::size_t logical = 0; logical < state_.source().num_nodes(); ++logical) {
        delta.add_feature_work(static_cast<std::uint64_t>(
                state_.chain(static_cast<Node>(logical)).size()));
    }
    work_.charge(delta, work_limits_);
    StateSnapshot result;
    result.generation = generation_;
    result.occupancy = state_.occupancy();
    result.max_occupancy = state_.max_occupancy();
    result.total_excess_occupancy = state_.total_excess_occupancy();
    result.used_target_nodes = state_.used_target_nodes();
    result.missing_source_edges = state_.missing_source_edges();
    result.valid = state_.is_valid_embedding();
    result.chains.reserve(state_.source().num_nodes());
    result.conflicts.reserve(state_.source().num_nodes());
    result.missing_incident_edges.reserve(state_.source().num_nodes());
    for (std::size_t logical = 0; logical < state_.source().num_nodes(); ++logical) {
        const auto node = static_cast<Node>(logical);
        result.chains.push_back(state_.chain(node));
        result.conflicts.push_back(state_.conflict_count(node));
        result.missing_incident_edges.push_back(state_.missing_incident_edges(node));
    }
    return result;
}

NativeWorkCounters SearchSession::work_counters() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return work_;
}

std::unique_ptr<SearchSession> SearchSession::fork() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return fork_locked();
}

std::unique_ptr<SearchSession> SearchSession::fork_with_seed(
        std::uint64_t random_seed) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto branch = fork_locked();
    branch->random_.seed(random_seed);
    return branch;
}

std::unique_ptr<SearchSession> SearchSession::fork_locked() const {
    if (outstanding_) {
        throw std::logic_error("cannot fork a session with an outstanding proposal");
    }
    auto branch = std::make_unique<SearchSession>(state_.source(), state_.target(), 0,
                                                  max_candidates_, work_limits_);
    branch->state_ = state_;
    branch->random_ = random_;
    branch->generation_ = generation_;
    branch->next_token_ = next_token_;
    branch->initial_assignment_complete_ = initial_assignment_complete_;
    branch->outstanding_.reset();
    branch->work_ = work_;
    return branch;
}

std::vector<SearchSession::Node> SearchSession::eligible_variables() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Node> eligible;
    for (std::size_t logical = 0; logical < state_.source().num_nodes(); ++logical) {
        const auto node = static_cast<Node>(logical);
        if (state_.chain(node).empty() || state_.conflict_count(node) != 0 ||
            state_.missing_incident_edges(node) != 0) {
            eligible.push_back(node);
        }
    }
    return eligible;
}

void SearchSession::perturb(const std::vector<Node>& logicals) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (logicals.empty()) {
        throw std::invalid_argument("a perturbation neighborhood cannot be empty");
    }
    if (state_.target().num_nodes() == 0) {
        throw std::invalid_argument("cannot perturb chains into an empty target graph");
    }

    std::vector<bool> selected(state_.source().num_nodes(), false);
    for (const Node logical : logicals) {
        static_cast<void>(state_.chain(logical));
        const auto index = static_cast<std::size_t>(logical);
        if (selected[index]) {
            throw std::invalid_argument("a perturbation neighborhood cannot contain duplicates");
        }
        selected[index] = true;
    }

    NativeWorkCounters delta;
    delta.add_restart_work();
    work_.charge(delta, work_limits_);

    EmbeddingState perturbed = state_;
    for (const Node logical : logicals) {
        perturbed.remove_chain(logical);
    }

    std::vector<Node> targets(perturbed.target().num_nodes());
    std::iota(targets.begin(), targets.end(), 0);
    std::shuffle(targets.begin(), targets.end(), random_);
    const auto& occupancy = perturbed.occupancy();
    std::stable_sort(targets.begin(), targets.end(), [&occupancy](Node first, Node second) {
        return occupancy[static_cast<std::size_t>(first)] <
               occupancy[static_cast<std::size_t>(second)];
    });

    for (std::size_t index = 0; index < logicals.size(); ++index) {
        perturbed.replace_chain(logicals[index], {targets[index % targets.size()]});
    }
    state_ = std::move(perturbed);
    outstanding_.reset();
    ++generation_;
}

void SearchSession::restart() {
    const std::lock_guard<std::mutex> lock(mutex_);
    const bool charge_internal_restart = initial_assignment_complete_;
    if (charge_internal_restart) {
        NativeWorkCounters delta;
        delta.add_restart_work();
        work_.charge(delta, work_limits_);
    }
    EmbeddingState fresh(state_.source(), state_.target());
    if (state_.target().num_nodes() != 0) {
        std::uniform_int_distribution<std::size_t> target_node(0,
                                                               state_.target().num_nodes() - 1);
        for (std::size_t logical = 0; logical < state_.source().num_nodes(); ++logical) {
            fresh.replace_chain(static_cast<Node>(logical),
                                {static_cast<Node>(target_node(random_))});
        }
    }
    state_ = std::move(fresh);
    outstanding_.reset();
    ++generation_;
    initial_assignment_complete_ = true;
}

}  // namespace lac::minorminer

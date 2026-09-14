#pragma once

#include "lac_minorminer/candidate.hpp"
#include "lac_minorminer/work_counters.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <vector>

namespace lac::minorminer {

struct StateSnapshot {
    std::uint64_t generation = 0;
    std::vector<EmbeddingState::Chain> chains;
    std::vector<std::size_t> occupancy;
    std::vector<std::size_t> conflicts;
    std::vector<std::size_t> missing_incident_edges;
    std::size_t max_occupancy = 0;
    std::size_t total_excess_occupancy = 0;
    std::size_t used_target_nodes = 0;
    std::size_t missing_source_edges = 0;
    bool valid = false;
};

struct CandidateBatch {
    std::uint64_t session_id = 0;
    std::uint64_t generation = 0;
    std::uint64_t token = 0;
    Graph::Node logical = -1;
    std::vector<Candidate> candidates;
};

struct AuditedCandidateBatches {
    CandidateBatch audit_batch;
    CandidateBatch decision_batch;
};

class SearchSession {
public:
    using Node = Graph::Node;

    SearchSession(Graph source, Graph target, std::uint64_t random_seed,
                  std::size_t max_candidates,
                  NativeWorkLimits work_limits = {});

    [[nodiscard]] static std::unique_ptr<SearchSession> from_chains(
            Graph source, Graph target,
            const std::vector<EmbeddingState::Chain>& chains,
            std::uint64_t random_seed, std::size_t max_candidates,
            NativeWorkLimits work_limits = {});

    [[nodiscard]] StateSnapshot snapshot() const;
    [[nodiscard]] NativeWorkCounters work_counters() const;
    [[nodiscard]] std::uint64_t session_id() const noexcept { return session_id_; }
    [[nodiscard]] std::unique_ptr<SearchSession> fork() const;
    [[nodiscard]] std::unique_ptr<SearchSession> fork_with_seed(std::uint64_t random_seed) const;
    [[nodiscard]] std::vector<Node> eligible_variables() const;
    [[nodiscard]] CandidateBatch propose(Node logical,
                                         const std::vector<double>& target_costs = {});
    [[nodiscard]] CandidateBatch propose_applicable(
            Node logical, std::size_t scoring_candidates,
            const std::vector<double>& target_costs = {});
    [[nodiscard]] AuditedCandidateBatches propose_with_audit(
            Node logical, std::size_t audit_candidates,
            const std::vector<double>& target_costs = {});
    [[nodiscard]] CandidateBatch materialize(
            Node logical, const std::vector<EmbeddingState::Chain>& chains);
    [[nodiscard]] bool candidate_is_valid(const CandidateBatch& batch,
                                          std::size_t index) const;
    void apply(const CandidateBatch& batch, std::size_t index);
    void discard(const CandidateBatch& batch);
    void perturb(const std::vector<Node>& logicals);
    void restart();

private:
    [[nodiscard]] std::unique_ptr<SearchSession> fork_locked() const;
    [[nodiscard]] CandidateBatch issue(Node logical, std::vector<Candidate> candidates);
    void validate_handle(const CandidateBatch& batch) const;

    EmbeddingState state_;
    std::mt19937_64 random_;
    std::size_t max_candidates_;
    std::uint64_t session_id_;
    std::uint64_t generation_ = 0;
    std::uint64_t next_token_ = 1;
    bool initial_assignment_complete_ = false;
    std::optional<CandidateBatch> outstanding_;
    CandidateGenerator generator_;
    NativeWorkLimits work_limits_;
    mutable NativeWorkCounters work_;
    mutable std::mutex mutex_;
};

}  // namespace lac::minorminer

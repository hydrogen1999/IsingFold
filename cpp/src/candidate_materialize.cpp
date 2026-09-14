#include "lac_minorminer/candidate.hpp"

#include <set>
#include <stdexcept>

namespace lac::minorminer {

std::vector<Candidate> CandidateGenerator::materialize(
        const EmbeddingState& state, Node logical,
        const std::vector<EmbeddingState::Chain>& chains,
        NativeWorkCounters* work,
        const NativeWorkLimits* work_limits) const {
    static_cast<void>(state.chain(logical));
    std::set<EmbeddingState::Chain> seen;
    std::vector<Candidate> candidates;
    candidates.reserve(chains.size());
    for (const auto& supplied_chain : chains) {
        NativeWorkCounters materialization_delta;
        materialization_delta.add_materializations();
        if (work != nullptr && work_limits != nullptr) {
            work->require(materialization_delta, *work_limits);
        }
        EmbeddingState trial = state;
        trial.replace_chain(logical, supplied_chain);
        if (trial.missing_incident_edges(logical) != 0) {
            throw std::invalid_argument(
                    "a supplied chain does not connect to every placed logical neighbor");
        }
        const auto& chain = trial.chain(logical);
        // The successor exists before duplicate rejection, so rejected duplicates are charged.
        if (work != nullptr) {
            work->add_materializations(1, work_limits);
        }
        if (!seen.insert(chain).second) {
            throw std::invalid_argument("supplied candidate chains must be unique");
        }
        candidates.push_back(
                {chain, make_rank(state, logical, chain, 0.0, work, work_limits),
                 candidates.size()});
    }
    return candidates;
}

}  // namespace lac::minorminer

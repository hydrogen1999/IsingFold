#include "lac_minorminer/search_session.hpp"

#include <stdexcept>
#include <utility>

namespace lac::minorminer {

CandidateBatch SearchSession::issue(Node logical, std::vector<Candidate> candidates) {
    if (outstanding_) {
        throw std::logic_error("discard or apply the outstanding proposal first");
    }
    CandidateBatch batch{session_id_, generation_, next_token_++, logical,
                         std::move(candidates)};
    outstanding_ = batch;
    return batch;
}

CandidateBatch SearchSession::propose(Node logical, const std::vector<double>& target_costs) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (outstanding_) {
        throw std::logic_error("discard or apply the outstanding proposal first");
    }
    NativeWorkCounters decision_delta;
    decision_delta.add_decisions();
    work_.require(decision_delta, work_limits_);
    return issue(logical,
                 generator_.propose(state_, logical, max_candidates_, target_costs,
                                    &work_, &work_limits_));
}

CandidateBatch SearchSession::propose_applicable(
        Node logical, std::size_t scoring_candidates,
        const std::vector<double>& target_costs) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (outstanding_) {
        throw std::logic_error("discard or apply the outstanding proposal first");
    }
    if (scoring_candidates < max_candidates_) {
        throw std::invalid_argument(
                "scoring candidate bound cannot be smaller than the decision bound");
    }
    NativeWorkCounters decision_delta;
    decision_delta.add_decisions();
    work_.require(decision_delta, work_limits_);
    return issue(
            logical,
            generator_.propose(state_, logical, scoring_candidates, target_costs,
                               &work_, &work_limits_));
}

AuditedCandidateBatches SearchSession::propose_with_audit(
        Node logical, std::size_t audit_candidates,
        const std::vector<double>& target_costs) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (outstanding_) {
        throw std::logic_error("discard or apply the outstanding proposal first");
    }
    if (audit_candidates < max_candidates_) {
        throw std::invalid_argument(
                "audit candidate bound cannot be smaller than the decision bound");
    }
    NativeWorkCounters decision_delta;
    decision_delta.add_decisions();
    work_.require(decision_delta, work_limits_);
    auto audit = generator_.propose(state_, logical, audit_candidates, target_costs,
                                     &work_, &work_limits_);
    auto decision = audit;
    if (decision.size() > max_candidates_) {
        decision.resize(max_candidates_);
    }
    CandidateBatch audit_batch{session_id_, generation_, 0, logical, std::move(audit)};
    auto decision_batch = issue(logical, std::move(decision));
    return {std::move(audit_batch), std::move(decision_batch)};
}

CandidateBatch SearchSession::materialize(
        Node logical, const std::vector<EmbeddingState::Chain>& chains) {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (outstanding_) {
        throw std::logic_error("discard or apply the outstanding proposal first");
    }
    if (chains.size() > max_candidates_) {
        throw std::invalid_argument("supplied chains exceed the session candidate bound");
    }
    NativeWorkCounters decision_delta;
    decision_delta.add_decisions();
    work_.require(decision_delta, work_limits_);
    return issue(logical, generator_.materialize(state_, logical, chains,
                                                  &work_, &work_limits_));
}

bool SearchSession::candidate_is_valid(const CandidateBatch& batch,
                                       std::size_t index) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    validate_handle(batch);
    if (index >= outstanding_->candidates.size()) {
        throw std::out_of_range("candidate index is outside the proposal batch");
    }
    NativeWorkCounters delta;
    delta.add_validator_calls();
    work_.charge(delta, work_limits_);
    EmbeddingState trial = state_;
    trial.replace_chain(outstanding_->logical, outstanding_->candidates[index].chain);
    return trial.is_valid_embedding();
}

void SearchSession::validate_handle(const CandidateBatch& batch) const {
    if (!outstanding_ || batch.session_id != session_id_ ||
        batch.generation != outstanding_->generation || batch.token != outstanding_->token ||
        batch.logical != outstanding_->logical) {
        throw std::logic_error("proposal is stale, foreign, or already consumed");
    }
}

void SearchSession::apply(const CandidateBatch& batch, std::size_t index) {
    const std::lock_guard<std::mutex> lock(mutex_);
    validate_handle(batch);
    if (index >= outstanding_->candidates.size()) {
        throw std::out_of_range("candidate index is outside the proposal batch");
    }
    NativeWorkCounters delta;
    delta.add_decisions();
    work_.charge(delta, work_limits_);
    state_.replace_chain(outstanding_->logical, outstanding_->candidates[index].chain);
    outstanding_.reset();
    ++generation_;
}

void SearchSession::discard(const CandidateBatch& batch) {
    const std::lock_guard<std::mutex> lock(mutex_);
    validate_handle(batch);
    NativeWorkCounters delta;
    delta.add_decisions();
    work_.charge(delta, work_limits_);
    outstanding_.reset();
}

}  // namespace lac::minorminer

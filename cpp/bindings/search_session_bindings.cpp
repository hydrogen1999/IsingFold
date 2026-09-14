#include "bindings.hpp"

#include "lac_minorminer/search_session.hpp"

#include <pybind11/stl.h>

namespace py = pybind11;

namespace lac::minorminer::bindings {

void bind_search_session(py::module_& module) {
    py::register_exception<WorkBudgetExceeded>(module, "WorkBudgetExceeded");

    py::class_<NativeWorkLimits>(module, "NativeWorkLimits")
            .def(py::init<>())
            .def_readwrite("decisions", &NativeWorkLimits::decisions)
            .def_readwrite("route_expansions", &NativeWorkLimits::route_expansions)
            .def_readwrite("materializations", &NativeWorkLimits::materializations)
            .def_readwrite("compiler_calls", &NativeWorkLimits::compiler_calls)
            .def_readwrite("validator_calls", &NativeWorkLimits::validator_calls)
            .def_readwrite("cut_edge_visits", &NativeWorkLimits::cut_edge_visits)
            .def_readwrite("restart_work", &NativeWorkLimits::restart_work)
            .def_readwrite("evaluator_reads", &NativeWorkLimits::evaluator_reads)
            .def_readwrite("feature_work", &NativeWorkLimits::feature_work);

    py::class_<NativeWorkCounters>(module, "NativeWorkCounters")
            .def_readonly("decisions", &NativeWorkCounters::decisions)
            .def_readonly("route_expansions", &NativeWorkCounters::route_expansions)
            .def_readonly("materializations", &NativeWorkCounters::materializations)
            .def_readonly("compiler_calls", &NativeWorkCounters::compiler_calls)
            .def_readonly("validator_calls", &NativeWorkCounters::validator_calls)
            .def_readonly("cut_edge_visits", &NativeWorkCounters::cut_edge_visits)
            .def_readonly("restart_work", &NativeWorkCounters::restart_work)
            .def_readonly("evaluator_reads", &NativeWorkCounters::evaluator_reads)
            .def_readonly("feature_work", &NativeWorkCounters::feature_work);

    py::class_<ResourceRank>(module, "ResourceRank")
            .def_readonly("max_occupancy", &ResourceRank::max_occupancy)
            .def_readonly("total_excess_occupancy", &ResourceRank::total_excess_occupancy)
            .def_readonly("used_target_nodes", &ResourceRank::used_target_nodes)
            .def_readonly("chain_length", &ResourceRank::chain_length)
            .def_readonly("route_cost", &ResourceRank::route_cost);

    py::class_<Candidate>(module, "Candidate")
            .def_readonly("candidate_id", &Candidate::candidate_id)
            .def_readonly("chain", &Candidate::chain)
            .def_readonly("rank", &Candidate::rank);

    py::class_<CandidateBatch>(module, "CandidateBatch")
            .def_readonly("session_id", &CandidateBatch::session_id)
            .def_readonly("generation", &CandidateBatch::generation)
            .def_readonly("token", &CandidateBatch::token)
            .def_readonly("logical", &CandidateBatch::logical)
            .def_readonly("candidates", &CandidateBatch::candidates);

    py::class_<AuditedCandidateBatches>(module, "AuditedCandidateBatches")
            .def_readonly("audit_batch", &AuditedCandidateBatches::audit_batch)
            .def_readonly("decision_batch", &AuditedCandidateBatches::decision_batch);

    py::class_<StateSnapshot>(module, "StateSnapshot")
            .def_readonly("generation", &StateSnapshot::generation)
            .def_readonly("chains", &StateSnapshot::chains)
            .def_readonly("occupancy", &StateSnapshot::occupancy)
            .def_readonly("conflicts", &StateSnapshot::conflicts)
            .def_readonly("missing_incident_edges", &StateSnapshot::missing_incident_edges)
            .def_readonly("max_occupancy", &StateSnapshot::max_occupancy)
            .def_readonly("total_excess_occupancy", &StateSnapshot::total_excess_occupancy)
            .def_readonly("used_target_nodes", &StateSnapshot::used_target_nodes)
            .def_readonly("missing_source_edges", &StateSnapshot::missing_source_edges)
            .def_readonly("valid", &StateSnapshot::valid);

    py::class_<SearchSession>(module, "SearchSession")
            .def(py::init<Graph, Graph, std::uint64_t, std::size_t, NativeWorkLimits>(),
                 py::arg("source"), py::arg("target"), py::arg("random_seed"),
                 py::arg("max_candidates"), py::arg("work_limits") = NativeWorkLimits{})
            .def_static("from_chains", &SearchSession::from_chains,
                 py::arg("source"), py::arg("target"), py::arg("chains"),
                 py::arg("random_seed"), py::arg("max_candidates"),
                 py::arg("work_limits") = NativeWorkLimits{},
                 py::call_guard<py::gil_scoped_release>())
            .def_property_readonly("session_id", &SearchSession::session_id)
            .def("work_counters", &SearchSession::work_counters,
                 py::call_guard<py::gil_scoped_release>())
            .def("snapshot", &SearchSession::snapshot,
                 py::call_guard<py::gil_scoped_release>())
            .def("fork", &SearchSession::fork,
                 py::call_guard<py::gil_scoped_release>())
            .def("fork_with_seed", &SearchSession::fork_with_seed,
                 py::arg("random_seed"), py::call_guard<py::gil_scoped_release>())
            .def("eligible_variables", &SearchSession::eligible_variables,
                 py::call_guard<py::gil_scoped_release>())
            .def("propose", &SearchSession::propose, py::arg("logical"),
                 py::arg("target_costs") = std::vector<double>{},
                 py::call_guard<py::gil_scoped_release>())
            .def("propose_applicable", &SearchSession::propose_applicable,
                 py::arg("logical"), py::arg("scoring_candidates"),
                 py::arg("target_costs") = std::vector<double>{},
                 py::call_guard<py::gil_scoped_release>())
            .def("propose_with_audit", &SearchSession::propose_with_audit,
                 py::arg("logical"), py::arg("audit_candidates"),
                 py::arg("target_costs") = std::vector<double>{},
                 py::call_guard<py::gil_scoped_release>())
            .def("materialize", &SearchSession::materialize, py::arg("logical"),
                 py::arg("chains"), py::call_guard<py::gil_scoped_release>())
            .def("candidate_is_valid", &SearchSession::candidate_is_valid,
                 py::arg("batch"), py::arg("index"),
                 py::call_guard<py::gil_scoped_release>())
            .def("apply", &SearchSession::apply, py::arg("batch"), py::arg("index"),
                 py::call_guard<py::gil_scoped_release>())
            .def("discard", &SearchSession::discard, py::arg("batch"),
                 py::call_guard<py::gil_scoped_release>())
            .def("perturb", &SearchSession::perturb, py::arg("logicals"),
                 py::call_guard<py::gil_scoped_release>())
            .def("restart", &SearchSession::restart,
                 py::call_guard<py::gil_scoped_release>());
}

}  // namespace lac::minorminer::bindings

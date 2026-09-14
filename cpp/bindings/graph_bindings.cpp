#include "bindings.hpp"

#include "lac_minorminer/graph.hpp"

#include <pybind11/stl.h>

namespace py = pybind11;

namespace lac::minorminer::bindings {

void bind_graph(py::module_& module) {
    py::class_<Graph>(module, "Graph")
            .def(py::init<std::size_t, std::vector<Graph::Edge>>(), py::arg("num_nodes"),
                 py::arg("edges"))
            .def_property_readonly("num_nodes", &Graph::num_nodes)
            .def_property_readonly("num_edges", &Graph::num_edges)
            .def("neighbors", &Graph::neighbors, py::arg("node"))
            .def("adjacent", &Graph::adjacent, py::arg("first"), py::arg("second"));
}

}  // namespace lac::minorminer::bindings

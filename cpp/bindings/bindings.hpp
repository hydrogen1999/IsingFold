#pragma once

#include <pybind11/pybind11.h>

namespace lac::minorminer::bindings {

void bind_graph(pybind11::module_& module);
void bind_process_isolation(pybind11::module_& module);
void bind_search_session(pybind11::module_& module);

}  // namespace lac::minorminer::bindings

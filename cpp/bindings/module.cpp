#include <pybind11/pybind11.h>

#include "bindings.hpp"
#include "lac_minorminer/work_counters.hpp"

namespace py = pybind11;

#ifndef LAC_MINORMINER_VERSION
#define LAC_MINORMINER_VERSION "unknown"
#endif

namespace {

const char* compiler_family() noexcept {
#if defined(__clang__)
    return "clang";
#elif defined(_MSC_VER)
    return "msvc";
#elif defined(__GNUC__)
    return "gcc";
#else
    return "unknown";
#endif
}

constexpr bool assertions_enabled() noexcept {
#ifdef NDEBUG
    return false;
#else
    return true;
#endif
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "Native operations for lac_minorminer";
    module.def("backend_info", []() {
        py::dict info;
        info["backend"] = "lac_minorminer_cpp";
        info["package_version"] = LAC_MINORMINER_VERSION;
        info["cpp_standard"] = 17;
        info["compiler_family"] = compiler_family();
        info["assertions_enabled"] = assertions_enabled();
        info["stock_minorminer_linked"] = false;
        info["work_counter_schema"] = lac::minorminer::kWorkCounterSchema;
        info["work_counter_version"] = lac::minorminer::kWorkCounterVersion;
        return info;
    });
    lac::minorminer::bindings::bind_graph(module);
    lac::minorminer::bindings::bind_process_isolation(module);
    lac::minorminer::bindings::bind_search_session(module);
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <initializer_list>
#include <stdexcept>

#include "bindings.hpp"
#include "lac_minorminer/process_isolation.hpp"

namespace py = pybind11;

namespace lac::minorminer::bindings {
namespace {

py::list string_list(const std::initializer_list<const char*> values) {
    py::list result;
    for (const char* value : values) {
        result.append(value);
    }
    return result;
}

py::dict support_payload(const isolation::Support& support) {
    py::dict result;
    result["landlock_abi"] = support.landlock_abi;
    result["machine"] = support.machine;
    result["minimum_landlock_abi"] = isolation::kMinimumLandlockAbi;
    result["native_boundary_version"] = isolation::kBoundaryVersion;
    result["platform"] = support.platform;
    result["reason"] = support.reason;
    result["supported"] = support.supported;
    return result;
}

py::dict receipt_payload(const isolation::Receipt& receipt) {
    py::dict result;
    result["attempt_kind"] = receipt.attempt_kind;
    result["backend"] = receipt.backend;
    result["closed_descriptor_floor"] = 3;
    result["handled_filesystem_rights"] = string_list({
        "execute", "write_file", "read_file", "read_dir", "remove_dir", "remove_file",
        "make_char", "make_dir", "make_reg", "make_sock", "make_fifo", "make_block",
        "make_sym", "refer", "truncate"});
    result["landlock_abi"] = receipt.landlock_abi;
    result["landlock_enforced"] = true;
    result["landlock_network_rights"] = string_list({"bind_tcp", "connect_tcp"});
    result["machine"] = "x86_64";
    result["native_boundary_version"] = isolation::kBoundaryVersion;
    result["no_new_privs"] = true;
    result["no_preexisting_child_processes_at_install"] =
        receipt.no_preexisting_child_processes_at_install;
    result["pid"] = receipt.pid;
    result["platform"] = "linux";
    result["policy_inherited"] = true;
    result["policy_logical_sha256"] = receipt.policy_logical_sha256;
    result["read_execute_path_count"] = receipt.read_execute_path_count;
    result["read_only_path_count"] = receipt.read_only_path_count;
    result["scheduler_evidence"] = receipt.scheduler_evidence;
    result["scheduler_mode"] = receipt.scheduler_mode;
    result["seccomp_denied_syscalls"] = string_list({
        "accept", "accept4", "bind", "connect", "getpeername", "getsockname", "getsockopt",
        "io_uring_enter", "io_uring_register", "io_uring_setup", "listen", "recvfrom",
        "recvmmsg", "recvmsg", "sendmmsg", "sendmsg", "sendto", "setsockopt", "shutdown",
        "socket", "socketpair"});
    result["seccomp_enforced"] = true;
    result["selected_artifact_root"] = receipt.selected_artifact_root;
    result["single_threaded_at_install"] = true;
    result["stderr_target"] = receipt.stderr_target;
    result["stdin_target"] = receipt.stdin_target;
    result["stdout_target"] = receipt.stdout_target;
    result["writable_path_count"] = receipt.writable_path_count;
    return result;
}

py::dict build_receipt_payload(const isolation::BuildReceipt& receipt) {
    py::dict result;
    result["attempt_index"] = receipt.attempt_index;
    result["attempt_root"] = receipt.attempt_root;
    result["backend"] = "goose";
    result["closed_descriptor_floor"] = 3;
    result["handled_filesystem_rights"] = string_list({
        "execute", "write_file", "read_file", "read_dir", "remove_dir", "remove_file",
        "make_char", "make_dir", "make_reg", "make_sock", "make_fifo", "make_block",
        "make_sym", "refer", "truncate"});
    result["landlock_abi"] = receipt.landlock_abi;
    result["landlock_enforced"] = true;
    result["landlock_network_rights"] = string_list({"bind_tcp", "connect_tcp"});
    result["machine"] = "x86_64";
    result["native_boundary_version"] = isolation::kBoundaryVersion;
    result["no_new_privs"] = true;
    result["no_preexisting_child_processes_at_install"] =
        receipt.no_preexisting_child_processes_at_install;
    result["peer_attempt_root"] = receipt.peer_attempt_root;
    result["pid"] = receipt.pid;
    result["platform"] = "linux";
    result["policy_inherited"] = true;
    result["policy_logical_sha256"] = receipt.policy_logical_sha256;
    result["read_execute_path_count"] = receipt.read_execute_path_count;
    result["read_only_path_count"] = receipt.read_only_path_count;
    result["role_kind"] = "wheel-build";
    result["scheduler_evidence"] = receipt.scheduler_evidence;
    result["scheduler_mode"] = receipt.scheduler_mode;
    result["seccomp_denied_syscalls"] = string_list({
        "accept", "accept4", "bind", "connect", "getpeername", "getsockname", "getsockopt",
        "io_uring_enter", "io_uring_register", "io_uring_setup", "listen", "recvfrom",
        "recvmmsg", "recvmsg", "sendmmsg", "sendmsg", "sendto", "setsockopt", "shutdown",
        "socket", "socketpair"});
    result["seccomp_enforced"] = true;
    result["single_threaded_at_install"] = true;
    result["source_root"] = receipt.source_root;
    result["stderr_target"] = receipt.stderr_target;
    result["stdin_target"] = receipt.stdin_target;
    result["stdout_target"] = receipt.stdout_target;
    result["wheelhouse_root"] = receipt.wheelhouse_root;
    result["writable_path_count"] = receipt.writable_path_count;
    return result;
}

}  // namespace

void bind_process_isolation(py::module_& module) {
    py::class_<isolation::ProcessIsolationCapability>(module,
                                                       "D1ProcessIsolationCapability")
        .def("receipt", [](const isolation::ProcessIsolationCapability& capability) {
            return receipt_payload(capability.receipt());
        })
        .def("validate_current_process",
             &isolation::ProcessIsolationCapability::validate_current_process)
        .def("__reduce_ex__", [](const isolation::ProcessIsolationCapability&, int) -> py::object {
            throw py::type_error("native process-isolation capabilities cannot be serialized");
        });

    py::class_<isolation::BuildProcessIsolationCapability>(
        module, "D1BuildProcessIsolationCapability")
        .def("receipt", [](const isolation::BuildProcessIsolationCapability& capability) {
            return build_receipt_payload(capability.receipt());
        })
        .def("validate_current_process",
             &isolation::BuildProcessIsolationCapability::validate_current_process)
        .def("__reduce_ex__",
             [](const isolation::BuildProcessIsolationCapability&, int) -> py::object {
                 throw py::type_error(
                     "native build process-isolation capabilities cannot be serialized");
             });

    module.def("query_d1_process_isolation_support", []() {
        return support_payload(isolation::query_support());
    });
    module.def(
        "install_d1_process_isolation_v1",
        [](const std::string& policy_logical_sha256, const std::string& backend,
           const std::string& attempt_kind, const std::string& worker_root,
           const std::string& selected_artifact_root, const std::string& peer_artifact_root,
           const std::vector<std::string>& read_execute_paths,
           const std::vector<std::string>& read_only_paths,
           const std::vector<std::string>& writable_paths,
           const std::vector<std::string>& negative_probe_paths) {
            isolation::Request request{
                policy_logical_sha256,
                backend,
                attempt_kind,
                worker_root,
                selected_artifact_root,
                peer_artifact_root,
                read_execute_paths,
                read_only_paths,
                writable_paths,
                negative_probe_paths,
            };
            return isolation::install(request);
        },
        py::arg("policy_logical_sha256"), py::arg("backend"), py::arg("attempt_kind"),
        py::arg("worker_root"), py::arg("selected_artifact_root"),
        py::arg("peer_artifact_root"), py::arg("read_execute_paths"),
        py::arg("read_only_paths"), py::arg("writable_paths"),
        py::arg("negative_probe_paths"));
    module.def(
        "install_d1_build_process_isolation_v1",
        [](const std::string& policy_logical_sha256, const int attempt_index,
           const std::string& attempt_root, const std::string& peer_attempt_root,
           const std::string& source_root, const std::string& wheelhouse_root,
           const std::vector<std::string>& read_execute_paths,
           const std::vector<std::string>& read_only_paths,
           const std::vector<std::string>& writable_paths,
           const std::vector<std::string>& negative_probe_paths) {
            isolation::BuildRequest request{
                policy_logical_sha256,
                attempt_index,
                attempt_root,
                peer_attempt_root,
                source_root,
                wheelhouse_root,
                read_execute_paths,
                read_only_paths,
                writable_paths,
                negative_probe_paths,
            };
            return isolation::install_build(request);
        },
        py::arg("policy_logical_sha256"), py::arg("attempt_index"),
        py::arg("attempt_root"), py::arg("peer_attempt_root"), py::arg("source_root"),
        py::arg("wheelhouse_root"), py::arg("read_execute_paths"),
        py::arg("read_only_paths"), py::arg("writable_paths"),
        py::arg("negative_probe_paths"));
}

}  // namespace lac::minorminer::bindings

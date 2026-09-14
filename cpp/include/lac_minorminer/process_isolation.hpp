#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lac::minorminer::isolation {

inline constexpr const char* kBoundaryVersion = "landlock-seccomp-v1";
inline constexpr int kMinimumLandlockAbi = 4;

struct Support {
    bool supported{false};
    int landlock_abi{0};
    std::string platform;
    std::string machine;
    std::string reason;
};

struct Request {
    std::string policy_logical_sha256;
    std::string backend;
    std::string attempt_kind;
    std::string worker_root;
    std::string selected_artifact_root;
    std::string peer_artifact_root;
    std::vector<std::string> read_execute_paths;
    std::vector<std::string> read_only_paths;
    std::vector<std::string> writable_paths;
    std::vector<std::string> negative_probe_paths;
};

struct Receipt {
    std::string policy_logical_sha256;
    std::string backend;
    std::string attempt_kind;
    std::string selected_artifact_root;
    std::string scheduler_mode;
    std::string scheduler_evidence;
    std::int64_t pid{0};
    int landlock_abi{0};
    bool no_preexisting_child_processes_at_install{false};
    std::string stdin_target;
    std::string stdout_target;
    std::string stderr_target;
    std::size_t read_execute_path_count{0};
    std::size_t read_only_path_count{0};
    std::size_t writable_path_count{0};
};

struct BuildRequest {
    std::string policy_logical_sha256;
    int attempt_index{-1};
    std::string attempt_root;
    std::string peer_attempt_root;
    std::string source_root;
    std::string wheelhouse_root;
    std::vector<std::string> read_execute_paths;
    std::vector<std::string> read_only_paths;
    std::vector<std::string> writable_paths;
    std::vector<std::string> negative_probe_paths;
};

struct BuildReceipt {
    std::string policy_logical_sha256;
    int attempt_index{-1};
    std::string attempt_root;
    std::string peer_attempt_root;
    std::string source_root;
    std::string wheelhouse_root;
    std::string scheduler_mode;
    std::string scheduler_evidence;
    std::int64_t pid{0};
    int landlock_abi{0};
    bool no_preexisting_child_processes_at_install{false};
    std::string stdin_target;
    std::string stdout_target;
    std::string stderr_target;
    std::size_t read_execute_path_count{0};
    std::size_t read_only_path_count{0};
    std::size_t writable_path_count{0};
};

class ProcessIsolationCapability final {
public:
    ProcessIsolationCapability(const ProcessIsolationCapability&) = delete;
    ProcessIsolationCapability& operator=(const ProcessIsolationCapability&) = delete;
    ProcessIsolationCapability(ProcessIsolationCapability&&) = delete;
    ProcessIsolationCapability& operator=(ProcessIsolationCapability&&) = delete;
    ~ProcessIsolationCapability() = default;

    [[nodiscard]] const Receipt& receipt() const noexcept;
    [[nodiscard]] bool validate_current_process() const noexcept;

private:
    explicit ProcessIsolationCapability(Receipt receipt);
    Receipt receipt_;

    friend std::unique_ptr<ProcessIsolationCapability> install(const Request& request);
};

class BuildProcessIsolationCapability final {
public:
    BuildProcessIsolationCapability(const BuildProcessIsolationCapability&) = delete;
    BuildProcessIsolationCapability& operator=(const BuildProcessIsolationCapability&) = delete;
    BuildProcessIsolationCapability(BuildProcessIsolationCapability&&) = delete;
    BuildProcessIsolationCapability& operator=(BuildProcessIsolationCapability&&) = delete;
    ~BuildProcessIsolationCapability() = default;

    [[nodiscard]] const BuildReceipt& receipt() const noexcept;
    [[nodiscard]] bool validate_current_process() const noexcept;

private:
    explicit BuildProcessIsolationCapability(BuildReceipt receipt);
    BuildReceipt receipt_;

    friend std::unique_ptr<BuildProcessIsolationCapability> install_build(
        const BuildRequest& request);
};

[[nodiscard]] Support query_support() noexcept;
[[nodiscard]] std::unique_ptr<ProcessIsolationCapability> install(const Request& request);
[[nodiscard]] std::unique_ptr<BuildProcessIsolationCapability> install_build(
    const BuildRequest& request);

}  // namespace lac::minorminer::isolation

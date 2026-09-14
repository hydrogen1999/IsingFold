#include "lac_minorminer/process_isolation.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__) && defined(__x86_64__)
#include <dirent.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <process.h>
#endif

namespace lac::minorminer::isolation {
namespace {

#if defined(__linux__) && defined(__x86_64__)
constexpr std::uint64_t kFsExecute = 1ULL << 0;
constexpr std::uint64_t kFsWriteFile = 1ULL << 1;
constexpr std::uint64_t kFsReadFile = 1ULL << 2;
constexpr std::uint64_t kFsReadDir = 1ULL << 3;
constexpr std::uint64_t kFsRemoveDir = 1ULL << 4;
constexpr std::uint64_t kFsRemoveFile = 1ULL << 5;
constexpr std::uint64_t kFsMakeChar = 1ULL << 6;
constexpr std::uint64_t kFsMakeDir = 1ULL << 7;
constexpr std::uint64_t kFsMakeReg = 1ULL << 8;
constexpr std::uint64_t kFsMakeSock = 1ULL << 9;
constexpr std::uint64_t kFsMakeFifo = 1ULL << 10;
constexpr std::uint64_t kFsMakeBlock = 1ULL << 11;
constexpr std::uint64_t kFsMakeSym = 1ULL << 12;
constexpr std::uint64_t kFsRefer = 1ULL << 13;
constexpr std::uint64_t kFsTruncate = 1ULL << 14;
constexpr std::uint64_t kAllFilesystemRights =
    kFsExecute | kFsWriteFile | kFsReadFile | kFsReadDir | kFsRemoveDir | kFsRemoveFile |
    kFsMakeChar | kFsMakeDir | kFsMakeReg | kFsMakeSock | kFsMakeFifo | kFsMakeBlock |
    kFsMakeSym | kFsRefer | kFsTruncate;
constexpr std::uint64_t kReadExecuteRights = kFsExecute | kFsReadFile | kFsReadDir;
constexpr std::uint64_t kReadOnlyRights = kFsReadFile | kFsReadDir;
constexpr std::uint64_t kSelectedRootRights =
    kFsWriteFile | kFsReadFile | kFsReadDir | kFsRemoveDir | kFsRemoveFile | kFsMakeDir |
    kFsMakeReg | kFsRefer | kFsTruncate;
constexpr std::uint64_t kBuildWritableRights = kSelectedRootRights | kFsExecute;
constexpr std::uint64_t kAllNetworkRights = (1ULL << 0) | (1ULL << 1);
#endif

std::atomic<std::int64_t> installed_pid{0};

[[nodiscard]] std::int64_t current_pid() noexcept {
#if defined(_WIN32)
    return static_cast<std::int64_t>(_getpid());
#else
    return static_cast<std::int64_t>(getpid());
#endif
}

[[nodiscard]] bool is_lower_sha256(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](const unsigned char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

[[nodiscard]] bool is_normalized_absolute_path(const std::string& value) {
    if (value.size() < 2 || value.front() != '/' || value.back() == '/' ||
        value.find("//") != std::string::npos || value.find('\\') != std::string::npos) {
        return false;
    }
    std::size_t start = 1;
    while (start < value.size()) {
        const auto end = value.find('/', start);
        const auto length = (end == std::string::npos ? value.size() : end) - start;
        const auto component = value.substr(start, length);
        if (component.empty() || component == "." || component == "..") {
            return false;
        }
        for (const unsigned char character : component) {
            if (character < 32 || character == 127) {
                return false;
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return true;
}

[[nodiscard]] bool path_within(const std::string& path, const std::string& parent) {
    return path == parent ||
           (path.size() > parent.size() && path.compare(0, parent.size(), parent) == 0 &&
            path[parent.size()] == '/');
}

[[nodiscard]] bool paths_overlap(const std::string& first, const std::string& second) {
    return path_within(first, second) || path_within(second, first);
}

void validate_path_list(const std::vector<std::string>& paths, const char* name) {
    if (paths.empty() || !std::is_sorted(paths.begin(), paths.end()) ||
        std::adjacent_find(paths.begin(), paths.end()) != paths.end()) {
        throw std::invalid_argument(std::string(name) + " must be nonempty, unique, and sorted");
    }
    for (std::size_t index = 0; index < paths.size(); ++index) {
        if (!is_normalized_absolute_path(paths[index])) {
            throw std::invalid_argument(std::string(name) + " contains an unsafe path");
        }
        for (std::size_t other = index + 1; other < paths.size(); ++other) {
            if (paths_overlap(paths[index], paths[other])) {
                throw std::invalid_argument(std::string(name) + " contains overlapping paths");
            }
        }
    }
}

void validate_request(const Request& request) {
    if (!is_lower_sha256(request.policy_logical_sha256)) {
        throw std::invalid_argument("policy digest must be one lowercase SHA-256 value");
    }
    if ((request.backend != "apollo" && request.backend != "goose") ||
        (request.attempt_kind != "outcome" && request.attempt_kind != "replay")) {
        throw std::invalid_argument("backend and attempt kind must name one registered role");
    }
    for (const auto* path : {&request.worker_root, &request.selected_artifact_root,
                             &request.peer_artifact_root}) {
        if (!is_normalized_absolute_path(*path)) {
            throw std::invalid_argument("role roots must be normalized absolute paths below root");
        }
    }
    if (paths_overlap(request.selected_artifact_root, request.peer_artifact_root)) {
        throw std::invalid_argument("selected and peer artifact roots must not overlap");
    }
    if (paths_overlap(request.worker_root, request.selected_artifact_root) ||
        paths_overlap(request.worker_root, request.peer_artifact_root)) {
        throw std::invalid_argument("worker and artifact roots must not overlap");
    }
    validate_path_list(request.read_execute_paths, "read/execute paths");
    validate_path_list(request.read_only_paths, "read-only paths");
    validate_path_list(request.writable_paths, "writable paths");
    validate_path_list(request.negative_probe_paths, "negative-probe paths");
    if (request.writable_paths != std::vector<std::string>{request.selected_artifact_root}) {
        throw std::invalid_argument("only the selected artifact root may be writable");
    }
    if (std::find(request.read_only_paths.begin(), request.read_only_paths.end(),
                  request.worker_root) == request.read_only_paths.end()) {
        throw std::invalid_argument("read-only paths must contain the worker root");
    }
    if (std::find(request.negative_probe_paths.begin(), request.negative_probe_paths.end(),
                  request.peer_artifact_root) == request.negative_probe_paths.end()) {
        throw std::invalid_argument("negative probes must contain the peer artifact root");
    }
    for (const auto& path : request.read_execute_paths) {
        if (paths_overlap(path, request.worker_root)) {
            throw std::invalid_argument("worker root cannot receive execute permission");
        }
        for (const auto& read_only : request.read_only_paths) {
            if (paths_overlap(path, read_only) &&
                !(path != read_only && path_within(path, read_only))) {
                throw std::invalid_argument(
                    "overlapping read/execute and read-only rules would widen permissions");
            }
        }
    }
    std::vector<std::string> allowed = request.read_execute_paths;
    allowed.insert(allowed.end(), request.read_only_paths.begin(), request.read_only_paths.end());
    for (const auto& path : allowed) {
        if (paths_overlap(path, request.selected_artifact_root) ||
            paths_overlap(path, request.peer_artifact_root)) {
            throw std::invalid_argument("immutable allowed paths overlap an artifact root");
        }
    }
    for (const auto& negative : request.negative_probe_paths) {
        if (paths_overlap(negative, request.selected_artifact_root)) {
            throw std::invalid_argument("a negative probe overlaps the selected artifact root");
        }
        for (const auto& path : allowed) {
            if (paths_overlap(negative, path)) {
                throw std::invalid_argument("a negative probe overlaps an allowed path");
            }
        }
    }
}

void require_no_cross_category_overlap(const std::vector<std::string>& first,
                                       const std::vector<std::string>& second) {
    for (const auto& first_path : first) {
        for (const auto& second_path : second) {
            if (paths_overlap(first_path, second_path)) {
                throw std::invalid_argument("build isolation authority classes must not overlap");
            }
        }
    }
}

void validate_build_execute_read_only_overlap(
    const std::vector<std::string>& read_execute_paths,
    const std::vector<std::string>& read_only_paths) {
    for (const auto& executable : read_execute_paths) {
        for (const auto& immutable : read_only_paths) {
            if (paths_overlap(executable, immutable) &&
                !(executable != immutable && path_within(executable, immutable))) {
                throw std::invalid_argument(
                    "build read/execute and read-only rules overlap unsafely");
            }
        }
    }
}

void validate_build_request(const BuildRequest& request) {
    if (!is_lower_sha256(request.policy_logical_sha256)) {
        throw std::invalid_argument("build policy digest must be one lowercase SHA-256 value");
    }
    if (request.attempt_index != 0 && request.attempt_index != 1) {
        throw std::invalid_argument("build attempt index must be exactly zero or one");
    }
    for (const auto* path : {&request.attempt_root, &request.peer_attempt_root,
                             &request.source_root, &request.wheelhouse_root}) {
        if (!is_normalized_absolute_path(*path)) {
            throw std::invalid_argument(
                "build isolation roots must be normalized absolute paths below root");
        }
    }
    const std::string suffix = "/attempt-" + std::to_string(request.attempt_index);
    if (request.attempt_root.size() <= suffix.size() ||
        request.attempt_root.compare(request.attempt_root.size() - suffix.size(), suffix.size(),
                                     suffix) != 0) {
        throw std::invalid_argument("build attempt root has the wrong registered suffix");
    }
    const std::string output_root =
        request.attempt_root.substr(0, request.attempt_root.size() - suffix.size());
    const std::string expected_peer =
        output_root + "/attempt-" + std::to_string(1 - request.attempt_index);
    if (request.peer_attempt_root != expected_peer ||
        request.source_root != request.attempt_root + "/source") {
        throw std::invalid_argument("build peer/source roots differ from the registered layout");
    }
    if (paths_overlap(request.attempt_root, request.peer_attempt_root) ||
        paths_overlap(request.wheelhouse_root, request.attempt_root) ||
        paths_overlap(request.wheelhouse_root, request.peer_attempt_root)) {
        throw std::invalid_argument("build immutable and attempt roots must not overlap");
    }

    validate_path_list(request.read_execute_paths, "build read/execute paths");
    validate_path_list(request.read_only_paths, "build read-only paths");
    validate_path_list(request.writable_paths, "build writable paths");
    validate_path_list(request.negative_probe_paths, "build negative-probe paths");
    const std::vector<std::string> expected_writable = {
        request.attempt_root + "/build",
        request.attempt_root + "/build-env",
        request.attempt_root + "/dist",
        request.attempt_root + "/reports",
        request.attempt_root + "/validation",
    };
    if (request.writable_paths != expected_writable) {
        throw std::invalid_argument(
            "build writable paths must be exactly the five registered subtrees");
    }
    if (std::find(request.read_only_paths.begin(), request.read_only_paths.end(),
                  request.source_root) == request.read_only_paths.end() ||
        std::find(request.read_only_paths.begin(), request.read_only_paths.end(),
                  request.wheelhouse_root) == request.read_only_paths.end()) {
        throw std::invalid_argument("build source and wheelhouse roots must both be read-only");
    }
    if (std::find(request.negative_probe_paths.begin(), request.negative_probe_paths.end(),
                  request.peer_attempt_root) == request.negative_probe_paths.end()) {
        throw std::invalid_argument("build negative probes must contain the peer attempt root");
    }
    // Landlock path rules in one layer are additive.  A strictly-descendant
    // executable may therefore receive EXECUTE from its exact rule while its
    // immutable parent supplies read-only data access.  Equal paths or a broad
    // executable ancestor would widen authority and remain forbidden.
    validate_build_execute_read_only_overlap(request.read_execute_paths,
                                             request.read_only_paths);
    require_no_cross_category_overlap(request.read_execute_paths, request.writable_paths);
    require_no_cross_category_overlap(request.read_only_paths, request.writable_paths);
    for (const auto& path : request.read_execute_paths) {
        if (paths_overlap(path, request.attempt_root) ||
            paths_overlap(path, request.peer_attempt_root)) {
            throw std::invalid_argument("host build executables must remain outside attempt roots");
        }
    }
    for (const auto& path : request.read_only_paths) {
        if ((paths_overlap(path, request.attempt_root) && path != request.source_root) ||
            paths_overlap(path, request.peer_attempt_root)) {
            throw std::invalid_argument(
                "only the selected source may be read-only below the build output root");
        }
    }
    std::vector<std::string> allowed = request.read_execute_paths;
    allowed.insert(allowed.end(), request.read_only_paths.begin(), request.read_only_paths.end());
    allowed.insert(allowed.end(), request.writable_paths.begin(), request.writable_paths.end());
    for (const auto& negative : request.negative_probe_paths) {
        for (const auto& path : allowed) {
            if (paths_overlap(negative, path)) {
                throw std::invalid_argument("build negative probe overlaps an allowed path");
            }
        }
    }
}

#if defined(__linux__) && defined(__x86_64__)

#ifndef __NR_landlock_create_ruleset
#define __NR_landlock_create_ruleset 444
#define __NR_landlock_add_rule 445
#define __NR_landlock_restrict_self 446
#endif
#ifndef __NR_close_range
#define __NR_close_range 436
#endif
#ifndef __NR_openat2
#define __NR_openat2 437
#endif

constexpr unsigned int kLandlockCreateRulesetVersion = 1;
constexpr int kLandlockRulePathBeneath = 1;
constexpr std::uint64_t kResolveNoMagicLinks = 0x02;
constexpr std::uint64_t kResolveNoSymlinks = 0x04;

struct [[gnu::packed]] LandlockRulesetAttr {
    std::uint64_t handled_access_fs;
    std::uint64_t handled_access_net;
};

struct [[gnu::packed]] LandlockPathBeneathAttr {
    std::uint64_t allowed_access;
    std::int32_t parent_fd;
};

struct OpenHow {
    std::uint64_t flags;
    std::uint64_t mode;
    std::uint64_t resolve;
};

class FileDescriptor final {
public:
    explicit FileDescriptor(const int value = -1) noexcept : value_(value) {}
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept : value_(std::exchange(other.value_, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            reset();
            value_ = std::exchange(other.value_, -1);
        }
        return *this;
    }
    ~FileDescriptor() { reset(); }
    [[nodiscard]] int get() const noexcept { return value_; }
    void reset() noexcept {
        if (value_ >= 0) {
            close(value_);
            value_ = -1;
        }
    }

private:
    int value_;
};

[[nodiscard]] std::string errno_message(const char* operation) {
    return std::string(operation) + " failed: " + std::strerror(errno);
}

[[nodiscard]] int query_landlock_abi() noexcept {
    errno = 0;
    const long result = syscall(__NR_landlock_create_ruleset, nullptr, 0,
                                kLandlockCreateRulesetVersion);
    if (result < 0 || result > INT_MAX) {
        return 0;
    }
    return static_cast<int>(result);
}

[[nodiscard]] std::size_t thread_count() {
    DIR* directory = opendir("/proc/self/task");
    if (directory == nullptr) {
        throw std::runtime_error(errno_message("open /proc/self/task"));
    }
    std::size_t count = 0;
    errno = 0;
    while (const dirent* entry = readdir(directory)) {
        if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9' &&
            std::all_of(entry->d_name, entry->d_name + std::strlen(entry->d_name),
                        [](const unsigned char value) { return value >= '0' && value <= '9'; })) {
            ++count;
        }
        errno = 0;
    }
    const int read_error = errno;
    if (closedir(directory) != 0 && read_error == 0) {
        throw std::runtime_error(errno_message("close /proc/self/task"));
    }
    if (read_error != 0) {
        errno = read_error;
        throw std::runtime_error(errno_message("read /proc/self/task"));
    }
    return count;
}

void require_no_child_processes() {
    const std::string path = "/proc/self/task/" + std::to_string(current_pid()) + "/children";
    FileDescriptor descriptor(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (descriptor.get() < 0) {
        throw std::runtime_error(errno_message("open current-thread children"));
    }
    char buffer[2]{};
    const auto count = read(descriptor.get(), buffer, sizeof(buffer));
    if (count < 0) {
        throw std::runtime_error(errno_message("read current-thread children"));
    }
    if (count != 0) {
        throw std::runtime_error("native isolation requires no pre-existing child processes");
    }
}

[[nodiscard]] std::string descriptor_target(const int descriptor) {
    const std::string path = "/proc/self/fd/" + std::to_string(descriptor);
    std::vector<char> buffer(4097, '\0');
    const auto size = readlink(path.c_str(), buffer.data(), buffer.size() - 1);
    if (size <= 0 || size >= static_cast<ssize_t>(buffer.size() - 1)) {
        throw std::runtime_error("standard-stream target is absent or too long");
    }
    return std::string(buffer.data(), static_cast<std::size_t>(size));
}

[[nodiscard]] bool canonical_decimal(const char* value, const bool positive) {
    if (value == nullptr || *value == '\0') {
        return false;
    }
    const std::size_t size = std::strlen(value);
    if (size > 18 || (size > 1 && value[0] == '0') || (positive && value[0] == '0')) {
        return false;
    }
    return std::all_of(value, value + size,
                       [](const unsigned char character) { return character >= '0' && character <= '9'; });
}

[[nodiscard]] std::string read_cgroup_identity() {
    FileDescriptor descriptor(open("/proc/self/cgroup", O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (descriptor.get() < 0) {
        throw std::runtime_error(errno_message("open /proc/self/cgroup"));
    }
    std::string result;
    result.reserve(256);
    char buffer[1024];
    while (true) {
        const auto count = read(descriptor.get(), buffer, sizeof(buffer));
        if (count < 0) {
            throw std::runtime_error(errno_message("read /proc/self/cgroup"));
        }
        if (count == 0) {
            break;
        }
        if (result.size() + static_cast<std::size_t>(count) > 64 * 1024) {
            throw std::runtime_error("/proc/self/cgroup exceeds the validation bound");
        }
        result.append(buffer, static_cast<std::size_t>(count));
    }
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    if (result.empty() || result.find('\n') != std::string::npos ||
        std::any_of(result.begin(), result.end(), [](const unsigned char character) {
            return character < 32 || character == 127;
        })) {
        throw std::runtime_error("unified cgroup identity is not one canonical line");
    }
    return result;
}

template <typename ReceiptType>
void verify_scheduler_boundary(const std::string& backend, ReceiptType& receipt) {
    constexpr const char* slurm_names[] = {
        "SLURM_JOB_ID",
        "SLURM_JOBID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
    };
    if (backend == "apollo") {
        for (const char* name : slurm_names) {
            const char* value = std::getenv(name);
            if (value != nullptr && *value != '\0') {
                throw std::runtime_error("Apollo direct execution rejects Slurm job variables");
            }
        }
        receipt.scheduler_mode = "bounded-direct-no-slurm";
        receipt.scheduler_evidence = "slurm-job-environment-absent";
        return;
    }

    const char* job_id = std::getenv("SLURM_JOB_ID");
    const char* array_job_id = std::getenv("SLURM_ARRAY_JOB_ID");
    const char* array_task_id = std::getenv("SLURM_ARRAY_TASK_ID");
    if (!canonical_decimal(job_id, true) || !canonical_decimal(array_job_id, true) ||
        !canonical_decimal(array_task_id, false)) {
        throw std::runtime_error("Goose isolation requires canonical Slurm array identifiers");
    }
    const std::string cgroup = read_cgroup_identity();
    const std::string expected = "0::/system.slice/slurmstepd.scope/job_" +
                                 std::string(job_id) + "/step_batch/user/task_0";
    if (cgroup != expected) {
        throw std::runtime_error("Goose process is not in its claimed Slurm batch-job cgroup");
    }
    receipt.scheduler_mode = "slurm-allocation-required";
    receipt.scheduler_evidence = cgroup;
}

template <typename ReceiptType>
void validate_standard_streams(ReceiptType& receipt) {
    struct stat null_metadata {};
    struct stat input_metadata {};
    const int input_flags = fcntl(STDIN_FILENO, F_GETFL);
    if (stat("/dev/null", &null_metadata) != 0 || fstat(STDIN_FILENO, &input_metadata) != 0 ||
        !S_ISCHR(input_metadata.st_mode) || input_metadata.st_rdev != null_metadata.st_rdev ||
        input_flags < 0 || (input_flags & O_ACCMODE) != O_RDONLY) {
        throw std::runtime_error("standard input must be read-only /dev/null");
    }
    receipt.stdin_target = "/dev/null";
    for (const int descriptor : {STDOUT_FILENO, STDERR_FILENO}) {
        struct stat metadata {};
        const int flags = fcntl(descriptor, F_GETFL);
        if (fstat(descriptor, &metadata) != 0 || S_ISSOCK(metadata.st_mode) || flags < 0 ||
            (flags & O_ACCMODE) == O_RDONLY) {
            throw std::runtime_error("stdout and stderr must be writable non-socket channels");
        }
    }
    receipt.stdout_target = descriptor_target(STDOUT_FILENO);
    receipt.stderr_target = descriptor_target(STDERR_FILENO);
}

void close_inherited_descriptors() {
    if (syscall(__NR_close_range, 3U, UINT_MAX, 0U) != 0) {
        throw std::runtime_error(errno_message("close_range"));
    }
    errno = 0;
    if (fcntl(3, F_GETFD) != -1 || errno != EBADF) {
        throw std::runtime_error("descriptor sweep did not close descriptor 3");
    }
}

[[nodiscard]] FileDescriptor open_physical_path(const std::string& path) {
    OpenHow how{
        static_cast<std::uint64_t>(O_PATH | O_CLOEXEC | O_NOFOLLOW),
        0,
        kResolveNoMagicLinks | kResolveNoSymlinks,
    };
    const long descriptor = syscall(__NR_openat2, AT_FDCWD, path.c_str(), &how, sizeof(how));
    if (descriptor < 0 || descriptor > INT_MAX) {
        throw std::runtime_error(errno_message("openat2 isolation path"));
    }
    return FileDescriptor(static_cast<int>(descriptor));
}

void add_path_rule(const int ruleset, const std::string& path, std::uint64_t rights,
                   const bool require_directory) {
    FileDescriptor parent = open_physical_path(path);
    struct stat metadata {};
    if (fstat(parent.get(), &metadata) != 0) {
        throw std::runtime_error(errno_message("fstat isolation path"));
    }
    if (require_directory && !S_ISDIR(metadata.st_mode)) {
        throw std::runtime_error("writable isolation path must be a directory");
    }
    if (S_ISREG(metadata.st_mode)) {
        rights &= kFsExecute | kFsWriteFile | kFsReadFile | kFsTruncate;
    } else if (S_ISCHR(metadata.st_mode)) {
        if (path != "/dev/null" || require_directory || rights != kReadOnlyRights) {
            throw std::runtime_error(
                "only exact read-only /dev/null may be a build isolation device rule");
        }
        rights = kFsReadFile;
    } else if (!S_ISDIR(metadata.st_mode)) {
        throw std::runtime_error("isolation paths must be regular files or directories");
    }
    LandlockPathBeneathAttr attribute{rights, parent.get()};
    if (syscall(__NR_landlock_add_rule, ruleset, kLandlockRulePathBeneath, &attribute, 0) != 0) {
        throw std::runtime_error(errno_message("landlock_add_rule"));
    }
}

[[nodiscard]] FileDescriptor build_ruleset(
    const std::vector<std::string>& read_execute_paths,
    const std::vector<std::string>& read_only_paths,
    const std::vector<std::string>& writable_paths,
    const std::uint64_t writable_rights) {
    LandlockRulesetAttr attribute{kAllFilesystemRights, kAllNetworkRights};
    const long descriptor =
        syscall(__NR_landlock_create_ruleset, &attribute, sizeof(attribute), 0);
    if (descriptor < 0 || descriptor > INT_MAX) {
        throw std::runtime_error(errno_message("landlock_create_ruleset"));
    }
    FileDescriptor ruleset(static_cast<int>(descriptor));
    for (const auto& path : read_execute_paths) {
        add_path_rule(ruleset.get(), path, kReadExecuteRights, false);
    }
    for (const auto& path : read_only_paths) {
        add_path_rule(ruleset.get(), path, kReadOnlyRights, false);
    }
    for (const auto& path : writable_paths) {
        add_path_rule(ruleset.get(), path, writable_rights, true);
    }
    return ruleset;
}

[[nodiscard]] sock_filter statement(const std::uint16_t code, const std::uint32_t value) {
    return sock_filter{code, 0, 0, value};
}

[[nodiscard]] sock_filter jump(const std::uint16_t code, const std::uint32_t value,
                               const std::uint8_t on_true, const std::uint8_t on_false) {
    return sock_filter{code, on_true, on_false, value};
}

[[nodiscard]] std::vector<int> denied_syscalls() {
    return {
        __NR_accept,          __NR_accept4,         __NR_bind,
        __NR_connect,         __NR_getpeername,     __NR_getsockname,
        __NR_getsockopt,      __NR_io_uring_enter,  __NR_io_uring_register,
        __NR_io_uring_setup,  __NR_listen,          __NR_recvfrom,
        __NR_recvmmsg,        __NR_recvmsg,         __NR_sendmmsg,
        __NR_sendmsg,         __NR_sendto,          __NR_setsockopt,
        __NR_shutdown,        __NR_socket,          __NR_socketpair,
    };
}

void install_seccomp_filter() {
    std::vector<sock_filter> filter;
    const auto syscalls = denied_syscalls();
    filter.reserve(7 + syscalls.size() * 2);
    filter.push_back(statement(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)));
    filter.push_back(jump(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0));
    filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
    filter.push_back(statement(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)));
    filter.push_back(jump(BPF_JMP | BPF_JGE | BPF_K, 0x40000000U, 0, 1));
    filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS));
    for (const int syscall_number : syscalls) {
        filter.push_back(
            jump(BPF_JMP | BPF_JEQ | BPF_K, static_cast<std::uint32_t>(syscall_number), 0, 1));
        filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM));
    }
    filter.push_back(statement(BPF_RET | BPF_K, SECCOMP_RET_ALLOW));
    if (filter.size() > USHRT_MAX) {
        throw std::runtime_error("seccomp filter exceeds the kernel instruction-count field");
    }
    sock_fprog program{
        static_cast<unsigned short>(filter.size()),
        filter.data(),
    };
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
        throw std::runtime_error(errno_message("PR_SET_SECCOMP"));
    }
}

[[noreturn]] void terminate_partial_installation() noexcept { _exit(126); }

#endif

}  // namespace

ProcessIsolationCapability::ProcessIsolationCapability(Receipt receipt)
    : receipt_(std::move(receipt)) {}

const Receipt& ProcessIsolationCapability::receipt() const noexcept { return receipt_; }

bool ProcessIsolationCapability::validate_current_process() const noexcept {
    if (receipt_.pid != current_pid() || installed_pid.load() != receipt_.pid) {
        return false;
    }
#if defined(__linux__) && defined(__x86_64__)
    return prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1 &&
           prctl(PR_GET_SECCOMP, 0, 0, 0, 0) == SECCOMP_MODE_FILTER;
#else
    return false;
#endif
}

BuildProcessIsolationCapability::BuildProcessIsolationCapability(BuildReceipt receipt)
    : receipt_(std::move(receipt)) {}

const BuildReceipt& BuildProcessIsolationCapability::receipt() const noexcept {
    return receipt_;
}

bool BuildProcessIsolationCapability::validate_current_process() const noexcept {
    if (receipt_.pid != current_pid() || installed_pid.load() != receipt_.pid) {
        return false;
    }
#if defined(__linux__) && defined(__x86_64__)
    return prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1 &&
           prctl(PR_GET_SECCOMP, 0, 0, 0, 0) == SECCOMP_MODE_FILTER;
#else
    return false;
#endif
}

Support query_support() noexcept {
#if defined(__linux__) && defined(__x86_64__)
    const int abi = query_landlock_abi();
    if (abi < kMinimumLandlockAbi) {
        return Support{false, abi, "linux", "x86_64",
                       "Landlock ABI 4 or newer is unavailable"};
    }
    return Support{true, abi, "linux", "x86_64",
                   "exact Linux x86-64 Landlock/seccomp prerequisites are available"};
#elif defined(__APPLE__) && TARGET_OS_OSX
#if defined(__aarch64__) || defined(__arm64__)
    return Support{false, 0, "darwin", "arm64", "Landlock is Linux-only"};
#else
    return Support{false, 0, "darwin", "x86_64", "Landlock is Linux-only"};
#endif
#elif defined(_WIN32)
    return Support{false, 0, "windows", "unknown", "Landlock is Linux-only"};
#else
    return Support{false, 0, "unsupported", "unknown", "unsupported operating system"};
#endif
}

std::unique_ptr<ProcessIsolationCapability> install(const Request& request) {
    validate_request(request);
#if defined(__linux__) && defined(__x86_64__)
    const auto support = query_support();
    if (!support.supported) {
        throw std::runtime_error(support.reason);
    }
    std::int64_t expected = 0;
    if (!installed_pid.compare_exchange_strong(expected, -1)) {
        throw std::runtime_error("process isolation may be installed exactly once per process");
    }
    bool irreversible = false;
    try {
        Receipt receipt;
        receipt.policy_logical_sha256 = request.policy_logical_sha256;
        receipt.backend = request.backend;
        receipt.attempt_kind = request.attempt_kind;
        receipt.selected_artifact_root = request.selected_artifact_root;
        verify_scheduler_boundary(request.backend, receipt);
        receipt.pid = current_pid();
        receipt.landlock_abi = support.landlock_abi;
        receipt.read_execute_path_count = request.read_execute_paths.size();
        receipt.read_only_path_count = request.read_only_paths.size();
        receipt.writable_path_count = request.writable_paths.size();

        if (thread_count() != 1) {
            throw std::runtime_error("native isolation must be installed while single-threaded");
        }
        require_no_child_processes();
        receipt.no_preexisting_child_processes_at_install = true;
        validate_standard_streams(receipt);
        // From this point onward, any failure terminates the process.  Returning
        // to Python after an FD sweep or partial kernel setup would permit a
        // caller to catch the exception and continue without the full boundary.
        irreversible = true;
        close_inherited_descriptors();
        FileDescriptor ruleset =
            build_ruleset(request.read_execute_paths, request.read_only_paths,
                          request.writable_paths, kSelectedRootRights);
        if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
            throw std::runtime_error(errno_message("PR_SET_NO_NEW_PRIVS"));
        }
        if (syscall(__NR_landlock_restrict_self, ruleset.get(), 0) != 0) {
            throw std::runtime_error(errno_message("landlock_restrict_self"));
        }
        ruleset.reset();
        try {
            install_seccomp_filter();
        } catch (...) {
            terminate_partial_installation();
        }
        installed_pid.store(receipt.pid);
        return std::unique_ptr<ProcessIsolationCapability>(
            new ProcessIsolationCapability(std::move(receipt)));
    } catch (...) {
        if (irreversible) {
            terminate_partial_installation();
        }
        installed_pid.store(0);
        throw;
    }
#else
    (void)request;
    throw std::runtime_error("native D1 process isolation requires Linux x86-64");
#endif
}

std::unique_ptr<BuildProcessIsolationCapability> install_build(
    const BuildRequest& request) {
    validate_build_request(request);
#if defined(__linux__) && defined(__x86_64__)
    const auto support = query_support();
    if (!support.supported) {
        throw std::runtime_error(support.reason);
    }
    std::int64_t expected = 0;
    if (!installed_pid.compare_exchange_strong(expected, -1)) {
        throw std::runtime_error("process isolation may be installed exactly once per process");
    }
    bool irreversible = false;
    try {
        BuildReceipt receipt;
        receipt.policy_logical_sha256 = request.policy_logical_sha256;
        receipt.attempt_index = request.attempt_index;
        receipt.attempt_root = request.attempt_root;
        receipt.peer_attempt_root = request.peer_attempt_root;
        receipt.source_root = request.source_root;
        receipt.wheelhouse_root = request.wheelhouse_root;
        verify_scheduler_boundary("goose", receipt);
        receipt.pid = current_pid();
        receipt.landlock_abi = support.landlock_abi;
        receipt.read_execute_path_count = request.read_execute_paths.size();
        receipt.read_only_path_count = request.read_only_paths.size();
        receipt.writable_path_count = request.writable_paths.size();

        if (thread_count() != 1) {
            throw std::runtime_error(
                "native build isolation must be installed while single-threaded");
        }
        require_no_child_processes();
        receipt.no_preexisting_child_processes_at_install = true;
        validate_standard_streams(receipt);
        irreversible = true;
        close_inherited_descriptors();
        FileDescriptor ruleset =
            build_ruleset(request.read_execute_paths, request.read_only_paths,
                          request.writable_paths, kBuildWritableRights);
        if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
            throw std::runtime_error(errno_message("PR_SET_NO_NEW_PRIVS"));
        }
        if (syscall(__NR_landlock_restrict_self, ruleset.get(), 0) != 0) {
            throw std::runtime_error(errno_message("landlock_restrict_self"));
        }
        ruleset.reset();
        try {
            install_seccomp_filter();
        } catch (...) {
            terminate_partial_installation();
        }
        installed_pid.store(receipt.pid);
        return std::unique_ptr<BuildProcessIsolationCapability>(
            new BuildProcessIsolationCapability(std::move(receipt)));
    } catch (...) {
        if (irreversible) {
            terminate_partial_installation();
        }
        installed_pid.store(0);
        throw;
    }
#else
    (void)request;
    throw std::runtime_error("native D1 build process isolation requires Linux x86-64");
#endif
}

}  // namespace lac::minorminer::isolation

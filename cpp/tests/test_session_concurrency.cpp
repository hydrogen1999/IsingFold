#include "lac_minorminer/search_session.hpp"

#include <atomic>
#include <iostream>
#include <thread>

int main() {
    using lac::minorminer::Graph;
    using lac::minorminer::SearchSession;

    SearchSession session(Graph(4, {}), Graph(4, {{0, 1}, {1, 2}, {2, 3}}), 23, 4);
    std::atomic<bool> failed{false};
    std::thread reader([&] {
        try {
            for (int iteration = 0; iteration < 2'000; ++iteration) {
                const auto snapshot = session.snapshot();
                if (snapshot.chains.size() != 4 || snapshot.occupancy.size() != 4) {
                    failed.store(true);
                }
            }
        } catch (...) {
            failed.store(true);
        }
    });

    for (int iteration = 0; iteration < 2'000; ++iteration) {
        session.restart();
    }
    reader.join();
    if (failed.load()) {
        std::cerr << "concurrent reads observed a partial restart\n";
        return 1;
    }
    const auto work = session.work_counters();
    if (work.validator_calls != 2'000 || work.restart_work != 2'000 ||
        work.feature_work != 42'000) {
        std::cerr << "concurrent work ledger lost or duplicated a registered operation\n";
        return 1;
    }
    return 0;
}

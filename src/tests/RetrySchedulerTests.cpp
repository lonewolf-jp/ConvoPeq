#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <mutex>

#include "audioengine/RetryScheduler.h"
#include "audioengine/RetrySchedulerTypes.h"
#include "core/RebuildTypes.h"

struct MockSink {
    std::atomic<int> callCount{0};
    std::vector<RebuildTelemetryReason> reasons;
    std::mutex mtx;
    void onDispatch(const RetryScheduleRequest& req) noexcept {
        callCount.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(mtx);
        reasons.push_back(req.reason);
    }
};

static RetryScheduleRequest makeReq(RebuildTelemetryReason r = RebuildTelemetryReason::EnqueueSnapshotCommand) {
    return RetryScheduleRequest{ convo::RebuildKind::Structural, r, RebuildTelemetryClass::Structural, RebuildTelemetryPolicy::Replaceable };
}

static bool testDelay50() {
    MockSink sink;
    RetryScheduler sched([&](const RetryScheduleRequest& req){ sink.onDispatch(req); });
    auto req = makeReq();
    sched.schedule(req, std::chrono::milliseconds(50));
    if (sched.pendingCount() != 1) { std::cerr << "T1: pendingCount != 1\n"; return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (sink.callCount.load() != 0) { std::cerr << "T1: dispatched before deadline\n"; return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    if (sink.callCount.load() != 1) { std::cerr << "T1: not dispatched after deadline\n"; return false; }
    sched.shutdown();
    std::cout << "T1 PASS\n"; return true;
}

static bool testZeroDelay() {
    MockSink sink;
    RetryScheduler sched([&](const RetryScheduleRequest& req){ sink.onDispatch(req); });
    auto req = makeReq();
    sched.schedule(req, std::chrono::milliseconds(0));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    if (sink.callCount.load() != 1) { std::cerr << "T2: zero delay not dispatched\n"; return false; }
    sched.shutdown();
    std::cout << "T2 PASS\n"; return true;
}

static bool testDeadlineOrdering() {
    MockSink sink;
    RetryScheduler sched([&](const RetryScheduleRequest& req){ sink.onDispatch(req); });
    auto r1 = makeReq(RebuildTelemetryReason::ConvolverParamsChanged);
    auto r2 = makeReq(RebuildTelemetryReason::HashDedup);
    auto r3 = makeReq(RebuildTelemetryReason::SnapshotEnqueued);
    sched.schedule(r1, std::chrono::milliseconds(100));
    sched.schedule(r2, std::chrono::milliseconds(20));
    sched.schedule(r3, std::chrono::milliseconds(50));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::lock_guard<std::mutex> lk(sink.mtx);
    if (sink.reasons.size() != 3) { std::cerr << "T4: reasons size " << sink.reasons.size() << "\n"; return false; }
    if (sink.reasons[0] != RebuildTelemetryReason::HashDedup) { std::cerr << "T4: order 0 wrong\n"; return false; }
    if (sink.reasons[1] != RebuildTelemetryReason::SnapshotEnqueued) { std::cerr << "T4: order 1 wrong\n"; return false; }
    if (sink.reasons[2] != RebuildTelemetryReason::ConvolverParamsChanged) { std::cerr << "T4: order 2 wrong\n"; return false; }
    sched.shutdown();
    std::cout << "T4 PASS\n"; return true;
}

static bool testMultipleProducers() {
    MockSink sink;
    RetryScheduler sched([&](const RetryScheduleRequest& req){ sink.onDispatch(req); });
    std::atomic<bool> failed{false};
    auto producer = [&](int base) {
        for (int i = 0; i < 4; ++i) {
            auto r = makeReq();
            (void)base;
            sched.schedule(r, std::chrono::milliseconds(10 + i * 5));
        }
    };
    std::thread t1([&]{ producer(0); });
    std::thread t2([&]{ producer(10); });
    t1.join(); t2.join();
    if (sched.pendingCount() > 8) { std::cerr << "T5: queue corruption\n"; failed = true; }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (sink.callCount.load() > 8) { std::cerr << "T5: too many dispatches\n"; failed = true; }
    sched.shutdown();
    if (failed) return false;
    std::cout << "T5 PASS\n"; return true;
}

static bool testQueueFull() {
    MockSink sink;
    RetryScheduler sched([&](const RetryScheduleRequest& req){ sink.onDispatch(req); });
    for (int i = 0; i < 8; ++i) {
        sched.schedule(makeReq(), std::chrono::milliseconds(500));
    }
    if (sched.pendingCount() != 8) { std::cerr << "T6a: pending != 8\n"; return false; }
    uint64_t before = sched.rejectCount();
    sched.schedule(makeReq(), std::chrono::milliseconds(500));
    if (sched.pendingCount() != 8) { std::cerr << "T6b: pending changed after reject\n"; return false; }
    if (sched.rejectCount() != before + 1) { std::cerr << "T6c: rejectCount not incremented\n"; return false; }
    sched.shutdown();
    std::cout << "T6 PASS\n"; return true;
}

static bool testShutdown() {
    MockSink sink;
    RetryScheduler sched([&](const RetryScheduleRequest& req){ sink.onDispatch(req); });
    sched.schedule(makeReq(), std::chrono::milliseconds(500));
    if (sched.pendingCount() != 1) { std::cerr << "T7a\n"; return false; }
    sched.shutdown();
    if (sched.pendingCount() != 0) { std::cerr << "T7b\n"; return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (sink.callCount.load() != 0) { std::cerr << "T7c: dispatched after shutdown\n"; return false; }
    sched.shutdown();
    std::cout << "T7 PASS\n"; return true;
}

static bool testConcurrentShutdown() {
    MockSink sink;
    RetryScheduler sched([&](const RetryScheduleRequest& req){ sink.onDispatch(req); });
    std::thread t([&]{ sched.schedule(makeReq(), std::chrono::milliseconds(0)); });
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    sched.shutdown();
    t.join();
    std::cout << "T8 PASS\n"; return true;
}

int main() {
    bool ok = true;
    ok &= testDelay50();
    ok &= testZeroDelay();
    ok &= testDeadlineOrdering();
    ok &= testMultipleProducers();
    ok &= testQueueFull();
    ok &= testShutdown();
    ok &= testConcurrentShutdown();
    std::cout << (ok ? "ALL PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

#pragma once
// RetryScheduler.h — D-5-2 Step 4-A: RetryScheduler minimal scheduler
// RetryScheduleRequest + PendingRetry + RetryScheduler class declarations.
// Depends only on RebuildKind (core/RebuildTypes.h) + telemetry enums.
// Engine dispatch is via injected callback to avoid AudioEngine.h dependency.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "core/RebuildTypes.h"
#include "RetrySchedulerTypes.h"

struct RetryScheduleRequest
{
    convo::RebuildKind kind;
    RebuildTelemetryReason reason;
    RebuildTelemetryClass rebuildClass;
    RebuildTelemetryPolicy collapsePolicy;
};

struct PendingRetry
{
    RetryScheduleRequest request;
    std::chrono::steady_clock::time_point deadline;
};

class RetryScheduler
{
public:
    using DispatchFn = std::function<void(const RetryScheduleRequest&)>;

    explicit RetryScheduler(DispatchFn dispatch) noexcept;
    ~RetryScheduler() noexcept;

    void schedule(RetryScheduleRequest request,
                  std::chrono::milliseconds delay) noexcept;

    void shutdown() noexcept;

    [[nodiscard]] std::size_t pendingCount() const noexcept;
    [[nodiscard]] std::uint64_t rejectCount() const noexcept;

private:
    void run() noexcept;

    DispatchFn dispatch_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<PendingRetry> queue_;

    std::atomic<bool> shouldExit_{false};
    std::atomic<uint64_t> rejectCount_{0};

    std::thread worker_;

    static constexpr std::size_t kCapacity = 8;
};

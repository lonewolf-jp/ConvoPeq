#include "RetryScheduler.h"

RetryScheduler::RetryScheduler(DispatchFn dispatch) noexcept
    : dispatch_(std::move(dispatch))
{
    try {
        worker_ = std::thread(&RetryScheduler::run, this);
    } catch (...) {
        shouldExit_ = true;
    }
}

RetryScheduler::~RetryScheduler() noexcept
{
    shutdown();
}

void RetryScheduler::schedule(RetryScheduleRequest request,
                              std::chrono::milliseconds delay) noexcept
{
    const auto deadline = std::chrono::steady_clock::now() + delay;
    const PendingRetry entry{ request, deadline };

    std::unique_lock<std::mutex> lock(mutex_);
    if (shouldExit_) {
        ++rejectCount_;
        return;
    }
    if (queue_.size() >= kCapacity) {
        ++rejectCount_;
        return;
    }
    try {
        // deadline ascending insertion (stable for equal deadlines)
        auto it = queue_.begin();
        for (; it != queue_.end(); ++it) {
            if (deadline < it->deadline) break;
        }
        queue_.insert(it, entry);
    } catch (...) {
        ++rejectCount_;
        return;
    }
    lock.unlock();
    cv_.notify_all();
}

void RetryScheduler::shutdown() noexcept
{
    bool doJoin = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shouldExit_) return;
        shouldExit_ = true;
        queue_.clear();
        doJoin = true;
    }
    cv_.notify_all();
    if (doJoin && worker_.joinable()) {
        try { worker_.join(); } catch (...) {}
    }
}

std::size_t RetryScheduler::pendingCount() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

uint64_t RetryScheduler::rejectCount() const noexcept
{
    return rejectCount_.load(std::memory_order_relaxed);
}

void RetryScheduler::run() noexcept
{
    while (true) {
        PendingRetry pending{};
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (!shouldExit_ && queue_.empty()) {
                cv_.wait(lock);
            }
            if (shouldExit_) return;
            // wait until front deadline
            const auto deadline = queue_.front().deadline;
            const auto status = cv_.wait_until(lock, deadline);
            if (shouldExit_) return;
            if (status == std::cv_status::timeout) {
                // deadline reached if still front and deadline <= now
                const auto now = std::chrono::steady_clock::now();
                if (queue_.empty()) continue;
                if (queue_.front().deadline > now) continue; // spurious or earlier wake due to new earlier deadline
                pending = queue_.front();
                queue_.pop_front();
            } else {
                // woken by schedule() with earlier deadline or shutdown
                continue;
            }
        }
        // unlock before dispatch to avoid lock-order coupling (single production path via DispatchFn)
        if (dispatch_) {
            dispatch_(pending.request);
        }
    }
}

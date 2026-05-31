#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

namespace convo {

struct DeferredRetireFallbackEntry
{
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    std::uint64_t epoch = 0;
};

class DeferredRetireFallbackQueue
{
public:
    DeferredRetireFallbackQueue() = default;
    DeferredRetireFallbackQueue(const DeferredRetireFallbackQueue&) = delete;
    DeferredRetireFallbackQueue& operator=(const DeferredRetireFallbackQueue&) = delete;

    [[nodiscard]] std::size_t push(DeferredRetireFallbackEntry entry)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(entry);
        return queue_.size();
    }

    [[nodiscard]] std::vector<DeferredRetireFallbackEntry> popAll()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<DeferredRetireFallbackEntry> pending;
        pending.swap(queue_);
        return pending;
    }

    [[nodiscard]] std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    [[nodiscard]] bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::vector<DeferredRetireFallbackEntry> queue_;
};

} // namespace convo

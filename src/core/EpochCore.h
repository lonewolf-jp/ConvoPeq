#pragma once

#include <atomic>
#include <cstdint>
#include <array>
#include <limits>

#include "audioengine/AtomicAccess.h"

namespace convo {

class EpochCore {
public:
    static constexpr int kMaxReaders = 8;
    static constexpr uint64_t kIdleEpoch = 0;

    EpochCore() : epoch(1) {
        for (auto& e : readerEpochs) {
            convo::publishAtomic(e, kIdleEpoch, std::memory_order_release);
        }
    }

    uint64_t current() const noexcept {
        return convo::consumeAtomic(epoch, std::memory_order_acquire);
    }

    uint64_t publish() noexcept {
        return epoch.fetch_add(1, std::memory_order_acq_rel);
    }

    uint64_t getMinReaderEpoch() const noexcept {
        uint64_t minEpoch = std::numeric_limits<uint64_t>::max();
        bool hasActiveReader = false;

        for (const auto& e : readerEpochs) {
            uint64_t r = convo::consumeAtomic(e, std::memory_order_acquire);
            if (r != kIdleEpoch) {
                if (!hasActiveReader) {
                    minEpoch = r;
                    hasActiveReader = true;
                } else if (isOlder(r, minEpoch)) {
                    minEpoch = r;
                }
            }
        }

        if (!hasActiveReader)
            return convo::consumeAtomic(epoch, std::memory_order_acquire);

        return minEpoch;
    }

    static inline bool isOlder(uint64_t a, uint64_t b) noexcept {
        return static_cast<int64_t>(a - b) < 0;
    }

    void enterReader(int readerIndex) noexcept {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return;
        convo::publishAtomic(readerEpochs[static_cast<size_t>(readerIndex)], current(), std::memory_order_release);
    }

    void exitReader(int readerIndex) noexcept {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return;
        convo::publishAtomic(readerEpochs[static_cast<size_t>(readerIndex)], kIdleEpoch, std::memory_order_release);
    }

private:
    std::atomic<uint64_t> epoch{1};
    std::array<std::atomic<uint64_t>, kMaxReaders> readerEpochs;
};

class EpochCoreReaderGuard {
public:
    EpochCoreReaderGuard(EpochCore& coreIn, int readerIndexIn) noexcept
        : core(coreIn), readerIndex(readerIndexIn)
    {
        core.enterReader(readerIndex);
    }

    ~EpochCoreReaderGuard() noexcept
    {
        core.exitReader(readerIndex);
    }

    EpochCoreReaderGuard(const EpochCoreReaderGuard&) = delete;
    EpochCoreReaderGuard& operator=(const EpochCoreReaderGuard&) = delete;

private:
    EpochCore& core;
    int readerIndex;
};

} // namespace convo

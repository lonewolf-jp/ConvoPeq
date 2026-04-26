#pragma once

#include <atomic>
#include <cstdint>
#include <array>
#include <limits>   // ← 追加

namespace convo {

class EpochCore {
public:
    static constexpr int kMaxReaders = 8;
    static constexpr uint64_t kIdleEpoch = 0;

    EpochCore() : epoch(1) {
        for (auto& e : readerEpochs) {
            e.store(kIdleEpoch, std::memory_order_relaxed);
        }
    }

    uint64_t current() const noexcept {
        return epoch.load(std::memory_order_acquire);
    }

    uint64_t publish() noexcept {
        return epoch.fetch_add(1, std::memory_order_acq_rel);
    }

    void enterReader(int index) noexcept {
        if (index >= 0 && index < kMaxReaders) {
            // relaxed load of global epoch is sufficient; release store publishes participation.
            uint64_t e = epoch.load(std::memory_order_relaxed);
            readerEpochs[index].store(e, std::memory_order_release);
        }
    }

    void exitReader(int index) noexcept {
        if (index >= 0 && index < kMaxReaders) {
            // relaxed store is sufficient for exiting participant.
            readerEpochs[index].store(kIdleEpoch, std::memory_order_relaxed);
        }
    }

    uint64_t getMinReaderEpoch() const noexcept {
        uint64_t minEpoch = std::numeric_limits<uint64_t>::max();
        bool hasActiveReader = false;

        for (const auto& e : readerEpochs) {
            uint64_t r = e.load(std::memory_order_acquire);
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
            return epoch.load(std::memory_order_relaxed);

        return minEpoch;
    }

    static inline bool isOlder(uint64_t a, uint64_t b) noexcept {
        return static_cast<int64_t>(a - b) < 0;
    }

private:
    std::atomic<uint64_t> epoch{1};
    std::array<std::atomic<uint64_t>, kMaxReaders> readerEpochs;
};

} // namespace convo

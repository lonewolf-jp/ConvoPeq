//==============================================================================
// EpochManager.h
// ConvoPeq RCU v17.15 - 単一エポック管理 + Watchdog + 登録凍結
// 
// 設計原則:
//   1. スレッド登録は prepareToPlay() 完了時に freeze され、以降の新規登録は abort
//   2. minActiveEpoch 計算から reader を除外することは絶対にない（UAF 防止）
//   3. Reader stuck は watchdog で検出するが、degraded mode で graceful handling
//==============================================================================
#pragma once

#include <atomic>
#include <vector>
#include <mutex>
#include <cstdint>
#include <thread>
#include <chrono>

#include <JuceHeader.h>

namespace convo {

struct alignas(64) ThreadEpoch {
    std::atomic<uint64_t> epoch {0};
    std::atomic<uint64_t> state {0}; // even = inactive, odd = active
    std::atomic<uint64_t> lastSeenTimestamp {0}; // milliseconds
};

class EpochManager {
public:
    static EpochManager& instance();

    ThreadEpoch& registerThread();
    void freezeRegistration();

    uint64_t currentEpoch() const noexcept {
        return m_globalEpoch.load(std::memory_order_acquire);
    }

    void advanceEpoch();
    uint64_t minActiveEpochFast() const noexcept;
    uint64_t minActiveEpochFull();

    std::atomic<bool> shutdownPhase {false};

private:
    EpochManager() = default;
    ~EpochManager() = default;

    std::atomic<uint64_t> m_globalEpoch {1};
    std::vector<ThreadEpoch*> m_threads;
    std::mutex m_threadMutex;
    std::atomic<bool> m_registrationClosed {false};
    std::atomic<uint64_t> m_cachedMinEpoch {1};

    static constexpr uint64_t READER_WATCHDOG_MS = 500;
};

inline void EpochManager::advanceEpoch() {
    m_globalEpoch.fetch_add(1, std::memory_order_acq_rel);
}

inline uint64_t EpochManager::minActiveEpochFast() const noexcept {
    return m_cachedMinEpoch.load(std::memory_order_acquire);
}

} // namespace convo

//==============================================================================
// ReclaimerThread.h
// ConvoPeq RCU v17.15 - 削除スレッド（Graceful Degradation 対応）
// 
// 設計原則:
//   1. Audio Thread は retire を呼ばない（convo::retire() が abort でガード）
//   2. Reader stuck 時には degraded mode へ移行し、stash 急増を抑制
//   3. stash が致命的限界に達した場合は abort で強制終了
//==============================================================================
#pragma once

#include "DeferredDeletionQueue.h"
#include "EpochManager.h"

#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

#include <JuceHeader.h>

namespace convo {

class ReclaimerThread {
public:
    ReclaimerThread();
    ~ReclaimerThread();
    
    void start();
    void shutdown();
    
    // 単一グローバルインスタンスへのアクセス
    static ReclaimerThread& instance();

private:
    static constexpr size_t MAX_BATCH = 256;
    static constexpr size_t STASH_WARN_THRESHOLD = 8192;
    static constexpr size_t STASH_CRITICAL_LIMIT = 1'000'000;
    static constexpr auto SHUTDOWN_TIMEOUT = std::chrono::seconds(1);
    static constexpr std::chrono::milliseconds EPOCH_STALL_THRESHOLD {10};
    static constexpr auto STALL_GRACE_PERIOD = std::chrono::seconds(10);

    enum class Mode { Normal, Degraded };

    void run();
    void drainAllEntries();

    Mode m_mode = Mode::Normal;
    uint64_t m_lastMinEpoch = 0;
    std::chrono::steady_clock::time_point m_lastAdvanceTime;
    
    std::atomic<bool> m_running {false};
    std::thread m_thread;
    std::vector<Retired> m_stash;
};

// グローバル deletion queue への参照
extern DeferredDeletionQueue g_deletionQueue;

} // namespace convo

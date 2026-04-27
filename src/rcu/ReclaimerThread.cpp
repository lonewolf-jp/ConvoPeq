//==============================================================================
// ReclaimerThread.cpp
// ConvoPeq RCU v17.15 - 削除スレッド実装
//==============================================================================
#include "ReclaimerThread.h"

namespace convo {

// グローバル deletion queue の定義
DeferredDeletionQueue g_deletionQueue;

ReclaimerThread& ReclaimerThread::instance() {
    static ReclaimerThread inst;
    return inst;
}

ReclaimerThread::ReclaimerThread() = default;

ReclaimerThread::~ReclaimerThread() {
    if (m_running.load(std::memory_order_acquire)) {
        shutdown();
    }
}

void ReclaimerThread::start() {
    if (m_running.load(std::memory_order_acquire)) {
        return;
    }
    m_running.store(true, std::memory_order_release);
    m_thread = std::thread([this]() { run(); });
}

void ReclaimerThread::shutdown() {
    auto& mgr = EpochManager::instance();
    mgr.shutdownPhase.store(true, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    uint64_t target = mgr.currentEpoch();
    auto start = std::chrono::steady_clock::now();
    
    while (mgr.minActiveEpochFull() <= target) {
        mgr.advanceEpoch();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        if (std::chrono::steady_clock::now() - start > SHUTDOWN_TIMEOUT) {
            juce::Logger::writeToLog("RCU shutdown timeout – aborting to prevent UAF");
            std::abort();
        }
    }
    
    mgr.advanceEpoch();
    drainAllEntries();
    
    m_running.store(false, std::memory_order_release);
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void ReclaimerThread::run() {
    m_lastMinEpoch = 0;
    m_lastAdvanceTime = std::chrono::steady_clock::now();

    while (m_running.load(std::memory_order_acquire)) {
        uint64_t minEpoch = EpochManager::instance().minActiveEpochFull();

        // ---- 停滞検出とモード遷移 ----
        if (minEpoch == m_lastMinEpoch) {
            if (m_mode == Mode::Normal) {
                auto now = std::chrono::steady_clock::now();
                if (now - m_lastAdvanceTime >= STALL_GRACE_PERIOD) {
                    juce::Logger::writeToLog(
                        "ReclaimerThread: entering degraded mode (reclamation stalled)");
                    m_mode = Mode::Degraded;
                }
            }
            
            auto now = std::chrono::steady_clock::now();
            if (now - m_lastAdvanceTime >= EPOCH_STALL_THRESHOLD) {
                EpochManager::instance().advanceEpoch();
                m_lastAdvanceTime = now;
            }
        } else {
            m_lastMinEpoch = minEpoch;
            m_lastAdvanceTime = std::chrono::steady_clock::now();
            
            if (m_mode == Mode::Degraded) {
                juce::Logger::writeToLog("ReclaimerThread: resuming normal mode");
                m_mode = Mode::Normal;
            }
        }

        // ---- 通常の解放処理（Normal モードのみ）----
        if (m_mode == Mode::Normal) {
            auto it = m_stash.begin();
            while (it != m_stash.end()) {
                if (it->epoch < minEpoch) {
                    it->deleter(it->ptr);
                    it = m_stash.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // ---- キューから stash への移動（degraded 時は制限制御）----
        size_t processed = 0;
        Retired r;
        size_t batchLimit = (m_mode == Mode::Normal) ? MAX_BATCH : 1;
        
        while (processed < batchLimit && g_deletionQueue.dequeue(r)) {
            if (m_mode == Mode::Normal && r.epoch < minEpoch) {
                r.deleter(r.ptr);
                ++processed;
            } else {
                if (m_stash.size() >= STASH_CRITICAL_LIMIT) {
                    juce::Logger::writeToLog(
                        "ReclaimerThread: stash critical limit reached – aborting");
                    std::abort();
                }
                m_stash.push_back(r);
            }
        }

        if (m_stash.size() > STASH_WARN_THRESHOLD) {
            juce::Logger::writeToLog("ReclaimerThread: stash size " +
                juce::String(m_stash.size()) + " (stalled?)");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void ReclaimerThread::drainAllEntries() {
    Retired r;
    while (g_deletionQueue.dequeue(r)) {
        r.deleter(r.ptr);
    }
    for (auto& s : m_stash) {
        s.deleter(s.ptr);
    }
    m_stash.clear();
}

} // namespace convo

//==============================================================================
// EpochManager.cpp
// ConvoPeq RCU v17.15 - エポック管理実装
//==============================================================================
#include "EpochManager.h"

namespace convo {

EpochManager& EpochManager::instance() {
    static EpochManager inst;
    return inst;
}

ThreadEpoch& EpochManager::registerThread() {
    std::lock_guard<std::mutex> lock(m_threadMutex);
    
    if (m_registrationClosed.load(std::memory_order_acquire)) {
        jassertfalse; // 凍結後のスレッド登録は致命的エラー
        Logger::writeToLog("EpochManager: Thread registration after freeze - aborting");
        std::abort();
    }
    
    auto* thread = new ThreadEpoch();
    thread->epoch.store(m_globalEpoch.load(std::memory_order_relaxed), std::memory_order_relaxed);
    thread->state.store(0, std::memory_order_relaxed);
    thread->lastSeenTimestamp.store(juce::Time::getMillisecondCounter(), std::memory_order_relaxed);
    m_threads.push_back(thread);
    return *thread;
}

void EpochManager::freezeRegistration() {
    m_registrationClosed.store(true, std::memory_order_release);
}

uint64_t EpochManager::minActiveEpochFull() {
    jassert(m_registrationClosed.load(std::memory_order_acquire));
    
    uint64_t now = juce::Time::getMillisecondCounter();
    uint64_t min = m_globalEpoch.load(std::memory_order_acquire);
    
    std::lock_guard<std::mutex> lock(m_threadMutex);
    for (auto* t : m_threads) {
        uint64_t s1 = t->state.load(std::memory_order_acquire);
        if (s1 & 1) { // active (odd)
            uint64_t lastSeen = t->lastSeenTimestamp.load(std::memory_order_acquire);
            if (now - lastSeen > READER_WATCHDOG_MS) {
                Logger::writeToLog("EpochManager: Reader possibly stuck – not excluding from minEpoch");
            }
            uint64_t e = t->epoch.load(std::memory_order_acquire);
            uint64_t s2 = t->state.load(std::memory_order_acquire);
            if (s1 == s2 && (s1 & 1)) { // state unchanged and still active
                if (e < min) min = e;
            }
        }
    }
    
    m_cachedMinEpoch.store(min, std::memory_order_release);
    return min;
}

} // namespace convo

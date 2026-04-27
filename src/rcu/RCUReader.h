//==============================================================================
// RCUReader.h
// ConvoPeq RCU v17.15 - Reader Guard（値オブジェクト、ネスト対応）
// 
// 使用法:
//   thread_local RCUReader tls_reader;
//   
//   void audioCallback() {
//       RCUReaderGuard guard(tls_reader);
//       // ... RCU protected read section ...
//   }
//==============================================================================
#pragma once

#include "EpochManager.h"

namespace convo {

class RCUReader {
    ThreadEpoch& m_epoch;
public:
    RCUReader() : m_epoch(EpochManager::instance().registerThread()) {}
    ThreadEpoch& epoch() { return m_epoch; }
};

class RCUReaderGuard {
    ThreadEpoch& m_epoch;
    static thread_local int s_depth;

public:
    RCUReaderGuard(RCUReader& reader) : m_epoch(reader.epoch()) {
        if (s_depth++ == 0) {
            uint64_t e = EpochManager::instance().currentEpoch();
            m_epoch.epoch.store(e, std::memory_order_release);
            m_epoch.lastSeenTimestamp.store(
                juce::Time::getMillisecondCounter(), std::memory_order_release);
            m_epoch.state.fetch_add(1, std::memory_order_release);  // enter (odd)
        }
    }
    
    ~RCUReaderGuard() {
        if (--s_depth == 0) {
            m_epoch.state.fetch_add(1, std::memory_order_release);  // exit (even)
        }
    }
    
    void notifyAlive() noexcept {
        m_epoch.lastSeenTimestamp.store(
            juce::Time::getMillisecondCounter(), std::memory_order_release);
    }
    
    RCUReaderGuard(const RCUReaderGuard&) = delete;
    RCUReaderGuard& operator=(const RCUReaderGuard&) = delete;
};

thread_local int RCUReaderGuard::s_depth = 0;

} // namespace convo

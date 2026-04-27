//==============================================================================
// retire.h
// ConvoPeq RCU v17.15 - 型付き retire API
// 
// 使用法:
//   convo::retire(ptr);
// 
// 制約:
//   - Audio Thread から呼ぶと abort（isAudioThread() でガード）
//   - ptr は最外殻オブジェクト（デストラクタで全内部リソースを解放）
//   - デストラクタ内で retire を呼ばない
//==============================================================================
#pragma once

#include "EpochManager.h"
#include "DeferredDeletionQueue.h"
#include "ReclaimerThread.h"

#include <cstdint>
#include <atomic>

namespace convo {

// Audio Thread 判定関数（AudioEngine.cpp で定義）
bool isAudioThread() noexcept;

template <typename T>
struct TypedDeleter {
    static void destroy(void* p) noexcept {
        delete static_cast<T*>(p);
    }
};

inline std::atomic<size_t>& retireCounter() {
    static std::atomic<size_t> counter {0};
    return counter;
}

template <typename T>
void retire(T* ptr) noexcept {
    if (isAudioThread()) {
        jassertfalse;
        std::abort();  // Audio Thread からの retire は致命的エラー
    }
    
    if (!ptr) return;
    
    auto& mgr = EpochManager::instance();
    if (mgr.shutdownPhase.load(std::memory_order_acquire)) {
        std::abort();  // シャットダウン中の retire もエラー
    }
    
    uint64_t e = mgr.currentEpoch();
    Retired entry {
        static_cast<void*>(ptr),
        &TypedDeleter<T>::destroy,
        e
    };
    
    if (g_deletionQueue.try_enqueue(entry)) {
        if (++retireCounter() % 1024 == 0) {
            mgr.advanceEpoch();
        }
        return;
    }
    
    // fallback: 確実に投入（スピン＋yield）
    g_deletionQueue.enqueue(entry);
    if (++retireCounter() % 1024 == 0) {
        mgr.advanceEpoch();
    }
}

} // namespace convo

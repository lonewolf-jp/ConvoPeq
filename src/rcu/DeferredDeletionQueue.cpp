//==============================================================================
// DeferredDeletionQueue.cpp
// ConvoPeq RCU v17.15 - Vyukov 型 Bounded MPSC キュー実装
//==============================================================================
#include "DeferredDeletionQueue.h"
#include <thread>

namespace convo {

DeferredDeletionQueue::DeferredDeletionQueue() {
    for (size_t i = 0; i < CAPACITY; ++i) {
        m_sequence[i].store(i, std::memory_order_relaxed);
    }
}

void DeferredDeletionQueue::enqueue(const Retired& r) noexcept {
    // fetch_add で一意のスロットを取得
    size_t head = m_head.fetch_add(1, std::memory_order_acq_rel);
    int spin = 0;
    
    // 前のデータが消費者に読まれるのを待つ
    while (m_sequence[head & MASK].load(std::memory_order_acquire) != head) {
        cpu_pause();
        if (++spin > SPIN_LIMIT) {
            std::this_thread::yield();
            spin = 0;
        }
    }
    
    // スロットが空いたのでデータを書き込む
    m_buffer[head & MASK] = r;
    m_sequence[head & MASK].store(head + 1, std::memory_order_release);
}

bool DeferredDeletionQueue::try_enqueue(const Retired& r) noexcept {
    size_t head = m_head.load(std::memory_order_relaxed);
    
    while (true) {
        size_t tail = m_tail.load(std::memory_order_acquire);
        
        // 満杯チェック
        if (head - tail >= CAPACITY) {
            return false;
        }
        
        // スロットが解放済みか（sequence == head なら書き込み可能）
        if (m_sequence[head & MASK].load(std::memory_order_acquire) != head) {
            return false;  // まだ古いデータが消費されていない
        }
        
        // CAS で head を進め、スロットを確保
        if (m_head.compare_exchange_weak(head, head + 1,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
            // スロットを確保したので、必ずデータを書き込む
            m_buffer[head & MASK] = r;
            m_sequence[head & MASK].store(head + 1, std::memory_order_release);
            return true;
        }
        // CAS に失敗した場合、head は更新されているので再試行
    }
}

bool DeferredDeletionQueue::dequeue(Retired& r) noexcept {
    size_t tail = m_tail.load(std::memory_order_relaxed);
    
    // データがあるかチェック
    if (m_sequence[tail & MASK].load(std::memory_order_acquire) != tail + 1) {
        return false;
    }
    
    r = m_buffer[tail & MASK];
    m_sequence[tail & MASK].store(tail + CAPACITY, std::memory_order_release);
    m_tail.store(tail + 1, std::memory_order_release);
    return true;
}

size_t DeferredDeletionQueue::size() const noexcept {
    return m_head.load(std::memory_order_acquire) - m_tail.load(std::memory_order_acquire);
}

} // namespace convo

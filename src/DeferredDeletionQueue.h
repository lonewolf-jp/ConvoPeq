//============================================================================
// DeferredDeletionQueue.h
//
// ロックフリー MPMC 削除キュー
//
// B23: 外部依存関係の明示と RT 保証の静的チェック
// - 本クラスは将来的に moodycamel::ConcurrentQueue への移行を想定しています。
// - 現状はカスタム MPMC 実装を使用していますが、エントリのトリビアルコピー可能性を維持します。
//============================================================================

#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <type_traits>

// DeletionEntry: 削除対象のエントリ
// B23: static_assert を通すため、std::atomic を含まずトリビアルコピー可能である必要があります。
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
};

// B23: リアルタイム安全性のための静的チェック
static_assert(std::is_trivially_copyable_v<DeletionEntry>,
    "DeletionEntry must be trivially copyable for lock-free queue operations");
static_assert(std::is_trivially_destructible_v<DeletionEntry>,
    "DeletionEntry must be trivially destructible");
static_assert(std::atomic<size_t>::is_always_lock_free,
    "std::atomic<size_t> must be lock-free for real-time safety");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
    "std::atomic<uint64_t> must be lock-free for real-time safety");

#pragma warning(push)
#pragma warning(disable : 4324) // 「構造体がパッドされました」を無視
/**
 * DeferredDeletionQueue: ロックフリー MPMC 削除キュー (B9 修正版)
 *
 * Dmitry Vyukov の bounded MPMC queue アルゴリズムを採用。
 * シーケンス番号をエントリ外に配置することで、エントリ自体のトリビアルコピー可能性を確保しています。
 */
class DeferredDeletionQueue {
public:
    DeferredDeletionQueue() noexcept {
        for (uint32_t i = 0; i < kQueueSize; ++i)
            sequences[i].store(i, std::memory_order_relaxed);
    }

    // Audio Thread から呼ばれる。ロックフリー。
    bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept {
        uint32_t pos = enqueuePos.load(std::memory_order_relaxed);
        while (true) {
            auto& seq_atom = sequences[pos & kMask];
            uint32_t seq = seq_atom.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                if (enqueuePos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    auto& entry = ringBuffer[pos & kMask];
                    entry.ptr = ptr;
                    entry.deleter = deleter;
                    entry.epoch = epoch;
                    seq_atom.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false; // Full
            } else {
                pos = enqueuePos.load(std::memory_order_relaxed);
            }
        }
    }

    // Message Thread / Timer から呼ばれる。
    void reclaim(uint64_t minEpoch) {
        uint32_t pos = dequeuePos.load(std::memory_order_relaxed);
        while (true) {
            auto& seq_atom = sequences[pos & kMask];
            uint32_t seq = seq_atom.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                auto& entry = ringBuffer[pos & kMask];
                if (entry.epoch < minEpoch) {
                    if (dequeuePos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                        if (entry.deleter && entry.ptr) {
                            entry.deleter(entry.ptr);
                        }
                        entry.ptr = nullptr;
                        entry.deleter = nullptr;
                        seq_atom.store(pos + kQueueSize, std::memory_order_release);
                        pos++;
                    }
                } else {
                    break; // Grace period
                }
            } else {
                break; // Empty
            }
        }
    }

    // Shutdown 専用: epoch 判定を無視して全エントリを回収する。
    void reclaimAllIgnoringEpoch() {
        uint32_t pos = dequeuePos.load(std::memory_order_relaxed);
        while (true) {
            auto& seq_atom = sequences[pos & kMask];
            uint32_t seq = seq_atom.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                auto& entry = ringBuffer[pos & kMask];
                if (dequeuePos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    if (entry.deleter && entry.ptr) {
                        entry.deleter(entry.ptr);
                    }
                    entry.ptr = nullptr;
                    entry.deleter = nullptr;
                    seq_atom.store(pos + kQueueSize, std::memory_order_release);
                    pos++;
                }
            } else {
                break; // Empty
            }
        }
    }

private:
    static constexpr uint32_t kQueueSize = 4096;
    static constexpr uint32_t kMask = kQueueSize - 1;

    alignas(64) std::array<DeletionEntry, kQueueSize> ringBuffer;
    alignas(64) std::array<std::atomic<uint32_t>, kQueueSize> sequences;
    alignas(64) std::atomic<uint32_t> enqueuePos{0};
    alignas(64) std::atomic<uint32_t> dequeuePos{0};
};
#pragma warning(pop)

extern DeferredDeletionQueue g_deletionQueue;

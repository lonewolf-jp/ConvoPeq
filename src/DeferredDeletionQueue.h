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

#include "audioengine/AtomicAccess.h"

enum class DeletionEntryType : uint8_t {
    Generic = 0
};

// DeletionEntry: 削除対象のエントリ
// B23: static_assert を通すため、std::atomic を含まずトリビアルコピー可能である必要があります。
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    uint64_t publicationSequenceId{0};  // ★ P2-2: 出版-退役の因果追跡用
    uint64_t generation{0};             // ★ P2-2: generation 追跡
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

#pragma warning(push) // C4324 suppression scope begin: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
#pragma warning(disable : 4324) // Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容
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
            convo::publishAtomic(sequences[i], i, std::memory_order_release); // release: enqueue/dequeue の seq acquire と HB (初期化後の最初の観測を保証)
    }

    // Audio Thread から呼ばれる。ロックフリー。
    bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept {
        return enqueue(ptr, deleter, epoch, DeletionEntryType::Generic, 0, 0);
    }

    // Audio Thread から呼ばれる。ロックフリー。
    bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type) noexcept {
        return enqueue(ptr, deleter, epoch, type, 0, 0);
    }

    // ★ P2-2: publicationSequenceId + generation 付き enqueue
    bool enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type,
                 uint64_t publicationSequenceId, uint64_t generation) noexcept {
        uint32_t pos = convo::consumeAtomic(enqueuePos, std::memory_order_acquire); // acquire: 前回 enqueue の CAS acq_rel と HB し最新の enqueuePos を観測
        while (true) {
            auto& seq_atom = sequences[pos & kMask];
            uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire); // acquire: dequeue の seq release と HB しスロット解放を観測
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                if (convo::compareExchangeAtomic(enqueuePos,
                                                 pos,
                                                 static_cast<uint32_t>(pos + 1),
                                                 std::memory_order_acq_rel,  // 成功時 acq_rel: acquire で dequeue release と HB; release で次回 enqueue acquire と HB
                                                 std::memory_order_acquire)) { // 失敗時 acquire: 最新 enqueuePos を観測して再試行
                    auto& entry = ringBuffer[pos & kMask];
                    entry.ptr = ptr;
                    entry.deleter = deleter;
                    entry.epoch = epoch;
                    entry.type = type;
                    entry.publicationSequenceId = publicationSequenceId;
                    entry.generation = generation;
                    convo::publishAtomic(seq_atom, pos + 1, std::memory_order_release); // release: dequeue の seq acquire と HB しエントリ書き込み完了を公知
                    return true;
                }
            } else if (diff < 0) {
                return false; // Full
            } else {
                pos = convo::consumeAtomic(enqueuePos, std::memory_order_acquire); // acquire: CAS 失敗後 retry — 最新 enqueuePos を再観測
            }
        }
    }

    // Message Thread / Timer から呼ばれる。
    void reclaim(uint64_t minReaderEpoch) {
        constexpr int kMaxScan = 1024;
        uint32_t deqPos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire); // acquire: 前回 dequeue の CAS release と HB し最新の dequeuePos を観測
        uint32_t scanPos = deqPos;
        int scanned = 0;

        while (scanned < kMaxScan) {
            auto& seq_atom = sequences[scanPos & kMask];
            const uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire); // acquire: enqueue の seq release と HB しエントリ書き込み完了を観測
            const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(scanPos + 1);

            if (diff != 0) {
                break; // Empty
            }

            auto& entry = ringBuffer[scanPos & kMask];
            bool canDelete = false;

            if (isOlder(entry.epoch, minReaderEpoch)) {
                canDelete = true;
            }

            // FIFO を維持するため、現在の dequeue 先頭と一致した時だけ削除する。
            if (canDelete && scanPos == deqPos) {
                if (convo::compareExchangeAtomic(dequeuePos,
                                                 deqPos,
                                                 static_cast<uint32_t>(deqPos + 1),
                                                 std::memory_order_release,  // 成功時 release: 次回 dequeue/drainAllUnsafe の acquire と HB しスロット解放を公知
                                                 std::memory_order_acquire)) { // 失敗時 acquire: 最新 dequeuePos を観測して再試行
                    if (entry.deleter && entry.ptr) {
                        entry.deleter(entry.ptr);
                    }
                    entry.ptr = nullptr;
                    entry.deleter = nullptr;
                    entry.type = DeletionEntryType::Generic;
                    convo::publishAtomic(seq_atom, scanPos + kQueueSize, std::memory_order_release); // release: enqueue の seq acquire と HB しスロット再利用可能を公知

                    ++deqPos;        // [BUG-03] dequeuePos の新値 (deqPos+1) に追従
                    scanPos = deqPos;
                    scanned = 0;
                } else {
                    deqPos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire); // acquire: CAS 失敗後の再観測 — 最新 dequeuePos を取得
                    scanPos = deqPos;
                    scanned = 0;
                }
            } else {
                if (scanPos - deqPos > static_cast<uint32_t>(kMaxScan)) {
                    scanPos = deqPos;
                } else {
                    ++scanPos;
                }
                ++scanned;
            }
        }
    }

    // Shutdown 専用: epoch 判定を無視して全エントリを回収する。
    void drainAllUnsafe() {
        uint32_t pos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire); // acquire: 前回 dequeue の CAS release と HB し最新 dequeuePos を観測
        while (true) {
            auto& seq_atom = sequences[pos & kMask];
            uint32_t seq = convo::consumeAtomic(seq_atom, std::memory_order_acquire); // acquire: enqueue の seq release と HB しエントリ書き込み完了を確認
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                auto& entry = ringBuffer[pos & kMask];
                if (convo::compareExchangeAtomic(dequeuePos,
                                                 pos,
                                                 static_cast<uint32_t>(pos + 1),
                                                 std::memory_order_acq_rel,  // 成功時 acq_rel: acquire で enqueue acq_rel と HB; release で次回ドレインの acquire と HB
                                                 std::memory_order_acquire)) { // 失敗時 acquire: 最新 dequeuePos を観測して retry
                    if (entry.deleter && entry.ptr) {
                        entry.deleter(entry.ptr);
                    }
                    entry.ptr = nullptr;
                    entry.deleter = nullptr;
                    entry.type = DeletionEntryType::Generic;
                    convo::publishAtomic(seq_atom, pos + kQueueSize, std::memory_order_release); // release: 次の enqueue の seq acquire と HB しスロット再利用可能を公知
                    pos++;
                }
            } else {
                break; // Empty
            }
        }
    }

    [[nodiscard]] uint32_t sizeApprox() const noexcept
    {
        const uint32_t enq = convo::consumeAtomic(enqueuePos, std::memory_order_acquire);
        const uint32_t deq = convo::consumeAtomic(dequeuePos, std::memory_order_acquire);
        return static_cast<uint32_t>(enq - deq);
    }

    // ★ P1-7: 最大滞留時間 (us) 追跡
    [[nodiscard]] uint64_t getMaxRetireAgeUs() const noexcept {
        return convo::consumeAtomic(maxRetireAgeUs_, std::memory_order_acquire);
    }

    void updateMaxRetireAge(uint64_t ageUs) noexcept {
        uint64_t current = convo::consumeAtomic(maxRetireAgeUs_, std::memory_order_acquire);
        while (ageUs > current) {
            if (convo::compareExchangeAtomic(maxRetireAgeUs_, current, ageUs,
                    std::memory_order_acq_rel, std::memory_order_acquire))
                break;
        }
    }

    void clearMaxRetireAge() noexcept {
        convo::publishAtomic(maxRetireAgeUs_, static_cast<uint64_t>(0), std::memory_order_release);
    }

private:
    static inline bool isOlder(uint64_t a, uint64_t b) noexcept
    {
        return static_cast<int64_t>(a - b) < 0;
    }

    static constexpr uint32_t kQueueSize = 4096;
    static constexpr uint32_t kMask = kQueueSize - 1;

    alignas(64) std::array<DeletionEntry, kQueueSize> ringBuffer;
    alignas(64) std::array<std::atomic<uint32_t>, kQueueSize> sequences;
    alignas(64) std::atomic<uint32_t> enqueuePos{0};
    alignas(64) std::atomic<uint32_t> dequeuePos{0};
    alignas(64) std::atomic<uint64_t> maxRetireAgeUs_{0};
};
#pragma warning(pop) // C4324 suppression scope end: Intentional alignas padding for cache-line isolation / alignas による意図的なパディングを許容

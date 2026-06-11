#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>

#include "../DeferredDeletionQueue.h"
#include "IEpochProvider.h"
#include "audioengine/AtomicAccess.h"

namespace convo {

class EpochDomain : public IEpochProvider
{
public:
    static constexpr int kMaxReaders = 64;
    static constexpr uint64_t kInactiveEpoch = std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t kReservedEpoch = std::numeric_limits<uint64_t>::max() - 1;

    EpochDomain() : globalEpoch(1)
    {
        for (auto& slot : readers)
        {
            // release: コンストラクタ内で単一スレッドから初期化するが、
            //          完了後に他スレッドがオブジェクトを取得する際に acquire で可視性を保証するため release。
            convo::publishAtomic(slot.epoch, kInactiveEpoch, std::memory_order_release);
            convo::publishAtomic(slot.depth, static_cast<uint32_t>(0), std::memory_order_release);
        }
    }

    int registerReaderThread() noexcept override
    {
        for (int i = 0; i < kMaxReaders; ++i)
        {
            uint64_t expected = kInactiveEpoch;
            // acq_rel/acquire: 成功側 release で slot 取得を他スレッドに公開し、
            //                  failure 側 acquire で競合の write を観測してループを継続。
            if (convo::compareExchangeAtomic(readers[static_cast<size_t>(i)].epoch,
                                             expected,
                                             kReservedEpoch,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
            {
                // release: depth ゼロ化を slot 取得後に他スレッドが観測できるよう publish。
                convo::publishAtomic(readers[static_cast<size_t>(i)].depth,
                                     static_cast<uint32_t>(0),
                                     std::memory_order_release);
                return i;
            }
        }

        return -1;
    }

    bool reserveReaderThread(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return false;

        uint64_t expected = kInactiveEpoch;
        // acq_rel/acquire: registerReaderThread と同じ HB 保証が必要。
        //   成功側 release で予約を公開し、failure 側 acquire で競合書き込みを観測。
        const bool reserved = convo::compareExchangeAtomic(
            readers[static_cast<size_t>(readerIndex)].epoch,
            expected,
            kReservedEpoch,
            std::memory_order_acq_rel,
            std::memory_order_acquire);

        if (reserved)
        {
            // release: depth ゼロ化を他スレッドが観測できるよう予約成功後に publish。
            convo::publishAtomic(readers[static_cast<size_t>(readerIndex)].depth,
                                 static_cast<uint32_t>(0),
                                 std::memory_order_release);
        }

        return reserved;
    }

    [[deprecated("Use RCUReader::enter() instead. See refactoring_plan.md P1-18.")]]
    void enterReader(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return;

        auto& slot = readers[static_cast<size_t>(readerIndex)];
        // acq_rel: 取得側 acquire で直前の exitReader release を観測し、
        //          放出側 release で後続の epoch load が depth > 0 可視後に行われることを保証。
        const uint32_t previousDepth = convo::fetchAddAtomic(slot.depth,
                                                              static_cast<uint32_t>(1),
                                                              std::memory_order_acq_rel);
        if (previousDepth > 0)
            return;

        const uint64_t epoch = currentEpoch();
        // release: epoch を publish することで reclaimers が slot.epoch の safe-below 判定に使用可能となる。
        convo::publishAtomic(slot.epoch, epoch, std::memory_order_release);
    }

    [[deprecated("Use RCUReader::exit() instead. See refactoring_plan.md P1-18.")]]
    void exitReader(int readerIndex) noexcept override
    {
        if (readerIndex < 0 || readerIndex >= kMaxReaders)
            return;

        auto& slot = readers[static_cast<size_t>(readerIndex)];
        // acq_rel: 取得側 acquire で enterReader 以降の読み取りが完了していることを観測し、
        //          放出側 release でその読み取りが slot.epoch の inactive 化より先に完了することを保証。
        const uint32_t previousDepth = convo::fetchSubAtomic(slot.depth,
                                                              static_cast<uint32_t>(1),
                                                              std::memory_order_acq_rel);
        if (previousDepth == 0)
        {
            convo::publishAtomic(slot.depth, static_cast<uint32_t>(0), std::memory_order_release);
            return;
        }

        if (previousDepth > 1)
            return;

        // release: epoch を kInactiveEpoch に戻し、reclaimers がこのスロットを safe-below 判定から除外可能にする。
        convo::publishAtomic(slot.epoch, kInactiveEpoch, std::memory_order_release);
    }

    uint64_t currentEpoch() const noexcept override
    {
        // acquire: advanceEpoch の acq_rel release-side と HB し、最新 epoch を観測する。
        return convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
    }

    // [work21] [[deprecated]] — transitional; callers migrated to publishEpoch()
    [[deprecated("Use Router::publishEpoch() instead. See refactoring_plan.md P0-8.")]]
    uint64_t advanceEpoch() noexcept
    {
        return convo::fetchAddAtomic(globalEpoch,
                                     static_cast<uint64_t>(1),
                                     std::memory_order_acq_rel);
    }

    // [work21] IEpochProvider::publishEpoch — inline advance to avoid deprecated call
    uint64_t publishEpoch() noexcept override
    {
        return convo::fetchAddAtomic(globalEpoch,
                                     static_cast<uint64_t>(1),
                                     std::memory_order_acq_rel);
    }

    uint64_t current() const noexcept
    {
        return currentEpoch();
    }

    uint64_t publish() noexcept
    {
        return publishEpoch();
    }

    uint64_t getMinReaderEpoch() const noexcept override
    {
        uint64_t minEpoch = currentEpoch();

        for (const auto& slot : readers)
        {
            // acquire: enterReader release の depth 書き込みと HB し、depth 読み取り後に epoch を読む。
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
            if (depth == 0)
                continue;

            // acquire: enterReader の epoch publish release と HB し、安全に epoch 値を取得。
            const uint64_t epoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            if (epoch == kInactiveEpoch || epoch == kReservedEpoch)
                continue;

            if (isOlder(epoch, minEpoch))
                minEpoch = epoch;
        }

        return minEpoch;
    }

    uint32_t activeReaderCount() const noexcept override
    {
        uint32_t count = 0;

        for (const auto& slot : readers)
        {
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
            if (depth != 0)
                ++count;
        }

        return count;
    }

    [[deprecated("Use ISR RuntimePublicationCoordinator::enqueueRetire")]] bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override
    {
        return deferredDeletionQueue.enqueue(ptr, deleter, epoch);
    }

    // [work21] [[deprecated]] — transitional; callers migrated to Router 4-param overload
    [[deprecated("Use ISR RuntimePublicationCoordinator::enqueueRetire")]] bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type) noexcept
    {
        return deferredDeletionQueue.enqueue(ptr, deleter, epoch, type);
    }

    // [work21] [[deprecated]] — transitional; callers migrated to tryReclaim()
    [[deprecated("Use requestReclaim() instead. See refactoring_plan.md P0-6.")]]
    void reclaimRetired() noexcept
    {
        deferredDeletionQueue.reclaim(getMinReaderEpoch());
    }

    // [work21] IEpochProvider::tryReclaim — inline reclaim to avoid deprecated call
    void tryReclaim() noexcept override
    {
        deferredDeletionQueue.reclaim(getMinReaderEpoch());
    }

    void drainAll() noexcept
    {
        deferredDeletionQueue.drainAllUnsafe();
    }

    [[nodiscard]] uint32_t pendingRetireCount() const noexcept
    {
        return deferredDeletionQueue.sizeApprox();
    }

    static bool isOlder(uint64_t a, uint64_t b) noexcept
    {
        return static_cast<int64_t>(a - b) < 0;
    }

private:
    struct ReaderSlot
    {
        std::atomic<uint64_t> epoch { kInactiveEpoch };
        std::atomic<uint32_t> depth { 0 };
        std::atomic<uint64_t> enterCount { 0 };  // ★ P3-1: enter 回数のみカウント（軽量）
        std::atomic<uint64_t> residencyStartTimestampUs { 0 }; // ★ P4.5: steady_clock ベースの滞留開始時刻
    };

    // ★ P3-1: 複合判定による Reader Stuck 検出
    struct StuckReaderInfo {
        int readerIndex{-1};
        uint64_t readerEpoch{0};
        uint64_t enterCount{0};
        uint64_t currentEpoch{0};
        uint64_t minReaderEpoch{0};
        uint32_t pendingRetireCount{0};
        bool isStuck{false};
        uint64_t residencyTimeUs{0}; // ★ P4.5: 実時間ベース滞留時間
    };

    [[nodiscard]] StuckReaderInfo detectStuckReaders(uint64_t stuckThreshold) const noexcept {
        StuckReaderInfo info;
        info.currentEpoch = convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
        info.minReaderEpoch = getMinReaderEpoch();
        info.pendingRetireCount = deferredDeletionQueue.sizeApprox();

        const auto nowUs = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());

        for (int i = 0; i < kMaxReaders; ++i) {
            const auto& slot = readers[i];
            const uint64_t readerEpoch = convo::consumeAtomic(slot.epoch, std::memory_order_acquire);
            if (readerEpoch == kInactiveEpoch)
                continue;

            const uint64_t ec = slot.enterCount.load(std::memory_order_relaxed);
            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);

            // ★ P4.5: residencyTime を実時間ベースで計算（epoch差ではなくsteady_clock）
            const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, std::memory_order_acquire);
            const uint64_t residencyUs = (startUs != 0 && depth > 0) ? (nowUs - startUs) : 0;

            // 複合判定: enterCount + epoch 長時間滞留 + depth + pendingRetire
            if (depth > 0 && readerEpoch < info.currentEpoch) {
                const uint64_t epochGap = info.currentEpoch - readerEpoch;
                if (epochGap > stuckThreshold) {
                    info.readerIndex = i;
                    info.readerEpoch = readerEpoch;
                    info.enterCount = ec;
                    info.isStuck = true;
                    info.residencyTimeUs = residencyUs;
                    break;
                }
            }
        }
        return info;
    }

    std::atomic<uint64_t> globalEpoch;
    std::array<ReaderSlot, kMaxReaders> readers;
    DeferredDeletionQueue deferredDeletionQueue;
};

} // namespace convo

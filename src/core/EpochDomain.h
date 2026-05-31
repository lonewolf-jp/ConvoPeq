#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>

#include "../DeferredDeletionQueue.h"
#include "audioengine/AtomicAccess.h"

namespace convo {

class EpochDomain
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

    int registerReaderThread() noexcept
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

    bool reserveReaderThread(int readerIndex) noexcept
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

    void enterReader(int readerIndex) noexcept
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

    void exitReader(int readerIndex) noexcept
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

    uint64_t currentEpoch() const noexcept
    {
        // acquire: advanceEpoch の acq_rel release-side と HB し、最新 epoch を観測する。
        return convo::consumeAtomic(globalEpoch, std::memory_order_acquire);
    }

    uint64_t advanceEpoch() noexcept
    {
        // acq_rel: 取得側 acquire で直前の retire 操作を観測し、
        //          解放側 release で新 epoch 値を他スレッドに可視化する。
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
        return advanceEpoch();
    }

    uint64_t getMinReaderEpoch() const noexcept
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

    uint32_t activeReaderCount() const noexcept
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

    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
    {
        return deferredDeletionQueue.enqueue(ptr, deleter, epoch);
    }

    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type) noexcept
    {
        return deferredDeletionQueue.enqueue(ptr, deleter, epoch, type);
    }

    void reclaimRetired() noexcept
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
    };

    std::atomic<uint64_t> globalEpoch;
    std::array<ReaderSlot, kMaxReaders> readers;
    DeferredDeletionQueue deferredDeletionQueue;
};

class EpochDomainReaderGuard
{
public:
    EpochDomainReaderGuard(EpochDomain& domainIn, int readerIndexIn) noexcept
        : domain(&domainIn), readerIndex(readerIndexIn), active(true)
    {
        domain->enterReader(readerIndex);
    }

    ~EpochDomainReaderGuard() noexcept
    {
        if (active && domain != nullptr)
            domain->exitReader(readerIndex);
    }

    EpochDomainReaderGuard(const EpochDomainReaderGuard&) = delete;
    EpochDomainReaderGuard& operator=(const EpochDomainReaderGuard&) = delete;

    EpochDomainReaderGuard(EpochDomainReaderGuard&& other) noexcept
        : domain(other.domain), readerIndex(other.readerIndex), active(other.active)
    {
        other.domain = nullptr;
        other.readerIndex = -1;
        other.active = false;
    }

    EpochDomainReaderGuard& operator=(EpochDomainReaderGuard&& other) noexcept
    {
        if (this == &other)
            return *this;

        if (active && domain != nullptr)
            domain->exitReader(readerIndex);

        domain = other.domain;
        readerIndex = other.readerIndex;
        active = other.active;

        other.domain = nullptr;
        other.readerIndex = -1;
        other.active = false;
        return *this;
    }

private:
    EpochDomain* domain = nullptr;
    int readerIndex = -1;
    bool active = false;
};

} // namespace convo

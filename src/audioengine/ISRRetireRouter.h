#pragma once

#include <atomic>
#include <cstdint>
#include <cassert>
#include <functional>

// ISR P1-19: 公開APIに EpochDomain 型を露出しない。
// コンストラクタは IEpochProvider& を受け取り、内部でダウンキャストする。
#include "DeferredDeletionQueue.h" // DeletionEntryType
#include "core/IEpochProvider.h"
#include "ISRAuthorityClass.h"

namespace convo {
namespace isr {

// [work21 P1-3] Retire lifecycle states
enum class RetireState : uint8_t
{
    Created = 0,
    Active,
    PendingRetire,
    Retiring,
    Reclaimed
};

// [work21] Forward declarations for Policy lanes
class DSPRetirePolicy;
class SnapshotRetirePolicy;
class DeferredRetirePolicy;

/**
 * ISRRetireRouter — Thin stateless dispatcher for retire operations.
 *
 * [work21 P0-1] Design constraints:
 *   Allowed: route / enqueue / observer factory
 *   Forbidden: state / policy / decision (delegated to Policy lanes)
 *
 * This class is the SINGLE public entry point for all retire operations.
 * It wraps EpochDomain internally so that callers do not need direct
 * EpochDomain reference. Policy lanes (DSPRetirePolicy, SnapshotRetirePolicy,
 * DeferredRetirePolicy) handle actual execution logic.
 *
 * Phase-C target: All EpochDomain direct call sites migrate to this API.
 *
 * ISR P1-19 conformance: EpochDomain 完全型は .cpp のみでインクルード。
 *   .h では前方宣言のみで十分（コンストラクタの参照パラメータとポインタメンバ）。
 */
class ISRRetireRouter : public convo::IEpochProvider
{
public:
    explicit ISRRetireRouter(convo::IEpochProvider& provider) noexcept;

    ISRRetireRouter(const ISRRetireRouter&) = delete;
    ISRRetireRouter& operator=(const ISRRetireRouter&) = delete;
    ISRRetireRouter(ISRRetireRouter&&) = delete;
    ISRRetireRouter& operator=(ISRRetireRouter&&) = delete;

    // ── Epoch API (Router経由でEpochDomainを間接参照、実装は .cpp) ──

    uint64_t snapshotEpoch() const noexcept;
    uint64_t publishEpoch() noexcept override;
    uint32_t activeReaderCount() const noexcept override;
    int readerCapacity() const noexcept override;
    uint64_t currentEpoch() const noexcept override;
    uint64_t getMinReaderEpoch() const noexcept override;
    int registerReaderThread() noexcept override;
    bool reserveReaderThread(int readerIndex) noexcept override;
    void enterReader(int readerIndex) noexcept override;
    void exitReader(int readerIndex) noexcept override;
    uint64_t minReaderEpoch() const noexcept;

    // ★ B-1: Reader Slot 詳細取得 (delegates to provider)
    [[nodiscard]] ReaderSlotDetail getReaderSlotDetail(int readerIndex) const noexcept override;

    // ★ Practical-1: Reader Stuck 診断 (delegates to provider)
    [[nodiscard]] StuckReaderInfo detectStuckReaders(uint64_t stuckThreshold) const noexcept override;

    // ★ A-2: EBR Queue Visibility 統計 (delegates to provider)
    [[nodiscard]] uint64_t reclaimAttemptCount() const noexcept override;
    [[nodiscard]] uint64_t reclaimSuccessCount() const noexcept override;

    // ── Retire API (実装は .cpp) ──

    RetireEnqueueResult enqueueRetire(void* ptr,
                                      void (*deleter)(void*),
                                      uint64_t epoch,
                                      DeletionEntryType type) noexcept;
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override;
    void tryReclaim() noexcept override;
    uint32_t pendingRetireCount() const noexcept override;
    void drainAll() noexcept override;

    // ★ Practical-3: Overflow レート監視用カウンター
    [[nodiscard]] uint64_t overflowCount() const noexcept {
        return convo::consumeAtomic(m_overflowCount_, std::memory_order_acquire);
    }
    [[nodiscard]] const std::atomic<uint64_t>* getOverflowCountRef() const noexcept {
        return &m_overflowCount_;
    }

    // ★ Practical-4: Forced reclaim 時刻（enqueueRetire QueuePressure 時の即時 tryReclaim 用）
    void setLastForcedReclaimTimeUs(uint64_t t) noexcept {
        convo::publishAtomic(m_lastForcedReclaimTimeUs_, t, std::memory_order_release);
    }
    [[nodiscard]] uint64_t lastForcedReclaimTimeUs() const noexcept {
        return convo::consumeAtomic(m_lastForcedReclaimTimeUs_, std::memory_order_acquire);
    }

    // ★ Phase 3: Reader Quarantine 委譲（IEpochProvider → EpochDomain）
    [[nodiscard]] bool quarantineReader(int readerIndex) noexcept override
    {
        return provider_->quarantineReader(readerIndex);
    }

    void unquarantineAllReaders() noexcept override
    {
        provider_->unquarantineAllReaders();
    }

    [[nodiscard]] int quarantinedReaderCount() const noexcept override
    {
        return provider_->quarantinedReaderCount();
    }

private:
    convo::IEpochProvider* provider_ = nullptr;
    std::atomic<uint64_t> m_overflowCount_{0};          // ★ Practical-3: enqueueRetire QueuePressure 回数
    std::atomic<uint64_t> m_lastForcedReclaimTimeUs_{0}; // ★ Practical-4: 最終強制 reclaim 時刻
};

} // namespace isr
} // namespace convo

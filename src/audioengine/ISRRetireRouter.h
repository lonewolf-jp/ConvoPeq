#pragma once

#include <atomic>
#include <cstdint>
#include <cassert>
#include <functional>

#include "core/EpochDomain.h"
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
 */
class ISRRetireRouter : public convo::IEpochProvider
{
public:
    explicit ISRRetireRouter(EpochDomain& epochDomain) noexcept
        : epochDomain_(&epochDomain)
    {
    }

    ISRRetireRouter(const ISRRetireRouter&) = delete;
    ISRRetireRouter& operator=(const ISRRetireRouter&) = delete;
    ISRRetireRouter(ISRRetireRouter&&) = delete;
    ISRRetireRouter& operator=(ISRRetireRouter&&) = delete;

    // ── Epoch API (Router経由でEpochDomainを間接参照) ──

    /** 現在のepoch番号を取得 (従来の currentEpoch/current に相当) */
    uint64_t snapshotEpoch() const noexcept
    {
        assert(epochDomain_ != nullptr);
        return epochDomain_->currentEpoch();
    }

    /** Epochを進捗 (従来の advanceEpoch/publish に相当) */
    uint64_t publishEpoch() noexcept override
    {
        assert(epochDomain_ != nullptr);
#pragma warning(push)
#pragma warning(disable : 4996) // [[deprecated]] — transitional, Router wraps EpochDomain
        return epochDomain_->advanceEpoch();
#pragma warning(pop)
    }

    /** アクティブReader数を診断用に取得 */
    uint32_t activeReaderCount() const noexcept override
    {
        assert(epochDomain_ != nullptr);
        return epochDomain_->activeReaderCount();
    }

    uint64_t currentEpoch() const noexcept override
    {
        return snapshotEpoch();
    }

    uint64_t getMinReaderEpoch() const noexcept override
    {
        return minReaderEpoch();
    }

    int registerReaderThread() noexcept override
    {
        assert(epochDomain_ != nullptr);
        return epochDomain_->registerReaderThread();
    }

    bool reserveReaderThread(int readerIndex) noexcept override
    {
        assert(epochDomain_ != nullptr);
        return epochDomain_->reserveReaderThread(readerIndex);
    }

    void enterReader(int readerIndex) noexcept override
    {
        assert(epochDomain_ != nullptr);
#pragma warning(push)
#pragma warning(disable : 4996)
        epochDomain_->enterReader(readerIndex);
#pragma warning(pop)
    }

    void exitReader(int readerIndex) noexcept override
    {
        assert(epochDomain_ != nullptr);
#pragma warning(push)
#pragma warning(disable : 4996)
        epochDomain_->exitReader(readerIndex);
#pragma warning(pop)
    }

    /** 最小Reader epochを取得 (reclaim判定用) */
    uint64_t minReaderEpoch() const noexcept
    {
        assert(epochDomain_ != nullptr);
        return epochDomain_->getMinReaderEpoch();
    }

    // ── Retire API (単一入口) ──

    /** 単一のretire enqueue — Policy Laneに振り分ける */
    RetireEnqueueResult enqueueRetire(void* ptr,
                                      void (*deleter)(void*),
                                      uint64_t epoch,
                                      DeletionEntryType type) noexcept
    {
        assert(epochDomain_ != nullptr);
        if (ptr == nullptr || deleter == nullptr)
            return RetireEnqueueResult::Success;

        // Route to EpochDomain's deferred deletion queue as primary target.
        // [work21 P0-1] Future: delegate to DSPRetirePolicy / SnapshotRetirePolicy / DeferredRetirePolicy
#pragma warning(push)
#pragma warning(disable : 4996) // [[deprecated]] — transitional, Router wraps EpochDomain
        if (epochDomain_->enqueueRetire(ptr, deleter, epoch, type))
            return RetireEnqueueResult::Success;
#pragma warning(pop)

        return RetireEnqueueResult::QueuePressure;
    }

    // [work21] IRetireProvider::enqueueRetire — wraps to 4-param overload with Generic type.
    // Use the 4-param overload directly when you need RetireEnqueueResult comparison.
    bool enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept override
    {
        return enqueueRetire(ptr, deleter, epoch, DeletionEntryType::Generic)
            == RetireEnqueueResult::Success;
    }

    /** Try to reclaim retired objects (delegates to EpochDomain::reclaimRetired) */
    void tryReclaim() noexcept override
    {
        assert(epochDomain_ != nullptr);
#pragma warning(push)
#pragma warning(disable : 4996) // [[deprecated]] — transitional, Router wraps EpochDomain
        epochDomain_->reclaimRetired();
#pragma warning(pop)
    }

    /** Pending retire count (diagnostic) */
    uint32_t pendingRetireCount() const noexcept
    {
        assert(epochDomain_ != nullptr);
        return epochDomain_->pendingRetireCount();
    }

private:
    EpochDomain* epochDomain_ = nullptr;
};

} // namespace isr
} // namespace convo

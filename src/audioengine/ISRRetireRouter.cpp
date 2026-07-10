//============================================================================
// ISRRetireRouter.cpp — EpochDomain ラッパーの実装
//
// ISR P1-19: EpochDomain の完全型はこの .cpp に閉じ込め、
// .h では前方宣言のみとする。これにより公開APIへの EpochDomain 露出を排除する。
//============================================================================

#include "ISRRetireRouter.h"
#include "core/TimeUtils.h"     // ★ Practical-4: getCurrentTimeUs

namespace convo {
namespace isr {

ISRRetireRouter::ISRRetireRouter(IEpochProvider& provider) noexcept
    : provider_(&provider)
{
}

uint64_t ISRRetireRouter::snapshotEpoch() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->currentEpoch();
}

uint64_t ISRRetireRouter::publishEpoch() noexcept
{
    assert(provider_ != nullptr);
    return provider_->publishEpoch();
}

uint32_t ISRRetireRouter::activeReaderCount() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->activeReaderCount();
}

uint64_t ISRRetireRouter::currentEpoch() const noexcept
{
    return snapshotEpoch();
}

uint64_t ISRRetireRouter::getMinReaderEpoch() const noexcept
{
    return minReaderEpoch();
}

int ISRRetireRouter::registerReaderThread() noexcept
{
    assert(provider_ != nullptr);
    return provider_->registerReaderThread();
}

bool ISRRetireRouter::reserveReaderThread(int readerIndex) noexcept
{
    assert(provider_ != nullptr);
    return provider_->reserveReaderThread(readerIndex);
}

void ISRRetireRouter::enterReader(int readerIndex) noexcept
{
    assert(provider_ != nullptr);
    provider_->enterReader(readerIndex);
}

void ISRRetireRouter::exitReader(int readerIndex) noexcept
{
    assert(provider_ != nullptr);
    provider_->exitReader(readerIndex);
}

convo::ReaderSlotDetail ISRRetireRouter::getReaderSlotDetail(int readerIndex) const noexcept
{
    assert(provider_ != nullptr);
    return provider_->getReaderSlotDetail(readerIndex);
}

uint64_t ISRRetireRouter::minReaderEpoch() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->getMinReaderEpoch();
}

int ISRRetireRouter::readerCapacity() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->readerCapacity();
}

StuckReaderInfo ISRRetireRouter::detectStuckReaders(uint64_t stuckThreshold) const noexcept
{
    // ★ Practical-1: IEpochProvider の virtual detectStuckReaders 経由で委譲
    //   dynamic_cast 不要。ISR P1-19 / P0-A 完全準拠。
    assert(provider_ != nullptr);
    return provider_->detectStuckReaders(stuckThreshold);
}

RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;

    // Route through IEpochProvider interface.
    if (provider_->enqueueRetire(ptr, deleter, epoch))
    {
        // ★ work70: サイズ追跡は enqueue 時に objectBytes が設定されている場合のみ。
        //   現在の呼び出し元は objectBytes=0 のため trackedRatio=0% となる。
        //   将来、特定の呼び出し元でサイズ設定する場合に対応。
        return RetireEnqueueResult::Success;
    }

    // ★ Practical-4: QueueFull → 同期的 tryReclaim を１度だけ試行（レート制限付き）
    const uint64_t nowUs = convo::getCurrentTimeUs();
    const uint64_t lastReclaim = convo::consumeAtomic(m_lastForcedReclaimTimeUs_, std::memory_order_acquire);
    if (nowUs - lastReclaim > 500'000) // 500ms cooldown
    {
        convo::publishAtomic(m_lastForcedReclaimTimeUs_, nowUs, std::memory_order_release);
        provider_->tryReclaim();

        // 再試行: reclaim 後に空きができたか確認
        if (provider_->enqueueRetire(ptr, deleter, epoch))
            return RetireEnqueueResult::Success;
    }

    // ★ Practical-3: Overflow カウンター増加（Rate監視用）
    convo::fetchAddAtomic(m_overflowCount_, uint64_t{1}, std::memory_order_release);
    return RetireEnqueueResult::QueuePressure;
}

bool ISRRetireRouter::enqueueRetire(void* ptr, void (*deleter)(void*), uint64_t epoch) noexcept
{
    return enqueueRetire(ptr, deleter, epoch, DeletionEntryType::Generic)
        == RetireEnqueueResult::Success;
}

void ISRRetireRouter::tryReclaim() noexcept
{
    assert(provider_ != nullptr);
    provider_->tryReclaim();
}

uint32_t ISRRetireRouter::pendingRetireCount() const noexcept
{
    // ★ P0-A: IRetireProvider 経由で委譲（dynamic_cast 不要）
    assert(provider_ != nullptr);
    return provider_->pendingRetireCount();
}

void ISRRetireRouter::drainAll() noexcept
{
    // ★ P0-A: IRetireProvider 経由で委譲（dynamic_cast 不要）
    assert(provider_ != nullptr);
    provider_->drainAll();
}

// ★ A-2: EBR Queue Visibility 統計委譲
uint64_t ISRRetireRouter::reclaimAttemptCount() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->reclaimAttemptCount();
}

uint64_t ISRRetireRouter::reclaimSuccessCount() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->reclaimSuccessCount();
}

} // namespace isr
} // namespace convo

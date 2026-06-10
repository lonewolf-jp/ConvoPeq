//============================================================================
// ISRRetireRouter.cpp — EpochDomain ラッパーの実装
//
// ISR P1-19: EpochDomain の完全型はこの .cpp に閉じ込め、
// .h では前方宣言のみとする。これにより公開APIへの EpochDomain 露出を排除する。
//============================================================================

#include "ISRRetireRouter.h"
#include "core/EpochDomain.h" // 完全型は .cpp のみでインクルード

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

uint64_t ISRRetireRouter::minReaderEpoch() const noexcept
{
    assert(provider_ != nullptr);
    return provider_->getMinReaderEpoch();
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
        return RetireEnqueueResult::Success;

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
    // pendingRetireCount は EpochDomain 固有メソッド。
    // プロバイダが EpochDomain の場合のみ dynamic_cast で取得。
    assert(provider_ != nullptr);
    auto* ed = dynamic_cast<EpochDomain*>(provider_);
    return ed ? ed->pendingRetireCount() : 0;
}

void ISRRetireRouter::drainAll() noexcept
{
    // drainAll は EpochDomain 固有メソッド。
    // プロバイダが EpochDomain の場合のみ dynamic_cast で実行。
    assert(provider_ != nullptr);
    if (auto* ed = dynamic_cast<EpochDomain*>(provider_))
        ed->drainAll();
}

} // namespace isr
} // namespace convo

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

// ★ R-1: IRetireRouter::retireRT — 単発 enqueue、リトライなし（RT-safe）
bool ISRRetireRouter::retireRT(void* ptr, void (*deleter)(void*)) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return true;
    return provider_->enqueueRetire(ptr, deleter, provider_->currentEpoch());
}

// ★ R-1: IRetireRouter::retire — リトライ込み（NonRT）
void ISRRetireRouter::retire(void* ptr, void (*deleter)(void*)) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return;
    const auto result = enqueueWithRetry(ptr, deleter, provider_->currentEpoch(), DeletionEntryType::Generic);
    if (result != RetireEnqueueResult::Success) {
        // ★ Future: RuntimeHealthMonitor へ通知
    }
}

// ★ Bug2 Phase1: リトライロジックを Router に集約。呼び出し元はリトライループ不要。
RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(void* ptr,
                                                        void (*deleter)(void*),
                                                        uint64_t epoch,
                                                        DeletionEntryType type) noexcept
{
    // 1. 通常の enqueue を試行 (内部で 500ms クールダウン付き tryReclaim を1回実行)
    auto result = enqueueRetire(ptr, deleter, epoch, type);
    if (result == RetireEnqueueResult::Success)
        return result;

    // 2. 追加リトライ: tryReclaim → enqueue（最大 2 回）
    constexpr int kMaxRetry = 2;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        provider_->tryReclaim();   // Router 内部で完結。呼び出し元は意識しない。
        result = enqueueRetire(ptr, deleter, epoch, type);
        if (result == RetireEnqueueResult::Success)
            return result;
        if (result != RetireEnqueueResult::QueuePressure)
            break;  // QueuePressure 以外（Shutdown 等）は即座に終了
    }

    // 3. 全リトライ失敗 → QueuePressure。Router 内部で RuntimeHealthMonitor へ通知する。
    //    （呼び出し側はこの戻り値をもとに動作。PolicyEngine へは HealthMonitor 経由。）
    //    ★ BUG-015/027 (work88): 退避ストアへ移送（directDelete しない）。
    //      queue full は RT 参照中の可能性が高いため、即時解放は UAF を生む。
    //      RetireQuarantineStore で安全保持し、epoch 安全到達後に定期 drain で解放する。
    //      Shutdown 結果はシャットダウン経路（drainAllQuarantineStore）が処理するため移送しない。
    if (result == RetireEnqueueResult::QueuePressure || result == RetireEnqueueResult::QueueFull)
    {
        const bool stored = m_retireQuarantine.quarantine(
            ptr, deleter, epoch, type, "enqueueWithRetry:QueuePressure",
            /*publicationSequenceId=*/0, /*generation=*/0);
        if (!stored)
        {
            // ★ 三次レビュー: store full 時に delete は絶対しない（UAF 構造的排除）。
            //   capacity exhaustion は health escalation（AudioEngine 側の
            //   quarantineOverflowCount 監視）で先行検知する。ここでは EBR 破綻として
            //   assert で異常を検出する（Release ではリーク検出は overflowCount 監視に委ねる）。
            assert(false && "RetireQuarantineStore capacity exhaustion - EBR 破綻の可能性");
        }
    }
    //    ★ Future: runtimeHealth_->notifyQueuePressure(QueuePressureInfo{...});
    return result;
}

void ISRRetireRouter::tryReclaim() noexcept
{
    assert(provider_ != nullptr);
    provider_->tryReclaim();
    // ★ BUG-015/027 (work88): tryReclaim 直後に退避ストアを drain（epoch 安全到達分のみ deleter 実行）
    drainQuarantineStore();
}

// ★ BUG-015/027 (work88): 退避ストアを drain。
//   epoch 比較は EpochDomain::isOlder と同一セマンティクス（wraparound 安全）を
//   インライン実装（ISR P1-19: EpochDomain 完全型を .h に露出しない）。
void ISRRetireRouter::drainQuarantineStore() noexcept
{
    const uint64_t minReader = minReaderEpoch();
    m_retireQuarantine.drain(minReader, [](uint64_t a, uint64_t b) noexcept {
        return static_cast<int64_t>(a - b) < 0;  // == EpochDomain::isOlder(a, b)
    });
}

// ★ BUG-015/027 (work88): Router API — 退避ストアへの移送（directDelete しない）。
//   store full 時は deleter を実行せず false を返す（UAF 構造的排除）。
//   capacity exhaustion は health escalation（AudioEngine 側が quarantineResidentCount /
//   quarantineOverflowCount を監視）で先行検知する。
bool ISRRetireRouter::quarantineRetire(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                       DeletionEntryType type, const char* reason,
                                       uint64_t publicationSequenceId, uint64_t generation) noexcept
{
    return m_retireQuarantine.quarantine(ptr, deleter, epoch, type, reason,
                                         publicationSequenceId, generation);
}

std::size_t ISRRetireRouter::quarantineResidentCount() const noexcept
{
    return m_retireQuarantine.residentCount();
}

std::uint64_t ISRRetireRouter::quarantineOverflowCount() const noexcept
{
    return m_retireQuarantine.overflowCount();
}

// ★ BUG-015/027: shutdown 時 — Audio Thread 停止後のみ全強制解放（drainAllUnsafe と同契約）
void ISRRetireRouter::drainAllQuarantineStore() noexcept
{
    m_retireQuarantine.drainAllUnsafe();
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
    // ★ BUG-015/027: shutdown 時は退避ストアも全強制解放（Audio Thread 停止後）
    drainAllQuarantineStore();
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

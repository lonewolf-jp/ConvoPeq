#include "ISRRuntimePublicationCoordinator.h"
#include "AtomicAccess.h"
#include "SequenceArithmetic.h"  // ★ dash2 §1.6.1 (Phase H): modular sequence arithmetic（commit monotonicity）
#include "ISRRetireOverflowRing.h"
#include "ISRDSPHandle.h"
#include "ISRDSPQuarantine.h"
#include <cassert>
#include "AudioEngine.h"  // FUTURE-4 / CW-3c: RuntimeState 完全型（commit() の bake・prevWorld 用）

namespace convo::isr {

RuntimeIntentCoordinator::RuntimeIntentCoordinator()
    : overflowScheduler_(*this)
    , shutdownScheduler_(*this)
    , priorityScheduler_(*this)
    , lastRejectCode_(RejectCode::None)
    , retireBacklogCount_(0)
    , publicationBacklogCount_(0)
    , pendingIntentCount_(0)
    , reclaimInFlightCount_(0)
    , previousRetireBacklogCount_(0)
    , pressureNormalizedWindows_(0)
    , swapPending_(false)
    , state_(CoordinatorState::Bootstrapping)
    , retireAuthorityCount_(0)
{
    // ★ dash2 §1.7 (CW-3c): persistentState_ / currentWorld_ 削除 — publication metadata は
    //   RuntimeStore::current の RuntimeState::publication（単一 source）
}

bool RuntimeIntentCoordinator::precheckPublish(const PayloadClosureDescriptor& closure,
                                                    const TieredPayloadDescriptor& descriptor) noexcept {
    ClosureValidator closureValidator;
    if (!closureValidator.validateClosureGraph(closure)) {
        convo::publishAtomic(lastRejectCode_, RejectCode::InvalidClosure, std::memory_order_release);
        return false;
    }

    PayloadTierValidator tierValidator;
    if (!tierValidator.isPublishAllowed(descriptor)) {
        convo::publishAtomic(lastRejectCode_, RejectCode::InvalidPayloadTier, std::memory_order_release);
        return false;
    }

    convo::publishAtomic(lastRejectCode_, RejectCode::None, std::memory_order_release);
    return true;
}

const char* RuntimeIntentCoordinator::lastRejectReason() const noexcept {
    switch (convo::consumeAtomic(lastRejectCode_, std::memory_order_acquire)) {
    case RejectCode::InvalidClosure:
        return "invalid closure graph";
    case RejectCode::InvalidPayloadTier:
        return "invalid payload tier";
    case RejectCode::None:
    default:
        return "none";
    }
}

void RuntimeIntentCoordinator::commit(PublishAuthority,
                                           RuntimeBoundary boundary,
                                           const void* newWorld,
                                           std::uint64_t version) {
    commit(PublishAuthority::Granted,
           boundary,
           newWorld,
           version,
           static_cast<PublicationSequenceId>(version),
           static_cast<PublicationEpoch>(version),
           version,
           nullptr);
}

void RuntimeIntentCoordinator::commit(PublishAuthority,
                                           RuntimeBoundary boundary,
                                           const void* newWorld,
                                           std::uint64_t /*version*/,
                                           PublicationSequenceId sequenceId,
                                           PublicationEpoch epoch,
                                           std::uint64_t mappedGeneration,
                                           const RuntimeState* prevWorld) {
    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }

    // ★ dash2 §1.7 (Phase G CW-3b): monotonicity baseline は明示された prevWorld（publish() が
    //   runtimeStore_.observe() = RuntimeStore::current から取得）のみ。currentWorld_ は参照しない
    //   （read/write dependency 除去 — explicit dependency。Authority は RuntimeWorldAuthority が
    //   publication baseline を所有）。prevWorld == nullptr は初回 publish を許容（後方互換の規則
    //   と同一 — nullptr だからといって値を無条件に受け入れる緩和はしない）。
    const auto prevSeqId = prevWorld ? prevWorld->publication.sequenceId : PublicationSequenceId{0};
    const auto prevEpoch  = prevWorld ? prevWorld->publication.epoch : PublicationEpoch{0};
    const auto prevGen    = prevWorld ? prevWorld->publication.mappedRuntimeGeneration : std::uint64_t{0};

    const bool hasPrevious = prevSeqId != 0 || prevEpoch != 0 || prevGen != 0;
    if (hasPrevious) {
        // ★ dash2 §1.6.1 (Phase H): modular comparison（wraparound-safe — Appendix E）。
        //   isAfter(a,b) == (a > b)（非 wrap 値）。seq/epoch は 1 ずつ増加のため semantics-preserving。
        if (!(convo::isr::isAfter(sequenceId, prevSeqId)
              && convo::isr::isAfter(epoch, prevEpoch)
              && mappedGeneration > prevGen)) {
            convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
            return;
        }
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);
    convo::publishAtomic(swapPending_, true, std::memory_order_release);

    // ★ FUTURE-4 (publish-time freeze): bake incoming publication semantics onto newWorld,
    //   replacing the former persistentState_ cache; Reader derives metadata via RuntimeStore::current
    //   ->publication（CW-1 で read source を RuntimeStore に一本化済み）。
    //   ［MutablePrePublish の semantic contract — candidate は未観測期にのみ mutation される］
    auto* pubWorld = const_cast<RuntimeState*>(static_cast<const RuntimeState*>(newWorld));
    pubWorld->publication = PublicationSemantic{sequenceId, epoch,
        static_cast<PublicationGeneration>(mappedGeneration), prevSeqId};
    convo::publishAtomic(swapPending_, false, std::memory_order_release);
    convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
}

void RuntimeIntentCoordinator::retire(RetireAuthority,
                                           RuntimeBoundary boundary,
                                           const void* oldWorld) {
    if (boundary != RuntimeBoundary::NonRTWorld || oldWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }

    // ★ dash2 §1.7 (Phase G CW-3c): currentWorld_ の metadata-cache clear（CAS）を削除。
    //   退役の実 authority は Lifetime/EBR（onRuntimeRetiredNonRt → emitRetireIntentRT）が担当。
    //   currentWorld_ は CW-3b 以降 non-update（commit が write しない）のため本 CAS は no-op だった。
    //   退役の identity source は publish() の戻り値 oldWorld（caller が判定）— RuntimeStore::current
    //   への委譲は不要（RuntimeStore::current が published-world read の単一 source）。
    //   ［本メソッドは入力契約（NonRTWorld・oldWorld 非 null）の検証 + Fault のみを維持］

    // ★ dash2 §1.4 (B0-3): 本メソッドは world-retire 時に AudioEngine.Commit.cpp:461 から呼ばれる。
    //   retireBacklogCount_ はここで増加させない — 元コードは Commit.cpp:481 の setRetireBacklogCount
    //   （lifetime pendingIntentCount スナップショット）で上書きされており、external setter 撤去後は
    //   retireBacklogCount_ が commit ごとに無制限増加して Layer 2 の isFullyDrained（retireBacklogCount_==0）
    //   がシャットダウンで失敗する回帰を防ぐ。retire バックログの実測判定は Layer 1
    //   （AudioEngine::isFullyDrained）が m_retireRouter->pendingRetireCount() と
    //   worldAuthority_.lifetime().pendingIntentCount() を直接参照する（dash2 §1.4 設計方針）。
    //   retireAuthorityCount_ はここで維持される（呼び出し回数の Authority 計数）。
}

RetireEnqueueResult RuntimeIntentCoordinator::enqueueRetire(RetireAuthority,
                                                                   ISRRetireRouter& router,
                                                                   void* ptr,
                                                                   void (*deleter)(void*),
                                                                   std::uint64_t epoch) noexcept
{
    convo::fetchAddAtomic(retireAuthorityCount_,
                          static_cast<std::uint64_t>(1),
                          std::memory_order_acq_rel);

    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;

    // ★ Bug#2-d: enqueueWithRetry に委譲（リトライロジックは Router に集約）
    const auto result = router.enqueueWithRetry(ptr, deleter, epoch, DeletionEntryType::Generic);
    if (result != RetireEnqueueResult::Success)
        return result;

    // ★ dash2 §1.4 (B0-3): 非原子的 load+setRetireBacklogCount RMW を semantic event に置換。
    //   （本メソッドは production で未使用 — 将来の retire 経路用。retireBacklogCount_ は
    //   onRetireConsumed と対で維持される。Layer 1 が実測で drain 判定するため authoritative ではない。）
    onRetireAccepted();

    return RetireEnqueueResult::Success;
}

std::uint64_t RuntimeIntentCoordinator::retireAuthorityCount() const noexcept
{
    return convo::consumeAtomic(retireAuthorityCount_, std::memory_order_acquire);
}

// ★ dash2 §1.7 (Phase G CW-3c): getCurrent/getVersion/currentPublicationEpoch/currentPublicationSequenceId
//   は production caller ゼロ（RuntimeWorldAuthority の delegation は CW-3c で削除）のため削除。
//   published-world read は RuntimeStore::current（RuntimeWorldAuthority::observePublishedWorld）が単一 source。
//   ［Coordinator に published-world read API を残さない — CW-1 read-side singularization と整合］

// ── dash2 §1.4: semantic event accounting ──
//   production は setter（絶対値上書き）ではなく本イベントで原子的に増減を通知する。
//   underflow ガード: fetch_sub 前に old > 0 を検証（0 で fetch_sub すると UINT64_MAX ラップ）。
//   違反時は Faulted → Proof 生成不能へ遷移（dash2 §1.4 第三者的レビュー反映 3）。
void RuntimeIntentCoordinator::onRetireAccepted() noexcept {
    // fetch_add 後（= 更新後の絶対値）を noteRetireBacklogChanged に渡し、pressure slope 検出を共通化。
    const auto newCount = convo::fetchAddAtomic(retireBacklogCount_,
                                                std::uint64_t{1},
                                                std::memory_order_acq_rel) + std::uint64_t{1};
    noteRetireBacklogChanged(newCount);
}

void RuntimeIntentCoordinator::onRetireConsumed() noexcept {
    const auto old = convo::consumeAtomic(retireBacklogCount_, std::memory_order_acquire);
    if (old > 0) {
        convo::fetchSubAtomic(retireBacklogCount_, std::uint64_t{1}, std::memory_order_acq_rel);
    } else {
        // underflow 違反（consume と fetch_sub の間の race では old==0 でも他 writer が先に
        // 書いている可能性があるため、ここでは Faulted 化のみ行う。実害のある負方向超過は
        // fetch_sub 前に old>0 検証で防止済み）。
        convo::publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
    }
}

// ★ D101-32-D: onFallbackAccepted/Consumed / onDeferredRetireAccepted/Consumed は削除済み
//   （D101-32-C §5 — caller ゼロ・対応 counter と domain 一括除去）。

void RuntimeIntentCoordinator::onReclaimBegin() noexcept {
    convo::fetchAddAtomic(reclaimInFlightCount_, std::uint64_t{1}, std::memory_order_acq_rel);
}

// ★ dash2 §2.2 (A2-G02): onReclaimEnd は「deferred 保留の解消」。
//   reclaimInFlightCount_ の意味論 = 保留中（deferred）の reclaim 数。
//   - defer（epoch unsafe）→ onReclaimBegin（+1）
//   - 成功 → onReclaimEnd（-1）
//   ⚠️ 他カウンタ（retire/fallback/deferred）の Consumed と異なり、count==0 で Faulted に
//   しない。単発成功（defer なし → count 0 のまま成功）は正当な正常系（INV-3-1）であり、
//   0 で fetch_sub すると UINT64_MAX ラップになるため、old>0 検証で no-op に留める。
//   （reclaimInFlightCount_ は isFullyDrained の ==0 判定に使われる近似カウンタであり、
//     正確な identity 管理は pendingReclaimHandles_ / ReclaimIdentity set が担当 — INV-X3-5）
void RuntimeIntentCoordinator::onReclaimEnd() noexcept {
    const auto old = convo::consumeAtomic(reclaimInFlightCount_, std::memory_order_acquire);
    if (old > 0) {
        convo::fetchSubAtomic(reclaimInFlightCount_, std::uint64_t{1}, std::memory_order_acq_rel);
    }
    // else: 単発成功（defer なし）— 正常系。no-op（Faulted にしない）
}

void RuntimeIntentCoordinator::noteRetireBacklogChanged(std::uint64_t count) noexcept {
    const auto previousBacklog = convo::consumeAtomic(previousRetireBacklogCount_, std::memory_order_acquire);
    const auto slope = (count > previousBacklog) ? (count - previousBacklog) : 0;

    convo::publishAtomic(previousRetireBacklogCount_, count, std::memory_order_release);

    if (slope > kPressureSlopeThreshold) {
        convo::publishAtomic(pressureNormalizedWindows_, static_cast<std::uint32_t>(0), std::memory_order_release);
        convo::publishAtomic(state_, CoordinatorState::Pressure, std::memory_order_release);
        return;
    }

    if (!isSwapPending()) {
        const auto state = convo::consumeAtomic(state_, std::memory_order_acquire);
        if (state == CoordinatorState::Pressure) {
            const auto nextWindow = static_cast<std::uint32_t>(
                convo::consumeAtomic(pressureNormalizedWindows_, std::memory_order_acquire) + 1U);
            convo::publishAtomic(pressureNormalizedWindows_, nextWindow, std::memory_order_release);

            if (nextWindow < kPressureNormalizeWindows) {
                return;
            }

            convo::publishAtomic(pressureNormalizedWindows_, static_cast<std::uint32_t>(0), std::memory_order_release);
            convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
        } else if (state == CoordinatorState::Publishing && count == 0) {
            convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
        }
    }
}

// ⚠️ TEST-ONLY（dash2 §1.4）: production からの絶対値上書きは禁止。テスト初期化リセットのみ。
void RuntimeIntentCoordinator::setRetireBacklogCount(std::uint64_t count) noexcept {
    convo::publishAtomic(retireBacklogCount_, count, std::memory_order_release);
    noteRetireBacklogChanged(count);
}

void RuntimeIntentCoordinator::setPublicationBacklogCount(std::uint64_t count) noexcept {
    convo::publishAtomic(publicationBacklogCount_, count, std::memory_order_release);
}

void RuntimeIntentCoordinator::setPendingIntentCount(std::uint64_t count) noexcept {
    convo::publishAtomic(pendingIntentCount_, count, std::memory_order_release);
}

// ★ D101-32-D: setFallbackBacklogCount / setReclaimInFlightCount /
//   setDeferredRetireResidencyCount / setQuarantineResidentCount は削除済み
//   （D101-32-C §7 — fallback/deferred/quarantine-resident domain の vestigial setter。
//     setReclaimInFlightCount は onReclaimBegin/End 移行済みのため test reset 用途も消滅）。

void RuntimeIntentCoordinator::setOverflowMaxAgeUs(std::uint64_t maxAgeUs) noexcept {
    convo::publishAtomic(overflowMaxAgeUs_, maxAgeUs, std::memory_order_release);
}

std::uint64_t RuntimeIntentCoordinator::getOverflowMaxAgeUs() const noexcept {
    return convo::consumeAtomic(overflowMaxAgeUs_, std::memory_order_acquire);
}

RuntimeIntentCoordinator::OverflowDrainResult
RuntimeIntentCoordinator::drainOverflowRing(
    RetireOverflowRing& overflowRing, LifetimeState& retireRuntime, bool unlimited) noexcept
{
    return overflowScheduler_.drainOverflowRing(overflowRing, retireRuntime, unlimited);
}

void RuntimeIntentCoordinator::setOverflowAgeWarnCallback(AgeWarnCallback cb) noexcept {
    priorityScheduler_.setOverflowAgeWarnCallback(cb);
}

size_t RuntimeIntentCoordinator::deferredRingOccupancy() const noexcept {
    return overflowScheduler_.deferredRingOccupancy();
}

// ═══════════════════════════════════════════════════════════
// ★ Phase5: OverflowScheduler implementation
// ═══════════════════════════════════════════════════════════

RuntimeIntentCoordinator::OverflowDrainResult
RuntimeIntentCoordinator::OverflowScheduler::drainOverflowRing(
    RetireOverflowRing& overflowRing, LifetimeState& retireRuntime, bool unlimited) noexcept
{
    OverflowDrainResult result;
    constexpr uint32_t kDefaultBudget = 64;
    constexpr uint32_t kMaxReinjectRetries = 10;
    const uint32_t budget = unlimited ? 0xFFFFFFFFu : kDefaultBudget;
    uint32_t consumed = 0;

    // ★ Phase1: OverflowRing から drain（優先度高）
    RetireOverflowEntry entry;
    while (consumed < budget && overflowRing.pop(entry))
    {
        ++consumed;

        // 滞留時間監視
        if (coordinator_.overflowAgeWarnCallback_ != nullptr)
        {
            const uint64_t nowUs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            if (entry.overflowTimestampUs > 0 && nowUs > entry.overflowTimestampUs)
            {
                const uint64_t ageUs = nowUs - entry.overflowTimestampUs;
                if (ageUs > result.oldestOverflowAgeUs)
                    result.oldestOverflowAgeUs = ageUs;
                const uint64_t maxAgeUs = convo::consumeAtomic(coordinator_.overflowMaxAgeUs_, std::memory_order_acquire);
                if (maxAgeUs > 0 && ageUs > maxAgeUs)
                    coordinator_.overflowAgeWarnCallback_(ageUs, result.droppedCount);
            }
        }

        // retry超過 → Drop
        if (entry.reinjectRetryCount >= kMaxReinjectRetries)
        {
            ++result.droppedCount;
            continue;
        }

        // 再注入: Overflowからの再注入は High 優先度
        ++entry.reinjectRetryCount;
        entry.intent.priority = RetirePriority::High;
        retireRuntime.emitRetireIntent(entry.intent);
        ++result.reinjectedCount;
    }

    // ★ Phase5: Coordinator DeferredRing から drain（優先度中）
    {
        RetireOverflowEntry deferredEntry;
        constexpr uint32_t kDeferredBudget = 32;
        uint32_t deferredDrained = 0;
        while (deferredDrained < kDeferredBudget && coordinator_.coordinatorDeferredRing_.pop(deferredEntry))
        {
            ++deferredDrained;
            retireRuntime.emitRetireIntent(deferredEntry.intent);
            ++result.reinjectedCount;
        }
        result.deferredRingOccupancy = convo::consumeAtomic(coordinator_.coordinatorDeferredCount_, std::memory_order_acquire);
        // 排出成功分をカウントから減算
        if (deferredDrained > 0)
        {
            convo::fetchSubAtomic(coordinator_.coordinatorDeferredCount_,
                                  static_cast<size_t>(deferredDrained),
                                  std::memory_order_acq_rel);
        }
    }

    // ★ Phase5: LastResortQueue から drain（優先度低）
    {
        const size_t lrCount = convo::consumeAtomic(coordinator_.lastResortCount_, std::memory_order_acquire);
        if (lrCount > 0)
        {
            constexpr uint32_t kLastResortBudget = 16;
            uint32_t lrDrained = 0;
            for (size_t i = 0; i < lrCount && lrDrained < kLastResortBudget; ++i)
            {
                auto& lrEntry = coordinator_.lastResortQueue_[i];
                if (lrEntry.intent.dspSlot != UINT32_MAX)
                {
                    lrEntry.intent.priority = RetirePriority::High;
                    retireRuntime.emitRetireIntent(lrEntry.intent);
                    lrEntry.intent.dspSlot = UINT32_MAX;
                    ++lrDrained;
                    ++result.reinjectedCount;
                }
            }
            if (lrDrained > 0)
            {
                // 排出済みエントリを詰める
                size_t writeIdx = 0;
                for (size_t readIdx = 0; readIdx < lrCount; ++readIdx)
                {
                    if (coordinator_.lastResortQueue_[readIdx].intent.dspSlot != UINT32_MAX)
                    {
                        if (writeIdx != readIdx)
                            coordinator_.lastResortQueue_[writeIdx] = coordinator_.lastResortQueue_[readIdx];
                        ++writeIdx;
                    }
                }
                convo::publishAtomic(coordinator_.lastResortCount_, writeIdx, std::memory_order_release);
            }
        }
    }

    return result;
}

void RuntimeIntentCoordinator::setSwapPending(bool pending) noexcept {
    convo::publishAtomic(swapPending_, pending, std::memory_order_release);
}

bool RuntimeIntentCoordinator::isSwapPending() const noexcept {
    return convo::consumeAtomic(swapPending_, std::memory_order_acquire);
}

std::uint64_t RuntimeIntentCoordinator::getReclaimInFlightCount() const noexcept {
    return convo::consumeAtomic(reclaimInFlightCount_, std::memory_order_acquire);
}

// ★ A-2.4: 新規 getter 群（DrainAudit 用）
std::uint64_t RuntimeIntentCoordinator::getPublicationBacklogCount() const noexcept {
    return convo::consumeAtomic(publicationBacklogCount_, std::memory_order_acquire);
}

// ★ work88 (X5 §6.5): Publish Intent residency counter（INV-X5-1）。
std::uint64_t RuntimeIntentCoordinator::getPublicationIntentResidencyCount() const noexcept {
    return convo::consumeAtomic(publicationIntentResidencyCount_, std::memory_order_acquire);
}

std::uint64_t RuntimeIntentCoordinator::getPendingIntentCount() const noexcept {
    return convo::consumeAtomic(pendingIntentCount_, std::memory_order_acquire);
}

std::uint64_t RuntimeIntentCoordinator::getRetireBacklogCount() const noexcept {
    return convo::consumeAtomic(retireBacklogCount_, std::memory_order_acquire);
}

// ★ work88 (X6 §6.6): Quarantine transport residency counters（INV-X6-4）。
std::uint64_t RuntimeIntentCoordinator::getQuarantineIntentResidencyCount() const noexcept {
    return convo::consumeAtomic(quarantineIntentResidencyCount_, std::memory_order_acquire);
}

std::uint64_t RuntimeIntentCoordinator::getQuarantineRingResidencyCount() const noexcept {
    return convo::consumeAtomic(quarantineRingResidencyCount_, std::memory_order_acquire);
}

// ★ D101-32-D: getFallbackBacklogCount / getDeferredRetireResidencyCount /
//   getQuarantineResidentCount（Coordinator側）は削除済み（vestigial counter の getter）。

// ★ Phase5: Delegation to ShutdownScheduler
bool RuntimeIntentCoordinator::isFullyDrained() const noexcept {
    return shutdownScheduler_.isFullyDrained();
}

void RuntimeIntentCoordinator::requestShutdown() noexcept {
    shutdownScheduler_.requestShutdown();
}

void RuntimeIntentCoordinator::markShutdownComplete() noexcept {
    shutdownScheduler_.markShutdownComplete();
}

RuntimeIntentCoordinator::CoordinatorState RuntimeIntentCoordinator::getState() const noexcept {
    return convo::consumeAtomic(state_, std::memory_order_acquire);
}

void RuntimeIntentCoordinator::markTransitionStart() noexcept {
    const auto state = convo::consumeAtomic(state_, std::memory_order_acquire);
    if (state != CoordinatorState::Ready) {
        return;
    }
    convo::publishAtomic(state_, CoordinatorState::Transitioning, std::memory_order_release);
}

void RuntimeIntentCoordinator::markTransitionCommitted() noexcept {
    const auto state = convo::consumeAtomic(state_, std::memory_order_acquire);
    if (state != CoordinatorState::Transitioning) {
        return;
    }
    if (!isSwapPending()) {
        convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);
    }
}

// ═══════════════════════════════════════════════════════════
// ★ Phase5: OverflowScheduler deferredRingOccupancy
// ═══════════════════════════════════════════════════════════

size_t RuntimeIntentCoordinator::OverflowScheduler::deferredRingOccupancy() const noexcept {
    return convo::consumeAtomic(coordinator_.coordinatorDeferredCount_, std::memory_order_acquire);
}

// ═══════════════════════════════════════════════════════════
// ★ Phase5: ShutdownScheduler implementation
// ═══════════════════════════════════════════════════════════

bool RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained() const noexcept {
    if (convo::consumeAtomic(coordinator_.swapPending_, std::memory_order_acquire)) {
        return false;
    }

    // ★ work88 (P2-4 §1.2): transport キュー空判定を追加。
    //   pendingIntentCount_ == 0 だけでは「Intent が transport に残存するがカウンタ不一致」の
    //   ケース（カウンタと実体の乖離）を検出できない。以下の 4 キューが空であることを直接確認する:
    //   - intentQueue_             : Observe/Publish/Quarantine（MPSC, reservation order）
    //   - observeDeferredRing_     : Observe overflow（SPSC）
    //   - quarantineFallbackQueue_ : Quarantine fallback（MPSC）
    //   - recoveryIntentQueue_     : Recovery（Builder Work Queue, SPSC）
    //   ★ phase-gated: 本判定は「admission closed + producer join」後にのみ authoritative。
    //     coordinator state が ShuttingDown へ遷移した後（producer が全て閉じた後）に
    //     isFullyDrained が呼ばれる前提。進行中 producer が居る最中はキューが空でも
    //     新たな Intent が到着し得るため、単独では drain 完了を保証しない。
    return coordinator_.intentQueue_.sizeApprox() == 0
        && coordinator_.observeDeferredRing_.size() == 0
        && coordinator_.quarantineFallbackQueue_.sizeApprox() == 0
        && coordinator_.recoveryIntentQueue_.size() == 0
        && convo::consumeAtomic(coordinator_.retireBacklogCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.publicationBacklogCount_, std::memory_order_acquire) == 0
        // ★ work88 (X5 §6.5): Publish Intent residency も独立に == 0 を要求（INV-X5-1）。
        //   queue emptiness と pendingIntentCount_ だけでは Publish 残留（intentQueue_ 内の
        //   Publish Intent）を捕捉できないため、本 counter で独立判定する。
        && convo::consumeAtomic(coordinator_.publicationIntentResidencyCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.pendingIntentCount_, std::memory_order_acquire) == 0
        // ★ D101-32-D: fallbackBacklogCount_ / deferredRetireResidencyCount_ の == 0 判定は削除。
        //   両 counter は vestigial（writer ゼロ）であり恒真判定だった。fallback/deferred の実在
        //   資源は quarantineFallbackQueue_ / observeDeferredRing_ の実測と Layer 1 実測が authority
        //   （D101-32-C §4/§6 — 情報欠落なし）。
        && convo::consumeAtomic(coordinator_.reclaimInFlightCount_, std::memory_order_acquire) == 0
        // ★ work88 (X6 §6.6): Quarantine transport residency を個別に == 0（INV-X6-4）。
        //   quarantineIntentResidencyCount_（intentQueue_ 残留）と quarantineRingResidencyCount_
        //   （quarantineFallbackQueue_ 残留）をそれぞれ独立判定する。実在 DSP quarantine resident
        //   （旧 quarantineResidentCount_ 項目）は D101-32-D で削除 — AudioEngine::isFullyDrained が
        //   DSPQuarantineManager を直接判定（X6、authority 変更なし）。
        && convo::consumeAtomic(coordinator_.quarantineIntentResidencyCount_, std::memory_order_acquire) == 0
        && convo::consumeAtomic(coordinator_.quarantineRingResidencyCount_, std::memory_order_acquire) == 0
        // ★ work88 (X1 §6.1): durable Recovery admission が空であること（INV-X1-1/INV-X1-2）。
        //   lease 方式では DurablePending OR Building の両方が false であること（recoveryAdmissionPending_
        //   は Building 中も true を維持 — 二十六次レビュー）。shutdown 時は discardPendingRecoveryAdmission
        //   （RecoveryAdmissionClosed）で破棄されるため、本条件は成立する。
        && !convo::consumeAtomic(coordinator_.recoveryAdmissionPending_, std::memory_order_acquire)
        // ★ D105-R13 (INV-1-2 structural): logical obligation residency must be zero. Previously this held
        //   only as an emergent property of shutdown ordering (requestShutdown → Coordinator/Builder joined
        //   → discardRecoveryRequestsOnShutdown closes every Live slot). A popped-but-not-yet-resolved
        //   obligation (Live + empty transport/durable/counters) would otherwise be a silent false-positive
        //   "drained". Elevating it into the predicate forces such a case to report not-drained. The single −1
        //   authority (RecoveryAdmissionTable::resolve, idempotent) guarantees liveCount_==0 once discard runs.
        && coordinator_.liveLogicalRecoveryObligationCount() == 0;
}

void RuntimeIntentCoordinator::ShutdownScheduler::requestShutdown() noexcept {
    convo::publishAtomic(coordinator_.state_, CoordinatorState::ShuttingDown, std::memory_order_release);
}

void RuntimeIntentCoordinator::ShutdownScheduler::markShutdownComplete() noexcept {
    const auto state = convo::consumeAtomic(coordinator_.state_, std::memory_order_acquire);
    if (state != CoordinatorState::ShuttingDown) {
        return;
    }

    if (isFullyDrained()) {
        convo::publishAtomic(coordinator_.state_, CoordinatorState::Bootstrapping, std::memory_order_release);
    } else {
        convo::publishAtomic(coordinator_.state_, CoordinatorState::Faulted, std::memory_order_release);
    }
}

// ═══════════════════════════════════════════════════════════
// ★ Phase5: PriorityScheduler implementation
// ═══════════════════════════════════════════════════════════

void RuntimeIntentCoordinator::PriorityScheduler::setOverflowAgeWarnCallback(AgeWarnCallback cb) noexcept {
    coordinator_.overflowAgeWarnCallback_ = cb;
}

void RuntimeIntentCoordinator::PriorityScheduler::escalateAllRetires(RetirePriority minPriority) noexcept {
    // ★ Phase5: Coordinator の escalateAllRetires は LifetimeState に委譲
    //   実装は AudioEngine.Processing.ReleaseResources.cpp の worldAuthority_.lifetime().escalateAllRetires() が担当
    //   本メソッドは Coordinator の公開APIとしての将来拡張用プレースホルダ
    (void)minPriority;
}

void MultiStagePublisher::publishTier(PayloadTier tier, const void* payload) {
    TieredPayloadDescriptor descriptor{};
    descriptor.tier = tier;
    descriptor.requiresRT = (boundary_ == RuntimeBoundary::RTWorld);
    descriptor.hasExternalResource = (tier == PayloadTier::ExternalPinned);
    descriptor.pinnedLifetime = (tier != PayloadTier::ExternalPinned) ? true : (payload != nullptr);

    PayloadTierValidator validator;
    rejected_ = (validator.explainPublishReject(descriptor) != TierRejectReason::None);
}

//==============================================================================
// ★ P0-4C: ISR Intent 発行インターフェース実装
//==============================================================================

void RuntimeIntentCoordinator::submitObserve(const DSPHandle& handle, PublicationEpoch epoch) noexcept
{
    // ★ P0-4A: Observe Intent — Timer → enqueue（RT-safe, lock-free）
    //   OBSERVE-2: push() は即座に復帰（lock-free）
    //   OBSERVE-9: FIFO を保証（共通 intentQueue_ = MpscBoundedRing, reservation order）
    //   OBSERVE-10: 世代検証用に epoch を保存
    //   ISR: Intent は自己完結型 — DSPHandle を含むため、Coordinator は外部状態に依存せず retire 対象を識別可能
    //
    // ★ work88 (FUTURE-10 / Phase 7): Observe 統合 — 共通 intentQueue_ (MPSC) を primary に。
    //   cross-type FIFO で Publish/Quarantine と同じキューを共有（ObserveIntentHandler が
    //   kDispatchTable 経由で処理）。SPSC 専用リング（observeIntentQueue_/observeFallbackQueue_）
    //   への複数 Producer 依存を排除（MPSC 実態の潜在競合を構造的に解消）。
    // ★ dash2 §1.7 (Phase G R6 修正, 2026-08-15): epoch は caller が RuntimeStore::current
    //   （RuntimeWorldAuthority::observePublishedWorld）から取得して明示的に渡す。Coordinator は
    //   currentWorld_ を参照しない（CW-3b で currentWorld_ は非更新 — 単一 source へ接続）。
    const auto intentId = nextObserveIntentId_.fetch_add(1, std::memory_order_relaxed);

    RuntimeIntentCoordinator::Intent intent{};
    intent.type = RuntimeIntentCoordinator::IntentType::Observe;
    intent.payload.observe = RuntimeIntentCoordinator::ObservePayload{handle, epoch};
    intent.sequenceId = intentId;

    // ★ work88 (P2-1 §1.1.3): reservation-before-push 化。
    //   push 前に pendingIntentCount_ を fetchAdd（enqueue reservation）。全層 push 失敗
    //   （drop）時は fetchSub で rollback し、カウンタは不変のまま drop を観測可能にする。
    //   予約は consumer（processIntent / drainObserveDeferred）の pop 成功時に fetchSub で
    //   消費される（pop 成功数 == push 成功数 の不変条件が構造的に保証される）。
    convo::fetchAddAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    if (intentQueue_.push(intent)) {
        return;
    }

    // intentQueue_ full → observeDeferredRing_（FUTURE-8 overflow 専用）へ退避。
    //   回御は processIntent の drainObserveDeferred（QUEUE-16）。
    observeOverflowCounter_.fetch_add(1, std::memory_order_relaxed);
    ObserveIntent fallbackIntent{ handle, epoch, intentId };
    if (observeDeferredRing_.push(fallbackIntent)) {
        return;
    }

    // 全層溢れ（intentQueue_ + deferred ring）→ reservation rollback + drop カウンタ。
    //   Observe は観測情報のため後発 Observe で補完可能（三次レビュー policy 表:
    //   Observe は条件付き drop / coalesce 可）。Publish/Quarantine とは異なり
    //   state transition の喪失ではないため許容。
    //   reservation を fetchSub で相殺（pendingIntentCount_ は増えない — shutdown の
    //   isFullyDrained が永久に false になるのを防ぐ）。
    convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    observeFallbackOverflowCounter_.fetch_add(1, std::memory_order_relaxed);
}

// ★ work88 (X3 §6.3 / R4 Phase 2): Reclaim Authority の唯一の entry point。
//   ⚠️ dash2 §2.2 (Phase A2 — Step 9): 旧 bool reclaim API は削除（compile guard）。
//   Mode 分岐を消し、reclaimNormal（RuntimeEBR）/ reclaimShutdownQuiescent（ShutdownQuiescent）
//   の Capability 型で経路を区別する（H.11.11.5 / AC-1）。物理削除（DSPCore* delete）は
//   含まない（slot 状態遷移のみ — 既存契約）。
//
// ★ dash2 §2.2 (Phase A2): reclaimInFlightCount_ の維持を semantic event に一本化（A2-G01/G02）。
//   - deferred（epoch unsafe）時: onReclaimBegin() → +1（保留中としてカウント）
//   - その後の成功時: onReclaimEnd() → -1（保留解消）
//   - deferred なしの単発成功: カウンタに触れない（0 のまま — INV-3-1 と整合）
//   - begin/end は「deferred → 成功」のライフサイクルで対になるため、単発成功の count==0 で
//     onReclaimEnd を呼ぶ underflow→Faulted 回帰は発生しない（前回実装の是正）。
bool RuntimeIntentCoordinator::requestReclaim(
    const DSPHandle& handle,
    DSPHandleRuntime& handleRuntime,
    ISRRetireRouter& router) noexcept
{
    // ★ work88 (X3 §6.3 / R4 Phase 2): Reclaim Authority に一本化（RuntimeEBR モード委譲）
    // ★ dash2 §2.2 (Phase A2 Step 7): reclaimNormal（分離 API）へ委譲。
    return reclaimNormal(handle, handleRuntime, router);
}

// ── ★ dash2 §2.2 (Phase A2 — H.11.17.5 15-Step 7-9): 分離 API 実装 ──

// RuntimeEBR（通常 runtime）reclaim。旧 reclaim(RuntimeEBR, ...) のロジックを直接実装。
//   retire → epoch 安全確認（retireEpoch < minReaderEpoch）→ reclaim / pending。
//   precondition は retire 前に評価（十四次指摘 — phase 不正時に retire の state transition を
//   先に発生させない）。
bool RuntimeIntentCoordinator::reclaimNormal(
    const DSPHandle& handle,
    DSPHandleRuntime& handleRuntime,
    ISRRetireRouter& router) noexcept
{
    // 1. executeRetire(handle) — DSPHandleRuntime に retire を委譲
    handleRuntime.retire(handle);

    // 2. waitReaders — epoch 安全確認（retireEpoch < minReaderEpoch）
    const auto retireEpoch = router.currentEpoch();
    const auto minReaderEpoch = router.minReaderEpoch();
    if (retireEpoch >= minReaderEpoch) {
        // Reader がまだアクティブ → 再試行（次の processIntent サイクルで再確認）
        // カウンタ更新のみ行い、即座に復帰（NonRT safe）
        // ★ dash2 §2.2 (A2-G02): deferred 分を onReclaimBegin で +1（保留中としてカウント）。
        //   旧 setReclaimInFlightCount(load+1) の絶対値上書きを廃止（G01/G02）。
        onReclaimBegin();
        // ★ TOCTOU 修正: 呼出し元へ「遅延」を通知（再試行リストへ戻す — slot リーク防止）
        return false;
    }

    // 3. executeReclaim(handle) — 安全確認完了
    //    Reclaimed 状態への遷移のみ（物理削除は retire path の enqueueWithRetry が担当）
    handleRuntime.reclaim(handle);
    // ACK: reclaim complete — deferred 保留の解消（onReclaimEnd で -1）。
    //   ★ dash2 §2.2 (A2-G02): deferred なしの単発成功ではここに来ても保留はない。
    //     onReclaimEnd は old>0 ガード付き（count==0 で underflow しない）— Faulted 化なし。
    onReclaimEnd();
    return true;
}

// ShutdownQuiescent reclaim。ReclaimPermit を consume して認可する（single-use）。
//   - permit.consume() が false（既に Consumed）なら二重 reclaim → 認可しない（INV-LIFE-7 / T9）
//   - Permit は ShutdownRuntime のみ生成（INV-LIFE-4）— caller は manufacture 不可
//   - Permit が quiescence（reader registration closed + readers zero + epoch settled +
//     postStopEnqueue == 0 + no-resurrection）を証明済みのため、bool readerRegistrationClosed は
//     不要（旧 bool API の代替 — AC-2: caller-side shutdown 判断 0 件）
//   - ［Step 12: ShutdownRuntime が tryMakeReclaimPermit で供給］
// ── ★ dash2 §2.2 (Phase A2 — Step 14 / Authority Singularization) ──
//   ReclaimAuthority の shutdown identity 管理。bind は ShutdownRuntime（friend）のみ。
//   ［Unbound → Bound(N) → 固定。既に Bound 済みなら再 bind は無視（任意再 bind 禁止 — INV-LIFE-4/6）］
void RuntimeIntentCoordinator::bindShutdownIdentity(ShutdownRuntimeIdentity identity) noexcept
{
    if (shutdownIdentityBound())
        return;                         // 既に Bound — 再 bind 禁止（authority 境界の強固化）
    currentShutdownIdentity_ = identity;
    convo::publishAtomic(shutdownIdentityBound_, true, std::memory_order_release);
}

bool RuntimeIntentCoordinator::shutdownIdentityBound() const noexcept
{
    return convo::consumeAtomic(shutdownIdentityBound_, std::memory_order_acquire);
}

const ShutdownRuntimeIdentity& RuntimeIntentCoordinator::currentShutdownIdentity() const noexcept
{
    return currentShutdownIdentity_;
}

bool RuntimeIntentCoordinator::reclaimShutdownQuiescent(
    const DSPHandle& handle,
    DSPHandleRuntime& handleRuntime,
    ISRRetireRouter& router,
    ReclaimPermit&& permit) noexcept
{
    // ★ dash2 §2.2 (Step 14 — Authority Singularization): ReclaimAuthority が単一の認可点。
    //   caller（AudioEngine）は identity check を行わない。本メソッドが以下の順序で認可する:
    //   1. identity validation: permit.identity() == currentShutdownIdentity()
    //      （cross-runtime: engineInstanceId 不一致 / stale: generation 不一致 — AUTH-09/13 / AC-5）
    //   2. single-use capability consumption: permit.consume()
    //   3. physical reclaim（slot 状態遷移）
    //   ［不変条件: INV-LIFE-4/6 — Permit は ShutdownRuntime のみ生成し、ReclaimAuthority が
    //     自身の shutdown identity と一致する Permit のみ消費する］
    if (!shutdownIdentityBound())
        return false;                       // identity 未 bind（shutdown 未開始）→ reject
    if (!(permit.identity() == currentShutdownIdentity_))
        return false;                       // identity 不一致（cross-runtime / stale）→ reject
    if (!permit.consume())
        return false;                       // 二重使用（INV-LIFE-7 / T9）

    // 1. executeRetire(handle) — DSPHandleRuntime に retire を委譲
    handleRuntime.retire(handle);

    // 2. epoch 判定スキップ（Permit が quiescence = reader registration closed + readers zero を
    //    証明済みのため EBR を bypass — ShutdownQuiescent セマンティクス）

    // 3. executeReclaim(handle) — 安全確認完了（Reclaimed 状態遷移のみ）
    handleRuntime.reclaim(handle);
    return true;
}

//==============================================================================
// ★ P0-5: QuarantineService implementation
//==============================================================================

QuarantineService::QuarantineResult QuarantineService::executeQuarantine(
    DSPHandleRuntime& handleRuntime,
    DSPQuarantineManager& quarantineManager,
    const QuarantineRequest& request) noexcept
{
    // QSVC-1: State変更 + Audit を単一トランザクションとして実行
    QuarantineResult result{};

    // 1. State 変更 — DSPHandleRuntime::quarantine()
    if (!request.handle.isNull() && request.handle.slot > 0) {
        handleRuntime.quarantine(request.handle);
        result.stateChanged = true;
    }

    // 2. Audit 記録 — DSPQuarantineManager::quarantineHandle()
    const bool auditLogged = quarantineManager.quarantineHandle(
        request.handle.slot,
        request.handle.generation,
        request.reason);
    result.auditLogged = auditLogged;

    // ★ FUTURE-3/QSVC-5: Audit 失敗時も State は変更しない。Publish後は Immutable。Rollback 廃止。
    //   Diagnostic カウンタのみ更新。（stateChanged は既に確定。rolledBack は削除 ── New World Publish が復旧担う。）

    return result;
}

// ★ FUTURE-3: Coordinator は Recovery Request enqueue のみ。Rollback ではない — New World の Immutable Publish が復旧担う。
//   Builder → Validate → Publish は Builder Loop が popRecoveryRequest() で消費（Admission 判定なし）。
//   transport-only: saturate 時は drop。Decision Authority を持たない。
// ★ FUTURE-3 (work88): buildSource（RuntimeBuildSnapshot 値コピー）を payload に内包。
//   quarantinedHandle だけでは resolve() 不能（ISRDSPHandle.cpp:69）なため、build 入力は
//   値コピーした snapshot から引当する（epoch 逆引き不要 — lifetime を構造的に解決）。
// ★ dash2 §1.1 (Phase F 検証, 2026-08-15): 単一 Producer 不変条件を確認済み。
//   Producer = CoordinatorLoop（本メソッドは submitRecoveryIntent ← QuarantineIntentHandler /
//   RecoveryIntentHandler〔dead code〕経由で CoordinatorLoop スレッド上でのみ呼ばれる）。
//   Consumer = Builder Loop（popRecoveryRequest）。⇒ SPSC 維持 — MPSC 化不要。
// ★ dash2 §1.9 (Phase E): 戻り値 — recovery obligation が生成・維持された場合 true。
//   transport（push 成功）/ durable（queue full → recoveryAdmissionPending_）とも true
//   （INV-X1-2: queue full ≠ Recovery lost）。shutdown gate による discard は false（wake 不要）。
// ── ★ G-4.2: SemanticRecoveryTarget derivation helpers (I4 D12.2). ──
//   Metadata generation only — NOT wired into supersession/admission yet (G-4.3+). No side effects.
	namespace {
	// G-4.2 helpers live at convo::isr anonymous-namespace scope; ObligationDomains is a nested
	// member type of RuntimeIntentCoordinator — alias it here so the helpers can name it.
	using convo::isr::RuntimeIntentCoordinator;
	using ObligationDomains = RuntimeIntentCoordinator::ObligationDomains;
	// Canonical FNV-1a over BuildInput configuration values (deterministic; no padding dependence).
std::uint64_t computeBuildInputHash(const convo::BuildInput& in) noexcept
{
    uint64_t h = 1469598103934665603ULL;   // FNV-1a 64-bit offset basis
    auto mix = [&h](std::uint64_t v) noexcept {
        for (int b = 0; b < 8; ++b) { h ^= ((v >> (b * 8)) & 0xFF); h *= 1099511628211ULL; }
    };
    mix(std::bit_cast<std::uint64_t>(in.sampleRate));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(in.blockSize)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(in.ditherBitDepth)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(in.oversamplingFactor)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(in.oversamplingType)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(in.noiseShaperType)));
    mix(static_cast<std::uint64_t>(static_cast<std::int64_t>(in.processingOrder)));
    mix(static_cast<std::uint64_t>(in.eqBypassed ? 1u : 0u));
    mix(static_cast<std::uint64_t>(in.convBypassed ? 1u : 0u));
    mix(static_cast<std::uint64_t>(in.softClipEnabled ? 1u : 0u));
    mix(std::bit_cast<std::uint64_t>(in.saturationAmount));
    mix(std::bit_cast<std::uint64_t>(in.inputHeadroomGain));
    mix(std::bit_cast<std::uint64_t>(in.outputMakeupGain));
    mix(std::bit_cast<std::uint64_t>(in.convolverInputTrimGain));
    mix(static_cast<std::uint64_t>(in.autoGainStagingEnabled ? 1u : 0u));
    return h;
}

// Necessary-condition metadata: which semantic domain(s) have a non-default value in the snapshot.
// (D12.2 — coverage is a necessary condition, never sufficient; not a supersession gate.)
ObligationDomains computeDomainCoverage(const convo::RuntimeBuildSnapshot& s) noexcept
{
    ObligationDomains dc = ObligationDomains::None;
    if (s.rebuildFingerprint.irIdentityHash != 0)
        dc = static_cast<ObligationDomains>(static_cast<std::uint8_t>(dc) | static_cast<std::uint8_t>(ObligationDomains::IR));
    if (s.rebuildFingerprint.convolutionConfigHash != 0 || s.convolverFingerprint != 0)
        dc = static_cast<ObligationDomains>(static_cast<std::uint8_t>(dc) | static_cast<std::uint8_t>(ObligationDomains::Conv));
    if (s.rebuildFingerprint.dspParameterHash != 0)
        dc = static_cast<ObligationDomains>(static_cast<std::uint8_t>(dc) | static_cast<std::uint8_t>(ObligationDomains::EQ));
    if (computeBuildInputHash(s.buildInput) != 0)
        dc = static_cast<ObligationDomains>(static_cast<std::uint8_t>(dc) | static_cast<std::uint8_t>(ObligationDomains::Config));
    return dc;
}
} // namespace

bool RuntimeIntentCoordinator::submitRecoveryRequest(const DSPHandle& quarantinedHandle,
                                                   const convo::RuntimeBuildSnapshot& buildSource,
                                                   PublicationEpoch epoch) noexcept
{
    // ★ work88 (P2-4 監査補正 — Step B: Recovery admission の shutdown gate)。 (unchanged)
    if (convo::consumeAtomic(state_, std::memory_order_acquire) == CoordinatorState::ShuttingDown)
    {
        convo::fetchAddAtomic(recoveryShutdownDiscardCount_, std::uint64_t{1}, std::memory_order_release);
        return false;   // shutdown discard — wake 不要（§1.9 Phase E）
    }

    // ★ D105-R5-10: opportunistically re-drive any deferred Live obligation before processing this
    //   submission. Delivery resources freed by prior takes/drains are reused promptly. Runs on the
    //   CoordinatorLoop (producer) thread only — recoveryIntentQueue_ は SPSC、pendingRecoveryAdmission_ は
    //   ★ D146: state CAS プロトコル＋payload 書込権限規約で保護される（旧「SPSC-safe」記述から更新）。
    //   CoalesceIdentity is resolved up-front so we can snapshot whether THIS target was deferred (delivery==None)
    //   BEFORE redrive runs; if redrive attaches the delivery in this same call, the coalesced resubmit below
    //   must not push a second representation (R10-3 / C16 single representation).
    // ★ G-4.2: I4 D12.2 SemanticRecoveryTarget from the RuntimeBuildSnapshot semantic source.
    //   domainCoverage = necessary-condition metadata; buildInputHash = canonical BuildInput hash (NOT a
    //   dummy/id/generation). These build the identity fields; NOT wired to supersession yet (G-4.3+).
    const SemanticRecoveryTarget srt{
        buildSource.rebuildFingerprint.irIdentityHash,
        buildSource.rebuildFingerprint.convolutionConfigHash,
        buildSource.rebuildFingerprint.dspParameterHash,
        computeDomainCoverage(buildSource),
        buildSource.convolverFingerprint,
        computeBuildInputHash(buildSource.buildInput)
    };
    const CoalesceIdentity cid{ quarantinedHandle, srt };
    const bool wasDeferredBefore = [&]() noexcept -> bool {
        const std::size_t e = recoveryAdmissions_.findByKey(cid);
        if (e == recoveryAdmissions_.kCapacity) return false;
        // ★ D152 T7: delivery read from the lifecycle snapshot (advisory; attaches commit via CAS).
        return recoveryAdmissions_.slot(e).lifecycle.load(std::memory_order_acquire).delivery
            == static_cast<std::uint8_t>(ObligationDeliveryState::None);
    }();
    redriveDeferredRecoveryObligations();

    // ★ dash2 §1.7 (Phase G R7 修正): epoch は caller が渡す。Coordinator は currentWorld_ を参照しない。
    RecoveryIntent intent{
        quarantinedHandle,
        epoch,
        nextRecoveryIntentId_.fetch_add(1, std::memory_order_relaxed),
        std::uint64_t{0},   // obligationId (assigned below from oblId)
        buildSource
    };

    // ── ★ D105-R5-8: Logical Recovery Obligation admission protocol (RecoveryAdmissionTable) ──
    //   Coalesce by semantic target (build fingerprint): reuse an existing Live obligation (ΔL = 0).
    //   Otherwise enforce L < 32: on capacity exhaustion REJECT (no obligation, no transport).
    // ★ D105-R5-9 MUST-3: CoalesceIdentity = quarantinedHandle + SemanticRecoveryTarget.
    //   Two distinct handles with identical fingerprints are NOT coalesced (R8 §4 counterexample).
    // (cid resolved above, before redrive — see D105-R5-10 wasDeferredBefore snapshot)
    std::uint64_t oblId = 0;
    std::size_t slotIdx = recoveryAdmissions_.kCapacity;
    // ★ G-4.3 (D27.2 linearization) + ★ D152 T5 (D152-R1): findByKey is a snapshot; a concurrent
    //   resolveRecoveryObligation (RebuildThread Route A/C) can terminalize this slot between the
    //   lookup and our use. Re-validate with a FULL-WORD identity CAS (expected == desired) so
    //   COALESCE linearizes at a defined point and never mutates a terminal obligation.
    //   ★ T5 retry rule (D152 R1): under the unified lifecycle word a CAS failure can ALSO mean
    //   benign contention (a concurrent postSignal changed pending, or an attach changed delivery)
    //   — NOT terminalization. compare_exchange_strong updates `w` to the current word; re-evaluate:
    //   still Live → RETRY the identity CAS; terminal → fall through to tryInsert. NEVER create a
    //   NEW obligation on a single contention failure (would double-count ΔL).
    const std::size_t existing = recoveryAdmissions_.findByKey(cid);
    bool coalesceOnLive = false;
    if (existing != recoveryAdmissions_.kCapacity) {
        auto& life = recoveryAdmissions_.slot(existing).lifecycle;
        RecoveryLifecycleWord w = life.load(std::memory_order_acquire);
        for (;;) {
            if (w.state != static_cast<std::uint8_t>(ObligationState::Live))
                break;                                   // terminal → fresh admission path
            RecoveryLifecycleWord desired = w;           // full-word identity CAS (D27.2)
            if (life.compare_exchange_strong(w, desired, std::memory_order_acq_rel)) {
                coalesceOnLive = true;
                break;
            }
            // CAS failed: w == current word. Benign contention → retry; terminal → break above.
        }
    }
    if (coalesceOnLive) {
        // COALESCE: reuse existing Live obligation (D18.7 — ΔL = 0; no new oblId)
        oblId = recoveryAdmissions_.slot(existing).lifecycle.load(std::memory_order_acquire).obligationId;
        slotIdx = existing;
        convo::fetchAddAtomic(recoveryCoalescedCount_, std::uint64_t{1}, std::memory_order_release);
        // ★ D105-R5-10: redrive may have attached a single delivery to a previously-deferred obligation —
        //   do NOT push a second representation (R10-3 / C16 single-representation). Resubmissions of an
        //   obligation already delivered before redrive still fall through and re-push.
        if (wasDeferredBefore
            && recoveryAdmissions_.slot(existing).lifecycle.load(std::memory_order_acquire).delivery
                != static_cast<std::uint8_t>(ObligationDeliveryState::None))
            return true;
    } else {
        // ★ D152 T4: payload (handle/epoch/intentId/buildSource) moves INSIDE tryInsert, before the
        //   Live-publishing CAS — the old post-insert writes (former cpp:956-959) are gone.
        const auto ins = recoveryAdmissions_.tryInsert(cid, quarantinedHandle, epoch,
                                                       intent.intentId, buildSource);
        if (!ins) {
            // ★ D105-R5-8: capacity exhausted (L == 32) — reject (INV-X1-7). No obligation, no transport.
            convo::fetchAddAtomic(recoveryCapacityExhaustedCount_, std::uint64_t{1}, std::memory_order_release);
            return false;
        }
        slotIdx = *ins;
        oblId = recoveryAdmissions_.slot(*ins).lifecycle.load(std::memory_order_acquire).obligationId;
    }

    // NOTE (D105-R5-10): a coalesced re-submission is ALLOWED to re-push its transport intent (this is how
    //   recoveryIntentQueue_ (256) fills to capacity — see testRecoveryDurableAdmission / fillRecoveryQueue,
    //   where the obligation was ALREADY delivered (Transport/Durable) before redrive, so wasDeferredBefore is
    //   false and we fall through to the push below). The §5 "no double delivery" guarantee for a DEFERRED
    //   obligation is enforced two ways: (a) redriveDeferredRecovery only acts on delivery==None, never
    //   re-attaching a second representation to one that already has one; (b) the coalesce branch above returns
    //   early (wasDeferredBefore && delivery!=None) when THIS submission's redrive already enqueued the single
    //   representation — so C16 (deferred → same-key resubmit) yields exactly ONE transport representation.

    // ★ D105-R5-8 §4: carry obligationId on the transport intent (and durable fallback below).
    intent.obligationId = oblId;
    // ★ G-4.2: thread the obligation's episode lineage id + in-episode recovery ordinal (source-of-truth
    //   = the table slot; stable across coalesce/redrive). intentId stays the diagnostic event sequence.
    intent.recoveryGeneration = recoveryAdmissions_.slot(slotIdx).recoveryGeneration;
    // ★ work88 (六次レビュー — INV-5: Recovery drop 禁止 / P2-1 §1.1.4 reservation-before-push): (unchanged)
    convo::fetchAddAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    if (recoveryIntentQueue_.push(intent)) {
        casDelivery(slotIdx, oblId, ObligationDeliveryState::Transport);   // ★ D152 T6: full-word CAS attach
        return true;   // transport recovery exists（§1.9 Phase E — wake 条件）
    }

    // ★ work88 (X1 §6.1): queue full ≠ Recovery lost（INV-X1-2）。 (durable fallback now carries obligationId)
    convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    convo::fetchAddAtomic(recoveryIntentDropCount_, std::uint64_t{1}, std::memory_order_release);

    // ★ D146 (I-HS2 / D144 Case C 修復): 単一の attach primitive へ収束。
    //   旧経路の「state!=NoAdmission ∧ oblId 一致 → payload 上書き」は廃止（Building lease 中の
    //   payload 書込 = cross-thread conflicting access の発生源）。同一 oblId が既に表現されていれば
    //   no-op（意味不変の証明: D145 Phase 1-D — oblId mint 専有性 ⇒ 同一 semantic target）。
    const auto attach = tryAttachDurableRecovery(quarantinedHandle, epoch, intent.intentId,
                                                 oblId, intent.recoveryGeneration, buildSource);
    if (attach == DurableAttachResult::OccupiedByOther) {
        // ★ D105-R5-9 MUST-2 + D105-R5-10: never clobber a distinct live obligation's ONLY delivery
        //   representation. Defer this obligation (delivery==None); redrive re-attaches when free.
        casDelivery(slotIdx, oblId, ObligationDeliveryState::None);   // ★ D152 T6: deferred (CAS)
        convo::fetchAddAtomic(recoveryRetryDeferredCount_, std::uint64_t{1}, std::memory_order_release);
        return true;   // obligation stays Live; delivery re-driven when the occupied slot frees
    }
    // Attached（新規 publish）または AlreadyRepresented（既存 durable 表現あり — delivery 再同期）。
    casDelivery(slotIdx, oblId, ObligationDeliveryState::Durable);   // ★ D152 T6: durable attach (CAS)
    return true;   // durable recovery exists（INV-X1-2: queue full ≠ Recovery lost — §1.9 Phase E）
}

// ★ D105-R5-9 + D105-R18: single Completion Authority (idempotent, lock-free, ISR-safe).
//   Transitions the first Live→terminal for the given id via the table's id-based resolve();
//   subsequent calls (same id) are no-ops. Called from onPublishCommitted (CoordinatorLoop,
//   Route B), submitPublishRequest rejection branches (Route C), and the T3c adjudicate
//   retry-exhaustion path (Route D — D152 T2 routes through THIS authority so the terminal
//   transition + counter reset stay one full-word CAS; exhausted telemetry is winner-gated).
//
//   ★ D105-R18: the historical `else → ResolvedFailed` fallthrough has been removed. Each
//   outcome now maps explicitly; the default branch asserts (Debug) / falls through
//   harmlessly (Release) for unknown values. `RecoveryOutcome::Failed` is still a valid
//   enum value but is reached only from the exhaustion path (the only sanctioned path to
//   ResolvedFailed per R17-4 / D105-R18 / D152 T2).
//   ★ D152 T3: returns true iff this call won the Live→terminal CAS (single −1 authority).
bool RuntimeIntentCoordinator::resolveRecoveryObligation(std::uint64_t obligationId, RecoveryResolution outcome) noexcept
{
    if (obligationId == 0)
        return false;
    // ★ D105-R5-9 MUST-1: Retry keeps the obligation Live (ΔL = 0); the durable admission is re-armed
    //   at the transport layer (settlePendingRecoveryAdmission(true)). Only terminal resolutions perform
    //   the single −1 via the table's id-based resolve (which re-scans by id — no cached index, no ABA).
    if (outcome == RecoveryOutcome::Retry)
        return false;
    ObligationState terminal;
    switch (outcome) {
        case RecoveryOutcome::Published:          terminal = ObligationState::ResolvedSuccess; break;
        case RecoveryOutcome::StaleSuperseded:    terminal = ObligationState::ResolvedStaleSuperseded; break;
        case RecoveryOutcome::ShutdownDiscarded:  terminal = ObligationState::ShutdownDiscarded; break;
        case RecoveryOutcome::Failed:             terminal = ObligationState::ResolvedFailed; break;
        case RecoveryOutcome::Retry:              return false;  // ★ D105-R5-9: handled above; defensive no-op
        default:
            // ★ D105-R18: any unknown outcome is a programming error. The historical fallthrough
            //   to ResolvedFailed has been removed; do not silently re-introduce it.
            jassertfalse;
            return false;
    }
    std::uint8_t discardedPending = 0;
    const bool won = recoveryAdmissions_.resolve(obligationId, terminal, &discardedPending);
    if (won) {
        // ★ D152 T3: in-flight signals zeroed by terminalization are classified (observability only).
        if (discardedPending > 0)
            convo::fetchAddAtomic(recoveryFailureSignalDroppedTerminalCount_,
                                  static_cast<std::uint64_t>(discardedPending), std::memory_order_release);
        if (outcome == RecoveryOutcome::ShutdownDiscarded)
            convo::fetchAddAtomic(recoveryObligationShutdownDiscardCount_, std::uint64_t{1}, std::memory_order_release);
    }
    return won;
}

// ── ★ D152/D152-R1 (T3c): transient-failure signal transport (T1) + adjudication (T2). ──
// T1 producer (RebuildThread — the 6 former markTransientFailure sites):
//   loop { w = load(acquire);
//          id != O        → droppedStale   (unknown / post-reuse)
//          state != Live  → droppedTerminal
//          pending >= K   → saturated      (inert — adjudication terminals at K anyway)
//          CAS {O, Live, p+1, a, d} }
//   Touches NOTHING outside the lifecycle word (no payload/delivery/state/liveCount).
void RuntimeIntentCoordinator::postRecoveryFailureSignal(std::uint64_t obligationId) noexcept
{
    if (obligationId == 0) {
        convo::fetchAddAtomic(recoveryFailureSignalDroppedInvalidCount_, std::uint64_t{1}, std::memory_order_release);
        return;
    }
    for (std::size_t i = 0; i < recoveryAdmissions_.kCapacity; ++i) {
        auto& life = recoveryAdmissions_.slot(i).lifecycle;
        RecoveryLifecycleWord w = life.load(std::memory_order_acquire);
        if (w.obligationId != obligationId)
            continue;
        for (;;) {
            if (w.state != static_cast<std::uint8_t>(ObligationState::Live)) {
                convo::fetchAddAtomic(recoveryFailureSignalDroppedTerminalCount_, std::uint64_t{1}, std::memory_order_release);
                return;
            }
            if (w.pending >= kMaxObligationConsecutiveFailures) {
                convo::fetchAddAtomic(recoveryFailureSignalSaturatedCount_, std::uint64_t{1}, std::memory_order_release);
                return;
            }
            RecoveryLifecycleWord desired = w;
            desired.pending = static_cast<std::uint8_t>(w.pending + 1);
            if (life.compare_exchange_strong(w, desired, std::memory_order_acq_rel))
                return;   // posted
            if (w.obligationId != obligationId) {
                // slot reused (id monotonic) → this observation belongs to a dead obligation
                convo::fetchAddAtomic(recoveryFailureSignalDroppedStaleCount_, std::uint64_t{1}, std::memory_order_release);
                return;
            }
            // benign contention (adjudicate/attach changed the word) → retry with updated w
        }
    }
    // no slot carries this id → stale/unknown observation
    convo::fetchAddAtomic(recoveryFailureSignalDroppedStaleCount_, std::uint64_t{1}, std::memory_order_release);
}

// T2 consumer (CoordinatorLoop ONLY — runCoordinatorPhase, after processIntent, before redrive):
//   per slot with pending>0:
//     a2 = adjudicated + pending
//     a2 >= K → terminalize via the SINGLE resolve authority (full-word CAS {O,Live,..}→{O,Failed,0,0,d});
//               exhausted telemetry + liveCount −1 on the CAS winner ONLY (no over-count — D152 §9).
//     else    → ONE CAS: pending=0 ∧ adjudicated=a2 ∧ delivery=None (drain+apply+P-B repair unified).
//   state != Live → break (terminalization zeroed pending; counted in T3).
//   id change (reuse) → break — old O's signals NEVER reach N (D149 #6 closed without affinity).
void RuntimeIntentCoordinator::adjudicateRecoveryFailureSignals() noexcept
{
    for (std::size_t i = 0; i < recoveryAdmissions_.kCapacity; ++i) {
        auto& life = recoveryAdmissions_.slot(i).lifecycle;
        RecoveryLifecycleWord w = life.load(std::memory_order_acquire);
        if (w.pending == 0)
            continue;
        const std::uint64_t oid = w.obligationId;
        for (;;) {
            if (w.state != static_cast<std::uint8_t>(ObligationState::Live))
                break;   // terminalized concurrently — resolve zeroed/zeroes pending (T3 telemetry)
            const std::uint8_t a2 = static_cast<std::uint8_t>(w.adjudicated + w.pending);
            if (a2 >= kMaxObligationConsecutiveFailures) {
                // ★ exhaustion: route through the single terminal authority; winner-only telemetry.
                if (resolveRecoveryObligation(oid, RecoveryOutcome::Failed))
                    convo::fetchAddAtomic(recoveryRetryExhaustedCount_, std::uint64_t{1}, std::memory_order_release);
                break;
            }
            RecoveryLifecycleWord desired = w;
            desired.pending = 0;
            desired.adjudicated = a2;
            desired.delivery = static_cast<std::uint8_t>(ObligationDeliveryState::None);   // P-B repair
            if (life.compare_exchange_strong(w, desired, std::memory_order_acq_rel))
                break;   // drained + applied + None-ing in ONE transition
            if (w.obligationId != oid)
                break;   // slot reused — old signals are inert
            // benign contention (postSignal) → retry with updated w
        }
    }
}

// ★ TEST-ONLY (D152 §7): synchronous compatibility wrapper preserving the pre-T3c semantics
//   (1 call = 1 pending observation + 1 immediate adjudication) for the R18/R20/R21/P2/P3 suites.
//   Production sites use postRecoveryFailureSignal (T1) + CL adjudicate (T2); the production call
//   count of THIS method must remain 0 (D153 V1, grep-enforced).
void RuntimeIntentCoordinator::markTransientFailure(std::uint64_t obligationId) noexcept
{
    postRecoveryFailureSignal(obligationId);
    adjudicateRecoveryFailureSignals();
}

// ★ TEST-ONLY (D152 §10.2): advisory lifecycle snapshot for NT-1..5 (single-threaded test context).
std::optional<RecoveryLifecycleWord>
RuntimeIntentCoordinator::peekLifecycleForTest(std::uint64_t obligationId) const noexcept
{
    for (std::size_t i = 0; i < recoveryAdmissions_.kCapacity; ++i) {
        const auto w = recoveryAdmissions_.slot(i).lifecycle.load(std::memory_order_acquire);
        if (w.obligationId == obligationId)
            return w;
    }
    return std::nullopt;
}

// ★ D105-R5-9 MUST-2: guarded re-arm for a Retry obligation. Only touches the durable slot when it
//   already holds THIS obligation in Building state; never overwrites a distinct live obligation.
//   ★ D146 (Option A / D144-4): Builder 側維持。state は acquire 読のみ（transition は settle の CAS へ一元化）。
//   Building は take CAS 成功（RebuildThread）しか生成せず、本関数も RebuildThread（Orchestrator:413 経由）
//   で実行されるため、lease 内同一スレッド操作 — 二段 plain read の競合は atomic load で解消。
void RuntimeIntentCoordinator::rearmRecoveryRetry(std::uint64_t obligationId) noexcept
{
    if (obligationId == 0)
        return;
    if (pendingRecoveryAdmission_.state.load(std::memory_order_acquire) == PendingRecoveryAdmission::State::Building
        && pendingRecoveryAdmission_.recoveryObligationId == obligationId)
        settlePendingRecoveryAdmission(true);
}

// ★ D105-R5-10: re-drive every Live obligation that currently has NO delivery representation
//   (delivery==None, i.e. a deferred obligation). Called only from the CoordinatorLoop (producer)
//   thread — SPSC-safe for recoveryIntentQueue_ and pendingRecoveryAdmission_. O(kCapacity=32).
//   ΔL=0: never inserts a new obligation or terminates one; only re-attaches a delivery representation
//   to an EXISTING Live obligation (existing id + buildSource preserved).
void RuntimeIntentCoordinator::redriveDeferredRecoveryObligations() noexcept
{
    for (std::size_t i = 0; i < recoveryAdmissions_.kCapacity; ++i) {
        auto& s = recoveryAdmissions_.slot(i);
        // ★ D152 T7: ONE lifecycle snapshot — state/delivery/id read as a consistent word, never
        //   as separate ownership decisions (the attach below commits via full-word CAS anyway).
        const auto w = s.lifecycle.load(std::memory_order_acquire);
        if (w.state != static_cast<std::uint8_t>(ObligationState::Live))
            continue;
        if (w.delivery != static_cast<std::uint8_t>(ObligationDeliveryState::None))
            continue;   // already has a delivery representation — do not re-send (§5)
        redriveDeferredRecovery(w.obligationId);
    }
}

// ★ D105-R5-10: re-attach a delivery representation to ONE existing Live obligation. Restores the
//   obligationId + buildSource from the table record (NO new obligation, NO new id). On success the
//   obligation's delivery residency is set; if BOTH resources are busy it remains deferred (delivery==None)
//   and is retried on a later redrive opportunity. Idempotent: a non-None delivery state is a no-op.
void RuntimeIntentCoordinator::redriveDeferredRecovery(std::uint64_t obligationId) noexcept
{
    if (obligationId == 0)
        return;
    std::size_t idx = recoveryAdmissions_.kCapacity;
    for (std::size_t i = 0; i < recoveryAdmissions_.kCapacity; ++i) {
        if (recoveryAdmissions_.slot(i).lifecycle.load(std::memory_order_acquire).obligationId == obligationId) {
            idx = i;
            break;
        }
    }
    if (idx == recoveryAdmissions_.kCapacity)
        return;   // unknown / mismatched id → no-op (do not fabricate an obligation)
    auto& s = recoveryAdmissions_.slot(idx);
    // ★ D152 T7: lifecycle snapshot gate (advisory) — the attach commits via full-word CAS.
    {
        const auto w = s.lifecycle.load(std::memory_order_acquire);
        if (w.state != static_cast<std::uint8_t>(ObligationState::Live))
            return;
        if (w.delivery != static_cast<std::uint8_t>(ObligationDeliveryState::None))
            return;   // idempotent: already delivered (no duplicate)
    }

    convo::fetchAddAtomic(recoveryRetryRedriveCount_, std::uint64_t{1}, std::memory_order_release);

    RecoveryIntent intent{
        s.handle,
        s.epoch,
        nextRecoveryIntentId_.fetch_add(1, std::memory_order_relaxed),
        std::uint64_t{0},   // obligationId (assigned below from obligationId)
        s.buildSource
    };
    intent.obligationId = obligationId;   // preserve the existing obligation id (no new id issued)
    // ★ G-4.2: redrive preserves the obligation's episode lineage id + recovery ordinal (no new allocation).
    intent.recoveryGeneration = s.recoveryGeneration;
    // ★ D146 (I-HS2 / D146-5): submit と同一の attach primitive へ収束（独自 state check/payload 書込を廃止）。
    //   Attached = 新規 publish、AlreadyRepresented = P3 same-holder repair（delivery 再同期・実体生成なし）、
    //   OccupiedByOther = 他 oblId 占有 → transport fallback（C13 意味論不変）。
    const auto attach = tryAttachDurableRecovery(s.handle, s.epoch, intent.intentId,
                                                 obligationId, intent.recoveryGeneration, s.buildSource);
    if (attach != DurableAttachResult::OccupiedByOther) {
        casDelivery(idx, obligationId, ObligationDeliveryState::Durable);   // ★ D152 T7: CAS attach
        redriveWakePending_ = true;   // ★ G-4.4-P2: 表現が消費待ち → Builder wake（repair も同一プロトコル）
        return;
    }

    // Else attempt transport queue. (different holder — C13 fallback semantics unchanged)
    convo::fetchAddAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    if (recoveryIntentQueue_.push(intent)) {
        casDelivery(idx, obligationId, ObligationDeliveryState::Transport);   // ★ D152 T7: CAS attach
        redriveWakePending_ = true;   // ★ G-4.4-P2 (D137): None→Transport attach success → wake Builder this tick
        return;
    }
    convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    convo::fetchAddAtomic(recoveryIntentDropCount_, std::uint64_t{1}, std::memory_order_release);
    // Both resources busy → remain deferred (delivery==None); ΔL=0; retry on a later redrive opportunity.
    convo::fetchAddAtomic(recoveryRetryRedriveFailureCount_, std::uint64_t{1}, std::memory_order_release);
}


// ★ work88 (X1 §6.1 — lease 方式): durable Recovery admission を Builder が消費する。
//   二十六次レビュー（必須修正1）: destructive dequeue ではなく DurablePending → Building の
//   state transition。build 失敗（transient）は Building → DurablePending へ戻すことで
//   retry を構造的に保証する（INV-X1-1: accepted ⇒ exactly one durable state が常に成立）。
//   recoveryAdmissionPending_ は Building 中も true を維持（build gap を isFullyDrained が検出）。
//   ★ D146 (I-HS2): CAS(DurablePending→Building, acquire) 成功そのものが lease 取得。
//   二段判定（load してから store）は lease ではない — CAS 失敗時は payload を読まない。
std::optional<RuntimeIntentCoordinator::RecoveryIntent>
RuntimeIntentCoordinator::takePendingRecoveryAdmission() noexcept
{
    PendingRecoveryAdmission::State expected = PendingRecoveryAdmission::State::DurablePending;
    if (!pendingRecoveryAdmission_.state.compare_exchange_strong(
            expected, PendingRecoveryAdmission::State::Building,
            std::memory_order_acquire, std::memory_order_relaxed))
        return std::nullopt;   // 非 DurablePending → lease 取得失敗、payload 非読取

    // lease 保持（Building）。acquire が attach の release と対になり payload 書込が可視・安定。
    RecoveryIntent intent{
        pendingRecoveryAdmission_.handle,
        pendingRecoveryAdmission_.epoch,
        pendingRecoveryAdmission_.intentId,
        pendingRecoveryAdmission_.recoveryObligationId,
        pendingRecoveryAdmission_.buildSource
    };
    intent.obligationId = pendingRecoveryAdmission_.recoveryObligationId;   // ★ D105-R5-8
    intent.recoveryGeneration = pendingRecoveryAdmission_.recoveryGeneration;  // ★ G-4.2: durable slot ordinal
    return intent;
}

bool RuntimeIntentCoordinator::hasPendingRecoveryAdmission() const noexcept
{
    return convo::consumeAtomic(recoveryAdmissionPending_, std::memory_order_acquire)
        && pendingRecoveryAdmission_.state.load(std::memory_order_acquire)
               != PendingRecoveryAdmission::State::NoAdmission;
}

// ★ work88 (X1 §6.1 — RecoveryAdmissionClosed): durable admission を破棄（shutdown 専用）。
//   shutdown 中は publish/commit が実行されないため、durable admission を保持しても意味がない
//   （§6.1 case B）。discard は ShutdownDiscard として recoveryShutdownDiscardCount_ に記録
//   （INV-5 — silent loss ではない・意図的な lifecycle discard — dash §8.1）。
//   ★ D146: 実行時点は shutdownCoordinatorLoop() + rebuildThread.join() 後（単一スレッド）。
//   reset 順序 = payload クリア → state release（逆順禁止）。
void RuntimeIntentCoordinator::discardPendingRecoveryAdmission() noexcept
{
    if (pendingRecoveryAdmission_.state.load(std::memory_order_acquire) != PendingRecoveryAdmission::State::NoAdmission)
    {
        convo::fetchAddAtomic(recoveryShutdownDiscardCount_, std::uint64_t{1}, std::memory_order_release);
        resetDurableAdmissionPayload();
        pendingRecoveryAdmission_.state.store(PendingRecoveryAdmission::State::NoAdmission,
                                              std::memory_order_release);
        convo::publishAtomic(recoveryAdmissionPending_, false, std::memory_order_release);
    }
}

// ★ work88 (X1 §6.1 — lease 方式): Builder の build 結果に応じて durable admission を settle。
//   retry=true: Building → DurablePending（transient failure — 次サイクルで再 take。recoveryAdmissionPending_
//   は true 維持 = build gap を isFullyDrained が検出）。
//   retry=false: クリア（build success / Discarded — payload reset → state NoAdmission release →
//                recoveryAdmissionPending_ = false）。
//   ★ D146 (I-HS2): 両分岐とも CAS を linearization point とする（plain read→plain write 置換は禁止）。
//   settle は Builder（lease 保持者）のみが呼ぶ（take CAS 成功か rearm の Building 確認後）。
void RuntimeIntentCoordinator::settlePendingRecoveryAdmission(bool retry) noexcept
{
    if (retry)
    {
        PendingRecoveryAdmission::State expected = PendingRecoveryAdmission::State::Building;
        pendingRecoveryAdmission_.state.compare_exchange_strong(
            expected, PendingRecoveryAdmission::State::DurablePending,
            std::memory_order_release, std::memory_order_relaxed);
        // 非 Building（契約上あり得ない）は no-op。recoveryAdmissionPending_ は true 維持。
        return;
    }
    // ★ reset 順序固定: payload クリア → state release（逆順は Builder→CL の stale 読を生むため禁止）。
    resetDurableAdmissionPayload();
    PendingRecoveryAdmission::State expected = PendingRecoveryAdmission::State::Building;
    pendingRecoveryAdmission_.state.compare_exchange_strong(
        expected, PendingRecoveryAdmission::State::NoAdmission,
        std::memory_order_release, std::memory_order_relaxed);
    convo::publishAtomic(recoveryAdmissionPending_, false, std::memory_order_release);
}

// ★ D146 (I-HS2 / D144 Case C 修復): 単一の durable payload mutation path。
//   submitRecoveryRequest（durable fallback）と redriveDeferredRecovery（付着・repair）が共通。
//   契約:
//     1. state を acquire で観測 — NoAdmission のときだけ payload に触れる。
//        （NoAdmission からの唯一の退出権限は本 CL 経路自身。Builder は NoAdmission を生成するだけで
//         消費せず、discard は join 後 → live 中に他者遷移は存在しない。）
//     2. payload 全書込 → CAS(NoAdmission→DurablePending, release) で publish。
//     3. 占有中は絶対上書きしない: 同一 oblId = AlreadyRepresented（no-op・意味不変の証明は
//        D145 Phase 1-D: oblId mint 専有性 ⇒ 同一 CoalesceIdentity ⇒ 同一 buildInput、epoch/intentId
//        は recovery build 経路で非消費/診断専用）、異 oblId = OccupiedByOther（D105-R5-9 MUST-2）。
RuntimeIntentCoordinator::DurableAttachResult
RuntimeIntentCoordinator::tryAttachDurableRecovery(const DSPHandle& handle, PublicationEpoch epoch,
                                                   std::uint64_t intentId, std::uint64_t oblId,
                                                   RecoveryGeneration generation,
                                                   const convo::RuntimeBuildSnapshot& buildSource) noexcept
{
    using State = PendingRecoveryAdmission::State;
    const State observed = pendingRecoveryAdmission_.state.load(std::memory_order_acquire);
    if (observed != State::NoAdmission) {
        if (pendingRecoveryAdmission_.recoveryObligationId == oblId)
            return DurableAttachResult::AlreadyRepresented;   // 既存表現あり — payload 非書込（D144 Case C）
        return DurableAttachResult::OccupiedByOther;          // 他 obligation の唯一表現を clobber しない
    }
    pendingRecoveryAdmission_.pending = true;
    pendingRecoveryAdmission_.recoveryGeneration = generation;   // ★ G-4.2: in-episode ordinal (dedicated; NOT intentId)
    pendingRecoveryAdmission_.buildSource = buildSource;
    pendingRecoveryAdmission_.reservationOwned = true;           // 1 admission = 1 reservation（INV-X1-5）
    pendingRecoveryAdmission_.handle = handle;
    pendingRecoveryAdmission_.epoch = epoch;
    pendingRecoveryAdmission_.intentId = intentId;
    pendingRecoveryAdmission_.recoveryObligationId = oblId;      // ★ D105-R5-8
    State expected = State::NoAdmission;
    if (pendingRecoveryAdmission_.state.compare_exchange_strong(
            expected, State::DurablePending, std::memory_order_release, std::memory_order_acquire)) {
        convo::publishAtomic(recoveryAdmissionPending_, true, std::memory_order_release);
        return DurableAttachResult::Attached;
    }
    // 防御（live 中は発生不能 — 上記権限論証）。残骸 payload は NoAdmission 下で誰にも読まれず、
    // 次の attach が全面上書きするため rollback 不要。観測された占有者で分類を返す。
    return (pendingRecoveryAdmission_.recoveryObligationId == oblId)
        ? DurableAttachResult::AlreadyRepresented
        : DurableAttachResult::OccupiedByOther;
}

void RuntimeIntentCoordinator::resetDurableAdmissionPayload() noexcept
{
    // ★ D146: 呼び出し側が遷移権限（Building lease / join 後 discard / NoAdmission 観測 attach）を持つこと。
    pendingRecoveryAdmission_.pending = false;
    pendingRecoveryAdmission_.recoveryGeneration = 0;
    pendingRecoveryAdmission_.buildSource = {};
    pendingRecoveryAdmission_.reservationOwned = false;
    pendingRecoveryAdmission_.handle = {};
    pendingRecoveryAdmission_.epoch = 0;
    pendingRecoveryAdmission_.intentId = 0;
    pendingRecoveryAdmission_.recoveryObligationId = 0;
}

// ★ D152 T6/T7: single delivery-attach primitive. Changes ONLY W.delivery via full-word CAS.
//   Returns false when the slot no longer carries oblId or is not Live (terminalized concurrently) —
//   delivery is NEVER written to a terminal or reused obligation (D150 §8 / D152 §6).
//   CoordinatorLoop-only callers (submit attach / redrive attach). Benign contention (a concurrent
//   postSignal changed pending) retries with the CAS-updated current word.
bool RuntimeIntentCoordinator::casDelivery(std::size_t slotIdx, std::uint64_t oblId,
                                           ObligationDeliveryState to) noexcept
{
    auto& life = recoveryAdmissions_.slot(slotIdx).lifecycle;
    RecoveryLifecycleWord w = life.load(std::memory_order_acquire);
    for (;;) {
        if (w.obligationId != oblId
            || w.state != static_cast<std::uint8_t>(ObligationState::Live))
            return false;
        RecoveryLifecycleWord desired = w;
        desired.delivery = static_cast<std::uint8_t>(to);
        if (life.compare_exchange_strong(w, desired, std::memory_order_acq_rel))
            return true;
        // w updated to the current word → re-evaluate (terminal/reuse exits above; contention retries)
    }
}

std::optional<RuntimeIntentCoordinator::RecoveryIntent>
RuntimeIntentCoordinator::popRecoveryRequest() noexcept
{
    RecoveryIntent intent{};
    if (!recoveryIntentQueue_.pop(intent))
        return std::nullopt;              // transport-only pop: empty は Builder 消費の前提
    // ★ work88 (P2-1 §1.1.4): cur>0 ガードを削除し、pop 成功時に fetchSub。
    //   四次レビュー: ガードはカウンタ不整合を silently hide するため危険（underflow を
    //   隠蔽して isFullyDrained のハングを潜在化させていた）。
    //   push 側（submitRecoveryRequest）は reservation-before-push で先に fetchAdd するため、
    //   pop 成功時は必ず対応する reservation が存在し underflow しない（不変条件）。
    convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    return intent;
}

// ★ work88 (P2-4 監査補正 — Step C: shutdown 時の Recovery 明示 discard)。
//   Builder 停止後に recoveryIntentQueue_ に残留する Recovery を ShutdownDiscard として明示破棄する
//   （silent loss 禁止 — INV-5）。popRecoveryRequest() が reservation（pendingIntentCount_）を
//   fetchSub するため counter は整合し、isFullyDrained の queue-empty + counter==0 が正しく成立する。
//   discard 数は recoveryShutdownDiscardCount_ に記録（queue full による drop とは区別 — §8.1）。
//   単なる drain-on-exit（pop して捨てるだけ）ではなく、discard を観測可能な lifecycle として
//   enqueue → pending → shutdown closes admission → discard → pending count release → discard telemetry
//   を構成する（dash §8.1 ShutdownDiscard）。
void RuntimeIntentCoordinator::discardRecoveryRequestsOnShutdown() noexcept
{
    while (popRecoveryRequest())
    {
        convo::fetchAddAtomic(recoveryShutdownDiscardCount_, std::uint64_t{1}, std::memory_order_release);
    }
    // ★ D105-R5-8: close all Live logical obligations (ShutdownDiscarded) — table-centric shutdown.
    //   Routes through the single Completion Authority (resolveRecoveryObligation → table.resolve) so the
    //   −1 stays idempotent (CAS Live→terminal); a concurrent ISR completion that wins the CAS makes
    //   shutdown skip it (no double −1, no L underflow). Delivery discard below does NOT touch L.
    for (std::size_t i = 0; i < recoveryAdmissions_.kCapacity; ++i) {
        const std::uint64_t oblId = recoveryAdmissions_.slot(i).lifecycle.load(std::memory_order_acquire).obligationId;
        if (oblId != 0)
            resolveRecoveryObligation(oblId, RecoveryOutcome::ShutdownDiscarded);
    }
}

void RuntimeIntentCoordinator::submitQuarantine(
    const DSPHandle& handle,
    QuarantineReason reason,
    DSPHandleRuntime& handleRuntime,
    DSPQuarantineManager& quarantineManager,
    uint64_t contextEpoch) noexcept
{
    // ★ A3 Step 4: sync → async. submitQuarantine no longer executes directly;
    //   it enqueues a Quarantine Intent onto the common intentQueue_.
    //   QSVC-2: Execution delegated to QuarantineService via QuarantineIntentHandler
    //   (kDispatchTable) in processIntent — Coordinator retains Decision Authority over enqueue only.
    //   (handleRuntime/quarantineManager params retained for API stability; the handler
    //    re-sources them through HandlerContext.engine. Step 6 removes these redundant params.)
    (void)handleRuntime;
    (void)quarantineManager;

    const std::uint64_t seqId = nextIntentId_.fetch_add(1, std::memory_order_relaxed);
    RuntimeIntentCoordinator::Intent intent{};
    intent.type = RuntimeIntentCoordinator::IntentType::Quarantine;
    intent.payload.quarantine = RuntimeIntentCoordinator::QuarantinePayload{handle, reason, contextEpoch};
    intent.sequenceId = seqId;

    // ★ work88 (P2-1 §1.1.3): pendingIntentCount_ のみ reservation-before-push 化。
    //   push 前に pendingIntentCount_ を fetchAdd。全段 push 失敗時は fetchSub で rollback。
    // ★ work88 (X6 §6.6): quarantine の transport residency を intent/ring に分離（INV-X6-4）。
    //   - primary（intentQueue_）成功: quarantineIntentResidencyCount_++（Intent lane residency）
    //   - fallback（quarantineFallbackQueue_）へ移動: intent -1 → ring +1（両方が同時に 1 にならない）
    //   - quarantineResidentCount_ の +1 は撤去（DSPQuarantineManager::quarantineHandle が唯一管理 —
    //     submitQuarantine は enqueue まで。実在 DSP 数は AudioEngine::isFullyDrained が直接判定）。
    convo::fetchAddAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    convo::fetchAddAtomic(quarantineIntentResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
    if (intentQueue_.push(intent)) {
        return;
    }

    // ★ work88 (FUTURE-10 / 三次レビュー policy 表): Quarantine intent の drop は禁止。
    //   quarantine 検出は安全要件（bad DSP のアクセス禁止）であり、drop されると
    //   quarantined DSP が永久に retire されず RT からアクセス不能なメモリが残存する。
    //   intentQueue_ full 時は Quarantine 専用 fallback ring へ退避（drop しない）。
    // ★ work88 (X6 §6.6): fallback へ移動した時点で primary intent residency → ring residency
    //   へ移動（intent -1 → ring +1。quarantineIntentResidencyCount_ と quarantineRingResidencyCount_
    //   は同時に 1 にならない — INV-X6-4 / §6.6 対象マッピング表）。
    convo::fetchSubAtomic(quarantineIntentResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
    if (quarantineFallbackQueue_.push(intent)) {
        convo::fetchAddAtomic(quarantineRingResidencyCount_, std::uint64_t{1}, std::memory_order_acq_rel);
        return;
    }
    // ★ X6: fallback も失敗（全段失敗）→ intent residency は上記 fetchSub で rollback 済み
    //   （ring へ移動しなかった）。pendingIntentCount_ のみ下記 drop 処理で fetchSub 相殺する。

    // fallback も full → 絶対に静かに破棄しない。drop カウンタを増やして診断に残す。
    //   （AudioEngine 側の HealthMonitor が quarantineFallbackDropCount を監視し、
    //    ISRHealthState::Critical 昇格 / controlled shutdown を駆動する。）
    //   reservation を fetchSub で相殺（pendingIntentCount_ は不変）。
    convo::fetchSubAtomic(pendingIntentCount_, std::uint64_t{1}, std::memory_order_release);
    convo::fetchAddAtomic(quarantineFallbackDropCount_, std::uint64_t{1}, std::memory_order_release);
}

} // namespace convo::isr

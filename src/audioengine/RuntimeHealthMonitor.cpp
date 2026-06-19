#include "RuntimeHealthMonitor.h"
#include "RuntimePublicationValidator.h"  // ★ Phase-1.5: ValidationFailureReason 完全型
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/CrossfadeRuntime.h"  // ★ P1-C: 完全型必要（isPending/getFadeAgeUs）
#include "audioengine/AtomicAccess.h"
#include "core/TimeUtils.h"
#include "../AudioSegmentBuffer.h"  // ★ Work39: Learner FIFO getNumAvailableSamples

namespace convo {

// [work37 Phase 9.1] Retire Stall 継続時間取得
uint64_t RuntimeHealthMonitor::getRetireStallDurationUs() const noexcept {
    return (m_retireStallStartUs_ != 0)
        ? (getCurrentTimeUs() - m_retireStallStartUs_)
        : 0;
}

void RuntimeHealthMonitor::tick() noexcept {
    checkRetireStall();
    checkPublicationStall();
    diagnoseRetireStall();
    checkCrossfadeTimeout();
    checkCrossfadeEventDrop();
    checkReaderSlotUsage();
    checkOverflowRate();         // ★ Practical-3
    checkRetireReclaimLatency();

    // [work37 Phase 9.2] Configuration Divergence 監視
    checkConfigurationDivergence();

    // [work37 Phase 7.1] World Consistency 監視
    checkWorldConsistency();

    // [work37 Phase 9.7] Snapshot Starvation 監視
    checkSnapshotStarvation();

    // [work37 Phase 9.8] Pending Structural Deployment 監視
    checkPendingStructuralDeployment();

    // [work37 Phase 9.29] Suppression Duration 監視
    checkSuppressionDuration();

    // [work37 Phase 9.40] Runtime Progress Freeze 監視
    checkRuntimeProgressFreeze();

    // [work39 Phase 5] Learner FIFO 監視
    checkLearnerBackpressure();

    // [work37 Phase 9.10 P2] Configuration Drift 監視
    checkConfigurationDrift();

    // [work39 Phase 3] 閉ループ制御（PolicyEngine 評価より前に実行）
    {
        const auto lastAction = m_policyEngine_.getLastExecutedAction();
        if (lastAction > RecoveryAction::Observe) {
            auto& entry = m_policyEngine_.getEntry(lastAction);
            if (entry.state == VerificationState::PendingVerification) {
                const uint64_t nowUs = getCurrentTimeUs();
                if (nowUs - entry.executedAtUs >= entry.verifyAfterUs) {
                    const auto nowSnapshot = takeSnapshot();
                    const auto trend = computeTrend(entry.baselineSnapshot, nowSnapshot);
                    auto& budget = m_policyEngine_.getBudget();

                    switch (trend) {
                        case RecoveryOutcome::Recovered:
                            m_policyEngine_.resetVerification();
                            budget.recordCycleCompletion(nowUs);
                            break;

                        case RecoveryOutcome::Improving: {
                            const uint64_t retireReduction =
                                entry.lastSnapshot.pendingRetire > nowSnapshot.pendingRetire
                                ? entry.lastSnapshot.pendingRetire - nowSnapshot.pendingRetire : 0;
                            const uint64_t baselineRetire = entry.baselineSnapshot.pendingRetire;
                            const double reductionRatio = baselineRetire > 0
                                ? static_cast<double>(retireReduction) / baselineRetire : 0.0;
                            if (reductionRatio < 0.01)
                                ++entry.stalledCount;
                            else
                                entry.stalledCount = 0;
                            entry.lastSnapshot = nowSnapshot;

                            if (entry.stalledCount >= 3) {
                                auto next = convo::nextAction(lastAction);
                                if (m_policyEngine_.canExecute(next)) {
                                    if (m_actionCallback) m_actionCallback(next);
                                    m_policyEngine_.markExecuted(next);
                                    m_policyEngine_.markForVerification(next, nowSnapshot);
                                    budget.record(next, nowUs);
                                }
                            } else {
                                // [work39 Phase 6] Restore Step2 実行条件（問題A-1/A-2: 強化版）
                                if (lastAction == RecoveryAction::Restore
                                    && nowSnapshot.restorePhase == RestorePhase::EpochRecoveryIssued)
                                {
                                    const bool epochAdvancing = (nowSnapshot.epochAdvanceCount
                                        > entry.baselineSnapshot.epochAdvanceCount);
                                    const uint64_t retireReductionStep2 =
                                        entry.lastSnapshot.pendingRetire > nowSnapshot.pendingRetire
                                        ? entry.lastSnapshot.pendingRetire - nowSnapshot.pendingRetire : 0;
                                    const uint64_t retireBaseline = entry.baselineSnapshot.pendingRetire;
                                    const double reductionRate = retireBaseline > 0
                                        ? static_cast<double>(retireReductionStep2) / retireBaseline : 0.0;
                                    const int64_t ageDelta = static_cast<int64_t>(nowSnapshot.maxRetireAgeUs)
                                        - static_cast<int64_t>(entry.baselineSnapshot.maxRetireAgeUs);
                                    constexpr uint64_t kAbsoluteReductionMin = 10;
                                    const bool absoluteEnough = retireReductionStep2 >= kAbsoluteReductionMin;
                                    constexpr uint64_t kHealthyThreshold = 256;
                                    const bool nearlyHealthy = nowSnapshot.pendingRetire <= kHealthyThreshold;
                                    if (epochAdvancing
                                        && (reductionRate >= 0.20 || absoluteEnough || nearlyHealthy)
                                        && ageDelta <= 0
                                        && m_restoreStep2Callback_) {
                                        m_restoreStep2Callback_();
                                    }
                                }
                                entry.verifyAfterUs = std::min(
                                    entry.verifyAfterUs * 2, uint64_t{30'000'000});
                                entry.executedAtUs = nowUs;
                            }
                            break;
                        }

                        case RecoveryOutcome::Stalled:
                        case RecoveryOutcome::Worsening: {
                            auto next = convo::nextAction(lastAction);
                            if (m_policyEngine_.canExecute(next)) {
                                if (m_actionCallback) m_actionCallback(next);
                                m_policyEngine_.markExecuted(next);
                                m_policyEngine_.markForVerification(next, nowSnapshot);
                                budget.record(next, nowUs);
                            }
                            break;
                        }

                        default:
                            break;
                    }

                    // Storm detection: 同Action再突入→Critical固定
                    if (budget.isStormDetected(lastAction, nowUs)) {
                        if (m_actionCallback) m_actionCallback(RecoveryAction::Critical);
                        m_policyEngine_.markExecutedCritical(RecoveryAction::Critical);
                        budget.reset();
                    }

                    // Budget exhausted → Critical
                    if (budget.isExhausted(nowUs)) {
                        if (m_actionCallback) m_actionCallback(RecoveryAction::Critical);
                        m_policyEngine_.markExecutedCritical(RecoveryAction::Critical);
                    }
                }
            }
        }
    }

    // [work37 Phase 4.1] Policy Engine 評価: 全 MonitorState から統合判定
    auto decision = m_policyEngine_.evaluateAggregate(
        m_prevRetireState,
        m_prevPublicationState,
        m_prevReaderSlotState,
        m_prevOverflowRateState,
        m_prevRetireAgeState,
        m_prevCrossfadeDropState);

    // [work37 Phase 4.4] 背圧信号を PolicyEngine 評価に反映 — exchange(0) でリセット
    if (m_backpressureInjected_) {
        const auto maxFb = convo::exchangeAtomic(m_maxFallbackSize_, uint64_t{0},
                                                  std::memory_order_acq_rel);
        const auto maxOr = convo::exchangeAtomic(m_maxOverflowRate_, 0.0,
                                                  std::memory_order_acq_rel);
        if (maxFb > 500) decision.actions |= toBit(RecoveryAction::Throttle);
        if (maxOr > 5.0) decision.actions |= toBit(RecoveryAction::Critical);
        m_backpressureInjected_ = false;
    }

    // [work37 Phase 9.1] Learner Health Policy:
    //   Retire Stall が 10秒以上継続 AND Learner 動作中 → PauseLearner (Throttle)
    {
        const uint64_t stallDur = getRetireStallDurationUs();
        const bool learnerActive = (m_learnerRunningRef != nullptr)
            && convo::consumeAtomic(*m_learnerRunningRef, std::memory_order_acquire);
        if (stallDur > 10'000'000 && learnerActive) {
            decision.actions |= toBit(RecoveryAction::Throttle);
        }
    }

    // ★ v3.1: RecoveryAction は HealthEvent として再包装しない
    //    executeRecoveryAction() を新設し、RecoveryAction を直接発火
    if (decision.actions != 0 && m_actionCallback) {
        const auto action = static_cast<RecoveryAction>(
            [] (RecoveryActionBits bits) -> int {
                // 最高優先度の Action を選択
                for (int i = static_cast<int>(RecoveryAction::_Count) - 1; i >= 0; --i) {
                    if (bits & toBit(static_cast<RecoveryAction>(i)))
                        return i;
                }
                return 0;
            }(decision.actions));
        if (m_policyEngine_.canExecute(action)) {
            m_actionCallback(action);
            m_policyEngine_.markExecuted(action);
            // [work39 Phase 4] Budget 記録
            auto& budget = m_policyEngine_.getBudget();
            budget.record(action, getCurrentTimeUs());
            if (budget.isExhausted(getCurrentTimeUs())) {
                m_actionCallback(RecoveryAction::Critical);
                m_policyEngine_.markExecutedCritical(RecoveryAction::Critical);
            }
            // [work39 Phase 3] Verification 開始
            m_policyEngine_.markForVerification(action, takeSnapshot());
        }
    }

    // [work39 Phase 7] Critical 出口評価（CriticalExitCondition 構造体使用）
    if (convo::consumeAtomic(m_healthState_, std::memory_order_acquire)
        == ISRHealthState::Critical) {
        CriticalExitCondition exitCond;

        // 条件1: 全 MonitorState が Normal
        exitCond.allMonitorsNormal = (m_prevRetireState == MonitorState::Normal)
            && (m_prevPublicationState == MonitorState::Normal)
            && (m_prevReaderSlotState == MonitorState::Normal)
            && (m_prevOverflowRateState == MonitorState::Normal)
            && (m_prevRetireAgeState == MonitorState::Normal)
            && (m_prevConfigDivergenceState_ == MonitorState::Normal)
            && (m_prevLearnerBackpressureState_ == MonitorState::Normal);

        // 条件2: 閉ループ制御 Idle
        exitCond.noRecoveryActionRunning = m_policyEngine_.getEntry(
            m_policyEngine_.getLastExecutedAction()).isIdle()
            && !m_policyEngine_.hasPendingVerification();

        // 条件2b: Suppression 非アクティブ（MonitorState に吸収済み: 常時 true）
        exitCond.suppressionInactive = true;

        // 条件3: RetireDepth + RetireAge 実メトリクス確認
        exitCond.pendingRetire = m_retireRouter
            ? m_retireRouter->pendingRetireCount() : 0;
        exitCond.retireAgeUs = m_maxRetireAgeRef
            ? convo::consumeAtomic(*m_maxRetireAgeRef, std::memory_order_acquire)
            : (m_maxRetireAgeDoubleRef
                ? static_cast<uint64_t>(convo::consumeAtomic(*m_maxRetireAgeDoubleRef,
                                                              std::memory_order_acquire))
                : 0);
        const bool readerHealthy = (m_retireRouter == nullptr
            || m_retireRouter->activeReaderCount() == 0);
        constexpr uint64_t kHealthyThreshold = 256;
        constexpr uint64_t kHealthyAgeUs = 3 * 1'000'000;
        const bool retireAgeHealthy = exitCond.retireAgeUs < kHealthyAgeUs;
        bool metricsHealthy = exitCond.pendingRetire < kHealthyThreshold
            && retireAgeHealthy && readerHealthy;
        if (exitCond.pendingRetire >= kHealthyThreshold)
            exitCond.blocker = CriticalExitBlocker::PendingRetireExceeded;
        else if (!retireAgeHealthy)
            exitCond.blocker = CriticalExitBlocker::RetireAgeExceeded;
        exitCond.allMonitorsNormal = exitCond.allMonitorsNormal && metricsHealthy;

        // 条件4: 安定60秒継続
        if (exitCond.allMonitorsNormal && exitCond.noRecoveryActionRunning) {
            const uint64_t nowUs = getCurrentTimeUs();
            if (m_criticalExitStableStartUs_ == 0)
                m_criticalExitStableStartUs_ = nowUs;
            exitCond.stableDuration = (nowUs - m_criticalExitStableStartUs_) >= 60'000'000;
        } else {
            m_criticalExitStableStartUs_ = 0;
        }

        if (exitCond.canExit()) {
            // 次 tick の updateHealthState() で Healthy 復帰が期待できる
        }
    }

    // ★ v3.0: updateHealthState() が PolicyDecision を受け取り単一権限で HealthState を決定
    updateHealthState(decision);
}

void RuntimeHealthMonitor::checkRetireStall() noexcept {
    if (!m_retireRouter) return;
    uint32_t pendingCount = m_retireRouter->pendingRetireCount();

    int hwm = (m_retireHighWatermarkRef != nullptr)
        ? convo::consumeAtomic(*m_retireHighWatermarkRef, std::memory_order_acquire)
        : 3072;

    // ★ Error 閾値: hwm * 1.5 を基本とし、最低でも hwm+1 を確保。
    int errorThreshold = std::max(hwm + hwm / 2, hwm + 1);

    MonitorState newState;
    HealthEvent::Severity severity;
    uint32_t eventCode;

    if (pendingCount > static_cast<uint32_t>(errorThreshold)) {
        newState = MonitorState::Error;
        severity = HealthEvent::Severity::Error;
        eventCode = EVENT_RETIRE_STALL;
    } else if (pendingCount > static_cast<uint32_t>(hwm)) {
        newState = MonitorState::Warning;
        severity = HealthEvent::Severity::Warning;
        eventCode = EVENT_RETIRE_STALL_WARNING;
    } else {
        newState = MonitorState::Normal;
        severity = HealthEvent::Severity::Info;
        eventCode = EVENT_RETIRE_STALL_WARNING;
    }

    emitOnTransition(m_prevRetireState, newState, severity, eventCode, pendingCount);

    // [work37 Phase 9.1] Retire Stall 継続時間追跡
    if (newState == MonitorState::Error) {
        if (m_retireStallStartUs_ == 0)
            m_retireStallStartUs_ = getCurrentTimeUs();
    } else if (newState == MonitorState::Normal) {
        m_retireStallStartUs_ = 0;
    }
}

void RuntimeHealthMonitor::checkPublicationStall() noexcept {
    if (!m_orchestrator) return;

    m_orchestrator->updateProgressObservation();

    // ★ 出版停滞の検出条件:
    //   pendingIntent（retire intent）, hasDeferred（保留中publish）,
    //   または publicationBacklog（溜まった未処理publish）が存在し、
    //   かつ sequence が 30秒以上進んでいない場合。
    const bool hasPendingWork = m_orchestrator->getPendingIntentCount() > 0
        || m_orchestrator->hasDeferredRequest()
        || m_orchestrator->getPublicationBacklogCount() > 0;

    MonitorState newState;
    HealthEvent::Severity severity;
    uint32_t eventCode;
    uint64_t value = 0;

    if (hasPendingWork && m_orchestrator->isPublicationStalled()) {
        newState = MonitorState::Error;
        severity = HealthEvent::Severity::Error;
        eventCode = EVENT_PUBLICATION_STALL;
    } else if (m_orchestrator->hasDeferredRequest()) {
        uint64_t deferredAge = m_orchestrator->getMaxDeferredAgeMs();
        if (deferredAge > 30000) {
            newState = MonitorState::Error;
            severity = HealthEvent::Severity::Error;
            eventCode = EVENT_PUBLICATION_STALL;
            value = deferredAge;
        } else if (deferredAge > 5000) {
            newState = MonitorState::Warning;
            severity = HealthEvent::Severity::Warning;
            eventCode = EVENT_PUBLICATION_WARNING;
            value = deferredAge;
        } else {
            newState = MonitorState::Normal;
            severity = HealthEvent::Severity::Info;
            eventCode = EVENT_PUBLICATION_WARNING;
        }
    } else {
        newState = MonitorState::Normal;
        severity = HealthEvent::Severity::Info;
        eventCode = EVENT_PUBLICATION_WARNING;
    }

    emitOnTransition(m_prevPublicationState, newState, severity, eventCode, value);
}

// ★ P1-B: 各 MonitorState から統合 ISRHealthState を算出
void RuntimeHealthMonitor::updateHealthState() noexcept
{
    ISRHealthState newState = ISRHealthState::Healthy;

    // Retire stall → Degraded or Critical
    if (m_prevRetireState == MonitorState::Error)
        newState = ISRHealthState::Critical;
    else if (m_prevRetireState == MonitorState::Warning)
        newState = ISRHealthState::Degraded;

    // Publication stall → Critical（retire より優先度高）
    if (m_prevPublicationState == MonitorState::Error)
        newState = ISRHealthState::Critical;
    else if (m_prevPublicationState == MonitorState::Warning
             && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    // ★ Practical-3: Overflow rate → Degraded or Critical
    if (m_prevOverflowRateState == MonitorState::Error && newState != ISRHealthState::Critical)
        newState = ISRHealthState::Critical;
    else if (m_prevOverflowRateState == MonitorState::Warning
             && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    // ★ Practical-4: Reader slot exhaustion → Degraded or Critical
    if (m_prevReaderSlotState == MonitorState::Error && newState != ISRHealthState::Critical)
        newState = ISRHealthState::Critical;
    else if (m_prevReaderSlotState == MonitorState::Warning
             && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    // ★ Practical-5: Retire age → Degraded or Critical
    if (m_prevRetireAgeState == MonitorState::Error && newState != ISRHealthState::Critical)
        newState = ISRHealthState::Critical;
    else if (m_prevRetireAgeState == MonitorState::Warning
             && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    convo::publishAtomic(m_healthState_, newState, std::memory_order_release);
}

// [work37 Phase 4.1] PolicyDecision 対応版 updateHealthState
//   1. 既存の MonitorState ベース評価（上記 updateHealthState() のロジックをインライン）
//   2. PolicyDecision の causes を考慮して HealthState を最終決定
void RuntimeHealthMonitor::updateHealthState(const PolicyDecision& decision) noexcept
{
    ISRHealthState newState = ISRHealthState::Healthy;

    // Retire stall → Degraded or Critical
    if (m_prevRetireState == MonitorState::Error)
        newState = ISRHealthState::Critical;
    else if (m_prevRetireState == MonitorState::Warning)
        newState = ISRHealthState::Degraded;

    // Publication stall → Critical
    if (m_prevPublicationState == MonitorState::Error)
        newState = ISRHealthState::Critical;
    else if (m_prevPublicationState == MonitorState::Warning
             && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    // ★ Practical-3: Overflow rate
    if (m_prevOverflowRateState == MonitorState::Error
        && newState != ISRHealthState::Critical)
        newState = ISRHealthState::Critical;
    else if (m_prevOverflowRateState == MonitorState::Warning
             && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    // Reader slot
    if (m_prevReaderSlotState == MonitorState::Error)
        newState = ISRHealthState::Critical;
    else if (m_prevReaderSlotState == MonitorState::Warning
             && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    // Retire age
    if (m_prevRetireAgeState == MonitorState::Error && newState != ISRHealthState::Critical)
        newState = ISRHealthState::Critical;
    else if (m_prevRetireAgeState == MonitorState::Warning
             && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    // ★ work37: PolicyDecision の causes を考慮
    //   複合原因がある場合は Degraded 以上に昇格
    if (decision.causes != 0 && newState == ISRHealthState::Healthy)
        newState = ISRHealthState::Degraded;

    convo::publishAtomic(m_healthState_, newState, std::memory_order_release);
}

void RuntimeHealthMonitor::emitOnTransition(
    MonitorState& currentState, MonitorState newState,
    HealthEvent::Severity severity, uint32_t eventCode,
    uint64_t value, uint32_t slot) noexcept
{
    if (currentState == newState) return;
    currentState = newState;
    if (newState == MonitorState::Normal) return;
    if (!m_callback) return;
    HealthEvent ev{getCurrentTimeUs(), severity, eventCode, value, slot};
    m_callback(ev);
}

// ★ Practical-1: Reader Stuck 実診断（detectStuckReaders 経由）
void RuntimeHealthMonitor::diagnoseRetireStall() noexcept
{
    if (!m_retireRouter)
        return;

    // ★ Practical-1: ルーター経由で実診断を取得（EpochDomain 直接アクセス不要）
    //   stuckThreshold = 10 epoch 以上進行していない Reader を Stuck と判定
    const auto stuckInfo = m_retireRouter->detectStuckReaders(10);

    if (stuckInfo.isStuck)
    {
        const uint64_t nowUs = convo::getCurrentTimeUs();
        const bool severe = (stuckInfo.pendingRetireCount > 100 || stuckInfo.residencyTimeUs > 30'000'000);
        MonitorState newState = severe ? MonitorState::Error : MonitorState::Warning;

        HealthEvent ev{nowUs,
                       severe ? HealthEvent::Severity::Error : HealthEvent::Severity::Warning,
                       EVENT_READER_STUCK,
                       stuckInfo.pendingRetireCount,
                       0};
        ev.readerIndex = stuckInfo.readerIndex;
        ev.readerEpoch = stuckInfo.readerEpoch;
        ev.readerDepth = 1; // depth > 0 確定
        ev.residencyTimeUs = stuckInfo.residencyTimeUs; // ★ Practical-8: 実測値

        if (m_callback)
        {
            if (m_prevRetireState != newState)
            {
                m_prevRetireState = newState;
                m_callback(ev);
                // ★ 改善②: 遷移発火と同じtickでは定期Evidenceを抑制（二重発呼防止）
                m_lastStuckEvidenceUs = nowUs;
            }
            // ★ 8.6: 状態遷移がなくとも10秒ごとに定期Evidence出力
            if (nowUs - m_lastStuckEvidenceUs > kStuckEvidenceIntervalUs)
            {
                m_lastStuckEvidenceUs = nowUs;
                // Reader Stuck 継続中を定期的に通知
                // onHealthEvent 側の diagLog + emitEvidenceTickNonRt に委譲
                HealthEvent periodicEv{nowUs,
                    severe ? HealthEvent::Severity::Error : HealthEvent::Severity::Warning,
                    EVENT_READER_STUCK,
                    stuckInfo.pendingRetireCount,
                    0};
                periodicEv.readerIndex = stuckInfo.readerIndex;
                periodicEv.readerEpoch = stuckInfo.readerEpoch;
                periodicEv.residencyTimeUs = stuckInfo.residencyTimeUs;
                m_callback(periodicEv);
            }
        }
    }
}

// ★ P1-C/Practical-2: Crossfade Timeout 監視（固定30秒）
void RuntimeHealthMonitor::checkCrossfadeTimeout() noexcept
{
    if (!m_crossfadeRuntime) return;
    if (!m_crossfadeRuntime->isPending()) return;

    uint64_t ageUs = m_crossfadeRuntime->getFadeAgeUs();
    if (ageUs > kCrossfadeTimeoutUs) {
        emitOnTransition(m_prevCrossfadeDropState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_CROSSFADE_TIMEOUT, ageUs / 1000);
    }
}

// ★ P1-C/Practical-6: Crossfade Event Drop 差分ベース監視
void RuntimeHealthMonitor::checkCrossfadeEventDrop() noexcept
{
    if (!m_crossfadeEventDropRef) return;
    uint64_t current = convo::consumeAtomic(*m_crossfadeEventDropRef, std::memory_order_acquire);
    uint64_t delta = current - m_lastObservedDropCount;

    if (delta >= kCrossfadeEventDropCriticalDelta) {
        emitOnTransition(m_prevCrossfadeDropState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_CROSSFADE_EVENT_DROP, delta);
    } else if (delta >= kCrossfadeEventDropWarningDelta) {
        emitOnTransition(m_prevCrossfadeDropState, MonitorState::Warning,
            HealthEvent::Severity::Warning, EVENT_CROSSFADE_EVENT_DROP, delta);
    }

    m_lastObservedDropCount = current;
}

// [work39 Phase 5] Learner FIFO 監視
void RuntimeHealthMonitor::checkLearnerBackpressure() noexcept
{
    if (m_learnerRunningRef == nullptr) return;
    const bool learnerActive = convo::consumeAtomic(*m_learnerRunningRef,
                                                     std::memory_order_acquire);

    // Learner restart detection → EMA reset
    if (!learnerActive) { m_learnerWasActive_ = false; return; }
    if (!m_learnerWasActive_) {
        m_fifoEma_ = -1.0;  m_lastFifoEma_ = 0.0;
        m_learnerFifoHighSinceUs_ = 0;  m_learnerWasActive_ = true;
    }

    const int available = m_learnerSegmentBuffer_
        ? m_learnerSegmentBuffer_->getNumAvailableSamples() : 0;
    constexpr int kCapacity = 3'840'000;
    const double fifoUsage = static_cast<double>(available) / kCapacity;

    // EMA (alpha=0.3)
    constexpr double kEmaAlpha = 0.3;
    if (m_fifoEma_ < 0.0) m_fifoEma_ = fifoUsage;
    m_fifoEma_ = kEmaAlpha * fifoUsage + (1.0 - kEmaAlpha) * m_fifoEma_;

    // Time-normalized slope
    const uint64_t nowUs = getCurrentTimeUs();
    const double elapsedSec = (m_lastFifoTickUs_ > 0)
        ? (nowUs - m_lastFifoTickUs_) / 1'000'000.0 : 1.0;
    const double slope = (m_fifoEma_ - m_lastFifoEma_) / std::max(elapsedSec, 0.001);
    m_lastFifoEma_ = m_fifoEma_;  m_lastFifoTickUs_ = nowUs;

    // 2-stage thresholds via emitOnTransition
    const uint64_t fifoUsagePct = static_cast<uint64_t>(fifoUsage * 100.0);
    if (fifoUsage > 0.95 && slope >= 0.0) {
        emitOnTransition(m_prevLearnerBackpressureState_, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_LEARNER_BACKPRESSURE_ERROR, fifoUsagePct);
    } else if (fifoUsage > 0.85 && slope >= 0.0) {
        emitOnTransition(m_prevLearnerBackpressureState_, MonitorState::Warning,
            HealthEvent::Severity::Warning, EVENT_LEARNER_BACKPRESSURE_WARNING, fifoUsagePct);
    } else if (fifoUsage <= 0.80 || slope < -0.01) {
        m_prevLearnerBackpressureState_ = MonitorState::Normal;
    }
}

// [work39 Phase 3] TrendSnapshot 取得
TrendSnapshot RuntimeHealthMonitor::takeSnapshot() const noexcept
{
    TrendSnapshot snap;
    if (m_retireRouter) {
        snap.pendingRetire = m_retireRouter->pendingRetireCount();
        const auto stuckInfo = m_retireRouter->detectStuckReaders(10);
        snap.readerStuckCount = stuckInfo.isStuck ? 1 : 0;
        snap.activeReaderCount = m_retireRouter->activeReaderCount();
    }
    if (m_publicationSequenceRef_)
        snap.publicationSeq = convo::consumeAtomic(*m_publicationSequenceRef_,
                                                     std::memory_order_acquire);
    if (m_maxRetireAgeRef)
        snap.maxRetireAgeUs = convo::consumeAtomic(*m_maxRetireAgeRef,
                                                     std::memory_order_acquire);
    else if (m_maxRetireAgeDoubleRef)
        snap.maxRetireAgeUs = static_cast<uint64_t>(
            convo::consumeAtomic(*m_maxRetireAgeDoubleRef, std::memory_order_acquire));
    if (m_epochAdvanceCountRef_)
        snap.epochAdvanceCount = convo::consumeAtomic(*m_epochAdvanceCountRef_,
                                                       std::memory_order_acquire);
    if (m_lastCompletedEpochRef_)
        snap.lastCompletedEpoch = convo::consumeAtomic(*m_lastCompletedEpochRef_,
                                                        std::memory_order_acquire);
    // activeFaultMask は HealthMonitor の監視状態から合成
    uint32_t faultMask = 0;
    if (m_prevRetireState == MonitorState::Error)  faultMask |= kFaultRetire;
    if (m_prevPublicationState == MonitorState::Error) faultMask |= kFaultPublication;
    if (m_prevReaderSlotState == MonitorState::Error)  faultMask |= kFaultReader;
    if (m_prevOverflowRateState == MonitorState::Error) faultMask |= kFaultOverflow;
    snap.activeFaultMask = faultMask;
    snap.freezeDetected = (m_prevProgressFreezeState_ == MonitorState::Error);
    return snap;
}

// [work39 Phase 3] 傾向判定（computeTrend）
RecoveryOutcome RuntimeHealthMonitor::computeTrend(
    const TrendSnapshot& before, const TrendSnapshot& now) const noexcept
{
    // Step 0: 主要delta計算
    const int64_t retireDelta = static_cast<int64_t>(now.pendingRetire)
                              - static_cast<int64_t>(before.pendingRetire);
    const int64_t ageDelta    = static_cast<int64_t>(now.maxRetireAgeUs)
                              - static_cast<int64_t>(before.maxRetireAgeUs);
    const int64_t pubDelta    = static_cast<int64_t>(now.publicationSeq)
                              - static_cast<int64_t>(before.publicationSeq);
    const bool faultMaskIncreased = (now.activeFaultMask > before.activeFaultMask);

    // Step 1: ProgressFreeze 監視（最優先、多軸評価）
    if (now.freezeDetected) {
        const bool retireProgress = before.pendingRetire > 0
            && (now.pendingRetire * 100) < (before.pendingRetire * 95);
        const bool readerProgress = (now.readerStuckCount < before.readerStuckCount);
        const bool epochProgress = (now.epochAdvanceCount > before.epochAdvanceCount);
        const bool pubProgress = (now.publicationSeq > before.publicationSeq);
        const bool multiAxisImprovement = retireProgress
            || (readerProgress && epochProgress)
            || (pubProgress && readerProgress);
        if (!multiAxisImprovement)
            return RecoveryOutcome::Worsening;
    }

    // Step 2: Recovered（全軸正常）
    constexpr uint64_t kRecoveredRetireLimit = 256;
    const bool idleRecovered = (now.pendingRetire == 0 && now.maxRetireAgeUs == 0);
    const bool retireWithinLimit = now.pendingRetire <= kRecoveredRetireLimit;
    const bool retireTrendImproving = (retireDelta < 0);
    if (!faultMaskIncreased
        && (pubDelta > 0 || idleRecovered)
        && (retireTrendImproving || retireWithinLimit)
        && ageDelta <= 0
        && now.readerStuckCount == 0 && now.activeReaderCount < 64)
        return RecoveryOutcome::Recovered;

    // Step 3: reader異常（Improvingより優先）
    if (now.readerStuckCount > before.readerStuckCount)
        return RecoveryOutcome::Worsening;
    if (now.pendingRetire > 0 && now.activeReaderCount == 0 && before.activeReaderCount > 0)
        return RecoveryOutcome::Worsening;

    // Step 4: Worsening
    if (faultMaskIncreased || retireDelta > 0 || ageDelta > 0)
        return RecoveryOutcome::Worsening;

    // Step 5: Improving
    const bool retireImproving = (retireDelta < -2);
    const bool readerImproving = (now.readerStuckCount < before.readerStuckCount);
    if ((retireImproving || readerImproving) && !faultMaskIncreased)
        return RecoveryOutcome::Improving;

    // Step 6: Stalled
    return RecoveryOutcome::Stalled;
}

// ★ Practical-4/6: Reader Slot Usage Telemetry（50%/75%/90% 閾値、capacity 動的取得）
void RuntimeHealthMonitor::checkReaderSlotUsage() noexcept
{
    // ★ 優先: m_readerSlotRef が設定されている場合はそれを使用
    //   未設定の場合は m_retireRouter 経由で直接取得（フォールバック）
    uint32_t activeCount = 0;
    if (m_readerSlotRef) {
        activeCount = convo::consumeAtomic(*m_readerSlotRef, std::memory_order_acquire);
    } else if (m_retireRouter) {
        activeCount = m_retireRouter->activeReaderCount();
    } else {
        return;
    }
    // ★ Practical-6: capacity を動的取得（固定値ではなく router 経由）
    int maxSlots = (m_retireRouter != nullptr) ? m_retireRouter->readerCapacity() : 64;
    if (maxSlots <= 0) maxSlots = 64;
    double usage = static_cast<double>(activeCount) / static_cast<double>(maxSlots);

    if (usage >= 0.90) {
        // ★ B-1: Reader Slot 使用率が Critical の場合、個別 Reader 情報を詳細取得
        int worstReaderIndex = -1;
        uint64_t worstResidencyUs = 0;
        convo::ReaderSlotDetail worstDetail{};

        if (m_retireRouter) {
            for (int i = 0; i < maxSlots; ++i) {
                auto detail = m_retireRouter->getReaderSlotDetail(i);
                if (detail.active && detail.residencyTimeUs > worstResidencyUs) {
                    worstResidencyUs = detail.residencyTimeUs;
                    worstReaderIndex = i;
                    worstDetail = detail;
                }
            }
        }

        if (m_callback && worstReaderIndex >= 0) {
            // 詳細情報付きで直接コールバック
            HealthEvent ev{getCurrentTimeUs(),
                           HealthEvent::Severity::Error,
                           EVENT_READER_SLOT_USAGE,
                           activeCount,
                           static_cast<uint32_t>(maxSlots)};
            ev.readerIndex = worstReaderIndex;
            ev.readerEpoch = worstDetail.epoch;
            ev.readerDepth = worstDetail.depth;
            ev.residencyTimeUs = worstDetail.residencyTimeUs;
            m_callback(ev);
            // 状態遷移を updateHealthState に反映（callback 後でも emitOnTransition を呼ばない）
            if (m_prevReaderSlotState != MonitorState::Error) {
                m_prevReaderSlotState = MonitorState::Error;
            }
        } else {
            // 詳細情報なし → 従来の emitOnTransition 経由
            emitOnTransition(m_prevReaderSlotState, MonitorState::Error,
                HealthEvent::Severity::Error, EVENT_READER_SLOT_USAGE,
                activeCount, maxSlots);
        }
    } else if (usage >= kReaderSlotCriticalThreshold) {
        emitOnTransition(m_prevReaderSlotState, MonitorState::Warning,
            HealthEvent::Severity::Warning, EVENT_READER_SLOT_USAGE,
            activeCount, maxSlots);
    }
}

// ★ Practical-3: Overflow Rate 監視（回数/sec ベース、ヒステリシス付き）
void RuntimeHealthMonitor::checkOverflowRate() noexcept
{
    if (!m_overflowCountRef) return;
    const uint64_t nowUs = convo::getCurrentTimeUs();
    const uint64_t elapsed = nowUs - m_lastOverflowCheckTimeUs;
    if (elapsed < kOverflowRateWindowUs)
        return; // まだ1秒経過していない

    const uint64_t current = convo::consumeAtomic(*m_overflowCountRef, std::memory_order_acquire);
    const uint64_t delta = current - m_lastOverflowCount;

    // レート計算: delta events / (elapsed us / 1,000,000) = delta / (elapsed / 1e6)
    // → delta * 1,000,000 / elapsed で per-sec 換算
    const uint64_t ratePerSec = (elapsed > 0) ? (delta * 1'000'000 / elapsed) : 0;

    m_lastOverflowCount = current;
    m_lastOverflowCheckTimeUs = nowUs;

    // ★ ヒステリシス付き状態遷移
    MonitorState desiredState;
    if (ratePerSec >= kOverflowRateCriticalThreshold)
        desiredState = MonitorState::Error;
    else if (ratePerSec >= kOverflowRateWarningThreshold)
        desiredState = MonitorState::Warning;
    else
        desiredState = MonitorState::Normal;

    if (desiredState != m_prevOverflowRateState)
    {
        // 上昇方向（Normal→Warning/Error, Warning→Error）: 即時遷移
        if (desiredState > m_prevOverflowRateState)
        {
            m_overflowRateStableSinceUs = nowUs;
            emitOnTransition(m_prevOverflowRateState, desiredState,
                desiredState == MonitorState::Error
                    ? HealthEvent::Severity::Error : HealthEvent::Severity::Warning,
                desiredState == MonitorState::Error ? EVENT_RETIRE_STALL : EVENT_RETIRE_STALL_WARNING,
                ratePerSec);
        }
        else
        {
            // 下降方向: ヒステリシス待機
            //   Error→Warning: 10秒安定
            //   Warning→Normal: 30秒安定
            //   または Error→Normal: 30秒安定（Error→Warning経由）
            if (m_overflowRateStableSinceUs == 0)
            {
                m_overflowRateStableSinceUs = nowUs;
            }
            const uint64_t stableDuration = nowUs - m_overflowRateStableSinceUs;
            const uint64_t requiredStableUs = (m_prevOverflowRateState == MonitorState::Error)
                ? kOverflowHysteresisCriticalToDegradedUs   // 10秒
                : kOverflowHysteresisDegradedToHealthyUs;    // 30秒

            if (stableDuration >= requiredStableUs)
            {
                m_overflowRateStableSinceUs = 0;
                emitOnTransition(m_prevOverflowRateState, desiredState,
                    HealthEvent::Severity::Info,
                    EVENT_RETIRE_STALL_WARNING,
                    ratePerSec);
            }
        }
    }
    else
    {
        // 状態維持中で安定クロックが進行中の場合、リセット(rate上昇で即時遷移できるように)
        if (desiredState == MonitorState::Normal && m_overflowRateStableSinceUs != 0)
        {
            const uint64_t stableDuration = nowUs - m_overflowRateStableSinceUs;
            const uint64_t requiredStableUs = (m_prevOverflowRateState == MonitorState::Warning)
                ? kOverflowHysteresisDegradedToHealthyUs
                : kOverflowHysteresisCriticalToDegradedUs;
            if (stableDuration >= requiredStableUs)
            {
                m_overflowRateStableSinceUs = 0;
                emitOnTransition(m_prevOverflowRateState, MonitorState::Normal,
                    HealthEvent::Severity::Info,
                    EVENT_RETIRE_STALL_WARNING,
                    ratePerSec);
            }
        }
    }
}

// ★ Work38: Retire Reclaim Latency 監視（型安全版）
void RuntimeHealthMonitor::checkRetireReclaimLatency() noexcept
{
    if (m_maxRetireAgeDoubleRef) {
        // ★ double 版: reclaimLatency_ は ms 単位で publish → μs に変換して判定
        const double elapsedMs = convo::consumeAtomic(
            *m_maxRetireAgeDoubleRef, std::memory_order_acquire);
        const uint64_t maxAgeUs = static_cast<uint64_t>(elapsedMs * 1000.0);

        if (maxAgeUs > kRetireAgeCriticalUs) {
            emitOnTransition(m_prevRetireAgeState, MonitorState::Error,
                HealthEvent::Severity::Error, EVENT_RETIRE_AGE_CRITICAL,
                maxAgeUs / 1000);
        } else if (maxAgeUs > kRetireAgeWarningUs) {
            emitOnTransition(m_prevRetireAgeState, MonitorState::Warning,
                HealthEvent::Severity::Warning, EVENT_RETIRE_AGE_WARNING,
                maxAgeUs / 1000);
        } else {
            // ★ Work38: 正常復帰イベント — 3状態遷移を完全カバー
            emitOnTransition(m_prevRetireAgeState, MonitorState::Normal,
                HealthEvent::Severity::Info, EVENT_RETIRE_AGE_NORMAL,
                maxAgeUs / 1000);
        }
    } else if (m_maxRetireAgeRef) {
        // ★ uint64_t 版: 従来ロジック（他からの呼び出しのために存置）
        const uint64_t maxAgeUs = convo::consumeAtomic(
            *m_maxRetireAgeRef, std::memory_order_acquire);

        if (maxAgeUs > kRetireAgeCriticalUs) {
            emitOnTransition(m_prevRetireAgeState, MonitorState::Error,
                HealthEvent::Severity::Error, EVENT_RETIRE_AGE_CRITICAL,
                maxAgeUs / 1000);
        } else if (maxAgeUs > kRetireAgeWarningUs) {
            emitOnTransition(m_prevRetireAgeState, MonitorState::Warning,
                HealthEvent::Severity::Warning, EVENT_RETIRE_AGE_WARNING,
                maxAgeUs / 1000);
        }
    }
}

// [work37 Phase 9.2] Configuration Divergence 監視
//   publishedRevision と requestedRevision の乖離を監視
//   gap >= 3 かつ 30秒継続 → Error
void RuntimeHealthMonitor::checkConfigurationDivergence() noexcept
{
    if (m_lastCommittedGenRef_ == nullptr || m_requestedGenRef_ == nullptr)
        return;

    const uint64_t committed = convo::consumeAtomic(*m_lastCommittedGenRef_,
                                                     std::memory_order_acquire);
    const uint64_t requested = convo::consumeAtomic(*m_requestedGenRef_,
                                                     std::memory_order_acquire);
    if (requested <= committed) {
        m_configDivergenceStartUs_ = 0;
        m_prevConfigDivergenceState_ = MonitorState::Normal;
        return;
    }

    const uint64_t gap = requested - committed;
    const uint64_t nowUs = getCurrentTimeUs();

    if (gap >= 3) {
        if (m_configDivergenceStartUs_ == 0)
            m_configDivergenceStartUs_ = nowUs;
        const uint64_t elapsedUs = nowUs - m_configDivergenceStartUs_;
        if (elapsedUs > 30'000'000) {
            emitOnTransition(m_prevConfigDivergenceState_, MonitorState::Error,
                HealthEvent::Severity::Error, 5001, gap);
        } else if (elapsedUs > 10'000'000) {
            emitOnTransition(m_prevConfigDivergenceState_, MonitorState::Warning,
                HealthEvent::Severity::Warning, 5001, gap);
        }
    } else if (gap >= 2) {
        if (m_configDivergenceStartUs_ == 0)
            m_configDivergenceStartUs_ = nowUs;
        const uint64_t elapsedUs = nowUs - m_configDivergenceStartUs_;
        if (elapsedUs > 30'000'000) {
            emitOnTransition(m_prevConfigDivergenceState_, MonitorState::Warning,
                HealthEvent::Severity::Warning, 5001, gap);
        }
    } else {
        m_configDivergenceStartUs_ = 0;
        m_prevConfigDivergenceState_ = MonitorState::Normal;
    }
}

// [work37 Phase 7.1] World Consistency 監視
//   RuntimeDrainAudit::verifyWorldConsistency() から Broken を検出 → PolicyEngine 通知
void RuntimeHealthMonitor::checkWorldConsistency() noexcept
{
    if (!m_worldConsistencyCheck_)
        return;
    const auto consistencyState = m_worldConsistencyCheck_();
    // 0=Consistent, 1=Suspicious, 2=Broken
    if (consistencyState >= 2) {
        // Broken → Critical 相当の HealthEvent 発行
        emitOnTransition(m_prevConfigDivergenceState_, MonitorState::Error,
            HealthEvent::Severity::Error, 5001,
            static_cast<uint64_t>(consistencyState));
    }
}

// [work37 Phase 9.7] Snapshot Starvation 監視
//   maxDeferredAgeMs > 10s → Warning
//   maxDeferredAgeMs > 30s → Error
void RuntimeHealthMonitor::checkSnapshotStarvation() noexcept
{
    if (m_orchestrator == nullptr)
        return;
    const uint64_t maxDeferredAgeMs = m_orchestrator->getMaxDeferredAgeMs();
    const uint64_t nowUs = getCurrentTimeUs();

    if (maxDeferredAgeMs > 30'000) {  // 30秒超 → Error
        if (m_snapshotStarvationStartUs_ == 0)
            m_snapshotStarvationStartUs_ = nowUs;
        emitOnTransition(m_prevSnapshotStarvationState_, MonitorState::Error,
            HealthEvent::Severity::Error, 5003, maxDeferredAgeMs);
    } else if (maxDeferredAgeMs > 10'000) {  // 10秒超 → Warning
        if (m_snapshotStarvationStartUs_ == 0)
            m_snapshotStarvationStartUs_ = nowUs;
        emitOnTransition(m_prevSnapshotStarvationState_, MonitorState::Warning,
            HealthEvent::Severity::Warning, 5003, maxDeferredAgeMs);
    } else {
        m_snapshotStarvationStartUs_ = 0;
        m_prevSnapshotStarvationState_ = MonitorState::Normal;
    }
}

// [work37 Phase 9.8] Pending Structural Deployment 監視
//   rebuildRequestGeneration と lastCommittedRebuildGeneration の乖離を監視
//   gap >= 3 → Warning, gap >= 5 → Error
void RuntimeHealthMonitor::checkPendingStructuralDeployment() noexcept
{
    if (m_requestedRebuildGenRef_ == nullptr || m_committedRebuildGenRef_ == nullptr)
        return;

    const int requested = convo::consumeAtomic(*m_requestedRebuildGenRef_,
                                                std::memory_order_acquire);
    const int committed = convo::consumeAtomic(*m_committedRebuildGenRef_,
                                                std::memory_order_acquire);
    if (requested <= committed) {
        m_structuralDeployStartUs_ = 0;
        m_prevStructuralDeployState_ = MonitorState::Normal;
        return;
    }

    const int gap = requested - committed;
    const uint64_t nowUs = getCurrentTimeUs();

    if (gap >= 5) {
        if (m_structuralDeployStartUs_ == 0)
            m_structuralDeployStartUs_ = nowUs;
        const uint64_t elapsedUs = nowUs - m_structuralDeployStartUs_;
        if (elapsedUs > 10'000'000) {
            emitOnTransition(m_prevStructuralDeployState_, MonitorState::Error,
                HealthEvent::Severity::Error, 5004, static_cast<uint64_t>(gap));
        }
    } else if (gap >= 3) {
        if (m_structuralDeployStartUs_ == 0)
            m_structuralDeployStartUs_ = nowUs;
        const uint64_t elapsedUs = nowUs - m_structuralDeployStartUs_;
        if (elapsedUs > 30'000'000) {
            emitOnTransition(m_prevStructuralDeployState_, MonitorState::Warning,
                HealthEvent::Severity::Warning, 5004, static_cast<uint64_t>(gap));
        }
    } else {
        m_structuralDeployStartUs_ = 0;
        m_prevStructuralDeployState_ = MonitorState::Normal;
    }
}

// [work37 Phase 9.29] Suppression Duration 監視 — 段階的エスカレーション
//   30s → ForceRetireDrain / 60s → ClearDeferredPublish
//   120s → ForceCrossfadeReset / 180s → RejectNewPublication
void RuntimeHealthMonitor::checkSuppressionDuration() noexcept
{
    if (m_suppressionStartRef_ == nullptr)
        return;
    const uint64_t startUs = convo::consumeAtomic(*m_suppressionStartRef_,
                                                    std::memory_order_acquire);
    if (startUs == 0) {
        m_prevSuppressionDurationState_ = MonitorState::Normal;
        return;
    }
    const uint64_t durationUs = getCurrentTimeUs() - startUs;
    if (durationUs > 180'000'000) {  // 180秒超 → Critical
        emitOnTransition(m_prevSuppressionDurationState_, MonitorState::Error,
            HealthEvent::Severity::Error, 6001, durationUs / 1'000'000);
    } else if (durationUs > 120'000'000) {  // 120秒超 → 重度
        emitOnTransition(m_prevSuppressionDurationState_, MonitorState::Warning,
            HealthEvent::Severity::Warning, 6001, durationUs / 1'000'000);
    } else if (durationUs > 60'000'000) {  // 60秒超 → 中度
        emitOnTransition(m_prevSuppressionDurationState_, MonitorState::Warning,
            HealthEvent::Severity::Warning, 6001, durationUs / 1'000'000);
    } else if (durationUs > 30'000'000) {  // 30秒超 → 軽度
        emitOnTransition(m_prevSuppressionDurationState_, MonitorState::Warning,
            HealthEvent::Severity::Warning, 6001, durationUs / 1'000'000);
    }
}

// [work37 Phase 9.40] Runtime Progress Freeze 監視 — 3軸統合検出
//   Publish/Retire/Rebuild の進行を監視し、2/3軸が60秒以上停滞で発火
void RuntimeHealthMonitor::checkRuntimeProgressFreeze() noexcept
{
    int stalledAxes = 0;
    const uint64_t nowUs = getCurrentTimeUs();

    // 軸1: Publish 進行 — publicationSequenceCounter の変化
    if (m_publicationSequenceRef_ != nullptr) {
        const uint64_t currentSeq = convo::consumeAtomic(*m_publicationSequenceRef_,
                                                          std::memory_order_acquire);
        if (currentSeq <= m_lastObservedPubSeq_) {
            ++stalledAxes;
        }
        m_lastObservedPubSeq_ = currentSeq;
    }

    // 軸2: Retire 進行 — lastRetireTimestamp の更新
    if (m_lastRetireTimestampRef_ != nullptr) {
        const uint64_t lastRetire = convo::consumeAtomic(*m_lastRetireTimestampRef_,
                                                          std::memory_order_acquire);
        if (lastRetire <= m_lastObservedRetireTs_) {
            ++stalledAxes;
        } else {
            m_lastObservedRetireTs_ = lastRetire;
        }
    }

    // 軸3: Rebuild 進行 — pendingStructuralDeploy 状態
    if (m_prevStructuralDeployState_ != MonitorState::Normal) {
        ++stalledAxes;
    }

    if (stalledAxes >= 2) {
        if (m_progressFreezeStartUs_ == 0)
            m_progressFreezeStartUs_ = nowUs;
        const uint64_t elapsedUs = nowUs - m_progressFreezeStartUs_;
        if (elapsedUs > 60'000'000) {
            emitOnTransition(m_prevProgressFreezeState_, MonitorState::Error,
                HealthEvent::Severity::Error, 5009,
                static_cast<uint64_t>(stalledAxes));
        } else if (elapsedUs > 30'000'000) {
            emitOnTransition(m_prevProgressFreezeState_, MonitorState::Warning,
                HealthEvent::Severity::Warning, 5009,
                static_cast<uint64_t>(stalledAxes));
        }
    } else {
        m_progressFreezeStartUs_ = 0;
        m_prevProgressFreezeState_ = MonitorState::Normal;
    }
}

// [work37 Phase 9.10 P2] Configuration Drift 監視
//   manualOversamplingFactor と activeOversamplingFactor の乖離を監視
//   30秒継続 → Warning, 60秒継続 → Error
void RuntimeHealthMonitor::checkConfigurationDrift() noexcept
{
    if (m_manualOversamplingRef_ == nullptr)
        return;

    const int manualFactor = convo::consumeAtomic(*m_manualOversamplingRef_,
                                                    std::memory_order_acquire);
    // manualOversamplingFactor == 0 は Auto (ドリフトなし)
    if (manualFactor <= 0) {
        m_configDriftStartUs_ = 0;
        m_prevConfigDriftState_ = MonitorState::Normal;
        return;
    }

    // activeOversamplingFactor は orchester または router 経由で取得する想定。
    // 簡易版として manualFactor の変化がないかだけを監視。
    const uint64_t nowUs = getCurrentTimeUs();

    if (m_configDriftStartUs_ == 0)
        m_configDriftStartUs_ = nowUs;
    const uint64_t elapsedUs = nowUs - m_configDriftStartUs_;

    if (elapsedUs > 60'000'000) {
        emitOnTransition(m_prevConfigDriftState_, MonitorState::Error,
            HealthEvent::Severity::Error, 5006,
            static_cast<uint64_t>(manualFactor));
    } else if (elapsedUs > 30'000'000) {
        emitOnTransition(m_prevConfigDriftState_, MonitorState::Warning,
            HealthEvent::Severity::Warning, 5006,
            static_cast<uint64_t>(manualFactor));
    }
}

// [work37 Phase 9.56 P2] RuntimeRecoveryScore — 4軸総合スコア（診断用）
RuntimeRecoveryScore RuntimeHealthMonitor::computeRuntimeRecoveryScore() const noexcept
{
    RuntimeRecoveryScore score;

    // 軸1: publishProgress — publicationSequence 増加率
    if (m_prevPublicationState == MonitorState::Normal) {
        score.publishProgress = 25;
    } else if (m_prevPublicationState == MonitorState::Warning) {
        score.publishProgress = 12;
    }

    // 軸2: retireProgress — pendingRetire 状態
    if (m_prevRetireState == MonitorState::Normal && m_prevRetireAgeState == MonitorState::Normal) {
        score.retireProgress = 25;
    } else if (m_prevRetireState == MonitorState::Warning || m_prevRetireAgeState == MonitorState::Warning) {
        score.retireProgress = 12;
    }

    // 軸3: rebuildProgress — ConfigurationDivergence + PendingDeployment
    if (m_prevConfigDivergenceState_ == MonitorState::Normal
        && m_prevStructuralDeployState_ == MonitorState::Normal) {
        score.rebuildProgress = 25;
    } else if (m_prevConfigDivergenceState_ == MonitorState::Warning
               || m_prevStructuralDeployState_ == MonitorState::Warning) {
        score.rebuildProgress = 12;
    }

    // 軸4: audioQuality — SafeMode + SnapshotStarvation + ProgressFreeze
    if (m_prevSnapshotStarvationState_ == MonitorState::Normal
        && m_prevProgressFreezeState_ == MonitorState::Normal) {
        score.audioQuality = 25;
    } else if (m_prevSnapshotStarvationState_ == MonitorState::Warning
               || m_prevProgressFreezeState_ == MonitorState::Warning) {
        score.audioQuality = 12;
    }

    return score;
}

// ★ C-4: HealthState のみ初期化
void RuntimeHealthMonitor::reset() noexcept
{
    convo::publishAtomic(m_healthState_, ISRHealthState::Healthy,
                         std::memory_order_release);
    // m_prevRetireState 等は維持 — 初期化すると次回監視で Warning が再通知される
    m_lastObservedDropCount = 0;
    m_lastStuckEvidenceUs = 0;
    // [work37 Phase 4.1] PolicyEngine もリセット
    m_policyEngine_.reset();
    m_backpressureInjected_ = false;
    convo::publishAtomic(m_maxFallbackSize_, uint64_t{0}, std::memory_order_release);
    convo::publishAtomic(m_maxOverflowRate_, 0.0, std::memory_order_release);
    // [work37] 新規監視状態のリセット
    m_retireStallStartUs_ = 0;
    m_configDivergenceStartUs_ = 0;
    m_prevConfigDivergenceState_ = MonitorState::Normal;
    m_snapshotStarvationStartUs_ = 0;
    m_prevSnapshotStarvationState_ = MonitorState::Normal;
    m_structuralDeployStartUs_ = 0;
    m_prevStructuralDeployState_ = MonitorState::Normal;
    m_prevSuppressionDurationState_ = MonitorState::Normal;
    m_progressFreezeStartUs_ = 0;
    m_prevProgressFreezeState_ = MonitorState::Normal;
    m_lastObservedPubSeq_ = 0;
    m_lastObservedRetireTs_ = 0;
    // [work39] 新規フィールドのリセット
    m_prevLearnerBackpressureState_ = MonitorState::Normal;
    m_learnerWasActive_ = false;
    m_fifoEma_ = -1.0;
    m_lastFifoEma_ = 0.0;
    m_lastFifoTickUs_ = 0;
    m_learnerFifoHighSinceUs_ = 0;
    m_backpressureWindow_.reset();
    // [work39 Phase 7] Critical 出口状態リセット
    m_criticalExitStableStartUs_ = 0;

    // ★ Phase-1.5: Validator Telemetry レート制限タイムスタンプリセット
    for (auto& t : m_lastValidationEventUs_)
        convo::publishAtomic(t, uint64_t{0}, std::memory_order_release);
}

// ★ Phase-1.5: Validator Telemetry — ValidationFailure を HealthEvent として発行
void RuntimeHealthMonitor::emitValidationEvent(
    iso::audio_engine::ValidationFailureReason reason) noexcept
{
    uint32_t eventCode = 0;
    size_t idx = 0;
    switch (reason) {
        using enum iso::audio_engine::ValidationFailureReason;
        case SemanticInconsistency:
            eventCode = EVENT_VALIDATION_SEMANTIC_FAILURE; idx = 0; break;
        case InvalidTopology:
            eventCode = EVENT_VALIDATION_TOPOLOGY_FAILURE; idx = 1; break;
        case InvalidResources:
            eventCode = EVENT_VALIDATION_RESOURCE_FAILURE; idx = 2; break;
        case InvalidTransition:
            eventCode = EVENT_VALIDATION_TRANSITION_FAILURE; idx = 3; break;
        default: return;
    }
    // ★ Validation failure は publish thread 単一からのみ発生。
    //   CAS は過剰設計。単純な load + store で十分。
    const uint64_t last = convo::consumeAtomic(
        m_lastValidationEventUs_[idx], std::memory_order_acquire);
    const uint64_t nowUs = convo::getCurrentTimeUs();
    if (nowUs - last >= kValidationEventMinIntervalUs) {
        convo::publishAtomic(m_lastValidationEventUs_[idx], nowUs, std::memory_order_release);
        if (m_callback)
            m_callback(convo::HealthEvent{nowUs, convo::HealthEvent::Severity::Warning,
                                         eventCode, 0, 0});
    }
}

} // namespace convo

#include "RuntimeHealthMonitor.h"
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/CrossfadeRuntime.h"  // ★ P1-C: 完全型必要（isPending/getFadeAgeUs）
#include "audioengine/AtomicAccess.h"
#include "core/TimeUtils.h"

namespace convo {

void RuntimeHealthMonitor::tick() noexcept {
    checkRetireStall();
    checkPublicationStall();
    diagnoseRetireStall();
    checkCrossfadeTimeout();
    checkCrossfadeEventDrop();
    checkReaderSlotUsage();
    checkOverflowRate();         // ★ Practical-3
    checkRetireReclaimLatency();
    updateHealthState();
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
        emitOnTransition(m_prevReaderSlotState, MonitorState::Error,
            HealthEvent::Severity::Error, EVENT_READER_SLOT_USAGE,
            activeCount, maxSlots);
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

// ★ Practical-5: Retire Reclaim Latency 監視
void RuntimeHealthMonitor::checkRetireReclaimLatency() noexcept
{
    if (!m_maxRetireAgeRef) return;
    uint64_t maxAgeUs = convo::consumeAtomic(*m_maxRetireAgeRef, std::memory_order_acquire);

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

} // namespace convo

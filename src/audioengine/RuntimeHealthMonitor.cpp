#include "RuntimeHealthMonitor.h"
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/AtomicAccess.h"
#include "core/TimeUtils.h"

namespace convo {

void RuntimeHealthMonitor::tick() noexcept {
    checkRetireStall();
    checkPublicationStall();
    diagnoseRetireStall(); // ★ P4.5
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

// ★ P4.5: Reader Stuck 診断
// router の情報から Reader 残留の可能性を診断し、HealthEvent として報告する。
// 完全な detectStuckReaders には EpochDomain 直接アクセスが必要。
void RuntimeHealthMonitor::diagnoseRetireStall() noexcept
{
    if (!m_retireRouter)
        return;

    const uint64_t pendingCount = m_retireRouter->pendingRetireCount();
    const uint64_t minEpoch = m_retireRouter->getMinReaderEpoch();
    const uint64_t curEpoch = m_retireRouter->currentEpoch();
    const uint64_t epochGap = (curEpoch > minEpoch) ? (curEpoch - minEpoch) : 0;

    // epochGap が大きく pendingRetire が滞留 → Reader 残留の可能性
    if (pendingCount > 0 && epochGap > 10)
    {
        const uint64_t nowUs = convo::getCurrentTimeUs();
        MonitorState newState = MonitorState::Warning;
        const bool severe = (pendingCount > 100 || epochGap > 100);

        if (severe)
            newState = MonitorState::Error;
        else
            newState = MonitorState::Warning;

        HealthEvent ev{nowUs,
                       severe ? HealthEvent::Severity::Error : HealthEvent::Severity::Warning,
                       EVENT_READER_STUCK,
                       pendingCount,
                       0};
        ev.readerIndex = -1; // 完全な Reader 特定は EpochDomain 直接アクセスが必要
        ev.readerEpoch = minEpoch;
        ev.readerDepth = 0;
        ev.residencyTimeUs = 0;

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

} // namespace convo

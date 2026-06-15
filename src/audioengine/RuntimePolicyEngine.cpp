// [work37 Phase 0] RuntimePolicyEngine — MonitorState 駆動型回復行動選択器
// 実装詳細

#include "RuntimePolicyEngine.h"
#include "RuntimeHealthMonitor.h"  // MonitorState の完全定義のため

#include <chrono>

namespace convo {

RuntimePolicyEngine::RuntimePolicyEngine() noexcept
{
    // 各 RecoveryAction のデフォルト Cooldown を設定（μs）
    m_cooldowns[static_cast<size_t>(RecoveryAction::Observe)].cooldownUs  = 0;
    m_cooldowns[static_cast<size_t>(RecoveryAction::Throttle)].cooldownUs = 1'000'000;     // 1秒
    m_cooldowns[static_cast<size_t>(RecoveryAction::Recover)].cooldownUs = 10'000'000;    // ★ Work38: 5→10秒（誤検出時の無駄なAction発行抑制）
    m_cooldowns[static_cast<size_t>(RecoveryAction::Restore)].cooldownUs = 30'000'000;    // 30秒
    m_cooldowns[static_cast<size_t>(RecoveryAction::Safe)].cooldownUs   = 60'000'000;    // 60秒
    m_cooldowns[static_cast<size_t>(RecoveryAction::Critical)].cooldownUs = 120'000'000;  // 120秒
}

uint64_t RuntimePolicyEngine::getNowUs() const noexcept
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool RuntimePolicyEngine::canExecute(RecoveryAction action) const noexcept
{
    const auto idx = static_cast<size_t>(action);
    if (idx >= static_cast<size_t>(RecoveryAction::_Count))
        return false;
    const auto& entry = m_cooldowns[idx];
    const uint64_t nowUs = getNowUs();
    return (nowUs - entry.lastExecutedUs) >= entry.cooldownUs;
}

void RuntimePolicyEngine::markExecuted(RecoveryAction action) noexcept
{
    const auto idx = static_cast<size_t>(action);
    if (idx >= static_cast<size_t>(RecoveryAction::_Count))
        return;
    m_cooldowns[idx].lastExecutedUs = getNowUs();
}

RecoveryAction RuntimePolicyEngine::selectHighestPriority(
    RecoveryActionBits bits) const noexcept
{
    // 高い enum 値 = 高い優先度
    for (int i = static_cast<int>(RecoveryAction::_Count) - 1; i >= 0; --i) {
        if (bits & toBit(static_cast<RecoveryAction>(i)))
            return static_cast<RecoveryAction>(i);
    }
    return RecoveryAction::Observe;
}

PolicyDecision RuntimePolicyEngine::evaluateAggregate(
    MonitorState retireStall,
    MonitorState publicationStall,
    MonitorState readerSlotUsage,
    MonitorState overflowRate,
    MonitorState retireAge,
    MonitorState crossfadeDrop) noexcept
{
    PolicyDecision decision;
    HealthCauseBits causes = 0;

    // 各 MonitorState から RecoveryAction を選択
    // レベル1 (Throttle): 軽度の抑制
    if (retireStall == MonitorState::Warning || readerSlotUsage == MonitorState::Warning) {
        decision.actions |= toBit(RecoveryAction::Throttle);
    }

    // レベル2 (Recover): 中度の能動的回復
    if (retireStall == MonitorState::Error || retireAge == MonitorState::Error) {
        decision.actions |= toBit(RecoveryAction::Recover);
        causes |= static_cast<HealthCauseBits>(HealthCause::RetireStall);
    }
    if (publicationStall == MonitorState::Error) {
        decision.actions |= toBit(RecoveryAction::Recover);
        causes |= static_cast<HealthCauseBits>(HealthCause::PublicationStall);
    }
    if (overflowRate == MonitorState::Error) {
        decision.actions |= toBit(RecoveryAction::Throttle);
        causes |= static_cast<HealthCauseBits>(HealthCause::OverflowRate);
    }
    if (crossfadeDrop == MonitorState::Error) {
        decision.actions |= toBit(RecoveryAction::Recover);
        causes |= static_cast<HealthCauseBits>(HealthCause::CrossfadeEventDrop);
    }

    // レベル5 (Critical): ReaderSlot 枯渇
    if (readerSlotUsage == MonitorState::Error) {
        decision.actions |= toBit(RecoveryAction::Critical);
        causes |= static_cast<HealthCauseBits>(HealthCause::ReaderSlotExhaustion);
    }

    // 最高優先度の Action のみ残す（複数Action同時発行防止）
    // ★ Observe のときは actions を 0 にクリア（toBit(Observe)=1 が誤って action=0 を発行するのを防止）
    const auto highest = selectHighestPriority(decision.actions);
    if (highest == RecoveryAction::Observe) {
        decision.actions = 0;
    } else {
        decision.actions = toBit(highest);
        // Cooldown チェック
        if (!canExecute(highest))
            decision.actions = 0;
    }
    decision.causes = causes;

    return decision;
}

PolicyDecision RuntimePolicyEngine::evaluateEvent(
    PolicySource source,
    ConsistencyFailureType consistencyType) noexcept
{
    PolicyDecision decision;

    switch (source) {
        case PolicySource::WorldConsistency:
            if (consistencyType == ConsistencyFailureType::WorldLeak) {
                decision.actions = toBit(RecoveryAction::Critical);
                decision.causes = static_cast<HealthCauseBits>(HealthCause::WorldLeak);
            } else if (consistencyType == ConsistencyFailureType::DoubleRetire) {
                decision.actions = toBit(RecoveryAction::Critical);
                decision.causes = static_cast<HealthCauseBits>(HealthCause::WorldLeak);
            }
            break;

        case PolicySource::EmergencyCondition:
            decision.actions = toBit(RecoveryAction::Critical);
            decision.causes = static_cast<HealthCauseBits>(HealthCause::EmergencyDrain);
            break;

        case PolicySource::LearnerAnomaly:
            decision.actions = toBit(RecoveryAction::Throttle);
            decision.causes = static_cast<HealthCauseBits>(HealthCause::LearnerBackpressure);
            break;

        case PolicySource::AudioOutputAnomaly:
            decision.actions = toBit(RecoveryAction::Restore);
            break;

        case PolicySource::SafeModeState:
            decision.actions = toBit(RecoveryAction::Safe);
            break;

        default:
            break;
    }

    // Cooldown チェック
    const auto action = selectHighestPriority(decision.actions);
    if (action != RecoveryAction::Observe && !canExecute(action)) {
        decision.actions = 0;
    }

    return decision;
}

void RuntimePolicyEngine::reset() noexcept
{
    for (auto& entry : m_cooldowns) {
        entry.lastExecutedUs = 0;
    }
    m_lastAction_ = RecoveryAction::Observe;
    for (auto& ve : m_verificationEntries_)
        ve.state = VerificationState::Idle;
    m_budget_.reset();
}

// [work39 Phase 3] Verification
void RuntimePolicyEngine::markForVerification(RecoveryAction action, const TrendSnapshot& snapshot) noexcept
{
    const auto idx = static_cast<size_t>(action);
    if (idx >= static_cast<size_t>(RecoveryAction::_Count)) return;
    m_lastAction_ = action;
    auto& entry = m_verificationEntries_[idx];
    entry.state = VerificationState::PendingVerification;
    entry.executedAtUs = getNowUs();
    entry.verifyAfterUs = 50'000;  // 50ms初期値
    entry.restoreGeneration = snapshot.restoreGeneration;
    entry.baselineSnapshot = snapshot;
    entry.lastSnapshot = snapshot;
    entry.stalledCount = 0;
}

VerificationEntry& RuntimePolicyEngine::getEntry(RecoveryAction action) noexcept
{
    return m_verificationEntries_[static_cast<size_t>(action)];
}

const VerificationEntry& RuntimePolicyEngine::getEntry(RecoveryAction action) const noexcept
{
    return m_verificationEntries_[static_cast<size_t>(action)];
}

void RuntimePolicyEngine::resetVerification() noexcept
{
    for (auto& ve : m_verificationEntries_) {
        ve.state = VerificationState::Idle;
        ve.stalledCount = 0;
    }
    m_lastAction_ = RecoveryAction::Observe;
}

bool RuntimePolicyEngine::hasPendingVerification() const noexcept
{
    for (const auto& ve : m_verificationEntries_) {
        if (ve.state == VerificationState::PendingVerification)
            return true;
    }
    return false;
}

void RuntimePolicyEngine::markExecutedCritical(RecoveryAction action) noexcept
{
    // Critical時: verification をクリア（hasPendingVerification()=false）
    const auto idx = static_cast<size_t>(action);
    if (idx < static_cast<size_t>(RecoveryAction::_Count)) {
        m_verificationEntries_[idx].state = VerificationState::Idle;
    }
    m_lastAction_ = action;
    markExecuted(action);
}

// [work39 Phase 4] Budget アクセス
RecoveryBudget& RuntimePolicyEngine::getBudget() noexcept
{
    return m_budget_;
}

const RecoveryBudget& RuntimePolicyEngine::getBudget() const noexcept
{
    return m_budget_;
}

// [work39 Phase 4] RecoveryBudget 実装
bool RecoveryBudget::isExhausted(uint64_t nowUs) const noexcept
{
    if (latched) return true;
    if (nowUs - windowStartUs > kBudgetWindowUs) return false;
    return cycleCountInWindow >= kMaxCyclesPerWindow
        || criticalCount >= kMaxCriticalCount
        || recoverCount >= kMaxRecoverCount;
}

bool RecoveryBudget::isStormDetected(RecoveryAction action, uint64_t nowUs) const noexcept
{
    return escalationTracker.isStormDetected(action, nowUs);
}

void RecoveryBudget::record(RecoveryAction action, uint64_t nowUs) noexcept
{
    if (nowUs - windowStartUs > kBudgetWindowUs) {
        // window 期限切れ → リセット
        windowStartUs = nowUs;
        cycleCountInWindow = 0;
        criticalCount = 0;
        recoverCount = 0;
        latched = false;
    }
    if (action == RecoveryAction::Critical) ++criticalCount;
    if (action == RecoveryAction::Recover) ++recoverCount;
    lastEscalationUs = nowUs;
    escalationTracker.record(action, nowUs);
}

void RecoveryBudget::recordCycleCompletion(uint64_t nowUs) noexcept
{
    ++cycleCountInWindow;
    ladderStep = 0;
    lastRecoverySuccessUs = nowUs;
    // latched は回復
    latched = false;
}

void RecoveryBudget::recordHeavyReach(uint64_t nowUs) noexcept
{
    lastEscalationUs = nowUs;
}

void RecoveryBudget::reset() noexcept
{
    cycleCountInWindow = 0;
    criticalCount = 0;
    recoverCount = 0;
    ladderStep = 0;
    windowStartUs = 0;
    lastRecoverySuccessUs = 0;
    lastEscalationUs = 0;
    escalationTracker.reset();
    latched = false;
}

} // namespace convo

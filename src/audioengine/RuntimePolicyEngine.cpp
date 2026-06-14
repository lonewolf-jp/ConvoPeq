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
    const auto highest = selectHighestPriority(decision.actions);
    decision.actions = toBit(highest);
    decision.causes = causes;

    // Cooldown チェック
    if (highest != RecoveryAction::Observe && !canExecute(highest)) {
        decision.actions = 0;  // Cooldown 中は Action を発行しない
    }

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
}

} // namespace convo

#pragma once

// [work37 Phase 0] RuntimePolicyEngine — MonitorState 駆動型回復行動選択器
//
// 設計方針:
//   - HealthMonitor の既存 MonitorState (Normal/Warning/Error) から RecoveryAction を選択
//   - Cooldown 制御で同一 Action の連続実行を防止
//   - PolicyDecision を updateHealthState() に渡し、HealthState は HealthMonitor が最終決定
//   - 新たな Severity/Persistence/BlastRadius 体系は導入しない
//
// ★ v3.0: PolicyEngine は HealthState を直接書き換えない
//
// 使用法:
//   RuntimePolicyEngine engine;
//   auto decision = engine.evaluateAggregate(retireStall, pubStall, ...);
//   if (decision.actions != RecoveryActionBits{0}) { executeRecoveryAction(decision); }

#include <array>
#include <atomic>
#include <cstdint>

namespace convo {

enum class MonitorState : uint8_t;  // forward declare from RuntimeHealthMonitor.h

// [work37 Phase 0] PolicySource — m_contexts のキー。eventCode 非依存。
// ★ v6.6統合版: 72→14分類
enum class PolicySource : uint8_t {
    RetireStall,            // Retire系: RetireStall/RetireAge/Overflow/RebuildSuppressionTTL
    PublicationStall,       // 出版系: PublicationStall/ProgressFreeze/ConfigurationDeadlock
    ReaderStuck,            // Reader系: ReaderSlotUsage/EpochAdvanceBlocked/ActiveReaderBlocker
    CrossfadeTimeout,       // Crossfade系: CrossfadeTimeout/CrossfadeEventDrop
    LearnerAnomaly,         // Learner系: LearnerBackpressure/LearnerStall/LearnerDivergence
    WorldConsistency,       // World整合性: WorldLeak/WorldConsistency/ConfigurationDivergence
    AudioOutputAnomaly,     // 音響系: DC Offset/Peak Clipping/RMS Jump/Noise Floor
    EmergencyCondition,     // 緊急系: EmergencyDrain/ShutdownTimeout
    RecoveryOutcome,        // 回復結果: Success/NoEffect/Failed
    SafeModeState,          // SafeMode系: SoftActive/HardActive/RecoveryReady
    _Count                  // 要素数 (= 10)
};

// [work39 Phase 1] RestorePhase — Restore Step進捗管理
enum class RestorePhase : uint8_t {
    None,                    // Restore未実行
    EpochRecoveryIssued,     // Step1完了（Epoch Recovery + Learner Rollback）
    LearnerRollbackDone,     // Learner復元完了
    IdleWorldPublished       // Step2完了（publishIdleWorldOnly）
};

// [work37 Phase 0] RecoveryAction — 6レベル階層化 (v6.6統合版)
enum class RecoveryAction : uint8_t {
    Observe,                // Level 0: 監視のみ。HealthEvent記録。
    Throttle,               // Level 1: 抑制。admissionStrict/PauseLearner/Suppress。
    Recover,                // Level 2: 回復。ForceRetireDrain/ForceSnapshotPublish/Escalate。
    Restore,                // Level 3: 復元。Rollback/LearnerRollback/CheckpointRestore。
    Safe,                   // Level 4: 安全確保。SoftSafeMode(ConvByPass+LearnerStop)/HardSafeMode(1x+FlatEQ)。
    Critical,               // Level 5: 重大。RejectNewPublication/EmergencyDrain/Shutdown。
    _Count                  // 要素数 (= 6)
};

// [work39 Phase 3] RecoveryOutcome — 閉ループ制御用（再定義）
enum class RecoveryOutcome : uint8_t {
    None,          // 未評価
    Improving,     // 改善傾向 — 維持、昇格禁止
    Recovered,     // 正常復帰 — Observe へ移行
    Stalled,       // 停滞 — 次段階へ昇格
    Worsening      // 悪化 — 即時昇格
};

// [work39 Phase 1/3] TrendSnapshot — 閉ループ制御用スナップショット
struct TrendSnapshot {
    uint64_t pendingRetire{0};
    uint64_t publicationSeq{0};
    uint64_t maxRetireAgeUs{0};
    uint32_t activeReaderCount{0};
    uint32_t readerStuckCount{0};
    bool     freezeDetected{false};
    uint32_t activeFaultMask{0};
    uint64_t restoreGeneration{0};
    uint64_t epochAdvanceCount{0};       // Epoch advance 累積回数
    uint64_t lastCompletedEpoch{0};      // 最終完了Epoch ID
    uint64_t publicationGeneration{0};   // Publication世代
    RestorePhase restorePhase{RestorePhase::None};
};

// activeFaultMask ビット定義
static constexpr uint32_t kFaultRetire       = 1u << 0;
static constexpr uint32_t kFaultPublication  = 1u << 1;
static constexpr uint32_t kFaultReader       = 1u << 2;
static constexpr uint32_t kFaultOverflow     = 1u << 3;
static constexpr uint32_t kFaultCrossfade    = 1u << 4;
static constexpr uint32_t kFaultLearner      = 1u << 5;
static constexpr uint32_t kFaultConfigDiverg = 1u << 6;
static constexpr uint32_t kFaultEpochAdvance = 1u << 7;  // Epoch Advance Blocked

// [work39 Phase 3] EpochAdvanceHealth（問題G）
struct EpochAdvanceHealth {
    uint64_t lastAdvanceUs{0};
    uint64_t currentEpoch{0};
    uint64_t completedEpoch{0};

    static constexpr uint64_t kEpochAdvanceStallWindowUs = 1 * 1'000'000;  // 1秒間進まなければStall

    [[nodiscard]] bool isBlocked(uint64_t nowUs) const noexcept {
        return nowUs - lastAdvanceUs > kEpochAdvanceStallWindowUs;
    }
};

// [work39 Phase 3] VerificationState + VerificationEntry
enum class VerificationState : uint8_t { Idle, PendingVerification };

struct VerificationEntry {
    VerificationState state{VerificationState::Idle};
    uint64_t executedAtUs{0};
    uint64_t verifyAfterUs{0};       // 初期値50ms推奨
    uint64_t restoreGeneration{0};
    TrendSnapshot baselineSnapshot;
    TrendSnapshot lastSnapshot;
    uint8_t stalledCount{0};         // 上限3回

    [[nodiscard]] bool isIdle() const noexcept {
        return state == VerificationState::Idle;
    }
};

// [work39 Phase 4] toRecoveryLevel()
[[nodiscard]] constexpr uint8_t toRecoveryLevel(RecoveryAction action) noexcept {
    switch (action) {
        case RecoveryAction::Observe:  return 0;
        case RecoveryAction::Throttle: return 1;
        case RecoveryAction::Recover:  return 2;
        case RecoveryAction::Restore:  return 3;
        case RecoveryAction::Safe:     return 4;
        case RecoveryAction::Critical: return 5;
        default:                       return 0;
    }
}

// [work39 Phase 3] nextAction() — Ladder 昇格
[[nodiscard]] inline RecoveryAction nextAction(RecoveryAction current) noexcept {
    switch (current) {
        case RecoveryAction::Throttle: return RecoveryAction::Recover;
        case RecoveryAction::Recover:  return RecoveryAction::Restore;
        case RecoveryAction::Restore:  return RecoveryAction::Safe;
        case RecoveryAction::Safe:     return RecoveryAction::Critical;
        default:                       return RecoveryAction::Critical;
    }
}

// [work39 Phase 4] EscalationTracker（問題C/A-2）
struct EscalationTracker {
    static constexpr uint8_t toIndex(RecoveryAction action) noexcept {
        switch (action) {
            case RecoveryAction::Observe:  return 0;
            case RecoveryAction::Throttle: return 1;
            case RecoveryAction::Recover:  return 2;
            case RecoveryAction::Restore:  return 3;
            case RecoveryAction::Safe:     return 4;
            case RecoveryAction::Critical: return 5;
            default:                       return 0;
        }
    }

    uint64_t lastActionUs[6]{0,0,0,0,0,0};
    uint32_t repeatCount[6]{0,0,0,0,0,0};
    uint8_t  lastLevel{0};
    uint32_t levelOscillationCount{0};
    uint32_t transitionPairCount[3]{0,0,0};

    static constexpr uint64_t kStormWindowUs = 30 * 1'000'000;
    static constexpr uint32_t kMaxRepeatBeforeStorm = 3;
    static constexpr uint32_t kMaxOscillationBeforeStorm = 4;

    [[nodiscard]] bool isStormDetected(RecoveryAction action, uint64_t nowUs) const noexcept {
        const auto idx = toIndex(action);
        if (lastActionUs[idx] == 0) return false;
        return (nowUs - lastActionUs[idx]) < kStormWindowUs
            && repeatCount[idx] >= kMaxRepeatBeforeStorm;
    }

    void record(RecoveryAction action, uint64_t nowUs) noexcept {
        const auto idx = toIndex(action);
        if (nowUs - lastActionUs[idx] < kStormWindowUs)
            ++repeatCount[idx];
        else
            repeatCount[idx] = 0;
        lastActionUs[idx] = nowUs;

        const uint8_t currentLevel = idx;
        if (lastLevel != 0 && currentLevel != lastLevel
            && std::abs(static_cast<int>(currentLevel) - static_cast<int>(lastLevel)) == 1) {
            ++levelOscillationCount;
            const uint8_t pairIdx = std::min(lastLevel, currentLevel);
            if (pairIdx >= 1 && pairIdx <= 3)
                ++transitionPairCount[pairIdx - 1];
        } else {
            levelOscillationCount = 0;
        }
        lastLevel = currentLevel;
    }

    void reset() noexcept {
        for (auto& tc : lastActionUs) tc = 0;
        for (auto& rc : repeatCount) rc = 0;
        lastLevel = 0;
        levelOscillationCount = 0;
        for (auto& tc : transitionPairCount) tc = 0;
    }
};

// [work39 Phase 4] RecoveryBudget
struct RecoveryBudget {
    uint32_t cycleCountInWindow{0};
    uint32_t criticalCount{0};
    uint32_t recoverCount{0};
    uint8_t  ladderStep{0};
    uint64_t windowStartUs{0};
    uint64_t lastRecoverySuccessUs{0};
    uint64_t lastEscalationUs{0};
    EscalationTracker escalationTracker;
    bool     latched{false};

    static constexpr uint64_t kBudgetWindowUs = 10 * 60 * 1'000'000;       // 10分
    static constexpr uint64_t kStableResetUs  = 15 * 60 * 1'000'000;       // 15分
    static constexpr uint32_t kMaxCyclesPerWindow = 3;
    static constexpr uint32_t kMaxCriticalCount = 5;
    static constexpr uint32_t kMaxRecoverCount = 20;

    [[nodiscard]] bool isExhausted(uint64_t nowUs) const noexcept;
    [[nodiscard]] bool isStormDetected(RecoveryAction action, uint64_t nowUs) const noexcept;
    void record(RecoveryAction action, uint64_t nowUs) noexcept;
    void recordCycleCompletion(uint64_t nowUs) noexcept;
    void recordHeavyReach(uint64_t nowUs) noexcept;
    void reset() noexcept;
};

// ★ v3.0: RecoveryActionBits — ビットマスク（複数Action同時発行可能）
using RecoveryActionBits = uint8_t;

constexpr RecoveryActionBits toBit(RecoveryAction action) noexcept {
    return static_cast<RecoveryActionBits>(1) << static_cast<uint8_t>(action);
}

// [work37 Phase 0] HealthCause — Critical/Degraded の原因を特定
// ★ v3.2: uint64_t causeBits で複合原因に対応
enum class HealthCause : uint64_t {
    None                  = 0,
    RetireStall           = 1ull << 0,
    PublicationStall      = 1ull << 1,
    ReaderStuck           = 1ull << 2,
    ReaderSlotExhaustion  = 1ull << 3,
    OverflowRate          = 1ull << 4,
    RetireAged            = 1ull << 5,
    CrossfadeTimeout      = 1ull << 6,
    CrossfadeEventDrop    = 1ull << 7,
    WorldLeak             = 1ull << 8,
    WorldConsistencyBad   = 1ull << 9,
    EmergencyDrain        = 1ull << 10,
    FallbackQueueOverflow = 1ull << 11,
    LearnerBackpressure   = 1ull << 12,
    ConfigurationDivergence = 1ull << 13
};

// HealthCauseBits: OR可能な複合原因
using HealthCauseBits = uint64_t;

// [work37 Phase 0] ConsistencyFailureType — World整合性異常の種類
enum class ConsistencyFailureType : uint8_t {
    None,
    AuditMismatch,     // 監査レコード欠損（published-retired≠active）— Warning
    WorldLeak,         // 実Worldリーク（retired>published）— Critical
    DoubleRetire,      // 二重retire検出 — Critical
    Unknown
};

// [work37 Phase 0] PolicyDecision — 統合評価結果
struct PolicyDecision {
    RecoveryActionBits actions{0};  // ビットマスク（複数Action同時発行可能）
    uint32_t cooldownUs{0};         // 同一 Action 再実行間隔
    HealthCauseBits causes{0};      // 複合原因（OR可能）
    // ★ v3.6: targetHealth は持たない。HealthStateは HealthMonitor 専権。
};

// [work37 Phase 0] CooldownEntry — 同一 Action の連続実行防止
struct CooldownEntry {
    uint64_t lastExecutedUs{0};
    uint64_t cooldownUs{0};
};

// [work37 Phase 9.56 P2] RuntimeRecoveryScore — 4軸総合スコア（診断用）
struct RuntimeRecoveryScore {
    uint8_t publishProgress{0};    // publicationSequence の増加率 (0-25)
    uint8_t retireProgress{0};     // pendingRetire の減少率 (0-25)
    uint8_t rebuildProgress{0};    // pendingIntent + deferredAge (0-25)
    uint8_t audioQuality{0};       // AudioOutputAnomaly + LearnerOutputDivergence (0-25)

    [[nodiscard]] uint8_t total() const noexcept {
        return publishProgress + retireProgress + rebuildProgress + audioQuality;
    }
    static constexpr uint8_t kHealthyThreshold = 80;
    [[nodiscard]] bool isHealthy() const noexcept { return total() >= kHealthyThreshold; }
};

// [work37 Phase 0] RuntimePolicyEngine — MonitorState → 最優先RecoveryAction の選択器
//   複数Actionを同時発行せず、最高優先度のActionのみ返す。
//   Action優先順位（高い順）:
//      1. Critical (EmergencyDrain, RejectNewPublication)
//      2. Safe (EnterSafeMode)
//      3. Restore (RollbackToLastHealthyWorld)
//      4. Recover (ForceRetireDrain, ForceSnapshotPublish)
//      5. Throttle (ThrottleRebuild, PauseLearner)
//      6. Observe (None)
//   HealthState の決定は RuntimeHealthMonitor::updateHealthState() が唯一の権限。
//
//   スレッド安全性: AudioEngine::timerCallback() (Message Thread) からのみ呼ばれるため、
//   m_cooldowns 配列の読み書きにデータ競合は発生しない。

class RuntimePolicyEngine {
public:
    RuntimePolicyEngine() noexcept;

    // 全 MonitorState から統合評価（HealthMonitor::tick から呼ばれる）
    PolicyDecision evaluateAggregate(
        MonitorState retireStall,
        MonitorState publicationStall,
        MonitorState readerSlotUsage,
        MonitorState overflowRate,
        MonitorState retireAge,
        MonitorState crossfadeDrop) noexcept;

    // 特定 PolicySource に対する評価
    PolicyDecision evaluateEvent(PolicySource source,
                                  ConsistencyFailureType consistencyType
                                      = ConsistencyFailureType::None) noexcept;

    // Cooldown 制御
    bool canExecute(RecoveryAction action) const noexcept;
    void markExecuted(RecoveryAction action) noexcept;

    // リセット
    void reset() noexcept;

    // [work39 Phase 3] Verification（閉ループ制御）
    void markForVerification(RecoveryAction action, const TrendSnapshot& snapshot) noexcept;
    VerificationEntry& getEntry(RecoveryAction action) noexcept;
    const VerificationEntry& getEntry(RecoveryAction action) const noexcept;
    void resetVerification() noexcept;
    [[nodiscard]] bool hasPendingVerification() const noexcept;
    RecoveryAction getLastExecutedAction() const noexcept { return m_lastAction_; }
    void markExecutedCritical(RecoveryAction action) noexcept;
    // [work39 Phase 4] Budget アクセス
    struct RecoveryBudget& getBudget() noexcept;
    const struct RecoveryBudget& getBudget() const noexcept;

private:
    // 内部 Action 優先順位評価
    RecoveryAction selectHighestPriority(RecoveryActionBits bits) const noexcept;

    // 時間取得
    uint64_t getNowUs() const noexcept;

    std::array<CooldownEntry, static_cast<size_t>(RecoveryAction::_Count)> m_cooldowns;

    // [work39 Phase 3] Verification 状態
    RecoveryAction m_lastAction_{RecoveryAction::Observe};
    std::array<VerificationEntry, static_cast<size_t>(RecoveryAction::_Count)> m_verificationEntries_;
    // [work39 Phase 4] Budget
    struct RecoveryBudget m_budget_;
};

} // namespace convo

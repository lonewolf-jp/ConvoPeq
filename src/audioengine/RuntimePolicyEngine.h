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

// [work37 Phase 9.45] RecoveryOutcome — 回復成功/失敗の閉ループ制御
enum class RecoveryOutcome : uint8_t {
    None,           // 未評価
    Success,        // 回復成功（状態改善確認済み）
    NoEffect,       // 効果なし（状態不変）
    Failed,         // 状態悪化
    Unsafe          // 安全条件違反で発動中止
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

private:
    // 内部 Action 優先順位評価
    RecoveryAction selectHighestPriority(RecoveryActionBits bits) const noexcept;

    // 時間取得
    uint64_t getNowUs() const noexcept;

    std::array<CooldownEntry, static_cast<size_t>(RecoveryAction::_Count)> m_cooldowns;
};

} // namespace convo

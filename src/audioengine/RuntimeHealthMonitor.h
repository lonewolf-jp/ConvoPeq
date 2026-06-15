#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include "AtomicAccess.h"
#include "RuntimePolicyEngine.h"  // ★ work37 Phase 4: PolicyEngine 連携

class AudioSegmentBuffer;  // ★ Work39: Learner FIFO 監視用（global scope）

namespace convo {

struct HealthEvent {
    enum class Severity : uint8_t { Info, Warning, Error };
    uint64_t timestampUs;
    Severity severity;
    uint32_t eventCode;
    uint64_t value;
    uint32_t slot;
    // ★ P4.5: Reader Residency Diagnostics
    int32_t readerIndex{-1};
    uint64_t readerEpoch{0};
    uint32_t readerDepth{0};
    uint64_t residencyTimeUs{0};
};

namespace isr {
class ISRRetireRouter;
class RuntimePublicationOrchestrator;
class CrossfadeRuntime;  // ★ P1-C: 前方宣言
}

// ★ P1-B: ISR Health State — HealthMonitor が一元管理し Admission へ公開
enum class ISRHealthState : uint8_t {
    Healthy = 0,    // 正常
    Degraded,       // 軽度障害（Reader枯渇 / Retire backlog増加）
    Critical        // 重度障害（Publication stall / Reader stuck / Timeout）
};

static constexpr uint32_t EVENT_RETIRE_STALL         = 1001;
static constexpr uint32_t EVENT_RETIRE_STALL_WARNING = 1002;
static constexpr uint32_t EVENT_PUBLICATION_STALL    = 2001;
static constexpr uint32_t EVENT_PUBLICATION_WARNING  = 2002;
static constexpr uint32_t EVENT_READER_STUCK         = 3001;
static constexpr uint32_t EVENT_READER_SLOT_USAGE     = 3010;  // ★ P1-B/Practical-4
static constexpr uint32_t EVENT_CROSSFADE_TIMEOUT    = 4001;  // ★ P1-C/Practical-2
static constexpr uint32_t EVENT_CROSSFADE_EVENT_DROP = 4002;  // ★ P1-C/Practical-6
// ★ Work39: Learner FIFO Backpressure
static constexpr uint32_t EVENT_LEARNER_BACKPRESSURE_WARNING = 5001;  // FIFO 85%+
static constexpr uint32_t EVENT_LEARNER_BACKPRESSURE_ERROR   = 5002;  // FIFO 95%+
// ★ Work38: Retire Age 正常復帰イベント（emitOnTransition 3状態遷移完全カバー）
static constexpr uint32_t EVENT_RETIRE_AGE_NORMAL     = 1009;  // ★ Work38
static constexpr uint32_t EVENT_RETIRE_AGE_WARNING   = 1010;  // ★ Practical-5
static constexpr uint32_t EVENT_RETIRE_AGE_CRITICAL  = 1011;  // ★ Practical-5

// ★ P1-C: Crossfade Timeout（固定30秒）
static constexpr uint64_t kCrossfadeTimeoutUs = 30'000'000;
// ★ P1-C: Crossfade Event Drop 閾値（差分ベース）
static constexpr uint64_t kCrossfadeEventDropCriticalDelta = 10;
static constexpr uint64_t kCrossfadeEventDropWarningDelta  = 1;
// ★ Practical-4: Reader Slot 閾値
static constexpr double kReaderSlotWarningThreshold  = 0.50;
static constexpr double kReaderSlotCriticalThreshold = 0.75;
// ★ Practical-5: Retire Age 閾値
static constexpr uint64_t kRetireAgeWarningUs  = 5'000'000;   // 5秒
static constexpr uint64_t kRetireAgeCriticalUs = 30'000'000;  // 30秒

enum class MonitorState : uint8_t { Normal, Warning, Error };
using HealthEventCallback = std::function<void(const HealthEvent&)>;

// [work39 Phase 7] CriticalExitBlocker — Critical 出口ブロック理由（診断用）
enum class CriticalExitBlocker : uint8_t {
    None,
    MonitorNotNormal,
    SuppressionActive,
    RecoveryRunning,
    StableDurationInsufficient,
    PendingRetireExceeded,
    RetireAgeExceeded
};

// [work39 Phase 7] CriticalExitCondition — Critical 出口評価構造体
struct CriticalExitCondition {
    bool allMonitorsNormal{false};
    bool suppressionInactive{false};
    bool noRecoveryActionRunning{false};
    bool stableDuration{false};
    CriticalExitBlocker blocker{CriticalExitBlocker::None};
    uint64_t pendingRetire{0};
    uint64_t retireAgeUs{0};

    [[nodiscard]] bool canExit() noexcept {
        if (!allMonitorsNormal) { blocker = CriticalExitBlocker::MonitorNotNormal; return false; }
        if (!suppressionInactive) { blocker = CriticalExitBlocker::SuppressionActive; return false; }
        if (!noRecoveryActionRunning) { blocker = CriticalExitBlocker::RecoveryRunning; return false; }
        if (!stableDuration) { blocker = CriticalExitBlocker::StableDurationInsufficient; return false; }
        blocker = CriticalExitBlocker::None;
        return true;
    }
};

/**
 * RuntimeHealthMonitor: Pull型監視エンジン。
 *
 * Phase 1 スコープ:
 *   - Retire Backlog 監視（queue depth ベース、状態遷移検出）
 *   - Publication Stall 監視（sequence 進捗＋deferred age ベース）
 *
 * 設計メモ:
 *   - callback は std::function を使用（Phase 1 では十分）
 *   - 将来 allocation 懸念が出た場合は AudioEngine* 直接参照に置き換え
 */
class RuntimeHealthMonitor {
public:
    // ★ P1-C/Practical-2: CrossfadeRuntime 参照設定
    void setRetireRouter(isr::ISRRetireRouter* router) noexcept { m_retireRouter = router; }
    void setOrchestrator(isr::RuntimePublicationOrchestrator* orch) noexcept { m_orchestrator = orch; }
    void setRetireHighWatermarkRef(const std::atomic<int>* ref) noexcept { m_retireHighWatermarkRef = ref; }
    void setEventCallback(HealthEventCallback cb) noexcept { m_callback = std::move(cb); }
    // ★ P1-C/Practical-2: CrossfadeRuntime 参照設定（crossfade timeout + event drop 監視用）
    void setCrossfadeRuntime(const isr::CrossfadeRuntime* rt) noexcept { m_crossfadeRuntime = rt; }
    // ★ Practical-2/5/6: 診断用参照設定
    void setCrossfadeEventDropRef(const std::atomic<uint64_t>* ref) noexcept { m_crossfadeEventDropRef = ref; }
    void setMaxRetireAgeRef(const std::atomic<uint64_t>* ref) noexcept {
        m_maxRetireAgeRef = ref;
        m_maxRetireAgeDoubleRef = nullptr;
    }
    // ★ Work38: double 版オーバーロード — reclaimLatency_ 用（型安全）
    void setMaxRetireAgeRef(const std::atomic<double>* ref) noexcept {
        m_maxRetireAgeDoubleRef = ref;
        m_maxRetireAgeRef = nullptr;
    }
    void setReaderSlotRef(const std::atomic<uint32_t>* ref) noexcept { m_readerSlotRef = ref; }
    void setOverflowCountRef(const std::atomic<uint64_t>* ref) noexcept { m_overflowCountRef = ref; }

    void tick() noexcept;

    // ★ C-4: HealthState のみ初期化（m_prev*State は維持 — イベント再通知防止）
    void reset() noexcept;

    // [work37 Phase 4.1] PolicyEngine 連携
    //   RecoveryAction を受け取る callback（AudioEngine::executeRecoveryAction）
    using RecoveryActionCallback = std::function<void(RecoveryAction)>;
    void setActionCallback(RecoveryActionCallback cb) noexcept { m_actionCallback = std::move(cb); }

    // [work39 Phase 6] RestoreStep2 callback — publishIdleWorldOnly(HardReset) 発行用
    using RestoreStep2Callback = std::function<void()>;
    void setRestoreStep2Callback(RestoreStep2Callback cb) noexcept { m_restoreStep2Callback_ = std::move(cb); }

    // [work37 Phase 9.1] Learner Health Policy 用 — Retire Stall の継続時間を追跡
    void setLearnerRunningRef(const std::atomic<bool>* ref) noexcept { m_learnerRunningRef = ref; }
    [[nodiscard]] uint64_t getRetireStallDurationUs() const noexcept;

    // [work39 Phase 5] Learner FIFO 監視用セッター
    void setLearnerSegmentBuffer(::AudioSegmentBuffer* buf) noexcept { m_learnerSegmentBuffer_ = buf; }
    void setEpochAdvanceCountRef(const std::atomic<uint64_t>* ref) noexcept { m_epochAdvanceCountRef_ = ref; }
    void setLastCompletedEpochRef(const std::atomic<uint64_t>* ref) noexcept { m_lastCompletedEpochRef_ = ref; }

    // [work37 Phase 9.2] Configuration Divergence 監視用参照設定
    void setCommittedGenRef(const std::atomic<uint64_t>* ref) noexcept { m_lastCommittedGenRef_ = ref; }
    void setRequestedGenRef(const std::atomic<uint64_t>* ref) noexcept { m_requestedGenRef_ = ref; }

    // [work37 Phase 7.1] World Consistency 監視用コールバック
    //   AudioEngine が collectDrainAudit().verifyWorldConsistency() をラップして提供
    using WorldConsistencyCheck = std::function<uint8_t()>;  // 0=Consistent, 1=Suspicious, 2=Broken
    void setWorldConsistencyCheck(WorldConsistencyCheck cb) noexcept { m_worldConsistencyCheck_ = std::move(cb); }

    // [work37 Phase 9.8] Pending Deployment 監視用参照設定
    void setRequestedRebuildGenRef(const std::atomic<int>* ref) noexcept { m_requestedRebuildGenRef_ = ref; }
    void setCommittedRebuildGenRef(const std::atomic<int>* ref) noexcept { m_committedRebuildGenRef_ = ref; }

    // [work37 Phase 9.29] Suppression Duration 監視用 — 抑制開始時刻の参照
    //   AudioEngine 側で retirePressureAdmissionStrict_ 設定時に更新する原子を提供
    void setSuppressionStartRef(const std::atomic<uint64_t>* ref) noexcept {
        m_suppressionStartRef_ = ref;
    }

    // [work37 Phase 9.40] Progress Freeze 監視用参照
    void setLastRetireTimestampRef(const std::atomic<uint64_t>* ref) noexcept {
        m_lastRetireTimestampRef_ = ref;
    }
    void setPublicationSequenceRef(const std::atomic<uint64_t>* ref) noexcept {
        m_publicationSequenceRef_ = ref;
    }

    // [work37 Phase 9.10 P2] Configuration Drift 監視用参照
    void setManualOversamplingRef(const std::atomic<int>* ref) noexcept {
        m_manualOversamplingRef_ = ref;
    }

    // [work37 Phase 4.4] 背圧信号注入（drainDeferredRetireQueues から）— CAS max update
    void injectBackpressureSignal(std::size_t fallbackSize, double overflowRate) noexcept {
        // atomic CAS max update for fallbackSize
        uint64_t current = convo::consumeAtomic(m_maxFallbackSize_, std::memory_order_acquire);
        while (fallbackSize > current) {
            if (convo::compareExchangeAtomic(m_maxFallbackSize_, current,
                static_cast<uint64_t>(fallbackSize), std::memory_order_acq_rel,
                std::memory_order_acquire)) break;
            current = convo::consumeAtomic(m_maxFallbackSize_, std::memory_order_acquire);
        }
        // atomic CAS max update for overflowRate
        double rateCurrent = convo::consumeAtomic(m_maxOverflowRate_, std::memory_order_acquire);
        while (overflowRate > rateCurrent) {
            if (convo::compareExchangeAtomic(m_maxOverflowRate_, rateCurrent,
                overflowRate, std::memory_order_acq_rel, std::memory_order_acquire)) break;
            rateCurrent = convo::consumeAtomic(m_maxOverflowRate_, std::memory_order_acquire);
        }
        m_backpressureInjected_ = true;
    }

    // [work37 Phase 8.2] EmergencyDrain 要求/確認
    void requestEmergencyDrain() noexcept {
        convo::publishAtomic(m_emergencyDrainRequested_, true, std::memory_order_release);
    }
    [[nodiscard]] bool isEmergencyDrainRequested() const noexcept {
        return convo::consumeAtomic(m_emergencyDrainRequested_, std::memory_order_acquire);
    }
    void clearEmergencyDrain() noexcept {
        convo::publishAtomic(m_emergencyDrainRequested_, false, std::memory_order_release);
    }

    // [work37 Phase 4.4] PolicyEngine へのアクセス（Phase 9 拡張用）
    RuntimePolicyEngine& getPolicyEngine() noexcept { return m_policyEngine_; }
    const RuntimePolicyEngine& getPolicyEngine() const noexcept { return m_policyEngine_; }

    // ★ 8.6: ReaderStuck 定期Evidence 出力用定数
    static constexpr uint64_t kStuckEvidenceIntervalUs = 10'000'000; // 10秒間隔

    // ★ P1-B: HealthState 公開 — Admission が参照する
    [[nodiscard]] ISRHealthState getHealthState() const noexcept {
        return convo::consumeAtomic(m_healthState_, std::memory_order_acquire);
    }

    // ★ P1-B: Admission が HealthState を参照するための生ポインタ公開
    [[nodiscard]] const std::atomic<ISRHealthState>* getHealthStateRef() const noexcept {
        return &m_healthState_;
    }

private:
    void checkRetireStall() noexcept;
    void checkPublicationStall() noexcept;
    void diagnoseRetireStall() noexcept;
    void updateHealthState() noexcept;
    // [work37 Phase 4.1] PolicyDecision 対応版
    void updateHealthState(const PolicyDecision& decision) noexcept;
    // [work37 Phase 9.2] Configuration Divergence 監視
    void checkConfigurationDivergence() noexcept;
    // [work37 Phase 7.1] World Consistency 監視
    void checkWorldConsistency() noexcept;
    // [work37 Phase 9.7] Snapshot Starvation 監視（deferred publish age > 10s/30s）
    void checkSnapshotStarvation() noexcept;
    // [work37 Phase 9.8] Pending Structural Deployment 監視（rebuild generation gap）
    void checkPendingStructuralDeployment() noexcept;
    // [work37 Phase 9.29] Suppression Duration 監視（段階的エスカレーション）
    void checkSuppressionDuration() noexcept;
    // [work37 Phase 9.40] Runtime Progress Freeze 監視（3軸統合）
    void checkRuntimeProgressFreeze() noexcept;
    // [work37 Phase 9.10 P2] Configuration Drift 監視（oversamplingFactor 乖離）
    void checkConfigurationDrift() noexcept;
    // [work39 Phase 3] 閉ループ制御
    [[nodiscard]] TrendSnapshot takeSnapshot() const noexcept;
    [[nodiscard]] RecoveryOutcome computeTrend(const TrendSnapshot& before,
                                                const TrendSnapshot& now) const noexcept;
    // [work39 Phase 5] Learner FIFO 監視
    void checkLearnerBackpressure() noexcept;
    // ★ P1-C/Practical-2/4/5/6: 追加監視
    void checkCrossfadeTimeout() noexcept;
    void checkCrossfadeEventDrop() noexcept;
    void checkReaderSlotUsage() noexcept;
    void checkOverflowRate() noexcept;
    void checkRetireReclaimLatency() noexcept;
    void emitOnTransition(MonitorState& currentState, MonitorState newState,
                          HealthEvent::Severity severity, uint32_t eventCode,
                          uint64_t value, uint32_t slot = 0) noexcept;

    isr::ISRRetireRouter* m_retireRouter = nullptr;
    isr::RuntimePublicationOrchestrator* m_orchestrator = nullptr;
    const std::atomic<int>* m_retireHighWatermarkRef = nullptr;
    HealthEventCallback m_callback;
    MonitorState m_prevRetireState { MonitorState::Normal };
    MonitorState m_prevPublicationState { MonitorState::Normal };
    MonitorState m_prevCrossfadeDropState { MonitorState::Normal }; // ★ P1-C
    MonitorState m_prevReaderSlotState { MonitorState::Normal };    // ★ Practical-4
    MonitorState m_prevOverflowRateState { MonitorState::Normal };  // ★ Practical-3
    MonitorState m_prevRetireAgeState { MonitorState::Normal };     // ★ Practical-5
    std::atomic<ISRHealthState> m_healthState_{ISRHealthState::Healthy};
    // ★ P1-C/Practical-2/4/5/6: 監視用参照
    const convo::isr::CrossfadeRuntime* m_crossfadeRuntime = nullptr;
    const std::atomic<uint64_t>* m_crossfadeEventDropRef = nullptr;
    const std::atomic<uint64_t>* m_maxRetireAgeRef = nullptr;
    // ★ Work38: double 版参照（reclaimLatency_ 用）
    const std::atomic<double>* m_maxRetireAgeDoubleRef{nullptr};
    const std::atomic<uint32_t>* m_readerSlotRef = nullptr;
    const std::atomic<uint64_t>* m_overflowCountRef = nullptr;   // ★ Practical-3
    // ★ P1-C drop: 差分検出用ローカル状態
    uint64_t m_lastObservedDropCount = 0;

    // ★ Practical-3: Overflow rate monitoring
    uint64_t m_lastOverflowCount = 0;
    uint64_t m_lastOverflowCheckTimeUs = 0;
    uint64_t m_overflowRateStableSinceUs = 0; // ★ ヒステリシス: 安定状態到達時刻
    static constexpr uint64_t kOverflowRateWindowUs = 1'000'000; // 1秒窓
    static constexpr uint32_t kOverflowRateCriticalThreshold = 5; // 5回/秒超 → Critical
    static constexpr uint32_t kOverflowRateWarningThreshold = 1;  // 1回/秒以上 → Warning
    static constexpr uint64_t kOverflowHysteresisCriticalToDegradedUs = 10'000'000; // Critical→Degraded: 10秒安定
    static constexpr uint64_t kOverflowHysteresisDegradedToHealthyUs = 30'000'000; // Degraded→Healthy: 30秒安定

    // ★ Practical-4: Reclaim rate limit
    uint64_t m_lastForcedReclaimTimeUs = 0;
    static constexpr uint64_t kForcedReclaimCooldownUs = 500'000; // 500ms以内は再試行禁止
    // ★ 8.6: ReaderStuck 定期Evidence 出力タイムスタンプ
    uint64_t m_lastStuckEvidenceUs = 0;

    // [work37 Phase 4.1/9.1] PolicyEngine メンバ
    RuntimePolicyEngine m_policyEngine_;
    RecoveryActionCallback m_actionCallback;
    RestoreStep2Callback m_restoreStep2Callback_;  // [work39 Phase 6] Restore Step2

    // [work37 Phase 9.1] Retire Stall 継続時間追跡
    uint64_t m_retireStallStartUs_{0};
    const std::atomic<bool>* m_learnerRunningRef{nullptr};

    // [work39 Phase 5] Learner FIFO 監視
    MonitorState m_prevLearnerBackpressureState_{MonitorState::Normal};
    bool     m_learnerWasActive_{false};
    double   m_fifoEma_{-1.0};        // -1.0 = uninitialized
    double   m_lastFifoEma_{0.0};
    uint64_t m_lastFifoTickUs_{0};
    uint64_t m_learnerFifoHighSinceUs_{0};
    ::AudioSegmentBuffer* m_learnerSegmentBuffer_{nullptr};
    // epochAdvance / lastCompletedEpoch（問題A-1: Restore効果測定用）
    const std::atomic<uint64_t>* m_epochAdvanceCountRef_{nullptr};
    const std::atomic<uint64_t>* m_lastCompletedEpochRef_{nullptr};

    // [work39 Phase 7] Critical 出口 安定60秒継続追跡
    uint64_t m_criticalExitStableStartUs_{0};

    // [work37 Phase 9.2] Configuration Divergence 追跡
    MonitorState m_prevConfigDivergenceState_{MonitorState::Normal};
    uint64_t m_configDivergenceStartUs_{0};
    const std::atomic<uint64_t>* m_lastCommittedGenRef_{nullptr};
    const std::atomic<uint64_t>* m_requestedGenRef_{nullptr};

    // [work37 Phase 7.1] World Consistency 監視
    WorldConsistencyCheck m_worldConsistencyCheck_;

    // [work37 Phase 9.7] Snapshot Starvation 追跡
    MonitorState m_prevSnapshotStarvationState_{MonitorState::Normal};
    uint64_t m_snapshotStarvationStartUs_{0};

    // [work37 Phase 9.8] Pending Deployment 追跡
    MonitorState m_prevStructuralDeployState_{MonitorState::Normal};
    uint64_t m_structuralDeployStartUs_{0};
    const std::atomic<int>* m_requestedRebuildGenRef_{nullptr};
    const std::atomic<int>* m_committedRebuildGenRef_{nullptr};

    // [work37 Phase 9.29] Suppression Duration 追跡
    MonitorState m_prevSuppressionDurationState_{MonitorState::Normal};
    const std::atomic<uint64_t>* m_suppressionStartRef_{nullptr};

    // [work37 Phase 9.40] Runtime Progress Freeze 追跡
    MonitorState m_prevProgressFreezeState_{MonitorState::Normal};
    uint64_t m_progressFreezeStartUs_{0};
    uint64_t m_lastObservedPubSeq_{0};
    uint64_t m_lastObservedRetireTs_{0};
    const std::atomic<uint64_t>* m_lastRetireTimestampRef_{nullptr};
    const std::atomic<uint64_t>* m_publicationSequenceRef_{nullptr};

    // [work37 Phase 9.10 P2] Configuration Drift 追跡
    MonitorState m_prevConfigDriftState_{MonitorState::Normal};
    uint64_t m_configDriftStartUs_{0};
    const std::atomic<int>* m_manualOversamplingRef_{nullptr};

    // [work37 Phase 9.56 P2] RuntimeRecoveryScore 計算
    RuntimeRecoveryScore computeRuntimeRecoveryScore() const noexcept;

    // [work37 Phase 4.4] 背圧信号（drainDeferredRetireQueues から注入）
    // ★ Work39: CAS max update 版 + BackpressureWindow
    bool m_backpressureInjected_{false};
    std::atomic<uint64_t> m_maxFallbackSize_{0};
    std::atomic<double> m_maxOverflowRate_{0.0};

    // BackpressureWindow（問題F: peak + average + count の統計モデル）
    struct BackpressureWindow {
        uint64_t maxSize{0};
        uint64_t sumSize{0};
        uint32_t sampleCount{0};
        double   maxRate{0.0};
        double   sumRate{0.0};

        void record(std::size_t size, double rate) noexcept {
            if (size > maxSize) maxSize = size;
            maxRate = (rate > maxRate) ? rate : maxRate;
            sumSize += size;
            sumRate += rate;
            ++sampleCount;
        }

        [[nodiscard]] double averageSize() const noexcept {
            return sampleCount > 0
                ? static_cast<double>(sumSize) / sampleCount : 0.0;
        }

        [[nodiscard]] double averageRate() const noexcept {
            return sampleCount > 0 ? sumRate / sampleCount : 0.0;
        }

        void reset() noexcept {
            maxSize = 0; sumSize = 0; sampleCount = 0;
            maxRate = 0.0; sumRate = 0.0;
        }
    };
    BackpressureWindow m_backpressureWindow_;

    // [work37 Phase 8.2] EmergencyDrain 実行時制御
    std::atomic<bool> m_emergencyDrainRequested_{false};
};

} // namespace convo

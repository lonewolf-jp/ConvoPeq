#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include "AtomicAccess.h"

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
    void setMaxRetireAgeRef(const std::atomic<uint64_t>* ref) noexcept { m_maxRetireAgeRef = ref; }
    void setReaderSlotRef(const std::atomic<uint32_t>* ref) noexcept { m_readerSlotRef = ref; }
    void setOverflowCountRef(const std::atomic<uint64_t>* ref) noexcept { m_overflowCountRef = ref; }

    void tick() noexcept;

    // ★ C-4: HealthState のみ初期化（m_prev*State は維持 — イベント再通知防止）
    void reset() noexcept;

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
};

} // namespace convo

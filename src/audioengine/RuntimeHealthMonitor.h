#pragma once
#include <atomic>
#include <cstdint>
#include <functional>

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
}

// ★ イベントコード定数
static constexpr uint32_t EVENT_RETIRE_STALL         = 1001;
static constexpr uint32_t EVENT_RETIRE_STALL_WARNING = 1002;
static constexpr uint32_t EVENT_PUBLICATION_STALL    = 2001;
static constexpr uint32_t EVENT_PUBLICATION_WARNING  = 2002;
static constexpr uint32_t EVENT_READER_STUCK         = 3001; // ★ P4.5

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
    void setRetireRouter(isr::ISRRetireRouter* router) noexcept { m_retireRouter = router; }
    void setOrchestrator(isr::RuntimePublicationOrchestrator* orch) noexcept { m_orchestrator = orch; }
    void setRetireHighWatermarkRef(const std::atomic<int>* ref) noexcept { m_retireHighWatermarkRef = ref; }
    void setEventCallback(HealthEventCallback cb) noexcept { m_callback = std::move(cb); }

    void tick() noexcept;

private:
    void checkRetireStall() noexcept;
    void checkPublicationStall() noexcept;
    // ★ P4.5: Reader Stuck 診断（detectStuckReaders からの情報を HealthEvent に反映）
    void diagnoseRetireStall() noexcept;
    void emitOnTransition(MonitorState& currentState, MonitorState newState,
                          HealthEvent::Severity severity, uint32_t eventCode,
                          uint64_t value, uint32_t slot = 0) noexcept;

    isr::ISRRetireRouter* m_retireRouter = nullptr;
    isr::RuntimePublicationOrchestrator* m_orchestrator = nullptr;
    const std::atomic<int>* m_retireHighWatermarkRef = nullptr;
    HealthEventCallback m_callback;
    MonitorState m_prevRetireState { MonitorState::Normal };
    MonitorState m_prevPublicationState { MonitorState::Normal };
};

} // namespace convo

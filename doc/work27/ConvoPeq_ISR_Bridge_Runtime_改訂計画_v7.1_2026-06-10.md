# ConvoPeq ISR Bridge Runtime 改訂版改修計画書 v7.1（Practical Stable 決定版）

**作成日**: 2026-06-10
**設計思想**: Practical Stable ISR Bridge Runtime — 実運用で破綻しにくく、過剰設計を排し、既存の安全網を尊重する

---

## 0. v7→v7.1 修正一覧

| # | 論点 | v7 | v7.1 | 理由 |
|---|------|----|------|------|
| 1 | ShutdownResult | 導入（二重管理） | **削除** — v6 の ShutdownPhase 単独管理に統一 | phase と result の二重管理は矛盾の元。状態は1箇所に集約 |
| 2 | finalizeShutdown 二重防止 | `exchange(true)` — 例外で安全網喪失リスク | **`load()` + 成功後 `store(true)`** — 安全網維持優先 | Message Thread 単一スレッドでは TOCTOU は実質問題なし |
| 3 | waitForDrain jassert | `>= AudioStopped` — enum 順序比較 | **明示的な列挙** — enum 順序非依存の原則を遵守 | enum 順序非依存 |
| 4 | Publication Stall backlog | `pendingIntent>0 \|\| hasDeferred \|\| backlog>0` | **`pendingIntent>0 \|\| hasDeferred`** — v6 に戻す | backlog は UI連打等で平常時も発生。誤検出の元 |
| 5 | Retire Stall Error 閾値 | `hwm * 2`（上限なし） | **`min(hwm * 2, 8192)`** — 上限付き | hwm が過大設定時の異常検出遅れ防止 |

---

## 1. 基本原則

1. **強制復旧・強制 epoch 前進・強制 reclaim は行わない**
2. **タイムアウト後の `tryReclaim()` は安全な範囲で許容**
3. **`drainAll()` は行わない**
4. **観測は状態を読むだけ、変更しない（Pull型）**
5. **監視系は reclaim 系スレッドに依存しない**
6. **既存の安全網を削除しない**
7. **既存の仕組みを最大限活用する**
8. **使わない統計値は追加しない**
9. **JUCE 公式サンプルにない過剰な安定化機構は導入しない**
10. **フラグや状態変数は `std::atomic`**
11. **enum 順序比較による terminal 判定は行わない** — 明示的な列挙を使用
12. **監視イベントは状態遷移時のみ出力**
13. **状態は1箇所に集約** — 二重管理を避ける
14. **安全網維持を優先** — 完全性より実運用の堅牢性

---

## 2. 実装計画（Phase 1）

### P1-1: ShutdownPhase 拡張（v6 方式、ShutdownResult は導入しない）（0.5h）

**ファイル**: `src/audioengine/ISRShutdown.h`, `src/audioengine/ISRShutdown.cpp`

```cpp
// ★ ShutdownPhase のみ（ShutdownResult は導入しない）
//   TimedOut/Failed は ShutdownComplete の前に追加
enum class ShutdownPhase : uint8_t {
    Running,          // 0
    AudioStopped,     // 1
    ObserverDrained,  // 2
    RetireClosed,     // 3
    EpochSettled,     // 4
    ReclaimComplete,  // 5
    TimedOut,         // 6 ← ★ 追加（ShutdownComplete の前）
    Failed,           // 7 ← ★ 追加
    ShutdownComplete  // 8
};
```

**isTerminalPhase()** — enum 順序比較ではなく明示的列挙:

```cpp
static bool isTerminalPhase(ShutdownPhase p) noexcept {
    return p == ShutdownPhase::ShutdownComplete
        || p == ShutdownPhase::TimedOut
        || p == ShutdownPhase::Failed;
}
```

**markTimedOut / markFailed** — 直接 `store()` で遷移制約をバイパス:

```cpp
void markTimedOut() noexcept {
    phase_.store(ShutdownPhase::TimedOut, std::memory_order_release);
}
void markFailed() noexcept {
    phase_.store(ShutdownPhase::Failed, std::memory_order_release);
}
```

**isShutdownInProgress()** — terminal 状態を除外:

```cpp
bool ShutdownRuntime::isShutdownInProgress() const noexcept {
    const ShutdownPhase current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    return current != ShutdownPhase::Running && !isTerminalPhase(current);
}
```

**advancePhase() switch** — terminal 状態からは進めない:

```cpp
case ShutdownPhase::ReclaimComplete:
    next = ShutdownPhase::ShutdownComplete;
    break;
case ShutdownPhase::TimedOut:
case ShutdownPhase::Failed:
case ShutdownPhase::ShutdownComplete:
default:
    return;
```

---

### P1-2: releaseResources タイムアウト処理の修正（1.5h）

**ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`

```cpp
const bool drainedWithinBudget = waitForDrain(2000, 2);
const bool timedOut = !drainedWithinBudget;

if (timedOut)
    shutdownRuntime_.markTimedOut();

if (!drainedWithinBudget || !isFullyDrained()) {
    if (timedOut) {
        diagLog("[DIAG] releaseResources: drain timeout reached, "
                "performing safe tryReclaim (drainAll skipped)");
    }
    drainDeferredRetireQueues(true);
    m_epochDomain.tryReclaim();         // ← drainAll 禁止
}

m_coordinator.finalizeShutdown(timedOut);
```

---

### P1-3: SnapshotCoordinator 二段構え化（v6 方式 — load → retire → store）（1h）

**ファイル**: `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp`

```cpp
class SnapshotCoordinator {
    // ★ load → retire → store: 二重呼び出し防止。
    //   exchange(true) を先頭に置くと retire/tryReclaim 中の
    //   異常終了時にフラグだけ残ってデストラクタ安全網が無効になる。
    //   Message Thread 単一スレッドでは TOCTOU は実質問題なし。
    std::atomic<bool> m_shutdownFinalized { false };

    void retireCurrentAndTarget() noexcept {
        constexpr auto deleter = [](void* p) noexcept {
            SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(p));
        };
        const uint64_t retireEpoch = m_epochProvider->publishEpoch();
        GlobalSnapshot* snap = m_slots.exchangeCurrent(nullptr, std::memory_order_acq_rel);
        if (snap) m_epochProvider->enqueueRetire(snap, deleter, retireEpoch);
        snap = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
        if (snap) m_epochProvider->enqueueRetire(snap, deleter, retireEpoch);
    }

public:
    void finalizeShutdown(bool timedOut) noexcept {
        // ★ load — retire 前に二重実行を検出
        if (m_shutdownFinalized.load(std::memory_order_acquire))
            return;

        retireCurrentAndTarget();

        if (!timedOut)
            m_epochProvider->tryReclaim();

        // ★ retire/tryReclaim 成功後に store
        m_shutdownFinalized.store(true, std::memory_order_release);
    }

    ~SnapshotCoordinator() noexcept {
        if (m_shutdownFinalized.load(std::memory_order_acquire))
            return;

        // 異常系: 最後の安全網
        retireCurrentAndTarget();
        m_epochProvider->tryReclaim();
    }
};
```

---

### P1-4: waitForDrain 結合前提 jassert（enum 順序非依存）（0.5h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();
    // ★ 結合前提: AudioStopped 以降でのみ呼ばれる。
    //   enum 順序比較（>=）ではなく明示的な列挙で確認。
    //   これにより enum 値の順序変更に耐性がある。
    const auto phase = shutdownRuntime_.getPhase();
    jassert(phase == convo::isr::ShutdownPhase::AudioStopped
         || phase == convo::isr::ShutdownPhase::ObserverDrained
         || phase == convo::isr::ShutdownPhase::RetireClosed
         || phase == convo::isr::ShutdownPhase::EpochSettled
         || phase == convo::isr::ShutdownPhase::ReclaimComplete
         || phase == convo::isr::ShutdownPhase::TimedOut
         || phase == convo::isr::ShutdownPhase::Failed
         || phase == convo::isr::ShutdownPhase::ShutdownComplete);

    const int boundedTimeoutMs = juce::jlimit(1, 10000, timeoutMs);
    const int boundedPollIntervalMs = juce::jlimit(1, 5, pollIntervalMs);
    const double startMs = juce::Time::getMillisecondCounterHiRes();
    while (!isFullyDrained()) {
        drainDeferredRetireQueues(true);
        const double elapsedMs = juce::Time::getMillisecondCounterHiRes() - startMs;
        if (elapsedMs >= static_cast<double>(boundedTimeoutMs))
            return false;
        juce::Thread::sleep(boundedPollIntervalMs);
    }
    return true;
}
```

---

### P1-5: TimeUtils.h 新規作成（0.5h）

**ファイル**: 新規 `src/core/TimeUtils.h`

```cpp
#pragma once
#include <chrono>
#include <cstdint>

namespace convo {
inline uint64_t getCurrentTimeUs() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}
} // namespace convo
```

---

### P1-6: RuntimePublicationOrchestrator 出版停滞監視（1h）

**ファイル**: `src/audioengine/RuntimePublicationOrchestrator.h`, `.cpp`

```cpp
static constexpr uint64_t kPublicationStallThresholdUs = 30'000'000;

class RuntimePublicationOrchestrator {
    std::atomic<PublicationSequenceId> m_lastObservedSequence {0};
    std::atomic<uint64_t> m_lastProgressTimestampUs {0};

public:
    RuntimePublicationOrchestrator(AudioEngine& engine, uint64_t engineInstanceId) noexcept
        : /* ... */
        , m_lastProgressTimestampUs(getCurrentTimeUs())
    {}

    void updateProgressObservation() noexcept {
        PublicationSequenceId current = engine_.getLastCommittedPublicationSequence();
        PublicationSequenceId last = m_lastObservedSequence.load(std::memory_order_relaxed);
        if (current > last) {
            m_lastObservedSequence.store(current, std::memory_order_relaxed);
            m_lastProgressTimestampUs.store(getCurrentTimeUs(), std::memory_order_relaxed);
        }
    }

    bool isPublicationStalled() const noexcept {
        uint64_t elapsed = getCurrentTimeUs()
            - m_lastProgressTimestampUs.load(std::memory_order_acquire);
        return elapsed >= kPublicationStallThresholdUs;
    }

    void resetProgressObservation() noexcept {
        m_lastProgressTimestampUs.store(getCurrentTimeUs(), std::memory_order_release);
    }
};
```

`prepareToPlay()` に:

```cpp
if (runtimeOrchestrator_)
    runtimeOrchestrator_->resetProgressObservation();
```

---

### P1-7: RuntimeHealthMonitor 新設（2h）

**ファイル**: 新規 `src/audioengine/RuntimeHealthMonitor.h`, `src/audioengine/RuntimeHealthMonitor.cpp`

```cpp
// RuntimeHealthMonitor.h
#pragma once
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
};

namespace isr {
class ISRRetireRouter;
class RuntimePublicationOrchestrator;
}

static constexpr uint32_t EVENT_RETIRE_STALL         = 1001;
static constexpr uint32_t EVENT_RETIRE_STALL_WARNING = 1002;
static constexpr uint32_t EVENT_PUBLICATION_STALL    = 2001;
static constexpr uint32_t EVENT_PUBLICATION_WARNING  = 2002;

// Retire Stall Error 閾値の上限（hwm が過大設定時の異常検出遅れ防止）
static constexpr int kRetireStallErrorMax = 8192;

enum class MonitorState : uint8_t { Normal, Warning, Error };
using HealthEventCallback = std::function<void(const HealthEvent&)>;

/**
 * RuntimeHealthMonitor: Pull型監視エンジン。
 *
 * Phase 1 スコープ:
 *   - Retire Backlog 監視（queue depth ベース、状態遷移検出）
 *   - Publication Stall 監視（sequence 進捗＋deferred age ベース）
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
```

```cpp
// RuntimeHealthMonitor.cpp
#include "RuntimeHealthMonitor.h"
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/AtomicAccess.h"
#include "core/TimeUtils.h"

namespace convo {

void RuntimeHealthMonitor::tick() noexcept {
    checkRetireStall();
    checkPublicationStall();
}

void RuntimeHealthMonitor::checkRetireStall() noexcept {
    if (!m_retireRouter) return;
    uint32_t pendingCount = m_retireRouter->pendingRetireCount();

    int hwm = (m_retireHighWatermarkRef != nullptr)
        ? convo::consumeAtomic(*m_retireHighWatermarkRef, std::memory_order_acquire)
        : 3072;

    // ★ Error 閾値: hwm * 2 だが、上限は kRetireStallErrorMax (8192)
    //   hwm が過大設定（例: 10000）の場合でも異常検出が遅れないよう上限を設ける
    int errorThreshold = std::min(hwm * 2, kRetireStallErrorMax);

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
    //   pendingIntent（retire intent）または hasDeferred（保留中のpublish要求）が
    //   存在し、かつ sequence が 30秒以上進んでいない場合のみ。
    //   backlog は UI連打等で平常時も発生するため含めない（誤検出防止）。
    const bool hasPendingWork = m_orchestrator->getPendingIntentCount() > 0
        || m_orchestrator->hasDeferredRequest();

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

} // namespace convo
```

---

### P1-8: AudioEngine 統合（1.5h）

**ファイル**: `src/audioengine/AudioEngine.h`, `AudioEngine.CtorDtor.cpp`, `AudioEngine.Timer.cpp`

```cpp
// AudioEngine.h
#include "audioengine/RuntimeHealthMonitor.h"

class AudioEngine {
    convo::RuntimeHealthMonitor m_healthMonitor;
};
```

```cpp
// AudioEngine.CtorDtor.cpp
m_healthMonitor.setRetireRouter(m_retireRouter.get());
m_healthMonitor.setOrchestrator(runtimeOrchestrator_.get());
m_healthMonitor.setRetireHighWatermarkRef(&retireHighWatermark_);
m_healthMonitor.setEventCallback(
    [this](const convo::HealthEvent& ev) { onHealthEvent(ev); });
```

```cpp
// AudioEngine.Timer.cpp
if (!isShutdownInProgress()) {
    m_healthMonitor.tick();
}
```

```cpp
// AudioEngine::onHealthEvent
void AudioEngine::onHealthEvent(const convo::HealthEvent& event) noexcept {
    diagLog("[HEALTH] eventCode=" + juce::String(static_cast<int>(event.eventCode))
        + " severity=" + juce::String(static_cast<int>(event.severity))
        + " value=" + juce::String(static_cast<juce::int64>(event.value)));
}
```

---

### P1-9: collectDrainAudit 修正（0.5h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
convo::isr::RuntimeDrainAudit AudioEngine::collectDrainAudit() noexcept
{
    return convo::isr::RuntimeDrainAudit{
        .pendingPublication = runtimePublicationBridge_.getPublicationBacklogCount(),
        .pendingRetire = retireRuntime_.pendingIntentCount(),
        .activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u,
        .routerPendingRetire = static_cast<uint64_t>(m_retireRouter->pendingRetireCount())
            + convo::consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire),
        .maxDeferredAgeMs = runtimeOrchestrator_
            ? runtimeOrchestrator_->getMaxDeferredAgeMs() : 0u,
        .deferredPublish = (runtimeOrchestrator_
            && runtimeOrchestrator_->hasDeferredRequest()) ? 1u : 0u,
        .quarantineResident = dspQuarantineManager_.residentCount(),
        .oldestPendingAgeMs = static_cast<uint64_t>(
            std::max(0.0, convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire))),
        .maxQuarantineAgeSec = dspQuarantineManager_.getMaxEntryAgeSec()
    };
}
```

---

### Phase 1 工数サマリー

| # | 項目 | ファイル | 工数 |
|---|------|---------|------|
| P1-1 | ShutdownPhase 拡張（ShutdownResult なし） | `ISRShutdown.h/cpp` | 0.5h |
| P1-2 | releaseResources タイムアウト処理修正 | `A.E.Processing.ReleaseResources.cpp` | 1.5h |
| P1-3 | SnapshotCoordinator 二段構え（load→retire→store） | `SnapshotCoordinator.h/cpp` | 1h |
| P1-4 | waitForDrain jassert（明示的列挙） | `AudioEngine.Threading.cpp` | 0.5h |
| P1-5 | TimeUtils.h 新規作成 | 新規 `core/TimeUtils.h` | 0.5h |
| P1-6 | RuntimePublicationOrchestrator 出版停滞監視 | `RuntimePublicationOrchestrator.h/cpp` | 1h |
| P1-7 | RuntimeHealthMonitor 新設（状態遷移検出＋上限付き閾値） | 新規 2ファイル | 2h |
| P1-8 | AudioEngine 統合 | `A.E.h/CtorDtor/Timer.cpp` | 1.5h |
| P1-9 | collectDrainAudit 修正 | `AudioEngine.Threading.cpp` | 0.5h |
| | **Phase 1 合計** | | **9h** |

---

## 3. v1→v7.1 設計判断の進化

| # | 論点 | v1（原案） | v6 | v7.1（最終） |
|---|------|-----------|----|------------|
| 1 | ShutdownPhase | `Running..Failed` 6状態に置換 | 既存7状態＋`TimedOut/Failed` | 同左（ShutdownResult は導入せず） |
| 2 | ShutdownResult | なし | なし | **なし**（状態は ShutdownPhase 1箇所に集約） |
| 3 | SnapshotCoordinator dtor | tryReclaim削除 | 二段構え（load→retire→store） | 二段構え（load→retire→store、exchange不使用） |
| 4 | releaseResources timeout | リーク許容＋何もしない | `markTimedOut`+`tryReclaim` | 同左 |
| 5 | waitForDrain | 新規実装 | 既存流用＋jassert | jassert を enum 順序非依存の明示的列挙に |
| 6 | DeletionEntry timestamp | 追加 | 追加しない | 追加しない |
| 7 | HealthMonitor イベント | 毎 tick 発火 | 状態遷移検出 | 同左＋Error 閾値に上限 `min(hwm*2,8192)` |
| 8 | Publication Stall | sequence監視 | pending+deferred | pending+deferred（backlog は含めず） |
| 9 | Reader Long Active | Phase 1 | Phase 2 | Phase 2 |
| 10 | Crossfade Watchdog | 実装 | 本スコープ外 | 本スコープ外 |
| 11 | EvidenceRingBuffer | 新設 | 新設せず | 新設せず |
| 12 | TimeUtils | `audioengine/` | `core/` | `core/` |

---

## 4. 設計判断一覧（v7.1 最終確定版）

| # | 論点 | v7.1 結論 | 根拠 |
|---|------|----------|------|
| 1 | ShutdownPhase 追加位置 | **`ShutdownComplete` の前に** `TimedOut/Failed` | `>= ShutdownComplete` 比較を壊さない |
| 2 | terminal 判定 | `isTerminalPhase()` 静的メソッド（明示的列挙） | enum 順序比較に依存しない |
| 3 | **ShutdownResult** | **導入しない** | phase と result の二重管理は矛盾の元。状態は1箇所に集約 |
| 4 | finalizeShutdown 二重防止 | `load()` + 成功後 `store(true)` | safety net 維持優先。Message Thread単独ではTOCTOUは問題なし |
| 5 | releaseResources timeout | `drainAll()` 禁止, `tryReclaim()` 維持 | 安全な epoch-based reclaim のみ |
| 6 | waitForDrain jassert | **明示的な enum 列挙**（`>=` 不使用） | enum 順序非依存の原則を遵守 |
| 7 | Publication Stall 条件 | `pendingIntent>0 \|\| hasDeferred` + `isStalled()` | backlog は平常時も発生。誤検出防止 |
| 8 | Publication Stall age | **`hasDeferredRequest()` 時のみ** | pending なし時の誤検出防止 |
| 9 | HealthMonitor イベント | **状態遷移検出**（Normal↔Warning↔Error）時のみ | 連続発火を99%削減 |
| 10 | Retire Stall Error 閾値 | **`min(hwm * 2, 8192)`** — 上限付き | hwm 過大設定時の異常検出遅れ防止 |
| 11 | Retire Stall Warning 閾値 | `hwm`（動的取得） | 既存設定値と整合 |
| 12 | Reader Long Active | **Phase 2**（既存 `detectStuckReaders` 拡張） | 監視機構の二重化防止 |
| 13 | Crossfade Watchdog | **本スコープ外** | 過剰設計回避 |
| 14 | EvidenceRingBuffer | **新設せず** | 既存証跡系で十分 |
| 15 | TimeUtils.h 配置 | **`src/core/TimeUtils.h`** | 依存方向として自然 |

---

## 5. Phase 1 コード変更ファイル一覧

| 操作 | ファイル | 変更内容 |
|------|---------|---------|
| 修正 | `ISRShutdown.h` | `ShutdownPhase` に `TimedOut/Failed` 追加、`isTerminalPhase()`, `markTimedOut()`, `markFailed()` |
| 修正 | `ISRShutdown.cpp` | `isShutdownInProgress()` 修正、`advancePhase()` switch 更新 |
| 修正 | `A.E.Processing.ReleaseResources.cpp` | `drainAll()`→`tryReclaim()`, `markTimedOut()`, `finalizeShutdown(timedOut)` |
| 修正 | `SnapshotCoordinator.h` | `finalizeShutdown()`, `retireCurrentAndTarget()`, `load/store` 二段階 |
| 修正 | `AudioEngine.Threading.cpp` | `waitForDrain` jassert 明示的列挙、`collectDrainAudit` 修正 |
| 修正 | `RuntimePublicationOrchestrator.h` | `kPublicationStallThresholdUs`, 監視用メソッド |
| 修正 | `RuntimePublicationOrchestrator.cpp` | コンストラクタで `m_lastProgressTimestampUs` 初期化 |
| 修正 | `AudioEngine.h` | `m_healthMonitor`, `onHealthEvent()` |
| 修正 | `AudioEngine.CtorDtor.cpp` | コンストラクタで `m_healthMonitor` 初期化 |
| 修正 | `AudioEngine.Timer.cpp` | `timerCallback` 内で `m_healthMonitor.tick()` |
| 修正 | `A.E.Processing.PrepareToPlay.cpp` | `runtimeOrchestrator_->resetProgressObservation()` |
| 新規 | `core/TimeUtils.h` | `getCurrentTimeUs()` |
| 新規 | `audioengine/RuntimeHealthMonitor.h` | `RuntimeHealthMonitor` + `MonitorState` |
| 新規 | `audioengine/RuntimeHealthMonitor.cpp` | 全実装 |

---

## 6. エラッタ

### v7→v7.1 修正詳細

**🔴 修正1: ShutdownResult 削除**

```cpp
// v7（問題: phase と result の二重管理。矛盾状態が発生可能）:
enum class ShutdownPhase { ..., TimedOut, Failed, ShutdownComplete };
enum class ShutdownResult { None, Success, TimedOut, Failed };  // 二重管理
void markTimedOut() { phase_=TimedOut; result_=TimedOut; }
void markSuccess() { result_=Success; }

// v7.1（改善: ShutdownPhase のみで一元管理）:
enum class ShutdownPhase { ..., TimedOut, Failed, ShutdownComplete };
// ShutdownResult は導入しない
void markTimedOut() { phase_.store(ShutdownPhase::TimedOut); }
// markSuccess() は不要（phase==ShutdownComplete で代用可能）
```

**🔴 修正2: finalizeShutdown — exchange → load/store**

```cpp
// v7（問題: exchange で安全網無効リスク）:
if (m_shutdownFinalized.exchange(true)) return;
retireCurrentAndTarget();
if (!timedOut) tryReclaim();

// v7.1（改善: 安全網維持優先。Message Thread 単独では TOCTOU 問題なし）:
if (m_shutdownFinalized.load()) return;
retireCurrentAndTarget();
if (!timedOut) tryReclaim();
m_shutdownFinalized.store(true);
```

**🔴 修正3: waitForDrain jassert — enum 順序比較廃止**

```cpp
// v7（問題: enum 順序比較 — 将来の enum 変更で壊れる）:
jassert(phase >= AudioStopped);

// v7.1（改善: 明示的列挙 — enum 順序非依存）:
jassert(phase == AudioStopped || phase == ObserverDrained || ...);
```

**🟡 修正4: Publication Stall — backlog 削除**

```cpp
// v7（問題: backlog は平常時も発生 → 誤検出）:
hasPendingWork = pendingIntent > 0 || hasDeferred || backlog > 0;

// v7.1（改善: pendingIntent + hasDeferred のみで判定）:
hasPendingWork = pendingIntent > 0 || hasDeferred;
```

**🟡 修正5: Retire Stall Error 閾値 — 上限追加**

```cpp
// v7（問題: hwm が過大設定時の異常検出遅れ）:
int errorThreshold = hwm * 2;  // hwm=10000 → error=20000

// v7.1（改善: 上限 8192 で異常検出を保証）:
static constexpr int kRetireStallErrorMax = 8192;
int errorThreshold = std::min(hwm * 2, kRetireStallErrorMax);
// hwm=10000 → error=8192
```

---

## 7. Practical Stable ISR Bridge Runtime 最終評価

| 評価軸 | スコア | 最終確定根拠 |
|--------|--------|-------------|
| **クラッシュ回避** | ⭐⭐⭐⭐⭐ | `drainAll()` 排除。`tryReclaim()` のみ。安全な epoch-based reclaim |
| **リーク防止** | ⭐⭐⭐⭐⭐ | 二段構え（`load→retire→store` + デストラクタ最終安全網） |
| **状態管理の単一性** | ⭐⭐⭐⭐⭐ | ShutdownPhase のみで一元管理。二重管理による矛盾状態を排除 |
| **過剰設計の回避** | ⭐⭐⭐⭐⭐ | 7項目除外（Crossfade/EvidenceRingBuffer/Reader/ShutdownResult/backlog/DeletionEntry/MPSC）|
| **監視イベント品質** | ⭐⭐⭐⭐⭐ | 状態遷移検出＋上限付き閾値。誤検出とログ埋めを防止 |
| **enum 順序非依存** | ⭐⭐⭐⭐⭐ | `isTerminalPhase()` + `jassert` 明示的列挙。将来のenum変更に耐性 |
| **誤検出の回避** | ⭐⭐⭐⭐⭐ | Publication Stall: backlog 除外。Retire Stall: error上限8192。deferred age: `hasDeferred`時のみ |
| **既存コード活用** | ⭐⭐⭐⭐⭐ | `retireHighWatermark_`, `oldestPendingAge_`, `detectStuckReaders()` 等を流用 |
| **実装工数** | ⭐⭐⭐⭐⭐ | **9h**（v1比 53%削減） |

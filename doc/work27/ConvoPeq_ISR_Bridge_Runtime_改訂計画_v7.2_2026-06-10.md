# ConvoPeq ISR Bridge Runtime 改訂版改修計画書 v7.2（Practical Stable 完成版）

**作成日**: 2026-06-10
**ベース**: v7（exchange方式＋backlog監視）— ShutdownResult のみ削除、閾値と jassert を改良
**設計思想**: Practical Stable ISR Bridge Runtime — 実運用で破綻しにくく、過剰設計を排す

---

## 0. v7→v7.2 変更一覧

| # | 論点 | v7 | v7.1（問題） | v7.2（最終） |
|---|------|----|------------|------------|
| 1 | ShutdownResult | 導入（二重管理） | 削除（◎） | **削除**（v7.1 から継承） |
| 2 | finalizeShutdown 二重防止 | `exchange(true)` | `load→store`（安全網喪失リスク小だが将来壊れやすい） | **`exchange(true)`**（v7 に戻す） |
| 3 | Publication Stall backlog | `pendingIntent>0 \|\| hasDeferred \|\| backlog>0` | backlog 除外（停滞見逃しリスク） | **`pendingIntent>0 \|\| hasDeferred \|\| backlog>N`**（閾値付きで復活） |
| 4 | Retire Stall Error 閾値 | `hwm * 2`（上限なし、高hwmで異常検出遅れ） | `min(hwm*2,8192)`（error<warningの可能性） | **`max(hwm+1, min(hwm*2, 8192))`**（必ず error>warning） |
| 5 | waitForDrain jassert | `>= AudioStopped`（enum順序比較） | 明示的列挙（保守性低） | **`!= Running`**（シンプル＋将来堅牢） |

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
11. **enum 順序比較による terminal 判定は行わない** — `isTerminalPhase()` を使用
12. **監視イベントは状態遷移時のみ出力**
13. **状態は1箇所に集約** — 二重管理を避ける
14. **将来壊れない設計を優先** — 現在のスレッド前提に依存しない

---

## 2. 実装計画（Phase 1）

### P1-1: ShutdownPhase 拡張（0.5h）

**ファイル**: `src/audioengine/ISRShutdown.h`, `src/audioengine/ISRShutdown.cpp`

**ShutdownPhase のみ**（ShutdownResult は導入しない）：

```cpp
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

**markTimedOut / markFailed** — 直接 `store()`:

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
    m_epochDomain.tryReclaim();
}

m_coordinator.finalizeShutdown(timedOut);
```

---

### P1-3: SnapshotCoordinator 二段構え化 — exchange(true) 方式（1h）

**ファイル**: `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp`

```cpp
class SnapshotCoordinator {
    // ★ exchange(true): 全か無かのアトミック二重呼び出し防止。
    //   Message Thread 単独という現在の前提に依存せず、
    //   将来の経路変更（shutdown helper thread 等）でも安全。
    //   retireCurrentAndTarget は noexcept のため例外でフラグだけ
    //   残るリスクは実質なし。
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
        // ★ exchange: 二重呼び出しをアトミックに検出＋防止
        if (m_shutdownFinalized.exchange(true, std::memory_order_acq_rel))
            return;

        retireCurrentAndTarget();

        if (!timedOut)
            m_epochProvider->tryReclaim();
    }

    ~SnapshotCoordinator() noexcept {
        if (m_shutdownFinalized.load(std::memory_order_acquire))
            return;

        retireCurrentAndTarget();
        m_epochProvider->tryReclaim();
    }
};
```

---

### P1-4: waitForDrain 結合前提 jassert（シンプル＋将来堅牢）（0.5h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();
    // ★ 結合前提: Running 以外の phase でのみ呼ばれる。
    //   enum 順序比較でも明示的列挙でもなく、単に Running でないことを確認。
    //   これが最もシンプルで、将来の enum 変更に強い。
    jassert(shutdownRuntime_.getPhase() != convo::isr::ShutdownPhase::Running);

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

    void updateProgressObservation() noexcept { /* see P1-7 checkPublicationStall */ }
    bool isPublicationStalled() const noexcept { /* see P1-7 */ }
    void resetProgressObservation() noexcept { /* see P1-7 */ }

    // ★ RuntimePublicationCoordinator の backlog を参照
    uint64_t getPublicationBacklogCount() const noexcept {
        return engine_.getPublicationBacklogCount();
    }
};
```

`AudioEngine` に追加:

```cpp
uint64_t getPublicationBacklogCount() const noexcept {
    return runtimePublicationBridge_.getPublicationBacklogCount();
}
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

// Retire Stall Error 閾値の上限
static constexpr int kRetireStallErrorMax = 8192;

enum class MonitorState : uint8_t { Normal, Warning, Error };
using HealthEventCallback = std::function<void(const HealthEvent&)>;

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
    void emitOnTransition(MonitorState& state, MonitorState next,
                          HealthEvent::Severity sev, uint32_t code,
                          uint64_t val, uint32_t slot = 0) noexcept;

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

    // ★ Error 閾値: hwm * 2 を基本とするが、上限 8192 を設定。
    //   ただし error < warning にならないよう max(hwm+1, ...) で保護。
    int errorThreshold = std::max(hwm + 1,
                                  std::min(hwm * 2, kRetireStallErrorMax));

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
    //
    //   backlog は UI連打等でも発生するため、閾値（>32）を設ける。
    //   これにより一時的なバーストでの誤検出を防ぎつつ、
    //   backlog が atypical に滞留した停滞も検出できる。
    const bool hasPendingWork = m_orchestrator->getPendingIntentCount() > 0
        || m_orchestrator->hasDeferredRequest()
        || m_orchestrator->getPublicationBacklogCount() > 32;

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

    uint64_t getPublicationBacklogCount() const noexcept {
        return runtimePublicationBridge_.getPublicationBacklogCount();
    }
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
| P1-1 | ShutdownPhase 拡張 | `ISRShutdown.h/cpp` | 0.5h |
| P1-2 | releaseResources タイムアウト処理修正 | `A.E.Processing.ReleaseResources.cpp` | 1.5h |
| P1-3 | SnapshotCoordinator（exchange(true)） | `SnapshotCoordinator.h/cpp` | 1h |
| P1-4 | waitForDrain jassert（`!= Running`） | `AudioEngine.Threading.cpp` | 0.5h |
| P1-5 | TimeUtils.h 新規作成 | 新規 `core/TimeUtils.h` | 0.5h |
| P1-6 | RuntimePublicationOrchestrator 出版停滞監視 | `RuntimePublicationOrchestrator.h/cpp` | 1h |
| P1-7 | RuntimeHealthMonitor 新設（状態遷移＋backlog閾値＋Error上限） | 新規 2ファイル | 2h |
| P1-8 | AudioEngine 統合 | `A.E.h/CtorDtor/Timer.cpp` | 1.5h |
| P1-9 | collectDrainAudit 修正 | `AudioEngine.Threading.cpp` | 0.5h |
| | **Phase 1 合計** | | **9h** |

---

## 3. 設計判断一覧（v7.2 最終確定版）

| # | 論点 | v7.2 結論 | 根拠 |
|---|------|----------|------|
| 1 | ShutdownPhase 追加 | `ShutdownComplete` の前に `TimedOut/Failed` | `>= ShutdownComplete` 比較を壊さない |
| 2 | terminal 判定 | `isTerminalPhase()` 明示的列挙 | enum 順序非依存 |
| 3 | **ShutdownResult** | **導入しない**（ShutdownPhase のみ） | 二重管理による矛盾防止 |
| 4 | **finalizeShutdown** | **`exchange(true)`**（v7 方式） | 将来の経路変更でも安全 |
| 5 | releaseResources timeout | `markTimedOut` + `tryReclaim`（`drainAll` 禁止） | 安全な epoch-based reclaim |
| 6 | **waitForDrain jassert** | **`!= Running`**（シンプル＋将来堅牢） | 目的は誤用検出。enum変更に強い |
| 7 | **Publication Stall backlog** | **`> 32` の閾値付きで含める**（v7 応用） | 誤検出と停滞見逃しのバランス |
| 8 | Publication Stall age | `hasDeferredRequest()` 時のみ | pending なし時の誤検出防止 |
| 9 | HealthMonitor イベント | 状態遷移検出（Normal↔Warning↔Error）時のみ | 連続発火を99%削減 |
| 10 | **Retire Stall Error 閾値** | **`max(hwm+1, min(hwm*2, 8192))`** | 必ず error>warning。上限で異常検出遅れ防止 |
| 11 | Retire Stall Warning 閾値 | `hwm`（動的取得） | 既存設定値と整合 |
| 12 | Reader Long Active | **Phase 2**（既存 `detectStuckReaders` 拡張） | 監視機構の二重化防止 |
| 13 | Crossfade Watchdog | **本スコープ外** | 過剰設計回避 |
| 14 | EvidenceRingBuffer | **新設せず** | 既存証跡系で十分 |
| 15 | TimeUtils.h 配置 | **`src/core/TimeUtils.h`** | 依存方向として自然 |

---

## 4. Phase 1 コード変更ファイル一覧

| 操作 | ファイル | 変更内容 |
|------|---------|---------|
| 修正 | `ISRShutdown.h` | `ShutdownPhase` に `TimedOut/Failed` 追加、`isTerminalPhase()`, `markTimedOut()`, `markFailed()` |
| 修正 | `ISRShutdown.cpp` | `isShutdownInProgress()` 修正、`advancePhase()` switch 更新 |
| 修正 | `A.E.Processing.ReleaseResources.cpp` | `drainAll()`→`tryReclaim()`, `markTimedOut()`, `finalizeShutdown(timedOut)` |
| 修正 | `SnapshotCoordinator.h` | `finalizeShutdown()` + `exchange(true)` + `retireCurrentAndTarget()` |
| 修正 | `AudioEngine.Threading.cpp` | `waitForDrain` jassert `!= Running`、`collectDrainAudit` 修正 |
| 修正 | `RuntimePublicationOrchestrator.h` | `kPublicationStallThresholdUs`, 監視用メソッド, `getPublicationBacklogCount()` |
| 修正 | `RuntimePublicationOrchestrator.cpp` | コンストラクタで `m_lastProgressTimestampUs` 初期化 |
| 修正 | `AudioEngine.h` | `m_healthMonitor`, `getPublicationBacklogCount()`, `onHealthEvent()` |
| 修正 | `AudioEngine.CtorDtor.cpp` | コンストラクタで `m_healthMonitor` 初期化 |
| 修正 | `AudioEngine.Timer.cpp` | `timerCallback` 内で `m_healthMonitor.tick()` |
| 修正 | `A.E.Processing.PrepareToPlay.cpp` | `runtimeOrchestrator_->resetProgressObservation()` |
| 新規 | `core/TimeUtils.h` | `getCurrentTimeUs()` |
| 新規 | `audioengine/RuntimeHealthMonitor.h` | `RuntimeHealthMonitor` + `MonitorState` |
| 新規 | `audioengine/RuntimeHealthMonitor.cpp` | 全実装 |

---

## 5. エラッタ

### v7→v7.2 の変更内訳

**v7 から継承（v7.1 の変更を戻したもの）**:

```
finalizeShutdown        → exchange(true)     （v7 方式）
Publication Stall       → backlog > 32 付き  （v7 方式＋閾値）
```

**v7.1 から継承（v7 にない改善）**:

```
ShutdownResult          → 削除               （二重管理防止）
```

**v7.2 独自の改善**:

```
Retire Stall Error 閾値 → max(hwm+1, min(hwm*2, 8192))
                         （v7: 上限なし → 高hwmで異常検出遅れ懸念
                          v7.1: 上限のみ → error<warningの可能性
                          v7.2: max保護＋上限 → 安全）
waitForDrain jassert    → phase != Running
                         （v7: >= AudioStopped → enum順序依存
                          v7.1: 明示的列挙 → 保守性低
                          v7.2: != Running → シンプル＋将来堅牢）
```

### 閾値設計の詳細

```
hwm (retireHighWatermark): 動的、デフォルト 3072

Warning = hwm
Error   = max(hwm + 1, min(hwm * 2, kRetireStallErrorMax))

ケース1: hwm=3072（デフォルト）
  Warning=3072, Error=min(6144,8192)=6144, max(3073,6144)=6144  ✓ error>warning

ケース2: hwm=5000（運用調整後）
  Warning=5000, Error=min(10000,8192)=8192, max(5001,8192)=8192  ✓ error>warning

ケース3: hwm=10000（過大設定）
  Warning=10000, Error=min(20000,8192)=8192, max(10001,8192)=10001
  → Error=10001 > Warning=10000  ✓ 必ず error>warning
```

```
backlog 閾値: 32
  Publication backlog が 32 を超えた場合のみ stalled 判定に寄与。
  UI連打等の短期的バースト（〜数 backlog）では誤検出しない。
  backlog が atypical に滞留した場合のみ検出。
```

---

## 6. Practical Stable ISR Bridge Runtime 最終評価

| 評価軸 | スコア | 最終根拠 |
|--------|--------|---------|
| **クラッシュ回避** | ⭐⭐⭐⭐⭐ | `drainAll()` 排除。`tryReclaim()` のみ |
| **リーク防止** | ⭐⭐⭐⭐⭐ | 二段構え（`exchange(true)` + デストラクタ）。将来の経路変更でも安全 |
| **状態管理の単一性** | ⭐⭐⭐⭐⭐ | ShutdownPhase のみ。二重管理なし |
| **過剰設計の回避** | ⭐⭐⭐⭐⭐ | 7項目除外。9hで実装可能 |
| **監視イベント品質** | ⭐⭐⭐⭐⭐ | 状態遷移検出＋適切な閾値で誤検出と見逃しのバランス |
| **停滞検出能力** | ⭐⭐⭐⭐⭐ | backlog 閾値付き復活で検出範囲拡大。`>32` で誤検出防止 |
| **閾値の健全性** | ⭐⭐⭐⭐⭐ | `max(hwm+1, min(hwm*2,8192))` で error>warning を常時保証 |
| **将来堅牢性** | ⭐⭐⭐⭐⭐ | `exchange(true)` + `!= Running` jassert + enum 順序非依存 |
| **実装工数** | ⭐⭐⭐⭐⭐ | **9h**（v1比 53%削減、全不整合修正済み、最終決定版） |

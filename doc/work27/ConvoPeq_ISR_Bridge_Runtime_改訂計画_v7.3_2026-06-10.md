# ConvoPeq ISR Bridge Runtime 改訂版改修計画書 v7.3（Practical Stable 完成版）

**作成日**: 2026-06-10
**設計思想**: Practical Stable ISR Bridge Runtime — 実運用で破綻しにくく、過剰設計を排す

---

## 0. v7.2→v7.3 修正一覧

| # | 論点 | v7.2 | v7.3 | 優先度 |
|---|------|------|------|--------|
| 1 | SnapshotCoordinator 二重防止 | `exchange(true)` — 異常時安全網喪失リスク | **`load()` + 成功後 `store(true)`** | 🔴 必須 |
| 2 | Publication Stall backlog | `> 32` — 根拠不足の閾値 | **`> 0`** — v7 方式（混雑と停滞を区別せず、どちらも異常として検出） | 🔴 必須 |
| 3 | Retire Stall Error 閾値 | `max(hwm+1, min(hwm*2,8192))` — hwm=10000でwarning≒error | **`max(hwm + hwm/2, hwm + 1)`** — 常に有意な gap | 🟡 推奨 |
| 4 | waitForDrain jassert | `!= Running` — 緩すぎ | **明示的列挙（v7.1 方式）** + 保守コメント | 🔴 必須 |
| 5 | markTimedOut 時の phase 消失 | 上書きで停止位置が不明 | **`lastNonTerminalPhase` 追跡追加** | 🟡 推奨 |
| 6 | HealthMonitor callback | `std::function` — 将来の allocation 懸念 | 設計メモ追加（Phase 1 では現状維持） | 🔵 任意 |

---

## 1. 基本原則

1. **強制復旧・強制 epoch 前進・強制 reclaim は行わない**
2. **タイムアウト後の `tryReclaim()` は安全な範囲で許容**
3. **`drainAll()` は行わない**
4. **観測は状態を読むだけ、変更しない（Pull型）**
5. **監視系は reclaim 系スレッドに依存しない**
6. **既存の安全網を削除しない** — 安全網維持を最優先
7. **既存の仕組みを最大限活用する**
8. **使わない統計値は追加しない**
9. **JUCE 公式サンプルにない過剰な安定化機構は導入しない**
10. **フラグや状態変数は `std::atomic`**
11. **enum 順序比較による terminal 判定は行わない** — `isTerminalPhase()` を使用
12. **監視イベントは状態遷移時のみ出力**
13. **状態は1箇所に集約** — 二重管理を避ける
14. **例外安全より安全網維持を優先** — 異常時はリークしてでも安全網を残す

---

## 2. 実装計画（Phase 1）

### P1-1: ShutdownPhase 拡張 + lastNonTerminalPhase 追跡（1h）

**ファイル**: `src/audioengine/ISRShutdown.h`, `src/audioengine/ISRShutdown.cpp`

```cpp
enum class ShutdownPhase : uint8_t {
    Running,          // 0
    AudioStopped,     // 1
    ObserverDrained,  // 2
    RetireClosed,     // 3
    EpochSettled,     // 4
    ReclaimComplete,  // 5
    TimedOut,         // 6 ← 追加
    Failed,           // 7 ← 追加
    ShutdownComplete  // 8
};

class ShutdownRuntime {
    std::atomic<ShutdownPhase> phase_{ShutdownPhase::Running};
    // ★ 追加: TimedOut/Failed 上書き前の最終フェーズを記録
    //   障害解析で「どの段階で停止したか」を特定するために使用
    std::atomic<ShutdownPhase> lastNonTerminalPhase_{ShutdownPhase::Running};
    // ... 既存メンバ ...

public:
    static bool isTerminalPhase(ShutdownPhase p) noexcept {
        return p == ShutdownPhase::ShutdownComplete
            || p == ShutdownPhase::TimedOut
            || p == ShutdownPhase::Failed;
    }

    ShutdownPhase getPhase() const noexcept { /* ...既存... */ }
    ShutdownPhase getLastNonTerminalPhase() const noexcept {  // ★ 追加
        return convo::consumeAtomic(lastNonTerminalPhase_, std::memory_order_acquire);
    }

    void markTimedOut() noexcept {
        // ★ 上書き前に現在の phase を保存
        convo::publishAtomic(lastNonTerminalPhase_,
                             convo::consumeAtomic(phase_, std::memory_order_acquire),
                             std::memory_order_release);
        phase_.store(ShutdownPhase::TimedOut, std::memory_order_release);
    }
    void markFailed() noexcept {
        convo::publishAtomic(lastNonTerminalPhase_,
                             convo::consumeAtomic(phase_, std::memory_order_acquire),
                             std::memory_order_release);
        phase_.store(ShutdownPhase::Failed, std::memory_order_release);
    }

    // ★ transitionTo でも記録（通常遷移用）
    bool transitionTo(ShutdownPhase target) noexcept {
        // ...既存の遷移チェック...
        convo::publishAtomic(lastNonTerminalPhase_, phase_.load(), std::memory_order_release);
        convo::publishAtomic(phase_, target, std::memory_order_release);
        return true;
    }
};
```

**isShutdownInProgress()**:

```cpp
bool ShutdownRuntime::isShutdownInProgress() const noexcept {
    const ShutdownPhase current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    return current != ShutdownPhase::Running && !isTerminalPhase(current);
}
```

**advancePhase() switch**:

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

**emitShutdownTrace() switch の修正** — `TimedOut` と `Failed` の case を追加:

```cpp
switch (phase) {
case ShutdownPhase::Running: phaseName = "Running"; break;
case ShutdownPhase::AudioStopped: phaseName = "AudioStopped"; break;
case ShutdownPhase::ObserverDrained: phaseName = "ObserverDrained"; break;
case ShutdownPhase::RetireClosed: phaseName = "RetireClosed"; break;
case ShutdownPhase::EpochSettled: phaseName = "EpochSettled"; break;
case ShutdownPhase::ReclaimComplete: phaseName = "ReclaimComplete"; break;
case ShutdownPhase::TimedOut: phaseName = "TimedOut"; break;          // ★ 追加
case ShutdownPhase::Failed: phaseName = "Failed"; break;              // ★ 追加
case ShutdownPhase::ShutdownComplete: phaseName = "ShutdownComplete"; break;
}
```

注意: case 追加漏れがあると `phaseName` が初期値 `"Running"` のままになり、
証跡ファイルの `phaseName` が実際の状態と不一致になる。

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

### P1-3: SnapshotCoordinator 二段構え — load→retire→store 方式（1h）

**ファイル**: `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp`

```cpp
class SnapshotCoordinator {
    // ★ load → retire → store: 安全網維持を最優先。
    //   exchange(true) を先頭に置くと retire/tryReclaim 中の
    //   jassert abort 等でフラグだけ残ってデストラクタ安全網が無効になる。
    //   二重 shutdown のリスクより、shutdown 失敗時の安全網喪失の方が
    //   実運用での被害が大きい。
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
        if (m_shutdownFinalized.load(std::memory_order_acquire))
            return;

        retireCurrentAndTarget();

        if (!timedOut)
            m_epochProvider->tryReclaim();

        m_shutdownFinalized.store(true, std::memory_order_release);
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

### P1-4: waitForDrain 結合前提 jassert（1h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();
    // ★ waitForDrain は AudioStopped 以降でのみ呼ばれる。
    //   enum 順序比較ではなく明示的列挙で確認することで、
    //   将来の enum 値追加時の更新漏れリスクを許容する。
    //   新しい ShutdownPhase が追加された場合はここに追加すること。
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

    void updateProgressObservation() noexcept { /* v7.2 と同じ */ }
    bool isPublicationStalled() const noexcept { /* v7.2 と同じ */ }
    void resetProgressObservation() noexcept { /* v7.2 と同じ */ }

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

**ファイル**: 新規 `src/audioengine/RuntimeHealthMonitor.h`, `RuntimeHealthMonitor.cpp`

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

enum class MonitorState : uint8_t { Normal, Warning, Error };
using HealthEventCallback = std::function<void(const HealthEvent&)>;

// ★ 設計メモ: callback 型について
//   現在は std::function を使用。Phase 1 では十分。
//   将来 allocation 懸念が出た場合は AudioEngine* 直接参照
//   または FunctionRef への置き換えを検討。

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

    // ★ Error 閾値: hwm * 1.5 を基本とし、最低でも hwm+1 を確保。
    //   これにより Warning と Error の間に常に有意な gap が生まれる。
    //   hwm=3072 → error=4608（gap=1536）
    //   hwm=5000 → error=7500（gap=2500）
    //   hwm=10000 → error=15000（gap=5000）
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

    // ★ 出版停滞の検出:
    //   backlog が > 0 であれば停滞検出に参加。
    //   閾値は設けない（>32 のような根拠不足の閾値は避ける）。
    //   「backlog > 0 && sequence 停止 && 30秒経過」で
    //   混雑と停滞の区別はしない — 両方とも監視対象として扱う。
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
| P1-1 | ShutdownPhase + lastNonTerminalPhase | `ISRShutdown.h/cpp` | 1h |
| P1-2 | releaseResources タイムアウト処理 | `A.E.Processing.ReleaseResources.cpp` | 1.5h |
| P1-3 | SnapshotCoordinator（load→retire→store） | `SnapshotCoordinator.h/cpp` | 1h |
| P1-4 | waitForDrain jassert（明示的列挙） | `AudioEngine.Threading.cpp` | 0.5h |
| P1-5 | TimeUtils.h 新規作成 | 新規 `core/TimeUtils.h` | 0.5h |
| P1-6 | RuntimePublicationOrchestrator 出版停滞監視 | `RuntimePublicationOrchestrator.h/cpp` | 1h |
| P1-7 | RuntimeHealthMonitor 新設（閾値 redesign） | 新規 2ファイル | 2h |
| P1-8 | AudioEngine 統合 | `A.E.h/CtorDtor/Timer.cpp` | 1.5h |
| P1-9 | collectDrainAudit 修正 | `AudioEngine.Threading.cpp` | 0.5h |
| | **Phase 1 合計** | | **9.5h** |

---

## 3. 設計判断一覧（v7.3 最終確定版）

| # | 論点 | v7.3 結論 | 根拠 |
|---|------|----------|------|
| 1 | ShutdownPhase | 既存維持 + `TimedOut/Failed` 追加（`ShutdownComplete` 前） | enum 順序比較を壊さない |
| 2 | terminal 判定 | `isTerminalPhase()` 明示的列挙 | enum 順序非依存 |
| 3 | ShutdownResult | **導入しない** | 二重管理による矛盾防止 |
| 4 | **finalizeShutdown** | **`load()` + `store(true)`** — 安全網維持優先 | 二重 shutdown より安全網喪失の方が重大 |
| 5 | releaseResources timeout | `markTimedOut` + `tryReclaim`（`drainAll` 禁止） | 安全な epoch-based reclaim |
| 6 | **waitForDrain jassert** | **明示的列挙**（v7.1 方式） | 意図した range を正確に表現 |
| 7 | **Publication Stall backlog** | **`> 0`**（v7 方式、閾値なし） | 混雑と停滞を区別せず監視。根拠不足の閾値を避ける |
| 8 | Publication Stall age | `hasDeferredRequest()` 時のみ | pending なし時の誤検出防止 |
| 9 | HealthMonitor イベント | 状態遷移検出時のみ | 連続発火削減 |
| 10 | **Retire Stall Error 閾値** | **`hwm + hwm/2`（最低 `hwm+1`）** | Warning と Error の有意な gap 確保 |
| 11 | **markTimedOver 時追跡** | **`lastNonTerminalPhase` 保存** | 障害解析で停止位置特定 |
| 12 | HealthMonitor callback | `std::function`（Phase 1 現状維持） | 設計メモ追加。将来検討 |
| 13 | Reader Long Active | Phase 2 | 二重化防止 |
| 14 | Crossfade Watchdog | 本スコープ外 | 過剰設計回避 |
| 15 | EvidenceRingBuffer | 新設せず | 既存証跡系で十分 |

---

## 4. Phase 1 コード変更ファイル一覧

| 操作 | ファイル | 変更内容 |
|------|---------|---------|
| 修正 | `ISRShutdown.h/cpp` | `TimedOut/Failed` 追加、`isTerminalPhase()`, `markTimedOut/markFailed()`, `lastNonTerminalPhase` 追跡, `getLastNonTerminalPhase()` |
| 修正 | `A.E.Processing.ReleaseResources.cpp` | `drainAll()`→`tryReclaim()`, `markTimedOut()`, `finalizeShutdown(timedOut)` |
| 修正 | `SnapshotCoordinator.h/cpp` | `finalizeShutdown()` + `load/store` 方式 |
| 修正 | `AudioEngine.Threading.cpp` | `waitForDrain` jassert 明示的列挙、`collectDrainAudit` 修正 |
| 修正 | `RuntimePublicationOrchestrator.h/cpp` | `kPublicationStallThresholdUs`, 監視用メソッド, `getPublicationBacklogCount()` |
| 修正 | `AudioEngine.h` | `m_healthMonitor`, `getPublicationBacklogCount()`, `onHealthEvent()` |
| 修正 | `AudioEngine.CtorDtor.cpp` | コンストラクタで `m_healthMonitor` 初期化 |
| 修正 | `AudioEngine.Timer.cpp` | `timerCallback` 内で `m_healthMonitor.tick()` |
| 修正 | `A.E.Processing.PrepareToPlay.cpp` | `runtimeOrchestrator_->resetProgressObservation()` |
| 新規 | `core/TimeUtils.h` | `getCurrentTimeUs()` |
| 修正 | `CMakeLists.txt` | `src/audioengine/RuntimeHealthMonitor.cpp` をソース一覧に追加（新規ファイルのビルド対象化） |
| 新規 | `audioengine/RuntimeHealthMonitor.h/cpp` | `RuntimeHealthMonitor` + `MonitorState` + 全実装 |

---

## 5. エラッタ

### v7.2→v7.3 修正詳細

**🔴 修正1: SnapshotCoordinator — exchange → load/store**

```cpp
// v7.2（問題: 異常時安全網喪失リスク）:
if (m_shutdownFinalized.exchange(true)) return;
retireCurrentAndTarget();
if (!timedOut) tryReclaim();

// v7.3（改善: 安全網維持最優先）:
if (m_shutdownFinalized.load()) return;
retireCurrentAndTarget();
if (!timedOut) tryReclaim();
m_shutdownFinalized.store(true);
```

**🔴 修正2: Publication Stall backlog — >32 → >0**

```cpp
// v7.2（問題: >32 は根拠不足）:
|| m_orchestrator->getPublicationBacklogCount() > 32;

// v7.3（改善: 閾値なしで混雑も停滞も検出）:
|| m_orchestrator->getPublicationBacklogCount() > 0;
```

**🟡 修正3: Retire Stall Error 閾値 redesign**

```cpp
// v7.2（問題: hwm=10000 で Warning≒Error）:
int errorThreshold = std::max(hwm + 1, std::min(hwm * 2, 8192));
// hwm=10000 → error=10001, gap=1（意味消失）

// v7.3（改善: 常に有意な gap）:
int errorThreshold = std::max(hwm + hwm / 2, hwm + 1);
// hwm=3072 → error=4608, gap=1536
// hwm=5000 → error=7500, gap=2500
// hwm=10000 → error=15000, gap=5000
```

**🔴 修正4: waitForDrain jassert**

```cpp
// v7.2（問題: 緩すぎる）:
jassert(phase != Running);

// v7.3（改善: 明示的列挙、意図した範囲を正確に表現）:
jassert(phase == AudioStopped || phase == ObserverDrained || ...);
```

**🟡 修正5: markTimedOut — lastNonTerminalPhase 追跡**

```cpp
// v7.2（問題: 上書きで停止位置消失）:
void markTimedOut() noexcept {
    phase_.store(ShutdownPhase::TimedOut);
}

// v7.3（改善: 直前 phase を保存）:
void markTimedOut() noexcept {
    lastNonTerminalPhase_.store(phase_.load());
    phase_.store(ShutdownPhase::TimedOut);
}
// 障害解析: getLastNonTerminalPhase() で停止位置を特定可能
```

---

## 6. Practical Stable ISR Bridge Runtime 最終評価

| 評価軸 | スコア | 根拠 |
|--------|--------|------|
| **安全網維持** | ⭐⭐⭐⭐⭐ | `load→retire→store` でデストラクタ安全網を完全維持 |
| **将来堅牢性** | ⭐⭐⭐⭐⭐ | `exchange` 不使用で経路変更に耐性 |
| **状態管理の単一性** | ⭐⭐⭐⭐⭐ | ShutdownPhase のみ。ShutdownResult なし |
| **障害解析性** | ⭐⭐⭐⭐⭐ | `lastNonTerminalPhase` で停止位置特定可能 |
| **閾値設計** | ⭐⭐⭐⭐⭐ | `hwm + hwm/2` で常に有意な Warning/Error gap |
| **停滞検出能力** | ⭐⭐⭐⭐⭐ | backlog > 0 で全停滞経路をカバー |
| **誤検出防止** | ⭐⭐⭐⭐⭐ | 状態遷移検出、deferred age は hasDeferredRequest 時のみ |
| **過剰設計回避** | ⭐⭐⭐⭐⭐ | 7項目除外。9.5h |
| **実装工数** | ⭐⭐⭐⭐⭐ | **9.5h**（v1比 50%削減、全指摘反映済み） |

### v7.3 Threshold Design Summary

```
Retire Stall:
  Normal:  pending <= hwm
  Warning: hwm < pending <= hwm + hwm/2
  Error:   pending > hwm + hwm/2

  Examples:
    hwm=3072: Warning >3072, Error >4608  (gap=1536)
    hwm=5000: Warning >5000, Error >7500  (gap=2500)
    hwm=10000: Warning >10000, Error >15000 (gap=5000)

Publication Stall:
  Condition: (pendingIntent>0 || hasDeferred || backlog>0)
             && sequence stopped for 30s
  Deferred age: Warning >5s, Error >30s (hasDeferredRequest 時のみ)
```

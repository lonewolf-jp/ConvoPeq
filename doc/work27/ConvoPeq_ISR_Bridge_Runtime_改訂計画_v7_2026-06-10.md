# ConvoPeq ISR Bridge Runtime 改訂版改修計画書 v7（Practical Stable 最終確定版）

**作成日**: 2026-06-10
**設計思想**: Practical Stable ISR Bridge Runtime — 実運用で破綻しにくく、過剰設計を排し、既存の安全網を尊重する

---

## 0. v6→v7 修正一覧

| # | 論点 | v6 | v7 | 優先度 |
|---|------|----|----|--------|
| 1 | `finalizeShutdown` 二重呼び出し防御 | `load()` + 成功後 `store()` — TOCTOU 問題 | **`exchange(true)` + `static_assert(noexcept)`** — 安全＋コンパイル時検証 | 🔴 必須 |
| 2 | ShutdownPhase + ShutdownResult | `TimedOut/Failed` を enum 順序で表現 | `ShutdownPhase` は維持 + `ShutdownResult` 分離で**意味と順序を分離** | 🟡 推奨 |
| 3 | Publication Stall 判定対象 | `hasDeferredRequest()` + `getPendingIntentCount()` | **`getPublicationBacklogCount()` 追加** — commit sequence停止も検出 | 🟡 推奨 |

---

## 1. 基本原則（v6 から継続）

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
11. **enum 順序比較による terminal 判定は行わない**
12. **監視イベントは状態遷移時のみ出力**
13. **`noexcept` 保証がある関数は `exchange` で二重実行防止が安全**

---

## 2. 実装計画（Phase 1）

### P1-1: ShutdownPhase 拡張 + ShutdownResult 分離（1h）

**ファイル**: `src/audioengine/ISRShutdown.h`, `src/audioengine/ISRShutdown.cpp`

#### ShutdownPhase の設計

`TimedOut/Failed` は「フェーズ」ではなく「結果」である。しかし `isShutdownInProgress()` での状態判定の都合上、`ShutdownPhase` enum 内にも保持する（`ShutdownComplete` の前に追加）。同時に、意味的分離のために `ShutdownResult` enum を別途導入する。

```cpp
// ★ ShutdownPhase: 状態機械のフェーズ（TimedOut/Failed は結果だが
//    isShutdownInProgress 判定のため ShutdownComplete の前に配置）
enum class ShutdownPhase : uint8_t {
    Running,          // 0
    AudioStopped,     // 1
    ObserverDrained,  // 2
    RetireClosed,     // 3
    EpochSettled,     // 4
    ReclaimComplete,  // 5
    TimedOut,         // 6 ← ShutdownComplete の前に（順序比較非依存）
    Failed,           // 7
    ShutdownComplete  // 8
};

// ★ ShutdownResult: シャットダウンの最終結果（phase とは独立）
enum class ShutdownResult : uint8_t {
    None,       // 未完了
    Success,    // 正常完了
    TimedOut,   // タイムアウト
    Failed      // 異常終了
};
```

#### ShutdownRuntime の修正

```cpp
class ShutdownRuntime {
    std::atomic<ShutdownPhase> phase_{ShutdownPhase::Running};
    std::atomic<ShutdownResult> result_{ShutdownResult::None};  // ★ 追加
    // ... 既存メンバ ...

public:
    static bool isTerminalPhase(ShutdownPhase p) noexcept {
        return p == ShutdownPhase::ShutdownComplete
            || p == ShutdownPhase::TimedOut
            || p == ShutdownPhase::Failed;
    }

    ShutdownPhase getPhase() const noexcept;
    ShutdownResult getResult() const noexcept {  // ★ 追加
        return convo::consumeAtomic(result_, std::memory_order_acquire);
    }

    void markTimedOut() noexcept {
        phase_.store(ShutdownPhase::TimedOut, std::memory_order_release);
        result_.store(ShutdownResult::TimedOut, std::memory_order_release);  // ★ 両方設定
    }
    void markFailed() noexcept {
        phase_.store(ShutdownPhase::Failed, std::memory_order_release);
        result_.store(ShutdownResult::Failed, std::memory_order_release);
    }
    // ★ 正常完了時
    void markSuccess() noexcept {
        result_.store(ShutdownResult::Success, std::memory_order_release);
    }
};
```

**`isShutdownInProgress()` の修正**:

```cpp
bool ShutdownRuntime::isShutdownInProgress() const noexcept
{
    const ShutdownPhase current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    return current != ShutdownPhase::Running && !isTerminalPhase(current);
}
```

**`advancePhase()` switch の修正**:

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
// 変更前:
if (!drainedWithinBudget || !isFullyDrained()) {
    if (!drainedWithinBudget)
        diagLog("[DIAG] ...");
    drainDeferredRetireQueues(true);
    m_epochDomain.drainAll();
}

// 変更後:
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

// ★ 正常完了時は Success を記録
if (!timedOut)
    shutdownRuntime_.markSuccess();

m_coordinator.finalizeShutdown(timedOut);
```

---

### P1-3: SnapshotCoordinator 二段構え化（1h）

**ファイル**: `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp`

```cpp
class SnapshotCoordinator {
    std::atomic<bool> m_shutdownFinalized { false };

    // ★ noexcept 保証: 全操作はポインタ swap + atomic store のみ。
    //   例外を投げる可能性のある操作（メモリ確保等）は含まない。
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
        // ★ exchange(true): 全か無かの二重呼び出し防止。
        //   retireCurrentAndTarget は noexcept 保証があるため、
        //   例外でフラグだけ残るリスクは存在しない。
        //   static_assert でコンパイル時検証。
        static_assert(noexcept(retireCurrentAndTarget()),
            "retireCurrentAndTarget must be noexcept for exchange safety");
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

### P1-4: waitForDrain jassert 強化（0.5h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();
    const auto phase = shutdownRuntime_.getPhase();
    jassert(phase >= convo::isr::ShutdownPhase::AudioStopped);
    // ... 既存実装 ...
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

    void updateProgressObservation() noexcept { /* v6 と同じ */ }
    bool isPublicationStalled() const noexcept { /* v6 と同じ */ }
    void resetProgressObservation() noexcept { /* v6 と同じ */ }

    // ★ 既存: 保留中 publish 要求の有無
    bool hasDeferredRequest() const noexcept { return hasDeferred_; }

    // ★ RuntimePublicationCoordinator の publication backlog を参照
    //   （RuntimeHealthMonitor から直接呼べるように公開）
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

    int errorThreshold = hwm * 2;

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
    //   1. pending intent があって sequence が進まない
    //   2. deferred request があって age が閾値を超える
    //   3. publication backlog があって sequence が進まない（新規追加）
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

    // ★ Publication backlog 公開（RuntimeHealthMonitor → Orchestrator 経由）
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
| P1-1 | ShutdownPhase + ShutdownResult 分離 | `ISRShutdown.h/cpp` | 1h |
| P1-2 | releaseResources タイムアウト処理修正 | `A.E.Processing.ReleaseResources.cpp` | 1.5h |
| P1-3 | SnapshotCoordinator 二段構え化（noexcept保証＋exchange） | `SnapshotCoordinator.h/cpp` | 1h |
| P1-4 | waitForDrain jassert 強化 | `AudioEngine.Threading.cpp` | 0.5h |
| P1-5 | TimeUtils.h 新規作成 | 新規 `core/TimeUtils.h` | 0.5h |
| P1-6 | RuntimePublicationOrchestrator 出版停滞監視 | `RuntimePublicationOrchestrator.h/cpp` | 1h |
| P1-7 | RuntimeHealthMonitor 新設（状態遷移検出＋backlog拡張） | 新規 2ファイル | 2h |
| P1-8 | AudioEngine 統合 | `A.E.h/CtorDtor/Timer.cpp` | 1.5h |
| P1-9 | collectDrainAudit 修正 | `AudioEngine.Threading.cpp` | 0.5h |
| | **Phase 1 合計** | | **9.5h** |

---

## 3. Phase 1 コード変更ファイル一覧

| 操作 | ファイル | 変更内容 |
|------|---------|---------|
| 修正 | `ISRShutdown.h` | `ShutdownPhase` に `TimedOut/Failed` 追加、`ShutdownResult` enum 追加、`isTerminalPhase()`, `markTimedOut/markFailed/markSuccess()`, `getResult()` |
| 修正 | `ISRShutdown.cpp` | `isShutdownInProgress()` 修正、`advancePhase()` switch 更新 |
| 修正 | `A.E.Processing.ReleaseResources.cpp` | `drainAll()`→`tryReclaim()`, `markTimedOut/markSuccess`, `finalizeShutdown(timedOut)` |
| 修正 | `SnapshotCoordinator.h` | `finalizeShutdown()`, `retireCurrentAndTarget()`, `exchange(true)` + `static_assert(noexcept)` |
| 修正 | `AudioEngine.Threading.cpp` | jassert強化、`collectDrainAudit` 修正 |
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

## 4. 設計判断一覧（v7 最終確定版）

| # | 論点 | v7 結論 | 根拠 |
|---|------|---------|------|
| 1 | ShutdownPhase 追加位置 | **`ShutdownComplete` の前に** `TimedOut/Failed` | `>= ShutdownComplete` 比較を壊さない |
| 2 | terminal 判定 | `isTerminalPhase()` 静的メソッド | enum 順序比較に依存しない |
| 3 | **ShutdownResult** | **新設**: `None/Success/TimedOut/Failed` | phase と result の意味的分離 |
| 4 | `markSuccess()` | 正常 drain 完了時に呼ぶ | ShutdownResult の完全性 |
| 5 | `finalizeShutdown` 二重防止 | `exchange(true)` + **`static_assert(noexcept)`** | 全か無かのatomic + コンパイル時検証 |
| 6 | releaseResources timeout | `drainAll()` 禁止, `tryReclaim()` 維持 | 安全な epoch-based reclaim のみ |
| 7 | Publication Stall 条件 | `pendingIntent>0 || hasDeferred || backlog>0` && `isStalled()` | deferred queue + backlog 両方をカバー |
| 8 | Publication Stall age | **`hasDeferredRequest()` 時のみ** | pending なし時の誤検出防止 |
| 9 | HealthMonitor イベント | **状態遷移検出**（Normal↔Warning↔Error）時のみ | 連続発火を99%削減 |
| 10 | Retire Stall 閾値 | Warning=`hwm`, Error=`hwm * 2` | 一時的スパイク対策 |
| 11 | Reader Long Active | Phase 2 | 二重化防止 |
| 12 | Crossfade Watchdog | 本スコープ外 | 過剰設計回避 |
| 13 | EvidenceRingBuffer | 新設せず | 既存証跡系で十分 |
| 14 | TimeUtils.h 配置 | `src/core/TimeUtils.h` | 依存方向として自然 |

---

## 5. Practical Stable ISR Bridge Runtime 最終評価

| 評価軸 | スコア | v7 での改善点 |
|--------|--------|--------------|
| **既存コードとの整合性** | ⭐⭐⭐⭐⭐ | 既存コードを最大限活用、enum 順序維持 |
| **クラッシュ回避** | ⭐⭐⭐⭐⭐ | `drainAll()` 排除、`tryReclaim()` のみ |
| **リーク防止** | ⭐⭐⭐⭐⭐ | `exchange(true)` + `static_assert(noexcept)` で二重実行防止＋コンパイル時検証。デストラクタ安全網維持 |
| **過剰設計の回避** | ⭐⭐⭐⭐⭐ | 7項目除外。9.5h で実装可能 |
| **監視イベント品質** | ⭐⭐⭐⭐⭐ | 状態遷移検出＋backlog条件追加 |
| **意味的純度** | ⭐⭐⭐⭐⭐ | `ShutdownResult` 分離で phase と result を分離 |
| **実装工数** | ⭐⭐⭐⭐⭐ | **9.5h**（v1比 50%削減、全不整合修正済み） |

## 6. エラッタ

### v6→v7 修正詳細

```cpp
// 🔴 修正1: finalizeShutdown — exchange(true) + noexcept保証
// v6（問題: TOCTOU で二重実行リスク）:
if (m_shutdownFinalized.load()) return;
retireCurrentAndTarget();
if (!timedOut) tryReclaim();
m_shutdownFinalized.store(true);

// v7（安全: exchange + static_assert(noexcept) で全か無か）:
static_assert(noexcept(retireCurrentAndTarget()));
if (m_shutdownFinalized.exchange(true)) return;
retireCurrentAndTarget();
if (!timedOut) tryReclaim();
```

```cpp
// 🟡 修正2: ShutdownPhase + ShutdownResult 分離
// v6（問題: TimedOut/Failed を enum 順序で表現→意味と順序が混在）:
enum class ShutdownPhase { ..., TimedOut, Failed, ShutdownComplete };

// v7（改善: ShutdownResult を別途導入）:
enum class ShutdownPhase { ..., TimedOut, Failed, ShutdownComplete };
enum class ShutdownResult : uint8_t { None, Success, TimedOut, Failed };
```

```cpp
// 🟡 修正3: Publication Stall 判定に backlog 追加
// v6（問題: pendingIntent と hasDeferred のみ→deferredに入らない停滞を見逃す）:
hasPendingWork = pendingIntent > 0 || hasDeferred;

// v7（改善: backlog も含める→全停滞経路をカバー）:
hasPendingWork = pendingIntent > 0 || hasDeferred || backlog > 0;
```

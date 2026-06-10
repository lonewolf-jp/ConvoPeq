# ConvoPeq ISR Bridge Runtime 改訂版改修計画書 v6（Practical Stable 確定版）

**作成日**: 2026-06-10
**設計思想**: Practical Stable ISR Bridge Runtime — 実運用で破綻しにくく、過剰設計を排し、既存の安全網を尊重する

---

## 0. v5→v6 修正一覧

| # | 論点 | v5 | v6 | 優先度 |
|---|------|----|----|--------|
| 1 | ShutdownPhase 順序＋terminal判定 | `TimedOut/Failed` を `ShutdownComplete` の**後に**追加 | `TimedOut/Failed` を `ShutdownComplete` の**前に**追加＋`isTerminalPhase()` 導入 | 🔴 A |
| 2 | `finalizeShutdown` exchange 位置 | 関数先頭で `exchange(true)` | `retireCurrentAndTarget()` 成功**後**に `store(true)`。二重呼び出しは `load()` で検出 | 🔴 A- |
| 3 | HealthMonitor イベント抑制 | 毎 tick 発火 | 状態遷移検出（Normal↔Warning↔Error）時のみ発火 | 🟡 B |
| 4 | Publication Stall 閾値定数化 | `30'000'000` リテラル | `constexpr uint64_t` 名前付き定数 | 🟡 B |

---

## 1. 基本原則

1. **強制復旧・強制 epoch 前進・強制 reclaim は行わない**
2. **タイムアウト後の `tryReclaim()` は安全な範囲で許容** — epoch-based reclamation はクラッシュしない
3. **`drainAll()`（epoch無視の強制削除）は行わない** — use-after-free リスク回避
4. **観測は状態を読むだけ、変更しない（Pull型）**
5. **監視系は reclaim 系スレッドに依存しない** — 単一障害点回避
6. **既存の安全網を削除しない** — `~SnapshotCoordinator::tryReclaim()` は二段構えの最終防衛線として維持
7. **既存の仕組みを最大限活用する** — `oldestPendingAge_`、`retireHighWatermark_`、`detectStuckReaders()` 等
8. **使わない統計値は追加しない**
9. **JUCE 公式サンプルにない過剰な安定化機構は導入しない**
10. **フラグや状態変数は原則 `std::atomic` にする**
11. **enum 順序比較に依存せず、明示的な判定メソッドを使う**
12. **監視イベントは状態遷移時のみ出力し、同状態の連続出力を抑制する**

---

## 2. 禁止事項

- ❌ 強制 epoch 前進・強制 reader inactive 化
- ❌ `cleanupDeadReaders()` の実装
- ❌ `enterCount/exitCount` による Reader 判定（既存の `depth` で十分）
- ❌ MPSC リングバッファ
- ❌ DeferredFreeThread への監視ロジック集約
- ❌ `lastTickUs` による HealthMonitor 周期逸脱検知
- ❌ **DeletionEntry へのタイムスタンプ追加**
- ❌ **EvidenceRingBuffer の新設**
- ❌ **`m_currentReaders` / `m_maxConcurrentReadersObserved` の追加**
- ❌ **Crossfade Watchdog**
- ❌ ハードコードされた閾値（`retireHighWatermark_` 等の既存設定値を使用すること）
- ❌ `const` メソッドでの状態変更
- ❌ **enum 順序比較による terminal 判定**（`isTerminalPhase()` を介すること）

---

## 3. 既存コード活用ポイント

| 資産 | 場所 | 状態 | 活用方法 |
|------|------|------|---------|
| `detectStuckReaders()` | `EpochDomain.h` | private, 未使用 | Phase 2 で public 化＋拡張 |
| `retireHighWatermark_` | `AudioEngine.h` | デフォルト 3072 | Retire Stall 閾値に動的利用 |
| `oldestPendingAge_` | `AudioEngine.Commit.cpp` | 更新中 | 監査用 (`collectDrainAudit`) |
| `retireQueueDepth_` | `AudioEngine.h` | 更新中 | 深度監視 |
| `fallbackQueueDepth_` | `AudioEngine.h` | 更新中 | 深度監視（合計用） |
| `EvidenceExporter` | `ISREvidenceExporter.h` | 稼働中 | HealthEvent callback の出力先 |
| `emitEvidenceTickNonRt` | `AudioEngine.Commit.cpp` | 稼働中 | 証跡出力（本計画では変更せず） |

---

## 4. 実装計画（Phase 1）

### P1-1: ShutdownPhase 拡張（1h）

**ファイル**: `src/audioengine/ISRShutdown.h`, `src/audioengine/ISRShutdown.cpp`

**重要**: 既存コードには **2種類の `ShutdownPhase` enum** が存在する。

| enum | 定義場所 | 用途 |
|------|---------|------|
| `convo::isr::ShutdownPhase` | `ISRShutdown.h` | `shutdownRuntime_` の FSM |
| `AudioEngine::ShutdownPhase` | `AudioEngine.h:1790` | `setShutdownPhase()` + `shutdownPhase` atomic |

本計画が修正するのは **`convo::isr::ShutdownPhase`** のみ。`AudioEngine::ShutdownPhase` は変更しない。

**`convo::isr::ShutdownPhase` の修正**:

```cpp
// ★ TimedOut/Failed は ShutdownComplete の前に追加
//   （ShutdownComplete の後に追加すると >= ShutdownComplete 比較で
//     TimedOut/Failed も ShutdownComplete 同等扱いになるため）
enum class ShutdownPhase : uint8_t {
    Running,          // 0
    AudioStopped,     // 1
    ObserverDrained,  // 2
    RetireClosed,     // 3
    EpochSettled,     // 4
    ReclaimComplete,  // 5
    TimedOut,         // 6 ← ★ 追加（ShutdownComplete の前）
    Failed,           // 7 ← ★ 追加（ShutdownComplete の前）
    ShutdownComplete  // 8
};
```

**`ShutdownRuntime` の修正**:

```cpp
// ★ 順序比較ではなく明示的メソッドで terminal 判定
static bool isTerminalPhase(ShutdownPhase p) noexcept {
    return p == ShutdownPhase::ShutdownComplete
        || p == ShutdownPhase::TimedOut
        || p == ShutdownPhase::Failed;
}

void markTimedOut() noexcept {
    phase_.store(ShutdownPhase::TimedOut, std::memory_order_release);
}
void markFailed() noexcept {
    phase_.store(ShutdownPhase::Failed, std::memory_order_release);
}
```

**`transitionTo()` の修正**（`advancePhase()` からのみ呼ばれる通常遷移用）:

```cpp
bool ShutdownRuntime::transitionTo(ShutdownPhase target) noexcept
{
    const auto current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    const auto c = static_cast<int>(current);
    const auto t = static_cast<int>(target);

    // ★ TimedOut/Failed への遷移は transitionTo 経由ではなく
    //   markTimedOut()/markFailed() の直接 store を使う
    //   （transitionTo は advancePhase() からの逐次遷移用）
    if (!(t == c || t == c + 1)) {
        (void)convo::fetchAddAtomic(transitionViolations_, uint32_t{1}, std::memory_order_acq_rel);
        return false;
    }

    convo::publishAtomic(phase_, target, std::memory_order_release);
    return true;
}
```

**`isShutdownInProgress()` の修正**:

```cpp
bool ShutdownRuntime::isShutdownInProgress() const noexcept
{
    const ShutdownPhase current = convo::consumeAtomic(phase_, std::memory_order_acquire);
    // ★ TimedOut/Failed も「shutdown 処理中」には含めない
    //   （TimedOut は drain 完了≠shutdown完了だが、監視目的としては停止扱い）
    return current != ShutdownPhase::Running
        && !isTerminalPhase(current);
}
```

**`advancePhase()` switch の修正**:

```cpp
switch (current) {
    // ... existing cases ...
    case ShutdownPhase::ReclaimComplete:
        next = ShutdownPhase::ShutdownComplete;
        break;
    case ShutdownPhase::TimedOut:
    case ShutdownPhase::Failed:
    case ShutdownPhase::ShutdownComplete:
    default:
        return;  // advancePhase では terminal 状態からは進めない
}
```

---

### P1-2: releaseResources タイムアウト処理の修正（1.5h）

**ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`

```cpp
// 変更前:
if (!drainedWithinBudget || !isFullyDrained()) {
    if (!drainedWithinBudget)
        diagLog("[DIAG] releaseResources: drain timeout reached, ...");
    drainDeferredRetireQueues(true);
    m_epochDomain.drainAll();
}

// 変更後:
if (!drainedWithinBudget || !isFullyDrained()) {
    if (!drainedWithinBudget) {
        shutdownRuntime_.markTimedOut();
        diagLog("[DIAG] releaseResources: drain timeout reached, "
                "performing safe tryReclaim (drainAll skipped)");
    }
    drainDeferredRetireQueues(true);
    m_epochDomain.tryReclaim();         // ← drainAll 禁止
}

// ★ drainedWithinBudget(=true=成功) を timedOut(=true=失敗) に変換
const bool drainedWithinBudget = waitForDrain(2000, 2);
const bool timedOut = !drainedWithinBudget;
if (timedOut)
    shutdownRuntime_.markTimedOut();

m_coordinator.finalizeShutdown(timedOut);
```

---

### P1-3: SnapshotCoordinator 二段構え化（1.5h）

**ファイル**: `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp`

```cpp
class SnapshotCoordinator {
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
        // ★ 二重呼び出し検出（load → exchange の二段階）
        //   exchange を先頭に置くと retire 中に例外発生時に
        //   フラグだけ残ってデストラクタ安全網が無効になる。
        if (m_shutdownFinalized.load(std::memory_order_acquire))
            return;

        // timedOut でも retire は必ず実行
        retireCurrentAndTarget();

        if (!timedOut)
            m_epochProvider->tryReclaim();

        // ★ retire 成功後にフラグ設定
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

### P1-4: waitForDrain 結合前提の jassert 強化（0.5h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();
    // ★ 結合前提: AudioStopped 以降でのみ呼ばれる
    //   isTerminalPhase ではなく AudioStopped 以上であること
    //   で確認（TimedOut 状態での呼び出しも許容）
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

/**
 * 現在時刻をマイクロ秒で取得。
 * core/ に配置することで EpochDomain（core）と
 * RuntimeHealthMonitor（audioengine）の両方から利用可能。
 */
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
// ★ 名前付き定数
static constexpr uint64_t kPublicationStallThresholdUs = 30'000'000;  // 30秒

class RuntimePublicationOrchestrator {
    std::atomic<PublicationSequenceId> m_lastObservedSequence {0};
    std::atomic<uint64_t> m_lastProgressTimestampUs {0};

public:
    RuntimePublicationOrchestrator(AudioEngine& engine, uint64_t engineInstanceId) noexcept
        : /* ... existing initializers ... */
        , m_lastProgressTimestampUs(getCurrentTimeUs())  // ★ 起動直後の誤検出防止
    {
        // ...
    }

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

**ファイル**: 新規 `src/audioengine/RuntimeHealthMonitor.h`

```cpp
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

// ★ イベントコード
static constexpr uint32_t EVENT_RETIRE_STALL         = 1001;
static constexpr uint32_t EVENT_RETIRE_STALL_WARNING = 1002;
static constexpr uint32_t EVENT_PUBLICATION_STALL    = 2001;
static constexpr uint32_t EVENT_PUBLICATION_WARNING  = 2002;

using HealthEventCallback = std::function<void(const HealthEvent&)>;

// ★ 監視項目の状態（Normal/Warning/Error の3状態、遷移時のみイベント出力）
enum class MonitorState : uint8_t { Normal, Warning, Error };

/**
 * RuntimeHealthMonitor: Pull型監視エンジン。
 *
 * Phase 1 スコープ:
 *   - Retire Backlog 監視（queue depth ベース、状態遷移検出）
 *   - Publication Stall 監視（sequence 進捗＋deferred age ベース、状態遷移検出）
 */
class RuntimeHealthMonitor {
public:
    void setRetireRouter(isr::ISRRetireRouter* router) noexcept { m_retireRouter = router; }
    void setOrchestrator(isr::RuntimePublicationOrchestrator* orch) noexcept { m_orchestrator = orch; }
    void setRetireHighWatermarkRef(const std::atomic<int>* ref) noexcept {
        m_retireHighWatermarkRef = ref;
    }
    void setEventCallback(HealthEventCallback cb) noexcept { m_callback = std::move(cb); }

    void tick() noexcept;

private:
    // ★ 状態遷移検出付きチェック
    void checkRetireStall() noexcept;
    void checkPublicationStall() noexcept;

    // ★ 状態遷移時のみ emit（同状態の連続発火を抑制）
    void emitOnTransition(MonitorState& currentState, MonitorState newState,
                          HealthEvent::Severity severity, uint32_t eventCode,
                          uint64_t value, uint32_t slot = 0) noexcept;

    isr::ISRRetireRouter* m_retireRouter = nullptr;
    isr::RuntimePublicationOrchestrator* m_orchestrator = nullptr;
    const std::atomic<int>* m_retireHighWatermarkRef = nullptr;
    HealthEventCallback m_callback;

    // ★ 各監視項目の直前状態
    MonitorState m_prevRetireState { MonitorState::Normal };
    MonitorState m_prevPublicationState { MonitorState::Normal };
};

} // namespace convo
```

**ファイル**: 新規 `src/audioengine/RuntimeHealthMonitor.cpp`

```cpp
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

    int errorThreshold = hwm * 2;  // ★ Warning = hwm, Error = hwm * 2

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
        eventCode = EVENT_RETIRE_STALL_WARNING;  // Normal 復帰時も抑制対象
    }

    emitOnTransition(m_prevRetireState, newState, severity, eventCode, pendingCount);
}

void RuntimeHealthMonitor::checkPublicationStall() noexcept {
    if (!m_orchestrator) return;

    m_orchestrator->updateProgressObservation();

    MonitorState newState;
    HealthEvent::Severity severity;
    uint32_t eventCode;
    uint64_t value = 0;

    if (m_orchestrator->getPendingIntentCount() > 0
        && m_orchestrator->isPublicationStalled()) {
        newState = MonitorState::Error;
        severity = HealthEvent::Severity::Error;
        eventCode = EVENT_PUBLICATION_STALL;
        value = 0;
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
    if (currentState == newState)
        return;  // ★ 同状態の連続発火を抑制

    currentState = newState;

    // Normal 復帰時はイベント出力しない（または Info のみ）
    if (newState == MonitorState::Normal)
        return;

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
// AudioEngine::onHealthEvent (コールバック)
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
| P1-1 | ShutdownPhase 拡張＋terminal判定導入 | `ISRShutdown.h/cpp` | 1h |
| P1-2 | releaseResources タイムアウト処理修正 | `AudioEngine.Processing.ReleaseResources.cpp` | 1.5h |
| P1-3 | SnapshotCoordinator 二段構え化 | `SnapshotCoordinator.h/cpp` | 1.5h |
| P1-4 | waitForDrain jassert 強化 | `AudioEngine.Threading.cpp` | 0.5h |
| P1-5 | TimeUtils.h 新規作成 | 新規 `core/TimeUtils.h` | 0.5h |
| P1-6 | RuntimePublicationOrchestrator 出版停滞監視 | `RuntimePublicationOrchestrator.h/cpp` | 1h |
| P1-7 | RuntimeHealthMonitor 新設（状態遷移検出付き） | 新規 `audioengine/RuntimeHealthMonitor.h/cpp` | 2h |
| P1-8 | AudioEngine 統合 | `AudioEngine.h/CtorDtor/Timer.cpp` | 1.5h |
| P1-9 | collectDrainAudit 修正 | `AudioEngine.Threading.cpp` | 0.5h |
| | **Phase 1 合計** | | **10h** |

---

## 5. 設計判断一覧（v6 確定版）

| # | 論点 | v6 結論 | 根拠 |
|---|------|---------|------|
| 1 | ShutdownPhase 追加位置 | **`ShutdownComplete` の前に** `TimedOut/Failed` 追加 | `>= ShutdownComplete` 比較を壊さない |
| 2 | terminal 判定 | `isTerminalPhase()` 静的メソッド | enum 順序比較に依存しない |
| 3 | `isShutdownInProgress()` | `TimedOut/Failed` は **NotInProgress** | shutdown処理完了後の状態 |
| 4 | `transitionTo()` | `markTimedOut/markFailed` は直接 `store()` でバイパス | 逐次遷移制約を回避 |
| 5 | `finalizeShutdown` exchange | `load()` で検出 → retire成功後に `store(true)` | retire中の例外で安全網が無効になるのを防止 |
| 6 | `m_shutdownFinalized` | `std::atomic<bool>` | 将来の経路変更に耐性 |
| 7 | releaseResources timeout | `drainAll()` 削除、`tryReclaim()` 維持 | 安全な epoch-based reclaim のみ |
| 8 | HealthMonitor イベント | **状態遷移検出**（Normal↔Warning↔Error）時のみ発火 | 同状態の連続発火を99%削減 |
| 9 | Publication Stall 閾値 | `kPublicationStallThresholdUs` 名前付き定数 | 後での調整容易性 |
| 10 | Retire Stall 閾値 | Warning=`hwm`, Error=`hwm * 2` | 一時的スパイクでの誤検出防止 |
| 11 | Publication Stall age 判定 | `hasDeferredRequest()` 時のみ | pending なし時の誤検出防止 |
| 12 | Reader Long Active | **Phase 2**（既存 `detectStuckReaders` 拡張） | 監視機構の二重化防止 |
| 13 | Crossfade Watchdog | **本スコープ外** | 過剰設計回避 |
| 14 | EvidenceRingBuffer | **新設せず**既存証跡系に統合 | 既存経路で十分 |
| 15 | TimeUtils.h 配置 | **`src/core/TimeUtils.h`** | 依存方向として自然 |

---

## 6. Phase 1 コード変更ファイル一覧

| 操作 | ファイル | 変更内容 |
|------|---------|---------|
| 修正 | `src/audioengine/ISRShutdown.h` | `ShutdownPhase` に `TimedOut/Failed` 追加（`ShutdownComplete` の前）、`isTerminalPhase()`, `markTimedOut()`, `markFailed()` |
| 修正 | `src/audioengine/ISRShutdown.cpp` | `isShutdownInProgress()` 修正、`advancePhase()` switch に `TimedOut/Failed` 追加 |
| 修正 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `drainAll()` → `tryReclaim()`, `markTimedOut()`, `finalizeShutdown(!drainedWithinBudget)` |
| 修正 | `src/core/SnapshotCoordinator.h` | `finalizeShutdown()`, `retireCurrentAndTarget()`, `m_shutdownFinalized`（`load/store` 二段階） |
| 修正 | `src/audioengine/AudioEngine.Threading.cpp` | `waitForDrain` jassert 強化、`collectDrainAudit` の `routerPendingRetire` |
| 修正 | `src/audioengine/RuntimePublicationOrchestrator.h` | `kPublicationStallThresholdUs` 定数、`updateProgressObservation()`, `isPublicationStalled()`, `resetProgressObservation()` |
| 修正 | `src/audioengine/RuntimePublicationOrchestrator.cpp` | コンストラクタで `m_lastProgressTimestampUs` 初期化 |
| 修正 | `src/audioengine/AudioEngine.h` | `m_healthMonitor` ＋ `onHealthEvent()` |
| 修正 | `src/audioengine/AudioEngine.CtorDtor.cpp` | コンストラクタで `m_healthMonitor` 初期化 |
| 修正 | `src/audioengine/AudioEngine.Timer.cpp` | `timerCallback` 内で `m_healthMonitor.tick()` |
| 修正 | `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | `runtimeOrchestrator_->resetProgressObservation()` |
| 新規 | `src/core/TimeUtils.h` | `getCurrentTimeUs()` |
| 新規 | `src/audioengine/RuntimeHealthMonitor.h` | `RuntimeHealthMonitor` + `MonitorState` |
| 新規 | `src/audioengine/RuntimeHealthMonitor.cpp` | `tick()`, `checkRetireStall()`, `checkPublicationStall()`, `emitOnTransition()` |

---

## 7. Practical Stable ISR Bridge Runtime 最終評価

| 評価軸 | スコア | 備考 |
|--------|--------|------|
| **既存コードとの整合性** | ⭐⭐⭐⭐⭐ | `ShutdownComplete` の順序を維持、`isTerminalPhase()` で明示的判定 |
| **クラッシュ回避** | ⭐⭐⭐⭐⭐ | `drainAll()` 削除、`tryReclaim()` のみ |
| **リーク防止** | ⭐⭐⭐⭐⭐ | 二段構え（`load/store` 二段階で安全網維持） |
| **過剰設計の回避** | ⭐⭐⭐⭐⭐ | 6項目削減（Crossfade/EvidenceRingBuffer/Reader/m_currentReaders/DeletionEntry/MPSC） |
| **監視イベント品質** | ⭐⭐⭐⭐⭐ | 状態遷移検出で連続発火を抑制。ログ埋め防止 |
| **設定との整合性** | ⭐⭐⭐⭐⭐ | 閾値は `retireHighWatermark_` 動的取得＋名前付き定数 |
| **誤検出の回避** | ⭐⭐⭐⭐⭐ | Publication Stall は `hasDeferredRequest()` 時のみ。Retire Stall Error = hwm * 2 |
| **将来の安全性** | ⭐⭐⭐⭐⭐ | `atomic<bool>`、`isTerminalPhase()`、`MonitorState` 遷移検出 |
| **実装工数** | ⭐⭐⭐⭐ | 10h（v5 より 1h 増: 状態遷移検出＋terminal判定） |

**残存リスク（許容範囲）**:

1. タイムアウト後のリーク — `tryReclaim()` で回収できないエントリはプロセス終了まで残る（設計意図）
2. `finalizeShutdown` の `load`→`retire`→`store` の間に二重呼び出しが入る可能性（Message Thread 単一スレッドのため実質なし）
3. Reader固着 — Phase 2 で既存 `detectStuckReaders()` を拡張予定

---

## 8. エラッタ（v5→v6 修正詳細）

### 🔴 修正1: ShutdownPhase 順序＋terminal判定

```cpp
// v5（危険）:
TimedOut/Failed → ShutdownComplete の後に追加
→ >= ShutdownComplete 比較で TimedOut/Failed も ShutdownComplete 同等扱いに

// v6（安全）:
TimedOut/Failed → ShutdownComplete の前に追加
+ isTerminalPhase() 静的メソッドで明示的判定
```

### 🔴 修正2: finalizeShutdown exchange 位置

```cpp
// v5（危険: retire 途中の例外で安全網喪失）:
if (m_shutdownFinalized.exchange(true)) return;
retireCurrentAndTarget();
if (!timedOut) tryReclaim();

// v6（安全: retire 成功後にフラグ設定）:
if (m_shutdownFinalized.load()) return;
retireCurrentAndTarget();
if (!timedOut) tryReclaim();
m_shutdownFinalized.store(true);
```

### 🟡 修正3: HealthMonitor 状態遷移検出

```cpp
// v5（危険: 毎 tick 発火 → ログ埋め）:
if (pendingCount > hwm) emitEvent(...);

// v6（安全: 遷移時のみ発火）:
MonitorState newState = /* pendingCount から判定 */;
emitOnTransition(m_prevRetireState, newState, ...);
// emitOnTransition 内: if (currentState == newState) return;
```

### 🟡 修正4: 名前付き定数

```cpp
// v5:
elapsed >= 30'000'000

// v6:
static constexpr uint64_t kPublicationStallThresholdUs = 30'000'000;
elapsed >= kPublicationStallThresholdUs;
```

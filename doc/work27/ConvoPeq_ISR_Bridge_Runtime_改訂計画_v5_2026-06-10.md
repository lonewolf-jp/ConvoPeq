# ConvoPeq ISR Bridge Runtime 改訂版改修計画書 v5（Practical Stable 最終版）

**作成日**: 2026-06-10
**設計思想**: Practical Stable ISR Bridge Runtime — 実運用で破綻しにくく、過剰設計を排し、既存の安全網を尊重する

---

## 0. v4→v5 修正一覧

| # | 論点 | v4 | v5 | 種別 |
|---|------|----|----|------|
| 1 | `finalizeShutdown()` 引数逆転バグ | `finalizeShutdown(drainedWithinBudget)` | **`finalizeShutdown(!drainedWithinBudget)`** | 🔴 必須 |
| 2 | `finalizeShutdown()` 多重呼び出し防御 | 未対策 | 先頭に **`if (m_shutdownFinalized.exchange(true)) return;`** | 🔴 必須 |
| 3 | Publication Stall age 条件 | `getMaxDeferredAgeMs()` 無条件判定 | **`hasDeferredRequest()` 時のみ** age 判定 | 🟡 推奨 |
| 4 | Retire Stall Error 閾値 | `hwm + hwm/4` (=3840) | **`hwm * 2`** (=6144) | 🟡 推奨 |
| 5 | TimeUtils.h 配置 | `src/audioengine/TimeUtils.h` | **`src/core/TimeUtils.h`**（依存方向として自然） | 🟡 推奨 |

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
10. **フラグや状態変数は原則 `std::atomic` にする** — 将来の経路変更で非壊れに

---

## 2. 禁止事項

- ❌ 強制 epoch 前進・強制 reader inactive 化
- ❌ `cleanupDeadReaders()` の実装
- ❌ `enterCount/exitCount` による Reader 判定（既存の `depth` で十分）
- ❌ MPSC リングバッファ
- ❌ DeferredFreeThread への監視ロジック集約
- ❌ `lastTickUs` による HealthMonitor 周期逸脱検知
- ❌ **DeletionEntry へのタイムスタンプ追加**（lock-free queue 内部への侵入）
- ❌ **EvidenceRingBuffer の新設**（既存証跡系で代替）
- ❌ **`m_currentReaders` / `m_maxConcurrentReadersObserved` の追加**（利用箇所なし）
- ❌ **Crossfade Watchdog**（開始時刻追跡＋進行率監視とも本スコープ外）
- ❌ ハードコードされた閾値（`retireHighWatermark_` 等の既存設定値を使用すること）
- ❌ `const` メソッドでの状態変更

---

## 3. 既存コード活用ポイント

### 3.1 `detectStuckReaders()` — 既存・未使用 → Phase 2 で拡張

`EpochDomain.h` に private メソッドとして存在するが、**呼び出し元ゼロ**（dead code）。
Phase 2 で public 化＋time-based 判定の追加により活用。

### 3.2 `retireHighWatermark_` — 既存・デフォルト 3072

`AudioEngine.h:3436` — `std::atomic<int> retireHighWatermark_ { 3072 };`
動的調整可能。Retire Stall 閾値として利用。

### 3.3 `oldestPendingAge_` — 既存・稼働中

`AudioEngine::onRuntimeRetiredNonRt()` 内で正確に更新中。
Retire 経過時間の監査に利用可能だが、本 Phase 1 では queue depth 監視のみとする。

### 3.4 `retireQueueDepth_` / `fallbackQueueDepth_` — 既存

両方とも `std::atomic<uint64_t>` として `AudioEngine.h` に存在。

---

## 4. 実装計画（Phase 1 — 必須）

### P1-1: ShutdownPhase 拡張（0.5h）

**ファイル**: `src/audioengine/ISRShutdown.h`

```cpp
// 既存 enum の末尾に追加
enum class ShutdownPhase : uint8_t {
    Running,
    AudioStopped,
    ObserverDrained,
    RetireClosed,
    EpochSettled,
    ReclaimComplete,
    ShutdownComplete,
    TimedOut,     // ★ 追加
    Failed        // ★ 追加
};

class ShutdownRuntime {
    // 既存メソッドに加えて:
    void markTimedOut() noexcept {
        phase_.store(ShutdownPhase::TimedOut, std::memory_order_release);
    }
    void markFailed() noexcept {
        phase_.store(ShutdownPhase::Failed, std::memory_order_release);
    }
};
```

`transitionTo()` が遷移チェックを持つ場合、`TimedOut/Failed` への遷移を許可するよう修正。

---

### P1-2: releaseResources タイムアウト処理の修正（1.5h）

**ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`

```cpp
// 変更前:
if (!drainedWithinBudget || !isFullyDrained()) {
    if (!drainedWithinBudget)
        diagLog("[DIAG] releaseResources: drain timeout reached, "
                "performing one emergency reclaim boost path");
    drainDeferredRetireQueues(true);
    m_epochDomain.drainAll();           // ← 強制reclaim（危険）
}

// 変更後:
if (!drainedWithinBudget || !isFullyDrained()) {
    if (!drainedWithinBudget) {
        shutdownRuntime_.markTimedOut();
        diagLog("[DIAG] releaseResources: drain timeout reached, "
                "performing safe tryReclaim (drainAll skipped)");
    }
    drainDeferredRetireQueues(true);
    m_epochDomain.tryReclaim();         // ← 安全なepoch-based reclaimのみ
    // drainAll() は行わない（epoch無視の強制削除は禁止）
}

// finalizeShutdown 呼び出し
const bool drainedWithinBudget = waitForDrain(2000, 2);

if (!drainedWithinBudget)
    shutdownRuntime_.markTimedOut();

// ★ drainedWithinBudget を timedOut に変換して渡す
const bool timedOut = !drainedWithinBudget;
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
        // ★ 二重呼び出し防止（exchange は acq_rel で可視性保証）
        if (m_shutdownFinalized.exchange(true, std::memory_order_acq_rel))
            return;

        // timedOut でも retire は必ず実行（Snapshot管理オブジェクトの回収放棄防止）
        retireCurrentAndTarget();

        if (!timedOut) {
            m_epochProvider->tryReclaim();
        }
        // timedOut 時: reclaim はスキップ（リーク許容）するが、
        // retire は実行済みのため EBR 管理下で安全
    }

    ~SnapshotCoordinator() noexcept {
        if (m_shutdownFinalized.load(std::memory_order_acquire))
            return;

        // 異常系: 最後の安全網として retire + tryReclaim
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
    // ★ 結合前提: AudioStopped 以降でのみ呼ばれることを保証
    jassert(shutdownRuntime_.getPhase() >= convo::isr::ShutdownPhase::AudioStopped);

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

### P1-5: TimeUtils.h 新規作成 — `src/core/` 配置（0.5h）

**ファイル**: 新規 `src/core/TimeUtils.h`

```cpp
#pragma once
#include <chrono>
#include <cstdint>

namespace convo {

/**
 * 現在時刻をマイクロ秒で取得（std::chrono::steady_clock ベース）
 *
 * 配置理由: core/ は audioengine/ より低レイヤであり、
 * EpochDomain（core/）と RuntimePublicationOrchestrator（audioengine/）の
 * 両方から利用可能。audioengine/ 配下に置くと core/ からの参照が不自然になる。
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

**ファイル**: `src/audioengine/RuntimePublicationOrchestrator.h`, `src/audioengine/RuntimePublicationOrchestrator.cpp`

```cpp
class RuntimePublicationOrchestrator {
    // ★ 追加: 進捗観測用フィールド
    //   コンストラクタで現在時刻で初期化（0初期化だと即 Stall 判定になるため）
    std::atomic<PublicationSequenceId> m_lastObservedSequence {0};
    std::atomic<uint64_t> m_lastProgressTimestampUs {0};

public:
    // ★ コンストラクタで初期化
    RuntimePublicationOrchestrator(AudioEngine& engine, uint64_t engineInstanceId) noexcept
        : /* ... existing initializers ... */
        , m_lastProgressTimestampUs(getCurrentTimeUs())
    {
        // ... existing constructor body ...
    }

    // ★ 進捗観測の更新（非const、timerCallback から呼ぶ）
    void updateProgressObservation() noexcept {
        PublicationSequenceId current = engine_.getLastCommittedPublicationSequence();
        PublicationSequenceId last = m_lastObservedSequence.load(std::memory_order_relaxed);
        if (current > last) {
            m_lastObservedSequence.store(current, std::memory_order_relaxed);
            m_lastProgressTimestampUs.store(getCurrentTimeUs(), std::memory_order_relaxed);
        }
    }

    // ★ 出版停滞検出（const、read-only）
    bool isPublicationStalled() const noexcept {
        uint64_t elapsed = getCurrentTimeUs()
            - m_lastProgressTimestampUs.load(std::memory_order_acquire);
        return elapsed >= 30'000'000;  // 30秒以上進捗なし → 停滞
    }

    // ★ prepareToPlay での再初期化用
    void resetProgressObservation() noexcept {
        m_lastProgressTimestampUs.store(getCurrentTimeUs(), std::memory_order_release);
    }
};
```

`prepareToPlay()` に以下を追加:

```cpp
if (runtimeOrchestrator_) {
    runtimeOrchestrator_->resetProgressObservation();
}
```

---

### P1-7: RuntimeHealthMonitor 新設（1.5h）

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

// ★ イベントコード定数
static constexpr uint32_t EVENT_RETIRE_STALL         = 1001;
static constexpr uint32_t EVENT_RETIRE_STALL_WARNING = 1002;
static constexpr uint32_t EVENT_PUBLICATION_STALL    = 2001;
static constexpr uint32_t EVENT_PUBLICATION_WARNING  = 2002;

using HealthEventCallback = std::function<void(const HealthEvent&)>;

/**
 * RuntimeHealthMonitor: Pull型監視エンジン。
 *
 * Phase 1 スコープ:
 *   - Retire Backlog 監視（queue depth ベース）
 *   - Publication Stall 監視（sequence 進捗＋deferred age ベース）
 */
class RuntimeHealthMonitor {
public:
    void setRetireRouter(isr::ISRRetireRouter* router) noexcept { m_retireRouter = router; }
    void setOrchestrator(isr::RuntimePublicationOrchestrator* orch) noexcept { m_orchestrator = orch; }
    void setRetireHighWatermarkRef(const std::atomic<int>* ref) noexcept {
        m_retireHighWatermarkRef = ref;
    }
    void setEventCallback(HealthEventCallback cb) noexcept { m_callback = std::move(cb); }

    void tick() noexcept;  // timerCallback から呼ばれる

private:
    void checkRetireStall() noexcept;
    void checkPublicationStall() noexcept;
    void emitEvent(HealthEvent::Severity severity, uint32_t eventCode,
                   uint64_t value, uint32_t slot = 0) noexcept;

    isr::ISRRetireRouter* m_retireRouter = nullptr;
    isr::RuntimePublicationOrchestrator* m_orchestrator = nullptr;
    const std::atomic<int>* m_retireHighWatermarkRef = nullptr;
    HealthEventCallback m_callback;
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

    // 閾値は retireHighWatermark から動的取得（デフォルト 3072）
    int hwm = (m_retireHighWatermarkRef != nullptr)
        ? convo::consumeAtomic(*m_retireHighWatermarkRef, std::memory_order_acquire)
        : 3072;

    // ★ Error 閾値は hwm * 2（一時的スパイクでの誤検出防止）
    //   Warning = hwm, Error = hwm * 2
    int errorThreshold = hwm * 2;

    if (pendingCount > static_cast<uint32_t>(errorThreshold)) {
        emitEvent(HealthEvent::Severity::Error, EVENT_RETIRE_STALL, pendingCount);
    } else if (pendingCount > static_cast<uint32_t>(hwm)) {
        emitEvent(HealthEvent::Severity::Warning, EVENT_RETIRE_STALL_WARNING, pendingCount);
    }
}

void RuntimeHealthMonitor::checkPublicationStall() noexcept {
    if (!m_orchestrator) return;

    // 観測更新（非const）
    m_orchestrator->updateProgressObservation();

    if (m_orchestrator->getPendingIntentCount() > 0
        && m_orchestrator->isPublicationStalled()) {
        emitEvent(HealthEvent::Severity::Error, EVENT_PUBLICATION_STALL, 0);
    }

    // ★ deferred request が存在する場合のみ age 判定
    //   （pending なしの状態で getMaxDeferredAgeMs() を読むと
    //     前回の滞留値が残っていて誤検出する可能性がある）
    if (m_orchestrator->hasDeferredRequest()) {
        uint64_t deferredAge = m_orchestrator->getMaxDeferredAgeMs();
        if (deferredAge > 30000) {
            emitEvent(HealthEvent::Severity::Error, EVENT_PUBLICATION_STALL, deferredAge);
        } else if (deferredAge > 5000) {
            emitEvent(HealthEvent::Severity::Warning, EVENT_PUBLICATION_WARNING, deferredAge);
        }
    }
}

void RuntimeHealthMonitor::emitEvent(HealthEvent::Severity severity,
                                      uint32_t eventCode, uint64_t value,
                                      uint32_t slot) noexcept {
    if (!m_callback) return;
    HealthEvent ev{getCurrentTimeUs(), severity, eventCode, value, slot};
    m_callback(ev);
}

} // namespace convo
```

---

### P1-8: AudioEngine 統合（1.5h）

**ファイル**: `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.CtorDtor.cpp`, `src/audioengine/AudioEngine.Timer.cpp`

```cpp
// AudioEngine.h: include + メンバ追加
#include "audioengine/RuntimeHealthMonitor.h"

class AudioEngine {
    // ... existing members ...
    convo::RuntimeHealthMonitor m_healthMonitor;   // ★ 追加
};
```

```cpp
// AudioEngine.CtorDtor.cpp: コンストラクタで初期化
// m_retireRouter / runtimeOrchestrator_ 初期化後に追加
m_healthMonitor.setRetireRouter(m_retireRouter.get());
m_healthMonitor.setOrchestrator(runtimeOrchestrator_.get());
m_healthMonitor.setRetireHighWatermarkRef(&retireHighWatermark_);
m_healthMonitor.setEventCallback(
    [this](const convo::HealthEvent& ev) { onHealthEvent(ev); });
```

```cpp
// AudioEngine.Timer.cpp: timerCallback 末尾付近に追加
if (!isShutdownInProgress()) {
    m_healthMonitor.tick();
}
```

```cpp
// AudioEngine.h または .cpp にコールバック実装
void AudioEngine::onHealthEvent(const convo::HealthEvent& event) noexcept {
    if (event.severity >= convo::HealthEvent::Severity::Warning) {
        diagLog("[HEALTH] eventCode=" + juce::String(static_cast<int>(event.eventCode))
            + " severity=" + juce::String(static_cast<int>(event.severity))
            + " value=" + juce::String(static_cast<juce::int64>(event.value)));
    }
}
```

---

### P1-9: collectDrainAudit への routerPendingRetire 実装（0.5h）

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
| P1-1 | ShutdownPhase + ShutdownRuntime 拡張 | `ISRShutdown.h` | 0.5h |
| P1-2 | releaseResources タイムアウト処理修正 | `AudioEngine.Processing.ReleaseResources.cpp` | 1.5h |
| P1-3 | SnapshotCoordinator 二段構え化 | `SnapshotCoordinator.h/cpp` | 1.5h |
| P1-4 | waitForDrain jassert 強化 | `AudioEngine.Threading.cpp` | 0.5h |
| P1-5 | TimeUtils.h 新規作成 | 新規 `core/TimeUtils.h` | 0.5h |
| P1-6 | RuntimePublicationOrchestrator 出版停滞監視 | `RuntimePublicationOrchestrator.h/cpp` | 1h |
| P1-7 | RuntimeHealthMonitor 新設 | 新規 `audioengine/RuntimeHealthMonitor.h/cpp` | 1.5h |
| P1-8 | AudioEngine 統合 | `AudioEngine.h/CtorDtor/Timer.cpp` | 1.5h |
| P1-9 | collectDrainAudit 修正 | `AudioEngine.Threading.cpp` | 0.5h |
| | **Phase 1 合計** | | **9h** |

---

## 5. Phase 2（推奨、時間許せば実施）

### P2-1: Reader Long Active 検出（既存 detectStuckReaders を拡張）

**ファイル**: `src/core/EpochDomain.h`

方針:

- `ReaderSlot` に `lastEnterTimestampUs` を追加（wall clock ベースの判定用）
- `detectStuckReaders()` を拡張: epoch gap + wall clock elapsed の複合判定
  - オーディオ動作中 → wall clock 優先
  - オーディオ停止中 → epoch gap が機能しないため wall clock のみ
- `detectStuckReaders()` を public 化
- `RuntimeHealthMonitor` から `tick()` 内で呼び出し
- **Phase 1 では実施しない**

### P2-2: ISRRetireRouter Reader API 削減

`registerReaderThread` / `enterReader` / `exitReader` / `activeReaderCount` の全呼び出し元を `RCUReader` 経由に移行後、API を削除。

---

## 6. 設計判断一覧（v5 確定版）

| # | 論点 | v5 結論 | 根拠 |
|---|------|---------|------|
| 1 | ShutdownPhase | 既存 enum 維持 + `TimedOut/Failed` 追加 | 既存FSMを破壊しない |
| 2 | ShutdownRuntime | `markTimedOut()` `markFailed()` のみ追加 | 既存の充実した実装を流用 |
| 3 | SnapshotCoordinator dtor | **維持**（二段構え、`atomic<bool>` フラグ） | 異常系の最終安全網 |
| 4 | `finalizeShutdown(timedOut)` | **`!drainedWithinBudget`** で呼び出し。多重呼び出しは `exchange(true)` で防止 | 引数逆転バグ修正＋二重実行防止 |
| 5 | `m_shutdownFinalized` | `std::atomic<bool>`、`exchange()` 使用 | 将来の経路変更で非壊れに |
| 6 | releaseResources timeout | `drainAll()` 削除、`tryReclaim()` は維持 | 安全な epoch-based reclaim のみ |
| 7 | waitForDrain jassert | `getPhase() >= AudioStopped` | より正確な結合前提 |
| 8 | DeletionEntry timestamp | **追加しない** | 既存 `oldestPendingAge_` で代替 |
| 9 | pendingRetireCount | `collectDrainAudit` で ring+fallback 合計 | 集計は監査時のみ |
| 10 | `m_currentReaders` / `m_maxConcurrentReadersObserved` | **追加しない** | 利用箇所なし |
| 11 | Reader Long Active | **Phase 2**（既存 `detectStuckReaders` 拡張） | 監視機構の二重化防止 |
| 12 | Publication Stall 初期化 | コンストラクタ＋`prepareToPlay` で `m_lastProgressTimestampUs` 初期化 | 起動直後の誤検出防止 |
| 13 | Publication Stall age 判定 | **`hasDeferredRequest()` 時のみ** | deferred なし時の誤検出防止 |
| 14 | Publication Stall const性 | `updateProgressObservation()` + `isPublicationStalled()` 分離 | 観測と状態変更の分離 |
| 15 | Retire Stall 閾値 | Warning: `hwm`, Error: **`hwm * 2`** | 一時的スパイクでの誤検出防止 |
| 16 | Crossfade Watchdog | **本スコープ外** | 過剰設計回避 |
| 17 | EvidenceRingBuffer | **新設せず**既存証跡系に統合 | `EvidenceExporter` / `emitEvidenceTickNonRt` で十分 |
| 18 | RuntimeHealthMonitor 範囲 | Retire + Publication のみ（Reader は Phase 2） | 責務過多の是正 |
| 19 | `detectStuckReaders()` | 既存 private 死コード → Phase 2 で拡張＋public化 | 二重実装防止 |
| 20 | TimeUtils.h 配置 | **`src/core/TimeUtils.h`** | core は audioengine より低レイヤ。両方から利用可能 |

---

## 7. Phase 1 コード変更ファイル一覧

| 操作 | ファイル | 変更内容 |
|------|---------|---------|
| 修正 | `src/audioengine/ISRShutdown.h` | `ShutdownPhase` に `TimedOut/Failed` 追加、`markTimedOut()/markFailed()` 追加 |
| 修正 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `drainAll()` → `tryReclaim()`, `markTimedOut()`, `finalizeShutdown(!drainedWithinBudget)` |
| 修正 | `src/core/SnapshotCoordinator.h` | `finalizeShutdown()`, `retireCurrentAndTarget()`, `std::atomic<bool> m_shutdownFinalized` + `exchange()` 防御。デストラクタ修正 |
| 修正 | `src/audioengine/AudioEngine.Threading.cpp` | `waitForDrain` jassert 強化、`collectDrainAudit` の `routerPendingRetire` |
| 修正 | `src/audioengine/RuntimePublicationOrchestrator.h` | `updateProgressObservation()`, `isPublicationStalled()`, `resetProgressObservation()` ＋初期化修正 |
| 修正 | `src/audioengine/RuntimePublicationOrchestrator.cpp` | コンストラクタで `m_lastProgressTimestampUs` 初期化 |
| 修正 | `src/audioengine/AudioEngine.h` | `m_healthMonitor` ＋ `onHealthEvent()` 宣言 |
| 修正 | `src/audioengine/AudioEngine.CtorDtor.cpp` | コンストラクタで `m_healthMonitor` 初期化 |
| 修正 | `src/audioengine/AudioEngine.Timer.cpp` | `timerCallback` 内で `m_healthMonitor.tick()` |
| 修正 | `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | `runtimeOrchestrator_->resetProgressObservation()` |
| 新規 | `src/core/TimeUtils.h` | `getCurrentTimeUs()` インライン関数 |
| 新規 | `src/audioengine/RuntimeHealthMonitor.h` | `RuntimeHealthMonitor` クラス |
| 新規 | `src/audioengine/RuntimeHealthMonitor.cpp` | `tick()`, `checkRetireStall()`, `checkPublicationStall()`, `emitEvent()` |

---

## 8. Practical Stable ISR Bridge Runtime 最終評価

| 評価軸 | スコア | 備考 |
|--------|--------|------|
| **既存コードとの整合性** | ⭐⭐⭐⭐⭐ | 既存コードを最大限活用、削除より拡張 |
| **クラッシュ回避** | ⭐⭐⭐⭐⭐ | `drainAll()` 削除、`tryReclaim()` のみで安全 |
| **リーク防止** | ⭐⭐⭐⭐⭐ | 二段構えの安全網。`exchange()` で二重実行防止。timedOut でも retire は実行 |
| **過剰設計の回避** | ⭐⭐⭐⭐⭐ | JUCE公式サンプルにない機構は最小限。Crossfade Watchdog/EvidenceRingBuffer/Reader監視をPhase1から除外 |
| **観測の純粋性** | ⭐⭐⭐⭐⭐ | const/non-const 分離完了。callback 経由で非侵入 |
| **設定との整合性** | ⭐⭐⭐⭐⭐ | 閾値は `retireHighWatermark_` から動的取得。Error = hwm * 2 |
| **誤検出の回避** | ⭐⭐⭐⭐⭐ | Publication Stall age は `hasDeferredRequest()` 時のみ判定。Retire Stall 閾値を緩和 |
| **将来の安全性** | ⭐⭐⭐⭐⭐ | `atomic<bool>` ＋ `exchange()` 採用で多重呼び出し・経路変更に耐性 |
| **実装工数** | ⭐⭐⭐⭐⭐ | 9h（新規ファイル3、修正ファイル9） |

**残存リスク（許容範囲）**:

1. タイムアウト後のリーク — `tryReclaim()` で回収できないエントリはプロセス終了まで残る（設計意図）
2. Snapshot の timedOut retire — reclaim されないが EBR 管理下で安全（設計意図）
3. Reader固着 — Phase 2 で既存 `detectStuckReaders()` を拡張予定

---

## 9. エラッタ

### v4 からの修正点詳細

**🔴 修正1: `finalizeShutdown()` 引数逆転バグ**

```cpp
// 誤 (v4):
m_coordinator.finalizeShutdown(drainedWithinBudget);
// → drainedWithinBudget=true(成功) なのに timedOut=true として渡る

// 正 (v5):
const bool timedOut = !drainedWithinBudget;
m_coordinator.finalizeShutdown(timedOut);
```

**🔴 修正2: 多重呼び出し防御**

```cpp
// v5 で追加:
void finalizeShutdown(bool timedOut) noexcept {
    if (m_shutdownFinalized.exchange(true, std::memory_order_acq_rel))
        return;  // 二重呼び出しは無視
    // ...
}
```

**🟡 修正3: Publication Stall age 条件**

```cpp
// v4: 無条件
// v5: hasDeferredRequest() 時のみ
if (m_orchestrator->hasDeferredRequest()) {
    uint64_t deferredAge = m_orchestrator->getMaxDeferredAgeMs();
    // ...
}
```

**🟡 修正4: Retire Stall Error 閾値**

```cpp
// v4: hwm + hwm/4 (=3840) — 一時的スパイクで到達
// v5: hwm * 2 (=6144) — 実運用で扱いやすい
```

**🟡 修正5: TimeUtils.h 配置**

```cpp
// v4: src/audioengine/TimeUtils.h
// v5: src/core/TimeUtils.h  — 依存方向として自然
```

# ConvoPeq ISR Bridge Runtime 改訂版改修計画書 v3（Practical Stable 反映版）

**作成日**: 2026-06-10
**ベース**: v2計画書 + コードベース実地調査（13ファイル + 5 MCPツール） + レビュー指摘反映
**設計思想**: Practical Stable ISR Bridge Runtime — 実運用で破綻しにくく、かつ過剰設計を排する

---

## 0. 本版での主要変更（v2→v3）

| # | 論点 | v2 | v3 | 理由 |
|---|------|----|----|------|
| A | SnapshotCoordinator dtor | `tryReclaim()` **削除** | `tryReclaim()` **維持**（二段構え） | 異常系の最後の安全網を残す |
| B | releaseResources timeout | `markTimedOut()` + 何もしない | `markTimedOut()` + `tryReclaim()`（`drainAll()` は削除） | クラッシュ回避と無制限リーク防止のバランス |
| C | DeletionEntry timestamp | 新規追加（非atomic） | **追加しない** | 既存 `oldestPendingAge_` で代替可能＋lock-free queue内部汚染回避 |
| D | isPublicationProgressing | constメソッドで状態変更 | `updateProgressObservation()` + `isPublicationStalled()` に分離 | 「観測は状態変更しない」原則を遵守 |
| E | Crossfade Watchdog | 実装必須 | **低優先度／保留** | 進行率監視の方が有効。開始時刻追跡は過剰 |
| F | EvidenceRingBuffer | 新規 `SPSRingBuffer` | **新設せず**既存証跡系を拡張 | 既存 `EvidenceExporter` + `emitEvidenceTickNonRt` で十分 |
| G | Reader Long Active | 固定30秒＋depthのみ | `audioCallbackActiveCount > 0` AND条件追加 | 誤検出防止（prepareToPlay/device restart） |
| H | 全体スコープ | 19項目 | **11項目に集約**（監視系は最小限） | JUCE公式サンプルにない過剰な安定化機構を導入しない |

---

## 1. 基本原則

1. **強制復旧・強制 epoch 前進・強制 reclaim は行わない**
2. **ただしタイムアウト後の `tryReclaim()` は安全な範囲で許容** — epoch-based reclamation はクラッシュしない
3. **`drainAll()`（epoch無視の強制削除）は行わない** — 無制御削除による use-after-free リスク回避
4. **観測は状態を読むだけ、変更しない（Pull型）**
5. **監視は reclaim 系スレッドに依存しない** — 単一障害点回避
6. **既存の安全網を削除しない** — `~SnapshotCoordinator::tryReclaim()` は二段構えの最終防衛線として維持
7. **既存の仕組みを最大限活用する** — `oldestPendingAge_`、`retireQueueDepth_`、`fallbackQueueDepth_` 等
8. **JUCE 公式サンプルにない過剰な安定化機構は導入しない**

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
- ❌ **Crossfade Watchdog の開始時刻方式**（進行率監視に限定。ただし本v3では Crossfade Watchdog 全体を保留）
- ❌ フォールバックキューを `peekOldestTimestampUs()` の監視対象に含める

---

## 3. 既存コード調査に基づく判断

### 3.1 `DeferredDeletionQueue` のタイムスタンプ → 追加しない

**判断**: `DeletionEntry` に `enqueueTimestampUs` を追加しない。

**理由**:

1. `DeletionEntry` は `is_trivially_copyable_v` の static_assert あり。`std::atomic<uint64_t>` は違反
2. 非atomicにしても lock-free queue の中間状態（enqueue 中の書き込み完了前）を `peekOldestTimestampUs()` が読む可能性がある
3. **既存の代替手段が揃っている**:
   - `AudioEngine::oldestPendingAge_`（`std::atomic<double>`）— 最古保留中の retire intent 経過時間（ms）
   - `AudioEngine::retireQueueDepth_` — retire queue 深度
   - `AudioEngine::fallbackQueueDepth_` — fallback queue 深度
   - `DeferredDeletionQueue::sizeApprox()` — ring buffer 深度
   - `DeferredDeletionQueue::getMaxRetireAgeUs()` — 過去最大滞留時間（us、参考値）

### 3.2 `oldestPendingAge_` の仕組み（既存）

`AudioEngine::onRuntimeRetiredNonRt()` 内で更新:

```cpp
// 保留中 retire intent がある場合
oldestPendingAge_ = nowMs - oldestPendingFirstSeenMs_;
// 保留中がない場合
oldestPendingAge_ = 0.0;
```

→ これが**正確に最古保留中の経過時間を表す**。新規機構不要。

### 3.3 SnapshotCoordinator デストラクタ（現状維持＋拡張）

現状:

```cpp
~SnapshotCoordinator() noexcept {
    // retire both current + target snapshots
    m_epochProvider->tryReclaim();  // ← 最終安全網
}
```

**方針**: `tryReclaim()` を維持しつつ、`finalizeShutdown()` フラグで二重実行を防止する。

```cpp
~SnapshotCoordinator() noexcept {
    if (!m_shutdownFinalized) {
        // retire + tryReclaim (same as current)
    }
}
```

### 3.4 既存の `waitForDrain()` は流用

`AudioEngine.Threading.cpp:80` — 既存実装をそのまま使用。
変更点: 呼び出し前に `jassert` を追加（結合前提の担保のみ）。

### 3.5 既存証跡系

以下の経路が既に存在:

- `EvidenceExporter::exportEvidence()` — テレメトリファイル出力
- `emitEvidenceTickNonRt(bool force)` — 1秒間隔制御付き証跡出力
- `retireRuntimeEx_.emitRetireTimeline()` — retire タイムライン出力
- `collectDrainAudit()` — shutdown 時監査情報収集

→ `RuntimeHealthMonitor` は「監視」のみを行い、証跡出力は既存経路を拡張して利用する。

---

## 4. 実装計画

### Phase 1（最優先・実装必須）— 監視機構追加

#### P1-1: ShutdownPhase 拡張（0.5h）

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

**注意**: `transitionTo()` が既存7状態間の遷移チェックを持つ場合、`TimedOut/Failed` への遷移を許可するよう修正。

#### P1-2: releaseResources タイムアウト処理の修正（1.5h）

**ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`

**変更内容**: タイムアウト時、`drainAll()` を削除し `tryReclaim()` に置換。

```cpp
// 変更前
if (!drainedWithinBudget || !isFullyDrained()) {
    if (!drainedWithinBudget)
        diagLog("[DIAG] releaseResources: drain timeout reached, ...");
    drainDeferredRetireQueues(true);
    m_epochDomain.drainAll();           // ← 強制reclaim（危険）
}

// 変更後
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
```

**根拠**: `tryReclaim()` は epoch ベースで安全に削除可能なエントリのみ処理する。`drainAll()` は epoch を無視するため、use-after-free リスクがある。タイムアウト後のリークは許容範囲。

#### P1-3: SnapshotCoordinator 二段構え化（1h）

**ファイル**: `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp`

```cpp
class SnapshotCoordinator {
    bool m_shutdownFinalized = false;   // ★ 追加
public:
    // ★ 追加: releaseResources から呼ばれる
    void finalizeShutdown(bool timedOut) noexcept {
        if (timedOut) {
            m_shutdownFinalized = true;  // リーク許容、何もしない
            return;
        }
        // 通常: 両方のスナップをretire + tryReclaim
        constexpr auto deleter = [](void* p) noexcept {
            SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(p));
        };
        const uint64_t retireEpoch = m_epochProvider->publishEpoch();
        GlobalSnapshot* snap = m_slots.exchangeCurrent(nullptr, std::memory_order_acq_rel);
        if (snap) m_epochProvider->enqueueRetire(snap, deleter, retireEpoch);
        snap = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
        if (snap) m_epochProvider->enqueueRetire(snap, deleter, retireEpoch);
        m_epochProvider->tryReclaim();
        m_shutdownFinalized = true;
    }

    // ★ デストラクタは現状維持 + フラグチェック
    ~SnapshotCoordinator() noexcept {
        if (m_shutdownFinalized)
            return;  // finalizeShutdown で処理済み
        // 異常系: 最後の安全網として retire + tryReclaim
        constexpr auto deleter = [](void* p) noexcept {
            SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(p));
        };
        const uint64_t retireEpoch = m_epochProvider->publishEpoch();
        GlobalSnapshot* snap = m_slots.exchangeCurrent(nullptr, std::memory_order_acq_rel);
        if (snap) m_epochProvider->enqueueRetire(snap, deleter, retireEpoch);
        snap = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
        if (snap) m_epochProvider->enqueueRetire(snap, deleter, retireEpoch);
        m_epochProvider->tryReclaim();
    }
};
```

```cpp
// releaseResources での呼び出し
const bool drainedWithinBudget = waitForDrain(2000, 2);
if (!drainedWithinBudget)
    shutdownRuntime_.markTimedOut();
m_coordinator.finalizeShutdown(drainedWithinBudget);  // ★ 追加
```

#### P1-4: waitForDrain 結合前提の jassert 追加（0.5h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();
    // ★ 結合前提: オーディオコールバック停止後にのみ呼ばれることを保証
    jassert(convo::consumeAtomic(rtLocalState_.audioCallbackActiveCount,
                                 std::memory_order_acquire) == 0);
    // ... 既存実装 ...
}
```

#### P1-5: EpochDomain ReaderSlot 拡張（1.5h）

**ファイル**: `src/core/EpochDomain.h`

```cpp
struct ReaderSlot {
    std::atomic<uint64_t> epoch { kInactiveEpoch };
    std::atomic<uint32_t> depth { 0 };
    std::atomic<uint64_t> enterCount { 0 };              // 既存維持
    std::atomic<uint64_t> lastEnterTimestampUs { 0 };     // ★ 追加
};

// ★ 追加: 同時アクティブReader統計
std::atomic<uint32_t> m_currentReaders { 0 };
std::atomic<uint32_t> m_maxConcurrentReadersObserved { 0 };

// getReaderSnapshot メソッド追加
struct ReaderSlotSnapshot {
    uint64_t lastEnterTimestampUs;
    uint32_t depth;
};
ReaderSlotSnapshot getReaderSnapshot(int idx) const noexcept {
    return {
        convo::consumeAtomic(readers[idx].lastEnterTimestampUs, std::memory_order_acquire),
        convo::consumeAtomic(readers[idx].depth, std::memory_order_acquire)
    };
}
```

**enterReader の修正**（`prevDepth == 0` の分岐内のみ）:

```cpp
if (previousDepth > 0) return;  // ネスト: 何もしない

// ★ 追加: depth 0→1 の時のみタイムスタンプ記録 + カウンタ更新
convo::publishAtomic(slot.lastEnterTimestampUs, getCurrentTimeUs(), std::memory_order_release);
uint32_t cur = m_currentReaders.fetch_add(1u, std::memory_order_relaxed) + 1u;
uint32_t max = m_maxConcurrentReadersObserved.load(std::memory_order_relaxed);
while (cur > max && !m_maxConcurrentReadersObserved.compare_exchange_weak(
    max, cur, std::memory_order_relaxed, std::memory_order_relaxed)) {}

const uint64_t epoch = currentEpoch();
convo::publishAtomic(slot.epoch, epoch, std::memory_order_release);
```

**exitReader の修正**（`prevDepth == 1` の分岐内のみ）:

```cpp
if (previousDepth > 1) return;  // ネスト: 何もしない

// ★ 追加: depth 1→0 の時のみカウンタ減少
m_currentReaders.fetch_sub(1u, std::memory_order_relaxed);
convo::publishAtomic(slot.epoch, kInactiveEpoch, std::memory_order_release);
```

#### P1-6: RuntimeHealthMonitor 新設（2h）

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

class EpochDomain;
namespace isr {
class ISRRetireRouter;
class RuntimePublicationOrchestrator;
}

// ★ イベントコード定数（RuntimeHealthMonitor.h に定義）
static constexpr uint32_t EVENT_RETIRE_STALL           = 1001;
static constexpr uint32_t EVENT_RETIRE_STALL_WARNING   = 1002;
static constexpr uint32_t EVENT_PUBLICATION_STALL      = 2001;
static constexpr uint32_t EVENT_PUBLICATION_WARNING    = 2002;
static constexpr uint32_t EVENT_READER_LONG_ACTIVE     = 3001;

using HealthEventCallback = std::function<void(const HealthEvent&)>;

/**
 * RuntimeHealthMonitor: Pull型監視エンジン。
 * timerCallback から tick() を呼び出し、各チェックを実行する。
 * 検出イベントは callback 経由で通知（既存証跡系へ統合）。
 *
 * 設計原則:
 * - 観測は状態を読むだけ、変更しない
 * - reclaim系スレッドに依存しない
 * - 強制復旧・強制reclaimは行わない
 */
class RuntimeHealthMonitor {
public:
    void setEpochDomain(EpochDomain* domain) noexcept { m_epochDomain = domain; }
    void setRetireRouter(isr::ISRRetireRouter* router) noexcept { m_retireRouter = router; }
    void setOrchestrator(isr::RuntimePublicationOrchestrator* orch) noexcept { m_orchestrator = orch; }
    void setEventCallback(HealthEventCallback cb) noexcept { m_callback = std::move(cb); }

    // AudioEngine::getAudioCallbackActiveCount() への参照
    void setAudioCallbackActiveCountRef(const std::atomic<uint32_t>* ref) noexcept {
        m_audioCallbackActiveCountRef = ref;
    }

    void tick() noexcept;  // timerCallback から呼ばれる

private:
    void checkRetireStall() noexcept;
    void checkPublicationStall() noexcept;
    void checkReaderLongActive() noexcept;
    void emitEvent(HealthEvent::Severity severity, uint32_t eventCode,
                   uint64_t value, uint32_t slot = 0) noexcept;

    EpochDomain* m_epochDomain = nullptr;
    isr::ISRRetireRouter* m_retireRouter = nullptr;
    isr::RuntimePublicationOrchestrator* m_orchestrator = nullptr;
    const std::atomic<uint32_t>* m_audioCallbackActiveCountRef = nullptr;
    HealthEventCallback m_callback;
};

} // namespace convo
```

**ファイル**: 新規 `src/audioengine/RuntimeHealthMonitor.cpp`

```cpp
#include "RuntimeHealthMonitor.h"
#include "core/EpochDomain.h"
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/TimeUtils.h"
#include "audioengine/AtomicAccess.h"

namespace convo {

void RuntimeHealthMonitor::tick() noexcept {
    checkRetireStall();
    checkPublicationStall();
    checkReaderLongActive();
}

void RuntimeHealthMonitor::checkRetireStall() noexcept {
    if (!m_retireRouter) return;
    uint32_t pendingCount = m_retireRouter->pendingRetireCount();

    // queue depth による簡易監視（タイムスタンプ不要）
    // 閾値: 3072 (retireHighWatermark のデフォルト値に準拠)
    if (pendingCount > 3072) {
        emitEvent(HealthEvent::Severity::Warning, EVENT_RETIRE_STALL_WARNING,
                  pendingCount);
    }
    if (pendingCount > 4096) {
        emitEvent(HealthEvent::Severity::Error, EVENT_RETIRE_STALL,
                  pendingCount);
    }
    // ※ 経過時間監視は AudioEngine::oldestPendingAge_ を
    //    collectDrainAudit() 経由で参照（RuntimeHealthMonitor の責務外）
}

void RuntimeHealthMonitor::checkPublicationStall() noexcept {
    if (!m_orchestrator) return;

    // 観測フェーズと状態更新フェーズを分離
    m_orchestrator->updateProgressObservation();

    if (m_orchestrator->getPendingIntentCount() > 0
        && m_orchestrator->isPublicationStalled()) {
        emitEvent(HealthEvent::Severity::Error, EVENT_PUBLICATION_STALL, 0);
    }

    uint64_t deferredAge = m_orchestrator->getMaxDeferredAgeMs();
    if (deferredAge > 30000) {
        emitEvent(HealthEvent::Severity::Error, EVENT_PUBLICATION_STALL, deferredAge);
    } else if (deferredAge > 5000) {
        emitEvent(HealthEvent::Severity::Warning, EVENT_PUBLICATION_WARNING, deferredAge);
    }
}

void RuntimeHealthMonitor::checkReaderLongActive() noexcept {
    if (!m_epochDomain) return;

    // Audio コールバックがアクティブでない場合は判定しない
    // （prepareToPlay 待機中や device restart 時の誤検出防止）
    bool audioActive = (m_audioCallbackActiveCountRef != nullptr)
        && convo::consumeAtomic(*m_audioCallbackActiveCountRef,
                                std::memory_order_acquire) > 0;
    if (!audioActive) return;

    uint64_t nowUs = getCurrentTimeUs();
    for (int slot = 0; slot < EpochDomain::kMaxReaders; ++slot) {
        auto snap = m_epochDomain->getReaderSnapshot(slot);
        // depth > 0 かつ 30秒以上経過
        if (snap.depth > 0 && (nowUs - snap.lastEnterTimestampUs) > 30'000'000) {
            emitEvent(HealthEvent::Severity::Warning, EVENT_READER_LONG_ACTIVE,
                      (nowUs - snap.lastEnterTimestampUs) / 1000, slot);
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

#### P1-7: RuntimePublicationOrchestrator 監視機能（1h）

**ファイル**: `src/audioengine/RuntimePublicationOrchestrator.h`

```cpp
// const メソッドと非constメソッドを分離（「観測は状態変更しない」原則）

class RuntimePublicationOrchestrator {
    // ★ 追加: 進捗観測用フィールド
    std::atomic<PublicationSequenceId> m_lastObservedSequence {0};
    std::atomic<uint64_t> m_lastProgressTimestampUs {0};

public:
    // ★ 追加: 進捗観測の更新（非const、timerCallback から呼ぶ）
    void updateProgressObservation() noexcept {
        PublicationSequenceId current = engine_.getLastCommittedPublicationSequence();
        PublicationSequenceId last = m_lastObservedSequence.load(std::memory_order_relaxed);
        if (current > last) {
            m_lastObservedSequence.store(current, std::memory_order_relaxed);
            m_lastProgressTimestampUs.store(getCurrentTimeUs(), std::memory_order_relaxed);
        }
    }

    // ★ 追加: 出版停滞検出（const、read-only）
    bool isPublicationStalled() const noexcept {
        uint64_t elapsed = getCurrentTimeUs()
            - m_lastProgressTimestampUs.load(std::memory_order_acquire);
        return elapsed >= 30'000'000;  // 30秒以上進捗なし → 停滞
    }
};
```

#### P1-8: TimeUtils.h 新規作成（0.5h）

**ファイル**: 新規 `src/audioengine/TimeUtils.h`

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

#### P1-9: AudioEngine 統合（1.5h）

**ファイル**: `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.CtorDtor.cpp`, `src/audioengine/AudioEngine.Timer.cpp`

```cpp
// AudioEngine.h: メンバ追加
#include "audioengine/RuntimeHealthMonitor.h"

// AudioEngine クラス内
convo::RuntimeHealthMonitor m_healthMonitor;

// AudioEngine クラス内にコールバック追加
void onHealthEvent(const convo::HealthEvent& event) noexcept;
```

```cpp
// AudioEngine.CtorDtor.cpp: コンストラクタで初期化
// m_retireRouter / runtimeOrchestrator_ 初期化後に追加
m_healthMonitor.setEpochDomain(&m_epochDomain);
m_healthMonitor.setRetireRouter(m_retireRouter.get());
m_healthMonitor.setOrchestrator(runtimeOrchestrator_.get());
m_healthMonitor.setAudioCallbackActiveCountRef(
    &rtLocalState_.audioCallbackActiveCount);
m_healthMonitor.setEventCallback(
    [this](const convo::HealthEvent& ev) { onHealthEvent(ev); });
```

```cpp
// AudioEngine.Timer.cpp: timerCallback 末尾に追加
// ※ emitEvidenceTickNonRt の直前または直後に配置
if (!isShutdownInProgress()) {
    m_healthMonitor.tick();
}
```

```cpp
// AudioEngine.h または .cpp にコールバック実装
void AudioEngine::onHealthEvent(const convo::HealthEvent& event) noexcept {
    // 既存証跡系へ統合
    // 例: diagLog / DBG 出力
    // 必要に応じて evidenceExporter_ へ渡すことも可能
    if (event.severity >= convo::HealthEvent::Severity::Warning) {
        diagLog("[HEALTH] eventCode=" + juce::String(static_cast<int>(event.eventCode))
            + " severity=" + juce::String(static_cast<int>(event.severity))
            + " value=" + juce::String(static_cast<juce::int64>(event.value))
            + " slot=" + juce::String(static_cast<int>(event.slot)));
    }
}
```

#### P1-10: collectDrainAudit への routerPendingRetire 追加（0.5h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
convo::isr::RuntimeDrainAudit AudioEngine::collectDrainAudit() noexcept
{
    return convo::isr::RuntimeDrainAudit{
        .pendingPublication = runtimePublicationBridge_.getPublicationBacklogCount(),
        .pendingRetire = retireRuntime_.pendingIntentCount(),
        .activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u,
        .routerPendingRetire = static_cast<uint64_t>(m_retireRouter->pendingRetireCount())
            + convo::consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire),  // ★ 修正
        .maxDeferredAgeMs = runtimeOrchestrator_
            ? runtimeOrchestrator_->getMaxDeferredAgeMs() : 0u,
        // ...
    };
}
```

---

### Phase 1 工数サマリー

| # | 項目 | 工数 |
|---|------|------|
| P1-1 | ShutdownPhase + ShutdownRuntime 拡張 | 0.5h |
| P1-2 | releaseResources タイムアウト処理修正 | 1.5h |
| P1-3 | SnapshotCoordinator 二段構え化 | 1h |
| P1-4 | waitForDrain jassert 追加 | 0.5h |
| P1-5 | EpochDomain ReaderSlot 拡張 | 1.5h |
| P1-6 | RuntimeHealthMonitor 新設 | 2h |
| P1-7 | RuntimePublicationOrchestrator 監視機能 | 1h |
| P1-8 | TimeUtils.h 新規作成 | 0.5h |
| P1-9 | AudioEngine 統合 | 1.5h |
| P1-10 | collectDrainAudit 修正 | 0.5h |
| | **Phase 1 合計** | **10.5h** |

---

## 5. 設計判断一覧（v3 版）

| # | 論点 | v3 結論 | 根拠 |
|---|------|---------|------|
| 1 | ShutdownPhase | 既存 enum 維持 + `TimedOut/Failed` 追加 | 既存FSMを破壊しない |
| 2 | ShutdownRuntime | `markTimedOut()` `markFailed()` のみ追加 | 既存の充実した実装を流用 |
| 3 | SnapshotCoordinator dtor | `tryReclaim()` **維持**（二段構え） | 異常系の最終安全網を残す |
| 4 | releaseResources timeout | `drainAll()` 削除、`tryReclaim()` は維持 | 安全な epoch-based reclaim のみ |
| 5 | waitForDrain | 既存実装を流用 + `jassert` | 結合前提の確認のみ |
| 6 | DeletionEntry timestamp | **追加しない** | 既存 `oldestPendingAge_` で代替 |
| 7 | pendingRetireCount | `collectDrainAudit` で ring+fallback 合計 | 集計は監査時のみ |
| 8 | EpochDomain ReaderSlot | `lastEnterTimestampUs` + `m_currentReaders` 追加 | Reader Long Active 検出に必要 |
| 9 | Reader Long Active | `audioCallbackActiveCount > 0` AND 条件 | 誤検出防止 |
| 10 | Publication Watchdog | `updateProgressObservation` + `isPublicationStalled` 分離 | const違反の解消 |
| 11 | Crossfade Watchdog | **保留** | 開始時刻追跡より進行率監視の方が有効だが、本スコープでは過剰 |
| 12 | EvidenceRingBuffer | **新設せず**既存証跡系に統合 | `EvidenceExporter` / `emitEvidenceTickNonRt` で十分 |
| 13 | RuntimeHealthMonitor | 新設、callback で既存証跡系へ統合 | 監視と証跡出力の責務分離 |

---

## 6. コードベース照合チェックリスト（最終版）

- [x] `DeferredDeletionQueue::maxRetireAgeUs_` 存在確認（`getMaxRetireAgeUs` / `updateMaxRetireAge`／**ただし呼び出し元なし＝dead API**）
- [x] `AudioEngine::oldestPendingAge_` 存在確認（`onRuntimeRetiredNonRt` で更新中、正確に最古保留中経過時間を追跡）
- [x] `AudioEngine::oldestPendingFirstSeenMs_` 存在確認
- [x] `AudioEngine::retireQueueDepth_` / `fallbackQueueDepth_` 存在確認
- [x] `SnapshotCoordinator::~SnapshotCoordinator()` が `tryReclaim()` を呼んでいることを確認
- [x] `EpochDomain::ReaderSlot::enterCount` 存在確認
- [x] `EpochDomain::detectStuckReaders()` 存在確認
- [x] `ShutdownRuntime` の充実した実装を確認
- [x] `waitForDrain()` の既存実装を確認
- [x] `releaseResources()` の既存タイムアウト処理（`drainAll()` 呼び出し中）を確認
- [x] `RuntimePublicationOrchestrator::deferredSlot_` / `getMaxDeferredAgeMs()` 存在確認
- [x] `ISREvidenceExporter` / `emitEvidenceTickNonRt` 存在確認
- [x] `AudioEngine::rtLocalState_.audioCallbackActiveCount` 存在確認
- [x] `DeletionEntry` の `is_trivially_copyable_v` static_assert 確認
- [x] `IEpochProvider` インターフェースの純粋仮想関数一覧確認（`currentEpoch`, `publishEpoch`, `getMinReaderEpoch`, `enqueueRetire`, `tryReclaim`, `registerReaderThread`, `reserveReaderThread`, `enterReader`, `exitReader`, `activeReaderCount`）

---

## 7. v2 からの削減項目と理由

| v2 項目 | v3 状態 | 削減理由 |
|---------|---------|---------|
| `DeletionEntry::enqueueTimestampUs` 追加 | **削除** | 既存 `oldestPendingAge_` で代替、lock-free queue 内部汚染回避 |
| `EvidenceRingBuffer` (SPSRingBuffer) 新設 | **削除** | 既存証跡系で十分。新規リングバッファは過剰 |
| `CrossfadeRuntime::m_startTimestampUs` | **保留** | 開始時刻追跡より進行率監視の方が有効だが、Crossfade Watchdog 全体を本スコープ外に |
| Crossfade Watchdog (checkCrossfadeStall) | **保留** | 同上 |
| `ISRRetireRouter::getOldestPendingRetireAgeMs()` | **削除** | `oldestPendingAge_` + queue depth で代替 |
| Retire Stall Monitor (timestamp-based) | **簡略化** | queue depth 監視に変更（タイムスタンプ不要） |
| `EpochDomain::peekOldestDeferredTimestampUs()` | **削除** | `DeletionEntry` にタイムスタンプを追加しないため不要 |

---

## 8. Practical Stable ISR Bridge Runtime 評価

| 評価軸 | スコア | 備考 |
|--------|--------|------|
| **既存コードとの整合性** | ⭐⭐⭐⭐⭐ | 既存コードを最大限活用、削除より拡張 |
| **クラッシュ回避** | ⭐⭐⭐⭐⭐ | `drainAll()` 削除、`tryReclaim()` のみで安全 |
| **リーク防止** | ⭐⭐⭐⭐ | 二段構えの安全網（正常系 + デストラクタ） |
| **過剰設計の回避** | ⭐⭐⭐⭐⭐ | JUCE公式サンプルにない機構は最小限 |
| **観測の純粋性** | ⭐⭐⭐⭐⭐ | const/non-const 分離完了。callback 経由で非侵入 |
| **実装工数** | ⭐⭐⭐⭐ | v2 の 19h → v3 の 10.5h に削減 |

**残存リスク**:

1. タイムアウト後のリーク（`tryReclaim()` で回収できないエントリ）はプロセス終了まで残る → 許容範囲（設計意図）
2. `m_shutdownFinalized` フラグのスレッド安全性 → いずれも同一スレッド（Message Thread）から呼ばれるため安全
3. `oldestPendingAge_` は retire intent の経過時間であり、`DeferredDeletionQueue` 内部エントリの経過時間ではない → 実用上十分な近似値

---

## 9. 今後の課題（本スコープ外）

- Crossfade 進行率監視（既存 `LinearRamp` の進行状況を確認する方式）
- Evidence 出力の拡張（`RuntimeHealthMonitor` の callback を `EvidenceExporter` に接続）
- `DeferredDeletionQueue::updateMaxRetireAge()` の呼び出し元追加（現状 dead API）

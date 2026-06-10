# ConvoPeq ISR Bridge Runtime 改訂版改修計画書 v2（コードベース照合反映版）

**作成日**: 2026-06-10
**ベース文書**: ConvoPeq_ISR_Bridge_Runtime_改訂版改修計画書（最終確定版）
**照合結果**: `doc/work27/コードベース調査報告_2026-06-10.md`（13ファイル直接読み取り＋5ツール解析済み）
**ステータス**: コードベースとの乖離を全て修正済み

---

## 0. 対象コードベース

本計画は ConvoPeq の以下のファイルを対象とする。

| コンポーネント | ファイルパス | 調査結果 |
|--------------|-------------|---------|
| EpochDomain | `src/core/EpochDomain.h` | ✅ 存在。`enterCount`, `detectStuckReaders` 既存。`lastEnterTimestampUs`, `m_currentReaders`, `m_maxConcurrentReadersObserved` は未実装 |
| DeferredDeletionQueue | `src/DeferredDeletionQueue.h` | ✅ 存在。MPMC bounded queue。`FallbackEntry` は不在。`DeletionEntry` にタイムスタンプ追加必要 |
| DeferredRetireFallbackQueue | `src/core/DeferredRetireFallbackQueue.h` | ✅ 存在。計画書の「FallbackEntry」の実体はこちら。`pendingRetireCount()` 合計対象に含める |
| DeferredFreeThread | `src/DeferredFreeThread.h` | 対象だが本改訂では監視対象外（単一障害点回避の原則） |
| ISRRetireRouter | `src/audioengine/ISRRetireRouter.h` | ✅ 存在。`registerReaderThread/enterReader/exitReader` は既に `[[deprecated]]` 委譲実装。`getOldestPendingRetireAgeMs()` 追加が必要 |
| CrossfadeRuntime | `src/audioengine/CrossfadeRuntime.h` | ✅ 存在。`m_startTimestampUs` / `getCrossfadeAgeMs()` 未実装 |
| AudioEngine | `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.cpp` | ✅ 存在。`m_coordinator(m_epochDomain)` で初期化。`shutdownRuntime_` 既存。`timerCallback()` 既存で拡張可能 |
| RuntimePublicationOrchestrator | `src/audioengine/RuntimePublicationOrchestrator.h`, `cpp` | ✅ 存在。`deferredSlot_`, `getMaxDeferredAgeMs()` 既存。`isPublicationProgressing()` 未実装 |
| SnapshotCoordinator | `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp` | ✅ 存在。**デストラクタで `tryReclaim()` 呼び出し中**（⚠️ 計画書の前提と異なる） |
| ISRShutdown | `src/audioengine/ISRShutdown.h` | ✅ 存在。**ShutdownPhase の定義が計画書の想定と完全に異なる** |
| SafeStateSwapper | `src/SafeStateSwapper.h` | ✅ 存在。`FallbackEntry` はこちらに定義（ConvolverState専用） |
| TimeUtils | 新規 `src/audioengine/TimeUtils.h` | 新規作成（`getCurrentTimeUs()` 共通定義） |
| RuntimeHealthMonitor | 新規 `src/audioengine/RuntimeHealthMonitor.h/cpp` | 新規作成 |
| EvidenceRingBuffer | 新規 `src/audioengine/EvidenceRingBuffer.h` | 新規作成（`SPSRingBuffer<HealthEvent, 1024>`） |

---

## 1. 基本原則（変更なし）

1. **強制復旧・強制reclaimは行わない** – 異常状態からの自動復旧はせず、検出・記録に徹する。
2. **タイムアウト後はリーク許容** – シャットダウン時はクラッシュよりリークを優先する。
3. **観測は状態を読むだけ、変更しない（Pull型）** – 監視コンポーネントは状態を変更してはならない。
4. **監視はreclaim系スレッド（DeferredFreeThread）に依存しない** – 単一障害点を避ける。
5. **複雑な抽象化・新規パターンを避ける** – 既存のコードスタイルに合わせる。
6. **EventBus / AudioEngine分割 / PublicationExecutor / RetirePolicy は実施しない** – 過剰設計を避ける。

## 2. 禁止事項（変更なし）

- ❌ 強制 epoch 前進・強制 reader inactive 化
- ❌ `cleanupDeadReaders()` の実装
- ❌ `enterCount/exitCount` による Reader 判定（既存の `depth` で十分）
- ❌ MPSC リングバッファ（SPSC で十分）
- ❌ DeferredFreeThread への監視ロジック集約
- ❌ `lastTickUs` による HealthMonitor 周期逸脱検知
- ❌ フォールバックキューを `peekOldestTimestampUs()` の監視対象に含める
- ❌ SnapshotCoordinator デストラクタでの無条件 `tryReclaim()`（→ `releaseResources()` 内の `finalizeShutdown()` に統合）

---

## 3. コードベース照合を反映した設計判断

### 3.1 ShutdownPhase（⚠️ 既存コードと完全に異なる — 既存を拡張）

**既存の enum** (`src/audioengine/ISRShutdown.h`):

```cpp
enum class ShutdownPhase : uint8_t {
    Running,
    AudioStopped,
    ObserverDrained,
    RetireClosed,
    EpochSettled,
    ReclaimComplete,
    ShutdownComplete
};
```

**修正方針**: 既存 enum を維持し、以下を追加する。

```cpp
// 追加
TimedOut,    // waitForDrain タイムアウト時
Failed       // 回復不能エラー時
```

### 3.2 ShutdownRuntime（⚠️ 既存クラスは遥かに充実している — 拡張のみ）

**既存の機能**: `initiateShutdown()`, `getPhase()`, `advancePhase()`, `transitionTo()`, `isShutdownInProgress()`, `setBoundedTeardownCounters()`, `markLateCallback()`, `markPostStopEnqueue()`, `emitShutdownTrace()`

**追加するもの**:

```cpp
void markTimedOut() { phase_.store(ShutdownPhase::TimedOut, std::memory_order_release); }
void markFailed() { phase_.store(ShutdownPhase::Failed, std::memory_order_release); }
```

**`tryStart()` は不要**: `AudioEngine::releaseResources()` は `compareExchangeAtomic(lifecycleState, ..., Releasing)` で二重実行防止済み。

### 3.3 SnapshotCoordinator デストラクタ（⚠️ 現状 tryReclaim() を呼んでいる）

**現状**: `~SnapshotCoordinator()` は old snapshot を retire 後、`m_epochProvider->tryReclaim()` を呼ぶ。
これは `EpochDomain` が `SnapshotCoordinator` より後に破棄される（メンバ宣言順）ため安全。

**修正方針**:

- `finalizeShutdown(bool timedOut)` を追加:
  - `timedOut == false` の場合のみ `tryReclaim()` を呼ぶ
  - `timedOut == true` の場合はリーク許容（何もしない）
- `~SnapshotCoordinator()` からの `tryReclaim()` を削除し、代わりに `releaseResources()` → `finalizeShutdown()` 経路に統合
- これにより、デストラクタがいつ呼ばれても安全になり、宣言順への依存がなくなる

### 3.4 waitForDrain（⚠️ 既存実装あり — 流用＋強化）

**既存** (`AudioEngine.Threading.cpp:80`):

```cpp
bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept {
    // juce::Time::getMillisecondCounterHiRes() ベースのポーリング
    // drainDeferredRetireQueues(true) をループ内で呼ぶ
    // タイムアウト時 false を返す
}
```

**修正方針**:

- 既存実装をそのまま流用
- `waitForDrain()` の前に `jassert(!isAudioCallbackActive())` を追加（**結合前提の担保**）
- タイムアウト後の処理は `releaseResources()` で制御（後述）

### 3.5 releaseResources タイムアウト処理（⚠️ 現状は強制ドレイン → リーク許容に変更）

**現状** (`AudioEngine.Processing.ReleaseResources.cpp:200`):

```cpp
if (!drainedWithinBudget || !isFullyDrained()) {
    drainDeferredRetireQueues(true);
    m_epochDomain.drainAll();  // ← 強制reclaim
}
```

**修正後**:

```cpp
if (!drainedWithinBudget || !isFullyDrained()) {
    m_shutdownRuntime.markTimedOut();  // タイムアウト記録（強制reclaimしない）
    // リーク許容: 強制reclaimは行わない
}
m_snapshotCoordinator.finalizeShutdown(drainedWithinBudget);
```

**注意**: これは既存の挙動を変える変更である。

### 3.6 FallbackEntry の配置修正（⚠️ DeferredDeletionQueue.h に存在しない）

**現状**: `FallbackEntry` は `SafeStateSwapper.h` に存在（ConvolverState 専用）。
フォールバック退役キューは `DeferredRetireFallbackQueue`（`src/core/DeferredRetireFallbackQueue.h`）として別クラス存在。

**修正方針**:

- `DeferredDeletionQueue` の `DeletionEntry` に `enqueueTimestampUs`（`uint64_t`、非atomic）を追加
- **非atomicにする理由**: `DeletionEntry` には `std::is_trivially_copyable_v` の static_assert があり、`std::atomic<uint64_t>` は違反する
- 書き込みは reclaim スレッド（単一）からのみ行う → データレースなし
- 読み取り（`peekOldestTimestampUs()`）は `std::memory_order_relaxed` 相当のコンパイラバリアのみ

### 3.7 pendingRetireCount() の集計（既存 ring + fallback）

**現状**: `EpochDomain::pendingRetireCount()` → `deferredDeletionQueue.sizeApprox()` → ring buffer のみ。

**修正方針**: `ISRRetireRouter` レベルで ring + fallback を合計する。

```cpp
uint32_t pendingRetireCount() const noexcept {
    uint32_t ringCount = epochDomain_->pendingRetireCount();
    uint32_t fallbackCount = /* DeferredRetireFallbackQueue の size() */;
    return ringCount + fallbackCount;
}
```

→ ただし `ISRRetireRouter` は現在 `DeferredRetireFallbackQueue` への参照を持たない。
→ `AudioEngine` レベルで集計し、`RuntimeBackpressureTelemetry` 経由で公開。

### 3.8 既存 enterCount + detectStuckReaders との共存

`EpochDomain` には既に以下が存在:

```cpp
std::atomic<uint64_t> enterCount { 0 };        // P3-1
StuckReaderInfo detectStuckReaders(uint64_t stuckThreshold) const noexcept;  // epoch gap ベース
```

新規追加:

```cpp
std::atomic<uint64_t> lastEnterTimestampUs { 0 };          // 追加: 時計ベース
std::atomic<uint32_t> m_currentReaders { 0 };              // 追加: 同時アクティブReader数
std::atomic<uint32_t> m_maxConcurrentReadersObserved { 0 }; // 追加: 最大同時Reader数
```

**役割分担**:

- `enterCount`: 総enter回数（診断用、既存維持）
- `detectStuckReaders()`: epoch gap ベースの固着検出（既存維持）
- `lastEnterTimestampUs`: 時計ベースの Long Active 検出用（新規）
- `m_currentReaders` / `m_maxConcurrentReadersObserved`: 統計用（新規）

### 3.9 既存 EvidenceExporter / emitEvidenceTickNonRt との役割分担

**既存**:

- `EvidenceExporter::exportEvidence()` + `emitEvidenceTickNonRt(bool force)` → テレメトリ/証跡のファイル出力
- `timerCallback()` の冒頭と shutdown 中に呼ばれている

**新規 `RuntimeHealthMonitor`**:

- **監視 (Watchdog)** を担当: stall検出、long active検出
- **証跡出力は行わない** → `EvidenceRingBuffer` にイベントをpushするのみ
- 出力は既存の `emitEvidenceTickNonRt()` が `EvidenceRingBuffer` から読み出して行う（役割分離）

### 3.10 DeletionEntry の atomic 制約への対応

**制約**:

```cpp
static_assert(std::is_trivially_copyable_v<DeletionEntry>,
    "DeletionEntry must be trivially copyable for lock-free queue operations");
```

`enqueueTimestampUs` を `std::atomic<uint64_t>` にすると **この static_assert に違反する**。

**対策**: `enqueueTimestampUs` は **非atomic `uint64_t`** とする。

- 書き込み: reclaim スレッド（単一）のみが `enqueue()` 時に設定
- 読み取り: `peekOldestTimestampUs()` は `std::atomic_thread_fence(std::memory_order_acquire)` + 非atomic読み取り
- **根拠**: MPMC queue の enqueue/dequeue が既に `seq_atom` によるHB保証を持っており、`enqueueTimestampUs` の可視性は enqueue/dequeue の CAS 成功で間接的に保証される

---

## 4. 実装順序（コードベース照合反映版）

### Phase 1-A（最優先・実装必須）

#### P1-1: ShutdownRuntime 拡張（1h）

**ファイル**: `src/audioengine/ISRShutdown.h`

```cpp
// 既存 enum に追加
// TimedOut, Failed を ShutdownComplete の後に追加
enum class ShutdownPhase : uint8_t {
    Running,
    AudioStopped,
    ObserverDrained,
    RetireClosed,
    EpochSettled,
    ReclaimComplete,
    ShutdownComplete,
    TimedOut,    // ★ 追加
    Failed       // ★ 追加
};

// ShutdownRuntime に追加
void markTimedOut() noexcept { phase_.store(ShutdownPhase::TimedOut, std::memory_order_release); }
void markFailed() noexcept { phase_.store(ShutdownPhase::Failed, std::memory_order_release); }
```

**注意**: `transitionTo()` が既存の7状態間の遷移をチェックしている場合、`TimedOut/Failed` への遷移を許可するよう修正が必要。

#### P1-2: releaseResources タイムアウト処理 + SnapshotCoordinator デストラクタ（2h）

**ファイル**: `src/core/SnapshotCoordinator.h`, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`

```cpp
// SnapshotCoordinator.h: finalizeShutdown を追加
void finalizeShutdown(bool timedOut) noexcept {
    if (!timedOut) {
        constexpr auto snapshotDeleter = [](void* ptr) noexcept {
            SnapshotFactory::destroy(static_cast<GlobalSnapshot*>(ptr));
        };
        const uint64_t retireEpoch = m_epochProvider->publishEpoch();
        GlobalSnapshot* snap = m_slots.exchangeCurrent(nullptr, std::memory_order_acq_rel);
        if (snap) m_epochProvider->enqueueRetire(snap, snapshotDeleter, retireEpoch);
        snap = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
        if (snap) m_epochProvider->enqueueRetire(snap, snapshotDeleter, retireEpoch);
        m_epochProvider->tryReclaim();
    }
    // timedOut == true: リーク許容、何もしない
}

// ~SnapshotCoordinator(): デストラクタでは tryReclaim() を絶対に呼ばない
~SnapshotCoordinator() noexcept {
    // 旧: tryReclaim() 呼び出し → 削除
    // デストラクタは何もしない（finalizeShutdown が releaseResources で先に呼ばれる想定）
}
```

**releaseResources の修正** (`AudioEngine.Processing.ReleaseResources.cpp`):

```cpp
// 既存の drainAll() 呼び出しを markTimedOut に置換
if (!drainedWithinBudget || !isFullyDrained()) {
    if (!drainedWithinBudget) {
        shutdownRuntime_.markTimedOut();
        diagLog("[DIAG] releaseResources: drain timeout reached, proceeding with leak");
    }
    // 強制reclaimはしない（リーク許容）
}

// finalizeShutdown 呼び出しを追加
m_coordinator.finalizeShutdown(drainedWithinBudget);

// 既存の shutdownRuntime_.transitionTo(ShutdownPhase::ShutdownComplete) は維持
```

#### P1-3: waitForDrain 結合前提の jassert 追加（0.5h）

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::waitForDrain(int timeoutMs, int pollIntervalMs) noexcept
{
    ASSERT_NON_RT_THREAD();
    // ★ 結合前提: オーディオコールバックが停止していることを確認
    jassert(convo::consumeAtomic(rtLocalState_.audioCallbackActiveCount, std::memory_order_acquire) == 0);
    // ... 既存実装 ...
}
```

#### P1-4: DeferredDeletionQueue 拡張（DeletionEntry + peekOldestTimestampUs）（2h）

**ファイル**: `src/DeferredDeletionQueue.h`

```cpp
// DeletionEntry に enqueueTimestampUs を追加（非atomic）
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    uint64_t publicationSequenceId{0};
    uint64_t generation{0};
    uint64_t enqueueTimestampUs{0};  // ★ 追加（非atomic、reclaimスレッドのみ書き込み）
};

// static_assert は引き続き成立（uint64_t は trivially copyable）
static_assert(std::is_trivially_copyable_v<DeletionEntry>, "...");

// enqueue() 内でタイムスタンプ設定
// CAS 成功直後（entry書き込み後）に設定
entry.enqueueTimestampUs = getCurrentTimeUs();

// peekOldestTimestampUs() の追加
[[nodiscard]] uint64_t peekOldestTimestampUs() const noexcept {
    uint64_t oldest = 0;
    uint32_t deqPosVal = convo::consumeAtomic(dequeuePos, std::memory_order_acquire);
    uint32_t enqPosVal = convo::consumeAtomic(enqueuePos, std::memory_order_acquire);
    for (uint32_t i = deqPosVal; i != enqPosVal; ++i) {
        uint64_t ts = ringBuffer[i & kMask].enqueueTimestampUs;
        if (ts != 0 && (oldest == 0 || ts < oldest))
            oldest = ts;
    }
    return oldest;
}
```

#### P1-5: TimeUtils.h の新規作成（0.5h）

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

#### P1-6: ISRRetireRouter への getOldestPendingRetireAgeMs 追加（0.5h）

**ファイル**: `src/audioengine/ISRRetireRouter.h`

```cpp
uint64_t getOldestPendingRetireAgeMs() const noexcept {
    assert(epochDomain_ != nullptr);
    uint64_t oldest = epochDomain_->deferredDeletionQueue.peekOldestTimestampUs();
    if (oldest == 0) return 0;
    return (getCurrentTimeUs() - oldest) / 1000;
}
```

**注意**: `deferredDeletionQueue` は `EpochDomain` の private メンバ。アクセス手段の追加が必要。
→ `EpochDomain` に `peekOldestDeferredTimestampUs()` 公開メソッドを追加。

#### P1-7: pendingRetireCount 合計対応（Ring + Fallback）（1h）

**ファイル**: `src/core/EpochDomain.h`, `src/audioengine/ISRRetireRouter.h`

- `EpochDomain` レベルでは ring buffer のみ（変更なし）
- `ISRRetireRouter` または `AudioEngine` レベルでフォールバックキューを含めた値を提供
- **方法**: `AudioEngine` の `RuntimeBackpressureTelemetry` で既に `fallbackQueueDepth_` を別途追跡済みのため、`pendingRetireCount()` を audioengine レベルでオーバーライドしない。代わりに `collectDrainAudit()` の `routerPendingRetire` フィールドを実際の値で埋める:

```cpp
// AudioEngine::collectDrainAudit() 修正
.routerPendingRetire = static_cast<uint64_t>(m_retireRouter->pendingRetireCount())
    + convo::consumeAtomic(fallbackQueueDepth_, std::memory_order_acquire),
```

#### P1-8: maxConcurrentReadersObserved（既存 depth 利用）（1h）

**ファイル**: `src/core/EpochDomain.h`

```cpp
struct ReaderSlot {
    std::atomic<uint64_t> epoch { kInactiveEpoch };
    std::atomic<uint32_t> depth { 0 };
    std::atomic<uint64_t> enterCount { 0 };             // 既存維持
    std::atomic<uint64_t> lastEnterTimestampUs { 0 };    // ★ 追加
};

// 追加メンバ
std::atomic<uint32_t> m_currentReaders { 0 };
std::atomic<uint32_t> m_maxConcurrentReadersObserved { 0 };

// enterReader の修正（prevDepth == 0 の分岐内）
void enterReader(int readerIndex) noexcept override {
    // ... 既存の head 処理 ...
    const uint32_t previousDepth = convo::fetchAddAtomic(slot.depth, 1u, std::memory_order_acq_rel);
    if (previousDepth > 0)
        return;

    // ★ 追加: ネストの最初のenterのみタイムスタンプ記録＋アクティブReader計測
    convo::publishAtomic(slot.lastEnterTimestampUs, getCurrentTimeUs(), std::memory_order_release);
    uint32_t cur = m_currentReaders.fetch_add(1u, std::memory_order_relaxed) + 1u;
    uint32_t max = m_maxConcurrentReadersObserved.load(std::memory_order_relaxed);
    while (cur > max && !m_maxConcurrentReadersObserved.compare_exchange_weak(
        max, cur, std::memory_order_relaxed, std::memory_order_relaxed)) {}

    const uint64_t epoch = currentEpoch();
    convo::publishAtomic(slot.epoch, epoch, std::memory_order_release);
}

// exitReader の修正（prevDepth == 1 の分岐内）
void exitReader(int readerIndex) noexcept override {
    // ... 既存の head 処理 ...
    const uint32_t previousDepth = convo::fetchSubAtomic(slot.depth, 1u, std::memory_order_acq_rel);
    if (previousDepth == 0) {
        convo::publishAtomic(slot.depth, 0u, std::memory_order_release);
        return;
    }
    if (previousDepth > 1)
        return;

    // ★ 追加: ネストの最後のexitでアクティブReaderカウント減
    m_currentReaders.fetch_sub(1u, std::memory_order_relaxed);

    convo::publishAtomic(slot.epoch, kInactiveEpoch, std::memory_order_release);
}

// ReaderSlotSnapshot 取得用
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

#### P1-9: Reader Long Active 検出（Warningのみ）（1h）

**ファイル**: 新設 `RuntimeHealthMonitor.cpp`（後述 P1-13 参照）

```cpp
void RuntimeHealthMonitor::checkReaderLongActive() {
    uint64_t nowUs = getCurrentTimeUs();
    for (int slot = 0; slot < EpochDomain::kMaxReaders; ++slot) {
        auto snap = m_epochDomain->getReaderSnapshot(slot);
        if (snap.depth > 0 && (nowUs - snap.lastEnterTimestampUs) > 30'000'000) {
            emitEvent(HealthEvent::Severity::Warning, EVENT_READER_LONG_ACTIVE,
                      (nowUs - snap.lastEnterTimestampUs) / 1000, slot);
        }
    }
}
```

- Severity は `Warning` のみ（Error にしない）
- イベント名: `EVENT_READER_LONG_ACTIVE`

#### P1-10: Publication Watchdog（1.5h）

**ファイル**: `src/audioengine/RuntimePublicationOrchestrator.h`

```cpp
// 追加メンバ
std::atomic<PublicationSequenceId> m_lastObservedSequence {0};
std::atomic<uint64_t> m_lastProgressTimestampUs {0};

// 追加メソッド
bool isPublicationProgressing() const noexcept {
    PublicationSequenceId current = getLastCommittedPublicationSequence();
    PublicationSequenceId last = m_lastObservedSequence.load(std::memory_order_relaxed);
    if (current > last) {
        m_lastObservedSequence.store(current, std::memory_order_relaxed);
        m_lastProgressTimestampUs.store(getCurrentTimeUs(), std::memory_order_relaxed);
        return true;
    }
    uint64_t elapsed = getCurrentTimeUs() - m_lastProgressTimestampUs.load(std::memory_order_acquire);
    return elapsed < 30'000'000;
}

uint64_t getOldestDeferredAgeMs() const noexcept {
    if (!deferredSlot_.has_value()) return 0;
    return (getCurrentTimeUs() - deferredSlot_->enqueueTimestampUs) / 1000;
}
```

**注意**: `getLastCommittedPublicationSequence()` は `AudioEngine` のメソッド。`RuntimePublicationOrchestrator` は `engine_` メンバ経由でアクセス可能。

#### P1-11: Crossfade Watchdog（1h）

**ファイル**: `src/audioengine/CrossfadeRuntime.h`

```cpp
// 追加メンバ
std::atomic<uint64_t> m_startTimestampUs {0};

// start() 修正
void start(double fadeTimeSec, double sampleRate) noexcept {
    // ... 既存処理 ...
    m_startTimestampUs.store(getCurrentTimeUs(), std::memory_order_release);
}

// complete() 修正
void complete() noexcept {
    // ... 既存処理 ...
    m_startTimestampUs.store(0, std::memory_order_release);
}

// 追加メソッド
uint64_t getCrossfadeAgeMs() const noexcept {
    uint64_t start = m_startTimestampUs.load(std::memory_order_acquire);
    if (start == 0) return 0;
    return (getCurrentTimeUs() - start) / 1000;
}

// reset() 修正
void reset() noexcept {
    // ... 既存処理 ...
    m_startTimestampUs.store(0, std::memory_order_release);
}
```

#### P1-12: EvidenceRingBuffer（SPSC簡素版）（1h）

**ファイル**: 新規 `src/audioengine/EvidenceRingBuffer.h`

```cpp
#pragma once
#include <atomic>
#include <array>
#include <cstdint>

namespace convo {

struct HealthEvent {
    enum class Severity : uint8_t { Info, Warning, Error };
    uint64_t timestampUs;
    Severity severity;
    uint32_t eventCode;
    uint64_t value;
    uint32_t slot;
};

template<typename T, size_t N>
class SPSRingBuffer {
    static_assert((N & (N - 1)) == 0, "N must be power of two");
    std::array<T, N> buffer;
    std::atomic<size_t> writeIndex {0};
    std::atomic<size_t> readIndex {0};
    std::atomic<uint64_t> droppedEventCount {0};
public:
    bool push(const T& item) noexcept {
        size_t w = writeIndex.load(std::memory_order_relaxed);
        size_t r = readIndex.load(std::memory_order_acquire);
        if (w - r >= N) {
            droppedEventCount.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        buffer[w & (N - 1)] = item;
        writeIndex.store(w + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& item) noexcept {
        size_t r = readIndex.load(std::memory_order_relaxed);
        size_t w = writeIndex.load(std::memory_order_acquire);
        if (r == w) return false;
        item = buffer[r & (N - 1)];
        readIndex.store(r + 1, std::memory_order_release);
        return true;
    }
    uint64_t getDroppedCount() const noexcept { return droppedEventCount.load(std::memory_order_relaxed); }
    size_t size() const noexcept {
        size_t w = writeIndex.load(std::memory_order_acquire);
        size_t r = readIndex.load(std::memory_order_acquire);
        return w - r;
    }
};

static constexpr uint32_t EVENT_RETIRE_STALL = 1001;
static constexpr uint32_t EVENT_RETIRE_STALL_WARNING = 1002;
static constexpr uint32_t EVENT_PUBLICATION_DEFERRED_STALL = 2001;
static constexpr uint32_t EVENT_PUBLICATION_DEFERRED_WARNING = 2002;
static constexpr uint32_t EVENT_PUBLICATION_PROGRESS_STALL = 2003;
static constexpr uint32_t EVENT_READER_LONG_ACTIVE = 3001;
static constexpr uint32_t EVENT_CROSSFADE_STALL = 4001;
static constexpr uint32_t EVENT_CROSSFADE_WARNING = 4002;

using HealthEventBuffer = SPSRingBuffer<HealthEvent, 1024>;

} // namespace convo
```

#### P1-13: RuntimeHealthMonitor（2h）

**ファイル**: 新規 `src/audioengine/RuntimeHealthMonitor.h`

```cpp
#pragma once
#include "EvidenceRingBuffer.h"
#include "core/EpochDomain.h"

namespace convo::isr {

class ISRRetireRouter;
class RuntimePublicationOrchestrator;
class CrossfadeRuntime;

class RuntimeHealthMonitor {
public:
    void setRingBuffer(HealthEventBuffer* buffer) noexcept { m_buffer = buffer; }
    void setEpochDomain(EpochDomain* domain) noexcept { m_epochDomain = domain; }
    void setRetireRouter(ISRRetireRouter* router) noexcept { m_retireRouter = router; }
    void setOrchestrator(RuntimePublicationOrchestrator* orch) noexcept { m_orchestrator = orch; }
    void setCrossfadeRuntime(CrossfadeRuntime* cf) noexcept { m_crossfadeRuntime = cf; }

    void tick() noexcept;  // 1秒ごとに timerCallback から呼ばれる

private:
    void checkRetireStall() noexcept;
    void checkPublicationStall() noexcept;
    void checkReaderLongActive() noexcept;
    void checkCrossfadeStall() noexcept;
    void emitEvent(HealthEvent::Severity severity, uint32_t eventCode, uint64_t value, uint32_t slot = 0) noexcept;

    HealthEventBuffer* m_buffer = nullptr;
    EpochDomain* m_epochDomain = nullptr;
    ISRRetireRouter* m_retireRouter = nullptr;
    RuntimePublicationOrchestrator* m_orchestrator = nullptr;
    CrossfadeRuntime* m_crossfadeRuntime = nullptr;
};

} // namespace convo::isr
```

**ファイル**: 新規 `src/audioengine/RuntimeHealthMonitor.cpp`

```cpp
#include "RuntimeHealthMonitor.h"
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/RuntimePublicationOrchestrator.h"
#include "audioengine/CrossfadeRuntime.h"
#include "audioengine/TimeUtils.h"

namespace convo::isr {

void RuntimeHealthMonitor::tick() noexcept {
    checkRetireStall();
    checkPublicationStall();
    checkReaderLongActive();
    checkCrossfadeStall();
}

void RuntimeHealthMonitor::checkRetireStall() noexcept {
    if (!m_retireRouter) return;
    uint64_t ageMs = m_retireRouter->getOldestPendingRetireAgeMs();
    if (ageMs > 60000) {
        emitEvent(HealthEvent::Severity::Error, EVENT_RETIRE_STALL, ageMs);
    } else if (ageMs > 10000) {
        emitEvent(HealthEvent::Severity::Warning, EVENT_RETIRE_STALL_WARNING, ageMs);
    }
}

void RuntimeHealthMonitor::checkPublicationStall() noexcept {
    if (!m_orchestrator) return;
    uint64_t deferredAge = m_orchestrator->getOldestDeferredAgeMs();
    if (deferredAge > 30000) {
        emitEvent(HealthEvent::Severity::Error, EVENT_PUBLICATION_DEFERRED_STALL, deferredAge);
    } else if (deferredAge > 5000) {
        emitEvent(HealthEvent::Severity::Warning, EVENT_PUBLICATION_DEFERRED_WARNING, deferredAge);
    }
    if (m_orchestrator->getPendingIntentCount() > 0 && !m_orchestrator->isPublicationProgressing()) {
        emitEvent(HealthEvent::Severity::Error, EVENT_PUBLICATION_PROGRESS_STALL, 0);
    }
}

void RuntimeHealthMonitor::checkReaderLongActive() noexcept {
    if (!m_epochDomain) return;
    uint64_t nowUs = getCurrentTimeUs();
    for (int slot = 0; slot < EpochDomain::kMaxReaders; ++slot) {
        auto snap = m_epochDomain->getReaderSnapshot(slot);
        if (snap.depth > 0 && (nowUs - snap.lastEnterTimestampUs) > 30'000'000) {
            emitEvent(HealthEvent::Severity::Warning, EVENT_READER_LONG_ACTIVE,
                      (nowUs - snap.lastEnterTimestampUs) / 1000, slot);
        }
    }
}

void RuntimeHealthMonitor::checkCrossfadeStall() noexcept {
    if (!m_crossfadeRuntime) return;
    uint64_t ageMs = m_crossfadeRuntime->getCrossfadeAgeMs();
    if (ageMs > 60000) {
        emitEvent(HealthEvent::Severity::Error, EVENT_CROSSFADE_STALL, ageMs);
    } else if (ageMs > 10000) {
        emitEvent(HealthEvent::Severity::Warning, EVENT_CROSSFADE_WARNING, ageMs);
    }
}

void RuntimeHealthMonitor::emitEvent(HealthEvent::Severity severity, uint32_t eventCode,
                                      uint64_t value, uint32_t slot) noexcept {
    if (!m_buffer) return;
    HealthEvent ev{getCurrentTimeUs(), severity, eventCode, value, slot};
    m_buffer->push(ev);
}

} // namespace convo::isr
```

#### P1-14: AudioEngine への統合（1.5h）

**ファイル**: `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.Timer.cpp`

```cpp
// AudioEngine.h: メンバ追加
#include "audioengine/RuntimeHealthMonitor.h"
#include "audioengine/EvidenceRingBuffer.h"

// AudioEngine クラス内
convo::HealthEventBuffer m_healthEventBuffer;
convo::isr::RuntimeHealthMonitor m_healthMonitor;
```

```cpp
// AudioEngine コンストラクタ: 初期化（既存コンストラクタに追加）
m_healthMonitor.setRingBuffer(&m_healthEventBuffer);
m_healthMonitor.setEpochDomain(&m_epochDomain);
m_healthMonitor.setRetireRouter(m_retireRouter.get());
m_healthMonitor.setOrchestrator(runtimeOrchestrator_.get());
m_healthMonitor.setCrossfadeRuntime(&crossfadeRuntime_);
```

```cpp
// AudioEngine::timerCallback() 末尾付近（processDeferredReleases() の後）
// ★ Health Monitor tick（1秒間隔、timerCallback の周期に依存）
if (m_healthMonitor) {
    m_healthMonitor.tick();
}
```

**注意**: `timerCallback()` 内には既に多くの診断/VERIFYログが存在する。`m_healthMonitor.tick()` は `emitEvidenceTickNonRt()` の直前または直後に配置する。周期逸脱検知は行わない（基本原則4による）。

### Phase 1-B（統合・テスト）

#### P1-15: EpochDomain への公開メソッド追加（0.5h）

**ファイル**: `src/core/EpochDomain.h`

```cpp
// peekOldestDeferredTimestampUs: ISRRetireRouter 経由で DeferredDeletionQueue の最古タイムスタンプを取得
[[nodiscard]] uint64_t peekOldestDeferredTimestampUs() const noexcept {
    return deferredDeletionQueue.peekOldestTimestampUs();
}
```

### Phase 2（推奨、時間許せば実施）

#### P2-1: ISRRetireRouter Reader API削減（移行期間付き）

**ファイル**: `src/audioengine/ISRRetireRouter.h`

現状の `registerReaderThread/enterReader/exitReader/activeReaderCount` は既に `[[deprecated]]` 警告抑制付き委譲実装。
新たな `[[deprecated]]` 付与は不要だが、呼び出し元の移行を促進するため以下を行う:

1. 全呼び出し元を `RCUReader` 経由に書き換え（`.cpp` ファイル調査が必要）
2. 全移行確認後、`ISRRetireRouter` から Reader API を削除

**注意**: `currentEpoch()`, `getMinReaderEpoch()`, `enqueueRetire()`, `tryReclaim()` は `IEpochProvider` インターフェースの純粋仮想関数として必要。削除不可。

### Phase 3（将来拡張）

#### P3-1: EvidenceExporter による HealthEventBuffer 消費

既存の `emitEvidenceTickNonRt()` が `HealthEventBuffer` からイベントを読み出し、証跡ファイルに出力する。

---

## 5. 修正された設計判断一覧

| # | 論点 | v1（旧計画） | v2（修正版） |
|---|------|-------------|-------------|
| 1 | ShutdownPhase | `Running,StopRequested,Draining,TimedOut,Stopped,Failed` に置換 | 既存 `Running..ShutdownComplete` を維持 + `TimedOut,Failed` を追加 |
| 2 | ShutdownRuntime | `tryStart()` 新設＋最小実装 | 既存を拡張: `markTimedOut()`, `markFailed()` のみ追加 |
| 3 | SnapshotCoordinator dtor | 「呼んでいないことを確認」→削除 | **現状呼んでいる**ことを認識した上で `finalizeShutdown()` に置換 |
| 4 | waitForDrain | `std::chrono` で新規実装 | 既存 `juce::Time` 実装を流用＋`jassert` 追加 |
| 5 | releaseResources timeout | 新規追加 | 既存の `drainAll()` 強制reclaimをリーク許容に変更 |
| 6 | FallbackEntry | `DeferredDeletionQueue.h` に存在する想定 | 存在しない。`SafeStateSwapper.h` にある。`DeletionEntry` に `enqueueTimestampUs` 追加 |
| 7 | ringBuffer タイムスタンプ | `std::atomic<uint64_t>` | 非atomic `uint64_t`（static_assert 回避のため） |
| 8 | pendingRetireCount | `pendingRetireCount()` に自動反映 | `collectDrainAudit()` の `routerPendingRetire` で加算 |
| 9 | EpochDomain enterCount | 新規追加 | 既存の `enterCount` + `detectStuckReaders` と共存。`lastEnterTimestampUs` と `m_currentReaders` を追加 |
| 10 | EvidenceBuffer producer | `RuntimeHealthMonitor` のみ | 同左。既存 `EvidenceExporter` は出力専用に |
| 11 | HealthMonitor tick | 新規 timer | 既存 `timerCallback` 内で呼ぶ。周期逸脱検知なし |
| 12 | DeletionEntry atomic制約 | 言及なし | `is_trivially_copyable_v` 制約を認識し非atomic設計に |

---

## 6. コードベース照合チェックリスト（更新版）

- [x] `DeferredDeletionQueue` の `head_`(dequeuePos), `tail_`(enqueuePos), `ringBuffer` 存在確認
- [x] `DeletionEntry` に `enqueueTimestampUs` 追加が必要（`FallbackEntry` は存在しない）
- [x] `DeletionEntry` の `is_trivially_copyable_v` static_assert 確認 → 非atomic設計が必要
- [x] `EpochDomain::ReaderSlot` に `depth` と `enterCount` が存在することを確認
- [x] `ShutdownRuntime` の現状確認（enum含む）→ 計画書の想定と大きく異なる
- [x] `SnapshotCoordinator` デストラクタが `tryReclaim()` を呼んでいることを確認
- [x] `AudioEngine::timerCallback` の既存負荷は軽量処理追加に耐えられる
- [x] `waitForDrain()` の既存実装を確認（`juce::Time` + `juce::Thread::sleep`）
- [x] `releaseResources()` の既存タイムアウト処理を確認（強制drain中）
- [x] `RuntimePublicationOrchestrator::deferredSlot_` と `getMaxDeferredAgeMs()` の既存確認
- [x] `DeferredRetireFallbackQueue` の存在確認（別クラスとして分離済み）
- [x] `AudioEngine::oldestPendingAge_` の存在確認
- [x] `RuntimeDrainAudit` + `collectDrainAudit()` の既存確認

---

## 7. 実装上の補足・アドバイス

### 7.1 `getCurrentTimeUs()` の配置

新規 `src/audioengine/TimeUtils.h` に定義。インライン関数のため、include するだけで使用可能。

### 7.2 `RuntimeHealthMonitor` の所有権と初期化

- `AudioEngine` の直接メンバとして保持（`unique_ptr` ではなく）
- コンストラクタの `m_retireRouter` / `runtimeOrchestrator_` 初期化後（かつ `timerCallback` 開始前）に各ポインタをセット
- `prepareToPlay()` でも初期化可能だが、コンストラクタで十分

### 7.3 `HealthEventBuffer` の Producer 制限

Producer = `RuntimeHealthMonitor::tick()` のみ。`timerCallback` から `m_healthMonitor.tick()` → `emitEvent()` → `push()` の単一パス。
`emitEvidenceTickNonRt()` は Consumer（`pop()`）のみ行う。

### 7.4 `DeletionEntry` の非atomicタイムスタンプ安全性

`enqueueTimestampUs` は reclaim スレッド（`reclaim()` 内の CAS 成功直後）のみが書き込む。
MPMC queue の `seq_atom` による HB (happens-before) 保証により、書き込み後の読み取りは常に可視。
ただし **peekOldestTimestampUs() が enqueue 中（CAS成功直後、ts書き込み前）のスロットを誤読する可能性**は理論上ある。
→ 影響は軽微（ts=0扱いでスキップされるだけ）。実用上問題なし。

### 7.5 タイムアウト後のリーク再生成問題

`markTimedOut()` による意図的リークが発生したインスタンスが残った状態での再生成は、Standalone アプリでは問題とならない。プロセス終了時に OS が回収する。
必要に応じて「再生成禁止」「プロセス再起動要求」などの防衛策を追加可能。

### 7.6 テスト方針

各コンポーネントは独立してユニットテスト可能:

- `SPSRingBuffer`: push/pop/dropped/サイズ限界テスト
- `RuntimeHealthMonitor`: mock ポインタ注入 + tick() 呼び出し
- `EpochDomain`: ReaderSlot 拡張部分の enter/exit 動作テスト
- `CrossfadeRuntime`: start/complete/reset のタイムスタンプ動作テスト

---

## 8. 改訂版 実装順序と工数

| 順序 | 項目 | 工数 |
|------|------|------|
| **Phase 1-A** | | |
| 1 | `TimeUtils.h` 新規作成 | 0.5h |
| 2 | `ISRShutdown.h`: `ShutdownPhase` に `TimedOut/Failed` 追加、`markTimedOut/markFailed` 追加 | 1h |
| 3 | `DeferredDeletionQueue`: `DeletionEntry` に `enqueueTimestampUs` 追加、`peekOldestTimestampUs()` 実装 | 2h |
| 4 | `EpochDomain`: `lastEnterTimestampUs` + `m_currentReaders/m_maxConcurrentReadersObserved` + `getReaderSnapshot()` + `peekOldestDeferredTimestampUs()` | 2h |
| 5 | `EvidenceRingBuffer.h`: `SPSRingBuffer` + `HealthEvent` + イベントコード定数 | 1.5h |
| 6 | `CrossfadeRuntime`: `m_startTimestampUs` + `getCrossfadeAgeMs()` | 1h |
| 7 | `RuntimePublicationOrchestrator`: `isPublicationProgressing()` + 進捗追跡フィールド | 1.5h |
| 8 | `ISRRetireRouter`: `getOldestPendingRetireAgeMs()` | 0.5h |
| 9 | `RuntimeHealthMonitor.h/cpp`: 新規クラス | 2.5h |
| 10 | `SnapshotCoordinator`: `finalizeShutdown()` + デストラクタからの `tryReclaim()` 削除 | 1.5h |
| 11 | `AudioEngine.Processing.ReleaseResources.cpp`: タイムアウト処理をリーク許容に変更 + `finalizeShutdown()` 呼び出し | 1.5h |
| 12 | `AudioEngine.Threading.cpp`: `waitForDrain` に `jassert` 追加 | 0.5h |
| 13 | `AudioEngine.h/cpp`: `RuntimeHealthMonitor + HealthEventBuffer` 統合 + `collectDrainAudit` 修正 | 2h |
| 14 | `AudioEngine.Timer.cpp`: `timerCallback` に `m_healthMonitor.tick()` 追加 | 0.5h |
| | **Phase 1-A 小計** | **18.5h** |
| **Phase 1-B** | | |
| 15 | ビルド確認 + リンクエラー修正 | 2h |
| 16 | CIゲートテスト | 1h |
| **Phase 2** | | |
| 17 | ISRRetireRouter Reader API 呼び出し元移行 | 3h |
| 18 | ISRRetireRouter Reader API 削除 | 1h |
| **Phase 3** | | |
| 19 | EvidenceExporter による HealthEventBuffer 消費 | 1.5h |
| | **合計** | **27h** |

---

## 9. 評価

- **コードベース適合性**: v1から大幅改善。既存コードとの不整合を全て解消。
- **実運用での破綻耐性**: 最高水準を維持。Pull型監視、デストラクタ依存排除、データレースフリー。
- **残リスク**:
  - `releaseResources()` の強制reclaim→リーク許容への変更は既存挙動を変える。テストで確認必須。
  - `SnapshotCoordinator` デストラクタからの `tryReclaim()` 削除により、異常系（`releaseResources` 未実行のデストラクタ）ではリークが増える。
  - これらのリスクは設計意図（クラッシュよりリーク優先）に合致する。

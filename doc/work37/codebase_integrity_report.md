# 現行コード監視系の実装状況と v4.1 設計との整合性報告

> **日付**: 2026-06-14
> **調査方法**: Serena MCP, CodeGraph MCP (Full Index 51K entities), graphify MCP (15K nodes), grep/Select-String, 実ファイル読み取り
> **確認範囲**: RuntimeHealthMonitor(.h/.cpp), AudioEngine.Timer.cpp, AudioEngine.CtorDtor.cpp, ISRRetireRouter(.h/.cpp), ISRShutdown(.h/.cpp), WorldLifecycleAudit(.h/.cpp), RuntimeDrainAudit.h, DSPLifetimeManager.h, SnapshotCoordinator(.cpp/.h), RefCountedDeferred.h, EQProcessor.Core/.Parameters/.Coefficients.cpp, DeferredDeletionQueue.h, DeferredRetireFallbackQueue.h, RuntimePublicationState.h, RuntimePublicationOrchestrator(.h/.cpp), DspNumericPolicy.h

---

## 1. HealthMonitor 実装状況と設計との整合性

### 1.1 tick() 内の監視項目（9種類、すべて存在）

| tick 内関数 | 設計でのイベント | コード存在 | 設計と整合 |
|-------------|----------------|-----------|-----------|
| `checkRetireStall()` | EVENT_RETIRE_STALL | ✅ `RuntimeHealthMonitor.cpp:30` | ✅ 閾値: hwm*1.5 |
| `checkPublicationStall()` | EVENT_PUBLICATION_STALL | ✅ `RuntimeHealthMonitor.cpp:57` | ✅ 30秒停滞 |
| `diagnoseRetireStall()` | EVENT_READER_STUCK | ✅ `RuntimeHealthMonitor.cpp:157` | ⚠️ 現在epoch差のみ。residencyはsevere判定のみ |
| `checkCrossfadeTimeout()` | EVENT_CROSSFADE_TIMEOUT | ✅ `RuntimeHealthMonitor.cpp:215` | ✅ 30秒固定 |
| `checkCrossfadeEventDrop()` | EVENT_CROSSFADE_EVENT_DROP | ✅ `RuntimeHealthMonitor.cpp:226` | ✅ 差分ベース |
| `checkReaderSlotUsage()` | EVENT_READER_SLOT_USAGE | ✅ `RuntimeHealthMonitor.cpp:244` | ✅ 75%/90%閾値 |
| `checkOverflowRate()` | なし（暗黙的Critical） | ✅ `RuntimeHealthMonitor.cpp:308` | ✅ 1回/秒 Warning, 5回/秒 Critical |
| `checkRetireReclaimLatency()` | EVENT_RETIRE_AGE_CRITICAL | ✅ `RuntimeHealthMonitor.cpp:394` | ✅ 5秒/30秒 |
| `updateHealthState()` | ISRHealthState統合 | ✅ `RuntimeHealthMonitor.cpp:103` | **v4.0 設計の単一権限と一致** |

**重要発見**: `updateHealthState()` は既に全 `m_prev*State` を集約して ISRHealthState を決定している。v4.0 の「HealthState は HealthMonitor 専権」という設計は現状コードと完全に一致。

### 1.2 MonitorState の状態管理（6系統）

| 状態変数 | 用途 | 設計と整合 |
|----------|------|-----------|
| `m_prevRetireState` | Retire stall | ✅ PolicyEngineの入力として使用 |
| `m_prevPublicationState` | Publication stall | ✅ |
| `m_prevCrossfadeDropState` | Crossfade timeout/drop | ✅ |
| `m_prevReaderSlotState` | Reader slot usage | ✅ |
| `m_prevOverflowRateState` | Overflow rate | ✅ |
| `m_prevRetireAgeState` | Retire reclaim latency | ✅ |

**v4.0 との一致点**: PolicyEngine はこれらの MonitorState を入力として受け取り、RecoveryAction を選択するだけでよい。

### 1.3 EventCallback の実態

`AudioEngine.CtorDtor.cpp:56`:

```cpp
m_healthMonitor.setEventCallback(
    [this](const convo::HealthEvent& ev) { onHealthEvent(ev); });
```

`AudioEngine.Timer.cpp:541`:

```cpp
void AudioEngine::onHealthEvent(const convo::HealthEvent& event) noexcept {
    // イベントコードで分岐して直接 Recovery Action を実行
    if (event.eventCode == convo::EVENT_READER_SLOT_USAGE && ...) {
        convo::publishAtomic(retirePressureAdmissionStrict_, true, ...);
        emitEvidenceTickNonRt(true);
    }
    if (event.eventCode == convo::EVENT_PUBLICATION_STALL && ...) {
        runtimeOrchestrator_->clearDeferredForShutdown();
    }
    if (event.eventCode == convo::EVENT_RETIRE_STALL && ...) {
        convo::publishAtomic(retirePressureAdmissionStrict_, true, ...);
        tryReclaimResources();
    }
    if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT) {
        // crossfade強制完了
    }
    if (event.eventCode == convo::EVENT_READER_STUCK) {
        emitEvidenceTickNonRt(true);
    }
}
```

**v4.0 との整合性**: ✅ `onHealthEvent` は既に Recovery Action の実装を持っている。v4.0 の PolicyEngine 導入で、Action 選択ロジックを PolicyEngine に移管し、`onHealthEvent` は Action 実装のみに専念させる形になる。

---

## 2. enqueueRetire 戻り値調査結果

### 2.1 ISRRetireRouter 経由（tryReclaim+再試行あり）

| 呼び出し元 | ファイル | 戻り値 | 保護 |
|-----------|----------|--------|------|
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult()` | `AudioEngine.h:3233` | ✅ `RetireEnqueueResult` チェック + drain retry | ✅ 二重保護 |
| `DSPLifetimeManager::retire()` | `DSPLifetimeManager.h:44` | ❌ **未チェック** | ⚠️ Router保護のみ |
| `RuntimePublicationCoordinator::enqueueRetire()` | `ISRRuntimePublicationCoordinator.cpp:137` | ✅ `Success`/`QueueFull` 変換 | ✅ |
| `EQProcessor::enqueueDeferredDeleteWithFallback()` | `EQProcessor.Core.cpp:26` | ⚠️ チェックするがvoidで消失 | ⚠️ 要修正 |

### 2.2 IEpochProvider 直結（ISRRetireRouter 未経由）

| 呼び出し元 | ファイル | 戻り値 | リスク |
|-----------|----------|--------|--------|
| `SnapshotCoordinator::startFade()` | `SnapshotCoordinator.cpp:36` | ❌ **未チェック** | **GlobalSnapshotリーク** |
| `SnapshotCoordinator::resetFadeStateAndRetireTarget()` | `SnapshotCoordinator.cpp:67` | ❌ **未チェック** | **GlobalSnapshotリーク** |
| `SnapshotCoordinator::completeFade()` | `SnapshotCoordinator.cpp:87` | ❌ **未チェック** | **GlobalSnapshotリーク** |
| `SnapshotCoordinator::retireCurrentAndTarget()` | `SnapshotCoordinator.h:161-162` | ❌ **未チェック** | **GlobalSnapshotリーク** |
| `RefCountedDeferred::release()` | `RefCountedDeferred.h:23` | ❌ **未チェック** | **テンプレート型リーク** |

**v4.0 との整合性**: ✅ Phase 1.1 (DSPLifetimeManager) + Phase 1.4 (EQProcessor) は ISRRetireRouter 経由で保護あり。Phase 1.2 (SnapshotCoordinator) は却下に決定。Phase 1.3 (RefCountedDeferred) は `canBlock()` 導入で対応予定。

---

## 3. DeferredDeletionQueue 現状

| 項目 | 値 |
|------|-----|
| `kQueueSize` | **4096**（`DeferredDeletionQueue.h:221`） |
| 種類 | MPMC ロックフリー |
| `enqueue()` 戻り値 | `bool`（full時はfalse） |

**確認**: Queue が full の場合、`EpochDomain::enqueueRetire()` は `false` を返す。これが `ISRRetireRouter` → `tryReclaim()` → 再試行 のトリガーになる。

---

## 4. DeferredRetireFallbackQueue 現状

| 項目 | 値 | v4.0 設計との差異 |
|------|-----|------------------|
| `push()` 戻り値 | `size_t`（現在サイズ） | ❌ v4.0 では `bool` が必要 |
| メモリサイズ追跡 | なし | ❌ v4.0 では `estimatedBytes()` が必要 |
| retryCount | なし | ❌ v4.0 では必要 |
| HardLimit | なし | ❌ v4.0 では必要 |

**結論**: DeferredRetireFallbackQueue は v4.0 設計での拡張が必要。

---

## 5. ReaderStuck 現状の課題

`EpochDomain.h` の `detectStuckReaders()`:

```cpp
// 現在: epoch差のみでStuck判定（residencyは収集するが判定未使用）
if (epochGap > stuckThreshold) {
    info.isStuck = true;
    info.residencyTimeUs = residencyUs;
    break;
}
```

**v4.0 設計との差異**:

- ❌ epoch差のみ（v4.0 では epoch差 AND residency 条件が必要）
- ⚠️ `stuckThreshold=10` は v4.0 設計と一致
- ❌ `pendingRetireCount` 条件なし（v4.0 では `pendingRetire > 0` が必要）
- ⚠️ `severe` 条件で `residency > 30s` を使用中 → v4.0 では `10s` に短縮予定

---

## 6. WorldLifecycleAudit 現状

| 機能 | 状態 | v4.0 設計 |
|------|------|-----------|
| `activeWorldCount` 追跡 | ✅ 実装済み | ✅ 利用 |
| `doubleRetireCount` | ✅ 実装済み | 🆕 `onFallbackOverflow()` 追加必要 |
| HealthMonitor 連携 | ❌ **なし**（ファイル監査のみ） | 🆕 連携必要 |
| `emitSnapshot()` | ✅ JSON出力 | Phase 8.1 の FallbackOverflow 検出に活用 |

---

## 7. ShutdownRuntime 現状

| 機能 | 状態 | v4.0 設計 |
|------|------|-----------|
| `ShutdownPhase::EmergencyDrain` | ✅ `ISRShutdown.h:31` に存在 | ✅ 既存、但し `#ifdef` で保護 |
| `requestEmergencyDrain()` | ❌ **存在しない** | 🆕 新規追加必要 |
| `isEmergencyDrainRequested()` | ❌ **存在しない** | 🆕 新規追加必要 |
| `ShutdownResult` 型 | ❌ **存在しない** | 🆕 Phase 3 で追加 |
| `collectResult()` | ❌ **存在しない** | 🆕 Phase 3 で追加 |

---

## 8. HealthMonitor の Recovery Action 実装（現状）

`onHealthEvent()` はすでに4種類の Recovery Action を実装済み:

| イベント | 現状のAction | v4.0 での移行先 |
|----------|-------------|-----------------|
| EVENT_READER_SLOT_USAGE | `admissionStrict_=true` + 診断ダンプ | PolicyEngine → `executeRecoveryAction()` |
| EVENT_PUBLICATION_STALL | `clearDeferredForShutdown()` | 同上 |
| EVENT_RETIRE_STALL | `admissionStrict_=true` + `tryReclaimResources()` | 同上 |
| EVENT_CROSSFADE_TIMEOUT | DSP退役 + crossfade解放 + idle publish | 同上 |
| EVENT_READER_STUCK | 診断ダンプのみ | 同上（`ForceRetireDrain` に強化可能） |

**重要**: 実装は既に存在する。v4.0 では「呼び出し元」を `onHealthEvent` から `executeRecoveryAction` に変更するだけ。

---

## 9. 全体整合性評価

| 設計項目 | コード状態 | ギャップ | 必要変更量 |
|---------|-----------|---------|-----------|
| Phase 0: PolicyEngine | ❌ 未実装 | 新規クラス | 50行（新規ファイル） |
| Phase 1.1: DSPLifetimeManager | ⚠️ 未チェック | 戻り値チェック追加 | 5行 |
| Phase 1.2: SnapshotCoordinator | ❌ 未チェック | **却下（現状維持）** | 0行 |
| Phase 1.3: RefCountedDeferred | ❌ 未チェック | `canBlock()` | 10行 |
| Phase 1.4: EQProcessor | ⚠️ void消失 | `enqueueDeferredDeleteWithFallback` に retry | 10行 |
| Phase 1.5: FallbackQueue | ❌ 要拡張 | `push()`→bool, `estimatedBytes`, `retryCount` | 20行 |
| Phase 2: ReaderStuck | ⚠️ epoch差のみ | residency条件追加 | 5行 |
| Phase 3: ShutdownResult | ❌ 不在 | 新規構造体+collectResult | 30行 |
| Phase 4.1: executeRecoveryAction | ✅ Action実装済み | コールバック経路変更のみ | 15行 |
| Phase 4.2: WorldLifecycle → HealthMonitor | ❌ 未連携 | `onFallbackOverflow()`, `injectEvent()` | 10行 |
| **Phase 4.4: 背圧機構統一** | ❌ **3重書き込み** | **3経路→単一権限化（最重要）** | **20行** |
| Phase 6: TTL | ❌ 不在 | `DiscardReason::Expired` | 5行 |
| Phase 7: WorldConsistency | ⚠️ diagLogのみ | Event注入 | 10行 |
| Phase 8.1: FallbackQueue OOM | ❌ 不在 | 容量ベース監視+HealthMonitor連携 | 15行 |
| Phase 8.2: EmergencyDrain runtime化 | ⚠️ `#ifdef` 保護 | ShutdownRuntime への移動 | 15行 |
| Phase 8.3: TTL通知 | ❌ 不在 | UI通知パス | 10行 |

**総合**: 設計とコードの間に大きな矛盾はない。必要な変更はすべて追加/修正であり、現状コードの削除は最小限。v4.0 の設計方向性は ConvoPeq の現状コードと整合している。

---

## 10. 【最重要】三重の独立した背圧機構 — retirePressureAdmissionStrict_ の3経路

### 10.1 発見の概要

`retirePressureAdmissionStrict_` は同一 Timer tick 内で**3つの独立した経路**から書き込まれている。
どの経路も調整・優先順位・排他制御なしで同一の `std::atomic<bool>` を `publishAtomic(true)` で上書きする（last-write-wins）。

### 10.2 3経路の詳細

#### 経路1: drainDeferredRetireQueues() → evaluateRetirePressureLevelNoRt()（独立背圧）

**ファイル**: `src/audioengine/AudioEngine.Retire.cpp:119-154` および `src/audioengine/AudioEngine.Threading.cpp:20`
**タイミング**: timerCallback() 内、HealthMonitor::tick() より前（約450行目）
**発火条件**: フォールバックキューサイズ > kFallbackWarningThreshold

```cpp
// AudioEngine.Retire.cpp:119
auto level = evaluateRetirePressureLevelNoRt();
if (level >= 2) {
    convo::publishAtomic(retirePressureAdmissionStrict_, true,
                         std::memory_order_release);
}
```

この `evaluateRetirePressureLevelNoRt()` はフォールバックキューのサイズとoverflow頻度を独自に評価し、
Level 1/2/3 を返す。HealthMonitor の状態とは**完全に独立**している。

#### 経路2: onHealthEvent(EVENT_READER_SLOT_USAGE) → admissionStrict_

**ファイル**: `src/audioengine/AudioEngine.Timer.cpp:557`
**タイミング**: timerCallback() 内、HealthMonitor::tick() → onHealthEvent() 経由（約540行目）
**発火条件**: ReaderSlot使用率 > 90%

```cpp
if (event.eventCode == convo::EVENT_READER_SLOT_USAGE && ...) {
    convo::publishAtomic(retirePressureAdmissionStrict_, true, ...);
}
```

#### 経路3: onHealthEvent(EVENT_RETIRE_STALL) → admissionStrict_

**ファイル**: `src/audioengine/AudioEngine.Timer.cpp:588`
**タイミング**: 同上、経路2と同じ onHealthEvent() 内、別条件分岐
**発火条件**: pendingRetireCount > hwm*1.5 かつ Warning状態

```cpp
if (event.eventCode == convo::EVENT_RETIRE_STALL && ...) {
    convo::publishAtomic(retirePressureAdmissionStrict_, true, ...);
    tryReclaimResources();
}
```

### 10.3 timerCallback() 内の実行順（約450〜600行目）

```
1. processDeferredReleases()  [~450行目]
   └→ drainDeferredRetireQueues()
       └→ evaluateRetirePressureLevelNoRt()  [経路1: 独立背圧]
           └→ retirePressureAdmissionStrict_ = true (独立判定)

2. m_healthMonitor.tick()      [~532行目]
   └→ 9種類の check*()
   └→ onHealthEvent() callback [~540行目]
       ├→ EVENT_READER_SLOT_USAGE  [経路2]
       │   └→ retirePressureAdmissionStrict_ = true
       └→ EVENT_RETIRE_STALL       [経路3]
           └→ retirePressureAdmissionStrict_ = true
```

### 10.4 問題点

| 問題 | 詳細 |
|------|------|
| 重複書き込み | 3経路が独立して同じ変数を上書き。最終書き込みのみ有効 |
| 判断の分散 | 経路1はHealthMonitor非依存、経路2/3はHealthMonitor依存。統合判断なし |
| 調整不足 | 経路1が admission 設定→経路2/3が上書き→経路1の意図が消失し得る |
| 監査困難 | どの経路が設定したか区別不可 |
| 解除タイミング | 各経路が個別に解除条件を持つ。=true 設定後に誰が =false に戻すか未定義 |

### 10.5 PolicyEngine による解決策

PolicyEngine 導入後は:

1. **経路2と経路3**: HealthMonitor 経由のため自然に PolicyEngine に統合される。
   `onHealthEvent` 内の `publishAtomic(admissionStrict_)` は削除され、
   `executeRecoveryAction(ThrottleRebuild)` に置き換わる。

2. **経路1**: `evaluateRetirePressureLevelNoRt()` の結果を `PolicySource::RetireBackpressure` として
   PolicyEngine に渡す。PolicyEngine が MonitorState（Warning/Error）に変換し、
   `ThrottleRebuild` Action として統一発行する。

3. **統合結果**: `retirePressureAdmissionStrict_` への書き込みは PolicyEngine の
   `executeRecoveryAction()` からのみ行われる（単一権限）。

---

## 11. 深掘り調査結果 — 全未確定事項の確定

### 11.1 releaseResources() 呼び出し連鎖と ShutdownResult 伝播

**ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`, `src/audioengine/AudioEngine.h:778`

**調査結果**:

- `releaseResources()` は `void AudioEngine::releaseResources()` override → **戻り値 void は変更不可**（JUCE AudioProcessor 規約）
- `emitShutdownTrace()` は ReleaseResources.cpp:372 で既に呼ばれている → `evidence/shutdown_trace.json` に出力
- `collectDrainAudit()` は既に `.healthState = m_healthMonitor.getHealthState()` を含む（`AudioEngine.Threading.cpp:84`）
- ただし `emitShutdownTrace()` の JSON 出力には healthState が含まれていない

**確定**: ✅ **ShutdownResult 上位伝播は変更不要。collectDrainAudit() 経由で既に healthState 取得可能。Phase 3 では emitShutdownTrace() の JSON に healthState を追加するのみ。releaseResources() の戻り値 void は維持。**

### 11.2 isAudioThread() / canBlock() — ExecutionClass 不要

**ファイル**: `src/DspNumericPolicy.h:116`

**調査結果**:

- `isAudioThread()` は Thread Tag 比較ベースで実装済み
- `canBlock()` および `ExecutionClass` はコードベースに存在しない

```
inline bool isAudioThread() noexcept {
    const uint64_t tag = currentThreadTag();
    for (auto& slot : audioThreadSlots()) {
        if (convo::consumeAtomic(slot.tag, std::memory_order_acquire) == tag)
            return true;
    }
    return false;
}
```

**確定**: ✅ **ExecutionClass/canBlock は不要。`isAudioThread()` で十分。RefCountedDeferred::release() は `!isAudioThread()` で tryReclaim 許容を判定すればよい。**

### 11.3 SnapshotCoordinator — スレッド安全性の再確認

**ファイル**: `src/core/SnapshotCoordinator.cpp:36`, `src/audioengine/AudioEngine.Timer.cpp:209`

**調査結果**:

- `startFade()` の呼び出し経路: Timer callback (Non-RT Message Thread) → `createSnapshotFromCurrentState()` → `m_coordinator.startFade()` → `m_epochProvider->enqueueRetire()`
- **Non-RT からのみ呼ばれる** → `tryReclaim()` は安全
- `updateFade()` (SnapshotCoordinator.h:99) は Audio Thread (RT) から呼ばれるが、これは enqueueRetire を含まない

**重要**: Phase 1.2 の「却下」判断は再評価が必要。startFade は Non-RT のみ → **tryReclaim の追加は安全**。却下理由だった「IEpochProvider インターフェースの汚染」は、`SnapshotCoordinator` 内に static ヘルパー `enqueueWithRetry()` を置くことで解決可能。

**確定**: ✅ **Phase 1.2 は再評価: 却下→実施可能。ただしリスクは低いため P1 に格下げ。**

### 11.4 PolicyContext 型 — 設計のみの未定義型

**調査結果**: `PolicyContext` 型はコードベースに存在しない。設計上のコンセプトのみ。

```cpp
// RuntimePolicyEngine 設計コード — PolicyContext を参照しているが未定義
std::array<PolicyContext, kPolicySourceCount> m_contexts;
```

**確定**: ✅ **PolicyContext は定義不要。PolicyEngine は MonitorState(enum class) を直接入力として受け取る。`m_contexts` 配列は削除する。**

### 11.5 WorldLifecycleAudit — 不足メソッドの確認

**ファイル**: `src/audioengine/WorldLifecycleAudit.h`

**調査結果**:

- 既存メソッド: `onWorldPublished()`, `onWorldRetired()`, `activeWorldCount()`, `publishedCount()`, `retiredCount()`, `doubleRetireCount()`, `emitSnapshot()`, `tryDumpPeriodic()`
- 不足: `onFallbackOverflow()` ❌, `injectEvent()` ❌
- Private メンバ: `ringBuffer_(4096)`, `activeWorldCount_`, `publishedCount_`, `retiredCount_`, `doubleRetireCount_`, `lastDumpTimeUs_`, `kDumpIntervalUs(60s)`

**確定**: ✅ **`onFallbackOverflow()` と `injectEvent()` は新規追加が必要。ただし `onWorldRetired()` に `doubleRetireCount_` インクリメントが既にあるため、`onFallbackOverflow()` は HealthMonitor への通知ラッパーとして実装可能。**

### 11.6 DeferredRetireFallbackQueue — 拡張要確認

**ファイル**: `src/core/DeferredRetireFallbackQueue.h`

**調査結果**:

- 既存メソッド: `push()` (→size_t), `popAll()` (→vector), `size()`, `empty()`
- 不足: `estimatedBytes()` ❌, `overflowRate()` ❌, `notifyOverflow()` ❌
- エントリ構造体: `DeferredRetireFallbackEntry { ptr, deleter, epoch }` — `retryCount` フィールドなし
- ロック: `std::mutex` + `std::vector` ベース（上限なし）

**確定**: ✅ **拡張内容を確認: (1) push() 戻り値を bool に変更, (2) retryCount フィールド追加, (3) HardLimit(2000) 追加, (4) estimatedBytes() はエントリ数の `* sizeof(DeferredRetireFallbackEntry)` で代替可能**

### 11.7 emitShutdownTrace() — JSON と healthState

**ファイル**: `src/audioengine/ISRShutdown.cpp:135`

**調査結果**:

- 出力: schema, phase, phaseName, blockingReason, blockingReasonCode, transitionViolations, sh1-sh6 counters, verified
- 不足: healthState ❌, EmergencyDrain フラグ ❌
- `collectDrainAudit()` は既に healthState を取得している（`AudioEngine.Threading.cpp:84`）

**確定**: ✅ **Phase 3 では emitShutdownTrace() の JSON に `healthState` フィールドを追加。collectDrainAudit() の結果を渡せばよい。**

### 11.8 ShutdownBlockingReason — 多値化の要否

**ファイル**: `src/audioengine/ISRShutdown.h:46-53`

**調査結果**:

- 既存値 (8種類): None, PendingPublication, PendingRetire, ActiveCrossfade, DeferredPublish, QuarantineResident, RouterPendingRetire, ReaderActive, Unknown
- `RuntimeDrainAudit::BlockingReason` も同一の値セット
- `getPrimaryBlockingReason()` で既に優先順位付き単一理由を返せる

**確定**: ✅ **Primary+Secondary の多値化は不要。既存の単一 ShutdownBlockingReason で十分。Phase 5 は P2→P3 に格下げ（優先度最低）。**

### 11.9 DiscardReason — Expired 値の有無

**ファイル**: `src/audioengine/RuntimePublicationState.h:8-13`

**調査結果**:

- 既存値: None, ShutdownDiscard, StaleDiscard, SupersededDiscard
- 不足: Expired ❌

**確定**: ✅ **Phase 6 で `DiscardReason::Expired` を追加。DeferredPublishSlot の `enqueueTimestampUs` フィールドは既存（TTL 判定に利用可能）。**

### 11.10 CONVOPEQ_EMERGENCY_DRAIN — 実装状態と不整合

**ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:196-246`, `src/audioengine/ISRShutdown.cpp:80`

**調査結果**:

- `releaseResources()` は常に `EmergencyDrain` phase に遷移（194行目）→ `#ifdef` で実装内容のみ保護
- `advancePhase()` は `ReclaimComplete→VerifyDrained`（EmergencyDrain スキップ）→ **ifdef 保護なし**
- 不整合: `releaseResources()` は EmergencyDrain を経由するが、`advancePhase()` はスキップする

**確定**: ✅ **Phase 8.2 で以下を修正: (1) advancePhase() に `#ifdef CONVOPEQ_EMERGENCY_DRAIN` を追加して条件分岐を統一, (2) または advancePhase のスキップ動作を設計として明文化**

### 11.11 EQProcessor retire 呼び出し元の実数

**ファイル**: `src/eqprocessor/EQProcessor.Core.cpp`, `src/eqprocessor/EQProcessor.Parameters.cpp`, `src/eqprocessor/EQProcessor.Coefficients.cpp`

**調査結果**:

- `retireEQStateDeferred`: **13箇所**（Core.cpp: 3, Parameters.cpp: 10）← 設計書の「34箇所」は誤り
- `retireBandNodeDeferred`: **4箇所**（Core.cpp: 3, Coefficients.cpp: 1）← 設計書の「25箇所」は誤り
- 合計: **17箇所** ← 設計書の「59箇所」の約29%

**確定**: ✅ **影響範囲が設計想定の約1/3。戻り値 (void) キャスト追加のコストは大幅低減。設計書の数値を訂正。**

### 11.12 collectDrainAudit() — 既存 healthState 連携の確認

**ファイル**: `src/audioengine/AudioEngine.Threading.cpp:54-85`

**調査結果**:

```cpp
return convo::isr::RuntimeDrainAudit{
    .pendingPublication = ...,
    .pendingRetire = ...,
    ...
    // ★ B-2: HealthState 診断情報
    .healthState = m_healthMonitor.getHealthState()
};
```

**既に healthState を収集している**。`emitShutdownTrace()` の JSON 出力に含まれていないのみ。

**確定**: ✅ **`collectDrainAudit()` は既に健康状態を収集している。Phase 3 では emitShutdownTrace() への連携のみ追加。**

### 11.13 全未確定事項の確定サマリ

| # | 事項 | 旧状態 | 確定結果 |
|---|------|--------|----------|
| 1 | ShutdownResult 上位伝播 | ⏳ 部分解決 | ✅ **変更不要**。collectDrainAudit 経由で既に伝播 |
| 2 | ExecutionClass/canBlock | ❌ 未実装 | ✅ **不要**。isAudioThread() で十分 |
| 3 | SnapshotCoordinator retry | ❌ 却下 | ✅ **再評価: 実施可能**。Non-RT のみ |
| 4 | PolicyContext 型 | ⚠️ 未定義 | ✅ **不要**。MonitorState を直接入力 |
| 5 | WorldLifecycleAudit 不足メソッド | ❌ 不明 | ✅ **onFallbackOverflow/injectEvent 追加必要** |
| 6 | DeferredRetireFallbackQueue 拡張 | ❌ 不明 | ✅ **push→bool/retryCount/HardLimit 追加** |
| 7 | emitShutdownTrace + healthState | ⚠️ 未連携 | ✅ **JSON に healthState 追加。低コスト** |
| 8 | ShutdownBlockingReason 多値化 | ⚠️ 検討中 | ✅ **不要**。単一で十分。P2→P3 格下げ |
| 9 | DiscardReason::Expired | ❌ 不在 | ✅ **追加必要**。DeferredPublishSlot は TTL 対応可 |
| 10 | CONVOPEQ_EMERGENCY_DRAIN | ⚠️ ifdef不整合 | ✅ **advancePhase の ifdef 追加 or 明文化** |
| 11 | EQProcessor 呼び出し元数 | ⚠️ 59箇所推定 | ✅ **実測17箇所**（推定の29%） |
| 12 | collectDrainAudit healthState | ⚠️ 不明 | ✅ **既に実装済み** |

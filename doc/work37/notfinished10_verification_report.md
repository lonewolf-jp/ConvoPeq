# notfinished10.md 検証報告

> **日付**: 2026-06-13
> **対象**: doc/work37/notfinished10.md「実運用で破綻しにくい Practical Stable ISR Bridge Runtime」未達箇所レビュー
> **検証方法**: AiDex MCP, Serena MCP, grep/Select-String, codegraph MCP による全該当シンボルの実コード確認

---

## 調査サマリ

- 確認したソースファイル: 22 ファイル（.h/.cpp）
- 確認したシンボル: 40+ クラス/構造体/関数
- 検証ツール: `aidex_query`, `aidex_signature`, `serena find_symbol`, `serena get_symbols_overview`, `codegraph query_codebase`, `grep_search`

---

## 各レビュー項目の整合性検証

### ① HealthState が Shutdown Authority に組み込まれていない ✅ **正確**

| 項目 | 内容 |
|------|------|
| 該当箇所 | `RuntimeDrainAudit.h` L45-46, `AudioEngine.Processing.ReleaseResources.cpp` |
| 確認結果 | `healthState` は `diagLog` 出力のみ。Shutdown 完了判定 (`isFullyDrained()` 等) では未参照 |
| コメント | `// ★ B-2: HealthState（診断情報としてのみ保持。canShutdown 条件にはしない）` |
| 評価 | レビュー通り。Shutdown Authority に healthState は組み込まれていない |

---

### ② WorldLifecycleAudit が監査専用でリーク防止になっていない ✅ **正確**

| 項目 | 内容 |
|------|------|
| 該当箇所 | `WorldLifecycleAudit.h` L23, `AudioEngine.Threading.cpp` collectDrainAudit() |
| 確認結果 | `activeWorldCount` は DrainAudit に収集されるが、Shutdown Authority 条件ではない |
| コメント | `// ★ P3-B: World ライフサイクル監査（Diagnostic 限定） Shutdown 完了判定の Authority にはしない` |
| 評価 | レビュー通り |

---

### ③ Retire Queue Overflow が回復不能 ⚠️ **部分的に正確（改善あり）＋追加問題発見**

**ISRRetireRouter::enqueueRetire() の実際の動作:**

```
1. enqueueRetire 失敗
2. → tryReclaim を 1 回試行（500ms クールダウン付き）
3. → 再試行成功 → Success 返却
4. → 再試行失敗 → QueuePressure 返却 + overflowCount 増加
```

つまり完全な「回復不能」ではなく、tryReclaim+再試行は実装済み。

しかし **DeferredRetireFallbackQueue は未使用**。`src/core/DeferredRetireFallbackQueue.h` にクラスは存在するが、ISRRetireRouter の enqueueRetire 経路では使われていない。

**🔴 [NEW-A] 追加問題発見: EQProcessor 経路でのサイレントドロップ**

`EQProcessor.Core.cpp` L54-56:

```cpp
auto result = m_retireCoordinator->enqueueRetire(
    convo::isr::RetireAuthority::Granted,
    stackRouter, ptr, deleter, retireEpoch);
if (result == convo::isr::RetireEnqueueResult::Success)
    return true;
return false; // ← QueuePressure/QueueFull でデータが静かに消失
```

EQProcessor の retire が QueuePressure で失敗すると、再試行なしで `false` を返す → **retire データ消失**。実運用での最大リスクの一つ。

---

### ④ Deferred Publish が単一スロット ✅ **正確**

`RuntimePublicationOrchestrator.h`:

```cpp
std::optional<DeferredPublishSlot> deferredSlot_;
bool hasDeferred_ = false;
```

1 スロットのみ。高速連続変更 (A→B→C→D) で A,B,C が上書き消失。`deferredOverwriteCount_` で消失回数は監視可能だが、復元不可能。

---

### ⑤ Reader Stuck 判定が Epoch差のみ ⚠️ **部分的に正確（改善あり）**

**ReaderSlot の現状:**

- `residencyStartTimestampUs` は **存在する**（`EpochDomain.h` L120, L149, L257, L286）
- `enterReader()` で設定、`exitReader()` でクリア

**detectStuckReaders() の実際の判定:**

```cpp
// residencyTimeUs は収集される（StuckReaderInfo に格納）
const uint64_t startUs = convo::consumeAtomic(slot.residencyStartTimestampUs, ...);
const uint64_t residencyUs = (startUs != 0 && depth > 0) ? (nowUs - startUs) : 0;
info.residencyTimeUs = residencyUs;

// しかし Stuck 判定条件は epoch 差のみ
if (epochGap > stuckThreshold) {
    info.isStuck = true;
    break;
}
```

**HealthMonitor::diagnoseRetireStall() では residencyTimeUs を severity 判定に使用:**

```cpp
const bool severe = (stuckInfo.pendingRetireCount > 100 || stuckInfo.residencyTimeUs > 30'000'000);
```

つまり residencyTime は **severity 判定には使われるが、Stuck の有無判定には使われていない**。

---

### ⑥ RuntimeHealthMonitor が「検出のみ」→ Admission を止めない ❌ **不正確（すでに改善済み）**

`PublicationAdmission.cpp`:

```cpp
if (m_healthStateRef) {
    auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
    if (health == ISRHealthState::Critical) {
        return Decision::RejectedPressure;  // ← Critical で全 publish 拒否
    }
    if (health == ISRHealthState::Degraded) {
        return Decision::RejectedPressure;
    }
}
```

`AudioEngine.Threading.cpp` の `shouldRejectRebuildAdmissionForPressure()` でも HealthState Critical チェックあり:

```cpp
if (m_healthMonitor.getHealthState() == convo::ISRHealthState::Critical)
    return true;
```

**確認**: Critical 時に Admission で publish を拒否する回路は最新コードで実装済み。レビュー作成時点より後に修正された可能性が高い。

ただし **Degraded/Critical の区別が `RejectedPressure` で同一** である点は粒度不足（#10 の指摘）。

---

### ⑦ WorldLifecycleAudit が診断ファイル依存 ✅ **正確**

`WorldLifecycleAudit::emitSnapshot()`:

- 出力先: `evidence/world_lifecycle_audit.json`
- HealthMonitor へのイベント送信: **なし**

`tryDumpPeriodic()`: 60秒間隔で JSON ダンプするのみ。

**🔴 [NEW-C] 追加問題: doubleRetireCount が HealthMonitor 非連携**

二重 retire 検出カウンタ (`doubleRetireCount_`) は存在するが、HealthMonitor へのイベントとして通知されない。Reader 経由でない異常（二重 retire）を検出する唯一の手段がファイル監査のみ。

---

### ⑧ Publish Epoch と Retire Epoch の完全整合保証がない ⚠️ **部分的に正確**

`WorldLifecycleRecord`:

```cpp
struct WorldLifecycleRecord {
    uint64_t worldId;
    uint64_t publishEpoch;
    uint64_t retireEpoch;        // 0 = 未退役
    uint64_t publishTimestampUs;
    uint64_t retireTimestampUs;
    CorrelationId correlationId;  // ← 存在する
};
```

**既存の追跡手段:**

- `correlationId` は WorldLifecycleRecord に記録される → Publish→Retire の因果追跡の基盤は存在
- リングバッファは追記専用のため `retireEpoch` は常に 0（更新不可）
- 直近の retire 情報は別途 `lastRetiredWorldId_`/`lastRetireEpoch_` で追跡

レビュー提案の `publicationSequence` 追加は有効だが、`correlationId` による部分的な追跡基盤は既にある。

---

### ⑨ World Consistency が Shutdown Authority になっていない ✅ **正確**

`AudioEngine.Processing.ReleaseResources.cpp`:

```cpp
const auto cs = audit.verifyWorldConsistency();
if (cs != ConsistencyState::Consistent) {
    diagLog("[AUDIT] VerifyDrained: world consistency=" ...);
    // ★ shutdown は続行
}
emitEvidenceTickNonRt(true);
```

**確認**: 診断のみ。`diagLog` + `emitEvidenceTickNonRt` で終了。Shutdown は続行される。

---

### ⑩ HealthState と Admission の粒度が粗い ✅ **正確**

`ISRHealthState` は `Healthy`/`Degraded`/`Critical` の 3 段階のみ。

**`HealthCause` はコードベース全体に存在しない**。

Admission の拒否理由も Degraded と Critical で同一 (`RejectedPressure`)。各圧力タイプ（Retire/Reader/Crossfade/Publication）で異なる制御ができない。

---

### ⑪ Deferred Publish の滞留時間上限が拒否条件に使われていない ✅ **正確**

**`timeToLive` / TTL はコードベース全体に存在しない**。

`notifyTransitionComplete()` では age チェックなしに deferred publish を実行する:

```cpp
// 年齢確認なし → 30秒以上経過した古い要求も実行される
auto req = deferred.request;
submitPublishRequest(req);
```

---

### ⑫ Router Pending Retire に Hard Limit が無い ✅ **正確 + 追加指摘**

`routerPendingRetire` は `DrainAudit` で収集されるが、`RuntimeHealthMonitor::updateHealthState()` の Critical 条件に含まれていない。

**🔴 [NEW-B] 追加問題: RouterPendingRetire → Critical の連鎖がない**

Reader Stuck → Retire 滞留 → Router 蓄積 の連鎖で、RouterPendingRetire が増加しても HealthMonitor は Critical に遷移しない。Monitor の Tick 関数一覧:

```cpp
void RuntimeHealthMonitor::tick() noexcept {
    checkRetireStall();        // pendingRetireCount ベース
    checkPublicationStall();   // deferred/stall ベース
    diagnoseRetireStall();     // Reader stuck ベース
    checkCrossfadeTimeout();   // crossfade age ベース
    checkCrossfadeEventDrop();
    checkReaderSlotUsage();
    checkOverflowRate();       // overflow レートベース
    checkRetireReclaimLatency();
    updateHealthState();
    // ← routerPendingRetire のチェックなし
}
```

---

### ⑬ CorrelationId が Publish→Retire 完全追跡に使われていない ⚠️ **部分的に正確**

`WorldLifecycleRecord` に `correlationId` は含まれる:

```cpp
void onWorldPublished(uint64_t worldId, uint64_t epoch, CorrelationId cid) noexcept
```

`onWorldRetired()` では correlationId が引数にないため、retire 時にどの publish の retire かが追跡できない:

```cpp
void onWorldRetired(uint64_t worldId, uint64_t epoch) noexcept
// ← correlationId なし
```

ライフサイクル全体（Publish→Activate→Retire→Reclaim）を一本の ID で追跡する枠組みは不足。

---

### ⑭ Shutdown BlockingReason の優先順位が不十分 ✅ **正確**

```cpp
BlockingReason getPrimaryBlockingReason() const noexcept {
    if (pendingPublication > 0)    return PendingPublication;
    if (pendingRetire > 0)         return PendingRetire;
    if (activeCrossfadeCount > 0)  return ActiveCrossfade;
    // ... 単一理由のみ返却（vector/primary+secondary なし）
    return BlockingReason::Unknown;
}
```

`vector<BlockingReason>` や Primary/Secondary の区別はなし。複数異常時に根本原因が隠れる。

---

### ⑮ HealthMonitor が自動回復戦略を持たない ✅ **正確**

**`RecoveryAction` はコードベース全体に存在しない**。

HealthMonitor の責務は:

1. ✅ 検出（各種 `check*` / `diagnose*`）
2. ✅ 状態変更（`updateHealthState()` / `emitOnTransition()`）
3. ❌ 緩和（なし）
4. ❌ 回復（なし）
5. ✅ 最終手段（HealthState Critical → Admission 拒否）

例: RetirePressure 検出後、Publish Throttle → 改善なければ Critical の段階制御がない。

---

## レビュー未指摘の追加問題（NEW）

### [NEW-A] enqueueRetire 戻り値無視の系統的問題 🔴 **最重要（v2.0 で再定義）**

#### 初期報告の評価

`EQProcessor.Core.cpp` L54-56:

```cpp
auto result = m_retireCoordinator->enqueueRetire(...);
if (result == convo::isr::RetireEnqueueResult::Success)
    return true;
return false; // ← QueuePressure/QueueFull でサイレントドロップ
```

#### v2.0 深掘り調査の結果 — 問題は EQProcessor 固有ではない

**全ての enqueueRetire 呼び出し元を調査**した結果、`RetireEnqueueResult` の戻り値をチェックしている箇所は **一箇所も存在しない**:

| 呼び出し元 | ファイル | 戻り値処理 |
|-----------|----------|-----------|
| `DSPLifetimeManager::retire()` | `DSPLifetimeManager.h` L44 | ❌ 未チェック |
| `SnapshotCoordinator::setTarget()` | `SnapshotCoordinator.cpp` L36 | ❌ 未チェック |
| `SnapshotCoordinator::resetFadeStateAndRetireTarget()` | `SnapshotCoordinator.cpp` L67 | ❌ 未チェック |
| `SnapshotCoordinator::completeFade()` | `SnapshotCoordinator.cpp` L87 | ❌ 未チェック |
| `EQProcessor::enqueueDeferredDeleteWithFallback()` | `EQProcessor.Core.cpp` L50 | ❌ return false するが呼び出し元(retireEQStateDeferred)は void で結果無視 |
| `RefCountedDeferred` | `RefCountedDeferred.h` L23 | ❌ 未チェック |

つまり **`RetireEnqueueResult` は常に無視されている**。`Success` / `QueuePressure` / `QueueFull` / `Shutdown` の区別なく、全ての呼び出し元が戻り値を破棄している。

#### 本質的な問題

```cpp
// [P0-5] enqueue failure -> drop + telemetry (RT-safe).
// Non-RT 側の定期的な reclaim が backlog を消化することを期待。
```

コメントにある通り、設計思想は **「定期的な reclaim に期待する」楽観的モデル**。契約による信頼性（Contract-based reliability）ではなく、**期待による信頼性（Hope-based reliability）** で動作している。

`RetireEnqueueResult` の各値は `ISRAuthorityClass.h` できちんと定義されている:

- `Success`: bounded retire queue に enqueue 成功
- `QueuePressure`: フォールバック経路で受付（coordinator による対応必要）
- `QueueFull`: 高フォールバック深度到達（coordinator による即時対応必要）
- `Shutdown`: shutdown フェーズで new work 拒否

しかし **定義と実装が一致していない** — 値は定義されているが、誰も検査しない。

#### 修正された評価

| 項目 | 内容 |
|------|------|
| 問題の種類 | ❌ **単一バグではなく系統的設計問題** |
| 実害 | 低頻度だが QueueFull 時に retire が静かに消失。メモリリーク + データ喪失 |
| 真因 | enqueueRetire の戻り値契約（RetireEnqueueResult）が定義されているが、全呼び出し元が無視 |
| 推奨対応 | enqueueRetire の契約を見直し（失敗しない設計にする OR 全呼び出し元で結果を検査する） |
| 優先度 | **🔴 P0**（但し NEW-J が上位）。単純な fallback queue 追加ではなく、契約の再設計が必要 |

### [NEW-B] RouterPendingRetire が HealthMonitor Critical 条件に含まれていない 🟡

| 項目 | 内容 |
|------|------|
| 該当箇所 | `RuntimeHealthMonitor::updateHealthState()` |
| 問題 | RouterPendingRetire 増加が HealthState Critical に反映されない |
| 影響 | Reader Stuck → Retire滞留 → Router蓄積の連鎖を検出不可 |
| 推奨対応 | HealthMonitor に `checkRouterPendingRetire()` 追加 |

### [NEW-C] WorldLifecycleAudit::doubleRetireCount が HealthMonitor 非連携 🟡

| 項目 | 内容 |
|------|------|
| 該当箇所 | `WorldLifecycleAudit.h` L76（doubleRetireCount_） |
| 問題 | 二重 retire 検出カウンタが HealthMonitor へ通知されない |
| 影響 | 二重 retire を検出する唯一の手段がファイル監査のみ |
| 推奨対応 | doubleRetireCount > 0 で HealthMonitor へ Degraded/Critical イベント発火 |

### [NEW-D] Deferred Publish に TTL/age チェックがない 🟡

| 項目 | 内容 |
|------|------|
| 該当箇所 | `RuntimePublicationOrchestrator::notifyTransitionComplete()` |
| 問題 | 滞留 age 確認なしに古い publish 要求を実行 |
| 影響 | 30秒以上経過した古い publish が crossfade 解除後に実行される |
| 推奨対応 | DeferredPublishSlot に maxAgeUs 追加、超過時は DiscardReason::Expired |

### [NEW-E] レビュー#6 の主張は最新コードでは既に解決済み 🟢

| 項目 | 内容 |
|------|------|
| 該当箇所 | `PublicationAdmission.cpp` L27-31 |
| 進捗 | HealthState Critical 時の Admission 拒否は実装済み |
| 備考 | ただし粒度は粗い（Degraded も Critical も同じ RejectedPressure） |

### [NEW-K] enqueueRetire 戻り値契約の死文化 🔴 **P0（NEW-A の深掘り結果）**

`RetireEnqueueResult` の定義（`ISRAuthorityClass.h`）:

```cpp
enum class RetireEnqueueResult : std::uint8_t {
    Success = 0,      // bounded retire queue に enqueue 成功
    QueuePressure,     // フォールバック経路で受付（coordinator 対応必要）
    QueueFull,         // 高フォールバック深度（coordinator 即時対応必要）
    Shutdown           // shutdown フェーズで new work 拒否
};
```

**全呼び出し元の戻り値処理状況**（v2.0 深掘り）:

| 呼び出し元 | ファイル | 戻り値 |
|-----------|----------|--------|
| `DSPLifetimeManager::retire()` | `DSPLifetimeManager.h:44` | ❌ 未チェック |
| `SnapshotCoordinator::setTarget()` | `SnapshotCoordinator.cpp:36` | ❌ 未チェック |
| `SnapshotCoordinator::resetFadeStateAndRetireTarget()` | `SnapshotCoordinator.cpp:67` | ❌ 未チェック |
| `SnapshotCoordinator::completeFade()` | `SnapshotCoordinator.cpp:87` | ❌ 未チェック |
| `EQProcessor::enqueueDeferredDeleteWithFallback()` | `EQProcessor.Core.cpp:50` | ❌ return false するが呼び出し元 void |
| `RefCountedDeferred` | `RefCountedDeferred.h:23` | ❌ 未チェック |

**全ての呼び出し元が戻り値を無視**している。契約（enum 定義）は存在するが死文化している。

設計思想は「Non-RT 側の定期 reclaim に期待する」楽観的モデル（Hope-based reliability）。この契約違反が NEW-A の真因。|

---

# 追加検証: 回答者フィードバック NEW-F〜NEW-J に対する実コード分析

---

## NEW-F: WorldLifecycleAudit activeWorldCount の Authority 化は危険

**主張**: Crossfade中は正常状態でも Old + New の2つのWorldが共存するため、`activeWorldCount > 1 → failShutdown` は誤検知。正しくは `activeWorldCount - activeCrossfadeCount` を見るべき。

### 実コード検証

`RuntimePublicationCoordinator::publishWorld()` の処理フロー:

```cpp
auto* newWorld = worldOwner.release();                  // (1) 新World作成
auto* oldWorld = writeAccess_.publishAndSwap(newWorld);  // (2) アトミックスワップ
bridge_.didPublishRuntimeNonRt(*newWorld);                // (3) → onWorldPublished (+1)
bridge_.willRetireRuntimeNonRt(oldWorld);                 // (4) 退役準備通知
bridge_.retireRuntimePublishWorldNonRt(oldWorld, false);  // (5) → onWorldRetired (-1)
```

**activeWorldCount の推移**: 1 → (3で+1→2) → (5で-1→1)

つまり**publish と retire は同一関数内で同期的に実行**される。Crossfade の有無にかかわらず、`activeWorldCount` が 2 になるのは publish→retire のごく短い間だけ（同一関数コールスタック内）。

`CrossfadeRuntime` は DSP レベルのフェードオーバーレイであり、`RuntimePublishWorld` のライフサイクルとは独立している。

### 結論

| 論点 | 判定 |
|------|------|
| Crossfade中に2個の World が同時に存在するか | ❌ **現状の実装では NO**。publishWorld() 内で retire も同期的に完了する |
| crossfadeCount を減算する式は必要か | ⚠️ 現状は実害なし。ただし将来のアーキテクチャ変更で2World保持に変わった場合に備えた設計原則として有効 |
| 何を監視すべきか | `activeWorldCount` 単体ではなく、`verifyWorldConsistency()` の結果（published - retired == activeWorldCount の一貫性） |

**現行コードでは NEW-F の危険性は顕在化していない** が、設計原則として `activeWorldCount - crossfadeCount` 式を採用することは安全側の設計であり、**P2 で検討価値あり**。

---

## NEW-G: HealthState を Shutdown Authority に入れるのは半分正しい

**主張**: Shutdown をブロックしてはいけない。異常でも「止まれる」ことが優先。`ShutdownResult::CompletedWithCriticalHealth` のように完了結果に異常を記録すべき。

### 実コード検証

**ShutdownResult 型はコードベースに存在しない**。`ShutdownPhase` には `TimedOut` と `Failed` があるのみ:

```cpp
enum class ShutdownPhase : uint8_t {
    Running, AudioStopped, ObserverDrained, RetireClosed,
    EpochSettled, ReclaimComplete, EmergencyDrain,
    VerifyDrained, TimedOut, Failed, ShutdownComplete
};
```

`emitShutdownTrace()` は JSON ファイルに shutdown 結果を書き込むが、**呼び出し元への戻り値として HealthState 異常を伝える仕組みはない**。

完了結果に異常を記録する型（`ShutdownResult::CompletedWithCriticalHealth` 等）は存在しない。

### 結論

| 論点 | 判定 |
|------|------|
| HealthState で Shutdown をブロックすべきか | ❌ **ブロックすべきでない**。「異常でも止まれる」が優先。「Fail Operational > Fail Safe > Shutdown」の順 |
| ShutdownResult 型は存在するか | ❌ **存在しない** |
| 現在の完了記録手段 | `emitShutdownTrace()` で JSON ファイル出力のみ |
| 推奨方向 | `ShutdownResult` 型を新設。完了結果に `healthState` + `blockingReason` を含め、異常でも停止完了を保証 |
| 備考 | Reader Stuck 等で「shutdownできない」状態はメモリリークより危険 |
| 優先度 | **🔴 P0**（Fail-Safe の根幹に関わる。NEW-J 確定後に並行着手可） |

---

## NEW-H: Deferred Publish の多段キュー化は不要

**主張**: ISR Runtime では最新値優先が自然。queue化すると crossfade 負荷が増える。単一 Slot 維持 + overwriteCount/maxDeferredAge 監視で十分。

### 実コード検証

`DeferredPublishSlot` の現状:

```cpp
std::optional<DeferredPublishSlot> deferredSlot_;  // 単一スロット
bool hasDeferred_ = false;
```

上書き時の動作:

```cpp
if (hasDeferred_)                                   // 前回が残っていれば
    convo::fetchAddAtomic(deferredOverwriteCount_, 1);  // カウントのみ
deferredSlot_ = DeferredPublishSlot{...};            // 上書き
```

**最新値上書き**が現在の動作。前回の値は `discardReason=SupersededDiscard` の明示なしに消える → 未追跡の上書き消失。

### 結論

| 論点 | 判定 |
|------|------|
| 多段キュー化が必要か | ❌ **不要**。最新値優先が ISR の正しい動作 |
| 改善すべき点 | `DiscardReason::SupersededDiscard` を上書き時に明示的に記録し、telemetry で追跡可能にする |
| 優先度 | **P3**（単なるトレーサビリティ改善） |

---

## NEW-I: BlockingReason 多値化は Primary + Secondary で十分

**主張**: `vector<BlockingReason>` は過剰。診断用に `PrimaryReason + SecondaryReason` で十分。

### 実コード検証

```cpp
enum class ShutdownBlockingReason : uint8_t {
    None, PendingPublication, PendingRetire, ActiveCrossfade,
    DeferredPublish, QuarantineResident, RouterPendingRetire, ReaderActive, Unknown
};
```

現在は `ShutdownBlockingReason` 単一値のみ。`blockingReason_` は `markTimedOut()` / `markFailed()` で設定される。

**複数理由が同時に発生するケース**:

- Reader Stuck → Retire 滞留 → `PendingRetire` が表面化、真因の `ReaderActive` が隠れる
- Deferred Publish + Active Crossfade → `ActiveCrossfade` のみ表示

### 結論

| 論点 | 判定 |
|------|------|
| vector は必要か | ❌ **過剰**。Primary + Secondary で十分 |
| 推奨 | `ShutdownBlockingReason primary`, `ShutdownBlockingReason secondary` の2値 |
| 優先度 | **P2** |

---

## NEW-J: 最大の未解決問題 — 「Diagnostic→Policy→Authority 中間層欠如」（v2.0 で再定義）

**初期版の評価**: "classification gap"（監査系コンポーネントが未分類）
**v2.0 の再定義**: "escalation policy gap"（分類は存在するが政策層が不在）

### 実コード検証

**既存の分類フレームワーク**:

1. `ISRAuthorityClass.h`: `AuthorityClass` enum 定義あり（`Authoritative` / `Derived` / `Diagnostic` / `ExecutorLocal`）
2. `ISRRuntimeSemanticSchema.h`: `RuntimeAuthorityClass`, `SemanticCategory` enum 定義あり
3. `AudioEngine.h` の `RuntimePublishWorld`: `kFieldDescriptors` + `kRuntimeAuthorityInventory` で全21フィールドを **Authoritative / Derived / Diagnostic に分類済み**
4. `RuntimeGraph.h`: 同様に `kAuthorityInventory` で7フィールド分類済み
5. `config/authority_inventory.json`: 上記の集約 JSON。正確に生成されている

**不足しているのは Policy 層**:

現在のアーキテクチャ:

```text
Diagnostic (RuntimeDrainAudit / WorldLifecycleAudit / HealthMonitor)
    ↓ (直接)
HealthState (Critical / Degraded / Healthy)
    ↓ (直接)
Admission (Reject / Accept)
```

あるべきアーキテクチャ:

```text
Diagnostic
    ↓
Policy Engine (Severity × Persistence × Blast Radius)  ← ★ 欠落
    ↓
Recovery Action (Throttle / Reject / Drain / ...)
    ↓
Authority Escalation (条件を満たした場合のみ HealthState 変更)
    ↓
Fail-Safe (Shutdown はブロックせず結果に記録)
```

**例: 現在の ConvoPeq が Policy 層なしに直面する問題**:

| 事象 | 現状の扱い | Policy 層があるべき扱い |
|------|-----------|----------------------|
| doubleRetireCount 検出 | ファイル監査のみ | Severity=Critical → 即 HealthState 変更 |
| overflowCount=1 | ファイル監査のみ | Severity=Info → 無視 |
| overflowRate 高 (5回/秒) | HealthMonitor が暗黙的に Critical に昇格 | Policy: Blast Radius大 → Degraded → 改善なければ Critical |
| routerPendingRetire=100 | 監視のみ | Policy: Persistence 確認 → 30秒継続で Escalation |
| WorldConsistency Failure | diagLog のみ | Policy: Blast Radius最大 → 即 HealthCritical |

**classification gap ではなく escalation policy gap である理由**:

- `ISRAuthorityClass.h` の `AuthorityClass::Diagnostic` → 値が存在するだけで**未使用**。つまり分類そのものではなく「分類後の処理」が不足
- `RuntimeHealthMonitor::updateHealthState()` には **暗黙の昇格規則** が散在（retire stall→Critical, overflow rate→Critical 等）。Policy として明文化されていない
- `config/runtime_graph_baseline.json` の `"authoritative_fields": []` は、分類が空なのではなく **Policy 評価結果が未記入**

### 結論

| 論点 | 判定 |
|------|------|
| 分類フレームワークは存在するか | ✅ **存在する**（ISRAuthorityClass.h / ISRRuntimeSemanticSchema.h / kAuthorityInventory） |
| RuntimePublishWorld フィールドは分類されているか | ✅ **されている**（全21フィールドに RuntimeAuthorityClass 付与） |
| 不足しているもの | ❌ **Policy Engine 層**。Severity × Persistence × Blast Radius の評価と Recovery Action の決定 |
| 現在の昇格規則 | ⚠️ HealthMonitor の updateHealthState() に暗黙的に散在。明文化されていない |
| 推奨 | 1. `RuntimePolicyEngine` クラスを新設<br>2. Severity / Persistence / Blast Radius の3軸評価を導入<br>3. RecoveryAction 列挙型を定義（None / ThrottlePublication / RejectNewPublication / ForceRetireDrain / SuspendCrossfade / EmergencyShutdown）<br>4. 異常分類テーブルを Policy として明文化 |
| 優先度 | **🔴 P0**（全異常系改修の前提条件） |

| 優先度 | 項目番号 | タイトル | レビュー評価 | 検証結論 |
|--------|----------|----------|--------------|----------|
| **🔴 P0** | NEW-J | Diagnostic→Policy→Authority 昇格体系 | v2.0再定義 | ✅ **Policy Engine 層が欠落**。分類FW(ISRAuthorityClass)は存在。不足は「分類後の処理規則」 |
| **🔴 P0** | NEW-A | enqueueRetire 戻り値無視（系統的） | v2.0深掘り | ✅ **EQProcessor固有ではなく全呼び出し元が未チェック**。RetireEnqueueResult 契約の死文化 |
| **🔴 P0** | NEW-G | ShutdownResult 導入 | 新規 | ✅ **Fail-Safe 根幹**。HealthState で Shutdown をブロックするのは設計危険 |
| **🔴 P0** | #5 | Reader Stuck 判定改善 | 正確 | ✅ **Retire停止→Router蓄積→Shutdown遅延の起点**。上流異常を先に潰す |
| **🔴 P0** | #9 | WorldConsistency/Shutdown分離 | 正確 | ✅ → NEW-J の Policy 層で解決 |
| **🟡 P1** | #1 | HealthState/Shutdown分離 | 正確 | ⚠️ NEW-G により方針変更。ShutdownResult 連携が正解 |
| **🟡 P1** | NEW-G | HealthState→ShutdownResult 連携 | 新規 | ✅ ShutdownResult 型が不在。完了後異常記録の仕組みなし |
| **🟡 P1** | #3 | Retire Overflow退避不足 | 部分的に正確 | ⚠️ tryReclaim改善あり。但し NEW-A が本質 |
| **🟡 P1** | #5 | Reader Stuck判定不完全 | 部分的に正確 | ⚠️ residency収集あり、判定条件未使用 |
| **🟡 P1** | NEW-D | Deferred Publish TTLなし | 新規 | ✅ 単一Slot維持 + age チェック追加が必要 |
| **🟡 P1** | NEW-C | doubleRetireCount Health連携 | 新規 | ✅ → NEW-J の一部として解決 |
| **🟡 P1** | #8/#13 | 因果証跡（CorrelationId/Epoch） | 部分的に正確 | ⚠️ onWorldRetired に correlationId 引数不足 |
| **🟢 P2** | #10 | Health粒度不足（HealthCause欠如） | 正確 | ✅ HealthCause enum 不在 |
| **🟢 P2** | #14+NEW-I | BlockingReason改善 | 正確 | ✅ Primary+Secondary が適切。vector は過剰 |
| **🟢 P2** | NEW-F | WorldLifecycleAudit crossfade考慮 | 新規 | ⚠️ 実装上 activeWorldCount≯1 だが設計原則として重要 |
| **🟢 P2** | #2 | Worldリーク検出権限なし | 正確 | ⚠️ NEW-F により安全な式が必要と判明 |
| **🟢 P2** | #15 | HealthMonitor回復戦略なし | 正確 | ⚠️ NEW-J 策定後の検討事項 |
| **🟢 P2** | #7 | WorldLifecycleAuditファイル依存 | 正確 | ⚠️ → NEW-C で部分解決可能 |
| **🔵 P3** | #4+NEW-H | Deferred Publish多段キュー化 | 正確 | ❌ **不要と判断**。単一Slot維持 |
| **🟢 解決済** | #6 | HealthMonitor→Admission連携 | 不正確 | ❌ **最新コードでは既に解決** |

---

## コードベース完成度評価（v2.0 回答者フィードバック反映版）

| 領域 | 評価 | 備考 |
|------|------|------|
| 正常系 Publish/Retire/Shutdown | 95% | 基本的なフローは完成 |
| 監視・検出（HealthMonitor） | 90% | 多角的な監視あり。routerPendingRetire のみ未対応 |
| 診断・証跡 | 85% | WorldLifecycleAudit の retireEpoch 追跡に制限あり |
| 異常系ガバナンス（強制力） | **75〜80%** | ❌ **v2.0 で上方修正**。分類FW(ISRAuthorityClass)は存在するため「全くの無対策」ではない。「Policy 層の不在」が本質 |
| 自己回復能力 | 30% | tryReclaim 再試行のみ。RecoveryAction 不在。段階的緩和戦略なし |
| 契約遵守（戻り値検査） | **10%** | 🔴 **v2.0 新規評価**。RetireEnqueueResult の全呼び出し元が戻り値を無視。Hope-based reliability |

**総合**: 正常系 95%、監査系 90%、ガバナンス枠組 75〜80%。自己回復 30%、契約遵守 10%。
異常系ガバナンスは **Framework はあるが Policy がない** 状態。最終的な完成度評価は **83〜87%** 程度。

---

## Practical Stable ISR Bridge Runtime: 5-Layer Architecture Proposal

回答者のフィードバックに基づき、不足している Policy 層を含む推奨アーキテクチャ:

```text
Layer 0: Diagnostic (既存)
    WorldLifecycleAudit / RuntimeDrainAudit / RuntimeHealthMonitor
    異常検出のみ。制御権なし。

Layer 1: Policy Evaluation (新設: RuntimePolicyEngine)
    Severity (Info / Warning / Error / Critical)
        × Persistence (瞬間的 / 継続的 / 慢性)
        × Blast Radius (局所 / 限定的 / 全域)
    → 統一的な異常評価

Layer 2: Recovery Action (新設)
    enum class RecoveryAction {
        None,
        ThrottlePublication,
        RejectNewPublication,
        ForceRetireDrain,
        SuspendCrossfade,
        EmergencyShutdown
    };
    段階的制御: 緩和→改善なければ Critical

Layer 3: Authority Escalation (既存の AuthorityClass を活用)
    条件を満たした異常のみが HealthState を変更
    例: routerPendingRetire>100 + 30秒継続 → Authority

Layer 4: Fail-Safe (ShutdownResult 分離、NEW-G)
    Shutdown をブロックしない
    完了結果に healthState + blockingReason を記録
```

### 異常分類テーブル（提案）

| Event | Severity | Recovery | Authority | 現在の状態 |
|-------|----------|----------|-----------|-----------|
| overflowCount++ | Warning | None | No | ファイル監査のみ |
| overflowRate 高 | Error | ThrottlePublication | No | HealthMonitor が暗黙昇格 |
| routerPendingRetire > threshold | Error | RejectPublication | 条件付き | 監視のみ |
| ReaderStuck | Critical | ForceRetireDrain | Yes | Epoch差のみ判定、residency利用せず |
| doubleRetire | Critical | 即 HealthCritical | Yes | ファイル監査のみ |
| WorldConsistencyFailure | Critical | 即 HealthCritical | Yes | diagLog のみ |
| DeferredTTLExpired | Warning | Discard | No | TTLなし |
| CrossfadeTimeout | Error | SuspendCrossfade | 条件付き | 検出のみ |
| CorrelationMismatch | Critical | HealthCritical | Yes | 追跡基盤不十分 |
| enqueueRetire失敗 | Critical | ForceRetireDrain | Yes | **全呼び出し元が未チェック** |

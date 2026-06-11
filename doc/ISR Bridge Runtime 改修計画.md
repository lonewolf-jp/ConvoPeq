# Practical Stable ISR Bridge Runtime 改修計画

> 作成日: 2026-06-11
> ベース: `doc/Practical Stable ISR Bridge Runtime.md`（理想定義）
> 現状調査: Serena MCP + CodeGraph MCP + grep/Select-String による全ファイル解析

---

## エグゼクティブサマリー

ConvoPeq は既に `ISR 10層 Architecture` に基づく堅牢な設計に達しています。しかし理想形
**「RTスレッドは絶対に待たない。絶対に解放しない。絶対に判断しない。すべての危険操作はNonRTへ橋渡し(Bridge)し、状態遷移は観測可能で、停止時には完全排水(Drain)を保証する」**

と比較すると、以下の **7つの成熟度指標** のうち一部にギャップが存在します。

| # | 指標 | 現状 | 目標 |
|----|------|------|------|
| 1 | RTでdeleteしない | ✅ 達成 | RT path に delete なし確認済み |
| 2 | RTでlockしない | ✅ 達成 | RT path に mutex lock なし確認済み |
| 3 | Retireが必ずEpochを通る | ⚠️ 一部 | retireDSP() 直接呼び出しが残存 |
| 4 | Shutdownが完全Drain保証 | ⚠️ 一部 | VerifyDrained が明示的フェーズでない |
| 5 | Overflowがデータ喪失に直結しない | ❌ 未達 | 256固定キュー、溢れで intent をドロップ |
| 6 | HealthMonitorがDetect+Diagnose+Report可能 | ❌ 未達 | Detect のみ、Diagnose/Report なし |
| 7 | Coordinatorが唯一のAuthority | ⚠️ 一部 | retireDSP()直接呼び出しが Timer/CtorDtor/RebuildDispatch に残存 |

---

## Phase 1: Overflow管理の多段階化（最重要）

### 現状

- `ISRRetireRuntime::emitRetireIntent()`: `RETIRE_INTENT_QUEUE_SIZE=256` の固定MPSCキュー
- 満杯時: `droppedIntentCount_` をインクリメントして **intentをドロップ**
- `DeferredRetireFallbackQueue`（別ファイル）は存在するが **RetireRuntimeと未統合**
- Quarantine は最終セーフティネットとして機能している

### 理想

```
Intent Queue (lock-free 256)
       │
       ▼ (overflow)
 Fallback Queue (mutex-protected, unbounded vector)
       │
       ▼ (overflow)
 Quarantine (safety net, 管理対象)
```

> **注意**: 3段階で十分。EpochDomain に fallback queue を持たせるのは責務汚染。
> EpochDomain の責務は epoch tracking / reader tracking / reclaim timing のみに限定する。
> Overflow 管理は別コンポーネント（Queue Manager）で行う。
>
> **Fallback Queue は有界（bounded）にする**: 無制限の vector は Backpressure を消失させる。
> UI 暴走 → 数十万 Intent 蓄積 → Shutdown 数分 を防ぐため、上限を動的設定にする。
> 固定 4096 ではなく `retireHighWatermark * 2` を基本値とし、小規模環境・大規模IR切替・大量Crossfade
> に応じて調整可能にする。上限超過時は `QueueFull` として Coordinator へ escalation する。

### タスク一覧

#### T1-1: RetireRuntime の FallbackQueue 統合

- **ファイル**: `src/audioengine/ISRRetire.h`, `src/audioengine/ISRRetire.cpp`
- **内容**: `ISRRetireRuntime` に `DeferredRetireFallbackQueue*` の参照を追加
- **変更**:
  - `emitRetireIntent()`: MPSCキュー満杯時、即ドロップせず `fallbackQueue_->push()` を試行
  - `dequeuePendingRetireIntents()`: fallback queue も同時に排出
  - `fallbackDrainRequired_` フラグ追加（Monitor が認識可能）
  - **必須メトリクス追加**:
    - `fallbackOccupancy` — 現在の Fallback 使用量
    - `fallbackHighWatermark` — Fallback 使用量の最大値（リセット可能）
    - `fallbackOverflowCount` — Fallback 溢れ回数
- **検証**: OverflowCount と DroppedIntentCount の差が減少し、fallbackOccupancy/fallbackHighWatermark が HealthEvent で観測可能であることを確認

> **削除理由**: EpochDomain は epoch tracking / reader tracking / reclaim timing のみを責務とする。
> Overflow 管理を EpochDomain に持たせると責務汚染になる。Intent → Fallback(4096) → QueueFull Escalation で十分。

#### T1-2: QueueFull → Coordinator Escalation（IntentOverflowRegistry は新設しない）

- **ファイル**: `src/audioengine/ISRRetire.h`, `src/audioengine/ISRRetire.cpp`, `src/audioengine/ISRAuthorityClass.h`
- **内容**: Overflow が発生した場合、第3キュー（IntentOverflowRegistry）を追加するのではなく、
  既存の `RetireEnqueueResult::QueueFull`（`ISRAuthorityClass.h` に既存）を上位へ伝播させる。
- **設計意図**:
  - 現行 `RetireEnqueueResult` は既に `Success → QueuePressure → QueueFull → Shutdown` の段階を持つ
  - IntentOverflowRegistry のような第3キューは実質的な無制限キューとなり、Backpressure を消失させる
  - Backpressure 消失 → UI暴走 → 数万Intent蓄積 → Shutdown数十秒 のリスク
  - Practical Stable ISR では「overflow を保存すること」より「なぜ overflow したか」を診断・上位通知する方が重要
- **変更**:
  - `ISRRetireRouter::enqueueRetire()` が `QueueFull` を返した場合の escalation 経路を確立:

    ```
    ISRRetireRouter → HealthMonitor → Coordinator (Admission PressureLevel)
    ```

  - **Coordinator を監視システムにしない**: Coordinator 経由の escalation は循環依存（Coordinator → HealthMonitor → Coordinator）を招く。
  - 正しい経路: `RetireRuntime` が `QueueFull` を `HealthMonitor` に通知 → `HealthMonitor` が診断 → `Coordinator` の Admission PressureLevel を引き上げ
  - HealthMonitor が `QueueFull` の継続を検出 → Diagnose → Report (Escalate)
  - **IntentOverflowRegistry は新設しない**。代わりに既存の `droppedIntentCount_` / `overflowCount_` + `RetireEnqueueResult` で監視する。
- **検証**: QueueFull 発生時に Coordinator の Admission PressureLevel が適切に引き上げられることを確認

---

## Phase 2: Retire Pipeline の Epoch 一元化（PR-3 完了の前提）

### 現状

- `retireDSP()` が Timer/CtorDtor/RebuildDispatch から直接呼ばれている
- `DSPLifetimeManager::retire()` 経路は releaseResources のみで使用
- Epoch を通らない retire 経路が存在する

### 理想

```
retireDSP() → DSPLifetimeManager → Coordinator → EpochDomain → Reclaim
```

すべての retire が EpochDomain の `enqueueRetire()` を通ること。

### タスク一覧

#### T2-1: DSPLifetimeManager の全面適用

- **ファイル**:
  - `src/audioengine/AudioEngine.CtorDtor.cpp`
  - `src/audioengine/AudioEngine.RebuildDispatch.cpp`
  - `src/audioengine/AudioEngine.Timer.cpp`
  - `src/audioengine/DSPLifetimeManager.h`
- **内容**: `retireDSP()` の全呼び出し箇所を `DSPLifetimeManager::retire()` 経由に置換
- **変更**:
  - `CtorDtor.cpp`: デストラクタ内の `retireDSP()` → `lifetimeManager.retire()` に変更
  - `RebuildDispatch.cpp`: rebuild完了後の `retireDSP()` → 同
  - `Timer.cpp`: クロスフェード完了後の `retireDSP()` → 同
  - `DSPLifetimeManager` に `enqueueRetireWithEpoch()` 追加（epoch 管理を隠蔽）
- **検証**: `retireDSP` の全 grep 一致がゼロになったことを確認

#### T2-2: EpochDomain 経由の Reclaim 一元化

- **ファイル**:
  - `src/audioengine/DSPLifetimeManager.h`
  - `src/core/EpochDomain.h`
- **内容**: `DSPLifetimeManager::retire()` が必ず `ISRRetireRouter` → `EpochDomain::enqueueRetire()` を通ることを保証
- **変更**:
  - `DSPLifetimeManager` コンストラクタに `ISRRetireRouter*` を必須化
  - `retire()` 内部で `enqueueRetireEpochBounded()` を必ず呼ぶ
  - `deleteRetiredDSP()`（AlignedObjectDeleter）を deleter として渡す
- **検証**: EpochDomain の `pendingRetireCount()` が retire 呼び出しと一致することを確認

#### T2-3: retireDSP 禁止 CI 追加（Phase2.5）

- **ファイル**: `.github/scripts/verify-isr-maturity.ps1`（新規、または既存 verifier に統合）
- **内容**: Phase2 完了直後から、`retireDSP` の直接呼び出しを CI で禁止する。
- **変更**:
  - CI スクリプトに retireDSP 直接呼び出し検出を追加:
    ```powershell
    # retireDSP direct call gate (post-unification)
    $retireDspMatches = Select-String -Path @(Get-ChildItem -Recurse -Filter "*.cpp","*.h" src | % FullName) -Pattern "retireDSP\("
    if ($retireDspMatches.Count -gt 0) { exit 1 }
    ```
  - `DSPGuard` 等のラッパ経由も検出対象とする
- **検証**: `retireDSP` が1箇所でも残ったら CI Fail

---

## Phase 3: Shutdown Pipeline の完全 Drain 保証

### 現状

- ShutdownRuntime FSM: `Running → AudioStopped → ObserverDrained → RetireClosed → EpochSettled → ReclaimComplete → ShutdownComplete`
- タイムアウト時: `markTimedOut()` で `TimedOut` フェーズに移行（データ損失の可能性あり）
- Verify Empty が明示的フェーズでない
- `isFullyDrained()` は存在するが drain パイプラインに統合されていない

### 理想

```
StopAccepting → StopAudio → DrainIntent → DrainRetire → EpochSettled → ReclaimComplete → VerifyDrained → ShutdownComplete
```

> **注意**: `StopReader`（強制 unregister）は追加しない。RCUReader は
> `preferredThreadId / activeThreadId / ownerThreadToken` の再利用設計であり、
> 強制 unregister を入れると Shutdown 中の Late Callback との競合が起きる。
> `ReaderCount==0` を待つ方針を採ること。
>
> **VerifyDrained の位置**: ReclaimComplete の**後**に配置する。理由:
> VerifyDrained 内部で `pendingRetireCount` を確認するなら、reclaim が未完了の可能性がある。
> Verify は最終監査であり、ReclaimComplete 後に配置すべき。
> 現行 `EpochDomain::pendingRetireCount()` で reclaim 完了を確認した上で、VerifyDrained で最終監査を行う。
>
> ```
> EpochSettled → ReclaimComplete → VerifyDrained → ShutdownComplete
> ```

### タスク一覧

#### T3-1: VerifyDrained フェーズ追加

- **ファイル**: `src/audioengine/ISRShutdown.h`, `src/audioengine/ISRShutdown.cpp`
- **内容**: `ShutdownPhase` 列挙に `VerifyDrained` を追加
- **変更**:
  - `Running(0) → StopAcceptingWork(1) → StopAudio(2) → StopPublish(3) → ObserverDrained(4) → RetireClosed(5) → EpochSettled(6) → ReclaimComplete(7) → VerifyDrained(8) → ShutdownComplete(9)`
  - **VerifyDrained は ReclaimComplete の後に配置**: `EpochSettled → ReclaimComplete → VerifyDrained → ShutdownComplete` の順序。Verify は最終監査であり、reclaim 完了後に pendingRetireCount を確認する。
  - `advancePhase()` に対応するケース追加
  - `TimedOut(10)`, `Failed(11)` は terminal 状態として維持
  - **StopReader フェーズは追加しない**: RCUReader の強制 unregister は Late Callback との競合を招く。代わりに ObserverDrained で ReaderCount==0 到達を待つ。
- **検証**: shutdown trace に VerifyDrained が含まれることを確認

#### T3-2: VerifyDrained の具体実装（DrainAudit = Evidence, isFullyDrained = Authority）

- **ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
- **内容**: `VerifyDrained` フェーズで `collectDrainAudit()` + `isFullyDrained()` を実行。
- **変更**:
  - `releaseResources()` 内で `setShutdownPhase(ShutdownPhase::VerifyDrained)` 後に `collectDrainAudit()` を実行
  - DrainAudit が以下の**全項目**をカバーすることを確認:
    - `pendingPublication` / `pendingRetire` / `activeCrossfadeCount`
    - `deferredPublish` / `routerPendingRetire`
    - `quarantineResident`（監査のみ）
  - **isFullyDrained() を唯一の判定 Authority とする**: 現状 `isFullyDrained()` は `runtimePublicationBridge_.isFullyDrained()` へ集約されており、DrainAudit + Bridge状態 + Fallback状態 を統合している。DrainAudit はこの判断の Evidence Source として使用する。
  - `isFullyDrained() == false` の場合、`markTimedOut()` でタイムアウト記録
  - **強制 drain は実施しない**: 強制 drain は RCU 系で `advanceEpoch()` の強制実行として実装されるリスクがある。Reader 残留中に epoch を強制進行すると UAF を引き起こす。
  - タイムアウト後は診断のみ（BlockingReason 特定 + diagnostic log 出力）。回復は上位層（次回起動時または運用者判断）に委ねる。
- **設計意図**: `collectDrainAudit()` が Evidence（証拠）を収集し、`isFullyDrained()` が Authority（判定）を行う。
  両者を分離することで、監査ログと判定ロジックの責務を明確にする。
- **検証**: 正常時は `ShutdownComplete` 到達、異常時は `TimedOut` または `Failed` に到達

> **StopReader 非推奨の理由**: RCUReader は `preferredThreadId / activeThreadId / ownerThreadToken`
> の再利用設計であり、強制 unregister を入れると Shutdown 中の Late Callback との競合が起きる。
> 代わりに `ObserverDrained` フェーズで ReaderCount==0 への到達を待つ方針を維持する。

---

## Phase 4: HealthMonitor の Diagnose → Recover 拡張

### 現状

- `RuntimeHealthMonitor::tick()`: `checkRetireStall()` + `checkPublicationStall()`
- 状態遷移検出時: `emitOnTransition()` で callback 発火のみ
- 回復処理: なし

### 理想

```
tick() → checkRetireStall() → diagnoseRetireStall() → report()
tick() → checkPublicationStall() → diagnosePublicationStall() → report()
```

> **重要**: HealthMonitor は **Detect → Diagnose → Report (Escalate)** まで。
> **Recover / VerifyRecovery は実装しない**。
>
> - recoverRetireStall() での強制 `advanceEpoch()` は RCU 原則違反。
>   Reader が生きている可能性がある中で epoch を強制進行すると UAF を引き起こす。
> - Quarantine への強制移動も危険。Stall の原因が Reader 残留なら問題は解決しない。
> - 回復は上位層（運用者判断または外部監視）に委ねる。

### タスク一覧

#### T4-1: diagnoseRetireStall の実装

- **ファイル**: `src/audioengine/RuntimeHealthMonitor.cpp`, `src/audioengine/RuntimeHealthMonitor.h`
- **内容**: Retire Stall の原因診断
- **変更**:
  - `diagnoseRetireStall()` 追加: EpochDomain の `detectStuckReaders()` を呼び reader stuck の有無を確認
  - `getMinReaderEpoch()` と `currentEpoch()` の差から epoch 進行停止の原因を特定
  - 診断結果を `HealthEvent` の `detailCode` に反映
- **検証**: テストで reader stuck 時の診断結果が正しいことを確認

> **T4-2/T4-3/T4-4（Recover / VerifyRecovery）は削除**。
> 理由: recoverRetireStall での強制 advanceEpoch は RCU 原則違反により UAF を引き起こす可能性がある。
> quarantine への強制移動も Stall 原因（Reader 残留）の解決にならない。
> HealthMonitor の責務は Detect → Diagnose → Report (Escalate) までに限定する。

---

## Phase 4.5: Reader Residency Diagnostics（新規）

### 現状

- `EpochDomain::detectStuckReaders()` は存在するが、検出した Reader の詳細（threadId / ownerToken / epoch / residencyTime）を HealthEvent に出力していない
- `HealthMonitor::diagnoseRetireStall()` の診断結果が粗い
- 実運用で最も危険な `ReaderCount > 0 → Epoch進まず → Retire詰まり → Shutdownタイムアウト` の診断力が弱い

### 理想

HealthMonitor が以下の Reader 残留情報を出力可能であること:

```
ReaderIndex, threadId, ownerToken, epoch, depth, residencyTimeUs, isStuck
```

### タスク一覧

#### T4-5-1: ReaderSlot 拡張（steady_clock timestamp 方式）

- **ファイル**: `src/core/EpochDomain.h`
- **内容**: `ReaderSlot` に `residencyStartTimestampUs`（steady_clock ベース）を追加
- **変更**:
  - `ReaderSlot` に `std::atomic<uint64_t> residencyStartTimestampUs{0}` を追加
  - `enterReader()` / `RCUReader::enter()` で `residencyStartTimestampUs = getCurrentTimeUs()` を更新
  - `exitReader()` / `RCUReader::exit()` で `residencyStartTimestampUs = 0` をクリア
  - `detectStuckReaders()` で `residencyTimeUs = getCurrentTimeUs() - residencyStartTimestampUs` を計算
- **設計意図**: Epoch ベースの計算（`epochGap × epochDuration`）は不正確。epoch は publish 頻度に依存し、UI 停止中は進まず、大量 publish 中は見かけ上過大になる。`steady_clock` の絶対タイムスタンプを使用することで実時間での滞留時間を正確に測定する。
- **検証**: StuckReaderInfo に residencyTimeUs（実時間ベース）が含まれることを確認

#### T4-5-2: HealthEvent への Reader 詳細追加

- **ファイル**: `src/audioengine/RuntimeHealthMonitor.h`, `src/audioengine/RuntimeHealthMonitor.cpp`
- **内容**: HealthEvent に Reader 詳細フィールドを追加
- **変更**:
  - `HealthEvent` に `readerIndex`, `readerEpoch`, `readerDepth`, `residencyTimeUs` を追加
  - `diagnoseRetireStall()` で `detectStuckReaders()` から取得した情報を HealthEvent に反映
  - 全 Reader の残留状態を定期的に Report として出力（Normal 時でも情報提供）
- **検証**: HealthMonitor.tick() の出力に Reader 詳細が含まれることを確認

#### T4-5-3: Shutdown 時の Reader 残留診断強化

- **ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
- **内容**: Shutdown 中に Reader が残留している場合の診断を強化
- **変更**:
  - `ObserverDrained` フェーズで `activeReaderCount() > 0` の場合、`detectStuckReaders()` を呼び全 Reader 情報を diagnostic log に出力
  - 残留 Reader の threadId / ownerToken を特定可能にする
- **検証**: shutdown trace に Reader 残留情報が含まれることを確認

---

## Phase 5: Coordinator 単一 Authority の完全化

### 現状

- `ISRRuntimePublicationCoordinator` が publish の Authority
- しかし retire は `retireDSP()` 直接呼び出しが複数箇所に残存
- `RuntimePublicationOrchestrator` と `ISRRuntimePublicationCoordinator` の責務境界が不明確

### 理想

```
Coordinator(Publish) → Orchestrator(Admission → Executor → Transition)
Coordinator(Retire)  → LifetimeManager(Retire → Epoch → Reclaim)
```

両方の Coordinator が唯一の Authority。

### タスク一覧

#### T5-1: RuntimePublicationCoordinator の責務明確化（retire Authority 確立が主目的）

- **ファイル**: `src/core/RuntimePublicationCoordinator.h`, 検証スクリプト群
- **内容**: Publication と Retire の Authority を明確化。ただし **Authority Framework 自体は既存**（`validateAuthorityInventorySet()` / `validateAuthorityInventoryAgainstDescriptors()` が既に存在）。不足しているのは **retire Authority の確立**。
- **変更**:
  - `kRuntimeAuthorityInventory` と `kRuntimeReadAuthorityInventory` のコメント強化
  - 各メソッドに `// Authority: PublicationCoordinator` または `// Authority: LifetimeManager` のタグ追加
  - 既存の `validateAuthorityInventorySet()` / `validateAuthorityInventoryAgainstDescriptors()` を使った CI 検証を強化
  - テストファイル `RuntimeSemanticSchemaValidationTests.cpp` に対応する assert 追加
  - **retireDSP の Authority を DSPLifetimeManager に一元化**（Phase2 完了が前提）
- **検証**: CI で retire Authority 違反（retireDSP 直接呼び出し）が検出可能になること

#### T5-2: retireDSP 全削除（RetireDeleter 新設は任意）

- **ファイル**:
  - `src/audioengine/AudioEngine.h`（`retireDSP` 宣言）
  - `src/audioengine/AudioEngine.Retire.cpp`
  - `src/audioengine/DSPLifetimeManager.h`
- **内容**: T2-1 完了後、`retireDSP()` メソッドそのものを削除。`deleteRetiredDSP()` の管理先は状況に応じて選択。
- **変更**:
  - `AudioEngine.h` から `retireDSP()` 宣言を削除
  - `AudioEngine.Retire.cpp` から `retireDSP()` 実装を削除
  - `deleteRetiredDSP()` の管理先は以下から選択:
    - **案A（推奨）**: `ISRRetireRouter` または `EpochDomain` に閉じ込める。現行の deleter 注入型 `ISRRetireRouter::enqueueRetire(ptr, deleter, epoch)` を活用すれば新規クラス不要。
    - **案B（オプション）**: `RetireDeleter` 新規クラス（責務分離は綺麗だが4コンポーネント化による運用コスト増）
- **設計意図**: `RetireDeleter` 新設は必須ではない。現状の `enqueueRetire(ptr, deleter, epoch)` の deleter 注入型が既に Reclaim Authority を表現している。
  `DSPLifetimeManager` → `ISRRetireRouter(delete実体)` → `EpochDomain` で十分明快。
- **検証**: 全ファイルで `retireDSP` の参照がゼロになったことを確認

#### T5-3: Orchestrator/Coordinator 境界明確化

- **ファイル**:
  - `src/audioengine/RuntimePublicationOrchestrator.h`
  - `src/audioengine/ISRRuntimePublicationCoordinator.h`
  - `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
- **内容**: 責務境界の文書化とインターフェース整理
- **変更**:
  - `Orchestrator`: UI/タイマーからの publish 要求を受け付け、Admission 判定、Executor 委譲、DSPTransition 起動
  - `Coordinator`: RuntimeWorld の publish/retire 権限のみ（world pointer 操作、Epoch 経由の retire）
  - `Orchestrator` は `Coordinator` を内部で保持（委譲）
  - 循環依存を避けるため `Orchestrator` → `Coordinator` の一方方向のみ
- **検証**: 全 publish/retire 操作が Orchestrator 経由であることを確認

---

## Phase 6: RT 判断の最小化

### 現状

- RT は `getNextAudioBlock()` 内で以下の判断を行っている:
  - `lifecycleState` の確認
  - `isShutdownInProgress()` の確認
  - クロスフェード進行状態の参照
  - `runtimeUuid` の確認（stale world 検出）

### 理想

RT は **読むだけ**:

```cpp
RuntimeWorld* world = currentWorld.load();
world->dsp->process();
if (retireNeeded) emitRetireIntent();
```

### タスク一覧

#### T6-1: lifecycleState 判断の隔離

- **ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- **内容**: `lifecycleState` の RT チェックを最小化
- **変更**:
  - `LifecycleIsolationRuntime::isAudioCallbackAllowed()` に委譲（既存の `lifecycleRuntime_.enterAudioCallback()` で代用可能）
  - `lifecycleState` の直接 atomic read を排除（LifecycleIsolationRuntime 経由に統一）
- **検証**: Debug ビルドで `lifecycleState` の直接 read がゼロになったことを確認

#### T6-2: shutdown 判断の隔離

- **ファイル**: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- **内容**: `isShutdownInProgress()` の RT 呼び出しを最適化
- **変更**:
  - `ShutdownRuntime::isShutdownInProgress()` の代わりに、`LifecycleIsolationRuntime` が提供する `isActive()` で代用
  - `shutdownRuntime_.markLateCallback()` の呼び出しは維持（bounded teardown 保証のため）
- **検証**: 呼び出し経路が 1 箇所（`getNextAudioBlock` 冒頭）のみであることを確認

#### T6-3: クロスフェード判断の隔離（限定版）

- **ファイル**:
  - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- **内容**: RT のクロスフェード判断のうち、RuntimeWorld の情報で代替可能なものを移行
- **制約**: `CrossfadeRuntime` の atomic read 自体は **RT 違反ではない**。
  `CrossfadeRuntime` は Execution State Cache であり、`atomic<bool>` / `LinearRamp` の読み取りは
  RT 安全である。そのため **完全排除までは行わない**。
- **変更**（限定）:
  - `RuntimeState`（RuntimePublishWorld）に `transition.active` フラグを追加（既存）
  - 可能な範囲で RT は `world->execution` や `world->overlap` の値も参照する
  - ただし `crossfadeRuntime_.getGain().isSmoothing()` や `getDryScaleGain().getNextValue()` の
    RT からの読み取りは維持（LinearRamp はレンダリングに必須であり、RT 安全）
- **検証**: RT の `crossfadeRuntime_` 読み取りが `getGain()` / `getDryScaleGain()` に限定され、
  その他の判断用 atomic（`isPending()` / `useDryAsOld()` 等）が RuntimeWorld 経由になったことを確認

---

## Phase 7: Publication Pipeline の Observe 明示化

### 現状

- `RuntimeReaderContext` と `ObservedRuntime` は存在する
- Publication Pipeline: Build → Validate → Admission → Publish → (暗黙の Observe) → Retire Old
- Observe 完了の明示的な検証がない

### 理想

```
Build → Validate → Admission → Publish → Observe (explicit) → Retire Old
```

Publish 後、新 world が全 reader に観測されるまで Retire Old を待機する。

> **⚠️ 実装非推奨**: Observe 完了待機は RCU の基本思想と衝突する。
> RCU の本質は **Grace Period を待つが、Reader 個別の観測完了は待たない** ことにある。
> `isObservedByAllReaders()` は実質的に Reader Tracking System 化を意味し、
> RCU → Reference Counting 方向へ近づく。
>
> Practical Stable ISR Bridge Runtime としては **実装しない** ことを推奨する。
> 代わりに既存の Epoch ベースの Grace Period 待機（`getMinReaderEpoch()` による間接観測）で十分。
>
> 理論的興味としての記述は残すが、タスク化は行わない。

---

## Phase 8: 改善の検証基盤

### T8-1: 7 条件の自動検証スクリプト

- **ファイル**: `.github/scripts/verify-isr-maturity.ps1`（新規）
- **内容**: 7つの成熟度指標を自動検証
- **検証**:
  1. RTで delete がないこと: `src` 配下の RT ファイル（getNextAudioBlock/processBlockDouble を含む）で `delete` キーワードがゼロ
  2. RTで lock がないこと: 同上で `mutex`, `lock()`, `condition_variable` がゼロ
  3. Retire が必ず Epoch を通ること: `retireDSP` の直接呼び出しがゼロ
  4. Shutdown が完全 Drain を保証すること: shutdown trace が `ShutdownComplete` または `TimedOut` に到達
  5. Overflow がデータ喪失に直結しないこと: `droppedIntentCount_` が常に 0
  6. HealthMonitor が Detect + Diagnose + Report を実装していること
  7. Coordinator が唯一の Authority: `retireDSP` 直接呼び出しがゼロ + 全 retire が `DSPLifetimeManager` 経由
  8. **EpochDomain 直接呼び出し監査**: 以下の EpochDomain メソッドの呼び出し元が ISRRetireRouter 経由であること:
     - `enqueueRetire()` — Router 以外から直接呼ばれていないこと
     - `publishEpoch()` — Router 以外から直接呼ばれていないこと
     - `tryReclaim()` — Router 以外から直接呼ばれていないこと
     - `pendingRetireCount()` — 監査・診断以外から直接呼ばれていないこと

### T8-2: 既存検証スクリプトの拡張

- **ファイル**: `.github/scripts/isr-verify-common.ps1` 等
- **内容**: 既存 verifier に上記検証を追加
- **変更**:
  - T8-1 の検証項目を phase2 の verifier 群に統合
  - `isr-run-tiered-verification.ps1` の smoke 層に追加

---

## 改修の推奨順序（再評価版）

```
Sランク（実施推奨）
┌─────────────────┐
│ Phase 2         │
│ Retire Epoch統一 │
└─────────────────┘
┌─────────────────────────────┐
│ Phase 4.5                   │
│ Reader Residency Diagnostics│
└─────────────────────────────┘
┌─────────────────┐
│ Phase 8         │
│ 検証基盤         │
└─────────────────┘

Aランク
┌─────────────────────────────────┐
│ Phase 4                         │
│ Diagnose拡張 (Detect→Diagnose→Report)│
└─────────────────────────────────┘
┌─────────────────────────────────┐
│ Phase 3                         │
│ VerifyDrained (DrainAudit唯一)   │
└─────────────────────────────────┘

Bランク
┌─────────────────┐
│ Phase 1         │
│ Overflow改善    │
│ (Fallback4096   │
│  +QueueFull Esc)│
└─────────────────┘

Cランク（Authority文書化）
┌─────────────────┐
│ Phase 5         │
│ Coordinator     │
│ Authority文書化  │
└─────────────────┘

Dランク（最適化）
┌─────────────────┐
│ Phase 6         │
│ RT判断削減・限定版│
└─────────────────┘

実装非推奨
  Phase 4 Recover系     — RCU原則違反（epoch強制進行でUAF）
  Phase 7 Observe       — RCU思想と衝突（Reader Tracking System化）
  IntentOverflowRegistry — Backpressure消失（QueueFull Escalationで代替）
  RetireDeleter新設     — 必須ではない（Routerのdeleter注入型で十分）
  StopReader            — Late Callback との競合
```

### 優先度理由（最終版）

#### Sランク（実施推奨）

> **最重要**: Phase 2 が完了するまでは他の Phase に着手しないことを強く推奨する。
> 現状の最大の構造的不整合は `retireDSP()` 直呼び出しの分散にあり、これが Coordinator Authority
> の未確立と Epoch 保護のバイパスの両方を引き起こしている。Phase 2 を最初に完了することで、
> 後続 Phase の実装基盤が安定する。

1. **Phase 2（Retire Epoch統一）**: `retireDSP()` 直接呼び出し排除による Coordinator Authority 確立。ISRRetireRouter が既存のため統合リスク最小。DSPLifetimeManager が現状単なるラッパ（`engine_.retireDSP(dsp)`）であることも確認済み。
    Phase2 完了直後の T2-3（retireDSP 禁止 CI）も必須。
2. **Phase 4.5（Reader Residency Diagnostics）★新規**: 現状最も危険な `ReaderCount>0 → Epoch進まず → Shutdownタイムアウト` の診断力を強化。`detectStuckReaders()` は存在するが HealthEvent への詳細出力が不足。実運用で最大の事故原因に対処する。
3. **Phase 8（検証基盤）**: 7条件自動検証スクリプトの整備。改修の進行管理と回帰防止に必須。

#### Aランク

1. **Phase 4（Diagnose拡張）**: 現状の Detect のみから Diagnose/Report まで拡張。Recover系は含めない。
2. **Phase 3（VerifyDrained）**: `collectDrainAudit()` + `isFullyDrained()` を唯一の drain 完了判定とする。位置は `EpochSettled → ReclaimComplete → VerifyDrained → ShutdownComplete`。StopReader は追加しない。

#### Bランク

1. **Phase 1（Overflow改善、限定版）**: Intent → Fallback（有界 4096） → QueueFull Escalation の3段階。
   Fallback 無制限化は Backpressure 消失を招くため禁止。EpochDomain への責務追加は行わない。
   IntentOverflowRegistry は新設しない。Quarantine拡張も行わない。

#### Cランク

1. **Phase 5（Coordinator Authority文書化）**: 責務境界の文書化。`retireDSP` 削除は Phase 2 で対応済みのため、Phase 5 は主に文書化作業。`RetireDeleter` 新設は任意（必須ではない）。

#### Dランク（最適化）

1. **Phase 6（RT判断最小化、限定版）**: CrossfadeRuntime の atomic read 自体は RT 安全であるため最適化扱い。
   `getGain().isSmoothing()` / `getDryScaleGain().getNextValue()` の RT 読み取りは維持。

#### 実装非推奨

1. **Phase 4（Recover系）**: 強制 `advanceEpoch()` は RCU 原則違反により UAF を引き起こす可能性がある。
2. **Phase 7（Observe明示化）**: RCU の Grace Period モデルと衝突する。
3. **IntentOverflowRegistry（旧T1-2）**: 新設しない。第3キューは Backpressure 消失を招く。QueueFull Escalation で代替。
4. **RetireDeleter新設**: 必須ではない。Router の deleter 注入型 `enqueueRetire(ptr, deleter, epoch)` で十分対応可能。
5. **StopReader強制解除**: Late Callback との競合リスクのため実施しない。

---

## 付録: 現状調査で確認した主要ファイル一覧

### コア ISR インフラ

| ファイル | 役割 | 状態 |
|---------|------|------|
| `src/audioengine/AudioEngine.h` | エンジン本体宣言 | ✅ 確認済み |
| `src/audioengine/ISRShutdown.h/.cpp` | ShutdownRuntime FSM | ✅ 確認済み |
| `src/audioengine/ISRRetire.h/.cpp` | RetireRuntime (intent queue) | ✅ 確認済み |
| `src/audioengine/ISRRetireRouter.h/.cpp` | RetireRouter (epoch routing) | ✅ 確認済み |
| `src/audioengine/ISRRetireRuntimeEx.h/.cpp` | RetireRuntimeEx (lifecycle tracking) | ✅ 確認済み |
| `src/audioengine/ISRRetireLane.h` | RetireLane enum | ✅ 確認済み |
| `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp` | Coordinator | ✅ 確認済み |
| `src/audioengine/RuntimePublicationOrchestrator.h/.cpp` | Orchestrator | ✅ 確認済み |
| `src/audioengine/PublicationAdmission.h/.cpp` | Admission | ✅ 確認済み |
| `src/audioengine/PublicationExecutor.h/.cpp` | Executor | ✅ 確認済み |
| `src/audioengine/RuntimeHealthMonitor.h/.cpp` | HealthMonitor | ✅ 確認済み |
| `src/audioengine/DSPLifetimeManager.h` | DSP寿命管理 | ✅ 確認済み |
| `src/audioengine/DSPTransition.h` | DSP遷移 | ✅ 確認済み |
| `src/audioengine/ISRDSPQuarantine.h/.cpp` | DSP隔離 | ✅ 確認済み |
| `src/audioengine/CrossfadeRuntime.h` | クロスフェードruntime | ✅ 確認済み |
| `src/audioengine/RuntimeDrainAudit.h` | Drain監査 | ✅ 確認済み |
| `src/audioengine/ShutdownScope.h` | ShutdownScope | ✅ 確認済み |

### コア基盤

| ファイル | 役割 | 状態 |
|---------|------|------|
| `src/core/EpochDomain.h` | EpochDomain (RCU) | ✅ 確認済み |
| `src/core/RCUReader.h` | RCUReader | ✅ 確認済み |
| `src/core/RuntimePublicationCoordinator.h` | RuntimePublicationCoordinator | ✅ 確認済み |
| `src/core/DeferredRetireFallbackQueue.h` | Fallback queue | ✅ 確認済み |
| `src/DeferredDeletionQueue.h` | Lock-free MPMC deletion queue | ✅ 確認済み |

### RTパス

| ファイル | 役割 | delete確認 | lock確認 |
|---------|------|-----------|---------|
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | getNextAudioBlock | ✅ なし | ✅ なし |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | processBlockDouble | ✅ なし | ✅ なし |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | DSPCore double処理 | ✅ なし | - |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | DSPCore float処理 | ✅ なし | - |

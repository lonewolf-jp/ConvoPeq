# Practical Stable ISR Bridge Runtime — 改修計画書

**バージョン**: 5.0
**日付**: 2026-06-12
**ベース文書**: `doc/work33/notfinished9.md`, `doc/work33/notfinished9_validation_report.md`
**検証ツール**: grep, Serena MCP, CodeGraph MCP, graphify MCP (15,779 nodes), AiDex MCP (275 files, 3,952 methods), semble CLI

---

## 目次

1. [改修項目一覧](#1-改修項目一覧)
2. [Priority A: 最優先](#2-priority-a-最優先)
3. [Priority B: 次優先](#3-priority-b-次優先)
4. [Priority C: 計画的対応](#4-priority-c-計画的対応)
5. [検出済みイベントと未ハンドラ対応表](#5-検出済みイベントと未ハンドラ対応表)
6. [改修対象ファイル一覧](#6-改修対象ファイル一覧)
7. [依存関係と実施順序](#7-依存関係と実施順序)
8. [Appendix: 調査過程で発見した追加乖離](#8-appendix-調査過程で発見した追加乖離)

---

## 1. 改修項目一覧

```
ID     | Priority | 領域                       | 概要
───────┼──────────┼────────────────────────────┼──────────────────────────────────────────
A-3    | A        | detectStuckReaders→Shutdown | 漏れ10: Reader検出とFSM断絶の解消
A-2    | A        | DrainAudit Reader統合       | activeReaderCount/stuckReaderCount 追加
A-5    | A        | Double-Retire Telemetry     | WorldLifecycleAudit二重retireのRelease証跡化
───────┼──────────┼────────────────────────────┼──────────────────────────────────────────
A-4    | B        | Crossfade経路統一（設計継続）  | publishIdleWorldOnly 方式（実装前に再設計必須）
B-1    | B        | World Consistency           | Evidence/Telemetryのみ（HealthState変更なし）
B-2    | B        | HealthState統合              | BlockingReason補助情報として（canShutdown条件にはしない）
───────┼──────────┼────────────────────────────┼──────────────────────────────────────────
8.6    | B        | ReaderStuck Evidence定期出力  | diagnoseRetireStall の定期Evidence再通知
───────┼──────────┼────────────────────────────┼──────────────────────────────────────────
B-3    | C        | Warmup Validation           | validateRuntimeIntegrity()（効果限定）
A-1    | -        | Reader状態機械              | 将来検討 — 実施せず
C-2    | C        | EmergencyDrain Phase        | ShutdownPhase 追加（Reader解放禁止）
C-4    | C        | HealthState Reset           | prepareToPlay 開始時に m_healthState_ のみ初期化
```

---

## 2. Priority A: 最優先 — Runtime 健全性に直結

### A-3: `detectStuckReaders()` → `ShutdownBlockingReason` 結合（漏れ10解消）

**最重要項目**: 実コード上で `EVENT_READER_STUCK` は発行されているが `onHealthEvent()` にハンドラがなく、
`ShutdownBlockingReason::ReaderActive` も定義のみで未使用。最も明確な実装ギャップ。

**問題**: `detectStuckReaders()` と `ShutdownBlockingReason::ReaderActive` が完全に断絶。Reader 異常を検出しても Shutdown FSM が「Reader により停止中」と認識できない。

**現状**:

- `EVENT_READER_STUCK` は `RuntimeHealthMonitor::diagnoseRetireStall()` で発行される
- しかし `AudioEngine::onHealthEvent()` にハンドラがなく「検出→HealthEvent発行→誰も処理しない」状態
- `ShutdownBlockingReason::ReaderActive` は誰もセットしない

**改修内容**:

1. **`onHealthEvent()` での処理（診断用）**:
   - `EVENT_READER_STUCK` ハンドラを追加し、Evidence 出力のみ行う
   - **ShutdownBlockingReason への変換は行わない** — HealthEvent は診断イベントであり責務が異なる

2. **`collectDrainAudit()` での処理（Shutdown Authority 用）**:
   `collectDrainAudit()` 内で `detectStuckReaders()` を直接呼び出し、Stuck Reader 数を収集:

   ```cpp
   .activeReaderCount = m_retireRouter->activeReaderCount(),
   .stuckReaderCount = /* detectStuckReaders から stuck 数を算出 */,
   .maxReaderResidencyUs = /* 最大滞留時間 */,
   ```

3. **`getPrimaryBlockingReason()` での判定**:
   `RuntimeDrainAudit::getPrimaryBlockingReason()` の優先順位に `ReaderActive` を追加:

   ```cpp
   if (stuckReaderCount > 0) return BlockingReason::ReaderActive;
   ```

4. **`VerifyDrained` フェーズでの動作**:
   Reader が Stuck している場合、`markTimedOut(ReaderActive)` する

**`activeReaderCount` と `stuckReaderCount` の優先順位**:

- `stuckReaderCount > 0` → `ReaderActive`（Reader 異常滞留による shutdown ブロック）
- `activeReaderCount > 0 && stuckReaderCount == 0` → 正常な Reader 稼働中。shutdown はブロックしない（graceful drain の `activeReaderCount()==0` とは別経路）
- 既存の graceful drain は `activeReaderCount()==0` を監視しているため、`stuckReaderCount` はそれより上位の異常検出として機能する

**責務の分離**:

- `onHealthEvent()` / `HealthEvent` → **診断イベント**（Evidence 出力）
- `collectDrainAudit()` / `ShutdownBlockingReason` → **Shutdown 状態判定**
- 両者は責務が異なる。ShutdownBlockingReason への変換は `collectDrainAudit()` 側で行う

---

### A-5: WorldLifecycleAudit Double-Retire Telemetry

**最重要昇格項目**: Appendix 8.5 から昇格。

**問題**: `WorldLifecycleAudit::onWorldRetired()` の二重 retire 検出が `assert(false)` のみで、
Release ビルドでは何も起こらない。実運用で二重 retire が完全に無視される。

**コード照合結果** (コード確定):

```cpp
void onWorldRetired(uint64_t worldId, uint64_t epoch) noexcept
{
    uint64_t prev = convo::fetchSubAtomic(activeWorldCount_, 1u,
        std::memory_order_acq_rel);
    if (prev == 0) {
        assert(false);  // 二重 retire 検出 — Release ビルドで消失
        convo::publishAtomic(activeWorldCount_, uint64_t{0},
            std::memory_order_release);
    }
    // ... retire record ...
}
```

Release ビルドでは `assert(false)` が除去され、`activeWorldCount_` の補正（0に固定）のみが残る。
二重 retire の発生自体が完全にサイレントになる。

**改修内容**:

1. `WorldLifecycleAudit` に `doubleRetireCount` カウンタ追加（`std::atomic<uint64_t>`）
2. `onWorldRetired()` の二重 retire 検出時（`prev == 0`）に以下を実行:
   - `fetchAddAtomic(doubleRetireCount_, 1)` — telemetry カウンタ
   - `emitWorldConsistencyEvidence()` — evidence JSON 出力（worldId, epoch, correlationId を含む）
3. `assert(false)` は維持（Debug ビルドでの即時検出用）
4. `markFailed()` のような Runtime 停止方向にはしない — WorldLifecycleAudit は診断系であり Authority ではない

**該当ファイル**:

- `src/audioengine/WorldLifecycleAudit.h` (onWorldRetired, doubleRetireCount)
- `src/audioengine/WorldLifecycleAudit.cpp` (evidence 出力)

---

### A-2: DrainAudit に Reader 状態統合

**問題**: `RuntimeDrainAudit` に `activeReaderCount` / `stuckReaderCount` が存在しない。
Shutdown 最終監査 (VerifyDrained) で Reader 状態が考慮されない。

**現状のコード** (`src/audioengine/RuntimeDrainAudit.h`):

```cpp
struct RuntimeDrainAudit {
    // ... existing fields ...
    // activeReaderCount:  ✗ 欠落
    // stuckReaderCount:   ✗ 欠落
};
```

**改修内容**:

1. `RuntimeDrainAudit` に以下を追加:

   ```cpp
   uint64_t activeReaderCount{0};
   uint64_t stuckReaderCount{0};
   uint64_t maxReaderResidencyUs{0};
   ```

2. `collectDrainAudit()` に Reader 状態収集を追加:

   ```cpp
   .activeReaderCount = m_retireRouter->activeReaderCount(),
   .stuckReaderCount = /* detectStuckReaders から算出 */,
   .maxReaderResidencyUs = /* 最大滞留時間 */,
   ```

3. `isAllZero()` に Reader 条件を追加しない（故意的 — A-3 でブロッキング理由として扱う）

**該当ファイル**:

- `src/audioengine/RuntimeDrainAudit.h` (struct)
- `src/audioengine/AudioEngine.Threading.cpp` (collectDrainAudit)

---

## 3. Priority B: 次優先 — 監査・Authority 強化

### A-4: Crossfade 経路統一 — 設計継続（Priority A→B 降格）

**重要性**: 正しい方向性だが、実装時期尚早。

**問題**: 3経路（Timer / DSPTransition / Timeout Recovery）が分岐しており、
Timeout Recovery では Idle World の publish が行われない。

#### コード照合: `notifyTransitionComplete()` の全責務

`RuntimePublicationOrchestrator::notifyTransitionComplete()`（`Orchestrator.cpp:243`）は
**デッドコードではあるが**、設計意図として以下の4責務を内包する:

1. **Transition Completion**: `transition_.onTransitionComplete(currentAfterFade)`
2. **Shutdown Guard**: `engine_.isShutdownInProgress()` 時は deferred をキャンセル
3. **Stale Discard**（二重検査）:
   - **Generation Guard**: `deferred.guard.generation != currentGen` → StaleDiscard
   - **Publication Sequence Guard**: `deferred.guard.sequence < currentPubSeq` → StaleDiscard
4. **Deferred Publish Submit**: 有効な deferred を `submitPublishRequest(req)`

#### `publishIdleWorldOnly()` が継承しない責務

計画書の `publishIdleWorldOnly()` は **上記1〜4の責務を一切引き継がない**。
Idle World の publish のみを責務とする純粋関数であり、
`notifyTransitionComplete()` の「Transition Completion + Deferred Publish Authority」
という複合責務を補完しない。

#### リスク: 将来の Timer 経路変更で再乖離

Timer `fadeCompleted` が将来 `notifyTransitionComplete()` 経由に変更された場合、
Timeout Recovery 側だけ `publishIdleWorldOnly()` となり、
再び経路差分が発生する。つまり:

- 現状では正しい（Timer は inline、DSPTransition も inline、Timeout Recovery のみ publish 欠落）
- しかしアーキテクチャ的には完成ではない

#### 結論

A-4 は「実装推奨」ではなく **「設計継続」** として位置づける。
以下の条件が揃うまで実装を開始しない:

1. A-3 + A-2 完了（Reader→Shutdown 結合の確立）
2. `notifyTransitionComplete()` の責務を継承する統一経路の再設計完了
3. 3経路の完全差分表 + 責務境界の明確化

**Priority**: A → B へ降格。A-3/A-2/A-5 完了後に着手すべき。

**該当ファイル**:

- `src/audioengine/AudioEngine.Timer.cpp` (Timeout Recovery, fadeCompleted)
- `src/audioengine/RuntimePublicationOrchestrator.cpp` (notifyTransitionComplete 責務定義)
- `src/audioengine/DSPTransition.h` (onTransitionComplete)

---

### B-1: World Consistency Verification

**問題**: `collectDrainAudit()` は `activeWorldCount`/`publishedCount`/`retiredCount` を収集するが、
`isAllZero()` や `VerifyDrained` で判定に使用していない。

```cpp
// isAllZero() — World カウンタを完全に無視
bool isAllZero() const noexcept {
    return pendingPublication == 0
        && pendingRetire == 0
        && activeCrossfadeCount == 0
        && deferredPublish == 0;
}
```

**改修内容**:

1. `RuntimeDrainAudit` に `verifyWorldConsistency()` メソッド追加: `Diagnostic` 用途。`bool` ではなく段階的診断が可能な enum を使用:

   ```cpp
   enum class ConsistencyState : uint8_t { Consistent, Suspicious, Broken };
   [[nodiscard]] ConsistencyState verifyWorldConsistency() const noexcept {
       if (publishedCount >= retiredCount
           && (publishedCount - retiredCount) == activeWorldCount)
           return ConsistencyState::Consistent;
       if (retiredCount <= publishedCount)
           return ConsistencyState::Suspicious;  // publish/retire/active が非atomic読み取りの不整合
       return ConsistencyState::Broken;  // retired > published → 二重retireの可能性（最も信頼性の高い指標）
   }
   ```

2. **重要: Shutdown 完了判定には使用しない**。`WorldLifecycleAudit` は診断系であり、
   `published=100, retired=99, active=0` が診断バグなのか Runtime バグなのか区別できないため。
3. `VerifyDrained` フェーズでの動作:
   - `verifyWorldConsistency()` が失敗 → Evidence 記録（JSON ダンプ）
   - **HealthState は変更しない**（診断系の不整合で Runtime HealthState を変えるのは強すぎる）
   - Telemetry 記録（`doubleRetireCount` 等のカウンタ更新）
   - **Shutdown をブロックしない**

**該当ファイル**:

- `src/audioengine/RuntimeDrainAudit.h`
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` (VerifyDrained)

---

### B-2: HealthState 統合（診断情報としてのみ保持）

**問題**: HealthMonitor の `ISRHealthState` が `RuntimeDrainAudit` に統合されていない。

**改修内容**:

1. `RuntimeDrainAudit` に `ISRHealthState healthState` フィールド追加
2. `collectDrainAudit()` で `m_healthMonitor.getHealthState()` を設定
3. **重要: `healthState != Critical` を Shutdown 可否判定には使用しない**。
   例えば Crossfade timeout recovery 成功後、activeCrossfade=0, pendingRetire=0 でも
   HealthState だけ Critical が残るケースがあり得る。
4. HealthState は `diagnosticHint` として保持し、**`getPrimaryBlockingReason()` には組み込まない**。
   HealthState Critical の原因は Reader stuck / Publication stall / Crossfade timeout / Retire stall
   など複数あり、`BlockingReason::Unknown` に変換すると情報が失われる。

**該当ファイル**:

- `src/audioengine/RuntimeDrainAudit.h`
- `src/audioengine/AudioEngine.Threading.cpp` (collectDrainAudit)
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` (VerifyDrained)

---

### B-3: Warmup Validation 強化 — Priority C

**位置づけ**: Priority A から **C へ降格**。現提案では実質 `prepareCompleted` 確認しか増えず、
Runtime 健全性向上効果は非常に小さい。

**問題**: `validateWarmup()` が `isIRLoaded() && !isIRFinalized()` の2条件のみ。
DSP Runtime の健全性をほぼ確認していない。

**現状のコード** (`src/audioengine/RuntimeBuilder.cpp`):

```cpp
BuildError RuntimeBuilder::validateWarmup(const AudioEngine::DSPCore& runtime) const noexcept
{
    if (runtime.convolverRt().isIRLoaded() && !runtime.convolverRt().isIRFinalized())
        return BuildError::WarmupFailed;
    return BuildError::None;
}
```

**改修内容**:

1. `validateRuntimeIntegrity()` を新規追加（**`rebuildThreadLoop()` 側で呼び出す**）:

   ```cpp
   BuildError RuntimeBuilder::validateRuntimeIntegrity(
       const AudioEngine::DSPCore& runtime,
       bool prepareCompleted) const noexcept
   {
       // ★ 検査範囲は構造整合性に限定
       //   sampleRate/eqState などの DSPCore 公開メンバの有無は実コードで確認すること
       //   runtime は参照型のため nullptr チェック不要

       // 1. Prepare 完了済み（DSPCore::prepare() が正常終了している）
       //    確認には BuildResult.prepared を使用（DSPCore メンバには prepared は存在しない）
       if (!prepareCompleted)
           return BuildError::PrepareFailure;

       // 2. Convolver Finalized（IR 読み込み完了かつ確定）
       //    既存の validateWarmup() と同等
       if (runtime.convolverRt().isIRLoaded() && !runtime.convolverRt().isIRFinalized())
           return BuildError::WarmupFailed;

       return BuildError::None;
   }
   ```

   **注意**: 現行 `validateWarmup()` との実質的差分は `prepareCompleted` の確認のみであり、
   検証強度は限定的である。DSPCore の `sampleRate`（`AudioEngine.h:696`）や
   `maxSamplesPerBlock`（`AudioEngine.h:703`）は public メンバとして存在するが、
   placeholder DSP や minimal runtime を許容する将来設計を考慮し、現状では検査対象としない。
   Practical Stable 観点では、今後 `RuntimeSemanticSchemaValidator`
   のような構造的整合性検証レイヤーを追加することを検討すべき。

   ```cpp
   // 現状の rebuildThreadLoop フロー:
   //   1. build() → BuildResult.runtime, BuildResult.prepared
   //   2. rebuildAllIRsSynchronous()
   //   3. validateWarmup()          ← ここで validateRuntimeIntegrity() も呼ぶ
   //   4. publish via coordinator
   ```

   ```cpp
   // ★ 追加: 構造整合性検査
   const auto integrityError = runtimeBuilder.validateRuntimeIntegrity(
       *newDSP, buildResult.prepared);
   if (integrityError != convo::BuildError::None) {
       diagLog("[DIAG] rebuildThreadLoop: integrity check failed: "
           + juce::String(convo::toString(integrityError)));
       continue;
   }
   ```

2. `BuildError` enum に必要に応じて項目追加

**該当ファイル**:

- `src/audioengine/RuntimeBuilder.h` (BuildError enum, validateRuntimeIntegrity 宣言)
- `src/audioengine/RuntimeBuilder.cpp` (validateRuntimeIntegrity 実装)
- `src/audioengine/AudioEngine.RebuildDispatch.cpp` (rebuildThreadLoop — 呼び出し追加)

---

## 4. Priority C: 計画的対応

### A-1: Reader 状態機械（隔離なし）— 将来検討

**結論**: **本改修は実施しないこと。**

**問題**: `EpochDomain::ReaderSlot` に状態機械がなく、Stuck Reader の進行状況を追跡できない。

**実施しない理由**:

- 既に `detectStuckReaders()` が readerIndex, readerEpoch, residencyTimeUs を返している
- Practical Stable Runtime の目的（安全な停止）には A-2 + A-3 で十分対応可能
- `Inactive → Active → Suspect → ZombieCandidate` の状態機械自体は shutdown 完了に寄与しない
- コードベースの複雑性を増やすだけで実運用上のメリットが小さい

**代わりに**: A-3 + A-5 の telemetry/evidence 強化で Reader 状態の診断は十分。
将来 Reader 状態の可視化要件が具体化した場合に再検討する。

**制約 ⚠️**: **Reader slot の強制解放（epoch 書き換え / depth クリア）は一切実装しないこと。**

理由:

- `epoch = kInactiveEpoch; depth = 0;` の強制実行は、まだ Reader が実行中の場合に Use After Free を引き起こす
- RCU/Epoch の根本原則違反となる
- Practical Stable どころか Runtime 安全性を破壊する

**改修内容**:

1. `ReaderSlot` に `std::atomic<ReaderSlotState> state` を追加
2. 状態遷移: `Inactive → Active → Suspect → ZombieCandidate`
   - **Active**: 通常稼働中
   - **Suspect**: `detectStuckReaders()` で閾値(10 epoch)超過 → 警告状態
   - **ZombieCandidate**: 30秒以上滞留 → 隔離予約状態
3. ZombieCandidate 到達時の措置:
   - `HealthState Critical` へ遷移促進
   - `ShutdownBlockingReason::ReaderActive` 設定
   - Evidence Dump（Reader 詳細情報 JSON 出力）
   - **slot の解放は行わない** — 運用者判断に委ねる
4. `detectStuckReaders()` の戻り値で Suspect/ZombieCandidate を区別できるように拡張

**安全上の注意**:

- `quarantineReaderSlot()` は実装しない
- `ManualOverride` 状態は **不要** — これは Runtime 状態ではなく運用状態であり、状態機械に含めない
- 強制解放は不可。運用者判断に委ねる（Evidence Dump により判断材料を提供）

**該当ファイル**:

- `src/core/EpochDomain.h` (ReaderSlot, detectStuckReaders)
- `src/core/IEpochProvider.h` (StuckReaderInfo)

### C-1: Overflow Freeze State — ⚠️ 実施非推奨

**結論**: **この改修は実施しないこと。**

**理由**:

1. 現在 `HealthState Critical → Admission Reject / Builder Reject / Crossfade Reject` が既に成立
2. さらに `Frozen` を追加すると状態機械が `HealthState / PressureLevel / Freeze` の3系統になり複雑化
3. Practical Stable 観点では現状の間接制御で十分

**代わりに**: 現状維持。

---

### C-2: EmergencyDrain Phase

**問題**: Shutdown FSM に Emergency Phase がなく、収束しない場合の最終手段がない。

**現状**: `releaseResources()` は既に graceful drain + tryReclaim + drainDeferredRetireQueues + VerifyDrained を持っており、かなり強力な回復処理が実装済み。

**改修内容**:

1. `ShutdownPhase` に `EmergencyDrain` を追加（**Optional / CompileFlag / DiagnosticMode として実装**）:

   ```cpp
   ... → ReclaimComplete → EmergencyDrain → VerifyDrained → ShutdownComplete
   ```

2. EmergencyDrain フェーズの処理（**Reader 強制解放は禁止**）:
   - `clearDeferredForShutdown()` — Deferred publish クリア
   - `m_epochDomain.tryReclaim()` — 安全な tryReclaim（drainAll 禁止）
   - Crossfade timeout recovery の強制実行
3. 実装方式:
   - デフォルトは **無効**（既存の graceful drain で十分）
   - `#ifdef CONVOPEQ_EMERGENCY_DRAIN` またはランタイムフラグで有効化
   - DiagnosticMode として evidence 出力のみ行い、実際の強制処理は抑制可能
4. EmergencyDrain の制約:
   - maxDuration = 500ms
   - 結果を `DrainResult` に記録（evidence 出力）
   - **Reader slot の epoch/depth 強制書き換えは一切禁止**

**該当ファイル**:

- `src/audioengine/ISRShutdown.h` (ShutdownPhase enum)
- `src/audioengine/ISRShutdown.cpp` (advancePhase)
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
- `src/audioengine/AudioEngine.Threading.cpp` (waitForDrain)

---

### C-3: ReaderActive BlockingReason の活用

**問題**: `ShutdownBlockingReason::ReaderActive` が誰もセットしない。

**現状**: enum 定義のみ。

**改修内容**:

1. A-3 の改修により自然に解消される（`detectStuckReaders()` → `ReaderActive` の経路が確立される）
2. 追加対応として `emitShutdownTrace()` の JSON 出力に ReaderActive を反映

**該当ファイル**:

- `src/audioengine/ISRShutdown.cpp` (emitShutdownTrace)

---

### C-4: HealthState Reset 整理

**問題**: `ShutdownRuntime::transitionTo(ShutdownComplete)` の前後で HealthMonitor の状態をクリアしない。
DAW 環境で `prepareToPlay()` → `releaseResources()` → `prepareToPlay()` を繰り返すと
HealthState が Critical のまま再初期化される可能性がある。

**改修内容**:

1. `RuntimeHealthMonitor` に `reset()` メソッド追加（**HealthState のみ初期化**）:

   ```cpp
   void reset() noexcept {
       // ★ HealthState のみ初期化。m_prev*State は維持する（イベント再通知防止）
       convo::publishAtomic(m_healthState_, ISRHealthState::Healthy,
                            std::memory_order_release);
       // m_prevRetireState 等は維持 — 初期化すると次回監視で Warning が再通知される
       m_lastObservedDropCount = 0;
   }
   ```

   **`m_prev*State` を初期化しない理由**:
   `m_prevRetireState` 等を `Normal` に戻すと、次回 `tick()` で同じ状態が
   `Normal → Warning` の遷移として再検出され、不要なイベントが再発行される。
   HealthState だけ `Healthy` に戻せば、診断状態はリセットされつつ
   監視ロジックは「既に Warning 状態」と認識したまま継続するため、
   再通知が発生しない。

2. **`prepareToPlay()` 開始時に `m_healthMonitor.reset()` を呼び出す**。
   **`releaseResources()` 直後には呼ばない** — Shutdown 診断情報を観測する前に消えるのを防ぐため。

**注意点**: 上記の修正により `m_prev*State` が維持されるため、
`prepareToPlay()` 直後の監視サイクルで Warning が再通知されるリスクは解消されている。

**該当ファイル**:

- `src/audioengine/RuntimeHealthMonitor.h` (reset 宣言)
- `src/audioengine/RuntimeHealthMonitor.cpp` (reset 実装)
- `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` (呼び出し)

---

## 5. 検出済みイベントと未ハンドラ対応表

HealthMonitor が発行する HealthEvent のうち、`onHealthEvent()` で具体的な回復処理が実装されているもの:

| イベントコード | 発行元 | ハンドラ有無 | 回復処理 |
|---------------|--------|------------|---------|
| `EVENT_RETIRE_STALL` | checkRetireStall | ✅ | throttle + reclaim |
| `EVENT_RETIRE_STALL_WARNING` | checkRetireStall | ❌ | なし（ログのみ） |
| `EVENT_RETIRE_AGE_WARNING` | checkRetireReclaimLatency | ❌ | なし（ログのみ） |
| `EVENT_RETIRE_AGE_CRITICAL` | checkRetireReclaimLatency | ✅ | throttle + reclaim |
| `EVENT_PUBLICATION_STALL` | checkPublicationStall | ✅ | clear deferred |
| `EVENT_PUBLICATION_WARNING` | checkPublicationStall | ❌ | なし（ログのみ） |
| `EVENT_READER_STUCK` | diagnoseRetireStall | ❌ | **なし** |
| `EVENT_READER_SLOT_USAGE` | checkReaderSlotUsage | ✅ | admission stop |
| `EVENT_CROSSFADE_TIMEOUT` | checkCrossfadeTimeout | ✅ | recovery（ただし不完全） |
| `EVENT_CROSSFADE_EVENT_DROP` | checkCrossfadeEventDrop | ❌ | **なし** |

**未ハンドライベントのうち重要なもの**:

- **`EVENT_READER_STUCK`**: A-1/A-3 で対応予定
- **`EVENT_CROSSFADE_EVENT_DROP`**: HealthState に間接的に影響するが、直接の回復処理なし

---

## 6. 改修対象ファイル一覧

| ファイル | A | B | C | 改修内容 |
|---------|---|---|---|----------|
| `src/core/EpochDomain.h` | - | - | - | (変更なし — A-1は将来検討のため実施せず) |
| `src/core/IEpochProvider.h` | - | - | - | (変更なし — A-1は将来検討のため実施せず) |
| `src/audioengine/RuntimeDrainAudit.h` | A-2,A-3 | B-1,B-2 | - | フィールド追加, verifyWorldConsistency |
| `src/audioengine/AudioEngine.Threading.cpp` | A-2,A-3 | B-2 | - | collectDrainAudit 拡張 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | A-3 | B-1,B-2 | C-2,C-4 | VerifyDrained 強化, EmergencyDrain |
| `src/audioengine/AudioEngine.Timer.cpp` | A-3 | A-4 | - | EVENT_READER_STUCKハンドラ, Crossfade経路統一(設計継続) |
| `src/audioengine/DSPTransition.h` | - | A-4 | - | (変更なし) |
| `src/audioengine/RuntimeBuilder.h` | - | - | B-3 | validateRuntimeIntegrity 宣言 |
| `src/audioengine/RuntimeBuilder.cpp` | - | - | B-3 | validateRuntimeIntegrity 実装 |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | - | A-4 | - | notifyTransitionComplete 責務定義（既存デッドコード） |
| `src/audioengine/ISRShutdown.h` | - | - | C-2 | EmergencyDrain Phase |
| `src/audioengine/ISRShutdown.cpp` | - | - | C-2,C-3 | EmergencyDrain 遷移, emitShutdownTrace |
| `src/audioengine/RuntimeHealthMonitor.h` | - | 8.6 | C-4 | reset() 宣言 + stuckReaderEvidence定期出力 |
| `src/audioengine/RuntimeHealthMonitor.cpp` | - | 8.6 | C-4 | reset() 実装 + stuckReaderEvidence定期出力 |
| `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | - | - | C-4 | HealthState Reset |
| `src/audioengine/WorldLifecycleAudit.h` | A-5 | - | - | 二重retire telemetry |
| `src/audioengine/WorldLifecycleAudit.cpp` | A-5 | - | - | evidence 出力 |

---

## 7. 依存関係と実施順序

```
A-3 (Reader→Shutdown) ── A-2 (DrainAudit Reader統合)
│
A-5 (Double-Retire Telemetry) ── (独立)
│
8.6 (ReaderStuck Evidence) ── (A-3前提、A-3完了後)
│
B-2 (HealthState統合) ── B-1 (World Consistency) ── C-4 (HealthState Reset)
│
A-4 (Crossfade経路統一) ── (設計継続中、A-3+A-2完了後)
│
B-3 (Warmup Validation) ── (独立、検証強度は限定的)
C-2 (EmergencyDrain) ── (optional/compile-flag)
```

**優先順位（Practical Stable ISR Bridge Runtime 観点）**:

```
Priority A（実装推奨）
  A-3 → Reader検出とShutdown FSMの結合（最も明確な実装ギャップ）
  A-2 → DrainAudit Reader状態統合（A-3 と密接に関連）
  A-5 → WorldLifecycleAudit Double-Retire Telemetry（Release証跡化）

Priority B（設計継続 + 監査強化）
  8.6 → ReaderStuck Evidence定期出力（運用監査証跡）
  B-2 → HealthState 診断情報統合
  B-1 → World Consistency Evidence化
  A-4 → Crossfade経路統一（設計継続 — コード調査+責務分析後に実装判断）
  C-4 → HealthState Reset（prepareToPlay 開始時）

Priority C（計画的な強化）
  B-3 → Warmup Validation（現提案では検証強度が限定的）
  C-2 → EmergencyDrain（optional/compile-flag）
```

**推奨実施順序**:

1. **Phase 1** (A-3): EVENT_READER_STUCK ハンドラ追加 + collectDrainAudit→ReaderActive 結合
   - **実装前必須確認**: `detectStuckReaders()` を `onHealthEvent()` 経由ではなく `collectDrainAudit()` 側で直接収集する設計を維持
2. **Phase 2** (A-2): DrainAudit Reader状態統合
3. **Phase 3** (A-5): WorldLifecycleAudit Double-Retire Telemetry
   - Release ビルドで二重 retire がサイレントになる問題を解消
4. **Phase 4** (8.6): ReaderStuck Evidence定期出力
   - `diagnoseRetireStall()` 内で10秒間隔のEvidence JSON出力を追加
5. **Phase 5** (B-2 + B-1): HealthState統合 + World Consistency Evidence化
6. **Phase 6** (C-4): HealthState Reset（prepareToPlay 開始時）
7. **Phase 7** (A-4): Crossfade経路統一 — **設計継続**
   - **⚠️ 実装開始前に以下を完了すること**:
     - notifyTransitionComplete の4責務（Transition/ShutdownGuard/StaleDiscard/DeferredSubmit）の定義
     - publishIdleWorldOnly が継承しない責務の明確化
     - 3経路完全差分表 + 責務境界文書
     - currentAfterFade として渡す DSP の明確化
8. **Phase 8** (B-3): Warmup Validation 強化（検証強度は限定的）
9. **Phase 9** (C-2): EmergencyDrain（optional/compile-flag）

---

## 8. Appendix: 調査過程で発見した追加乖離

本調査で、notfinished9.md および前版検証報告書で指摘されていない以下の乖離を追加発見した:

### 8.1 `EVENT_READER_STUCK` 未ハンドラ

`RuntimeHealthMonitor::diagnoseRetireStall()` は `EVENT_READER_STUCK` を発行するが、
`AudioEngine::onHealthEvent()` にこのイベントを処理するハンドラが存在しない。
Reader Stuck を検出しても具体的な回復処理が一切実行されない。

**該当**: `src/audioengine/AudioEngine.Timer.cpp` (onHealthEvent)

### 8.2 `EVENT_CROSSFADE_EVENT_DROP` 未ハンドラ

`EVENT_CROSSFADE_EVENT_DROP` は HealthState に間接的に影響するが、
直接の回復処理がない。crossfade event drop が発生しても何もしない。

**該当**: `src/audioengine/AudioEngine.Timer.cpp` (onHealthEvent)

### 8.3 `EVENT_RETIRE_STALL_WARNING` / `EVENT_PUBLICATION_WARNING` / `EVENT_RETIRE_AGE_WARNING` 未ハンドラ

Warning レベルのイベントはすべて未ハンドラ。HealthState 遷移には影響するが、
具体的な予防措置が実行されない。

**該当**: `src/audioengine/AudioEngine.Timer.cpp` (onHealthEvent)

### 8.4 `notifyTransitionComplete` の呼び出し経路が1箇所のみ

`RuntimePublicationOrchestrator::notifyTransitionComplete()` は
`DSPTransition::onTransitionComplete()` をラップしているが、
timer の通常 fade completion 経路（`tryCompleteFade` 成功後）では
`notifyTransitionComplete` を経由せず、直接同等の処理をインラインで行っている。

つまり `notifyTransitionComplete` 経由の処理と Timer のインライン処理が重複しており、
片方だけ修正すると不整合が発生する。

**該当**:

- `src/audioengine/RuntimePublicationOrchestrator.cpp` (notifyTransitionComplete)
- `src/audioengine/AudioEngine.Timer.cpp` (fadeCompleted ブロック)

### 8.5 `WorldLifecycleAudit::onWorldRetired()` の二重 retire 検出が assert のみ

`onWorldRetired()` は二重 retire を `assert(false)` で検出するが、
Release ビルドでは何も起こらない。Practical Stable 観点では
少なくとも telemetry/evidence/diagnostic counter のいずれかが必要。

**推奨**: telemetry 記録 + evidence JSON 出力を追加する。
ただし `markFailed()` のような Runtime 停止方向にはしないこと。
WorldLifecycleAudit は診断系であり Authority ではないため。

**該当**: `src/audioengine/WorldLifecycleAudit.h` (onWorldRetired)

**優先度**: **Priority A（確定）** — A-5 として昇格。Release ビルドで二重 retire が完全に無視される問題は実運用で重大。

---

### 8.6 `RuntimeHealthMonitor::diagnoseRetireStall()` が最初の1回しか通知しない

`diagnoseRetireStall()` は以下のコードで状態遷移時のみイベントを発行する:

```cpp
if (m_prevRetireState != newState)
{
    m_prevRetireState = newState;
    m_callback(ev);
}
```

つまり同じ Stuck Reader が30分存在しても、最初の1回しか `EVENT_READER_STUCK` が通知されない。
State が Warning→Error に変化しない限り再通知されない。

**問題**: Shutdown Authority は A-3 で `collectDrainAudit()` が毎回 Reader 状態を観測するため
Runtime 安全性には影響しないが、**運用監査の観点では Reader 異常の Evidence が不足**する。

**推奨**: 以下のいずれかを追加する:

- **定期再発行**: 一定間隔（例: 10秒ごと）に同じ Stuck Reader 情報を再通知
- **Severity Escalation**: 滞留時間に応じて Warning（10秒）→ Error（60秒）へ自動エスカレーション
- **Evidence 定期出力**: `diagnoseRetireStall()` 内で定期的に Evidence JSON 出力（`emitEvidenceTickNonRt` 相当）

**設計方針**:

```cpp
// ★ 定期Evidence出力（10秒間隔）
const uint64_t nowUs = convo::getCurrentTimeUs();
if (stuckInfo.isStuck && m_callback) {
    // 状態遷移時: イベント発行（既存）
    if (m_prevRetireState != newState) {
        m_prevRetireState = newState;
        m_callback(ev);
    }
    // 定期Evidence出力: 状態遷移がなくとも10秒ごとに出力
    if (stuckInfo.isStuck && (nowUs - m_lastStuckEvidenceUs > kStuckEvidenceIntervalUs)) {
        m_lastStuckEvidenceUs = nowUs;
        // Evidence JSON 出力（ReaderIndex, residencyTimeUs, pendingRetireCount）
        emitStuckReaderEvidence(...);
    }
}
```

**優先度**: **Priority B**（運用監査・Evidence品質）。A-3/A-2/A-5 完了後、A-4 より先に着手すべき。
A-3 が `collectDrainAudit()` で毎回 Reader 状態を観測するため Runtime 安全性には影響しないが、
運用監査の観点から Reader 異常が継続している証跡を残すことが Practical Stable には重要。

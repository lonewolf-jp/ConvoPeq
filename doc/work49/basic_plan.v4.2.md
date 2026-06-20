# Practical Stable ISR Bridge Runtime — 設計書 v4.2（最終確定版）

**Document Version:** 4.2
**Date:** 2026-06-19
**Based on:** v4.1 + 深堀調査6項目の確定結果
**Status:** 最終（全未確定事項の調査完了）

---

## 検証プロセス総括

| サイクル | 成果物 | ツール | 確定項目数 |
|---|---|---|---|
| 1st (v2.0→v3.0) | validation_report.md | 6種 | 12の実装済み項目を確認 |
| 2nd (v3.0→deep) | design_deep_investigation_report.md | 6種 | 7つの未確定事項を確定 |
| 3rd (v4.0→v4.1) | basic_plan.v4.1.md | 6種 | 4設計改善点を反映 |
| **4th (v4.1→v4.2)** | **本ドキュメント** | **6種** | **6つの追加深堀項目を確定** |

### 使用ツール（最終確認）

| ツール | 確認内容 |
|---|---|
| **Serena MCP** | `get_symbols_overview` で Orchestrator/Coordinator/Shutdown の完全なメソッド一覧を取得 |
| **AiDex MCP** | `aidex_query` で currentWorld_/PublicationExecutor/CrossfadeAuthorityRuntime/ShutdownRuntime の全出現箇所を特定 |
| **CodeGraph MCP** | `get_file_structure` で ISRShutdown.h/PublicationExecutor/ISRDSPHandle の完全な構造を把握 |
| **graphify MCP** | アーキテクチャ中心ノード確認 |
| **semble** | semantic search で reconcile/AuthorityState のパターンを確認 |
| **Select-String** | Orchestrator メソッド一覧、currentWorld_ 使用箇所の最終確認 |

---

## 第0章: 深堀結果サマリー

### 発見1: commit() の currentWorld_ 読取は存在しない（誤認識の修正）

v4.0 では「currentWorld_ は commit() 内で読み取り使用される」と記述していたが、
実際のコードでは **commit() は currentWorld_ を読み取らず、書き込みのみ行う**。

| 行 | 実際のコード | v4.0 の記述 | 正誤 |
|---|---|---|---|
| .cpp:109 | `const auto previousSequenceId = convo::consumeAtomic(publicationSequenceId_, ...)` | 「古い currentWorld_ を読み取り」 | **誤り**（publicationSequenceId_ の読み取り） |
| .cpp:123 | `convo::publishAtomic(currentWorld_, newWorld, ...)` | publishAtomic で newWorld を格納 | **正しい** |
| .cpp:126 | `auto observedCurrent = convo::consumeAtomic(currentWorld_, ...)` in `retire()` | compareExchangeAtomic で nullptr へのCAS | **正しい** |

**修正**: currentWorld_ の使用は以下の4箇所（6箇所から修正）：

| # | ファイル | 行 | 用途 |
|---|---|---|---|
| 1 | ISRRuntimePublicationCoordinator.h | 90 | 宣言 |
| 2 | ISRRuntimePublicationCoordinator.cpp | 10 | コンストラクタ初期化 |
| 3 | ISRRuntimePublicationCoordinator.cpp | 123 | **commit()**: publishAtomic で write（read はしていない） |
| 4 | ISRRuntimePublicationCoordinator.cpp | 126 | **retire()**: consumeAtomic + compareExchangeAtomic で読取+CAS |
| 5 | ISRRuntimePublicationCoordinator.cpp | 165 | **getCurrent()**: consumeAtomic で読み取り |

### 発見2: CrossfadeAuthorityRuntime は CrossfadeAuthority とは別責務

| コンポーネント | 責務 | 依存 |
|---|---|---|
| **CrossfadeAuthority** (`CrossfadeAuthority.h`) | **意思決定**: evaluate(old, new, policy) → Decision | Pure Function |
| **CrossfadeAuthorityRuntime** (`ISRDSPHandle.h`) | **実行管理**: registerCrossfade(from, to) → CrossfadeId | DSPHandle のライフサイクル管理 |
| **CrossfadeRuntime** (`CrossfadeRuntime.h`) | **状態実行**: start/fade/complete + timeout | SPSC queue, LinearRamp |

v4.1 の設計は CrossfadeAuthority の Pure Function 化のみ言及していたが、
CrossfadeAuthorityRuntime の責務も明確に分離されていることを確認。

### 発見3: Orchestrator パイプラインは7段階

```
RuntimePublicationOrchestrator::trySubmit():
  1. Admission::evaluate()      → 7種のDecision
  2. DSPHandle解決              → DSPCore* 取得
  3. RuntimeBuilder.build()     → world生成
  4. CrossfadeAuthority.evaluate()  → Pure Function
  5. HealthState Critical 抑制  → Orchestratorレベル
  6. PublicationExecutor.publish() → Coordinator経由
  7. DSPTransition.onPublishCompleted() → activate + crossfade/retire
  8. advanceRetireEpoch()       → epoch進捗
```

### 発見4: Shutdown FSM は10フェーズ

```
Running → AudioStopped → ObserverDrained → RetireClosed →
EpochSettled → ReclaimComplete → [EmergencyDrain] →
VerifyDrained → [TimedOut | Failed | ShutdownComplete]
```

各フェーズで `ShutdownBlockingReason` と `BlockingReasonStats` により
完了阻害要因を追跡。ISR-AUTH-002 の Recovery 後状態同値性と
Shutdown 後の状態同値性は整合していることを確認。

### 発見5: PublicationAdmission は7種のDecision

| Decision | 条件 |
|---|---|
| Accepted | 通常受理 |
| DeferredFadingActive | クロスフェード中は保留 |
| RejectedStaleGeneration | 世代が古い |
| RejectedNotFinalized | world 未完了 |
| RejectedPressure | バックプレッシャー |
| RejectedShutdown | シャットダウン中 |
| RejectedLowPriority | 低優先度要求を圧迫時拒否 |

### 発見6: TelemetryRecorder は完全なステージ追跡

```
TelemetryRecorder::recordProgress(correlationId, generation, 0,
    PublishStage::Submitted → Built → Validated → Published)
```

各ステージのタイムスタンプを記録。FailureStage と FailureReason も
独立して記録される。

---

## 第1章: アーキテクチャ（v4.1 から継承）

### 1.1 3層の Coordinator 構造（v4.2 で明確化）

```
┌─────────────────────────────────────────────────────────────────┐
│ AudioEngine                                                      │
│                                                                   │
│  Layer 1: RuntimeStore (RuntimeStore template) ──── SSOT          │
│      consumePublishedWorld() / WriteAccess::publishAndSwap()      │
│                                                                   │
│  Layer 2: RuntimePublicationCoordinator (template)                │
│      publishWorld() → validate → seal → publishAndSwap → retire  │
│      RuntimePublicationBridge を介して AudioEngine と疎結合       │
│                                                                   │
│  Layer 3: ISRRuntimePublicationCoordinator (metadata tracker)     │
│      backlog counts, state machine, pressure detection            │
│      currentWorld_ (REDUNDANT → A-4 で削除)                       │
│      publicationSequenceId_ / epoch_ / generation_                │
│          → A-1 で PersistentStateBlock に統合                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 RuntimePublicationOrchestrator パイプライン（確定）

```
AudioEngine.Commit.cpp
  → submitPublishRequest(PublishRequest)
    → trySubmit():
      1. Admission::evaluate()
         ├─ Accepted → 続行
         ├─ DeferredFadingActive → enqueueDeferred()
         ├─ Rejected* → 即時 return
         └─ RejectedShutdown → return

      2. DSPHandle → DSPCore* 解決
      3. RuntimeBuilder.buildRuntimePublishWorld()
      4. CrossfadeAuthority.evaluate(oldWorld, newWorld, policy)
      5. HealthState Critical 抑制（Orchestratorレベル）
      6. PublicationExecutor.publish()
         → coordinator.publishWorld()
           → Bridge.validatePublicationNonRt()
           → RuntimeStore.publishAndSwap()
           → Bridge.didPublishRuntimeNonRt()
           → Bridge.willRetireRuntimeNonRt()
           → Bridge.retireRuntimePublishWorldNonRt()
      7. DSPTransition.onPublishCompleted()
         ├─ HealthState::Critical → Emergency Override
         └─ 通常 → activate → crossfade/retire
      8. advanceRetireEpoch()
```

### 1.3 PublishIdleWorldOnly の位置づけ（確定）

DSPTransition::onTransitionComplete() と onHealthEvent (Timer) からのみ呼ばれる。
どちらも `makeRuntimePublicationCoordinator().publishWorld()` を経由するため、
Coordinator のバイパスではない。

---

## 第2章: 設計コンポーネント（v4.1 から継承・補強）

### 2.1 PersistentStateBlock（論理スナップショット版）

v4.1 からの変更なし。version 付き read-version→read-fields→read-version 方式。

### 2.2 deriveAuthorityState + reconcileAuthorityState

v4.1 からの変更なし。3層構造（derive / deriveExpected / reconcile）。

### 2.3 AuthorityDescriptor（Domain × Reason）

v4.1 からの変更なし。

---

## 第3章: 新規追加事項（v4.2 で確定）

### 3.1 ShutdownRecovery 協調設計

Shutdown FSM と Recovery Architecture の協調は以下の設計とする：

```
Shutdown 開始
  → initiateShutdown() で ShutdownPhase::Running → AudioStopped
  → Recovery は Shutdown 検知時、executeRecoveryAction() の昇格を停止
  → Shutdown 完了後は Recovery 不要（RuntimeStore は null）
```

**ISR-AUTH-002 の拡張解釈**:
Recovery 後の状態は通常 Publish 経路で到達可能な状態と同値でなければならない。
Shutdown 後の `null world` も「publishWorld(nullptr)」により通常経路で
到達可能であるため、ISR-AUTH-002 は Shutdown 状態にも適用可能。

### 3.2 Crossfade 3層責務の確定

```
評価 (Evaluation):     CrossfadeAuthority::evaluate()
  Pure Function、AudioEngine 非依存、dspProjection のみ参照

登録 (Registration):   CrossfadeAuthorityRuntime::registerCrossfade()
  CrossfadeId 発行、レコード管理

実行 (Execution):      CrossfadeRuntime::{start, notifyFadeComplete, complete, reset}
  LinearRamp、SPSC queue、timeout 監視

抑制 (Suppression):    Orchestrator レベル（RuntimePublicationOrchestrator.cpp:108-115）
  HealthState::Critical 時、evaluate 結果を上書き

緊急 (Emergency):      DSPTransition::onPublishCompleted() 内
  HealthState::Critical 時、activate → complete → retire → incrementAbortCount
```

### 3.3 CoordinatorState 完全定義

ISRRuntimePublicationCoordinator の状態遷移：

```
Bootstrapping → Ready → Publishing → Ready
                                              ↘ Transitioning
                                                ↘ Pressure → Ready
                                                  ↘ ShuttingDown
                                                    ↘ Faulted
```

`commit()` 内での単調増加チェック（sequenceId/epoch/generation）が
Faulted 遷移の唯一のトリガー。Faulted はリカバリ不能な状態であり、
RuntimeStore から回復する必要がある。

### 3.4 currentWorld_ 削除戦略（最終版）

**削除後の getCurrent() の代替**:

`ISRRuntimePublicationCoordinator::getCurrent()` はテスト(ISRSemanticValidationTests.cpp)の
17箇所で使用されている。削除後は以下のいずれかに置き換え：

```cpp
// 案A（推奨）: RuntimePublicationCoordinator::consumePublishedWorld を使用
if (RuntimePublicationCoordinator::consumePublishedWorld(store) != &world1)

// 案B: AudioEngine::observePublishedWorld() を使用（既存のラッパー）
if (engine.observePublishedWorld() != &world1)
```

**削除手順**:

```
STEP 1: ISRRuntimePublicationCoordinator に RuntimeStore への参照を注入
  → コンストラクタまたは setter で runtimeStore ポインタを受け取る

STEP 2: getCurrent() の実装を変更
  → return runtimeStore_->observe();

STEP 3: retire() の compareExchangeAtomic(currentWorld_, ...) を削除
  → RuntimeStore が既に world ポインタを管理しているため不要

STEP 4: commit() の publishAtomic(currentWorld_, ...) を削除
  → PersistentStateBlock 導入後、状態更新は update() に移譲

STEP 5: currentWorld_ メンバ変数を削除
  → 宣言とコンストラクタ初期化を削除
```

### 3.5 TelemetryRecorder 統合設計

TelemetryRecorder はすでに Orchestrator 内で publish ライフサイクルの
各ステージを記録している。AuthoritySource/Domain 導入時はこれと統合する：

```cpp
// TelemetryRecorder 拡張案（最小影響）
struct AuthorityTelemetry {
    std::atomic<uint64_t> domainCount[7]{};  // Domain 単位集計
};

// TelemetryRecorder とは別管理（関心の分離）
// TelemetryRecorder = publish 進捗の時系列記録
// AuthorityTelemetry = 発行元の統計集計
```

---

## 第4章: 改訂 Phase 順序（最終確定）

```
Week 1: A-1 PersistentStateBlock（論理スナップショット版）
        A-2 AuthorityDescriptor + Telemetry

Week 2: A-4 currentWorld_ 削除（STEP 1-5）
        B-1 Validator エッジケース（7ケース）

Week 3: A-3 deriveAuthorityState + deriveExpectedState + reconcileAuthorityState
        B-3 Property Test（Publish+Retire+Recover+Shutdown）

Week 4: A-5 Recovery 統合（reconcileAuthorityState 接続）
        B-2 障害注入テスト（4シナリオ）

Week 5: 全体検証 + CI ゲート追加（ISR-AUTH-001/002）
```

---

## 第5章: ISR-AUTH Invariant（確定版）

### ISR-AUTH-001（変更なし）

Authority State は PersistentStateBlock からのみ再構築可能でなければならない。

### ISR-AUTH-002（拡張）

Recovery 後の状態は通常の Publish 経路で到達可能な状態と同値でなければならない。

**拡張解釈**: Shutdown 後の `null world` も通常経路で到達可能。
Emergency Override 後の Immediate World も到達可能。

### ISR-AUTH-003（新規）

```
ISR-AUTH-003

Publish 経路は Orchestrator → Coordinator の唯一経路のみ。
DSPTransition / HealthMonitor / CrossfadeRuntime は直接 publish してはならない。
```

これは現在既に遵守されており、CI で監査可能：

```powershell
# CI ゲート: ISR-AUTH-003 違反検出
# DSPTransition/HealthMonitor/CrossfadeRuntime から publishWorld の直接呼び出しを禁止
Select-String -Path "src\audioengine\DSPTransition.h","src\audioengine\RuntimeHealthMonitor.cpp","src\audioengine\CrossfadeRuntime.h" -Pattern "publishWorld|submitPublishRequest" | ForEach-Object {
    if ($_ -notmatch "engine_\.publishIdleWorldOnly|//|comment") {
        Write-Error "ISR-AUTH-003 violation: direct publish call detected"
    }
}
```

---

## 第6章: 最終状態の達成予測

```
現状:
  92-95%（v2.0 の評価）
  ↓
  95-97%（v3.0 検証後、12の実装済み項目確認）
  ↓
Phase-A 完了後:
  98-99%
  ↓
Phase-B 完了後:
  99-100%
```

### 完了条件

| 条件 | 確認方法 | 判定 |
|---|---|---|
| PersistentStateBlock 導入 | `grep PersistentStateBlock src/core/` | 1件以上 |
| AuthorityDescriptor 導入 | `grep AuthorityDomain src/core/` | 1件以上 |
| currentWorld_ 削除 | `grep currentWorld_ src/audioengine/ISRRuntimePublicationCoordinator.*` | 0件 |
| deriveAuthorityState 実装 | `grep deriveAuthorityState src/core/` | 1件以上 |
| reconcileAuthorityState 実装 | `grep reconcileAuthorityState src/core/` | 1件以上 |
| ISR-AUTH-001 CI ゲート | `.github/scripts/isr-verify-auth-001.ps1` | PASS |
| ISR-AUTH-002 CI ゲート | `.github/scripts/isr-verify-auth-002.ps1` | PASS |
| ISR-AUTH-003 CI ゲート | 上記 Select-String スクリプト | PASS |
| Validator テスト | `Select-String TEST_F\|TEST\( src/tests/PublicationValidatorIsolationTests.cpp` | 45+ 件 |
| 障害注入テスト | 4シナリオ全パス | PASS |
| Property Test | 10,000回ランダムシーケンス | PASS |

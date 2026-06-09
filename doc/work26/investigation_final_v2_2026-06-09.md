# 最終棚卸し確定レポート

**作成日**: 2026-06-09
**調査対象**: 全未確定事項・要調査事項
**使用ツール**: Serena MCP, CodeGraph MCP, grep (全30+回), 直接読取 (20+ファイル)

---

## ① 全 publish 経路マッピング — 確定

### 3つの独立した publish 経路

| 経路 | 入口 | Admission通過 | Validator通過 | 呼び出し元 |
| --- | --- | :---: | :---: | --- |
| **A** | `Orchestrator::trySubmit()` | ✅ Yes | ✅ Yes (coordinator内) | commitNewDSP (唯一) |
| **B** | `AudioEngine::publishWorld()` | ❌ **No** | ✅ Yes (coordinator内) | public API |
| **C** | `makeRuntimePublicationCoordinator()` → `publishWorld()` | ❌ **No** | ✅ Yes (coordinator内) | 7箇所 |

### 経路Cの7箇所の呼び出し元

| ファイル | 行 | 状況 |
| --- | --- | --- |
| `AudioEngine.Init.cpp:46` | bootstrap publish | 起動時のみ |
| `AudioEngine.CtorDtor.cpp:127` | shutdown clear | 終了時のみ |
| `AudioEngine.Processing.PrepareToPlay.cpp:124,236` | prepareToPlay x2 | 再生準備 |
| `AudioEngine.Processing.ReleaseResources.cpp:124,196` | releaseResources x2 | リソース解放 |
| `AudioEngine.Timer.cpp:404` | クロスフェード完了後 | Timer定期実行 |
| `DSPTransition.h:115` | DSP遷移完了後 | onPublishCompleted内 |

### 確定: 3経路とも Coordinator の publishWorld() を経由するため Validator は通過するが、**Admission チェックは経路B/Cではバイパスされる。**

**補正**: "Authority完全崩壊"ではなく **"Authority一元化未完了"** が正確。現状は経路B/CからのAdmissionバイパスが存在するが、Orchestrator 経路Aは正しくAdmissionを通過する。

---

## ② runtimeStore 全アクセスパターン — 確定

### 読み取りアクセス (9箇所)

| メソッド | ファイル | 方法 |
|----------|----------|------|
| `getRuntimeGraph()` | AudioEngine.h:905 | consumeWorldHandle |
| `makeRuntimeReadHandle()` | AudioEngine.h:2334 | consumeWorldHandle |
| `computeRuntimePublishComputation()` | AudioEngine.h:2600 | consumeWorldHandle |
| `logRuntimeTransitionEvent()` | AudioEngine.h:2768 | consumeWorldHandle |
| `Orchestrator::trySubmit()` | RuntimePublicationOrchestrator.cpp:67 | **store.observe() 直接** |
| `onRuntimeRetiredNonRt()` | AudioEngine.Commit.cpp:519 | consumeWorldHandle |

### 書き込みアクセス (publishAndSwap)

| 経路 | 方法 |
|------|------|
| Coordinator::publishWorld() | WriteAccess::publishAndSwap() |
| Orchestrator → Executor → Coordinator | 同上 (間接) |
| AudioEngine::publishWorld() → Coordinator | 同上 (直接) |

### Store 保護機構

```cpp
// RuntimeStore.h — Owner = RuntimePublicationCoordinator<...>
friend Owner;  // WriteAccess::acquireWriteAccess() は friend Owner 経由のみ

// しかし Store 自体は AudioEngine の public メンバ:
RuntimePublishStore runtimeStore;  // AudioEngine.h:2716 — public セクション
```

**確定: Store の WriteAccess は Coordinator のみが取得可能。問題の本質は Store 保護不備ではなく、Coordinator 生成権限の漏洩（誰でも makeRuntimePublicationCoordinator() で Coordinator を生成できること）にある。**

---

## ③ Validator 呼び出し経路 — 確定 (重要)

### Validator の呼び出し連鎖

```
coordinator.publishWorld(worldOwner)         // RuntimePublicationCoordinator.h:89
  → worldOwner->sealRecursively()             // 不変性確保
  → bridge_.validatePublicationNonRt(world)   // RuntimePublicationCoordinator.h:99
    → RuntimePublicationBridge::validatePublicationNonRt()  // AudioEngine.h:2657
      → validator_->validatePublication(world)  // AudioEngine.h:2660
        → validateSemanticConsistency()         // 実装あり (crossfade params check)
        → validateTopology()                    // return true (ダミー)
        → validateResources()                   // return true (ダミー)
        → checkNoConflictingTransitions()       // return true (ダミー)
  → writeAccess_.publishAndSwap(newWorld)       // Store 書換
```

### 重要な訂正: Validator は未統合ではない

**v4 最終版では「Validator が統合されていない」と評価したが、実際にはすべての publish 経路が Coordinator::publishWorld() を通るため、Validator は必ず呼ばれる。**

問題なのは:

1. Validator の内部が 4段階中 3段階ダミー (63: ダミー率 75% → 正)
2. Validator の結果は bool (pass/fail) のみで構造化エラー情報がない
3. Validator 通過後でも Admission とは独立した判断 (Validator≠Admission)

### Admission の evaluate() との関係

```
Orchestrator::trySubmit() のみ:
  → Admission::evaluate()     ← Shutdown/Stale/Pressure/Defer チェック
  → (build world)
  → Executor::publish()       ← 内部で coordinator.publishWorld() → Validator呼び出し

publishWorld() / makeRuntimePublicationCoordinator():
  → coordinator.publishWorld()  ← Validatorは通るが、Admissionは通らない
```

**確定: Validator の呼び出し自体は全経路で担保されている。問題は Validator の内部実装の未完成 (ダミー率75%) と、Admission チェックとの非連携。v4 の「Validator 未統合」評価は撤回する — 正しくは「Validator 実装未完成」。**

---

## ④ 削除キュー追跡ID — 確定

### DeletionEntry の構造

```cpp
// DeferredDeletionQueue.h
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
};
```

| 追跡ID | 存在 | フィールド |
|--------|:----:|-----------|
| PublicationSequenceId | ✅ | RuntimePublishWorld::publication.sequenceId |
| worldId | ✅ (診断専用) | RuntimePublishWorld::worldId |
| RetireId | ❌ **不在** | — |
| ReclaimId | ❌ **不在** | — |
| PublicationId → RetireId 紐付け | ❌ **不在** | — |

### 診断専用と明記

```cpp
// ISRRuntimeIdentityGenerators.h
// DIAGNOSTIC ONLY: RuntimeWorldIdGenerator produces identifiers for
// trace/correlation/diagnostic purposes. Must NOT be used for:
// - Authority decisions (branch, condition, ordering)
// - Publication ordering
// - Retire ordering
```

**確定: Publication → Retire → Reclaim の因果チェーンを追跡するIDはコードベースに存在しない。**

---

## ⑤ Shutdown Drain Audit — 確定

### Shutdown フェーズ

```
ShutdownPhase::Running
  → StopAcceptingWork      ← 新規work拒否
  → StopAudio              ← Audio停止 (releaseResources)
  → StopWorkers            ← Workerスレッド停止
  → ForceEpochAdvance      ← 強制epoch進捗
  → DrainRetire            ← 全キュー強制Drain
  → Destroy                ← 破棄
```

### drainAll() の実装

```cpp
// EpochDomain.h
void drainAll() noexcept {
    deferredDeletionQueue.drainAllUnsafe();
}
// DeferredDeletionQueue.h — epoch 無視で全回収
```

### 現在の監査機構

```cpp
// AudioEngine.h
[[nodiscard]] bool isFullyDrained() noexcept;
[[nodiscard]] bool waitForDrain(int timeoutMs = 2000, int pollIntervalMs = 2) noexcept;
[[nodiscard]] convo::isr::RuntimeDrainAudit collectDrainAudit() noexcept;
```

`isFullyDrained()` と `markShutdownComplete()` による Drain 判定機構は**既に存在する**:

```cpp
// ISRRuntimePublicationCoordinator.cpp (推定)
void markShutdownComplete() noexcept {
    if (isFullyDrained())
        state_ = CoordinatorState::Bootstrapping;
    else
        state_ = CoordinatorState::Faulted;
}
```

**不足: Shutdown Drain Guarantee ではなく Shutdown Drain Audit が未完成。** Evidence への実データ反映と、shutdown 終了時に必ず監査を実行する仕組みが不足。

### EvidenceExporter の shutdown_trace.json

`ISREvidenceExporter.cpp:177`:

```cpp
{"shutdown_trace.json",
 "{\"artifact\":\"shutdown_trace.json\",\"status\":\"generated\",\"phase\":0,
   \"verified\":true,\"sh1_callbackCount\":0,\"sh2_activeCrossfade\":0,
   \"sh3_pendingRetire\":0,\"sh4_observerCount\":0,
   \"sh5_lateCallbackCount\":0,\"sh6_postStopEnqueueCount\":0}"}
```

**すべての値がハードコードされた 0 / true。実際の Runtime 状態を反映していない。**

**確定: Drain 機構は存在する。Drain 完了の自動検証と Evidence への実データ反映が不足。**

---

## ⑥ ReaderContext 誤用調査 — 確定

### `RuntimeReaderContext{ reader, channel }` の直接構築

**全コードベースで2箇所のみ**:

| 箇所 | コード | 正誤 |
| --- | --- | :---: |
| `RuntimePublicationOrchestrator.cpp:23` | `RuntimeReaderContext{ publicationReader, ObserveChannel::Publication }` | ✅ 正しい |
| `AudioEngine.Processing.ReleaseResources.cpp:92` | `RuntimeReaderContext{ messageThreadRcuReader, ObserveChannel::Message }` | ✅ 正しい |

### ヘルパー関数の使用

| 箇所 | 使用したヘルパー | 正誤 |
|------|-----------------|:----:|
| `NoiseShaperLearner.cpp:1034` | `makeWorkerReaderContext(rcuReader, 0)` | ✅ |
| `SpectrumAnalyzerComponent.cpp:278` | `makeMessageReaderContext(rcuReader)` | ✅ |

### 間接使用 (makeRuntimeReadHandle 経由)

`makeRuntimeReadHandle(const RuntimeReaderContext& ctx)` が9箇所から呼ばれている。各呼び出し元は適切な reader/channel を構築済みの ctx を渡す。

### 確定

**RuntimeReaderContext の誤用はコードベースのどこにも存在しない。** 理論上の型安全性の問題はあるが、実際の障害原因にはなっていない。P3 で妥当。

---

## ⑦ EpochDomain 残留呼び出し — 確定

### AudioEngine の m_epochDomain 直接参照 (7箇所)

| ファイル | 行数 | メソッド | 内容 |
| --- | --- | --- | --- |
| `AudioEngine.h` | 3366 | フィールド宣言 | 本体 |
| `AudioEngine.h` | 3370 | RCUReader初期化 | audioThreadRcuReader |
| `AudioEngine.h` | 3372 | RCUReader初期化 | messageThreadRcuReader |
| `AudioEngine.CtorDtor.cpp` | 21 | コンストラクタ | m_coordinator(m_epochDomain) — **置換可能** |
| `AudioEngine.CtorDtor.cpp` | 26 | コンストラクタ | make_unique<ISRRetireRouter>(m_epochDomain) — **Router構築は不可避** |
| `AudioEngine.CtorDtor.cpp` | 131 | デストラクタ | drainAll() |
| `AudioEngine.Processing.ReleaseResources.cpp` | 208 | releaseResources | drainAll() |

### 改善可能な箇所

`m_coordinator(m_epochDomain)` → `m_coordinator(*m_retireRouter)` に変更可能。IEpochProvider インターフェースを通すことで EpochDomain 依存を除去できる。

`drainAll()` は EpochDomain 固有APIであり、IEpochProvider には存在しない。ただし shutdown 時にのみ呼ばれるため、Router に委譲メソッドを追加することで隠蔽可能。

### EQProcessor の m_epochDomain 直接参照 (10箇所)

**こちらは独立した内部管理用 EpochDomain であり、Router 化の対象外。** ただし ISR 全体の移行計画上は注意が必要。

### 確定

AudioEngine の EpochDomain 直接参照は **7箇所**。うち 2箇所は RCUReader 初期化 (IEpochProvider 経由に変更済みだが初期化時に EpochDomain 参照が必要)、1箇所は Router 構築 (やむを得ない)、1箇所は m_coordinator (置換可能)、2箇所は drainAll (委譲可能)、1箇所はフィールド宣言。

---

## ⑧ 削除キューサイズと溢れリスク — 確定

### DeferredDeletionQueue

```cpp
static constexpr uint32_t kQueueSize = 4096;  // DeferredDeletionQueue.h
```

ロックフリー MPMC。4096 エントリ。Audio Thread からの enqueue はノンブロッキング。溢れ时は `enqueue()` が `false` を返し、呼び出し元が `QueuePressure` または `QueueFull` として処理。

### DeletionQueue (旧実装)

```cpp
static constexpr size_t kCapacity = 128;  // DeletionQueue.h
std::mutex mutex;
```

128エントリ + mutex。**旧実装。現在の Runtime 経路では使用されていない可能性が高い** (EpochDomain は DeferredDeletionQueue を使用)。

### SafeStateSwapper リングバッファ

```cpp
static constexpr size_t kMaxRetired = 64;  // SafeStateSwapper.h (推定)
```

溢れ時は `fallbackMutex` + `fallbackQueue` (priority_queue) にフォールバック。

### 確定

主要な retire キュー (DeferredDeletionQueue, 4096エントリ) は十分な容量を持つ。SafeStateSwapper のリングバッファ (64) は溢れうるが、mutex fallback で保護されている。

---

## 実運用観点での最終優先順位（確定版）

Practical Stable ISR Bridge Runtime の観点から再評価した最終確定優先順位:

### P0（実運用事故に直結）

1. **PublicationExecutor 完成** — Success しか返さない空実装
2. **Admission bypass 除去** — 経路B/CからもAdmissionを通過させる
3. **Coordinator 生成権限封鎖** — makeRuntimePublicationCoordinator() を friend-only に
4. **RuntimeStore の公開範囲縮小** — Store を private 化

### P1（長期運用で破綻）

1. RuntimePublicationValidator 実装完成 — Topology/Resource/Conflict のダミー解消
2. ISRRetireRouter 実体化 — EpochDomain直接転送からの脱却
3. RetirePolicy 実装 — 前方宣言のみの Policy を実体化
4. EpochDomain 直接依存除去 — 7箇所中4箇所は Router/IEpochProvider で置換可能
5. Evidence System 実データ化 — 静的テンプレートからの脱却
6. QueuePressure 回復戦略 — 検知はあるが回復ポリシーが弱い
7. Reclaim Progress Monitoring — 回収停滞の検知

### P2（監査不能）

1. Shutdown Audit 自動化 — isFullyDrained()+markShutdownComplete()既存、Evidence連携不足
2. Publication→Retire→Reclaim 追跡 — RetireId/ReclaimId 不在。実運用上の追跡不能≠設計欠陥
3. ObserveToken 拡張 — generation/pubId/worldId 追加

### P3（将来リスク）

1. Reader Stuck Detection — RAII guardで正常系OK。クラッシュ時のみ
2. RuntimeReaderContext 型安全化 — 誤用実績ゼロ
3. ReaderSlotPool 化 — 11/64 使用。枯渇リスクなし

---

## 管理者レビューによる補正一覧

| レポート記載 | 補正後 | 根拠 |
| --- | --- | --- |
| "Authority完全崩壊" | **"Authority一元化未完了"** | 経路Aは正しくAdmission通過 |
| "Store保護なし" | **"Coordinator生成権限の漏洩"** | Store自体はfriend Ownerで保護 |
| "Validator未統合" (v4) | **撤回済み。正しくは"Validator実装未完成"** | Coordinator→Bridge→Validator の呼び出し連鎖が存在 |
| "Drain完了の自動検証なし" | **"Shutdown Drain Audit未完成"** | isFullyDrained()+markShutdownComplete() 既存 |
| "RetireId/ReclaimId: P0" | **P2に降格** | 追跡不能≠設計欠陥。監査性向上策 |
| "EpochDomain: P0-P1" | **P1に統一** | 依存そのものは事故要因ではない |
| "QueuePressure: 検知なし" | **"検知あり、回復ポリシーが弱い"** | RuntimePublicationCoordinatorにPressure状態遷移実装済み |

---

## ツール使用実績

| ツール | 使用回数 | 主な用途 |
| --- | --- | --- |
| Serena MCP | 10回 | シンボル定義/参照解析 |
| CodeGraph MCP | 3回 | モジュール構造解析、インデックス |
| grep/Select-String | 30回以上 | 経路/Store/Validator/EpochDomain 網羅検索 |
| Graphify MCP | 1回 | God Nodes確認 |
| 直接ファイル読取 | 20ファイル以上 | 全調査対象ファイル |

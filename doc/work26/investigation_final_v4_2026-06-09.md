# 未確定事項 最終確定レポート v4

**作成日**: 2026-06-09
**調査手段**: CodeGraph MCP, Serena MCP, grep, 直接読取

---

## 調査結果一覧

| # | 項目 | 確定内容 | 改修影響 |
| --- | --- | --- | --- |
| 1 | Orchestrator publish 流れ | Admission → Build → Crossfade → Executor → Transition → EpochAdvance の6段階確定 | ✅ P0-1/P0-2 の基礎 |
| 2 | `runtimePublicationBridge_` 型 | `convo::isr::RuntimePublicationCoordinator` (ISR版、テンプレート版とは別) | ✅ P0-4 Store封鎖で考慮必要 |
| 3 | Evidence テンプレート | 8種類 + manifest。**動的データは sealViolationCount のみ** | ✅ P1-5 の範囲確定 |
| 4 | ObserveToken 拡張 | 現在: guard + ptr + ownerThreadId。generation/pubId/worldId なし | ✅ P2-3 の範囲確定 |
| 5 | P0-1 Validator統合経路 | `PublicationExecutor::publish()` 内で validator を呼ぶ必要あり | P0-1 改修手順確定 |
| 6 | runtimePublicationBridge_ 使用箇所 | commit/retire/backlog/shutdown — 20箇所 | P0-4 移行対象 |

---

## 詳細

### 1. Orchestrator publish 6段階パイプライン

`RuntimePublicationOrchestrator::trySubmit()` の実行順序:

```
Phase 1: Admission::evaluate()          ← 許否判定 (Shutdown/Stale/Pressure/Defer)
Phase 2: Build + Publish (activate前)
  2a: resolveDSPHandle()                ← DSPHandle → DSPCore* 解決
  2b: buildRuntimePublishWorld()        ← RuntimeWorld構築
  2c: CrossfadeAuthority::evaluate()    ← クロスフェード要否判定
  2d: PublicationExecutor::publish()    ← validate → coordinator.publishWorld
Phase 3: DSPTransition::onPublishCompleted() ← DSP活性化
Phase 4: advanceRetireEpoch()           ← Epoch進捗 + retire queue drain
```

### 2. `runtimePublicationBridge_` の実体

```cpp
// AudioEngine.h:3455
convo::isr::RuntimePublicationCoordinator runtimePublicationBridge_;
```

これは **ISR版 (`src/audioengine/ISRRuntimePublicationCoordinator.h`)** であり、テンプレート版 (`core/RuntimePublicationCoordinator.h`) とは異なる。両者は共存している。

**ISR版の主なメソッド**:

- `precheckPublish()` — Closure + PayloadTier 検証
- `commit()` — publish確定 (PublishAuthority必須)
- `retire()` — retire確定 (RetireAuthority必須)
- `enqueueRetire()` — 遅延削除キュー投入
- `isFullyDrained()` — 完全Drain確認
- `markShutdownComplete()` — シャットダウン完了
- backlog setter/getter 群

### 3. Evidence テンプレート — 8種類の詳細

| 成果物 | 実データ反映部分 | 静的部分 |
| --- | --- | --- |
| `closure_graph.json` | なし | `nodeCount:0, edgeCount:0, descriptorCoverageComplete:true` |
| `mutation_fault_trace.json` | ⚠️ `sealViolationCount` のみ動的 | 他は固定 |
| `hb_graph_trace.json` | なし | `eventCount:0` |
| `hb_violation_report.json` | なし | `violations:[]` |
| `retire_timeline.json` | なし | `totalTransitions:0, rollbackReady:true` |
| `shutdown_trace.json` | なし | `phase:0, verified:true, sh*:全て0` |
| `retire_latency_report.json` | なし | `withinThreshold:true` |
| `payload_tier_report.json` | なし | `violations:0` + 固定 family 配列 |

**唯一の動的データ**: `sealViolationCountValue()` — シール違反カウントのみ。

### 4. ObserveToken (ObservedRuntime)

```cpp
struct ObservedRuntime {
    RCUReaderGuard guard;          // Reader lifetime管理
    const GlobalSnapshot* ptr;     // Snapshot ポインタ
    // (Debug only) std::thread::id ownerThreadId;

    // 欠落:
    // - generation (publish generation)
    // - publicationSequenceId
    // - worldId
    // - epoch
};
```

### 5. `runtimePublicationBridge_` アクセスパターン (20箇所)

| カテゴリ | 件数 | ファイル |
| --- | --- | --- |
| commit() | 1 | AudioEngine.Commit.cpp |
| retire() | 1 | AudioEngine.Commit.cpp |
| setRetireBacklogCount() | 4 | AudioEngine.h, Commit.cpp, Retire.cpp |
| setPendingIntentCount() | 3 | Commit.cpp, Threading.cpp |
| setReclaimInFlightCount() | 4 | Retire.cpp |
| setFallbackBacklogCount() | 1 | Retire.cpp |
| setDeferredRetireResidencyCount() | 1 | Retire.cpp |
| getPublicationBacklogCount() | 1 | Threading.cpp |
| requestShutdown() | 2 | CtorDtor.cpp, ReleaseResources.cpp |
| markShutdownComplete() | 2 | CtorDtor.cpp, ReleaseResources.cpp |

---

## 改修計画への反映

| 発見事項 | 反映先 |
|---------|--------|
| Orchestrator 6段階パイプライン | P0-1/P0-2 の設計前提として追記 |
| runtimePublicationBridge_ 型の分離 | P0-4 Store封鎖で考慮 |
| Evidence 8種類中7種類が静的テンプレート | P1-5 のスコープ確定 |
| ObserveToken 欠落フィールド特定 | P2-3 の改修範囲確定 |

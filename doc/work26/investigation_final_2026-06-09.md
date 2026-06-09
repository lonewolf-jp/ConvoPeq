# 未確定事項・要調査事項 最終確定レポート

**作成日**: 2026-06-09
**目的**: v4 最終版で未確定だった事項を全ツール使用して確定

---

## A. Publication/Retire 因果追跡 — 確定: 未完成 (P1相当)

### PublicationSequenceId の存在

`PublicationSequenceId` は `ISRRuntimeSemanticSchema.h` で定義:

```cpp
using PublicationSequenceId = std::uint64_t;
```

以下の場所で使用:

| 場所 | 用途 |
| --- | --- |
| `RuntimePublishWorld::publication.sequenceId` | World に付与 |
| `ISRRuntimePublicationCoordinator::commit()` | 引数として受取・記録 |
| `RuntimePublicationOrchestrator::DeferredGuard` | Deferred publish のガード |
| `AudioEngine::publicationSequenceCounter_` | 発番カウンタ |
| `AudioEngine::lastCommittedPublicationSequence_` | 最終確定値 |

### RuntimeWorldIdGenerator の存在

`ISRRuntimeIdentityGenerators.h`: worldId を発番。ただし:

```cpp
// DIAGNOSTIC ONLY: RuntimeWorldIdGenerator produces identifiers for
// trace/correlation/diagnostic purposes. Must NOT be used for:
// - Authority decisions (branch, condition, ordering)
// - Publication ordering
// - Retire ordering
// - Hash keys in semantic structures
```

**診断専用**であり、Authority 決定や Retire 順序付けには使用禁止と明記されている。

### 欠落: RetireId / ReclaimId

**`RetireId`、`reclaimId` はコードベースのどこにも存在しない**（grep 0件）。

`DeletionEntry`（`DeferredDeletionQueue.h`）の構造:

```cpp
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
};
```

因果追跡に必要な情報:

| 情報 | 保持箇所 | 状態 |
| --- | --- | --- |
| PublicationSequenceId | RuntimePublishWorld | ✅ |
| worldId | RuntimePublishWorld | ✅ (診断専用) |
| RetireId | なし | ❌ **未実装** |
| ReclaimId | なし | ❌ **未実装** |
| Publication→Retire 紐付け | なし | ❌ **未実装** |

**確定: PublicationId → RetireId → ReclaimId の因果追跡は存在しない。**

---

## B. World Lifetime 所有権 — 確定: 複数所有者が混在 (P0-P1相当)

### World 生成

`RuntimeBuilder::buildRuntimePublishWorld()` が `aligned_unique_ptr<RuntimePublishWorld>` を返す:

```cpp
auto worldOwner = RuntimePublishWorld::createForBuilder(BuilderToken{});
// ... 構築 ...
worldOwner->freeze();
return worldOwner;
```

### World 所有権の流れ

```
RuntimeBuilder::buildRuntimePublishWorld()
  → aligned_unique_ptr として返却
    → Orchestrator::trySubmit() で受け取り
      → PublicationExecutor::publish() へ move
        → coordinator.publishWorld(move(worldOwner))
          → Store::WriteAccess::publishAndSwap() で Store へ
            → 旧 World が返却される
              → Bridge::retireRuntimePublishWorldNonRt() で retire
```

### 問題: 所有権が複数経路

**RuntimeStore の設計**:

```cpp
// RuntimeStore.h: コメント
// write authority は owner のみが取得可能（rule4: publish authority 集約）。
friend Owner;  // Owner = RuntimePublicationCoordinator
```

所有権は Store が持つが、**Store が public メンバであるため誰でも publishAndSwap できる。**

さらに:

```cpp
// AudioEngine.h: public メソッド
inline void publishWorld(aligned_unique_ptr<RuntimePublishWorld> worldOwner) noexcept {
    auto coordinator = makeRuntimePublicationCoordinator();
    coordinator.publishWorld(std::move(worldOwner));
}
```

**World 所有権の Single Source Of Lifetime Truth が確立していない。**

### 確定

**RuntimeStore 内の atomic ポインタが唯一の「公開中 World」所有権だが、誰でも書き換え可能。** 真の Single Source Of Lifetime Truth は Orchestrator 内に閉じるべき。

---

## C. Admission/Validator 責務境界 — 確定: 未分離 (P0-P1相当)

### 現状の責務配置

| コンポーネント | 責務 | 実装状態 |
| --- | --- | --- |
| `RuntimePublicationValidator` | Semantic/Topology/Resource/Conflict 検証 | Semantic のみ実装、他はダミー |
| `PublicationAdmission` | publish可否判定（Shutdown/Stale/Pressure/Defer） | 実装済み |
| `PublicationExecutor` | publish 実行パイプライン | **未完成**（coordinator丸投げ） |
| `RuntimePublicationCoordinator` (テンプレート) | Store+Bridge 管理 | 実装済み |
| `ISRRuntimePublicationCoordinator` | ISR版 Coordinator（Closure検証） | 実装済み |
| `RuntimePublicationOrchestrator` | 上位オーケストレータ | 実装済み（唯一の正しい入口） |

### パイプラインの現状

```
Orchestrator::trySubmit()
  ├── Admission::evaluate()                    ← shutdown/stale/pressure/defer
  ├── RuntimeBuilder::buildRuntimePublishWorld() ← World構築
  ├── CrossfadeAuthority::evaluate()           ← クロスフェード判定
  ├── PublicationExecutor::publish()           ← 未完成（coordinator丸投げ）
  │     └── coordinator.publishWorld()         ← 内部で sealRecursively → publishAndSwap
  ├── DSPTransition::onPublishCompleted()      ← DSP活性化
  └── advanceRetireEpoch()                     ← Epoch進捗
```

### Validator 未統合

**`RuntimePublicationValidator` は Orchestrator パイプラインのどこからも呼ばれていない。**

```cpp
// RuntimePublicationOrchestrator.cpp
auto coordinator = engine.makeRuntimePublicationCoordinator();
// ↑ この coordinator は内部で RuntimePublicationBridge から Validator を参照可能だが、
//    Executor::publish() は coordinator.publishWorld() を呼ぶだけ
//    validatePublication() は呼ばれていない
```

`RuntimePublicationBridge` は `RuntimePublicationCoordinator` テンプレートの Bridge 型として Validator を保持しているが、`publishWorld()` 内部でのみ使用される。`RuntimePublicationValidator::validatePublication()` は **誰からも呼ばれていない経路**である。

### 確定

**Validator, Admission, Executor, Coordinator の責務境界は曖昧。特に Validator の検証結果が Orchestrator の Admission 判断に統合されていない。**

---

## D. Reader Slot 使用状況 — 確定: P3（将来拡張課題で十分）

`kMaxReaders = 64`（EpochDomain.h:17）

現在の Reader 割り当て:

| Reader | 割当方法 | スロット数 |
| --- | --- | --- |
| Audio Thread | `registerReaderThread()` | 1 |
| Message Thread | `registerReaderThread()` | 1 |
| Publication Reader | `RCUReader(provider)` | 1 |
| Worker Threads (最大8) | `registerReaderThread()` | 最大8 |
| ConvolverProcessor 内部 | 独自 EpochDomain | 別管理 |
| 余剰 | — | 53 |

**現状の使用スロット: 最大11 / 64。枯渇リスクは現時点では極めて低い。** したがって P3（将来拡張課題）で妥当。

---

## E. RuntimeReaderContext 誤用箇所 — 確定: P3（優先度過大ではない）

### 全使用箇所

| 使用方法 | ファイル | 件数 |
| --- | --- | --- |
| ヘルパー関数経由（安全） | `NoiseShaperLearner.cpp`, `SpectrumAnalyzerComponent.cpp` | 2 |
| `makeRuntimeReadHandle(ctx)` 経由 | `AudioEngine.h`, `AudioEngine.*.cpp` 各所 | 9 |
| 直接構築（目視で正しい組合せ） | `RuntimePublicationOrchestrator.cpp:23` | 1 |
| 直接構築（目視で正しい組合せ） | `AudioEngine.Processing.ReleaseResources.cpp:92` | 1 |

**直接構築は2箇所のみで、いずれも目視で正しい reader/channel の組合せ。**

### 確定

**誤用箇所は発見されず。** 型安全性の欠如は理論上の問題だが、実際のコードベースで障害を引き起こしている証拠はない。P3 で妥当。

---

## F. 残存 Fallback 経路 棚卸し — 確定: 2系統 (P1)

| # | 経路 | ファイル | 条件 | 状態 |
| --- | --- | --- | --- | --- |
| 1 | EQProcessor → EpochDomain直接 | `EQProcessor.Core.cpp:56-66` | coordinator未設定時 | ❌ 削除予定 (Phase-B) |
| 2 | SafeStateSwapper → mutex priority_queue | `SafeStateSwapper.h:112-115` | リングバッファ溢れ時 | ❌ 恒久的対策未定 |

### #1 EQProcessor Fallback の詳細

```cpp
// EQProcessor.Core.cpp
bool EQProcessor::enqueueDeferredDeleteWithFallback(...) noexcept {
    if (m_retireCoordinator != nullptr) {
        // Coordinator 経路（優先）
        ...
        if (result == RetireEnqueueResult::Success)
            return true;
        return false;  // drop
    }

    // Fallback: direct EpochDomain path
    // [Phase-B] coordinator 常時設定確認後、この経路は削除
    m_epochDomain.enqueueRetire(ptr, deleter, epoch);
}
```

`m_retireCoordinator` は `setRetireCoordinator()` で設定されるが、**constructor では設定されない**。設定漏れがあると常に Fallback 経路を通る。

### #2 SafeStateSwapper Fallback の詳細

```cpp
// SafeStateSwapper.h
if (next == convo::consumeAtomic(head, std::memory_order_acquire)) {
    // バッファ溢れ: フォールバックキュー（非 RT パスなのでロック可）
    std::lock_guard<std::mutex> lock(fallbackMutex);
    fallbackQueue.push({oldState, epoch1});
    return;
}
```

リングバッファサイズは `kMaxRetired`。溢れた場合のみ mutex fallback に移行。**通常運用では発生しないが、大量の ConvolverState 切り替えが発生した場合にボトルネックとなる可能性。**

### 確定

**2系統の Fallback 経路が残存。特に #1 は coordinator 設定漏れのリスクがあり、P1 相当。**

---

## G. Evidence System 詳細調査 — 確定: P1（v4 修正後の評価で妥当）

### EvidenceExporter 生成物の実データ反映状況

| 成果物 | ファイル | schema | 実データ反映 |
| --- | --- | --- | --- |
| `closure_graph.json` | evidence/ | closure_graph_v1 | ❌ 静的テンプレート `nodeCount:0` |
| `mutation_fault_trace.json` | evidence/ | mutation_fault_trace_v1 | ⚠️ sealViolationCount のみ動的 |
| `hb_graph_trace.json` | evidence/ | hb_trace_v1 | ❌ 静的テンプレート `eventCount:0` |
| `hb_violation_report.json` | evidence/ | hb_violation_report_v1 | ❌ 静的テンプレート `violations:[]` |
| `retire_timeline.json` | evidence/ | retire_timeline_v1 | ❌ 静的テンプレート `totalTransitions:0` |
| `shutdown_trace.json` | evidence/ | shutdown_trace_v1 | ❌ 静的テンプレート `verified:true` |
| `retire_latency_report.json` | evidence/ | retire_latency_report_v1 | ❌ 静的テンプレート `withinThreshold:true` |
| `payload_tier_report.json` | evidence/ | payload_tier_report_v1 | ❌ 静的テンプレート（固定値） |
| `runtime_budget_report.json` | evidence/ | runtime_budget_report_v1 | ✅ artifactTotalBytes のみ動的 |
| `recovery_trace.json` | evidence/ | (独自) | ⚠️ ファイル存在確認のみ |
| `runtime_snapshot.json` | evidence/ | (独自) | ❌ 静的テンプレート |

### lastRejectReason の制限

`ISRRuntimePublicationCoordinator.cpp`:

```cpp
const char* lastRejectReason() const noexcept {
    switch (lastRejectCode_) {
    case RejectCode::InvalidClosure:    return "invalid closure graph";
    case RejectCode::InvalidPayloadTier: return "invalid payload tier";
    case RejectCode::None:              return "none";
    }
}
```

**PublicationAdmission の拒否理由（6種類）は全く追跡できない。**

### 確定

**11種類の evidence 成果物のうち、動的データを反映しているのは実質 0 に等しい。** 障害発生時の原因追跡には事実上使えない。P1 相当（v4 修正後）で妥当。

---

## 総合確定表

| # | 調査項目 | 確定区分 | 優先度 | 備考 |
| --- | --- | --- | --- | --- |
| A | Publication/Retire因果追跡 | **RetireId/ReclaimId 未実装** | P1 | DeletionEntry に追跡用IDなし |
| B | World Lifetime所有権 | **Single Source Of Truth 未確立** | P0-P1 | Store が public, publishWorld() が public |
| C | Admission/Validator責務境界 | **Validator 未統合** | P0-P1 | validatePublication() が誰からも呼ばれていない |
| D | Reader Slot使用状況 | **現状十分（11/64）** | P3 | 枯渇リスクなし |
| E | ReaderContext誤用箇所 | **誤用なし（2箇所とも正しい）** | P3 | 理論上の問題のみ |
| F | Fallback経路 | **2系統残存** | P1 | EQProcessor + SafeStateSwapper |
| G | Evidence System | **11種類中ほぼ全て静的テンプレート** | P1 | lastRejectReason も限定 |

### 未確定だった項目の「確定」状況

| 項目 | v4 での扱い | 確定結果 |
| --- | --- | --- |
| Publication/Retire因果追跡 | 「弱い」と記述 | **RetireId/ReclaimId はコードベース全体で存在せず確定** |
| World Lifetime所有権 | 「近い」と記述 | **Store::publishAndSwap が public パスから呼べることを確定** |
| Admission/Validator責務境界 | 「曖昧」と記述 | **Validator::validatePublication() が誰からも呼ばれていないことを確定** |
| Reader Slot固定64 | P3妥当と判断 | **使用率 11/64 でP3妥当と確定** |
| ReaderContext誤用 | P3妥当と判断 | **直接構築2箇所とも正しく、誤用なしと確定** |
| Fallback経路 | P1と評価 | **2系統の存在と条件を確定** |
| Evidence System | P1と評価 | **11種類中ほぼ全て静的テンプレートと確定** |

### 調査に使用したツール

| ツール | 使用回数 | 主な用途 |
| --- | --- | --- |
| Serena MCP (find_symbol, find_referencing_symbols) | 8回 | 構造体/メソッド定義・参照解析 |
| CodeGraph MCP (analyze_module_structure) | 2回 | モジュール構造解析 |
| Graphify MCP (god_nodes) | 1回 | 中心ノード確認 |
| grep/Select-String | 12回以上 | RetireId/ReclaimId/worldId/ReaderContext等の網羅検索 |
| 直接ファイル読取 | 10ファイル以上 | RuntimeBuilder/PublicationAdmission/EvidenceExporter 等 |

---

## 管理者レビューによる優先度再評価

調査結果の優先度を Practical Stable ISR Bridge Runtime の観点で再評価:

| 調査項目 | 確定結果 | 調査時の評価 | 再評価後 | 理由 |
| --- | --- | --- | --- | --- |
| A: 因果追跡 | RetireId/ReclaimId 不在 | P1 | **P1維持** | 因果連鎖欠落は長期運用で致命的 |
| B: World Lifetime所有権 | Store が public | P0-P1 | **P0昇格** | Authority破れは実運用事故に直結 |
| C: Validator未統合 | validatePublication() 未呼び出し | P0-P1 | **P0昇格** | 完成したValidatorを呼ばない状態が最も危険 |
| D: Reader Slot | 11/64 使用 | P3 | P3維持 | — |
| E: ReaderContext誤用 | 誤用なし | P3 | P3維持 | — |
| F: Fallback経路 | 2系統残存 | P1 | P1維持 | — |
| G: Evidence System | 静的テンプレートのみ | P1 | P1維持 | — |

### 最終 20項目 4階層（確定版）

#### P0（実運用事故に直結）— 5項目

1. **PublicationExecutor完成** — Success しか返さない空実装
2. **Authority単一化** — 3経路の publish 経路が並立
3. **Coordinator bypass除去** — makeRuntimePublicationCoordinator() が public
4. **World Lifetime Authority確立** — Store が public、誰でも publishAndSwap 可能
5. **Validator強制統合** — validatePublication() が pipeline 入口で強制されていない

#### P1（長期運用で破綻）— 8項目

1. RuntimePublicationValidator完成 — Topology/Resource/Conflict がダミー
2. Reclaim Progress Guarantee — tryReclaim() 呼び出し依存、回収停滞検知不能
3. RetirePolicy実装 — DSPRetirePolicy 等は前方宣言のみ
4. ISRRetireRouter実体化 — 全メソッドが epochDomain_->xxx() の直接転送
5. Deprecated API全廃 — 5つの[[deprecated]] + 8箇所の抑制
6. Fallback経路除去 — EQProcessor + SafeStateSwapper の2系統
7. **Publication→Retire→Reclaim因果追跡** — RetireId/ReclaimId が存在せず、因果連鎖が完全欠落
8. Evidence System実運用化 — 11種類中ほぼ全て静的テンプレート

#### P2（監査不能）— 4項目

1. Shutdown Drain Audit
2. ObserveToken拡張
3. RuntimeReadHandle統合
4. Queue Pressure Recovery

#### P3（将来リスク）— 3項目

1. Reader Stuck Detection
2. RuntimeReaderContext型安全化
3. ReaderSlotPool化

### 達成度評価

**Practical Stable ISR Bridge Runtime 達成度: 60〜65%**

根拠: Publication 系の Authority が複数経路で破れており、Publication→Retire→Reclaim の完全な因果チェーンが成立していない。最大の欠陥は **「Validator未完成」ではなく「Authority・Lifetime・Publication Pipeline がまだ単一の真実源(Source of Truth)に収束していないこと」**。

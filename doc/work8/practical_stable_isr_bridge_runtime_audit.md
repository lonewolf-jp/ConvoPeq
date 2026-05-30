# Practical Stable ISR Bridge Runtime 調査レポート

## 目的

実運用で破綻しにくい **Practical Stable ISR Bridge Runtime** への移行状況を、ソースコードと MCP 系ツール（Serena / CodeGraph）で再調査し、未達項目の妥当性を検証する。

## 調査方法

### 使用した主な観点

- `AudioEngine` の publication / retire / snapshot / runtime 読取経路
- `RuntimePublicationCoordinator` / `RuntimeStore` / `SnapshotCoordinator` / `ObservedRuntime`
- `ISRRuntimeSemanticSchema` / `ISRRetireRuntimeEx` / `ISRDebugRuntime`
- 参照関係・依存関係の追跡（Serena / CodeGraph）
- CodeGraph のインデックス整合性確認（フル再インデックスを実施）

### 実施した MCP 調査

- Serena
  - `mcp_oraios_serena_initial_instructions`
  - `mcp_oraios_serena_find_symbol`
  - `mcp_oraios_serena_find_referencing_symbols`
- CodeGraph
  - `mcp_codegraph_get_file_structure`
  - `mcp_codegraph_find_dependencies`
  - `mcp_codegraph_local_search`
  - `mcp_codegraph_global_search`
  - `mcp_codegraph_query_codebase`
- 追加実行
  - `CodeGraph Incremental Index`
  - `CodeGraph Full Index`
  - `CodeGraph Stats`

## 主な調査結果

### 1. RuntimeWorld 単一 Authority 化

**判定:** 部分達成・未完了

#### 1.1 根拠

- `AudioEngine.h` に `RuntimeState` (= `RuntimePublishWorld`) があり、以下を一括保持している。
  - `generation`
  - `graph`
  - `publication`
  - `retire`
  - `overlap`
  - `timing`
  - `latency`
  - `scheduling`
  - `resource`
  - `affinity`
  - `automation`
  - `coefficient`
- 一方で `activeRuntimeDSPSlot` / `fadingRuntimeDSPSlot` が残っており、`getActiveRuntimeDSP()` 経由の参照が複数箇所に残存している。
- `SnapshotCoordinator` も `GlobalSnapshot` の current/target を独立に保持している。

#### 1.2 実運用上の評価

- Runtime の可視状態は world 側に寄っているが、**真実の単一化は未完了**。
- Legacy slot と world の二重系が残るため、長時間運転時の不整合リスクは残存。

---

### 2. Legacy Runtime Semantic 除去

**判定:** 未完了

#### 2.1 根拠

- `ObservedRuntime` / `ObserveToken` は存在し、`SnapshotCoordinator::observeCurrentRuntime()` も定義済み。
- しかし `AudioEngine` では依然として `activeRuntimeDSPSlot` を正規経路として扱うコードが残る。
- Serena の参照追跡で `getActiveRuntimeDSP()` の参照が `commitNewDSP` / `releaseResources` / `prepareToPlay` / `logRuntimeTransitionEvent` などに多数存在することを確認。

#### 2.2 実運用上の評価

- 「現在 Runtime を `observeCurrentRuntime()` のみで取得する」状態には到達していない。
- そのため、semantic split brain を完全に排除できる状態ではない。

---

### 3. Publication Sequence 統治

**判定:** 概ね達成

#### 3.1 根拠

- `ISRRuntimeSemanticSchema.h` に以下が定義されている。
  - `PublicationSequenceId`
  - `PublicationEpoch`
  - `PublicationSemantic`
- `AudioEngine::buildRuntimePublishWorld()` で以下を設定している。
  - `publication.sequenceId`
  - `publication.epoch`
  - `publication.mappedRuntimeGeneration`
  - `publication.previousSequenceId`
- `AudioEngine::runPublicationPrecheckNonRt()` で `previousSequenceId < sequenceId` 等を検証。
- `RuntimePublicationCoordinator::commit()` でも version / sequence / epoch / mapped generation の整合性を確認。

#### 3.2 実運用上の評価

- publish 順序の追跡可能性は確保されている。
- この項目はレビュー文面よりも実装が進んでいる。

---

### 4. Visibility Monotonicity 保証

**判定:** 部分達成・未完了

#### 4.1 根拠

- `DebugRuntime::recordShadowCompareObservation()` が `sequenceId <= lastSequenceId_` を検出し、`monotonicViolationCount_` を増加させる。
- ただしこれは主に **診断/観測** であり、読取側が `lastSeenGeneration` を保持して後退を強制拒否するような防御機構にはなっていない。
- `RuntimePublicationCoordinator::commit()` には publish 側の単調性チェックがあるが、読取パスの防御とは別。

#### 4.2 実運用上の評価

- 後退の検知はあるが、**運用保護としての monotonic guarantee は不足**。
- 長時間運転での「戻ったように見える」異常を、より積極的に隔離する仕組みが必要。

---

### 5. Runtime Semantic Schema

**判定:** 部分達成

#### 5.1 根拠

- `ISRRuntimeSemanticSchema.h` に semantic 構造群が定義済み。
- `AuthorityClass` もあり、`RuntimeState` 内ではコメントで各フィールドの分類が付与されている。
- ただし、指摘にあるような `RuntimeFieldDescriptor` ベースの機械可読な field governance は確認できない。

#### 5.2 実運用上の評価

- 方向性は正しい。
- ただし、開発者が新規フィールドを追加したときの逸脱防止はまだコメント依存が中心。

---

### 6. Retire Governance

**判定:** 実装進展あり・ただしレビューの懸念は概ね妥当

#### 6.1 根拠

- `AudioEngine.Threading.cpp` に以下がある。
  - `retireQueueDepth_`
  - `fallbackQueueDepth_`
  - `retireHighWatermark_`
  - `retireLowWatermark_`
  - saturation / recovery / emergency reclaim boost
- `ISRRetireRuntimeEx` では lane / lifecycle / rollback フラグ / quarantine resident を管理している。

#### 6.2 実運用上の評価

- 完全未実装ではない。
- ただし、レビューで挙げられたような「deferral epochs や pending retire count を明示的に統治する」姿とは一致していない。
- したがって、**概念としての懸念は妥当**。

---

### 7. Shadow Compare

**判定:** 概ね達成

#### 7.1 根拠

- `AudioEngine::onRuntimePublishedNonRt()` で `debugRuntime_.recordShadowCompareObservation(world.publication.sequenceId, world.semanticHash);` を呼んでいる。
- `DebugRuntime::recordShadowCompareObservation()` は sequence monotonicity と semantic hash の差分を計測し、`emitShadowCompareCadenceReport()` を出す。

#### 7.2 実運用上の評価

- 旧 runtime と新 runtime の比較観測は実装済み。
- ただし、比較結果を publish 制御に直結する fail-safe までは薄い。

---

### 8. Rollback Framework

**判定:** 未完了

#### 8.1 根拠

- `RetireRuntimeEx` に `requestRollback()` / `canRollback()` / `setRollbackMode()` は存在する。
- しかし Serena の参照追跡では `requestRollback()` の実運用呼び出し元が見つからず、配線が実質未接続。
- `AudioEngine` の publish / retire パスにも、異常検知時に直前 runtime を再 publish する SoftRollback 経路は確認できない。

#### 8.2 実運用上の評価

- API だけがある状態。
- レビューの「ほぼ未実装」は妥当。

---

## MCP 追加調査で分かったこと

### Serena での参照追跡

- `getActiveRuntimeDSP()` は以下で参照されている。
  - `AudioEngine.Commit.cpp`
  - `AudioEngine.CtorDtor.cpp`
  - `AudioEngine.Processing.PrepareToPlay.cpp`
  - `AudioEngine.Processing.ReleaseResources.cpp`
  - `logRuntimeTransitionEvent`
- `observeCurrentRuntime()` は `AudioEngine::makeRuntimeReadView()` から使用される。
- `requestRollback()` は参照元なし。
- `recordShadowCompareObservation()` は `onRuntimePublishedNonRt()` から参照される。

### CodeGraph 再インデックス

- Incremental index は 0 件だったため、フル再インデックスを実施。
- その後 `CodeGraph Stats` で以下を確認。
  - Entities: 170312
  - Relations: 442857
  - Files: 11125

### CodeGraph から得た補強

- `AudioEngine::onRuntimePublishedNonRt()` → `RuntimePublicationCoordinator::commit()` → `recordShadowCompareObservation()` の publish 観測経路は成立。
- `AudioEngine::runPublicationPrecheckNonRt()` → `validatePublicationNonRt()` の precheck 経路も成立。
- `RetireRuntimeEx::requestRollback()` は存在するが、呼び出し配線は見当たらない。

## 総合結論

現状は **Practical Stable ISR Bridge Runtime に近づいているが、まだ中核要件を満たし切っていない**。

### 必須4項目の到達判定

1. RuntimeWorld 単一 Authority 化 → **未完了**
2. Legacy Runtime Semantic 除去 → **未完了**
3. Publication Sequence 統治 → **達成寄り**
4. Visibility Monotonicity 保証 → **未完了**

### 評価

- レビューの危機感は妥当。
- 特に、**Legacy 経路の残存** と **monotonicity の防御不足** と **rollback の未配線** は、実運用での破綻回避という観点では重要な未達点。
- 一方で、`Publication Sequence` と `Shadow Compare` はすでに一定水準まで実装されている。

## 次に進めるなら

優先度の高い順は以下。

1. `activeRuntimeDSPSlot` を「移行用」に封じ込め、read path の正規経路を world 側へ寄せる
2. `lastSeenGeneration` 型の monotonic guard を read path に追加する
3. SoftRollback の最小実装を配線する
4. Runtime field schema を機械可読化する

---

保存日: 2026-05-30
調査対象: `AudioEngine`, `RuntimePublicationCoordinator`, `SnapshotCoordinator`, `ISRRuntimeSemanticSchema`, `ISRRetireRuntimeEx`, `ISRDebugRuntime`

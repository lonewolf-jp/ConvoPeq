# ISR Bridge Runtime 実装タスク化詳細設計 rev13（1ファイル完結・最終凍結版）

作成日: 2026-05-30

関連資料（履歴参照）:

- `doc/work9/isr_bridge_runtime_implementation_tasks_phase_file_2026-05-30_rev12.md`

---

## 1. 目的と適用範囲

本書は、`rev12` の有効条項に対して、実装時の誤実装リスクと将来保守時の解釈揺れを最終補強した **単独完結版** である。以降の実装は本書のみを規範とし、差分参照を前提にしない。

対象システム前提:

- Single Audio Thread
- Immutable RuntimeWorld
- Runtime Retire Queue
- Bridge Runtime
- Generation Based Transition

---

## 2. コア設計原則

1. RT authoritative source は `RuntimeWorld` のみ
2. publish は authority 単一経路
3. world は Build->Validate->Freeze->Publish 後に不変
4. retire/free 判定責務は retire coordinator に一元化
5. Snapshot は builder 入力専用（RuntimeWorld から切断）
6. 実装収束のため、Validation内容・rollback検出・ID/Generation採番・observe用途を凍結条項化する

---

## 3. RuntimeWorld モデル

`RuntimeWorld` は次で固定する。

- `RuntimeWorldId worldId`
- `RuntimeGeneration generation`
- `RuntimeGraph graph`
- `RuntimeExecutionDescriptor executionDescriptor`
- `RuntimeMetadata metadata`

`RuntimeMetadata` 必須:

- `schemaVersion`
- `publicationSequence`

禁止:

- `RuntimeWorld` ↔ `RuntimeSnapshot` の参照/保持
- runtime instance 実体（active/fading）の world 所有
- world 内 `mutable`
- authoritative 判定に影響する world 内 atomic
- snapshot identity（`sourceSnapshotId` 等）の world 保持

---

## 4. Identity / Sequence / Generation 規約

監査単位は identity tuple:

- `worldId`
- `generation`
- `publicationSequence`

### 4.1 WorldId 規約

- `uint64_t` 相当単調増加
- 再利用禁止
- generation / publicationSequence と独立
- `worldId=generation` / `worldId=publicationSequence` 禁止
- ランダム UUID を既定採用しない

### 4.2 WorldId 発番主体（単一化）

- `RuntimeWorldIdGenerator` **唯一**が発番主体
- authority は generator を利用するのみ
- builder / publication coordinator の独自発番禁止

### 4.3 RuntimeWorldIdGenerator 並行安全性

`RuntimeWorldIdGenerator` は以下を満たすこと。

1. 重複発番ゼロ
2. 並行呼び出し下でも単調増加維持
3. 発番順が publish 順序と衝突しない

実装方式（atomic/mutex）は任意だが、上記 1〜3 の満足を優先する。

### 4.4 publicationSequence 規約

- authority 管理下の **global monotonic sequence**
- validation 成功後、commit 直前に採番
- commit 成功時に確定
- 確定値を published world metadata へコピー
- validation 失敗で確定禁止

### 4.5 Generation 採番主体（rev13 固定）

- `RuntimeGenerationGenerator` **唯一**が generation 採番主体
- authority は generator を利用して publish 値を決定する
- builder / publication coordinator / external precheck の独自 generation 採番禁止
- `if (newGeneration > oldGeneration)` の単純比較のみで rollback 検出を実装してはならない

### 4.6 Generation 運用規約

- publish 順で単調増加
- rollback/reuse 禁止
- 既定運用: 新規 published world は新規 generation
- 同一 generation で複数 world publish は既定禁止（別紙で明示許可時のみ）

### 4.7 rollback 検出規約（実装拘束）

rollback 検出は、最低でも以下を満たすこと。

1. `lastCommittedGeneration` を authority 側で保持し、publish 判定に使用する
2. 過去 generation の再受理を fail-closed で拒否する
3. 連続 publish で generation が単調増加しない入力を拒否する

受入ゲート必須テスト:

- `publish(100) -> publish(101) -> publish(100)` が失敗
- `publish(100) -> publish(101) -> publish(102) -> publish(101)` が失敗

---

## 5. Graph / Execution / Snapshot 境界

### 5.1 Routing と Graph

- `RuntimeGraph = Nodes + Edges + Routing`
- world 直下の独立 routing descriptor 禁止

### 5.2 ExecutionDescriptor

許可:

- transition policy
- fade duration
- latency policy

禁止:

- progress 値（fade position / sample counter 等）
- active/fading runtime 所有
- live counters

### 5.3 Snapshot

- `Snapshot -> WorldBuilder -> RuntimeWorld` のみ
- snapshot由来情報は「参照形式」で world に保持禁止
- copy済み値のみ許可

---

## 6. Handle / Ownership 意味論

公開型:

- `using RuntimeWorldHandle = std::shared_ptr<const RuntimeWorld>;`

### 6.1 二層 ownership

- **Authoritative ownership**: 責務概念（参照数ではない）
- **Temporary observer reference**: 観測用一時参照

明確化:

- authoritative ownership は `shared_ptr` use_count を意味しない
- use_count で authoritative owner 判定禁止

### 6.2 Authoritative owner

- Build中: `WorldBuilder`
- Publish後: `RuntimeWorldAuthority`
- Retire後: `RuntimeRetireCoordinator`

### 6.3 RetireCoordinator の所有権意味論（rev13 固定）

- `RuntimeRetireCoordinator` は retire 対象 world の **authoritative lifecycle owner** である
- observer 側の `shared_ptr` 保持は許容される
- ただし free 判定責務（RetirePending->Free 判定、解放要求確定）は `RuntimeRetireCoordinator` のみが持つ
- `RuntimeRetireCoordinator` を `shared_ptr` の唯一所有者とみなしてはならない

### 6.4 observeWorldHandle() 規約（誤用防止強化）

用途限定:

- 診断/UI用途のみ許可
- リアルタイム状態保持用途禁止
- 制御系の恒久状態源として利用禁止

API 制約:

- `observeWorldHandle()` は `[[nodiscard]]` を付与する
- 可能な限り名前を診断用途明示へ寄せる（例: `observeDiagnosticWorldHandle()`）

保持制限:

- call scope 利用のみ許可
- メンバ保存禁止
- キャッシュ禁止
- コンテナ蓄積禁止
- 非同期 capture 持続化禁止（`std::async` / lambda）

検証方式:

- 「キャッシュゼロ証明」ではなく、禁止パターン未検出を静的検査で確認

---

## 7. Observe / Publish API 境界

`RuntimeWorldAuthority` 構成:

- `RuntimeStore& store`
- `RuntimePublicationCoordinator& publication`
- `RuntimeRetireCoordinator& retire`
- `RuntimeWorldIdGenerator& worldIdGenerator`
- `RuntimeGenerationGenerator& generationGenerator`

公開 API:

- `RuntimeWorldHandle observeWorldHandle() const`（非RT、診断/UI用途限定）
- `const RuntimeWorld* observeWorldRt() const`（RT専用）
- `void publishWorld(RuntimeWorldHandle world)`

非公開責務:

- retire 登録
- publication 直接 publish

禁止:

- `retireWorld()` 外部公開
- `RuntimePublicationCoordinator::publish(...)` 直接利用

### Publication Contract

- 同一 world の再Publish禁止
- publish済み world は retire 方向のみ
- 再Publish検出責務は authority 側

---

## 8. メモリオーダ規約

Publish:

- store は `release`
- `seq_cst` 既定使用禁止

Observe:

- load は `acquire`
- `seq_cst` 使用禁止

禁止:

- acquire/release 片側のみ適用
- RT path で refcount変動を伴う共有所有権操作

---

## 9. Build / Validate / Freeze / Publish 規約

順序:

`Build Complete -> Validation -> Freeze -> Publish`

### 9.1 Builder Rule

- 未完成 world の公開禁止
- 未完成 world の validation 入力禁止
- Build Complete まで builder 内部に閉じる

### 9.2 Constructor 封鎖

- `RuntimeWorld` constructor は `private/protected`
- `friend WorldBuilder` で生成権限限定
- public constructor 残置禁止

### 9.3 Validation 主体

- `RuntimeWorldAuthority` 内部 validation 経路のみ
- builder 独自最終判定禁止
- external precheck バイパス禁止

### 9.4 Validation 必須項目

1. `schemaVersion` 互換性
2. Graph 整合性
   - Node ID 重複なし
   - Edge 参照先 node 実在
   - 不正自己循環/禁止循環の検出
3. Routing 整合性
   - routing entry の node/edge 参照整合
   - routing graph と `RuntimeGraph` の一貫性
4. ExecutionDescriptor 整合性
   - 許可項目のみ使用
   - progress/ownership/live counter の混入なし
5. Identity 整合性
   - `worldId / generation / publicationSequence`
6. 再Publish禁止契約
7. Snapshot 非保持規約（RT到達経路 0 を含む）

### 9.5 Freeze 規約

- Freeze 完了時点で `std::shared_ptr<const RuntimeWorld>` 化
- `shared_ptr<RuntimeWorld>` のまま publish 禁止
- Freeze 後の const 解除経路禁止

---

## 10. Lifecycle / GracePeriod / Retire / Free

Lifecycle:

`Published -> Observed -> Unobserved -> GracePeriod -> RetirePending -> Free`

### 10.1 GracePeriod 開始条件

対象は以下すべて満たす world のみ:

- publish対象から除外済み
- retire set 登録済み
- `world != currentPublishedWorld`

### 10.2 single-reader 最適化の前提（rev13 固定）

ConvoPeq の GracePeriod 判定は **single-reader 最適化** である。

- reader count は multi-reader registry ではない
- RT callback active flag と同義
- callback実行中=1 / callback外=0

以下は禁止:

- 本規約を無断で multi-reader registry へ拡張すること
- `oldestObservedGeneration` 主導の reader registry 判定へ置換すること

### 10.3 GracePeriod 完了条件

対象 generation を $G$ とし、以下いずれかで完了:

1. `maxObservedGeneration > G`
2. callback active flag = 0

判定規約:

- current 値の瞬間比較のみで判定しない
- observation event ベースで扱う
- 永続イベントログ導入禁止

設計注記:

- 本方式は single-reader 最適化であり、世代スキップ（例: 100->200）時にも規約上許容される

### 10.4 RetirePending 定義

- GracePeriod 完了後の free 実行待ち論理状態
- 診断上の論理状態であり、必須独立実装状態とは限らない

### 10.5 RetirePending -> Free 条件

以下すべて必須:

1. GracePeriod 完了
2. retire coordinator が対象 world を pending 集合から正当に取り出し
3. authoritative ownership 放棄済み

### 10.6 Free / Destroy 分離

- retire coordinator は「解放要求確定」の責務主体
- 実 destroy は `shared_ptr` 参照数 0 で発生
- 即時 raw delete/free 禁止

---

## 11. Quarantine 規約

- Quarantine は retire queue 上の論理状態として扱う
- 専用 subsystem 新設禁止（Queue/Manager/Registry 等）

### 11.1 Retire queue 順序規約（rev13 固定）

- retire queue の処理順序は実装依存とする
- FIFO を規範として固定しない
- ただし、**GracePeriod 条件を満たした world のみ Free 遷移可能** を厳守する

---

## 12. Metrics / 監査指標

必須:

- `lastDroppedGeneration`
- `oldestPendingGeneration`
- `newestPendingGeneration`
- `oldestRetirePendingGeneration`
- `pendingRetireGenerationCount`
- `oldestPendingAge`
- `oldestPublishedGeneration`
- `youngestPublishedGeneration`
- `oldestObservedGeneration`
- `youngestObservedGeneration`
- `oldestRetiredGeneration`
- `retireDepth`
- `retirePressure`
- `publishedWorldCount`
- `retiredWorldCount`

定義:

- `publishedWorldCount`: 累積 publish 成功回数
- `retiredWorldCount`: 累積 retire 完了回数

主要判定:

- `youngestPublishedGeneration - oldestObservedGeneration`
- `publishedWorldCount - retiredWorldCount`
- `oldestPendingAge`

---

## 13. 受け入れゲート（凍結判定）

以下をすべて満たすこと:

1. Publish/Observe source count が各1
2. RT は `const RuntimeWorld*` のみ
3. RT path で handle copy なし
4. `retireWorld()` 非公開
5. publication 直接 publish 経路なし
6. world immutable（mutableなし）
7. world/snapshot 双方向参照なし
8. RT->Snapshot 到達経路 0
9. schemaVersion 互換チェック強制
10. identity tuple 監査可能
11. WorldId 発番主体が generator 単一
12. Generation 発番主体が `RuntimeGenerationGenerator` 単一
13. publicationSequence が global monotonic で commit時確定
14. generation 単調増加（rollback/reuseなし）
15. rollback 検出テスト fail-closed（100→101→100 / 100→101→102→101）
16. constructor 封鎖がコードで enforce
17. validation 主体が authority 内部に単一化
18. validation 必須項目（Graph/Routing/Node/Edge/ExecutionDescriptor）を検証
19. Freeze 時に `shared_ptr<const RuntimeWorld>` 化
20. observe API が診断/UI用途限定で、`[[nodiscard]]` と保持禁止静的検査が適用済み
21. GracePeriod 判定が single-reader 最適化規約に準拠
22. GracePeriod 完了判定が maxObservedGeneration/active flag 規約準拠
23. RetirePending->Free 条件が retire coordinator 管理下で明示
24. RetireCoordinator が lifecycle owner であり、shared_ptr 唯一所有者ではないことが明示
25. retire queue 順序依存でなく、GracePeriod 条件で Free 正当性を担保
26. free責務と実destroy条件が分離定義
27. pending/age系メトリクス監査可能
28. RuntimeWorld が WorldBuilder 以外から生成されない

---

## 14. 実装時の注意

- Audio Thread で lock/alloc/blocking を導入しない
- `JUCE/` と `r8brain-free-src/` は編集禁止
- 過剰な汎用RCU基盤化（hazard pointer/epoch table等）を導入しない
- 本書は「アーキテクチャ変更」ではなく「実装統治条項」の最終固定版である

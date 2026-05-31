# Practical Stable ISR Bridge Runtime 完全実装 詳細設計（2026-05-31）

## 0. 目的

レビュー指摘（到達率 70〜75% 評価）を踏まえ、ConvoPeq の Runtime を

- **Bridge Runtimeとして動作**する状態から
- **長期運用で破綻しにくい Practical Stable ISR Bridge Runtime 契約準拠状態**

へ移行するための、実装可能な詳細設計を定義する。

本設計は以下を基準に整合を取る。

- `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`
- `doc/work6/base_plan.md`（v2.3）
- 現行実装（`src/audioengine/AudioEngine.h` / `AudioEngine.Commit.cpp` / `ISRRuntimeSemanticSchema.h` / `src/core/RuntimePublicationCoordinator.h`）

> 補足: リポジトリ内に `ConvoPeq.md` 同名ファイルは確認できなかったため、本設計では上記計画書を最新版基準として扱う。

---

## 1. 最終到達像（Definition of Done）

唯一の authority は `RuntimeWorld`（`RuntimePublishWorld`）のみ。

許可される意味パスを以下に固定する。

1. Build
2. RuntimeWorld 構築
3. `publish(world)`
4. `observe(const world*)`
5. `retire(world)`

これ以外の経路（field-level publish / snapshot authority / overlap多重 authority / retire曖昧化）は禁止し、verifier で fail-closed する。

---

## 2. 現状との主要ギャップ（設計入力）

1. RuntimeSemanticSchema 完成不足（authority inventory 不整合）
2. Publication Atomicity 未完成（publish境界の概念分散）
3. Partial Publication 検証不足（schema completeness reject が弱い）
4. Observe Collapse 未完成（world以外の観測経路混在）
5. Overlap Authority 未完成（overlap意味の多重化）
6. Retire Pressure Contract 未完成（観測はあるが制御が弱い）
7. Starvation Contract 未完成（epoch上限契約不足）
8. Semantic Hash 運用不足（nightly drift配線不足）
9. Shadow Compare 基盤不足（比較軸と判定統治が弱い）
10. Soak Governance 不足（PR/Nightly/Release 3層不足）

### 2.1 `plan_review.md` 妥当性判定（要約）

本レビューは網羅性が高く、特に以下は**妥当（採用）**と判定する。

- RuntimeWorld 統合先/降格先の明示不足
- LegacyTemporary の期限・所有者管理不足
- PublicationIntent の authority 逆流リスク
- Observe verifier の実装可能性担保（禁止型/禁止参照契約）
- Overlap authority と ExecutorLocal の境界不足
- Immutable world / generation monotonicity / visibility monotonicity
- RuntimeWorld self-contained / build isolation / rollback fail-safe
- retire starvation の wall-clock 補助契約
- failure taxonomy / world lifecycle / commit point 定義不足
- memory ordering / ownership transfer / ABA / RT boundary 契約不足

一方、以下は**段階導入（後続フェーズ）**とする。

- Runtime replay / audit trail / archaeology / advanced governance metrics
- semantic entropy / blast radius / churn budget の運用自動最適化

理由: 価値は高いが、まず authority 収束と publication/observe/retire 契約を先に閉じる必要があるため。

### 2.2 追加レビュー（本改訂での採否）

今回提示レビューは、前版に残っていた「実装解釈の余地」を埋める観点で妥当性が高い。以下を**採用**する。

- RuntimeWorld Self-Contained Contract
- Semantic Dependency Graph（一方向依存）
- Retire 強制回収条件の矛盾修正（即reclaimではなく escalation）
- Semantic Transaction の `Rejected` 状態追加
- Canonical Semantic Form
- Derived Semantic Non-Persistence
- Authority Exhaustiveness
- Semantic Equivalence（Equivalent/Compatible/Different）
- RuntimeWorld Replacement Atomicity
- Publication Queue Ordering Contract
- RuntimeWorld Identity Contract
- Partial Semantic Update Prohibition
- Semantic Validity + Admission 4段階モデル
- Construction Determinism Contract
- Semantic Conflict Contract

以下は**段階導入**とする（価値は高いが後続フェーズ）。

- Runtime capability の高度化（将来機能の網羅モデル最適化）
- 契約ドキュメント自動生成、長期考古学系の拡張運用

---

## 3. アーキテクチャ設計

### 3.1 Authority Inventory 正規化

`RuntimeAuthorityInventory` を導入し、runtime field を必ず次の5分類に固定する。

- Authoritative
- Derived
- Diagnostic
- ExecutorLocal
- LegacyTemporary

### 3.1.1 ルール

- 未分類 field は build fail
- `RuntimeSemanticSchema` と inventory の不一致は build fail

### 3.2 Schema 完全性契約

`RuntimeState::kFieldDescriptors` を「schema authority の完全一致」に再定義。

必須追加:

- `routing`
- `execution`
- `overlap`
- `topology`
- `generationSemantic`
- 必要に応じ `publication/retire` の構造整合強化

`SchemaCompletenessVerifier` を追加し、

- schemaの required authority field 欠落
- descriptor 側の孤立 authority field

を reject する。

### 3.3 Publication 境界統合

semantic publish API を1つに固定する。

- 許可: `publish(world)`
- 禁止: graph/fade/snapshot/transition 単位 publish

`PublicationIntent` は transport queue としてのみ扱い、authority source には含めない。

### 3.4 Partial Publication Reject

`runPublicationPrecheckNonRt()` に加え、publish直前に

- `validateSemanticCompleteness(world)`

を必須化。

reject条件例:

- required semantic field 欠落
- generation/publication/retire mapping 不整合
- projection freshness 契約違反

### 3.5 Observe Path 一本化

Audio Thread の branch source を `RuntimeWorld` のみに固定。

- snapshot は projection-only
- snapshot 由来 branch authority 禁止

`ObservePathVerifier` を追加し、Audio callback 内の world以外観測起点を検出して fail。

### 3.6 Overlap Authority 単一化

overlap branch authority は `world.overlap` のみ。

`CrossfadePreparedSnapshot` や atomic群は ExecutorLocal として維持可能だが、branch authority は持たせない。

`OverlapAuthorityVerifier` で、world.overlap 以外の overlap branch を fail。

### 3.7 Retire Pressure Feedback 完成

`RetirePressurePolicy` を実装（閾値は初期値、調整可能）。

- Mild（>=75%）: rebuild coalescing
- Medium（>=90%）: publication throttle
- Severe（>=95%）: rebuild reject / admission strict
- Critical（継続）: emergency drain + protective mode

すべて evidence に出力。

### 3.8 Starvation Contract

`maxRetireDeferralEpochs` を導入。

- `epochNow - retireEpoch > maxRetireDeferralEpochs` で強制 reclaim
- 「eventual」ではなく bounded reclaim を保証

### 3.9 Semantic Hash + Shadow Compare

`RuntimeSemanticHash` は nightly drift pipeline に接続。

比較項目（v2.3準拠）:

- generation
- topology
- overlap decision
- retire ordering
- execution ordering
- publication timing
- visibility delay
- semantic hash

判定:

- hash mismatch は mismatch
- hash match は十分条件にしない（direct checks併用）

### 3.10 Soak Governance 3層

- PR: static verifier + short churn
- Nightly: 24h/72h soak
- Release: 1week相当 + full matrix

監視必須:

- publication drift
- retire drift
- semantic drift
- backlog slope
- visibility percentile

### 3.11 RuntimeWorld 統合・降格マップ（新規）

`RuntimeWorld` を唯一 authority container とし、以下へ収束する。

```cpp
RuntimeWorld
{
    RuntimeTopology   topology;
    RuntimeRouting    routing;
    RuntimeExecution  execution;
    RuntimePublication publication;
    RuntimeOverlap    overlap;
    RuntimeRetire     retire;
}
```

降格方針（明示）:

- `RuntimeGraph` → topology projection（non-authoritative）
- `RuntimeBuildSnapshot` → build artifact（immutable, pre-publish only）
- `TransitionState` → ExecutorLocal
- `CrossfadePreparedSnapshot` → ExecutorLocal
- `PublicationIntent` → transport queue（semantic authority禁止）

### 3.11.1 RuntimeWorld Self-Contained Contract（新規）

Published RuntimeWorld は、意味決定に必要な情報を world 内に閉じる。

禁止:

- `world + external mutable state` による意味決定
- global registry / singleton mutable state / external authority table 参照

許可:

- value
- immutable snapshot
- runtime-owned immutable object

### 3.11.2 Semantic Dependency Graph（新規）

依存方向を固定し、逆方向参照を禁止する。

```text
Topology
 ↓
Routing
 ↓
Execution
 ↓
Publication
 ↓
Retire
```

禁止:

- `publication -> routing` のような逆依存
- semantic layer の循環依存

### 3.12 Immutable / Monotonic / Visibility 契約（新規）

- Published `RuntimeWorld` は strict immutable
- generation は strict monotonic: $g_{n+1} > g_n$
- visible generation は monotonic non-decreasing: $v_{n+1} \ge v_n$
- publication request ordering を保持（後着publishによる逆転禁止）

### 3.12.1 RuntimeWorld Identity Contract（新規）

- `worldId`: world instance identity
- `generation`: 公開世代の単調識別子
- `semanticHash`: 内容指紋

Authoritative semantic 変更時は必ず new world を生成し、差し替え publish する。

### 3.12.2 Partial Semantic Update Prohibition（新規）

禁止:

- `world.field = ...` による published world の部分更新

許可:

- `new RuntimeWorld -> publish(newWorld)` の置換のみ

### 3.13 Build Isolation / Commit Point / Rollback 契約（新規）

- Build は current published world を branch source として参照禁止（build isolation）
- Publication Commit Point は world ポインタ可視化の原子更新成功時点
- Commit Point 以降は mutation 不可
- publish reject 時は previous world 継続（fail-safe continuity）

### 3.13.1 RuntimeWorld Replacement Atomicity（新規）

必須順序:

1. new world visible
2. old world retire start

禁止順序:

1. old world retire
2. new world visible

Visibility gap を禁止する。

### 3.13.2 Publication Queue Ordering Contract（新規）

`PublicationIntent -> Queue -> Publish` 経路において、publish順は generation順と一致しなければならない。

禁止:

- enqueue順/世代順を破る publish
- 例: `g101` publish 後に `g100` publish

### 3.14 Retire Safety / Starvation 契約（新規）

- retire前提条件: No Reader / No Executor Reference / No Pending Transition
- starvation 契約は dual-threshold

starvation 契約は dual-threshold とする。

- `maxRetireDeferralEpochs`
- `maxRetireWallClockMs`

### 3.14.1 Retire Escalation Contract（矛盾修正）

`maxRetireDeferralEpochs` / `maxRetireWallClockMs` 超過時は**即 reclaim**ではなく、まず escalation に遷移する。

reclaim 実行は以下成立時のみ許可:

- No Reader
- No Executor Reference
- No Pending Transition

### 3.15 PublicationIntent 制限契約（新規）

`PublicationIntent` に以下の保持を禁止する。

- `generation`
- `routing`
- `execution`
- `overlap`

許可は transport 最小情報のみ（例: requestId, targetWorldId, enqueueTime）。

### 3.16 Observe 禁止型契約（新規）

AudioThread の禁止参照型を定義し、静的/動的検査を併用する。

- `RuntimeGraph*`
- `RuntimeBuildSnapshot*`
- `PublicationIntent*`
- `TransitionState*`

### 3.17 Memory Ordering / Ownership / ABA 契約（新規）

- publish: release / observe: acquire を既定契約化
- ownership transfer matrix を定義（Build→Publish→Retire→Destroy）
- ABA hazard 対策として `worldId + generation + epoch` の組を識別子に採用

### 3.17.1 RT Safety Boundary Contract（新規）

主要コンポーネントは属性を持つ。

- `RTSafe`
- `NonRTOnly`
- `Mixed`

`NonRTOnly` ロジックが AudioThread から呼ばれた場合は verifier fail。

### 3.17.2 Executor Snapshot Freshness Contract（新規）

ExecutorLocal snapshot は world generation と整合しなければならない。

判定基準:

- `executorSnapshotGeneration == worldGeneration`

不一致は drift として検出し、publish/execute admission で扱う。

### 3.17.3 Construction Determinism Contract（新規）

同一入力（同一 semantic inputs）からは同一 canonical world を構築可能でなければならない。

`DeterministicBuildVerifier` で検証し、非決定要素（timestamp等）は diagnostic 領域へ隔離する。

### 3.18 反回帰統治（新規）

- Authority Regression Gate（未登録 authority 追加をCI fail）
- RuntimeWorld Expansion Gate（authoritative semantic 以外の field 追加禁止）
- Hidden Authority Detector（branch source の棚卸し）
- Semantic Alias Prohibited
- One Authority = One Writer（multi-writer禁止）
- Projection → Authority rehydration 禁止
- Governance Coverage Metrics（Authority/Verifier/Test/Evidence カバレッジ）

### 3.19 Canonical / Validity / Admission 契約（新規）

#### 3.19.1 Canonical Semantic Form

One Semantic = One Canonical Representation を強制する。

例: `fadeSamples` と `fadeMs` の同時 authority 化を禁止し、変換は projection で実施。

#### 3.19.2 Derived Semantic Non-Persistence

Derived semantic は persisted authority として保持してはならない。

#### 3.19.3 Semantic Validity

Completeness とは別に validity を検証する。

- 例: 範囲、符号、整合条件（`OverlapDisabled` なのに `fadeDuration > 0` 等）

#### 3.19.4 RuntimeWorld Admission Contract

`Completeness -> Validity -> Admission -> Publish` の4段階モデルを採用する。

Admission 必須条件:

- Topology valid
- Routing valid
- Semantic complete
- No semantic conflict
- No authority leak

#### 3.19.5 Semantic Transaction（拡張）

許可状態:

- `Building -> Validated -> Committed -> Published`
- `Rejected`

`Rejected` は evidence 出力を必須とする。

#### 3.19.6 Semantic Equivalence

Shadow compare 判定は次で定義する。

- `Equivalent`
- `Compatible`
- `Different`

hash は補助指標であり、単独判定を禁止する。

---

## 4. 実装対象ファイル（第一候補）

- `src/audioengine/ISRRuntimeSemanticSchema.h`
  - authority契約の必須項目明確化
- `src/audioengine/AudioEngine.h`
  - `RuntimeState::kFieldDescriptors` 拡張
  - authority classification 補強
- `src/audioengine/AudioEngine.Commit.cpp`
  - `validateSemanticCompleteness(world)` 組み込み
- `src/core/RuntimePublicationCoordinator.h`
  - publish境界統合方針に合わせた責務明確化
- `src/audioengine/ISRRuntimePublicationCoordinator.h`
  - semantic publish contract の単一化
- `src/tests/`（新規）
  - partial publication reject
  - observe single-source
  - overlap authority singular
  - retire pressure / starvation
  - shadow compare契約

---

## 5. verifier 設計（優先）

### 5.1 必須（最優先）

1. schema completeness verifier
2. partial publication verifier
3. publication single-source verifier
4. observe path verifier
5. overlap authority verifier

### 5.2 早期導入

1. retire pressure verifier
2. retire starvation verifier
3. semantic hash drift verifier
4. shadow compare contract verifier

### 5.3 運用統治

1. soak governance verifier（PR/Nightly/Release wiring）

### 5.4 追加 verifier（plan_review 反映）

1. immutable world verifier
2. generation monotonicity verifier
3. visibility monotonicity verifier
4. publication ordering verifier
5. build isolation verifier
6. retire safety verifier
7. observe forbidden-type verifier
8. publication failure taxonomy verifier
9. memory ordering contract verifier
10. ownership transfer contract verifier
11. ABA hazard verifier
12. RT boundary verifier
13. hidden authority verifier
14. semantic alias verifier
15. multi-writer prohibition verifier

### 5.4.1 追加 verifier（本改訂で採用）

1. self-contained world verifier
2. semantic dependency graph verifier（acyclic + direction）
3. retire escalation safety verifier
4. publication queue ordering verifier
5. runtime world identity verifier
6. partial semantic update prohibition verifier
7. semantic validity verifier
8. runtime admission verifier
9. semantic conflict verifier
10. authority exhaustiveness verifier
11. semantic equivalence verifier
12. replacement atomicity verifier
13. executor snapshot freshness verifier
14. deterministic build verifier

### 5.5 Verifier 実行層（新規）

- Compile-time
- Unit
- Integration
- Soak
- Gate tier（PR / Nightly / Release）

各 verifier は `Warning / Error / Fatal` の severity table を必須化し、fail-closed 運用を実装する。

---

## 6. PR分割（優先7項目対応）

### PR-01 Partial Publication 完全排除

- `validateSemanticCompleteness(world)` 導入
- precheck mandatory 化
- reject reason evidence 出力

### PR-02 Observe Path 完全一本化

- Audio Thread の world外 authority branch 除去
- observe verifier 導入

### PR-03 Overlap Authority 単一化

- overlap branch を world.overlap のみに統制
- overlap leakage verifier 追加

### PR-04 Retire Pressure Feedback 完成

- 75/90/95% policy 実装
- throttle/reject/drain 制御配線

### PR-05 Starvation Contract 完成

- maxRetireDeferralEpochs 導入
- 強制 reclaim 経路実装

### PR-06 Semantic Hash + Shadow Compare 完成

- nightly drift 配線
- mismatch taxonomy 運用

### PR-07 Soak Governance 完成

- PR/Nightly/Release 3層の閾値と判定を固定
- 24h/72h/1week系 evidence 保存を標準化

### PR-08 RuntimeWorld 完全性・自己完結化

- Self-Contained Contract 実装
- Semantic Dependency Graph 固定
- Partial semantic update prohibition 実装

### PR-09 Admission/Validity/Queue順序の強化

- Completeness/Validity/Admission 4段階導入
- Publication Queue Ordering 契約実装
- Semantic conflict / equivalence / determinism の verifier 追加

---

## 7. 受け入れ基準（Exit Criteria）

- Partial publication: 0
- Audio Thread observe source: RuntimeWorld only
- overlap authority leakage: 0
- retire pressure policy: 75/90/95%で期待制御
- starvation violation: 0
- nightly drift detection: 稼働
- 24h/72h soak で critical mismatch: 0
- Authority Leak: 0
- Observe Leak: 0
- Projection Freshness Violation: 0
- Semantic Drift: 0
- Generation rollback: 0
- Visibility rollback: 0
- RuntimeWorld mutable-after-publish: 0
- Publication ordering violation: 0
- Self-contained violation: 0
- Dependency graph cycle: 0
- Retire escalation safety violation: 0
- Partial semantic update: 0
- Admission violation: 0
- Semantic validity violation: 0
- Semantic conflict: 0
- Deterministic build mismatch: 0

---

## 8. リスクと緩和

1. **Coordinator責務肥大化**
   - 緩和: semantic coordinator と execution helper を分離
2. **false positive増加（verifier過剰）**
   - 緩和: taxonomy + severity + tier（PR/Nightly/Release）
3. **long-runテストコスト増**
   - 緩和: PRはcheap verifier中心、nightly/releaseで重い検証

---

## 9. 実装方針メモ（運用）

- fail-closed を最優先
- authority source の増殖禁止
- legacy temporary は期限付きmanifest必須
- hashは指紋（単独で同値判定しない）
- RuntimeWorld 以外は projection/diagnostic/executor-local に限定
- Semantic Transaction: `Building -> Validated -> Committed -> Published` と `Rejected` のみ許可
- PartiallyBuilt / PartiallyCommitted の外部可視化禁止
- RuntimeWorld lifecycle: `Building -> Prepared -> Published -> Observed -> Retiring -> Retired -> Destroyed`
- Derived semantic の persistent authority 化を禁止
- Canonical form を唯一表現として強制（重複意味表現禁止）
- publish queue は generation順を必須化

---

## 10. 改訂履歴

- 2026-05-31: 初版作成（レビュー差分反映、v3.1/v2.3整合）
- 2026-05-31: `plan_review.md` 妥当指摘を反映（統合/降格マップ、Immutable/Monotonic、Build Isolation、Retire Safety、Memory Ordering、反回帰統治を追加）
- 2026-05-31: 追加レビュー反映（Self-Contained、Dependency Graph、Retire Escalation矛盾修正、Admission/Validity、Queue Ordering、Identity、Partial Update禁止、Determinism を追加）

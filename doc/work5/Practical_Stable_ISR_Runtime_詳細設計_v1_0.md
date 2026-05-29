# ConvoPeq Practical Stable ISR Runtime 詳細設計 v1.0

本書は以下を上位拘束条件として策定する。

- 基本計画: `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`
- 実装統治規約: `doc/work5/ISR_Runtime_実装統治規約_v1_1.md`

本書の目的は、実装者（人間/AI）が同一の実装判断で進められるよう、Runtime 収束戦略を **実装単位** に分解し、非機能・検証・運用を含めて定義することである。

---

## 0. 設計原則（上位規約の具体化）

### 0.1 判断優先順位（Safety-First）

実装判断は次の順で行う。

1. 実運用安全性（XRUN/破音/RT stall/deadlock/leak/suspend-resume/host互換性）
2. authority singularization
3. observe path singularization
4. legacy authority reduction
5. beautification / abstraction

### 0.2 収束ゴール（不変）

最終状態は以下を同時に満たす。

- observe 単一（Audio Thread observe = `RuntimeWorld` のみ）
- authority 単一
- publication 単一（`publish(RuntimeWorld*)` のみ）
- generation 単一（`RuntimeGeneration` のみ authoritative）
- retire 単一（retire ownership の一元化）

### 0.3 禁止（実装時）

- 新 authority source / observe path の追加
- field-level partial publication
- non-authoritative branching（`runtimeVersion`/`transitionId`/debug flag）
- RuntimeCoordinator の責務肥大化（DSP ownership, async IO 等）
- executor-local state の publish/export

---

## 1. 対象範囲と非対象

### 1.1 対象

- AudioEngine Runtime publication/consume/retire 経路
- RuntimeCoordinator の authority 判定・publish 判定・retire 制御
- crossfade/transition の semantic source 統合
- verifier/CI wiring（Tier 実行）

### 1.2 非対象

- 音響アルゴリズム自体の改善
- UI/UX 改良
- 汎用フレームワーク化
- vendor ソース改変（`JUCE/`, `r8brain-free-src/`）

---

## 2. 論理アーキテクチャ

### 2.1 コンポーネント責務

| コンポーネント | 役割 | 禁止責務 |
| --- | --- | --- |
| `RuntimeCoordinator` | authoritative state 判定、publish/retire 制御、generation ordering 局所化 | DSP ownership、UI orchestration、async IO、cache ownership |
| `RuntimeWorld` | Audio Thread が observe する唯一 immutable snapshot | publish 後 mutation |
| `Executor` | `RuntimeWorld` を入力にオーディオ処理を実行（crossfade は executor-local detail） | semantic source の外部公開 |
| `RetireManager` | retire queue 管理、pressure フィードバック、Non-RT drain 連携 | silent drop、hidden queue expansion |
| `Verification Layer` | tiered verifier 実行・fail-closed 判定 | warning only での通過 |

### 2.2 スレッドドメイン

| ドメイン | 許可 | 禁止 |
| --- | --- | --- |
| Audio Thread | `consume(RuntimeWorld*)` + `executor.process(world)` | lock/blocking wait/alloc/free/file IO/MessageManager/SEH/例外伝播 |
| Non-RT Worker | world 構築、publish 準備、retire drain、evidence 収集 | Audio Thread 条件の逸脱を前提にした設計 |
| Message/UI Thread | UI 更新、運用可視化 | Runtime authority の直接操作 |

---

## 3. データモデル詳細

### 3.1 AuthorityClass 分類ルール

すべての runtime-related state は以下のいずれかに必ず分類する。

- `Authoritative`
- `Derived`
- `Diagnostic`
- `ExecutorLocal`
- `LegacyTemporary`

未分類 state は CI fail。

### 3.2 RuntimeGeneration 規約

- authoritative generation は `RuntimeGeneration` のみ
- runtime branching は `==` / `!=` を原則
- `isNewer(a,b)` は RuntimeCoordinator 内の単一実装のみ許可
- `runtimeVersion` は diagnostic only、`transitionId` は trace only

### 3.3 RuntimeWorld スキーマ要件

`RuntimeWorld` は publish 時点で完全整合状態であること。

- 参照可能情報は publish 後不変
- observe で必要な情報は world 内に閉じる
- crossfade は世界観意味ではなく executor-local 実行 detail として扱う

### 3.4 RetireEnqueueResult

`RetireEnqueueResult`:

- `Success`
- `QueuePressure`
- `QueueFull`
- `Shutdown`

`QueuePressure` は異常ではなく backpressure signal。Coordinator は coalescing/throttling/restart を選択する。

---

## 4. API 契約（詳細設計レベル）

### 4.1 Consume 契約（Audio Thread）

- 入力: なし
- 出力: `const RuntimeWorld*`
- 事後条件:
  - non-null world を返す（silent fallback を含め必ず有効世界）
  - consume 自体は lock-free/exception-free

### 4.2 Publish 契約（Non-RT）

- 許可 API: `publish(RuntimeWorld*)` のみ
- 事前条件:
  - world は immutable 構築済み
  - authority inventory 更新済み
  - verification impact 分析済み
- 事後条件:
  - publication は single unit
  - partial publication 不可

### 4.3 Retire 契約

- retire owner は一意
- enqueue 結果に応じて coordinator が制御
- `QueuePressure` を受けた場合、silent drop せず明示制御を実施

### 4.4 BreakGlassOverride 契約

例外運用時のみ許可。必須属性:

- `id`
- `owner`
- `reason`
- `expiration`
- `rollback_plan`

`expiration` 超過は CI fail。

---

## 5. Inventory / Manifest 設計

### 5.1 Current/Post-Migration Authority Inventory

変更前後で機械可読 inventory を提出する。

必須フィールド:

- `state`
- `authority_class`
- `owner`
- `readers`
- `writers`
- `thread_domain`
- `publication_path`
- `observe_path`
- `retirement_owner`

### 5.2 LegacyTemporary Manifest

管理ファイル: `.github/isr-legacy-temporary.json`

必須フィールド:

- `symbol`
- `owner`
- `replacement_authority`
- `removal_phase`
- `deadline`
- `scope`

未登録 legacy は fail-closed。

---

## 6. フェーズ別詳細設計

### 6.1 Phase 1: Authority Freeze

実装項目:

- generation authoritative source を `RuntimeGeneration` に固定
- publication authority を単一化
- AuthorityClass 注釈の全域導入
- non-authoritative branch の封じ込み

DoD:

- authority source 増加 0
- non-authoritative branch 0
- smoke/standard tier pass

### 6.2 Phase 2: Observe Path Unification

実装項目:

- Audio Thread observe path を world 1本へ収束
- observe shim/side-channel を削減

DoD:

- Audio Thread observe = RuntimeWorld only
- observe path 増加 0

### 6.3 Phase 3: Legacy Authority Removal

実装項目:

- dual authority の片系停止
- legacy observe/export の撤去

DoD:

- dual authority coexistence 終了
- legacy temporary の期限内縮退

### 6.4 Phase 4: Crossfade Executor-local Migration

実装項目:

- crossfade semantic source の world 外化
- overlap handling を reject/coalesce/restart に固定

DoD:

- semantic merge 0
- executor-local leakage 0

### 6.5 Phase 5: Publication Atomicity Completion

実装項目:

- `publish(RuntimeWorld*)` 以外の publish 経路を撤去

DoD:

- partial publication 経路 0

### 6.6 Phase 6: Retire Pressure Governance

実装項目:

- retire backlog を pressure-aware 制御へ統一
- shutdown reclaim ルール固定

DoD:

- silent drop 0
- backlog slope stable
- retention leak non-regression

---

## 7. Verification 設計

### 7.1 実行ポリシー

- runtime 変更では必ず `isr-run-tiered-verification.ps1` を起点実行
- Tier は `smoke` / `standard` / `exhaustive`
- default は `standard`、release 前は `exhaustive` を必須

### 7.2 変更種別→必須 verifier

詳細は `ISR_Runtime_実装統治規約_v1_1.md` 第12章に従う。特に:

- RuntimeWorld/生成系変更: immutability + generation drift + observe shim
- retire系変更: retire lane + bridge + v7.3 reclaim/residency
- crossfade系変更: crossfade observable + rtmutable boundary

### 7.3 fail-closed 判定

いずれか fail でマージ不可。warning での暫定通過は禁止。

---

## 8. Soak / 非機能検証設計

### 8.1 必須シナリオ

- IR reload storm（10Hz〜50Hz, 4h）
- automation storm
- suspend/resume storm
- sample rate churn
- UI attach/detach storm

### 8.2 主要メトリクス（非悪化）

- XRUN count
- stale observe count
- retire backlog slope
- world leak count
- publication latency drift
- overlap rejection count

### 8.3 failure taxonomy

- Class-A: audio corruption
- Class-B: generation drift
- Class-C: stale observe
- Class-D: retire backlog divergence
- Class-E: world retention leak
- Class-F: authority duplication regression

---

## 9. 運用手順（実装PR単位）

### 9.1 実装前

1. Current Authority Inventory 作成
2. authority impact analysis 作成
3. verification impact 作成

### 9.2 実装中

1. phase 境界を超えない
2. 新state追加は原則禁止
3. 例外は BreakGlassOverride + deadline + CI guard

### 9.3 実装後

1. Post-Migration Authority Inventory 更新
2. legacy manifest 更新（該当時）
3. tiered verification 実行
4. soak 影響分析記録

---

## 10. Break-glass 運用詳細

### 10.1 許可条件

- 緊急障害回避に限定
- rollback 計画が明確
- 期限が明確
- release branch は承認必須

### 10.2 禁止

- 無期限 temporary
- undocumented bypass
- permanent suppression

### 10.3 解除条件

- 代替 authority 実装完了
- soak 非悪化確認
- manifest から legacy 削除

---

## 11. プロジェクト固有ハード制約

- `JUCE/` / `r8brain-free-src/` 編集禁止（例外は明示承認付き break-glass のみ）
- Audio Thread で allocation/deallocation/lock/blocking/file IO/MessageManager/sleep/condition_variable wait/SEH/例外伝播を禁止
- oneMKL/SIMD メモリは non-RT で 64-byte aligned allocation を必須
- ISR runtime path は exception-free

---

## 12. 受入基準（詳細設計としての完了条件）

本詳細設計は、以下を満たした時点で採用可能とする。

1. v3.1 基本計画の5単一（observe/authority/publication/generation/retire）に矛盾しない
2. v1.1 規約（Safety-first, Break-glass, Verification Matrix, Hard Constraints）に矛盾しない
3. 実装前後提出物（authority inventory, impact analysis, verification impact）が定義済み
4. Tiered verification と soak 検証が運用手順として閉じている

---

## 13. 最終判断原則

すべての変更提案は、最終的に次の問いで判定する。

> この変更は、実運用安全性を維持したまま authority topology を収束させるか？

Yes の場合のみ採用候補とする。

# ConvoPeq Practical Stable ISR Runtime 詳細設計 v1.1

本書は v1.0 に対する差分マージ版である。上位拘束条件は下記2点。

- 基本計画: `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`
- 実装統治規約: `doc/work5/ISR_Runtime_実装統治規約_v1_1.md`

v1.0 からの主要変更（Must 6項目）:

1. Documentation Scope Rule を §9 に正式取り込み
2. Governance Budget を §6 各 Phase DoD に紐付け
3. State Addition Exception の全必須条件を §9.2 に明示
4. RT 側 Retire QueueFull / Shutdown の挙動規定を §4.3 に追加
5. Crossfade 境界（world含む/含まない state）を §6.4 / §3.3 に明示
6. Phase DoD と検証スクリプトの紐付け表を §7.4 に追加

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

### 3.3 RuntimeWorld スキーマ要件（境界の明示）

`RuntimeWorld` は publish 時点で完全整合状態であること。

**world に含めるべき state（authoritative / derived）:**

- generation 識別子（`RuntimeGeneration`）
- 現在の DSP graph snapshot（immutable）
- パラメータ snapshot（immutable）
- latency / sample rate / block size の確定値
- crossfade の **意味的属性のみ**（例: 「fade対象であるか」「fade先 graph」など authority に属する事実）

**world に含めてはならない state（executor-local）:**

- fade progression（経過比率、サンプルカウント、エンベロープ位置）
- interpolation phase
- meter / level 計測値
- executor 内部のスムージング状態
- DSP 内部リソース（mkl handle、テンポラリバッファ等）

**判定原則:**

- Audio Thread 以外がその値を見る必要があるか → No なら executor-local
- publish 後に変化する性質があるか → Yes なら executor-local（world外）
- generation を越えて引き継がれるか → No なら executor-local

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
- 事前条件:
  - 初回 publish 完了済み（初期 world は Non-RT 側で構築 → publish 完了後にのみ Audio Thread が起動）
- 事後条件:
  - 常に non-null world を返す
  - 初期化未完了時は Audio Thread を起動しない（consume の戻り値で初期化未完了を表現しない）
  - consume 自身は lock-free / exception-free / wait-free（O(1)）

### 4.2 Publish 契約（Non-RT）

- 許可 API: `publish(RuntimeWorld*)` のみ
- 事前条件:
  - world は immutable 構築済み
  - authority inventory 更新済み
  - verification impact 分析済み
- 事後条件:
  - publication は single unit
  - partial publication 不可
  - publish 完了後は同 world オブジェクトを mutation してはならない

### 4.3 Retire 契約（RT/Non-RT 双方）

- retire owner は一意（`RetireManager`）
- enqueue 結果に応じた **確定挙動**:

| 結果 | RT 側挙動 | Non-RT / Coordinator 側挙動 |
| --- | --- | --- |
| `Success` | 何もしない | 通常 drain |
| `QueuePressure` | 何もしない（RT は backpressure を観測しない） | coalescing / throttling / restart を実施 |
| `QueueFull` | **RT は block/alloc/log を行わず即時 return**。世界は最新 publish 済 world を継続使用 | Non-RT で overflow lane に確実回収し、次 publish までに drain。silent drop 禁止 |
| `Shutdown` | RT は新規 publish を期待せず、最後に observe した world で processing 終了まで継続 | Non-RT が全 outstanding world を回収。retention leak を発生させない |

- RT 側で enqueue 失敗時に **代替 publish を行ってはならない**（=部分 publish 禁止の派生）
- RT 側のフォールバックは「直前 world の継続 observe」のみ許可

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

## 6. フェーズ別詳細設計（Governance Budget 統合）

各 Phase の DoD には、規約 v1.1 §5.11 の Governance Budget を **数値で組み込む**。

共通 Budget:

- Authority Migration Budget: phase 内で `authority_source 増加 = 0`
- Observe Growth Budget: phase 内で `observe_path 増加 ≤ 0`
- Legacy Lifetime Cap: 任意 LegacyTemporary の存続 phase 数 ≤ 2
- Semantic Duplication Budget: 同一 semantic state の同時並列箇所 ≤ 2

### 6.1 Phase 1: Authority Freeze

実装項目:

- generation authoritative source を `RuntimeGeneration` に固定
- publication authority を単一化
- AuthorityClass 注釈の全域導入
- non-authoritative branch の封じ込み

DoD（Budget込み）:

- authority source 増加 = 0
- observe path 増加 ≤ 0
- non-authoritative branch 0
- AuthorityClass 未分類 state 0
- smoke / standard tier pass

### 6.2 Phase 2: Observe Path Unification

実装項目:

- Audio Thread observe path を world 1本へ収束
- observe shim/side-channel を削減

DoD（Budget込み）:

- Audio Thread observe = RuntimeWorld only
- observe path 増加 ≤ 0（純減目標）
- Semantic Duplication ≤ 2
- standard tier pass

### 6.3 Phase 3: Legacy Authority Removal

実装項目:

- dual authority の片系停止
- legacy observe/export の撤去

DoD（Budget込み）:

- dual authority coexistence 終了
- LegacyTemporary 存続 ≤ 2 phase
- legacy manifest と実態一致（差分0）

### 6.4 Phase 4: Crossfade Executor-local Migration

実装項目:

- crossfade semantic source の world 外化（§3.3 の境界に従う）
- overlap handling を reject/coalesce/restart に固定

**境界（再掲・固定）:**

- world 内に残してよいもの: 「fade の有無」「fade先 graph」など authoritative な事実
- world から外すもの: fade progression、interpolation phase、サンプルカウント、エンベロープ位置

DoD（Budget込み）:

- semantic merge 0
- executor-local leakage 0
- observe path 増加 = 0
- Semantic Duplication ≤ 2

### 6.5 Phase 5: Publication Atomicity Completion

実装項目:

- `publish(RuntimeWorld*)` 以外の publish 経路を撤去

DoD（Budget込み）:

- partial publication 経路 0
- publish API 数 = 1

### 6.6 Phase 6: Retire Pressure Governance

実装項目:

- retire backlog を pressure-aware 制御へ統一
- shutdown reclaim ルール固定（§4.3）

DoD（Budget込み）:

- silent drop 0
- backlog slope stable（soak で非悪化）
- retention leak 非悪化
- LegacyTemporary 残存 0（移行完了時点）

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

### 7.4 Phase DoD ↔ 必須 verifier 紐付け表

Phase 完了判定では下記の必須スクリプトが pass していることを最低条件とする。

| Phase | DoD 主要項目 | 必須 verifier（最低） |
| --- | --- | --- |
| Phase 1: Authority Freeze | authoritative generation singularization / AuthorityClass 完備 / non-authoritative branch 0 | `isr-verify-v1-immutability.ps1` / `isr-verify-phase4-generation-drift.ps1` / `isr-verify-gate-wiring.ps1` / `isr-verify-validator-tiering.ps1` |
| Phase 2: Observe Path Unification | Audio Thread observe = RuntimeWorld only | `isr-verify-observe-shim-usage.ps1` / `isr-verify-rtmutable-boundary.ps1` / `isr-verify-facade-bypass.ps1` |
| Phase 3: Legacy Authority Removal | dual authority 終了 / legacy manifest 整合 | `isr-verify-cleanup-deferred.ps1` / `isr-verify-trigger-cleanup-readiness.ps1` / `isr-verify-trigger-cleanup-completion.ps1` / `isr-verify-rollback-matrix.ps1` |
| Phase 4: Crossfade Executor-local Migration | semantic merge 0 / executor-local leakage 0 | `isr-verify-crossfade-observable-state.ps1` / `isr-verify-observe-shim-usage.ps1` / `isr-verify-rtmutable-boundary.ps1` |
| Phase 5: Publication Atomicity Completion | `publish(RuntimeWorld*)` 単一 | `isr-verify-v1-immutability.ps1` / `isr-verify-v3-runtime-graph-immutability.ps1` / `isr-verify-v4.ps1` |
| Phase 6: Retire Pressure Governance | silent drop 0 / backlog slope stable / retention leak 非悪化 | `isr-verify-v5-retire-authority-lane.ps1` / `isr-verify-v7-rt-nonrt-retire-bridge.ps1` / `isr-verify-v73-admission-funnel.ps1` / `isr-verify-v73-shutdown-reclaim.ps1` / `isr-verify-v73-residency-telemetry.ps1` |

すべての Phase で起点は `isr-run-tiered-verification.ps1 -Tier <standard|exhaustive>` とする。

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

非悪化判定は「直前マージ時点ベースライン以下」を CI 上で記録・比較する。

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

1. phase 境界を超えない（越境は §10 Break-glass のみ）
2. 新 state 追加は原則禁止
3. 例外的に新 state を追加する場合、**以下 5 条件すべて**を満たすこと:
   - `BreakGlassOverride` の発行（必須属性完備）
   - `deadline`（期限）設定
   - `removal_phase`（撤去予定 Phase）設定
   - CI guard（該当 state を対象とする検出ルール）設定
   - soak validation（影響範囲の非悪化確認）実施
   - 追加 state は `LegacyTemporary` に分類し manifest 登録

### 9.3 実装後

1. Post-Migration Authority Inventory 更新
2. legacy manifest 更新（該当時）
3. tiered verification 実行（標準は `standard`、release 前は `exhaustive`）
4. soak 影響分析記録

### 9.4 Documentation Scope Rule（必須更新対象）

runtime semantics を変更した PR では、最低限以下を同一 PR 内で更新する。

- `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`（方針差分があれば）
- `doc/work5/ISR_Runtime_実装統治規約_v1_1.md`（規約差分があれば）
- 本詳細設計（`doc/work5/Practical_Stable_ISR_Runtime_詳細設計_v1_1.md`）
- topology 差分文書
- authority inventory（前後）
- `.github/isr-legacy-temporary.json`（legacy 変更時）
- verification matrix（変更種別に該当する場合）

コードのみ変更で PR を閉じることは禁止。

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

### 10.4 期限到来時の自動挙動

- `expiration` 超過の `BreakGlassOverride` は CI fail（warning 不可）
- 期限延長は新規 override 発行に等しく、再承認必須

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
2. v1.1 規約（Safety-first, Break-glass, Verification Matrix, Hard Constraints, Documentation Scope, State Addition Exception）に矛盾しない
3. 実装前後提出物（authority inventory, impact analysis, verification impact）が定義済み
4. Tiered verification と soak 検証が運用手順として閉じている
5. Phase DoD と Governance Budget が紐付いている
6. Phase DoD と verifier が紐付いている

---

## 13. 最終判断原則

すべての変更提案は、最終的に次の問いで判定する。

> この変更は、実運用安全性を維持したまま authority topology を収束させるか？

Yes の場合のみ採用候補とする。

---

## 付録A: v1.0 → v1.1 差分サマリ

| 項目 | v1.0 | v1.1 |
| --- | --- | --- |
| Documentation Scope Rule | 言及のみ | §9.4 に必須更新対象を明文化 |
| Governance Budget | 未収録 | §6 各 Phase DoD に数値統合 |
| State Addition Exception | 言及のみ | §9.2 に 5 必須条件を列挙 |
| RT Retire 失敗時挙動 | 未定義 | §4.3 に QueueFull / Shutdown の確定挙動を表で定義 |
| Crossfade 境界 | 抽象記述 | §3.3 / §6.4 で world含む/含まない state を明示列挙 |
| Phase DoD ↔ verifier | 紐付けなし | §7.4 に Phase 別必須スクリプト表を追加 |
| 期限到来時 CI 挙動 | 未記載 | §10.4 に fail-closed を明記 |

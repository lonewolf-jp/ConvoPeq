# ConvoPeq Practical Stable ISR Runtime フェーズ別実装タスク分解 v1.0

本書は以下の設計図書を拘束条件として、実装作業をフェーズ単位で分解した実行計画である。

- 基本計画: `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`
- 実装統治規約: `doc/work5/ISR_Runtime_実装統治規約_v1_1.md`
- 詳細設計: `doc/work5/Practical_Stable_ISR_Runtime_詳細設計_v1_2.md`

---

## 0. 実行共通ルール（全フェーズ適用）

### 0.1 絶対遵守

1. Safety-First（実運用安全性最優先）
2. 5単一（observe / authority / publication / generation / retire）への収束
3. fail-closed（warning通過禁止）
4. Runtime semantics 変更時は Documentation Scope Rule 必須更新

### 0.2 Governance Budget（毎フェーズで必須判定）

- Authority Migration Budget: authority source 増加 = 0
- Observe Growth Budget: observe path 増加 ≤ 0
- Legacy Lifetime Cap: LegacyTemporary 存続 ≤ 2 phase
- Semantic Duplication Budget: 同一 semantic state 並列 ≤ 2

### 0.3 必須成果物（PRごと）

- `Current Authority Inventory`
- `Post-Migration Authority Inventory`
- `inventory_diff_report`
- authority impact analysis
- verification impact
- safety regression 判定表（SafetyPass）

### 0.4 PRクラス/Tier（v1.2準拠）

- Class-S: smoke
- Class-A: standard
- Class-B/C/D: exhaustive

release 直前は全クラス exhaustive。

---

## 1. Phase 1: Authority Freeze（権威源固定）

### 1.1 目的

- generation authoritative source を `RuntimeGeneration` に固定
- authority classification を完備
- non-authoritative branch を除去

### 1.2 実装タスク

- [P1-T01] Runtime state inventory の current 生成
- [P1-T02] 全 runtime-related state へ `AuthorityClass` を付与（未分類ゼロ化）
- [P1-T03] generation 参照経路の一本化（`RuntimeGeneration` 以外を diagnostic/trace へ格下げ）
- [P1-T04] `runtimeVersion` / `transitionId` / debug flag を分岐条件に使う箇所の除去
- [P1-T05] `isNewer(a,b)` を RuntimeCoordinator 内の単一実装へ隔離
- [P1-T06] Post inventory 生成 + diff 作成

### 1.3 検証タスク

- [P1-V01] `isr-run-tiered-verification.ps1 -Tier standard`
- [P1-V02] `isr-verify-v1-immutability.ps1`
- [P1-V03] `isr-verify-phase4-generation-drift.ps1`
- [P1-V04] `isr-verify-gate-wiring.ps1`
- [P1-V05] `isr-verify-validator-tiering.ps1`

### 1.4 DoD

- authoritative generation singularization 達成
- AuthorityClass 未分類 0
- non-authoritative branch 0
- Governance Budget 全項目 pass

---

## 2. Phase 2: Observe Path Unification（観測経路一本化）

### 2.1 目的

- Audio Thread observe を `RuntimeWorld` のみに収束

### 2.2 実装タスク

- [P2-T01] Audio Thread の observe path を棚卸し（atomic/slot/flag を列挙）
- [P2-T02] `consume(RuntimeWorld*)` 経由へ順次統一
- [P2-T03] observe shim/side-channel の削除または read-only 診断化
- [P2-T04] 初回 publish 前に Audio Thread を起動しない起動順制約を固定
- [P2-T05] inventory diff 更新（observe_path 差分確認）

### 2.3 検証タスク

- [P2-V01] `isr-run-tiered-verification.ps1 -Tier standard`
- [P2-V02] `isr-verify-observe-shim-usage.ps1`
- [P2-V03] `isr-verify-rtmutable-boundary.ps1`
- [P2-V04] `isr-verify-facade-bypass.ps1`

### 2.4 DoD

- Audio Thread observe = RuntimeWorld only
- observe path 増加 ≤ 0（純減を推奨）
- stale observe 非悪化

---

## 3. Phase 3: Legacy Authority Removal（暫定権威撤去）

### 3.1 目的

- dual authority coexistence の終了
- LegacyTemporary の計画撤去

### 3.2 実装タスク

- [P3-T01] `.github/isr-legacy-temporary.json` と実コードの突合
- [P3-T02] replacement_authority 準拠で legacy reader/writer を段階停止
- [P3-T03] removal_phase 到来項目を撤去
- [P3-T04] deadline 超過項目の fail-closed 検知を有効化
- [P3-T05] legacy 変更に伴う topology 差分文書更新

### 3.3 検証タスク

- [P3-V01] `isr-run-tiered-verification.ps1 -Tier standard`
- [P3-V02] `isr-verify-cleanup-deferred.ps1`
- [P3-V03] `isr-verify-trigger-cleanup-readiness.ps1`
- [P3-V04] `isr-verify-trigger-cleanup-completion.ps1`
- [P3-V05] `isr-verify-rollback-matrix.ps1`

### 3.4 DoD

- dual authority coexistence 終了
- LegacyTemporary 存続 >2 phase の項目 0
- manifest と実態差分 0

---

## 4. Phase 4: Crossfade Executor-local Migration

### 4.1 目的

- crossfade の semantic source を world 外へ移し executor-local 化

### 4.2 実装タスク

- [P4-T01] crossfade 関連 state を「world内/外」へ再分類
- [P4-T02] world 内に残すのは authoritative 事実（fade有無・fade先 graph 等）のみへ制限
- [P4-T03] world 外へ移す state（progression/interpolation phase/accumulator）を executor-local に隔離
- [P4-T04] overlap handling を `reject/coalesce/restart` のみに固定
- [P4-T05] semantic merge 経路を削除

### 4.3 検証タスク

- [P4-V01] `isr-run-tiered-verification.ps1 -Tier exhaustive`
- [P4-V02] `isr-verify-crossfade-observable-state.ps1`
- [P4-V03] `isr-verify-observe-shim-usage.ps1`
- [P4-V04] `isr-verify-rtmutable-boundary.ps1`

### 4.4 DoD

- semantic merge 0
- executor-local leakage 0
- Class-A/B failure 非悪化

---

## 5. Phase 5: Publication Atomicity Completion

### 5.1 目的

- publication 経路を `publish(RuntimeWorld*)` 単一へ収束

### 5.2 実装タスク

- [P5-T01] publish API 呼び出し元を全列挙
- [P5-T02] 分割 publish（graph/fade/transition/snapshot 単位）を撤去
- [P5-T03] publish 前 immutable 完了チェックを導入
- [P5-T04] publish 後 mutation を静的検知対象へ追加

### 5.3 検証タスク

- [P5-V01] `isr-run-tiered-verification.ps1 -Tier exhaustive`
- [P5-V02] `isr-verify-v1-immutability.ps1`
- [P5-V03] `isr-verify-v3-runtime-graph-immutability.ps1`
- [P5-V04] `isr-verify-v4.ps1`

### 5.4 DoD

- publish API 数 = 1
- partial publication 経路 0
- Class-B/C failure 非悪化

---

## 6. Phase 6: Retire Pressure Governance

### 6.1 目的

- retire queue 圧力制御を完成
- shutdown reclaim を保証

### 6.2 実装タスク

- [P6-T01] `RetireEnqueueResult` 分岐の実装整合（Success/QueuePressure/QueueFull/Shutdown）
- [P6-T02] RT 側は QueueFull/Shutdown 時に block/alloc/log なし即時 return を保証
- [P6-T03] Non-RT overflow lane 回収 + deferred drain を実装
- [P6-T04] shutdown 時 outstanding world 回収完了を保証
- [P6-T05] residency telemetry を収集・比較

### 6.3 検証タスク

- [P6-V01] `isr-run-tiered-verification.ps1 -Tier exhaustive`
- [P6-V02] `isr-verify-v5-retire-authority-lane.ps1`
- [P6-V03] `isr-verify-v7-rt-nonrt-retire-bridge.ps1`
- [P6-V04] `isr-verify-v73-admission-funnel.ps1`
- [P6-V05] `isr-verify-v73-shutdown-reclaim.ps1`
- [P6-V06] `isr-verify-v73-residency-telemetry.ps1`

### 6.4 DoD

- silent drop 0
- backlog slope stable（非悪化）
- retention leak 非悪化（理想はゼロ）
- Class-D/E/F failure 非悪化

---

## 7. 横断タスク（全Phaseで継続）

### 7.1 Inventory 自動化

- [X-T01] current/post/diff 生成を CI ジョブ化
- [X-T02] 未分類 state / legacy未登録 / observe増加 / authority増加を fail 条件化
- [X-T03] PR artifact に inventory diff を必須添付

### 7.2 RuntimeCoordinator 状態機械実装

- [X-T04] `Bootstrapping/Ready/Publishing/Transitioning/Pressure/ShuttingDown/Faulted` の enum/state 実装
- [X-T05] 遷移ガードと fail-closed 遷移（Faulted）を統一実装
- [X-T06] 不変条件（non-null observe, monotonic generation, retire owner single）を検査可能化

### 7.3 Safety Regression 運用

- [X-T07] baseline 保存（直前 main）
- [X-T08] 指標比較（XRUN/stale/backlog/leak/p99）自動化
- [X-T09] `SafetyPass` 判定レポートを PR 必須添付

### 7.4 Documentation Scope Rule 運用

- [X-T10] runtime semantics 変更PRで以下更新をチェック:
  - `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`（必要時）
  - `doc/work5/ISR_Runtime_実装統治規約_v1_1.md`（必要時）
  - `doc/work5/Practical_Stable_ISR_Runtime_詳細設計_v1_2.md`（必要時）
  - topology差分文書
  - authority inventory
  - `.github/isr-legacy-temporary.json`（該当時）
  - verification matrix

---

## 8. 実行順序（推奨）

1. Phase 1（authority/source 固定）
2. Phase 2（observe 経路一本化）
3. Phase 3（legacy 撤去）
4. Phase 4（crossfade executor-local 化）
5. Phase 5（publish 単一化）
6. Phase 6（retire pressure 完成）

横断タスク（X-T01〜X-T10）は Phase 1 開始時点から並行着手し、Phase 3 完了までに基盤化する。

---

## 9. チケット化テンプレート（1タスク単位）

各タスクは次フォーマットで起票する。

- Task ID: `P{phase}-Txx` / `P{phase}-Vxx` / `X-Txx`
- Objective
- Scope（変更対象ファイル）
- Non-Scope
- Risks
- Required Verifiers
- Required Artifacts
- DoD
- Rollback Plan

---

## 10. 受入判定（本タスク分解書）

本分解書は以下を満たしたため受入可能とする。

1. 設計図書3点の拘束（v3.1 / 規約v1.1 / 詳細v1.2）に矛盾しない
2. フェーズDoD・検証・成果物・Budgetが1対1で追跡可能
3. MUST事項を再変更せず、MUST以外の運用/実装ギャップを実行タスクへ落とし込んでいる

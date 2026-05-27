# ConvoPeq ISR Bridge Runtime 詳細設計（キックオフ版）

最終更新: 2026-05-24
対象: `main`（dual-authority bridge runtime）
基準計画: `doc/work/bridge_runtime_migration_plan.md`
規約（厳守）: `doc/work/ISR_Bridge_Runtime_AI_暴走防止規約.md`

---

## 1. 本詳細設計の目的

本書は、bridge runtime の事故率を下げるための詳細設計開始点である。

本書の目的は次の 3 点に限定する。

1. **1PR=1責務** で実装可能な作業単位へ分解する
2. **rollback 可能** な前進のみを許可する
3. **機械判定可能な完了条件** を各作業に付与する

---

## 2. 非目標（明示）

以下は本フェーズで実施しない。

- 全面 rewrite
- subsystem 一括置換
- crossfade 全面刷新
- RuntimeGraph の責務追加
- validator の削除
- cleanup 先行実施

---

## 3. 固定設計原則（この文書で再確認）

### 3.1 最上位

- purity ではなく **controlled bridge runtime** を優先する
- dual authority は「統制下で許容」する

### 3.2 実装分割

- 1PR で変更してよい責務は 1 つ
- 動作変更と構造変更を同時に行わない

### 3.3 観測系

- IR-A（callback 中 snapshot 固定）を最優先
- observe path は `RuntimeExecutionView{snapshot, local}` へ収束

### 3.4 mutable 制御

新設 mutable は次の 2 種のみ許可。

- `RTLocalState`
- `RTAuxMutable`（counter/timing/telemetry/debug scalar のみ）

---

## 4. 実装パッケージ設計（Phase 0～1 着手分）

> すべて「1パッケージ = 1責務」。

## P0-1: ObserveToken formalization（責務: observe enter/exit の形式化のみ）

### P0-1 目的

- ObserveToken の責務を「generation pin / observe enter/exit」に固定する

### P0-1 変更種別

- **構造変更のみ**（動作変更なし）

### P0-1 完了条件（機械判定）

- retire/publish/graph mutation を ObserveToken 内で実施していない
- callback 中 snapshot 固定に関する既存検証が退行しない

### P0-1 rollback

- `isr.observe_token_formalization.enabled` フラグで無効化可能

---

## P0-2: RuntimeExecutionView 読み取り経路収束（責務: observe path 整流のみ）

### P0-2 目的

- callback での読み取り経路を `RuntimeExecutionView{snapshot, local}` に寄せる

### P0-2 変更種別

- **読取経路の構造変更のみ**（同期モデル変更なし）

### P0-2 完了条件（機械判定）

- 許可外の新規 observe path 追加 0
- callback 中 direct snapshot 取得の増加 0

### P0-2 rollback

- `isr.observe_path_converge.enabled` フラグで無効化可能

---

## P0-3: RTLocalState / RTAuxMutable 境界固定（責務: mutable の分類固定のみ）

### P0-3 目的

- RT-visible mutable の増殖を防ぐ境界を先に固定する

### P0-3 変更種別

- **構造変更のみ**（DSP アルゴリズム変更なし）

### P0-3 完了条件（機械判定）

- 新規 mutable が `RTLocalState` / `RTAuxMutable` 以外に存在しない
- `RTAuxMutable` に pointer / ownership / cache / graph / DSP handle が存在しない

### P0-3 rollback

- `isr.mutable_boundary_enforcement.enabled` フラグで無効化可能

---

## P1-1: trigger 機械判定化（責務: CI ルール導入のみ）

### P1-1 目的

- 「十分整理されたら削除」を禁止し、trigger を機械判定へ統一する

### P1-1 変更種別

- **CI/検証のみ**（runtime ロジック変更なし）

### P1-1 完了条件（機械判定）

- trigger 定義に自然言語のみの条件が残っていない
- CI rule self-test が追加され、rename/alias 耐性を確認できる

### P1-1 rollback

- `isr.trigger_gate.enabled` フラグで段階無効化可能

---

## 5. authority migration 適用順序（強制）

authority transfer は必ず次順序で実施する。

1. 新authority導入
2. read path 切替
3. write path 切替
4. metrics 確認
5. trigger 達成確認
6. 旧authority削除

逆順は禁止。

---

## 6. crossfade 取り扱い

crossfade は危険領域として独立サブシステムで扱う。

本キックオフで許可するのは次のみ。

- observable state 固定
- metrics 追加
- generation drift 検出

本キックオフで禁止するもの。

- crossfade path 全面置換
- latency alignment 全面変更
- ramp algorithm 一括変更

---

## 7. validator / CI 設計方針

### 7.1 validator

- tier 維持: smoke / standard / exhaustive
- hard dependency 化禁止
- 不要そうという理由で削除禁止

### 7.2 CI ルール

- grep は暫定（恒久化しない）
- AST / symbol reference へ移行前提
- 新規ルールは self-test 必須

---

## 8. metrics 最小セット（開始時）

追加可能メトリクスは最小限とし、次属性を必須にする。

- owner
- threshold
- retention
- action

### 初期候補

- XRUN delta
- callback jitter
- retire latency
- crossfade peak

「取るだけメトリクス」は禁止。

---

## 9. PR レビューゲート（固定順）

1. XRUN 悪化なし
2. click/pop 悪化なし
3. rollback 可能
4. dual authority 暴走なし
5. observe path 増殖なし
6. RT-visible mutation 増加なし
7. cleanup 先走りなし
8. purity 追求への逸脱なし

---

## 10. 直近着手順（実作業）

1. P0-1 の差分設計（ObserveToken の責務境界定義）
2. P0-1 の CI 検証項目定義（責務逸脱検知）
3. P0-1 実装 PR（構造変更のみ）
4. P0-1 検証通過後に P0-2 へ進む

P0-1 専用差分設計: `doc/work/P0-1_ObserveToken_formalization_PR差分設計_2026-05-24.md`

---

## 11. 変更管理ルール

- cleanup は trigger 達成確認と同一 PR でのみ許可
- 1PR で複数責務を混在させない
- 動作変更と構造変更を分離する
- rollback 不可能な変更を禁止する

---

## 12. 結論

本詳細設計開始点は、理想ISRの完成を急ぐためではない。

目的は一貫して次である。

```text
bridge runtime を長期間崩壊させない統制システム
```

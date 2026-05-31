# Practical Stable ISR Bridge Runtime AI実装統治規約 v1.1（草案）

作成日: 2026-05-31
適用範囲（Production Runtime Tree）:

- `src/audioengine/**`
- `src/convolver/**`
- `src/eqprocessor/**`
- `src/core/**`

根拠文書:

- `doc/work12/practical_stable_isr_bridge_runtime_masterplan_detailed_design_and_findings_2026-05-31.md`
- `doc/work12/authority_inventory.md`
- `doc/work12/runtime-coordinator-topology-decision.md`

---

## 0. この版の位置づけ（v1.0との差分）

本 v1.1 は、v1.0 提案を「採用 / 修正採用 / 統合削除」へ再編した運用版草案である。

- 採用強化: Rule-31, 33, 34, 35, 36, 42, 43
- 修正採用: Rule-02, 03, 10, 16, 17, 26, 39, 40
- 統合整理: Rule-04+40、Rule-05+33、Rule-07/08/09+35

---

## 1. 基本原則

### Rule-01（維持）

本改修は単なる局所リファクタリングではなく、Authority Topology Collapse と Practical Stable ISR Bridge Runtime への移行である。
局所修正のみで完了扱いしてはならない。

### Rule-31（新規・最重要）

Authority Rename 禁止。
名称変更で責務を温存した状態を「完了」と判定してはならない。

判定は名前ではなく「権威責務」で行う。

---

## 2. 探索規約（Evidence-first）

### Rule-02（修正採用）

原則として、探索は以下3系統を併用する。

- grep
- Serena
- CodeGraph

ただしツール障害時は一時例外を認める。

例外時必須:

1. 障害記録（Unavailable Evidence Waiver）
2. 代替証跡（grep + 実コード断面 + 手動callsite表）
3. 復旧後の再検証コミット

### Rule-03（修正採用）

CodeGraph 再インデックスは毎回必須ではなく、以下条件で必須。

1. フェーズ開始時（Phase0, 0.5, 1, 8-A, 8-B, 5, 7）
2. 主要シンボル移動/大量変更後
3. 検索結果と実コード不整合を検知した場合

### Rule-32（新規）

Semantic Grep 義務。名称検索だけでなく責務語彙を含める。

必須語彙:

- `commit`, `publish`, `retire`, `build`, `snapshot`, `transition`, `generation`
- `rollback`, `activate`, `deactivate`, `pending`, `prepare`, `execute`, `swap`, `drain`

### Rule-33（新規・最重要）

削除/到達不能化判断前に、Call Graph 全域確認を実施する。

必須観点:

- direct caller
- indirect caller
- callee
- ライフサイクル境界（startup/shutdown/timer/worker）

### Rule-40（修正採用）

`grep 0件` 単独で完了判定してはならない。
最低限、grep + Serena + CodeGraph + 実コード断面確認を要求する。

---

## 3. Authority解析規約

### Rule-07/08/09/35（統合）

Authority Inventory は各フェーズ終了時に更新し、以下3分類で再監査する。

- Authority Source
- Mirror Source
- Legacy Source

最低調査カテゴリ:

- publish / commit / retire / build
- generation / snapshot / transition
- RuntimeState / coordinator / retire / rollback 関連

一覧化前の削除を禁止する。

### Rule-15（維持）

RuntimeState 外へ実行権威（Execution Authority）を追加してはならない。

### Rule-41（新規）

Phase2 以降に RuntimeState へ追加するフィールドは、必ず次のいずれかへ分類する。

- Authority
- Projection
- Diagnostic

分類不能な追加を禁止する。

---

## 4. フェーズ統治規約

### Rule-10（修正採用）

フェーズ順序は原則固定（マスタープラン準拠）。

- 原則: `Phase0 -> 0.5 -> 1 -> 8-A -> 8-B -> 2 -> 3 -> 4 -> 5-Gate -> 5 -> 6 -> 7 -> 9 -> 10`
- 例外: セキュリティ/ビルド破綻修復のみ
- 例外時: 「フェーズ外修正記録 + 次フェーズ開始前の整合再確認」を必須化

### Rule-11（維持）

開始条件未達で次フェーズへ進んではならない。

### Rule-36（新規・最重要）

フェーズ越境実装禁止。
現在フェーズ外の設計変更・実装を「ついで」で行ってはならない。

### Rule-37（新規）

後続フェーズ完了条件の先行達成禁止。
例: Phase1 中に `transition.active` 撤去を開始してはならない。

### Rule-44（新規）

以下は実装前レビュー承認を必須とする。

- Phase0.5
- Phase5-Gate
- Phase7

---

## 5. 実装規約

### Rule-13（維持）

新規Authority追加禁止。

### Rule-14（維持）

新規 Mutable SoT 追加禁止（静的mutable、singleton SoT、権威キャッシュ等）。

### Rule-16 + Rule-17 + Rule-42（統合修正）

Temporary 実装・二重稼働は原則禁止。
ただし互換ブリッジは時限で許可し、導入時に以下を必須化する。

- 削除フェーズ
- 削除条件
- 削除対象
- owner

期限未設定の temporary を禁止する。

### Rule-38（新規）

Runtime Contract 変更時は文書化必須。

対象:

- publish contract
- retire contract
- generation contract
- snapshot contract
- rollback contract

---

## 6. Convolver統合規約

### Rule-18（維持）

Phase8-A 完了前に Phase8-B 着手禁止。

### Rule-19（維持）

RuntimeBuilder は Convolver 型へ依存してはならない。

### Rule-20（維持）

以下は最終的に Production Runtime Tree から消滅させる。

- `SafeStateSwapper`
- `PendingParams`
- `PreparedIRState`

### Rule-BSA-Exit（新規）

Phase8-A の出口条件を明示:

- Exit-1: RuntimeBuilder が `ConvolverProcessor` 型を参照しない
- Exit-2: `RuntimeBuilder.h/.cpp` から `BuildSnapshot` 参照が消滅

---

## 7. DSP Selection規約

### Rule-21（維持）

`transition.active` の先行削除禁止。

### Rule-22（維持）

先に state machine を設計する。

必須状態:

- Stable
- Entering
- Retiring

### Rule-23（維持）

AudioThread 分岐置換表を作成する。

### Rule-Phase5-Gate（新規）

Phase5 実装開始条件:

1. 状態図
2. 分岐置換表
3. 互換期間フェイルセーフ

のレビュー承認済み。

---

## 8. Retire統治規約

### Rule-24（維持）

RetireManager は新規設計案件として扱う（既存実装と誤認禁止）。

### Rule-25（維持）

統合対象:

- `audioThreadRetireOverflowPtr`
- `deferredDeleteFallbackQueue`（AudioEngine）
- `deferredDeleteFallbackQueue`（EQProcessor）
- `DeferredFreeThread`
- `ISRRetireRuntimeEx`

### Rule-34（新規・最重要）

旧経路削除前に到達不能証明を必須化する。

手順:

1. call graph で到達不能確認
2. 証跡保存
3. 削除実施

---

## 9. 検証・完了判定規約

### Rule-12（維持）

Phase出口条件は機械検証で証明する（推測禁止）。

### Rule-26（修正採用）

完了宣言前に3系統再検証（grep/Serena/CodeGraph）を実施。
障害時は Rule-02 例外運用に従う。

### Rule-27（維持）

`C1〜C15` を証跡付きで提示する。

### Rule-28（維持）

完了条件未達で完了宣言を禁止。

### Rule-39（修正採用）

規約追加時は、重要度に応じて CI 検証を同時追加または期限付きで追加する。

- Blocker級規約: 同時CI必須
- Warning級規約: 次フェーズ開始前までにCI化

---

## 10. AI自己監査規約

### Rule-29（維持）

各修正ごとに以下を提示する。

- 探索結果
- 修正理由
- 影響範囲
- 関連シンボル
- 完了条件への寄与

### Rule-30（維持）

以下判断を禁止。

- 「たぶん未使用」
- 「おそらく不要」
- 「grepで出なかったので削除」
- 「呼ばれていないように見える」

証跡必須（grep / Serena / CodeGraph）。

### Rule-43（新規・最重要）

各フェーズ完了時に提出する証跡テンプレートを固定する。

1. 探索証跡（grep/Serena/CodeGraph）
2. Authority差分（Before/After）
3. Call Graph差分（Before/After）
4. 完了条件判定（C1〜C15）

---

## 11. 違反時取り扱い（v1.1新設）

規約違反は次の3段階で扱う。

- Blocker: 実装停止・差し戻し（例: Rule-34, 35, 36, 43）
- Warning: 継続可だが是正期限必須（例: Rule-39 の期限内CI化）
- Advisory: 次フェーズで是正

---

## 12. 付録A: Ruleマッピング（v1.0→v1.1）

- そのまま採用: 01, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30
- 修正採用: 02, 03, 10, 16, 17, 26, 39, 40
- 統合: (04+40), (05+33), (07+08+09+35), (16+17+42)
- 新規追加: 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, Rule-BSA-Exit

---

## 13. 要約

v1.1 は、v1.0 の厳格性を維持しつつ、運用不能になる過剰拘束（毎回再インデックス等）を修正し、
「証跡で止める」ための必須ルール（Rule-34/35/36/43）を中核に据えた実行版草案である。

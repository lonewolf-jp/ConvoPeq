# ConvoPeq Bridge Runtime 長期安全移行計画書（統制設計版）

最終更新: 2026-05-24
対象: `ConvoPeq` 現行 `main`（dual-authority bridge runtime）

関連規約: `doc/work/ISR_Bridge_Runtime_AI_暴走防止規約.md`（AI向け暴走防止規約）
詳細設計（開始）: `doc/work/ISR_Bridge_Runtime_詳細設計キックオフ_2026-05-24.md`

---

## 1. 本計画の目的

本計画の主目的は **理想ISRの即時完成** ではなく、移行期間（6か月〜1年）における次のリスクを継続的に封じ込めること。

- UAF（Use-After-Free）
- RT blocking
- partial publish visibility
- hidden mutable access
- stale pointer
- dual authority divergence

キーワードは **runtime ambiguity reduction**。

---

## 2. 設計原則（固定）

### 2.1 原則A: 削除より流入遮断

`legacy field` / `legacy path` は即削除しない。順序は必ず以下:

1. 新規参照禁止
2. write禁止（必要なら read互換維持）
3. dead path化
4. 最終削除

### 2.2 原則B: controlled coexistence

bridge phase は「悪」ではない。危険なのは無制御bridge。
したがって dual path は短期許容し、CI・flag・triggerで統制する。

### 2.3 原則C: 責務二軸で統制

移行対象は常に以下の二軸:

- Authority（誰が publish/retire/ownership を持つか）
- Observation topology（誰がいつ何を observe できるか）

---

## 3. 目標アーキテクチャ（運用版）

### 3.1 最終目標（簡潔）

- UI thread: immutable snapshot build / publish request
- Audio thread: `RuntimeExecutionView{snapshot, local}` のみ参照
- Retire thread: retired snapshot queue のみ処理

### 3.2 RuntimeGraph の責務固定

`RuntimeGraph` は immutable processing description のみ。
以下を禁止:

- ownership coordination
- retire coordination
- runtime mutation repair

### 3.3 ObserveToken の極小責務

ObserveToken は次のみ許可:

- generation pin
- observe enter/exit

以下は禁止:

- retire logic
- publish repair
- graph mutation
- cache ownership

---

## 4. Invariant（最小セット）

### IR-A

Audio callback 中に observe snapshot は固定。

### IR-B

RT thread は blocking 禁止。

### IR-C

published object の RT-visible mutate 禁止。

### IR-D

retire object の再参照禁止。

### IR-E

`snapshot pointer + generation` は不可分公開（publication atomicity）。

### IR-F

RT mutation whitelist 違反なし。

### IR-G

facade bypass なし（許可ディレクトリ外 direct call 0）。

---

## 5. RT mutation policy

### 5.1 許可カテゴリ

1. `RTLocalState`（オーディオ処理に必要なRTローカル可変）
2. `RTAuxMutable`（補助可変）
   - 例: perf counter / callback timing / XRUN telemetry / debug stats

### 5.2 `RTAuxMutable` の厳格制約

以下を禁止:

- ownership
- lifetime管理
- pointer保持

許可する型は scalar / counter / timestamp 系のみ。

---

## 6. フェーズ計画（実運用優先）

## Phase 0: 統制基盤の先行導入

目的: 「壊れ方」を先に可視化。

- ObserveToken 仕様導入（最小責務）
- RTLocalState 分離開始（互換維持）
- 新規 legacy 参照禁止（CI）
- IR-A〜E の smoke 検証追加

完了条件:

- 新規PRで legacy direct 参照増加 0
- callback中 snapshot 固定違反 0

---

## Phase 1: 計測可能トリガー化

目的: deferred を永久化させない。

- trigger を semantic ではなく tool-detectable へ統一
- grep/lint ルール運用開始
- allowlist lifecycle policy 導入（owner/expiry/issue/rationale）

完了条件:

- deferred 項目に機械判定トリガー 100%
- expiry超過 allowlist は CI fail

---

## Phase 2: enforcement 高度化（grep→AST）

目的: textual検査の抜けを段階的に削減。

- Phase 2-1: grep/lint（既存）
- Phase 2-2: clang-tidy custom rule 追加
- Phase 2-3: symbol reference ベース検査へ移行

完了条件:

- 主要禁止ルールの AST 検査化率 80%以上

---

## Phase 3: facade 統制

目的: facade を導入して終わりにしない。

- retire facade 導入
- facade bypass CI 導入（許可ディレクトリ限定）
- facade runtime execution metrics 導入

完了条件:

- direct dependency 0（許可外）
- runtime execution count で bypass 経路 0

---

## Phase 4: crossfade 専用移行

目的: 最も壊れやすい経路を分離統制。

1. crossfade observable state 固定
2. callback 固定範囲定義
3. generation drift 検出
4. latency alignment stabilization（独立フェーズ）
5. ownership 移行

完了条件:

- pop/click 回帰基準内
- latency discontinuity 既知閾値以下

---

## Phase 5: rollback hierarchy 導入

目的: forward-only 失敗を防止。

- global rollback flag
- subsystem rollback flags
  - publication only
  - crossfade only
  - retire path only
- rollback compatibility matrix を管理

完了条件:

- 主要障害シナリオで subsystem rollback が機能

---

## Phase 6: cleanup（trigger達成後）

目的: fossil化回避。

- facade removal trigger 達成後に撤去
- legacy field 最終削除
- validator hard dependency 解除済み維持

完了条件:

- trigger 達成項目の未削除 0

---

## 7. CI / 検証運用

## 7.1 Validator tiering

- smoke: PR（早期破壊検知）
- standard: nightly（bridge anomaly 監視）
- exhaustive: weekly（深層整合検査）

### exhaustive fail SLA（必須）

- HB violation: 24h
- payload mismatch: 72h

## 7.2 PR canary metrics

- XRUN delta
- callback jitter
- retire latency
- crossfade peak

canary は baseline window normalization を実施（CPU/熱/OS揺れを正規化）。

## 7.3 CI rule self-test

CIルール自身の回帰テストを持ち、rename/aliasで空振りしないことを保証。

### 2026-05-27 実装追記（自己検証強化）

- `isr-verify-gate-wiring.ps1` に validator tiering の不変条件自己検証を追加。
  - runtime source root が `src` 全体走査であること
  - runtime source への forbidden hard dependency pattern（verifier script/evidence artifact/strict env flag）維持
- `isr-verify-gate-wiring.ps1` に trigger cleanup completion の不変条件自己検証を追加。
  - source scan root が `src` であること
  - `policyEvaluations` 検証を保持していること
  - cleanup complete 判定の必須 metrics field 群を保持していること
  - retired helper 名スキャン（legacy helper 残存検出）を保持していること

---

## 8. メトリクス運用規約（Metric Governance）

各メトリクスに以下を必須化:

- blocking: yes/no
- owner
- retention
- alert threshold
- action（閾値超過時の運用手順）

「取得するだけ」のメトリクスは禁止。

---

## 9. Trigger 一覧（機械判定）

## 9.1 代表トリガー

- `activeDSP` 削除開始: 許可外参照 0（symbol usage）
- `fadingOutDSP` 削除開始: write 0（AST rule）
- retire facade 撤去開始: direct dependency 0 + runtime execution 0
- observe shim 撤去開始: direct snapshot observe 100%

## 9.2 判定原則

trigger は必ず CI で自動判定可能であること。

---

## 10. allowlist 運用ポリシー

allowlist entry 必須属性:

- owner
- expiry
- issue
- rationale

expiry超過は fail（warning不可）。

---

## 11. 既知リスクと抑止

1. metrics explosion
→ governance 属性必須化。

2. flag interaction hell
→ feature flag dependency graph と compatibility matrix を維持。

3. RTAuxMutable 肥大化
→ pointer/ownership/lifetime 禁止。

4. validator decay
→ nightly mandatory + SLA。

5. rollback 逆依存崩壊
→ subsystem別 matrix 検証を定期実行。

---

## 12. この計画での「完成」定義

本計画における完成は「完全ISR純度」ではなく、以下の成立:

1. bridge runtime が制御不能化しない
2. CIで逸脱を機械検知できる
3. rollback が subsystem 粒度で機能する
4. deferred が trigger 駆動で確実に縮退する
5. 主要事故（UAF/RT blocking/partial visibility/stale access）が実運用で増加しない

---

## 13. 実行順（短縮版）

1. ObserveToken formalization（極小）
2. RTLocalState + RTAuxMutable 分離
3. 新規legacy参照禁止（CI）
4. callback中snapshot固定 + IR-E
5. grep/lint enforcement
6. clang-tidy/symbol enforcement
7. retire facade + bypass CI + execution metrics
8. crossfade専用移行（latency分離含む）
9. rollback hierarchy + matrix
10. validator tiering + SLA
11. trigger達成項目のcleanup

---

## 14. 結論

ConvoPeq 現状における最適戦略は、

- 完全ISRへの直行ではなく
- bridge runtime の事故率を継続的に下げる統制運用

である。
本計画はそのための長期移行オペレーティングシステムとして定義する。

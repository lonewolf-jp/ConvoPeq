# ConvoPeq ISR Runtime Reduction Strategy

## 目的

本書は R18（Verification Pipeline）の補助正本として、
ISR の安全性強化に伴う runtime 複雑化を抑制するための **runtime budget** を定義する。

本書は新規 R を追加しない。R18 の `Spec-Fixed -> Closed` 昇格に必要な運用制約のみを扱う。

---

## 基本原則

- 安全性強化は必須だが、runtime 複雑度の無制限増加は許容しない
- validator の追加は「証跡品質改善」と「実行コスト増」の両面評価を必須化する
- RT（Audio Thread）側での監視/証跡生成は zero-allocation / lock-free を維持する

---

## Runtime Budget（上限制約）

| 項目 | 制約 |
| --- | --- |
| closure traversal cost | O(N)（N=publish graph node count） |
| publish validation cost | bounded（入力サイズに対し線形上限） |
| RT instrumentation | zero alloc / no lock / no blocking |
| artifact size | bounded（保存上限を定義） |
| retire latency monitor overhead | bounded（callback jitter 悪化を許容域内に維持） |

`bounded` の具体値は CI/運用実測で決める。未確定時は conservative default を適用する。

---

## Validator 増殖抑制ルール

- 同一違反カテゴリに対して重複 validator を追加しない
- 追加時は既存 validator 統合を優先する
- 新規 validator 提案には以下を必須添付する
  - 目的（どの R の Closed 条件を満たすか）
  - 既存 validator との差分
  - runtime/CI の追加コスト見積
  - 既存 validator への統合不可理由

---

## Metadata 圧縮ルール

- Closure metadata は必要最小限の canonical field に限定する
- 同義フィールドの多重保持を禁止する
- proof artifact 生成時にのみ展開する派生情報は runtime 常駐させない

---

## RT 計測の制約

- RT で全量 trace を禁止する（sampling / ring-buffer 集約を使用）
- RT 側は event emission のみ行い、重い集計は NonRT へ委譲する
- RT 側での JSON 直列化は禁止する

---

## CI 実行予算

- CI pipeline は merge blocker を維持しつつ、実行時間予算を監視する
- 予算超過時は validator 削除ではなく、以下を優先する
  - artifact 生成の差分化
  - 非RT計測の集約
  - 重複ステージの統合

---

## R18 との接続

本書の達成条件は R18 の補助条件として扱う。

- runtime-generated proof artifacts を安定生成できる
- CI evaluator が証跡を判定できる
- 安全性を落とさず runtime/CI コストが予算内に収まる

---

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（予算値の実測固定と運用証跡が未完）

# ConvoPeq ISR 最小フェーズ0（推奨）

## 目的

`Spec-Fixed` 済みだが実装未充足の形式保証コアを先行導入し、
Phase A/B の実装を **検証可能な状態** で開始できるようにする。

本フェーズは「ISR完成」を目的としない。目的は以下の2点のみ:

- RuntimePublishWorld 周辺の重大乖離（R11/R12/R14/R15/R18）を最小コストで封じる
- 後続実装の品質判定を文書規律ではなく CI/テストで機械化する

---

## スコープ（最小）

以下5項目のみをフェーズ0対象とする。

1. R11: Closure Descriptor System
2. R12: Payload Tier System
3. R14: Deferred Retire Intent Queue
4. R15: Shutdown HB FSM（最小段階）
5. R18: CI Verification Pipeline（先行3段 + 拡張枠）

非スコープ（Phase 0ではやらない）:

- R16 reorder simulator の完全実装
- R17 epoch abstraction の全面切替
- R10 shared/split 実測評価
- Runtime matrix 全再編

---

## 入口条件（Start Gate）

- `plan5.md` がハブ専用（要約+リンク）を維持している
- `ISR_Completeness_Risk_Backlog.md` の R11/R12/R14/R15/R18 が `Spec-Fixed`
- 既存 CI（`audioengine-lint.yml`, `list-compliance.yml`）が green

---

## 実装パッケージ

### P0-1: Closure Descriptor 骨格（R11）

目的:

- publish直前に closure 検証ポイントを必須化する

最小実装:

- `PayloadClosureDescriptor`（最小ノード）を導入
- ノード属性: `kind`, `ownership`, `mutability`, `lifetime`, `hbDomain`, `authority`, `allocatorFamily`
- `validateRuntimeClosure(...)` の呼び出しを publish 経路に mandatory で挿入

受入条件:

- descriptor未登録 payload を publish しようとすると reject される
- external mutable dependency を検出して失敗させる

### P0-2: Payload Tier 骨格（R12）

目的:

- payload boundary を静的に識別可能にする

最小実装:

- `PayloadTier` を定義
  - `InlineImmutable`
  - `ImmutableShared`
  - `ExternalPinned`
  - `RTLocalOnly`
  - `Forbidden`
- publish 対象 family に tier 割当表を作成
- 検証ルール:
  - `Forbidden` は publish payload 禁止
  - `RTLocalOnly` は RuntimePublishWorld 禁止

受入条件:

- tier未割当 object family が検出される
- Forbidden/RTLocalOnly混入で CI が fail する

### P0-3: RetireIntent Bridge（R14）

目的:

- RT detect と retire authority 実行を明示分離する

最小実装:

- `RetireIntent` を導入（RTは emission のみ）
- NonRT coordinator が dequeue 後に authority enqueue 実行
- RT で retire/reclaim/delete を行わない監視を追加

受入条件:

- RT経路から direct enqueue が禁止される
- `RT detect -> NonRT dequeue -> authority enqueue` を再現試験で確認

### P0-4: Shutdown FSM 最小整流（R15）

目的:

- 現行 shutdown phase と spec phase の乖離を最小で縮める

最小実装:

- phase enum を以下に合わせる（互換マッピング可）
  - `Running`
  - `StopAudioCallbacks`
  - `DrainObservers`
  - `StopRetireIngress`
  - `EpochSettlement`
  - `CompleteReclaim`
  - `AllocatorShutdown`
  - `Finalized`
- 禁止遷移（逆行・飛び越し）を reject
- late callback / post-stop enqueue の検知フックを追加

受入条件:

- phase逆行/飛び越し遷移が fail する
- post-stop enqueue を shutdown verifier が検出する

### P0-5: CI 先行パイプライン（R18）

目的:

- フェーズ0成果を merge blocker 化する

最小実装（先行3段）:

1. Atomic Dot-Call Scan（既存活用）
2. Ownership Closure Validator（新規）
3. Shutdown Sequencing Verifier（新規）

拡張枠（Phase Cで追加）:

- Post-publish Mutation Detector
- HB Reorder Simulator
- Retire Latency Monitor

受入条件:

- 先行3段が CI 統合され、失敗時に PR merge を停止する
- 実行ログ/レポートを artifact として保存する

---

## 実行順（推奨）

1. P0-1 Closure Descriptor
2. P0-2 Payload Tier
3. P0-3 RetireIntent Bridge
4. P0-4 Shutdown FSM
5. P0-5 CI Pipeline

理由:

- 先に「何を publish してよいか」を定義（P0-1/P0-2）
- 次に「いつ/どこで retire するか」を分離（P0-3）
- 最後に終了順序を固定（P0-4）し、CIで封止（P0-5）

---

## 完了判定（Exit Gate）

以下をすべて満たしたら最小フェーズ0完了。

- R11/R12/R14/R15/R18 が `Spec-Fixed` から「実装+検証証跡あり」へ進捗
- 先行3段 CI が main で安定稼働
- publish/retire/shutdown の違反注入ケースで fail-fast が確認済み
- `plan5.md` に本書への参照が追加され、ハブ責務が維持されている

---

## リスクと回避

- リスク: 既存 phase 名変更で運用ログが読みにくくなる
  - 回避: 互換表示名（旧->新）をログに併記
- リスク: ルール導入直後に false positive が増える
  - 回避: 1週間は fail閾値を段階適用（ただし merge blocker化は維持）
- リスク: 実装先行で文書が遅れる
  - 回避: PR テンプレに「R番号・検証証跡」を必須欄として追加

---

## 成果物

- 本書: `doc/work/ISR_Minimal_Phase0_Recommended.md`
- 参照更新: `doc/work/plan5.md`（ハブリンク追加）
- トラッキング: `doc/work/ISR_Completeness_Risk_Backlog.md`（進捗更新）

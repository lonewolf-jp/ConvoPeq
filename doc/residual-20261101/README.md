# ConvoPeq 残課題解消フェーズ — 成果物索引（2026-11-01）

Owner 指示「残課題解消フェーズへの移行」（2026-11-01）に基づく調査・設計の成果物。
本ディレクトリは **文書のみ**（製品コード・テスト・合格条件を変更しない）。

| 文書 | 内容 | 対応する作業 |
|---|---|---|
| [RESIDUAL-FINAL-INVENTORY-20261101.md](RESIDUAL-FINAL-INVENTORY-20261101.md) | 全セッション・PR・ブランチ・worktree・stash・未追跡・台帳・main ソースを横断した**残課題の最終一覧**（RES-01〜RES-12、各 12 項目） | Owner 指示 A |
| [ROADMAP-20261101.md](ROADMAP-20261101.md) | 残課題の 4 区分（実装可能 / 設計待ち / Owner 判断待ち / HOLD）と**優先順位付き実行計画** | Owner 指示 A |
| [OWNER-DECISIONS-20261101.md](OWNER-DECISIONS-20261101.md) | **Owner 判断が必要な項目**（D-01〜D-07）と推奨・根拠・影響範囲 | Owner 指示 A |
| [U5-MAX-MEASUREMENT-PROPOSAL-20261101.md](U5-MAX-MEASUREMENT-PROPOSAL-20261101.md) | U5 `max` 測定構造の**技術提案**（問題分析・原因候補・分離方法・改善案・限界・回帰リスク・検証計画） | Owner 指示 B |

## 結論の要約

1. **main に取り込むべき未統合の製品ソース変更は 0 件**（112 ref・8 worktree・76 PR の多証拠判定）。
   未統合に見えるものは (a) squash 済みの重複、(b) 計測・診断専用、(c) 生成物のいずれか。
2. セッション単位の差分記録は**構造的に取得不能**
   （`session_v2.summary_files = 0` が 165 件すべて、`summary_diffs` は全件 NULL、
   `storage/session_diff` は 7 ファイルすべて 2 B）。
3. 未完成事項は **RES-01〜RES-12** の 12 件。新規発見は **RES-06（STG-8 のフレーク）**で、
   ローカル検証の再現性を損なっている。
4. U5 は **FAIL・既定 OFF** を維持。測定構造の改善案は「合格条件を変えない案 1〜5」と
   「変える案 6（未提案）」に分離して提示した。

## 変更しないもの（明記）

- U5 = FAIL / 既定 OFF
- U3 / U4 = HOLD
- B-1 / B-2 = 未承認
- 現行の許容値 +50 µs・p99 基準 = 不変
- 同一測定構造での追加反復 = 未承認
- 未追跡 18 件・backup 6 件・証拠ファイル = 削除しない
- 他セッションの worktree = 読み取り専用

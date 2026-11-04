# main 統合履歴・ビルド／テスト結果（2026-11-04）

## 1. 統合履歴（main のマージコミット）

| PR | コミット | 内容 |
|---|---|---|
| #105 | `7318e44f` | R-101: H2R1 測定 workflow（dispatch-only） |
| #106 | `a83d10ba` | R-102: R-49 診断を Windows ビルドへ移植 |
| #107 | `cf835829` | STG8 mid-run clear テストの駆動点修正 |
| #108 | `7dfb21d4` | R-107 U4: 非 RT ロック計測（既定 OFF） |
| #109 | `bd7bb14d` | 生成物・U4 検証行の更新 |
| #110 | `64ace991` | R-101 測定・U4 オーバーヘッド記録 |
| #111 | `1d5ed1bd` | 7 回目の dispatch・oneAPI 失敗の記録 |
| #112 | `f5f4a6d8` | 16 回目の dispatch・STG-8 flaky の記録 |
| #113 | `cab2ef05` | R-101 合格 5 件の記録・検証クローズ |
| #114 | `e89c66fe` | R-113/R-111/oneAPI 堅牢化/U3 プローブ |
| #115 | `bfaa9c9e` | ConvoPeq.md 再生成・記録確定 |
| #116 | `3fa7090f` | R-106 指標再定義の技術判断案 |
| #117 | `6afdee60` | 台帳の U3 実装済み化・ブランチ内訳 |
| #118 | `5d88dc95` | metric v2 事前登録・測定ドライバ |
| #119 | `ecfdb244` | metric v2 新規 24 実行 → PASS |
| #120 | `7a34fa40` | R-101/R-102 の現行 main 再確認 |
| #121 | `8561091d` | R-101 oneAPI キャッシュ |
| #122 | `7392ca52` | R-108 ブランチ整理・最終監査・台帳 §11 |
| #123 | `6eeb1d42` | 監査記録の記述修正 |
| #124 | `28cd3d1e` | R-13 Reader guard 順序修正 |
| #125 | `38075a7a` | ConvoPeq.md 再生成（R-13 反映） |
| #126 | `e69f20c3` | 全セッション調査・統合履歴・残課題一覧（本文書を含む 5 ファイル） |

`main` は本記録の対象時点（#126 マージ後）で `e69f20c3`（origin と一致）。調査開始時点は `38075a7a`。以降の記録更新は PR 単位で本節に追記し、SHA は各時点の値として扱う。

## 2. ビルド・テスト結果

| 検証 | 結果 | 根拠 |
|---|---|---|
| Release ビルド（Harness） | rc=0（588 s） | R-13 修正時 |
| Release ビルド（アプリ全体） | rc=0（1,162 s） | 同上 |
| `ScopedEpochReaderStructuralTests` | **PASS 15/15**（新規 4: guard 順序・fail-closed） | 同上 |
| `AudioEngineHarness` | rc=0・297.12 s・100% passed | 同上 |
| ASan / TSan | PR #124 で success（14 pass / 0 fail） | CI |
| R-101（キャッシュヒット） | run `41751003210`：51.4 分・5/5 PASS・復元 50 秒 | CI |
| R-101（キャッシュミス） | run `41736861395`：導入 9.35 分・918 MB 保存 | CI |
| R-102 診断 | run `41665198158`：`1/1 Test #49 AudioEngineHarness Passed 319.11 s` | CI |
| metric v2 | 24/24 有効・silence 100.50% / tone1k 99.89%・PASS | #119 |
| ブランチ整理 | 13 本＋リモート 49 本を削除、タグ 42 本で保存 | #122 |
| PR #126 の CI | **12 pass / 0 fail / 0 pending**（ASan・TSan・Release・CodeQL・isr-verify ほか） | CI |

## 3. main とローカルの同一性

| 確認 | 結果 |
|---|---|
| `main` / `origin/main` | `e69f20c3` / `e69f20c3`（差分 0、#126 マージ後） |
| ローカル HEAD | `main` をチェックアウト |
| 作業ツリー | 未コミット・未追跡 **0 件**、worktree **1 本** |
| ConvoPeq.md | 6,482,434 B / sha256 `a3784c44…`、`--check` で **FRESH**（`NEWER_SRC_COUNT: 0`） |

## 4. ブランチ・worktree の状態

| 区分 | 状態 |
|---|---|
| ローカルブランチ | `main`、`measure/u5-20261102`（U5ATTR 保持） |
| リモートブランチ | `main` のみ |
| worktree | 1 本（`C:\VSC_Project\ConvoPeq`） |
| アーカイブタグ | 42 本（削除済みブランチの SHA を保持） |

## 5. 配信（Dropbox）

`/ConvoPeq-stage3-20261026/investigation-20261102/` に ConvoPeq.md（LF 6,482,434 B）・分割 4 パート・`SHA256SUMS.txt`・R-13 記録・台帳・最終報告を配信し、ダウンロード照合で一致を確認（`concat == full: True`）。

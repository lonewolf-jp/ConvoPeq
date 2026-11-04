# 全セッション成果の調査報告書（2026-11-04）

Owner 指示（ConvoPeq 全開発成果の統合・作業環境一本化・残課題整理）の Phase A の記録。

## 1. 調査方法

| 手段 | 内容 |
|---|---|
| セッション列挙 | OpenChamber の `session.list`（全 14 セッション） |
| Git 状態 | `git rev-parse` / `git status` / `git branch` / `git ls-remote` / `git worktree list` / `git log` |
| 成果の照合 | main のマージ履歴（PR 番号付き）、アーカイブタグ、台帳・記録文書 |
| 未コミット成果 | 作業ツリー、worktree、ローカルブランチ、リモートブランチの全数を確認 |

調査時点の `main` は `38075a7a`（origin と一致、clean、worktree 1 本）。

## 2. セッション一覧（14）

| # | セッション | 種別 | リポジトリ成果 | 根拠 |
|---|---|---|---|---|
| 1 | **main-develop** | ConvoPeq 開発（本セッション） | PR #105〜#125、ブランチ整理、測定、記録 | main のマージ履歴、台帳 |
| 2 | A12 blind undetermined items | Notion 監査（A12） | なし | タイトル・成果物なし |
| 3 | A12 blind gate judgment | Notion 監査（A12） | なし | 同上 |
| 4 | A12 blind arithmetic residual | Notion 監査（A12） | なし | 同上 |
| 5 | A12 blind provenance series | Notion 監査（A12） | なし | 同上 |
| 6 | A12 blind references | Notion 監査（A12） | なし | 同上 |
| 7 | A12 blind provenance series（2） | Notion 監査（A12） | なし | 同上 |
| 8 | A12 blind undetermined items（2） | Notion 監査（A12） | なし | 同上 |
| 9 | A12 blind arithmetic residual（2） | Notion 監査（A12） | なし | 同上 |
| 10 | A12 blind gate judgment（2） | Notion 監査（A12） | なし | 同上 |
| 11 | A12 blind references（2） | Notion 監査（A12） | なし | 同上 |
| 12 | format test | 出力形式の検証 | なし | タイトル・成果物なし |
| 13 | format test 2 | 出力形式の検証 | なし | 同上 |
| 14 | Exact reply instruction | 応答指示の検証 | なし | 同上 |

**判断根拠**: リポジトリに痕跡を残す作業は worktree・ブランチ・コミット・未コミット変更のいずれかに現れる。調査時点で worktree は 1 本（本セッションの作業領域）、ローカルブランチは `main` と `measure/u5-20261102`、リモートは `main` のみ、未コミット 0 件であり、A12・format・exact-reply の各セッションに対応するリポジトリ成果は存在しない。これらのセッションの成果物は Notion 側の文書であり、本リポジトリの対象外である。

### 2.1 他セッションの作業痕跡（読み取り調査）

13 セッションの `directory` を問い合わせたところ、すべてが本セッションと同じ作業領域 `C:\Users\user\.local\share\opencode\worktree\f34893\j195-notion-report` を指していた。A12 系は監査用の subagent セッションで、同じ領域の `.openchamber\phase-d-evidence\` 配下の Markdown を読み書きしている。

| 確認 | 結果 |
|---|---|
| セッションのメッセージ走査（13 セッション、計 38 メッセージ） | 参照パスに `ConvoPeq\src`・`CMakeLists.txt`・`build.bat` は **0 件**。参照は `.openchamber\phase-d-evidence\*.md` と `Temp\opencode\a12_bodies\*.txt` のみ |
| 現作業領域の `.openchamber` | **存在しない**（`doc`・`evidence` のみ） |
| アーカイブ側（`Temp\opencode\worktree\f34893\j195-notion-report\.openchamber`） | ディレクトリ構造のみでファイル 0 件 |
| `Temp\opencode\a12_bodies` | ディレクトリのみでファイル 0 件 |
| A11-3／A12 の監査 Markdown | ディスク上に残存せず。内容はセッションのメッセージ（transcript）にのみ残る |

本作業では削除操作を一切行っていない。したがって「他セッションの作業内容を破棄しない」という制約に抵触しない。A12 系の成果は Notion 監査の証跡であり、ConvoPeq のソースコード・ブランチ・コミットには影響していない（コードパス参照 0 件、リポジトリの未コミット変更 0 件）。

### 2.2 他の複製の確認

「唯一の最新ソースコード置き場」を担保するため、ディスク上の ConvoPeq 複製を走査した。

| 場所 | 状態 |
|---|---|
| `C:\VSC_Project\ConvoPeq` | 正本。`main` をチェックアウト、未コミット 0 件 |
| `C:\VSC_Project\ConvoPeq-preserved` | git リポジトリではない。`A7-8-bright-falcon-doc-20261022-232616` の文書スナップショットのみ |
| `Temp\opencode\gen2〜gen8\ConvoPeq`、`proposal-stage`、`reconstruct-check` ほか | `.git` は `HEAD`・`config` を欠く残骸（`objects` 等のみ）で `rev-parse` が失敗する。独立した未マージ履歴を持たない |

したがって、正本以外に未統合のコミットや未コミット変更を抱える複製は存在しない。

## 3. 成果一覧（main 統合済み）

| ID | セッション | 内容 | 状態 | Git 状態 | 依存 | 必要な対応 | 根拠 |
|---|---|---|---|---|---|---|---|
| B-01 | main-develop | R-101: h2r1 測定 workflow を dispatch-only で追加 | 完成 | 統合済み | なし | なし | #105 `7318e44f` |
| B-02 | main-develop | R-102: R-49 診断を Windows ビルドへ移植 | 完成 | 統合済み | B-01 | なし | #106 `a83d10ba` |
| B-03 | main-develop | STG8 mid-run clear テストの駆動点修正 | 完成 | 統合済み | なし | なし | #107 `cf835829` |
| B-04 | main-develop | R-107 U4: 非 RT ロック計測（既定 OFF） | 完成 | 統合済み | なし | なし | #108 `7dfb21d4` |
| B-05 | main-develop | 記録・生成物の更新（R-101 測定、U4 検証） | 完成 | 統合済み | B-01,B-04 | なし | #109〜#113 |
| B-06 | main-develop | R-113/R-111/oneAPI 堅牢化/U3 プローブ | 完成 | 統合済み | B-01,B-02 | なし | #114 `e89c66fe` |
| B-07 | main-develop | ConvoPeq.md 再生成・記録確定 | 完成 | 統合済み | B-06 | なし | #115 `bfaa9c9e` |
| B-08 | main-develop | R-106 指標再定義の技術判断案（metric v2） | 完成 | 統合済み | なし | なし | #116 `3fa7090f` |
| B-09 | main-develop | 台帳の U3 実装済み化・ブランチ内訳 | 完成 | 統合済み | B-06 | なし | #117 `6afdee60` |
| B-10 | main-develop | metric v2 の事前登録と測定ドライバ | 完成 | 統合済み | B-08 | なし | #118 `5d88dc95` |
| B-11 | main-develop | metric v2 新規 24 実行 → **PASS**（既定 OFF 維持） | 完成 | 統合済み | B-10 | なし | #119 `ecfdb244` |
| B-12 | main-develop | R-101/R-102 の現行 main 再確認（5/5・1/1） | 完成 | 統合済み | B-01,B-02 | なし | #120 `7a34fa40` |
| B-13 | main-develop | R-101 oneAPI キャッシュ（918 MB・復元 50 秒） | 完成 | 統合済み | B-01 | なし | #121 `8561091d` |
| B-14 | main-develop | R-108 ブランチ整理・最終監査・台帳 §11 | 完成 | 統合済み | 全体 | なし | #122 `7392ca52` |
| B-15 | main-develop | 監査記録の記述修正 | 完成 | 統合済み | B-14 | なし | #123 `6eeb1d42` |
| B-16 | main-develop | R-13 Reader guard 順序修正・構造テスト追加 | 完成 | 統合済み | なし | なし | #124 `28cd3d1e` |
| B-17 | main-develop | ConvoPeq.md 再生成（R-13 反映） | 完成 | 統合済み | B-16 | なし | #125 `38075a7a` |
| B-18 | main-develop | 旧ブランチ 62 本の整理（アーカイブタグ 42 本） | 完成 | 該当なし（履歴操作） | B-14 | なし | `R108-BRANCH-CLEANUP-20261104.md` |
| B-19 | main-develop | U5 測定専用の `[U5ATTR]` 計装 | 保持（意図的未統合） | ブランチ `measure/u5-20261102` | なし | 保持 | 台帳 §11.4 |

## 4. 未統合・未コミットの成果

| 区分 | 結果 |
|---|---|
| 未コミット変更 | **0 件** |
| 未マージの完成済み成果 | **なし**（B-01〜B-17 はすべて main に統合済み） |
| 意図的に未統合 | B-19（U5ATTR、測定専用）のみ |
| 未完成・放置 | なし（詳細は `RESIDUAL-ISSUES-20261104.md`） |

## 5. 結論

- 全 14 セッションのうち、リポジトリ成果を持つのは `main-develop` のみで、その成果は **PR #105〜#125 として main に統合済み**。
- 他 13 セッションは Notion 監査・形式検証・応答検証であり、リポジトリ成果物を持たない。
- 未コミット変更・未マージの完成済み成果は存在しない。

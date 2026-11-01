# 全セッション統合台帳（2026-11-01）

Owner 指示「全 OpenChamber Desktop セッションの変更・未完了事項を把握し、統合台帳に整理する」
（第 1 段階）の成果物。**読み取りのみで作成し、製品コードは変更していない。**

- 調査基準: `C:\VSC_Project\ConvoPeq`（`HEAD` = `origin/main` = `6bb7c8d5`）
- 調査日時: 2026-11-01
- 対象: ローカル全ブランチ 76、worktree 全 8、未統合ブランチ 19（祖先比較）＋ origin のみ 10

## 1. 調査方法（再現可能な形で記載）

| 判定 | コマンド |
|---|---|
| 未統合ブランチ（祖先比較） | `git branch --merged main` との差集合 |
| 未統合ブランチ（remote のみ） | `git branch -r --no-merged main` |
| **main へ未取込のコミット** | **`git cherry main <branch>`**（パッチ同一性で判定。squash 統合も検出） |
| worktree の未コミット変更 | `git -C <wt> status --porcelain` |
| 変更ファイルが main と同一か | `git -C <wt> hash-object` と `git rev-parse main:<path>` の比較 |
| 内容の新旧判定 | 実ファイル mtime と `git log -1 main -- <path>` の日時比較＋`git -C <wt> diff main --numstat` |
| 同一作業の別 SHA 取込 | `git log main --grep=<キーワード>` と main のソース実測 |

> 祖先比較だけでは「squash / cherry-pick で既に取り込まれた作業」を未統合と誤判定する。
> そのため `git cherry` と main 側のキーワード・ソース実測を併用した。

## 2. リポジトリ状態

| 項目 | 値 |
|---|---|
| ブランチ | `main` |
| `HEAD` = `origin/main` | `6bb7c8d5` |
| 追跡差分 | 0 |
| 未追跡 | 47（既存 18 ＋ 本作業 29。削除・整理はしていない） |
| stash | なし |
| open PR | 0 件（`gh pr list`） |

## 3. worktree（セッション）別の状態

| worktree | HEAD | 未コミット | 未追跡 | 判定 |
|---|---|---|---|---|
| `C:\VSC_Project\ConvoPeq`（main） | `6bb7c8d5` | 0 | 47 | **作業の唯一の場所** |
| `j195-notion-report` | `bda43034` | 5 ファイル | 763 | 変更は main より**古い**（下記 §5） |
| `j237-baseline` | `d79b8be4` | 8 ファイル | 0 | **1 件に未統合の有効変更**（下記 §5） |
| `j237-baseline-clean` | `d79b8be4` | 0 | 0 | クリーン |
| `bright-falcon` | `d79b8be4` | 3 ファイル | 243 | 変更は main に別 SHA で取込済み |
| `crisp-zebra` | `bda43034` | 0 | 0 | クリーン（修正は main に取込済み） |
| `wt-g3` | `812abcc8` | 0 | 0 | クリーン（取込済み） |
| `wt-r16` | `3f3225d7` | 0 | 0 | クリーン（取込済み） |

## 4. ブランチ棚卸し（19 件）

`git cherry main <branch>` の「+」＝ main に取り込まれていないコミット数。

| ブランチ | + | 判定 | 根拠 |
|---|---|---|---|
| `bright-falcon` | 2 | **取込済み（別 SHA）** | main に `8d4a776f`（Phase 1 lastUsedTime）、`aa65170d`（Phase 2 validator） |
| `crisp-zebra` | 1 | **取込済み（別 SHA）** | main に `08f8f19e`（`cherry-picked from bda43034`） |
| `main-pre-sync-20261021` | 1 | 同上 | 同一コミット `bda43034` |
| `integration/session-consolidation` | 0 | 取込済み（パッチ同一） | — |
| `fix/r51-f1b-pin-protection-20261030` | 0 | 取込済み | PR で統合済み |
| `review/r51-merge-rehearsal-20261030` | 0 | 取込済み | リハーサルのみ |
| `feat/b2-stage1-t1-20261028` | 0 | 取込済み | — |
| `fix/r52-f1b-defer-observation-20261029` | 0 | 取込済み | — |
| `chore/convopeqmd-5057968c-20261030` | 0 | 取込済み | — |
| `work/g3-snapshot-ignore-20261026` | 0 | 取込済み | main に `63a198d9` 系 |
| `work/r16-i4-sync-20261026` | 0 | 取込済み | main に `0266fef4` 系 |
| `ci/r51-two-engine-on-pr82` | 1 | **陳腐化（測定専用）** | main の恒久テスト `b15c92ed`（PR #95）で置換 |
| `ci/r51-two-engine-repro-20261030` | 1 | 同上 | 同上 |
| `verify/r51-two-engine-on-merged-main-20261030` | 1 | 同上（証拠のみ） | 同上 |
| `ci/r51-diag-only-20261030` | 1 | 同上（計測用計装） | 同上 |
| `ci/h2r1-release-measurement-20261030` | 3 | **CI 定義変更 → 保留** | Owner 承認外（CI 定義変更は今回未承認） |
| `ci/r49-selected-tests-probe-20261029` | 2 | **CI 定義変更 → 保留** | 同上（main に r49 修正 `201bfbab` / `751ef379` あり） |
| `ci/r50-asan-vbcable-20261029` | 2 | **取込済み／CI 領域** | `b7311107` は main の `5057968c` と同一メッセージ。`fcf383a9` は R-50 系クラスタ（`ff483c2c`）に包含 |
| `chore/convoPeqmd-023ae16f-20261029` | 1 | **陳腐化** | `ConvoPeq.md` 再生成。main の最新（`6bb7c8d5`）が後続 |

**未統合ユニークコミット（重複除く）: 15 件。うち機能実装として未取込のものは 0 件。**

## 5. worktree の未コミット変更（実測）

### 5.1 判定結果

| worktree / ファイル | mtime | main 最終更新 | 差分 | 判定 |
|---|---|---|---|---|
| `j195` AGENTS.md | 10-30 00:13 | 10-28 11:48 | +17/-0 | **セッション固有の運用メモ（semble 復旧記録）**。製品コードではない |
| `j195` CMakeLists.txt | 10-21 20:59 | 10-30 22:07 | +0/-41 | 陳腐化（main が後続） |
| `j195` ISRDSPHandle.cpp | 10-24 08:38 | 10-30 13:59 | +19/-352 | 陳腐化 |
| `j195` RuntimePublicationOrchestrator.cpp | 10-24 08:50 | 10-26 17:35 | +16/-73 | 陳腐化 |
| `j195` opencode.json | 10-22 20:48 | 10-24 02:24 | 同一 | 差分なし |
| `j237` AudioEngine.h ほか 6 件 | 10-14〜10-16 | 10-20〜10-30 | 大半が削除方向 | 陳腐化 |
| **`j237` ISRShutdown.h** | 10-16 21:15 | 10-26 01:47 | **+4/-0** | **未統合の有効変更（下記 5.2）** |
| `bright-falcon` 3 件 | 10-14 | 10-28〜11-01 | 大半が削除方向 | 陳腐化（main に別 SHA で取込済み） |

### 5.2 統合候補（唯一の有効な未コミット変更）

`j237-baseline` の `src/audioengine/ISRShutdown.h` に、テスト専用 friend を
`#if defined(CONVOPEQ_UNIT_TESTS)` で囲う変更がある（前方宣言・friend 宣言の 3 箇所）。

- main には**このガードが無い**（`src/audioengine/ISRShutdown.h:21` / `:378`）。
- `AudioEngine.h:3985` の既存コメントは「`#if defined(CONVOPEQ_UNIT_TESTS)` が**データメンバ**を
  含んでいたため ODR 問題になった」と記録している。今回の対象は**宣言のみ**でデータメンバを含まない。
- 効果: 本番ビルド（`CONVOPEQ_UNIT_TESTS` 未定義）でテスト用 friend を露出しない。
- リスク: 低（レイアウト不変）。ただし**ビルド検証が必要**。

### 5.3 未追跡ファイル

`j195` 763 件、`bright-falcon` 243 件。ソース系は `bright-falcon` の
`scripts/aggregate_preflight.ps1` / `verify_canonical_source.ps1` / `verify_git_state.ps1` の 3 件のみで、
残りは証拠・ログ・スナップショット。**削除・移動はしていない**（Owner 指示どおり保全）。

## 6. 統合方針（提案）

| 優先 | 対象 | 方針 | 前提 |
|---|---|---|---|
| 1 | B-2 実装 | 承認済み設計どおり `main` から作業ブランチを切って実装 | 本台帳の次段階 |
| 2 | A2→A3-a 実装 | B-2 の検証完了後に着手 | B-2 の PASS |
| 3 | `j237` ISRShutdown.h のガード | ビルド検証のうえ小規模コミットで統合 | build / test の実行可否 |
| 4 | `j195` AGENTS.md の運用メモ | 文書として main の AGENTS.md へ反映するか判断 | 製品コードではない |
| 5 | CI 系ブランチ 5 本 | **保留**（CI 定義変更は今回の承認外） | 別途承認 |
| 6 | 陳腐化ブランチ | 削除しない（記録として保持）。必要なら後日まとめて整理 | Owner 判断 |

**削除・強制 push・履歴改変は行わない。** 未追跡ファイルも無断で削除しない。

## 7. 次段階への接続

- 台帳上、main に取り込まれていない**機能実装は無い**。
  したがって「他セッションの未完成機能を main へ統合する」作業は、現時点では
  §5.2 の 1 件と文書 1 件に限られる。
- 実装の主対象は **B-2**（設計・実装前調査は完了済み）と **A2→A3-a**。
- B-2 の実装順序は Owner 指示の 8 段階に従う。

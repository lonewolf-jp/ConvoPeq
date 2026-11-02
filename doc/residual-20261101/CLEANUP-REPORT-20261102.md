# 整理作業の報告（2026-11-02）

Owner 承認（ASan 記録 / 陳腐化 12 本の削除 / ci/r50・R-51 4 本の整理 / ci/r49・h2r1 保持 /
未追跡 12 件の追跡化 / `AGENTS.md` 統合）に基づく実施記録。

- 作業場所: `C:\VSC_Project\ConvoPeq`（唯一）
- セッション: `main-develop`
- 実施前の `HEAD`: `20aa61f2`（`origin/main` と一致）
- 本書を追加する直前の `HEAD`: `23d006a7`（`origin/main` と一致。本書の commit で進む）

> SHA は「いつ時点か」を表すために置いている。台帳 §9 と同じく、文書自身の commit で
> 先頭が変わるため、最終状態は `git rev-parse HEAD` と `git rev-parse origin/main` の
> 一致で確認する。

## 1. 削除・保持したブランチ

### 1.1 削除したブランチ（16 本、tip はアーカイブタグで保全）

| # | ブランチ | ローカル | リモート | tip |
|---|---|---|---|---|
| 1 | `integration/session-consolidation` | 削除 | 削除 | `e8aa1f6f` |
| 2 | `fix/r51-f1b-pin-protection-20261030` | 削除 | 削除 | `ed11a6c3` |
| 3 | `review/r51-merge-rehearsal-20261030` | 削除 | （無し） | `e9613d67` |
| 4 | `feat/b2-stage1-t1-20261028` | 削除 | 削除 | `ec80b9d3` |
| 5 | `fix/r52-f1b-defer-observation-20261029` | 削除 | 削除 | `7114440d` |
| 6 | `chore/convoPeqmd-023ae16f-20261029` | 削除 | 削除 | `fd0c5153` |
| 7 | `chore/convopeqmd-5057968c-20261030` | 削除 | 削除 | `e7237372` |
| 8 | `main-pre-sync-20261021` | 削除 | （無し） | `bda43034` |
| 9 | `ci/r51-two-engine-on-pr82` | 削除 | 削除 | `c56f571d` |
| 10 | `ci/r51-two-engine-repro-20261030` | 削除 | 削除 | `857372eb` |
| 11 | `verify/r51-two-engine-on-merged-main-20261030` | 削除 | 削除 | `3327b484` |
| 12 | `ci/r51-diag-only-20261030` | 削除 | 削除 | `481db70d` |
| 13 | `bright-falcon` | **残存**（worktree がチェックアウト中） | 削除 | `d79b8be4` |
| 14 | `crisp-zebra` | **残存**（同上） | 削除 | `bda43034` |
| 15 | `work/g3-snapshot-ignore-20261026` | **残存**（同上） | 削除 | `812abcc8` |
| 16 | `work/r16-i4-sync-20261026` | **残存**（同上） | 削除 | `3f3225d7` |

- **ローカル削除 12 本 / リモート削除 14 本**。
- 残存 4 本は **他の worktree がチェックアウトしているため削除不可**（他セッションは読み取り専用の
  ため、worktree の detach も行っていない）。**tip はアーカイブタグで保全済み**。
- アーカイブタグ 16 個（`archive/<branch>-<short>`）。削除で到達不能になるコミットは無い。

### 1.2 保持したブランチ

| ブランチ | 理由 |
|---|---|
| `ci/r49-selected-tests-probe-20261029` | Owner 承認により保持（旧ベース由来で 2 ワークフローを削除する差分を含む） |
| `ci/h2r1-release-measurement-20261030` | Owner 承認により保持（新規 workflow の恒久追加は CI 定義変更） |
| `ci/r50-asan-vbcable-20261029` | **削除を保留**。main に取込済みの内容のほかに **policy 1 行の未解決差分**が残るため |

## 2. Git 管理に追加した資料（12 件）

| # | ファイル | サイズ |
|---|---|---|
| 1 | `doc/work113/REPORT-20261101-design-phase.md` | 8,967 B |
| 2 | `doc/work113/REPORT-20261101-abc-investigation.md` | 23,129 B |
| 3 | `doc/work113/REPORT-20261101-attach-plan.md` | 6,183 B |
| 4 | `doc/work113/REPORT-20261101-attach-contract.md` | 7,363 B |
| 5 | `doc/work113/REPORT-20261101-lifetime-proof.md` | 9,497 B |
| 6 | `doc/work113/P1-5-IR-P2_STG-11-D13_FRESH-DISCOVERY_20261001.md` | 6,691 B |
| 7 | `doc/work113/P1-5-IR-P2_STG-11-D14_FRESH-DISCOVERY_20261001.md` | 6,422 B |
| 8 | `doc/work113/P1-5-IR-P2_STG-11-D15_FRESH-DISCOVERY_20261001.md` | 5,620 B |
| 9 | `doc/work113/P1-5-IR-P2_STG-11-D16_FRESH-DISCOVERY_20261001.md` | 5,691 B |
| 10 | `doc/work113/P1-5-IR-P2_STG-11-D17_FRESH-DISCOVERY_20261001.md` | 4,844 B |
| 11 | `doc/work113/P1-5-IR-P2_STG-11-D19_FRESH-DISCOVERY_20261001.md` | 21,078 B |
| 12 | `.memories/repo/tool-versions-2026-09-26.md` | 2,193 B |

追加前チェック: 機密パターン（`ghp_` / `sk-` / 秘密鍵 / AWS / Slack / Bearer 等）**一致 0**、
既存追跡ファイルとの同名衝突**なし**、格納先は既存の `doc/work113/` 規則に一致。

## 3. 保管・削除判断を保留したファイル（14 件）

| 分類 | 件数 | 内容 | 現状 |
|---|---|---|---|
| 旧分割ステージング | 5 | `.openchamber/convopeq-md-split-20261030/` の `part01〜04` ＋ `SHA256SUMS.txt` | 保持 |
| 旧分割の検証キット | 2 | `doc/ConvoPeq-f8aab49b-split/{README.md,SHA256SUMS.txt}` | 保持 |
| 設定バックアップ | 6 | `opencode.json.bak-py314-20261022-*.bak` | 保持 |
| 生成物 | 1 | `vc140.pdb`（69,632 B） | 保持 |

いずれも**削除・移動・上書きはしていない**。不要性の確認後に推奨案を提出する。
→ **提出済み**: `DISPOSAL-RECOMMENDATION-20261102.md`（14 件／6,537,003 B の実測と推奨）。

## 4. `AGENTS.md` の変更内容

新規節を作らず、既存の `## Python 3.14 → 3.15 移行（2026-10-20・MCP 全停止の復旧）` の
「復旧の手順」へ **17 行**を統合した。

- 6 項の末尾に `※ 2026-10-30 訂正: semble は例外だった` を追加
- 7 項として「semble の異常停止（2026-10-30 復旧）」を追加
  （`pyvenv.cfg` が削除済み `C:\Python314` を指す件、`uv tool install "semble[mcp]==0.6.0" --python 3.13 --force`、
  直接起動への変更、`HF_HUB_OFFLINE=1`、`mcp-probe` の結果、複数ディレクトリ同時起動の不安定さ）
- 行数 544 → **561**、`pyvenv.cfg` / `HF_HUB_OFFLINE` の記述は各 1 箇所（重複なし）
- 作業場所・セッション・main 同期・承認境界・禁止事項の記述は**変更していない**

## 5. commit・push の結果

| コミット | 内容 |
|---|---|
| `b20506fa` | 資料 12 件の追跡化 |
| `23d006a7` | `AGENTS.md` へ semble 記述を統合 |
| `c093543b` | 整理報告（本書） |
| `53b15a13` | 本書と台帳 §9 の `HEAD` 固定をやめ、台帳へ §9 を新設 |
| `5dfe1f81` | 保留 14 件の処分・保管 推奨案（`DISPOSAL-RECOMMENDATION-20261102.md`） |
| `655187f8` | `ci/h2r1` の判断案（`CI-H2R1-DISPOSITION-20261102.md`） |
| push | `20aa61f2` → `655187f8`（`origin/main`） |

## 6. GitHub main とローカル main の一致

- `HEAD` = `origin/main`（`git rev-parse` で一致、追跡差分 **0**）
- **`git status` だけでは足りないため、全追跡ファイルの内容を実測した**:
  `git hash-object` の結果と `HEAD` の blob SHA-1 を 7,042 ファイルすべてで照合し、
  **内容が異なるファイル 0 件・作業ツリーから欠落 0 件**。`assume-unchanged` /
  `skip-worktree` の登録も 0 件（`git ls-files -v` で確認）

## 7. Dropbox 正規ソースとの一致

| 対象 | サイズ | SHA-256 |
|---|---|---|
| git blob `HEAD:ConvoPeq.md` | 6,451,711 B | `c12a532e70770ab40430d34052af87407291e0a12c2a9e68f10229fd5760e8a9` |
| Dropbox `/ConvoPeq.md` | 6,451,711 B | 同上 |
| 判定 | — | **identical = True** |

今回は生成対象（`src` / `CMakeLists.txt` / `build.bat`）に変更が無いため、`ConvoPeq.md` の
再生成は行っていない（`git diff 0dfa39e3..HEAD -- src CMakeLists.txt build.bat` = **0 件**）。

**作業ツリー上の `ConvoPeq.md` は CRLF 表現**（6,597,645 B / CRLF 145,934 個）で、git blob や
Dropbox の LF 正規形（6,451,711 B）とは生バイトが異なる。ただし
`*.md text eol=lf`（`.gitattributes:16`）により git の clean 変換後は同一 blob になり、
LF 正規化後の SHA-256 も `c12a532e…` と一致する。配信スクリプト（`post_merge_sync.py`）は
ディスクの生バイトを読んで LF に正規化してから送る設計（同スクリプト L78〜L80）なので、
**三者は内容として一致している**。生バイト比較で不一致と判定しないよう注意すること。

## 8. CI workflow の変更

`git diff --name-only 6bb7c8d5..HEAD -- .github/workflows` = **0 件**。
policy・検査スクリプトも今回の作業では変更していない（allowlist 1 行の同期は PR #103 に含まれ、
既に main にある）。

## 9. 統合台帳

`doc/residual-20261101/SESSION-CONSOLIDATION-LEDGER-20261101.md` を更新した。

- §2 に「本節は調査時点のスナップショット。現在の状態は §9 を参照」の注記を追加
- **§9「整理の実施結果（2026-11-02）」を新設**し、現在のリポジトリ状態、削除 16 本の一覧、
  保持 3 本、追跡化 12 件、保留 14 件、`AGENTS.md` 統合、§8 の 9 項目一覧との対応を記載
- §8 の末尾に「#3・#4・#9・#13・#14 は §9 で決着済み」の参照を追加

これにより台帳は「調査時の状態（§1〜§8）」と「整理後の状態（§9）」を併記する形になった。

## 10. 新たに残った Owner 判断事項

| # | 項目 | 必要な判断 |
|---|---|---|
| 1 | ローカル 4 本（`bright-falcon` / `crisp-zebra` / `work/g3` / `work/r16`） | これらをチェックアウトしている worktree を閉じてよいか（閉じれば削除可能）。tip はタグで保全済み |
| 2 | 未追跡 14 件 | 削除可否。根拠と推奨は `DISPOSAL-RECOMMENDATION-20261102.md`（旧分割 5＋検証キット 2 は Dropbox にハッシュ検証済みの同一コピーあり、`.bak` 6 は 3 種のみで全て旧内容、`vc140.pdb` は再生成可能） |
| 3 | `ci/r50` | policy 1 行の差分をどう扱うか（削除するか、差分を破棄して削除するか） |
| 4 | `ci/h2r1` | 恒久化・削除・現状維持の選択。判断材料は `CI-H2R1-DISPOSITION-20261102.md`（測定は 5/5 PASS で完了、トリガは main/PR に発火しない、実行ログは失効するため計測値の記録を先に行うべき） |
| 5 | `AGENTS.md` | 統合結果の確認（矛盾・重複が無いことの承認） |
| 6 | アーカイブタグ 16 個 | **ローカルのみ**（リモート 0）。リモートへ push して保全を恒久化するか。push しない場合、削除した 16 本のうち PR を持たないブランチの tip はローカルにしか残らない |
| 7 | `opencode.json:121` の uvx パス | `C:/Python314/Scripts/uvx.exe` を Python315 ネイティブの uvx へ差し替えるか（現状は動作中。故障はしていない） |
| 8 | `.gitignore` 1 行目 | 先頭が `` ``` ``（Markdown フェンスの貼り付け痕）。削除するか（実害なし） |

U5 = FAIL / 既定 OFF、U3/U4 = HOLD、CI 定義（workflow / policy / 検査スクリプト）の変更、
リリース・外部公開・既定 OFF 有効化は**未実施のまま**。

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

> 本節は**調査時点（2026-11-02 前半）のスナップショット**である。
> その後の整理作業で `HEAD`・未追跡件数・ブランチ構成が変わった。
> **現在の状態は §9 を参照**（§9 が本台帳の最新状態）。

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

## 8. 全セッション統合台帳（Owner 指定 9 項目・2026-11-02 最終）

Owner 指示 2 の 9 項目（セッション名 / 課題・機能 / 実装状況 / 変更対象ファイル / テスト状況 /
main 統合状態 / 未完了の理由 / 依存関係 / 次に行うべき作業）で一覧化する。

| # | セッション（worktree / ブランチ） | 課題・機能 | 実装状況 | 変更対象ファイル | テスト状況 | main 統合 | 未完了の理由 | 依存関係 | 次の作業 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `main-develop` / `fix/b2-diag-sink-20261101` | B-2 診断 sink の所有権 | **完了** | `AudioEngine.h`、Threading / Init / CtorDtor / DSPCoreLifecycle / DSPCoreFloat・Double・IO、`ConvolverProcessor.h`・Runtime、`B2DiagSinkTests.cpp` | Release・Debug・ASan で専用テスト PASS、既定スイート PASS | **統合済み**（`1f258686`） | — | A2→A3-a と同一ブランチ（変更単位は分離） | — |
| 2 | 同上 | A2→A3-a（assert 判定位置・deferred マーク） | **完了** | `ISRDSPHandle.h`・cpp、`A2A3aQuarantineTests.cpp` | V1/V2 とも PASS（両構成） | **統合済み**（`1f258686`） | — | B-2 と同一ブランチ | — |
| 3 | `j237-baseline` / `fix/isrshutdown-test-seam-guard-20261102` | テスト専用 friend の本番露出 | **完了** | `ISRShutdown.h`（+4 行） | Release ビルド＋スイート PASS、CI の本番構成でビルド | **統合済み**（`0dfa39e3`） | ASan の CI のみ pending（他 12 PASS） | — | ASan 完了確認のみ |
| 4 | `j195-notion-report` | `AGENTS.md` の運用メモ 17 行 | 未着手 | `AGENTS.md` | 対象外（文書） | **未統合** | 製品コードではないため反映可否が未決 | — | 文書として採用するか判断 |
| 5 | `j195-notion-report` | 未追跡 763 件（証拠・ログ・スナップショット） | 保留 | —（未追跡） | — | 未統合 | 由来の調査が未了 | — | 内容と由来を調査（削除しない） |
| 6 | `bright-falcon` | Phase 1/2 persistent cache | **main に別 SHA で取込済み** | `MixedPhasePersistentCache.*` ほか | main 側で検証済み | 取込済み（`8d4a776f` / `aa65170d`） | — | — | 不要（記録のみ） |
| 7 | `crisp-zebra` / `main-pre-sync-20261021` | STG-11-D18 crossfade 修正 | **main に cherry-pick 済み** | audioengine 6 ファイル | main 側で検証済み | 取込済み（`08f8f19e`） | — | — | 不要（記録のみ） |
| 8 | `bright-falcon` | 未追跡 243 件（`scripts/*.ps1` 3 件を含む） | 保留 | `scripts/aggregate_preflight.ps1` ほか | — | 未統合 | 由来の調査が未了 | — | 内容と由来を調査（削除しない） |
| 9 | `ci/r51-two-engine-on-pr82` / `-repro` / `verify` / `ci/r51-diag-only` | R-51 二重起動の測定・計装 | 陳腐化 | テスト・計装 | — | 未統合 | main の恒久テスト `b15c92ed`（PR #95）で置換済み | R-51 恒久テスト | 不要（記録のみ） |
| 10 | `ci/h2r1-release-measurement-20261030` | H-2-R1 の CI 計測 | 保留 | `.github/workflows` | — | 未統合 | **CI 定義変更に該当**（今回の承認外） | — | 別途承認 |
| 11 | `ci/r49-selected-tests-probe-20261029` | R-49 の CI プローブ | 保留 | `.github/workflows` | — | 未統合 | 同上 | — | 別途承認 |
| 12 | `ci/r50-asan-vbcable-20261029` | ASan ジョブの VB-CABLE 導入 | 取込済み／CI 領域 | `.github/workflows` | — | 一部取込（`5057968c`） | `fcf383a9` は CI 領域 | — | 別途承認 |
| 13 | `chore/convoPeqmd-023ae16f-20261029` ほか | `ConvoPeq.md` 再生成 | 陳腐化 | `ConvoPeq.md` | — | 未統合 | 後続の再生成（`3d8ea758`）が最新 | — | 不要 |
| 14 | `integration/session-consolidation` / `fix/r51-f1b-pin-protection` / `feat/b2-stage1-t1` / `fix/r52-f1b-defer-observation` / `work/g3` / `work/r16` | 各種 | 取込済み（パッチ同一） | — | — | 取込済み | — | — | 不要 |

**未統合として残るのは #3（ASan 待ちのみ）、#4、#5、#8、#10〜#12 の 6 群**であり、
このうち機能実装は **0 件**。

> この 6 群のうち **#3・#4・#9・#13・#14 は §9 で決着済み**（ASan 完了、`AGENTS.md` 統合、
> ブランチ削除）。現在の残りは §9.4 を参照。

## 9. 整理の実施結果（2026-11-02・Owner 承認分の完了）

Owner 回答案（陳腐化 12 本の削除 / `ci/r50`・R-51 4 本の整理 / `ci/r49`・`ci/h2r1` の保持 /
未追跡 12 件の追跡化 / `AGENTS.md` 統合）に基づく実施結果。**本節が最新状態である。**

### 9.1 現在のリポジトリ状態

| 項目 | 値 |
|---|---|
| ブランチ | `main` |
| `HEAD` = `origin/main` | 一致（本書を commit するたびに進むため、SHA は固定しない） |
| 整理作業のコミット | `b20506fa`（資料 12 件の追跡化）、`23d006a7`（`AGENTS.md` 統合）、`c093543b`（整理報告） |
| 追跡差分 | **0** |
| 未追跡 | **14**（削除・移動はしていない） |
| ローカルブランチ | 66 本 |
| リモートブランチ（`origin`） | 48 本 |
| アーカイブタグ | 16 個（`archive/<branch>-<short>`） |
| CI workflow の変更（`6bb7c8d5..HEAD`） | **0 件** |
| policy の変更 | allowlist 1 行のみ（`c9e74603`・既承認） |
| 検査スクリプトの変更 | **0 件** |

### 9.2 削除したブランチ（16 本）

`git cherry main <branch>` の未取込が 0 でない 8 本（`bright-falcon` / `crisp-zebra` /
`main-pre-sync-20261021` / `chore/convoPeqmd-023ae16f-20261029` と R-51 系 4 本）は、
**内容が main に別 SHA で存在**することを確認したうえで、tip をアーカイブタグに保全して削除した。
削除で到達不能になったコミットは無い。

| ブランチ | ローカル | リモート | tip |
|---|---|---|---|
| `integration/session-consolidation` | 削除 | 削除 | `e8aa1f6f` |
| `fix/r51-f1b-pin-protection-20261030` | 削除 | 削除 | `ed11a6c3` |
| `review/r51-merge-rehearsal-20261030` | 削除 | （無し） | `e9613d67` |
| `feat/b2-stage1-t1-20261028` | 削除 | 削除 | `ec80b9d3` |
| `fix/r52-f1b-defer-observation-20261029` | 削除 | 削除 | `7114440d` |
| `chore/convoPeqmd-023ae16f-20261029` | 削除 | 削除 | `fd0c5153` |
| `chore/convopeqmd-5057968c-20261030` | 削除 | 削除 | `e7237372` |
| `main-pre-sync-20261021` | 削除 | （無し） | `bda43034` |
| `ci/r51-two-engine-on-pr82` | 削除 | 削除 | `c56f571d` |
| `ci/r51-two-engine-repro-20261030` | 削除 | 削除 | `857372eb` |
| `verify/r51-two-engine-on-merged-main-20261030` | 削除 | 削除 | `3327b484` |
| `ci/r51-diag-only-20261030` | 削除 | 削除 | `481db70d` |
| `bright-falcon` | **残存**（他 worktree が使用中） | 削除 | `d79b8be4` |
| `crisp-zebra` | **残存**（同上） | 削除 | `bda43034` |
| `work/g3-snapshot-ignore-20261026` | **残存**（同上） | 削除 | `812abcc8` |
| `work/r16-i4-sync-20261026` | **残存**（同上） | 削除 | `3f3225d7` |

- ローカル削除 **12 本** / リモート削除 **14 本**。
- 残存 4 本は `git branch -D` が `used by worktree` で拒否されたもの。
  他セッションは読み取り専用のため worktree の detach は行っていない。tip はタグで保全済み。

### 9.3 保持したブランチ

| ブランチ | 理由 |
|---|---|
| `ci/r49-selected-tests-probe-20261029` | Owner 承認により保持。旧ベース由来で `sanitizer-ci.yml` と `vbcable-probe.yml` を削除する差分を含むため統合不可 |
| `ci/h2r1-release-measurement-20261030` | Owner 承認により保持。新規 workflow 235 行の恒久追加は CI 定義変更に該当 |
| `ci/r50-asan-vbcable-20261029` | **削除を保留**。workflow 差分は main に取込済み（`5057968c`）だが、policy 1 行の未解決差分が残る |

### 9.4 Git 管理へ追加した資料（12 件・`b20506fa`）

`doc/work113/REPORT-20261101-{design-phase,abc-investigation,attach-plan,attach-contract,lifetime-proof}.md`（5）、
`doc/work113/P1-5-IR-P2_STG-11-D13〜D17・D19_FRESH-DISCOVERY_20261001.md`（6）、
`.memories/repo/tool-versions-2026-09-26.md`（1）。

追加前確認: 機密パターン一致 **0**、既存追跡ファイルとの同名衝突 **なし**、
格納先は既存の `doc/work113/` 規則に一致、生成物は含まない。

### 9.5 保留した未追跡 14 件

| 分類 | 件数 | 内容 |
|---|---|---|
| 旧分割ステージング | 5 | `.openchamber/convopeq-md-split-20261030/` の `part01〜04` ＋ `SHA256SUMS.txt` |
| 旧分割の検証キット | 2 | `doc/ConvoPeq-f8aab49b-split/{README.md,SHA256SUMS.txt}` |
| 設定バックアップ | 6 | `opencode.json.bak-py314-20261022-*.bak` |
| 生成物 | 1 | `vc140.pdb`（69,632 B） |

### 9.6 `AGENTS.md` の統合（`23d006a7`）

新規節を作らず、既存の `## Python 3.14 → 3.15 移行（2026-10-20・MCP 全停止の復旧）` の
「復旧の手順」へ 17 行を統合した（6 項末尾の訂正 1 行 ＋ 7 項の semble 復旧記録）。
行数 544 → **561**、`pyvenv.cfg` / `HF_HUB_OFFLINE` の記述は各 1 箇所で重複なし。
作業場所・セッション・main 同期・承認境界・禁止事項の記述は変更していない。

### 9.7 §8 の 9 項目一覧に対する更新

| §8 の # | 旧状態 | §9 での確定 |
|---|---|---|
| 3（`j237` ISRShutdown.h） | ASan の CI のみ pending | **ASan を含め 13/13 PASS**（PR #104 `0dfa39e3`） |
| 4（`AGENTS.md` 17 行） | 未統合 | **統合済み**（`23d006a7`） |
| 9（R-51 測定 4 本） | 未統合（陳腐化） | **削除済み**（tip はタグ保全） |
| 10（`ci/h2r1`） | 保留 | **保持**（恒久化は別途承認） |
| 11（`ci/r49`） | 保留 | **保持**（統合しない） |
| 12（`ci/r50`） | 一部取込 | **削除保留**（policy 1 行が未解決） |
| 13（`ConvoPeq.md` 再生成系） | 陳腐化 | **削除済み** |
| 14（取込済み 6 本） | 取込済み | **削除済み**（`bright-falcon` / `crisp-zebra` / `work/g3` / `work/r16` のローカル ref は残存） |
| 5・8（未追跡 763 / 243 件） | 調査未了 | 本作業の対象は main worktree の 26 件。**12 件を追跡化、14 件を保留** |

### 9.8 変わっていないもの

- 製品コード・テストコード: 未変更（`git diff 0dfa39e3..HEAD -- src CMakeLists.txt build.bat` = 0 件）
- CI workflow・policy・検査スクリプト: workflow 0 件、スクリプト 0 件、policy は既承認の 1 行のみ
- **U5 = FAIL / 既定 OFF、U3/U4 = HOLD** を維持
- リリース・外部公開・既定 OFF 機能の有効化: **未実施**
- `ConvoPeq.md`: 生成対象が未変更のため再生成しない（git blob と Dropbox が一致）


## 10. 整理作業ラウンド 2（2026-11-02・Owner 指示の追補）

§9 に続く第 2 ラウンド。**worktree の閉鎖は承認待ちのため未実施**。

### 10.1 リポジトリ状態

| 項目 | 値 |
|---|---|
| ブランチ | `main`（`HEAD` = `origin/main`） |
| 追跡差分 | 0 |
| 未追跡 | **0**（保留 14 件を削除したため） |
| ローカルブランチ | 65 本 |
| リモートブランチ | 47 本 |
| アーカイブタグ | **17 個**（ローカル・リモートとも 17） |
| CI workflow / 検査スクリプトの変更 | 0 件 / 0 件 |
| policy の変更 | 既承認 1 行（`c9e74603`）のみ |

### 10.2 実施内容

| 項目 | 結果 |
|---|---|
| archive タグのリモート保全 | 16 個を「タグ名・対象 SHA・リモート未存在」で検証し push（force なし）。リモート 16/16 |
| `ci/r50` の整理 | 残差分は**承認済み allowlist を巻き戻す policy 1 行のみ**、workflow 差分 0。tip `fcf383a9` にタグを作成して push し、ブランチを削除（ローカル・リモート） |
| 保留 14 ファイル | サイズとハッシュを再検証（不一致 0）して削除。残存 0・未追跡 0 |
| `.gitignore` | 先頭の Markdown フェンス 1 行を除去し、`*.pdb` を追加（既存規則との重複なし・追跡済み `.pdb` は影響なし） |
| `opencode.json` | ddgs の uvx を Python315 版へ 1 行差し替え。変更前後でサーバー名 `ddgs-search`・6 tools が一致 |
| `ci/h2r1` | 現状維持。測定記録を `H2R1-MEASUREMENT-RECORD-20261102.md` に追跡文書化 |

### 10.3 worktree 閉鎖候補（未実施）

| worktree | 未コミット | 未追跡 | 判定 |
|---|---|---|---|
| `crisp-zebra` | 0 | 0 | 閉鎖可 |
| `wt-g3`（Temp） | 0 | 0 | 閉鎖可 |
| `wt-r16`（Temp） | 0 | 0 | 閉鎖可 |
| `bright-falcon` | 3 | 243 | パッチ保存と証跡退避の後に閉鎖可 |

4 本とも接続プロセス無し・lock 無し・stash 0。

### 10.4 残件

- worktree 4 本の閉鎖（承認待ち）と、それに伴うローカルブランチ 4 本の削除
- `ci/r50` 系の残り 3 本（ローカルのみ・PR は MERGED）の削除可否
- `ci/h2r1` の統合後再測定（CI 定義変更の承認が必要）
- 次回サーバー再起動後の ddgs 接続確認（6 tools）

### 10.5 変わっていないもの

- 製品コード・テスト: 未変更（`src` / `CMakeLists.txt` / `build.bat` の差分 0）
- CI workflow・検査スクリプト: 未変更
- U5 = FAIL / 既定 OFF、U3/U4 = HOLD: 維持
- リリース・外部公開・既定 OFF 機能の有効化: 未実施
- `ConvoPeq.md`: 生成対象が未変更のため再生成せず（git blob と Dropbox が一致）

## 11. 整理作業ラウンド 3（2026-11-02・Owner 判断の実行）

Owner 判断（APPROVED WITH CONDITIONS）に基づく実行。対象は worktree 4 本の閉鎖、
ローカルブランチ 7 本の削除、`bright-falcon` の退避、読み取り専用の接続確認に限る。

### 11.1 リポジトリ状態

| 項目 | 値 |
|---|---|
| ブランチ | `main`（`HEAD` = `origin/main`） |
| ローカルブランチ | **58 本**（65 → 61 → 58） |
| リモートブランチ | **47 本**（本ラウンドで削除なし） |
| アーカイブタグ | **17 個**（ローカル・リモートとも） |
| worktree | 4 本（main ＋ detached 3 本） |
| 追跡差分 / 未追跡 | 0 / 0 |
| CI workflow / 検査スクリプト | 変更 0 件 / 0 件 |
| policy | 既承認 1 行のみ |

### 11.2 実施内容

| 項目 | 結果 |
|---|---|
| worktree 閉鎖 | 4 本（`crisp-zebra` / `wt-g3` / `wt-r16` は通常削除、`bright-falcon` は退避・検証後に `--force`） |
| `bright-falcon` の退避 | パッチ 27,687 B・未追跡 243 件 ZIP 6,965,024 B・MANIFEST・SHA256SUMS・README。Dropbox `/archive-bright-falcon-20261102/` に配信し内容一致を確認 |
| ブランチ削除（worktree 由来 4 本） | tip と archive タグの一致 4/4 を確認して削除。リモート削除なし |
| ブランチ削除（`ci/r50` 系 3 本） | PR MERGED（#84 / #83 / #80）・main の祖先・未使用を確認して削除。リモート削除なし |
| `ci/h2r1` | 保留。workflow・記録とも未変更 |
| uvx 切替の確認 | 新経路の稼働を確認（`Python315` uvx、生成 11:47:04）。サーバー名 `ddgs-search`・tools 6・tool 名完全一致。リポジトリ設定の `Python314` 参照 0 |

### 11.3 判定指標の訂正

`ci/r50` 系 3 本の初回検証で `git diff main..<branch>` の差分数を「残差」と誤用し、中止判定にした。
main が 88〜96 コミット先行しているため当然に差分が出る指標である。判定を
`git merge-base --is-ancestor <branch> main`（完全に取り込み済み）へ訂正して再実行し、削除した。
中止判定時は 1 本も削除していない。

### 11.4 残存事項

- セッション worktree `…/j195-notion-report/opencode.json` が `Python314` を 1 箇所参照（本リポジトリ外のため未変更）
- `ci/h2r1` の統合後再測定（別の Owner 判断が必要）
- U5 = FAIL / 既定 OFF、U3/U4 = HOLD、リリース・外部公開・既定 OFF 有効化は未実施


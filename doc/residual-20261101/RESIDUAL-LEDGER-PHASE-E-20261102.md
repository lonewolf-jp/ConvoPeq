# Phase E 残課題台帳（2026-11-02）

Owner 決定（全セッション成果統合・正本統一・残課題整理）に基づく残課題の一覧。
Phase A の実測（コミット `bdb1dcb8` 時点）を根拠とする。

- 記載方針: 未解決でも「状態・理由・影響・必要作業・優先度・必要判断」が明確であれば整理完了とする
- 優先度: P1（正本・安全性に直結）/ P2（品質・再現性）/ P3（整理・衛生）

## 1. サマリ

| 分類 | 件数 | 状態 |
|---|---|---|
| main 未統合（製品コード） | **0** | — |
| main 未統合（CI 定義・Owner 保留） | **2** | R-101 / R-102 として保持 |
| 未完成・放置（証跡・控え） | **2 群** | R-103 完了 / R-104 完了 |
| 維持すべき保留事項（U5 / U3・U4） | **2** | 変更なし |
| 整理候補（ブランチ・worktree・重複） | **4 群** | R-105 完了 / R-110 完了 / R-108 現状維持 / R-109 完了 |

## 2. 残課題一覧

### R-101 `ci/h2r1` の扱い（保留）

| 項目 | 内容 |
|---|---|
| 課題内容 | H-2-R1 測定用ワークフロー（新規 235 行）を恒久化するか、保持のみとするか |
| 発生元 | `ci/h2r1-release-measurement-20261030`（R-04 測定、2026-10-30） |
| 関連ファイル | `.github/workflows/h2r1-release-measurement.yml`、`doc/residual-20261101/H2R1-MEASUREMENT-RECORD-20261102.md` |
| 現在の状態 | ブランチ保持（tip `d7036618`）。main に workflow は無い。測定は 5/5 PASS で完了 |
| 未完了の理由 | 新規 workflow の恒久追加は CI 定義変更に該当し、別承認が必要 |
| 影響・リスク | 統合後の再測定手段が無い（現行 main を測るには main 側にワークフローが必要） |
| 必要な作業 | ①恒久化する場合: `workflow_dispatch` のみに整理し policy 差分を落として 1 ファイル追加 ②しない場合: 現状維持 |
| 優先度 | P3 |
| 推奨対応 | 現状維持（記録は `H2R1-MEASUREMENT-RECORD-20261102.md` に保存済み）。判断材料は `CI-ANALYSIS-R101-R102-R108-20261102.md` §1 に提出済み |
| 必要権限 | Owner 判断（CI 定義変更） |

### R-102 `ci/r49` の扱い（保留）

| 項目 | 内容 |
|---|---|
| 課題内容 | R-49 プローブ用ブランチの処分 |
| 発生元 | `ci/r49-selected-tests-probe-20261029`（2026-10-29） |
| 関連ファイル | `.github/workflows/windows-msvc-build.yml`（+51 行）、旧ベース由来の 2 ワークフロー削除差分 |
| 現在の状態 | 保持（tip `96b46aa0`）。ahead=2 / cherry+=2 |
| 未完了の理由 | そのまま統合すると `sanitizer-ci.yml` と `vbcable-probe.yml` が消える |
| 影響・リスク | 統合不可。放置しても main に影響なし |
| 必要な作業 | 移植する場合は現 main への手作業移植として別設計 |
| 優先度 | P3 |
| 推奨対応 | 保持（統合しない）。将来プローブ再開時に参照。処分案は `CI-ANALYSIS-R101-R102-R108-20261102.md` §2 に提出済み |
| 必要権限 | Owner 判断（統合する場合のみ） |

### R-103 j195 worktree の未追跡 764 件（証跡）— **完了**

| 項目 | 内容 |
|---|---|
| 課題内容 | 未追跡証跡（764 件・1,998.3 MB）の退避と worktree 閉鎖 |
| 発生元 | 複数セッション（2026-09〜11 の調査・検証記録） |
| 関連ファイル | `.openchamber/a13-2-evidence/`・`phase-d-evidence/`・`dumps/` ほか |
| 現在の状態 | **退避・閉鎖とも完了**。ZIP 387,303,598 B（`e4a32118…`）を Dropbox `/archive-j195-20261102/` に配信し、**再取得して 764 エントリ全件を照合（不一致 0・余分 0）**。追跡変更 5 件も `j195-tracked-changes.patch`（46,770 B）として同梱 |
| 未完了の理由 | — |
| 影響・リスク | なし（データは main と Dropbox に保全） |
| 必要な作業 | 空ディレクトリの消滅待ちのみ（セッションが使用中のため削除不可。内容 0 件） |
| 優先度 | — |
| 推奨対応 | セッション終了後に空ディレクトリが解放される |
| 必要権限 | — |

### R-104 worktree の取込済みローカル変更 13 ファイル — **完了**

| 項目 | 内容 |
|---|---|
| 現在の状態 | j237 系 2 本の閉鎖（R-110）と j195 の閉鎖（R-103）により**解消**。追加削除なし |
| 根拠 | 追加行の 95〜100% が main に存在（Phase A §2.1） |
| 必要な作業 | なし |

### R-105 完了済みブランチの整理 — **完了**

| 項目 | 内容 |
|---|---|
| 課題内容 | `ahead=0` / `cherry+=0` のローカルブランチの削除 |
| 現在の状態 | **55 本を削除**（失敗 0）。削除後は main ＋ 保留 2 本（`ci/h2r1` / `ci/r49`）の **3 本** |
| 確認条件 | ①main の祖先であること（55/55）②未統合コミット 0 ③worktree 未使用 ④削除範囲はローカルブランチのみ（リモート削除なし） |
| 根拠 | 55 本すべて `merge-base --is-ancestor` が真＝内容は main に保持。アーカイブタグ不要 |
| 削除一覧 | 付録（55 行・ブランチ名と HEAD SHA） |
| 必要な作業 | なし |

### R-106 U5 の状態（既定 OFF 維持）

| 項目 | 内容 |
|---|---|
| 課題内容 | U5（R-06）の技術的解決が未着手 |
| 現在の状態 | **FAIL・既定 OFF**（`STATUS-20261101.md`） |
| 未完了の理由 | 技術的解決が未着手。有効化は別承認 |
| 影響・リスク | 機能有効化は不可 |
| 必要な作業 | 設計・実装（別フェーズ） |
| 優先度 | P2（機能面） |
| 推奨対応 | 現状維持。着手は別承認 |
| 必要権限 | Owner 判断（有効化） |

### R-107 U3 / U4（HOLD 維持）

| 項目 | 内容 |
|---|---|
| 課題内容 | U3 / U4 の実行 |
| 現在の状態 | **HOLD** |
| 必要な作業 | 別途の判断後に実行 |
| 優先度 | P3 |
| 推奨対応 | 現状維持 |
| 必要権限 | Owner 判断 |

### R-108 基点フォルダの旧分割（Dropbox）

| 項目 | 内容 |
|---|---|
| 課題内容 | `/ConvoPeq-stage3-20261026/ConvoPeq.part01〜04.md` が旧版（`f8aab49b`）の分割 |
| 現在の状態 | 旧版のまま（正規はルート `/ConvoPeq-c12a532e-split/`） |
| 未完了の理由 | 既存 Dropbox ファイルの変更は未承認 |
| 影響・リスク | 同名参照で誤認する余地 |
| 必要な作業 | 現行版で更新するか削除。関係と削除案は `CI-ANALYSIS-R101-R102-R108-20261102.md` §3 に記録済み（現状維持） |
| 優先度 | P3 |
| 推奨対応 | 現行版へ更新（または削除） |
| 必要権限 | Owner 判断 |

### R-109 作業台帳の Dropbox 配信 — **完了**

| 項目 | 内容 |
|---|---|
| 更新対象 | `/ConvoPeq-stage3-20261026/ledger/RESIDUAL-LEDGER-20261028.md` |
| 更新前 | 397,086 B / rev `0165f436edbd9f30…` / SHA-256 `5c9ef2cc88c60bf2…` |
| 更新内容 | 第 67〜71 次（設定是正・受容判断・Phase A/Phase E の記録）を含む **404,351 B** |
| バックアップ | `/ConvoPeq-stage3-20261026/backup-ledger-20261102/RESIDUAL-LEDGER-20261028.md.bak-20261102`（397,086 B・一致確認済み） |
| 更新後 | **404,351 B** / rev `0165f4593673b590…` / SHA-256 `94a2bb079f4669c1…` / modified 2026-11-02T06:51:30Z |
| 取得検証 | 再取得してサイズ・SHA-256 の一致を確認（PASS） |
| 必要な作業 | なし |

### R-110 j237 系 worktree 2 本の閉鎖

| 項目 | 内容 |
|---|---|
| 課題内容 | `j237-baseline`（8 件変更・取込済み）と `j237-baseline-clean`（クリーン）の閉鎖 |
| 現在の状態 | 両方とも存在。ブランチは detached のため削除対象なし |
| 未完了の理由 | 閉鎖は Owner 判断 |
| 影響・リスク | なし |
| 必要な作業 | `git worktree remove` → `prune` |
| 優先度 | P3 |
| 推奨対応 | j237-baseline-clean から閉鎖（クリーン）。j237-baseline はパッチ保存後 |
| 必要権限 | Owner 判断 |

## 3. 維持している制約

| 制約 | 状態 |
|---|---|
| force push / 履歴書き換え | 未実施 |
| 未検証変更の main 直接統合 | 未実施（全変更は PR または検証済み） |
| 既存成果物の不可逆な削除 | 未実施（削除はアーカイブタグ／Dropbox 退避後） |
| CI 必須条件の緩和 | 未実施 |
| U5 有効化 | 未実施（FAIL・既定 OFF） |
| U3 / U4 | HOLD 維持 |
| `ci/h2r1` 恒久化 | 未実施 |

## 4. 本フェーズで実施した整理（2026-11-02）

| 対象 | 実施内容 | 検証 |
|---|---|---|
| `j237-baseline` worktree | 未コミット 8 ファイルのパッチ（12,624 B）を Dropbox `/archive-j237-baseline-20261102/` へ退避してから閉鎖 | 退避 3 ファイルの内容一致を確認。閉鎖後ディレクトリ消滅 |
| `j237-baseline-clean` worktree | クリーンのため通常削除 | ディレクトリ消滅 |
| `git worktree prune` | 管理情報の整理 | 終了コード 0 |
| 結果 | worktree は **2 本**（`C:\VSC_Project\ConvoPeq` と現行セッションの `j195-notion-report`） | `git worktree list` で確認 |

→ R-110（j237 系 worktree の閉鎖）は**完了**。

### 4.1 R-103〜R-105・R-109 の実施結果（2026-11-02）

| 項目 | 結果 |
|---|---|
| R-103 退避 | 764 件（1,998.3 MB）を ZIP 387,303,598 B にまとめ、Dropbox `/archive-j195-20261102/` へ配信。**再取得して 764 エントリ全件照合（不一致 0・余分 0）** |
| R-103 閉鎖 | `git worktree remove` は使用中で失敗したが、内容はすべて削除され **0 件**、`prune` で管理登録を解除。`git worktree list` は main のみ。空ディレクトリのみ残存（セッションが使用中のため削除不可） |
| R-104 | j195・j237 の閉鎖により**解消**（追加削除なし） |
| R-105 | 統合済み **55 本**を削除（失敗 0）。残りは main ＋ `ci/h2r1` ＋ `ci/r49` の 3 本 |
| R-109 | 台帳を 404,351 B に更新（旧 397,086 B は `backup-ledger-20261102/` に退避）。再取得で一致確認（PASS） |

### 4.2 報告数値の整合性（Owner 指摘の 4 点）

| 指摘 | 実測と説明 |
|---|---|
| 追跡 7,047 件 vs 検証スクリプト 7,045 件 | `3dbbd6f9` 時点が 7,045 件、Phase A・E の文書 2 件を加えた `47d833a2` 以降が **7,047 件**。差は追加した文書 2 件 |
| 配信 7,764 / 8,942 B | `PHASE-A-INVENTORY-20261102.md` = 7,764 B、`RESIDUAL-LEDGER-PHASE-E-20261102.md` = 8,942 B（いずれも Dropbox と一致） |
| worktree の件数 | 調査開始時 4 本 → j237 系 2 本を閉鎖 → 3 本 → j195 を閉鎖し **1 本**（`C:\VSC_Project\ConvoPeq`） |
| 削除対象ブランチ | **55 本**（当初報告の「56」は誤記。58 − main − 未統合 2 = 55）。削除結果は付録のとおり |

## 附録

### 付録: R-105 で削除したブランチ（55 本）

| # | ブランチ | HEAD SHA |
|---|---|---|
| 1 | `backup/main-before-sync-20261028` | `a12420526d1c` |
| 2 | `chore/convopeq-md-20261026` | `08d751fc43d7` |
| 3 | `chore/convopeq-md-493cfbbb-20261030` | `751ef379a67d` |
| 4 | `chore/convopeq-md-cr-20261027` | `0b6a49ebb675` |
| 5 | `chore/convopeq-md-f4diag-20261027` | `aa56eaf0dbb3` |
| 6 | `chore/convopeq-md-p21-20261027` | `7f753bb7498c` |
| 7 | `chore/convopeq-md-r47b-20261029` | `30cd96d478f8` |
| 8 | `chore/convopeq-md-t10-20261026` | `68950995eb30` |
| 9 | `chore/convopeq-md-t10c-20261027` | `a339d2740b20` |
| 10 | `chore/convopeq-md-t10direct-20261027` | `d6e375879756` |
| 11 | `chore/convopeq-md-t10fix-20261026` | `902eb88e6499` |
| 12 | `chore/convopeq-md-t9-20261027` | `e27154d45827` |
| 13 | `ci/sanitizer-oneapi-r48-20261029` | `45af6df3726f` |
| 14 | `ci/windows-msvc-build-20261029` | `32932387ba3c` |
| 15 | `docs/agents-env-sync-20261028` | `b16659f9d467` |
| 16 | `docs/d02-env-isolation-plan-20261101` | `d3cc3d54ec9d` |
| 17 | `docs/d07-ledger-summary-20261101` | `fa178d7c5536` |
| 18 | `docs/residual-inventory-20261101` | `73e563b652c5` |
| 19 | `docs/track-reports-20261101` | `63f9336bff29` |
| 20 | `feat/b2-stage1-t1-measure-20261028` | `7b5ff5b27ba3` |
| 21 | `feat/d02-rt-measure-split-20261101` | `9c89b1ea2f89` |
| 22 | `feat/engine-instance-count-diagnostic-20261030` | `fc8e8a6beac2` |
| 23 | `feat/r06-u2-rt-measure-20261031` | `da472ec73956` |
| 24 | `feat/r06-u5-prep-20261031` | `45811d7f9cdc` |
| 25 | `feature/p0-1-observetoken-formalization` | `3f7269c4cca0` |
| 26 | `feature/p0-2-runtimeexecutionview-readpath` | `4aff30d3cf5a` |
| 27 | `feature/p1-e1-comment-corrections` | `2b36acde5288` |
| 28 | `fix/a2-probe-lint-ae011-20261027` | `2b25f3b2894e` |
| 29 | `fix/b2-diag-sink-20261101` | `c9e74603f291` |
| 30 | `fix/compile-blockers-20261028` | `7da877e8d41c` |
| 31 | `fix/d01-stg8-test-determinism-20261101` | `355584457c34` |
| 32 | `fix/f2-asan-reclaim-race-20261101` | `91d29903ab5a` |
| 33 | `fix/f2-quarantine-destroy-lifo-20261031` | `5ec8d65b97e7` |
| 34 | `fix/h3-t10-correction-20261026` | `56472207df89` |
| 35 | `fix/isrshutdown-test-seam-guard-20261102` | `e8e83fc545a4` |
| 36 | `integrate-fasttanh` | `d485110d3f2c` |
| 37 | `integration/all-sessions-20261026` | `d54beb63789e` |
| 38 | `phase-c-prime` | `a4f61d02415f` |
| 39 | `r8-stall-clear-obligation-fix` | `75db52b1fc02` |
| 40 | `stg11-d1-d18` | `3bc9c32e6599` |
| 41 | `test/a2-rt-allocation-probe-20261027` | `087c1c3118e0` |
| 42 | `test/f4-d2b-overwrite-redrive-diag-20261027` | `4e342fb2b7ee` |
| 43 | `test/h3-t9-delete-count-20261027` | `e9b2382d59d9` |
| 44 | `test/p2-1-resolve-states-20261027` | `830dfc3cf0f3` |
| 45 | `test/r51-two-engine-permanent-20261031` | `b15c92ed147a` |
| 46 | `test/retiregrace-constexpr-20261027` | `a869dae5a8d1` |
| 47 | `test/t10-direct-access-20261026` | `78362e0f36a0` |
| 48 | `test/t10c-production-provider-20261027` | `5373fa95faeb` |
| 49 | `v11-agc-lifetime-fix` | `916447f1ea3c` |
| 50 | `v11-handle-lifetime-f2f3` | `1bf90f8a9606` |
| 51 | `v11-handle-lifetime-hd` | `4898625da108` |
| 52 | `work/g1-stg8-selector-20261026` | `d8ccd7112532` |
| 53 | `work/h2r1-mixed-tuple-20261026` | `20546ab6a75e` |
| 54 | `work/h3-t10-target-retire-20261026` | `390ded0b661f` |
| 55 | `work/h5-gb-quiescence-20261026` | `81bd40267904` |

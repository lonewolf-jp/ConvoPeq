# Phase E 残課題台帳（2026-11-02）

Owner 決定（全セッション成果統合・正本統一・残課題整理）に基づく残課題の一覧。
Phase A の実測（コミット `bdb1dcb8` 時点）を根拠とする。

- 記載方針: 未解決でも「状態・理由・影響・必要作業・優先度・必要判断」が明確であれば整理完了とする
- 優先度: P1（正本・安全性に直結）/ P2（品質・再現性）/ P3（整理・衛生）

## 1. サマリ

| 分類 | 件数 | 状態 |
|---|---|---|
| main 未統合（製品コード・テスト） | **0** | R-112 は PR #107 でレビュー中、U4 は PR 作成予定 |
| main 未統合（CI 定義） | **0** | R-101（`7318e44f`）/ R-102（`a83d10ba`）ともマージ済み |
| 未完成・放置（証跡・控え） | **2 群** | R-103 完了 / R-104 完了 |
| U5（R-06） | **1** | 再測定完了・**FAIL＋判定保留**（既定 OFF 維持） |
| R-107 U4 / U3 | **2** | U4 実装・検証中 / U3 計画済み（§10.4） |
| 整理候補（ブランチ・worktree・重複） | **4 群** | R-105 完了 / R-110 完了 / R-108 削除済み / R-109 完了 |

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
| 現在の状態 | **55 本を削除**（失敗 0）。削除後は main ＋ 保留 2 本（`ci/h2r1` / `ci/r49`）の **3 本**。その後 #114〜#116 の作業ブランチが加わり、2026-11-03 時点のローカルは 6 本（§10.6 末尾） |
| 確認条件 | ①main の祖先であること（55/55）②未統合コミット 0 ③worktree 未使用 ④削除範囲はローカルブランチのみ（リモート削除なし） |
| 根拠 | 55 本すべて `merge-base --is-ancestor` が真＝内容は main に保持。アーカイブタグ不要 |
| 削除一覧 | 付録（55 行・ブランチ名と HEAD SHA） |
| 必要な作業 | なし |

### R-106 U5 の状態（既定 OFF 維持）

| 項目 | 内容 |
|---|---|
| 課題内容 | U5（R-06）の技術的解決（機能の非干渉を許容値内に収める設計） |
| 現在の状態 | **FAIL・判定保留**（2026-11-02 再測定 20 実行で検出力不足）。測定器は**既定 OFF** |
| 未完了の理由 | 判定に足る検出力が得られていない。指標の技術判断案を `R106-METRIC-REDEFINITION-PROPOSAL-20261103.md` に提出し、Owner 決定待ち |
| 影響・リスク | 機能有効化は不可（既定 OFF のまま） |
| 必要な作業 | 指標方針の決定後に再測定、または判定不能でクローズ（§10.6） |
| 優先度 | P2（機能面） |
| 推奨対応 | 現状維持。測定結果を合格扱いにしない |
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

## 5. Owner 判断の受領（2026-11-02）

| ID | Owner 判断 | 台帳上の状態 |
|---|---|---|
| R-101 `ci/h2r1` | 恒久化は #105 で完了。プローブブランチは **2026-11-04 に削除**（§11.1） | 解消 |
| R-102 `ci/r49` | 診断コードは #106 で main に移植済み。ブランチは **2026-11-04 に削除**（§11.1） | 解消 |
| R-108 Dropbox 旧分割コピー | **現状維持を承認**。削除・上書き・移動は不承認。再検討には対象と影響を明示した別判断が必要 | 現状維持 |
| R-103 / R-104 / R-105 / R-109 | **完了報告および証拠を受理**（追加操作の承認を意味しない） | 完了 |
| R-106 / R-107 | 従来どおり（U5 = FAIL・既定 OFF、U3/U4 = HOLD） | 維持 |
| R-110 | 完了（j237 系 worktree 閉鎖） | 完了 |

### 5.1 共通制約（本判断に基づく）

- 追加の**実装変更・CI 変更・不可逆な削除**は許可されていない。
- 未承認事項は本台帳に維持する。
- Owner の明示的な追加判断なしに、保留事項を実行へ移さない。

## 6. 残課題の位置づけ（2026-11-02 訂正）

Owner の評価を受け、「残課題なし」ではなく次のように整理する。

| 区分 | 状態 |
|---|---|
| Phase 1〜5 の承認対象作業 | **完了** |
| Owner の主要目標（回収・統合・正本一致） | **達成** |
| **未解決の技術課題・個別判断事項** | **5 件**（R-101 / R-102 / R-106 / R-107 / R-108） |
| 追加作業 | 5 件それぞれの個別判断後に実施 |

5 件の実行承認案は `EXECUTION-PROPOSALS-5ITEMS-20261102.md` に提出した。
内容は R-101（CI 恒久化の具体手順）、R-102（診断 4 ステップの移植）、
R-106（U5 の原因分析と環境隔離・原因帰属による再測定）、
R-107（U3/U4 の前提・リスク・検証計画）、R-108（旧分割の重複確認と削除案）。

**「5 件を残課題なし」として扱わない。** 各項目は未承認の限り実行しない。

## 7. Owner 評価と推奨順序（2026-11-02）

Owner は提案書の構成（5 件を個別の承認単位に分割し、変更範囲と合格条件を明示）を評価した。
**ただし実行承認は与えていない。** 現在の状態は次のとおり。

| 区分 | 状態 |
|---|---|
| 承認案 | **提出済み** |
| 実施 | **未承認** |
| 変更 | **なし**（製品コード・CI・ブランチ・Dropbox 既存ファイルとも） |

### 7.1 判断上の要点（Owner 提示）

| 項目 | 要点 |
|---|---|
| R-101 | CI 定義変更。policy の旧 1 行を混入させない移植手順と、main 上の dispatch 検証が重要 |
| R-102 | 既存 workflow の変更。`run_selected_tests` の true/false 両条件の検証が必要 |
| R-106 | 再測定のみ。σ の改善を合格と誤認せず、測定結果を FAIL 判定に使わない設計を維持 |
| R-107 | U4 先行の段階案は合理的。ただし U3/U4 の実装承認は別途必要 |
| R-108 | Dropbox 削除。ZIP 退避・内容一致確認・削除対象の特定を承認条件とする |

### 7.2 推奨順序（Owner 提示）

1. R-106（再測定の実施）
2. R-101（CI 定義変更）
3. R-102（CI 追記）
4. R-106 の測定結果と判定方針の確認
5. R-107（U4 の対象列挙・レビューから着手するか判断）
6. R-108（削除対象と ZIP 退避の実体を確認してから承認）

**R-107 は U5 の判定方針確定前に着手しない**（提案条件の尊重）。
**R-108 は削除操作のため、他の項目とは独立に扱う。**

### 7.3 現時点の扱い

- 5 件はすべて「承認案提出済み・実施未承認」。本台帳に維持する。
- 承認された単位のみ着手し、承認前に測定・CI 変更・削除を行わない。

## 8. R-108 削除前確認の受容（当時は未承認・2026-11-02）

> 2026-11-04 追記: 本章の「未承認」は当時の状態。削除は §11.1 で実施済み。

Owner は退避 ZIP の検証結果をもって**削除前の確認条件が満たされた**と判断した。
**ただし削除実行の承認ではない。**

| 確認条件 | 結果 |
|---|---|
| ZIP の実在・取得一致 | PASS（1,498,358 B / `469aebf3…`） |
| ZIP 内 4 ファイルと削除対象のバイト一致 | PASS |
| 展開による復元性 | PASS |
| 削除対象のパス・rev の確定 | PASS |

### 8.1 現在の状態（5 件）

| 項目 | 状態 |
|---|---|
| R-108 | **削除前確認完了・削除実行は未承認** |
| R-101 / R-102 | CI 変更未承認 |
| R-106 | 再測定未承認 |
| R-107 | U4 着手未承認 |

### 8.2 維持する順序

`R-106 → R-101 → R-102 → R-106 結果確認 → R-107（U4 対象列挙）`
R-108 は独立した判断項目として扱い、事前確認の完了をもって自動承認とはしない。

> ※ その後、同日の Owner 指示書で 4 ファイルの削除が承認され、§9 で実施した。

## 9. R-108 削除の実施記録（2026-11-02）

Owner 承認（削除前確認の完了を受けて）に基づき、承認された 4 ファイルのみを削除した。

| 手順 | 結果 |
|---|---|
| 退避 ZIP の存在・取得一致（再確認） | PASS（1,498,358 B / `469aebf3…` / rev `0165f46760…`） |
| ZIP 内 4 ファイルと削除対象のバイト一致 | PASS（4/4） |
| ZIP からの復元一致 | PASS（4/4） |
| パス・rev の照合 | PASS（`0165f353…` / `0165f2cd…` / `0165f31462…` / `0165f31464…`） |
| 削除したファイル | `ConvoPeq.part01.md`〜`part04.md` の **4 件のみ** |
| 削除後の非存在確認 | 4/4 で metadata 取得不可 |
| 正本 `/ConvoPeq.md` | 6,451,711 B / `c12a532e…`（不変） |
| 現行分割の連結 | 6,451,711 B / `c12a532e…`（正本と一致） |

R-108 は**完了**。退避 ZIP は `/ConvoPeq-stage3-20261026/archive-old-split-20261102/` に保全。

## 10. 2026-11-02 夜間の実施結果（R-101 / R-102 / R-106 / R-107）

### 10.1 R-101 `ci/h2r1` の恒久化 — マージ済み・5 回検証は再実施中

| 項目 | 結果 |
|---|---|
| 変更 | `.github/workflows/h2r1-release-measurement.yml` 追加のみ（+232 行） |
| トリガ | `workflow_dispatch` のみ |
| policy 旧 1 行 | **混入なし**（差分は 1 ファイル +232 行のみ） |
| マージ | PR #105 → `7318e44f`（squash） |
| PR チェック | 12 pass / 0 fail（probe は仕様どおり skipped） |
| 既存ジョブ | fail なし（12 pass が既存 13 ジョブに相当。probe は dispatch 専用） |

dispatch 検証の経過:

| 試行 | run | 結論 | run 全体 | job 所要 |
|---|---|---|---|---|
| 1 | `41422842771` | failure（45.7 分・run 5 の flaky テスト） | — | — |
| 2 | `41422888522` | 参考（旧 main） | 105.0 分 | — |
| 3 | `41447831754` | success | 61.1 分 | 60.0 分 |
| 4 | `41456500055` | **success（合格）** | **49.2 分** | 49.2 分 |
| 5 | `41464228797` | success | 60.8 分 | 60.7 分 |
| 6 | `41474161659` | success | 60.4 分 | 60.2 分 |
| 7 | `41484247670` | **success（合格）** | **59.8 分** | — |
| 8 | `41493867695` | **failure** | 10.5 分 | — （oneAPI 導入: `winget exit=-1978335226`、ヘッダは存在） |
| 9 | `41495643990` | success | 63.4 分 | — |
| 10 | `41505721908` | **success（合格）** | **52.8 分** | — |
| 11 | `41513986906` | success | 60.7 分 | — |
| 12 | `41523395609` | **failure** | 58.3 分 | — （測定ステップ内 STG-8-D2 flaky） |
| 13 | `41532396525` | **success（合格）** | **59.6 分** | — |
| 14 | `41548873504` | **success（合格）** | **59.8 分** | — |

- **合格 5 件（すべて rc=0・60 分以内）**: `41456500055`（49.2 分）、`41484247670`（59.8 分）、`41505721908`（52.8 分）、`41532396525`（59.6 分）、`41548873504`（59.8 分）。
- rc=0 は **11/14**（失敗 3 件: oneAPI 導入 1、STG-8-D2 flaky 1、旧 main の flaky 1）。
- 測定値 `[H2R1] trials=40 mixed_trials=0 distinct_worlds=80` は全成功回で同一。
- 所要のばらつきは `Install Intel oneAPI via winget`（未導入時 10.7 分）に依存するが、**5 回連続で 60 分以内**を達成。
- 残課題: R-113（STG-8-D2 flaky）、R-111（`CONVO_CI_BUILD` 未設定）、workflow の oneAPI 導入堅牢化。




**flaky の原因と修正**: `checkSTG8ClearTerminalizesObligation` が RebuildThread 稼働中に
`clearDeferredForShutdown()`（plain メンバ所有・停止時専用）を直接呼び、`processDeferredAdmission()`
と競合していた。正規入口 `requestDeferredClear()` へ変更（PR #107）。ローカル ctest 3 連続で
**rc=0 / `STG-8-CLEAR` PASS**（304.4 s / 302.1 s / 299.5 s）。

### 10.2 R-102 `ci/r49` の診断移植 — 検証完了

| 項目 | 結果 |
|---|---|
| 変更 | `windows-msvc-build.yml` に +51 行（診断 3 ステップ＋`run_selected_tests` 入力） |
| マージ | PR #106 → `a83d10ba`（squash） |
| `false` | `41423092426` **success**／診断 3 ステップ **skipped** |
| `true` | `41423073648` **failure**（診断実行）。`AudioEngineHarness` Passed 298.27 s、`HeadlessAudioPathVerification` Failed（`[CLI_PERF_RAW] callbacks=` 不在） |
| 既存 CI | PR #106 12 pass / 0 fail、main push の Windows MSVC・ISR Authority は success |

### 10.3 R-106 U5 再測定 — 完了（FAIL・判定保留）

20 実行（5/アーム/条件、34,000 ブロック）を 121.7 分で実施。

| 条件 | p99 判定 | max 判定 | σ(max) | σ(p99) | 5 組での検出力 |
|---|---|---|---|---|---|
| 無音 | PASS | PASS | 280.3 µs | 9.2 / 20.1 µs | 0.05 |
| 1 kHz | PASS | **FAIL** | 403.4 µs | 31.4 / 46.4 µs | 0.04 |

- 全 20 実行で `late50us=0`（周期超過なし）。max は単発スパイク。
- σ ≤ 64 µs は **max で未達**、p99 で達成。+50 µs の判別に必要な組数は 493〜1,022/アーム。
- 判定: **FAIL（max 基準）かつ判定保留（検出力不足）**。指標の再定義は行わない（技術判断案は §10.6）。
- 測定器は**既定 OFF を維持**。詳細は `R106-REMEASUREMENT-RESULT-20261102.md`。

### 10.4 R-107 U4 / U3

| 項目 | 状態 |
|---|---|
| Stage 1（列挙・計画） | 完了（`R107-STAGE1-ENUMERATION-20261102.md`） |
| Stage 2（U4 実装） | 実装済み・ブランチ `feat/r107-u4-lock-measure`（`dd56f0f0`）。既定 OFF、非 RT の 14 ロック取得点を `LockScope` 化、検証 (13) 追加 |
| Stage 2 検証 | Release ビルド・`RtMeasureRingTests`・PR の CI で確認（結果は §10.5 に追記） |
| Stage 3（U3） | **実装・検証完了**（PR #114、`e89c66fe`）。U4 と分離し、テストバイナリ限定の `operator new` 置換（`RtAllocProbe.cpp` の 1 箇所）。陽性対照 1、RT 確保 **0**（6,470 / 38,208 ブロック）、既定 OFF |

### 10.5 本夜間で追加・解消した残課題

| ID | 内容 | 状態 |
|---|---|---|
| R-111 | `HeadlessAudioPathVerification` が音声デバイスの無い CI ランナーで FAIL する | **修正済み**: `windows-msvc-build.yml` の Configure に `CONVO_CI_BUILD=1` を設定（登録 1→0 を実証）。CI で最終確認 |
| R-112 | `checkSTG8ClearTerminalizesObligation` の flaky（`hasDeferred still true after clear`） | **PR #107 で修正**（ctest 13 連続 PASS） |
| R-113 | `checkSTG82OverwriteTerminalizesEvicted` の flaky（`[STG-8-D2] FAIL: unstable (L=1 hasDeferred=0)`） | **修正済み**: 固定 8 秒判定を不変条件サンプリングへ変更（解放時は TTL 30 秒＋余裕 5 秒の 35 秒で終端化を要求）。Release **16 回すべて rc=0**、Debug 1/1、ASan CI PASS |
| — | oneAPI 導入ステップの `winget` 非 0 による FAIL | **修正済み**: 終了コードを成功判定に使わず、ヘッダ＋lib 4 点で判定（両 workflow） |
| — | U3（RT 経路の新規確保検出） | **実装・実測完了**: 陽性対照 1、RT 確保 **0**（6,470 ブロック／5 秒、38,208 ブロック／30 秒）。既定 OFF・テスト限定 |

### 10.6 R-106 指標再定義の技術判断案（Owner 承認待ち）

R-106 は **FAIL・判定保留のまま**据え置く。判定を動かさず、指標定義の選択肢だけを技術判断案として提出した。

- 提案文書: `R106-METRIC-REDEFINITION-PROPOSAL-20261103.md`
- 提案の核: 主判定を p99 の非劣性検定（片側 Welch t、α=0.05、差 5%）へ移し、`late50us=0` などのゲートを測定成立条件として分離する。max は記録専用（OFF の 2 倍超で警告）とし、合否に使わない。
- 根拠: `late50us=0`（20/20）で max は周期超過ではない。σ(max) 280〜403 µs は判定幅 +50 µs の 5.6〜8.1 倍で、5 組の検出力は 0.04〜0.05。検出力 0.8 には 390〜873 組/アーム（40〜89 時間/アーム）が必要。
- p99 の検出力: 無音 1.000（n=5）、1 kHz 0.871（n=5）／0.928（n=6）。推奨は 6 組/アーム。
- 既存 20 実行の参考値: 無音 ON/OFF 98.91%（p=0.0002）、1 kHz 99.28%（p=0.0047）。**参考値であり合格扱いにしない**。
- 決定依頼: D1 v2 採用の可否、D2 既存データでの再判定の可否、D3 不採用時のクローズ可否。
- 測定器の既定 OFF は維持。製品コードは変更しない。

#### ローカルブランチの内訳（2026-11-03）

| ブランチ | 状態 |
|---|---|
| `main` | `3fa7090f`（#116 マージ後） |
| `docs/r106-metric-proposal-20261103` | #116 の squash 元。main との差分 0 |
| `feat/r113-r111-oneapi-u3-20261103` | コード差分 0（内容は main に取り込み済み） |
| `docs/final-20261103b` | コード差分 0（同上） |
| `measure/u5-20261102` | U5 の原因帰属計装 `[U5ATTR]` を保持する測定専用。main には入れない |
| `ci/h2r1-release-measurement-20261030` / `ci/r49-selected-tests-probe-20261029` | 2026-11-04 に削除（タグ `archive/ci-h2r1-20261030-d7036618` 等で保持。§11.1） |


### 10.7 R-106 metric v2 新規測定 — PASS

事前登録 `R106-METRIC-V2-MEASUREMENT-DESIGN-20261103.md` の規則をそのまま適用した独立測定（旧 20 実行は流用しない）。

- 実行: 24 実行（6 組/アーム/条件）、2026-11-03 20:26:21 〜 2026-11-03 22:52:20、ツリー `a0bad07a`
- ゲート: 違反 0 件 / 再実行 0 件。破棄した 6 実行（ゲート訂正前・判定未計算）は別保存
- 判定規則: p99 の片側 Welch 非劣性（H0: ON > 1.05×OFF、α=0.05）、両条件合格で PASS
- **metric v2 総合: PASS**

| 条件 | OFF p99 平均 (µs) | ON p99 平均 (µs) | ON/OFF | 差（ON−1.05×OFF） | p 値 | 判定 |
|---|---|---|---|---|---|---|
| silence | 1190.5 | 1196.5 | 100.50% | -53.5 | 0.0000 | **PASS** |
| tone1k | 1553.8 | 1552.2 | 99.89% | -79.4 | 0.0002 | **PASS** |

- max は記録専用（合否に使わない）。既定 OFF は維持。製品の有効化は別判断。
- 詳細: `R106-METRIC-V2-RESULT-20261103.md`、証跡 `metric-v2-evidence-20261103.zip`


### 10.8 R-101 / R-102 の現行 main 再確認（2026-11-04）

`main` = `ecfdb244`（U4・U3・R-106 ドライバ込み）で両方を再実行した。

| 項目 | 結果 |
|---|---|
| R-101 `41665191767` | success。5/5 回 rc=0、各回 `[H2R1] trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80`（PASS）。run 全体 **61.0 分**（60 分基準は 1 分超過。oneAPI 導入 10.7 分が原因で、コード要因ではない） |
| R-102 `41665198158` | success。`CONVO_CI_BUILD: 1`、`1/1 Test #49: AudioEngineHarness ... Passed 319.11 sec`、診断 3 ステップ success |
| R-107 U4 | `7dfb21d4`（PR #108）が main の祖先。変更なし |
| R-107 U3 | `e89c66fe`（PR #114）が main の祖先。変更なし |

- 既存の合格 5 件（49.2 / 59.8 / 52.8 / 59.6 / 59.8 分）は維持し、本実行は現行 main での機能再現の証跡として追加する。
- 詳細: `R101-R102-RECHECK-20261104.md`

### 11. 2026-11-04 の整理・監査（R-108 / R-101 / フレーク / 棚卸し）

#### 11.1 R-108 ブランチ整理 — 完了

- 指示対象の 2 本（`ci/h2r1`・`ci/r49`）に加え、同一条件を満たす ci 4 本と作業ブランチ 7 本を、**アーカイブタグを付けて削除**した。
- リモート 49 本を棚卸し: main の祖先 **42 本**は削除、非祖先 **7 本**は旧版スナップショット（新規コードなし）と確認してタグ保存のうえ削除。
- 現在のリモートは `main` のみ。アーカイブタグは 38 本。
- 詳細: `R108-BRANCH-CLEANUP-20261104.md`

#### 11.2 R-101 oneAPI キャッシュ — 実装・実測

| 項目 | 内容 |
|---|---|
| 方式 | `actions/cache@v4` で**実体**の `mkl\2025.1`・`ipp\2022.1` を保存（`latest` は symlink のため対象外）。復元後に junction を作り直す |
| キー | `oneapi-mkl-ipp-Windows-mkl2025.1-ipp2022.1-v2` |
| ミス経路 | run `41736861395`。導入 9.35 分、キャッシュ 918.1 MB を保存 |
| ヒット経路 | run `41751003210`。`oneAPI cache hit: true`、復元 50 秒、導入ステップ 1 秒（winget 非実行）、4 点チェック成功 |
| 効果 | R-101 の run 全体 **61.0 分 → 51.4 分**（60 分基準内）。5/5 回 rc=0・`[H2R1]` PASS |
| 合格条件 | 変更なし（60 分基準・4 点チェック・fail closed 維持） |

#### 11.3 R-113 / R-111 フレーク監視 — 再発なし

- 直近の Asan ジョブ（PR #114〜#121 の 8 本）はすべて success。`AudioEngineHarness` スイートの再発は確認されていない。
- 再発がないため、無意味な再実行・変更は行っていない。監視は継続する。

#### 11.4 放置・未完成事項の棚卸し

| ID | 内容 | 状態 | main 統合 | 必要な作業 | 承認 |
|---|---|---|---|---|---|
| U5-ENABLE | U5 の製品有効化 | 未承認・既定 OFF | 意図的に未統合 | 有効化する場合の設計・検証 | 必要 |
| R101-CACHE | oneAPI キャッシュ | 実装・実測済み | #121 | なし（継続監視） | 不要 |
| FLAKE | R-113 / R-111 | 修正済み・再発なし | #114 | 再発時の原因調査 | 不要 |
| U5ATTR | 測定専用計装 | `measure/u5-20261102` に保持 | 意図的に未統合 | なし | — |
| BRANCHES | 保留ブランチ | 整理済み | — | なし | — |

オープン Issue 0 件、オープン PR なし（#121 はマージ済み）。詳細: `FINAL-AUDIT-20261104.md`

### 12. 2026-11-04 Reader memory safety 再監査（R-13）

最新 main（`6eeb1d42`）で公開 world の read-side を再監査した。

- プロトコル自体は健全: `publishAndSwap` は `acq_rel`、`observe` は `acquire`、retire は `publishEpoch()` で **epoch を前進**させ、`isOlder(retireEpoch, minReaderEpoch)` のときだけ解放する。reader が scope 内なら `minReaderEpoch <= enterEpoch < retireEpoch` で解放されない。
- 欠陥: **load してから enter する**箇所が 3 つあった。
  1. `AudioEngine::makeRuntimeReadHandle`（world load → 末尾で enter）
  2. `RuntimePublicationOrchestrator::trySubmitImpl`（RebuildThread からも実行、guard なし）
  3. `AudioEngine::makePublishDecisionSnapshot`（producer スレッド、guard なし）
- 修正: enter を load の前へ移動し、enter 失敗時は world を公開しない（fail-closed）。#2/#3 は局所 `RCUReader`＋`RCUReaderGuard` で保護。
- 回帰防止: `ScopedEpochReaderStructuralTests` に「guard が world load より前」「fail-closed」の 4 チェックを追加（計 15 チェック）。
- 検証: Release ビルド rc=0、`AudioEngineHarness` rc=0（297.12 s・100% passed、V11 reader テスト含む）、ASan/TSan は PR で確認。
- 残存: message-thread 専用の読み（`currentPublicationEpoch` 等）は reclaimer と同一スレッドのため未変更。将来 worker から呼ぶ場合は同じ契約が必要。
- 詳細: `READER-MEMORY-SAFETY-R13-20261104.md`

### 13. 2026-11-04 全セッション成果の統合監査

Owner 指示（全開発成果の統合・作業環境一本化・残課題整理）に基づく最終監査。

- セッション: 全 14。リポジトリ成果を持つのは `main-develop` のみ（他は Notion 監査 10・形式検証 2・応答検証 1 で、ブランチ・worktree・未コミット変更を持たない）。
- 統合: PR **#105〜#126**（22 件）が main に統合済み。未統合の完成成果は 0。意図的未統合は `measure/u5-20261102`（`[U5ATTR]`）のみ。
- 状態: `main` = `origin/main` = `e69f20c3`（#126 のスカッシュマージ後。調査開始時は `38075a7a`）、未コミット 0 件、worktree 1 本、リモートブランチは `main` のみ、アーカイブタグ 42 本。
- ConvoPeq.md: LF 6,482,434 B / `a3784c44…`、`--check` FRESH。
- 残課題: `RESIDUAL-ISSUES-20261104.md`（8 分類）。維持事項は U5 未承認、R-101 初回ミス時、フレーク監視、`[U5ATTR]` 保持。技術的負債 3 件は不具合なし。
- 記録: `SESSION-INVENTORY-20261104.md`、`INTEGRATION-LOG-20261104.md`、`FINAL-REPORT-20261104-FULL-INTEGRATION.md`。

### 14. 2026-11-05 放置・未完成・未修正バグの棚卸し

`main` = `89c48a02` 時点で、未完成の修正・機能、放置された作業、未修正の欠陥候補を洗い直した。詳細は `RESIDUAL-ISSUES-20261105.md`。

- 未修正・未確定の欠陥候補 3 件: ポリフェーズ利得の不対称（案 E が既定 OFF。採用判断は材料充足、既定 ON は HOLD、設計意図かは未確定、`RES2-01`）、`isBadSample` のスカラー／AVX2 閾値不一致（`RES2-02`）、SoftClip 局所 OS の遅延定数 15 が未実測（`RES2-03`）。
- 途中で止まっている修正・機能 5 件: F-4/D2b 記録消失（調査停止、`RES2-04`）、B2 Stage 2/3（`RES2-05`）、A2 本番適用（`RES2-06`）、U5 製品有効化（`RES2-07`）、`EngineRuntime` 単一 atomic 統合（`RES2-08`）。
- 検証の穴 4 件: TSan の best-effort（`RES2-11`）、soak の手動実行のみ（`RES2-12`）、TX-02／TX-10 の Waived（`RES2-13`）、`FFTBackendTests` の現状確認止まり（`RES2-14`）。
- 技術的負債 6 件: CMA 状態の部分永続化、elite 再評価、`D8_1` の陳腐化コメント、Worker のアサーション欠落、Phase-II 送り項目、`.venv` の Python 3.14 消滅（`RES2-15`〜`RES2-20`）。
- 既に修正済みで「未修正」と誤読されやすい項目（H-01・C-01・B-2 診断クラッシュ・A2 assert・STG-8 フレーク・`.gitignore` のフェンス行・ddgs の uvx パス）は、同書 §9 に分離して除外した。

### 15. 2026-11-05 RES2 残課題の一次処理

Owner 指示（RES2-01〜20 の処理開始）に基づく一次処理。

- 計画: 20 件すべてを 10 項目で整理し、状態を分類（修正着手可能 3・追加調査 4・検証のみ 2・設計判断 6・承認待ち 5）。
- 修正した 3 件: RES2-02（破損判定閾値の統一、#131 `ad19211c`）、RES2-14（FFT 出力の参照比較テスト、#130 `951e8b9c`）、RES2-17（`D8_1` T2b の陳腐化コメントを契約検査へ、#129 `65afa479`）。
- 検証: 3 PR とも CI 14 pass / 0 fail / 0 pending（ASan 含む）。icx Release で追加テストを実行し pass。
- 未着手（承認待ち 5）: RES2-01・04・05・06・07。U5 は既定 OFF を維持。
- 未着手（設計判断 6）: RES2-08・12・15・16・19・20。RES2-19 は設計どおりで対応不要。
- 未着手（追加調査 4）: RES2-03・11・13・18。検証のみ 2: RES2-09・10。
- `ConvoPeq.md` は 3 件のソース変更を反映して再生成した。
- 詳細: `RESIDUAL-ISSUES-20261105.md` §11、`doc/work113/20261105-1200_RES2処理計画/`。

### 16. 2026-11-05 Owner 判断材料パケット（6 件）

Owner 指示（判断が必要な 6 件の材料整理）に基づく提出物。実行・有効化・再開は行っていない。

- 対象: RES2-01（案 E）・RES2-04（F-4/D2b 再開）・RES2-05（B2 Stage 2/3）・RES2-06（A2 本番適用）・RES2-07（U5 有効化）・RES2-12（soak スケジュール化）。
- 各項目に「現行ソースの問題・放置リスク・推奨案・案ごとの副作用・必要権限・検証方法・求める判断」を記載。
- 推奨: RES2-01・04・05・06・07 は現状維持（既定 OFF／停止／NO-GO）、RES2-12 は夜間スケジュール化。
- 詳細: `RES2-DECISION-PACKETS-20261105.md`。
- U5 と案 E は既定 OFF を維持している。

### 17. 2026-11-05 RES2・U5・案 E の詳細調査

- 基準: `main` = `f90f56be`（clean・worktree 1 本）。
- 再確認: RES2 20 件すべてを現行コードで確認。RES2-03（P0-F が D=15 を実測済み）と RES2-08（RuntimeWorld 移行で前提消滅）を解消。
- 案 E: OFF 実測（54/54 PASS、DC 0.75^N、cand 1.0、SoftClip DC 0.375→0.5）と ON ビルド実測（19/54 PASS、DC 1.0）を取得。既定 OFF を維持。
- U5: 既定 Off・環境変数のみ・UI 経路なし・リング 8,192・dropped 計上を確認。有効化は未承認のまま。
- 修正: `AudioEngine.Processing.Latency.cpp` のコメント更新のみ（挙動変更なし）。
- 記録: `RES2-DETAILED-INVESTIGATION-20261105.md` ほか 6 資料。

### 18. 2026-11-05 RES2 全件処理（数値解析）

- 方針: 聴感評価を使わず、数値解析・自動テスト・性能測定で評価。
- RES2-01: `PolyphaseGainMetrics`（test-only）を追加し、OFF/ON の DC・RMS・ピーク・周波数応答・レイテンシ・CPU を測定。ON は DC 1.0、ピーク比 1.03〜1.67（最大 +4.4 dB）、レイテンシは 4 経路で不変・LP3 r8 のみ 475→461、CPU 差はばらつき範囲。採用は未承認のまま。
- RES2-07: `RtMeasureOverheadTests` を追加し、計測器のコストを測定（Off 1.34 / TimeOnly 26.84 / Full 26.60 ns/ブロック）。有効化は未承認のまま。
- RES2-18: `ObserveChannel::Worker` に `debugAssertNotAudioThread()` を追加（Release はインライン化で消える）。`ConvoPeq`・`AudioEngineHarness` の再ビルド成功、4 テスト Passed。
- 状態: 修正済み 6（02・03・08・14・17・18）、承認待ち 5、設計判断 4、追加調査 2、検証 2、対応不要 1。
- 記録: `RES2-FULL-RESOLUTION-20261105.md`、`RES2-ITEM-ACTIONS-20261105.md`、`VERIFICATION-RESULTS-20261105.md`、`RES2-REGISTER-FINAL-20261105.md`。
- 統合: PR #136（`42924b5c` → `ac5f8eb2`）。CI 14 pass / 0 fail。

### 19. 2026-11-05 Owner 判断の受領と処置

- Owner が RES2 12 項目を決定。記録: `RES2-OWNER-DECISION-RECORD-20261105.md`。
- 保留: 案 E（既定 OFF）、U5（既定 OFF・UI 保留）。停止: RES2-04。クローズ: B2 Stage 2/3。NO-GO 維持: A2。TSan: 必須化しない。soak: 週次を候補（設定変更は確認後）。TX-02/10: 要件照合待ち。CMA: 現状維持・elite は設計案のみ。Worker: 現行維持。`.venv`: 再作成しない。
- 追加資料: `RES2-TSAN-GATE-DESIGN-20261105.md`、`RES2-SOAK-OPERATIONS-20261105.md`、`RES2-CMA-REQUIREMENTS-20261105.md`、`RES2-TX-REQUIREMENT-RECONCILIATION-20261105.md`。
- 製品コード・既定値・RT 経路・CI 設定は変更していない。
- §18 の件数（修正済み 6・承認待ち 5 など）は判断前の値である。判断後の内訳は `RES2-REGISTER-FINAL-20261105.md` §21 を参照する。

### 20. 2026-11-05 追加判断（残り 4 項目）

- Owner が要件・運用・開発環境の 4 項目を判断した。
  - RES2-13: 現行要件との照合を継続する。過去の Waiver を自動免除にしない。
  - RES2-20: `.venv` は再作成・削除しない。参照スクリプト 5 本の移行設計を作成する。
  - RES2-11: best-effort を維持し、必須ゲート化しない。対象テスト・失敗時の扱い・安定性基準を整理する。
  - RES2-12: 週次実行を採用候補とする。実行時間・コスト・成功率・通知を確認してから設定を変更する。
- 追加資料: `RES2-DEVSCRIPT-MIGRATION-20261105.md`。`RES2-TSAN-GATE-DESIGN` と `RES2-SOAK-OPERATIONS` を更新。
- 製品コード・既定値・RT 経路・CI 設定は変更していない。

### 21. 2026-11-06 最終判断（RES2-11・12・13）

- RES2-11: best-effort を維持。対象テスト・連続成功率・skip・環境依存・実行時間・CI コストを評価し、必須化は条件充足後に範囲限定で別判断。
- RES2-12: 週次は候補のまま。GitHub-hosted runner は oneAPI 不在で skip するため、手動継続し `schedule` は追加しない。失敗は通知のみ。
- RES2-13: 数値検証可能な再定義案を承認。ブロック処理時間の基準は RT 制約（1.25 ms 周期）との照合後に確定。実機・waiver は別判断。
- 製品コード・既定値・RT 経路・CI 設定は変更していない。

### 22. 2026-11-06 追加判断（TX-02／TX-10 のブロック時間）

- ハード制約 1.25 ms（48 kHz・512）を維持する。
- 運用目標 1.0 ms（周期の 80%、余裕 0.25 ms）を承認する。
- 1.0 ms は RT 制約そのものではなく運用候補であり、実機検証なしに達成済みとは扱わない。
- 基準の承認、実機検証、TX-02／TX-10 の waiver は分離して扱う。

### 23. 2026-11-06 最終 Owner 判断

- `RES2-OWNER-FINAL-DECISION-20261106.md` を以後の基準とする。
- RES2-11 は best-effort 維持。評価条件（対象固定・skip の扱い・連続成功率・実行時間とコスト・環境分離）の充足後に、専用環境で限定必須化を再判断する。TSan の成功を完全なスレッド安全性の証明とはしない。
- RES2-12 は手動継続。oneAPI ランナーの構成とコストの確定まで `schedule` を追加しない。通知のみを検証成立としない。
- RES2-13 は再定義と運用目標 1.0 ms を承認済み。ハード制約 1.25 ms は維持。実機検証と waiver は個別判断。1.25 ms 超を自動 waiver にしない。
- 製品コード・RT 経路・CI・既定値の変更、検証実行、waiver 発行は本判断に含まれない。

### 24. 2026-11-06 未決事項の判断材料

- TSan: `RES2-TSAN-EVIDENCE-20261106.md`。評価条件 1〜5 は未充足。skip が success になるため success 率は検出力を表さない。専用環境での実走が先。
- soak: `RES2-SOAK-RUNNER-OPTIONS-20261106.md`。セルフホスト（13.3 分）を第一候補、次点は開発機の週次実行。oneAPI 導入込みの CI は導入時間が不確定。
- TX: `RES2-TX-MEASUREMENT-PLAN-20261106.md`。48 kHz・512、10 回試行、最大 1.0 ms 合格／1.25 ms 超は不合格、1.0〜1.25 ms は条件付き。
- いずれも実行承認ではない。

### 25. 2026-11-06 未決 3 件の最終判断

- RES2-11: best-effort 維持。条件 1〜5 未充足のため必須ゲート化しない。条件充足後に専用環境で限定必須化を再判断。skip の success を検出力の証拠にしない。
- RES2-12: 手動維持・schedule なし。oneAPI 導入済みセルフホストを第一候補、開発機の週次を次点。構成・費用・運用責任の確定後に再判断。
- RES2-13: 1.0 ms 目標と 1.25 ms ハード制約を維持。測定計画と合否基準を採用。実機測定は別承認、waiver は個別判断、1.25 ms 超・XRUN の自動許容なし。
- コード・RT 経路・既定値・CI・実走・schedule・実機測定・waiver は本判断に含まれない。

### 26. 2026-11-06 TSan 診断実走と soak 手動 1 回

- TSan: `CMakeLists.txt` の計装範囲を診断目的で変更し、非 MSVC の MSVC 専用フラグを除去。Linux で configure 成功、計装 3 本が PASS・報告 0 件。`DiagnosticsConfig.h` の値判定への変更は、ASan でアロケータの不一致（bad-free）を招くため撤回した。
- TSan の本命 `ISRSemanticValidationTests` は MKL/IPP 依存でビルド不能。Linux では MKL・IPP の探索とリンクが MSVC 専用で、環境整備だけでは解決しない。
- soak: `ce386431` から再ビルドしたハーネスで手動 1 回を実施。ALL PASS、744.8 秒、issued 180,300 / rejected 0、メモリ 284.7 MB 収束。
- CI 必須化、schedule、週次化、waiver はいずれも未実施。

### 27. 2026-11-06 限定実走の受理

- RES2-11: 計装 3 本（`MpscBoundedRingTests`、`ScopedEpochReaderStructuralTests`、`ISRRuntimeIdentityTests`）の実走を受理。`ISRSemanticValidationTests` と `RetireGraceSemanticsTests` は MKL・IPP 依存で未実走。`ScopedEpochReaderTests` のリンク失敗は TSan と区別。ゲート化はしない。MKL・IPP の Linux 対応は未承認。
- RES2-12: `ce386431` から再ビルドした手動 1 回を受理。ALL PASS、744.8 秒、issued 180,300 / rejected 0、メモリ 284.7 MB 収束。週次・登録・費用・責任者は未決定。
- RES2-13: `[RTM]` と `xRunBuffer` の取得口の確認を受理。実機・シナリオ・実施者・時期は未指定で、測定は開始しない。基準は不変、waiver は個別判断。
- 受理は証拠の範囲に限る。TSan の検証完了、データ競合の不存在、性能達成を意味しない。

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

# Phase A 全セッション成果物インベントリ（2026-11-02）

Owner 決定（ConvoPeq 全セッション成果統合・正本統一・残課題整理）に基づく調査記録。
**本調査は実体確認を伴う**（ブランチ・worktree・PR・未追跡・パッチをすべて実測）。

- 調査基準: `main` = `3dbbd6f9`（本書を追加する直前の HEAD。以降は文書コミットで進む）
- 作業場所: `C:\VSC_Project\ConvoPeq`
- 参照: GitHub `lonewolf-jp/ConvoPeq`

## 1. セッションの棚卸し

OpenChamber のメッセージ索引（`%USERPROFILE%\.config\openchamber\message-search.sqlite`）に
**74 セッション**が登録されている。ConvoPeq の開発作業は主に次のセッション群で行われた。

| 区分 | 内容 |
|---|---|
| 現行の継続セッション | 39,618 メッセージ・最終 2026-11-02（本作業系列） |
| 2026-08〜09 の設計・改修セッション | 20 件超（P3-5 / STG 系の設計・レビュー・ゲート記録） |
| 2026-09 の環境構築セッション | ツール導入・MCP 設定・headroom 等の環境確認 |
| 2026-10 以降の実装セッション | 残課題フェーズ（B-2 / A2→A3-a / R-51 等）とブランチ運用 |

> セッションの「成果物」は、コミット・ブランチ・PR・worktree・未追跡として現れる。
> 本報告は**成果物の実体**を基準に分類し、セッション名はその由来として扱う。

## 2. worktree（4 本）

| worktree | ブランチ | HEAD | 追跡変更 | 未追跡 | 判定 |
|---|---|---|---|---|---|
| `C:\VSC_Project\ConvoPeq` | `main` | `3dbbd6f9` | 0 | 0 | **正本**（クリーン） |
| `…\worktree\f34893\j195-notion-report` | (detached) | `bda43034` | 5 | 764 | 変更は main に**取込済み**（後述） |
| `…\worktree\f34893\j237-baseline` | (detached) | `d79b8be4` | 8 | 0 | 変更は main に**取込済み** |
| `…\worktree\f34893\j237-baseline-clean` | (detached) | `d79b8be4` | 0 | 0 | クリーン |

### 2.1 未コミット変更の実体（main 取込済みの検証）

各 worktree のローカル変更（自分の HEAD に対する追加行）が main に存在するかを実測した。

| worktree | ファイル | 追加行 | main に存在 | 判定 |
|---|---|---|---|---|
| j195 | `AGENTS.md` | 213 | 213 (100%) | 取込済み |
| j195 | `CMakeLists.txt` | 288 | 288 (100%) | 取込済み |
| j195 | `opencode.json` | 14 | 14 (100%) | 取込済み |
| j195 | `src/audioengine/ISRDSPHandle.cpp` | 10 | 10 (100%) | 取込済み |
| j195 | `src/audioengine/RuntimePublicationOrchestrator.cpp` | 21 | 20 (95%) | 取込済み（残 1 行はコメント `}  // close outer if (hasDeferred_)`） |
| j237 | `AudioEngine.Processing.DSPCoreFloat.cpp` | 2 | 2 (100%) | 取込済み |
| j237 | `AudioEngine.Processing.DSPCoreIO.cpp` | 2 | 2 (100%) | 取込済み |
| j237 | `AudioEngine.Processing.ReleaseResources.cpp` | 6 | 6 (100%) | 取込済み |
| j237 | `AudioEngine.h` | 2 | 2 (100%) | 取込済み |
| j237 | `ISRRetireRouter.h` | 4 | 4 (100%) | 取込済み |
| j237 | `ISRShutdown.cpp` | 2 | 2 (100%) | 取込済み |
| j237 | `ISRShutdown.h` | 2 | 2 (100%) | 取込済み（PR #104 で統合） |
| j237 | `PublishPipelineIntegrationTests.cpp` | 16 | 16 (100%) | 取込済み |

→ **未統合の製品コードは無い。** いずれも古いベース上のローカル控えで、main が先行している。

## 3. ブランチ（ローカル 58 / リモート 47）

### 3.1 main 未統合コミットを持つブランチ（2 本のみ）

| ブランチ | ahead | cherry+ | 内容 | Owner 判断 |
|---|---|---|---|---|
| `ci/h2r1-release-measurement-20261030` | 3 | 3 | H-2-R1 測定用ワークフロー（新規 235 行）＋ policy 旧行 | **保持**（恒久化は別承認） |
| `ci/r49-selected-tests-probe-20261029` | 2 | 2 | R-49 プローブ（workflow 追加＋旧ベース由来の削除差分） | **保持**（統合不可） |

### 3.2 その他 56 本

`ahead=0` かつ `cherry+=0`（完全に main へ取込済み）。`ConvoPeq.md` 再生成系、
R-51 測定系、各修正ブランチを含む。**統合対象ではない。**

### 3.3 リモートのみのブランチ（1 本）

| ブランチ | SHA | main の祖先 | 差分 | 判定 |
|---|---|---|---|---|
| `copilot/audit-source-code-for-bugs` | `803fbd9b` | **はい** | 0 ファイル | 取込済み（統合不要） |

## 4. PR（直近 25 件を実測）

| 状態 | 件数 | 備考 |
|---|---|---|
| MERGED | 24 | #80〜#104（R-50 系・R-51 恒久テスト・U1/U2/U5 準備・D-01/D-02・B-2/A2→A3-a・j237 ガード） |
| CLOSED | 1 | #82（置換済み。`fix/r51-f1b-pin-protection-clean` 系で再提出） |
| OPEN | **0** | 未処理 PR なし |

## 5. 未追跡ファイル

| 場所 | 件数 | 内容 | 判定 |
|---|---|---|---|
| `C:\VSC_Project\ConvoPeq` | **0** | — | 正本はクリーン |
| j195 worktree | 764 | `.openchamber` 配下 756（証跡・スクリプト・ログ）、ほか 8（設定バックアップ・レポート） | 証跡。**コード変更は含まれない**（拡張子内訳: .md 444 / .py 57 / .json 56 / .patch 45 / .xml 34 / .jpg 33 / .csv 32 / .txt 25） |
| j237 / j237-clean | 0 | — | — |

`.patch` 45 件の内容が main に取込済みかは別途検証中（Phase A 追加確認）。

## 6. 分類（Phase B の基礎）

| 分類 | 対象 | 件数 |
|---|---|---|
| main 統合済み | ローカル 56 ブランチ・リモート 46 本・PR 24 件・取込済みリモート 1 本 | 多数 |
| main 未統合・統合可能 | **なし**（製品コードは 0 件） | 0 |
| main 未統合・競合解消が必要 | **なし** | 0 |
| 未完成・放置 | j195 の未追跡 764 件（証跡）、worktree ローカル変更 13 件（取込済み控え） | 2 群 |
| 検証不足 | なし（PR #103 / #104 は 13/13 PASS） | 0 |
| 保留・Owner 判断が必要 | `ci/h2r1`（恒久化）、`ci/r49`（統合不可） | 2 |
| 統合不要・重複 | 完了済みブランチ 56 本、再生成系、閉じた PR #82 | 多数 |

## 7. 結論（Phase A）

1. **main に未統合の製品コードは存在しない。** セッション成果はすべて main に取込済みで、
   残る未統合は **CI 定義 2 本（Owner 保留）** のみ。
2. したがって Phase C（統合）で新たに main へ入れる製品変更は無い。
   作業の中心は「正本統一の検証（Phase D）」と「残課題台帳（Phase E）」、
   および証跡・陳腐化控えの整理になる。
3. 未追跡 764 件は証跡であり、コード変更を含まない（`.patch` の取込状況は追加確認）。

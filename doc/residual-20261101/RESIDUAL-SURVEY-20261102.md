# 残課題調査報告（2026-11-02・読み取り専用）

- 作業場所: `C:\VSC_Project\ConvoPeq`（唯一）
- セッション: `main-develop`
- 状態: `HEAD` = `origin/main` = **`95cc0357`**、追跡差分 **0**、未追跡 **26**
- 本書は**調査と提案**であり、ソース・CI 定義・ブランチ・未追跡ファイルを変更していない
  （本書の追加のみ）。

## 1. ASan の結果（PR #104）

| 項目 | 実測 |
|---|---|
| PR #104 の状態 | **MERGED**（`0dfa39e3` / 2026-11-01T23:38:24Z） |
| チェック | **13 PASS / 0 pending / 0 fail** |
| 判定 | **ASan を含め全チェック PASS**（失敗の切り分けは不要） |

PR #103 も 13 PASS / 0 pending / 0 fail（ASan 1h19m28s、c-cpp 1h5m32s）で確定済み。
**両 PR とも全 CI 緑**であり、統合判断（ASan 完了前に統合）は結果的に条件を満たした。

## 2. CI 系ブランチ（7 本）の調査

`.github/` を変更するもの 3 本と、R-51 測定用（`.github/` を含まない）4 本に分かれる。

### 2.1 `.github/` を変更する 3 本

| ブランチ | 目的 | main との差分（`.github`） | 現 main の状態 | 必要性 | 放置した場合 | 選択肢 | 推奨 |
|---|---|---|---|---|---|---|---|
| `ci/h2r1-release-measurement-20261030` | H-2-R1 を Release CI で計測 | **新規 `h2r1-release-measurement.yml`（+235 行）** ＋ policy 1 行 | 該当ワークフロー**なし**（`H2R1` の記載 0 件） | 計測用の任意ワークフロー。恒久テストは main にあり（`20546ab6`） | 計測手段が失われる（製品挙動への影響なし） | ①統合 ②保持 ③破棄 | **保持**（計測は必要時に再作成可能。新規ワークフローの恒久追加は CI 定義変更のため個別承認） |
| `ci/r49-selected-tests-probe-20261029` | R-49 の dispatch プローブ | `windows-msvc-build.yml` +51 行、**加えて `sanitizer-ci.yml` と `vbcable-probe.yml` の削除**（旧ベース由来） | 両ワークフローは main に存在 | **そのまま統合すると 2 ワークフローが消える** | — | ①そのまま統合 ②現 main に手で移植 ③保持 | **保持**（統合不可。移植するなら別途設計） |
| `ci/r50-asan-vbcable-20261029` | ASan ジョブへの VB-CABLE 導入 | **policy 1 行のみ**（workflow 差分は main に取込済み `5057968c`） | `VB-CABLE` は main に 12 箇所あり | 実質差分なし | なし | ①破棄 ②保持 | **破棄候補**（内容は main に存在。ただし削除は個別承認） |

**共通の注意**: 3 本とも `.github/isr-ai-governance-policy.json` を 1 行変更しているが、
これは**今回の allowlist 同期（`c9e74603`）と衝突する旧内容**である。統合時は必ず手当てが要る。

### 2.2 R-51 測定用 4 本（`.github` を含まない）

| ブランチ | 内容 | main の状況 | 推奨 |
|---|---|---|---|
| `ci/r51-two-engine-on-pr82` | PR #82 上での二重起動再現（src 8 ファイル） | 恒久テスト `b15c92ed`（PR #95）で置換 | **破棄候補** |
| `ci/r51-two-engine-repro-20261030` | 二重起動の再現テスト（1 ファイル） | 同上 | **破棄候補** |
| `verify/r51-two-engine-on-merged-main-20261030` | 統合後 main での検証（1 ファイル） | 同上 | **破棄候補**（証跡としては PR/台帳に残る） |
| `ci/r51-diag-only-20261030` | 測定専用の計装（2 ファイル） | 同上 | **破棄候補** |

いずれも「測定・証跡」目的で、製品コードの修正ではない。削除は個別承認。

## 3. 陳腐化ブランチ（Owner 指定 8 本）

8 本はいずれも **`git cherry main <branch>` の未取込コミットが 0**（パッチ同一で main に存在）。

| ブランチ | HEAD | 分岐点 | 未統合 | 変更 | 重複 | 推奨 |
|---|---|---|---|---|---|---|
| `integration/session-consolidation` | `e8aa1f6f` | `3d4f9c40` | 0 | `scripts/mcp_probe.py`、`PublishPipelineIntegrationTests.cpp` | `fix/r51-*` と一部重複 | 破棄候補 |
| `fix/r51-f1b-pin-protection-20261030` | `ed11a6c3` | `493cfbbb` | 0 | audioengine 8 ファイル | `review/r51-merge-rehearsal` と同内容 | 破棄候補 |
| `review/r51-merge-rehearsal-20261030` | `e9613d67` | `9f81d5d5` | 0 | 同上（merge rehearsal） | 上記の複製 | 破棄候補 |
| `feat/b2-stage1-t1-20261028` | `ec80b9d3` | `f876bcb9` | 0 | テスト 2 ファイル | — | 破棄候補 |
| `fix/r52-f1b-defer-observation-20261029` | `7114440d` | `023ae16f` | 0 | `V11EngineRetireProtectionTests.cpp` | — | 破棄候補 |
| `chore/convopeqmd-5057968c-20261030` | `e7237372` | `5057968c` | 0 | `ConvoPeq.md` 再生成 | 後続の再生成（`3d8ea758`）が最新 | 破棄候補 |
| `work/g3-snapshot-ignore-20261026` | `812abcc8` | `812abcc8` | 0 | なし（base == HEAD） | — | 破棄候補 |
| `work/r16-i4-sync-20261026` | `3f3225d7` | `3f3225d7` | 0 | なし（base == HEAD） | — | 破棄候補 |

### 3.1 追加で確認した「内容が main にある」4 本

| ブランチ | 未取込（cherry+） | main 側の対応 | 推奨 |
|---|---|---|---|
| `bright-falcon` | 2 | `8d4a776f`（Phase 1）、`aa65170d`（Phase 2） | 破棄候補 |
| `crisp-zebra` | 1 | `08f8f19e`（`cherry-picked from bda43034`） | 破棄候補 |
| `main-pre-sync-20261021` | 1 | 同上（同一コミット） | 破棄候補 |
| `chore/convoPeqmd-023ae16f-20261029` | 1 | 後続の再生成が最新 | 破棄候補 |

**削除は行っていない。** 削除には個別承認が必要。

## 4. 未追跡ファイル 26 件

| 分類 | 件数 | 例 | 用途 | 推奨 |
|---|---|---|---|---|
| 調査報告（本作業） | 5 | `doc/work113/REPORT-20261101-{design-phase,abc-investigation,attach-plan,attach-contract,lifetime-proof}.md` | 設計・調査の記録 | **Git 管理を推奨**（台帳から参照している） |
| 過去の discovery 記録 | 6 | `doc/work113/P1-5-IR-P2_STG-11-D13〜D19_FRESH-DISCOVERY_20261001.md` | 2026-10-01 の STG-11 調査 | **Git 管理を推奨**（他の STG-11 文書は追跡済み） |
| 分割版ステージング | 4 | `.openchamber/convopeq-md-split-20261030/ConvoPeq.part01〜04.md` | 旧 `f8aab49b` の分割（6,436,359 B） | **管理しない**（再生成可能） |
| ハッシュ・手順書 | 2 | 同 `SHA256SUMS.txt`、`doc/ConvoPeq-f8aab49b-split/{README.md,SHA256SUMS.txt}`（計 3 だがうち 1 は上記） | 旧分割の検証キット | **管理しない**（旧版用） |
| バックアップ | 6 | `opencode.json.bak-py314-20261022-*.bak`（各 4,387〜4,410 B） | Python 3.14 時代の設定退避 | **破棄候補**（現行設定は `opencode.json` にあり） |
| ツール記録 | 1 | `.memories/repo/tool-versions-2026-09-26.md` | ツール版数の記録 | **Git 管理または `doc/` へ移動** |
| 生成物 | 1 | `vc140.pdb`（69,632 B） | MSVC の中間ファイル | **管理しない・破棄候補** |

**削除・移動・上書きは行っていない。** いずれも個別承認が必要。

## 5. `AGENTS.md` 運用メモ 17 行のレビュー

### 5.1 内容（事実）

追加されているのは **`semble` MCP の異常停止と復旧手順**。

- `uv\tools\semble\pyvenv.cfg` の `home` が削除済み `C:\Python314` を指し、`uvx` が exit 2
- 復旧: `uv tool install "semble[mcp]==0.6.0" --python 3.13 --force`
- 起動方法を `uvx` から `uv\tools\semble\Scripts\semble.exe` の直接起動へ変更
- `HF_HUB_OFFLINE=1` / `HF_HUB_DISABLE_TELEMETRY=1` を追加
- 検証: `mcp-probe.py` で `[OK] semble ... tools(2)`
- 既知の不安定: 複数ディレクトリ分を同時起動すると 1 本だけ exit 1（再現条件未特定）

### 5.2 評価

| 観点 | 評価 |
|---|---|
| Owner 要望との整合 | 「3 層パイプラインを常時使う」の前提（MCP が起動する）を守る情報であり**整合** |
| 作業場所の一元化 | 影響なし（環境依存の復旧手順） |
| GitHub main とソースの同期 | 影響なし |
| 承認境界の明確化 | 影響なし |
| 誤操作防止 | **有用**（古い Python を指す `pyvenv.cfg` の罠を記録） |
| 既存規約との重複・矛盾 | `## Python 3.14 → 3.15 移行` 節に **semble の項目が既にある**ため、**そこへ統合するのが自然**（新規節を作ると重複する） |

### 5.3 採用案（提案）

新規の節を作らず、既存の `## Python 3.14 → 3.15 移行（2026-10-20・MCP 全停止の復旧）` の
「7. **semble の異常停止（2026-10-30 復旧）**」として 17 行を追記する。
`※ 2026-10-30 訂正: semble は例外だった` の 1 行は、同節の「6.」の直後に置く。

**適用は未実施**（レビュー結果の確認後に適用する）。

## 6. 推奨アクション一覧（承認が必要なもの）

| # | 対象 | 推奨 | 必要な承認 |
|---|---|---|---|
| 1 | PR #104 ASan | 完了として記録する（追加作業なし） | 不要（結果確認済み） |
| 2 | `ci/r50-asan-vbcable-20261029` ほか R-51 測定 4 本 | 破棄 | ブランチ削除の個別承認 |
| 3 | 陳腐化 8 本 ＋ 内容一致 4 本 | 破棄（またはアーカイブタグ化） | 同上 |
| 4 | `ci/h2r1` / `ci/r49` | 保持（統合しない） | 不要（保持は現状維持） |
| 5 | 調査報告 5 件・discovery 6 件 | Git 管理に追加 | 追跡の個別承認 |
| 6 | 旧分割ステージング 7 件・`vc140.pdb` | 管理しない／破棄 | 削除の個別承認 |
| 7 | `opencode.json.bak` 6 件 | 破棄 | 削除の個別承認 |
| 8 | `AGENTS.md` の semble 記述 | 既存節へ統合 | 適用の承認 |
| 9 | CI 定義（workflow / policy / 検査スクリプト） | **変更しない** | 別途承認 |

## 7. 変更していないもの

- 製品コード・テストコード: 未変更（追跡差分 0）
- CI 定義（`.github/workflows` / policy / 検査スクリプト）: 未変更
- ブランチ: 削除・改名なし（8 本＋4 本＋CI 系 7 本を保持）
- 未追跡ファイル: 削除・移動・上書きなし（26 件を保全）
- U5 = FAIL / 既定 OFF、U3/U4 = HOLD: 維持
- リリース・外部公開・既定 OFF 有効化: 未実施

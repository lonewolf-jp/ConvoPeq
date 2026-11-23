# main マージ 事前調査 — 対象のブランチが存在しないため停止

- 調査日時: 2026-11-14 00:17 JST
- 対象: C3-D（head identity API 拡張＋Epoch 前進本体）、C9-A、retireBacklog 改名
- 種別: **第 1 段階（事前調査）のみ**。マージ・コミット・ビルド・テストは行っていない。

---

## 1. 結論（先に）

**マージ対象のブランチとコミット SHA は存在しない。** 3 件の変更は、いずれも `main` の
作業ツリー上に**未コミット**で存在する。したがって、指示された「ブランチのマージ」は実行できない。

加えて、作業ツリーには**今回の対象外である既存の未コミット変更**が 179 件あり、
対象 8 ソースのうち 3 ファイルは「今回の変更」と「既存の他タスク変更」が同一ファイル内に混在している。
指示の「対象外の機能変更を混ぜない」「未コミット変更が無いことを確認する」を満たせないため、
**推測でブランチを作らず、ここで停止**した。

## 2. 第 1 段階の結果

| # | 調査項目 | 結果 |
| --- | --- | --- |
| 1 | main の最新 HEAD | `173e69957b046290f7077797782f031ed8633fd0`（2026-11-12 の domain tag コミット。以後の更新なし） |
| 2 | 現在の作業ブランチ / HEAD | `main` / `173e6995`。worktree は 2 つ（`C:/VSC_Project/ConvoPeq` = main、`C:/VSC_Project/ConvoPeq-ci-fix` = `ci-fix/173e6995` @ `a05daf76`） |
| 3 | 各対象のブランチ名・コミット SHA | **存在しない**（ローカル・リモート 96 ブランチを検索。c3d / c9a / rename / epoch / drain / backlog に該当するものは無し。該当したのは 2026-10 の無関係な `retiregrace` / `h3-t10-target-retire` のみ） |
| 4 | 対象と main の差分 | 作業ツリー vs main の差分としてのみ存在（下記 §3） |
| 5 | main への取り込み済み有無 | **未取り込み**。新規シンボル `retireAcceptedCount_` / `peekHead` / `inspectRetireHead` / `advancedOnce` / `collectDrainBreakdown` はいずれも `git grep HEAD` で 0 件 |
| 6 | 未コミット変更 | **179 件**（modified 43 / untracked 136）。stash は空 |

## 3. 対象ファイルと混在の実態

### 3.1 今回の 3 件が触れたファイル（10 件）

| ファイル | 差分 | 内容 |
| --- | ---: | --- |
| `src/DeferredDeletionQueue.h` | +36 | A/C の `peekHead`、C3-D の out 引数拡張 |
| `src/core/EpochDomain.h` | +51 | `RetireHeadSnapshot` / `inspectRetireHead` |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | 45 | `DrainBreakdown`、改名 |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | 54 | `collectDrainBreakdown`、C9-A、改名 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | +115 | DRAIN_DIAG 2 行、`headSlot` / `headSeq` |
| `src/audioengine/AudioEngine.Threading.cpp` | +38 | C3-D の epoch 前進 |
| `src/audioengine/AudioEngine.h` | 11 | `pendingReclaimHandleCount()` |
| `src/tests/ISRSemanticValidationTests.cpp` | 205 | AC1・AC2・C9-A テスト、改名 |
| `AGENTS.md` | 103 | 本セッションの知見（複数タスク分） |
| `ConvoPeq.md` | 12,610 | 再生成（ワーキングツリー全体を反映） |

### 3.2 対象外の既存変更が混在しているファイル（3 件）

| ファイル | 混在している既存変更 | 由来 |
| --- | --- | --- |
| `src/audioengine/AudioEngine.Threading.cpp` | `#include "convolver/ConvolverStopProbe.h"`、`probeCoordStop` | 停止計測タスク（本セッション外） |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 同 include、`probeStopLearning` 2 箇所 | 同上 |
| `src/audioengine/AudioEngine.h` | `mutable std::mutex stateMutex`（BUG-1） | 別タスク |

### 3.3 文書側の混在

- `AGENTS.md` は本セッションの複数タスク（BUG-2、XML 上限、プリセット保存、停止計測、本件）の
  追記が 1 ファイルに同居している。
- `ConvoPeq.md` は**ワーキングツリー全体から生成**されるため、他タスクの未コミット変更も内容に含む。

## 4. 停止した理由（指示との不整合）

| 指示 | 現状 | 判定 |
| --- | --- | --- |
| 対象のブランチ名が特定できない場合は推測で選ばない | ブランチが存在しない | **停止条件に該当** |
| 未コミット変更が存在しないことを確認する | 179 件存在 | **前提を満たさない** |
| 対象外の機能変更を混ぜない | 3 ファイルに既存変更が混在 | ファイル単位のコミットでは満たせない |
| 既存の main の変更を巻き戻さない | 巻き戻しは無い | OK |

## 5. 進め方の選択肢（Owner 判断）

| 案 | 内容 | 長所 | 短所 |
| --- | --- | --- | --- |
| **A（推奨）** | `main` に直接コミットする。3 つの混在ファイルは**ハンク単位で分離**し、今回分だけをステージする | 履歴が単純。他の未コミット作業を巻き込まない | ハンク分離の作業が必要（`git apply --cached` でフィルタしたパッチを使う） |
| B | 一時ブランチ `impl/c3d-c9a-rename-20261114` を作り、A と同じ内容をコミットしてから `main` へ `--no-ff` マージ | 「マージコミット SHA」を残せる | 手順が 1 段増える。内容は A と同じ |
| C | 作業ツリー全体をコミット | 最速 | 対象外の変更（他タスク 179 件）を大量に含む。**禁止事項に抵触** |
| D | 対象ファイルだけを別ワークツリーへ複製し、そこで検証してから統合 | 既存作業を汚さない | 複製時点で混在ハンクが付いてくる。A の前処理が必要 |

**A か B を推奨**。どちらも「3 ファイルのハンク分離」が必須で、対象外の既存変更
（ConvolverStopProbe 系 5 ハンク、`stateMutex` 1 ハンク）を除外する必要がある。

## 6. 未実施

| # | 内容 | 理由 |
| --- | --- | --- |
| 1 | マージ / コミット | 対象ブランチが無く、指示の前提を満たさないため停止 |
| 2 | マージ後の Release ビルド・ctest・統合テスト | 新しい HEAD が存在しないため未実施 |
| 3 | ブランチの新規作成 | 「推測で選ばない」指示に従い、Owner 判断を待つ |

## 7. 参考: 直近の検証結果（新 HEAD のものではない）

作業ツリー（= main + 未コミット変更）に対する直近の検証は次のとおり。**マージ後の検証としては
流用できない**（指示どおり、新 HEAD で再取得が必要）。

| 項目 | 結果 | 記録 |
| --- | --- | --- |
| Release ビルド | 319 ターゲット・エラー 0 | `doc/work113/20261113-2342_改名実装/` |
| Release ctest | 61/61（152.20 s） | 同上 |
| CLI 3 回 | タイムアウト 0・約 4 秒 | 同上 |
| Debug ctest | 59/61（構成固有の 2 件） | `doc/work113/20261113-2325_Debug追加検証/` |

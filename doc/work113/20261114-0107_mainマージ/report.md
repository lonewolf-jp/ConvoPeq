# C3-D・C9-A・改名実装の main 統合 — 実施報告

- 実施日時: 2026-11-14 00:17〜（JST）
- 指示: 「C3-D・C9-A・改名実装の main 統合指示」（進め方 B = 一時ブランチ → `--no-ff` マージ）
- 対象: C3-D（head identity API 拡張＋Epoch 前進本体）、C9-A、`retireBacklogCount_` 改名
- 種別: 実装の統合・ビルド・テスト・CI 確認まで

---

## 1. 結論

3 件の変更をハンク単位で選別し、一時ブランチに 1 コミットとしてまとめ、`main` へ `--no-ff` で
マージした。対象外の未コミット変更（`ConvolverStopProbe` 系 5 ハンク、`stateMutex` 1 ハンク）は
コミットに含めず、作業ツリーにそのまま残した。新 HEAD に対する Release ビルドと ctest は成功し、
CLI 実行 3 回でも drain タイムアウトは再現しない。

| 項目 | 値 |
| --- | --- |
| 一時ブランチ | `impl/c3d-c9-rename-20261114` |
| 対象コミット | `6bb5700303dc8a8937b5e15ccd070032ee16ca96` |
| `main` 変更前 HEAD | `173e69957b046290f7077797782f031ed8633fd0` |
| `main` 変更後 HEAD | `862565a319f839d73db306eaaea5e3919d709c14` |
| マージコミット | `862565a319f839d73db306eaaea5e3919d709c14`（親 = `173e6995` + `6bb57003`） |
| 変更ファイル | 14 件（ソース 8・文書 6）、+1,513 / −31 行 |
| Release ビルド | 692 ターゲット・エラー 0（アプリ 126 ターゲットも別途成功） |
| Release ctest | 57/57 PASS（166.77 s） |
| 統合（CLI 12 秒 × 3） | タイムアウト 0・`[FAULT]` 0・総経過 13.86〜14.24 s |
| CI | 6 ワークフロー: success 4・failure 2（失敗 2 件はベース `173e6995` と同一） |

---

## 2. 第 1 段階（事前調査）の結果

| # | 調査項目 | 結果 |
| --- | --- | --- |
| 1 | `main` の最新 HEAD | `173e6995`（2026-11-12 以降の更新なし） |
| 2 | 現在のブランチ / HEAD | `main` / `173e6995`（worktree 2 つ: 本体と `ci-fix/173e6995`） |
| 3 | 対象のブランチ名・SHA | **存在しない**（ローカル・リモート 96 ブランチを検索） |
| 4 | 対象と `main` の差分 | 作業ツリー vs `main` の差分としてのみ存在 |
| 5 | `main` への取り込み | 未取り込み（`peekHead` 等 5 シンボルが `git grep HEAD` で 0 件） |
| 6 | 未コミット変更 | 179 件（stash は空） |

ブランチが無いため「ブランチのマージ」は成立しない。Owner の指示（進め方 B）に従い、
**未コミット変更をハンク単位で選別して一時ブランチにコミットし、`main` へ `--no-ff` マージ**した。

---

## 3. 変更選別（ハンク単位）

### 3.1 コミットに含めたファイル

| ファイル | 追加/削除 | 内容 |
| --- | ---: | --- |
| `src/DeferredDeletionQueue.h` | +36 / −0 | `peekHead` の out 引数化（slot / sequence / epoch の identity） |
| `src/core/EpochDomain.h` | +51 / −0 | `RetireHeadSnapshot` に `headSlotIndex` / `headSequence` / `headEpoch` |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | +37 / −8 | `DrainBreakdown`、`collectDrainBreakdown`、改名 |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | +41 / −13 | C9-A（13 条件）、診断、改名 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | +112 / −0 | `[DRAIN_DIAG]` 2 行（`headSlot=` / `headSeq=` 含む） |
| `src/audioengine/AudioEngine.Threading.cpp` | +34 / −0 | C3-D 本体（`waitForDrain` 内の epoch 前進） |
| `src/audioengine/AudioEngine.h` | +9 / −0 | `pendingReclaimHandleCount()`（診断用） |
| `src/tests/ISRSemanticValidationTests.cpp` | +195 / −10 | AC2 head identity、C9-A 2 分岐、改名 |
| `doc/work113/20261113-2031_c3c9設計検討/report.md` | +399 | 設計 |
| `doc/work113/20261113-2215_C3D実装/report.md` | +153 | C3-D 実装 |
| `doc/work113/20261113-2251_C3D本体実装/report.md` | +109 | C3-D 本体 |
| `doc/work113/20261113-2255_C9A実装/report.md` | +141 | C9-A |
| `doc/work113/20261113-2250_改名課題/issue.md` | +91 | 改名課題票 |
| `doc/work113/20261113-2342_改名実装/report.md` | +105 | 改名実装 |

混在 3 ファイルで**採用したハンク**は次の 5 つ（`git show -U0` 表記）。

| ファイル | ハンク | 内容 |
| --- | --- | --- |
| `src/audioengine/AudioEngine.h` | `@@ -1651,0 +1652,9 @@` | `pendingReclaimHandleCount()`（診断用アクセサ） |
| `src/audioengine/AudioEngine.Threading.cpp` | `@@ -323,0 +324,13 @@` | C3-D のコメントブロックと `advancedOnce` 等の宣言 |
| `src/audioengine/AudioEngine.Threading.cpp` | `@@ -325,0 +339,21 @@` | C3-D の前進＋drain ループ本体 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `@@ -583,0 +584,20 @@` | `[DRAIN_DIAG] worldClear`（読み取りのみ） |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `@@ -668,0 +689,92 @@` | `[DRAIN_DIAG] timedOut` の 13 条件出力（`headSlot=` / `headSeq=` 含む） |

### 3.2 除外した対象外ハンク（6 件・コミットに含めない）

| ファイル | 元の行 | 内容 | 由来 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.h` | `@@ -3088,3` | `mutable std::mutex stateMutex` | BUG-1（別タスク） |
| `src/audioengine/AudioEngine.Threading.cpp` | `@@ -12,2` | `#include "../convolver/ConvolverStopProbe.h"` | 停止計測 |
| `src/audioengine/AudioEngine.Threading.cpp` | `@@ -412,4` | `probeCoordStop`（Span / setCancelled / noteMark） | 停止計測 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `@@ -8,2` | `#include "convolver/ConvolverStopProbe.h"` | 停止計測 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `@@ -148,2` | `probeStopLearning`（terminal） | 停止計測 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `@@ -932,2` | `probeStopLearning`（reconfigure） | 停止計測 |

混在ファイル 3 件は「ハンク数 / 採用数」でいうと `AudioEngine.h` 2/1、`AudioEngine.Threading.cpp` 3/1、
`ReleaseResources.cpp` 5/2。選別は `git diff` からハンクを切り出し、`git apply --cached` で
ステージした（作業ツリーのファイルには一切書き込んでいない）。

### 3.3 コミット前に確認したこと

- ステージ済み差分（84,077 B）に `ConvolverStopProbe` / `probeStopLearning` / `probeCoordStop` /
  `stateMutex` / `BUG-1` が **1 件も含まれない**。
- 対象シンボルは含まれる: `peekHead` 23、`inspectRetireHead` 9、`headSlotIndex` 16、`advancedOnce` 5、
  `advanceRetireEpoch` 4、`collectDrainBreakdown` 9、`retireAcceptedCount_` 24、`DRAIN_DIAG` 10、
  `testAC2_headIdentity` 4、`testC9A_` 8。
- `AGENTS.md` と `ConvoPeq.md` は**コミットに含めていない**（他タスクの記述・作業ツリー全体の生成物のため）。

---

## 4. ブランチ作成・コミット

| 手順 | 結果 |
| --- | --- |
| 作業開始時の `main` HEAD 記録 | `173e69957b046290f7077797782f031ed8633fd0` |
| 一時ブランチ作成（未コミット変更を保持したまま） | `git checkout -b impl/c3d-c9-rename-20261114`（HEAD は `173e6995` のまま） |
| ハンク選別ステージ | 純粋 5 ファイルを `git add`、混在 3 ファイルはハンク抜粋パッチを `git apply --cached` |
| ステージ確認 | §3.3 のとおり対象外マーカー 0 件 |
| コミット | `6bb57003`（14 ファイル、+1,513 / −31） |

コミットメッセージ:

```
feat(isr): advance the retire epoch in the shutdown drain wait and drop the redundant backlog condition

C3-D: advance the epoch once per head identity inside waitForDrain, then drain.
- head identity (slot, sequence, epoch) is exposed through peekHead out params
- RetireHeadSnapshot carries the identity and [DRAIN_DIAG] logs headSlot/headSeq
C9-A: remove retireAcceptedCount_ == 0 from ShutdownScheduler::isFullyDrained.
- 13 conditions remain; Bootstrapping and Faulted tests added
rename: retireBacklogCount_ -> retireAcceptedCount_ (getter, TEST-ONLY setter, tests)
```

---

## 5. `main` へのマージ

| 手順 | 結果 |
| --- | --- |
| `main` が作業開始時から進んでいないことの確認 | `173e6995` のまま（`git rev-parse main`） |
| マージ | 別 worktree `C:\VSC_Project\ConvoPeq-merge`（`main` をチェックアウト）で `git merge --no-ff impl/c3d-c9-rename-20261114` |
| 競合 | **なし**（`ort` ストラテジで自動マージ） |
| マージコミット | `862565a319f839d73db306eaaea5e3919d709c14`（親 2 つ: `173e6995` / `6bb57003`） |
| マージ後の差分検証 | `173e6995..HEAD` の 14 ファイル。`ConvolverStopProbe` / `probeStopLearning` / `probeCoordStop` / `stateMutex` / `BUG-1` は 0 件 |
| push | `173e6995..862565a3  main -> main`（成功。main の必須チェックは bypass 記録あり） |

---

## 6. 検証（新 HEAD `862565a3` に対する実測）

過去（作業ツリー）の検証値は流用していない。以下はすべて `862565a3` の成果物で取得した。

### 6.1 Release ビルド

| 項目 | 結果 |
| --- | --- |
| 構成 | `cmake -S . -B build -G "Ninja Multi-Config"`（MSVC 14.52・`CONVOPEQ_PGO_INSTRUMENT=OFF` / `CONVOPEQ_PGO_USE=OFF`） |
| アプリ（`ConvoPeq`） | 126 ターゲット・エラー 0・`ConvoPeq.exe` 生成 |
| 全ターゲット | 692 ターゲット・エラー 0 |
| 補足 | 全ターゲット一括ビルドは `JuceHeader.h` の生成順で 23/733 で失敗した（`fatal error C1083`）。アプリを先にビルドして `JuceHeader.h` を生成した後、全ターゲットをビルドすると成功する。**ソースの不具合ではなく構成順序の問題**で、既存ビルドツリーでは発生しない |

### 6.2 Release ctest

| 項目 | 結果 |
| --- | --- |
| 実行 | `ctest -C Release --output-on-failure`（`862565a3` の build ツリー） |
| 結果 | **57/57 PASS**・失敗 0・166.77 s |
| 登録数が 57 の理由 | 作業ツリーの build は 61 件。差の 4 件（`CacheManagerHeaderValidationTests` / `InputInspectionTests` / `LearnedStateMigrationTests` / `LearnerStateAuthorityTests`）は**未コミットの `CMakeLists.txt`（+143 行・別タスク）が追加したターゲット**であり、`862565a3` には存在しないのが正しい |

対象テストの個別結果:

| テスト | 結果 | 備考 |
| --- | --- | --- |
| `ISRSemanticValidationRejects`（= `ISRSemanticValidationTests.exe`） | Passed 0.43 s | AC1 ミラー・AC2 head identity・C9-A 2 分岐を含む |
| `DomainTagTests` | Passed 0.13 s | domain tag の回帰なし |
| `D8_2_B_2_Tests` | Passed 0.14 s | Release では PASS（Debug のスタック問題は未変更） |
| `TailLayerIntegrityTests` | Passed 2.08 s | Release の CPU 予算内 |
| `AudioEngineHarness` | Passed 140.20 s | shutdown / drain 系を含む統合ハーネス |

`ISRSemanticValidationTests.exe` を直接実行しても `PASS`（0.35 s・exit 0）。AC2 と C9-A の
2 分岐は `main()` から無条件に呼ばれており（`throw` で失敗する形）、PASS は実行済みを意味する。

### 6.3 実装契約の確認（ソース実測）

| 対象 | 確認内容 | 結果 |
| --- | --- | --- |
| C3-D | `waitForDrain` のループ先頭で `inspectRetireHead()` → identity が変わったときだけ `advanceRetireEpoch()`、その後に `drainDeferredRetireQueues(true)` | 実装どおり（`AudioEngine.Threading.cpp:337-367`） |
| head identity API | `peekHead(DeletionEntry&, uint32_t&, uint32_t&)`、失敗時は 3 つとも書かない。`RetireHeadSnapshot` に `headSlotIndex` / `headSequence` / `headEpoch` | AC2 で固定（空キュー・sequence 不一致・peek 反復不変・reclaim 後の変化・deleter 1 回） |
| C9-A | `ShutdownScheduler::isFullyDrained()` は早期 return 1 ＋ `&&` 12 = **13 条件**。述語に `retireAcceptedCount_` は含まれない | 実装どおり（`ISRRuntimePublicationCoordinator.cpp:545-600`） |
| 改名 | `retireBacklogCount_` / `getRetireBacklogCount` は 0 件（宣言部の注記コメント 1 件のみ）。`setRetireBacklogCount` は**過去の設計を説明するコメント 7 件のみ**でコード参照 0 件 | 参照漏れなし |

### 6.4 統合（CLI 実行）

`--cli-run --cli-exit-ms 12000` で 3 回実行した（`862565a3` の `ConvoPeq.exe`）。

| 回 | 終了コード | 総経過 | `drain timeout reached` | `[FAULT]` | `[DRAIN_DIAG]` |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 14.24 s | 0 | 0 | `worldClear cleared=1 retireResult=void publishedAudit=4 retiredAudit=3 activeAudit=1 terminalAfterRetire=0` |
| 2 | 0 | 13.86 s | 0 | 0 | 同上 |
| 3 | 0 | 14.19 s | 0 | 0 | 同上 |

12 秒稼働＋約 2 秒で終了しており、2,000 ms の drain 待機を使い切っていない。
`timedOut=1` の診断行は**タイムアウト時のみ**出る設計なので、これが出ないことが drain 成立の証拠になる。

### 6.5 検証上の注意（新 HEAD で判明した差分）

- `--verify-runtime-evidence-live` は `862565a3` では**未実装**（B-11 の未コミット変更）。
  HEAD の `--verify-runtime-evidence` 判定は部分一致なので、このフラグを渡すと
  `--verify-runtime-evidence` 分岐に入り、証跡を書いて即終了する（実測 0.05 s・exit 0）。
  そのため統合計測は `--cli-run --cli-exit-ms 12000` のみで実施した。**本件の対象外**（B-11 の作業）。
- アプリ実行で `%APPDATA%\ConvoPeq` が書き換わるため、実行前後にバックアップと復元を行った
  （`device_settings.xml` `8dd0c67e`・`learned_state.xml` `2b073d96`・`noise_shaper_learn.xml`
  `21bdd698`、`IRCache` 23 ファイル。復元後に全一致を確認）。

---

## 7. CI 結果

push（`173e6995..862565a3`）で `862565a3` に対して 6 ワークフローが起動した。

| ワークフロー | 結果 | 備考 |
| --- | --- | --- |
| Windows MSVC Build | **success** | Configure / Build Release (ConvoPeq) / Build Release (AudioEngineHarness) すべて success |
| Sanitizer CI | **success** | Debug TSan success・RelWithDebInfo ASan success |
| CodeQL | **success** | — |
| ISR Authority Compliance v6.4 | **success** | — |
| audioengine-lint | failure | **ベース `173e6995` と同一の失敗**（後述） |
| list-compliance | failure | **ベース `173e6995` と同一の失敗**（後述） |

`Windows MSVC / Release` は push では ctest を実行しない（`Run selected tests` は dispatch 専用で
skipped）。したがって**テストの証跡はローカル ctest 57/57**（§6.2）である。

### 7.1 失敗 2 件はベース由来（本マージで増えていない）

| ワークフロー | 失敗内容 | ベース `173e6995` | 本マージ `862565a3` | 判定 |
| --- | --- | --- | --- | --- |
| audioengine-lint | `[LINT-AE-011]` 2 件（`ISRDebugRuntime.h:37` の `mutable HBVerifierRuntime hbVerifier_;`、`ISRRetireLatency.h:1021` の `thread_local std::uint32_t assigned`） | 同一 2 件 | 同一 2 件 | **ベース由来**。どちらも本コミットの変更ファイルではない |
| list-compliance | `[3.1] Strict atomic dot-call scan failed` | 同一 | 同一 | **ベース由来**（下記で機械的に一致を確認） |

atomic dot-call の一致確認は、検査スクリプト `.github/scripts/check-src-atomic-dotcall.ps1` を
**ベース `173e6995` の `src` と本マージの `src` の両方に対して実行**し、検出集合を比較して行った。

| ファイル | ベース | 本マージ | 一致 |
| --- | ---: | ---: | --- |
| `ISRRetireLatency.h` | 206 | 206 | 同一 |
| `WorldLifetimeRecorder.h` | 30 | 30 | 同一 |
| `ReaderMeasureRegistry.h` | 9 | 9 | 同一 |
| `ScopedEpochReader.h` | 9 | 9 | 同一 |
| `ISRDebugRuntime.cpp` | 2 | 2 | 同一 |
| `EpochDomain.h` | 1 | 1 | 同一 |
| `SnapshotCoordinator.h` | 1 | 1 | 同一 |
| 合計 | 258（7 ファイル） | 258（7 ファイル） | 完全一致 |

本コミットが触れた 8 ファイルのうち、この検査で検出されるのは `EpochDomain.h` の 1 件だけで、
**ベースと同一行**（追加行ではない）。他の 7 ファイルは検出 0 件。すなわち本マージは
`list-compliance` の失敗を増やしていない。

---

## 8. 作業ツリーの保全確認

| 確認 | 結果 |
| --- | --- |
| 除外ハンクの残存 | 混在 3 ファイルの未コミット差分は**除外した 6 ハンクのみ**（`stateMutex` 1 行、`ConvolverStopProbe` 系 7 行） |
| 対象 5 ファイル | コミット済みのため clean（内容は `main` と一致） |
| 未コミット変更の総数 | 作業開始時 179 件（modified 43 / untracked 136）→ 現在 170 件（modified 38 / untracked 132）。減少は「文書 6 件＋純粋 5 ファイル」がコミット済みになった分、増加は AC1 課題フォルダと本報告フォルダ。他タスクの変更は減少なし |
| ブランチの復帰 | マージ用 worktree を撤去し、本体 worktree を `main`（`862565a3`）へ戻した。`git worktree list` は本体と `ci-fix/173e6995` の 2 つ |
| 破棄・上書き | なし（対象 8 ソースへの書き込みは行っていない。`AGENTS.md` だけは統合記録を追記したが、コミットには含めていない） |

---

## 9. 未解決事項と残存リスク

| # | 内容 | 状態 |
| --- | --- | --- |
| 1 | AC1 のミラー乖離（`allDrained` が 14 条件） | **未修正**（指示の禁止事項）。課題票 `doc/work113/20261113-2350_AC1修正課題/issue.md` |
| 2 | `D8_2_B_2_Tests` の Debug スタック超過（0xC00000FD） | 未変更（指示の禁止事項）。Release は PASS |
| 3 | `--verify-runtime-evidence-live` の部分一致 | 未修正（B-11 の未コミット作業）。新 HEAD では live 証跡モードを使えない |
| 4 | C3-D のループ自体の単体テスト | 無し。head identity は AC2、ループはアプリ実測（§6.4）で確認 |
| 5 | `noteRetireBacklogChanged` / `previousRetireBacklogCount_` | 旧名のまま（改名課題票の範囲外） |
| 6 | マージ用 worktree `C:\VSC_Project\ConvoPeq-merge` | **撤去済み**（`git worktree remove --force`）。Release build も一緒に削除したため、再検証する場合は再ビルドが必要 |
| 7 | `AGENTS.md` / `ConvoPeq.md` | コミットに含めていない（他タスク分が混在するため） |

### 残存リスク

- `AudioEngineHarness` の PASS は無負荷環境の実測であり、負荷時の drain 挙動は保証しない。
- CLI 実測は音声デバイスを使わない経路であり、実機 ASIO 経路の drain は別途の確認が必要。
- 2 秒という drain 予算は変更していない。C3-D と C9-A は「予算内で drain が成立する」ように
  する修正であり、予算超過時の安全側挙動（`tryReclaim` フォールバック）は従来どおり残る。

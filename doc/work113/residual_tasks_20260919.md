# WORK113 残課題台帳（2026-09-19 時点スナップショット）

Phase 2-2 CLOSED（`3ea32089`）・test-instrumentation cleanup CLOSED（`6ae95149`）・受入記録（`f769a23e`）確定時点の残課題整理。
git 状態実測値: **ahead 3 / behind 0**・push 未実施。ファイル一覧は 2026-09-19 スナップショット（以降の作業で変動し得る）。

---

## 【2026-09-20 更新】本スナップショットの A / C / C-2 はすべて解消済み

実測 git 状態: **ahead 0 / behind 0**（`origin/main` と完全一致・全 commit push 済み）。worktree は clean（`git status --porcelain -uall` = 0、skip-worktree / assume-unchanged = 0、stash 空、進行中の merge/rebase なし）。

### A. push gate — CLOSED

`3ea32089` / `6ae95149` / `f769a23e` を含む全 commit は push 済み。

### C / C-2. 未 commit 分の系列別整理 — CLOSED

当時の「tracked 39 ファイル」は **C-1〜C-11 の 11 commit に系列別で分割され、ファイル数合計がちょうど 39 と一致**することを実測で確認した。

| 系列 | commit | files | subject |
| --- | --- | --- | --- |
| C-1 WORK105 IR runtime contract | `6bca71f6` | 11 | fix(work105): enforce IR runtime shape contract at build |
| C-2 WORK105/F1 processing geometry | `036d31e5` | 3 | fix(work105-f1): follow UI convolver processing geometry |
| C-3 WORK111/112 trim seam | `f7fa1aa8` | 2 | test(work111/112): add measurement hooks for transform-chain audit |
| C-4 WORK109 resample | `adfd29c0` | 1 | test(work109): add resample density trace |
| C-5 WORK113-13/14/15 filter authority | `18740c71` | 5 | fix(work113-13/14/15): establish single filter application authority |
| C-6 WORK107/108 geometry trace | `c89641c4` | 2 | feat(work107/108): add L0 geometry trace |
| C-7 WORK110 L0 correspondence | `106bc737` | 1 | feat(work110): record L0 write-storage correspondence |
| C-8 WORK113-16 EQ projection | `c95cb2ec` | 7 | feat(work113-16): project EQ parameters into RuntimeWorld |
| C-9 WORK113-17 totalGain projection | `8c4fdecd` | 2 | feat(work113-17): project World total gain into the active DSP |
| C-10 WORK104 harness test seam | `489ea22a` | 4 | test(work104): add harness measurement tap and buzz dispatch |
| C-11 WORK113-7b test access | `780a53b6` | 1 | test(work113-7b): expose NUPC L0 frequency-domain getters |
| C-12 evidence | `7ee62cb3` | 11 | chore(evidence): refresh generated verification artifacts |
| C-13 tooling config | `2bd74a68` | 1 | chore(config): add mslearn MCP server |
| C-14 audit docs | `7827560c` | 29 | docs(work103-113): add audit records for WORK103-113 |

C-5 の必須確認も実測済み: `applySpectrumFilter` は定義と `FilterSpec` を保持したまま**呼出のみ停止**（`MKLNonUniformConvolver.cpp:1167`）、IR への HC/LC 焼き込み fan-out も `AudioEngine.Parameters.cpp:668/682` で停止され、単一 Authority は conv 出力段の OutputFilter に集約。根拠 doc `doc/work113/filter_application_implementation_plan_20260918.md` は C-14 に含まれる。

untracked 三点仕分けの帰結: 監査記録群 → C-14、`IRRuntimeContract.h` / `IRTrimTestHooks.h` / `IRRuntimeContractTests.cpp` → C-1 / C-3、`sampledata/**` → housekeeping commit。旧 capture CSV（`7e_*` / `7f0_E_*` / `acc_*` / `irfreq_*`）および `.bak_7f0era` は **worktree に現存しない**（tracked にも ignored にも無し）ため選別判断は消滅。

### B-2. `--buzz-order=` silent fallback — CLOSED（2026-09-20）

fail-closed 化 + regression test を実装・検証済み。**新規明示値 `cte`（ConvolverThenEQ）を導入**（既存仕様ではなく `etc` の頭字語規約の対称適用）。詳細は §B-2。`PublicationValidatorIsolationTests` は別 work item（§F-1）。

### 既知の履歴逸脱（rewrite しない）

`06794615`（2026-09-20 01:50・message は "commit"）は post-series repository housekeeping commit。production `src/` 変更を**含まない**ため C-1〜C-14 の production provenance は汚していない。内容: `evidence/*.json` 137・`ConvoPeq.md`・`.gitignore`（`tmp/` 追加）・`sampledata/{impulse.wav, defaultdataset/**}`・`tools/fix-csdevkit-cpp-project-conflict.ps1` 追加、`_prev01_build_convopeq.cmd` / `_prev01_gate_check.cmd` / `_prev01_run_build.cmd` / `acc_11317b_C2_lc_fixed.csv` 削除。push 済み。**履歴 rewrite は実施しない**（既知の逸脱として固定）。

---

## A. 判断待ち（即時の操作対象）— 2026-09-20 更新で CLOSED（下記は当時の記録）

| 残課題 | 状態 | 次のアクション |
| --- | --- | --- |
| push gate の判断 | `3ea32089`（Phase 2-2）/ `6ae95149`（cleanup）/ `f769a23e`（受入記録）の **3 commits ahead**・behind 0 | push gate 手順での push 実行判断 |

## B. 他オーナーへ引継ぎ済みの OPEN（本クローズ範囲外・記録確定済み）

1. **EQ-on 経路の −5.17dB 減成の帰属** → **EQ DSP owner**（独立 work item）
   - 実測: `0.4912 = 0.891 × 0.5513`、totalGain 0dB・全 band 無効・AGC/engine staging off でも二連続再現・原因未帰属
   - 対応不要部分: rigcheck=eq 校準窓 `[0.486,0.496]` は regression tripwire として固定済み（`6ae95149`）
2. **`--buzz-order=` silent fallback** — **CLOSED（2026-09-20 / B-2）**
   - 実測（欠陥）: 旧実装 `(v == "etc") ? 1 : 0` は `"etc"` 以外をすべて黙って `0` に落とし、`orderMode` の既定 sentinel `-1`（= routing 未変更）を生成しなかった。typo・空値・`"1"` が probe routing を意図せず `ConvolverThenEQ` へ上書きし得た（`--buzz-eq/conv` が 2026-09-19 に誘発した誤測定と同型）
   - 実測（見送り根拠の失効）: リポジトリ内に `--buzz-order=` を呼ぶ script（`.cmd`/`.bat`/`.ps1`/`.sh`）は **0 件**。`test_instrumentation_cleanup_20260919.md:25` の「既存スクリプト互換優先」は committed な呼び出し元で裏付けられなかった
   - 対処: 純関数 `tryParseBuzzProcessingOrder()`（成功時のみ 0/1 を書き、失敗時は `out` を変更しない＝ sentinel 保全）+ CLI 層 fail-closed（`[BUZZ] FAIL` → `exit 2`、既存 `parseOnOff` と同規約）。regression test `runBuzzArgParserTests()`（8 ケース）を AudioEngineHarness 既定スイート先頭に登録
   - **★ 新規明示値 `cte` を導入（既存仕様ではない）**: `cte` はコード・doc とも 0 件で、既存 CLI 仕様として存在しなかった。根拠は既存 `etc` = **E**Q**T**hen**C**onvolver の頭字語という命名規約の**対称適用**で、`cte` = **C**onvolver**T**hen**EQ** = `ProcessingOrder::ConvolverThenEQ`（`src/core/Types.h:11-14` で `0`）。したがって **B-2 で新規追加した CLI 表記**であり、内部 enum 意味（0/1）の変更ではない
   - 受理集合: `cte` → 0 / `etc` → 1 / それ以外（空値・typo・大文字・数値・未知値）→ FAIL（exit 2）
   - 検証: Release build 149/149・`[BUZZ_PARSER] PASS: 8 cases`・AudioEngineHarness フルスイート PASS（exit 0）・CLI 実挙動 5 ケースすべて exit 2（`etc`/`cte` の受理は二重指定ケースで切り分け）
   - 非変更: production `src/`・`CMakeLists.txt`・`tools/build-debug.bat` はいずれも未変更。`PublicationValidatorIsolationTests` の移行・退役は**別 work item のまま**
   - 残観察（別パターン・低リスク・未修正）: `--buzz-probe=` は既に fail-closed。`--buzz-flip-eqgain=` は値を受理して破棄（設計上ステップ固定）。`parseHcIdx`/`parseLcIdx` と `stod`/`stoi`/`stof` は未捕捉例外で terminate
3. **timestamp-based capture**（評価注記・ユーザーレビューで「別課題」認定）
   - cleanup-3 の平均実効レート binding（`out.size() / 2.0s`）は非一様 callback rate を完全補正しない
   - 現行 transition probe の目的には十分（flipIndex 誤差 <0.5% 実測）。必要になった時点で別課題

## C. worktree の未 commit 分（最大の実作業残課題）— 2026-09-20 更新で CLOSED（下記は当時の記録）

実測: **tracked 39 ファイル modified + 大量 untracked**。複数 work item アークにまたがるため系列別整理が必要。

### C-1. tracked modified（系列別 commit 整理）

- **WORK113 系（Phase 2-1/2-2 関連の残り）**: `src/eqprocessor/EQProcessor.h`（Phase 2-1 `applyTotalGainDbNonRt`）・`src/audioengine/RuntimeBuilder.cpp`（totalGain 適用・485 行）・`src/audioengine/RuntimeBuildTypes.h`・`src/audioengine/AudioEngine.h`（bypass mirrors 等を含む大規模変更）・`AudioEngine.Parameters.cpp` / `.DSPCoreDouble.cpp` / `.DSPCoreFloat.cpp` / `.RebuildDispatch.cpp` / `.Timer.cpp`
- **WORK105/F1 系**: `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`（`reprepareUiConvolverForProcessingGeometry` — Phase 2-2 commit では hunk 分離済みで本 hunk が未 commit）
- **convolver/MKL 系**: `src/ConvolverProcessor.h`・`src/convolver/ConvolverProcessor.{Lifecycle,LoadPipeline,LoaderThread,Rebuild,StateAndUI}.cpp`・`src/MKLNonUniformConvolver.{cpp,h}`
- **tests 系**: `src/tests/AudioEngineHarness/{AudioEngineHarness.cpp,AudioEngineHarness.h,PublishPipelineIntegrationTests.cpp}`・`src/tests/BuildErrorClassificationTests.cpp`・`src/tests/NUPCTestAccess.h`
- **その他**: `src/audioengine/BuildErrorPolicy.h`・`src/audioengine/ISRRuntimeSemanticSchema.h`・`CMakeLists.txt`・`opencode.json`
- **evidence/*.json 9 件**: CI verify スクリプトの正常再生成を含む（`isr_runtime_world_identity_report` / `isr_semantic_validity_report` / `publication_atomicity_report` 等）→ commit するか运行時生成物として除外するかの方針確認

推奨: WORK113-15/16/17 のゲート完了済み系列から順に、Phase 2-2 で実施した hunk 分離 commit（`git apply --cached`）の要領で整理。

### C-2. untracked の三点仕分け（commit / 保持 / 廃棄）

- `doc/work103/`〜`doc/work112/`（丸ごと）+ `doc/work113/` の他 15 本（監査記録群）
- `src/audioengine/IRRuntimeContract.h`・`src/convolver/IRTrimTestHooks.h`・`src/tests/IRRuntimeContractTests.cpp`
- 旧 capture CSV: `7e_*`（12 件）・`7f0_E_*`（7 件）・`acc_11315_matrix.csv`・`acc_11317_{t4,t5b,t5c}_capture.csv`・`acc_11317b_{A,B,C1}_*.csv`（境界 jump 証跡）
- `sampledata/defaultdataset/`・`sampledata/impulse.wav`・`irfreq_7d_B.csv` / `irfreq_*.csv` 6 件
- `ConvoPeq.md`（監査用一時ファイル・ユーザー運用）

## D. 環境・運用の改善候補（優先度低・記録済み）

1. **build identity gate の M1/M2 欠陥**（E-G3-3 既知・未修正）
   - M2: commit 毎に `source_revision` 変更 → COHERENCE-4 fail-closed（対処: stamp 再作成・次回 `f769a23e` 基準で再 stamp 済み）
   - M1: ベアシェル/icx 起動時の cache 素字化
   - 恒久修正は別 work item
2. **headroom ランタイム統合** — 4 ランタイム混在
   - 稼働中: pip/Python3.14 + user-site fork litellm 1.81.13（`asyncio.iscoroutinefunction` → `inspect.iscoroutinefunction` パッチ適用済み・`6ae95149` とは別管理・fork 再インストール時は再適用要）
   - uv tool `headroom-ai` 0.37.0: extras=mcp のみで fastapi 無し → proxy 起動不可（統一するなら `uv tool install --force headroom-ai --with fastapi --with uvicorn`）
   - canary: `headroom-proxy.vbs` → `headroom-proxy-start.ps1`（project venv）+ Startup の .lnk / bat / vbs 3 起動体の整理
3. **memory 記録済みの運用手順**: `headroom-runtime-map`（ランタイムマップ+パッチ）/ `convopeq-msvc-build-env`（vcvars+MKL/IPP INCLUDE・stamp 運用・`cmd.exe //c` 注意）

## E. CLOSED（対応不要・対比列挙）

Phase 2-2（mirror = committed-state compatibility projection）・stale mirror audit / mirror transition・boundary jump（T5c 0.0386 現行条件非再現）・test-instrumentation cleanup 4 件・rigcheck=eq criterion 不整合・ConvoPeq.md 版ずれ（現行 = 2026-09-19 15:31:30 版）・headroom DeprecationWarning（litellm パッチ）— すべて監査記録・commit 済み。

## F. 新規棚卸し所見（2026-09-20 追加）

### F-1. `src/tests/PublicationValidatorIsolationTests.cpp` — read-only audit 結果

**判定: MIGRATE CASES THEN RETIRE**（ケース移行後に退役）

実測: 520 行・`TEST_F` 34 件 + `TEST` 4 件 = 38 ケース。2026-06-03（`313efbc3`）追加、最終更新 2026-07-12。include は `RuntimePublicationValidator.h` / `ISRRuntimeSemanticSchema.h` / `RuntimeBuilder.h` / `gtest/gtest.h`。

| 項目 | 実測結果 |
| --- | --- |
| CMake 登録 | **無し**。`add_executable` 全 39 ターゲットに該当なし。2026-06-03 の `313efbc3` で一度登録され、同日 `85ac377c` で削除。以降 8 か月間 未登録 |
| gtest | **リポジトリ内で gtest を使う唯一のファイル**。登録テストは全件 custom `main()` ハーネス方式。CMakeLists / build.bat / CMakePresets に gtest 依存（`find_package` / FetchContent）は一切無し → `#include <gtest/gtest.h>` が解決不能 |
| private アクセス | `validator_.checkNoConflictingTransitions(...)` を **9 箇所**で外部から呼ぶ。当該メソッドは現行 header で `private`（`RuntimePublicationValidator.h:101`）。`FRIEND_TEST` はリポジトリ全体に 0 件 → **コンパイル不可** |
| 意味論の妥当性 | 現行実装と**一致**。`validateResources`（`os ∈ [1,16]` かつ 2 の冪 / `ditherBitDepth ∈ {0,16,24,32}` / `noiseShaperType ∈ [0,3]`）、`checkNoConflictingTransitions`（policy 別 fade 可否、`!active` 時 `useDryAsOld` 拒否、未知 policy 拒否）、`validateTopology`（runtimeUuid==0 と transition/fading の排他、`fadingRuntimeUuid == runtimeUuid` 衝突拒否、policy ∈ [0,2]、processingOrder ∈ [0,1]）、`validateSemanticConsistency` いずれもテスト期待と同一 |
| 直接カバレッジ | `RuntimePublicationValidator` を参照する**登録テストは 0 件**（`git grep -l` で本ファイルのみ） |
| 間接カバレッジ | あり。production は `RuntimePublicationBridge`（`AudioEngine.h:3670` の `validator_->validatePublication`）経由で使用中。`AudioEngineHarness`（`PublishPipelineIntegrationTests` / `SoakPublishIntegrationTests` 等）がこの publish 経路を駆動 |
| `CrossfadeAuthority` | 同ファイルの `CrossfadeAuthorityRegressionTest` 4 ケースが**唯一のカバレッジ**。現行 API（`Decision{needsCrossfade, fadeTimeSec}` / `evaluate(old, new, policy)` — `CrossfadeAuthority.h:23-40`）は**変更なし**で、移植は機械的 |
| 既存テストへの移行実績 | **無し**。`OversamplingNotPowerOfTwo` / `IdentityCollision` / `DryAsOld` といった不変条件名は登録テストに出現しない（= 移植されていない） |
| `tools/build-debug.bat` | L29 が `--target PublicationValidatorIsolationTests` を指定。当該ターゲットは 2026-06-03 に削除済みであり、スクリプト作成（`20c28a2a` / 2026-07-12）時点で**既に無効**。作成時から一度も成立していない stale 参照 |

判定理由: **KEEP + REGISTER** は gtest 非依存化と private アクセス解消のためテスト本体改変が必須で、本段階では不可。**KEEP + RETIRE OLD SCRIPT REFERENCE** は validator 直接カバレッジ（登録テスト 0 件）と `CrossfadeAuthority` 唯一のカバレッジを失うため不可。よって **MIGRATE CASES THEN RETIRE**。

移行方針（未実施・別 work item）:

1. `CrossfadeAuthorityRegressionTest` 4 ケースを custom `main()` ハーネスへ移植（API 不変・最優先）。
2. validator 34 ケースを custom `main()` ハーネスへ移植。`CheckTransition_*` 6 件は `checkNoConflictingTransitions` が private のため**公開 `validatePublication` 経由**へ書き換え、残り 28 件は公開メソッド（`validateTopology` / `validateResources` / `validateSemanticConsistency` / `validatePublication`）を直接使用。
3. 移植完了後にファイルを退役し、`tools/build-debug.bat` の stale 行を除去。

**注意**: 上記は read-only audit の結果であり、テスト本体 / `CMakeLists.txt` / `tools/build-debug.bat` は本段階で一切変更していない。

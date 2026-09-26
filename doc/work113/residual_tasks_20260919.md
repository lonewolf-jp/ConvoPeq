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

1. **EQ-on 経路の −5.17dB 減成の帰属** → **`CustomInputOversampler` owner**（独立 work item）— **B-1 監査完了・原因 ATTRIBUTED**
   - 実測: `0.4912 = 0.891 × 0.5513`、totalGain 0dB・全 band 無効・AGC/engine staging off でも二連続再現・原因未帰属
   - 対応不要部分: rigcheck=eq 校準窓 `[0.486,0.496]` は regression tripwire として固定済み（`6ae95149`）
   - **[B-1] baseline 固定（再現済）**: bare `ratio=0.8846 thd=-149.2dB` / eq `ratio=0.4912 thd=-152.7dB`（inPeak=0.5000 固定）。線形減成（THD −152.7dB）。`0.891` の正体は `kOutputHeadroom = 0.8912509381337456`（−1.0dBFS、DSP 末尾で無条件乗算＋±同値クランプ）。現行 bare 実測は 0.8846 で記録の 0.891 より 0.75% 低い
   - **[B-1] 棄却済み（実測 or 数値再現による）**: EQ 内部 AGC（診断モードで強制 OFF でも不変）・filter structure（Serial/Parallel で不変）・totalGain（`totalGainTarget` 既定 1.0）・staging headroom/makeup（eq 側は 0/0）・saturation（0・線形）・EQ 内部バンド処理（active band 0 個で両構造とも unity）・**stage ② OutputFilter**（自ソースの `makeHPF`/`makeLPF` を数値再現 → 50Hz で 0.987442 = −0.110dB、fs 非依存。実測 −5.11dB と桁違い）
   - **[B-1] 新規識別子: サンプルレート依存**。`--buzz-sr=` 実測で 48000→0.3681 / 96000→0.3683 / 192000→0.4912（bare 比 −7.62/−7.61/−5.11dB）。**純ゲインではなくレート依存要素**だが、Hz 固定コーナーの単純フィルタでは説明不能（全 fs で同値になるはず）
   - **[B-1] 数値の正規化（記録上の注意）**: `[EQLEVEL]` は `outLinear`（peak 絶対値）と `published/input = outLinear/inLinear` の 2 量を出す。本監査で一度この取り違えにより「×0.5 異常」と誤報した（測定異常ではなく集計ミス）。**以降は `published/input` を用いる**
   - **[B-1] stage 別の棄却（すべて直接実測または実測ログによる）**:
     - EQProcessor 単体: base 192k/2048 および RT 相当 768k/4096（AGC 0/1 双方）すべて `ratio=1.000000`（unity）
     - AutoGainPlanner: `[AUTO_GAIN_ANALYSIS] eqMeasuredGainDb=0.00 eqUpperBoundGainDb=0.00 boundExcessDb=0.000` / `[AUTO_GAIN_PLAN] inputHeadroomDb=0.00 outputMakeupDb=0.00 trimDb=0.00`（恒等 EQ では 0dB 計画）
     - World automation: `headroom=1.0 makeup=1.0`、`softClip=1 satAmount=0.1`（閾値非接触・THD 線形）
     - OutputFilter ②: 直接駆動で 6 条件すべて `ratio=0.987442`（−0.110dB、lpMode・prepare レート非依存）→ 自ソース式の数値再現と 3 桁一致
     - bypass blend / crossfade / fade-in: 不発動（`requestedFullBypass=false`、crossfade gain `current=1.0`、fade 10.7ms ≪ 測定窓 2s）
     - `kOutputHeadroom`(0.8912509): publish 点より後・全経路共通
     - `rigcheck=ir` は `convBypassed=1` のまま dry 出力（wet 対照にならない。F-5 参照）
   - **★ [B-1] 主因（ATTRIBUTED）: `CustomInputOversampler` の up/down round-trip 利得欠陥**
     - 直接駆動（DSP を挟まない `processUp → processDown`）: ratio 1→1.000000 / 2→**0.750000** / 4→**0.562500** / 8→**0.421875**。厳密に `0.75^log2(ratio)`、preset（IIRLike/LinearPhase）非依存
     - 実装式: `prepareStage()` が `centerCoeff=0.5` ＋ 非center総和=`0.5` に正規化 → `interpolateStage()` は **conv 位相のみ** `convValue *= 2.0`（位相 DC gain が 0.5 / 1.0 の非対称）→ `decimateStage()` に対応する `×2` が無く両位相を 0.5/0.5 で混合 → **`0.5×0.5 + 0.5×1.0 = 0.75`**
     - 1 stage（ratio 2）の分離実測: `upGain=1.000000` / `downGain=0.750000` → **down（間引き）側に局在**
     - `prepareSingleStage(31, 90.0, …)`（SoftClip 局所2×OS の production 実引数）でも同一（up 1.0 / down 0.75）→ **構築経路に依存しない一様な convention**
     - engine 実測との照合: `0.98379（= inputHeadroom≈0.9963 × ②0.987442） × 0.75^max(log2 effOS, 1)` で 4 条件すべて **0.05% 以内**（eqos1 0.737826 / eqos2 0.737516 / eq@192k 0.553142 / eq@96k 0.414820）
     - **softClip 有効系の構造**: `softClipEnabled=false` → `0.75^log2(effOS)`。`true` → `0.75^max(log2(effOS), 1)`（`effOS==1` では主 OS が非実行の代わりに `softClipOS` 局所2×OS が 1 因子を供給。`AudioEngine.Processing.DSPCoreFloat.cpp:393-415` の分岐）。これにより「OS=1 ≡ OS=2」は**未検証の解決値仮説を用いず**説明できる
     - **契約不整合（確定）**: `isSymmetricUpDown=true` / `isLinearPhaseFIR=true` の宣言、および `AudioEngine.Processing.Latency.cpp` の `static_assert(..., "…symmetric linear-phase FIR with identical up/down taps")` という契約前提と、実装の非対称正規化が矛盾
   - **[B-1] 判定: ATTRIBUTED / ROOT CAUSE LOCALIZED**。**修正は別 work item**（全経路のレベルが最大 +7.5dB 変化し、rigcheck 窓 `[0.486,0.496]` を含む校正値・回帰基準の再取得が必要）。本監査を通じて production 変更 0
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

### F-2. harness 欠陥: `--buzz-rigcheck=eq` の実効 AGC が意図と一致しない（**B-1 帰属とは分離**）

`--buzz-rigcheck=eq` はコメント・文書上「EQ AGC off」を前提にしているが、**実効状態は EQ AGC = ON** である（実測 `gainpath: staging=0 eqAGC=1`）。原因は呼出順:

1. `configureProbeFlatEQ(e)`（`BassBuzzMeasurement.cpp:1370`）が `setEQAGCEnabled(false)` を実行する
2. 直後の `e.setAutoGainStagingEnabled(false)`（同 1374）が内部で `getEQProcessor().setAGCEnabled(!enabled)` = **`setAGCEnabled(true)`** を呼ぶ（`AudioEngine.h:1425`）
3. `getEQProcessor()` は `uiEqEditor` そのもの（`AudioEngine.h:1292`）であるため、1 の AGC OFF が上書きされる

`--buzz-probe` 経路は呼出順が逆（staging → configureProbeFlatEQ）なので AGC OFF で終わる。

- **B-1 の原因ではない**: AGC を強制 OFF にした診断モードでも ratio は 0.4912 のまま完全不変だった（§B-1 の棄却リスト参照）
- したがって `test_instrumentation_cleanup_20260919.md:38` の「EQ AGC off で二連続実測 0.4912」は**実効状態としては誤記**（実際は AGC ON）
- 同ファイル 1371–1373 行のコメント「staging ON だと ratio 0.4912」も実測と矛盾する（実測は staging OFF・AGC ON で 0.4912）
- 本件は **harness cleanup** として B-1 から分離して扱う。既存窓 `[0.486,0.496]` は変更しない

### F-3. EQ dry/wet 混合の潜在欠陥（R2-1 の read-only 所見・未修正）

`EQProcessor.Processing.cpp:980-1010` の bypass 遷移時ブレンド:

```cpp
const bool canBlendDry = (dryCopyBase != nullptr);
const double wetGainState = activeBypassRamp->getNextValue();
const double dryGain = 1.0 - wetGainState;
if (canBlendDry) out = wet * wetGainState + dry * dryGain;
else             out = wet * wetGainState;   // ← dry 補償なしの wet-only 減衰
```

`dryCopyBase` は `bypassTransitionActive` かつ EQ 自身の `dryBypassBuffer` が十分な容量で確保できた場合のみ充填される（同 570-580）。確保できない場合は `else` 側に落ち、**bypass 遷移が終わるまで dry 項なしのレベル低下**が生じる。遷移時のみの潜在欠陥であり、**今回の定常状態の −5.11dB の原因ではない**（定常では `bypassTransitionActive=false` でブレンド自体が実行されない）。記録のみ。

**R2-2（dry buffer / processed buffer / final output の同一 block 比較）は境界超過**: α は `EQProcessor::bypassFadeGain { 1.0 }`（`EQProcessor.h:676`）という **private メンバで公開 getter が存在しない**。DSPCore 側の `bypassFadeGainDouble/Float` と `dryBypassBufferDouble*` は crossfade dry-hold の別機構で、これも private。したがって内部観測には production 側（EQ ヘッダ等）への getter/tap 追加が必要となり、今回の「production src/ 変更禁止」境界を越える。代替案は「実 `EQProcessor` を直接駆動する新規テストの追加」だが、engine と同一の coefficient cache 構築の再現が必要で忠実性が未検証（既存 `EQProcessorMaxGainTests` は係数数学の**再実装**であり実 EQ を駆動していない）。判断は保留。

### F-4. `--buzz-rigcheck=ir` は convolver を有効化していない（harness 欠陥・2026-09-20）

`ir` モードは共通部の `setConvolverBypassRequested(true)` を残したまま `loadImpulseResponse()` するだけで `convBypassed` を解除しません（`BassBuzzMeasurement.cpp:1507` が唯一の設定箇所）。実測の World も `eqBypassed=1 convBypassed=1` で、出力は bypass blend の **dry コピー**（`published/input` が bare と同一の 0.498140）。したがって **`ir` は wet 経路の対照実験に使えません**（B-1 C2-D で「OS 段は無損失」と誤結論した原因）。既存窓 `[0.880,0.897]` は dry 測定に対する基準です。wet 対照が必要な場合は test-only 診断モード `irwet<digit>` を使用します。

### F-5. Serena MCP の python language server 起動不能（環境障害・2026-09-20 修復済）

`solidlsp` は `uvx -p 3.13 --from pyright==1.1.403 pyright-langserver --stdio` を起動しますが、PyPI `pyright` 1.1.403 が提供する LSP 実行ファイル名は **`pyright-python-langserver`** で不一致 → `Failed to spawn: program not found` → `LanguageServerTerminatedException` → 全 Serena ツールが `The language server manager is not initialized` で失敗。
修復: `.serena/project.yml` の `language_servers` を `python` → **`python_basedpyright`**（Serena 1.7.0 対応、`uvx --from basedpyright==1.39.9 basedpyright-langserver --stdio`、実行ファイル名が一致）へ変更。検証: cpp / python_basedpyright / bash の 3 LS すべて起動完了・例外 0。`.serena/project.yml` は untracked のため repo は未変更。
**本障害は B-1 の測定値とは無関係**（当該の「×0.5」は集計ミスであり測定異常ではない。§B-1 の正規化注記を参照）。

### F-6. B-1 計測用 test-only 診断資産（2026-09-20・production 変更 0）

`src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` に追加（`eq` の判定窓は不変）:
- 診断モード: `eqdiag` / `eqdiagser` / `eqos<digit>`（OS 明示固定）/ `irwet<digit>`（conv 有効の真の wet 対照）
- 観測行: `gainpath`（実効 staging/AGC/totalGain/headroom/makeup/struct）/ `[EQ_RTPATH]`（World の eqCoeffHash・eqParams・routing・eqLPFMode・RT cache 実体）/ `[XFADE]`（crossfade runtime 実値）/ `[EQLEVEL]`（`outputLevelLinear` の測定窓中最大値）
- 直接駆動測定（既定スイート登録）: `[EQ_DIRECT]`（実 EQProcessor、base 192k/2048 と RT 768k/4096）/ `[OF_DIRECT]`（実 OutputFilter ②、3 mode×2 rate）/ `[OS_DIRECT]`（実 `CustomInputOversampler` round-trip、ratio 1/2/4/8 × preset、up/down 分離、`prepareSingleStage(31,90)`）
- 計測上の注意: `getOutputLevel()` 等は**信号が流れている間**にサンプリングする（無音時は `measureLevel` が 0 を返す）。`published/input` と `outLinear` を取り違えない

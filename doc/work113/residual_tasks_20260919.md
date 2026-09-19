# WORK113 残課題台帳（2026-09-19 時点スナップショット）

Phase 2-2 CLOSED（`3ea32089`）・test-instrumentation cleanup CLOSED（`6ae95149`）・受入記録（`f769a23e`）確定時点の残課題整理。
git 状態実測値: **ahead 3 / behind 0**・push 未実施。ファイル一覧は 2026-09-19 スナップショット（以降の作業で変動し得る）。

## A. 判断待ち（即時の操作対象）

| 残課題 | 状態 | 次のアクション |
| --- | --- | --- |
| push gate の判断 | `3ea32089`（Phase 2-2）/ `6ae95149`（cleanup）/ `f769a23e`（受入記録）の **3 commits ahead**・behind 0 | push gate 手順での push 実行判断 |

## B. 他オーナーへ引継ぎ済みの OPEN（本クローズ範囲外・記録確定済み）

1. **EQ-on 経路の −5.17dB 減成の帰属** → **EQ DSP owner**（独立 work item）
   - 実測: `0.4912 = 0.891 × 0.5513`、totalGain 0dB・全 band 無効・AGC/engine staging off でも二連続再現・原因未帰属
   - 対応不要部分: rigcheck=eq 校準窓 `[0.486,0.496]` は regression tripwire として固定済み（`6ae95149`）
2. **`--buzz-order=` silent fallback** → 将来の harness cleanup
   - enum-like option のため既存 script 互換優先で未変更・OBSERVATION 記録済み
   - 候補実装: `parseProcessingOrder()` 等の明示的 enum parser 化
3. **timestamp-based capture**（評価注記・ユーザーレビューで「別課題」認定）
   - cleanup-3 の平均実効レート binding（`out.size() / 2.0s`）は非一様 callback rate を完全補正しない
   - 現行 transition probe の目的には十分（flipIndex 誤差 <0.5% 実測）。必要になった時点で別課題

## C. worktree の未 commit 分（最大の実作業残課題）

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

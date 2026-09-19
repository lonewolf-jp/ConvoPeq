# WORK113-C0 — Worktree provenance / commit partition audit（2026-09-19）

**read-only audit**（production source 無修正・reset/stash/clean/add なし・push なし）。
Snapshot: HEAD `f769a23e`、origin/main 比 ahead 3 / behind 0、tracked 39 files +3988/−226、staged 空、untracked 69 エントリ。

## Step 0 — 固定値

```text
git log: f769a23e (HEAD->main) / 6ae95149 / 3ea32089 / 0654e7b5 (origin/main, origin/HEAD) / cf8c286d
rev-list --left-right --count origin/main...HEAD = 0  3
git diff --stat total: 39 files changed, 3988 insertions(+), 226 deletions(-)
git diff --cached: empty
untracked-files=all: 69 entries
```

## Step 1 — tracked 39 files の hunk 別 provenance

### 系列 A — WORK113-16 Phase 1（EQ World projection A1）+ WORK113-17 Phase 2-1（totalGain）

| file | hunks | 内容 | 依存 | 行先 commit |
| --- | --- | --- | --- | --- |
| src/eqprocessor/EQProcessor.h | 1 hunk (9/0) | Phase 2-1 `applyTotalGainDbNonRt`（INV-EQ-GAIN-001/002/004/005） | なし | C-9 |
| src/audioengine/RuntimeBuildTypes.h | 2 hunks (3/0) | Phase 2-1? 記載は Phase 2-1 行も含む — 実際は `#include core/EQParameters.h` + `BuildInput.eqParams` 値 capture（コメント=[WORK113-16 Phase 1]） | core/EQParameters.h は HEAD 済み | C-8 |
| src/audioengine/RuntimeBuilder.cpp | 3 hunks (87/1) | H01=**B**（include IRRuntimeContract.h ★WORK105）/ H02=**A**（eqCoeffHash → cache getOrCreate・coefficient projection）/ H03=**A**（Phase 2-1 totalGain 適用 INV-EQ-GAIN-007） | H01 は untracked IRRuntimeContract.h に依存 | H01→C-1、H02→C-8、H03→C-9 |
| src/audioengine/AudioEngine.h | 4 hunks (47/0) | H01=**A**（EQ cache getOrCreate 37 行）/ H02=**B**（reprepareUiConvolverForProcessingGeometry 宣言）/ H03=**A**（E1 telemetry counter）/ H04=**A**（EQ World projection comment） | H02 は C-2 の定義と対 | H01/H03/H04→C-8、H02→C-2 |
| src/audioengine/AudioEngine.RebuildDispatch.cpp | 1 hunk (4/0) | Phase 2-1? 記載=[WORK113-16 Phase 1] EQ shadow 値 capture | なし | C-8 |
| src/audioengine/AudioEngine.Timer.cpp | 1 hunk (2/1) | `[WORK113-16 Phase 1]` sealed input 引継ぎ（`&currentBuildSnapshot_`） | なし | C-8 |

### 系列 B — WORK105/F1（IR runtime shape contract + UI convolver geometry）

| file | hunks | 内容 | 依存 | 行先 commit |
| --- | --- | --- | --- | --- |
| src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp | 1 hunk (6/1) | WORK105/F1: processing geometry で prepare | C-2 の定義・C-8 の宣言と対 | C-2 |
| src/audioengine/AudioEngine.Parameters.cpp | 5 hunks (33/12) | H01/H02/H03=**B**（OversamplingPolicy include・OS 変更時追従・reprepare 定義）/ H04・H05=**C**（WORK113-13 Phase 1: HC/LC 焼き込み停止） | — | H01-H03→C-2、H04/H05→C-5 |
| src/ConvolverProcessor.h | 9 hunks (49/7) | 全て **WORK105**（knownBlockSize / IR 形状追跡・RuntimeBuilder 検証用） | — | C-1 |
| src/convolver/ConvolverProcessor.Rebuild.cpp | 2 hunks (3/1) | WORK105（incremental 経路の 0=不明 刻印） | — | C-1 |
| src/convolver/ConvolverProcessor.LoadPipeline.cpp | 7 hunks (8/4) | WORK105（knownBlockSize plumbing） | — | C-1 |
| src/convolver/ConvolverProcessor.Lifecycle.cpp | 3 hunks (18/2) | WORK105（M-1 再利用の形状契約） | — | C-1 |
| src/convolver/ConvolverProcessor.LoaderThread.cpp | 18 hunks (194/31) | **混在**: H03/H04=WORK105（同期パス刻印）/ H01=WORK111（IRTrimTestHooks include）/ H15/H16=WORK111（tail fade hook）/ H02=WORK112（limits）/ H05・H10-H14・H17・H18=WORK112（transform chain checkpoint）/ H06-H09=WORK109（resample 世代ペア・密度） | H01 は untracked IRTrimTestHooks.h に依存 | WORK105 分→C-1、WORK111/112 分→C-3、WORK109 分→C-4（hunk 分離要） |
| src/audioengine/BuildErrorPolicy.h | 3 hunks (6/1) | WORK105（IRRateMismatch / IRBlockMismatch 追加） | — | C-1 |
| src/tests/BuildErrorClassificationTests.cpp | 8 hunks (10/7) | WORK105（8→10 value matrix） | BuildErrorPolicy.h と対 | C-1 |

### 系列 C — WORK113-13/14/15（filter application 単一 authority・NUC HC/LC 停止）

| file | hunks | 内容 | 依存 | 行先 commit |
| --- | --- | --- | --- | --- |
| src/audioengine/AudioEngine.Parameters.cpp | H04・H05 | WORK113-13 Phase 1（setConvHC/LCFilterMode の IR 焼き込み停止） | doc/work113/filter_application_implementation_plan §6/§10 をコメント引用 | C-5 |
| src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp | 4 hunks (21/9) | 全て [WORK113-15]（① conv 出力段 HC/LC exactly once・② EQ final stage） | `outputFilter` は HEAD 済み（AudioEngine.h:968） | C-5 |
| src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp | 5 hunks (23/9) | 同上（float path） | 同上 | C-5 |
| src/MKLNonUniformConvolver.cpp | 8 hunks (118/3) | **混在**: H03=**WORK113-14 Phase 2**（NUC 側 HC/LC 適用停止）/ H01・H04-H07=**WORK107/108**（geometry trace）/ H02=**WORK110**（110-2 L0 trace）/ H08=**WORK113 RT telemetry** → **C-8**（意味論上の所属: telemetry は Observer の観測責務・WORK113-16/17 の committed-state observation アーク。C-6 への同梱は不可 — ユーザー確定 2026-09-19） | — | H03→C-5、H01/H04-H07→C-6、H02→C-7、**H08→C-8** |
| src/convolver/ConvolverProcessor.StateAndUI.cpp | 1 hunk (5/2) | WORK113-13 Phase 1（nucHC/LCMode を structural hash から除外） | — | C-5 |

### 系列 D — Test / WORK104 / WORK113-7B / tooling

| file | hunks | 内容 | 行先 commit |
| --- | --- | --- | --- |
| src/tests/AudioEngineHarness/AudioEngineHarness.cpp | 3 hunks (24/1) | WORK104（測定タップ seam） | C-10 |
| src/tests/AudioEngineHarness/AudioEngineHarness.h | 5 hunks (14/0) | WORK104（HarnessTapFn / tapMutex_） | C-10 |
| src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp | 2 hunks (5/0) | WORK104（--buzz dispatch） | C-10 |
| src/tests/NUPCTestAccess.h | 1 hunk (20/0) | WORK113-7B（L0 周波数領域内部 buffer getter） | C-11 |
| CMakeLists.txt | 2 hunks (13/0) | H01=**WORK105**（IRRuntimeContractTests target）/ H02=**WORK104**（BassBuzzMeasurement を harness に追加） | H01→C-1、H02→C-10 |
| opencode.json | 1 hunk (5/0) | mslearn MCP server 追加（tooling config・work item 外） | C-13 |
| src/audioengine/ISRRuntimeSemanticSchema.h | 2 hunks (6/0) | [WORK113-16 Phase 1] A1 契約（CoefficientSemantic への EQ projection） | C-8 |

## Step 2 — evidence/*.json（実測 11 ファイル・全て生成物）

| 生成元 | ファイル | mtime |
| --- | --- | --- |
| CI verify（PowerShell・本日 16:26 実行分） | isr_runtime_world_identity_report / isr_semantic_validity_report / publication_atomicity_report | Sep 19 16:26 |
| engine runtime export（harness/soak 実行分） | epoch_reclaim_audit / evidence_manifest（ISREvidenceExporter・MainApplication）/ retire_timeline / retire_trace_shutdown_last（ReleaseResources）/ shadow_compare_cadence（ISRDebugRuntime）/ shutdown_trace（+isr-generate-runtime-evidence.ps1）/ world_lifecycle_audit（WorldLifecycleAudit）/ world_retirement_telemetry（Commit/Timer） | Sep 19 19:23 |

**方針（第一候補）**: 全 11 ファイル = **generated verification artifact**。production/work-item commit に混ぜず、**`chore(evidence): refresh generated artifacts` として独立 commit**（evidence/ は tracked のため .gitignore 変更は不要・実施せず）。

## Step 3 — untracked 69 エントリの分類

### ② production/test source → commit 候補（依存関係あり）

| file | work item | 依存元（commit 済みコードからの参照） | 判定 |
| --- | --- | --- | --- |
| src/audioengine/IRRuntimeContract.h | WORK105 | RuntimeBuilder.cpp（uncommitted include）・IRTrimTestHooks.h・IRRuntimeContractTests.cpp が include | **C-1 に必須**（先に commit） |
| src/convolver/IRTrimTestHooks.h | WORK111 | LoaderThread.cpp（uncommitted include）・BassBuzzMeasurement.cpp（commit済み）が include | C-3 に必須 |
| src/tests/IRRuntimeContractTests.cpp | WORK105 | CMakeLists.txt H01 が add_executable | **C-1 に必須** |

### ① 正式な監査記録 → commit 候補（全 28 ファイル）

- doc/work103: bass_noise_investigation / verification_round2（2）
- doc/work104: bass_buzz_measurement + analyze_buzz_csv.py（2）
- doc/work105: ir_runtime_contract_and_remeasure（1）
- doc/work106: e_isolation_delta_probe（1）
- doc/work107: l0_fdl_correspondence（1）
- doc/work108: effective_rate_origin（1）
- doc/work109: ir_resample_target_content_density（1）
- doc/work110: l0_partition_fdl_correspondence（1）
- doc/work111: real_ir_tail_fade_impact（1）
- doc/work112: real_ir_transform_chain_audit（1）
- doc/work113（16）: block_rate_sidebands / correct_reference_rebaseline_7f0 / engine_null_impulse_baseline / fdl_irfreq_ifft_interpretation / filter_application_{architecture_audit, contract_freeze, contract_reconciliation, implementation_plan, single_authority_design} / filterspec_{freqresponse_ccs_integrity, numpartsir_orthogonalization} / l0_internal_seam_7b / l0_temporal_continuity / nuc_standalone_isolation / phase1_state_ownership_migration / phase2_nuc_hc_lc_removal / residual_tasks_20260919
- 特記: `filter_application_implementation_plan_20260918.md` は **production コメント（Parameters.cpp H04/H05）から §6/§10 として参照される** → C-5 と同期 commit 推奨

### ③ 再現証跡（保持 / 必要最小限のみ commit 候補・合計 333MB）

| group | 件数/size | 参照 |
| --- | --- | --- |
| acc_11317b_{A,B,C1} | 3 × 23MB | 境界 jump 証跡（C-0 監査） |
| acc_11317_{t4,t5b,t5c} | 3 × 22MB | WORK113-17 T4/T5 系列 |
| 7f0_E_*.csv（.bak 除く 5 件） | 5 × 27.5MB | doc/work113（fdl_irfreq_ifft_interpretation・filterspec_freqresponse・contract_freeze）が引用 |
| 7e_*.csv | 12 × 0.06-0.3MB | 同上（7e 系列） |
| irfreq_*.csv | 6 × 0.05MB | 同上 |
| acc_11315_matrix.csv | 1KB | WORK113-15 |
| sampledata/impulse.wav + defaultdataset/ | harness 既定 IR 入力（BassBuzzMeasurement が参照） | commit 候補 |

### ④ 一時生成物 → 廃棄候補（削除はしない・一覧化のみ）

- `7f0_E_P1_s40.bak_7f0era`（28MB）/ `7f0_E_P1_s50.bak_7f0era`（28MB）— 世代 backup

## Step 4 — ConvoPeq.md freshness gate

```text
baseline Generated : 2026-09-19 18:24:23
NEWER_SRC_COUNT    : 2
STATUS             : STALE
  newer: src\tests\AudioEngineHarness\BassBuzzMeasurement.cpp  (19:22:25)
  newer: src\tests\AudioEngineHarness\TransitionMetrics.h      (19:12:02)
```

→ **再生成が必要（記録のみ・C-0 では実施せず）**。cleanup 2 ファイルが baseline 生成後に修正されたため。次回 `python output_sourcecode_markdown.py` 実行で解消。

## Step 5 — provenance matrix / 次の commit 単位の列挙

| commit 単位 | 系列 | 含むファイル（hunk） | 依存・順序 |
| --- | --- | --- | --- |
| C-1 `fix(work105): IR runtime shape contract` | B | IRRuntimeContract.h (新) / ConvolverProcessor.h / Rebuild / LoadPipeline / Lifecycle / LoaderThread(Work105 hunk) / BuildErrorPolicy.h / BuildErrorClassificationTests.cpp / IRRuntimeContractTests.cpp (新) / CMakeLists H01 | 自己完結・**最初に commit**（RuntimeBuilder が include） |
| C-2 `fix(work105-f1): UI convolver processing-geometry follow` | B | Parameters H01-H03 / AudioEngine.h H02 / PrepareToPlay hunk | 自己完結 |
| C-3 `test(work111/112): measurement hooks + transform chain audit` | — | IRTrimTestHooks.h (新) / LoaderThread の WORK111/112 hunks / （WORK109 hunks は C-4 またはここに同梱） | LoaderThread hunk 分離要 |
| C-4 `test(work109): resample density trace`（C-3 と統合可） | — | LoaderThread WORK109 hunks | 同上 |
| C-5 `fix(work113-13/14/15): filter application single authority` | C | Parameters H04/H05 / DSPCoreDouble / DSPCoreFloat / MKLNonUniformConvolver.cpp H03 / StateAndUI hunk / doc/work113/filter_application_implementation_plan（参照 doc 同期推奨） | outputFilter は HEAD 済み・自己完結 |
| C-6 `feat(work107/108): L0 geometry trace` | — | MKLNonUniformConvolver.h / MKLNonUniformConvolver.cpp H01/H04-H07（H08 要判断） | 自己完結 |
| C-7 `feat(work110): L0 write↔storage trace` | — | MKLNonUniformConvolver.cpp H02 | 自己完結 |
| C-8 `feat(work113-16-phase1): EQ World projection A1` | A | RuntimeBuildTypes.h / AudioEngine.h H01/H03/H04 / RebuildDispatch / Timer / ISRRuntimeSemanticSchema.h / RuntimeBuilder.cpp H02 | **C-1 の後**（RuntimeBuilder include 前提） |
| C-9 `feat(work113-17-phase2-1): totalGain world projection` | A | EQProcessor.h / RuntimeBuilder.cpp H03 | C-8 の後（同 file hunk 分離） |
| C-10 `test(work104): harness tap seam + buzz dispatch` | D | AudioEngineHarness.cpp/h / PublishPipelineIntegrationTests.cpp / CMakeLists H02 | 自己完結 |
| C-11 `test(work113-7b): NUPC L0 getters` | D | NUPCTestAccess.h | 自己完結 |
| C-12 `chore(evidence): refresh generated artifacts` | — | evidence/*.json 11 ファイル | work-item commit と分離 |
| C-13 `chore(config): opencode mslearn mcp` | — | opencode.json | 任意 |
| C-14 `docs(work103-113): audit records` | ① | doc/work103-112（11）+ doc/work113 16 本 | C-5 の参照 doc 同期に注意 |

**loader 系（LoaderThread.cpp）のみ 4 系列混在のため hunk 分離が必須**。それ以外は file 単位または 2-hunk 分離で適用可能。

## C-0 完了条件

```text
[PASS] tracked 39 files → hunk provenance 確定（上表・全 39 file / 約 70 hunk 分類完了）
[PASS] untracked → commit / retain / discard 分類（②3 file・①28 file・③7 group・④2 file）
[PASS] evidence/*.json → 全 11 ファイルが generated artifact と確定（独立 commit 方針）
[PASS] 次の commit 単位を列挙（C-1〜C-14・依存順 C-1 → C-8 → C-9 など）
```

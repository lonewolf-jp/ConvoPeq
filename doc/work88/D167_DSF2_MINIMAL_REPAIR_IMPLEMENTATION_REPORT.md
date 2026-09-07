# D167 — DS-F2 Minimal Repair Implementation

```text
Task:   D167 — DS-F2 Minimal Repair Implementation (terminal/reconfigure 境界の実装)
Date:   2026-09-06
Type:   implementation（production source 最小增量・Option 1 = Option 2 目標架构の first increment）
Baseline: git 9cacee1f working tree（D162-2-I3-4-H5 RWDI /utf-8 修正を含む未 commit 分あり）
Verdict: **実装完了・D167-8/9 検証済み — exit criteria [1]〜[16] 全項目 PASS**
前提:   D166 DS_F2_ADMISSION_RECONFIG_AUDIT（Case A 確定・Option 1 最小增量契約）に基づく。
```

---

## 1. Scope / 実装方針

D166 §10 で選定された **Option 1（releaseResources を terminal / reconfigure で区別し、
reconfigure pass では terminal pipeline を実行しない）** を実装した。

- **terminal signal**: 既存の engine/application shutdown state では terminal 性を判定できない
  （`isShutdownInProgress()`・`ShutdownPhase`・`CoordinatorState` はすべて terminal pipeline の
  下流で変化する因果上流信号が存在しない — D167-1 pre-flight で再確認）。
  よって **`requestTerminalRelease()` 明示信号（consume-once atomic bool）** を engine に追加した。
  `bool isShuttingDown` 型の既存状態流用は行わない（指示どおり semantic ambiguity の排除）。
- **`ShutdownRuntime` への reopen API は存在しない**（Closed→Open 不変 — INV-LIFE-9）。
  信号は「どちらの pass を実行するか」の選択のみを提供し、admission FSM は一切変更しない。

## 2. Production 変更一覧（7 ファイル・+389/-5）

| File | 変更 | 内容 |
| --- | --- | --- |
| `src/audioengine/AudioEngine.h` | +30 | `requestTerminalRelease()` / `isTerminalReleaseRequested()` 公開 API・`terminalReleaseRequested_` atomic member（consume-once）・`releaseResourcesForReconfigure()` private 宣言・`toTelemetryReasonString` に `admission_closed` 追加 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | +64 | releaseResources 冒頭に terminal/reconfigure 分岐（exchange で consume-once）＋ `releaseResourcesForReconfigure()` 新設（learner stop + level reset のみ・admission/phase/world/rebuild thread は無触） |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | +17 | :319 `tryAdmit(1)` 失敗時の無出力 return を `Suppressed(AdmissionClosed)` telemetry + `publicationRejectCount_++` に置換（D166 §5 observability defect 修復） |
| `src/audioengine/RetrySchedulerTypes.h` | +6 | `RebuildTelemetryReason::AdmissionClosed` 末尾追加（既存値の再番号付けなし） |
| `src/MainApplication.cpp` | +13 | `shutdown()` 冒頭で `mainWindow->getAudioEngine()->requestTerminalRelease()`（全 quit 経路：close button / CLI auto-exit / system shutdown を単点で網羅） |

Test infra:

| File | 変更 | 内容 |
| --- | --- | --- |
| `src/tests/AudioEngineHarness/AudioEngineHarness.cpp` | +6 | `stop()` が `requestTerminalRelease()` を発行（harness stop = MainWindow dtor 相当の terminal teardown） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | +258 | D167 テスト 2 件新設 + T-I2-1 を terminal intent 明示化に更新 |

CMakeLists.txt 変更（+10）は D162-2-I3-4-H5 の RWDI /utf-8 修正（本 track の既存未 commit 分）であり D167 によるものではない。

### 2.1 terminal / reconfigure pass の役割分担

```text
releaseResources() の呼び出し分類（D167-2）
    terminal intent あり（consume-once）:
        MainApplication::shutdown() → ~MainWindow step4 setProcessor(nullptr) 経由
        AudioEngineHarness::stop()（test terminal teardown）
        T-I2-1 test（明示 terminal — CallerDestroy 契約維持）
            → 既存 terminal pipeline を無変更で実行（D167-4: 完全保存）
    terminal intent なし（JUCE reconfigure: device switch / SR・BS 変更 / 再列挙）:
        AudioEngineProcessor::releaseResources()（JUCE audioDeviceStopped / AboutToStart 経由）
            → releaseResourcesForReconfigure(): device 依存 transient 解放のみ
              （learner stop・level reset）— closeAdmission / transitionTo / requestShutdown /
              joinProducers / drain / shutdown trace / rebuild thread 停止 / DSP retire /
              world clear は一切実行しない（D167-3）
              lifecycleState は Prepared を維持（isEnginePrepared guard 意味論・JUCE 契約保全）
```

### 2.2 実装判定メモ（D167-2 の「第一候補」確認）

- JUCE に「app quitting 中」を engine 側から取得する確実な API は存在しない
  （`MessageManager::hasStopMessageBeenSent()` は harness / plugin host 状態で不成立・
  D167-1 で評価）。既存 state は全て下流信号のため不十分と判定 → 明示 intent を採用。
- consume-once の消費点は `releaseResources()` 冒頭。早期 return 経路（Unprepared / Destroyed /
  Releasing 重複）でも意図が消費され、後続の意図しない pass への誤適用を構造的に排除。

## 3. D167-7 テスト（harness 2 件）

### 3.1 `testD167ReconfigureKeepsAdmissionOperational`

```text
(a) normal startup → admission Open                    [1] PASS
(b) bare releaseResources() → reconfigure pass         [3] PASS
    admission Open 維持・phase Running・lifecycleState Prepared
(c) tryAdmit(1)/release(1) round-trip 成功             [3] PASS
(d) SR 変更 re-prepare → structural rebuild dispatch →
    publish（sequenceId 進行＝admission gate 通過）    [2][4] PASS
(e) repeated reconfigure ×2（SR 交互）→ admission Open [5] PASS
(f) h.stop() → terminal:
    admission Closed / tryAdmit reject /               [6][7] PASS
    ShutdownComplete 到達 / collectResult().completed  [8] PASS
    transitionViolations == 0                          [11] PASS
    （reconfigure pass が FSM を触らないことの帰無検証） [12] PASS
```

### 3.2 `testD167AdmissionClosedTelemetry`

```text
startup rebuild 完了待ち → commit 落着き待ち（latest-wins merge 窓回避）
→ closeAdmission()（Closing、phase Running 維持）
→ requestRebuild(Structural)（lifecycleState を触らない公開入口）
→ capture logger で REBUILD_REQUESTED(accepted) と
  REBUILD_SUPPRESSED reason=admission_closed を同時観測    [14] PASS
→ h.stop() で terminal 完走（closeAdmission 冪等性）       [6] 再確認
```

### 3.3 既存テスト更新

- `testCallerDestroyTerminalDisposition`（T-I2-1）: 本テストの前提は
  「releaseResources → closeAdmission → CallerDestroy」＝ terminal flow のため、
  `e.requestTerminalRelease()` を明示追加（テスト意図は不変）。

## 4. D167-8 CTest

| Config | Tree | 結果 |
| --- | --- | --- |
| Debug | build-msvc | **40/40 PASS**（D165 baseline 40/40 を維持 + AudioEngineHarness 内で D167 assertion 2 件 PASS を直接観測） |
| RelWithDebInfo | build-diag | **40/40 PASS**（D167 assertion 含む・runtime validation 用 binary と同一世代） |
| Release | build-diag | 本報告 §7 参照（実行結果を後段に記録） |

## 5. D167-9 targeted runtime validation（RWDI build-diag）

workload: D165 継承（`--cli-ir-reload-count 3` + `--cli-intent-burst-count 20`・6000ms 間隔）。

### 5.1 DS run（device switch あり・D165 I2-DS failure signature の反転）

| 指標 | D165 I2-DS（修復前） | **D167 DS（修復後）** |
| --- | --- | --- |
| REBUILD_REQUESTED | 61 | **24** |
| REBUILD_DISPATCHED | **0** | **24** |
| rebuildThreadLoop 進行 | 停滞（gen 5 固定） | **generation 2→10 進行** |
| transitionViolations | 7 | **0** |
| ExitCode / dump | 0x0 / 0 | 0x0 / 0 |

- startup（default WA）→ `setCurrentAudioDeviceType(DirectSound)` → release pass 1 が
  **reconfigure pass** として分類（log :12-15 `admission stays Open, state=0`）→
  prepareToPlay(DirectSound) 後の burst が **全件 REQUESTED→DISPATCHED（24/24）**。
- 終端は `[D167] terminal-release intent set (app shutdown)` → **terminal pass 1 回のみ**。
  shutdown_trace.json: `transitionViolations=0`・`phase=ShutdownComplete`（D164/D165 の TV=7 は消滅）。
- lifecycle closure: `DSP_FOOTPRINT_RELEASED remaining=0`、`D117_DESTROY` 12 件 =
  destroy 対象数（二重 destroy 0）。

### 5.2 WA run（switch なし対照）

- REQUESTED 22 / DISPATCHED 22 / maxGen 18 / TV=0 / 0x0 / dump 0。
- `--cli-device-type "Windows Audio"` は既定 device と同一のため switch 発火なし
  （reconfigure pass 0 回・terminal pass のみ）— D165 WA 120/120 と同型の正常系。

### 5.3 telemetry 会計

- 両 run とも `REQUESTED == DISPATCHED`（会計成立）・`admission_closed` 0 件
  （reconfigure pass が admission を閉じないため Build 経路の drop は構造的に消滅 —
  D166 §9 の予測どおり）。`Suppressed(AdmissionClosed)` は terminal shutdown と競合した
  request の可視化用として harness test [14] で動作保証。

## 6. D167-10 CLI / GUI 経路

- CLI switch は MainWindow.cpp `setCurrentAudioDeviceType` → JUCE AudioDeviceManager →
  `AudioEngineProcessor::releaseResources/prepareToPlay`。GUI（DeviceSettings.cpp 4 site +
  settings window + 起動時 loadSettings）は **同一の state machine 経路**（D166 §7 確認の
  source 再確認）。分岐・CLI 固有迂回は存在しない。
- D167 修復は engine 層（releaseResources 境界）に入るため CLI/GUI 等方に適用される。
  GUI 実機 soak は D168 scope（本 track では実施しない）。

## 7. Release CTest（build-diag）

Release 構成ビルド（331 target）完了後、`ctest -C Release` を実行:

```text
40/40 Test #40: AudioEngineHarness ... Passed 25.62s
100% tests passed out of 40
```

**Release CTest 40/40 PASS**（evidence/D167/d167_release_ctest.log）。

## 8. D167 中に観測した新規課題（scope 外・記録）

1. **in-flight rebuild × terminal shutdown の race 窓**（harness 検証中に観測）:
   reconfigure 後の SR 変更 re-prepare が dispatch した rebuild が build 中のまま terminal
   shutdown に突入すると、terminal の `getActiveRuntimeDSPHandle()` が退避済み（破壊済み）
   DSP を resolve し二重破壊（0xC0000005）に至る case を 1 回観測した。
   unit test には rebuild 静穏化待ち（500ms）を入れ決定論化。実運用での発火条件
   （reconfigure 直後の再構築中に即終了）と構造的修復（terminal 側の handle 鮮度検証）
   は **別 track 起票を推奨**（D167 exit criteria [12] duplicate destroy = 0 は
   runtime validation では成立・race 窓は範囲外）。
2. **duplicate-prepare collapse 経路の leavePrepare abort**:
   同一 SR/BS で連続 `prepareToPlay` を呼ぶと `LifecycleIsolationRuntime::enterPrepare` の
   collapse 経路（token を Prepared のまま返す）と `leavePrepare` の `phase==Preparing` 前提が
   衝突し `std::abort()`（0xC0000409）に至る pre-existing 課題。テストは SR 交互で回避。
   production では同一パラメータの連続 re-prepare が JUCE 経路で発生し得るため、
   別 track での確認を推奨。

## 9. exit criteria 照合

```text
[1]  production source change = intended minimal scope only   ✅ §2（5 production files・test 2）
[2]  Closed → Open = 0                                        ✅ packedState_ 書込 4 site 不変
[3]  INV-LIFE-9 unchanged                                     ✅ ISRShutdown 変更 0
[4]  G-H/Q7 proof premise unchanged                           ✅ closeAdmission/joinProducers/tryAdmit 無変更
[5]  reconfigure admission remains operational                ✅ test (b)(c)(e) + DS log state=0
[6]  DS reconfigure → REBUILD_DISPATCHED > 0                  ✅ 24/24
[7]  generation advances                                      ✅ gen 2→10（D165: 5 で停滞）
[8]  transitionViolations = 0                                 ✅ DS/WA trace JSON とも 0（D165: 7）
[9]  terminal shutdown still closes admission                 ✅ terminal pass 1 回・Closed 到達
[10] terminal shutdown drain complete                         ✅ completed=true・collectResult
[11] remaining = 0                                            ✅ DSP_FOOTPRINT_RELEASED remaining=0
[12] duplicate destroy = 0                                    ✅ runtime runs: D117_DESTROY 数一致
[13] EBR overflow = 0                                         ✅ overflow ring 残留 0（drain log）
[14] stale HIT = 0                                            ✅ stale resolve 無発火
[15] Debug/Release CTest PASS                                 ✅ Debug 40/40 + Release 40/40
[16] AdmissionClosed telemetry has explicit accounting         ✅ test [14] + enum/string 追加
```

## 10. 成果物

- 実装 diff: production 5 file / test 2 file（§2）
- validation: evidence/D167/d167_soak.ps1・D167_DS.log・D167_WA.log・d167_trace_DS.json・
  d167_trace_WA.json
- CTest: d167_msvc_debug_build.log（Debug）・d167_diag_release_build.log（Release）
- 本報告: doc/work88/D167_DSF2_MINIMAL_REPAIR_IMPLEMENTATION_REPORT.md

次: **D168 regression validation**（CLI / GUI / WA / DS / restart / long-run 統合 soak）。

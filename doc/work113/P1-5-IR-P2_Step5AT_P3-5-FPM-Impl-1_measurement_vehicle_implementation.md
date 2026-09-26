# P1-5-IR-P2 — Step 5-AT / P3-5-FPM-Impl-1: Measurement Vehicle Implementation Gate

- **作成**: 2026-09-24 / work113 P1-5-IR Phase 2（P3-5-FPM-Impl-1）
- **種別**: **test-only implementation gate**。
  test source 追加・変更のみ。production 0・public API 0・getter/counter 0・
  shutdown／epoch／reclaim／Recovery semantics 0・CMake target 追加 0・実計測 0。
- **判定**: **FPM-IMPL-1-PASS**（実装＋build 確認まで。M0/M1/M2 未実行）。
- **入力**: R33-A-PASS＋R33-B-PASS＋R34-PASS（判定 C）＋FPM-PREP-1-PASS＋FPM-IMPL-PREP-PASS。
- **STOP 後停止**: 本 gate 完了後は実測に進まず STOP。次は FPM-Run-1（実行 gate・別指示）。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a（R33-A〜FPM-Impl-Prep1 と同一）
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF（継承）
ConvoPeq.md = 2026-09-23 23:33:38, Length 5535334（本 gate 開始時に再実測・同一）
R16/R19/R22/R27 counters／R23 vehicle／R30・R32 vehicle 保持（変更なし）
FPM-Impl-1 の差分 = test 2 件＋本ドキュメント（§3）
```

検証手段：Read／Grep＋test source 編集＋build（compile／link／target existence／CLI dispatch 確認）。
M0/M1/M2 の実行は行っていない。

---

## 2. Latest ConvoPeq.md Reconciliation

実装開始時に 7 領域を live source で再確認（FPM-Impl-Prep1 §2 の authority と一致・変更なし）：

```text
AudioEngineHarness（AudioEngineHarness.cpp/h）：start／stop（stopAudioOnly→
  requestTerminalRelease→releaseResources terminal pass）／stopAudioOnly／startAudioOnly。
  変更なし。
P1PolyphaseGainCharacterization（R30/R32 vehicle）：episode-driving（:1185-1242）／
  snap（p1RecReadSnap :1077-1093）／drain emit（p1RecDrainFields :1104-1120）／
  terminal capture（:1254-1273）。既存判定 terminalOk は変更なし（§10 境界）。
PublishPipelineIntegrationTests（CLI 分岐 :1124-1209）：--p1-recovery-origin 等の先例。
  本 gate で --fpm-m0／--fpm-m1／--fpm-m2 を同分岐に追加（test のみ）。
ISRShutdown（ISRShutdown.h/cpp）：getPhase／collectResult／transitionTo／
  isTerminalPhase。変更なし。
Threading（Threading.cpp:109-247）：collectDrainAudit／isFullyDrained／waitForDrain。
  変更なし。
ReleaseResources（:615-744）：waitForDrain→markTimedOut→finalize→ShutdownComplete。
  変更なし。
Coordinator（Coordinator.cpp:511-562）：P9-P22（P22 liveCount==0）。変更なし。
```

---

## 3. Implementation Diff

test source の変更のみ（production／CMake／build 設定の変更なし）：

```text
M  src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp（+約290 行・FPM vehicle 本体）
M  src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp（+14 行・CLI 宣言＋分岐）
```

`git diff --stat` 上の `src/audioengine/*`・`BassBuzzMeasurement.cpp` 等の差分は
R27 等の既存差分であり、本 gate の変更ではない（§11 で監査）。
`git diff --stat` 上の P1PolyphaseGainCharacterization の +1126 行表示は
既存 working-tree 差分（R30/R32 vehicle 等）を含む累積値であり、本 gate の新規は約 290 行。

### 本 gate の新規内容

1. 共通 T3b capture（`FpmT3bCapture`＋`fpmCaptureT3b`＋`fpmEmitT3b`＋`fpmClassify`）：
```text
phase = e.isrShutdownRuntime().getPhase()
result = e.isrShutdownRuntime().collectResult(Healthy, 0)
fullyDrained = e.isFullyDrained()
audit = e.collectDrainAudit()（diagnostic のみ）
t3b = phaseComplete && result.completed
   && result.blockingReason == ShutdownBlockingReason::None
   && result.transitionViolations == 0 && fullyDrained
class = t3b ? VALID : (phaseComplete ? INVALID-TERMINAL : ABORTED)
```
2. 共通 helper（`fpmStartupSettle`・`fpmSingleRecoveryEpisode`）：R30 手順の最小流用。
3. `runFpmM0`／`runFpmM1`／`runFpmM2`（§4-§6）。
4. CLI 宣言（`int runFpmM0();` 等 3 行）＋分岐（`--fpm-m0／--fpm-m1／--fpm-m2` 各 2 行）。

---

## 4. M0 Implementation

```text
start（h.start kSr/kBlock）→ fpmStartupSettle（authoritative runtime＋seq 前進＋backlog 0）
→ stop（h.stop）→ fpmCaptureT3b → fpmEmitT3b → summary（t3b＋class）
```

- Recovery なし。publication／crossfade／retire の追加操作なし。
- M0 が INVALID-TERMINAL でも exit code は 0（sequence 完遂を示す。T3b 成否は class で判定・捏造しない）。
- ABORT 経路（harness start 失敗／settle 失敗）は exit code 2。

---

## 5. M1 Implementation

```text
start → fpmStartupSettle → sleepPump(1000) → fpmSingleRecoveryEpisode（episode=1）
→ sleepPump(500) → stop → T3b capture → summary（episodeOk＋t3b＋class）
```

- episode-driving は R30 の pre snap→submitRecoveryIntent→seq 前進待ち（20s budget）→
  post snap→dSeq／dCoord／dCmt／dTake／dBld 確認の最小流用（`kEpisodes=1`）。
- **E2 winner なし**（winner／liveCount の露出なし）。
- **Running 中 waitForDrain なし**（R31 STOP-1 契約・R30 R32 修正の継承）。

---

## 6. M2 Implementation

```text
start → fpmStartupSettle → sleepPump(1000) → publish-1（requestRebuild Structural＋seq 前進確認）
→ crossfade settle（audit.activeCrossfadeCount==0・diagnostic 条件）
→ recovery episode（§5 と同一）→ publish-2（requestRebuild Structural＋seq 前進確認）
→ sleepPump(500) → stop → T3b capture → summary
```

- publish 操作は `requestRebuild(convo::RebuildKind::Structural)`（D167-5 と同一の公開入口・
  lifecycleState を触らない）。audio 実行中の `prepareToPlay` は harness 契約上使用しない
  （実装中に初版の prepareToPlay 案を同理由で requestRebuild に修正・test のみ）。
- crossfade settle は diagnostic 条件であり shutdown success authority ではない（T3b が authority）。
- 音質・buzz・limiter・NUC は扱わない（pipeline integrity のみ）。

---

## 7. T3b Capture Implementation

§3 の 5 点 conjunction のみを terminal success authority とする。
禁止形はいずれも実装していない：

```text
× result.completed 単独 → なし（grep 確認：t3b 式は 5 点 conjunction のみ）
× phase 単独 → なし
× R30/R32 型（phase&&completed&&violations）→ なし（FpmT3bCapture::t3b は blockingReason と
  fullyDrained を含む。既存 runP1RecoveryOrigin の terminalOk は変更していない）
× collectDrainAudit().isAllZero() の authority 化 → なし（audit は emit のみ・isAllZero 未使用）
```

---

## 8. Evidence Schema

各 run の `[FPM] <tag> terminal` 行（authority 5 点＋付帯）：

```text
phase／phaseComplete／completed／blockingReason／violations／fullyDrained／t3b／
lateCallbacks／postStopEnqueue
```

`[FPM] <tag> terminal_drain_audit` 行（diagnostic 13 項目）：

```text
pendPub／pendRetire／xfade／routerPending／deferred／quarRes／
activeWorlds／published／retired／activeReaders／stuckReaders／overflowRes
```

M1/M2 supplemental（episode 行・補助 evidence）：

```text
dSeq／dCoord／dCmt／dTake／dBld／seqAdvanced＋pre／post snap（req／que／dup／take／bld／
cmt／coord／drp／seq／blo）
```

counter 期待値だけで shutdown success を判定しない（terminal authority は T3b のみ）。

---

## 9. Classification

実装済み（`fpmClassify`）：

```text
VALID：t3b==true
INVALID-TERMINAL：phaseComplete==true && t3b==false
  例：ShutdownComplete／completed=true／blockingReason=Unknown／isFullyDrained=false
      → INVALID-TERMINAL（VALID に補正しない）
ABORTED：vehicle sequence 未完了（settle／episode／publish 未達・exit code 2）
CRASH／PROCESS FAILURE：process 異常終了等（T3b 取得不能・実行 gate で記録）
```

---

## 10. Build Result

```text
build 手段：vcvars64 環境での cmake --build build --config Release --target AudioEngineHarness
  （build.bat と同一の Ninja Multi-Config・MSVC。素の cmake --build は vcvars 未適用のため
   <cstdio>/<atomic> C1083 で失敗する環境要因。build.bat 経由と等価の vcvars 適用で解消）
結果：[1/4]→[3/4] Linking CXX executable Release\AudioEngineHarness.exe → PASS
  再実行：ninja: no work to do（最新）
成果物：build\Release\AudioEngineHarness.exe（2026-09-24 01:00:47・41216512 bytes）
CLI dispatch：binary 内に --fpm-m0／--fpm-m1／--fpm-m2 文字列を確認（findstr DISPATCH-STRINGS-OK）
```

M0/M1/M2 の実行は行っていない（指示 §11 の範囲どおり compile／link／target existence／
CLI dispatch まで）。

---

## 11. Production / Test / CMake Diff Audit

```text
production diff（src/audioengine 等）：本 gate の変更 0
  ※ git diff 上の src/audioengine 差分は R27 等の既存差分（本 gate で触っていない）。
     確認：本 gate の edit は P1PolyphaseGainCharacterization.cpp と
     PublishPipelineIntegrationTests.cpp の 2 件のみ。
getter diff：0（新規 getter なし・T3b は既存 public API のみ）
counter diff：0（新規 counter なし）
shutdown semantic diff：0／epoch-reclaim diff：0／Recovery semantic diff：0
CMake target 追加：0（既存 AudioEngineHarness executable 配下・CMakeLists.txt 無変更）
build 設定変更：0（build.bat／CMakePresets.json 無変更）
```

自己監査 15 項目：

```text
[1] production diff = 0 … PASS（§11）
[2] getter diff = 0 … PASS
[3] counter diff = 0 … PASS
[4] shutdown semantic diff = 0 … PASS
[5] epoch/reclaim diff = 0 … PASS
[6] Recovery semantic diff = 0 … PASS
[7] CMake target 追加 = 0 … PASS
[8] T3b 5 点を取得 … PASS（§3・§7）
[9] T3b 以外を authority にしていない … PASS（§7）
[10] collectDrainAudit を authority にしていない … PASS（emit のみ）
[11] M1 = 1 episode … PASS（§5）
[12] E2 winner なし … PASS
[13] Running 中 waitForDrain なし … PASS（grep 確認・新規 vehicle 内 0 件）
[14] M2 に buzz/limiter/NUC なし … PASS（grep 確認・新規 vehicle 内該当なし）
[15] build PASS … PASS（§10）
[16] M0/M1/M2 未実行 … PASS（実行なし）
```

---

## 12. Gate Result

```text
test vehicle implementation PASS＋build PASS＋production 変更 0＋API 変更 0＋
semantics 変更 0＋M0/M1/M2 未実行
→ FPM-IMPL-1-PASS
```

---

## 13. STOP / Next Gate

```text
FPM-Impl-1 PASS（本 gate・実装＋build 確認まで・未実行）
    │
    └── P3-5-FPM-Run-1 — Full-pipeline Measurement Execution Gate（別 gate・別指示）
          - M0 → M1 → M2 の順で実測
          - 各 run を §9 分類で記録
          - buzz／limiter／NUC attribution は含まない
```

- source は R27 production＋R23/R30/R32 test vehicle＋FPM M0/M1/M2 vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7 解釈制約・H-B 対象外を維持する。
- 本 gate はここで停止する。

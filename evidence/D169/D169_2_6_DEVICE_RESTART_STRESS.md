# D169-2-6 — Device Restart / Collapse Stress（Work Record）

```text
D169-2-6 — Device Restart / Stress（JUCE restart chain 反復 × collapse）
Date:      2026-09-07
Contract:  evidence/D169/D169_2_2_REPAIR_CONTRACT.md + D169-2-5 PASS（GO）
Preflight: 本文書 §1-P1（JUCE restart chain source 固定）先行実施
Verdict:   **PASS — D169-2-7 GO**
```

---

## P1 — JUCE restart chain 再確認（source 固定）

D169-2-3 P3 で読取済みの JUCE source を基準に、device restart の engine 到達 chain を固定:

```text
[JUCE] audioDeviceStopped
  └─ AudioProcessorPlayer::audioDeviceStopped (juce_AudioProcessorPlayer.cpp:375-389)
       └─ processor->releaseResources()                    (:379-380)
            └─ AudioEngineProcessor::releaseResources (isEnginePrepared guard)
                 └─ AudioEngine::releaseResources = reconfigure pass
                    （terminal intent 無し・lifecycleState/phase = Prepared 維持）
[JUCE] audioDeviceAboutToStart
  └─ AudioProcessorPlayer::audioDeviceAboutToStart (:344-373)
       └─ isPrepared==false（Stopped で解除済み）のため :367 を skip
       └─ setProcessor(nullptr) + setProcessor(old)       (:370-371)
            └─ setProcessor 内で processorToPlay->prepareToPlay(sampleRate, blockSize) (:180)
                 └─ AudioEngineProcessor::prepareToPlay
                      └─ AudioEngine::prepareToPlay(same SR/BS)
                           └─ enterPrepare: phase==Prepared && same → COLLAPSE token
                                └─ D169-2-4 分岐 → return（no-op）
```

- **same SR/BS → collapse 到達**: enterPrepare の collapse 条件（phase==Prepared &&
  lastPrepared==requested）が restart 後の同一 setup で成立。
- **SR/BS 変更時は非 collapse**: lastPrepared と不一致 → `Prepared→Preparing` 遷移 →
  完全 prepare（D167 テスト・D169-2-5 SR 変更観測で実証済み）。
- **restart 中の同時注入なし**: JUCE は device lifecycle を Message Thread で直列化し、
  本 stress も main thread 直列で実施（P2 準拠）。

## P2 — restart test protocol

- D169-2-5 修正版プロトコルを維持: audio 停止後のみ `prepareToPlay` を呼ぶ
  （`h.stopAudioOnly()` → `e.releaseResources()` reconfigure pass → collapse prepare）。
- restart 中の別 thread からの prepare 注入なし（main thread 直列）。
- capture logger は `CriticalSection` 直列化（Timer [MEM_SNAP] 並行書込み対策 — D169-2-5 実装）。
- **test infra deviation（合計 2 件目）**: `AudioEngineHarness` に `startAudioOnly(int)`
  seam を追加（audio thread のみ再開・engine prepare/release に触れない）。
  device restart cycle の「audio run resume」反復に必要。production source 無変更。

## P3 — 反復 stress（50 cycles・3 config）

`testD169DeviceRestartCollapseStress`（新設・runner entry 1 箇所）:

```text
1 cycle = JUCE device restart 相当
  h.stopAudioOnly()          ← audioDeviceStopped
  e.releaseResources()       ← reconfigure pass（Prepared 維持）
  e.prepareToPlay(512, 48000)← about-to-start → setProcessor swap → collapse
  （観測: collapse +1・body 副作用 0・Prepared/admission Open 維持）
  h.startAudioOnly(512)      ← audio resume
  （観測: blocksProcessed 進行 = audio callback 流動）
× 50 cycles
→ stress 全体の不変性確認（P5）→ h.stop() → ShutdownComplete 確認
```

| Config | 結果 |
| --- | --- |
| RelWithDebInfo | **50/50 cycles OK**・全テスト PASS（exit 0） |
| Debug | **50/50 cycles OK**・全テスト PASS（exit 0） |
| Release | **50/50 cycles OK**・全テスト PASS（exit 0） |

（基準 20 cycles を超過・目標 50 cycles を達成）

## P4 — 各 cycle の必須観測（全 cycle で assert 実施・PASS）

| 項目 | 観測方法 | 結果 |
| --- | --- | --- |
| process termination / `std::abort()` | テスト生存 = 到達 | **0**（50 cycles ×3 config） |
| exception | 同上 | **0** |
| collapse count = restart 回数 | `logger.countContains("duplicate-prepare collapsed")` を cycle 毎に +1 検証 | **一致**（50/50/cfg） |
| `Prepared` 維持 | `isEnginePrepared()` cycle 毎 | **true 維持** |
| leavePrepare collapse-side invocation | 間接観測（collapse が 50 回連続成立 = phase が Prepared のまま・leavePrepare が 通っていたら abort） | **0** |
| generation reset | stress 前後で `currentBuildGeneration` 比較 | **不変** |
| publication | `sequenceId` stress 前後比較 | **不変** |
| rebuild intent | capture 内 `REBUILD_TELEMETRY` 行数 | **0** |
| RuntimeWorld / active DSP 再生成 | `activeRuntimeDSPSlot` stress 前後比較 | **不変** |
| admission integrity | cycle 毎に `AdmissionState::Open` 検証 | **維持** |
| 最終 shutdown | `h.stop()` → Closed + `ShutdownComplete` | **到達** |

## P5 — stress 中の副作用監査（25 side effects の反復確認）

| side effect | stress 全体での観測 |
| --- | --- |
| rebuildRequestGeneration | 不変（baseline == final） |
| publication sequenceId | 不変（baseline == final） |
| placeholder slot | 不変（同一 pointer address 維持） |
| RuntimeWorld publication | 進行なし（idle publish #2 不発） |
| REBUILD_TELEMETRY | 0 行 |
| lifecycleState | Prepared 維持（cycle 毎確認） |
| admission | Open 維持（cycle 毎確認） |
| latency resources | realloc 不発（collapse 分岐より後ろのため実行されない） |
| analyzer/crossfade resources | re-init 不発（同上） |
| uiConvolverProcessor.prepareToPlay | 不発（同上） |
| leavePrepare | collapse 経路から到達不能（collapse 50 連続成立が直接証明） |

## P6 — pre-existing hazard の扱い

D169-2-5 で記録した `timerCallback → MEM_SNAP → getActiveRuntimeDSP() →
collectTrackedMemoryStatistics()` の dangling 参照リスクは**修正しない**（指示どおり）。
stress 実行中の新規 crash dump: **0 件**（CrashDumps の AudioEngineHarness dump は
調査時の既存 3 件のみ・stress 中の追加なし）→ 本 hazard は本 stress では顕在化せず、
**独立記録のまま**（D169-2-5 evidence 参照・別 track 起票候補）。

## 実行中の環境問題（記録）

OS フリーズ（2 回）に伴い: (1) Release ビルドで C1033（PDB lock 残骸）→ 全 vc140.pdb
削除で解消、(2) RWDI ビルドで LNK1285（MTNUPCMeasurement PDB 破損）→ 該当 PDB 削除で
解消。また bat の errorlevel 判定が LNK1285 を取りこぼし `[OK]` 誤表示 → **binary
freshness の確認（exe mtime vs obj mtime）を手順に追加**した（旧 binary での stress 実行を
1 回検出・やり直し）。両方とも環境起因であり source 起因ではない。

## PASS 基準照合

```text
JUCE device restart → same SR/BS prepare → collapse → NO prepare body
→ NO leavePrepare → NO generation reset → NO publication
→ Prepared / RuntimeWorld integrity maintained → audio resumes → repeat ×50
```

| 条件 | 結果 |
| --- | --- |
| abort / AV / lifecycle corruption / duplicate publication / duplicate rebuild | **全 0**（50 cycles ×3 config） |
| collapse count = restart 回数 | ✓ 50/50/cfg |
| Prepared / admission / world / slot integrity | ✓ |
| audio resume | ✓ cycle 毎に blocksProcessed 進行 |
| 最終 ShutdownComplete | ✓ |
| production source 追加変更 | 0（test infra のみ: stress test 1 関数 + startAudioOnly seam） |

## Verdict

**PASS — D169-2-7 GO。**

## Evidence

- stress logs: evidence/D169/d169_2_6_harness_{rwdi,debug,release}.log（全 exit 0・FAIL 0）
- CTest: evidence/D169/d169_2_6_ctest_{debug,release}.log（40/40 ×2・collapse test 込み）
- build logs: evidence/D169/d169_2_6_build*.log（Debug/Release/RWDI）
- test source: `PublishPipelineIntegrationTests.cpp`（stress test 1 関数 + runner entry）、
  `AudioEngineHarness.{h,cpp}`（`startAudioOnly` seam・deviation 2 件目）

次: **D169-2-7 Full Regression → D169-2 close**。

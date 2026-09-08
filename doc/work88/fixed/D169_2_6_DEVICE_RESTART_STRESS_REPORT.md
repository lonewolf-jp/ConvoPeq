# D169-2-6 — Device Restart / Stress（Work Report）

```text
Task:    D169-2-6 — Device Restart / Collapse Stress
Date:    2026-09-07
Type:    targeted stress（production 変更 0・test source deviation 合計 2 件を文書化）
Verdict: **PASS — D169-2-7 GO**
```

## 判定

JUCE device restart chain 相当（audioDeviceStopped → reconfigure release →
about-to-start → same SR/BS prepare → collapse → audio resume）を **1 cycle として
50 cycles 反復**し、3 config（Debug / Release / RelWithDebInfo）で全 PASS。

```text
1 cycle: h.stopAudioOnly() → e.releaseResources()（reconfigure pass）
         → e.prepareToPlay(512, 48000)（collapse 1 回発生）
         → h.startAudioOnly(512) → audio callback 流動確認
50 cycles × 3 config: abort/AV 0・collapse count = 50/50/cfg
stress 全体: generation / publication sequenceId / placeholder slot 不変
             REBUILD_TELEMETRY 0 行・Prepared/admission Open 維持
最終: h.stop() → admission Closed + ShutdownComplete 到達
```

## P1〜P7 照合

- **P1** JUCE restart chain を source 固定（`AudioProcessorPlayer::audioDeviceAboutToStart`
  :344-373 → setProcessor swap → prepareToPlay :180。isPrepared 解除により :367 skip →
  restart 毎に prepareToPlay が 1 回呼ばれ、same SR/BS で collapse 到達・SR/BS 変更時は
  非 collapse）。
- **P2** D169-2-5 修正版プロトコル維持（audio 停止後のみ prepare・main thread 直列・
  CriticalSection logger）。
- **P3** 50 cycles ×3 config 全 PASS（基準 20 を超過）。
- **P4** 各 cycle の必須観測 11 項目 全 PASS（テスト内 assert）。
- **P5** 25 side effects の反復非実行を stress 全体で確認（10 項目を個別照合）。
- **P6** pre-existing hazard（MEM_SNAP dangling 参照）は修正せず独立記録のまま。
  stress 中の新規 crash dump 0 件。
- **P7** Debug / Release / RelWithDebInfo の 3 config で実施。

## test source deviation（合計 2 件）

1. D169-2-5: `testD169DuplicatePrepareCollapseNoop` 追加
2. D169-2-6: `testD169DeviceRestartCollapseStress` 追加 + `AudioEngineHarness::
   startAudioOnly(int)` seam 追加（audio thread のみ再開・engine に触れない。
   restart cycle 毎の audio resume 反復に必要）

production source 変更は 0。

## 実行中の環境問題（記録）

OS フリーズ 2 回に伴う: C1033（Release PDB lock 残骸・全 vc140.pdb 削除で解消）、
LNK1285（RWDI MTNUPCMeasurement PDB 破損・該当 PDB 削除で解消）。**bat の errorlevel
判定が LNK1285 を取りこぼし `[OK]` 誤表示 → binary freshness 確認（exe mtime vs obj
mtime）を手順に追加**（旧 binary での stress 実行を 1 回検出・やり直し済み）。

## 成果物

- 正本: [evidence/D169/D169_2_6_DEVICE_RESTART_STRESS.md](C:\VSC_Project\ConvoPeq\evidence\D169\D169_2_6_DEVICE_RESTART_STRESS.md)
- 本報告: doc/work88/D169_2_6_DEVICE_RESTART_STRESS_REPORT.md
- logs: evidence/D169/d169_2_6_harness_{rwdi,debug,release}.log / d169_2_6_ctest_*.log / d169_2_6_build*.log

次: **D169-2-7 Full Regression → D169-2 close**。

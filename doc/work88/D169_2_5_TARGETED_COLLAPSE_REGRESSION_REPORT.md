# D169-2-5 — Targeted Collapse Regression（Work Report）

```text
Task:    D169-2-5 — Duplicate-Prepare Collapse Targeted Runtime Regression
Date:    2026-09-07
Type:    targeted runtime regression（production 変更 0・test source deviation 1 件を文書化して実施）
Verdict: **PASS — D169-2-6 GO**
```

## 判定

`testD169DuplicatePrepareCollapseNoop`（新設）により、same SR/BS duplicate prepare ×4 の
runtime 観測を実施し、D169-2-4 の「collapse = 真の no-op」を実証した。

```text
Test sequence:
  h.start(48000,512) → rebuild 完了待ち → stopAudioOnly + reconfigure release
  → ベースライン観測 → prepareToPlay(512,48000) ×4（collapse）
  → 観測比較 → prepareToPlay(512,44100)（非 collapse）→ h.stop()（terminal）

Observed collapse:    diagLog "duplicate-prepare collapsed" ×4/4
Abort/exception:      0（旧 code なら leavePrepare 前提違反で 0xC0000409）
Lifecycle before/after: Prepared 維持（isEnginePrepared true・admission Open）
Side-effect observations:
  generation 不変 / publication sequenceId 不変 / REBUILD_TELEMETRY 0 /
  prepare body 入口 log 0 / placeholder slot 不変
  → prepare body の 25 side effect（D169-2-3 P4）が一切再実行されていない
Non-collapse regression: SR 変更 re-prepare で publication 進行（collapse 分岐に入らない）
  + 既存 CTest Debug 40/40・Release 40/40（無変更実行）
Environment: build-diag Debug / Release / RWDI 全 config
Verdict: **PASS — D169-2-6 GO**
```

## test source deviation の記録

既存 runtime 経路では collapse に到達できないことを証明（CLI switch は startup 1 回のみ、
JUCE 同一 setup は early return、既存テストは全て release 後 prepare か SR 交互、soak は
prepare 0 件）したため、targeted regression 1 件を `PublishPipelineIntegrationTests.cpp`
に追加した（1 関数 + runner entry 1 箇所）。D169-2-1 R11 の恒久 coverage gap も同時に埋める。
production source 変更は 0。

## 実行中の crash 調査（解消済み・記録）

初版テストで harness が segfault。bisect（D169-2-4 分岐 stash → 同一 crash）で
production 修復は無関係と確定後、crash dump 解析（crash thread = `timerCallback`）と
テスト契約の再吟味により **初版テストの 2 つの契約違反**（audio thread 走行中の
prepareToPlay 呼出・capture logger の非同期書込み）と特定し、D167 と同一の
`stopAudioOnly + reconfigure release` パターン + CriticalSection 直列化 logger に修正。
修正後は 3 config harness + CTest ×2 が全 PASS（crash 完全消滅）。

調査過程で観測した pre-existing hazard（本 track では処置せず記録のみ）:
`timerCallback` の MEM_SNAP sampler（`Timer.cpp:1079-1088`）が pointer slot の値
（rebuild 後は dangling な placeholder address — D169-1 §3）を
`collectTrackedMemoryStatistics()` に渡す経路。timing 依存 flaky AV の潜在。別 track 起票候補。

## 成果物

- 正本: [evidence/D169/D169_2_5_TARGETED_COLLAPSE_REGRESSION.md](C:\VSC_Project\ConvoPeq\evidence\D169\D169_2_5_TARGETED_COLLAPSE_REGRESSION.md)
- 本報告: doc/work88/D169_2_5_TARGETED_COLLAPSE_REGRESSION_REPORT.md
- logs: evidence/D169/d169_2_5_harness_{rwdi,debug,release}.log / d169_2_5_ctest_{debug,release}.log / crash 調査系一式

次: **D169-2-6 Device restart / stress → D169-2-7 Full regression → D169-2 close**。

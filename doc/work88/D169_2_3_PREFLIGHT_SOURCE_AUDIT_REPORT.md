# D169-2-3 — Preflight Source Audit（Work Report）

```text
Task:    D169-2-3 — Preflight Source Audit（P1〜P5）
Date:    2026-09-07
Type:    read-only source audit（production/test/CMake/build script/tool/binary 変更 0）
Verdict: **P1〜P5 全 GO → D169-2-4 GO**
```

## 判定

D169-2-2 契約（RC-D169-2-1〜7）に基づき、D169-2-4 minimal implementation の
前提を P1〜P5 で source-level 再証明した。すべて GO。

### P1 — 判別子一意性 **GO**

`enterPrepare()` の return site は 2 箇所のみ（`ISRLifecycle.cpp:34` collapse / `:45` 通常）。
`expectedPhase == Prepared` を生成するのは **collapse 経路のみ**（通常は常に Preparing、
`enterAudioCallback` は AudioRunning・`enterRelease` は Releasing）。
`prepareToPlay()` 内の `expectedPhase` 読み取り 0 件・`leavePrepare` は token 未参照。

### P2 — 挿入位置 **GO**

挿入点 `PrepareToPlay.cpp:20` 直後（行番号は最新 source で不変を再確認）。
collapse return は lifecycleState CAS（:50-67）・rollbackPrepareFailure lambda（:30）・
全 allocation・`leavePrepare`（:327）のすべてより前に位置し、block/rollback/leavePrepare
と完全分離。窓（collapse + lifecycleState≠Prepared）は現行も block return で no-op であり、
修復後も観測可能挙動は同一 — 唯一の挙動変化は defect 経路（body 実行 → abort）の no-op 化。

### P3 — JUCE/engine 契約 **GO**

- `isEnginePrepared()` は lifecycleState 由来（AudioEngine.h:1146-1150）で collapse でも true 維持。
- latency buffer / analyzerFifo / crossfade buffer は engine member として生存し、
  reconfigure pass も触らない。同一 SR/BS では要求寸法も不変。
- `AudioEngineProcessor::prepareToPlay` の後段処理（setLatencySamples・cachedTailLength
  publish）は同値冪等。AudioProcessorPlayer の setProcessor swap → prepareToPlay chain で
  no-op return は「restart 直前と同一の processable state」への復帰として正常系成立。

### P4 — side effect 完全列挙 **GO**

enterPrepare return 〜 leavePrepare の制御フローを全行読取し **25 項目**を列挙
（RC-5 リストは例示・本列挙が権威。追加確認分: setShutdownPhase(Running)・SR/BS publish・
coeff bank select・level/bypass atomics・latency reset block・rtLocalState_ 書込・
**uiConvolverProcessor.prepareToPlay** 等）。いずれも挿入点より後ろに位置し、
early return 1 箇所で RC-5「side effect = 0」が完全達成される。

### P5 — expectedPhase 意味論 **GO**

LifecycleToken 生成は source-wide で 6 site（Prepared は collapse の 1 箇所のみ）。
`expectedPhase` の読み取りは **source 全体で 0 件**（判別子導入が最初の読み取り）。
default 構築 token は直ちに上書きされ未初期化値の読取経路なし。future ambiguity 無し →
STOP 条件不発。

## 変更範囲

production 0 / test 0 / CMake 0 / build script 0 / tool 0 / binary 0。

## 成果物

- 正本: [evidence/D169/D169_2_3_PREFLIGHT_SOURCE_AUDIT.md](C:\VSC_Project\ConvoPeq\evidence\D169\D169_2_3_PREFLIGHT_SOURCE_AUDIT.md)
- 本報告: doc/work88/D169_2_3_PREFLIGHT_SOURCE_AUDIT_REPORT.md

次: **D169-2-4 Minimal Implementation** — `AudioEngine.Processing.PrepareToPlay.cpp` の
enterPrepare return 直後 collapse detection 1 箇所のみ（ISRLifecycle.cpp / LifecycleToken /
FSM / rollback / placeholder / world / publish / rebuild / shutdown FSM は変更禁止）。

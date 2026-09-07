# D169-2-3 — Preflight Source Audit（P1〜P5）

```text
D169-2-3 — Preflight Source Audit for D169-2-4 minimal implementation
Date:     2026-09-07
Type:     read-only source audit（production/test/CMake/build script/tool/binary 変更 0）
Contract: evidence/D169/D169_2_2_REPAIR_CONTRACT.md（RC-D169-2-1〜7・RC-1〜RC-10）
Verdict:  **P1 GO / P2 GO / P3 GO / P4 GO / P5 GO → D169-2-4 GO**
```

---

## P1 — `expectedPhase == Prepared` の一意性 **GO**

### 確認結果（ISRLifecycle.cpp / .h 全走査）

| # | 項目 | 事実 | 位置 |
| --- | --- | --- | --- |
| 1 | `LifecycleToken::expectedPhase` 定義 | `LifecyclePhase expectedPhase;` — struct の第 2 field | `ISRLifecycle.h:31-35` |
| 2 | `enterPrepare()` の全 return site | **2 箇所のみ**: :34（collapse）と :45（通常） | `ISRLifecycle.cpp:34/:45` |
| 3 | `expectedPhase = Prepared` 生成箇所 | **:34 のみ**（collapse 経路） | `:27-36` |
| 4 | 通常経路の token | `newPhase = transitionTo(LifecyclePhase::Preparing)` → **常に `Preparing`** | `:40-45` |
| 5 | `Prepared` token と collapse の同値性 | `enterPrepare` 内では :34 のみ。他の token 生成（`enterAudioCallback` :77 = AudioRunning、`enterRelease` :105 = Releasing）は Prepared を返さない | source-wide |
| 6 | `prepareToPlay()` 内の `expectedPhase` 読み取り | **0 件**（token は `leavePrepare` への pass-through のみ） | grep 実測 |
| 7 | `leavePrepare()` の token 参照 | **無し** — body は `phase_` のみ読み、token 引数は未使用 | `:48-58` |

### P1 判定条件の成立

> `expectedPhase == Prepared` を collapse 判別に使用しても、別の lifecycle state /
> 別の return path を誤って collapse と認識する可能性が source 上 0

**成立。** `enterPrepare` の 2 return site のうち Prepared を生成するのは collapse のみ、
通常経路は常に Preparing。判別子は `enterPrepare` の契約内で閉じている。

## P2 — early return 挿入位置の再検証 **GO**

挿入点: `PrepareToPlay.cpp:20`（`enterPrepare` 呼出）直後。最新 source で行番号不変を再確認
（D170 の変更は `ReleaseResources.cpp` のみ・PrepareToPlay.cpp は未改変）。

### P2-A — block path との分離 **GO**

```text
【collapse 経路（本修復）】                  【通常経路（無変更）】
enterPrepare(:20)                           enterPrepare(:20)
  ↓ expectedPhase==Prepared                   ↓ expectedPhase==Preparing
[early return :21 挿入]                      lifecycleState CAS(:50-67)
  ↓                                             ├─ Releasing/Preparing/Destroyed → block return(:53-59)
leavePrepare 未到達・lifecycleState 不変        └─ Prepared/他 → Preparing → body(:73-324)
                                                ↓ leavePrepare(:327)
```

- collapse return は CAS（:50）より前 → block semantics（Releasing/Preparing/Destroyed
  の早期 return）に到達しない = 既存 block 経路は無変更。
- **窓分析**: collapse が成立しかつ lifecycleState ≠ Prepared（Releasing/Preparing 窓）の
  場合、現行 code も block return（:53-59）で side effect 無し・leavePrepare 無しで抜ける。
  修復後は collapse return で抜ける — **観測可能挙動は同一の no-op**（差分は :57 の
  block ログが出ないのみ）。
- 唯一の挙動変化 = defect 経路（collapse + lifecycleState==Prepared → body 実行 →
  leavePrepare abort）が no-op に変わること。

### P2-B — rollback からの遮断 **GO**

`rollbackPrepareFailure` lambda は :30-44 で定義され、初回発火は :201（latency alloc 失敗）。
collapse return は :20 直後のため **lambda 定義自体に到達せず、全 allocation 経路
（latency realloc :186-204・placeholder construct :242）より前に抜ける**。
`collapse → allocation failure → rollback` の経路は構造的に存在しない。

### P2-C — leavePrepare 遮断 **GO**

`leavePrepare` は :327。collapse return（:21 挿入）はその前に存在し、
collapsed token が `leavePrepare` に渡る経路は 0 件となる（RC-3 達成）。

## P3 — JUCE / engine 契約の no-op 安全性 **GO**

### P3-A — isEnginePrepared() **GO**

`isEnginePrepared()` = `lifecycleState == Prepared`（`AudioEngine.h:1146-1150`）。
collapse return は lifecycleState を触らないため **true を維持**。
`AudioEngineProcessor::releaseResources` の duplicate-release guard（`isEnginePrepared` 前提）
とも整合（re-prepare が reconfigure pass と組む前提を壊さない）。

### P3-B — prepare 依存 resource の生存 **GO**

- latency buffer（`latencyBufOldL/R` 等・`AudioEngine.h:2101`）・analyzerFifo（:2221）・
  dspCrossfadeFloat/DoubleBuffer（:2380）は **engine member** であり、prepare 間で生存する。
  prepare body 内でのみ free/realloc される（collapse では実行されない）。
- 直前の reconfigure pass（`releaseResourcesForReconfigure`）は learner stop + level reset
  のみでこれらに触らない（D167-3・`ReleaseResources.cpp:761-783` 実測）。
  よって同一 SR/BS では requiredLatencyBufSize も不変（:178-179 の式は SR/BS のみに依存）。

### P3-C — RuntimeWorld / active DSP **GO**

collapse return は world・registry・handle に触れない。D167 reconfigure pass が
world/active DSP を device restart を跨いで生存させる設計（D166 §10）のため、
collapse no-op は「既存 prepared 状態の継続」として整合。

### P3-D — AudioProcessorPlayer / JUCE 契約 **GO**

caller chain（`juce_AudioProcessorPlayer.cpp` 実測）:
`audioDeviceStopped`（player.isPrepared=false → processor->releaseResources = engine
reconfigure pass）→ `audioDeviceAboutToStart`（:364-372 — isPrepared==false のため
:367 を skip → `setProcessor(nullptr)+setProcessor(old)` swap → `:180 processor->
prepareToPlay(sampleRate, blockSize)`）。

`AudioEngineProcessor::prepareToPlay`（`AudioEngineProcessor.cpp:35-53`）は engine
no-op return 後も:
- `setLatencySamples(audioEngine.getTotalLatencySamples())`（:38）— 値は前回 prepare
  から不変（state-based PDC alias）→ JUCE `setLatencySamples` は同値で冪等。
- `cachedTailLength` publish（:47/:51）— 同値の再 publish。

いずれも同値冪等であり契約矛盾なし。`audioDeviceIOCallbackWithContext` は
AboutToStart 完了後に開始され、engine は Prepared + world published =
**restart 直前と同一の processable state**。no-op return は正常系として成立。

## P4 — side effect 完全列挙 **GO**

`enterPrepare` return（:20）から `leavePrepare`（:327）までの制御フローを全行読取し、
状態変更・allocation・publication・rebuild dispatch・ownership 操作を列挙した。
**RC-5 の禁止リストは例示であり、以下が権威ある完全列挙である**（D169-2-5 検証の照合基準）:

| # | 位置 | side effect |
| --- | --- | --- |
| 1 | :22-23 ほか | diagLog（ログのみ・許容） |
| 2 | :30-44 | rollbackPrepareFailure lambda 定義（実行なし） |
| 3 | :48 | `m_healthMonitor.reset()` |
| 4 | :50-67 | lifecycleState CAS → **Preparing**（block return :53-59 を含む） |
| 5 | :73 | `setShutdownPhase(ShutdownPhase::Running)` |
| 6 | :77-88 | rebuild thread restart 分岐: `rebuildThreadShouldExit←false`・`hasPendingTask=false`・`publishRetryReady=false`・`pendingTask = RebuildTask{}`（rebuildMutex 下）・thread spawn |
| 7 | :96 | **`rebuildRequestGeneration ← 0`** |
| 8 | :100-102 | `runtimeOrchestrator_->resetProgressObservation()` |
| 9 | :124-125 | **`maxSamplesPerBlock` / `currentSampleRate` publish（SR/BS 更新）** |
| 10 | :127-132 | `m_irFadeTimeSec` publish |
| 11 | :133-135 | **`crossfadeRuntime_.reset()` + gain reset/setCurrentAndTargetValue** |
| 12 | :138-161 | **idle publish #2**（`commitRuntimePublication`・needsRegistration・publication） |
| 13 | :162 | `selectAdaptiveCoeffBankForCurrentSettings()` |
| 14 | :164-165 | dspCrossfadeFloat/DoubleBuffer `setSize` |
| 15 | :167 | **`analyzerFifo.prepare`** |
| 16 | :168-172 | input/outputLevelLinear・eqBypassActive・convBypassActive publish |
| 17 | :178-204 | **latency buffer realloc**（free+alloc・失敗時 rollback → Unprepared） |
| 18 | :206-216 | latency reset block（memset ×4・latencyWritePos=0・publishLatencyDelayAtomics・latencyResetPending・refreshCrossfadePreparedSnapshotFromAtomics・resetLatencyDelayRtState） |
| 19 | :222-229 | `rtLocalState_.expectedCallbackIntervalUs / cachedThreadId` 書込 |
| 20 | :230 | **`lifecycleState → Prepared` publish** |
| 21 | :233-304 | **placeholder branch**（DSPCore construct/prepare/setBypass/setFixedLatency・`setActiveRuntimeDSP`・lastCommitted publish ×2・**idle publish #3**・CallerDestroy 分岐 `destroyRolledBackDSP`+slot=null） |
| 22 | :307 | **`uiConvolverProcessor.prepareToPlay(safeSampleRate, bufferSize)`** |
| 23 | :308-309 | `uiConvolverProcessor.invalidatePendingLoads()`（rateChanged） |
| 24 | :311-323 | **`submitRebuildIntent(Structural)`**（rateChanged\|\|blockSizeChanged\|\|\!hasCurrentRuntime） |
| 25 | :327 | `leavePrepare(token)` |

RC-5 契約リスト（10 項目）は #4/#6/#7/#8/#11/#12/#15/#17/#20/#21 を包含。
完全列挙により追加確認された項目: #5（shutdownPhase）・#9（SR/BS publish）・#10・
#13（coeff bank）・#14・#16（level/bypass atomics）・#18（latency reset block）・
#19（rtLocalState_）・**#22（uiConvolverProcessor.prepareToPlay）**・#23。
いずれも collapse return（:21 挿入）より後ろに位置し、early return により
**全 25 項目が同時に不実行となる**（1 箇所の挿入で RC-5 完全達成）。

## P5 — `expectedPhase` 判別子の意味論 **GO**

### LifecycleToken 全生成箇所（source-wide grep 実測）

| site | expectedPhase 値 |
| --- | --- |
| `ISRLifecycle.cpp:34`（enterPrepare collapse） | **Prepared** |
| `ISRLifecycle.cpp:45`（enterPrepare 通常） | Preparing（transitionTo 戻り値） |
| `ISRLifecycle.cpp:77`（enterAudioCallback） | AudioRunning |
| `ISRLifecycle.cpp:105`（enterRelease） | Releasing（transitionTo 戻り値） |
| `AudioBlock.cpp:77` / `BlockDouble.cpp:79` | default 構築（Uninitialized）→ :82 で `enterAudioCallback()` 戻り値で上書き |

### consumer 実測

- `expectedPhase` の **読み取りは source 全体で 0 件**（宣言 `ISRLifecycle.h:34` のみ）。
  `leavePrepare` / `leaveAudioCallback` / `leaveRelease` はいずれも token を照合しない。
  唯一の token field 読み取りは `AudioBlock.cpp:301` の `lifecycleToken.epochId`（診断用）。
- default 構築 token は ctor initializer で直ちに `enterAudioCallback()` 戻り値で
  置換され、未初期化値（Uninitialized）が読まれる経路は無い。

### 判定

> `Prepared` を意味する token が将来別用途に使用される余地が現在 source 上存在しない

**成立。** Prepared-phase token の生成点は collapse の 1 箇所のみ。判別子導入は
`expectedPhase` の **最初の読み取り**であり、既存 consumer（0 件）との衝突はない。
ambiguity 無し → **STOP 条件不発・D169-2-4 GO**。

---

## 最終判定

```text
P1 GO   （expectedPhase==Prepared は collapse のみ・他経路誤認識の可能性 0）
P2 GO   （挿入点 :20 直後 — block/rollback/leavePrepare と完全分離・窓挙動同一）
P3 GO   （isEnginePrepared 維持・prepare 依存 resource 生存・world/DSP 無傷・
         AudioProcessorPlayer/Processor 契約は同値冪等で矛盾なし）
P4 GO   （side effect 完全列挙 25 項目 — 挿入 1 箇所で RC-5 完全達成・
         RC-5 リストは例示として本列挙が権威）
P5 GO   （expectedPhase 読み取り 0 件・Prepared token 生成点は collapse のみ・
         future ambiguity 無し）
      ↓
D169-2-4 GO — 変更は AudioEngine.Processing.PrepareToPlay.cpp の
            enterPrepare return 直後 collapse detection 1 箇所のみ
```

本 audit 中に production/test/CMake/build script/tool/binary への変更は 0。

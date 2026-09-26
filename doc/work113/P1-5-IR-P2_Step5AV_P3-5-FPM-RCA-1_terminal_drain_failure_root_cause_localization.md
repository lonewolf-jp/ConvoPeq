# P1-5-IR-P2 — Step 5-AV / P3-5-FPM-RCA-1: Terminal Drain Failure Root-Cause Localization Gate

- **作成**: 2026-09-24 / work113 P1-5-IR Phase 2
- **種別**: read-only RCA audit。コード変更 0、FPM vehicle 変更 0、production／test／CMake／build 設定変更 0、FPM 再実行 0。
- **入力**: P3-5-FPM-Run-1 の M0／M1／M2 `INVALID-TERMINAL` 固定結果。
- **判定**: **PARTIALLY-LOCALIZED**。
  - 確定: `AudioEngine::isFullyDrained()` の外側 predicate `retireDepth == 0` が false であった。
  - 確定: 観測された `routerPending=2` は、現行 source の writer 条件下で `DeferredDeletionQueue` の D-queue depth 2 に帰着する。`LifetimeState` の RetireIntent 数ではない。
  - 確定: `waitForDrain(2000,2)` timeout 後の `m_coordinator.finalizeShutdown(true)` は、同じ `m_epochDomain` に対して current／target snapshot の retire enqueue を行い、`timedOut` 時に provider reclaim を実行しない。観測数 2 と整合する late-enqueue／reclaim-suppression 機構がある。
  - 未確定: D-queue 内の 2 entry が `SnapshotCoordinator` の current＋target か、final DSP retire 等を含むか。FPM evidence には entry 種別の露出がない。
  - 未確定: `pendingReclaimHandles_`、router Q/E/T、bridge 内部の個別 state が同時に false だったか。FPM schema はこれらを直接出力しない。

本 gate は修正案を出さず、read-only source evidence で局在化範囲を固定する。

---

## 1. Scope / State Freeze

### 1.1 Git / source freeze

```text
HEAD = 1e9e63e34bed7adb9342ebc81259ded9689fc48a
ConvoPeq.md =
  mtime 2026-09-23 23:33:38
  size  5535334 bytes
  SHA256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
FPM vehicle = build\Release\AudioEngineHarness.exe
  mtime 2026-09-24 01:00:47
  size  41216512 bytes
本 gate の source/test/CMake/build 変更 = 0
本 gate の FPM-M0/M1/M2 再実行 = 0
```

`HEAD` と `ConvoPeq.md` は FPM-Run-1 の State Freeze と一致する。vehicle は source より新しく、FPM-Run-1 の実行成果物と同一である。

### 1.2 指定 aggregate の reconciliation

workspace roots には `ConvoPeq(3).md` という実体が存在しなかった。`C:\VSC_Project\ConvoPeq` 直下、workspace recursive search、`C:\VSC_Project` 配下を fdfind／filesystem search で確認したが、該当なし。指定名を推測して別 file に置換していない。

本 gate では、workspace に存在する唯一の最新 aggregate `ConvoPeq.md` と live source を照合した。`ConvoPeq.md` 内で以下を確認済みである。

- `AudioEngine::isFullyDrained()`: aggregate line 21070 付近
- `RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained()`: aggregate line 32004 付近
- `drainPendingRetireIntentsForShutdown()`: aggregate line 18305 付近

aggregate の predicate は live source と一致する。`ConvoPeq(3).md` が別環境で提供される場合の再照合は未実施項目として §12 に残す。

### 1.3 Index / tool state

```text
Serena initial_instructions: 実行済み
AiDex session: 実行済み。13 external changes を auto-reindex
AiDex current index: 2026-09-24 01:24:28
CocoIndex ccc: 96,891 chunks / 1,857 files、2026-09-24 00:53:21
Graphify graph: 2026-09-21 12:57:59（stale。結論の authority には不使用）
tgrep .tgtep: 2d ago（stale。交差確認には使用せず）
headroom doctor: proxy 127.0.0.1:8787 PASS
context-mode doctor: server PASS（plugin registration warning のみ）
rtk WSL: 0.49.0、Ubuntu-26.04
```

source の結論は stale index ではなく、直接 read した live source と既存 FPM evidence file を優先した。

### 1.4 Read-only boundary

```text
production source変更なし
test vehicle変更なし
getter／counter追加なし
shutdown／epoch／reclaim／router／Recovery変更なし
FPM-M0/M1/M2 rerunなし
Dr. Memory 実行なし（本 gate は FPM rerun 禁止のため。version 2.6.20434 の存在だけ確認）
```

---

## 2. Latest ConvoPeq.md reconciliation

`ConvoPeq.md` は source aggregate として以下の authority を持つ。

| live source | aggregate で確認した内容 | 判定 |
|---|---|---|
| `src/audioengine/AudioEngine.Threading.cpp:153-212` | `isFullyDrained()` の 9 条件 conjunction | 一致 |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp:511-562` | bridge の queue／counter／recovery conjunction | 一致 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:615-664` | `waitForDrain` → timeout reason → `finalizeShutdown` | 一致 |
| `src/core/SnapshotCoordinator.h:62-71,165-183` | timeout 時の `retireCurrentAndTarget` と reclaim skip | 一致 |

したがって本 RCA の source line citation は aggregate と live source の二重確認済みである。

---

## 3. FPM-Run-1 evidence recap

### 3.1 3 run の terminal signature

```text
Run: M0 / M1 / M2
Process exit code: 0 / 0 / 0
phase: 10
phaseComplete: 1
completed: 1
blockingReason: 9 (Unknown)
transitionViolations: 0
fullyDrained: 0
t3b: 0
lateCallbacks: 0
postStopEnqueue: 0
classification: INVALID-TERMINAL
```

M0／M1／M2 の terminal 5 点と主要 audit は同一である。

```text
field                         M0    M1    M2
pendPub                         0     0     0
pendRetire                      0     0     0
xfade                           0     0     0
routerPending                   2     2     2
deferred                        0     0     0
quarRes                         0     0     0
activeWorlds                    1     1     1
published                       3     4     6
retired                         2     3     5
activeReaders                   0     0     0
stuckReaders                    0     0     0
overflowRes                     0     0     0
```

### 3.2 FPM field の正確な意味

FPM vehicle の `fpmCaptureT3b()` は `isFullyDrained()` を先に呼び、その後 `collectDrainAudit()` を呼ぶ（`P1PolyphaseGainCharacterization.cpp:1317-1350`）。

```text
[FPM] fullyDrained
  = e.isFullyDrained()

[FPM] pendPub
  = audit.pendingPublication
[FPM] pendRetire
  = audit.pendingRetire
[FPM] routerPending
  = audit.routerPendingRetire
[FPM] deferred
  = audit.deferredPublish
[FPM] quarRes
  = audit.quarantineResident
[FPM] overflowRes
  = audit.overflowRingResident
```

したがって `overflowRes=0` は `DeferredDeletionQueue` の D-queue depth そのものではなく、`worldAuthority_.lifetime().getOverflowRing()->residentCount()` である。

### 3.3 既存 shutdown trace による独立 cross-check

FPM-Run-1 実行時に生成された既存 `evidence/shutdown_trace.json` を再実行せず read-only で parse した。

```text
phaseName          = ShutdownComplete
phase              = 10
blockingReason     = Unknown
blockingReasonCode = 9
sh1_callbackCount  = 0
sh2_activeCrossfade= 0
sh3_pendingRetire  = 2
sh4_observerCount  = 0
sh5_lateCallbackCount = 0
sh6_postStopEnqueueCount = 0
verified           = false
Unknown stats.count = 1
```

`shutdown_trace.json` の `sh3_pendingRetire=2` は `AudioEngine.Processing.ReleaseResources.cpp:726-736` で `m_retireRouter->pendingRetireCount()` を capture した値であり、FPM の `routerPending=2` と同一の D-queue 観測を別 artifact から支持する。

`evidence/retire_trace_shutdown_last.json` は EpochControl slot lifecycle（5 slot reclaimed）を示すだけで、D-queue 2 entry の pointer 種別・identity を示すものではない。混同しない。

---

## 4. `isFullyDrained()` predicate decomposition

### 4.1 現行 outer predicate

`AudioEngine::isFullyDrained()` は `AudioEngine.Threading.cpp:153-212` で以下の式を評価する。

```cpp
return !hasDeferredCommit
    && pendingReclaimEmpty
    && retireDepth == 0
    && lifetimeRetireIntentPending == 0
    && ringResident == 0
    && dspQuarantineResident == 0
    && retireQuarantineResident == 0
    && terminalReclaimResident == 0
    && runtimePublicationBridge_.isFullyDrained();
```

### 4.2 predicate ごとの source / observation

| predicate | source of truth | FPM evidence | 判定 |
|---|---|---:|---|
| `!hasDeferredCommit` | `runtimeOrchestrator_->hasDeferredRequest()` (`AudioEngine.Threading.cpp:155`) | `deferred=0` | 成立 |
| `pendingReclaimEmpty` | `pendingReclaimHandles_.empty()` under mutex (`AudioEngine.Threading.cpp:193-201`) | 出力なし | 未観測 |
| `retireDepth == 0` | `m_retireRouter->pendingRetireCount()` (`AudioEngine.Threading.cpp:170-171`) | `routerPending=2` と `fallbackQueueDepth_=0` の source 条件下で depth 2 | **不成立** |
| `lifetimeRetireIntentPending == 0` | `worldAuthority_.lifetime().pendingIntentCount()` (`AudioEngine.Threading.cpp:172`) | `pendRetire=0` | 成立 |
| `ringResident == 0` | `worldAuthority_.lifetime().getOverflowRing()->residentCount()` (`AudioEngine.Threading.cpp:178-179`) | `overflowRes=0` | 成立 |
| `dspQuarantineResident == 0` | `dspQuarantineManager_.residentCount()` (`AudioEngine.Threading.cpp:180`) | `quarRes=0` | 成立 |
| `retireQuarantineResident == 0` | `m_retireRouter->quarantineResidentCount()`、Q+EmergencyQ (`AudioEngine.Threading.cpp:181-182`) | 出力なし（`quarRes` ではない） | 未観測 |
| `terminalReclaimResident == 0` | `m_retireRouter->terminalReclaimResidentCount()` (`AudioEngine.Threading.cpp:190-191`) | 出力なし | 未観測 |
| bridge conjunction | `runtimePublicationBridge_.isFullyDrained()` | FPM 内部状態を直接出力しない | 未観測／source-compatible |

### 4.3 重要な queue 分離

FPM の `pendRetire=0` と `routerPending=2` は矛盾しない。

```text
LifetimeState::pendingIntentCount()
  = RetireIntent queue (Vyukov main + fallback) の件数
  = FPM pendRetire

m_retireRouter->pendingRetireCount()
  = provider_->pendingRetireCount()
  = EpochDomain::DeferredDeletionQueue::sizeApprox()
  = pointer-lifetime D queue の件数
  = FPM routerPending および shutdown_trace sh3_pendingRetire に関係
```

`drainPendingRetireIntentsForShutdown()` は前者の `LifetimeState` queue と fallback queue を drain するが、DeferredDeletionQueue D queue を直接 drain しない（`AudioEngine.Processing.ReleaseResources.cpp:803-887`）。このため `pendRetire=0` と D queue=2 は同時に成立できる。

---

## 5. `runtimePublicationBridge_.isFullyDrained()` trace

### 5.1 bridge の全条件

`RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained()` は `ISRRuntimePublicationCoordinator.cpp:511-562` で以下を conjunction 判定する。

```text
swapPending_ == false
intentQueue_.sizeApprox() == 0
observeDeferredRing_.size() == 0
quarantineFallbackQueue_.sizeApprox() == 0
recoveryIntentQueue_.size() == 0
retireBacklogCount_ == 0
publicationBacklogCount_ == 0
publicationIntentResidencyCount_ == 0
pendingIntentCount_ == 0
reclaimInFlightCount_ == 0
quarantineIntentResidencyCount_ == 0
quarantineRingResidencyCount_ == 0
recoveryAdmissionPending_ == false
liveLogicalRecoveryObligationCount() == 0
```

### 5.2 主要 state writer と shutdown clear

| bridge state | writer / clear path | M0 source expectation |
|---|---|---|
| `swapPending_` | `commit()` が `true` → metadata bake → `false` (`ISRRuntimePublicationCoordinator.cpp:109-120`) | publish receipt 後に false |
| `intentQueue_` | `enqueue*` push、`processIntent()` pop (`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:35-73`) | Coordinator join 後に空化 |
| `observeDeferredRing_` | `drainObserveDeferred()` pop (`...ProcessIntent.cpp:87-99`) | Coordinator 処理後に空化 |
| `quarantineFallbackQueue_` | quarantine enqueue、`processIntent()` pop | M0 は quarRes=0、quarantine 経路なし |
| `recoveryIntentQueue_` | recovery enqueue、`stopRebuildThread()` が `discardRecoveryRequestsOnShutdown()` (`AudioEngine.RebuildDispatch.cpp:825-831`) | M0 は Recovery episode なし。builder join 後も discard される |
| `publicationIntentResidencyCount_` | publish enqueue で +1、common queue publish pop で -1 (`...ProcessIntent.cpp:47-63`) | receipt 完了後に 0 |
| `pendingIntentCount_` | Observe/Quarantine/Recovery の reservation/pop | M0 は Recovery/Quarantine なし。停止後の producer はない |
| `recoveryAdmissionPending_` | durable attach true、discard false (`...Coordinator.cpp:1316-1325`) | `stopRebuildThread()` が discard (`...RebuildDispatch.cpp:833-839`) |
| `liveLogicalRecoveryObligationCount()` | `resolveRecoveryObligation()` の単一 `Live→terminal` authority | M0 は recovery obligation を作らない |
| `reclaimInFlightCount_` | deferred `requestReclaim()` 経路の begin/end (`...Coordinator.cpp:211-229,689-717`) | FPM に個別値がないため未観測 |

M0 は FPM source 上で Recovery episode を発行しない。起動時に生成された通常の publish／snapshot は stop 前に receipt まで到達するため、bridge の queue/counter を残す経路は特定されていない。

ただし `pendingReclaimHandles_` と `reclaimInFlightCount_` の個別値は FPM evidence に存在しないため、bridge が true であったこと、または同時に false であったことを source trace だけで断定しない。

### 5.3 bridge の terminal disposition

`runtimePublicationBridge_.markShutdownComplete()` は `ISRRuntimePublicationCoordinator.cpp:568-579` で bridge predicate を再評価する。

```text
bridge true  -> CoordinatorState::Bootstrapping
bridge false -> CoordinatorState::Faulted
```

これは `ShutdownRuntime` の `ShutdownPhase` を変更する API ではない。outer `AudioEngine::isFullyDrained()` の T3b gate でもない。したがって bridge state Faulted と `phase=ShutdownComplete` は併存し得る。

---

## 6. `ISRRetireRouter` pendingRetire trace

### 6.1 D queue の enqueue lifecycle

```text
DSPLifetimeManager::retire()
  -> engine_.retireDSPHandleForRuntime()
  -> router_->currentEpoch()
  -> ISRRetireRouter::enqueueWithRetry()
  -> EpochDomain::DeferredDeletionQueue::enqueue()
       (`DeferredDeletionQueue.h:65-107`)

SnapshotCoordinator::retireCurrentAndTarget()
  -> m_epochProvider->publishEpoch()
  -> current snapshot を enqueue
  -> target snapshot を enqueue
       (`SnapshotCoordinator.h:165-183`)

ISRRetireRouter::pendingRetireCount()
  -> provider_->pendingRetireCount()
  -> EpochDomain::DeferredDeletionQueue::sizeApprox()
       (`ISRRetireRouter.cpp:586-590`,
        `EpochDomain.h:430-433`,
        `DeferredDeletionQueue.h:219-224`)
```

`m_retireRouter` と `m_coordinator` は同一 `m_epochDomain` を provider として持つ。

```text
AudioEngine constructor:
  m_coordinator(m_epochDomain)       `AudioEngine.CtorDtor.cpp:26-27`
  m_retireRouter(m_epochDomain, ...) `AudioEngine.CtorDtor.cpp:35-40`
```

したがって SnapshotCoordinator が late enqueue した GlobalSnapshot と、DSPLifetimeManager が enqueue した DSPCore は、outer predicate から見ると同一の D-queue depth に加算される。

### 6.2 `pendingRetireCount` と fallback の混同回避

`collectDrainAudit()` の `routerPendingRetire` は `AudioEngine.Threading.cpp:120-121` で以下を足す。

```text
m_retireRouter->pendingRetireCount()
+ fallbackQueueDepth_
```

これは `LifetimeState::fallbackCount_` ではない。

- `fallbackQueueDepth_` は `AudioEngine.h:5064` で初期化 0。
- production writer は `AudioEngine.Retire.cpp:136-140` の `fallbackDepth = 0` の一箇所だけ。
- terminal 前に `drainDeferredRetireQueues(true)` が呼ばれ、最後の writer も 0。
- `LifetimeState::fallbackOccupancy()` は別の `fallbackCount_` であり、collectDrainAudit の `routerPendingRetire` には直接入っていない。

したがって現 source の writer inventory では、FPM の `routerPending=2` は `DeferredDeletionQueue::sizeApprox() == 2` と解釈できる。これは「routerPending が原因」と結論することではなく、audit field の実体を確定することである。

### 6.3 RetireIntent drain は D queue drain ではない

`drainPendingRetireIntentsForShutdown()` は `AudioEngine.Processing.ReleaseResources.cpp:803-887` で以下だけを行う。

```text
OverflowRing -> LifetimeState::emitRetireIntent
LifetimeState::dequeueOne/dequeueFallback
LifetimeState::reclaim(slot)
```

`m_retireRouter->pendingRetireCount()` の D queue を pop する処理ではない。FPM の `pendRetire=0` と `routerPending=2` を同時に観測できる source 的理由である。

---

## 7. `fallbackQueueDepth_` trace

```text
declaration: AudioEngine.h:5064
read:        AudioEngine.Threading.cpp:121
write:       AudioEngine.Retire.cpp:139 = 0
```

`collectDrainAudit()` のコメントは ring+fallback 合計と記載するが、現行 writer には非ゼロ fallback writer がない。`LifetimeState` の実 fallback queue は `ISRRetire.cpp:133-142,182-188` に別実装されている。

本 audit では `fallbackQueueDepth_` を router D queue と同一視していない。FPM の `routerPending=2` の source mapping は、この legacy atomic が 0 である(writer inventory と shutdown drain order より)という限定付きである。

---

## 8. `blockingReason=Unknown` trace

### 8.1 実際の writer

source literal inventory は以下のみである。

```text
markTimedOut(reason):
  `ISRShutdown.cpp:72-112`
markFailed(reason):
  `ISRShutdown.cpp:114-122`
blockingReason_ store:
  `ISRShutdown.cpp:106,117`
collectResult read:
  `ISRShutdown.cpp:168-175`
```

`markFailed()` の production caller は source inventory 上存在しない。`blockingReason=Unknown(9)` の writer は `releaseResources()` の timeout 分岐である。

### 8.2 releaseResources の Unknown 選択

`AudioEngine.Processing.ReleaseResources.cpp:624-633` は次の擬似 code を持つ。

```text
timedOut = !waitForDrain(2000, 2)
reason = Unknown
if audit.stuckReaderCount > 0:
    reason = ReaderActive
else if rebuildThreadIsRunning:
    reason = ActiveBuilder
shutdownRuntime_.markTimedOut(reason)
```

FPM 3 run では `stuckReaders=0` で、builder は stop/join 済みである。したがって `Unknown` が残る。

`RuntimeDrainAudit::getPrimaryBlockingReason()` は `RuntimeDrainAudit.h:65-73` に実装され、audit 値から `RouterPendingRetire` 等を返す設計だが、production `releaseResources.cpp` から呼ばれていない。現在の `Unknown` は「getPrimaryBlockingReason が具体 source を返せなかった」結果ではない。

したがって source 上の意味は次のとおりである。

```text
Unknown = timedOut 分岐の default/fallback reason
       ≠ terminal disposition 後に isFullyDrained が false になったことを独立检测した reason
       ≠ collectDrainAudit の主因分類
```

### 8.3 Unknown と ShutdownComplete の併存

`markTimedOut()` は reason を保存し phase を `TimedOut` にする（`ISRShutdown.cpp:105-112`）。後の `transitionTo(ShutdownComplete)` は `TimedOut`／`Failed` を terminal phase として skip できる（`ISRShutdown.cpp:124-150`）。

`collectResult()` は現在の phase が `ShutdownComplete` なら `completed=true` とし、reason は保存済みの `blockingReason_` をそのまま読む（`ISRShutdown.cpp:161-180`）。

よって以下は current source から直接導出できる。

```text
phase=ShutdownComplete
completed=true
blockingReason=Unknown
violations=0
```

これは `fullyDrained=true` を意味しない。FPM の T3b false との整合である。

---

## 9. `ShutdownComplete` 到達経路

### 9.1 実際の時系列

| 順序 | source path | 状態・処理 |
|---:|---|---|
| 1 | `AudioEngineHarness.cpp:38-53` | audio thread stop、`requestTerminalRelease()`、`releaseResources()` |
| 2 | `ReleaseResources.cpp:61-109` | lifecycle `Releasing`、`AudioStopped`、bridge `requestShutdown()`、`closeAdmission()` |
| 3 | `ReleaseResources.cpp:235-250` | Coordinator stop/join、Rebuild stop/join、producer reservation join |
| 4 | `ReleaseResources.cpp:259-265` | `ObserverDrained`、`advanceRetireEpoch()`、`RetireClosed`、`EpochSettled` |
| 5 | `ReleaseResources.cpp:269-376` | reader registration close、graceful drain、deferred drain、`ReclaimComplete` |
| 6 | `ReleaseResources.cpp:378-471` | `EmergencyDrain`、Q/E/T cleanup |
| 7 | `ReleaseResources.cpp:474-517` | `VerifyDrained`、active/fading handle の quiescent disposition |
| 8 | `ReleaseResources.cpp:532-606` | world clear、router Q/E/T drain、final DSPLifetimeManager retire、deferred clear |
| 9 | `ReleaseResources.cpp:615-617` | `waitForDrain(2000,2)` |
| 10 | `Threading.cpp:215-247` | `while (!isFullyDrained())`、各 loop で `drainDeferredRetireQueues(true)` |
| 11 | `ReleaseResources.cpp:622` | `drainPendingRetireIntentsForShutdown()`（LifetimeState queue のみ） |
| 12 | `ReleaseResources.cpp:624-633` | timeout なら `markTimedOut(Unknown)` |
| 13 | `ReleaseResources.cpp:654-662` | timeout または outer predicate false なら safe `tryReclaim` |
| 14 | `ReleaseResources.cpp:664` | `m_coordinator.finalizeShutdown(timedOut)` |
| 15 | `ReleaseResources.cpp:666-697` | OwnerChannel residual drain |
| 16 | `ReleaseResources.cpp:700-723` | audit collection / diagnostic logging |
| 17 | `ReleaseResources.cpp:724` | `runtimePublicationBridge_.markShutdownComplete()`（bridge のみ評価） |
| 18 | `ReleaseResources.cpp:726-736` | `pendingRetireCount` を shutdown trace へ capture |
| 19 | `ReleaseResources.cpp:744` | `transitionTo(ShutdownComplete)` |
| 20 | FPM `runFpmM0/M1/M2` | `h.stop()` 後に `isFullyDrained()` と audit を capture |

### 9.2 最後の outer predicate 評価と terminal capture の非同一性

`releaseResources()` で outer `isFullyDrained()` が最後に明示評価されるのは `ReleaseResources.cpp:654` の condition である。その後、

```text
m_coordinator.finalizeShutdown(timedOut)
  -> SnapshotCoordinator::retireCurrentAndTarget()
  -> current/target snapshot を同じ D queue へ enqueue
  -> timedOut=true なら m_epochProvider->tryReclaim() を skip
```

が実行される。`finalizeShutdown` 後には outer `isFullyDrained()` を再評価せず、`markShutdownComplete()` は bridge predicate だけを評価して、phase は無条件に `ShutdownComplete` へ遷移する。

したがって source 上の状態機械は、

```text
pre-finalize isFullyDrained=false
  → finalizeShutdown が D queue を変更し得る
  → terminal phase は ShutdownComplete へ進める
  → FPM terminal capture の isFullyDrained を再度観測
```

という順序であり、terminal phase 到達時点の state と FPM capture 時点を同一視できない。

### 9.3 `timedOut=true` の source 確定

FPM 3 run の `blockingReason=Unknown` は §8 の timeout writer に由来する。`shutdown_trace.json` の `Unknown count=1` も一致する。したがって本 run では `m_coordinator.finalizeShutdown(true)` が実行されたと source evidence から確定できる。

`SnapshotCoordinator::finalizeShutdown()` は `SnapshotCoordinator.h:60-71` で、

```text
retireCurrentAndTarget();
if (!timedOut)
    m_epochProvider->tryReclaim();
```

を実行する。`retireCurrentAndTarget()` は `publishEpoch()` 後に current と target の両方を enqueue する（`SnapshotCoordinator.h:165-183`）。D queue depth 2 という観測値と整合するが、FPM は current／target の occupancy を出力しないため、2 entry の 정확한 identity は source-only で断定しない。

---

## 10. Predicate-to-observation mapping

### 10.1 M0 を主対象にした mapping

```text
Observation                          Predicate relation                         Result
deferred=0                           !hasDeferredCommit                         pass
pendRetire=0                         lifetimeRetireIntentPending==0            pass
overflowRes=0                        ringResident==0                           pass
quarRes=0                            dspQuarantineResident==0                  pass
activeReaders=0                      reader diagnostic                         no stuck reader
stuckReaders=0                       Unknown fallback選択                       pass
routerPending=2                      retireDepth==0                             FAIL（source-mapped D queue=2）
sh3_pendingRetire=2                  terminal直前の D queue                     FAIL
blockingReason=9                     markTimedOut default                      explained
phase=10                             transition TimedOut→Complete              explained
```

### 10.2 未観測 predicate

```text
pendingReclaimEmpty
retireQuarantineResident
terminalReclaimResident
runtimePublicationBridge_.isFullyDrained() 内部の全 counter／queue
```

これらは FPM の `terminal_drain_audit` に直接含まれない。outer conjunction は短絡評価されるため、`retireDepth` が false を返すため、各未観測条件の値を事後的に source から逆算してはならない。

### 10.3 primary false predicate の確定

本 audit で確定する primary false predicate は次のとおりである。

```text
retireDepth == 0
```

根拠は二重要である。

1. FPM `routerPending=2` は `collectDrainAudit()` で `m_retireRouter->pendingRetireCount() + fallbackQueueDepth_`。
2. `fallbackQueueDepth_` は現行 production writer で 0 に書かれ、shutdown 前に `drainDeferredRetireQueues(true)` が実行される。
3. `m_retireRouter->pendingRetireCount()` は `DeferredDeletionQueue::sizeApprox()` であり、既存 `shutdown_trace.json` の `sh3_pendingRetire=2` が独立に一致する。

したがって `routerPending=2` を simply router bug と断定するのではなく、**outer drain predicate である D-queue depth が 2 であった**と確定する。

---

## 11. Root-cause localization verdict

### 11.1 判定

```text
PARTIALLY-LOCALIZED
```

### 11.2 確定した chain

```text
FPM M0/M1/M2
  ↓
phase=ShutdownComplete, blockingReason=Unknown, fullyDrained=0
  ↓
routerPending=2 / shutdown_trace sh3_pendingRetire=2
  ↓
source mapping: m_retireRouter->pendingRetireCount()
                = DeferredDeletionQueue::sizeApprox() = 2
  ↓
outer predicate retireDepth == 0 is false
  ↓
waitForDrain(2000,2) times out
  ↓
releaseResources selects Unknown and calls markTimedOut
  ↓
m_coordinator.finalizeShutdown(true)
  ├─ retireCurrentAndTarget() enqueues current/target into shared m_epochDomain D queue
  └─ timedOut=true skips m_epochProvider->tryReclaim()
  ↓
phase later transitions to ShutdownComplete without an outer T3b gate
```

この chain は「routerPending=2 が shutdown failure の原因である」という広義の結論ではない。確定事项は、**terminal capture 時に外側 drain predicate の D-queue depth が 2 であり、timeout 後の late finalization が D queue を再投入し、reclaim を skip する source path が存在すること**である。

### 11.3 なぜ LOCALIZED ではないか

以下の exact identity は現 evidence にない。

```text
D-queue 2 entries の object type
  - GlobalSnapshot current/target 2件か
  - DSPCore final retire を含むか
  - 他の pointer retire path を含むか
```

また以下は FPM output に直接ない。

```text
pendingReclaimHandles_.size()
retireQuarantineResidentCount()
terminalReclaimResidentCount()
reclaimInFlightCount_ / bridge queue・counter
```

したがって「primary false predicate」と「timeout 後の late enqueue／reclaim skip 機構」は source で局在化できたが、全 conjunction の simultaneous state と 2 entry の exact provenance までは確定していない。よって全体判定は `PARTIALLY-LOCALIZED` とする。

---

## 12. Unresolved items

1. `ConvoPeq(3).md` の実体が workspace に存在しない。最新 aggregate は `ConvoPeq.md` であり、aggregate/live source は照合済み。別 environment に同名 file がある場合は再 hash が必要。
2. D queue 2 entry の per-entry type／identity／enqueue site を FPM evidence から復元できない。既存 trace は slot lifecycle であり D queue entry identity ではない。
3. `pendingReclaimHandles_` が terminal capture 時に空だったか未観測。`activeReaders=0` だけでは epoch-safe 条件 `entry.epoch < minReaderEpoch` の証明にはならない。
4. `retireQuarantineResident` と `terminalReclaimResident` は FPM の `quarRes` に混在していない。個別値の直接観測がない。
5. bridge 内部 queue／counter の individual state は FPM schema にない。M0 の source path は bridge true と整合するが、measured proof はない。
6. `fallbackQueueDepth_` は legacy atomic で、現行 writer が 0 のみ。`LifetimeState::fallbackCount_` ではない。telemetry naming の不一致は残る。
7. 同一 audit の `pendingReclaimHandles_`／bridge state を追加観測する次は、production source／test vehicle を変更しない別 gate での-read-only evidence design が必要。本 gate では設計・修正を行わない。
8. Dr. Memory／FPM rerun は本 gate の禁止により未実施。cppcheck は JUCE macro configuration warnings、clang-tidy は compile DB 使用時に対象 source の警告なしを確認したが、runtime root cause の代替 evidence にはしていない。

---

## 13. STOP / next gate

```text
P3-5-FPM-Run-1
  ↓
INVALID-TERMINAL ×3
  ↓
P3-5-FPM-RCA-1
  ↓
primary false predicate: retireDepth == 0
  ↓
state: DeferredDeletionQueue D depth = 2
  ↓
transition: timeout後 finalizeShutdown(true) の late enqueue + reclaim skip
  ↓
PARTIALLY-LOCALIZED
```

本 gate はここで STOP する。

```text
production/test/CMake/build 変更 = 0
shutdown/epoch/reclaim/router/Recovery 修正 = 0
FPM vehicle 変更 = 0
FPM-M0/M1/M2 再実行 = 0
```

次 gate は、本 RCA の未観測項目を read-only で解消する evidence gate か、RCA 結果を受けた設計 gate を別指示で選定する。ここでは修正候補、source patch、API／counter 追加を先取りしない。

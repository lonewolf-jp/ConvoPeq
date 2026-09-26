# P3-5-FPM-RCA-10 — Reader Lifecycle / Epoch Provenance Static Closure

- Gate: `P3-5-FPM-RCA-10`
- Mode: `read-only source-only static closure`
- Date: 2026-09-24
- Input: `RCA-9 = PARTIALLY-LOCALIZED / STOP`
- Target: `RCA-7 first Engine DQueue reclaim: provenance of minReaderEpoch=9`
- Scope: `all production RCUReader/EpochDomain callsites, reader lifecycle, epoch publication/retirement chronology, shutdown ordering, Case A/B/C/D, and P9-P22`
- Verdict: `PARTIALLY-LOCALIZED / STOP`
- Disposition: `NO IMPLEMENTATION / NO CDB RETRY / NO M0 / NO M1 / NO M2 / NO BUILD / NO DR. MEMORY`
- M0 executions: `0`
- M1 executions: `0` (RCA-9's one failed M1 is inherited evidence, not a new execution)
- M2 executions: `0`
- Build: `not run`
- Dr. Memory: `not run`
- Production/test/CMake/getter/counter/telemetry/shutdown/epoch/reclaim/router/Recovery changes: `0`

## 1. Gate / Scope

RCA-10 asks a source-only question:

```text
Which production reader and epoch paths can establish minReaderEpoch=9,
and which of those paths remain compatible with the S7 observation?
```

This is not a DQueue repair task and does not assume that an active reader caused the shutdown timeout. The layers remain separate:

```text
direct cause      = head sequence is ready, but isOlder(9, 9) is false
upstream cause    = why the Engine DQueue observed minReaderEpoch=9
terminal result   = residual -> isFullyDrained false -> wait timeout -> Unknown
```

The gate permits only static source reconciliation. It does not permit a CDB rerun, correction of the RCA-9 script, M2, build, Dr. Memory, source edits, or implementation.

## 2. Source Identity and Evidence Boundary

The current generated source authority matches the RCA-8/RCA-9 freeze:

```text
Path       = ConvoPeq.md
Bytes      = 5535334
SHA-256    = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
LastWrite  = 2026-09-23T14:33:38.6388122Z
Identity   = MATCH
```

Current source hashes used by this closure:

```text
src/core/EpochDomain.h
  6EC7F1106FE029F862062A4BDB2C7A3F7A14D109E49CA0294B95E7A41F451E7D
src/core/RCUReader.h
  E70CE078A3B402891E9431223C2C0AE9BEE42808D8B3FB119E2B593DD6BEC8F8
src/DeferredDeletionQueue.h
  91143BB072C4D2B2EAE42A012783EF9F28DA0DA8EA7B67FBF70B56E34248A188
src/audioengine/AudioEngine.h
  A491E71A472F8F2C08CB08B415F455BF9CB56CBF360D2DCD8C55B46175214267
src/audioengine/AudioEngine.CtorDtor.cpp
  8DF473AAF5F56F1DB2A356201AA197E108718AD2C962956CF81A2184909C5AEA
src/audioengine/AudioEngine.Processing.ReleaseResources.cpp
  0675CF5137E5A1F917B86D3C517BBF3767550E64051FF9E1C64873B6BAE77E63
src/audioengine/AudioEngine.Threading.cpp
  0B2279DBF30A73F0A54E41C79185394D7B15705742E4128E7AF44558F0EE8E6C
src/audioengine/AudioEngine.Retire.cpp
  E86B8EC22BB80788A31455BE2AA4891590EFEA3756B1E3B3456EB47BD7791198
src/audioengine/DSPLifetimeManager.cpp
  7A0F6F406B69E9F2B21820FECA5BADBD2C819850B7A92BE3311525413F9E6982
src/audioengine/ISRRetireRouter.cpp
  7731E933A2D69BC30598F2173F106B725A884F9BBA235EB4B3D41004CD85B638
src/audioengine/ISRRuntimePublicationCoordinator.cpp
  1AD84A0C3F2FFB7E796A43F979B0D05A8827EBA2631E3A69E01894E5A32DD8C0
src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp
  4390798F3BCD66B9C3730194E633D56487E113BF9EF03B558F2A9A9CA0B454EA
src/core/SnapshotCoordinator.cpp
  DDE9088110F19430A8869E4A2B62CB78B7FF0D60EBE5376D0D258F6B7388A879
src/ConvolverProcessor.h
  DAC751E477F073B582AEA48BC89EBC60E4AB54FBEC7E190021C0D33E5BC16D55
src/convolver/ConvolverProcessor.Runtime.cpp
  11CA7A2EF49F3832937E85C9C0B2B7AF48E06BAA23E54051DDDED25C27A191D9
src/convolver/ConvolverProcessor.Lifecycle.cpp
  4B8B96BE011586B1EE6CA56029338B89C64D4F8D1DBE151C3A4AD0D715CF5A1E
src/convolver/ConvolverProcessor.StateAndUI.cpp
  82146B5E8C5ADB573132F341E4B38267EE894FFEEF3E5CB4957C8446B8F30BB9
src/convolver/ConvolverProcessor.LoadPipeline.cpp
  E7E8012A9FFE488B6536B2FCA1D69092774F2B384DC330075DCFCBDB14DC05EE
src/eqprocessor/EQProcessor.h
  BD71E3A33C24F8460C16D5DE8256C8CCD9C453FD13E5DCED6D0A3F551A7097F4
src/eqprocessor/EQProcessor.Core.cpp
  A9D66FDA2FB71678195766A3AD80F706C860CFC1FA76070FAA646DFA55E6D5F0
src/eqprocessor/EQProcessor.Processing.cpp
  246B1F76D21F24B024F9FB42C460D7B1B23AFC7809AD41E92CB07D8135BA4644
src/NoiseShaperLearner.cpp
  CF579397CA49101ED2ABB8AC3D75A566D93FBAF94474DA64122666E5E48E2B31
src/SpectrumAnalyzerComponent.cpp
  7BE903ED41E2EF0CFE3011F2EBA0C665B4D787AC0E939021A2E6B1FAB60962A7
src/audioengine/RuntimePublicationOrchestrator.cpp
  4F5CAF2FFDE3DCA79A94B675A2D4CA74109C83706487A727E92BC8C94072CCD9
src/audioengine/PublicationAdmission.cpp
  9EEA93F9784F288CD01E442ECABF64ADB273BF7F72458662D85120B4C744EA58
```

The working tree already contains unrelated/pre-existing modifications. They were preserved. No source or test file was changed by RCA-10.

## 3. Frozen S7 Evidence

RCA-7 established the direct first-call condition:

```text
globalEpoch       = 9
dequeuePos        = 4
enqueuePos        = 5
head sequence     = 5
expectedSequence  = 5
head entry epoch  = 9
getMinReaderEpoch = 9
sequence gate     = PASS
epoch gate        = BLOCK
isOlder(9, 9)     = false
dequeue CAS       = NOT REACHED
```

RCA-9 reached the Engine `getMinReaderEpoch` precondition but its CDB `.for` expression failed with `Bad register error` before slot data, return chronology, or reclaim chronology. Therefore this gate inherits no valid S7 reader-slot identity.

The equality `globalEpoch == minReaderEpoch == 9` is not, by itself, evidence of an active reader. The source initializes the minimum from the current epoch and only lowers it for an eligible older slot.

## 4. Domain Topology and Ownership

The repository has three production `EpochDomain` objects and one separate custom RCU implementation. They must not be conflated.

| domain / object | production location | reader / queue role | relation to RCA-7 Engine DQueue |
|---|---|---|---|
| `AudioEngine::m_epochDomain` | `src/audioengine/AudioEngine.h:5017` | global `ISRRetireRouter`, global `DeferredDeletionQueue`, `SnapshotCoordinator`, two persistent Engine `RCUReader`s | **Direct target domain** |
| `ISRRetireRouter` wrapper | `src/audioengine/ISRRetireRouter.cpp:148-223` | delegates global reader/epoch/reclaim calls to `AudioEngine::m_epochDomain` | Same global domain; wrapper is not a second domain |
| `SnapshotCoordinator` | `src/core/SnapshotCoordinator.h/.cpp` | global epoch provider supplied from `AudioEngine::m_epochDomain`; snapshot retirement/reclaim | Can enqueue into the same global DQueue |
| `ConvolverProcessor::m_epochDomain` | `src/ConvolverProcessor.h:1418` | `runtimeRcuReader` protects Convolver runtime objects; nested `DeferredDeletionQueue` | Separate from the Engine DQueue; not direct S7 evidence |
| `EQProcessor::m_epochDomain` | `src/eqprocessor/EQProcessor.h:488` | `rcuReader` protects EQ state/nodes; nested queue | Separate from the Engine DQueue; not direct S7 evidence |
| `SafeStateSwapper` | `src/SafeStateSwapper.h` | fixed 8-slot custom RCU for `ConvolverState`; `rcuSwapper` | Separate implementation and epoch space; not direct S7 evidence |

`RuntimePublicationOrchestrator::publicationReader`, `NoiseShaperLearner::rcuReader`, and `SpectrumAnalyzerComponent::rcuReader` all use the Engine `ISRRetireRouter`, so they are global-domain readers even though their owning components are separate.

## 5. Production RCUReader / Epoch Callsite Inventory

The inventory below covers current production `src` code and excludes test-only implementations. No additional production `RCUReader` construction or `makeRuntimeReadHandle` callsite was found.

### 5.1 Persistent `RCUReader` objects

| reader object | construction | production entry points | scope / thread contract | shutdown connection | static status |
|---|---|---|---|---|---|
| `AudioEngine::audioThreadRcuReader` | `AudioEngine.h:5024` with global `m_epochDomain` | `readAudioRuntimeView()` at `AudioEngine.h:3380-3383`; used by `getNextAudioBlock`, `processBlockDouble`, and `processWithSnapshot` | Audio callback; returned `RuntimeReadHandle` owns `ObservedRuntime`/`RCUReaderGuard` for function scope | Harness stops and joins audio thread before terminal release | Callsite complete; no long-lived handle found |
| `AudioEngine::messageThreadRcuReader` | `AudioEngine.h:5026` with global `m_epochDomain` | constructor health callback, destructor, learning commands, `prepareToPlay`, `releaseResources`, snapshot creation, Timer, fade/recovery paths | Non-RT Message/Timer/control paths; handles are local and lexical | Timer is stopped and producers are joined in normal terminal path; direct destructor fallback differs | Callsite complete; cross-thread execution is not runtime-proven |
| `RuntimePublicationOrchestrator::publicationReader` | `RuntimePublicationOrchestrator.cpp:20-29` from `engine.getRetireRouter()` | `trySubmitImpl()` at `RuntimePublicationOrchestrator.cpp:41-52` | Publication submission path; production callers resolve to RebuildThread/deferred handoff | `stopRebuildThread()` joins the producer before reader close | Callsite complete; no persistent handle found |
| `NoiseShaperLearner::rcuReader` | `NoiseShaperLearner.cpp:45-50` from `engine.getRetireRouter()` | `captureSessionSignature()` at `NoiseShaperLearner.cpp:1048-1059`, called by learner worker | Learner worker; guard is scoped to the runtime-world lookup | Normal terminal path calls `stopLearning()` and joins worker before reader close; direct `~AudioEngine` fallback relies on later member destruction | Callsite complete; fallback ordering is a source caveat |
| `SpectrumAnalyzerComponent::rcuReader` | `SpectrumAnalyzerComponent.cpp:50-53` from `audioEngine.getRetireRouter()` | `timerCallback()` at `SpectrumAnalyzerComponent.cpp:275-287` | JUCE Message/UI Timer | Destructor stops timer; `MainWindow` declares analyzer after `audioEngine`, so it is destroyed before the engine in the normal UI owner | Not present in `AudioEngineHarness`; external component lifetime is not an M0 path |
| `ConvolverProcessor::runtimeRcuReader` | `ConvolverProcessor.h:1418-1420` with nested Convolver domain | `ConvolverProcessor::process()` at `ConvolverProcessor.Runtime.cpp:238-240` | Audio thread; nested domain only | Audio thread stopped before global shutdown path | Callsite complete; excluded from Engine DQueue attribution |
| `EQProcessor::rcuReader` | `EQProcessor.h:488-497` with nested EQ domain | `EQProcessor::process()` at `EQProcessor.Processing.cpp:486-489` | Audio thread; nested domain only | EQ release/drain is separate from Engine global queue | Callsite complete; excluded from Engine DQueue attribution |

`ObservedRuntime` construction is owned by `SnapshotCoordinator::observeCurrentRuntime()` (`SnapshotCoordinator.h:77-81`); no production caller of that method was found. `RCUReaderGuard` construction is otherwise the `process()` callsites above or the `RuntimeReadHandle` path.

### 5.2 Direct Engine-domain fixed reader slots

`ConvolverProcessor` has a second, non-RAII path into the **Engine** domain:

```text
ConvolverProcessor::enterGlobalReader(index)
  -> AudioEngine::enterRcuReader(index)
  -> m_retireRouter->enterReader(index)
  -> m_epochDomain.enterReader(index)
```

The corresponding exit is `exitGlobalReader()` -> `AudioEngine::exitRcuReader()` -> `m_retireRouter->exitReader()`.

| fixed slot | production callsites | guard shape | source observation |
|---:|---|---|---|
| `2` | `ConvolverProcessor::prepareToPlay`, `reset`, `shareConvolutionEngineFrom`, `getLatencyBreakdown` | local `GlobalGuard` calls direct provider enter/exit | No `reserveReaderThread(2)` or `registerReaderThread()` owner is established by these callsites |
| `3` | `ConvolverProcessor::refreshLatency`, `ConvolverProcessor::timerCallback` | local `GlobalGuard` calls direct provider enter/exit | Background/loader and timer paths are not tied to an `RCUReader` owner token |

No production `startTimer()` callsite for `ConvolverProcessor` was found; `timerCallback` is therefore a compiled production function, not proof that this timer was active in the M0 episode.

The direct `EpochDomain::enterReader()` path has material static differences from `RCUReader::enter()`:

1. It writes `slot.epoch = currentEpoch()` and increments `slot.depth` directly.
2. It does not acquire an `RCUReader` thread slot, owner token, or preferred-slot reservation.
3. It does not test `registrationClosed_` or quarantine state.
4. It does not update `ownerThreadId`/`ownerTag`.
5. It permits fixed-slot overlap; the source does not provide a single-owner assertion for slots 2 and 3.
6. On a nested direct enter, it writes the current epoch before discovering `previousDepth > 0`, so overlapping guards can overwrite the slot epoch while an earlier guard remains active.

These facts make fixed slots 2/3 the only production source path that can create an Engine-domain reader without the `RCUReader` owner/reservation lifecycle. They do **not** prove that either slot was active at S7, and the lack of owner metadata means a future slot dump could not identify a direct-guard owner from `ownerThreadId`.

### 5.3 Separate custom reader slots

`ConvolverProcessor::rcuSwapper` uses `SafeStateSwapper`, not `EpochDomain`:

- `AudioEngine::createSnapshotFromCurrentState()` uses local slot `1` (`AudioEngine.Snapshot.cpp:22-26`).
- `ConvolverProcessor::isCacheEntrySafeToDelete()` uses local slot `2` (`ConvolverProcessor.LoadPipeline.cpp:202-215`).
- `SafeStateSwapper::getMinReaderEpoch()` scans its own eight `readerEpochs` and its own `globalEpoch` (`SafeStateSwapper.h:374-395`).

These callsites cannot explain the Engine DQueue's `minReaderEpoch=9` unless a separate cross-domain defect is introduced; no such connection is present in the inspected source.

### 5.4 Inventory conclusion

```text
PRODUCTION_RCU_CALLSITE_INVENTORY       = COMPLETE
GLOBAL_ENGINE_READER_PATHS             = MAPPED
DIRECT_FIXED_SLOT_PATHS                = MAPPED
NESTED_DOMAIN_SEPARATION               = PROVEN
UNPROVEN_RUNTIME_READER_IDENTITY        = REMAINS
```

## 6. Reader Lifecycle and Close Semantics

### 6.1 RAII path

`RCUReader::enter()` obtains or reuses a slot and calls provider `enterReader()` at `RCUReader.h:36-82`. `RCUReader::exit()` calls provider `exitReader()` only on the outer exit at `RCUReader.h:105-147`. The nested RAII depth prevents an inner guard from becoming a second provider reader.

`EpochDomain::enterReader()` publishes the sampled epoch before the depth increment (`EpochDomain.h:117-141`), and final exit clears the epoch to `kInactiveEpoch` (`EpochDomain.h:143-184`). This gives the normal RAII path a source-level balanced lifetime.

### 6.2 Registration close is not forced exit

`closeReaderRegistration()` only publishes `registrationClosed_` and increments the registration generation (`EpochDomain.h:626-639`). It does not walk slots or clear active epochs.

For an `RCUReader` that is already active, an in-flight guard can finish. For an `RCUReader` that has already exited, a later `enter()` cannot reserve/register a new slot after close; `acquireThreadSlot()` fails closed. This is a different contract from the direct fixed-slot path, which calls `EpochDomain::enterReader()` and bypasses `registrationClosed_` entirely.

Therefore:

```text
closeReaderRegistration blocks new RCUReader allocation
closeReaderRegistration does not force active RCUReader exit
closeReaderRegistration does not block direct EpochDomain::enterReader(2/3)
```

### 6.3 `getMinReaderEpoch` truth table

`EpochDomain::getMinReaderEpoch()` starts with `currentEpoch()` and scans each slot (`EpochDomain.h:214-248`):

| slot state at scan | contribution |
|---|---|
| quarantined flag set | skipped |
| depth zero | skipped |
| reserved/inactive epoch | skipped |
| eligible epoch older than current | lowers minimum |
| eligible epoch equal to or newer than current | does not lower minimum |
| no eligible slot | returns current epoch |

Thus the S7 equality remains compatible with both no eligible reader and an eligible reader at the same/newer epoch. The scan is not a single snapshot of all slot fields, so an enter/exit or epoch transition during the scan cannot be excluded as Case C.

## 7. Epoch Publication and DQueue Enqueue Chronology

### 7.1 Engine global epoch writers

| source path | operation | relation to DQueue |
|---|---|---|
| `AudioEngine.h:4466-4481` | `enqueueDeferredDeleteNonRt*()` calls `markRetireEpoch()` before the normal enqueue or shutdown ownership transfer | The normal branch can create a DQueue entry at the newly published current epoch; the shutdown branch transfers to shutdown/terminal authority |
| `RuntimePublishExecutor.h:104-114` | calls `ctx.engine.advanceRetireEpoch()` after publish execution tail | Advances global epoch after a committed publication; does not identify an existing DQueue head |
| `SnapshotCoordinator.cpp:81-119` | `resetFadeStateAndRetireTarget()` / `completeFade()` publish an epoch before enqueuing an old snapshot | A global snapshot entry can be stamped with the newly current epoch |
| `AudioEngine.Processing.ReleaseResources.cpp:262-265` | explicit `advanceRetireEpoch()` before `RetireClosed`/`EpochSettled` | Establishes the terminal-pass epoch baseline |
| `AudioEngine.Processing.ReleaseResources.cpp:305-306,322-323` | graceful-loop and timeout-loop `publishEpoch()` followed by safe reclaim | Can make older entries reclaimable, but only while that loop runs |
| `AudioEngine.CtorDtor.cpp:227-250` | destructor force advance and polling publishes | Abnormal destructor path; not the normal M0 `releaseResources()` path |
| `ConvolverProcessor.LoadPipeline.cpp:780-792` and `StateAndUI.cpp:1083-1089` | Engine provider `advanceRetireEpoch()` on Convolver engine/state swap | Global epoch writer reached through the Engine provider; separate from nested Convolver reader domain |
| `EQProcessor.Core.cpp:69-76` | `m_epochDomain.publishEpoch()` | Nested EQ domain only; excluded from Engine DQueue chronology |

`ISRRetireRouter::currentEpoch()` and `minReaderEpoch()` are reads/delegates, not epoch writers (`ISRRetireRouter.cpp:161-223`).

### 7.2 Engine DQueue entry families

The following production families can enqueue into the global `DeferredDeletionQueue`:

1. `RuntimePublicationBridge::retirePublishedRuntimeWorldNonRt()` and rejected-world paths through `AudioEngine::enqueueDeferredDeleteNonRt*()`.
2. `DSPLifetimeManager::retire()` / `retireByHandle()` through `ISRRetireRouter::enqueueWithRetry()`.
3. `SnapshotCoordinator` snapshot replacement/fade retirement.
4. Convolver IR/engine retirement through the Engine deferred-delete helper.
5. Router fallback/quarantine/terminal paths that ultimately transfer ownership to the global queue or terminal authority.

`DeferredDeletionQueue::reclaim()` uses strict `isOlder(entry.epoch, minReaderEpoch)` and stops at the FIFO head (`DeferredDeletionQueue.h:109-177`). The source does not expose the S7 entry's pointer/deleter/type, so static closure cannot identify which family created slot index 4.

### 7.3 Source-level epoch compatibility

An Engine entry with `epoch=9` is compatible with:

```text
markRetireEpoch() returned 9 and the entry was enqueued immediately;
currentEpoch() was 9 when a retirement path sampled it;
SnapshotCoordinator published 9 immediately before enqueue;
a reader entered at 9 and a concurrent baseline was 9;
no eligible reader existed while currentEpoch() was 9.
```

The strict epoch gate makes all of these conservative at the observed instant. Static source cannot distinguish them without the S7 enqueue call or slot state.

## 8. Shutdown Ordering and Reader Proof

### 8.1 Normal terminal path

The current `releaseResources()` order is:

| order | operation | reader/epoch consequence |
|---:|---|---|
| 1 | stop admission/request shutdown; stop learner and callbacks | stops normal new work; learner worker is joined on the normal path |
| 2 | join CoordinatorLoop and RebuildThread | removes publicationReader/rebuild producer paths before reader close |
| 3 | `advanceRetireEpoch()` at `ReleaseResources.cpp:262-265` | advances global epoch before terminal drain |
| 4 | `closeReaderRegistration()` at `:269-274` | blocks new RCUReader allocation; does not force active/direct slots out |
| 5 | graceful `publishEpoch()` + `tryReclaim()` loop | attempts epoch progress and safe DQueue reclaim |
| 6 | `drainDeferredRetireQueues(true)` and `ReclaimComplete` | can still leave an epoch-equal FIFO head because safe reclaim does not advance epoch |
| 7 | quarantine/terminal handling and `tryShutdownQuiescentReclaim()` | active/fading handles use Proof/Permit path; `tryShutdownQuiescentReclaim()` observes `activeReaderCount()==0` at one instant |
| 8 | `uiConvolverProcessor.releaseResources()` and `uiEqEditor.releaseResources()` | nested processor resources are released; their domains are separate |
| 9 | world clear and final DSP retire | can transfer additional global-retirement ownership; shutdown helper may publish another epoch |
| 10 | `waitForDrain(2000,2)` | calls `drainDeferredRetireQueues(true)` but does not call `publishEpoch()` |
| 11 | post-wait RetireIntent drain and timeout-path `m_epochDomain.tryReclaim()` | RCA-7/RCA-9 S7 first observed Engine reclaim occurred in this safe-retry boundary |
| 12 | coordinator finalization, OwnerChannel residual drain, and terminal finalization | terminal state does not repair the DQueue residual |

The S7 CDB log places the first observed Engine `getMinReaderEpoch` call after the `[DIAG] releaseResources: drain timeout reached` line (`rca9-cdb.log:316-330`). Static source therefore proves that the final wait is an epoch-equal-head opportunity, but it still does not prove which earlier enqueue created the head.

### 8.2 Quiescence proof boundary

`tryShutdownQuiescentReclaim()` fills `QuiescenceObservation` at `AudioEngine.h:4643-4655`:

- Q3 reads `readerRegistrationClosed()`.
- Q4 reads an instantaneous `activeReaderCount()==0`.
- Q5 `epochSettled` is set to `true` by the caller rather than derived from an epoch evidence object.
- Q6 `postStopEnqueueZero` is set to `true` by the caller.
- Q7 checks that admission is not open.

`tryMakeQuiescenceProof()` requires all Q0-Q7 (`ISRShutdown.cpp:350-397`), and `reclaimShutdownQuiescent()` then bypasses the epoch check after validating the permit identity and consuming it (`ISRRuntimePublicationCoordinator.cpp:748-778`). This is a source-level shutdown proof boundary, not evidence that the S7 head was created by that path.

The normal path joins the known producer threads before proof, but the direct fixed-slot path is not represented by an `RCUReader` owner token. A direct guard active at Q4, or a direct guard entered after Q4, remains a static lifecycle possibility; no runtime evidence selects it.

### 8.3 Abnormal destructor path

`~AudioEngine()` closes the global reader registration and drains at `AudioEngine.CtorDtor.cpp:224-292` before later destruction of `noiseShaperLearner`, `uiConvolverProcessor`, and other members. The normal harness path calls `releaseResources()` first, so this is not the primary M0 path. It remains a separate fallback lifecycle candidate and is not promoted to the S7 cause.

## 9. Case A/B/C/D Reclassification

| case | static source condition | compatibility with S7 | RCA-10 status |
|---|---|---|---|
| A — no eligible reader | all slots are depth-zero, reserved/inactive, or quarantined; `minEpoch` remains `currentEpoch()` | `currentEpoch=9` directly yields `minReaderEpoch=9` | **POSSIBLE / NOT PROVEN** |
| B — eligible reader at/above baseline | one or more eligible slots have epoch `9` or newer; strict `isOlder` does not lower the base | equality is expected; fixed slots 2/3 and RAII readers are candidates | **POSSIBLE / NOT PROVEN** |
| C — transition during scan | flags/depth/epoch are read separately; enter/exit/epoch changes can occur while the 64-slot scan is in progress | a pre/post return difference cannot be excluded without slot chronology | **POSSIBLE / NOT PROVEN** |
| D — epoch publication supplies baseline | `publishEpoch()`/retirement enqueue chronology makes current epoch 9; no reader identity is required | fully compatible with `minReaderEpoch=9` | **POSSIBLE / NOT PROVEN** |

The source closure does not make the cases mutually exclusive. It establishes that no-reader and reader-at-9 are observationally identical at the DQueue gate, and that the current source has multiple global epoch writers capable of producing the baseline.

## 10. RuntimeIntentCoordinator P9-P22 Reconfirmation

The coordinator predicate is a separate domain from the Engine DQueue. The current source mapping remains:

| ID | source predicate | production writer(s) | zero transition / consumer | RCA-10 status |
|---|---|---|---|---|
| P9 | `swapPending_ == false` (`Coordinator.cpp:512`) | `commit()` sets true then false (`:109-120`); test setter exists | commit clears; `markTransitionCommitted()` reads it | source mapped; runtime value not captured |
| P10 | `intentQueue_.sizeApprox() == 0` (`:527`) | `submitObserve`, `enqueuePublicationIntent`, `submitQuarantine` | `processIntent()` pops common intents (`ProcessIntent.cpp:47-73`) | source mapped; independent of DQueue |
| P11 | `observeDeferredRing_.size() == 0` (`:528`) | `submitObserve()` overflow fallback | `drainObserveDeferred()` pops and decrements (`ProcessIntent.cpp:87-99`) | source mapped |
| P12 | `quarantineFallbackQueue_.sizeApprox() == 0` (`:529`) | `submitQuarantine()` fallback push (`:1517-1518`) | fallback pop at `ProcessIntent.cpp:35-40` | source mapped |
| P13 | `recoveryIntentQueue_.size() == 0` (`:530`) | `submitRecoveryRequest()` transport push (`Coordinator.cpp:995-998`) | Builder pop; shutdown discard path | source mapped |
| P14 | `retireBacklogCount_ == 0` (`:531`) | legacy `enqueueRetire()` calls `onRetireAccepted()`; no production `onRetireConsumed()` caller found | setter is test-only; comments identify Layer 1 measurement as authoritative | source mapped; likely vestigial in production, not a DQueue term |
| P15 | `publicationBacklogCount_ == 0` (`:532`) | no production writer found; setter is test-only | no production zero transition found | source mapped; likely vestigial, runtime not captured |
| P16 | `publicationIntentResidencyCount_ == 0` (`:536`) | reservation-before-push in `enqueuePublicationIntent()` (`.h:789-799`) | Publish pop decrement (`ProcessIntent.cpp:55-63`) | source mapped |
| P17 | `pendingIntentCount_ == 0` (`:537`) | Observe/Recovery/Quarantine reservation-before-push | common/fallback/deferred/recovery pops and recovery admission settlement | source mapped |
| P18 | `reclaimInFlightCount_ == 0` (`:542`) | `onReclaimBegin()` for deferred normal reclaim | `onReclaimEnd()` on success; wrappers call both | source mapped; approximate counter by its own comments |
| P19 | `quarantineIntentResidencyCount_ == 0` (`:548`) | primary quarantine reservation (`:1503-1505`) | primary pop decrement (`ProcessIntent.cpp:64-66`) | source mapped |
| P20 | `quarantineRingResidencyCount_ == 0` (`:549`) | fallback ring increment (`:1517-1518`) | fallback pop decrement (`ProcessIntent.cpp:35-40`) | source mapped |
| P21 | `!recoveryAdmissionPending_` (`:554`) | durable attach sets true (`:1378-1390`) | Builder settle or shutdown discard clears it (`:1321-1325`, `:1347-1352`) | source mapped |
| P22 | `liveLogicalRecoveryObligationCount() == 0` (`:561`) | `RecoveryAdmissionTable::tryInsert()` increments only the Live CAS winner (`.h:461-499`) | `resolve()` decrements only the terminal CAS winner (`.h:513-529`) | source mapped |

No P9-P22 field is the Engine DQueue `entry.epoch`, `enqueuePos`, `dequeuePos`, or `getMinReaderEpoch`. The exact S7 coordinator false term remains unknown; source mapping must not be promoted to a same-cause claim.

## 11. Static Findings Relevant to minReaderEpoch=9

### Finding 1 — Direct fixed slots are the only non-RAII global path

`ConvolverProcessor::GlobalGuard` slots 2/3 call `EpochDomain::enterReader()` without `RCUReader` registration, owner token, or reservation. This is a concrete source distinction and the highest-priority static candidate for a future authorized runtime observation. It is not proof of S7 involvement.

### Finding 2 — Direct fixed-slot diagnostics lack owner identity

Because `EpochDomain::enterReader()` does not write `ownerThreadId`, a slot dump of a direct guard could show epoch/depth but not the owning thread. A future capture would need call-stack or guard identity correlation; this gate does not add instrumentation.

### Finding 3 — Close-registration is an allocation gate, not a reader barrier

Normal RCUReaders are lexically scoped and known worker joins make the common path plausible. Direct `enterReader(2/3)` remains callable after close, and Q4 is only an instantaneous observation. This is a lifecycle caveat, not a demonstrated S7 race.

### Finding 4 — Epoch-equal DQueue head is source-compatible without a stuck reader

The current epoch can be advanced by retirement, snapshot fade completion, Convolver provider swaps, publication execution, and shutdown loops. `DeferredDeletionQueue` requires strict older-than-min, and the final wait does not advance epoch. Therefore an entry stamped at the current epoch can block even when no reader is stuck.

### Finding 5 — Nested domains do not explain the Engine DQueue

Convolver `m_epochDomain`, Convolver `SafeStateSwapper`, and EQ `m_epochDomain` have separate reader arrays/epochs. Their activity cannot be used as the Engine `ReaderSlot` explanation without additional cross-domain evidence.

## 12. Direct Cause / Upstream Cause / Terminal Consequence

| layer | classification | RCA-10 statement |
|---|---|---|
| direct reclaim cause | **PROVEN** | S7 head sequence is ready, but `isOlder(9,9)` is false; CAS is not reached |
| upstream reader identity | **UNRESOLVED** | no valid S7 slot dump, owner identity, or pre/post reader chronology exists |
| upstream epoch provenance | **UNRESOLVED** | source has multiple global epoch writers and enqueue families; exact S7 writer is not captured |
| Case A | **NOT PROVEN** | no eligible reader is statically possible |
| Case B | **NOT PROVEN** | eligible reader at/above epoch 9 is statically possible, including fixed slots 2/3 |
| Case C | **NOT PROVEN** | non-snapshot slot scan permits transition ambiguity |
| Case D | **NOT PROVEN** | current-epoch publication/retirement baseline is statically possible without a reader |
| terminal consequence | **PROVEN AT CONTRACT LEVEL** | residual DQueue depth keeps `isFullyDrained()` false; `waitForDrain()` can time out; `completed=true`/`ShutdownComplete` does not mean drain success |
| coordinator relation | **UNRESOLVED** | P9-P22 are source-mapped and independent; exact false term is not captured |
| DQueue corruption | **NOT SUPPORTED** | sequence is ready and the later RCA-7 control CAS succeeds after epoch progress |

## 13. STOP Matrix and Disposition

```text
STOP-1_SOURCE_IDENTITY_MISMATCH       = NOT_TRIGGERED
STOP-2_PRODUCTION_CALLSITE_INCOMPLETE = NOT_TRIGGERED
STOP-3_EXACT_S7_SLOT_EPOCH_CHRONOLOGY = TRIGGERED
STOP-4_CASE_A_B_C_D_NOT_EXCLUDED     = TRIGGERED
STOP-5_SHUTDOWN_OR_COORDINATOR_LINK   = TRIGGERED
```

The source inventory is complete, but the exact S7 runtime state is not recoverable under this gate. A further source-only pass cannot turn `minReaderEpoch=9` into a reader identity or an enqueue-call identity. RCA-11 is therefore not authorized by this report.

```text
RCA10_GATE_DISPOSITION = PARTIALLY-LOCALIZED / STOP
READER_CALLSITE_INVENTORY = COMPLETE
GLOBAL_DOMAIN_TOPOLOGY = PROVEN
NESTED_DOMAIN_SEPARATION = PROVEN
CASE_A = POSSIBLE_NOT_PROVEN
CASE_B = POSSIBLE_NOT_PROVEN
CASE_C = POSSIBLE_NOT_PROVEN
CASE_D = POSSIBLE_NOT_PROVEN
MIN_READER_EPOCH_9_SOURCE = UNRESOLVED
READER_IDENTITY = UNRESOLVED
DIRECT_RECLAIM_CAUSE = PROVEN
UPSTREAM_CAUSE = UNRESOLVED
TERMINAL_CONSEQUENCE = PROVEN_AT_CONTRACT_LEVEL
P9_P22_SOURCE_MAPPING = PROVEN
EXACT_COORDINATOR_FALSE_TERM = UNRESOLVED
M0 = 0
M1 = 0 (RCA-9 inherited 1 failed M1)
M2 = 0
BUILD = NOT_RUN
DR_MEMORY = 0
CDB_RETRY = FORBIDDEN
IMPLEMENTATION = FORBIDDEN
FURTHER_CAPTURE = NOT_AUTHORIZED
```

## 14. Evidence Index

| evidence | location |
|---|---|
| Source identity | `ConvoPeq.md`, SHA-256 `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609`, timestamp `2026-09-23T14:33:38.6388122Z` |
| Epoch initialization/registration | `src/core/EpochDomain.h:26-114` |
| Reader enter/exit | `src/core/EpochDomain.h:117-184` |
| Current/publish epoch | `src/core/EpochDomain.h:186-202` |
| Minimum-reader scan | `src/core/EpochDomain.h:214-248` |
| Quarantine and close registration | `src/core/EpochDomain.h:271-325,626-639` |
| RAII reader lifecycle | `src/core/RCUReader.h:36-82,105-181` |
| Engine reader objects | `src/audioengine/AudioEngine.h:5017-5026` |
| Runtime read handle | `src/audioengine/AudioEngine.h:3380-3387` |
| Engine reader wrappers | `src/audioengine/AudioEngine.Reader.cpp:11-18` |
| Direct fixed-slot reader path | `src/convolver/ConvolverProcessor.Runtime.cpp:112-119,217-240`; `ConvolverProcessor.Lifecycle.cpp:146-153,234-240,520-529`; `StateAndUI.cpp:422-428,726-732` |
| Convolver provider epoch writers | `src/convolver/ConvolverProcessor.LoadPipeline.cpp:780-792`; `StateAndUI.cpp:1083-1089` |
| Learner reader lifecycle | `src/NoiseShaperLearner.cpp:45-50,185-213,734-760,891-897,1048-1059` |
| Analyzer reader lifecycle | `src/SpectrumAnalyzerComponent.cpp:50-53,119-124,275-287`; `src/MainWindow.h:64-73` |
| Nested Convolver reader | `src/ConvolverProcessor.h:1417-1420`; `ConvolverProcessor.Runtime.cpp:238-240` |
| Nested Convolver SafeStateSwapper | `src/SafeStateSwapper.h:151-179,374-395`; `AudioEngine.Snapshot.cpp:22-26`; `ConvolverProcessor.LoadPipeline.cpp:202-215` |
| Nested EQ reader/domain | `src/eqprocessor/EQProcessor.h:488-497`; `EQProcessor.Processing.cpp:486-489`; `EQProcessor.Core.cpp:26-76,138-205` |
| Global DQueue enqueue families | `src/audioengine/AudioEngine.h:4460-4497`; `DSPLifetimeManager.cpp:35-137`; `SnapshotCoordinator.cpp:53-119`; `ISRRetireRouter.cpp:239-342` |
| DQueue reclaim predicate | `src/DeferredDeletionQueue.h:109-177` |
| Shutdown ordering | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp:95-109,240-375,474-567,615-745` |
| Reader quiescence proof | `src/audioengine/AudioEngine.h:4638-4672`; `ISRShutdown.cpp:350-412`; `ISRRuntimePublicationCoordinator.cpp:748-778` |
| Drain predicate and wait | `src/audioengine/AudioEngine.Threading.cpp:153-247` |
| Drain audit semantics | `src/audioengine/RuntimeDrainAudit.h:11-84` |
| P9-P22 predicate | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:417-579,967-1020,1274-1400,1474-1530`; `ISRRuntimePublicationCoordinator_ProcessIntent.cpp:10-100` |
| P21/P22 recovery table | `src/audioengine/ISRRuntimePublicationCoordinator.h:450-539` |
| RCA-7 direct proof | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-7_DQueue_head_block_exact.md` |
| RCA-8 terminal reconciliation | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-8_EpochHold_terminal_contract_reconciliation.md` |
| RCA-9 failed M1 evidence | `C:\Users\user\AppData\Local\Temp\opencode\rca9-cdb.log:257-336` |
| RCA-9 script (not rerun) | `C:\Users\user\AppData\Local\Temp\opencode\rca9-cdb-capture.txt` |
| RCA-9 report | `doc/work113/P1-5-IR-P2_Step5BA_P3-5-FPM-RCA-9_EpochProvenance_reader_lifecycle_exact.md` |

No new runtime evidence was created by RCA-10. The report is a static closure and a STOP disposition, not an implementation authorization.
